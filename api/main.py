"""REST API: chats, messages, and streamed LLM answers grounded in the paper corpus.

Run:  .venv/bin/uvicorn api.main:app --reload        (docs at http://localhost:8000/docs)

POST /chats/{id}/messages answers with Server-Sent Events (text/event-stream):
    event: sources   data: [{"n": 1, "paper_id": ..., "title": ..., "url": ...}, ...]
    event: delta     data: {"text": "<next piece of the answer>"}     (repeated)
    event: error     data: {"detail": "..."}                           (only on failure)
    event: done      data: {"message_id": 42}                          (always last)
The user message is saved before streaming starts; the assistant message is saved when
the stream ends, including a partial answer if the LLM fails or the client disconnects.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal
from uuid import UUID

import anyio
import psycopg
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from . import llm, rag
from .config import ROOT, settings

log = logging.getLogger("api")


async def connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        settings.database_url, options=settings.db_options, autocommit=True, row_factory=dict_row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with await connect() as conn:
        await conn.execute((ROOT / "db/init/03-chat.sql").read_text())
    yield


app = FastAPI(title="HEP-TH RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["*"], allow_headers=["*"])


# --- models -------------------------------------------------------------------------

class ChatCreate(BaseModel):
    title: str | None = Field(None, max_length=200)


class Chat(BaseModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class Message(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    sources: list[dict] | None = None
    model: str | None = None
    created_at: datetime


class ChatDetail(Chat):
    messages: list[Message]


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    k: int = Field(6, ge=0, le=20, description="Chunks to retrieve; 0 = no retrieval")


# --- chats --------------------------------------------------------------------------

CHAT_COLS = "id, title, created_at, updated_at"


@app.post("/chats", status_code=201, response_model=Chat)
async def create_chat(body: ChatCreate | None = None):
    title = body.title if body else None
    async with await connect() as conn:
        cur = await conn.execute(f"INSERT INTO chats (title) VALUES (%s) RETURNING {CHAT_COLS}", (title,))
        return await cur.fetchone()


@app.get("/chats", response_model=list[Chat])
async def list_chats(limit: int = 50, offset: int = 0):
    async with await connect() as conn:
        cur = await conn.execute(
            f"SELECT {CHAT_COLS} FROM chats ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            (min(max(limit, 1), 200), max(offset, 0)))
        return await cur.fetchall()


@app.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: UUID):
    async with await connect() as conn:
        chat = await (await conn.execute(f"SELECT {CHAT_COLS} FROM chats WHERE id = %s", (chat_id,))).fetchone()
        if chat is None:
            raise HTTPException(404, "chat not found")
        cur = await conn.execute(
            "SELECT id, role, content, sources, model, created_at FROM messages "
            "WHERE chat_id = %s ORDER BY id", (chat_id,))
        return {**chat, "messages": await cur.fetchall()}


@app.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: UUID):
    async with await connect() as conn:
        cur = await conn.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "chat not found")
    return Response(status_code=204)


# --- messages -----------------------------------------------------------------------

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chats/{chat_id}/messages")
async def send_message(chat_id: UUID, body: MessageCreate):
    async with await connect() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "UPDATE chats SET updated_at = now(), title = coalesce(title, left(%s, 80)) "
                "WHERE id = %s RETURNING id", (" ".join(body.content.split()), chat_id))
            if await cur.fetchone() is None:
                raise HTTPException(404, "chat not found")
            cur = await conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
                (chat_id, settings.history_messages))
            history = list(reversed(await cur.fetchall()))
            await conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
                (chat_id, body.content))
    return StreamingResponse(
        answer_stream(chat_id, body, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def answer_stream(chat_id: UUID, body: MessageCreate, history: list[dict]):
    hits = []
    if body.k:
        try:
            hits = await run_in_threadpool(rag.retrieve, body.content, body.k)
        except Exception as e:
            log.exception("retrieval failed")
            yield sse("error", {"detail": f"retrieval failed: {e}"})
            yield sse("done", {"message_id": None})
            return
    sources = rag.sources(hits)
    yield sse("sources", sources)

    model = settings.ollama_model
    parts: list[str] = []
    error = None
    message_id = None
    try:
        try:
            async for text in llm.stream_chat(rag.build_messages(history, body.content, hits), model=model):
                parts.append(text)
                yield sse("delta", {"text": text})
        except llm.LLMError as e:
            error = str(e)
    finally:
        # Runs on success, LLM failure, and client disconnect (cancellation): keep what we have.
        if parts:
            with anyio.CancelScope(shield=True):
                message_id = await save_answer(chat_id, "".join(parts), sources, model)
    if error:
        yield sse("error", {"detail": error})
    yield sse("done", {"message_id": message_id})


async def save_answer(chat_id: UUID, content: str, sources: list[dict], model: str) -> int:
    async with await connect() as conn:
        cur = await conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, model) "
            "VALUES (%s, 'assistant', %s, %s, %s) RETURNING id",
            (chat_id, content, Jsonb(sources), model))
        row = await cur.fetchone()
        await conn.execute("UPDATE chats SET updated_at = now() WHERE id = %s", (chat_id,))
        return row["id"]


@app.get("/health")
async def health():
    out = {"model": settings.ollama_model}
    try:
        async with await connect() as conn:
            await conn.execute("SELECT 1")
        out["database"] = "ok"
    except Exception as e:
        out["database"] = f"error: {e}"
    try:
        import httpx
        async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=3) as c:
            tags = (await c.get("/api/tags")).json()
        names = {m["name"] for m in tags.get("models", [])}
        wanted = {settings.ollama_model, f"{settings.ollama_model}:latest"}
        out["ollama"] = "ok" if names & wanted else f"model not pulled ({settings.ollama_model})"
    except Exception:
        out["ollama"] = f"unreachable at {settings.ollama_url}"
    return out
