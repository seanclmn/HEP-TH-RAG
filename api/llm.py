"""Streaming chat completions from a local Ollama server (POST /api/chat).

Ollama streams newline-delimited JSON objects:
    {"message": {"role": "assistant", "content": "<piece>"}, "done": false}
    ...
    {"done": true, ...stats}
or {"error": "..."} on failure.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .config import settings

TIMEOUT = httpx.Timeout(10.0, read=300.0)  # a slow model can pause a while between tokens


class LLMError(Exception):
    pass


async def stream_chat(messages: list[dict], *, model: str | None = None,
                      client: httpx.AsyncClient | None = None) -> AsyncIterator[str]:
    """Yield the assistant's reply piece by piece. Raises LLMError on any failure."""
    model = model or settings.ollama_model
    # think=false is accepted by every model; think=true only by thinking models (qwen3, ...).
    # The reasoning itself arrives in message.thinking, which is not streamed to the client.
    payload = {"model": model, "messages": messages, "stream": True, "think": settings.ollama_think}
    owns_client = client is None
    client = client or httpx.AsyncClient(base_url=settings.ollama_url, timeout=TIMEOUT)
    try:
        async with client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                try:
                    body = json.loads(body).get("error", body)
                except ValueError:
                    pass
                hint = f" (run: ollama pull {model})" if r.status_code == 404 else ""
                raise LLMError(f"Ollama returned {r.status_code}: {body}{hint}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if err := data.get("error"):
                    raise LLMError(f"Ollama error: {err}")
                if text := data.get("message", {}).get("content"):
                    yield text
                if data.get("done"):
                    return
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot reach Ollama at {settings.ollama_url}; is `ollama serve` running?") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama request failed: {e!r}") from e
    finally:
        if owns_client:
            await client.aclose()
