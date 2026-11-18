"""The REST API against real Postgres, with fake retrieval and a fake LLM.

Each test gets a throwaway schema (the API's connections use it via search_path),
which is dropped afterwards.
"""

import json
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from api import llm, main, rag
from api.config import settings
from retrieval import Hit

pytestmark = pytest.mark.db

HITS = [Hit(paper_id="2607.28486", chunk_index=1, title="SYK and gravity", section_path="1 Intro",
            content=r"$H = \sum J_{ijkl}\chi_i\chi_j\chi_k\chi_l$", arxiv_abs_url=None, score=0.03)]


@pytest.fixture
def client(monkeypatch):
    schema = f"test_api_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(settings.database_url, autocommit=True)
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    admin.execute(f"CREATE SCHEMA {schema}")
    monkeypatch.setattr(settings, "db_options", f"-c search_path={schema},public")
    monkeypatch.setattr(rag, "retrieve", lambda question, k: HITS[:k])
    try:
        with TestClient(main.app) as c:   # lifespan applies 03-chat.sql in the temp schema
            yield c
    finally:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace Ollama; returns the list of message lists it was called with."""
    calls, script = [], {"pieces": ["The ", "answer [1]."], "fail_after": None}

    async def stream_chat(messages, *, model=None, client=None):
        calls.append(messages)
        for i, piece in enumerate(script["pieces"]):
            if script["fail_after"] == i:
                raise llm.LLMError("boom")
            yield piece

    monkeypatch.setattr(llm, "stream_chat", stream_chat)
    return calls, script


def events(response) -> list[tuple[str, object]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def new_chat(client, **body) -> str:
    r = client.post("/chats", json=body or None)
    assert r.status_code == 201
    return r.json()["id"]


class TestChats:
    def test_create_list_get_delete(self, client):
        a = new_chat(client, title="Holography")
        b = new_chat(client)
        listed = client.get("/chats").json()
        assert {c["id"] for c in listed} == {a, b}
        detail = client.get(f"/chats/{a}").json()
        assert detail["title"] == "Holography" and detail["messages"] == []
        assert client.delete(f"/chats/{a}").status_code == 204
        assert client.get(f"/chats/{a}").status_code == 404
        assert client.delete(f"/chats/{a}").status_code == 404

    def test_unknown_and_malformed_ids(self, client):
        assert client.get(f"/chats/{uuid.uuid4()}").status_code == 404
        assert client.get("/chats/not-a-uuid").status_code == 422


class TestMessages:
    def test_streams_sources_deltas_done_and_saves_both_messages(self, client, fake_llm):
        chat = new_chat(client)
        r = client.post(f"/chats/{chat}/messages", json={"content": "What is  the SYK model?"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = events(r)
        assert [e for e, _ in ev] == ["sources", "delta", "delta", "done"]
        assert ev[0][1][0]["paper_id"] == "2607.28486"
        assert "".join(d["text"] for e, d in ev if e == "delta") == "The answer [1]."

        detail = client.get(f"/chats/{chat}").json()
        assert detail["title"] == "What is the SYK model?"
        user, assistant = detail["messages"]
        assert (user["role"], user["content"]) == ("user", "What is  the SYK model?")
        assert assistant["role"] == "assistant" and assistant["content"] == "The answer [1]."
        assert assistant["id"] == ev[-1][1]["message_id"]
        assert assistant["sources"][0]["title"] == "SYK and gravity"
        assert assistant["model"] == settings.ollama_model

    def test_history_and_excerpts_reach_the_llm(self, client, fake_llm):
        calls, _ = fake_llm
        chat = new_chat(client)
        client.post(f"/chats/{chat}/messages", json={"content": "q1"})
        client.post(f"/chats/{chat}/messages", json={"content": "q2"})
        second = calls[1]
        assert second[0]["role"] == "system" and r"J_{ijkl}" in second[0]["content"]
        assert second[1:] == [{"role": "user", "content": "q1"},
                              {"role": "assistant", "content": "The answer [1]."},
                              {"role": "user", "content": "q2"}]

    def test_k_zero_skips_retrieval(self, client, fake_llm, monkeypatch):
        def no(*a):
            raise AssertionError("retrieval should not run")
        monkeypatch.setattr(rag, "retrieve", no)
        chat = new_chat(client)
        ev = events(client.post(f"/chats/{chat}/messages", json={"content": "hi", "k": 0}))
        assert ev[0] == ("sources", [])

    def test_llm_failure_keeps_partial_answer(self, client, fake_llm):
        _, script = fake_llm
        script["fail_after"] = 1
        chat = new_chat(client)
        ev = events(client.post(f"/chats/{chat}/messages", json={"content": "q"}))
        assert [e for e, _ in ev] == ["sources", "delta", "error", "done"]
        assert ev[2][1] == {"detail": "boom"}
        messages = client.get(f"/chats/{chat}").json()["messages"]
        assert [m["content"] for m in messages] == ["q", "The "]

    def test_llm_failure_before_any_text_saves_no_answer(self, client, fake_llm):
        _, script = fake_llm
        script["fail_after"] = 0
        chat = new_chat(client)
        ev = events(client.post(f"/chats/{chat}/messages", json={"content": "q"}))
        assert ev[-1] == ("done", {"message_id": None})
        assert [m["role"] for m in client.get(f"/chats/{chat}").json()["messages"]] == ["user"]

    def test_retrieval_failure_is_reported(self, client, fake_llm, monkeypatch):
        def broken(*a):
            raise RuntimeError("no index")
        monkeypatch.setattr(rag, "retrieve", broken)
        chat = new_chat(client)
        ev = events(client.post(f"/chats/{chat}/messages", json={"content": "q"}))
        assert ev == [("error", {"detail": "retrieval failed: no index"}), ("done", {"message_id": None})]

    def test_unknown_chat_and_validation(self, client, fake_llm):
        assert client.post(f"/chats/{uuid.uuid4()}/messages", json={"content": "q"}).status_code == 404
        chat = new_chat(client)
        assert client.post(f"/chats/{chat}/messages", json={"content": ""}).status_code == 422
        assert client.post(f"/chats/{chat}/messages", json={"content": "q", "k": 99}).status_code == 422
        assert client.get(f"/chats/{chat}").json()["messages"] == []

    def test_deleting_chat_deletes_messages(self, client, fake_llm):
        chat = new_chat(client)
        client.post(f"/chats/{chat}/messages", json={"content": "q"})
        client.delete(f"/chats/{chat}")
        with psycopg.connect(settings.database_url, options=settings.db_options) as conn:
            assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
