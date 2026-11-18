"""api/ pieces that need neither Postgres nor Ollama."""

import asyncio
import json

import httpx
import pytest

from api import llm, rag
from api.main import sse
from retrieval import Hit


def ollama(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))


def ndjson(*objs) -> bytes:
    return "".join(json.dumps(o) + "\n" for o in objs).encode()


def collect(client, messages=None):
    async def run():
        return [t async for t in llm.stream_chat(messages or [{"role": "user", "content": "hi"}],
                                                 model="m", client=client)]
    return asyncio.run(run())


class TestStreamChat:
    def test_yields_pieces_until_done(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=ndjson(
                {"message": {"role": "assistant", "content": "Hel"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": False},
                {"message": {"role": "assistant", "content": "lo"}, "done": False},
                {"done": True, "eval_count": 2},
            ))

        assert collect(ollama(handler)) == ["Hel", "lo"]
        assert seen == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True, "think": False}

    def test_error_line_raises(self):
        handler = lambda r: httpx.Response(200, content=ndjson(
            {"message": {"content": "a"}, "done": False}, {"error": "out of memory"}))
        with pytest.raises(llm.LLMError, match="out of memory"):
            collect(ollama(handler))

    def test_missing_model_hints_pull(self):
        handler = lambda r: httpx.Response(404, json={"error": "model 'm' not found"})
        with pytest.raises(llm.LLMError, match=r"model 'm' not found.*ollama pull m"):
            collect(ollama(handler))

    def test_unreachable(self):
        def handler(request):
            raise httpx.ConnectError("refused")
        with pytest.raises(llm.LLMError, match="Cannot reach Ollama"):
            collect(ollama(handler))


def hit(pid="2401.00001", content="The entropy is $S = A/4G_N$."):
    return Hit(paper_id=pid, chunk_index=3, title="Black holes", section_path="2 Entropy",
               content=content, arxiv_abs_url=f"https://arxiv.org/abs/{pid}", score=0.03,
               authors="A. One, B. Two, C. Three, D. Four")


class TestPrompt:
    def test_excerpts_in_system_prompt_then_history_then_question(self):
        history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        msgs = rag.build_messages(history, "q2", [hit()])
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert ('[1] "Black holes" by A. One, B. Two, C. Three et al. (arXiv:2401.00001)\n'
                "Section: 2 Entropy\nThe entropy is $S = A/4G_N$.") in msgs[0]["content"]
        assert "Never name or cite any other paper" in msgs[0]["content"]
        assert msgs[-1] == {"role": "user", "content": "q2"}

    def test_no_hits(self):
        assert "No excerpts" in rag.build_messages([], "q", [])[0]["content"]

    def test_sources_numbered_like_excerpts(self):
        src = rag.sources([hit("a"), hit("b", "x " * 500)])
        assert [s["n"] for s in src] == [1, 2] and src[1]["paper_id"] == "b"
        assert len(src[1]["snippet"]) <= 200 and "content" not in src[0]


def test_short_authors():
    assert rag.short_authors("A, B, C") == "A, B, C"
    assert rag.short_authors("A, B, C, D") == "A, B, C et al."
    assert rag.short_authors(None) == rag.short_authors(" ") == "unknown authors"


def test_sse_format():
    assert sse("delta", {"text": "a\nb"}) == 'event: delta\ndata: {"text": "a\\nb"}\n\n'
