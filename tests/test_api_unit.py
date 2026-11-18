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
        assert len(src[1]["snippet"]) <= 281 and "content" not in src[0]


class TestMathSnippet:
    def test_equation_environments_become_math(self):
        out = rag.math_snippet(r"so \begin{equation} E = mc^2 \label{eq:e} \end{equation} holds")
        assert out == "so $E = mc^2$ holds"
        out = rag.math_snippet(r"\begin{align*} a &= b \\ c &= d \end{align*}")
        assert out == r"$\begin{aligned}a &= b \\ c &= d\end{aligned}$"
        assert rag.math_snippet(r"\[ x \] and \( y \)") == "$x$ and $y$"

    def test_never_cuts_inside_a_formula(self):
        text = "word " * 50 + r"$\sum_{i=1}^{N} J_{ijkl}\chi_i\chi_j$ more words " * 5
        for width in range(240, 320, 7):
            out = rag.math_snippet(text, width)
            assert out.endswith("…")
            assert out.count("$") % 2 == 0, (width, out[-60:])
            assert len(out) <= width * 1.5 + 1

    def test_formula_cut_by_the_chunker_is_dropped(self):
        assert rag.math_snippet(r"mu_0$ in a scheme. Then $x$ holds") == "in a scheme. Then $x$ holds"
        assert rag.math_snippet(r"Then $x$ holds and $\sum_i a_i") == "Then $x$ holds and"
        assert rag.math_snippet(r"a = b \end{equation} so $x$") == "so $x$"
        assert rag.math_snippet(r"so $x$ and \begin{align} G = 1") == "so $x$ and"

    def test_unrepairable_falls_back_to_escaped_text(self):
        assert rag.math_snippet(r"a_1 + b$") == r"a_1 + b\$"   # nothing left after the repair
        long_formula = "$" + r"\alpha + " * 100 + "x$"
        out = rag.math_snippet(long_formula, 100)
        assert out.startswith(r"\$\alpha") and out.endswith("…")

    def test_short_text_unchanged(self):
        assert rag.math_snippet("  a\n b  ") == "a b"


def test_short_authors():
    assert rag.short_authors("A, B, C") == "A, B, C"
    assert rag.short_authors("A, B, C, D") == "A, B, C et al."
    assert rag.short_authors(None) == rag.short_authors(" ") == "unknown authors"


def test_sse_format():
    assert sse("delta", {"text": "a\nb"}) == 'event: delta\ndata: {"text": "a\\nb"}\n\n'
