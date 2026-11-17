"""Pure-Python tests: no database, no model."""

import numpy as np
import pytest

from retrieval import DIM, Hit, limit_per_paper, query_prompt, rrf, search, snippet


def hit(pid: str, idx: int, score: float = 0.5) -> Hit:
    return Hit(pid, idx, "T", "S", "c", None, score)


class TestQueryPrompt:
    def test_matches_qwen3_format(self):
        # Qwen3-Embedding reference format: f"Instruct: {task}\nQuery:{query}"
        assert query_prompt("Find papers") == "Instruct: Find papers\nQuery:"

    def test_collapses_whitespace(self):
        assert query_prompt("  Find\n  papers ") == "Instruct: Find papers\nQuery:"

    def test_default_is_well_formed(self):
        p = query_prompt()
        assert p.startswith("Instruct: ") and p.endswith("\nQuery:") and p.count("\n") == 1

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            query_prompt("   ")


class TestLimitPerPaper:
    def test_caps_each_paper_and_preserves_order(self):
        hits = [hit("a", 0), hit("a", 1), hit("b", 0), hit("a", 2), hit("b", 1), hit("c", 0)]
        out = limit_per_paper(hits, 1)
        assert [(h.paper_id, h.chunk_index) for h in out] == [("a", 0), ("b", 0), ("c", 0)]

    def test_limit_two(self):
        hits = [hit("a", 0), hit("a", 1), hit("a", 2), hit("b", 0)]
        out = limit_per_paper(hits, 2)
        assert [(h.paper_id, h.chunk_index) for h in out] == [("a", 0), ("a", 1), ("b", 0)]

    def test_empty(self):
        assert limit_per_paper([], 3) == []

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            limit_per_paper([hit("a", 0)], 0)


class TestRRF:
    def test_single_ranking_keeps_order(self):
        out = rrf([[(3, "a"), (1, "b"), (2, "c")]])
        assert [c.id for c in out] == [3, 1, 2]
        assert out[0].score == pytest.approx(1 / 61)

    def test_agreement_beats_one_strong_signal(self):
        # 2 is ranked 2nd by both lists; 1 and 3 are each 1st in one list but absent in the other.
        out = rrf([[(1, "a"), (2, "b")], [(3, "c"), (2, "b")]])
        assert out[0].id == 2
        assert out[0].score == pytest.approx(2 / 62)

    def test_ties_break_by_id(self):
        assert [c.id for c in rrf([[(5, "a")], [(4, "b")]])] == [4, 5]

    def test_carries_paper_id_and_handles_empty(self):
        assert rrf([[], [(7, "p")]])[0].paper_id == "p"
        assert rrf([[], []]) == []


class TestSnippet:
    def test_short_text_unchanged_except_whitespace(self):
        assert snippet("a  b\n\nc", 100) == "a b c"

    def test_cuts_at_word_boundary(self):
        out = snippet("alpha beta gamma delta", 12)
        assert out == "alpha beta…"
        assert len(out) <= 12

    def test_never_exceeds_width(self):
        text = "word " * 200
        for w in (5, 17, 80, 300):
            assert len(snippet(text, w)) <= w

    def test_single_long_word_is_hard_cut(self):
        assert snippet("x" * 50, 10) == "x" * 9 + "…"


class TestSearchValidation:
    """Argument checks happen before the connection is touched, so conn=None is fine."""

    def good(self):
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = 1
        return v

    @pytest.mark.parametrize("k", [0, -1])
    def test_bad_k(self, k):
        with pytest.raises(ValueError, match="k must"):
            search(None, self.good(), k=k)

    def test_bad_per_paper(self):
        with pytest.raises(ValueError, match="per_paper"):
            search(None, self.good(), per_paper=0)

    @pytest.mark.parametrize("shape", [(DIM - 1,), (1, DIM), (3,)])
    def test_wrong_shape(self, shape):
        with pytest.raises(ValueError, match="shape"):
            search(None, np.ones(shape, dtype=np.float32))

    def test_zero_vector(self):
        with pytest.raises(ValueError, match="non-zero"):
            search(None, np.zeros(DIM, dtype=np.float32))

    def test_needs_embedding_or_query(self):
        with pytest.raises(ValueError, match="embedding, a query"):
            search(None)

    def test_blank_query(self):
        with pytest.raises(ValueError, match="query"):
            search(None, query="   ")

    def test_nan(self):
        v = self.good()
        v[1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            search(None, v)
