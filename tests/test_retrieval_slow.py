"""Tests that load the real Qwen3 model. Run with: .venv/bin/python -m pytest -m slow"""

import numpy as np
import pytest

from retrieval import DIM, QueryEncoder, connect, search

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def encoder():
    return QueryEncoder()


def test_query_embedding_shape_and_norm(encoder):
    v = encoder.encode("What is the AdS/CFT correspondence?")
    assert v.shape == (DIM,) and v.dtype == np.float32
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


def test_batch_matches_single(encoder):
    qs = ["black hole entropy", "string compactification"]
    batch = encoder.encode(qs)
    assert batch.shape == (2, DIM)
    for q, row in zip(qs, batch):
        assert float(encoder.encode(q) @ row) == pytest.approx(1.0, abs=1e-3)


def test_instruction_is_applied(encoder):
    raw = encoder.model.encode(["black hole entropy"], normalize_embeddings=True)[0]
    assert float(raw @ encoder.encode("black hole entropy")) < 0.999


def test_rejects_blank(encoder):
    with pytest.raises(ValueError):
        encoder.encode("  ")


@pytest.mark.db
def test_title_query_retrieves_its_own_paper(encoder):
    """End-to-end on the real corpus: querying by a paper's title should surface it.

    Checks that query-side embedding is consistent with how 03_embed.py embedded
    documents (same model, normalisation, prompt convention).
    """
    try:
        conn = connect()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    with conn:
        papers = conn.execute(
            "SELECT paper_id, title FROM papers ORDER BY md5(paper_id) LIMIT 10").fetchall()
        if not papers:
            pytest.skip("no papers loaded")
        found = sum(
            pid in {h.paper_id for h in search(conn, encoder.encode(title), k=10, per_paper=1)}
            for pid, title in papers
        )
    assert found >= 8, f"only {found}/10 papers retrieved by their own title"


@pytest.mark.db
def test_hybrid_finds_syk_hamiltonian_by_formula(encoder):
    """The case that motivated hybrid search: vector search alone missed this paper."""
    try:
        conn = connect()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    q = r"H = \sum_{ijkl} J_{ijkl} \chi_i \chi_j \chi_k \chi_l"
    with conn:
        if not conn.execute("SELECT 1 FROM papers WHERE paper_id = '2607.28486'").fetchone():
            pytest.skip("SYK paper 2607.28486 not in corpus")
        hits = search(conn, encoder.encode(q), k=5, per_paper=1, query=q)
    assert "2607.28486" in [h.paper_id for h in hits[:3]]
