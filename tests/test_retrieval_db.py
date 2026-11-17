"""search() against real Postgres + pgvector, isolated from the real data.

Each test runs inside a transaction that creates a throwaway schema, applies
db/init/02-schema.sql there, inserts synthetic papers/chunks with hand-built
vectors, and is rolled back at the end, so nothing is left behind.
"""

import uuid

import numpy as np
import pytest

from lexical import index_lexical
from retrieval import DIM, ROOT, connect, search

pytestmark = pytest.mark.db


def unit(*pairs: tuple[int, float]) -> np.ndarray:
    """Unit vector with the given (axis, weight) components."""
    v = np.zeros(DIM, dtype=np.float32)
    for axis, w in pairs:
        v[axis] = w
    return v / np.linalg.norm(v)


@pytest.fixture
def conn():
    try:
        c = connect()
    except Exception as e:  # container down, bad .env, ...
        pytest.skip(f"database unavailable: {e}")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    c.execute(f"CREATE SCHEMA {schema}")
    c.execute(f"SET LOCAL search_path TO {schema}, public")
    c.execute((ROOT / "db/init/02-schema.sql").read_text())
    try:
        yield c
    finally:
        c.rollback()
        c.close()


def insert(conn, papers: dict[str, str], chunks: list[tuple], texts: dict | None = None):
    """chunks: (paper_id, chunk_index, vector); texts: optional {(paper_id, idx): content}."""
    texts = texts or {}
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO papers (paper_id, title, arxiv_abs_url) VALUES (%s, %s, %s)",
            [(pid, title, f"https://arxiv.org/abs/{pid}") for pid, title in papers.items()],
        )
        cur.executemany(
            """INSERT INTO chunks (paper_id, chunk_index, section_path, content, n_tokens, content_sha, embedding)
               VALUES (%s, %s, %s, %s, 1, 'x', %s)""",
            [(pid, idx, f"Sec {idx}", texts.get((pid, idx), f"{pid} chunk {idx}"), v)
             for pid, idx, v in chunks],
        )
    index_lexical(conn)


@pytest.fixture
def corpus(conn):
    # Query will be axis 0. Expected similarity order: A0 (1.0) > A1 > B0 > A2 > C0 (0.0).
    insert(
        conn,
        {"2401.00001": "Paper   A\n title", "2401.00002": "Paper B", "2401.00003": "Paper C"},
        [
            ("2401.00001", 0, unit((0, 1.0))),
            ("2401.00001", 1, unit((0, 1.0), (1, 0.3))),
            ("2401.00002", 0, unit((0, 1.0), (1, 0.6))),
            ("2401.00001", 2, unit((0, 1.0), (1, 1.0))),
            ("2401.00003", 0, unit((2, 1.0))),
        ],
    )
    return conn


def ids(hits):
    return [(h.paper_id[-1], h.chunk_index) for h in hits]


def test_orders_by_cosine_similarity(corpus):
    hits = search(corpus, unit((0, 1.0)), k=10)
    assert ids(hits) == [("1", 0), ("1", 1), ("2", 0), ("1", 2), ("3", 0)]
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    assert sims[0] == pytest.approx(1.0, abs=1e-5)
    assert sims[-1] == pytest.approx(0.0, abs=1e-5)
    assert sims[1] == pytest.approx(float(unit((0, 1.0)) @ unit((0, 1.0), (1, 0.3))), abs=1e-5)


def test_respects_k(corpus):
    assert ids(search(corpus, unit((0, 1.0)), k=2)) == [("1", 0), ("1", 1)]


def test_joins_paper_metadata(corpus):
    top = search(corpus, unit((0, 1.0)), k=1)[0]
    assert top.title == "Paper A title"  # whitespace collapsed
    assert top.arxiv_abs_url == "https://arxiv.org/abs/2401.00001"
    assert top.section_path == "Sec 0"
    assert top.content == "2401.00001 chunk 0"


def test_per_paper_diversifies(corpus):
    hits = search(corpus, unit((0, 1.0)), k=3, per_paper=1)
    assert ids(hits) == [("1", 0), ("2", 0), ("3", 0)]


def test_per_paper_two(corpus):
    hits = search(corpus, unit((0, 1.0)), k=10, per_paper=2)
    assert ids(hits) == [("1", 0), ("1", 1), ("2", 0), ("3", 0)]


def test_unnormalised_query_gives_same_ranking(corpus):
    # Cosine distance is scale-invariant.
    assert ids(search(corpus, 7.5 * unit((0, 1.0)), k=10)) == ids(search(corpus, unit((0, 1.0)), k=10))


def test_empty_table(conn):
    assert search(conn, unit((0, 1.0)), k=5) == []


def test_k_beyond_default_ef_search(conn):
    # HNSW returns at most hnsw.ef_search rows (default 40); search() must raise it.
    n = 120
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(n, DIM)).astype(np.float32)
    insert(conn, {"2401.09999": "Big"}, [("2401.09999", i, vecs[i]) for i in range(n)])
    conn.execute("SET LOCAL enable_seqscan = off")  # force the index path
    hits = search(conn, vecs[0], k=100)
    assert len(hits) == 100
    assert hits[0].chunk_index == 0


def test_uses_hnsw_index(conn):
    # Guard against a query rewrite that silently falls back to a sequential scan.
    from retrieval import VECTOR_SQL

    conn.execute("SET LOCAL enable_seqscan = off")
    plan = "\n".join(r[0] for r in conn.execute(
        "EXPLAIN " + VECTOR_SQL, {"q": unit((0, 1.0)), "n": 10}).fetchall())
    assert "chunks_embedding_hnsw" in plan, plan


def test_per_paper_widens_candidate_pool(conn):
    # Paper A owns the 30 nearest chunks; with k=3, per_paper=1 the first pool (15) holds
    # only A, so search() must widen it to reach papers B and C.
    q = unit((0, 1.0))
    chunks = [("2401.00001", i, unit((0, 1.0), (1 + i, 0.01 * (i + 1)))) for i in range(30)]
    chunks += [("2401.00002", 0, unit((0, 1.0), (200, 1.0))), ("2401.00003", 0, unit((0, 1.0), (201, 1.5)))]
    insert(conn, {"2401.00001": "A", "2401.00002": "B", "2401.00003": "C"}, chunks)
    hits = search(conn, q, k=3, per_paper=1)
    assert [h.paper_id[-1] for h in hits] == ["1", "2", "3"]


# ---------------------------------------------------------------- lexical + hybrid

SYK = r"The SYK Hamiltonian \begin{equation} H = \sum_{ijkl} J_{ijkl}\,\chi_{i}\chi_{j}\chi_{k}\chi_{l} \end{equation}"
COMMUTATOR = r"We need the constraints $[H, J^{z-}]$ and $[H, J^{\bar z-}]$ for the Hamiltonian H."
ISING = r"The Ising Hamiltonian is $H = -J \sum_{i} \sigma_i \sigma_{i+1}$ with coupling J."
BH = r"The Bekenstein--Hawking entropy is $S = \dfrac{A}{4G_{\rm N}}$ for a black hole."


@pytest.fixture
def latex_corpus(conn):
    # Vectors are deliberately unhelpful: COMMUTATOR is the vector-nearest chunk.
    insert(
        conn,
        {"2401.00011": "Hamiltonian SYK", "2401.00012": "Higher spin", "2401.00013": "Spin chains",
         "2401.00014": "Black holes"},
        [("2401.00011", 0, unit((0, 1.0), (1, 2.0))), ("2401.00012", 0, unit((0, 1.0))),
         ("2401.00013", 0, unit((0, 1.0), (2, 1.0))), ("2401.00014", 0, unit((3, 1.0)))],
        {("2401.00011", 0): SYK, ("2401.00012", 0): COMMUTATOR, ("2401.00013", 0): ISING,
         ("2401.00014", 0): BH},
    )
    return conn


def test_lexical_finds_formula_despite_notation(latex_corpus):
    hits = search(latex_corpus, query=r"H = \sum J_{ijkl} \chi_i \chi_j \chi_k \chi_l", k=4)
    assert hits[0].paper_id == "2401.00011"
    assert hits[0].bm25 is not None and hits[0].similarity is None
    assert [h.bm25 for h in hits] == sorted([h.bm25 for h in hits], reverse=True)


def test_lexical_matches_normalised_variant(latex_corpus):
    hits = search(latex_corpus, query=r"S = \frac{A}{4 G_N}", k=1)
    assert hits[0].paper_id == "2401.00014"


def test_lexical_no_matching_terms(latex_corpus):
    assert search(latex_corpus, query="supersymmetric localization", k=5) == []


def test_hybrid_surfaces_lexical_match_vector_missed(latex_corpus):
    q = r"\chi_i \chi_j \chi_k \chi_l"
    vector_only = search(latex_corpus, unit((0, 1.0)), k=1)
    assert vector_only[0].paper_id == "2401.00012"           # vector alone picks the commutator
    hybrid = search(latex_corpus, unit((0, 1.0)), k=4, query=q)
    assert hybrid[0].paper_id == "2401.00011"                 # BM25 + decent vector rank wins
    assert hybrid[0].similarity is not None and hybrid[0].bm25 is not None
    assert {h.paper_id for h in hybrid} == {"2401.00011", "2401.00012", "2401.00013", "2401.00014"}
    assert [h.score for h in hybrid] == sorted([h.score for h in hybrid], reverse=True)


def test_index_lexical_only_fills_missing_and_tracks_stats(latex_corpus):
    assert index_lexical(latex_corpus) == 0                    # already indexed by insert()
    n, avg = latex_corpus.execute("SELECT n_docs, avg_len FROM lex_corpus").fetchone()
    assert n == 4 and avg > 0
    df, = latex_corpus.execute(r"SELECT df FROM lex_terms WHERE term = 'hamiltonian'").fetchone()
    assert df == 3                                             # SYK, COMMUTATOR, ISING (BH lacks it)
    latex_corpus.execute("UPDATE chunks SET lex = NULL WHERE paper_id = '2401.00014'")
    assert index_lexical(latex_corpus) == 1
    assert index_lexical(latex_corpus, rebuild=True) == 4


def test_tsvector_literal_round_trips(conn):
    from lexical import tsvector_literal
    lit = tsvector_literal([r"\chi_i", "it's", r"\chi_i \chi_j", r"\chi_i"])
    rows = dict(conn.execute("SELECT lexeme, cardinality(positions) FROM unnest(%s::tsvector)", [lit]).fetchall())
    assert rows == {r"\chi_i": 2, "it's": 1, r"\chi_i \chi_j": 1}
