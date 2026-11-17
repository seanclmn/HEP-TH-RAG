"""Query-side retrieval: hybrid vector + lexical search over the chunks table.

Documents were embedded by 03_embed.py *without* an instruction. Qwen3-Embedding
queries must carry one, in the form "Instruct: <task>\\nQuery:<question>".
Both sides are L2-normalised, so cosine distance (<=>) is the metric and the
HNSW index (vector_cosine_ops) serves the ORDER BY.

The lexical (BM25) half lives in lexical.py; the two rankings are fused with RRF.

Imported by pipeline/04_search.py and, later, the API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import numpy as np

from lexical import lexical_candidates

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIM = 1024
DEFAULT_INSTRUCTION = (
    "Given a question about theoretical high-energy physics, "
    "retrieve passages from arXiv papers that answer the question"
)
CANDIDATE_FACTOR = 5   # candidates fetched per side = k * this (more depth for RRF)
# HNSW recall on this corpus at pgvector's default ef_search=40 is poor (recall@50 mean 0.67,
# worst 0.08 over 12 test queries); ef=200 gives 0.97 / 0.88 at ~3 ms. ef = max(EF_MIN, 2n).
HNSW_EF_MIN = 200
HNSW_EF_MAX = 1000     # pgvector's maximum ef_search
RRF_K = 60             # standard reciprocal-rank-fusion constant

T = TypeVar("T")


def database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("DATABASE_URL not set and not found in .env")


def connect(url: str | None = None):
    """Open a psycopg connection with the pgvector type adapter registered."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(url or database_url())
    register_vector(conn)
    return conn


def query_prompt(instruction: str = DEFAULT_INSTRUCTION) -> str:
    """The prefix Qwen3-Embedding expects before a query (no space after 'Query:')."""
    instruction = " ".join(instruction.split())
    if not instruction:
        raise ValueError("instruction must be non-empty")
    return f"Instruct: {instruction}\nQuery:"


class QueryEncoder:
    """Loads the embedding model once and encodes questions with the query prompt."""

    def __init__(self, model_name: str = MODEL, device: str | None = None,
                 instruction: str = DEFAULT_INSTRUCTION):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.prompt = query_prompt(instruction)
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, queries: str | Sequence[str]) -> np.ndarray:
        """One query -> shape (DIM,); a list of queries -> shape (n, DIM). Unit norm."""
        single = isinstance(queries, str)
        batch = [queries] if single else list(queries)
        if not batch or any(not q.strip() for q in batch):
            raise ValueError("queries must be non-empty strings")
        emb = self.model.encode(
            [q.strip() for q in batch], prompt=self.prompt, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        # The model runs in bfloat16, so its own normalisation is only good to ~0.4%.
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0] if single else emb


@dataclass(frozen=True)
class Hit:
    paper_id: str
    chunk_index: int
    title: str
    section_path: str
    content: str
    arxiv_abs_url: str | None
    score: float                     # fused RRF score; the ranking key
    similarity: float | None = None  # cosine similarity to the query (vector/hybrid)
    bm25: float | None = None        # lexical score, if the chunk was a lexical candidate


@dataclass(frozen=True)
class _Candidate:
    id: int
    paper_id: str
    score: float


VECTOR_SQL = """
SELECT id, paper_id FROM chunks
ORDER BY embedding <=> %(q)s
LIMIT %(n)s
"""

FETCH_SQL = """
SELECT c.id, c.paper_id, c.chunk_index, p.title, c.section_path, c.content, p.arxiv_abs_url,
       {similarity}
FROM chunks c JOIN papers p USING (paper_id)
WHERE c.id = ANY(%(ids)s)
"""


def limit_per_paper(hits: Iterable[T], n: int) -> list[T]:
    """Keep at most n hits per paper (anything with .paper_id), preserving rank order."""
    if n < 1:
        raise ValueError("per-paper limit must be >= 1")
    seen: dict[str, int] = {}
    out = []
    for h in hits:
        if seen.get(h.paper_id, 0) < n:
            seen[h.paper_id] = seen.get(h.paper_id, 0) + 1
            out.append(h)
    return out


def rrf(rankings: Sequence[Sequence[tuple[int, str]]], k: int = RRF_K) -> list[_Candidate]:
    """Reciprocal rank fusion of ranked (id, paper_id) lists: score = sum 1/(k + rank).

    Uses only ranks, so cosine similarities and BM25 scores never need calibrating.
    Ties break by id for determinism.
    """
    score: dict[int, float] = {}
    paper: dict[int, str] = {}
    for ranking in rankings:
        for rank, (cid, pid) in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            paper[cid] = pid
    return [_Candidate(c, paper[c], s) for c, s in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


def _check_embedding(embedding) -> np.ndarray:
    q = np.asarray(embedding, dtype=np.float32)
    if q.shape != (DIM,):
        raise ValueError(f"expected embedding of shape ({DIM},), got {q.shape}")
    if not np.all(np.isfinite(q)) or not np.any(q):
        raise ValueError("embedding must be finite and non-zero")
    return q


def search(conn, embedding: np.ndarray | None = None, k: int = 10, per_paper: int | None = None,
           *, query: str | None = None) -> list[Hit]:
    """Top-k chunks. Pass `embedding` for vector search, `query` for lexical (BM25) search,
    or both for hybrid search (the two rankings fused with RRF).

    Each side fetches k * CANDIDATE_FACTOR candidates. If a per_paper cap leaves fewer
    than k hits, the pool is doubled (up to HNSW_EF_MAX) until k hits are found or
    the candidates run out.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if per_paper is not None and per_paper < 1:
        raise ValueError("per_paper must be >= 1")
    if embedding is None and query is None:
        raise ValueError("pass an embedding, a query, or both")
    if query is not None and not query.strip():
        raise ValueError("query must be non-empty")
    q = _check_embedding(embedding) if embedding is not None else None

    n = min(k * CANDIDATE_FACTOR, HNSW_EF_MAX)
    while True:
        with conn.transaction(), conn.cursor() as cur:
            vec: list = []
            if q is not None:
                # HNSW returns at most ef_search rows, and recall needs headroom beyond n.
                ef = min(max(HNSW_EF_MIN, 2 * n), HNSW_EF_MAX)
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(ef)])
                vec = cur.execute(VECTOR_SQL, {"q": q, "n": n}).fetchall()
            lex = lexical_candidates(conn, query, n) if query is not None else []
        fused = rrf([[(r[0], r[1]) for r in vec], [(r[0], r[1]) for r in lex]])
        picked = (limit_per_paper(fused, per_paper) if per_paper else fused)[:k]
        exhausted = len(vec) < n and len(lex) < n
        if len(picked) >= k or exhausted or n >= HNSW_EF_MAX:
            break
        n = min(n * 2, HNSW_EF_MAX)

    if not picked:
        return []
    bm25 = {r[0]: float(r[2]) for r in lex}
    sim_sql = "1 - (c.embedding <=> %(q)s)" if q is not None else "NULL::float8"
    rows = conn.execute(FETCH_SQL.format(similarity=sim_sql),
                        {"ids": [c.id for c in picked], "q": q}).fetchall()
    by_id = {r[0]: r for r in rows}
    hits = []
    for c in picked:
        _, pid, idx, title, path, content, url, sim = by_id[c.id]
        hits.append(Hit(pid, idx, " ".join(title.split()), path, content, url, c.score,
                        None if sim is None else float(sim), bm25.get(c.id)))
    return hits


def snippet(text: str, width: int = 300) -> str:
    """Whitespace-collapsed text, cut at a word boundary to at most `width` chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    cut = text[: width - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"
