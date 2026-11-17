"""Embed chunks with Qwen3-Embedding-0.6B and load papers + chunks into pgvector.

Each chunk is embedded as
    Title: <paper title>
    Section: <section path>

    <content>
with no instruction (Qwen3 documents are embedded raw; queries use the "query" prompt).
Embeddings are L2-normalised, 1024-dim.

Resumable and idempotent: chunks already in the DB with the same content hash are
skipped, changed ones are re-embedded, and stale chunk rows for a paper are deleted.
Work is committed every BATCH chunks.
Finally, chunks without a lexical index entry (new or changed) are indexed for BM25
(see lexical.py).

Usage:  .venv/bin/python pipeline/03_embed.py [data/processed/chunks_latest_100.parquet [data/raw/hepth_latest_100.parquet]]
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import duckdb
import psycopg
import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from lexical import index_lexical

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH = 256  # chunks per encode + commit
ENCODE_BATCH = 16  # model batch size
MAX_SEQ = 1024  # chunks are <= ~530 tokens + short header


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("DATABASE_URL not set and not found in .env")


def main() -> int:
    chunks_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "data/processed/chunks_latest_100.parquet"
    )
    papers_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "data/raw/hepth_latest_100.parquet"
    )

    duck = duckdb.connect()
    papers = duck.execute(
        """SELECT paper_id, title, authors, abstract, categories, primary_category,
                  first_version_date, latest_version_date, doi, journal_ref, arxiv_abs_url
           FROM read_parquet(?) WHERE paper_id IN (SELECT paper_id FROM read_parquet(?))""",
        [str(papers_path), str(chunks_path)],
    ).fetchall()
    chunks = duck.execute(
        """SELECT c.paper_id, c.chunk_index, c.section_path, c.content, c.n_tokens, p.title
           FROM read_parquet(?) c JOIN read_parquet(?) p USING (paper_id)
           ORDER BY c.paper_id, c.chunk_index""",
        [str(chunks_path), str(papers_path)],
    ).fetchall()
    rows = []
    for pid, idx, path, content, ntok, title in chunks:
        text = f"Title: {' '.join(title.split())}\nSection: {path}\n\n{content}"
        rows.append(
            (
                pid,
                idx,
                path,
                content,
                ntok,
                hashlib.sha256(text.encode()).hexdigest(),
                text,
            )
        )
    log(f"{len(papers)} papers, {len(rows):,} chunks")

    with psycopg.connect(database_url()) as db:
        db.execute((ROOT / "db/init/02-schema.sql").read_text())
        register_vector(db)
        with db.cursor() as cur:
            cur.executemany(
                """INSERT INTO papers VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (paper_id) DO UPDATE SET
                     title=EXCLUDED.title, authors=EXCLUDED.authors, abstract=EXCLUDED.abstract,
                     categories=EXCLUDED.categories, primary_category=EXCLUDED.primary_category,
                     first_version_date=EXCLUDED.first_version_date,
                     latest_version_date=EXCLUDED.latest_version_date, doi=EXCLUDED.doi,
                     journal_ref=EXCLUDED.journal_ref, arxiv_abs_url=EXCLUDED.arxiv_abs_url""",
                papers,
            )
            # Remove chunk rows beyond each paper's current chunk count (chunking changed).
            counts = {}
            for r in rows:
                counts[r[0]] = counts.get(r[0], 0) + 1
            cur.executemany(
                "DELETE FROM chunks WHERE paper_id = %s AND chunk_index >= %s",
                list(counts.items()),
            )
            existing = set(
                cur.execute(
                    "SELECT paper_id, chunk_index, content_sha FROM chunks"
                ).fetchall()
            )
        db.commit()

        todo = [r for r in rows if (r[0], r[1], r[5]) not in existing]
        log(
            f"{len(rows) - len(todo):,} chunks already embedded, {len(todo):,} to embed"
        )
        if todo:
            embed(db, todo)
        log(f"lexical index: {index_lexical(db):,} chunks indexed")
        db.commit()
        n = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        log(f"done: {n:,} chunks in DB")
    return 0


def embed(db, todo: list[tuple]) -> None:
    """Embed chunks (rows built in main) and upsert them, committing every BATCH."""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SentenceTransformer(MODEL, device=device)
    model.max_seq_length = MAX_SEQ
    log(f"loaded {MODEL} on {device}")

    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        batch = todo[b : b + BATCH]
        emb = model.encode(
            [r[6] for r in batch],
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        with db.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (paper_id, chunk_index, section_path, content, n_tokens, content_sha, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (paper_id, chunk_index) DO UPDATE SET
                     section_path=EXCLUDED.section_path, content=EXCLUDED.content,
                     n_tokens=EXCLUDED.n_tokens, content_sha=EXCLUDED.content_sha,
                     embedding=EXCLUDED.embedding, lex=NULL, lex_len=NULL""",
                [(*r[:6], e) for r, e in zip(batch, emb)],
            )
        db.commit()
        done = b + len(batch)
        rate = done / (time.time() - t0)
        log(
            f"{done:,}/{len(todo):,} chunks ({rate:.1f}/s, ~{(len(todo) - done) / rate / 60:.0f} min left)"
        )


if __name__ == "__main__":
    sys.exit(main())
