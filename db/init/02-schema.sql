-- Papers and their embedded chunks. Idempotent: also applied by pipeline/03_embed.py,
-- since docker-entrypoint-initdb.d only runs on a fresh volume.
-- Embeddings: Qwen/Qwen3-Embedding-0.6B, 1024 dims, L2-normalised (use cosine / <=>).
-- Documents are embedded without an instruction; queries must use the model's
-- "query" prompt ("Instruct: ...\nQuery: ...").

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    paper_id            text PRIMARY KEY,
    title               text NOT NULL,
    authors             text,
    abstract            text,
    categories          text,
    primary_category    text,
    first_version_date  timestamp,
    latest_version_date timestamp,
    doi                 text,
    journal_ref         text,
    arxiv_abs_url       text
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    paper_id     text NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    chunk_index  int  NOT NULL,
    section_path text NOT NULL,
    content      text NOT NULL,
    n_tokens     int  NOT NULL,
    content_sha  text NOT NULL,   -- sha256 of the text that was embedded (header + content)
    embedding    vector(1024) NOT NULL,
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
