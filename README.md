# HEP-TH RAG

Question answering over arXiv **hep-th** papers, searching their full LaTeX source.
Postgres + pgvector for storage, Qwen3-Embedding for vectors. Generation (LLM answers),
the API (`api/`) and the web UI (`web/`) are not built yet.

## Usage

```sh
cp .env.example .env && docker compose up -d   # Postgres + pgvector
./scripts/test-db.sh                           # smoke test the database

.venv/bin/python pipeline/01b_download_latest.py   # newest 100 hep-th papers (01_download.py: all)
./scripts/reembed.sh                               # chunk -> embed -> lexical index

.venv/bin/python pipeline/04_search.py "your question"     # --mode hybrid|vector|lexical
.venv/bin/python -m pytest                                 # add -m slow for model tests
```

Every step is safe to re-run. `03_embed.py` only re-embeds chunks whose text changed.

## Design decisions

**Raw LaTeX, not PDF text.** In hep-th the math carries most of the meaning, and PDF
extraction mangles it. Keeping the source makes formulas searchable.

**Chunking (`02_chunk.py`).**
- Each paper's macros are expanded with that paper's own definitions (`macros.py`).
  A shared macro table is impossible because names clash between papers: `\be` is
  `\begin{equation}` in most papers but `\mathbf{e}` in others.
- Chunks are ~450 tokens, split along `\section`s, with math kept as LaTeX.

**Embeddings: Qwen3-Embedding-0.6B (1024 dimensions, runs locally).**
- Documents are embedded with no instruction. Queries **must** use the
  `Instruct: …\nQuery:` prefix (`retrieval.query_prompt`).
- The model runs in bfloat16, so query vectors are renormalised in float32.

**Hybrid search: vector + BM25, merged with reciprocal rank fusion (RRF).**
- **Why:** vector search finds a formula's topic but not its structure. On its own it
  never found the SYK Hamiltonian `\sum J_{ijkl}\chi_i\chi_j…`; with lexical search,
  that paper ranks #1.
- **Lexical side (`lexical.py`):** chunks and queries share one LaTeX normaliser, so
  `G_{\rm N}` = `G_N` and `\dfrac` = `\frac`.
  - It uses its own tokenizer, because Postgres' splits `\chi_i` into `chi` + `i`.
  - It ranks with BM25, because `ts_rank` has no IDF.
- **RRF** combines rankings only, so the two score scales never need calibrating.
- **Rejected:** formula-tree search (as in Approach0). It's a large build, and hybrid
  search covers exact and near-exact notation.

**Vector index.**
- HNSW with `ef_search ≥ 200`. At pgvector's default of 40, recall was poor: 0.67 on
  average, 0.08 at worst.
- When benchmarking, set `conn.prepare_threshold = None`. Otherwise psycopg reuses the
  first query plan, which can silently turn approximate search into exact search.

**Storage.**
- `content_sha` lets re-runs re-embed only the chunks that changed.
- The schema must stay idempotent, because `03_embed.py` re-applies it on every run.
- After changing `lexical.py`, run `03b_index_lexical.py --rebuild`.

## Known issues

- **Database is out of date.** It still holds chunks built without macro expansion.
  Run `./scripts/reembed.sh` (about 5.7k of 10.8k chunks will be re-embedded).
- **LaTeX noise.** TikZ and equation-heavy chunks can rank highly. They should be
  stripped in the chunker.
- **No relevance cutoff.** Off-topic queries still return results, with scores close
  to good hits.
- **No question benchmark yet.** Quality has been checked with spot queries and
  title/equation lookups only.
- **No dependency file.** The venv is managed by `uv`.
