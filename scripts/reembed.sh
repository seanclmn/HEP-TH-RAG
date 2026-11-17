#!/usr/bin/env bash
# Re-chunk, re-embed and re-index the corpus. Run after changing the chunker
# (02_chunk.py / macros.py), the embedded text format, or the lexical tokenizer.
#
# Usage: ./scripts/reembed.sh [data/raw/hepth_latest_100.parquet]
#
# 1. 02_chunk.py              re-chunks the raw papers (overwrites the chunks parquet)
# 2. 03_embed.py              re-embeds only chunks whose text changed (content_sha)
#                             and deletes chunk rows that no longer exist
# 3. 03b_index_lexical.py     rebuilds the BM25 index for every chunk, so all of them use
#                             the current tokenizer (~12 s per 10k chunks)
#
# Safe to re-run: if it is interrupted, run it again and finished work is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

raw="${1:-data/raw/hepth_latest_100.parquet}"
name="$(basename "$raw" .parquet)"
chunks="data/processed/chunks_${name#hepth_}.parquet"   # naming used by 02_chunk.py
py=.venv/bin/python

[ -f "$raw" ] || { echo "Missing $raw (run pipeline/01_download.py or 01b_download_latest.py)"; exit 1; }
[ -x "$py" ] || { echo "Missing $py (create the venv first)"; exit 1; }
status=$(docker inspect -f '{{.State.Health.Status}}' hep-th-rag-db 2>/dev/null || echo missing)
[ "$status" = healthy ] || { echo "Database is $status; start it with: docker compose up -d"; exit 1; }

step() { echo; echo "==> $*"; }
t0=$SECONDS

step "chunking $raw"
"$py" pipeline/02_chunk.py "$raw"

step "embedding $chunks"
"$py" pipeline/03_embed.py "$chunks" "$raw"

step "rebuilding lexical index"
"$py" pipeline/03b_index_lexical.py --rebuild

echo; echo "done in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
