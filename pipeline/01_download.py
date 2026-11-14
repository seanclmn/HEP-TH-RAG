"""Download the hep-th subset of the Hugging Face dataset secemp9/arxiv-complete.

Steps
  1. Download the `metadata` file (~1.6 GB), keep every paper whose space-separated
     `categories` contains "hep-th" (cross-lists included), and write
     data/raw/hepth_metadata.parquet. The full metadata file is deleted afterwards.
  2. For each of the 50 `paper_text` shards (~1.4 GB each): download it, keep only
     rows whose paper_id is in the hep-th set, write data/raw/text/hepth_text-NNNNN.parquet,
     then delete the downloaded shard. Peak disk use is roughly one shard.

Re-running is safe: finished outputs are skipped, and outputs are written to a temp
file and renamed, so an interrupted shard is redone from scratch.

Usage:  .venv/bin/python pipeline/01_download.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "secemp9/arxiv-complete"
CATEGORY = "hep-th"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TEXT_DIR = RAW / "text"
DL_DIR = RAW / "_download"  # temporary home for full-size downloads
META_OUT = RAW / "hepth_metadata.parquet"

MAX_ATTEMPTS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sql_path(p: Path) -> str:
    return str(p).replace("'", "''")


def download(filename: str) -> Path:
    """Download one repo file into DL_DIR, retrying with backoff on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=DL_DIR,
                )
            )
        except Exception as e:  # network errors, 429s, 5xx
            if attempt == MAX_ATTEMPTS:
                raise
            wait = min(60 * 2 ** (attempt - 1), 900)
            log(f"  download failed ({type(e).__name__}: {e}); retry {attempt}/{MAX_ATTEMPTS - 1} in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    spill = RAW / "_duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{sql_path(spill)}'")
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET preserve_insertion_order = false")
    return con


def list_files(prefix: str) -> list[str]:
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix + "/") and f.endswith(".parquet"))


def build_metadata(con: duckdb.DuckDBPyConnection) -> None:
    if META_OUT.exists():
        log(f"metadata: {META_OUT.name} exists, skipping")
        return
    (meta_file,) = list_files("metadata")
    log(f"metadata: downloading {meta_file}")
    src = download(meta_file)
    tmp = META_OUT.with_suffix(".parquet.tmp")
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{sql_path(src)}')
            WHERE list_contains(string_split(categories, ' '), '{CATEGORY}')
            ORDER BY paper_id
        ) TO '{sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    tmp.replace(META_OUT)
    src.unlink()
    total, primary = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE primary_category = '{CATEGORY}') "
        f"FROM read_parquet('{sql_path(META_OUT)}')"
    ).fetchone()
    log(f"metadata: {total:,} {CATEGORY} papers ({primary:,} primary) -> {META_OUT.relative_to(ROOT)}")


def filter_shard(con: duckdb.DuckDBPyConnection, shard_file: str, index: int) -> None:
    out = TEXT_DIR / f"hepth_text-{index:05d}.parquet"
    if out.exists():
        log(f"shard {index:02d}: done already, skipping")
        return
    t0 = time.time()
    log(f"shard {index:02d}: downloading {shard_file}")
    src = download(shard_file)
    size_gb = src.stat().st_size / 1e9
    tmp = out.with_suffix(".parquet.tmp")
    con.execute(
        f"""
        COPY (
            SELECT t.*, length(t.text) AS n_chars
            FROM read_parquet('{sql_path(src)}') AS t
            SEMI JOIN hepth_ids USING (paper_id)
        ) TO '{sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    tmp.replace(out)  # written even when empty, so every shard leaves a marker
    src.unlink()
    (kept,) = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(out)}')").fetchone()
    log(f"shard {index:02d}: kept {kept:,} rows from {size_gb:.2f} GB in {time.time() - t0:.0f}s")


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        try:
            HfApi().whoami()
        except Exception:
            log("warning: not logged in to Hugging Face; downloads may be rate-limited (run `hf auth login`)")

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in TEXT_DIR.glob("*.tmp"):
        stale.unlink()

    con = connect()
    build_metadata(con)
    con.execute(
        f"CREATE TEMP TABLE hepth_ids AS SELECT paper_id FROM read_parquet('{sql_path(META_OUT)}')"
    )

    shards = list_files("paper_text")
    log(f"paper_text: {len(shards)} shards")
    for i, shard in enumerate(shards):
        filter_shard(con, shard, i)

    done = len(list(TEXT_DIR.glob("hepth_text-*.parquet")))
    if done != len(shards):
        log(f"error: {done} of {len(shards)} shard outputs present")
        return 1
    shutil.rmtree(DL_DIR, ignore_errors=True)
    shutil.rmtree(RAW / "_duckdb_tmp", ignore_errors=True)
    log("all shards done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
