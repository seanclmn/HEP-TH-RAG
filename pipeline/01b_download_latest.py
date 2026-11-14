"""Download full text for only the N most recent hep-th papers (default 100).

The dataset's `metadata` runs a few weeks past its `paper_text` (e.g. metadata to
2026-08-27, text to 2607.*), so "most recent" here means the N hep-th papers
(cross-lists included) with the latest first_version_date *that have full text*.

How: shards of `paper_text` are sorted by paper_id with new-style IDs (YYMM.NNNNN)
first, so we walk new-style shards from the end, reading only the paper_id column
remotely, until we have N hep-th candidates. Then DuckDB reads just those rows from
the remote shards (row-group pruning on paper_id), so no full shards are downloaded.

Needs data/raw/hepth_metadata.parquet from 01_download.py.
Output: data/raw/hepth_latest_<N>.parquet, which has the metadata columns plus text columns.

Usage:  .venv/bin/python pipeline/01b_download_latest.py [N]
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("dl", Path(__file__).with_name("01_download.py"))
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

NEW_ID = re.compile(r"^\d{4}\.\d{4,5}$")


def hf_url(path: str) -> str:
    return f"hf://datasets/{dl.REPO_ID}/{path}"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = dl.RAW / f"hepth_latest_{n}.parquet"
    if not dl.META_OUT.exists():
        dl.log(f"error: {dl.META_OUT.name} missing; run 01_download.py first")
        return 1

    con = dl.connect()
    meta = f"read_parquet('{dl.sql_path(dl.META_OUT)}')"
    con.execute("CREATE TEMP TABLE avail (paper_id VARCHAR, shard VARCHAR)")

    # Walk shards from the end, skipping the old-style-ID tail, collecting available IDs.
    for shard in reversed(dl.list_files("paper_text")):
        url = hf_url(shard)
        lo = con.execute(
            f"SELECT min(stats_min_value) FROM parquet_metadata('{url}') WHERE path_in_schema = 'paper_id'"
        ).fetchone()[0]
        if not NEW_ID.match(lo):
            continue  # shard holds only old-style IDs (astro-ph/..., hep-th/...)
        con.execute(
            f"INSERT INTO avail SELECT paper_id, '{shard}' FROM read_parquet('{url}') WHERE paper_id < 'a'"
        )
        (have,) = con.execute(f"SELECT count(*) FROM {meta} SEMI JOIN avail USING (paper_id)").fetchone()
        dl.log(f"{shard}: IDs from {lo}; {have} hep-th papers with text so far")
        if have >= n:
            break

    con.execute(
        f"""
        CREATE TEMP TABLE target AS
        SELECT m.*, a.shard FROM {meta} m JOIN avail a USING (paper_id)
        ORDER BY m.first_version_date DESC, m.paper_id DESC
        LIMIT {n}
        """
    )
    lo, hi = con.execute("SELECT min(first_version_date), max(first_version_date) FROM target").fetchone()
    dl.log(f"target: {n} papers submitted {lo} .. {hi}")

    shards = [s for (s,) in con.execute("SELECT DISTINCT shard FROM target").fetchall()]
    urls = ", ".join(f"'{hf_url(s)}'" for s in shards)
    ids = ", ".join(f"'{i}'" for (i,) in con.execute("SELECT paper_id FROM target").fetchall())
    tmp = out.with_suffix(".parquet.tmp")
    dl.log(f"fetching text from {len(shards)} shard(s)")
    con.execute(
        f"""
        COPY (
            SELECT m.* EXCLUDE (shard),
                   t.text, t.main_file, t.resolution, t.n_tex_files, t.n_files_used,
                   t.n_unused_files, t.text_encoding, t.text_sha256,
                   length(t.text) AS n_chars
            FROM target m
            JOIN (SELECT * FROM read_parquet([{urls}]) WHERE paper_id IN ({ids})) t USING (paper_id)
            ORDER BY m.first_version_date DESC, m.paper_id DESC
        ) TO '{dl.sql_path(tmp)}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    tmp.replace(out)
    total, primary, chars = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE primary_category = '{dl.CATEGORY}'), sum(n_chars) "
        f"FROM read_parquet('{dl.sql_path(out)}')"
    ).fetchone()
    dl.log(f"wrote {out.relative_to(dl.ROOT)}: {total} papers ({primary} primary hep-th), {chars:,} chars of text")
    return 0 if total == n else 1


if __name__ == "__main__":
    sys.exit(main())
