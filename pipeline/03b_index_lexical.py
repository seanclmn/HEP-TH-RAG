"""(Re)build the lexical (BM25) index: chunks.lex, lex_terms, lex_corpus.

03_embed.py already indexes new or changed chunks. Run this with --rebuild after
changing the tokenizer in lexical.py, so every chunk is re-tokenized.

Usage:  .venv/bin/python pipeline/03b_index_lexical.py [--rebuild]
"""

from __future__ import annotations

import sys
import time

from lexical import index_lexical
from retrieval import ROOT, connect


def main() -> int:
    rebuild = "--rebuild" in sys.argv[1:]
    t0 = time.time()
    with connect() as conn:
        conn.execute((ROOT / "db/init/02-schema.sql").read_text())
        n = index_lexical(conn, rebuild=rebuild)
        terms, = conn.execute("SELECT count(*) FROM lex_terms").fetchone()
    print(f"indexed {n:,} chunks ({terms:,} distinct terms) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
