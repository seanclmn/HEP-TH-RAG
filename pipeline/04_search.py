r"""Search the embedded chunks from the command line.

Usage:
    .venv/bin/python pipeline/04_search.py "what is the swampland distance conjecture?"
    .venv/bin/python pipeline/04_search.py -k 5 --per-paper 1 "holographic entanglement entropy"
    .venv/bin/python pipeline/04_search.py --mode lexical 'J_{ijkl}\chi_i\chi_j'   # no model needed
    .venv/bin/python pipeline/04_search.py --json "..." | jq
    .venv/bin/python pipeline/04_search.py            # interactive: model loads once, ask many

Modes: hybrid (default; vector + BM25 fused with RRF), vector, lexical.
Prints rank, cosine similarity, BM25, arXiv id, title, section and a snippet of each hit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

from retrieval import DEFAULT_INSTRUCTION, Hit, QueryEncoder, connect, search, snippet


def render(hits: list[Hit], width: int, full: bool) -> str:
    if not hits:
        return "no results"
    lines = []
    for rank, h in enumerate(hits, 1):
        sim = "  -  " if h.similarity is None else f"{h.similarity:.3f}"
        bm = "   -" if h.bm25 is None else f"{h.bm25:4.1f}"
        lines.append(f"{rank:>2}. sim {sim}  bm25 {bm}  {h.paper_id}  {h.title}")
        lines.append(f"      § {h.section_path}  (chunk {h.chunk_index})")
        body = h.content.strip() if full else snippet(h.content, width)
        lines.extend(f"      {line}" for line in body.splitlines())
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="question; omit for interactive mode")
    ap.add_argument("--mode", choices=["hybrid", "vector", "lexical"], default="hybrid")
    ap.add_argument("-k", type=int, default=10, help="number of hits (default 10)")
    ap.add_argument("--per-paper", type=int, metavar="N", help="at most N hits from any one paper")
    ap.add_argument("--json", action="store_true", help="print hits as JSON")
    ap.add_argument("--full", action="store_true", help="print whole chunks instead of snippets")
    ap.add_argument("--width", type=int, default=300, help="snippet length in characters")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Qwen3 query instruction")
    args = ap.parse_args()
    if args.k < 1 or (args.per_paper is not None and args.per_paper < 1):
        ap.error("-k and --per-paper must be >= 1")

    encoder = None
    if args.mode != "lexical":
        t0 = time.time()
        encoder = QueryEncoder(instruction=args.instruction)
        print(f"[loaded model on {encoder.device} in {time.time() - t0:.1f}s]", file=sys.stderr)

    with connect() as conn:
        def run(q: str) -> None:
            t = time.time()
            hits = search(conn, encoder.encode(q) if encoder else None, k=args.k,
                          per_paper=args.per_paper, query=None if args.mode == "vector" else q)
            if args.json:
                print(json.dumps([asdict(h) for h in hits], indent=2, ensure_ascii=False))
            else:
                print(render(hits, args.width, args.full))
            print(f"[{len(hits)} hits in {(time.time() - t) * 1000:.0f} ms]", file=sys.stderr)

        if args.query:
            run(" ".join(args.query))
            return 0
        while True:
            try:
                q = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
            if q:
                run(q)


if __name__ == "__main__":
    sys.exit(main())
