"""Clean the LaTeX source of downloaded papers and split it into retrieval chunks.

Per paper:
  0. Expand the paper's own macros with its own definitions (macros.py), so chunks read
     \\mathcal{O} rather than \\cO, and \\be means whatever this paper defined it as.
  1. Keep only the document body (\\begin{document} .. \\end{document}); drop % comments,
     comment/\\iffalse blocks, bibliography, labels, citations, graphics and layout
     commands. Figures/tables are reduced to their captions. Math is kept as LaTeX.
  2. Split along \\section / \\subsection / \\subsubsection; text before the first section
     (title/author block) and acknowledgement sections are dropped. The metadata
     abstract becomes its own chunk (index 0).
  3. Pack paragraphs into chunks of ~TARGET tokens (Qwen3 tokenizer), carrying the
     tail of the previous chunk (up to OVERLAP tokens) into the next one. Paragraphs
     longer than TARGET are split into overlapping token windows.

Output: data/processed/chunks_<name>.parquet with paper_id, chunk_index, section_path,
content, n_tokens.

Usage:  .venv/bin/python pipeline/02_chunk.py [data/raw/hepth_latest_100.parquet]
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import duckdb
from transformers import AutoTokenizer

from macros import expand, extract_macros

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "raw" / "hepth_latest_100.parquet"
OUT_DIR = ROOT / "data" / "processed"
TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"

TARGET = 450   # tokens per chunk (content only; title/section header is added at embed time)
OVERLAP = 60   # max tokens carried over from the previous chunk
MIN_TOKENS = 25  # drop chunks smaller than this (e.g. a heading followed by a subsection)

tok = AutoTokenizer.from_pretrained(TOKENIZER)


def ntok(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def tail(s: str, n: int) -> str:
    ids = tok.encode(s, add_special_tokens=False)
    return tok.decode(ids[-n:]) if len(ids) > n else s


# ---------------------------------------------------------------- LaTeX helpers

def braced(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' -> (contents, index after the matching '}'). Tolerates unbalanced input."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    return s[i + 1 :], len(s)


def skip_ws_opt(s: str, i: int) -> int:
    """Skip whitespace, a '*', and one [...] optional argument starting at i."""
    while i < len(s) and s[i] in " \t*":
        i += 1
    if i < len(s) and s[i] == "[":
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    i = j + 1
                    break
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def replace_cmd(s: str, names: str, fn) -> str:
    """Replace \\name[opt]{arg} (all names in the regex alternation) with fn(arg)."""
    pat = re.compile(r"\\(?:" + names + r")(?![A-Za-z])")
    out, pos = [], 0
    for m in pat.finditer(s):
        if m.start() < pos:
            continue
        i = skip_ws_opt(s, m.end())
        if i < len(s) and s[i] == "{":
            arg, end = braced(s, i)
        else:
            arg, end = "", m.end()
        out.append(s[pos : m.start()])
        out.append(fn(arg))
        pos = end
    out.append(s[pos:])
    return "".join(out)


def strip_env(s: str, env: str, keep=lambda body: "") -> str:
    pat = re.compile(r"\\begin\{(" + env + r")\}(.*?)\\end\{\1\}", re.S)
    return pat.sub(lambda m: keep(m.group(2)), s)


def captions(body: str) -> str:
    caps = []
    replace_cmd(body, "caption", lambda a: caps.append(a) or "")
    return "\n\n" + "\n".join(f"[Caption: {c.strip()}]" for c in caps) + "\n\n" if caps else "\n"


def clean(tex: str, resolved: bool = False) -> str:
    tex = tex.replace("\r\n", "\n")
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end() :]
    m = re.search(r"\\end\{document\}", tex)
    if m:
        # Multi-file sources ("resolved") have the \input files appended after \end{document}.
        tex = tex[: m.start()] + ("\n\n" + tex[m.end() :] if resolved else "")
    tex = re.sub(r"(?<!\\)%.*", "", tex)                      # % comments
    tex = re.sub(r"\\iffalse\b.*?\\fi\b", "", tex, flags=re.S)
    tex = strip_env(tex, r"comment")
    tex = strip_env(tex, r"thebibliography")
    tex = strip_env(tex, r"abstract")                          # metadata abstract is chunk 0
    tex = strip_env(tex, r"figure\*?|table\*?|wrapfigure|subfigure", captions)
    tex = re.sub(r"\\(bibliography|bibliographystyle|input|include)\{[^}]*\}", "", tex)
    tex = replace_cmd(tex, r"label|includegraphics|vspace|hspace|bibitem|hypertarget", lambda a: "")
    tex = replace_cmd(tex, r"cite[pt]?|citep|citet|citealp|parencite", lambda a: "\x00")
    tex = re.sub(r"[ ~]*\x00+", "", tex)                   # drop cites and the space/~ before them
    tex = replace_cmd(tex, r"eqref|ref|autoref|cref|Cref", lambda a: "[ref]")
    tex = replace_cmd(tex, r"textbf|textit|emph|underline|texttt|textsc", lambda a: a)
    tex = re.sub(
        r"\\(maketitle|tableofcontents|newpage|clearpage|noindent|bigskip|medskip|smallskip"
        r"|centering|appendix|acknowledgments|acknowledgements|footnotesize|small|large)\b",
        lambda m: "\n\\appendix\n" if m.group(1) == "appendix" else "",
        tex,
    )
    tex = re.sub(r"[ \t]+\n", "\n", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


# ---------------------------------------------------------------- sectioning + chunking

SEC = re.compile(r"\\(section|subsection|subsubsection|appendix)\b")
LEVEL = {"section": 0, "subsection": 1, "subsubsection": 2}


def sections(body: str) -> list[tuple[str, str]]:
    """-> [(section_path, text)] for each piece between headings; frontmatter dropped."""
    out, path, in_app = [], [], False
    pos, cur = 0, None
    for m in SEC.finditer(body):
        if m.start() < pos:
            continue
        if cur is not None:
            out.append((cur, body[pos : m.start()]))
        if m.group(1) == "appendix":
            in_app, pos = True, m.end()
            cur = " > ".join(path) if path else None
            continue
        i = skip_ws_opt(body, m.end())
        title, pos = braced(body, i) if i < len(body) and body[i] == "{" else ("", m.end())
        title = re.sub(r"\\[A-Za-z]+\s*", "", title.replace("\\&", "&")).replace("{", "").replace("}", "")
        title = re.sub(r"\s+", " ", title).strip() or "(untitled)"
        lvl = LEVEL[m.group(1)]
        if lvl == 0 and in_app and not title.lower().startswith("appendix"):
            title = f"Appendix: {title}"
        path = path[:lvl] + [title]
        cur = " > ".join(path)
    if cur is not None:
        out.append((cur, body[pos:]))
    if not out:  # no sectioning commands at all: whole body is one section
        out = [("Body", body)]
    return [(p, t) for p, t in out if not re.match(r"(?i)acknowledg", p.split(" > ")[0])]


def windows(text: str) -> list[str]:
    ids = tok.encode(text, add_special_tokens=False)
    step = TARGET - OVERLAP
    return [tok.decode(ids[i : i + TARGET]) for i in range(0, max(len(ids) - OVERLAP, 1), step)]


def pack(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, cur_n = [], [], 0
    for p in paras:
        pn = ntok(p)
        if pn > TARGET:
            if cur:
                chunks.append("\n\n".join(cur))
            chunks.extend(windows(p))
            carry = tail(chunks[-1], OVERLAP)
            cur, cur_n = [carry], ntok(carry)
            continue
        if cur_n + pn > TARGET and cur:
            chunks.append("\n\n".join(cur))
            carry = tail(cur[-1], OVERLAP)
            cur, cur_n = [carry], ntok(carry)
        cur.append(p)
        cur_n += pn
    if cur and (not chunks or cur_n > OVERLAP):  # skip a trailing chunk that is only carry-over
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_paper(paper_id: str, abstract: str, tex: str, resolved: bool) -> list[dict]:
    macros, body = extract_macros(tex)
    # The metadata abstract can use the paper's macros too, so expand it with the same table.
    abstract = expand(re.sub(r"\s+", " ", abstract or "").strip(), macros)
    rows = [{"section_path": "Abstract", "content": abstract}]
    for path, text in sections(clean(expand(body, macros), resolved)):
        rows += [{"section_path": path, "content": c} for c in pack(text)]
    out = []
    for r in rows:
        n = ntok(r["content"])
        if n >= MIN_TOKENS:
            out.append({"paper_id": paper_id, "chunk_index": len(out), "n_tokens": n, **r})
    return out


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"chunks_{src.stem.removeprefix('hepth_')}.parquet"
    con = duckdb.connect()
    papers = con.execute(
        "SELECT paper_id, abstract, text, resolution = 'resolved' FROM read_parquet(?) WHERE text IS NOT NULL ORDER BY paper_id",
        [str(src)],
    ).fetchall()
    t0, rows = time.time(), []
    for pid, abstract, text, resolved in papers:
        rows += chunk_paper(pid, abstract, text, resolved)
    con.execute(
        "CREATE TABLE c (paper_id VARCHAR, chunk_index INTEGER, section_path VARCHAR, content VARCHAR, n_tokens INTEGER)"
    )
    con.executemany(
        "INSERT INTO c VALUES (?, ?, ?, ?, ?)",
        [(r["paper_id"], r["chunk_index"], r["section_path"], r["content"], r["n_tokens"]) for r in rows],
    )
    tmp = out.with_suffix(".parquet.tmp")
    con.execute(f"COPY (SELECT * FROM c ORDER BY paper_id, chunk_index) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out)
    n, avg, p50, mx = con.execute(
        "SELECT count(*), avg(n_tokens), median(n_tokens), max(n_tokens) FROM c"
    ).fetchone()
    print(f"{len(papers)} papers -> {n:,} chunks (avg {avg:.0f}, median {p50:.0f}, max {mx} tokens) "
          f"in {time.time() - t0:.0f}s -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
