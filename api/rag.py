"""Retrieval for the chat endpoint: hybrid search, then the prompt sent to the LLM."""

from __future__ import annotations

import re
import threading

import psycopg
from pgvector.psycopg import register_vector

from .config import settings
from retrieval import Hit, QueryEncoder, search  # pipeline/, via config

SYSTEM_PROMPT = """\
You are a research assistant for theoretical high-energy physics. You answer questions
using excerpts from a database of recent arXiv hep-th papers. The excerpts below are the
only papers you can see.

Rules:
- Cite excerpts inline by number, e.g. [1] or [2, 3]. Every statement taken from an
  excerpt needs a citation.
- Only mention papers that appear in the excerpts, with the exact title, authors and arXiv
  number given there. Never name or cite any other paper, author or arXiv number, even
  well-known ones, and never invent one.
- If the excerpts do not answer the question, say so plainly. You may then give brief
  general background, labelled as not coming from the database, without citing papers.
- When asked to find or list papers, list only the relevant papers among the excerpts, and
  say these are the closest matches retrieved, not a complete list.
- Write mathematics in LaTeX ($...$ inline, $$...$$ for display)."""

_encoder: QueryEncoder | None = None
_lock = threading.Lock()


def encoder() -> QueryEncoder:
    """The embedding model, loaded on first use (takes a few seconds)."""
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = QueryEncoder()
    return _encoder


def retrieve(question: str, k: int) -> list[Hit]:
    """Hybrid (vector + BM25) top-k chunks. Blocking: call from a worker thread."""
    embedding = encoder().encode(question)
    with psycopg.connect(settings.database_url, options=settings.db_options) as conn:
        register_vector(conn)
        return search(conn, embedding, k=k, per_paper=settings.per_paper, query=question)


# Math spans in chunk text ($$...$$ first, then $...$); \\. skips escaped characters like \$.
_MATH = re.compile(r"\$\$(?:\\.|[^$\\])+?\$\$|\$(?:\\.|[^$\\])+?\$", re.S)
_ENV = re.compile(r"\\begin\{(equation|align|gather|eqnarray|multline)(\*?)\}(.*?)\\end\{\1\2\}", re.S)


def _env_to_math(m: re.Match) -> str:
    body = m[3].strip()
    if "\\\\" in body or m[1] in ("align", "eqnarray"):  # multi-line or &-aligned
        body = f"\\begin{{aligned}}{body}\\end{{aligned}}"
    return f"${body}$"


def _plain_snippet(text: str, width: int) -> str:
    """Fallback: raw LaTeX as text, with $ escaped so nothing is rendered as math."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) > width:
        t = t[:width].rsplit(" ", 1)[0] + "…"
    return re.sub(r"(?<!\\)\$", r"\\$", t)


def math_snippet(text: str, width: int = 300) -> str:
    """A short preview for the UI that KaTeX can render: equation environments become
    $...$, labels are dropped, and the text is never cut inside a formula."""
    t = re.sub(r"\\label\{[^}]*\}|\\nonumber|\\notag", "", text)
    t = _ENV.sub(_env_to_math, t)
    t = re.sub(r"\\\[(.*?)\\\]|\\\((.*?)\\\)", lambda m: f"${(m[1] or m[2]).strip()}$", t, flags=re.S)
    t = re.sub(r"\s+", " ", t).strip()
    # Chunks can start or end inside a formula (the chunker splits on token counts).
    # Drop an environment cut in half, then a formula cut in half.
    envs = r"(?:equation|align|gather|eqnarray|multline)\*?"
    t = re.sub(rf"^.*?\\end\{{{envs}\}}", "", t)          # chunk began inside an environment
    t = re.sub(rf"\\begin\{{{envs}\}}.*$", "", t).strip()  # chunk ended inside one
    dollars = [m.start() for m in re.finditer(r"(?<!\\)\$", t)]
    if len(dollars) % 2:
        first = dollars[0]
        if first + 1 >= len(t) or t[first + 1] in " .,;:)]":  # first $ closes a formula
            t = t[first + 1:].lstrip(" .,;:)]")
        else:                                                  # last $ opens one
            t = t[:dollars[-1]].rstrip()
    if len(t) <= width:
        return _checked(t, text, width)
    spans = [m.span() for m in _MATH.finditer(t)]
    inside = lambda p: any(s < p < e for s, e in spans)
    # Cut points: spaces outside formulas, and formula ends. Take the last one that fits;
    # if a long formula straddles the limit, finish it unless that overruns by > 50%.
    cuts = [p for p, ch in enumerate(t) if ch == " " and not inside(p)] + [e for _, e in spans]
    fitting = [p for p in cuts if 0 < p <= width]
    cut = max(fitting) if fitting else min((p for p in cuts if p > 0), default=width)
    if cut > width * 1.5 and fitting == []:
        cut = width  # one huge formula: show nothing better than a hard cut
    return _checked(t[:cut].rstrip() + ("…" if cut < len(t) else ""), text, width)


def _checked(snippet: str, text: str, width: int) -> str:
    """Use the plain fallback if repairs left nothing, or an unpaired $ (e.g. one huge formula)."""
    if not snippet or len(re.findall(r"(?<!\\)\$", snippet)) % 2:
        return _plain_snippet(text, width)
    return snippet


def short_authors(authors: str | None, max_names: int = 3) -> str:
    names = [a.strip() for a in (authors or "").split(",") if a.strip()]
    if not names:
        return "unknown authors"
    return ", ".join(names[:max_names]) + (" et al." if len(names) > max_names else "")


def sources(hits: list[Hit]) -> list[dict]:
    """What the client sees (and what is stored on the assistant message)."""
    return [
        {"n": i, "paper_id": h.paper_id, "title": h.title, "authors": short_authors(h.authors),
         "section": h.section_path,
         "chunk_index": h.chunk_index, "url": h.arxiv_abs_url, "score": round(h.score, 5),
         "snippet": math_snippet(h.content, 280)}
        for i, h in enumerate(hits, 1)
    ]


def build_messages(history: list[dict], question: str, hits: list[Hit]) -> list[dict]:
    """System prompt + excerpts, earlier turns, then the new question."""
    system = SYSTEM_PROMPT
    if hits:
        excerpts = "\n\n".join(
            f'[{i}] "{h.title}" by {short_authors(h.authors)} (arXiv:{h.paper_id})\n'
            f"Section: {h.section_path}\n{h.content}"
            for i, h in enumerate(hits, 1)
        )
        system += f"\n\nExcerpts:\n\n{excerpts}"
    else:
        system += "\n\nNo excerpts were retrieved for this question, so no papers can be cited."
    return ([{"role": "system", "content": system}]
            + [{"role": m["role"], "content": m["content"]} for m in history]
            + [{"role": "user", "content": question}])
