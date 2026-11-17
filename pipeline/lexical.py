"""Lexical (BM25) search over normalised LaTeX: the keyword half of hybrid search.

Vector search captures what a passage is about, not its exact notation. This module
indexes the notation itself. Chunks and queries both go through
    normalize_latex() -> tokens() -> terms()
so that notation variants meet: G_{\\rm N} / G_N / G_{\\mathrm{N}}, \\dfrac / \\frac,
\\left( / (, $...$ / \\begin{equation}, and so on.

Why not Postgres' own text search:
  * Its parser splits \\chi_i into 'chi' + 'i' and G_{\\rm N} into 'g', 'rm', 'n'. Terms here
    are built in Python and stored as a literal tsvector, so tokens like \\chi_i stay whole.
  * ts_rank has no IDF, so common symbols would swamp rare ones. Ranking is BM25,
    using per-term document frequencies in lex_terms and corpus stats in lex_corpus.
    The GIN index on chunks.lex only selects candidates.

Terms: each atom (word, command, or symbol with fused sub/superscripts like \\chi_i,
J_ijkl, ads_3); the shorter prefixes of a fused atom (G_N^bare -> G, G_N), for recall
across index conventions; and bigrams of adjacent atoms, for structure and phrases.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

K1, B = 1.2, 0.75          # BM25 parameters (standard values)
MAX_DF_FRACTION = 0.5      # terms in more chunks than this don't select candidates
MAX_SCRIPT = 10            # longer sub/superscripts are not fused into one term
MAX_TERM = 200             # Postgres lexemes are limited to 2047 bytes
MAX_POS, MAX_POSITIONS = 16383, 256  # tsvector limits

# ---------------------------------------------------------------- normalisation

_WRAPPERS = (r"mathrm|mathit|mathbf|mathsf|mathtt|mathcal|mathscr|mathfrak|mathbb|boldsymbol|bm"
             r"|operatorname|text|textrm|textit|textbf|textsf|emph|rm|bf|it|cal|sf|tt")
_SPACING = (r"left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|bigg|Bigg|big|Big"
            r"|quad|qquad|nonumber|notag|displaystyle|textstyle|scriptstyle|limits|nolimits")
_SYNONYMS = {
    "dfrac": "frac", "tfrac": "frac", "le": "leq", "ge": "geq", "ne": "neq",
    "to": "rightarrow", "gets": "leftarrow", "varepsilon": "epsilon", "varphi": "phi",
    "vartheta": "theta", "varrho": "rho", "varsigma": "sigma", "varpi": "pi",
}
_STOP = frozenset(
    "the of and to in is that for we this with as by on are be which from an it at can its "
    "or these our has have not will was were been their there such also than then thus".split()
)

_NORMALISERS = [
    (re.compile(r"\\(?:begin|end)\s*\{[^}]*\}"), " "),                 # environment markers
    (re.compile(r"\$+|\\[()\[\]]"), " "),                              # math delimiters
    (re.compile(r"\\[{}%&_#$,;:!> ]"), " "),                           # escapes and thin spaces
    (re.compile(rf"\\(?:{_WRAPPERS})(?![A-Za-z])"), " "),               # font wrappers (keep arg)
    (re.compile(rf"\\(?:{_SPACING})(?![A-Za-z])"), " "),
    (re.compile(r"\\\\|[&~]"), " "),
]
_SYN = re.compile(r"\\(" + "|".join(sorted(_SYNONYMS, key=len, reverse=True)) + r")(?![A-Za-z])")


def normalize_latex(s: str) -> str:
    """Strip presentation-only LaTeX and map synonyms; keeps braces and scripts."""
    for pat, rep in _NORMALISERS:
        s = pat.sub(rep, s)
    s = _SYN.sub(lambda m: "\\" + _SYNONYMS[m.group(1)], s)
    return " ".join(s.split())


# ---------------------------------------------------------------- tokens and terms

_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[_^{}]")


@dataclass
class _Atom:
    text: str
    prefixes: tuple[str, ...] = ()  # shorter forms when scripts were fused: G_N^b -> (G, G_N)


def _parse(toks: list[str], i: int, depth: int) -> tuple[list[_Atom], int]:
    """Parse until the matching '}' (depth > 0) or the end; returns atoms and next index."""
    atoms: list[_Atom] = []
    while i < len(toks):
        t = toks[i]
        if t == "}":
            if depth:
                return atoms, i + 1
            i += 1  # stray close brace
        elif t == "{":
            inner, i = _parse(toks, i + 1, depth + 1)
            atoms.extend(inner)
        elif t in "_^":
            script, i, inner = _script(toks, i + 1, depth)
            fusable = (script is not None and len(script) <= MAX_SCRIPT
                       and not any(a.prefixes for a in inner))  # e^{-ip_1 x} is not one symbol
            if atoms and fusable:
                a = atoms[-1]
                atoms[-1] = _Atom(f"{a.text}{t}{script}", (*a.prefixes, a.text))
            else:
                atoms.extend(inner)
        else:
            atoms.append(_Atom(t.lower() if len(t) > 1 and t[0] != "\\" else t))
            i += 1
    return atoms, i


def _script(toks: list[str], i: int, depth: int) -> tuple[str | None, int, list[_Atom]]:
    """The argument of _ or ^: a {group} or (per LaTeX) the first character of a token."""
    if i >= len(toks):
        return None, i, []
    t = toks[i]
    if t == "{":
        inner, i = _parse(toks, i + 1, depth + 1)
        return "".join(a.text for a in inner), i, inner
    if t in "_^}":
        return None, i, []
    if t[0] == "\\":
        return t, i + 1, [_Atom(t)]
    head, rest = t[0], t[1:]
    if rest:  # x_ab means x_a b
        toks[i] = rest
        return head, i, [_Atom(head)]
    return head, i + 1, [_Atom(head)]


def _atoms(text: str) -> list[_Atom]:
    atoms, _ = _parse(_TOKEN.findall(normalize_latex(text)), 0, 0)
    return [a for a in atoms if a.text not in _STOP]


def tokens(text: str) -> list[str]:
    """Atoms of normalised text: words lower-cased, single letters and commands case-kept."""
    return [a.text for a in _atoms(text)]


def terms(text: str) -> list[str]:
    """Index/query terms in order: atoms, bases of fused atoms, and adjacent-atom bigrams."""
    atoms = _atoms(text)
    out = []
    for j, a in enumerate(atoms):
        out.append(a.text)
        out.extend(a.prefixes)
        if j + 1 < len(atoms):
            out.append(f"{a.text} {atoms[j + 1].text}")
    return [t for t in out if len(t) <= MAX_TERM]


def _quote(term: str) -> str:
    return "'" + term.replace("\\", "\\\\").replace("'", "''") + "'"


def tsvector_literal(ts: list[str]) -> str:
    """Postgres tsvector input text with our terms as exact lexemes (positions = order)."""
    pos: dict[str, list[int]] = {}
    for p, t in enumerate(ts, 1):
        lst = pos.setdefault(t, [])
        if len(lst) < MAX_POSITIONS:
            lst.append(min(p, MAX_POS))
    return " ".join(f"{_quote(t)}:{','.join(map(str, sorted(set(ps))))}" for t, ps in pos.items())


def tsquery_literal(ts: list[str]) -> str:
    return " | ".join(_quote(t) for t in dict.fromkeys(ts))


# ---------------------------------------------------------------- indexing

def refresh_stats(conn) -> None:
    """Recompute document frequencies and corpus stats used for BM25."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lex_terms")
        cur.execute("""INSERT INTO lex_terms (term, df)
                       SELECT word, ndoc FROM ts_stat('SELECT lex FROM chunks WHERE lex IS NOT NULL')""")
        cur.execute("DELETE FROM lex_corpus")
        cur.execute("""INSERT INTO lex_corpus (n_docs, avg_len)
                       SELECT count(*), coalesce(avg(lex_len), 0) FROM chunks WHERE lex IS NOT NULL""")


def index_lexical(conn, rebuild: bool = False, batch: int = 1000) -> int:
    """Fill chunks.lex for chunks missing it (or all, with rebuild), then refresh stats.

    Indexes title + section path + content, like the embedded text. Run with
    rebuild=True after changing anything above. Returns the number of chunks indexed.
    """
    sql = """SELECT c.id, p.title, c.section_path, c.content
             FROM chunks c JOIN papers p USING (paper_id)"""
    rows = conn.execute(sql + ("" if rebuild else " WHERE c.lex IS NULL")).fetchall()
    with conn.cursor() as cur:
        for b in range(0, len(rows), batch):
            upd = []
            for cid, title, path, content in rows[b : b + batch]:
                ts = terms(f"{title}\n{path}\n{content}")
                upd.append((tsvector_literal(ts), len(ts), cid))
            cur.executemany("UPDATE chunks SET lex = %s::tsvector, lex_len = %s WHERE id = %s", upd)
    refresh_stats(conn)
    return len(rows)


# ---------------------------------------------------------------- search

BM25_SQL = """
WITH q AS (SELECT * FROM unnest(%(terms)s::text[], %(idf)s::float8[]) AS q(term, idf)),
cand AS (SELECT id, paper_id, lex, lex_len FROM chunks WHERE lex @@ %(tsq)s::tsquery)
SELECT c.id, c.paper_id,
       sum(q.idf * u.tf * (%(k1)s + 1)
           / (u.tf + %(k1)s * (1 - %(b)s + %(b)s * c.lex_len / %(avg)s))) AS score
FROM cand c
CROSS JOIN LATERAL (SELECT lexeme, cardinality(positions) AS tf FROM unnest(c.lex)) u
JOIN q ON q.term = u.lexeme
GROUP BY c.id, c.paper_id
ORDER BY score DESC, c.id
LIMIT %(n)s
"""


def idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def lexical_candidates(conn, query: str, n: int) -> list[tuple[int, str, float]]:
    """Top-n (chunk id, paper_id, bm25) for the query's terms; [] if none are indexed."""
    qterms = list(dict.fromkeys(terms(query)))
    meta = conn.execute("SELECT n_docs, avg_len FROM lex_corpus").fetchone()
    if not qterms or not meta or not meta[0]:
        return []
    n_docs, avg_len = meta
    df = dict(conn.execute("SELECT term, df FROM lex_terms WHERE term = ANY(%s)", [qterms]).fetchall())
    present = [t for t in qterms if t in df]
    if not present:
        return []
    selective = [t for t in present if df[t] <= MAX_DF_FRACTION * n_docs] or present
    return conn.execute(BM25_SQL, {
        "terms": present, "idf": [idf(n_docs, df[t]) for t in present],
        "tsq": tsquery_literal(selective), "k1": K1, "b": B, "avg": avg_len or 1.0, "n": n,
    }).fetchall()
