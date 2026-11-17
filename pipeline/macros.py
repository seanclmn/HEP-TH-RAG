r"""Expand each paper's own LaTeX macros using its own definitions.

96 of the first 100 papers define macros, and 35% of chunks used one. Left unexpanded,
the same symbol is spelled differently per paper (\cO / {\cal O} / \mathcal{O}), and
one name means different things in different papers: \be is \begin{equation} in most,
but \mathbf{e} or \bar{e} in others. Embeddings, lexical search and the LLM all see
better text after expansion.

Handled:
  \newcommand / \renewcommand / \providecommand (starred too), with or without braces
  around the name, [n] arguments and an optional [default] first argument;
  \def\name{...} and \def\name#1#2{...} (undelimited parameters only);
  \DeclareMathOperator{\name}{text} -> \operatorname{text} (starred: \operatorname*).
Skipped (left as-is): \def with delimited parameters (\def\foo#1.{...}), bodies with
## or '@' (internals, nested definitions), and names in PROTECTED, which the chunker
relies on (\section, \cite, \label, ...).

Definitions are removed from the text. Expansion is recursive up to MAX_DEPTH passes,
and stops if the text grows past MAX_GROWTH times its size (runaway recursion).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_DEPTH = 10
MAX_GROWTH = 4

PROTECTED = frozenset(
    "section subsection subsubsection paragraph subparagraph chapter part appendix "
    "begin end item label ref eqref autoref cref Cref cite citep citet citealp parencite "
    "caption footnote input include bibliography bibliographystyle bibitem "
    "documentclass usepackage maketitle title author date abstract "
    "newcommand renewcommand providecommand def let DeclareMathOperator "
    "textbf textit emph underline texttt textsc includegraphics".split()
)


@dataclass(frozen=True)
class Macro:
    nargs: int
    default: str | None  # default for an optional first argument, if any
    body: str


_DEF = re.compile(
    r"\\(?P<kind>newcommand|renewcommand|providecommand|DeclareMathOperator|def)(?P<star>\*?)"
    r"(?![A-Za-z])\s*"
)
_NAME = re.compile(r"\{?\s*\\(?P<name>[A-Za-z]+)\s*\}?")


def _group(s: str, i: int) -> tuple[str, int] | None:
    """s[i] == '{' -> (contents, index after matching '}'); None if unbalanced."""
    if i >= len(s) or s[i] != "{":
        return None
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
    return None


def _bracket(s: str, i: int) -> tuple[str, int] | None:
    """s[i] == '[' -> (contents, index after matching ']'), respecting braces."""
    if i >= len(s) or s[i] != "[":
        return None
    depth, j = 0, i + 1
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "]" and depth == 0:
            return s[i + 1 : j], j + 1
        j += 1
    return None


def _ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def _parse_def(s: str, m: re.Match) -> tuple[str, Macro | None, int] | None:
    """Parse one definition starting at match m -> (name, macro or None if unusable, end)."""
    kind, i = m.group("kind"), m.end()
    nm = _NAME.match(s, i)
    if not nm:
        return None
    name, i = nm.group("name"), nm.end()

    if kind == "DeclareMathOperator":
        g = _group(s, _ws(s, i))
        if not g:
            return None
        op = "\\operatorname*" if m.group("star") else "\\operatorname"
        return name, Macro(0, None, f"{op}{{{g[0]}}}"), g[1]

    if kind == "def":
        params = re.match(r"(?:#[1-9])*", s[i:]).group(0)
        g = _group(s, i + len(params))
        if g is None:  # delimited parameters (\def\foo#1.{...}) or malformed
            return None
        expected = "".join(f"#{k}" for k in range(1, len(params) // 2 + 1))
        ok = params == expected
        return name, Macro(len(params) // 2, None, g[0]) if ok else None, g[1]

    nargs, default = 0, None
    j = _ws(s, i)
    b = _bracket(s, j)
    if b:
        if not b[0].strip().isdigit():
            return None
        nargs, j = int(b[0]), _ws(s, b[1])
        b = _bracket(s, j)
        if b:
            default, j = b[0], _ws(s, b[1])
    g = _group(s, j)
    if not g:
        return None
    return name, Macro(nargs, default, g[0]), g[1]


def extract_macros(tex: str) -> tuple[dict[str, Macro], str]:
    """Parse the macro definitions in tex -> ({name: Macro}, tex with definitions removed).

    Later definitions override earlier ones, except \\providecommand, which only
    defines a name that is not yet defined.
    """
    macros: dict[str, Macro] = {}
    out, pos = [], 0
    for m in _DEF.finditer(tex):
        if m.start() < pos:
            continue
        parsed = _parse_def(tex, m)
        if parsed is None:
            continue
        name, macro, end = parsed
        out.append(tex[pos : m.start()])
        pos = end
        if macro is None or name in PROTECTED or "##" in macro.body or "@" in macro.body:
            continue
        if m.group("kind") == "providecommand" and name in macros:
            continue
        macros[name] = macro
    out.append(tex[pos:])
    return macros, "".join(out)


def _arg(s: str, i: int) -> tuple[str, int] | None:
    """One mandatory TeX argument at i: a {group}, a control word, or a single character."""
    i = _ws(s, i)
    if i >= len(s):
        return None
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        m = re.match(r"\\(?:[A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    if s[i] in "}":
        return None
    return s[i], i + 1


def _substitute(body: str, args: list[str]) -> str:
    return re.sub(r"#([1-9])", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) <= len(args) else "", body)


def _expand_once(s: str, macros: dict[str, Macro], pat: re.Pattern) -> tuple[str, bool]:
    out, pos, changed = [], 0, False
    for m in pat.finditer(s):
        if m.start() < pos:
            continue
        mac = macros[m.group(1)]
        i, args = m.end(), []
        if mac.default is not None:
            b = _bracket(s, _ws(s, i)) if mac.nargs else None
            if b:
                args.append(b[0])
                i = b[1]
            else:
                args.append(mac.default)
        ok = True
        while len(args) < mac.nargs:
            a = _arg(s, i)
            if a is None:
                ok = False
                break
            args.append(a[0])
            i = a[1]
        if not ok:
            continue
        out.append(s[pos : m.start()])
        out.append(_substitute(mac.body, args))
        pos, changed = i, True
    out.append(s[pos:])
    return "".join(out), changed


def expand(tex: str, macros: dict[str, Macro]) -> str:
    """Expand uses of macros in tex (recursively, bounded)."""
    if not macros:
        return tex
    names = sorted(macros, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z])")
    limit = MAX_GROWTH * len(tex) + 10_000
    for _ in range(MAX_DEPTH):
        new, changed = _expand_once(tex, macros, pat)
        if not changed or len(new) > limit:
            return tex if len(new) > limit else new
        tex = new
    return tex


def expand_paper(tex: str) -> str:
    """Extract a paper's macro definitions, remove them, and expand their uses."""
    macros, body = extract_macros(tex)
    return expand(body, macros)
