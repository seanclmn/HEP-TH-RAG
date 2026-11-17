"""02_chunk.chunk_paper end to end (loads the Qwen3 tokenizer, which is quick)."""

import importlib.util

from retrieval import ROOT

_spec = importlib.util.spec_from_file_location("chunk", ROOT / "pipeline/02_chunk.py")
chunk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunk)

TEX = r"""
\documentclass{article}
\newcommand{\cO}{\mathcal{O}}
\def\be{\begin{equation}}\def\ee{\end{equation}}
\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}
\begin{document}
\maketitle
\section{Introduction}
We study the operator $\cO$ and its flow, governed by the equation below in detail.
\be \pd{\cO}{t} = [H, \cO] \ee
This equation is the starting point for everything that follows in this paper.
\end{document}
"""


def test_expands_paper_macros_in_body_and_abstract():
    abstract = (r"We study $\cO$ operators and their renormalization group flow in a large class "
                r"of strongly coupled theories, and compare with holographic predictions.")  # >= MIN_TOKENS
    rows = chunk.chunk_paper("2401.00001", abstract, TEX, False)
    text = "\n".join(r["content"] for r in rows)
    assert rows[0]["section_path"] == "Abstract" and r"$\mathcal{O}$" in rows[0]["content"]
    assert r"\frac{\partial \mathcal{O}}{\partial t} = [H, \mathcal{O}]" in text
    assert r"\begin{equation}" in text
    for leftover in (r"\cO", r"\pd", r"\be ", r"\newcommand", r"\def"):
        assert leftover not in text
