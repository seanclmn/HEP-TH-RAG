"""Per-paper macro expansion (pipeline/macros.py)."""

import pytest

from macros import Macro, expand, expand_paper, extract_macros


def run(defs: str, body: str) -> str:
    return expand_paper(defs + "\n" + body).strip()


class TestDefinitionForms:
    @pytest.mark.parametrize("defn", [
        r"\newcommand{\cO}{\mathcal{O}}",
        r"\newcommand\cO{\mathcal{O}}",
        r"\newcommand*{\cO}{\mathcal{O}}",
        r"\renewcommand{\cO}{\mathcal{O}}",
        r"\providecommand{\cO}{\mathcal{O}}",
        r"\def\cO{\mathcal{O}}",
        r"\newcommand { \cO } {\mathcal{O}}",
    ])
    def test_zero_arg(self, defn):
        assert run(defn, r"$\cO(x)$") == r"$\mathcal{O}(x)$"

    def test_multiline_body_with_nested_braces(self):
        defs = "\\newcommand{\\half}{\n  {\\frac{1}{2}}\n}"
        assert run(defs, r"\half x") == "{\\frac{1}{2}}\n x"  # run() strips the leading newline

    def test_declare_math_operator(self):
        assert run(r"\DeclareMathOperator{\Tr}{Tr}", r"\Tr\rho") == r"\operatorname{Tr}\rho"
        assert run(r"\DeclareMathOperator*{\argmax}{arg\,max}", r"\argmax") == r"\operatorname*{arg\,max}"

    def test_definitions_are_removed_from_text(self):
        out = expand_paper("\\newcommand{\\cO}{\\mathcal{O}}\n\\def\\be{\\begin{equation}}\nbody")
        assert "newcommand" not in out and "\\def" not in out and out.strip() == "body"


class TestArguments:
    PD = r"\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}"

    def test_braced_args(self):
        assert run(self.PD, r"\pd{f}{x}") == r"\frac{\partial f}{\partial x}"

    def test_single_token_args(self):
        # TeX takes one character or control word per undelimited argument.
        assert run(self.PD, r"\pd fx") == r"\frac{\partial f}{\partial x}"
        assert run(self.PD, r"\pd\phi t") == r"\frac{\partial \phi}{\partial t}"

    def test_optional_default_argument(self):
        defs = r"\newcommand{\ket}[2][\psi]{|#1\rangle_{#2}}"
        assert run(defs, r"\ket{A}") == r"|\psi\rangle_{A}"
        assert run(defs, r"\ket[\phi]{B}") == r"|\phi\rangle_{B}"

    def test_def_with_parameters(self):
        assert run(r"\def\comm#1#2{[#1,#2]}", r"\comm{A}{B}") == "[A,B]"

    def test_missing_argument_leaves_use_alone(self):
        assert run(self.PD, r"\pd{f}}") == r"\pd{f}}"

    def test_braces_in_args(self):
        assert run(self.PD, r"\pd{e^{x}}{x_{1}}") == r"\frac{\partial e^{x}}{\partial x_{1}}"


class TestPerPaperMeaning:
    """The same name means different things in different papers (seen in the corpus)."""

    def test_be_as_equation(self):
        out = run(r"\def\be{\begin{equation}}\def\ee{\end{equation}}", r"\be E=mc^2 \ee")
        assert out == r"\begin{equation} E=mc^2 \end{equation}"

    def test_be_as_bold_e(self):
        assert run(r"\newcommand{\be}{\mathbf{e}}", r"$\be_1$") == r"$\mathbf{e}_1$"

    def test_later_definition_wins_but_providecommand_does_not_override(self):
        assert run(r"\newcommand{\e}{\epsilon}\renewcommand{\e}{\varepsilon}", r"\e") == r"\varepsilon"
        assert run(r"\newcommand{\e}{\epsilon}\providecommand{\e}{e}", r"\e") == r"\epsilon"


class TestMatching:
    def test_prefix_names_do_not_collide(self):
        defs = r"\newcommand{\e}{E}\newcommand{\eps}{\epsilon}"
        assert run(defs, r"\e \eps \epsilon \ee") == r"E \epsilon \epsilon \ee"

    def test_space_after_macro_preserved(self):
        # Must not glue \epsilon onto the next word (\epsilonx would be a different command).
        assert run(r"\def\e{\epsilon}", r"\e x") == r"\epsilon x"

    def test_nested_expansion(self):
        assert run(r"\def\a{\b+\b}\def\b{x}", r"\a") == "x+x"

    def test_self_recursion_terminates(self):
        out = run(r"\def\a{\a x}", r"\a")
        assert out.endswith("x") and len(out) < 100

    def test_exponential_recursion_is_capped(self):
        out = run(r"\def\a{\a\a}", r"\a")
        assert len(out) < 50_000


class TestSkipped:
    def test_protected_names_not_expanded(self):
        out = run(r"\renewcommand{\section}{\oops}", r"\section{Intro}")
        assert out == r"\section{Intro}"

    def test_delimited_def_skipped(self):
        macros, _ = extract_macros(r"\def\foo#1.{[#1]}")
        assert "foo" not in macros

    def test_internal_macros_skipped(self):
        macros, _ = extract_macros(r"\def\x{\@foo}\newcommand{\y}{\def\z##1{##1}}")
        assert not macros

    def test_no_definitions_is_identity(self):
        tex = r"\section{A} $\chi_i$ text"
        assert expand_paper(tex) == tex

    def test_malformed_definition_ignored(self):
        assert expand_paper(r"\newcommand{\x}") == r"\newcommand{\x}"


def test_expand_with_explicit_table():
    assert expand(r"\cN=4", {"cN": Macro(0, None, r"\mathcal{N}")}) == r"\mathcal{N}=4"
