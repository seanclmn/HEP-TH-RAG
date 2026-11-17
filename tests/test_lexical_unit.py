"""LaTeX normalisation and term extraction: no database."""

import pytest

from lexical import idf, normalize_latex, terms, tokens, tsquery_literal, tsvector_literal


class TestNotationVariantsMatch:
    """Different ways of writing the same thing must produce the same tokens."""

    @pytest.mark.parametrize("variants", [
        [r"G_N", r"G_{N}", r"G_{\rm N}", r"G_{\mathrm{N}}", r"G_{\text{N}}"],
        [r"\frac{A}{4G_N}", r"\dfrac{A}{4 G_N}", r"\tfrac{A}{4G_{N}}"],
        [r"\left( x \right)", r"\big( x \big)", r"( x )", r"x"],
        [r"$\mathrm{AdS}_3$", r"AdS$_3$", r"\text{AdS}_{3}", r"AdS_3"],
        [r"\varepsilon \le 1", r"\epsilon \leq 1"],
        [r"\begin{equation} E = m \end{equation}", r"\[ E = m \]", r"$$E = m$$", r"E = m"],
        [r"a \, b \; c \quad d", r"a b c d"],
    ])
    def test_same_tokens(self, variants):
        expected = tokens(variants[0])
        assert expected
        for v in variants[1:]:
            assert tokens(v) == expected, v


class TestTokens:
    def test_fuses_scripts(self):
        assert tokens(r"\sum_{ijkl} J_{ijkl} \chi_i \chi_j") == [r"\sum_ijkl", "J_ijkl", r"\chi_i", r"\chi_j"]

    def test_sub_and_superscript(self):
        assert tokens(r"T^{\mu\nu} x_i^2") == [r"T^\mu\nu", "x_i^2"]

    def test_latex_script_takes_one_character(self):
        assert tokens("x_ab") == ["x_a", "b"]

    def test_long_or_nested_script_not_fused(self):
        assert tokens(r"e^{-ip_1\cdot x_1}") == ["e", "ip_1", r"\cdot", "x_1"]
        assert tokens(r"J_{i_1\cdots i_q}") == ["J", "i_1", r"\cdots", "i_q"]

    def test_case(self):
        # Prose words fold case; single-letter symbols and commands keep it.
        assert tokens(r"Black Hole S s \Phi \phi") == ["black", "hole", "S", "s", r"\Phi", r"\phi"]

    def test_drops_stopwords_and_punctuation(self):
        assert tokens("the entropy of a black hole, as computed.") == ["entropy", "a", "black", "hole", "computed"]

    def test_escapes_are_not_commands(self):
        assert tokens(r"\{\chi_i, \chi_j\} = \delta_{ij}") == [r"\chi_i", r"\chi_j", r"\delta_ij"]

    def test_unbalanced_braces_do_not_crash(self):
        assert tokens(r"x_{ab } } { y") == ["x_ab", "y"]
        assert tokens("x_") == ["x"]
        assert tokens("^2") == ["2"]

    def test_empty(self):
        assert tokens("") == [] and terms("") == []


class TestTerms:
    def test_unigrams_prefixes_and_bigrams(self):
        assert terms(r"\chi_i\chi_j") == [r"\chi_i", r"\chi", r"\chi_i \chi_j", r"\chi_j", r"\chi"]

    def test_all_prefixes_of_fused_atom(self):
        assert terms(r"G_N^{\text{bare}}") == ["G_N^bare", "G", "G_N"]

    def test_phrase_bigrams(self):
        assert "distance conjecture" in terms("the swampland distance conjecture")


class TestLiterals:
    def test_tsvector_escapes_and_positions(self):
        lit = tsvector_literal([r"\chi_i", "it's", r"\chi_i"])
        assert lit == r"'\\chi_i':1,3 'it''s':2"

    def test_tsvector_caps_positions(self):
        lit = tsvector_literal(["x"] * 300)
        assert lit.count(",") == 255  # 256 positions max per lexeme

    def test_tsquery_dedupes_and_ors(self):
        assert tsquery_literal(["a b", r"\chi", "a b"]) == r"'a b' | '\\chi'"


def test_idf_decreases_with_df_and_stays_positive():
    assert idf(1000, 1) > idf(1000, 10) > idf(1000, 900) > 0


def test_normalize_collapses_whitespace():
    assert normalize_latex("a\n\n  b\t c") == "a b c"
