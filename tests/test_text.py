"""Parity examples for the preparation repository's text functions."""

from __future__ import annotations

import pytest

from vilens.text import normalized_levenshtein, strip_diacritics, surface_norm


def test_strip_diacritics_examples() -> None:
    """Match lowercased and case-preserving Vietnamese transliteration."""
    assert strip_diacritics("Đường phố") == "duong pho"
    assert strip_diacritics("Tiếng Việt", preserve_case=True) == "Tieng Viet"
    assert strip_diacritics("ĐỨNG", preserve_case=True) == "DUNG"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("", "", 0.0), ("a", "", 1.0), ("same", "same", 0.0), ("cat", "cut", 1 / 3)],
)
def test_normalized_levenshtein(left: str, right: str, expected: float) -> None:
    """Check boundary cases and one substitution for normalized edit distance."""
    assert normalized_levenshtein(left, right) == pytest.approx(expected)


def test_surface_norm_examples() -> None:
    """Match the configured Vietnamese and non-Vietnamese surface normalization."""
    options = {"remove_chars": [" ", "-", "'"], "vi_replace": {"ph": "f"}}
    assert surface_norm("cà phê", vietnamese=True, **options) == "cafe"
    assert surface_norm("ra-đi-ô", vietnamese=True, **options) == "radio"
    assert surface_norm("café", vietnamese=False, **options) == "cafe"
