"""Tests for v1.4 answer normalization and A10 classification."""

from __future__ import annotations

from vilens.score import classify, matches, source_groups
from vilens.text import vi_orth_key


def _concept(
    concept_id: str,
    vi: str,
    *,
    en: str = "house",
    zh: str = "房子",
    fr: str = "maison",
    indonesian: str = "rumah",
    ru: str = "дом",
    vi_alts: list[str] | None = None,
) -> dict[str, object]:
    """Create a small hand-written concept row for scorer tests."""
    return {
        "concept_id": concept_id,
        "vi": vi,
        "vi_alts": vi_alts or [],
        "en": en,
        "zh": zh,
        "fr": fr,
        "id": indonesian,
        "ru": ru,
    }


def test_word_boundary_prefix_matching() -> None:
    """Accept punctuation boundaries and reject alphanumeric continuations."""
    assert matches("nước.", "nước")
    assert not matches("nướcngoài", "nước")


def test_vi_alternatives_match() -> None:
    """Count a released Vietnamese alternative as a VI answer."""
    concept = _concept("a", "nước", vi_alts=["nước non"])
    assert classify(
        "nước non!", concept, condition="diac", format="repetition", groups={}, concepts={"a": concept}
    ) == "VI"


def test_source_group_match_applies_only_to_translation() -> None:
    """Use shared-Russian-source Vietnamese forms for translation only."""
    first = _concept("a", "xe")
    second = _concept("b", "ô tô")
    concepts = {"a": first, "b": second}
    groups = source_groups(concepts)
    assert classify(
        "ô tô", first, condition="diac", format="translation", groups=groups, concepts=concepts
    ) == "VI"
    assert classify(
        "ô tô", first, condition="diac", format="repetition", groups=groups, concepts=concepts
    ) == "DAMAGE"


def test_nodiac_strips_diacritics_for_vietnamese_forms() -> None:
    """Match a diacritic-bearing answer to the frozen no-diacritic form."""
    concept = _concept("a", "nước")
    assert classify(
        "nước", concept, condition="nodiac", format="repetition", groups={}, concepts={"a": concept}
    ) == "VI"


def test_cjk_form_uses_plain_prefix_matching() -> None:
    """Allow a Chinese form prefix without a Latin word-boundary rule."""
    assert matches("中文词", "中文")


def test_language_order_and_damage_classification() -> None:
    """Check EN, OTHER-LANG, and fallback DAMAGE categories."""
    concept = _concept("a", "nước", en="house", fr="maison")
    concepts = {"a": concept}
    kwargs = {"condition": "diac", "format": "repetition", "groups": {}, "concepts": concepts}
    assert classify("house.", concept, **kwargs) == "EN"
    assert classify("maison!", concept, **kwargs) == "OTHER-LANG"
    assert classify("unrelated", concept, **kwargs) == "DAMAGE"
    assert classify("", concept, **kwargs) == "DAMAGE"


def test_source_groups_normalize_russian_yo_and_case() -> None:
    """Group Russian sources after NFC, case folding, and yo normalization."""
    concepts = {"a": _concept("a", "x", ru="Ёлка"), "b": _concept("b", "y", ru="елка")}
    assert source_groups(concepts) == {"елка": ["a", "b"]}


def test_vietnamese_orthographic_equivalence_is_vi_only() -> None:
    """Match v1.2 Vietnamese spelling variants while preserving other classes."""
    equivalent_pairs = (
        ("hoà bình", "hòa bình"),
        ("lý do", "lí do"),
        ("thuỷ", "thủy"),
    )
    for older, modern in equivalent_pairs:
        assert vi_orth_key(older) == vi_orth_key(modern)
        concept = _concept("a", modern)
        assert classify(
            older,
            concept,
            condition="diac",
            format="repetition",
            groups={},
            concepts={"a": concept},
        ) == "VI"

    assert vi_orth_key("kỳ nhông") != vi_orth_key("kì giông")
    concept = _concept("a", "kỳ nhông")
    assert classify(
        "kì giông",
        concept,
        condition="diac",
        format="repetition",
        groups={},
        concepts={"a": concept},
    ) == "DAMAGE"
    assert classify(
        "house.",
        _concept("b", "nhà"),
        condition="diac",
        format="repetition",
        groups={},
        concepts={"b": _concept("b", "nhà")},
    ) == "EN"
