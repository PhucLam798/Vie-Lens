"""Synthetic tests for frozen target, distractor, sampling and E8 helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vilens.derived import (
    e0_sample,
    e8_select,
    form_stats,
    match_distractors,
    compare_directories,
    refuse_nonempty_derived,
    target_string,
)
from vilens.tokenize import TokenClassifier


def byte_level_tokenizer() -> PreTrainedTokenizerFast:
    """Build the same tiny byte-level tokenizer used by the tokenization tests."""
    backend = Tokenizer(models.BPE(
        vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
        merges=[],
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _concept(concept_id: str, vi: str, *, zipf: float = 3.0, collapsed: bool = False) -> dict[str, Any]:
    """Return one small concept row with all target-language forms."""
    return {
        "concept_id": concept_id,
        "vi": vi,
        "vi_nodiac": vi.lower(),
        "en": f"en-{concept_id}",
        "fr": f"fr-{concept_id}",
        "id": f"id-{concept_id}",
        "zh": f"中{concept_id}",
        "zipf_en": zipf,
        "collapsed": collapsed,
    }


def _stats(
    family: str,
    concepts: dict[str, dict[str, Any]],
    counts: dict[str, int],
    first_ids: dict[str, int],
    lang: str = "vi",
    variant: str = "released",
) -> dict[tuple[str, str, str, str], tuple[int, int | None]]:
    """Build the precomputed cache expected by distractor matching."""
    return {
        (family, lang, variant, concept_id): (counts[concept_id], first_ids[concept_id])
        for concept_id in concepts
    }


@pytest.mark.parametrize(
    ("lang", "variant", "expected"),
    [
        ("vi", "released", " Đường Phố"),
        ("en", "released", " Street"),
        ("fr", "released", " Rue"),
        ("id", "released", " Jalan"),
        ("zh", "released", " 街道"),
        ("zh", "nospace", "街道"),
        ("vi", "nodiac", " Duong Pho"),
    ],
)
def test_target_string_all_configured_combinations(lang: str, variant: str, expected: str) -> None:
    """Build each released and derived target without losing Vietnamese case."""
    concept = {
        "vi": "Đường Phố", "en": "Street", "fr": "Rue", "id": "Jalan", "zh": "街道",
    }
    assert target_string(concept, lang, variant) == expected
    if (lang, variant) == ("zh", "nospace"):
        assert not expected.startswith(" ") and expected == concept["zh"]


def test_form_stats_counts_content_tokens_and_first_id() -> None:
    """Summarize a jointly encoded form with whitespace excluded from content."""
    tokenizer = byte_level_tokenizer()
    classifier = TokenClassifier(tokenizer)
    count, first_id = form_stats(tokenizer, classifier, "ab:", " cd")
    assert count == 2
    assert first_id is not None


def test_exact_count_match_is_preferred() -> None:
    """Choose a closer-zipf exact-count candidate over a nearby fallback."""
    concepts = {
        "a": _concept("a", "alpha", zipf=3.0),
        "b": _concept("b", "bravo", zipf=3.1),
        "c": _concept("c", "charlie", zipf=3.0),
    }
    stats = _stats("gemma", concepts, {"a": 2, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3})
    rows = match_distractors(
        concepts, stats, family="gemma", lang="vi", variant="released",
        pool_ids=list(concepts), seed=20260925,
    )
    row_a = next(row for row in rows if row["concept_id"] == "a")
    assert row_a["kind"] == "exact"
    assert row_a["distractor_id"] == "b"


def test_count_fallback_includes_both_equally_near_counts() -> None:
    """Use all candidates tied for the smallest count difference before zipf."""
    concepts = {
        "a": _concept("a", "alpha", zipf=3.0),
        "b": _concept("b", "bravo", zipf=2.5),
        "c": _concept("c", "charlie", zipf=3.5),
    }
    stats = _stats("gemma", concepts, {"a": 2, "b": 1, "c": 3}, {"a": 1, "b": 2, "c": 3})
    rows = match_distractors(
        concepts, stats, family="gemma", lang="vi", variant="released",
        pool_ids=list(concepts), seed=20260925,
    )
    row_a = next(row for row in rows if row["concept_id"] == "a")
    assert row_a["kind"] == "count_fallback"
    assert row_a["n_tied"] == 2
    assert row_a["distractor_id"] in {"b", "c"}


def test_self_identical_targets_and_first_ids_are_excluded() -> None:
    """Exclude self, exact duplicate forms, and candidates sharing the first ID."""
    concepts = {
        "a": _concept("a", "same", zipf=3.0),
        "b": _concept("b", "same", zipf=3.0),
        "c": _concept("c", "other", zipf=3.0),
        "d": _concept("d", "final", zipf=3.0),
    }
    stats = _stats("gemma", concepts, {key: 2 for key in concepts}, {"a": 10, "b": 20, "c": 10, "d": 30})
    rows = match_distractors(
        concepts, stats, family="gemma", lang="vi", variant="released",
        pool_ids=list(concepts), seed=20260925,
    )
    assert next(row for row in rows if row["concept_id"] == "a")["distractor_id"] == "d"
    assert all(row["concept_id"] != row["distractor_id"] for row in rows)


def test_collapsed_distractors_are_excluded_only_for_vi_nodiac() -> None:
    """Keep a collapsed candidate in released matching but exclude it for nodiac."""
    concepts = {
        "a": _concept("a", "alpha", zipf=0.0),
        "b": _concept("b", "bravo", zipf=0.1, collapsed=True),
        "c": _concept("c", "charlie", zipf=4.0),
    }
    stats: dict[tuple[str, str, str, str], tuple[int, int | None]] = {}
    for variant in ("released", "nodiac"):
        stats.update(_stats("gemma", concepts, {key: 2 for key in concepts}, {"a": 1, "b": 2, "c": 3}, variant=variant))
    released = match_distractors(
        concepts, stats, family="gemma", lang="vi", variant="released",
        pool_ids=list(concepts), seed=20260925,
    )
    nodiac = match_distractors(
        concepts, stats, family="gemma", lang="vi", variant="nodiac",
        pool_ids=list(concepts), seed=20260925,
    )
    assert next(row for row in released if row["concept_id"] == "a")["distractor_id"] == "b"
    assert next(row for row in nodiac if row["concept_id"] == "a")["distractor_id"] == "c"


def test_ties_are_stable_when_pool_input_is_shuffled() -> None:
    """Sort ties before seeded draws so shuffled source order has no effect."""
    concepts = {letter: _concept(letter, f"word-{letter}", zipf=2.0) for letter in "abcdefgh"}
    stats = _stats("qwen", concepts, {key: 2 for key in concepts}, {key: i for i, key in enumerate(concepts, 1)})
    kwargs = dict(family="qwen", lang="vi", variant="released", seed=20260925)
    first = match_distractors(concepts, stats, pool_ids=list(concepts), **kwargs)
    second = match_distractors(concepts, stats, pool_ids=list(reversed(concepts)), **kwargs)
    assert first == second
    assert all(row["n_tied"] == 7 for row in first)


def test_e8_select_uses_largest_remainder_and_skips_incompetent_items() -> None:
    """Allocate proportional quotas and honor seeded per-stratum rank order."""
    rows = [
        {"family": "gemma", "stratum": "a", "rank": i, "concept_id": f"a{i}"}
        for i in range(5)
    ] + [
        {"family": "gemma", "stratum": "b", "rank": i, "concept_id": f"b{i}"}
        for i in range(3)
    ]
    competent = {"a0", "a1", "a3", "b0", "b1", "b2", "not-in-order"}
    selected = e8_select(rows, competent, "gemma", max_items=4)
    assert selected == ["a0", "a1", "b0", "b1"]
    assert e8_select(rows, competent, "gemma", max_items=300) == ["a0", "a1", "a3", "b0", "b1", "b2"]
    assert e8_select(rows, competent, "gemma", max_items=4) == selected


def test_e0_sample_is_deterministic_and_distinct() -> None:
    """Sample 50 sorted directions IDs with the registered independent stream."""
    ids = [f"direction-{index:03}" for index in range(132)]
    first = e0_sample(ids, 20260925, 50)
    assert first == e0_sample(list(reversed(ids)), 20260925, 50)
    assert len(first) == len({row["concept_id"] for row in first}) == 50
    assert [row["rank"] for row in first] == list(range(50))


def test_generator_refuses_nonempty_derived_directory(tmp_path: Path) -> None:
    """Exercise the default-mode guard against overwriting frozen manifests."""
    derived_dir = tmp_path / "derived"
    derived_dir.mkdir()
    (derived_dir / "already-frozen.txt").write_text("frozen", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty derived directory"):
        refuse_nonempty_derived(derived_dir)


def test_check_directory_comparison_reports_byte_equality(tmp_path: Path) -> None:
    """Use the same directory comparator as --check on identical generated files."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    (first / "manifest.jsonl").write_bytes(b'{"id":"a"}\n')
    (second / "manifest.jsonl").write_bytes(b'{"id":"a"}\n')
    assert compare_directories(first, second) == (True, [])
