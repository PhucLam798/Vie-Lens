"""Tests for shared cognate filtering helpers."""

from __future__ import annotations

import pandas as pd

from vilens.cognates import (
    cognate_free_ids,
    distance_ok_ids,
    distinct_first_token_ids,
    id_list_sha256,
    surface_distance,
)

TEXT_CFG = {
    "surface_remove_chars": [" ", "-", "'"],
    "surface_vi_replace": {"ph": "f"},
    "cognate_max_distance": 0.5,
}


def test_surface_distance_identical_forms() -> None:
    """Identical normalized forms have zero distance."""
    assert surface_distance("drum", "drum", TEXT_CFG) == 0.0


def test_distance_threshold_equality_is_kept() -> None:
    """The distance filter includes forms exactly at its threshold."""
    concepts = pd.DataFrame([
        {"concept_id": "a", "en": "aa", "fr": "ab", "id": "ab"},
    ])
    _, _, accepted = distance_ok_ids(concepts, TEXT_CFG)
    assert accepted == {"a"}


def test_distinct_first_token_ids_requires_five_unique_values() -> None:
    """Drop repeated and missing first-content token IDs."""
    ids = {
        "ok": {"vi": 1, "en": 2, "zh": 3, "fr": 4, "id": 5},
        "repeat": {"vi": 1, "en": 2, "zh": 3, "fr": 4, "id": 4},
        "missing": {"vi": 1, "en": 2, "zh": 3, "fr": 4, "id": None},
    }
    assert distinct_first_token_ids(ids) == {"ok"}


def test_cognate_free_intersection_is_sorted_and_order_invariant() -> None:
    """Intersect every family and return stable sorted IDs regardless of input order."""
    distance = {"z", "a", "b"}
    families = {"gemma": {"b", "a"}, "qwen": {"a", "b"}, "llama": {"a", "b", "z"}}
    first = cognate_free_ids(distance, families)
    shuffled_families = {
        name: set(reversed(sorted(ids)))
        for name, ids in reversed(list(families.items()))
    }
    assert first == ["a", "b"]
    assert cognate_free_ids(set(reversed(sorted(distance))), shuffled_families) == first


def test_id_list_sha256_is_order_invariant() -> None:
    """Hash canonical sorted ids independently of input ordering."""
    assert id_list_sha256(["b", "a"]) == id_list_sha256(["a", "b"])
