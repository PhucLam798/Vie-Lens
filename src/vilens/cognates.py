"""Deterministic cognate-distance and shared-subset helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .text import normalized_levenshtein, surface_norm


def surface_distance(a: str, b: str, text_cfg: Mapping[str, Any]) -> float:
    """Compute normalized surface distance with the configured filter rules."""
    remove_chars = list(text_cfg["surface_remove_chars"])
    vi_replace = dict(text_cfg["surface_vi_replace"])
    normalized_a = surface_norm(a, vietnamese=False, remove_chars=remove_chars, vi_replace=vi_replace)
    normalized_b = surface_norm(b, vietnamese=False, remove_chars=remove_chars, vi_replace=vi_replace)
    return normalized_levenshtein(normalized_a, normalized_b)


def distance_ok_ids(
    main_concepts: Any,
    text_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], set[str]]:
    """Return en-fr and en-id distances and ids meeting both thresholds."""
    threshold = float(text_cfg["cognate_max_distance"])
    d_en_fr: dict[str, float] = {}
    d_en_id: dict[str, float] = {}
    accepted: set[str] = set()
    for row in main_concepts.itertuples(index=False):
        concept_id = str(row.concept_id)
        fr_distance = surface_distance(str(row.en), str(row.fr), text_cfg)
        id_distance = surface_distance(str(row.en), str(row.id), text_cfg)
        d_en_fr[concept_id] = fr_distance
        d_en_id[concept_id] = id_distance
        if fr_distance >= threshold and id_distance >= threshold:
            accepted.add(concept_id)
    return d_en_fr, d_en_id, accepted


def distinct_first_token_ids(first_ids: Mapping[str, Mapping[str, int | None]]) -> set[str]:
    """Return concepts whose five language first-content token ids are present and distinct."""
    languages = ("vi", "en", "zh", "fr", "id")
    distinct: set[str] = set()
    for concept_id, language_ids in first_ids.items():
        values = [language_ids.get(language) for language in languages]
        if all(value is not None for value in values) and len(set(values)) == len(values):
            distinct.add(concept_id)
    return distinct


def cognate_free_ids(
    distance_ok: set[str],
    distinct_by_family: Mapping[str, set[str]],
) -> list[str]:
    """Return sorted ids passing distance and distinct-token checks in every family."""
    if not distinct_by_family:
        return []
    shared = set.intersection(*(set(ids) for ids in distinct_by_family.values()))
    return sorted(set(distance_ok) & shared)


def id_list_sha256(ids: list[str] | set[str]) -> str:
    """Hash the sorted newline-joined ids without a trailing newline."""
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
