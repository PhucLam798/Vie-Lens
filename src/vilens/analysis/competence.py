"""A10 generation scoring and shared-Russian-source flags."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..score import classify


def competence_rows(
    generations: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Score generations through ``score.classify`` and flag shared ru sources."""
    groups: dict[str, list[str]] = defaultdict(list)
    for concept_id, concept in concepts.items():
        ru = concept.get("ru")
        if isinstance(ru, str) and ru.strip():
            key = unicodedata.normalize("NFC", ru).casefold().replace("ё", "е")
            groups[key].append(str(concept_id))
    shared = {key: sorted(ids) for key, ids in groups.items() if len(ids) > 1}
    by_concept: dict[str, str | None] = {}
    for concept_id, concept in concepts.items():
        ru = concept.get("ru")
        by_concept[str(concept_id)] = (
            unicodedata.normalize("NFC", ru).casefold().replace("ё", "е")
            if isinstance(ru, str) and ru.strip()
            else None
        )
    rows: list[dict[str, Any]] = []
    for generation in generations:
        concept_id = str(generation["concept_id"])
        concept = concepts[concept_id]
        fmt = str(generation["format"])
        condition = str(generation.get("condition", "diac"))
        key = by_concept.get(concept_id)
        flagged = fmt == "translation" and key in shared
        group_ids = shared.get(key, []) if flagged and key is not None else []
        output_class = classify(
            str(generation.get("answer", generation.get("generation", ""))),
            concept,
            condition=condition,
            format=fmt,
            groups=shared,
            concepts=concepts,
        )
        rows.append(
            {
                **dict(generation),
                "concept_id": concept_id,
                "competence_class": output_class,
                "competent": output_class == "VI",
                "shared_ru_source": bool(flagged),
                "shared_ru_group_n": len(group_ids),
            }
        )
    return pd.DataFrame.from_records(rows)


def competence_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Count A10 classes by model and format, retaining shared-source counts."""
    keys = [key for key in ("model", "format") if key in scored.columns]
    if not keys:
        raise ValueError("Generation rows need model and format columns")
    summary = (
        scored.groupby(keys + ["competence_class"], dropna=False, sort=True)
        .agg(n=("concept_id", "size"), shared_ru_source=("shared_ru_source", "sum"))
        .reset_index()
    )
    return summary

