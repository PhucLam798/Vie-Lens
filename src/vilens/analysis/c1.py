"""Paired concept resampling helpers for the descriptive C1 analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def paired_selection_draws(
    concept_ids: Sequence[str],
    eligible_ids: Iterable[str],
    indices: np.ndarray,
) -> list[np.ndarray]:
    """Filter each full-main bootstrap draw to its M1-eligible sampled concepts."""
    ordered_ids = tuple(sorted(map(str, concept_ids)))
    eligible = set(map(str, eligible_ids))
    draws = np.asarray(indices, dtype=np.int64)
    if not ordered_ids or draws.ndim != 2 or draws.shape[1] == 0:
        raise ValueError("selection effects need concept IDs and a non-empty draw matrix")
    if draws.min() < 0 or draws.max() >= len(ordered_ids):
        raise ValueError("selection-effect bootstrap index is outside the main concept set")
    eligible_positions = np.asarray(
        [position for position, concept_id in enumerate(ordered_ids) if concept_id in eligible],
        dtype=np.int64,
    )
    if eligible_positions.size == 0:
        raise ValueError("selection-effect eligible set is empty")
    eligible_mask = np.zeros(len(ordered_ids), dtype=bool)
    eligible_mask[eligible_positions] = True
    return [draw[eligible_mask[draw]] for draw in draws]
