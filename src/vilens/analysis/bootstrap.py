"""Deterministic concept-level bootstrap helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..derived import stream


def bootstrap_key(item_set: str, model: str, fmt: str, *, template: str) -> str:
    """Format the OPS-14 resampling key for one named item set."""
    try:
        key = template.format(item_set=str(item_set), model=str(model), format=str(fmt))
    except (KeyError, ValueError) as exc:
        raise ValueError("bootstrap key template must contain item_set, model, and format") from exc
    if not key:
        raise ValueError("bootstrap key must not be empty")
    return key


def concept_indices(
    concept_ids: Sequence[str],
    *,
    resamples: int = 1_000,
    seed: int = 20260925,
    key: str,
) -> np.ndarray:
    """Draw sorted concept indices with replacement under the registered RNG scheme."""
    ids = sorted(map(str, concept_ids))
    if not ids:
        raise ValueError("Cannot bootstrap an empty concept set")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = stream(int(seed), str(key))
    return rng.integers(0, len(ids), size=(int(resamples), len(ids)), endpoint=False)


def bootstrap_means(values: np.ndarray, indices: np.ndarray, *, chunk_size: int = 32) -> np.ndarray:
    """Return bootstrap means over axis 0 using a bounded temporary per chunk."""
    array = np.asarray(values, dtype=np.float64)
    draws = np.asarray(indices, dtype=np.int64)
    if array.ndim < 1 or draws.ndim != 2 or draws.shape[1] == 0:
        raise ValueError("values must have a concept axis and indices must be a non-empty matrix")
    if draws.min() < 0 or draws.max() >= array.shape[0]:
        raise ValueError("bootstrap index is outside the concept axis")
    output = np.empty((draws.shape[0], *array.shape[1:]), dtype=np.float64)
    for start in range(0, draws.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), draws.shape[0])
        output[start:stop] = array[draws[start:stop]].mean(axis=1)
    return output


def bootstrap_statistics(
    values: np.ndarray,
    indices: np.ndarray,
    statistic: Callable[[np.ndarray], Any],
    *,
    chunk_size: int = 32,
) -> np.ndarray:
    """Apply a scalar/vector statistic to shared concept-resampled arrays."""
    means = bootstrap_means(values, indices, chunk_size=chunk_size)
    return np.asarray([statistic(sample) for sample in means])


def percentile_interval(samples: np.ndarray, *, level: float = 0.95) -> tuple[float, float]:
    """Return a two-sided percentile interval from finite bootstrap samples."""
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or not 0.0 < level < 1.0:
        return float("nan"), float("nan")
    tail = (1.0 - level) * 50.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def percentile_interval_with_count(
    samples: Sequence[float | int | None] | np.ndarray, *, level: float = 0.95
) -> tuple[float, float, int]:
    """Return a percentile interval and its count of finite, defined draws."""
    values = np.asarray(
        [np.nan if value is None else float(value) for value in samples], dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    low, high = percentile_interval(finite, level=level)
    return low, high, int(finite.size)


def one_sided_positive_p(samples: np.ndarray) -> float:
    """Estimate P(null non-positive) with the registered plus-one correction."""
    values = np.asarray(samples, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float((1 + np.count_nonzero(finite <= 0.0)) / (1 + finite.size))
