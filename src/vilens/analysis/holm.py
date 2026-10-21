"""Holm–Bonferroni correction for preregistered one-sided families."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_means, percentile_interval


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm-adjusted p-values in the same order as the input family."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be a finite vector in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def holm_table(rows: pd.DataFrame, *, p_column: str = "p_value") -> pd.DataFrame:
    """Add the Holm-adjusted p-value and rejection flag to a family table."""
    if p_column not in rows:
        raise ValueError(f"Missing p-value column {p_column!r}")
    result = rows.copy()
    result["p_holm"] = holm_adjust(result[p_column].to_numpy(dtype=float))
    result["reject_0_05"] = result["p_holm"] <= 0.05
    return result


def window_mean(values: np.ndarray, layers: Sequence[int], window: Sequence[float]) -> np.ndarray:
    """Average the last axis over block outputs in an inclusive depth window."""
    array = np.asarray(values, dtype=np.float64)
    layer_values = np.asarray(layers, dtype=np.int64)
    bounds = np.asarray(window, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != len(layer_values):
        raise ValueError("the last values axis must align with layers")
    if bounds.shape != (2,) or not 0.0 <= bounds[0] < bounds[1] <= 1.0:
        raise ValueError("window must contain increasing normalized-depth bounds")
    if layer_values.size == 0 or np.any(layer_values < 1) or int(layer_values.max()) < 1:
        raise ValueError("layers must contain block outputs")
    n_layers = int(layer_values.max())
    selected = (layer_values / n_layers >= bounds[0]) & (layer_values / n_layers <= bounds[1])
    if not selected.any():
        raise ValueError("the analysis window contains no block outputs")
    return np.mean(array[..., selected], axis=-1, dtype=np.float64)


def window_positive_test(
    concept_curves: np.ndarray,
    *,
    layers: Sequence[int],
    window: Sequence[float],
    indices: np.ndarray,
) -> dict[str, float | int]:
    """Test the OPS-14 W_L window mean against a non-positive null."""
    values = np.asarray(concept_curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("concept_curves must be a non-empty concepts×layers matrix")
    per_concept_window = window_mean(values, layers, window)
    samples = bootstrap_means(per_concept_window[:, None], indices).reshape(-1)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {
            "W_L": float(np.mean(per_concept_window)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "valid_draws": 0,
        }
    low, high = percentile_interval(finite)
    return {
        "W_L": float(np.mean(per_concept_window, dtype=np.float64)),
        "ci_low": low,
        "ci_high": high,
        "p_value": float((1 + np.count_nonzero(finite <= 0.0)) / (1 + finite.size)),
        "valid_draws": int(finite.size),
    }
