"""Scale-free landmarks on concept-mean E3 curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def concept_layer(
    mean_scores: Mapping[str, Sequence[float]],
    *,
    layers: Sequence[int],
    threshold: float,
) -> int | None:
    """Find the first block where any readout reaches a fraction of final vi p."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if "vi" not in mean_scores:
        raise ValueError("Vietnamese scores are required for the anchor")
    layer_values = np.asarray(layers, dtype=int)
    if layer_values.ndim != 1 or len(layer_values) == 0 or np.any(layer_values < 1):
        raise ValueError("layers must be a non-empty vector of block indices")
    vi = np.asarray(mean_scores["vi"], dtype=np.float64)
    if vi.shape != layer_values.shape:
        raise ValueError("Vietnamese curve does not align with layers")
    final_anchor = float(vi[-1])
    cutoff = final_anchor + float(np.log(threshold))
    eligible: list[int] = []
    for language, values in mean_scores.items():
        curve = np.asarray(values, dtype=np.float64)
        if curve.shape != layer_values.shape:
            raise ValueError(f"Curve for {language} does not align with layers")
        eligible.extend(layer_values[np.flatnonzero(curve >= cutoff)].tolist())
    return min(eligible) if eligible else None


def probability_concept_layer(
    mean_probabilities: Mapping[str, Sequence[float]],
    *,
    layers: Sequence[int],
    threshold: float,
) -> int | None:
    """Find the first block reaching a fraction of the final vi mean probability."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if "vi" not in mean_probabilities:
        raise ValueError("Vietnamese probabilities are required for the anchor")
    layer_values = np.asarray(layers, dtype=np.int64)
    if layer_values.ndim != 1 or layer_values.size == 0 or np.any(layer_values < 1):
        raise ValueError("layers must be a non-empty vector of block indices")
    vi = np.asarray(mean_probabilities["vi"], dtype=np.float64)
    if vi.shape != layer_values.shape or np.any(vi < 0.0):
        raise ValueError("Vietnamese probability curve must be non-negative and align with layers")
    cutoff = float(vi[-1]) * float(threshold)
    eligible: list[int] = []
    for language, values in mean_probabilities.items():
        curve = np.asarray(values, dtype=np.float64)
        if curve.shape != layer_values.shape or np.any(curve < 0.0):
            raise ValueError(f"Probability curve for {language} must be non-negative and align with layers")
        eligible.extend(layer_values[np.flatnonzero(curve >= cutoff)].tolist())
    return min(eligible) if eligible else None


def landmarks(
    mean_scores: Mapping[str, Sequence[float]],
    mean_rise: Mapping[str, Sequence[float]],
    *,
    layers: Sequence[int],
    n_layers: int,
    readout_order: Sequence[str] = ("en", "zh", "fr", "id"),
    thresholds: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, int | float | str | None]:
    """Compute L_concept sensitivities, L_peak, L_cross, and pivot summaries."""
    block_layers = np.asarray(layers, dtype=int)
    if (
        np.any(block_layers < 1)
        or int(block_layers.max()) != int(n_layers)
        or np.any(np.diff(block_layers) <= 0)
    ):
        raise ValueError("landmark layers must be block outputs 1..N")
    threshold_values = tuple(map(float, thresholds))
    if not threshold_values or any(not 0.0 < value < 1.0 for value in threshold_values):
        raise ValueError("landmark thresholds must be fractions strictly between zero and one")
    for language, curve in mean_rise.items():
        if np.asarray(curve).shape != block_layers.shape:
            raise ValueError(f"Rise curve for {language} does not align with layers")
    non_vi = [language for language in readout_order if language in mean_rise]
    if not non_vi:
        raise ValueError("At least one non-Vietnamese readout is required")
    maxima = {language: float(np.max(mean_rise[language])) for language in non_vi}
    pivot_language = max(non_vi, key=lambda language: maxima[language])
    pivot_curve = np.asarray(mean_rise[pivot_language], dtype=np.float64)
    peak_index = int(np.argmax(pivot_curve))
    l_peak = int(block_layers[peak_index])
    vi_curve = np.asarray(mean_rise["vi"], dtype=np.float64)
    crossing = np.flatnonzero((block_layers > l_peak) & (vi_curve > pivot_curve))
    l_cross = int(block_layers[crossing[0]]) if crossing.size else None
    concept_values = {
        f"L_concept_{int(threshold * 100)}": concept_layer(
            mean_scores, layers=block_layers, threshold=threshold
        )
        for threshold in threshold_values
    }
    primary_concept = concept_values["L_concept_10"]
    width = (
        (l_cross - primary_concept) / float(n_layers)
        if l_cross is not None and primary_concept is not None
        else None
    )
    return {
        **concept_values,
        "L_peak": l_peak,
        "pivot_language": pivot_language,
        "L_cross": l_cross,
        "pivot_width": width,
        "pivot_magnitude": maxima[pivot_language],
    }
