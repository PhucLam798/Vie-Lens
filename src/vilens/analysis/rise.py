"""Matched-distractor rise curves and H1/H2 decision quantities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def concept_rise_curves(
    deltas: pd.DataFrame,
    *,
    early_depth_max: float = 0.15,
    value_column: str = "delta",
) -> pd.DataFrame:
    """Subtract each concept's early-block baseline from its blockwise delta.

    Layer 0 is excluded. The early set is the configured fraction of decoder
    depth and must contain at least one block output.
    """
    required = {"model", "format", "concept_id", "target_lang", "target_variant", "layer", value_column}
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Delta table is missing columns: {sorted(missing)}")
    frame = deltas.loc[pd.to_numeric(deltas["layer"]) >= 1].copy()
    if frame.empty:
        raise ValueError("No block-output layers are available")
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    frame["n_layers"] = frame.groupby("model")["layer"].transform("max").astype(int)
    frame["is_early"] = frame["layer"] / frame["n_layers"] <= float(early_depth_max)
    group_keys = ["model", "format", "concept_id", "target_lang", "target_variant"]
    baselines = (
        frame.loc[frame["is_early"]]
        .groupby(group_keys, sort=True, dropna=False)[value_column]
        .mean()
        .rename("early_baseline")
    )
    if baselines.empty:
        raise ValueError("The configured early-layer range contains no block outputs")
    frame = frame.join(baselines, on=group_keys, validate="many_to_one")
    has_any_score = frame.groupby(group_keys, sort=True, dropna=False)[value_column].transform(
        "count"
    ).gt(0)
    if frame.loc[has_any_score, "early_baseline"].isna().any():
        raise ValueError("At least one concept/readout has no early-layer baseline")
    frame["rise"] = frame[value_column] - frame["early_baseline"]
    return frame.drop(columns=["is_early"])


def mean_rise_curves(per_concept: pd.DataFrame) -> pd.DataFrame:
    """Average concept-level rise curves by model, format, readout and layer."""
    keys = ["model", "format", "target_lang", "target_variant", "layer"]
    return (
        per_concept.groupby(keys, sort=True, dropna=False)["rise"]
        .agg(mean_rise="mean", n_concepts="count")
        .reset_index()
    )


def maxima_by_language(curves: pd.DataFrame, *, column: str = "mean_rise") -> dict[str, float]:
    """Return each language's maximum rise over block outputs."""
    required = {"target_lang", "layer", column}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Rise curves are missing columns: {sorted(missing)}")
    block_curves = curves.loc[pd.to_numeric(curves["layer"]) >= 1]
    return {
        str(language): float(group[column].max())
        for language, group in block_curves.groupby("target_lang", sort=True)
    }


def decision_rule_quantities(maxima: Mapping[str, float]) -> dict[str, float | bool | None]:
    """Compute proposed H1 contrast/ratio and H2 positive 1.5× band quantities."""
    required = {"vi", "en", "zh", "fr", "id"}
    missing = required - set(maxima)
    if missing:
        raise ValueError(f"Missing five-way language maxima: {sorted(missing)}")
    values = {language: float(maxima[language]) for language in required}
    control_mean = (values["fr"] + values["id"]) / 2.0
    h1_q = values["en"] - 2.0 * control_mean
    ratio = values["en"] / control_mean if control_mean > 0.0 else None
    minimum = min(values.values())
    maximum = max(values.values())
    band_ratio = maximum / minimum if minimum > 0.0 else None
    return {
        "h1_q": h1_q,
        "h1_ratio": ratio,
        "h1_rise_holds": h1_q >= 0.0,
        "h2_band_ratio": band_ratio,
        "h2_band_holds": bool(minimum > 0.0 and band_ratio is not None and band_ratio <= 1.5),
    }
