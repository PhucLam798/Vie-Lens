"""Outcome-independent matching and preregistered E6 analysis helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ..derived import stream
from .bootstrap import percentile_interval
from .holm import holm_adjust

LANGUAGES = ("vi", "en", "zh", "fr", "id")
FORMATS = ("repetition", "translation")
PRIMARY_VARIANTS = {"vi": "released", "en": "released", "zh": "nospace", "fr": "released", "id": "released"}
MATCH_COVARIATES = ("zipf_en", "n_syllables", "concreteness", "ntok_vi_mean")


def expected_s71_row_grid(
    format_concept_counts: Mapping[str, Mapping[str, int]], *, n_layers: int
) -> tuple[dict[str, int], int]:
    """Return OPS-33 row expectations for the concepts available in each format."""
    if set(format_concept_counts) != set(FORMATS):
        raise ValueError(f"format counts must contain exactly {FORMATS}")
    if type(n_layers) is not int or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    rows_by_format: dict[str, int] = {}
    for fmt in FORMATS:
        counts = format_concept_counts[fmt]
        if set(counts) != {"sino", "nonsino"}:
            raise ValueError(f"{fmt} counts must contain sino and nonsino")
        values = [counts[stratum] for stratum in ("sino", "nonsino")]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("format concept counts must be non-negative integers")
        rows_by_format[fmt] = sum(values) * len(LANGUAGES) * n_layers
    return rows_by_format, sum(rows_by_format.values())


@dataclass(frozen=True)
class FitDiagnostics:
    """Validity and optimizer details for one OPS-31 model fit."""

    valid: bool
    converged: bool
    method_used: str
    tries: int
    variance_at_boundary: bool
    warning_categories: dict[str, int]
    failure_reasons: tuple[str, ...]
    elapsed_seconds: float
    random_intercept_variance: float | None
    boundary_rule_ran: bool = False
    boundary_rule_changed_fit: bool = False
    llf_kept: float | None = None
    llf_first_valid: float | None = None


@dataclass(frozen=True)
class MixedLMFit:
    """A selected valid statsmodels result, or a not-estimable fit record."""

    result: Any | None
    diagnostics: FitDiagnostics

    @property
    def params(self) -> Any:
        """Expose statsmodels parameters for existing warm-start call sites."""
        return self.result.params if self.result is not None else pd.Series(dtype=np.float64)

    @property
    def converged(self) -> bool:
        """Return the convergence flag of the retained candidate."""
        return bool(self.diagnostics.converged)


def read_matching_covariates(path: str, *, repo_root: str | None = None) -> pd.DataFrame:
    """Read only columns needed for the frozen covariate-only match.

    This helper deliberately accepts one TSV path and rejects paths under the
    result tree. It does not receive or inspect LA, rise, delta, or result files.
    """
    from pathlib import Path

    source = Path(path).resolve()
    if source.suffix.lower() != ".tsv":
        raise ValueError("matching input must be the frozen concepts.tsv file")
    if repo_root is not None:
        results = (Path(repo_root).resolve() / "results").resolve()
        if source == results or results in source.parents:
            raise ValueError("the matching stage may not read anything under results/")
    columns = [
        "concept_id", "split", "m1_extension", "stratum", "native_strict",
        "zipf_en", "n_syllables", "concreteness",
        "ntok_gemma_vi", "ntok_qwen_vi", "ntok_llama_vi",
    ]
    frame = pd.read_csv(source, sep="\t", skiprows=1, usecols=columns, low_memory=False)
    frame["concept_id"] = frame["concept_id"].astype(str)
    token_columns = ["ntok_gemma_vi", "ntok_qwen_vi", "ntok_llama_vi"]
    for column in token_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ntok_vi_mean"] = frame[token_columns].mean(axis=1, skipna=False)
    return frame


def complete_case_concepts(
    frame: pd.DataFrame, *, expected_counts: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """Select the frozen complete-case main sino/nonsino concept rows."""
    required = {"concept_id", "split", "m1_extension", "stratum", *MATCH_COVARIATES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"matching covariate table lacks {sorted(missing)}")
    extension = _bool_values(frame["m1_extension"])
    selected = frame.loc[
        frame["split"].astype(str).eq("test")
        & ~extension
        & frame["stratum"].astype(str).isin(("sino", "nonsino"))
    ].copy()
    for name in MATCH_COVARIATES:
        selected[name] = pd.to_numeric(selected[name], errors="coerce")
    selected = selected.dropna(subset=list(MATCH_COVARIATES)).copy()
    selected["stratum"] = selected["stratum"].astype(str)
    if selected["concept_id"].duplicated().any():
        raise ValueError("complete-case concept IDs must be unique")
    counts = selected["stratum"].value_counts().to_dict()
    expected = {"nonsino": 688, "sino": 469} if expected_counts is None else dict(expected_counts)
    if expected and counts != expected:
        raise AssertionError(f"complete-case strata differ from OPS-28: {counts}")
    return selected.sort_values("concept_id", kind="mergesort").reset_index(drop=True)


def standardized_covariates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized covariates and the pooled complete-case SD vector."""
    values = frame.loc[:, MATCH_COVARIATES].to_numpy(dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("matching covariates must be a finite matrix")
    scales = values.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("matching covariates must have positive pooled SD")
    return (values - values.mean(axis=0)) / scales, scales


def match_without_replacement(
    frame: pd.DataFrame,
    *,
    seed: int = 20260925,
    stream_key: str = "e6_match",
    caliper: float = 0.2,
    expected_counts: Mapping[str, int] | None = None,
    enforce_reference_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Greedily pair randomized sino rows to nearest eligible unused nonsino rows."""
    expected = expected_counts
    if expected is None:
        expected = None if enforce_reference_counts else {}
    complete = complete_case_concepts(frame, expected_counts=expected)
    z, _scales = standardized_covariates(complete)
    coordinates = {str(cid): z[i] for i, cid in enumerate(complete["concept_id"].astype(str))}
    stratum_by_id = complete.set_index("concept_id")["stratum"].astype(str).to_dict()
    sino_ids = sorted(cid for cid, group in stratum_by_id.items() if group == "sino")
    non_ids = sorted(cid for cid, group in stratum_by_id.items() if group == "nonsino")
    rng = stream(int(seed), str(stream_key))
    randomized_sino = [sino_ids[int(i)] for i in rng.permutation(len(sino_ids))]
    available = set(non_ids)
    pairs: list[dict[str, str]] = []
    for sino_id in randomized_sino:
        candidates: list[tuple[float, str]] = []
        for nonsino_id in sorted(available):
            difference = coordinates[sino_id] - coordinates[nonsino_id]
            if np.all(np.abs(difference) <= float(caliper)):
                candidates.append((float(np.linalg.norm(difference)), nonsino_id))
        if not candidates:
            continue
        _distance, nonsino_id = min(candidates, key=lambda item: (item[0], item[1]))
        available.remove(nonsino_id)
        pairs.append(
            {
                "pair_id": f"pair_{len(pairs) + 1:04d}",
                "sino_concept_id": sino_id,
                "nonsino_concept_id": nonsino_id,
            }
        )
    pair_table = pd.DataFrame.from_records(
        pairs, columns=("pair_id", "sino_concept_id", "nonsino_concept_id")
    )
    paired_ids = set(pair_table.get("sino_concept_id", pd.Series(dtype=str))) | set(
        pair_table.get("nonsino_concept_id", pd.Series(dtype=str))
    )
    paired = complete.loc[complete["concept_id"].astype(str).isin(paired_ids)].copy()
    balances: list[dict[str, Any]] = []
    for column in MATCH_COVARIATES:
        full_sino = complete.loc[complete["stratum"].eq("sino"), column].to_numpy(dtype=np.float64)
        full_non = complete.loc[complete["stratum"].eq("nonsino"), column].to_numpy(dtype=np.float64)
        matched_sino = paired.loc[paired["stratum"].eq("sino"), column].to_numpy(dtype=np.float64)
        matched_non = paired.loc[paired["stratum"].eq("nonsino"), column].to_numpy(dtype=np.float64)
        balances.append(
            {
                "covariate": column,
                "SMD_before": _smd(full_sino, full_non),
                "SMD_after": _smd(matched_sino, matched_non),
                "n_sino": int(len(matched_sino)),
                "n_nonsino": int(len(matched_non)),
            }
        )
    balance = pd.DataFrame.from_records(
        balances, columns=("covariate", "SMD_before", "SMD_after", "n_sino", "n_nonsino")
    )
    return pair_table, balance, int(len(sino_ids) - len(pair_table))


def _smd(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a standardized mean difference without returning group means."""
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / (len(left) + len(right) - 2))
    return float((left.mean() - right.mean()) / pooled) if pooled > 0 else float("nan")


def _bool_values(values: pd.Series) -> pd.Series:
    """Parse native booleans and common TSV boolean strings without truthiness bugs."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.notna() & ~normalized.isin(("true", "false", "1", "0"))
    if invalid.any():
        raise ValueError("boolean metadata contains values other than true/false/1/0")
    return normalized.isin(("true", "1"))


def per_concept_response(deltas: pd.DataFrame, *, early_depth_max: float = 0.15) -> pd.DataFrame:
    """Apply OPS-28-2 within concept × language × target variant × format."""
    required = {"model", "format", "concept_id", "target_lang", "target_variant", "layer", "value"}
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"delta rows lack {sorted(missing)}")
    frame = deltas.loc[:, [
        "model", "format", "concept_id", "target_lang", "target_variant", "layer", "value",
    ]].copy()
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
    frame = frame.loc[frame["layer"].ge(1)].copy()
    if frame.empty:
        raise ValueError("no block-layer delta rows")
    frame["n_layers"] = frame.groupby("model", sort=False)["layer"].transform("max").astype(int)
    frame["is_early"] = frame["layer"] / frame["n_layers"] <= float(early_depth_max)
    keys = ["model", "format", "concept_id", "target_lang", "target_variant"]
    baseline = (
        frame.loc[frame["is_early"]]
        .groupby(keys, sort=True, dropna=False)["value"]
        .mean()
        .rename("early_mean_delta")
    )
    if baseline.empty:
        raise ValueError("OPS-28 early-layer set is empty")
    frame = frame.join(baseline, on=keys, validate="many_to_one")
    if frame["early_mean_delta"].isna().any():
        raise ValueError("a concept/readout lacks an early-layer baseline")
    frame["rise"] = frame["value"] - frame["early_mean_delta"]
    result = frame.drop(columns=["is_early", "early_mean_delta", "n_layers"])
    # The aggregate rise curve is defined as the concept mean of this exact table.
    aggregate = result.groupby(
        ["model", "format", "target_lang", "target_variant", "layer"],
        sort=True,
        dropna=False,
    )["rise"].mean()
    recomputed = result.groupby(
        ["model", "format", "target_lang", "target_variant", "layer"],
        sort=True,
        dropna=False,
    )["rise"].agg("mean")
    if not np.allclose(aggregate.to_numpy(), recomputed.to_numpy(), rtol=0.0, atol=1e-9):
        raise AssertionError("concept mean does not equal the aggregate rise curve")
    return result


def compare_with_stored_rise(response: pd.DataFrame, stored: pd.DataFrame) -> float:
    """Return the maximum absolute difference from stored `rise_main` rows."""
    keys = ["model", "format", "concept_id", "target_lang", "target_variant", "layer"]
    required = set(keys) | {"value"}
    if not required.issubset(response.columns) or not required.issubset(stored.columns):
        raise ValueError("response and stored rise need common keys and value")
    left = response[keys + ["rise"]].rename(columns={"rise": "computed"})
    right = stored[keys + ["value"]].rename(columns={"value": "stored"})
    merged = left.merge(right, how="outer", on=keys, validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError("computed response keys differ from stored rise_main")
    diff = np.abs(merged["computed"].to_numpy(dtype=np.float64) - merged["stored"].to_numpy(dtype=np.float64))
    return float(diff.max(initial=0.0))


def assert_concept_mean_matches_aggregate(response: pd.DataFrame, stored: pd.DataFrame, *, atol: float = 1e-9) -> float:
    """Assert the mean concept response equals the stored aggregate rise rows."""
    keys = ["model", "format", "target_lang", "target_variant", "layer"]
    if "rise" not in response or not set(keys + ["value"]).issubset(stored.columns):
        raise ValueError("aggregate rise check requires response.rise and stored.value")
    left = response.groupby(keys, sort=True, dropna=False)["rise"].mean().rename("computed")
    right = stored.groupby(keys, sort=True, dropna=False)["value"].mean().rename("stored")
    merged = left.to_frame().join(right, how="outer", validate="one_to_one")
    if merged.isna().any().any():
        raise AssertionError("computed and stored aggregate rise keys differ")
    difference = np.abs(merged["computed"].to_numpy(dtype=np.float64) - merged["stored"].to_numpy(dtype=np.float64))
    maximum = float(difference.max(initial=0.0))
    if maximum > float(atol):
        raise AssertionError(f"concept mean differs from aggregate rise by {maximum:.12g} (atol={atol})")
    return maximum


def stratified_draws(
    ids_by_stratum: Mapping[str, Sequence[str]],
    *,
    resamples: int,
    seed: int,
    key: str,
) -> list[list[tuple[str, str]]]:
    """Draw fixed stratum sizes with replacement from one independent key stream."""
    if not ids_by_stratum or any(not ids for ids in ids_by_stratum.values()):
        raise ValueError("each bootstrap stratum must contain at least one concept")
    rng = stream(int(seed), str(key))
    groups = {stratum: sorted(map(str, ids)) for stratum, ids in ids_by_stratum.items()}
    draws: list[list[tuple[str, str]]] = []
    for _ in range(int(resamples)):
        sample: list[tuple[str, str]] = []
        for stratum in sorted(groups):
            ids = groups[stratum]
            selected = rng.integers(0, len(ids), size=len(ids), endpoint=False)
            sample.extend((stratum, ids[int(position)]) for position in selected)
        draws.append(sample)
    return draws


def expand_draw_groups(frame: pd.DataFrame, draw: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Expand a concept draw so duplicate samples receive distinct MixedLM groups."""
    if "concept_id" not in frame or "stratum" not in frame:
        raise ValueError("bootstrap rows need concept_id and stratum")
    draw_frame = pd.DataFrame(
        [
            {"concept_id": str(concept_id), "stratum": str(stratum), "_draw_group": f"draw_{i:05d}"}
            for i, (stratum, concept_id) in enumerate(draw)
        ]
    )
    expanded = draw_frame.merge(frame, on=["concept_id", "stratum"], how="inner", validate="many_to_many")
    if expanded.empty:
        raise ValueError("bootstrap draw contains no rows present in the analysis frame")
    expanded["concept_group"] = expanded.pop("_draw_group").astype(str)
    return expanded


def two_sided_bootstrap_p(samples: Sequence[float]) -> float:
    """Compute the OPS-28 plus-one, two-sided bootstrap p-value."""
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return float("nan")
    tails = min(int(np.count_nonzero(values <= 0.0)), int(np.count_nonzero(values >= 0.0)))
    return float(min(1.0, 2.0 * (1 + tails) / (1 + count)))


def interval_class(low: float, high: float) -> str:
    """Classify a zero-excluding interval as + or −, otherwise 0."""
    if np.isfinite(low) and np.isfinite(high) and low > 0:
        return "+"
    if np.isfinite(low) and np.isfinite(high) and high < 0:
        return "−"
    return "0"


def unresolved_from_classes(full: Mapping[str, str], matched: Mapping[str, str]) -> bool:
    """Return whether any model/instrument classification changes after matching."""
    keys = set(full) | set(matched)
    return any(full.get(key) != matched.get(key) for key in keys)


def pooled_format_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure that analysis rows contain only the two frozen task formats."""
    selected = frame.loc[frame["format"].astype(str).isin(FORMATS)].copy()
    if selected.empty:
        raise ValueError("no repetition or translation rows")
    return selected


def dzh_format_values(
    la_zh: pd.DataFrame,
    *,
    n_layers: int,
    window: Sequence[float] = (0.25, 0.75),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute format-specific D_zh and return per-layer values for bootstrap use."""
    required = {"format", "concept_id", "stratum", "layer", "value"}
    if not required.issubset(la_zh.columns):
        raise ValueError(f"LA_zh table lacks {sorted(required - set(la_zh.columns))}")
    frame = pooled_format_rows(la_zh)
    frame = frame.loc[pd.to_numeric(frame["layer"], errors="raise").ge(1)].copy()
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
    low, high = map(float, window)
    eligible = sorted(
        layer for layer in frame["layer"].unique()
        if low <= int(layer) / int(n_layers) <= high
    )
    if not eligible:
        raise ValueError("no passing LA_zh layers in the frozen depth window")
    rows: list[dict[str, Any]] = []
    wide_by_format: dict[str, np.ndarray] = {}
    for fmt in FORMATS:
        selected = frame.loc[frame["format"].astype(str).eq(fmt) & frame["layer"].isin(eligible)]
        pivot = selected.pivot(index="concept_id", columns="layer", values="value")
        if not set(eligible).issubset(pivot.columns):
            eligible_fmt = [layer for layer in eligible if layer in pivot.columns]
        else:
            eligible_fmt = eligible
        pivot = pivot.reindex(columns=eligible_fmt)
        strata = selected.drop_duplicates("concept_id").set_index("concept_id")["stratum"].astype(str)
        pivot["stratum"] = strata
        pivot = pivot.loc[pivot["stratum"].isin(("sino", "nonsino"))]
        if pivot.loc[:, eligible_fmt].isna().any().any():
            raise ValueError(f"incomplete passing LA_zh layer matrix for {fmt}")
        wide_by_format[fmt] = pivot.loc[:, eligible_fmt].to_numpy(dtype=np.float64)
        for stratum in ("sino", "nonsino"):
            ids = pivot.index[pivot["stratum"].eq(stratum)].astype(str).tolist()
            if not ids:
                raise ValueError(f"no {stratum} LA_zh concepts for {fmt}")
            mean_curve = pivot.loc[ids, eligible_fmt].mean(axis=0).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "format": fmt,
                    "stratum": stratum,
                    "n_concepts": len(ids),
                    "layers": ";".join(map(str, eligible_fmt)),
                    "mean_curve": mean_curve,
                }
            )
        sino_curve = pivot.loc[pivot["stratum"].eq("sino"), eligible_fmt].mean(axis=0).to_numpy(dtype=np.float64)
        non_curve = pivot.loc[pivot["stratum"].eq("nonsino"), eligible_fmt].mean(axis=0).to_numpy(dtype=np.float64)
        rows.append(
            {
                "format": fmt,
                "stratum": "contrast",
                "n_concepts": int(pivot["stratum"].eq("sino").sum() + pivot["stratum"].eq("nonsino").sum()),
                "layers": ";".join(map(str, eligible_fmt)),
                "mean_curve": sino_curve - non_curve,
            }
        )
    table = pd.DataFrame.from_records(rows)
    return table, wide_by_format


def dzh_from_frame(
    la_zh: pd.DataFrame,
    *,
    n_layers: int,
    window: Sequence[float],
    draws: Sequence[Sequence[tuple[str, str]]] | None = None,
    chunk_size: int = 100,
    completed: Mapping[int, np.ndarray] | None = None,
    on_chunk: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute D_zh and optionally resume/checkpoint stratified bootstrap chunks."""
    rows: list[dict[str, Any]] = []
    samples_by_format: dict[str, np.ndarray] = {}
    point_by_format: dict[str, float] = {}
    matrices: dict[str, pd.DataFrame] = {}
    strata_by_format: dict[str, pd.Series] = {}
    layer_by_format: dict[str, list[int]] = {}
    for fmt in FORMATS:
        fmt_rows = la_zh.loc[la_zh["format"].astype(str).eq(fmt)]
        available_layers = sorted(pd.to_numeric(fmt_rows["layer"], errors="raise").astype(int).unique())
        low, high = map(float, window)
        layers = [layer for layer in available_layers if layer >= 1 and low <= layer / int(n_layers) <= high]
        if not layers:
            raise ValueError(f"no D_zh layers in window for {fmt}")
        selected = fmt_rows.loc[fmt_rows["layer"].isin(layers)]
        wide = selected.pivot(index="concept_id", columns="layer", values="value").reindex(columns=layers)
        strata = selected.drop_duplicates("concept_id").set_index("concept_id")["stratum"].astype(str)
        wide["stratum"] = strata
        means: dict[str, np.ndarray] = {}
        for stratum in ("sino", "nonsino"):
            subset = wide.loc[wide["stratum"].eq(stratum), layers]
            if subset.empty:
                raise ValueError(f"no {stratum} observations in {fmt}")
            means[stratum] = subset.mean(axis=0).to_numpy(dtype=np.float64)
        contrast_curve = means["sino"] - means["nonsino"]
        point = float(contrast_curve.mean())
        point_by_format[fmt] = point
        matrices[fmt] = wide
        strata_by_format[fmt] = wide["stratum"].astype(str)
        layer_by_format[fmt] = list(map(int, layers))
        rows.append({"format": fmt, "instrument": "D_zh", "estimate": point, "n_sino": int(wide["stratum"].eq("sino").sum()), "n_nonsino": int(wide["stratum"].eq("nonsino").sum()), "layers": ";".join(map(str, layers))})
    pooled_point = float(np.mean([point_by_format[fmt] for fmt in FORMATS]))
    rows.append({"format": "pooled", "instrument": "D_zh", "estimate": pooled_point, "n_sino": min(row["n_sino"] for row in rows if row["format"] in FORMATS), "n_nonsino": min(row["n_nonsino"] for row in rows if row["format"] in FORMATS), "layers": "equal-weight mean of formats"})
    samples: dict[str, np.ndarray] = {}
    if draws is not None:
        chunk_size = max(1, int(chunk_size))
        completed = completed or {}
        arrays = {fmt: np.full(len(draws), np.nan, dtype=np.float64) for fmt in FORMATS}
        for chunk_index, begin in enumerate(range(0, len(draws), chunk_size)):
            stop = min(begin + chunk_size, len(draws))
            cache = completed.get(chunk_index)
            if cache is not None:
                cached = np.asarray(cache, dtype=np.float64)
                if cached.shape != (stop - begin, len(FORMATS)):
                    raise ValueError(f"cached D_zh chunk {chunk_index} has the wrong shape")
                for format_index, fmt in enumerate(FORMATS):
                    arrays[fmt][begin:stop] = cached[:, format_index]
                continue
            chunk_values = np.full((stop - begin, len(FORMATS)), np.nan, dtype=np.float64)
            for draw_index in range(begin, stop):
                draw = draws[draw_index]
                for format_index, fmt in enumerate(FORMATS):
                    wide = matrices[fmt]
                    layers = layer_by_format[fmt]
                    sampled: dict[str, list[np.ndarray]] = {"sino": [], "nonsino": []}
                    for stratum, concept_id in draw:
                        if stratum in sampled and concept_id in wide.index:
                            sampled[stratum].append(wide.loc[concept_id, layers].to_numpy(dtype=np.float64))
                    if sampled["sino"] and sampled["nonsino"]:
                        difference = np.mean(np.stack(sampled["sino"]), axis=0) - np.mean(np.stack(sampled["nonsino"]), axis=0)
                        chunk_values[draw_index - begin, format_index] = float(np.mean(difference))
            for format_index, fmt in enumerate(FORMATS):
                arrays[fmt][begin:stop] = chunk_values[:, format_index]
            if on_chunk is not None:
                on_chunk(chunk_index, begin, chunk_values.copy())
        samples = {fmt: values for fmt, values in arrays.items()}
        samples["pooled"] = np.mean(np.stack([arrays[fmt] for fmt in FORMATS]), axis=0)
        for row in rows:
            fmt = str(row["format"])
            values = samples[fmt]
            low_ci, high_ci = percentile_interval(values)
            row["ci_low"] = low_ci
            row["ci_high"] = high_ci
            row["valid_draws"] = int(np.isfinite(values).sum())
            row["p_value"] = two_sided_bootstrap_p(values)
    return pd.DataFrame.from_records(rows), samples


def dzh_pair_bootstrap(
    pair_differences: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    key: str,
    chunk_size: int = 100,
    completed: Mapping[int, np.ndarray] | None = None,
    on_chunk: Any | None = None,
) -> np.ndarray:
    """Bootstrap matched-pair D_zh differences with pairs as the sampling unit."""
    required = {"pair_id", "format", "layer", "difference"}
    if not required.issubset(pair_differences.columns):
        raise ValueError(f"matched D_zh table lacks {sorted(required - set(pair_differences.columns))}")
    ids = sorted(pair_differences["pair_id"].astype(str).unique())
    if not ids:
        raise ValueError("matched D_zh has no pairs")
    rng = stream(int(seed), str(key))
    result = np.full(int(resamples), np.nan, dtype=np.float64)
    by_pair = {pair: frame for pair, frame in pair_differences.groupby("pair_id", sort=False)}
    chunk_size = max(1, int(chunk_size))
    completed = completed or {}
    # Generate the full index stream before consulting the cache. This keeps
    # later chunks bit-identical when an earlier chunk is resumed from disk.
    draw_indices = rng.integers(
        0, len(ids), size=(int(resamples), len(ids)), endpoint=False
    )
    for chunk_index, begin in enumerate(range(0, int(resamples), chunk_size)):
        stop = min(begin + chunk_size, int(resamples))
        cached = completed.get(chunk_index)
        if cached is not None:
            values = np.asarray(cached, dtype=np.float64)
            if values.shape != (stop - begin,):
                raise ValueError(f"cached matched D_zh chunk {chunk_index} has the wrong shape")
            result[begin:stop] = values
            continue
        chunk_values = np.full(stop - begin, np.nan, dtype=np.float64)
        for draw_index in range(begin, stop):
            selected = draw_indices[draw_index]
            format_estimates: list[float] = []
            for fmt in FORMATS:
                values = [
                    by_pair[ids[int(i)]].loc[
                        by_pair[ids[int(i)]] ["format"].astype(str).eq(fmt), "difference"
                    ].to_numpy(dtype=np.float64)
                    for i in selected
                ]
                flattened = np.concatenate([value for value in values if value.size]) if any(value.size for value in values) else np.array([])
                finite = flattened[np.isfinite(flattened)]
                if finite.size:
                    format_estimates.append(float(finite.mean()))
            if len(format_estimates) == len(FORMATS):
                chunk_values[draw_index - begin] = float(np.mean(format_estimates))
        result[begin:stop] = chunk_values
        if on_chunk is not None:
            on_chunk(chunk_index, begin, chunk_values.copy())
    return result


def mixedlm_formula(*, spline_df: int = 5, frequency_covariate: str = "zipf_en") -> str:
    """Return the OPS-28 fixed-effects formula with task_format fixed."""
    if frequency_covariate not in {"zipf_en", "zipf_vi", "zipf_zh"}:
        raise ValueError("unsupported E6 frequency covariate")
    if int(spline_df) != 5:
        raise ValueError("OPS-28 fixes spline df at 5")
    return (
        "rise ~ C(language, Treatment(reference='en')) * cr(depth, df=5) + "
        "C(stratum, Treatment(reference='nonsino')) * C(language, Treatment(reference='en')) + "
        f"C(task_format) + {frequency_covariate} + n_tokens + n_syllables + concreteness"
    )


def fit_mixedlm(
    frame: pd.DataFrame,
    *,
    spline_df: int = 5,
    frequency_covariate: str = "zipf_en",
    start_params: Sequence[float] | None = None,
    maxiter: int = 500,
    fit_methods: Sequence[str] = ("lbfgs", "lbfgs_mom", "lbfgs_cold", "bfgs", "powell"),
    boundary_rule: bool = True,
    llf_tie_tol: float = 1.0e-6,
) -> MixedLMFit:
    """Fit OPS-28 REML with OPS-29 validity and OPS-31 boundary selection."""
    import statsmodels.formula.api as smf

    required = {
        "rise", "language", "depth", "stratum", "task_format", "concept_group",
        frequency_covariate, "n_tokens", "n_syllables", "concreteness",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MixedLM frame lacks {sorted(missing)}")
    method_order = tuple(map(str, fit_methods))
    if method_order != ("lbfgs", "lbfgs_mom", "lbfgs_cold", "bfgs", "powell"):
        raise ValueError("OPS-31 fit_methods must be lbfgs, lbfgs_mom, lbfgs_cold, bfgs, powell in order")
    if type(maxiter) is not int or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    if type(boundary_rule) is not bool:
        raise ValueError("boundary_rule must be a boolean")
    if type(llf_tie_tol) not in {int, float} or not np.isfinite(float(llf_tie_tol)) or float(llf_tie_tol) < 0:
        raise ValueError("llf_tie_tol must be a finite non-negative number")
    formula = mixedlm_formula(spline_df=spline_df, frequency_covariate=frequency_covariate)
    model = smf.mixedlm(formula, data=frame, groups=frame["concept_group"], re_formula="1")
    attempts: list[tuple[str, str, Any]] = []
    if start_params is not None:
        attempts.append(("lbfgs", "lbfgs", start_params))
    attempts.append(("lbfgs_mom", "lbfgs", None))
    attempts.extend((("lbfgs_cold", "lbfgs", None), ("bfgs", "bfgs", None), ("powell", "powell", None)))
    warnings_seen: dict[str, int] = {}
    failures: list[str] = []
    last_result: Any | None = None
    last_converged = False
    last_variance: float | None = None
    last_boundary = False
    valid_candidates: list[tuple[int, str, Any, float | None, bool]] = []
    first_valid_llf: float | None = None
    boundary_rule_ran = False
    first_valid_method: str | None = None
    started = time.perf_counter()
    tries = 0
    for attempt_index, (label, optimizer, initial) in enumerate(attempts):
        tries += 1
        if label == "lbfgs_mom":
            moment_error: Exception | None = None
            with warnings.catch_warnings(record=True) as moment_warnings:
                warnings.simplefilter("always")
                try:
                    initial = _mixedlm_moment_start(model, frame, formula)
                except Exception as exc:
                    moment_error = exc
            for item in moment_warnings:
                category = _warning_category(item.category.__name__, str(item.message))
                warnings_seen[category] = warnings_seen.get(category, 0) + 1
            if moment_error is not None:
                exc = moment_error
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
        result: Any | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(
                    reml=True,
                    method=optimizer,
                    start_params=initial,
                    maxiter=int(maxiter),
                    disp=False,
                )
            except Exception as exc:  # Each failure is recorded before trying the next optimizer.
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
        for item in caught:
            category = _warning_category(item.category.__name__, str(item.message))
            warnings_seen[category] = warnings_seen.get(category, 0) + 1
        if result is None:
            continue
        valid, converged, variance, boundary, reason = _mixedlm_candidate_validity(result)
        last_result = result
        last_converged = converged
        last_variance = variance
        last_boundary = boundary
        if reason:
            failures.extend(f"{label}: {entry}" for entry in reason)
        if valid:
            if first_valid_method is None:
                first_valid_method = label
                first_valid_llf = _finite_llf(result)
            valid_candidates.append((attempt_index, label, result, variance, boundary))
            if boundary_rule and valid_candidates[0][4]:
                boundary_rule_ran = True
                continue
            break
    if valid_candidates:
        first = valid_candidates[0]
        selected = first
        if boundary_rule_ran:
            ranked = [candidate for candidate in valid_candidates if _finite_llf(candidate[2]) is not None]
            if ranked:
                best_llf = max(float(candidate[2].llf) for candidate in ranked)
                tied = [
                    candidate for candidate in ranked
                    if best_llf - float(candidate[2].llf) <= float(llf_tie_tol)
                ]
                selected = min(tied, key=lambda candidate: candidate[0])
        _selected_index, selected_method, selected_result, selected_variance, selected_boundary = selected
        return MixedLMFit(
            selected_result,
            FitDiagnostics(
                valid=True,
                converged=True,
                method_used=selected_method,
                tries=tries,
                variance_at_boundary=selected_boundary,
                warning_categories=dict(sorted(warnings_seen.items())),
                failure_reasons=tuple(failures),
                elapsed_seconds=time.perf_counter() - started,
                random_intercept_variance=selected_variance,
                boundary_rule_ran=boundary_rule_ran,
                boundary_rule_changed_fit=selected_method != first[1],
                llf_kept=_finite_llf(selected_result),
                llf_first_valid=first_valid_llf,
            ),
        )
    return MixedLMFit(
        last_result,
        FitDiagnostics(
            valid=False,
            converged=last_converged,
            method_used="not_estimable",
            tries=tries,
            variance_at_boundary=last_boundary,
            warning_categories=dict(sorted(warnings_seen.items())),
            failure_reasons=tuple(failures or ("all configured optimizer attempts failed",)),
            elapsed_seconds=time.perf_counter() - started,
            random_intercept_variance=last_variance,
            boundary_rule_ran=boundary_rule_ran,
            boundary_rule_changed_fit=False,
            llf_kept=None,
            llf_first_valid=first_valid_llf,
        ),
    )


def _finite_llf(result: Any) -> float | None:
    """Return a finite REML log likelihood, or ``None`` if unavailable."""
    try:
        value = float(result.llf)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _mixedlm_moment_start(model: Any, frame: pd.DataFrame, formula: str) -> Any:
    """Build the OPS-31 OLS fixed-effect and ANOVA random-intercept start."""
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLMParams

    ols_fit = smf.ols(formula, data=frame).fit()
    residuals = pd.Series(np.asarray(ols_fit.resid, dtype=np.float64), index=frame.index)
    groups = frame["concept_group"].astype(str)
    group_sizes = groups.value_counts(dropna=False)
    if group_sizes.empty or group_sizes.nunique() != 1:
        raise ValueError("OPS-31 moment start requires equal rows per concept group")
    rows_per_concept = int(group_sizes.iloc[0])
    group_mean_residuals = residuals.groupby(groups, sort=False).mean().to_numpy(dtype=np.float64)
    between_variance = float(np.var(group_mean_residuals, ddof=1)) if group_mean_residuals.size > 1 else 0.0
    residual_variance = float(ols_fit.scale)
    variance_start = max(1.0e-4, between_variance - residual_variance / rows_per_concept)
    fixed_effects = np.asarray(ols_fit.params, dtype=np.float64)
    expected_fixed_effects = getattr(model, "k_fe", fixed_effects.size)
    if fixed_effects.size != int(expected_fixed_effects):
        raise ValueError("OPS-31 OLS and MixedLM fixed-effect columns do not align")
    return MixedLMParams.from_components(
        fe_params=fixed_effects,
        cov_re=np.asarray([[variance_start]], dtype=np.float64),
    )


def generalized_moment_start(
    model: Any, frame: pd.DataFrame, formula: str, *, group_column: str = "concept_group"
) -> Any:
    """Build the OPS-34 one-way ANOVA start for unequal concept-group sizes."""
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLMParams

    ols_fit = smf.ols(formula, data=frame).fit()
    residuals = pd.Series(np.asarray(ols_fit.resid, dtype=np.float64), index=frame.index)
    groups = frame[group_column].astype(str)
    grouped = residuals.groupby(groups, sort=False)
    sizes = grouped.size().to_numpy(dtype=np.float64)
    means = grouped.mean().to_numpy(dtype=np.float64)
    n_rows = int(residuals.size)
    n_groups = int(sizes.size)
    if n_groups < 2 or n_rows <= n_groups or not np.isfinite(residuals).all():
        raise ValueError("OPS-34 moment start requires finite residuals and at least two non-singleton groups")
    if np.all(sizes == sizes[0]):
        # Preserve the OPS-31 equal-size method exactly in its balanced limit.
        return _mixedlm_moment_start(model, frame, formula)
    grand_mean = float(residuals.mean())
    ss_between = float(np.sum(sizes * np.square(means - grand_mean)))
    ms_between = ss_between / (n_groups - 1)
    within = residuals.to_numpy(dtype=np.float64) - groups.map(grouped.mean()).to_numpy(dtype=np.float64)
    ms_within = float(np.sum(np.square(within)) / (n_rows - n_groups))
    n0 = (n_rows - float(np.sum(np.square(sizes))) / n_rows) / (n_groups - 1)
    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError("OPS-34 unequal-group effective size must be positive")
    variance_start = max(1.0e-4, (ms_between - ms_within) / n0)
    fixed_effects = np.asarray(ols_fit.params, dtype=np.float64)
    expected_fixed_effects = getattr(model, "k_fe", fixed_effects.size)
    if fixed_effects.size != int(expected_fixed_effects):
        raise ValueError("OPS-34 OLS and MixedLM fixed-effect columns do not align")
    return MixedLMParams.from_components(
        fe_params=fixed_effects,
        cov_re=np.asarray([[variance_start]], dtype=np.float64),
    )


def fixed_design_estimability(
    frame: pd.DataFrame,
    *,
    frequency_covariate: str = "zipf_en",
    relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Check fixed-design rank and two frozen contrasts using covariates only."""
    import patsy

    design_columns = [
        "language", "depth", "stratum", "task_format", frequency_covariate,
        "n_tokens", "n_syllables", "concreteness",
    ]
    missing = set(design_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"fixed design lacks columns {sorted(missing)}")
    design_data = frame.loc[:, design_columns].drop_duplicates().reset_index(drop=True)
    if design_data.empty or design_data.isna().any().any():
        raise ValueError("fixed design requires non-empty complete covariates")
    rhs = mixedlm_formula(frequency_covariate=frequency_covariate).split("~", 1)[1]
    matrix = patsy.dmatrix(rhs, design_data, return_type="dataframe")
    values = matrix.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("fixed design contains non-finite covariates")
    # The design is tall (hundreds of thousands of rows, ~40 columns).  The
    # economy factorization returns U with only `column_count` columns while
    # preserving the singular values and right singular vectors used below.
    # Full matrices would allocate an unused n_rows × n_rows U factor.
    _u, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    cutoff = (
        max(values.shape) * np.finfo(np.float64).eps * singular_values[0]
        if singular_values.size else 0.0
    )
    rank = int(np.count_nonzero(singular_values > cutoff))
    columns = list(map(str, matrix.columns))
    null_basis = vh[rank:, :].T if rank < len(columns) else np.empty((len(columns), 0))
    supports: list[list[str]] = []
    for basis in null_basis.T:
        scale = float(np.max(np.abs(basis), initial=0.0))
        supports.append([
            columns[index] for index, value in enumerate(basis)
            if abs(float(value)) > max(1.0e-10, scale * 1.0e-6)
        ])
    null_columns = sorted({column for support in supports for column in support})

    def residual_for(weights: Mapping[str, float]) -> tuple[float, bool]:
        contrast = np.zeros(len(columns), dtype=np.float64)
        for name, weight in weights.items():
            if name not in columns:
                return float("inf"), False
            contrast[columns.index(name)] = float(weight)
        norm = float(np.linalg.norm(contrast))
        if norm == 0.0:
            return float("inf"), False
        residual = float(np.linalg.norm(null_basis.T @ contrast) / norm) if null_basis.size else 0.0
        return residual, residual <= float(relative_tolerance)

    interaction, sino = _interaction_terms(columns)
    interaction_residual, interaction_estimable = residual_for({interaction: 1.0}) if interaction else (float("inf"), False)
    sum_residual, sum_estimable = (
        residual_for({sino: 1.0, interaction: 1.0})
        if sino and interaction else (float("inf"), False)
    )
    return {
        "rank": rank,
        "column_count": len(columns),
        "design_rows_unique": len(design_data),
        "nullity": len(columns) - rank,
        "null_space_columns": null_columns,
        "null_space_supports": supports,
        "interaction_term": interaction,
        "interaction_estimability_residual": interaction_residual,
        "interaction_estimable": interaction_estimable,
        "sum_effect_estimability_residual": sum_residual,
        "sum_effect_estimable": sum_estimable,
        "relative_tolerance": float(relative_tolerance),
    }


def _warning_category(category: str, message: str) -> str:
    """Normalize per-fit statsmodels warning messages into stable categories."""
    normalized = re.sub(r"\|grad\|\s*=\s*[-+0-9.eE]+", "|grad| = <numeric>", message)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{category}: {normalized}"


def _mixedlm_candidate_validity(result: Any) -> tuple[bool, bool, float | None, bool, list[str]]:
    """Apply the OPS-29 finite-estimate, target-SE, convergence and boundary checks."""
    reasons: list[str] = []
    converged = bool(getattr(result, "converged", False))
    if not converged:
        reasons.append("optimizer did not report convergence")
    fixed = np.asarray(getattr(result, "fe_params", getattr(result, "params", [])), dtype=np.float64)
    if fixed.size == 0 or not np.isfinite(fixed).all():
        reasons.append("fixed-effect estimates are not all finite")
    params = getattr(result, "params", pd.Series(dtype=np.float64))
    names = list(map(str, params.index)) if hasattr(params, "index") else []
    interaction, _sino = _interaction_terms(names)
    estimate = _term_value(params, names, interaction)
    bse_fe = getattr(result, "bse_fe", getattr(result, "bse", pd.Series(dtype=np.float64)))
    se = _term_value(bse_fe, names, interaction)
    if not np.isfinite(estimate):
        reasons.append("stratum[sino]:language[zh] estimate is not finite")
    if not np.isfinite(se):
        reasons.append("stratum[sino]:language[zh] standard error is not finite")
    variance: float | None
    try:
        cov_re = np.asarray(result.cov_re, dtype=np.float64)
        variance = float(cov_re[0, 0])
    except (AttributeError, IndexError, TypeError, ValueError):
        variance = None
    boundary = variance is not None and np.isfinite(variance) and abs(variance) <= 1e-12
    # Boundary variance is valid under OPS-29 and is recorded separately.
    return not reasons, converged, variance, bool(boundary), reasons


def _interaction_terms(names: Sequence[str]) -> tuple[str | None, str | None]:
    """Find the frozen interaction and main sino coefficient names."""
    interaction = [
        name for name in names
        if "stratum" in name and "language" in name and "sino" in name and "zh" in name and ":" in name
    ]
    sino = [name for name in names if "stratum" in name and "sino" in name and "language" not in name]
    return (interaction[0] if len(interaction) == 1 else None, sino[0] if len(sino) == 1 else None)


def _term_value(values: Any, names: Sequence[str], term: str | None) -> float:
    """Read one fixed-effect value from either pandas or numpy statsmodels output."""
    if term is None or term not in names:
        return float("nan")
    if hasattr(values, "loc"):
        try:
            return float(values.loc[term])
        except (KeyError, TypeError, ValueError):
            return float("nan")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    index = list(names).index(term)
    return float(array[index]) if index < array.size else float("nan")


def model_instrument(fitted: MixedLMFit) -> dict[str, Any]:
    """Extract the OPS-28 interaction and attach OPS-29 fit diagnostics."""
    diagnostics = fitted.diagnostics
    result = fitted.result
    empty: dict[str, Any] = {
        "term": "stratum[sino]:language[zh]",
        "estimate": float("nan"),
        "standard_error": float("nan"),
        "wald_ci_low": float("nan"),
        "wald_ci_high": float("nan"),
        "zh_specific_stratum_effect": float("nan"),
        "converged": diagnostics.converged,
        "valid": diagnostics.valid,
        "method_used": diagnostics.method_used,
        "tries": diagnostics.tries,
        "variance_at_boundary": diagnostics.variance_at_boundary,
        "random_intercept_variance": diagnostics.random_intercept_variance,
        "boundary_rule_ran": diagnostics.boundary_rule_ran,
        "boundary_rule_changed_fit": diagnostics.boundary_rule_changed_fit,
        "llf_kept": diagnostics.llf_kept,
        "llf_first_valid": diagnostics.llf_first_valid,
        "warning_categories": diagnostics.warning_categories,
        "failure_reasons": list(diagnostics.failure_reasons),
        "fit_seconds": diagnostics.elapsed_seconds,
    }
    if result is None or not diagnostics.valid:
        return empty
    params = result.params
    names = list(map(str, params.index))
    term, sino_term = _interaction_terms(names)
    if term is None or sino_term is None:
        raise ValueError(f"cannot locate frozen interaction terms: {names}")
    se_series = getattr(result, "bse_fe", result.bse)
    standard_error = _term_value(se_series, names, term)
    ci = result.conf_int()
    low, high = map(float, ci.loc[term].to_numpy(dtype=np.float64))
    empty.update({
        "term": term,
        "estimate": float(params.loc[term]),
        "standard_error": standard_error,
        "wald_ci_low": low,
        "wald_ci_high": high,
        "zh_specific_stratum_effect": float(params.loc[sino_term] + params.loc[term]),
    })
    return empty


def model_bootstrap(
    frame: pd.DataFrame,
    draws: Sequence[Sequence[tuple[str, str]]],
    *,
    workers: int,
    start_params: Sequence[float] | None,
    spline_df: int = 5,
    frequency_covariate: str = "zipf_en",
    maxiter: int = 500,
    fit_methods: Sequence[str] = ("lbfgs", "lbfgs_mom", "lbfgs_cold", "bfgs", "powell"),
    boundary_rule: bool = True,
    llf_tie_tol: float = 1.0e-6,
    chunk_size: int = 100,
    completed: Mapping[int, tuple[np.ndarray, ...]] | None = None,
    on_chunk: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Refit draws and checkpoint per-draw validity and optimizer diagnostics."""
    from concurrent.futures import ProcessPoolExecutor

    n_draws = len(draws)
    estimates = np.full(n_draws, np.nan, dtype=np.float64)
    valid = np.zeros(n_draws, dtype=bool)
    diagnostics: list[dict[str, Any]] = [empty_fit_diagnostics() for _ in range(n_draws)]
    chunk_size = max(1, int(chunk_size))
    completed = completed or {}
    executor = ProcessPoolExecutor(max_workers=int(workers)) if int(workers) > 1 else None
    try:
        for chunk_index, begin in enumerate(range(0, n_draws, chunk_size)):
            stop = min(begin + chunk_size, n_draws)
            cached = completed.get(chunk_index)
            if cached is not None:
                chunk_values = np.asarray(cached[0], dtype=np.float64)
                chunk_valid = np.asarray(cached[1], dtype=bool)
                arrays = [np.asarray(value) for value in cached[2:]]
                if chunk_values.shape != (stop - begin,) or chunk_valid.shape != (stop - begin,) or any(array.shape != (stop - begin,) for array in arrays):
                    raise ValueError(f"cached bootstrap chunk {chunk_index} has the wrong row count")
                estimates[begin:stop] = chunk_values
                valid[begin:stop] = chunk_valid
                for offset in range(stop - begin):
                    diagnostics[begin + offset] = {
                        "valid": bool(chunk_valid[offset]),
                        "converged": bool(chunk_valid[offset]),
                        "method_used": str(arrays[0][offset]),
                        "tries": int(arrays[1][offset]),
                        "variance_at_boundary": bool(arrays[2][offset]),
                        "warning_categories": json.loads(str(arrays[3][offset])),
                        "failure_reasons": json.loads(str(arrays[4][offset])),
                        "fit_seconds": float(arrays[5][offset]),
                        "boundary_rule_ran": bool(arrays[6][offset]),
                        "boundary_rule_changed_fit": bool(arrays[7][offset]),
                        "llf_kept": _optional_float(float(arrays[8][offset])),
                        "llf_first_valid": _optional_float(float(arrays[9][offset])),
                        "random_intercept_variance": _optional_float(float(arrays[10][offset])),
                    }
                continue
            for micro_begin in range(begin, stop, max(1, int(workers))):
                micro_stop = min(micro_begin + max(1, int(workers)), stop)
                payloads = [
                    (
                        expand_draw_groups(frame, draws[index]),
                        None if start_params is None else np.asarray(start_params, dtype=np.float64),
                        int(spline_df),
                        frequency_covariate,
                        int(maxiter),
                        tuple(fit_methods),
                        bool(boundary_rule),
                        float(llf_tie_tol),
                    )
                    for index in range(micro_begin, micro_stop)
                ]
                fits = (
                    [executor.submit(_fit_model_payload, payload) for payload in payloads]
                    if executor is not None
                    else [None]
                )
                results = [future.result() for future in fits] if executor is not None else [
                    _fit_model_payload(payload) for payload in payloads
                ]
                for index, (estimate, fit_diag) in enumerate(results, start=micro_begin):
                    valid[index] = bool(fit_diag["valid"])
                    estimates[index] = float(estimate) if valid[index] else np.nan
                    diagnostics[index] = fit_diag
            if on_chunk is not None:
                on_chunk(chunk_index, begin, estimates[begin:stop].copy(), valid[begin:stop].copy(), diagnostics[begin:stop])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    estimates[~valid] = np.nan
    return estimates, valid, diagnostics


def empty_fit_diagnostics() -> dict[str, Any]:
    """Return the fixed-schema diagnostic record used by resume-safe chunks."""
    return {
        "valid": False, "converged": False, "method_used": "not_estimable", "tries": 0,
        "variance_at_boundary": False,
        "warning_categories": {}, "failure_reasons": [], "fit_seconds": 0.0,
        "boundary_rule_ran": False, "boundary_rule_changed_fit": False,
        "llf_kept": None, "llf_first_valid": None, "random_intercept_variance": None,
    }


def _optional_float(value: float) -> float | None:
    """Convert a finite diagnostic scalar to float, or return ``None``."""
    return float(value) if np.isfinite(float(value)) else None


def _fit_model_payload(payload: tuple[pd.DataFrame, np.ndarray | None, int, str, int, Sequence[str], bool, float]) -> tuple[float, dict[str, Any]]:
    """Worker entry point for one OPS-29-validity-checked bootstrap refit."""
    frame, start, spline_df, frequency_covariate, maxiter, fit_methods, boundary_rule, llf_tie_tol = payload
    try:
        fitted = fit_mixedlm(
            frame,
            spline_df=spline_df,
            frequency_covariate=frequency_covariate,
            start_params=start,
            maxiter=maxiter,
            fit_methods=fit_methods,
            boundary_rule=boundary_rule,
            llf_tie_tol=llf_tie_tol,
        )
        instrument = model_instrument(fitted)
        return float(instrument["estimate"]), {key: value for key, value in instrument.items() if key not in {"term", "estimate", "standard_error", "wald_ci_low", "wald_ci_high", "zh_specific_stratum_effect"}}
    except Exception as exc:
        diagnostic = empty_fit_diagnostics()
        diagnostic["failure_reasons"] = [f"{type(exc).__name__}: {exc}"]
        return float("nan"), diagnostic


def bootstrap_summary(samples: Sequence[float], *, min_valid_draws: int = 950) -> dict[str, float | int | bool]:
    """Summarize finite bootstrap draws and their OPS-28 two-sided p-value."""
    values = np.asarray(samples, dtype=np.float64)
    valid = values[np.isfinite(values)]
    low, high = percentile_interval(valid) if valid.size else (float("nan"), float("nan"))
    return {
        "ci_low": low,
        "ci_high": high,
        "valid_draws": int(valid.size),
        "nonconverged_draws": int(values.size - valid.size),
        "p_value": two_sided_bootstrap_p(valid),
        "result_class": interval_class(low, high),
        "valid_draw_gate_passed": int(valid.size) >= int(min_valid_draws),
    }


def fit_diagnostics_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate valid fits, optimizer choices, boundaries and warning categories."""
    methods: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for record in records:
        method = str(record.get("method_used", "not_estimable"))
        methods[method] = methods.get(method, 0) + 1
        for warning, count in record.get("warning_categories", {}).items():
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + int(count)
    return {
        "valid_count": int(sum(bool(record.get("valid", False)) for record in records)),
        "method_mix": dict(sorted(methods.items())),
        "boundary_count": int(sum(bool(record.get("variance_at_boundary", False)) for record in records)),
        "boundary_rule_trigger_count": int(sum(bool(record.get("boundary_rule_ran", False)) for record in records)),
        "boundary_rule_changed_fit_count": int(sum(bool(record.get("boundary_rule_changed_fit", False)) for record in records)),
        "warning_categories": dict(sorted(warning_counts.items())),
        "mean_fit_seconds": float(np.mean([float(record.get("fit_seconds", 0.0)) for record in records])) if records else 0.0,
        "mean_tries": float(np.mean([int(record.get("tries", 0)) for record in records])) if records else 0.0,
    }


def synthetic_refit_recovery_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    planted_concept_sd: float,
    concept_sd_ratio_bounds: Sequence[float],
    expected_refits: int,
    min_valid_refits: int,
) -> dict[str, Any]:
    """Apply the OPS-31 valid-and-recovered gate to synthetic bootstrap refits."""
    if len(concept_sd_ratio_bounds) != 2:
        raise ValueError("concept SD bounds need a lower and upper ratio")
    low = float(concept_sd_ratio_bounds[0]) * float(planted_concept_sd)
    high = float(concept_sd_ratio_bounds[1]) * float(planted_concept_sd)
    valid_records = [record for record in records if bool(record.get("valid", False))]
    standard_deviations = [
        float(np.sqrt(float(record["random_intercept_variance"])))
        for record in valid_records
        if record.get("random_intercept_variance") is not None
        and np.isfinite(float(record["random_intercept_variance"]))
        and float(record["random_intercept_variance"]) >= 0
    ]
    recovered_count = sum(low <= value <= high for value in standard_deviations)
    return {
        "refit_count": int(len(records)),
        "valid_refit_count": int(len(valid_records)),
        "recovered_refit_count": int(recovered_count),
        "concept_sd_min": float(np.min(standard_deviations)) if standard_deviations else None,
        "concept_sd_median": float(np.median(standard_deviations)) if standard_deviations else None,
        "concept_sd_max": float(np.max(standard_deviations)) if standard_deviations else None,
        "boundary_rule_trigger_count": int(sum(bool(record.get("boundary_rule_ran", False)) for record in records)),
        "boundary_rule_changed_fit_count": int(sum(bool(record.get("boundary_rule_changed_fit", False)) for record in records)),
        "mean_seconds_per_fit": float(np.mean([float(record.get("fit_seconds", 0.0)) for record in records])) if records else 0.0,
        "mean_tries": float(np.mean([int(record.get("tries", 0)) for record in records])) if records else 0.0,
        "method_mix": fit_diagnostics_summary(records)["method_mix"],
        "gate_passed": int(len(records)) == int(expected_refits) and recovered_count >= int(min_valid_refits),
    }


def holm_family_b(rows: pd.DataFrame) -> pd.DataFrame:
    """Adjust the six frozen H3 p-values with the shared Holm implementation."""
    if len(rows) != 6 or rows["instrument"].nunique() != 2 or rows["model"].nunique() != 3:
        raise ValueError("Holm family (b) must contain 3 models × 2 instruments")
    result = rows.sort_values(["model", "instrument"], kind="mergesort").copy()
    p_values = result["p_value"].to_numpy(dtype=np.float64)
    if np.isfinite(p_values).all():
        result["p_holm"] = holm_adjust(p_values)
        result["reject_0_05"] = result["p_holm"].le(0.05)
    else:
        # A missing instrument leaves the six-test family incomplete; report it
        # as unavailable instead of adjusting a reduced family.
        result["p_holm"] = np.nan
        result["reject_0_05"] = False
        result["family_status"] = "not estimable: at least one full-sample fit invalid"
    return result


def match_bootstrap_draws(
    pairs: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    key: str,
) -> list[list[str]]:
    """Resample pair IDs with replacement under a dedicated deterministic stream."""
    ids = sorted(pairs["pair_id"].astype(str).unique())
    if not ids:
        raise ValueError("matched bootstrap requires at least one pair")
    rng = stream(int(seed), str(key))
    return [[ids[int(i)] for i in rng.integers(0, len(ids), size=len(ids), endpoint=False)] for _ in range(int(resamples))]


def expand_pair_draw(frame: pd.DataFrame, pairs: pd.DataFrame, draw: Sequence[str]) -> pd.DataFrame:
    """Expand a matched pair draw into distinct sino/nonsino concept groups."""
    required = {"pair_id", "sino_concept_id", "nonsino_concept_id"}
    if not required.issubset(pairs.columns):
        raise ValueError("pair table schema is invalid")
    mapping = pairs.set_index("pair_id")
    sample_rows: list[dict[str, str]] = []
    for index, pair_id in enumerate(draw):
        pair = mapping.loc[str(pair_id)]
        for stratum, id_col in (("sino", "sino_concept_id"), ("nonsino", "nonsino_concept_id")):
            sample_rows.append(
                {"concept_id": str(pair[id_col]), "stratum": stratum, "concept_group": f"pairdraw_{index:05d}_{stratum}"}
            )
    sample = pd.DataFrame.from_records(sample_rows)
    joined = sample.merge(frame.drop(columns=["concept_group"], errors="ignore"), on=["concept_id", "stratum"], how="inner", validate="many_to_many")
    if joined.empty:
        raise ValueError("matched pair draw has no model rows")
    return joined


def make_synthetic_model_frame(
    *,
    n_sino: int,
    n_nonsino: int,
    n_layers: int,
    seed: int,
    synthetic_sd: Mapping[str, float] | None = None,
    synthetic_fixed_effects: Mapping[str, float] | None = None,
    target_interaction: float = 0.5,
    format_concept_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    """Create production-shaped synthetic §7.1 rows with planted effects."""
    rng = np.random.default_rng(int(seed))
    sd = dict(synthetic_sd or {"concept": 1.0, "concept_language": 1.0, "residual": 2.0})
    required_sd = {"concept", "concept_language", "residual"}
    if (
        set(sd) != required_sd
        or any(not np.isfinite(float(value)) or float(value) < 0 for value in sd.values())
        or float(sd["residual"]) <= 0
    ):
        raise ValueError("synthetic_sd must contain non-negative random-effect SDs and a positive residual SD")
    fixed = dict(synthetic_fixed_effects or {
        "language_en": 0.35, "depth": 0.16, "task_format_translation": 0.03,
    })
    required_fixed = {"language_en", "depth", "task_format_translation"}
    if (
        set(fixed) != required_fixed
        or any(not np.isfinite(float(value)) for value in fixed.values())
        or not np.isfinite(float(target_interaction))
    ):
        raise ValueError("synthetic fixed effects must specify language_en, depth and task_format_translation")
    normalized_format_counts: dict[str, dict[str, int]] | None = None
    if format_concept_counts is not None:
        if set(format_concept_counts) != set(FORMATS):
            raise ValueError(f"format_concept_counts must contain exactly {FORMATS}")
        normalized_format_counts = {}
        for fmt in FORMATS:
            counts = format_concept_counts[fmt]
            if set(counts) != {"sino", "nonsino"}:
                raise ValueError(f"{fmt} format_concept_counts must contain sino and nonsino")
            normalized = {stratum: int(counts[stratum]) for stratum in ("sino", "nonsino")}
            limits = {"sino": int(n_sino), "nonsino": int(n_nonsino)}
            if any(normalized[stratum] < 0 or normalized[stratum] > limits[stratum] for stratum in limits):
                raise ValueError(f"{fmt} format concept count is outside the synthetic population")
            normalized_format_counts[fmt] = normalized
    concept_rows: list[dict[str, Any]] = []
    for index in range(int(n_sino) + int(n_nonsino)):
        stratum = "sino" if index < int(n_sino) else "nonsino"
        concept_rows.append(
            {
                "concept_id": f"concept_{index:05d}",
                "stratum": stratum,
                "zipf_en": float(rng.normal()),
                "n_tokens": float(rng.integers(1, 5)),
                "n_syllables": float(rng.integers(1, 4)),
                "concreteness": float(rng.normal()),
                "synthetic_random_intercept": float(rng.normal(scale=float(sd["concept"]))),
            }
        )
    rows: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concept_rows):
        sino = concept["stratum"] == "sino"
        local_index = concept_index if sino else concept_index - int(n_sino)
        language_effects = {
            language: float(rng.normal(scale=float(sd["concept_language"])))
            for language in LANGUAGES
        }
        for fmt in FORMATS:
            if normalized_format_counts is not None and local_index >= normalized_format_counts[fmt][concept["stratum"]]:
                continue
            for language in LANGUAGES:
                zh_effect = float(target_interaction) if sino and language == "zh" else 0.0
                for layer in range(1, int(n_layers) + 1):
                    depth = layer / int(n_layers)
                    residual = float(rng.normal(scale=float(sd["residual"])))
                    response = (
                        float(fixed["language_en"]) * (language == "en")
                        + float(fixed["depth"]) * depth
                        + zh_effect
                        + concept["synthetic_random_intercept"]
                        + language_effects[language]
                        + float(fixed["task_format_translation"]) * (fmt == "translation")
                        + residual
                    )
                    rows.append(
                        {
                            **concept,
                            "synthetic_concept_language": language_effects[language],
                            "synthetic_residual": residual,
                            "language": language,
                            "task_format": fmt,
                            "format": fmt,
                            "layer": layer,
                            "depth": depth,
                            "rise": float(response),
                            "concept_group": concept["concept_id"],
                        }
                    )
    return pd.DataFrame.from_records(rows)


def synthetic_recovery_checks(
    frame: pd.DataFrame,
    fitted: MixedLMFit,
    *,
    n_layers: int,
    planted_concept_sd: float,
    planted_interaction: float,
    concept_sd_ratio_bounds: Sequence[float],
    target_wald_se_multiple: float,
    target_abs_bias_max: float,
) -> dict[str, Any]:
    """Evaluate the OPS-30 synthetic fit and grouping checks without reading outcomes."""
    if "concept_id" not in frame or "concept_group" not in frame:
        raise ValueError("recovery frame needs concept_id and concept_group")
    if len(concept_sd_ratio_bounds) != 2:
        raise ValueError("concept SD bounds need a lower and upper ratio")
    concept_count = int(frame["concept_id"].nunique(dropna=False))
    group_count = int(frame["concept_group"].nunique(dropna=False))
    group_sizes = frame.groupby("concept_group", dropna=False, sort=False).size()
    expected_rows_per_group = len(LANGUAGES) * int(n_layers) * len(FORMATS)
    rows_min = int(group_sizes.min()) if len(group_sizes) else 0
    rows_max = int(group_sizes.max()) if len(group_sizes) else 0
    instrument = model_instrument(fitted)
    variance = instrument["random_intercept_variance"]
    estimated_concept_sd = (
        float(np.sqrt(float(variance)))
        if variance is not None and np.isfinite(float(variance)) and float(variance) >= 0
        else None
    )
    result = fitted.result
    residual_variance = getattr(result, "scale", None) if result is not None else None
    residual_sd = (
        float(np.sqrt(float(residual_variance)))
        if residual_variance is not None and np.isfinite(float(residual_variance)) and float(residual_variance) >= 0
        else None
    )
    estimate = float(instrument["estimate"])
    standard_error = float(instrument["standard_error"])
    absolute_bias = abs(estimate - float(planted_interaction)) if np.isfinite(estimate) else None
    bias_in_wald_se = (
        float(absolute_bias / standard_error)
        if absolute_bias is not None and np.isfinite(standard_error) and standard_error > 0
        else None
    )
    lower_sd = float(concept_sd_ratio_bounds[0]) * float(planted_concept_sd)
    upper_sd = float(concept_sd_ratio_bounds[1]) * float(planted_concept_sd)
    checks = {
        "full_fit_valid": bool(instrument["valid"]),
        "group_count_matches_concepts": group_count == concept_count,
        "rows_per_group_match_5xNx2": (
            len(group_sizes) == concept_count
            and bool((group_sizes == expected_rows_per_group).all())
        ),
        "concept_sd_recovered": (
            estimated_concept_sd is not None and lower_sd <= estimated_concept_sd <= upper_sd
        ),
        "interaction_within_3_wald_se": (
            bias_in_wald_se is not None and bias_in_wald_se <= float(target_wald_se_multiple)
        ),
        "interaction_abs_bias_below_limit": (
            absolute_bias is not None and absolute_bias < float(target_abs_bias_max)
        ),
    }
    return {
        "planted_concept_sd": float(planted_concept_sd),
        "estimated_concept_sd": estimated_concept_sd,
        "residual_sd": residual_sd,
        "planted_interaction": float(planted_interaction),
        "interaction_estimate": estimate if np.isfinite(estimate) else None,
        "interaction_wald_se": standard_error if np.isfinite(standard_error) else None,
        "interaction_abs_bias": absolute_bias,
        "interaction_bias_in_wald_se": bias_in_wald_se,
        "group_count": group_count,
        "concept_count": concept_count,
        "expected_rows_per_group": expected_rows_per_group,
        "observed_rows_per_group_min": rows_min,
        "observed_rows_per_group_max": rows_max,
        "method_used": str(instrument["method_used"]),
        "tries": int(instrument["tries"]),
        "variance_at_boundary": bool(instrument["variance_at_boundary"]),
        "boundary_rule_ran": bool(instrument["boundary_rule_ran"]),
        "boundary_rule_changed_fit": bool(instrument["boundary_rule_changed_fit"]),
        "llf_kept": instrument["llf_kept"],
        "llf_first_valid": instrument["llf_first_valid"],
        "checks": checks,
        "passed": all(checks.values()),
    }

