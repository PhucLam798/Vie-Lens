"""E7 component and attention-head attribution helpers (OPS-37)."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .derived import stream

LANGUAGES = ("vi", "en", "zh", "fr", "id")
DIRECTIONS = ("v_vi", "d_vi_en")


def replay_capture_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    stable_key: Callable[[Mapping[str, Any]], Any],
    shard_key: Callable[[Mapping[str, Any]], Hashable],
    batch_size: int,
    shard_order_key: Callable[[Hashable], Any] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild consecutive source-pass capture batches within each shard."""
    if batch_size <= 0:
        raise ValueError("replay capture batch size must be positive")
    grouped: dict[Hashable, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(shard_key(record), []).append(record)
    order_key = shard_order_key or (lambda value: value)
    batches: list[dict[str, Any]] = []
    for shard in sorted(grouped, key=order_key):
        ordered = sorted(grouped[shard], key=stable_key)
        for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
            batches.append({
                "shard": shard,
                "batch_index": batch_index,
                "start_index": start,
                "records": list(ordered[start : start + batch_size]),
            })
    return batches


def replay_unit_keys(
    *,
    family: str,
    format_name: str,
    pass_name: str,
    replay_source: str,
    batch_count: int,
    item_sets: Sequence[str],
) -> list[dict[str, str | int]]:
    """Enumerate result-unit keys for every replayed batch and item set."""
    if batch_count < 0:
        raise ValueError("replay batch count must be non-negative")
    if not family or not format_name or not pass_name or not replay_source:
        raise ValueError("replay unit key fields must be non-empty")
    if not item_sets or any(not value for value in item_sets):
        raise ValueError("replay unit keys require non-empty item-set names")
    return [
        {
            "family": family,
            "pass": pass_name,
            "format": format_name,
            "item_set": item_set,
            "chunk": chunk,
            "replay_source": replay_source,
        }
        for chunk in range(batch_count)
        for item_set in item_sets
    ]


def replay_unit_keys_from_summary(
    summary: Mapping[str, Any],
    *,
    family: str,
    format_name: str,
    pass_name: str,
    replay_source: str,
    item_sets: Sequence[str],
) -> list[dict[str, str | int]]:
    """Enumerate unit keys from the source-pass replay count recorded by a job."""
    try:
        batch_count = int(summary["families"][family][format_name]["replay_batch_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"summary lacks replay batch count for {family}/{format_name}") from exc
    return replay_unit_keys(
        family=family,
        format_name=format_name,
        pass_name=pass_name,
        replay_source=replay_source,
        batch_count=batch_count,
        item_sets=item_sets,
    )


def exact_replay_gate_summary(errors: Sequence[float] | np.ndarray, limit: float) -> dict[str, Any]:
    """Summarize the OPS-39 maximum relative-error gate using its inclusive bound."""
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    maximum = float(np.max(values[finite])) if finite.any() else float("nan")
    return {
        "maximum": maximum,
        "limit": float(limit),
        "passed": bool(values.size and finite.all() and maximum <= float(limit)),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return an fp32 unit vector, rejecting zero or non-finite directions."""
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("cannot normalize a zero or non-finite E7 direction")
    return value / np.float32(norm)


def rebuild_a6_directions(
    residuals: np.ndarray,
    index_rows: Sequence[Mapping[str, Any]],
    *,
    format_name: str,
    languages: Sequence[str] = LANGUAGES,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Rebuild unit v_vi and d=mu_vi-mu_en through the frozen A6 fit helper."""
    from .analysis.la import fit_layer_directions

    values = np.asarray(residuals)
    if values.ndim != 3 or values.dtype != np.float32:
        raise ValueError("P-D residuals must be an fp32 [rows, residual-layer, hidden] array")
    row_map: dict[int, Mapping[str, Any]] = {}
    for fallback, row in enumerate(index_rows):
        array_row = int(row.get("row", fallback))
        if array_row in row_map:
            raise ValueError(f"duplicate P-D row coordinate {array_row}")
        row_map[array_row] = row
    fit_rows = sorted(
        array_row
        for array_row, row in row_map.items()
        if str(row.get("format")) == format_name and str(row.get("split")) == "fit"
    )
    if not fit_rows or fit_rows[-1] >= values.shape[0]:
        raise ValueError(f"P-D fit rows are missing or out of range for {format_name}")
    labels = [str(row_map[row].get("lang")) for row in fit_rows]
    counts = {language: labels.count(language) for language in languages}
    if any(counts[language] == 0 for language in languages):
        raise ValueError(f"P-D fit split is missing a required language: {counts}")
    # Residual row zero is the embedding output.  OPS-37 projects block b's
    # update onto residual row b for b in 1..N, so row zero is retained only
    # as an unused zero sentinel in these block-direction arrays.
    v_vi = np.zeros((values.shape[1], values.shape[2]), dtype=np.float32)
    d_vi_en = np.zeros_like(v_vi)
    for layer in range(values.shape[1]):
        if layer == 0:
            continue
        fit = np.asarray(values[fit_rows, layer, :], dtype=np.float32)
        _center, directions, means = fit_layer_directions(fit, labels, class_order=languages)
        v_vi[layer] = _unit(directions["vi"])
        d_vi_en[layer] = _unit(np.asarray(means["vi"] - means["en"], dtype=np.float32))
    return v_vi, d_vi_en, counts


def direction_match_max_relative(d_vi_en: np.ndarray, e8_u_en_minus_vi: np.ndarray) -> float:
    """Return max|d-(-u)| divided by max|d| for the OPS-37 E8 sign check."""
    d = np.asarray(d_vi_en, dtype=np.float32)
    u = np.asarray(e8_u_en_minus_vi, dtype=np.float32)
    if d.shape != u.shape or d.ndim != 2:
        raise ValueError("E7 and E8 direction arrays must share [layer, hidden] shape")
    denominator = float(np.max(np.abs(d)))
    if denominator == 0 or not math.isfinite(denominator):
        raise ValueError("E7 contrast direction has no finite nonzero component")
    return float(np.max(np.abs(d + u)) / denominator)


def resolve_bands(
    *,
    n_layers: int,
    l_cross: int | None,
    l_peak: int | None,
    mean_delta_vi: Sequence[float],
    gate_pass: Mapping[int, bool],
    early_blocks: Sequence[int],
) -> dict[str, Any]:
    """Resolve OPS-43 primary and sensitivity bands from fixed landmarks and gates."""
    if n_layers < 1 or len(mean_delta_vi) != n_layers + 1:
        raise ValueError("mean_delta_vi must contain residual layers 0..N")
    if any(layer not in gate_pass for layer in range(1, n_layers + 1)):
        raise ValueError("gate_pass must name every block layer")
    early = sorted({int(layer) for layer in early_blocks})
    if any(layer < 1 or layer > n_layers for layer in early):
        raise ValueError("early_blocks must contain block layers in 1..N")
    early_max = max(early, default=0)
    primary: list[int] = []
    if l_cross is not None:
        crossing = int(l_cross)
        if not 1 <= crossing <= n_layers:
            raise ValueError("L_cross must be a block layer in 1..N")
        if l_peak is None or int(l_peak) > crossing:
            start = crossing - 2
        else:
            start = min(int(l_peak), crossing - 2)
        start = max(1, start)
        end = crossing
        if end - start + 1 < 3:
            start = max(1, end - 2)
        primary = list(range(start, end + 1))

    curve = np.asarray(mean_delta_vi, dtype=np.float64)
    slopes = curve[1:] - curve[:-1]
    if not np.isfinite(slopes).all():
        raise ValueError("the concept-mean delta_vi slope curve must be finite at layers 0..N")
    size = int(math.ceil(n_layers / 4))
    eligible = [
        block for block in range(1, n_layers + 1)
        if block > early_max and bool(gate_pass[block])
    ]

    def top_slopes(candidates: Sequence[int]) -> list[int]:
        """Select the largest slopes, with deeper blocks winning ties."""
        selected = sorted(
            candidates,
            key=lambda block: (float(slopes[block - 1]), block),
            reverse=True,
        )[: min(size, len(candidates))]
        return sorted(selected)

    eligible_slope = top_slopes(eligible)
    ops37_as_written = top_slopes(list(range(1, n_layers + 1)))
    if l_cross is None:
        primary = list(eligible_slope)

    def describe(
        blocks: Sequence[int], *, pool: Sequence[int], used_all_eligible: bool = False
    ) -> dict[str, Any]:
        selected = list(blocks)
        excluded = [block for block in selected if not gate_pass[block]]
        included = [block for block in selected if gate_pass[block]]
        hull = [min(selected), max(selected)] if selected else None
        return {
            "blocks": selected,
            "contiguous_hull": hull,
            "included_blocks": included,
            "excluded_gate_failed_blocks": excluded,
            "eligible_count": len(eligible),
            "requested_size": size,
            "selection_pool_count": len(pool),
            "used_all_eligible": bool(used_all_eligible),
        }

    used_all_eligible = len(eligible) < size
    return {
        "rule": "i" if l_cross is not None else "ii",
        "early_blocks": early,
        "eligible_blocks": eligible,
        "primary": describe(
            primary,
            pool=range(1, n_layers + 1) if l_cross is not None else eligible,
            used_all_eligible=l_cross is None and used_all_eligible,
        ),
        "slope_sensitivity": describe(
            eligible_slope, pool=eligible, used_all_eligible=used_all_eligible
        ),
        "ops37_as_written": describe(
            ops37_as_written, pool=range(1, n_layers + 1)
        ),
        "slope_values": {layer: float(slopes[layer - 1]) for layer in range(1, n_layers + 1)},
    }


def primary_landmark_point(frame: Any, *, model: str, format_name: str, metric: str) -> int | None:
    """Select exactly one OPS-37-5 primary-s, cf729, set-1 landmark point row."""
    import pandas as pd

    required = {
        "model", "format", "pass", "condition", "fewshot_set", "role", "item_set",
        "metric_variant", "metric", "value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"landmark table lacks explicit selection columns: {sorted(missing)}")
    selected = frame.loc[
        frame["model"].astype(str).eq(model)
        & frame["format"].astype(str).eq(format_name)
        & frame["pass"].astype(str).eq("analysis")
        & frame["condition"].astype(str).eq("diac")
        & pd.to_numeric(frame["fewshot_set"], errors="coerce").eq(1)
        & frame["role"].astype(str).eq("path")
        & frame["item_set"].astype(str).eq("cf729")
        & frame["metric_variant"].astype(str).eq("s")
        & frame["metric"].astype(str).eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"OPS-37-5 {metric} row count is {len(selected)} for {model}/{format_name}; expected exactly one"
        )
    raw = pd.to_numeric(selected["value"], errors="coerce").iloc[0]
    if pd.isna(raw):
        return None
    point = float(raw)
    return int(round(point)) if np.isfinite(point) else None


def attention_head_contributions(
    z: Any,
    o_proj_weight: Any,
    *,
    n_heads: int,
    head_dim: int,
    total_attention: Any | None = None,
    gemma_post_norm_weight: Any | None = None,
    gemma_post_norm_eps: float = 1e-6,
) -> Any:
    """Recover each head's o_proj contribution; linearize Gemma's post-attention RMSNorm."""
    import torch

    z32 = torch.as_tensor(z).float()
    weight32 = torch.as_tensor(o_proj_weight).float()
    if z32.shape[-1] != n_heads * head_dim:
        raise ValueError("o_proj input width differs from num_heads × head_dim")
    if weight32.ndim != 2 or weight32.shape[1] != n_heads * head_dim:
        raise ValueError("o_proj weight must have [hidden, num_heads × head_dim] shape")
    grouped = z32.reshape(*z32.shape[:-1], n_heads, head_dim)
    grouped_weight = weight32.reshape(weight32.shape[0], n_heads, head_dim)
    contributions = torch.einsum("...hd,ohd->...ho", grouped, grouped_weight)
    if gemma_post_norm_weight is not None:
        if total_attention is None:
            raise ValueError("Gemma post-norm attribution requires the total pre-norm attention output")
        total = torch.as_tensor(total_attention).float()
        gain_weight = 1.0 + torch.as_tensor(gemma_post_norm_weight).float()
        rms = torch.sqrt(total.square().mean(dim=-1, keepdim=True) + float(gemma_post_norm_eps))
        gain = gain_weight / rms
        contributions = contributions * gain.unsqueeze(-2)
    return contributions


def project_vectors(vectors: Any, directions: Any) -> Any:
    """Project vectors onto one or more unit directions in fp32."""
    import torch

    x = torch.as_tensor(vectors).float()
    d = torch.as_tensor(directions, device=x.device).float()
    if x.shape[-1] != d.shape[-1] or d.ndim != 2:
        raise ValueError("directions must be [n_directions, hidden] and match vector width")
    return torch.einsum("...d,qd->...q", x, d)


def vector_norms(vectors: Any) -> Any:
    """Return fp32 L2 norms along the final hidden dimension."""
    import torch

    return torch.linalg.vector_norm(torch.as_tensor(vectors).float(), dim=-1)


def relative_errors(reference: Any, estimate: Any, *, epsilon: float = 1e-12) -> np.ndarray:
    """Return rowwise L2 relative error with the reference norm as denominator."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape or ref.ndim < 1:
        raise ValueError("relative-error inputs must have equal shapes and a hidden dimension")
    numerator = np.linalg.norm((ref - est).reshape(-1, ref.shape[-1]), axis=1)
    denominator = np.linalg.norm(ref.reshape(-1, ref.shape[-1]), axis=1)
    return numerator / np.maximum(denominator, float(epsilon))


def captured_component_errors(
    resid_in: Any,
    attn: Any,
    mlp: Any,
    resid_out: Any,
    head_sum: Any,
) -> dict[str, np.ndarray]:
    """Compute E7 checks from the exact captured bf16 tensors before NumPy conversion."""
    import torch

    tensors = (resid_in, attn, mlp, resid_out)
    if any(not isinstance(value, torch.Tensor) for value in (*tensors, head_sum)):
        raise TypeError("E7 captured activations and head sum must be torch tensors")
    if any(value.dtype != torch.bfloat16 for value in tensors):
        raise TypeError("E7 model-order operands must be the captured bf16 tensors")
    if any(value.shape != resid_in.shape for value in (*tensors[1:], head_sum)):
        raise ValueError("E7 captured component tensors must have matching [batch, hidden] shapes")

    # Preserve the decoder's model-order bf16 additions before converting any operand.
    recon = (resid_in + attn) + mlp
    resid_in32 = resid_in.detach().float().cpu().numpy()
    attn32 = attn.detach().float().cpu().numpy()
    mlp32 = mlp.detach().float().cpu().numpy()
    resid_out32 = resid_out.detach().float().cpu().numpy()
    recon32 = recon.detach().float().cpu().numpy()
    head_sum32 = head_sum.detach().float().cpu().numpy()

    residual_delta32 = resid_out32 - resid_in32
    update_sum32 = attn32 + mlp32
    return {
        "head_sum": relative_errors(attn32, head_sum32),
        "residual_model_order": relative_errors(resid_out32, recon32),
        "residual_update_relative": relative_errors(residual_delta32, update_sum32),
    }


def error_summary(values: Sequence[float], *, median_limit: float, p99_limit: float | None = None) -> dict[str, float | bool]:
    """Summarize an integrity-error distribution and apply OPS-37 thresholds."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"median": float("nan"), "p99": float("nan"), "maximum": float("nan"), "passed": False}
    median = float(np.median(array))
    p99 = float(np.quantile(array, 0.99))
    passed = median <= median_limit and (p99_limit is None or p99 <= p99_limit)
    return {"median": median, "p99": p99, "maximum": float(np.max(array)), "passed": bool(passed)}


def attention_share(attention_sum: float, mlp_sum: float, *, epsilon: float = 1e-8) -> float:
    """Compute OPS-37 absolute attention share, returning NaN for a tiny denominator."""
    a = float(attention_sum)
    m = float(mlp_sum)
    denominator = abs(a) + abs(m)
    if not math.isfinite(denominator) or denominator < epsilon:
        return float("nan")
    return abs(a) / denominator


def bootstrap_indices(
    concept_ids: Sequence[str], *, seed: int, key: str, n_resamples: int = 1000
) -> np.ndarray:
    """Draw deterministic concept-level bootstrap indices from the registered stream."""
    ids = np.asarray(list(concept_ids), dtype=object)
    if ids.ndim != 1 or not ids.size or n_resamples <= 0:
        raise ValueError("bootstrap requires non-empty concepts and positive resamples")
    rng = stream(int(seed), str(key))
    return rng.integers(0, len(ids), size=(int(n_resamples), len(ids)), endpoint=False)


def percentile_ci(values: Sequence[float], *, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    """Return a finite percentile interval for bootstrap values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def _mean_ci(matrix: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    draws = matrix[indices].mean(axis=1, dtype=np.float64)
    return percentile_ci(draws)


def summarize_attribution(
    projections: pd.DataFrame,
    *,
    model: str,
    format_name: str,
    item_set: str,
    bands: Mapping[str, Mapping[str, Any]],
    seed: int,
    key_template: str,
    n_resamples: int = 1000,
    top_n: int = 10,
    share_epsilon: float = 1e-8,
    gate_pass: Mapping[int, bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build layer, band, within-band head-rank, and full-depth head summaries."""
    required = {"concept_id", "layer", "kind", "head", "direction", "projection", "update_norm"}
    if not required.issubset(projections.columns):
        raise ValueError(f"projection table lacks {sorted(required - set(projections.columns))}")
    frame = projections.copy()
    frame["concept_id"] = frame["concept_id"].astype(str)
    concepts = sorted(frame["concept_id"].unique())
    boot_indices = bootstrap_indices(
        concepts,
        seed=seed,
        key=key_template.format(model=model, format=format_name),
        n_resamples=n_resamples,
    )
    layers_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    component = frame[frame["kind"].isin(["attn", "mlp"])]
    for direction in DIRECTIONS:
        directed = component[component["direction"] == direction]
        layer_types = (
            directed.groupby("layer", sort=True)["layer_type"].first().to_dict()
            if "layer_type" in directed.columns else {}
        )
        pivot = directed.pivot_table(
            index=["concept_id", "layer"], columns="kind", values="projection", aggfunc="mean"
        )
        norm_pivot = directed.pivot_table(
            index=["concept_id", "layer"], columns="kind", values="update_norm", aggfunc="mean"
        )
        pivot = pivot.reindex(pd.MultiIndex.from_product([concepts, sorted(directed["layer"].unique())], names=["concept_id", "layer"]))
        norm_pivot = norm_pivot.reindex(pivot.index)
        for layer in sorted(directed["layer"].astype(int).unique()):
            block = pivot.xs(layer, level="layer").reindex(concepts)
            block_norm = norm_pivot.xs(layer, level="layer").reindex(concepts)
            for kind in ("attn", "mlp"):
                values = block[kind].to_numpy(dtype=np.float64)
                lo, hi = _mean_ci(values, boot_indices)
                norm_values = block_norm[kind].to_numpy(dtype=np.float64)
                layers_rows.append({
                    "model": model, "format": format_name, "item_set": item_set,
                    "direction": direction, "layer": int(layer),
                    "layer_type": str(layer_types.get(layer, "standard")),
                    "gate_pass": bool(gate_pass[layer]) if gate_pass is not None and layer in gate_pass else None,
                    "component": kind,
                    "mean_projection": float(np.mean(values)), "ci_low": lo, "ci_high": hi,
                    "mean_update_norm": float(np.mean(norm_values)), "concepts": len(concepts),
                })
            layer_share_draws = []
            a_values = block["attn"].to_numpy(dtype=np.float64)
            m_values = block["mlp"].to_numpy(dtype=np.float64)
            for selected in boot_indices:
                layer_share_draws.append(attention_share(float(a_values[selected].mean()), float(m_values[selected].mean()), epsilon=share_epsilon))
            share_ci = percentile_ci(layer_share_draws)
            layers_rows.append({
                "model": model, "format": format_name, "item_set": item_set,
                "direction": direction, "layer": int(layer),
                "layer_type": str(layer_types.get(layer, "standard")),
                "gate_pass": bool(gate_pass[layer]) if gate_pass is not None and layer in gate_pass else None,
                "component": "share",
                "mean_projection": attention_share(float(a_values.mean()), float(m_values.mean()), epsilon=share_epsilon),
                "ci_low": share_ci[0], "ci_high": share_ci[1],
                "mean_update_norm": np.nan, "concepts": len(concepts),
            })
        for band_name in ("primary", "slope_sensitivity", "ops37_as_written"):
            if band_name not in bands:
                raise ValueError(f"E7 band mapping is missing {band_name}")
            band_spec = bands[band_name]
            selected_blocks = sorted(set(int(value) for value in band_spec["blocks"]))
            included_blocks = sorted(set(int(value) for value in band_spec["included_blocks"]))
            excluded_blocks = sorted(set(int(value) for value in band_spec["excluded_gate_failed_blocks"]))
            metadata = {
                "model": model,
                "format": format_name,
                "item_set": item_set,
                "direction": direction,
                "band": band_name,
                "band_label": band_name,
                "blocks": ";".join(map(str, selected_blocks)),
                "included_blocks": ";".join(map(str, included_blocks)),
                "excluded_gate_failed_blocks": ";".join(map(str, excluded_blocks)),
                "contiguous_hull": ";".join(map(str, band_spec["contiguous_hull"]))
                if band_spec["contiguous_hull"] is not None else "",
                "eligible_count": int(band_spec["eligible_count"]),
                "requested_size": int(band_spec["requested_size"]),
                "selection_pool_count": int(band_spec["selection_pool_count"]),
                "used_all_eligible": bool(band_spec["used_all_eligible"]),
                "concepts": len(concepts),
            }
            band_frame = pivot.loc[
                pivot.index.get_level_values("layer").isin(included_blocks)
            ]
            if included_blocks:
                by_concept = band_frame.groupby(level="concept_id", sort=True).sum(min_count=1).reindex(concepts)
                a_values = by_concept["attn"].to_numpy(dtype=np.float64)
                m_values = by_concept["mlp"].to_numpy(dtype=np.float64)
                a_ci = _mean_ci(a_values, boot_indices)
                m_ci = _mean_ci(m_values, boot_indices)
                share_draws = [
                    attention_share(float(a_values[index].mean()), float(m_values[index].mean()), epsilon=share_epsilon)
                    for index in boot_indices
                ]
                share_ci = percentile_ci(share_draws)
                estimates = {
                    "attn": (float(a_values.mean()), a_ci, int(np.sign(a_values.mean()))),
                    "mlp": (float(m_values.mean()), m_ci, int(np.sign(m_values.mean()))),
                    "share": (
                        attention_share(float(a_values.mean()), float(m_values.mean()), epsilon=share_epsilon),
                        share_ci,
                        np.nan,
                    ),
                }
            else:
                estimates = {
                    component: (np.nan, (np.nan, np.nan), np.nan)
                    for component in ("attn", "mlp", "share")
                }
            for component_name, (estimate, ci, sign) in estimates.items():
                band_rows.append({
                    **metadata,
                    "component": component_name,
                    "estimate": estimate,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "sign": sign,
                })
    all_heads = frame[(frame["kind"] == "head") & (frame["direction"] == "d_vi_en")].copy()
    if not all_heads.empty and "layer_type" not in all_heads.columns:
        all_heads["layer_type"] = "standard"

    def pair_summaries(head_frame: pd.DataFrame, *, rank_pairs: bool) -> list[dict[str, Any]]:
        """Summarize distinct (layer, head) pairs, optionally ranking within each type."""
        if head_frame.empty:
            return []
        per_concept = head_frame.groupby(
            ["concept_id", "layer", "head", "layer_type"], as_index=False, sort=True
        )["projection"].mean()
        output_rows: list[dict[str, Any]] = []
        for layer_type in sorted(per_concept["layer_type"].astype(str).unique()):
            typed = per_concept[per_concept["layer_type"].astype(str) == layer_type]
            pair_keys = sorted({
                (int(row.layer), int(row.head))
                for row in typed[["layer", "head"]].itertuples(index=False)
            })
            pair_columns = pd.MultiIndex.from_tuples(pair_keys, names=["layer", "head"])
            matrix = typed.pivot(
                index="concept_id", columns=["layer", "head"], values="projection"
            ).reindex(index=concepts, columns=pair_columns)
            values = matrix.to_numpy(dtype=np.float64)
            means = values.mean(axis=0)
            positive_order: list[int] = []
            negative_order: list[int] = []
            top_pos_draws = np.zeros((n_resamples, len(pair_keys)), dtype=np.float64)
            top_neg_draws = np.zeros_like(top_pos_draws)
            if rank_pairs:
                positive_order = sorted(
                    (index for index, value in enumerate(means) if value > 0),
                    key=lambda index: (-means[index], pair_keys[index]),
                )[:top_n]
                negative_order = sorted(
                    (index for index, value in enumerate(means) if value < 0),
                    key=lambda index: (means[index], pair_keys[index]),
                )[:top_n]
                for draw_index, selected in enumerate(boot_indices):
                    draw_means = values[selected].mean(axis=0)
                    positive = sorted(
                        (index for index, value in enumerate(draw_means) if value > 0),
                        key=lambda index: (-draw_means[index], pair_keys[index]),
                    )[:top_n]
                    negative = sorted(
                        (index for index, value in enumerate(draw_means) if value < 0),
                        key=lambda index: (draw_means[index], pair_keys[index]),
                    )[:top_n]
                    top_pos_draws[draw_index, positive] = 1.0
                    top_neg_draws[draw_index, negative] = 1.0
            positive_rank = {pair_keys[index]: rank for rank, index in enumerate(positive_order, 1)}
            negative_rank = {pair_keys[index]: rank for rank, index in enumerate(negative_order, 1)}
            for column, pair in enumerate(pair_keys):
                positive_rank_value = positive_rank.get(pair)
                negative_rank_value = negative_rank.get(pair)
                if rank_pairs and positive_rank_value is None and negative_rank_value is None:
                    continue
                ci = _mean_ci(values[:, column], boot_indices)
                output_rows.append({
                    "model": model,
                    "format": format_name,
                    "item_set": item_set,
                    "direction": "d_vi_en",
                    "layer": pair[0],
                    "head": pair[1],
                    "layer_type": layer_type,
                    "mean_projection": float(means[column]),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "top_positive_inclusion": float(top_pos_draws[:, column].mean()) if rank_pairs else np.nan,
                    "top_negative_inclusion": float(top_neg_draws[:, column].mean()) if rank_pairs else np.nan,
                    "top_positive_rank": positive_rank_value,
                    "top_negative_rank": negative_rank_value,
                    "primary_band": ";".join(map(str, sorted(set(map(int, bands["primary"]["included_blocks"]))))),
                    "concepts": len(concepts),
                })
        return output_rows

    heads_all_rows = pair_summaries(all_heads, rank_pairs=False)
    primary_band = set(int(value) for value in bands["primary"]["included_blocks"])
    within_band_heads = all_heads[all_heads["layer"].astype(int).isin(primary_band)]
    head_rows.extend(pair_summaries(within_band_heads, rank_pairs=True))
    return (
        pd.DataFrame(layers_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(head_rows),
        pd.DataFrame(heads_all_rows),
    )


def gate_rows(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten named E7 check summaries into stable report rows."""
    return [
        {"check": name, **dict(summary)}
        for name, summary in sorted(results.items())
    ]
