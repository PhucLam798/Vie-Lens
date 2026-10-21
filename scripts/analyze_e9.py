#!/datastore/luannt/.cache/sb/bin/python
"""Analyze paired E9 nodiac and diac P-L results (descriptive)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens.analysis.bootstrap import (
    bootstrap_means,
    concept_indices,
    percentile_interval,
)
from vilens.config import load_config
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.io import unit_paths

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation", "cloze")
BLOCK_FORMATS = ("repetition", "translation")
LANGUAGES = ("vi", "en", "zh", "fr", "id")
LONG_COLUMNS = [
    "model", "model_revision", "pass", "format", "condition", "fewshot_set",
    "concept_id", "target_lang", "target_variant", "role", "ref_concept_id",
    "tok_idx", "token_id", "is_content", "layer", "metric", "value",
]


def _is_true(value: Any) -> bool:
    """Interpret the explicit JSON boolean representation used in source files."""
    return value is True or str(value).strip().lower() == "true"


def _records_for_e9(cfg: Mapping[str, Any], repo: Path) -> list[dict[str, Any]]:
    """Load the configured E9 records without writing or printing their text."""
    paths = prompt_file_paths(dict(cfg))
    rows: list[dict[str, Any]] = []
    for stem in cfg["pl"]["e9"]["files"]:
        path = Path(paths[str(stem)])
        if not path.is_absolute():
            path = repo / path
        rows.extend(load_prompt_file(path))
    return rows


def _records_by_format(
    records: Sequence[Mapping[str, Any]], main_ids: set[str]
) -> dict[str, set[str]]:
    """Select non-extension test IDs by registered format."""
    output = {fmt: set() for fmt in FORMATS}
    for record in records:
        fmt = str(record.get("format", ""))
        concept_id = str(record.get("concept_id", ""))
        if (
            fmt in output
            and str(record.get("split", "")) == "test"
            and concept_id in main_ids
            and not _is_true(record.get("m1_extension", False))
        ):
            output[fmt].add(concept_id)
    return output


def _paired_scopes(
    cfg: Mapping[str, Any], repo: Path
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Resolve paired main and cf729 scopes from the two configured prompt sets."""
    from scripts import analyze_core as core

    main_ids, cf_ids = core._primary_scope_ids(repo, cfg)
    diac_records = core._primary_prompt_records(cfg, repo)
    nodiac_records = _records_for_e9(cfg, repo)
    diac = _records_by_format(diac_records, main_ids)
    nodiac = _records_by_format(nodiac_records, main_ids)
    paired = {fmt: diac[fmt] & nodiac[fmt] for fmt in FORMATS}
    paired_cf = {fmt: paired[fmt] & cf_ids for fmt in FORMATS}
    expected = cfg["pl"]["e9"]["expected"]
    for fmt in FORMATS:
        wanted = int(expected[fmt])
        if len(nodiac[fmt]) != wanted:
            raise AssertionError(f"E9 {fmt} non-collapsed prompt count={len(nodiac[fmt])}, expected {wanted}")
        if len(paired[fmt]) != wanted:
            raise AssertionError(f"E9 {fmt} paired main count={len(paired[fmt])}, expected {wanted}")
        if len(paired_cf[fmt]) > len(paired[fmt]):
            raise AssertionError(f"E9 {fmt} cf729 scope exceeds paired main scope")
    return paired, paired_cf, diac, nodiac


def _unit_key(family: str, fmt: str, shard: int) -> dict[str, str | int]:
    """Build an E9 P-L unit key."""
    return {
        "family": family,
        "pass": "pl",
        "condition": "nodiac",
        "fewshot_set": "none" if fmt == "cloze" else "1",
        "format": fmt,
        "shard": shard,
    }


def _e9_paths(cfg: Mapping[str, Any], repo: Path, family: str, fmt: str) -> list[Path]:
    """Return the configured E9 P-L units in shard order."""
    root = repo / str(cfg["pl"]["e9"]["results_dir"])
    expected_prompts = int(cfg["pl"]["e9"]["expected"][fmt])
    shard_size = int(cfg["pl"]["shard_prompts"])
    count = (expected_prompts + shard_size - 1) // shard_size
    return [unit_paths(root, _unit_key(family, fmt, shard))[0] for shard in range(count)]


def _primary_paths(cfg: Mapping[str, Any], repo: Path, family: str, fmt: str) -> list[Path]:
    """Return primary set-1 P-L units for one model-format cell."""
    from scripts import analyze_core as core

    return [
        path for path in core._primary_paths(repo, cfg, family=family)
        if f"format={fmt}__" in path.name
    ]


def _index_ids(paths: Sequence[Path], expected_ids: set[str]) -> list[str]:
    """Read only row identifiers from residual index sidecars."""
    rows: list[str] = []
    for parquet_path in paths:
        index_path = parquet_path.with_suffix(".resid_index.jsonl")
        with index_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    concept_id = str(json.loads(line)["concept_id"])
                    if concept_id in expected_ids:
                        rows.append(concept_id)
    return rows


def _input_complete(parquet_paths: Sequence[Path], expected_ids: set[str]) -> bool:
    """Check all P-L companions and exact concept coverage for one condition."""
    if not parquet_paths:
        return False
    required_suffixes = (
        ".provenance.json", ".resid.npy", ".attn.npy", ".mlp.npy",
        ".resid_index.jsonl", ".generations.jsonl",
    )
    for parquet_path in parquet_paths:
        if not parquet_path.is_file() or any(
            not parquet_path.with_suffix(suffix).is_file() for suffix in required_suffixes
        ):
            return False
    ids = _index_ids(parquet_paths, expected_ids)
    return len(ids) == len(set(ids)) == len(expected_ids) and set(ids) == expected_ids


def preflight(
    cfg: Mapping[str, Any], *, repo: Path = REPO
) -> tuple[str, bool, dict[str, Any]]:
    """Check E9 paired scopes and complete E9/primary P-L units without analysis."""
    paired, paired_cf, _diac_records, _nodiac_records = _paired_scopes(cfg, repo)
    rows: list[str] = []
    all_ok = True
    for family in FAMILIES:
        for fmt in FORMATS:
            expected_ids = paired[fmt]
            e9_paths = _e9_paths(cfg, repo, family, fmt)
            primary_paths = _primary_paths(cfg, repo, family, fmt)
            e9_ok = _input_complete(e9_paths, expected_ids)
            primary_ok = _input_complete(primary_paths, expected_ids)
            e9_count = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in e9_paths if path.is_file())
            primary_count = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in primary_paths if path.is_file())
            scope_status = "PASS" if e9_ok and primary_ok else "MISSING/FAIL"
            all_ok &= e9_ok and primary_ok
            rows.append(
                f"{family} | {fmt} | {len(expected_ids)} | {len(paired_cf[fmt])} | "
                f"{e9_count} | {primary_count} | {len(e9_paths)} | {scope_status}"
            )
    output_paths = [repo / str(cfg["e9_analysis"][name]) for name in ("report", "csv")]
    outputs_absent = not any(path.exists() for path in output_paths)
    all_ok &= outputs_absent
    table = "model | format | paired_main | paired_cf729 | E9 long rows | diac long rows | E9 units | inputs\n"
    table += "--- | --- | ---: | ---: | ---: | ---: | ---: | ---\n" + "\n".join(rows)
    table += f"\nOutputs absent={outputs_absent}; P-L inputs complete={all_ok and outputs_absent}"
    bundle = {"paired": paired, "paired_cf": paired_cf}
    return table, all_ok and outputs_absent, bundle


def _read_long(paths: Sequence[Path], ids: set[str], fmt: str) -> pd.DataFrame:
    """Read selected long-format P-L rows for one cell, never retaining prompt text."""
    parts: list[pd.DataFrame] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=LONG_COLUMNS, batch_size=65_536):
            frame = batch.to_pandas()
            frame = frame.loc[
                frame["format"].astype(str).eq(fmt)
                & frame["concept_id"].astype(str).isin(ids)
                & frame["metric"].astype(str).eq("lens_logprob")
                & frame["role"].astype(str).isin({"correct", "distractor"})
            ]
            if not frame.empty:
                parts.append(frame)
    if not parts:
        raise ValueError(f"No P-L content rows for {fmt} selected scope")
    return pd.concat(parts, ignore_index=True)


def _read_generations(
    paths: Sequence[Path], *, family: str, fmt: str, condition: str, ids: set[str]
) -> list[dict[str, Any]]:
    """Read generation labels for selected concepts without exposing answer text."""
    output: list[dict[str, Any]] = []
    for parquet_path in paths:
        generation_path = parquet_path.with_suffix(".generations.jsonl")
        with generation_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                concept_id = str(record.get("concept_id", ""))
                if concept_id in ids and str(record.get("format")) == fmt:
                    if _is_true(record.get("m1_extension", False)):
                        raise AssertionError("E9 paired generation rows include an M1 extension")
                    output.append({
                        "model": family,
                        "format": fmt,
                        "condition": condition,
                        "fewshot_set": 1,
                        "concept_id": concept_id,
                        "answer": str(record.get("text", "")),
                    })
    observed = [str(row["concept_id"]) for row in output]
    if len(observed) != len(set(observed)) or set(observed) != ids:
        raise AssertionError(f"E9 {family}/{fmt}/{condition} generation lookup is not one-to-one")
    return output


def _paired_boot_key(model: str, fmt: str, scope: str) -> str:
    """Use the OPS-45 shared key for main, with its registered cf729 suffix."""
    base = f"bootstrap|e9|{model}|{fmt}"
    return base if scope == "paired_main" else base + "|cf729"


def _normalized_scores(raw: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create frozen score, delta, readout, and rise tables for one condition."""
    from scripts import analyze_core as core

    scores = core.content_token_scores(raw)
    if condition == "nodiac":
        is_vi = scores["target_lang"].astype(str).eq("vi")
        scores.loc[is_vi, "target_variant"] = "released"
    deltas = core.matched_distractor_deltas(scores)
    readouts = core._readout_rows(deltas)
    correct = scores.loc[scores["role"].astype(str).eq("correct")]
    score_readouts = core._readout_rows(correct)
    return scores, readouts, score_readouts, deltas


def _rise_for_metric(
    readouts: pd.DataFrame, cfg: Mapping[str, Any], value_column: str
) -> pd.DataFrame:
    from scripts import analyze_core as core

    return core.concept_rise_curves(
        readouts,
        early_depth_max=float(cfg["e1"]["early_depth_max"]),
        value_column=value_column,
    )


def _rule_values(
    rules: pd.DataFrame, family: str, fmt: str, metric_variant: str
) -> dict[str, float]:
    selected = rules.loc[
        rules["model"].astype(str).eq(family)
        & rules["format"].astype(str).eq(fmt)
        & rules["metric_variant"].astype(str).eq(metric_variant)
        & rules["target_lang"].astype(str).eq("__H1_H2__")
    ]
    if len(selected) != 1:
        raise AssertionError(f"E9 rule summary is non-unique for {family}/{fmt}/{metric_variant}")
    row = selected.iloc[0]
    return {
        "Q_H1": float(row["h1_q"]),
        "H1_ratio": float(row["h1_ratio"]) if pd.notna(row["h1_ratio"]) else np.nan,
        "H2_band_ratio": float(row["h2_band_ratio"]) if pd.notna(row["h2_band_ratio"]) else np.nan,
    }


def _rule_bootstrap_values(
    rise_frame: pd.DataFrame, indices: np.ndarray, family: str, fmt: str
) -> dict[str, np.ndarray]:
    """Calculate the E4 quantities on shared paired concept resamples."""
    from scripts import analyze_core as core

    ids, _layers, values = core._complete_matrix(
        rise_frame.loc[
            rise_frame["model"].astype(str).eq(family)
            & rise_frame["format"].astype(str).eq(fmt)
        ],
        languages=LANGUAGES,
        value="rise",
    )
    if len(ids) != indices.shape[1]:
        raise AssertionError("E9 bootstrap IDs do not match paired E4 rows")
    means = bootstrap_means(values, indices)
    maxima = means.max(axis=2)
    control = (maxima[:, 3] + maxima[:, 4]) / 2.0
    h1_ratio = np.divide(
        maxima[:, 1], control,
        out=np.full(len(maxima), np.nan, dtype=np.float64), where=control > 0,
    )
    minimum = maxima.min(axis=1)
    maximum = maxima.max(axis=1)
    h2_ratio = np.divide(
        maximum, minimum, out=np.full(len(maximum), np.nan), where=minimum > 0,
    )
    return {
        "Q_H1": maxima[:, 1] - maxima[:, 3] - maxima[:, 4],
        "H1_ratio": h1_ratio,
        "H2_band_ratio": h2_ratio,
    }


def _max_la_samples(
    la_long: pd.DataFrame, ids: Sequence[str], indices: np.ndarray, family: str, fmt: str
) -> np.ndarray:
    """Bootstrap the maximum concept-mean LA_en over passing blocks."""
    selected = la_long.loc[
        la_long["model"].astype(str).eq(family)
        & la_long["format"].astype(str).eq(fmt)
        & la_long["target_lang"].astype(str).eq("en")
        & la_long["metric"].astype(str).eq("LA")
    ]
    if selected.empty:
        return np.full(indices.shape[0], np.nan, dtype=np.float64)
    matrix_frame = selected.pivot(index="concept_id", columns="layer", values="value")
    layer_columns = sorted(int(value) for value in matrix_frame.columns)
    matrix = matrix_frame.reindex(index=list(ids), columns=layer_columns).to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise AssertionError("E9 LA rows are incomplete for a paired condition")
    return bootstrap_means(matrix, indices).max(axis=1)


def _a10_row(
    generations: Sequence[Mapping[str, Any]], concepts: Mapping[str, Mapping[str, Any]],
    *, family: str, fmt: str, condition: str, scope: str,
) -> dict[str, Any]:
    """Return an A10 count/rate row for one condition and paired scope."""
    from vilens.analysis.competence import competence_rows

    scored = competence_rows(generations, concepts)
    selected = scored.loc[
        scored["model"].astype(str).eq(family)
        & scored["format"].astype(str).eq(fmt)
        & scored["condition"].astype(str).eq(condition)
    ]
    n = len(selected)
    if n == 0:
        raise AssertionError(f"E9 A10 has no rows for {family}/{fmt}/{condition}/{scope}")
    vi_n = int(selected["competence_class"].astype(str).eq("VI").sum())
    return {
        "model": family, "format": fmt, "scope": scope, "condition": condition,
        "metric_variant": "s", "quantity": "A10_vi_rate", "value": vi_n / n,
        "layer": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "a10_vi_n": vi_n, "a10_denominator": n,
    }


def _summarize_scope(
    cfg: Mapping[str, Any], *, family: str, fmt: str, scope: str, ids: set[str],
    diac_raw: pd.DataFrame, nodiac_raw: pd.DataFrame,
    diac_la_long: pd.DataFrame, diac_la_summary: pd.DataFrame,
    nodiac_la_long: pd.DataFrame, nodiac_la_summary: pd.DataFrame,
    diac_generations: Sequence[Mapping[str, Any]],
    nodiac_generations: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute one paired E9 scope from real or synthetic P-L long tables."""
    from scripts import analyze_core as core

    if not ids:
        return []
    template = "bootstrap|e9|{model}|{format}" if scope == "paired_main" else "bootstrap|e9|{model}|{format}|cf729"
    analysis_cfg = {**dict(cfg["analysis"]), "bootstrap_key_template": template}
    resamples = int(analysis_cfg["bootstrap_resamples"])
    output_rows: list[dict[str, Any]] = []
    conditions = {
        "diac": (diac_raw, diac_la_long, diac_la_summary, diac_generations),
        "nodiac": (nodiac_raw, nodiac_la_long, nodiac_la_summary, nodiac_generations),
    }
    stat_by_condition: dict[str, dict[str, Any]] = {}
    for condition, (raw, la_long, la_summary, generations) in conditions.items():
        selected_raw = raw.loc[raw["concept_id"].astype(str).isin(ids)].copy()
        scores, readouts, score_readouts, _deltas = _normalized_scores(selected_raw, condition)
        if scores.empty:
            raise ValueError(f"E9 score table is empty for {family}/{fmt}/{condition}/{scope}")
        cell_model = sorted(scores["model"].astype(str).unique())
        if len(cell_model) != 1:
            raise AssertionError("E9 score rows mix model identities")
        model = cell_model[0]
        rise_s = _rise_for_metric(readouts, cfg, "delta_s")
        rise_s1 = _rise_for_metric(readouts, cfg, "delta_s1")
        rules_parts: list[pd.DataFrame] = []
        landmarks_parts: list[pd.DataFrame] = []
        rises_by_metric: dict[str, pd.DataFrame] = {}
        for metric_variant, rise, score_column in (("s", rise_s, "s"), ("s1", rise_s1, "s1")):
            rules, landmarks, _window = core._bootstrap_core(
                per_concept_rise=rise,
                per_concept_scores=score_readouts,
                source="e9",
                item_set=scope,
                seed=int(cfg["seed"]),
                analysis_config=analysis_cfg,
                score_column=score_column,
                metric_variant=metric_variant,
            )
            rules_parts.append(rules)
            landmarks_parts.append(landmarks)
            rises_by_metric[metric_variant] = rise
        rules_all = pd.concat(rules_parts, ignore_index=True)
        landmarks_all = pd.concat(landmarks_parts, ignore_index=True)
        rules_long = core._summary_metric_long(
            rules_all, source_rows=score_readouts, kind="rule"
        )
        checkpoint, _verdict = core._build_checkpoint_rules(rules_long, la_summary)
        if fmt in BLOCK_FORMATS:
            checkpoint_rows = checkpoint.loc[
                checkpoint["model"].astype(str).eq(model)
                & checkpoint["format"].astype(str).eq(fmt)
            ]
            if len(checkpoint_rows) != 1:
                raise AssertionError(f"E9 outcome lookup is non-unique for {family}/{fmt}/{condition}")
            checkpoint_row = checkpoint_rows.iloc[0]
            outcomes = {"s": checkpoint_row.get("Outcome"), "s1": checkpoint_row.get("s1_outcome")}
        else:
            outcomes = {"s": "descriptive_only", "s1": "descriptive_only"}

        condition_rows: list[dict[str, Any]] = []
        for metric_variant in ("s", "s1"):
            rule_rows = rules_all.loc[
                rules_all["model"].astype(str).eq(model)
                & rules_all["format"].astype(str).eq(fmt)
                & rules_all["metric_variant"].astype(str).eq(metric_variant)
                & rules_all["target_lang"].astype(str).eq("__H1_H2__")
            ]
            if len(rule_rows) != 1:
                raise AssertionError(f"E9 H1/H2 row is non-unique for {family}/{fmt}/{condition}/{metric_variant}")
            rule = rule_rows.iloc[0]
            for quantity, value_key, low_key, high_key in (
                ("Q_H1", "h1_q", "h1_q_ci_low", "h1_q_ci_high"),
                ("H1_ratio", "h1_ratio", "h1_ratio_ci_low", "h1_ratio_ci_high"),
                ("H2_band_ratio", "h2_band_ratio", "h2_band_ci_low", "h2_band_ci_high"),
            ):
                condition_rows.append({
                    "model": model, "format": fmt, "scope": scope, "condition": condition,
                    "metric_variant": metric_variant, "quantity": quantity,
                    "value": rule.get(value_key), "layer": np.nan,
                    "ci_low": rule.get(low_key), "ci_high": rule.get(high_key),
                    "a10_vi_n": np.nan, "a10_denominator": np.nan,
                    "outcome": outcomes[metric_variant],
                })
        la_en = la_summary.loc[
            la_summary["model"].astype(str).eq(model)
            & la_summary["format"].astype(str).eq(fmt)
            & la_summary["metric"].astype(str).eq("max_LA")
            & la_summary["target_lang"].astype(str).eq("en")
        ]
        if len(la_en) != 1:
            raise AssertionError(f"E9 LA_en row is non-unique for {family}/{fmt}/{condition}/{scope}")
        la_row = la_en.iloc[0]
        condition_rows.append({
            "model": model, "format": fmt, "scope": scope, "condition": condition,
            "metric_variant": "s", "quantity": "max_LA_en", "value": la_row.get("value"),
            "layer": la_row.get("layer"), "ci_low": la_row.get("ci_low"), "ci_high": la_row.get("ci_high"),
            "a10_vi_n": np.nan, "a10_denominator": np.nan,
            "outcome": outcomes["s"],
        })
        for landmark in ("L_peak", "L_concept_10", "L_cross"):
            landmark_rows = landmarks_all.loc[
                landmarks_all["model"].astype(str).eq(model)
                & landmarks_all["format"].astype(str).eq(fmt)
                & landmarks_all["metric_variant"].astype(str).isin({"s", "s1"})
                & landmarks_all["metric"].astype(str).eq(landmark)
            ]
            for landmark_row in landmark_rows.itertuples(index=False):
                condition_rows.append({
                    "model": model, "format": fmt, "scope": scope, "condition": condition,
                    "metric_variant": str(landmark_row.metric_variant), "quantity": landmark,
                    "value": landmark_row.value, "layer": landmark_row.value,
                    "ci_low": landmark_row.ci_low, "ci_high": landmark_row.ci_high,
                    "a10_vi_n": np.nan, "a10_denominator": np.nan,
                    "outcome": outcomes[str(landmark_row.metric_variant)],
                })
        if not ids.issubset(concepts):
            raise AssertionError(f"E9 concept metadata is incomplete for {family}/{fmt}/{scope}")
        a10 = _a10_row(
            [row for row in generations if str(row["concept_id"]) in ids], concepts,
            family=family, fmt=fmt, condition=condition, scope=scope,
        )
        a10["model"] = model
        condition_rows.append({**a10, "outcome": outcomes["s"]})
        stat_by_condition[condition] = {
            "rows": condition_rows,
            "rules": rules_all,
            "landmarks": landmarks_all,
            "rise": rises_by_metric,
            "la_long": la_long,
            "la_summary": la_summary,
            "model": model,
        }
        output_rows.extend(condition_rows)

    sorted_ids = sorted(ids)
    bootstrap_ids = sorted_ids
    cond_stats = {name: stat_by_condition[name] for name in ("diac", "nodiac")}
    analysis_model = cond_stats["nodiac"]["model"]
    indices = concept_indices(
        bootstrap_ids, resamples=resamples, seed=int(cfg["seed"]),
        key=_paired_boot_key(analysis_model, fmt, scope),
    )
    for metric_variant in ("s", "s1"):
        if metric_variant == "s1" and fmt not in BLOCK_FORMATS:
            # Cloze remains descriptive; the metric is still reported above, while its paired bootstrap is retained.
            pass
        cond_boot: dict[str, dict[str, np.ndarray]] = {}
        cond_point: dict[str, dict[str, float]] = {}
        for condition in ("diac", "nodiac"):
            status = cond_stats[condition]
            rise = status["rise"][metric_variant]
            # The long E4 frame uses normalized Vietnamese target variants in both conditions.
            cond_boot[condition] = _rule_bootstrap_values(rise, indices, analysis_model, fmt)
            cond_point[condition] = _rule_values(status["rules"], analysis_model, fmt, metric_variant)
        for quantity in ("Q_H1", "H1_ratio", "H2_band_ratio"):
            difference = cond_boot["nodiac"][quantity] - cond_boot["diac"][quantity]
            low, high = percentile_interval(difference)
            output_rows.append({
                "model": analysis_model, "format": fmt, "scope": scope,
                "condition": "nodiac_minus_diac", "metric_variant": metric_variant,
                "quantity": quantity,
                "value": cond_point["nodiac"][quantity] - cond_point["diac"][quantity],
                "layer": np.nan, "ci_low": low, "ci_high": high,
                "a10_vi_n": np.nan, "a10_denominator": np.nan,
                "outcome": "descriptive_only",
            })
    la_boot: dict[str, np.ndarray] = {}
    for condition in ("diac", "nodiac"):
        la_boot[condition] = _max_la_samples(
            cond_stats[condition]["la_long"], sorted_ids, indices, analysis_model, fmt
        )
    la_point = {
        condition: float(
            cond_stats[condition]["la_summary"].loc[
                cond_stats[condition]["la_summary"]["model"].astype(str).eq(cond_stats[condition]["model"])
                & cond_stats[condition]["la_summary"]["format"].astype(str).eq(fmt)
                & cond_stats[condition]["la_summary"]["metric"].astype(str).eq("max_LA")
                & cond_stats[condition]["la_summary"]["target_lang"].astype(str).eq("en"),
                "value",
            ].iloc[0]
        )
        for condition in ("diac", "nodiac")
    }
    delta_la = la_boot["nodiac"] - la_boot["diac"]
    low, high = percentile_interval(delta_la)
    output_rows.append({
        "model": analysis_model, "format": fmt, "scope": scope,
        "condition": "nodiac_minus_diac", "metric_variant": "s",
        "quantity": "max_LA_en", "value": la_point["nodiac"] - la_point["diac"],
        "layer": np.nan, "ci_low": low, "ci_high": high,
        "a10_vi_n": np.nan, "a10_denominator": np.nan,
        "outcome": "descriptive_only",
    })
    return output_rows


def _load_concepts(repo: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load scoring fields only for selected paired concepts."""
    path = repo / "data/external/viLens-concepts/concepts.jsonl"
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            concept_id = str(row["concept_id"])
            if concept_id in ids:
                selected[concept_id] = row
    if set(selected) != ids:
        raise AssertionError("E9 concept metadata does not cover paired prompts")
    return selected


def _load_scope_inputs(
    cfg: Mapping[str, Any], repo: Path, family: str, fmt: str,
    paired_ids: set[str], paired_cf_ids: set[str], main_ids: set[str],
) -> dict[str, Any]:
    """Read one E9 cell and compute each condition's LA with primary matched directions."""
    from scripts import analyze_core as core

    diac_paths = _primary_paths(cfg, repo, family, fmt)
    nodiac_paths = _e9_paths(cfg, repo, family, fmt)
    diac_raw = _read_long(diac_paths, paired_ids, fmt)
    nodiac_raw = _read_long(nodiac_paths, paired_ids, fmt)
    diac_generations = _read_generations(
        diac_paths, family=family, fmt=fmt, condition="diac", ids=paired_ids
    )
    nodiac_generations = _read_generations(
        nodiac_paths, family=family, fmt=fmt, condition="nodiac", ids=paired_ids
    )
    nodiac_cfg = copy.deepcopy(dict(cfg))
    nodiac_cfg["pl"]["results_dir"] = str(cfg["pl"]["e9"]["results_dir"])
    nodiac_cfg["pl"]["primary"] = {
        **dict(cfg["pl"]["primary"]),
        "condition": "nodiac",
        "fewshot_set": 1,
        "files": list(cfg["pl"]["e9"]["files"]),
    }
    output: dict[str, Any] = {}
    for scope, scope_ids in (("paired_main", paired_ids), ("paired_cf729", paired_cf_ids)):
        if not scope_ids:
            continue
        key_template = (
            "bootstrap|e9|{model}|{format}"
            if scope == "paired_main" else "bootstrap|e9|{model}|{format}|cf729"
        )
        expected_counts = {fmt: len(scope_ids)}
        _diac_la_long, diac_la_summary = core._compute_la_source(
            "primary", config=cfg, repo=repo, main_ids=main_ids,
            family_filter=family, formats_override=(fmt,), query_scope_ids=scope_ids,
            expected_query_counts=expected_counts, bootstrap_key_template=key_template,
        )
        _nodiac_la_long, nodiac_la_summary = core._compute_la_source(
            "primary", config=nodiac_cfg, repo=repo, main_ids=main_ids,
            family_filter=family, formats_override=(fmt,), query_scope_ids=scope_ids,
            expected_query_counts=expected_counts, bootstrap_key_template=key_template,
        )
        output[scope] = {
            "diac_raw": diac_raw,
            "nodiac_raw": nodiac_raw,
            "diac_la_long": _diac_la_long,
            "diac_la_summary": diac_la_summary,
            "nodiac_la_long": _nodiac_la_long,
            "nodiac_la_summary": nodiac_la_summary,
            "diac_generations": diac_generations,
            "nodiac_generations": nodiac_generations,
        }
    return output


def _write_outputs(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], repo: Path) -> None:
    """Write the descriptive E9 CSV and Markdown report, refusing replacement."""
    csv_path = repo / str(cfg["e9_analysis"]["csv"])
    md_path = repo / str(cfg["e9_analysis"]["report"])
    existing = [path for path in (csv_path, md_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to replace E9 outputs: " + ", ".join(str(p) for p in existing))
    frame = pd.DataFrame.from_records(rows)
    frame = frame.sort_values(
        ["model", "format", "scope", "condition", "quantity", "metric_variant"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    display = frame.loc[:, [
        "model", "format", "scope", "condition", "metric_variant", "quantity",
        "value", "layer", "ci_low", "ci_high", "outcome",
    ]].copy()
    lines = [
        "# E9 paired nodiac–diac analysis (descriptive)", "",
        "Read alongside A12. Values and paired concept-bootstrap intervals are reported by condition and as nodiac minus diac.",
        "Cloze rows are descriptive only.", "",
        "| Model | Format | Scope | Condition | Variant | Quantity | Value | Layer | CI low | CI high | Outcome |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in display.itertuples(index=False):
        value = "NA" if pd.isna(row.value) else f"{float(row.value):.8g}"
        layer = "NA" if pd.isna(row.layer) else f"{float(row.layer):g}"
        low = "NA" if pd.isna(row.ci_low) else f"{float(row.ci_low):.8g}"
        high = "NA" if pd.isna(row.ci_high) else f"{float(row.ci_high):.8g}"
        lines.append(
            f"| {row.model} | {row.format} | {row.scope} | {row.condition} | {row.metric_variant} | "
            f"{row.quantity} | {value} | {layer} | {low} | {high} | {row.outcome} |"
        )
    lines.extend(["", "Source values are in `reports/E9.csv`.", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    cfg: Mapping[str, Any], *, repo: Path = REPO,
    synthetic_inputs: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run the paired analysis and write its report/CSV outputs."""
    from scripts import analyze_core as core

    rows: list[dict[str, Any]] = []
    if synthetic_inputs is None:
        _table, okay, scopes = preflight(cfg, repo=repo)
        if not okay:
            raise RuntimeError("E9 analysis preflight failed; refusing analysis")
        main_ids, _cf_ids = core._primary_scope_ids(repo, cfg)
        concepts = _load_concepts(repo, main_ids)
        for family in FAMILIES:
            for fmt in FORMATS:
                ids = scopes["paired"][fmt]
                cf_ids = scopes["paired_cf"][fmt]
                inputs = _load_scope_inputs(cfg, repo, family, fmt, ids, cf_ids, main_ids)
                for scope, scope_ids in (("paired_main", ids), ("paired_cf729", cf_ids)):
                    cell = inputs.get(scope)
                    if not cell:
                        continue
                    rows.extend(_summarize_scope(
                        cfg, family=family, fmt=fmt, scope=scope, ids=scope_ids,
                        diac_raw=cell["diac_raw"], nodiac_raw=cell["nodiac_raw"],
                        diac_la_long=cell["diac_la_long"], diac_la_summary=cell["diac_la_summary"],
                        nodiac_la_long=cell["nodiac_la_long"], nodiac_la_summary=cell["nodiac_la_summary"],
                        diac_generations=cell["diac_generations"], nodiac_generations=cell["nodiac_generations"],
                        concepts=concepts,
                    ))
    else:
        for (family, fmt), cell in synthetic_inputs.items():
            for scope, scope_ids in (("paired_main", set(cell["paired_ids"])), ("paired_cf729", set(cell["cf_ids"]))):
                if not scope_ids:
                    continue
                rows.extend(_summarize_scope(
                    cfg, family=family, fmt=fmt, scope=scope, ids=scope_ids,
                    diac_raw=cell["diac_raw"], nodiac_raw=cell["nodiac_raw"],
                    diac_la_long=cell["diac_la_long"], diac_la_summary=cell["diac_la_summary"],
                    nodiac_la_long=cell["nodiac_la_long"], nodiac_la_summary=cell["nodiac_la_summary"],
                    diac_generations=cell["diac_generations"], nodiac_generations=cell["nodiac_generations"],
                    concepts=cell["concepts"],
                ))
    _write_outputs(rows, cfg, repo)
    return pd.DataFrame.from_records(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the E9 paired P-L input preflight or analysis."""
    env_guard.guard(offline=True)
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(REPO / args.config)
    table, okay, _scopes = preflight(cfg, repo=REPO)
    print(table)
    if args.preflight_only:
        return 0 if okay else 2
    if not okay:
        raise SystemExit("E9 analysis preflight failed; refusing analysis")
    analyze(cfg, repo=REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
