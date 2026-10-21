#!/datastore/luannt/.cache/sb/bin/python
"""Analyze the descriptive R1 few-shot robustness sets from completed P-L units."""

from __future__ import annotations

import argparse
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

from vilens.config import load_config
from vilens.io import unit_paths

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
LONG_COLUMNS = [
    "model", "model_revision", "pass", "format", "condition", "fewshot_set",
    "concept_id", "target_lang", "target_variant", "role", "ref_concept_id",
    "tok_idx", "token_id", "is_content", "layer", "metric", "value",
]


def _unit_key(family: str, fmt: str, set_id: int, shard: int) -> dict[str, str | int]:
    """Build the P-L unit identity for one R1 set/format/shard."""
    return {
        "family": family,
        "pass": "pl",
        "condition": "diac",
        "fewshot_set": str(set_id),
        "format": fmt,
        "shard": shard,
    }


def expected_unit_keys(cfg: Mapping[str, Any]) -> list[dict[str, str | int]]:
    """Enumerate all configured R1 units in set order, then format and shard order."""
    shard_size = int(cfg["pl"]["shard_prompts"])
    r1 = cfg["pl"]["r1"]
    output: list[dict[str, str | int]] = []
    for family in FAMILIES:
        for set_id in map(int, r1["fewshot_sets"]):
            for fmt in FORMATS:
                count = int(r1["expected"][fmt])
                for shard in range((count + shard_size - 1) // shard_size):
                    output.append(_unit_key(family, fmt, set_id, shard))
    return output


def r1_a10_summary(
    generations: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Summarize A10 Vietnamese competence by model, format, and few-shot set."""
    from vilens.analysis.competence import competence_rows

    scored = competence_rows(generations, concepts)
    if scored.empty:
        raise ValueError("R1 A10 requires generation rows")
    rows: list[dict[str, Any]] = []
    for (model, fmt, set_id), group in scored.groupby(
        ["model", "format", "fewshot_set"], sort=True, dropna=False
    ):
        counts = group["competence_class"].astype(str).value_counts().to_dict()
        denominator = int(len(group))
        vi_n = int(counts.get("VI", 0))
        rows.append({
            "model": str(model),
            "format": str(fmt),
            "fewshot_set": int(set_id),
            "metric_variant": "s",
            "quantity": "A10_vi_rate",
            "value": vi_n / denominator,
            "layer": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "a10_vi_n": vi_n,
            "a10_denominator": denominator,
            "a10_en_n": int(counts.get("EN", 0)),
            "a10_other_lang_n": int(counts.get("OTHER-LANG", 0)),
            "a10_damage_n": int(counts.get("DAMAGE", 0)),
            "a10_shared_ru_source_n": int(group["shared_ru_source"].astype(bool).sum()),
        })
    return pd.DataFrame.from_records(rows)


def _r1_generation_rows(
    cfg: Mapping[str, Any], *, repo: Path, family: str, set_id: int, fmt: str
) -> list[dict[str, Any]]:
    """Read one R1 cell's generation answers without emitting answer text."""
    expected = int(cfg["pl"]["r1"]["expected"][fmt])
    shard_size = int(cfg["pl"]["shard_prompts"])
    root = repo / str(cfg["pl"]["r1"]["results_dir"])
    generations: list[dict[str, Any]] = []
    for shard in range((expected + shard_size - 1) // shard_size):
        parquet_path, _sidecar = unit_paths(root, _unit_key(family, fmt, set_id, shard))
        generation_path = parquet_path.with_suffix(".generations.jsonl")
        with generation_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if bool(raw.get("m1_extension", False)):
                    raise AssertionError("R1 A10 generations unexpectedly include an M1 extension row")
                generations.append({
                    "model": family,
                    "format": fmt,
                    "condition": str(cfg["pl"]["r1"]["condition"]),
                    "fewshot_set": set_id,
                    "concept_id": str(raw["concept_id"]),
                    "answer": str(raw.get("text", "")),
                })
    ids = [str(row["concept_id"]) for row in generations]
    if len(generations) != expected or len(set(ids)) != expected:
        raise AssertionError(
            f"R1 {family}/set{set_id}/{fmt} generation rows={len(generations)}, unique={len(set(ids))}, expected={expected}"
        )
    return generations


def preflight(cfg: Mapping[str, Any], *, repo: Path = REPO) -> tuple[str, bool]:
    """Check R1 units, provenance, and generation-row alignment without displaying text."""
    root = repo / str(cfg["pl"]["r1"]["results_dir"])
    rows: list[tuple[str, str, str, str, str]] = []
    present_units = 0
    generation_enabled = cfg["pl"]["r1"].get("generation") is True
    for family in FAMILIES:
        for set_id in map(int, cfg["pl"]["r1"]["fewshot_sets"]):
            for fmt in FORMATS:
                expected_prompts = int(cfg["pl"]["r1"]["expected"][fmt])
                expected = int(
                    (expected_prompts + int(cfg["pl"]["shard_prompts"]) - 1)
                    // int(cfg["pl"]["shard_prompts"])
                )
                present = 0
                invalid = 0
                generation_count = 0
                for shard in range(expected):
                    key = _unit_key(family, fmt, set_id, shard)
                    parquet_path, sidecar_path = unit_paths(root, key)
                    required = (
                        parquet_path,
                        sidecar_path,
                        parquet_path.with_suffix(".resid.npy"),
                        parquet_path.with_suffix(".attn.npy"),
                        parquet_path.with_suffix(".mlp.npy"),
                        parquet_path.with_suffix(".resid_index.jsonl"),
                        parquet_path.with_suffix(".answer_topk.jsonl"),
                        parquet_path.with_suffix(".generations.jsonl"),
                    )
                    if not all(path.is_file() for path in required):
                        continue
                    try:
                        metadata_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
                        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                        index_rows = [
                            json.loads(line)
                            for line in parquet_path.with_suffix(".resid_index.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                        generated_rows = [
                            json.loads(line)
                            for line in parquet_path.with_suffix(".generations.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                        expected_shard_rows = min(
                            int(cfg["pl"]["shard_prompts"]),
                            expected_prompts - shard * int(cfg["pl"]["shard_prompts"]),
                        )
                        generation_match = (
                            len(generated_rows) == expected_shard_rows
                            and len(index_rows) == expected_shard_rows
                            and [str(row.get("concept_id", "")) for row in generated_rows]
                            == [str(row.get("concept_id", "")) for row in index_rows]
                            and all(not bool(row.get("m1_extension", False)) for row in generated_rows)
                        )
                        valid = (
                            metadata_rows > 0
                            and metadata_rows == sidecar.get("row_count")
                            and metadata_rows == sidecar.get("expected_row_count")
                            and sidecar.get("unit_key") == key
                            and generation_enabled
                            and generation_match
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        valid = False
                    if valid:
                        present += 1
                        present_units += 1
                        generation_count += len(generated_rows)
                    else:
                        invalid += 1
                status = (
                    "PASS"
                    if present == expected and invalid == 0 and generation_count == expected_prompts
                    else "MISSING"
                )
                rows.append((
                    family, str(set_id), fmt,
                    f"{present}/{expected}; invalid={invalid}; {status}",
                    f"{generation_count}/{expected_prompts}; greedy={'on' if generation_enabled else 'off'}",
                ))
    summary_path = root / "summary.json"
    summary_ok = False
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_ok = int(summary.get("exit_code", -1)) == 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            summary_ok = False
    headers = ("FAMILY", "SET", "FORMAT", "UNITS", "GREEDY_GENERATION_ROWS")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(5)]
    table = " | ".join(headers[i].ljust(widths[i]) for i in range(5))
    table += "\n" + "-+-".join("-" * width for width in widths)
    table += "\n" + "\n".join(" | ".join(row[i].ljust(widths[i]) for i in range(5)) for row in rows)
    table += f"\nR1 units present={present_units}/90; greedy generation expected={generation_enabled}; summary={'PASS' if summary_ok else 'MISSING/FAIL'}"
    return table, present_units == 90 and summary_ok and generation_enabled


def _metrics_for_cell(
    raw: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    family: str,
    set_id: int,
    baseline_rules: pd.DataFrame,
    baseline_landmarks: pd.DataFrame,
    baseline_la: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute frozen E3/E4 and E5 summaries for one few-shot set."""
    from scripts import analyze_core as core

    selected = raw.loc[pd.to_numeric(raw["fewshot_set"], errors="coerce").eq(set_id)].copy()
    counts = selected.groupby("format", sort=True)["concept_id"].nunique().to_dict()
    expected = {"repetition": 729, "translation": 721}
    if {fmt: int(counts.get(fmt, 0)) for fmt in FORMATS} != expected:
        raise AssertionError(f"R1 {family}/set{set_id} prompt counts differ from 729/721")
    scores = core.content_token_scores(selected)
    deltas = core.matched_distractor_deltas(scores)
    readouts = core._readout_rows(deltas)
    score_readouts = core._readout_rows(scores.loc[scores["role"].astype(str).eq("correct")])
    early = float(cfg["e1"]["early_depth_max"])
    rises_s = core.concept_rise_curves(readouts, early_depth_max=early, value_column="delta_s")
    rises_s1 = core.concept_rise_curves(readouts, early_depth_max=early, value_column="delta_s1")
    analysis_cfg_s = dict(cfg["analysis"])

    all_rules: list[pd.DataFrame] = []
    all_landmarks: list[pd.DataFrame] = []
    for variant, rises, score_column in (("s", rises_s, "s"), ("s1", rises_s1, "s1")):
        template = f"bootstrap|r1|{{model}}|{{format}}|set{set_id}"
        analysis_cfg = {**analysis_cfg_s, "bootstrap_key_template": template}
        rules, landmarks, _window_rows = core._bootstrap_core(
            per_concept_rise=rises,
            per_concept_scores=score_readouts,
            source="r1",
            item_set="cf729",
            seed=int(cfg["seed"]),
            analysis_config=analysis_cfg,
            score_column=score_column,
            metric_variant=variant,
        )
        all_rules.append(rules)
        all_landmarks.append(landmarks)

    rules_wide = pd.concat(all_rules, ignore_index=True)
    landmarks_wide = pd.concat(all_landmarks, ignore_index=True)
    la_long, la_summary = core._compute_la_source(
        "r1",
        config=cfg,
        repo=REPO,
        main_ids=None,
        family_filter=family,
        fewshot_set=set_id,
        bootstrap_key_template=f"bootstrap|r1|{{model}}|{{format}}|set{set_id}",
    )
    if la_summary.empty:
        raise ValueError(f"R1 E5 alignment summary is empty for {family}/set{set_id}")
    rules_long = core._summary_metric_long(rules_wide, source_rows=score_readouts, kind="rule")
    rules_long["fewshot_set"] = set_id
    landmarks_long = core._summary_metric_long(landmarks_wide, source_rows=score_readouts, kind="landmark")
    landmarks_long["fewshot_set"] = set_id
    checkpoint, _verdict = core._build_checkpoint_rules(rules_long, la_summary)
    if not checkpoint.empty:
        checkpoint["fewshot_set"] = set_id
    metric_rows: list[dict[str, Any]] = []
    for row in la_summary.loc[
        la_summary["metric"].astype(str).eq("max_LA")
        & la_summary["target_lang"].astype(str).eq("en")
    ].to_dict(orient="records"):
        metric_rows.append({
            "model": row["model"], "format": row["format"], "fewshot_set": set_id,
            "metric_variant": "s", "quantity": "max_LA_en", "value": row["value"],
            "layer": row["layer"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        })
    for row in rules_wide.loc[rules_wide["target_lang"].astype(str).eq("__H1_H2__")].to_dict(orient="records"):
        for quantity, value_key, low_key, high_key in (
            ("Q_H1", "h1_q", "h1_q_ci_low", "h1_q_ci_high"),
            ("H1_ratio", "h1_ratio", "h1_ratio_ci_low", "h1_ratio_ci_high"),
            ("H2_band_ratio", "h2_band_ratio", "h2_band_ci_low", "h2_band_ci_high"),
        ):
            metric_rows.append({
                "model": row["model"], "format": row["format"], "fewshot_set": set_id,
                "metric_variant": row["metric_variant"], "quantity": quantity,
                "value": row.get(value_key), "layer": None,
                "ci_low": row.get(low_key), "ci_high": row.get(high_key),
            })
    for row in landmarks_wide.to_dict(orient="records"):
        if str(row["metric"]) in {"L_peak", "L_concept_10", "L_cross"}:
            metric_rows.append({
                "model": row["model"], "format": row["format"], "fewshot_set": set_id,
                "metric_variant": row["metric_variant"], "quantity": row["metric"],
                "value": row["value"], "layer": row["value"],
                "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            })
    metric_frame = pd.DataFrame.from_records(metric_rows)
    outcome_frame = checkpoint.loc[:, [
        "model", "format", "fewshot_set", "Outcome", "s1_outcome", "verdict", "s1_verdict"
    ]].copy()
    return metric_frame, outcome_frame, pd.concat([la_long, la_summary], ignore_index=True, sort=False)


def analyze(cfg: Mapping[str, Any], *, repo: Path = REPO) -> None:
    """Write R1 metric and set-outcome tables using frozen E3/E4/E5 functions."""
    from scripts import analyze_core as core

    table, okay = preflight(cfg, repo=repo)
    if not okay:
        raise RuntimeError("R1 P-L units are incomplete; refusing R1 analysis")
    outputs = (repo / "reports/R1.md", repo / "reports/R1_metrics.csv", repo / "reports/R1_set_outcomes.csv")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to replace existing R1 outputs: " + ", ".join(map(str, existing)))
    baseline_rules = pd.read_csv(repo / "reports/E2-E5_rules.csv")
    baseline_landmarks = pd.read_csv(repo / "reports/E2-E5_landmarks.csv")
    metric_parts: list[pd.DataFrame] = []
    outcome_parts: list[pd.DataFrame] = []
    set1_checkpoints: list[pd.DataFrame] = []
    set1_la_parts: list[pd.DataFrame] = []
    set1_a10_parts: list[pd.DataFrame] = []
    main_ids, cognate_ids = core._primary_scope_ids(repo, cfg)
    r1_concepts = core._load_primary_concepts(repo, main_ids)
    for family in FAMILIES:
        _set1_la_long, baseline_la = core._compute_la_source(
            "primary",
            config=cfg,
            repo=repo,
            main_ids=main_ids,
            family_filter=family,
            formats_override=FORMATS,
            query_scope_ids=cognate_ids & main_ids,
            expected_query_counts={"repetition": 729, "translation": 721},
            bootstrap_key_template="bootstrap|r1|{model}|{format}|set1",
        )
        set1_la_parts.append(baseline_la)
        set1_rule_rows = baseline_rules.loc[
            baseline_rules["item_set"].astype(str).eq("cf729")
            & baseline_rules["metric_variant"].astype(str).isin({"s", "s1"})
        ].copy()
        set1_checkpoint, _set1_verdict = core._build_checkpoint_rules(set1_rule_rows, baseline_la)
        set1_checkpoints.append(set1_checkpoint)
        primary_generations = core._load_primary_generations(repo, cfg, main_ids, family=family)
        set1_generations = [
            {**row, "fewshot_set": 1}
            for row in primary_generations
            if str(row["concept_id"]) in cognate_ids
        ]
        set1_a10 = r1_a10_summary(set1_generations, r1_concepts)
        for fmt, expected_n in (("repetition", 729), ("translation", 721)):
            selected = set1_a10.loc[set1_a10["format"].astype(str).eq(fmt)]
            if len(selected) != 1 or int(selected.iloc[0]["a10_denominator"]) != expected_n:
                raise AssertionError(f"R1 set-1 A10 {family}/{fmt} denominator differs from {expected_n}")
        set1_a10_parts.append(set1_a10)
        for set_id in map(int, cfg["pl"]["r1"]["fewshot_sets"]):
            root = repo / str(cfg["pl"]["r1"]["results_dir"])
            paths = sorted(root.glob(
                f"condition=diac__family={family}__fewshot_set={set_id}__format=*__pass=pl__shard=*.parquet"
            ))
            batches = [pq.ParquetFile(path).read(columns=LONG_COLUMNS).to_pandas() for path in paths]
            raw = pd.concat(batches, ignore_index=True)
            metric, outcomes, _la = _metrics_for_cell(
                raw, cfg, family=family, set_id=set_id,
                baseline_rules=baseline_rules, baseline_landmarks=baseline_landmarks,
                baseline_la=baseline_la,
            )
            a10_parts = []
            for fmt in FORMATS:
                generation_rows = _r1_generation_rows(
                    cfg, repo=repo, family=family, set_id=set_id, fmt=fmt
                )
                a10_cell = r1_a10_summary(generation_rows, r1_concepts)
                expected_n = int(cfg["pl"]["r1"]["expected"][fmt])
                if len(a10_cell) != 1 or int(a10_cell.iloc[0]["a10_denominator"]) != expected_n:
                    raise AssertionError(f"R1 A10 {family}/set{set_id}/{fmt} denominator differs from {expected_n}")
                a10_parts.append(a10_cell)
            a10_metric = pd.concat(a10_parts, ignore_index=True)
            metric_parts.append(pd.concat([metric, a10_metric], ignore_index=True, sort=False))
            outcome_parts.append(outcomes)
            del batches, raw

    metrics = pd.concat(metric_parts, ignore_index=True)
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    set1_checkpoint = pd.concat(set1_checkpoints, ignore_index=True)
    set1_la_all = pd.concat(set1_la_parts, ignore_index=True)
    set1_a10_all = pd.concat(set1_a10_parts, ignore_index=True)
    baseline_by_key = {
        (str(row.model), str(row.format)): row
        for row in set1_checkpoint.itertuples(index=False)
    }
    metrics["set1_value"] = np.nan
    metrics["set1_layer"] = np.nan
    metrics["set1_ci_low"] = np.nan
    metrics["set1_ci_high"] = np.nan
    metrics["set1_a10_vi_n"] = np.nan
    metrics["set1_a10_denominator"] = np.nan
    for index, row in metrics.iterrows():
        model, fmt, quantity, variant = str(row.model), str(row.format), str(row.quantity), str(row.metric_variant)
        if quantity == "A10_vi_rate":
            selected = set1_a10_all.loc[
                set1_a10_all["model"].astype(str).eq(model)
                & set1_a10_all["format"].astype(str).eq(fmt)
                & set1_a10_all["fewshot_set"].astype(int).eq(1)
            ]
            if len(selected) != 1:
                raise AssertionError(f"R1 set-1 A10 lookup is non-unique for {model}/{fmt}: {len(selected)} rows")
            set1 = selected.iloc[0]
            set1_values = (set1["value"], np.nan, np.nan, np.nan)
            metrics.loc[index, "set1_a10_vi_n"] = int(set1["a10_vi_n"])
            metrics.loc[index, "set1_a10_denominator"] = int(set1["a10_denominator"])
        elif quantity == "max_LA_en":
            selected = set1_la_all.loc[
                set1_la_all["model"].astype(str).eq(model)
                & set1_la_all["format"].astype(str).eq(fmt)
                & set1_la_all["metric"].astype(str).eq("max_LA")
                & set1_la_all["target_lang"].astype(str).eq("en")
            ]
            set1 = selected.iloc[0] if len(selected) == 1 else None
            set1_values = (set1.get("value"), set1.get("layer"), set1.get("ci_low"), set1.get("ci_high")) if set1 is not None else (np.nan,) * 4
        elif quantity in {"Q_H1", "H1_ratio", "H2_band_ratio"}:
            source_metric = {"Q_H1": "h1_q", "H1_ratio": "h1_ratio", "H2_band_ratio": "h2_band_ratio"}[quantity]
            selected = baseline_rules.loc[
                baseline_rules["model"].astype(str).eq(model)
                & baseline_rules["format"].astype(str).eq(fmt)
                & baseline_rules["metric"].astype(str).eq(source_metric)
                & baseline_rules["metric_variant"].astype(str).eq(variant)
                & baseline_rules["target_lang"].astype(str).eq("__H1_H2__")
                & baseline_rules["item_set"].astype(str).eq("cf729")
            ]
            set1 = selected.iloc[0] if len(selected) == 1 else None
            set1_values = (set1.get("value"), np.nan, set1.get("ci_low"), set1.get("ci_high")) if set1 is not None else (np.nan,) * 4
        else:
            selected = baseline_landmarks.loc[
                baseline_landmarks["model"].astype(str).eq(model)
                & baseline_landmarks["format"].astype(str).eq(fmt)
                & baseline_landmarks["metric"].astype(str).eq(quantity)
                & baseline_landmarks["metric_variant"].astype(str).eq(variant)
                & baseline_landmarks["item_set"].astype(str).eq("cf729")
            ]
            set1 = selected.iloc[0] if len(selected) == 1 else None
            set1_values = (set1.get("value"), set1.get("value"), set1.get("ci_low"), set1.get("ci_high")) if set1 is not None else (np.nan,) * 4
        metrics.loc[index, ["set1_value", "set1_layer", "set1_ci_low", "set1_ci_high"]] = set1_values

    outcomes["set1_outcome"] = ""
    outcomes["set1_s1_outcome"] = ""
    for index, row in outcomes.iterrows():
        baseline = baseline_by_key.get((str(row.model), str(row.format)))
        if baseline is not None:
            outcomes.loc[index, "set1_outcome"] = str(baseline.Outcome)
            outcomes.loc[index, "set1_s1_outcome"] = str(baseline.s1_outcome)
    outcomes["matches_set1_s"] = outcomes["Outcome"].astype(str).eq(outcomes["set1_outcome"].astype(str))
    outcomes["matches_set1_s1"] = outcomes["s1_outcome"].astype(str).eq(outcomes["set1_s1_outcome"].astype(str))
    counts = outcomes.loc[outcomes["fewshot_set"].astype(int).between(2, 6)].groupby(
        ["model", "format"], sort=True
    ).agg(
        sets_equal_set1_s=("matches_set1_s", "sum"),
        sets_equal_set1_s1=("matches_set1_s1", "sum"),
    ).reset_index()
    outcomes = outcomes.merge(counts, on=["model", "format"], how="left", validate="many_to_one")
    csv_outputs = outputs[1:]
    for path, frame in zip(csv_outputs, (metrics, outcomes), strict=True):
        temp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(path)
    report = [
        "# R1 few-shot robustness (descriptive)", "",
        "R1 is descriptive robustness; it adds no Holm family or decision rule. The Checkpoint H verdicts remain primary.",
        "", "Capture inputs are the cognate-free repetition and translation items, sets 2–6, with separate resume-safe units.",
        "", "The tables report each set beside set 1, including the primary s metric and the s¹ sensitivity.",
        "", "| Family | Set | Format | Units |", "|---|---:|---|---:|",
    ]
    report.extend(
        f"| {family} | {set_id} | {fmt} | {int(cfg['pl']['r1']['expected'][fmt])} prompts |"
        for family in FAMILIES
        for set_id in map(int, cfg["pl"]["r1"]["fewshot_sets"])
        for fmt in FORMATS
    )
    a10_rows = metrics.loc[metrics["quantity"].astype(str).eq("A10_vi_rate")].sort_values(
        ["model", "format", "fewshot_set"], kind="stable"
    )
    report.extend([
        "", "## A10 competence rate (descriptive)", "",
        "The rate is the A10 VI-class count divided by the cognate-free prompt count; set 1 uses the same cf729 scope.",
        "", "| Model | Format | Set | VI / N | Rate | Set 1 VI / N | Set 1 rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    report.extend(
        f"| {row.model} | {row.format} | {int(row.fewshot_set)} | "
        f"{int(row.a10_vi_n)}/{int(row.a10_denominator)} | {float(row.value):.4f} | "
        f"{int(row.set1_a10_vi_n)}/{int(row.set1_a10_denominator)} | {float(row.set1_value):.4f} |"
        for row in a10_rows.itertuples(index=False)
    )
    report.extend(["", "Outputs: reports/R1_metrics.csv and reports/R1_set_outcomes.csv.", ""])
    report_path = outputs[0]
    temp_report = report_path.with_suffix(".md.tmp")
    temp_report.write_text("\n".join(report), encoding="utf-8")
    temp_report.replace(report_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the R1 metadata preflight or analysis."""
    env_guard.guard(offline=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(REPO / args.config)
    table, okay = preflight(cfg)
    print(table)
    if args.preflight_only:
        return 0 if okay else 2
    if not okay:
        raise SystemExit("R1 units are incomplete; refusing analysis")
    analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
