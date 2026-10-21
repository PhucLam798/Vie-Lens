#!/datastore/luannt/.cache/sb/bin/python
"""Compute the OPS-44 descriptive E4 robustness variants from stored P-L rows."""

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

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
NLLB_FLAGS = ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree")


def _truth(value: Any) -> bool:
    """Interpret the concepts JSONL's explicit string booleans."""
    return value is True or str(value).strip().lower() == "true"


def _load_metadata(repo: Path, main_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Read metadata for main items without printing any item fields."""
    path = repo / "data/external/viLens-concepts/concepts.jsonl"
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            concept_id = str(row["concept_id"])
            if concept_id in main_ids:
                selected[concept_id] = row
    if set(selected) != main_ids:
        raise AssertionError("Concept metadata does not cover main test ids")
    return selected


def variant_ids(
    variant: Mapping[str, Any],
    *,
    main_ids: set[str],
    cf_ids: set[str],
    metadata: Mapping[str, Mapping[str, Any]],
    competent_ids: set[str] | None = None,
    r11_ids: set[str] | None = None,
) -> set[str]:
    """Resolve one registered OPS-44 item filter to its concept ids."""
    scope = main_ids if variant["scope"] == "main" else cf_ids & main_ids
    filter_name = str(variant["filter"])
    if filter_name == "none":
        return set(scope)
    if filter_name == "bt_pass_strict_v1":
        return {item for item in scope if _truth(metadata[item].get("bt_pass_strict_v1"))}
    if filter_name == "external_attested":
        return {item for item in scope if _truth(metadata[item].get("external_attested"))}
    if filter_name == "all_three_nllb":
        return {
            item for item in scope
            if all(_truth(metadata[item].get(flag)) for flag in NLLB_FLAGS)
        }
    if filter_name == "min_surface_dist_ge_060":
        return {
            item for item in scope
            if metadata[item].get("min_surface_dist") not in (None, "")
            and float(metadata[item]["min_surface_dist"]) >= 0.60
        }
    if filter_name == "competent_vi":
        if competent_ids is None:
            raise ValueError("R7 requires model/format A10 competent ids")
        return scope & competent_ids
    if filter_name == "exclude_r11_ids":
        if r11_ids is None:
            raise ValueError("R11 requires the configured excluded ids")
        return scope - r11_ids
    raise ValueError(f"Unknown R-battery filter: {filter_name}")


def _variant_bootstrap_config(
    cfg: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    """Use the OPS-44 per-variant key for the frozen bootstrap helpers."""
    code = str(variant["code"]).lower()
    suffix = str(variant["name"]).removeprefix(f"{variant['code']}_")
    return {
        **dict(cfg["analysis"]),
        "bootstrap_key_template": f"bootstrap|{code}|{{model}}|{{format}}|{suffix}",
    }


def _summary_row(
    rules: pd.DataFrame, landmarks: pd.DataFrame, la: pd.DataFrame, checkpoint: pd.DataFrame,
    *, model: str, fmt: str, variant: str, item_count: int,
) -> dict[str, Any]:
    """Flatten frozen E3/E4/E5 summaries for one model-format scope."""
    row: dict[str, Any] = {"variant": variant, "model": model, "format": fmt, "item_count": item_count}
    rule_rows = rules.loc[rules["model"].astype(str).eq(model)
                          & rules["format"].astype(str).eq(fmt)
                          & rules["target_lang"].astype(str).eq("__H1_H2__")]
    for metric_variant in ("s", "s1"):
        selected = rule_rows.loc[rule_rows["metric_variant"].astype(str).eq(metric_variant)]
        if len(selected) != 1:
            raise AssertionError(f"Expected one {metric_variant} E4 summary for {variant}/{model}/{fmt}")
        source = selected.iloc[0]
        for output_name, value_name, low_name, high_name in (
            ("Q_H1", "h1_q", "h1_q_ci_low", "h1_q_ci_high"),
            ("H1_ratio", "h1_ratio", "h1_ratio_ci_low", "h1_ratio_ci_high"),
            ("H2_band_ratio", "h2_band_ratio", "h2_band_ci_low", "h2_band_ci_high"),
        ):
            prefix = f"{output_name}_{metric_variant}"
            row[prefix] = source.get(value_name)
            row[f"{prefix}_ci_low"] = source.get(low_name)
            row[f"{prefix}_ci_high"] = source.get(high_name)
        cp = checkpoint.loc[checkpoint["format"].astype(str).eq(fmt)]
        if len(cp) != 1:
            raise AssertionError(f"Expected one checkpoint outcome for {variant}/{model}/{fmt}")
        row[f"Outcome_{metric_variant}"] = str(
            cp.iloc[0]["Outcome" if metric_variant == "s" else "s1_outcome"]
        )

    for metric_variant in ("s", "s1"):
        metric_rows = landmarks.loc[
            landmarks["model"].astype(str).eq(model)
            & landmarks["format"].astype(str).eq(fmt)
            & landmarks["metric_variant"].astype(str).eq(metric_variant)
            & landmarks["metric"].astype(str).isin({"L_peak", "L_concept_10", "L_cross"})
        ]
        for metric in ("L_peak", "L_concept_10", "L_cross"):
            selected = metric_rows.loc[metric_rows["metric"].astype(str).eq(metric)]
            if len(selected) != 1:
                raise AssertionError(f"Expected one {metric} landmark for {variant}/{model}/{fmt}/{metric_variant}")
            source = selected.iloc[0]
            prefix = f"{metric}_{metric_variant}"
            row[prefix] = source.get("value")
            row[f"{prefix}_ci_low"] = source.get("ci_low")
            row[f"{prefix}_ci_high"] = source.get("ci_high")

    la_rows = la.loc[la["model"].astype(str).eq(model)
                     & la["format"].astype(str).eq(fmt)
                     & la["metric"].astype(str).eq("max_LA")
                     & la["target_lang"].astype(str).eq("en")]
    if len(la_rows) != 1:
        raise AssertionError(f"Expected one LA_en max for {variant}/{model}/{fmt}")
    la_row = la_rows.iloc[0]
    row.update(
        max_LA_en=la_row.get("value"), max_LA_en_layer=la_row.get("layer"),
        max_LA_en_ci_low=la_row.get("ci_low"), max_LA_en_ci_high=la_row.get("ci_high"),
    )
    return row


def _filter_for_cell(
    variant: Mapping[str, Any], *, family: str, fmt: str, main_ids: set[str],
    cf_ids: set[str], metadata: Mapping[str, Mapping[str, Any]],
    competent_by_cell: Mapping[tuple[str, str], set[str]], r11_ids: set[str],
    available_ids: set[str],
) -> set[str]:
    return variant_ids(
        variant, main_ids=main_ids, cf_ids=cf_ids, metadata=metadata,
        competent_ids=competent_by_cell.get((family, fmt)), r11_ids=r11_ids,
    ) & available_ids


def _prompt_ids_by_format(
    records: Sequence[Mapping[str, Any]], main_ids: set[str]
) -> dict[str, set[str]]:
    """Resolve format-specific primary test ids, excluding extension rows."""
    result = {fmt: set() for fmt in FORMATS}
    for record in records:
        fmt = str(record.get("format", ""))
        extension = record.get("m1_extension") is True or str(record.get("m1_extension", "false")).lower() == "true"
        if fmt in result and str(record.get("split", "")) == "test" and not extension:
            concept_id = str(record.get("concept_id", ""))
            if concept_id in main_ids:
                result[fmt].add(concept_id)
    return result


def _load_primary_long(cfg: Mapping[str, Any], repo: Path, family: str) -> pd.DataFrame:
    """Load one family's existing P-L primary long rows, without extension rows."""
    from scripts import analyze_core as core

    main_ids, _cf_ids = core._primary_scope_ids(repo, cfg)
    paths = [path for path in core._primary_paths(repo, cfg, family=family) if "format=cloze__" not in path.name]
    tables = [pq.ParquetFile(path).read().to_pandas() for path in paths]
    raw = pd.concat(tables, ignore_index=True)
    return raw.loc[raw["concept_id"].astype(str).isin(main_ids)].copy()


def preflight(cfg: Mapping[str, Any], *, repo: Path = REPO) -> tuple[str, bool]:
    """Check stored P-L inputs and report variant row counts only."""
    from scripts import analyze_core as core
    from vilens.analysis.competence import competence_rows

    main_ids, cf_ids = core._primary_scope_ids(repo, cfg)
    prompt_ids = _prompt_ids_by_format(core._primary_prompt_records(cfg, repo), main_ids)
    metadata = _load_metadata(repo, main_ids)
    competent_by_cell: dict[tuple[str, str], set[str]] = {}
    for family in FAMILIES:
        generations = core._load_primary_generations(repo, cfg, main_ids, family=family)
        scored = competence_rows(generations, metadata)
        for fmt in FORMATS:
            competent_by_cell[(family, fmt)] = set(
                scored.loc[scored["format"].astype(str).eq(fmt)
                           & scored["competent"].astype(bool), "concept_id"].astype(str)
            )
    r11_ids = set(map(str, cfg["e6"]["r11_concept_ids"]))
    rows: list[str] = []
    all_ok = True
    for family in FAMILIES:
        paths = [path for path in core._primary_paths(repo, cfg, family=family) if "format=cloze__" not in path.name]
        required_paths = [
            companion
            for path in paths
            for companion in (
                path,
                path.with_suffix(".provenance.json"),
                path.with_suffix(".resid.npy"),
                path.with_suffix(".generations.jsonl"),
            )
        ]
        present = bool(paths) and all(path.is_file() for path in required_paths)
        for variant in cfg["r_battery_e4"]["variants"]:
            for fmt in FORMATS:
                ids = _filter_for_cell(
                    variant, family=family, fmt=fmt, main_ids=main_ids, cf_ids=cf_ids,
                    metadata=metadata, competent_by_cell=competent_by_cell, r11_ids=r11_ids,
                    available_ids=prompt_ids[fmt],
                )
                expected = (
                    len(prompt_ids[fmt]) if variant["scope"] == "main"
                    else len(prompt_ids[fmt] & cf_ids)
                )
                required = 0 < len(ids) <= expected
                all_ok &= present and required
                rows.append(f"{family} | {fmt} | {variant['name']} | {len(ids)} | {'PASS' if present and required else 'FAIL'}")
    output_paths = [repo / cfg["r_battery_e4"]["outputs"][key] for key in ("report", "csv")]
    outputs_clear = not any(path.exists() for path in output_paths)
    all_ok &= outputs_clear
    header = "family | format | variant | item_count | inputs"
    lines = [header, "--- | --- | --- | ---: | ---", *rows,
             f"P-L inputs complete={all_ok and outputs_clear}; outputs_absent={outputs_clear}; R10=omitted (aggregation unresolved)"]
    return "\n".join(lines), all_ok and outputs_clear


def _compute_variant_cell(
    family_scores: pd.DataFrame, cfg: Mapping[str, Any], *, family: str, fmt: str,
    variant: Mapping[str, Any], ids: set[str], repo: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the frozen E3/E4 bootstrap and E5 path for one variant/cell."""
    from scripts import analyze_core as core

    scores = family_scores.loc[
        family_scores["format"].astype(str).eq(fmt)
        & family_scores["concept_id"].astype(str).isin(ids)
    ].copy()
    if variant["score_source"] == "raw":
        deltas = scores.loc[scores["role"].astype(str).eq("correct")].copy()
        score_readouts = core._readout_rows(deltas)
        rises = core.concept_rise_curves(
            score_readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="s"
        )
        rises_s1 = core.concept_rise_curves(
            score_readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="s1"
        )
    else:
        deltas = core.matched_distractor_deltas(scores)
        readouts = core._readout_rows(deltas)
        score_readouts = core._readout_rows(scores.loc[scores["role"].astype(str).eq("correct")])
        rises = core.concept_rise_curves(
            readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="delta_s"
        )
        rises_s1 = core.concept_rise_curves(
            readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="delta_s1"
        )
    if score_readouts.empty or rises.empty:
        raise ValueError(f"No E3/E4 readouts for {family}/{fmt}/{variant['name']}")

    analysis_cfg = _variant_bootstrap_config(cfg, variant)
    item_set = str(variant["scope"])
    metric_parts = []
    landmark_parts = []
    for metric_variant, rise_frame, score_column in (("s", rises, "s"), ("s1", rises_s1, "s1")):
        rules, landmarks_frame, _window = core._bootstrap_core(
            per_concept_rise=rise_frame,
            per_concept_scores=score_readouts,
            source="r_battery_e4",
            item_set=item_set,
            seed=int(cfg["seed"]),
            analysis_config=analysis_cfg,
            score_column=score_column,
            metric_variant=metric_variant,
        )
        metric_parts.append(rules)
        landmark_parts.append(landmarks_frame)
    rules_wide = pd.concat(metric_parts, ignore_index=True)
    landmarks_wide = pd.concat(landmark_parts, ignore_index=True)

    expected = len(ids)
    la_long, la_summary = core._compute_la_source(
        "primary", config=cfg, repo=repo, main_ids=set(ids), family_filter=family,
        bootstrap_key_template=f"bootstrap|{str(variant['code']).lower()}|{{model}}|{{format}}|{str(variant['name']).removeprefix(str(variant['code']) + '_')}",
        formats_override=(fmt,), query_scope_ids=set(ids), expected_query_counts={fmt: expected},
    )
    model_rows = score_readouts["model"].astype(str).unique()
    if len(model_rows) != 1:
        raise AssertionError("Variant cell mixes model ids")
    model = str(model_rows[0])
    rules_long = core._summary_metric_long(rules_wide, source_rows=score_readouts, kind="rule")
    checkpoint, _verdict = core._build_checkpoint_rules(rules_long, la_summary)
    row = _summary_row(
        rules_wide, landmarks_wide, la_summary, checkpoint, model=model, fmt=fmt,
        variant=str(variant["name"]), item_count=len(ids),
    )
    return row, la_long


def analyze(cfg: Mapping[str, Any], *, repo: Path = REPO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the OPS-44 E4 sensitivity report and return its summary frames."""
    from scripts import analyze_core as core
    from vilens.analysis.competence import competence_rows

    table, okay = preflight(cfg, repo=repo)
    if not okay:
        raise RuntimeError("R-battery E4 preflight failed; refusing analysis")
    output_csv = repo / str(cfg["r_battery_e4"]["outputs"]["csv"])
    output_md = repo / str(cfg["r_battery_e4"]["outputs"]["report"])
    main_ids, cf_ids = core._primary_scope_ids(repo, cfg)
    metadata = _load_metadata(repo, main_ids)
    competent_by_cell: dict[tuple[str, str], set[str]] = {}
    for family in FAMILIES:
        scored = competence_rows(core._load_primary_generations(repo, cfg, main_ids, family=family), metadata)
        for fmt in FORMATS:
            competent_by_cell[(family, fmt)] = set(
                scored.loc[scored["format"].astype(str).eq(fmt)
                           & scored["competent"].astype(bool), "concept_id"].astype(str)
            )
    r11_ids = set(map(str, cfg["e6"]["r11_concept_ids"]))
    baseline_rules = pd.read_csv(repo / "reports/E2-E5_rules.csv")
    baseline_landmarks = pd.read_csv(repo / "reports/E2-E5_landmarks.csv")
    all_rows: list[dict[str, Any]] = []
    baseline_map: dict[tuple[str, str], dict[str, Any]] = {}
    outcome_match_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        raw = _load_primary_long(cfg, repo, family)
        scores = core.content_token_scores(raw)
        correct_scores = scores.loc[scores["role"].astype(str).eq("correct")]
        for fmt in FORMATS:
            format_available = set(
                scores.loc[scores["format"].astype(str).eq(fmt), "concept_id"].astype(str)
            )
            base_rule = baseline_rules.loc[
                baseline_rules["item_set"].astype(str).eq("cf729")
                & baseline_rules["format"].astype(str).eq(fmt)
                & baseline_rules["metric_variant"].astype(str).isin({"s", "s1"})
            ].copy()
            base_landmark = baseline_landmarks.loc[
                baseline_landmarks["item_set"].astype(str).eq("cf729")
                & baseline_landmarks["format"].astype(str).eq(fmt)
                & baseline_landmarks["metric_variant"].astype(str).isin({"s", "s1"})
            ].copy()
            base_ids = cf_ids & main_ids & format_available
            _base_la_long, base_la = core._compute_la_source(
                "primary", config=cfg, repo=repo, main_ids=main_ids, family_filter=family,
                formats_override=(fmt,), query_scope_ids=base_ids,
                expected_query_counts={fmt: len(base_ids)},
                bootstrap_key_template="bootstrap|r1|{model}|{format}|set1",
            )
            model = str(correct_scores.loc[correct_scores["format"].astype(str).eq(fmt), "model"].iloc[0])
            base_checkpoint, _base_verdict = core._build_checkpoint_rules(base_rule, base_la)
            base_cell = base_checkpoint.loc[base_checkpoint["format"].astype(str).eq(fmt)]
            if len(base_cell) != 1:
                raise AssertionError(f"Primary baseline lookup failed for {family}/{fmt}")
            baseline_map[(family, fmt)] = {
                "model": model, "rules": base_rule, "landmarks": base_landmark,
                "la": base_la, "checkpoint": base_cell.iloc[0].to_dict(),
            }
            for variant in cfg["r_battery_e4"]["variants"]:
                ids = _filter_for_cell(
                    variant, family=family, fmt=fmt, main_ids=main_ids, cf_ids=cf_ids,
                    metadata=metadata, competent_by_cell=competent_by_cell, r11_ids=r11_ids,
                    available_ids=format_available,
                )
                row, _la_long = _compute_variant_cell(
                    scores, cfg, family=family, fmt=fmt, variant=variant, ids=ids, repo=repo,
                )
                baseline = baseline_map[(family, fmt)]
                cp_rows = baseline["checkpoint"]
                row.update(
                    primary_Outcome=cp_rows.get("Outcome"),
                    primary_s1_outcome=cp_rows.get("s1_outcome"),
                )
                # Store baseline values from the committed primary rule/landmark tables beside each variant.
                for metric_variant in ("s", "s1"):
                    for metric, output_name in (("h1_q", "Q_H1"), ("h1_ratio", "H1_ratio"), ("h2_band_ratio", "H2_band_ratio")):
                        selected = baseline["rules"].loc[
                            baseline["rules"]["model"].astype(str).eq(row["model"])
                            & baseline["rules"]["format"].astype(str).eq(fmt)
                            & baseline["rules"]["metric"].astype(str).eq(metric)
                            & baseline["rules"]["metric_variant"].astype(str).eq(metric_variant)
                            & baseline["rules"]["target_lang"].astype(str).eq("__H1_H2__")
                        ]
                        if len(selected) != 1:
                            raise AssertionError(f"Primary baseline rule missing: {metric}/{metric_variant}/{family}/{fmt}")
                        source = selected.iloc[0]
                        prefix = f"primary_{output_name}_{metric_variant}"
                        row[prefix] = source.get("value")
                        row[f"{prefix}_ci_low"] = source.get("ci_low")
                        row[f"{prefix}_ci_high"] = source.get("ci_high")
                    for metric in ("L_peak", "L_concept_10", "L_cross"):
                        selected = baseline["landmarks"].loc[
                            baseline["landmarks"]["model"].astype(str).eq(row["model"])
                            & baseline["landmarks"]["format"].astype(str).eq(fmt)
                            & baseline["landmarks"]["metric"].astype(str).eq(metric)
                            & baseline["landmarks"]["metric_variant"].astype(str).eq(metric_variant)
                        ]
                        if len(selected) != 1:
                            raise AssertionError(f"Primary baseline landmark missing: {metric}/{metric_variant}/{family}/{fmt}")
                        source = selected.iloc[0]
                        prefix = f"primary_{metric}_{metric_variant}"
                        row[prefix] = source.get("value")
                        row[f"{prefix}_ci_low"] = source.get("ci_low")
                        row[f"{prefix}_ci_high"] = source.get("ci_high")
                for output_name in ("max_LA_en",):
                    selected = baseline["la"].loc[
                        baseline["la"]["model"].astype(str).eq(row["model"])
                        & baseline["la"]["format"].astype(str).eq(fmt)
                        & baseline["la"]["metric"].astype(str).eq("max_LA")
                        & baseline["la"]["target_lang"].astype(str).eq("en")
                    ]
                    if len(selected) != 1:
                        raise AssertionError(f"Primary LA baseline missing for {family}/{fmt}")
                    source = selected.iloc[0]
                    row.update(primary_max_LA_en=source.get("value"), primary_max_LA_en_layer=source.get("layer"),
                               primary_max_LA_en_ci_low=source.get("ci_low"), primary_max_LA_en_ci_high=source.get("ci_high"))
                all_rows.append(row)
                outcome_match_rows.append({
                    "family": family, "format": fmt, "variant": variant["name"],
                    "match_s": row["Outcome_s"] == row["primary_Outcome"],
                    "match_s1": row["Outcome_s1"] == row["primary_s1_outcome"],
                })
        del raw, scores, correct_scores

    summary = pd.DataFrame.from_records(all_rows)
    matches = pd.DataFrame.from_records(outcome_match_rows)
    match_counts = matches.groupby("variant", sort=False).agg(
        cells_equal_primary_s=("match_s", "sum"), cells_equal_primary_s1=("match_s1", "sum")
    ).reset_index()
    summary = summary.merge(match_counts, on="variant", how="left", validate="many_to_one")
    summary.to_csv(output_csv, index=False)
    report_lines = [
        "# R-battery E4 sensitivities (OPS-44; descriptive)", "",
        "The variants add no Holm family or decision rule; Checkpoint H remains primary.",
        "R10 is omitted: the registered along-path score is position-specific, but does not fix how positions are aggregated into the E3/E4 concept-by-layer input.",
        "The CSV contains every variant beside the primary set-1/cf729 values, with s and s1 rules/landmarks, E5 max LA_en, and per-variant agreement counts.",
        "", "## Preflight item counts", "", table,
        "", "## Outcomes and agreement counts", "",
        "| Variant | Cells equal primary under s | Cells equal primary under s1 |", "|---|---:|---:|",
    ]
    report_lines.extend(
        f"| {row.variant} | {int(row.cells_equal_primary_s)}/6 | {int(row.cells_equal_primary_s1)}/6 |"
        for row in match_counts.itertuples(index=False)
    )
    output_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary, matches


def main(argv: Sequence[str] | None = None) -> int:
    """Run the R-battery input preflight or descriptive sensitivity analysis."""
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
        raise SystemExit("R-battery E4 preflight failed; refusing analysis")
    analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
