"""Summarize P-A E7 projections without loading study weights."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

env_guard.guard(offline=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/datastore/luannt/.cache/matplotlib")
os.environ.setdefault("CUDA_CACHE_PATH", "/datastore/luannt/.cache/nv")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens import attribute
from vilens.config import load_config
from vilens.io import unit_paths

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
T22B_OPS37_AS_WRITTEN = {
    ("gemma", "repetition"): [1, 2, 3, 22, 24, 25, 26, 27, 30],
    ("gemma", "translation"): [1, 23, 24, 25, 26, 27, 28, 29, 30],
    ("qwen", "repetition"): [23, 24, 25, 27, 29, 30, 32, 33, 34],
    ("qwen", "translation"): [22, 24, 25, 27, 29, 31, 32, 33, 34],
    ("llama", "repetition"): [3, 15, 16, 21, 22, 25, 27],
    ("llama", "translation"): [15, 16, 20, 23, 24, 25, 26],
}
PA_COLUMNS = [
    "model", "model_revision", "pass", "format", "item_set", "concept_id", "layer",
    "layer_type", "kind", "head", "direction", "projection", "update_norm",
]


def _expected_units(
    cfg: dict[str, Any], p_a_summary: dict[str, Any]
) -> list[dict[str, str | int]]:
    """Enumerate P-A output units using the captured source replay batch counts."""
    replay_source = str(cfg["e7"]["replay"]["p_a"]["source"])
    output: list[dict[str, str | int]] = []
    for family in FAMILIES:
        for fmt in FORMATS:
            try:
                output.extend(
                    attribute.replay_unit_keys_from_summary(
                        p_a_summary,
                        family=family,
                        format_name=fmt,
                        pass_name="pa",
                        replay_source=replay_source,
                        item_sets=("main", "cf729"),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"P-A summary unit plan invalid for {family}/{fmt}: {exc}") from exc
    return output


def preflight(cfg: dict[str, Any]) -> tuple[str, bool]:
    """Check inputs and P-A unit presence from metadata only; read no projection values."""
    pa_dir = REPO / cfg["e7"]["results_dir"]
    rows: list[tuple[str, str, str]] = []
    summary_path = pa_dir / "summary.json"
    p_a_summary: dict[str, Any] = {}
    summary_ok = False
    try:
        p_a_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_ok = int(p_a_summary.get("exit_code", -1)) == 0
        summary_detail = (
            f"job={p_a_summary.get('slurm_job_id')}; "
            f"exit_code={p_a_summary.get('exit_code')}"
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        summary_detail = f"unavailable or invalid ({type(exc).__name__}: {exc})"
    rows.append(("P-A run summary", summary_detail, "PASS" if summary_ok else "FAIL"))
    for label, path in (
        ("landmark summary", REPO / "reports/E2-E5_landmarks.csv"),
        ("A6 gate summary", REPO / "reports/E5_gate.csv"),
        ("cf729 manifest", REPO / "derived/cognate_flags.jsonl"),
    ):
        rows.append((label, "present" if path.is_file() else "missing", "PASS" if path.is_file() else "FAIL"))
    for family in FAMILIES:
        for fmt in FORMATS:
            delta_path = REPO / cfg["e7"]["analysis_results_dir"] / f"family={family}" / "deltas.parquet"
            if delta_path.is_file():
                parquet = pq.ParquetFile(delta_path)
                schema = set(parquet.schema_arrow.names)
                okay = {"format", "condition", "fewshot_set", "concept_id", "target_lang", "target_variant", "role", "layer", "metric", "value"}.issubset(schema)
                detail = f"deltas.parquet rows={parquet.metadata.num_rows}; required schema={'PASS' if okay else 'FAIL'}"
            else:
                okay, detail = False, "deltas.parquet missing"
            rows.append((f"delta curve input {family}/{fmt}", detail, "PASS" if okay else "FAIL"))
    try:
        landmarks = pd.read_csv(REPO / "reports/E2-E5_landmarks.csv")
    except (OSError, ValueError) as exc:
        landmarks = None
        landmark_error = f"{type(exc).__name__}: {exc}"
    else:
        landmark_error = ""
    try:
        gates = pd.read_csv(REPO / "reports/E5_gate.csv")
    except (OSError, ValueError) as exc:
        gates = None
        gate_error = f"{type(exc).__name__}: {exc}"
    else:
        gate_error = ""
    for family in FAMILIES:
        for fmt in FORMATS:
            try:
                if landmarks is None or gates is None:
                    raise FileNotFoundError(landmark_error or gate_error)
                resolved = _resolve_cell_bands(cfg, landmarks, gates, family, fmt)
                bands = resolved["bands"]
                n_layers = int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
                requested = (n_layers + 3) // 4
                early_max = max(resolved["early_blocks"], default=0)
                eligible = set(bands["eligible_blocks"])
                gate_pass = resolved["gate_pass"]
                if family == "llama" and fmt == "repetition":
                    primary_expected = (
                        bands["rule"] == "i"
                        and resolved["l_peak"] == 23
                        and resolved["l_cross"] == 27
                        and bands["primary"]["blocks"] == [23, 24, 25, 26, 27]
                    )
                else:
                    expected_size = min(requested, len(eligible))
                    primary_expected = (
                        bands["rule"] == "ii"
                        and resolved["l_cross"] is None
                        and bool(bands["primary"]["blocks"])
                        and len(bands["primary"]["blocks"]) == expected_size
                        and bands["primary"]["blocks"] == bands["slope_sensitivity"]["blocks"]
                        and all(
                            block > early_max and gate_pass[block]
                            for block in bands["primary"]["blocks"]
                        )
                    )
                for band_label in ("primary", "slope_sensitivity", "ops37_as_written"):
                    band = bands[band_label]
                    if band_label == "primary":
                        band_ok = primary_expected
                    elif band_label == "slope_sensitivity":
                        band_ok = (
                            len(band["blocks"]) == min(requested, len(eligible))
                            and all(block in eligible for block in band["blocks"])
                            and not band["excluded_gate_failed_blocks"]
                        )
                    else:
                        band_ok = (
                            band["blocks"] == T22B_OPS37_AS_WRITTEN[(family, fmt)]
                            and len(band["blocks"]) == requested
                        )
                    detail = (
                        f"L_peak={_format_layer(resolved['l_peak'])}; "
                        f"L_cross={_format_layer(resolved['l_cross'])}; rule={bands['rule']}; "
                        f"blocks={band['blocks']}; hull={band['contiguous_hull']}; "
                        f"excluded={band['excluded_gate_failed_blocks']}; "
                        f"eligible_count={band['eligible_count']}; requested={band['requested_size']}; "
                        f"pool_count={band['selection_pool_count']}; "
                        f"used_all_eligible={band['used_all_eligible']}"
                    )
                    rows.append((
                        f"band {family}/{fmt}/{band_label}", detail,
                        "PASS" if band_ok else "FAIL",
                    ))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                rows.append((f"band resolution {family}/{fmt}", f"FAIL ({type(exc).__name__}: {exc})", "FAIL"))
    missing = 0
    bad = 0
    try:
        expected_units = _expected_units(cfg, p_a_summary) if summary_ok else []
    except (KeyError, TypeError, ValueError) as exc:
        expected_units = []
        bad += 1
        rows.append(("P-A replay unit plan", f"invalid ({type(exc).__name__}: {exc})", "FAIL"))
    if not expected_units and summary_ok:
        bad += 1
        rows.append(("P-A replay unit plan", "no expected replay units", "FAIL"))
    for unit in expected_units:
        parquet_path, sidecar_path = unit_paths(pa_dir, unit)
        if not parquet_path.is_file() or not sidecar_path.is_file():
            missing += 1
            continue
        try:
            parquet = pq.ParquetFile(parquet_path)
            metadata_rows = int(parquet.metadata.num_rows)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            valid = (
                metadata_rows > 0
                and metadata_rows == sidecar.get("row_count")
                and metadata_rows == sidecar.get("expected_row_count")
                and sidecar.get("unit_key") == unit
                and set(PA_COLUMNS).issubset(parquet.schema_arrow.names)
            )
        except Exception:
            valid = False
        if not valid:
            bad += 1
    rows.append((
        "P-A unit files",
        f"expected={len(expected_units)}; missing={missing}; invalid row/provenance metadata={bad}",
        "PASS" if missing == 0 and bad == 0 else "MISSING" if missing else "FAIL",
    ))
    headers = ("CHECK", "DETAIL", "STATUS")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(3)]
    table = " | ".join(headers[i].ljust(widths[i]) for i in range(3))
    table += "\n" + "-+-".join("-" * widths[i] for i in range(3))
    table += "\n" + "\n".join(" | ".join(row[i].ljust(widths[i]) for i in range(3)) for row in rows)
    return table, all(row[2] == "PASS" for row in rows)


def _model_id(cfg: dict[str, Any], family: str) -> str:
    """Resolve the selected pinned model id for report-key matching."""
    group = cfg["models"][cfg["models"]["study_set"]]
    return str(group[family]["hf_id"])


def _cf_ids() -> set[str]:
    """Read the frozen cognate-free item IDs only."""
    path = REPO / "derived/cognate_flags.jsonl"
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cognate_free") is True or str(row.get("cognate_free", "")).lower() == "true":
                ids.add(str(row["concept_id"]))
    return ids


def _format_layer(value: int | None) -> str:
    """Format a landmark layer, preserving a missing landmark as NA."""
    return "NA" if value is None else str(value)


def _landmark_point(frame: pd.DataFrame, *, model: str, fmt: str, metric: str) -> int | None:
    """Resolve exactly the OPS-37 primary-s, cf729, set-1 point-estimate row."""
    return attribute.primary_landmark_point(frame, model=model, format_name=fmt, metric=metric)


def _delta_curve(cfg: dict[str, Any], family: str, fmt: str, cf_ids: set[str], n_layers: int) -> np.ndarray:
    """Read the primary-set Vietnamese delta-s rows and average them by residual layer."""
    path = REPO / cfg["e7"]["analysis_results_dir"] / f"family={family}" / "deltas.parquet"
    frame = pq.read_table(
        path,
        columns=["pass", "format", "condition", "fewshot_set", "concept_id", "target_lang", "target_variant", "role", "layer", "metric", "value"],
    ).to_pandas()
    selected = frame.loc[
        frame["pass"].astype(str).eq("pl")
        & frame["format"].astype(str).eq(fmt)
        & frame["condition"].astype(str).eq("diac")
        & pd.to_numeric(frame["fewshot_set"], errors="coerce").eq(1)
        & frame["concept_id"].astype(str).isin(cf_ids)
        & frame["target_lang"].astype(str).eq("vi")
        & frame["target_variant"].astype(str).eq("released")
        & frame["role"].astype(str).eq("path")
        & frame["metric"].astype(str).eq("delta")
    ]
    if selected.empty:
        raise ValueError(f"primary Δ_vi curve is empty for {family}/{fmt}")
    if selected.duplicated(["concept_id", "layer"]).any():
        raise ValueError(f"primary Δ_vi curve has duplicate concept/layer rows for {family}/{fmt}")
    means = selected.groupby("layer", sort=True)["value"].mean()
    curve = np.full(n_layers + 1, np.nan, dtype=np.float64)
    for layer, value in means.items():
        layer_i = int(layer)
        if 0 <= layer_i <= n_layers:
            curve[layer_i] = float(value)
    if not np.isfinite(curve).all():
        raise ValueError(f"primary Δ_vi curve lacks one or more residual layers for {family}/{fmt}")
    return curve


def _gate_map(gates: pd.DataFrame, family: str, fmt: str, n_layers: int) -> dict[int, bool]:
    """Resolve A6 pass flags for every block layer."""
    selected = gates.loc[
        gates["family"].astype(str).eq(family)
        & gates["fit_format"].astype(str).eq(fmt)
    ]
    if selected["layer"].duplicated().any() or len(selected) != n_layers:
        raise ValueError(f"A6 gate rows are not unique for every block layer in {family}/{fmt}")
    output: dict[int, bool] = {}
    for row in selected.itertuples(index=False):
        layer = int(row.layer)
        if 1 <= layer <= n_layers:
            if pd.isna(row.gate_pass):
                raise ValueError(f"A6 gate flag is missing for {family}/{fmt}/{layer}")
            output[layer] = bool(row.gate_pass)
    if set(output) != set(range(1, n_layers + 1)):
        raise ValueError(f"A6 gate table does not cover 1..N for {family}/{fmt}")
    return output


def _resolve_cell_bands(
    cfg: dict[str, Any], landmarks: pd.DataFrame, gates: pd.DataFrame, family: str, fmt: str
) -> dict[str, Any]:
    """Resolve the frozen E7 bands from primary landmarks, deltas, and A6 gates."""
    model = _model_id(cfg, family)
    n_layers = int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
    cf_ids = _cf_ids()
    l_cross = _landmark_point(landmarks, model=model, fmt=fmt, metric="L_cross")
    l_peak = _landmark_point(landmarks, model=model, fmt=fmt, metric="L_peak")
    delta_curve = _delta_curve(cfg, family, fmt, cf_ids, n_layers)
    gate_pass = _gate_map(gates, family, fmt, n_layers)
    early_limits = list(map(int, cfg["e1"]["expected_early_blocks"][family]))
    if len(early_limits) != 2 or early_limits[0] > early_limits[1]:
        raise ValueError(f"configured ℓ_early range is invalid for {family}: {early_limits}")
    early_blocks = list(range(early_limits[0], early_limits[1] + 1))
    bands = attribute.resolve_bands(
        n_layers=n_layers,
        l_cross=l_cross,
        l_peak=l_peak,
        mean_delta_vi=delta_curve,
        gate_pass=gate_pass,
        early_blocks=early_blocks,
    )
    return {
        "l_cross": l_cross,
        "l_peak": l_peak,
        "bands": bands,
        "early_blocks": early_blocks,
        "gate_pass": gate_pass,
    }


def analyze(cfg: dict[str, Any]) -> None:
    """Build exploratory E7 summaries and report from completed P-A scalar units."""
    pa_dir = REPO / cfg["e7"]["results_dir"]
    landmarks = pd.read_csv(REPO / "reports/E2-E5_landmarks.csv")
    gates = pd.read_csv(REPO / "reports/E5_gate.csv")
    cf_ids = _cf_ids()
    layers_out: list[pd.DataFrame] = []
    band_out: list[pd.DataFrame] = []
    heads_out: list[pd.DataFrame] = []
    heads_all_out: list[pd.DataFrame] = []
    report_lines = [
        "# E7 component attribution (exploratory)", "",
        "Run record: P-A projections summarized with OPS-37; results are exploratory and descriptive.",
        "Head ranking: v2: head ranking corrected to (layer, head) per OPS-37-7.", "",
    ]
    summary_path = pa_dir / "summary.json"
    p_a_summary: dict[str, Any] = {}
    if summary_path.is_file():
        p_a_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report_lines.extend([
            f"P-A job: {p_a_summary.get('slurm_job_id')}; code commit: {p_a_summary.get('git_commit')}; GPU: {p_a_summary.get('gpu_name')}; wall seconds: {p_a_summary.get('wall_seconds')}.",
            "",
        ])
        for family in FAMILIES:
            family_summary = p_a_summary.get("families", {}).get(family)
            if not family_summary:
                continue
            report_lines.append(
                f"P-A {family}: peak GPU memory bytes={family_summary.get('peak_gpu_memory_bytes')}; "
                f"family wall seconds={family_summary.get('wall_seconds')}."
            )
            for fmt in FORMATS:
                unit = family_summary.get(fmt, {})
                report_lines.append(
                    f"P-A {family}/{fmt}: main prompts={unit.get('main_prompts')}; cf729 prompts={unit.get('cf729_prompts')}; "
                    f"P-D fit overlap={unit.get('p_d_fit_e7_overlap')}; D12 note={unit.get('d12_note')}; "
                    f"direction relative error={unit.get('d_vs_negative_e8_max_relative')}; "
                    f"trial checks={json.dumps(unit.get('trial_checks', {}), sort_keys=True)}; "
                    f"full checks={json.dumps(unit.get('full_checks', {}), sort_keys=True)}; units={json.dumps(unit.get('units', {}), sort_keys=True)}."
                )
        report_lines.append("")
    for family in FAMILIES:
        for fmt in FORMATS:
            unit_paths_list = [unit_paths(pa_dir, key)[0] for key in _expected_units(cfg, p_a_summary) if key["family"] == family and key["format"] == fmt]
            existing = [path for path in unit_paths_list if path.is_file()]
            frame = pd.concat([pq.read_table(path, columns=PA_COLUMNS).to_pandas() for path in existing], ignore_index=True)
            if frame.empty:
                raise FileNotFoundError(f"No P-A projections for {family}/{fmt}")
            resolved = _resolve_cell_bands(cfg, landmarks, gates, family, fmt)
            bands = resolved["bands"]
            n_layers = int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
            gate_pass = _gate_map(gates, family, fmt, n_layers)
            report_lines.extend([
                f"## {family} — {fmt}",
                "",
                f"Primary band blocks: {bands['primary']['blocks']}; gate-excluded blocks: {bands['primary']['excluded_gate_failed_blocks']}.",
                f"OPS-43 slope sensitivity blocks: {bands['slope_sensitivity']['blocks']}; gate-excluded blocks: {bands['slope_sensitivity']['excluded_gate_failed_blocks']}.",
                f"OPS-37 as-written sensitivity blocks: {bands['ops37_as_written']['blocks']}; gate-excluded blocks: {bands['ops37_as_written']['excluded_gate_failed_blocks']}.",
                "",
            ])
            for item_set in ("cf729", "main"):
                item_frame = frame.loc[frame["item_set"].astype(str).eq(item_set)].copy()
                layer_summary, band_summary, head_summary, head_all_summary = attribute.summarize_attribution(
                    item_frame,
                    model=_model_id(cfg, family), format_name=fmt, item_set=item_set,
                    bands=bands,
                    seed=int(cfg["seed"]), key_template=str(cfg["e7"]["bootstrap_key_template"]),
                    n_resamples=int(cfg["e7"]["bootstrap_resamples"]),
                    top_n=int(cfg["e7"]["head_rank_n"]), share_epsilon=float(cfg["e7"]["share_epsilon"]),
                    gate_pass=gate_pass,
                )
                layers_out.append(layer_summary)
                band_out.append(band_summary)
                heads_out.append(head_summary)
                heads_all_out.append(head_all_summary)
                if item_set == "cf729":
                    report_lines.extend([
                        "### Primary-band head rankings",
                        "",
                        "Ranked by concept-mean projection onto d̂; each entry is layer.head, mean projection [95% concept-bootstrap CI], and top-10 inclusion frequency.",
                        "",
                    ])
                    for layer_type in sorted(head_summary["layer_type"].astype(str).unique()):
                        typed = head_summary[head_summary["layer_type"].astype(str).eq(layer_type)]
                        report_lines.extend([
                            f"Layer type: {layer_type}.",
                            "",
                            "| sign | rank | layer.head | mean [CI] | inclusion |",
                            "|---|---:|---:|---:|---:|",
                        ])
                        for sign_name, rank_column, inclusion_column in (
                            ("positive", "top_positive_rank", "top_positive_inclusion"),
                            ("negative", "top_negative_rank", "top_negative_inclusion"),
                        ):
                            ranked = typed.loc[typed[rank_column].notna()].sort_values(rank_column)
                            for row in ranked.itertuples(index=False):
                                rank_value = getattr(row, rank_column)
                                inclusion = getattr(row, inclusion_column)
                                report_lines.append(
                                    f"| {sign_name} | {int(rank_value)} | {int(row.layer)}.{int(row.head)} | "
                                    f"{row.mean_projection:.8g} [{row.ci_low:.8g}, {row.ci_high:.8g}] | {inclusion:.6f} |"
                                )
                        report_lines.append("")
    outputs = cfg["e7"]["outputs"]
    pd.concat(layers_out, ignore_index=True).to_csv(REPO / outputs["layers"], index=False)
    pd.concat(band_out, ignore_index=True).to_csv(REPO / outputs["band"], index=False)
    pd.concat(heads_out, ignore_index=True).to_csv(REPO / outputs["heads"], index=False)
    pd.concat(heads_all_out, ignore_index=True).to_csv(REPO / outputs["heads_all"], index=False)
    report_lines.extend([
        "## Output tables", "",
        f"- `{outputs['layers']}`: per-block component projections, update norms, attention shares, and concept-bootstrap CIs.",
        f"- `{outputs['band']}`: primary and slope-sensitivity band sums, signs, shares, and CIs.",
        f"- `{outputs['heads']}`: primary-band (layer, head) rankings and top-10 inclusion frequencies; Gemma types are separate.",
        f"- `{outputs['heads_all']}`: full-depth per-(layer, head) descriptive values and concept-bootstrap CIs.",
        "",
    ])
    (REPO / cfg["e7"]["report"]).write_text("\n".join(report_lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run metadata preflight or summarize completed P-A outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    table, okay = preflight(cfg)
    print(table)
    if args.preflight:
        if not okay:
            print("E7 preflight failed; no statistics computed.")
            return 2
        return 0
    if not okay:
        raise SystemExit("P-A outputs are incomplete; refusing analysis")
    analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
