#!/datastore/luannt/.cache/sb/bin/python
"""Compare FLORES-fit and format-matched directions using frozen E5/A6 code."""

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
POOLINGS = ("last", "mean")
LANGUAGES = ("vi", "en", "zh", "fr", "id")
OUTPUTS = (
    "reports/R2.md",
    "reports/R2_directions.csv",
    "reports/R2_classification.csv",
    "reports/R2_alignment.csv",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a finite fp32 unit vector, rejecting a zero vector."""
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a zero or non-finite direction")
    return values / norm


def flores_direction_cosines(
    matched_activations: np.ndarray,
    matched_labels: Sequence[str],
    flores_activations: np.ndarray,
    flores_labels: Sequence[str],
) -> dict[str, float]:
    """Compare the five centered class directions and vi−en contrast."""
    from vilens.analysis.la import fit_layer_directions

    matched_center, matched_dirs, matched_means = fit_layer_directions(
        matched_activations, matched_labels, class_order=LANGUAGES
    )
    flores_center, flores_dirs, flores_means = fit_layer_directions(
        flores_activations, flores_labels, class_order=LANGUAGES
    )
    del matched_center, flores_center
    output = {
        f"v_{language}": float(np.dot(_unit(matched_dirs[language]), _unit(flores_dirs[language])))
        for language in LANGUAGES
    }
    matched_contrast = matched_means["vi"] - matched_means["en"]
    flores_contrast = flores_means["vi"] - flores_means["en"]
    output["d_vi_minus_en"] = float(np.dot(_unit(matched_contrast), _unit(flores_contrast)))
    return output


def flores_classification(
    flores_fit_activations: np.ndarray,
    flores_fit_labels: Sequence[str],
    matched_validate_activations: np.ndarray,
    matched_validate_labels: Sequence[str],
) -> tuple[int, int, bool]:
    """Classify matched validate rows with directions fit on FLORES rows."""
    from vilens.analysis.la import classification_accuracy, fit_layer_directions

    center, directions, _means = fit_layer_directions(
        flores_fit_activations, flores_fit_labels, class_order=LANGUAGES
    )
    _accuracy, correct, total = classification_accuracy(
        matched_validate_activations,
        matched_validate_labels,
        center,
        directions,
    )
    return correct, total, correct >= 124


def r2_bootstrap_key(model: str, fmt: str, pooling: str) -> str:
    """Return the OPS-41 deterministic bootstrap key for one comparison cell."""
    if pooling not in POOLINGS or fmt not in FORMATS:
        raise ValueError("unknown R2 format or pooling")
    return f"bootstrap|r2|{model}|{fmt}|{pooling}"


def _npy_shape(path: Path) -> tuple[tuple[int, ...], str]:
    """Read only an NPY header through a memory map."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(map(int, array.shape)), str(array.dtype)


def preflight(cfg: Mapping[str, Any], *, repo: Path = REPO) -> tuple[str, bool]:
    """Check all P-D, P-L, E5 and Checkpoint-H inputs by metadata only."""
    from scripts import analyze_core as core

    rows: list[tuple[str, str, str]] = []
    pd_root = repo / str(cfg["pd"]["results_dir"])
    n_layers_by_family: dict[str, int] = {}
    width_by_family: dict[str, int] = {}
    for family in FAMILIES:
        family_ok = True
        observed = []
        for fmt in FORMATS:
            stem = f"family={family}__format={fmt}__pass=pd__source=matched"
            parquet = pd_root / f"{stem}.parquet"
            array_path = parquet.with_suffix(".resid.npy")
            try:
                table = pq.read_table(parquet, columns=["row", "lang", "split"])
                index = table.to_pandas()
                shape, dtype = _npy_shape(array_path)
                valid = (
                    len(index) == int(cfg["pd"]["expected_matched_records"]) // 2
                    and shape[0] == len(index)
                    and len(shape) == 3
                    and dtype == "float32"
                    and np.array_equal(index["row"].to_numpy(dtype=np.int64), np.arange(len(index)))
                )
                observed.append((shape, dtype))
                family_ok &= bool(valid)
            except (OSError, ValueError, KeyError, TypeError):
                family_ok = False
        if len(observed) == 2:
            shape, _dtype = observed[0]
            n_layers_by_family[family] = int(shape[1] - 1)
            width_by_family[family] = int(shape[2])
            family_ok &= observed[0] == observed[1]

        flores_parquet = pd_root / f"family={family}__pass=pd__source=flores.parquet"
        flores_ok = True
        flores_shapes: list[tuple[tuple[int, ...], str]] = []
        for pooling in POOLINGS:
            try:
                index = pq.read_table(flores_parquet, columns=["row", "lang", "split"]).to_pandas()
                path = pd_root / f"family={family}__pass=pd__source=flores.{pooling}.npy"
                shape, dtype = _npy_shape(path)
                okay = (
                    len(index) == int(cfg["pd"]["expected_flores_records"])
                    and shape[0] == len(index)
                    and len(shape) == 3
                    and dtype == "float32"
                    and np.array_equal(index["row"].to_numpy(dtype=np.int64), np.arange(len(index)))
                    and int(index["split"].astype(str).eq("fit").sum()) == 3_990
                    and int(index["split"].astype(str).eq("validate").sum()) == 995
                )
                flores_ok &= bool(okay)
                flores_shapes.append((shape, dtype))
            except (OSError, ValueError, KeyError, TypeError):
                flores_ok = False
        if len(flores_shapes) == 2:
            flores_ok &= flores_shapes[0] == flores_shapes[1]
            if observed:
                flores_ok &= flores_shapes[0][0][1:] == observed[0][0][1:]
        rows.append(("P-D arrays", family, "PASS" if family_ok else "FAIL"))
        rows.append(("P-D FLORES arrays", family, "PASS" if flores_ok else "FAIL"))

    try:
        gate = pd.read_csv(repo / "reports/E5_gate.csv")
        e5 = pd.read_csv(repo / "reports/E2-E5_e5.csv")
        gate_ok = True
        for family in FAMILIES:
            n_layers = n_layers_by_family.get(family, -1)
            for fmt in FORMATS:
                selected = gate.loc[
                    gate["family"].astype(str).eq(family)
                    & gate["fit_format"].astype(str).eq(fmt)
                ]
                gate_ok &= len(selected) == n_layers and set(
                    pd.to_numeric(selected["layer"], errors="coerce").dropna().astype(int)
                ) == set(range(1, n_layers + 1))
        e5_ok = not e5.empty and {"model", "format", "metric", "target_lang", "value"}.issubset(e5.columns)
    except (OSError, ValueError, KeyError, TypeError):
        gate_ok = e5_ok = False
    rows.append(("E5 gate and baseline LA", "all cells", "PASS" if gate_ok and e5_ok else "FAIL"))

    try:
        primary_records = core._primary_prompt_records(cfg, repo)
        primary_paths = list(core._primary_paths(repo, cfg, records=primary_records))
        primary_ok = len(primary_records) == 3_346 and bool(primary_paths)
        for family in FAMILIES:
            paths = core._source_residual_paths(
                "primary", repo, family=family, fmt="repetition", config=cfg,
                primary_records=primary_records,
            ) + core._source_residual_paths(
                "primary", repo, family=family, fmt="translation", config=cfg,
                primary_records=primary_records,
            )
            if not paths:
                primary_ok = False
            for path in paths:
                sidecar = path.with_suffix(".provenance.json")
                resid = path.with_suffix(".resid.npy")
                index = path.with_suffix(".resid_index.jsonl")
                if not all(item.is_file() for item in (sidecar, resid, index)):
                    primary_ok = False
                    continue
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    shape, dtype = _npy_shape(resid)
                    parquet_rows = int(pq.ParquetFile(path).metadata.num_rows)
                    index_rows = sum(1 for line in index.open("rb") if line.strip())
                    primary_ok &= (
                        parquet_rows > 0
                        and parquet_rows == int(metadata.get("row_count", -1))
                        and parquet_rows == int(metadata.get("expected_row_count", -2))
                        and len(shape) == 3
                        and dtype == "float32"
                        and shape[0] == index_rows
                        and shape[1:] == (n_layers_by_family[family] + 1, width_by_family[family])
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    primary_ok = False
    except (OSError, KeyError, ValueError, TypeError):
        primary_ok = False
    rows.append(("P-L primary residuals", "repetition + translation", "PASS" if primary_ok else "FAIL"))
    output_clear = all(not (repo / path).exists() for path in OUTPUTS)
    rows.append(("R2 output paths", "no overwrite", "PASS" if output_clear else "FAIL"))

    headers = ("CHECK", "SCOPE", "STATUS")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(3)]
    table = " | ".join(headers[i].ljust(widths[i]) for i in range(3))
    table += "\n" + "-+-".join("-" * width for width in widths)
    table += "\n" + "\n".join(" | ".join(row[i].ljust(widths[i]) for i in range(3)) for row in rows)
    return table, all(row[2] == "PASS" for row in rows)


def _read_index(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only non-text P-D index columns needed for directions and splits."""
    return pq.read_table(path, columns=list(columns)).to_pandas()


def _direction_tables(
    cfg: Mapping[str, Any], *, repo: Path, family: str, pooling: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute layerwise direction cosines and validate-row classification counts."""
    pd_root = repo / str(cfg["pd"]["results_dir"])
    rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    gate = pd.read_csv(repo / "reports/E5_gate.csv")
    for fmt in FORMATS:
        matched_path = pd_root / f"family={family}__format={fmt}__pass=pd__source=matched.parquet"
        matched_index = _read_index(matched_path, ["row", "lang", "split"])
        matched = np.load(matched_path.with_suffix(".resid.npy"), mmap_mode="r", allow_pickle=False)
        flores_path = pd_root / f"family={family}__pass=pd__source=flores.parquet"
        flores_index = _read_index(flores_path, ["row", "lang", "split"])
        flores = np.load(
            pd_root / f"family={family}__pass=pd__source=flores.{pooling}.npy",
            mmap_mode="r", allow_pickle=False,
        )
        fit_mask = matched_index["split"].astype(str).eq("fit").to_numpy()
        valid_mask = matched_index["split"].astype(str).eq("validate").to_numpy()
        fit_rows = matched_index.loc[fit_mask, "row"].to_numpy(dtype=np.int64)
        valid_rows = matched_index.loc[valid_mask, "row"].to_numpy(dtype=np.int64)
        fit_labels = matched_index.loc[fit_mask, "lang"].astype(str).to_numpy()
        valid_labels = matched_index.loc[valid_mask, "lang"].astype(str).to_numpy()
        flores_fit_mask = flores_index["split"].astype(str).eq("fit").to_numpy()
        flores_fit_rows = flores_index.loc[flores_fit_mask, "row"].to_numpy(dtype=np.int64)
        flores_labels = flores_index.loc[flores_fit_mask, "lang"].astype(str).to_numpy()
        n_layers = int(matched.shape[1] - 1)
        for layer in range(1, n_layers + 1):
            matched_layer = np.asarray(matched[fit_rows, layer, :], dtype=np.float32)
            flores_layer = np.asarray(flores[flores_fit_rows, layer, :], dtype=np.float32)
            direction_values = flores_direction_cosines(
                matched_layer, fit_labels, flores_layer, flores_labels
            )
            rows.append({
                "model": family, "format": fmt, "pooling": pooling, "layer": layer,
                **direction_values,
            })
            correct, total, passes = flores_classification(
                flores_layer,
                flores_labels,
                np.asarray(matched[valid_rows, layer, :], dtype=np.float32),
                valid_labels,
            )
            matched_gate = gate.loc[
                gate["family"].astype(str).eq(family)
                & gate["fit_format"].astype(str).eq(fmt)
                & pd.to_numeric(gate["layer"], errors="coerce").eq(layer)
            ]
            if len(matched_gate) != 1:
                raise AssertionError(f"Expected one matched E5 gate row for {family}/{fmt}/{layer}")
            matched_correct = int(matched_gate.iloc[0]["gate_correct"])
            classification_rows.append({
                "model": family, "format": fmt, "pooling": pooling, "layer": layer,
                "flores_correct": correct, "total": total, "meets_124_of_130": passes,
                "matched_correct": matched_correct,
                "matched_meets_124_of_130": matched_correct >= 124,
            })
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(classification_rows)


def analyze(cfg: Mapping[str, Any], *, repo: Path = REPO) -> None:
    """Write the OPS-41 vector, classification and primary LA comparisons."""
    from scripts import analyze_core as core

    table, okay = preflight(cfg, repo=repo)
    if not okay:
        raise RuntimeError("R2 inputs are incomplete; refusing analysis")
    existing = [repo / path for path in OUTPUTS if (repo / path).exists()]
    if existing:
        raise FileExistsError("refusing to replace existing R2 outputs: " + ", ".join(map(str, existing)))
    directions: list[pd.DataFrame] = []
    classifications: list[pd.DataFrame] = []
    alignment: list[pd.DataFrame] = []
    main_ids, _cognate_ids = core._primary_scope_ids(repo, cfg)
    model_specs = {family: spec for family, spec in core.study_models(dict(cfg)).items()}
    for family in FAMILIES:
        for pooling in POOLINGS:
            vector_frame, classification_frame = _direction_tables(
                cfg, repo=repo, family=family, pooling=pooling
            )
            directions.append(vector_frame)
            classifications.append(classification_frame)
            model_id = str(model_specs[family]["hf_id"])
            la_long, la_summary = core._compute_la_source(
                "primary",
                config=cfg,
                repo=repo,
                main_ids=main_ids,
                family_filter=family,
                direction_source="flores",
                pooling=pooling,
                bootstrap_key_template=f"bootstrap|r2|{model_id}|{{format}}|{pooling}",
                formats_override=FORMATS,
            )
            baseline = pd.read_csv(repo / "reports/E2-E5_e5.csv")
            for fmt in FORMATS:
                selected = la_summary.loc[
                    la_summary["format"].astype(str).eq(fmt)
                    & la_summary["metric"].astype(str).eq("max_LA")
                    & la_summary["target_lang"].astype(str).eq("en")
                ]
                primary = baseline.loc[
                    baseline["model"].astype(str).eq(model_id)
                    & baseline["format"].astype(str).eq(fmt)
                    & baseline["metric"].astype(str).eq("max_LA")
                    & baseline["target_lang"].astype(str).eq("en")
                ]
                if len(selected) != 1 or len(primary) != 1:
                    raise AssertionError(f"Expected one primary and R2 max_LA_en row for {family}/{fmt}")
                current = selected.iloc[0]
                original = primary.iloc[0]
                gate_rows = pd.read_csv(repo / "reports/E5_gate.csv")
                failed = gate_rows.loc[
                    gate_rows["family"].astype(str).eq(family)
                    & gate_rows["fit_format"].astype(str).eq(fmt)
                    & ~gate_rows["gate_pass"].astype(str).str.lower().eq("true"),
                    "layer",
                ].astype(int).tolist()
                alignment.append(pd.DataFrame.from_records([{
                    "model": model_id, "family": family, "format": fmt, "pooling": pooling,
                    "max_LA_en": current["value"], "layer": current["layer"],
                    "ci_low": current["ci_low"], "ci_high": current["ci_high"],
                    "primary_max_LA_en": original["value"],
                    "primary_layer": original["layer"],
                    "primary_ci_low": original.get("ci_low"),
                    "primary_ci_high": original.get("ci_high"),
                    "primary_A6_gate_failed_layers": ",".join(map(str, failed)),
                    "bootstrap_key": r2_bootstrap_key(model_id, fmt, pooling),
                }]))

    output_frames = (
        pd.concat(directions, ignore_index=True),
        pd.concat(classifications, ignore_index=True),
        pd.concat(alignment, ignore_index=True),
    )
    for path, frame in zip((repo / value for value in OUTPUTS[1:]), output_frames, strict=True):
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    report = [
        "# R2 FLORES-fit versus format-matched directions (descriptive)", "",
        "Last-token pooling is primary; mean pooling is a sensitivity. This analysis adds no Holm family or decision rule.",
        "The five language directions and the vi-minus-en contrast are compared by cosine. Matched-fit validate classification uses 130 rows per format and reports the 124/130 criterion beside the FLORES-fit result.",
        "Primary-item max LA for English is recomputed with FLORES-fit directions and shown beside the existing matched-direction value; the primary A6 layer gate is retained.",
        "", "| Model | Format | Pooling | FLORES-fit max LA_en | Layer | Primary max LA_en | Primary layer |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for frame in output_frames[2].itertuples(index=False):
        report.append(
            f"| {frame.model} | {frame.format} | {frame.pooling} | {frame.max_LA_en} | {frame.layer} | {frame.primary_max_LA_en} | {frame.primary_layer} |"
        )
    report.extend([
        "", "Outputs: reports/R2_directions.csv, reports/R2_classification.csv, and reports/R2_alignment.csv.", "",
    ])
    report_path = repo / OUTPUTS[0]
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(report), encoding="utf-8")
    temporary.replace(report_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the R2 metadata preflight or its registered analysis."""
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
        raise SystemExit("R2 inputs are incomplete; refusing R2 analysis")
    analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
