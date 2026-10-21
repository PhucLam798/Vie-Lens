#!/datastore/luannt/.cache/sb/bin/python
"""Run the frozen A6 direction-validity diagnostics on non-test P-D outputs."""

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard; env_guard.guard(offline=True)

import argparse
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens.analysis.la import classification_accuracy, fit_layer_directions, random_cosine_baseline
from vilens.cli import setup_logging
from vilens.config import load_config
from vilens.derived import stream

LANGUAGES = ("vi", "en", "zh", "fr", "id")
FORMATS = ("repetition", "translation")
POOLINGS = ("last", "mean")


def _matched_path(root: Path, family: str, fmt: str) -> Path:
    """Return the exact committed-style P-D matched filename."""
    return root / f"family={family}__format={fmt}__pass=pd__source=matched.parquet"


def _flores_path(root: Path, family: str, suffix: str) -> Path:
    """Return the exact P-D FLORES filename for the requested suffix."""
    return root / f"family={family}__pass=pd__source=flores{suffix}"


def _read_index(path: Path, *, expected_rows: int, columns: list[str]) -> pd.DataFrame:
    """Read a compact P-D parquet index and assert its stored row coordinates."""
    frame = pq.read_table(path, columns=columns).to_pandas()
    if len(frame) != expected_rows:
        raise AssertionError(f"{path.name} has {len(frame)} rows, expected {expected_rows}")
    if "row" not in frame or not np.array_equal(
        frame["row"].to_numpy(dtype=np.int64), np.arange(expected_rows, dtype=np.int64)
    ):
        raise AssertionError(f"{path.name} row coordinates are not contiguous array indices")
    return frame


def _ordered_rows(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Map a boolean row selection to the aligned activation-array coordinates."""
    return frame.loc[mask, "row"].to_numpy(dtype=np.int64)


def _evaluate_layer(
    *,
    family: str,
    fmt: str,
    layer: int,
    matched: pd.DataFrame,
    matched_resid: np.ndarray,
    other: pd.DataFrame,
    other_resid: np.ndarray,
    flores: pd.DataFrame,
    flores_arrays: Mapping[str, np.ndarray],
    seed: int,
    gate_min_correct: int,
    random_key_template: str,
) -> dict[str, Any]:
    """Fit, validate, and transfer one model/format/block direction set."""
    fit = matched["split"].astype(str).eq("fit").to_numpy()
    validate = matched["split"].astype(str).eq("validate").to_numpy()
    if int(fit.sum()) != 530 or int(validate.sum()) != 130:
        raise AssertionError(f"{family}/{fmt} fit/validate counts differ from 530/130")
    if set(matched.loc[fit, "lang"].astype(str)) != set(LANGUAGES):
        raise AssertionError(f"{family}/{fmt} fit data do not contain all five languages")

    fit_rows = _ordered_rows(matched, fit)
    fit_acts = np.asarray(matched_resid[fit_rows, layer, :], dtype=np.float32)
    fit_labels = matched.loc[fit, "lang"].astype(str).to_numpy()
    fit_counts = pd.Series(fit_labels).value_counts().to_dict()
    if any(int(fit_counts.get(language, 0)) != 106 for language in LANGUAGES):
        raise AssertionError(f"{family}/{fmt} fit labels are not 106 per language: {fit_counts}")
    if not np.isfinite(fit_acts).all():
        raise AssertionError(f"{family}/{fmt} fit activations contain non-finite values at layer {layer}")
    center, directions, _means = fit_layer_directions(
        fit_acts,
        fit_labels.tolist(),
        class_order=LANGUAGES,
    )
    valid_rows = _ordered_rows(matched, validate)
    valid_acts = np.asarray(matched_resid[valid_rows, layer, :], dtype=np.float32)
    valid_labels = matched.loc[validate, "lang"].astype(str).to_numpy()
    valid_counts = pd.Series(valid_labels).value_counts().to_dict()
    if any(int(valid_counts.get(language, 0)) != 26 for language in LANGUAGES):
        raise AssertionError(f"{family}/{fmt} validation labels are not 26 per language: {valid_counts}")
    if not np.isfinite(valid_acts).all():
        raise AssertionError(f"{family}/{fmt} validation activations contain non-finite values at layer {layer}")
    gate_accuracy, gate_correct, gate_n = classification_accuracy(
        valid_acts, valid_labels, center, directions
    )

    vi_validate = validate & matched["lang"].astype(str).eq("vi").to_numpy()
    vi_rows = _ordered_rows(matched, vi_validate)
    vi_acts = np.asarray(matched_resid[vi_rows, layer, :], dtype=np.float32) - center
    random_key = random_key_template.format(family=family, format=fmt, layer=layer)
    c_random = random_cosine_baseline(
        vi_acts,
        rng=stream(seed, random_key),
        n_directions=1_000,
    )

    other_validate = other["split"].astype(str).eq("validate").to_numpy()
    other_rows = _ordered_rows(other, other_validate)
    other_acts = np.asarray(other_resid[other_rows, layer, :], dtype=np.float32)
    other_labels = other.loc[other_validate, "lang"].astype(str).to_numpy()
    if len(other_rows) != 130 or set(other_labels) != set(LANGUAGES):
        raise AssertionError(f"{family}/{other_fmt} cross-format validation set is not 130 five-language rows")
    other_counts = pd.Series(other_labels).value_counts().to_dict()
    if any(int(other_counts.get(language, 0)) != 26 for language in LANGUAGES):
        raise AssertionError(
            f"{family}/{other_fmt} cross-format labels are not 26 per language: {other_counts}"
        )
    if not np.isfinite(other_acts).all():
        raise AssertionError(f"{family}/{other_fmt} cross-format activations contain non-finite values at layer {layer}")
    cross_accuracy, cross_correct, cross_n = classification_accuracy(
        other_acts, other_labels, center, directions
    )

    flores_validate = flores["split"].astype(str).eq("validate").to_numpy()
    if int(flores_validate.sum()) != 995:
        raise AssertionError(f"{family} FLORES validation count is {int(flores_validate.sum())}, expected 995")
    flores_labels = flores.loc[flores_validate, "lang"].astype(str).to_numpy()
    flores_counts = pd.Series(flores_labels).value_counts().to_dict()
    if any(int(flores_counts.get(language, 0)) != 199 for language in LANGUAGES):
        raise AssertionError(f"{family} FLORES validation rows are not 199 per language: {flores_counts}")
    flores_rows = _ordered_rows(flores, flores_validate)
    result: dict[str, Any] = {
        "family": family,
        "fit_format": fmt,
        "layer": int(layer),
        "gate_correct": gate_correct,
        "gate_n": gate_n,
        "gate_accuracy": gate_accuracy,
        "gate_pass": bool(gate_correct >= gate_min_correct),
        "cross_format": "translation" if fmt == "repetition" else "repetition",
        "cross_format_correct": cross_correct,
        "cross_format_n": cross_n,
        "cross_format_accuracy": cross_accuracy,
        "c_random": c_random,
        "c_random_input_n": int(len(vi_rows)),
        "c_random_directions": 1_000,
        "c_random_seed_key": random_key,
        "flores_n": int(flores_validate.sum()),
    }
    for pooling in POOLINGS:
        acts = np.asarray(flores_arrays[pooling][flores_rows, layer, :], dtype=np.float32)
        accuracy, correct, count = classification_accuracy(
            acts,
            flores_labels,
            center,
            directions,
        )
        if not np.isfinite(acts).all():
            raise AssertionError(f"{family} FLORES {pooling} activations contain non-finite values at layer {layer}")
        result[f"flores_{pooling}_correct"] = correct
        result[f"flores_{pooling}_n"] = count
        result[f"flores_{pooling}_accuracy"] = accuracy
        result[f"label_dominated_{pooling}"] = bool(accuracy < 0.95)
    return result


def collect_a6(cfg: Mapping[str, Any], *, repo: Path = REPO) -> pd.DataFrame:
    """Evaluate all P-D families and formats without opening any test-result path."""
    result_root = repo / str(cfg["pd"]["results_dir"])
    if result_root.resolve() != (repo / "results/pd").resolve():
        raise ValueError("A6 input root must be the frozen P-D directory")
    seed = int(cfg["seed"])
    gate_min_correct = int(cfg["analysis"]["gate_min_correct"])
    random_key_template = str(cfg["analysis"]["la_random_key_template"])
    rows: list[dict[str, Any]] = []
    for family in cfg["pd"]["families"]:
        flores_index = _read_index(
            _flores_path(result_root, family, ".parquet"),
            expected_rows=int(cfg["pd"]["expected_flores_records"]),
            columns=["row", "sentence_id", "lang", "split"],
        )
        flores_arrays = {
            pooling: np.load(
                _flores_path(result_root, family, f".{pooling}.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
            for pooling in POOLINGS
        }
        for array in flores_arrays.values():
            if array.ndim != 3 or array.shape[0] != len(flores_index) or array.dtype != np.float32:
                raise AssertionError(f"{family} FLORES array shape/dtype does not match its index")
        matched_by_format: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        for fmt in FORMATS:
            index = _read_index(
                _matched_path(result_root, family, fmt),
                expected_rows=int(cfg["pd"]["expected_matched_records"]) // 2,
                columns=["row", "concept_id", "lang", "format", "split"],
            )
            resid_path = _matched_path(result_root, family, fmt).with_name(
                _matched_path(result_root, family, fmt).name.replace(".parquet", ".resid.npy")
            )
            resid = np.load(resid_path, mmap_mode="r", allow_pickle=False)
            if resid.ndim != 3 or resid.shape[0] != len(index) or resid.dtype != np.float32:
                raise AssertionError(f"{family}/{fmt} matched array shape/dtype does not match its index")
            if not np.isfinite(np.asarray(resid[:, 1, :], dtype=np.float32)).all():
                raise AssertionError(f"{family}/{fmt} matched activations contain non-finite values")
            matched_by_format[fmt] = (index, resid)

        n_layers = int(matched_by_format[FORMATS[0]][1].shape[1] - 1)
        if any(array.shape[1] != n_layers + 1 for _, array in matched_by_format.values()):
            raise AssertionError(f"{family} P-D formats have different layer counts")
        if any(array.shape[1] != n_layers + 1 for array in flores_arrays.values()):
            raise AssertionError(f"{family} FLORES and matched layer counts differ")
        for fmt in FORMATS:
            other_fmt = "translation" if fmt == "repetition" else "repetition"
            matched, resid = matched_by_format[fmt]
            other, other_resid = matched_by_format[other_fmt]
            for layer in range(1, n_layers + 1):
                rows.append(
                    _evaluate_layer(
                        family=family,
                        fmt=fmt,
                        layer=layer,
                        matched=matched,
                        matched_resid=resid,
                        other=other,
                        other_resid=other_resid,
                        flores=flores_index,
                        flores_arrays=flores_arrays,
                        seed=seed,
                        gate_min_correct=gate_min_correct,
                        random_key_template=random_key_template,
                    )
                )
    return pd.DataFrame.from_records(rows).sort_values(
        ["family", "fit_format", "layer"], kind="stable"
    ).reset_index(drop=True)


def _passing_ranges(frame: pd.DataFrame) -> str:
    """Format the set of passing blocks as a compact list of intervals."""
    layers = frame.loc[frame["gate_pass"], "layer"].astype(int).tolist()
    if not layers:
        return "none"
    intervals: list[str] = []
    start = previous = layers[0]
    for layer in layers[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        intervals.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = layer
    intervals.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(intervals)


def write_report(frame: pd.DataFrame, *, report_md: Path, report_csv: Path) -> None:
    """Write the factual A6 per-layer data and compact run summary."""
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_csv, index=False, float_format="%.8g")
    lines = [
        "# E5 gate — P-D direction-validity diagnostics",
        "",
        "Inputs: P-D matched directions and FLORES+ dev validation activations only; no test-item result was opened.",
        "Layers are block outputs ℓ=1…N. Gate PASS is ≥124/130 (95%). FLORES validation is 199 ids × 5 languages = 995 records; primary `last` and diagnostic `mean` pooling are both shown.",
        "A layer is label-dominated when its FLORES transfer accuracy for the indicated pooling is <95%.",
        "`c_random` is the mean cosine from centered validate Vietnamese activations to 1,000 seeded Gaussian random directions, using `bootstrap`-scheme streams keyed by `la_random|family|format|layer`.",
        "",
        "## Summary by model and fit format",
        "",
        "| Model | Fit format | Gate layers PASS / N | Passing block ranges | FLORES last mean (range) | FLORES mean-pool mean (range) | Cross-format accuracy mean | c_random mean (max abs) | Label-dominated last layers |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (family, fmt), group in frame.groupby(["family", "fit_format"], sort=True):
        lines.append(
            "| {family} | {fmt} | {passed}/{total} | {ranges} | {last_mean:.4f} ({last_min:.4f}–{last_max:.4f}) | {mean_mean:.4f} ({mean_min:.4f}–{mean_max:.4f}) | {cross:.4f} | {crand:.3e} ({crand_abs:.3e}) | {dominated} |".format(
                family=family,
                fmt=fmt,
                passed=int(group["gate_pass"].sum()),
                total=len(group),
                ranges=_passing_ranges(group),
                last_mean=float(group["flores_last_accuracy"].mean()),
                last_min=float(group["flores_last_accuracy"].min()),
                last_max=float(group["flores_last_accuracy"].max()),
                mean_mean=float(group["flores_mean_accuracy"].mean()),
                mean_min=float(group["flores_mean_accuracy"].min()),
                mean_max=float(group["flores_mean_accuracy"].max()),
                cross=float(group["cross_format_accuracy"].mean()),
                crand=float(group["c_random"].mean()),
                crand_abs=float(group["c_random"].abs().max()),
                dominated=int(group["label_dominated_last"].sum()),
            )
        )
    lines.extend(
        [
            "",
            "## Per-layer results",
            "",
            "| Model | Fit format | Block | Gate correct / 130 | Gate accuracy | Gate | FLORES last correct / 995 | FLORES last accuracy | Label-dominated last | FLORES mean correct / 995 | FLORES mean accuracy | Label-dominated mean | Cross-format correct / 130 | Cross-format accuracy | c_random |",
            "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {family} | {fit_format} | {layer} | {gate_correct}/130 | {gate_accuracy:.4f} | {gate} | {flores_last_correct}/995 | {flores_last_accuracy:.4f} | {dom_last} | {flores_mean_correct}/995 | {flores_mean_accuracy:.4f} | {dom_mean} | {cross_format_correct}/130 | {cross_format_accuracy:.4f} | {c_random:.3e} |".format(
                **row,
                gate="PASS" if row["gate_pass"] else "FAIL",
                dom_last="yes" if row["label_dominated_last"] else "no",
                dom_mean="yes" if row["label_dominated_mean"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "D12 note: five translation-format Vietnamese directions records have byte-identical prompts to test prompts; P-D fit is unchanged and these five overlaps are reported, not filtered here.",
            "",
            "| Directions concept_id | Matching test concept_id |",
            "|---|---|",
            "| `9b523e0c2ba7` | `1516bff503d4` |",
            "| `c8cac6e6221a` | `24d4529a0170` |",
            "| `70d57c0663fa` | `84b2d9c93a58` |",
            "| `44b4b01dae07` | `cc2d2ff59954` |",
            "| `5fec3d7df089` | `fb2a9a229f1a` |",
            "Fit/validate shared-prompt footnote (retained): repetition zh concept `92286b25eb72` (fit) / `d014c4a6e97c` (validate).",
            "",
            "The test-item LA curves and `D_en` are specified for T13b and were not computed in this task.",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run A6 diagnostics and write the registered gate report files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--results", default="results/pd", help=argparse.SUPPRESS)
    args = parser.parse_args()
    logger = setup_logging("a6_directions", REPO / "logs")
    if args.results != "results/pd":
        parser.error("A6 is restricted to the frozen results/pd inputs")
    cfg = load_config(args.config)
    frame = collect_a6(cfg)
    write_report(frame, report_md=REPO / "reports/E5_gate.md", report_csv=REPO / "reports/E5_gate.csv")
    logger.info("A6 rows=%s; report=reports/E5_gate.md; csv=reports/E5_gate.csv", len(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
