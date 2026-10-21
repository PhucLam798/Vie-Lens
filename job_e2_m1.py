"""Compute the preregistered E2 prefix-sum M1 readout from saved prompt residuals."""

from __future__ import annotations

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

env_guard.guard(offline=True)

import argparse
import gc
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from vilens import io, models
from vilens.analysis.m1 import (
    absolute_delta_gate_passes,
    cyclic_shift_lens_control_deltas,
    first_token_lens_rows,
    lens_control_passes,
    prefix_token_ids,
    require_nonempty_prefix_token_ids,
    summarize_lens_control_deltas,
    summarize_report_only_lens_deltas,
    unicode_prefix_token_id,
    validate_probability_values,
    vocabulary_prefix_index,
)
from vilens.cli import StageTimer, setup_logging
from vilens.config import load_config
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.lens import LensHead
from vilens.preflight import Check, format_checks, run_preflight
from vilens.provenance import build as build_provenance
from vilens.provenance import capture_run_git
from vilens.passes import pl_unit_key, with_pl_shards

LANGUAGES = ("vi", "en", "zh", "fr", "id", "ru")
M1_READOUTS = (
    ("vi", "released"),
    ("en", "released"),
    ("zh", "nospace"),
    ("fr", "released"),
    ("id", "released"),
)
PRIMARY_FORMATS = ("repetition", "translation", "cloze")
GATE_A_ABS_TOLERANCE = 1.0e-2
GATE_A_REPORT_THRESHOLD = 1.0e-4
GATE_A_CONTROL_MINIMUM_MEDIAN = 0.1
_LOGGER = logging.getLogger("e2_m1")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an ordered JSON-lines input file."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _inputs_and_records(cfg: Mapping[str, Any], source: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Return preflight input paths and prompt records without touching result outputs."""
    root = Path(cfg["paths"]["repo_root"])
    inputs = [str(cfg["paths"]["concepts_jsonl"])]
    if source == "primary":
        stems = tuple(cfg["pl"]["primary"]["files"])
        records = []
        for stem in stems:
            path = prompt_file_paths(dict(cfg))[stem]
            if not path.is_absolute():
                path = root / path
            records.extend(load_prompt_file(path))
            inputs.append(path.relative_to(root).as_posix())
        records = with_pl_shards(records, int(cfg["pl"]["shard_prompts"]))
    else:
        path = prompt_file_paths(dict(cfg))["directions_matched"]
        if not path.is_absolute():
            path = root / path
        records = load_prompt_file(path)
        inputs.append(path.relative_to(root).as_posix())
    return inputs, records


def _preflight(cfg: dict[str, Any], source: str, *, gpu: bool) -> list[Any]:
    """Run standard login-node preflight without loading weights or reading P-L results."""
    inputs, records = _inputs_and_records(cfg, source)
    output_dir = Path("results/analysis/m1") / source
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E2",
        gpu=gpu,
        inputs=inputs,
        families=list(cfg["pd"]["families"]),
        records=records,
        output_dirs=[output_dir],
        entrypoint=Path(__file__),
    )
    if source == "primary":
        checks.append(
            _e2_inputs_resolve(
                records,
                list(cfg["pd"]["families"]),
                REPO / cfg_results_dir(source),
            )
        )
    return checks


def _unit_parquet_path(source: str, family: str, fmt: str, shard: int) -> Path:
    """Resolve a trial P-L output parquet from its registered unit key."""
    unit = {
        "condition": "diac",
        "family": family,
        "fewshot_set": "none",
        "format": fmt,
        "pass": "pl",
        "shard": shard,
    }
    results_dir = Path(cfg_results_dir(source))
    return io.unit_paths(REPO / results_dir, unit)[0]


def cfg_results_dir(source: str) -> str:
    """Return the registered P-L result root for primary or trial inputs."""
    return "results/pl" if source == "primary" else "results/pl_trial2"


def _expected_primary_units(
    records: Sequence[Mapping[str, Any]], family: str, results_dir: Path
) -> tuple[list[tuple[str, Path]], dict[str, tuple[int, int]], list[str]]:
    """Resolve primary unit paths from P-L's shared unit key and prompt shards."""
    grouped: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    prompt_counts: dict[str, int] = {}
    for record in records:
        fmt = str(record["format"])
        key = pl_unit_key(family, record)
        signature = tuple((name, str(value)) for name, value in sorted(key.items()))
        grouped.setdefault(signature, {"format": fmt, "key": key, "prompts": 0})["prompts"] += 1
        prompt_counts[fmt] = prompt_counts.get(fmt, 0) + 1
    units: list[tuple[str, Path]] = []
    unit_counts: dict[str, int] = {}
    missing: list[str] = []
    for signature in sorted(grouped):
        group = grouped[signature]
        fmt = str(group["format"])
        parquet_path, sidecar_path = io.unit_paths(results_dir, group["key"])
        units.append((fmt, parquet_path))
        unit_counts[fmt] = unit_counts.get(fmt, 0) + 1
        required = (
            parquet_path,
            sidecar_path,
            parquet_path.with_suffix(".resid.npy"),
            parquet_path.with_suffix(".resid_index.jsonl"),
            parquet_path.with_suffix(".answer_topk.jsonl"),
        )
        missing.extend(str(path) for path in required if not path.is_file())
    counts = {
        fmt: (prompt_counts.get(fmt, 0), unit_counts.get(fmt, 0))
        for fmt in sorted(set(prompt_counts) | set(unit_counts))
    }
    return units, counts, missing


def _e2_inputs_resolve(
    records: Sequence[Mapping[str, Any]], families: Sequence[str], results_dir: Path
) -> Check:
    """Check every expected primary P-L unit and its E2-required sidecars by path only."""
    summaries: list[str] = []
    missing: list[str] = []
    present_formats = {str(record.get("format")) for record in records}
    missing_formats = sorted(set(PRIMARY_FORMATS) - present_formats)
    if missing_formats:
        return Check(
            "e2_inputs_resolve",
            "FAIL",
            "primary prompt records missing formats: " + ", ".join(missing_formats),
        )
    try:
        for family in families:
            _units, counts, family_missing = _expected_primary_units(records, family, results_dir)
            missing.extend(family_missing)
            summaries.append(
                f"{family}: "
                + ", ".join(
                    f"{fmt}={prompt_count} prompts/{unit_count} units"
                    for fmt, (prompt_count, unit_count) in counts.items()
                )
            )
    except Exception as exc:
        return Check("e2_inputs_resolve", "FAIL", f"{type(exc).__name__}: {exc}")
    if missing:
        return Check(
            "e2_inputs_resolve",
            "FAIL",
            "; ".join(summaries) + "; missing=" + ", ".join(missing[:12]),
        )
    return Check("e2_inputs_resolve", "PASS", "; ".join(summaries))


def _result_units(
    source: str,
    family: str,
    formats: Sequence[str],
    *,
    primary_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[str, Path]]:
    """Enumerate P-L unit paths using the P-L plan for primary inputs."""
    root = REPO / cfg_results_dir(source)
    units: list[tuple[str, Path]] = []
    if source == "primary":
        if primary_records is None:
            raise ValueError("Primary P-L unit resolution requires the P-L-sharded prompt records")
        expected, _counts, missing = _expected_primary_units(primary_records, family, root)
        if missing:
            raise FileNotFoundError("Missing E2 primary P-L unit files: " + ", ".join(missing[:12]))
        return [(fmt, path) for fmt, path in expected if fmt in formats]
    for fmt in formats:
        parquet_path = _unit_parquet_path(source, family, fmt, 0)
        if parquet_path.is_file():
            units.append((fmt, parquet_path))
    if not units:
        raise FileNotFoundError(f"No P-L residual units were found under the registered {source} source")
    return units


def _load_target_forms() -> dict[tuple[str, str, str], str]:
    """Load frozen target forms by concept, language, and variant."""
    rows = _read_jsonl(REPO / "derived/target_forms.jsonl")
    return {
        (str(row["concept_id"]), str(row["target_lang"]), str(row["target_variant"])): str(row["target"])
        for row in rows
    }


def _eligible_concepts() -> dict[str, dict[str, Any]]:
    """Load concept flags needed to select the frozen family-specific M1 set."""
    concepts_path = REPO / "data/external/viLens-concepts/concepts.jsonl"
    return {str(row["concept_id"]): row for row in _read_jsonl(concepts_path)}


def _token_id_sets(
    tokenizer: Any,
    family: str,
    concept_ids: Sequence[str],
    target_forms: Mapping[tuple[str, str, str], str],
    concepts: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str, str], tuple[int, ...]]:
    """Build deduplicated Wendler prefix ids for the requested forms."""
    token_sets: dict[tuple[str, str, str], tuple[int, ...]] = {}
    prefix_index = vocabulary_prefix_index(tokenizer, family=family)
    for concept_id in concept_ids:
        for language in LANGUAGES:
            if language == "ru":
                variants = ("released",)
            elif language == "zh":
                variants = ("nospace",)
            else:
                variants = ("released",)
            for variant in variants:
                key = (concept_id, language, variant)
                target = target_forms.get(key)
                if target is None and language == "ru":
                    ru = concepts.get(concept_id, {}).get("ru")
                    if isinstance(ru, str) and ru.strip():
                        target = " " + ru
                if target is None:
                    if (language, variant) in M1_READOUTS:
                        raise ValueError(
                            f"Missing target string for concept_id={concept_id} "
                            f"readout={language}/{variant} family={family}"
                        )
                    continue
                if language == "ru" and not target.strip():
                    continue
                unicode_id = (
                    unicode_prefix_token_id(tokenizer, target, family=family)
                    if language in {"zh", "ru"}
                    else None
                )
                prefix_fn = (
                    require_nonempty_prefix_token_ids
                    if (language, variant) in M1_READOUTS
                    else prefix_token_ids
                )
                prefix_kwargs = (
                    {"item_description": f"concept_id={concept_id} readout={language}/{variant}"}
                    if prefix_fn is require_nonempty_prefix_token_ids
                    else {}
                )
                token_sets[key] = prefix_fn(
                    tokenizer,
                    target,
                    family=family,
                    language=language,
                    unicode_prefix_tokid=unicode_id,
                    prefix_index=prefix_index,
                    **prefix_kwargs,
                )
    return token_sets


def _format_for_resid_index(fmt: str) -> str:
    """Normalize the P-L format name for result-index matching."""
    return "cloze" if fmt == "cloze" else fmt


def _compute_unit(
    loaded: Any,
    *,
    source: str,
    fmt: str,
    fewshot_set: int | None,
    parquet_path: Path,
    concepts: Mapping[str, Mapping[str, Any]],
    token_sets: Mapping[tuple[str, str, str], tuple[int, ...]],
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """Score E2 sets and compare both P-L answer-topk and teacher-forced values."""
    device = next(loaded.model.parameters()).device
    resid_path = parquet_path.with_suffix(".resid.npy")
    index_path = parquet_path.with_suffix(".resid_index.jsonl")
    resid = np.load(resid_path, mmap_mode="r", allow_pickle=False)
    index = _read_jsonl(index_path)
    if resid.ndim != 3 or resid.shape[0] != len(index) or resid.shape[1] != loaded.n_layers + 1:
        raise AssertionError(f"Residual dimensions do not match index for {parquet_path.name}")
    selected: dict[str, tuple[int, bool, str]] = {}
    for row in index:
        concept_id = str(row["concept_id"])
        concept = concepts.get(concept_id)
        if concept is None:
            continue
        eligible = (
            source == "trial"
            or str(concept.get(f"m1_eligible_{loaded.family}", "false")).lower() == "true"
        )
        if not eligible:
            continue
        row_fmt = _format_for_resid_index(str(row["format"]))
        if row_fmt != fmt:
            continue
        array_row = int(row["array_row"])
        if concept_id not in selected or str(row.get("lang", "")) == "vi":
            selected[concept_id] = (
                array_row,
                str(row.get("m1_extension", "false")).lower() == "true",
                str(row.get("lang", "")),
            )

    if not selected:
        return (
            pd.DataFrame(columns=io.LENS_COLUMNS + ["m1_extension"]),
            0,
            {
                "gate_a": {
                    "prompt_layer_count": 0,
                    "missing_prompt_layers": [],
                    "matched_count": 0,
                    "max_abs_delta": None,
                    "max_relative_delta": None,
                    "failed_count": 0,
                    "count_over_1e-4": 0,
                    "largest_cases": [],
                    "failure_case": None,
                    "argmax_key": None,
                    "e2_value": None,
                    "reference_value": None,
                },
                "gate_b": {
                    "matched_count": 0,
                    "missing_count": 0,
                    "median_abs_delta": None,
                    "p95_abs_delta": None,
                    "p99_abs_delta": None,
                    "max_abs_delta": None,
                    "argmax_index": None,
                    "argmax_key": None,
                    "e2_value": None,
                    "reference_value": None,
                },
                "gate_b_deltas": np.asarray([], dtype=np.float64),
                "gate_a_control": summarize_lens_control_deltas(
                    np.asarray([], dtype=np.float64)
                ),
                "gate_a_control_deltas": np.asarray([], dtype=np.float64),
            },
        )
    readouts = M1_READOUTS
    expected_row_count = sum(
        (concept_id, language, variant) in token_sets
        for concept_id in selected
        for language, variant in readouts
    ) * (loaded.n_layers + 1)

    pl_rows = pq.read_table(
        parquet_path,
        columns=["concept_id", "target_lang", "target_variant", "role", "tok_idx", "layer", "metric", "token_id", "value"],
        filters=[
            ("metric", "=", "lens_logprob"),
            ("role", "in", ["correct", "path"]),
            ("tok_idx", "=", 0),
        ],
    ).to_pandas()
    consistency = first_token_lens_rows(
        pl_rows.to_dict(orient="records"),
        concept_ids=set(selected),
        readouts=readouts,
        n_layers=loaded.n_layers,
    )
    answer_topk_path = parquet_path.with_suffix(".answer_topk.jsonl")
    answer_topk_by_prompt_layer: dict[tuple[str, int], list[dict[str, Any]]] = {}
    answer_topk_rows = _read_jsonl(answer_topk_path) if answer_topk_path.is_file() else []
    for row in answer_topk_rows:
        concept_id = str(row.get("concept_id", ""))
        if concept_id not in selected or str(row.get("format", "")) != fmt:
            continue
        selected_lang = selected[concept_id][2]
        row_lang = str(row.get("target_lang", ""))
        if selected_lang and row_lang != selected_lang:
            continue
        layer = int(row.get("layer", -1))
        if not 0 <= layer <= loaded.n_layers:
            continue
        answer_topk_by_prompt_layer.setdefault((concept_id, layer), []).append(row)

    head = LensHead.from_loaded(loaded, "fp32")
    output: list[dict[str, Any]] = []
    gate_a_matched_count = 0
    gate_a_prompt_layer_count = len(selected) * (loaded.n_layers + 1)
    gate_a_missing_prompt_layers: list[dict[str, Any]] = []
    gate_a_max_abs_delta = -1.0
    gate_a_max_relative_delta = -1.0
    gate_a_failed_count = 0
    gate_a_count_over_1e4 = 0
    gate_a_largest_cases: list[dict[str, Any]] = []
    gate_a_failure_case: dict[str, Any] | None = None
    gate_a_failure_excess = -1.0
    gate_a_argmax_key: dict[str, Any] | None = None
    gate_a_e2_value: float | None = None
    gate_a_reference_value: float | None = None
    gate_b_deltas: list[float] = []
    gate_b_missing_count = 0
    gate_b_max_abs_delta = -1.0
    gate_b_argmax_key: dict[str, Any] | None = None
    gate_b_e2_value: float | None = None
    gate_b_reference_value: float | None = None
    gate_a_control_deltas: list[np.ndarray] = []
    selected_ids = sorted(selected)
    control_prompt_ids = list(selected)
    next_control_prompt_by_id = {
        prompt_id: control_prompt_ids[(index + 1) % len(control_prompt_ids)]
        for index, prompt_id in enumerate(control_prompt_ids)
    }
    prompt_batch = int(loaded.lens_position_chunk)
    for layer in range(loaded.n_layers + 1):
        for start in range(0, len(selected_ids), prompt_batch):
            batch_ids = selected_ids[start : start + prompt_batch]
            array_rows = [selected[concept_id][0] for concept_id in batch_ids]
            hidden = torch.as_tensor(np.asarray(resid[array_rows, layer, :]), dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits = head.logits(hidden).float()
                probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
                log_probabilities = torch.log_softmax(logits, dim=-1)
            if source == "trial" and layer >= 1:
                control_deltas = cyclic_shift_lens_control_deltas(
                    log_probabilities.detach().cpu().numpy(),
                    batch_ids,
                    next_control_prompt_by_id,
                    answer_topk_by_prompt_layer,
                    layer=layer,
                )
                if control_deltas.size:
                    gate_a_control_deltas.append(control_deltas)
            for local, concept_id in enumerate(batch_ids):
                extension = selected[concept_id][1]
                topk_rows = answer_topk_by_prompt_layer.get((concept_id, layer), [])
                valid_topk_count = 0
                if not topk_rows:
                    gate_a_missing_prompt_layers.append(
                        {
                            "family": loaded.family,
                            "format": fmt,
                            "concept": concept_id,
                            "readout": "prompt/answer_topk",
                            "layer": layer,
                        }
                    )
                else:
                    topk_ids = [int(row["token_id"]) for row in topk_rows]
                    if len(topk_ids) != len(set(topk_ids)):
                        duplicate_id = next(
                            token_id for token_id in topk_ids if topk_ids.count(token_id) > 1
                        )
                        duplicate_row = next(
                            row for row in topk_rows if int(row["token_id"]) == duplicate_id
                        )
                        duplicate_e2 = (
                            float(log_probabilities[local, duplicate_id].item())
                            if 0 <= duplicate_id < int(log_probabilities.shape[-1])
                            else float("nan")
                        )
                        _log_consistency_failure(
                            "Gate A duplicate answer_topk token",
                            key={
                                "family": loaded.family,
                                "format": fmt,
                                "concept": concept_id,
                                "readout": (
                                    f"{duplicate_row.get('target_lang', 'prompt')}/"
                                    f"{duplicate_row.get('target_variant', 'prompt')}"
                                ),
                                "layer": layer,
                                "token_id": duplicate_id,
                            },
                            e2_value=duplicate_e2,
                            reference_value=float(duplicate_row["value"]),
                            delta=abs(duplicate_e2 - float(duplicate_row["value"])),
                            tolerance=GATE_A_ABS_TOLERANCE,
                        )
                        raise AssertionError(
                            f"Duplicate P-L answer_topk token ids for {loaded.family}/{fmt}/"
                            f"{concept_id}/layer={layer}"
                        )
                    for topk_row, token_id in zip(topk_rows, topk_ids, strict=True):
                        if not 0 <= token_id < int(log_probabilities.shape[-1]):
                            _LOGGER.error(
                                "Gate A invalid answer_topk token: key=%s token_id=%s "
                                "e2_value=<out_of_vocab> answer_topk_value=%s matched_count=0",
                                json.dumps(
                                    {
                                        "family": loaded.family,
                                        "format": fmt,
                                        "concept": concept_id,
                                        "readout": (
                                            f"{topk_row.get('target_lang', 'prompt')}/"
                                            f"{topk_row.get('target_variant', 'prompt')}"
                                        ),
                                        "layer": layer,
                                        "token_id": token_id,
                                    },
                                    sort_keys=True,
                                ),
                                token_id,
                                topk_row.get("value"),
                            )
                            continue
                        e2_value = float(log_probabilities[local, token_id].item())
                        reference_value = float(topk_row["value"])
                        delta = (
                            abs(e2_value - reference_value)
                            if np.isfinite(e2_value) and np.isfinite(reference_value)
                            else float("inf")
                        )
                        within_gate = delta <= GATE_A_ABS_TOLERANCE
                        relative_delta = (
                            delta / abs(reference_value)
                            if np.isfinite(delta) and reference_value != 0.0
                            else 0.0 if delta == 0.0 else float("inf")
                        )
                        tolerance = GATE_A_ABS_TOLERANCE
                        key = {
                            "family": loaded.family,
                            "format": fmt,
                            "concept": concept_id,
                            "readout": (
                                f"{topk_row.get('target_lang', 'prompt')}/"
                                f"{topk_row.get('target_variant', 'prompt')}"
                            ),
                            "layer": layer,
                            "token_id": token_id,
                        }
                        relative_for_report: float | str = (
                            float(relative_delta) if np.isfinite(relative_delta) else "inf"
                        )
                        case = {
                            "key": key,
                            "e2_value": e2_value,
                            "reference_value": reference_value,
                            "abs_delta": delta,
                            "relative_delta": relative_for_report,
                            "tolerance": tolerance,
                            "within_gate": within_gate,
                        }
                        gate_a_largest_cases.append(case)
                        gate_a_largest_cases.sort(
                            key=lambda entry: (
                                -float(entry["abs_delta"]),
                                json.dumps(entry["key"], sort_keys=True),
                            )
                        )
                        del gate_a_largest_cases[5:]
                        if np.isfinite(relative_delta):
                            gate_a_max_relative_delta = max(
                                gate_a_max_relative_delta, relative_delta
                            )
                        elif relative_delta > gate_a_max_relative_delta:
                            gate_a_max_relative_delta = relative_delta
                        if delta > GATE_A_REPORT_THRESHOLD:
                            gate_a_count_over_1e4 += 1
                        if not within_gate:
                            gate_a_failed_count += 1
                            excess = delta - tolerance
                            if excess > gate_a_failure_excess:
                                gate_a_failure_excess = excess
                                gate_a_failure_case = {
                                    **case,
                                    "relative_delta": relative_for_report,
                                    "tolerance_excess": excess,
                                }
                        valid_topk_count += 1
                        gate_a_matched_count += 1
                        if delta > gate_a_max_abs_delta:
                            gate_a_max_abs_delta = delta
                            gate_a_argmax_key = key
                            gate_a_e2_value = e2_value
                            gate_a_reference_value = reference_value
                if topk_rows and valid_topk_count == 0:
                    gate_a_missing_prompt_layers.append(
                        {
                            "family": loaded.family,
                            "format": fmt,
                            "concept": concept_id,
                            "readout": "prompt/answer_topk",
                            "layer": layer,
                        }
                    )
                for language, variant in readouts:
                    token_key = (concept_id, language, variant)
                    ids = token_sets.get(token_key)
                    if token_key in token_sets:
                        probability = float(
                            probabilities[local, torch.as_tensor(ids, dtype=torch.long, device=device)]
                            .sum()
                            .item()
                        )
                        concept = concepts.get(concept_id, {})
                        output.append(
                            {
                                "model": loaded.spec.hf_id if loaded.spec else loaded.family,
                                "model_revision": loaded.spec.revision if loaded.spec else "synthetic",
                                "pass": "m1",
                                "format": fmt,
                                "condition": "diac",
                                "fewshot_set": fewshot_set,
                                "concept_id": concept_id,
                                "target_lang": language,
                                "target_variant": variant,
                                "role": "correct",
                                "ref_concept_id": concept_id,
                                "tok_idx": 0,
                                "token_id": int(ids[0]),
                                "is_content": True,
                                "layer": layer,
                                "metric": "prefix_sum_probability",
                                "value": probability,
                                "m1_extension": extension
                                or str(concept.get("m1_extension", "false")).lower() == "true",
                            }
                        )
                    consistency_row = consistency.get((concept_id, language, variant, layer))
                    if consistency_row is None:
                        gate_b_missing_count += 1
                        continue
                    first_token_id, pl_logprob = consistency_row
                    e2_value = float(log_probabilities[local, first_token_id].item())
                    delta = (
                        abs(e2_value - pl_logprob)
                        if np.isfinite(e2_value) and np.isfinite(pl_logprob)
                        else float("inf")
                    )
                    gate_b_deltas.append(delta)
                    if delta > gate_b_max_abs_delta:
                        gate_b_max_abs_delta = delta
                        gate_b_argmax_key = {
                            "family": loaded.family,
                            "format": fmt,
                            "concept": concept_id,
                            "readout": f"{language}/{variant}",
                            "layer": layer,
                            "token_id": first_token_id,
                        }
                        gate_b_e2_value = e2_value
                        gate_b_reference_value = pl_logprob
            del probabilities, log_probabilities, logits, hidden
    frame = pd.DataFrame.from_records(output, columns=io.LENS_COLUMNS + ["m1_extension"])
    if len(frame) != expected_row_count:
        raise AssertionError(
            f"M1 rows {len(frame)} differ from independently counted expected rows {expected_row_count}"
        )
    validate_probability_values(
        frame["value"].to_numpy(dtype=np.float64),
        expected_rows=expected_row_count,
        observed_rows=len(frame),
    )
    gate_b_deltas_array = np.asarray(gate_b_deltas, dtype=np.float64)
    gate_b_summary = summarize_report_only_lens_deltas(gate_b_deltas_array)
    gate_b_summary["missing_count"] = gate_b_missing_count
    gate_a_summary: dict[str, Any] = {
        "prompt_layer_count": gate_a_prompt_layer_count,
        "missing_prompt_layers": gate_a_missing_prompt_layers,
        "matched_count": gate_a_matched_count,
        "max_abs_delta": None if gate_a_max_abs_delta < 0.0 else gate_a_max_abs_delta,
        "max_relative_delta": (
            None
            if gate_a_max_relative_delta < 0.0
            else float(gate_a_max_relative_delta)
            if np.isfinite(gate_a_max_relative_delta)
            else "inf"
        ),
        "failed_count": gate_a_failed_count,
        "count_over_1e-4": gate_a_count_over_1e4,
        "largest_cases": gate_a_largest_cases,
        "failure_case": gate_a_failure_case,
        "argmax_key": gate_a_argmax_key,
        "e2_value": gate_a_e2_value,
        "reference_value": gate_a_reference_value,
    }
    gate_b_summary.update(
        argmax_key=gate_b_argmax_key,
        e2_value=gate_b_e2_value,
        reference_value=gate_b_reference_value,
    )
    gate_a_control_deltas_array = (
        np.concatenate(gate_a_control_deltas)
        if gate_a_control_deltas
        else np.asarray([], dtype=np.float64)
    )
    gate_a_control_summary = summarize_lens_control_deltas(gate_a_control_deltas_array)
    return frame, expected_row_count, {
        "gate_a": gate_a_summary,
        "gate_b": gate_b_summary,
        "gate_b_deltas": gate_b_deltas_array,
        "gate_a_control": gate_a_control_summary,
        "gate_a_control_deltas": gate_a_control_deltas_array,
    }


def _log_consistency_failure(
    gate_name: str,
    *,
    key: Mapping[str, Any],
    e2_value: float,
    reference_value: float,
    delta: float,
    tolerance: float,
) -> None:
    """Log the largest keyed mismatch with both numeric values."""
    _LOGGER.error(
        "%s failed: key=%s e2_value=%.9g reference_value=%.9g abs_delta=%.9g tolerance=%.8g",
        gate_name,
        json.dumps(dict(key), sort_keys=True),
        e2_value,
        reference_value,
        delta,
        tolerance,
    )


def _run_source_once(cfg: dict[str, Any], source: str, run_git: Any) -> dict[str, Any]:
    """Run checked M1 units for one source and return exact count summaries."""
    concepts = _eligible_concepts()
    target_forms = _load_target_forms()
    output_root = REPO / "results/analysis/m1" / source
    output_root.mkdir(parents=True, exist_ok=True)
    formats = PRIMARY_FORMATS if source == "primary" else ("repetition", "translation")
    primary_records = _inputs_and_records(cfg, "primary")[1] if source == "primary" else None
    unit_summaries: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    for family in cfg["pd"]["families"]:
        family_started = time.perf_counter()
        loaded = models.load(cfg, family, device=torch.device("cuda"))
        if source == "primary":
            relevant_ids = sorted(
                concept_id
                for concept_id, concept in concepts.items()
                if str(concept.get(f"m1_eligible_{family}", "false")).lower() == "true"
            )
        else:
            relevant_ids = sorted(
                {
                    str(row["concept_id"])
                    for fmt in formats
                    for _, path in _result_units(
                        source, family, (fmt,), primary_records=primary_records
                    )
                    for row in _read_jsonl(path.with_suffix(".resid_index.jsonl"))
                }
            )
        token_sets = _token_id_sets(
            loaded.tokenizer,
            family,
            relevant_ids,
            target_forms,
            concepts,
        )
        family_gate_a_matched_count = 0
        family_gate_a_prompt_layer_count = 0
        family_gate_a_missing_prompt_layers: list[dict[str, Any]] = []
        family_gate_a_max_abs_delta = -1.0
        family_gate_a_max_relative_delta = -1.0
        family_gate_a_failed_count = 0
        family_gate_a_count_over_1e4 = 0
        family_gate_a_largest_cases: list[dict[str, Any]] = []
        family_gate_a_failure_case: dict[str, Any] | None = None
        family_gate_a_failure_excess = -1.0
        family_gate_a_argmax_key: dict[str, Any] | None = None
        family_gate_a_e2_value: float | None = None
        family_gate_a_reference_value: float | None = None
        family_gate_b_deltas: list[np.ndarray] = []
        family_gate_b_missing_count = 0
        family_gate_b_max_abs_delta = -1.0
        family_gate_b_argmax_key: dict[str, Any] | None = None
        family_gate_b_e2_value: float | None = None
        family_gate_b_reference_value: float | None = None
        family_gate_a_control_deltas: list[np.ndarray] = []
        for fmt in formats:
            unit_frames: list[pd.DataFrame] = []
            expected_row_count = 0
            for _unit_format, path in _result_units(
                source, family, (fmt,), primary_records=primary_records
            ):
                unit_frame, unit_expected, consistency_metrics = _compute_unit(
                    loaded,
                    source=source,
                    fmt=fmt,
                    fewshot_set=(
                        int(cfg["pl"]["primary"]["fewshot_set"])
                        if source == "primary" and fmt != "cloze"
                        else None
                    ),
                    parquet_path=path,
                    concepts=concepts,
                    token_sets=token_sets,
                )
                unit_frames.append(unit_frame)
                expected_row_count += unit_expected
                gate_a = consistency_metrics["gate_a"]
                gate_b = consistency_metrics["gate_b"]
                gate_a_control = consistency_metrics["gate_a_control"]
                family_gate_a_matched_count += int(gate_a["matched_count"])
                family_gate_a_prompt_layer_count += int(gate_a["prompt_layer_count"])
                family_gate_a_missing_prompt_layers.extend(gate_a["missing_prompt_layers"])
                family_gate_a_failed_count += int(gate_a["failed_count"])
                family_gate_a_count_over_1e4 += int(gate_a["count_over_1e-4"])
                relative_delta = gate_a["max_relative_delta"]
                if relative_delta is not None:
                    family_gate_a_max_relative_delta = max(
                        family_gate_a_max_relative_delta, float(relative_delta)
                    )
                family_gate_a_largest_cases.extend(gate_a["largest_cases"])
                family_gate_a_largest_cases.sort(
                    key=lambda entry: (
                        -float(entry["abs_delta"]),
                        json.dumps(entry["key"], sort_keys=True),
                    )
                )
                del family_gate_a_largest_cases[5:]
                if gate_a["failure_case"] is not None:
                    excess = float(gate_a["failure_case"]["tolerance_excess"])
                    if excess > family_gate_a_failure_excess:
                        family_gate_a_failure_excess = excess
                        family_gate_a_failure_case = gate_a["failure_case"]
                if gate_a["max_abs_delta"] is not None and float(gate_a["max_abs_delta"]) > family_gate_a_max_abs_delta:
                    family_gate_a_max_abs_delta = float(gate_a["max_abs_delta"])
                    family_gate_a_argmax_key = gate_a["argmax_key"]
                    family_gate_a_e2_value = gate_a["e2_value"]
                    family_gate_a_reference_value = gate_a["reference_value"]
                gate_b_deltas = np.asarray(consistency_metrics["gate_b_deltas"], dtype=np.float64)
                control_deltas = np.asarray(
                    consistency_metrics["gate_a_control_deltas"], dtype=np.float64
                )
                if source == "trial" and control_deltas.size:
                    family_gate_a_control_deltas.append(control_deltas)
                family_gate_b_missing_count += int(gate_b["missing_count"])
                if gate_b_deltas.size:
                    family_gate_b_deltas.append(gate_b_deltas)
                if (
                    gate_b["argmax_key"] is not None
                    and (
                        family_gate_b_argmax_key is None
                        or float(gate_b["max_abs_delta"]) > family_gate_b_max_abs_delta
                    )
                ):
                    family_gate_b_max_abs_delta = float(gate_b["max_abs_delta"])
                    family_gate_b_argmax_key = gate_b["argmax_key"]
                    family_gate_b_e2_value = gate_b["e2_value"]
                    family_gate_b_reference_value = gate_b["reference_value"]
                _LOGGER.info(
                    "E2 unit checked: source=%s family=%s format=%s input=%s rows=%s expected=%s "
                    "gate_a_matched=%s gate_a_failed=%s gate_a_count_over_1e-4=%s "
                    "gate_a_prompt_layers=%s "
                    "gate_a_missing_prompt_layers=%s gate_a_max_abs_delta=%s "
                    "gate_a_max_relative_delta=%s gate_a_largest_cases=%s "
                    "gate_a_control_matched=%s gate_a_control_median=%s "
                    "gate_a_control_p05=%s gate_a_control_min=%s "
                    "gate_b_median_abs_delta=%s gate_b_p95_abs_delta=%s "
                    "gate_b_p99_abs_delta=%s gate_b_max_abs_delta=%s gate_b_argmax_key=%s",
                    source,
                    family,
                    fmt,
                    path.name,
                    len(unit_frame),
                    unit_expected,
                    gate_a["matched_count"],
                    gate_a["failed_count"],
                    gate_a["count_over_1e-4"],
                    gate_a["prompt_layer_count"],
                    len(gate_a["missing_prompt_layers"]),
                    gate_a["max_abs_delta"],
                    gate_a["max_relative_delta"],
                    json.dumps(gate_a["largest_cases"], sort_keys=True),
                    gate_a_control["matched_count"] if source == "trial" else "NOT_RUN",
                    gate_a_control["median_abs_delta"] if source == "trial" else "NOT_RUN",
                    gate_a_control["p05_abs_delta"] if source == "trial" else "NOT_RUN",
                    gate_a_control["min_abs_delta"] if source == "trial" else "NOT_RUN",
                    gate_b["median_abs_delta"],
                    gate_b["p95_abs_delta"],
                    gate_b["p99_abs_delta"],
                    gate_b["max_abs_delta"],
                    json.dumps(gate_b["argmax_key"], sort_keys=True),
                )
            frame = (
                pd.concat(unit_frames, ignore_index=True)
                if unit_frames
                else pd.DataFrame(columns=io.LENS_COLUMNS + ["m1_extension"])
            )
            validate_probability_values(
                frame["value"].to_numpy(dtype=np.float64),
                expected_rows=expected_row_count,
                observed_rows=len(frame),
            )
            if len(frame) != expected_row_count:
                raise AssertionError(
                    f"{family}/{fmt} has {len(frame)} M1 rows, expected {expected_row_count}"
                )
            frame = frame[io.LENS_COLUMNS].sort_values(
                ["format", "concept_id", "target_lang", "target_variant", "layer"], kind="stable"
            )
            model_id = loaded.spec.hf_id if loaded.spec else loaded.family
            revision = loaded.spec.revision if loaded.spec else "synthetic"
            unit_key = {"family": family, "format": fmt, "pass": "m1", "source": source}
            provenance = build_provenance(
                cfg,
                repo_root=REPO,
                unit_key=unit_key,
                entrypoint=Path(__file__),
                model_id=model_id,
                model_revision=revision,
                dtype="torch.bfloat16",
                batch_size=int(cfg["inference"]["batch_size"][family]),
                expected_row_count=expected_row_count,
                run_git=run_git,
                model_loads=[
                    {
                        "model_id": model_id,
                        "model_revision": revision,
                        "dtype": "torch.bfloat16",
                        "dtype_override": None,
                    }
                ],
            )
            filename = f"family={family}__format={fmt}__pass=m1__source={source}.parquet"
            complete, reason = io.unit_is_complete(
                output_root,
                unit_key,
                provenance,
                cfg["runtime"]["provenance_compare"],
                filename=filename,
            )
            if complete:
                _LOGGER.info("verified complete M1 unit %s/%s (%s rows)", family, fmt, expected_row_count)
            else:
                _LOGGER.info("write M1 unit %s/%s: %s", family, fmt, reason)
                io.write_unit(
                    frame,
                    output_root,
                    provenance,
                    columns=io.LENS_COLUMNS,
                    sort_by=["format", "concept_id", "target_lang", "target_variant", "layer"],
                    compression=cfg["runtime"]["parquet_compression"],
                    filename=filename,
                )
            unit_summaries.append(
                {
                    "family": family,
                    "format": fmt,
                    "rows": len(frame),
                    "expected_rows": expected_row_count,
                    "units": len(unit_frames),
                }
            )
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        family_gate_b_summary = summarize_report_only_lens_deltas(
            np.concatenate(family_gate_b_deltas)
            if family_gate_b_deltas
            else np.asarray([], dtype=np.float64)
        )
        family_gate_b_summary.update(
            missing_count=family_gate_b_missing_count,
            argmax_key=family_gate_b_argmax_key,
            e2_value=family_gate_b_e2_value,
            reference_value=family_gate_b_reference_value,
        )
        if source == "trial":
            family_gate_a_control_deltas_array = (
                np.concatenate(family_gate_a_control_deltas)
                if family_gate_a_control_deltas
                else np.asarray([], dtype=np.float64)
            )
            family_gate_a_control_summary = summarize_lens_control_deltas(
                family_gate_a_control_deltas_array
            )
            family_gate_a_control_passed = lens_control_passes(
                family_gate_a_control_summary,
                minimum_median=GATE_A_CONTROL_MINIMUM_MEDIAN,
            )
        else:
            family_gate_a_control_summary = {
                "matched_count": 0,
                "median_abs_delta": None,
                "p05_abs_delta": None,
                "min_abs_delta": None,
            }
            family_gate_a_control_passed = None
        family_gate_a_summary: dict[str, Any] = {
            "matched_count": family_gate_a_matched_count,
            "prompt_layer_count": family_gate_a_prompt_layer_count,
            "missing_prompt_layers": family_gate_a_missing_prompt_layers,
            "max_abs_delta": None if family_gate_a_max_abs_delta < 0.0 else family_gate_a_max_abs_delta,
            "max_relative_delta": (
                None
                if family_gate_a_max_relative_delta < 0.0
                else float(family_gate_a_max_relative_delta)
                if np.isfinite(family_gate_a_max_relative_delta)
                else "inf"
            ),
            "failed_count": family_gate_a_failed_count,
            "count_over_1e-4": family_gate_a_count_over_1e4,
            "largest_cases": family_gate_a_largest_cases,
            "failure_case": family_gate_a_failure_case,
            "argmax_key": family_gate_a_argmax_key,
            "e2_value": family_gate_a_e2_value,
            "reference_value": family_gate_a_reference_value,
        }
        gate_a_failed = (
            family_gate_a_prompt_layer_count == 0
            or bool(family_gate_a_missing_prompt_layers)
            or not absolute_delta_gate_passes(
                family_gate_a_summary["max_abs_delta"],
                tolerance=GATE_A_ABS_TOLERANCE,
            )
        )
        family_summary = {
            "family": family,
            "wall_time_seconds": time.perf_counter() - family_started,
            "gate_a": family_gate_a_summary,
            "gate_a_control": family_gate_a_control_summary,
            "gate_a_control_passed": family_gate_a_control_passed,
            "gate_b": family_gate_b_summary,
            "gate_status": (
                "REPORTED_NOT_GATED"
                if source != "trial"
                else "PASS"
                if not gate_a_failed and family_gate_a_control_passed
                else "FAIL"
            ),
        }
        family_summaries.append(family_summary)
        _LOGGER.info(
            "E2 consistency summary: source=%s family=%s wall_time_seconds=%.3f "
            "gate_a_matched=%s gate_a_failed=%s gate_a_count_over_1e-4=%s gate_a_prompt_layers=%s "
            "gate_a_missing_prompt_layers=%s gate_a_max_abs_delta=%s "
            "gate_a_max_relative_delta=%s gate_a_largest_cases=%s "
            "gate_a_control_matched=%s gate_a_control_median=%s gate_a_control_p05=%s "
            "gate_a_control_min=%s gate_a_control_status=%s "
            "gate_b_median_abs_delta=%s gate_b_p95_abs_delta=%s "
            "gate_b_p99_abs_delta=%s gate_b_max_abs_delta=%s gate_b_argmax_key=%s gate_status=%s",
            source,
            family,
            family_summary["wall_time_seconds"],
            family_gate_a_summary["matched_count"],
            family_gate_a_summary["failed_count"],
            family_gate_a_summary["count_over_1e-4"],
            family_gate_a_summary["prompt_layer_count"],
            len(family_gate_a_missing_prompt_layers),
            family_gate_a_summary["max_abs_delta"],
            family_gate_a_summary["max_relative_delta"],
            json.dumps(family_gate_a_summary["largest_cases"], sort_keys=True),
            family_gate_a_control_summary["matched_count"],
            family_gate_a_control_summary["median_abs_delta"],
            family_gate_a_control_summary["p05_abs_delta"],
            family_gate_a_control_summary["min_abs_delta"],
            (
                "NOT_RUN"
                if source != "trial"
                else "PASS" if family_gate_a_control_passed else "FAIL_NOT_DISCRIMINATIVE"
            ),
            family_gate_b_summary["median_abs_delta"],
            family_gate_b_summary["p95_abs_delta"],
            family_gate_b_summary["p99_abs_delta"],
            family_gate_b_summary["max_abs_delta"],
            json.dumps(family_gate_b_summary["argmax_key"], sort_keys=True),
            family_summary["gate_status"],
        )
        if source == "trial" and not family_gate_a_control_passed:
            _LOGGER.error(
                "Gate A-control is not discriminative: family=%s matched=%s median=%s "
                "p05=%s min=%s required_median_at_least=%s; cyclically shifted prompt rows "
                "must differ before any primary unit can run",
                family,
                family_gate_a_control_summary["matched_count"],
                family_gate_a_control_summary["median_abs_delta"],
                family_gate_a_control_summary["p05_abs_delta"],
                family_gate_a_control_summary["min_abs_delta"],
                GATE_A_CONTROL_MINIMUM_MEDIAN,
            )
        if source == "trial" and gate_a_failed:
            for missing_key in family_gate_a_missing_prompt_layers:
                _LOGGER.error(
                    "Gate A coverage failure: key=%s token_id=<none> e2_value=<none> "
                    "answer_topk_value=<none> matched_count=0",
                    json.dumps(missing_key, sort_keys=True),
                )
            if family_gate_a_failure_case is not None:
                _log_consistency_failure(
                    "Gate A",
                    key=family_gate_a_failure_case["key"],
                    e2_value=float(family_gate_a_failure_case["e2_value"]),
                    reference_value=float(family_gate_a_failure_case["reference_value"]),
                    delta=float(family_gate_a_failure_case["abs_delta"]),
                    tolerance=float(family_gate_a_failure_case["tolerance"]),
                )
            raise AssertionError(
                f"E2 trial Gate A failed for {family}: "
                f"Gate A max |Δ|={family_gate_a_summary['max_abs_delta']} exceeds "
                f"{GATE_A_ABS_TOLERANCE} "
                f"failed_count={family_gate_a_failed_count} "
                f"missing_prompt_layers={len(family_gate_a_missing_prompt_layers)}"
            )
        if source == "trial" and not family_gate_a_control_passed:
            raise AssertionError(
                f"E2 trial Gate A-control is not discriminative for {family}: "
                f"median |Δ|={family_gate_a_control_summary['median_abs_delta']} "
                f"required >= {GATE_A_CONTROL_MINIMUM_MEDIAN}"
            )
    return {
        "source": source,
        "units": unit_summaries,
        "families": family_summaries,
        "gate_a_matched_count": sum(
            int(item["gate_a"]["matched_count"]) for item in family_summaries
        ),
        "gate_a_failed_count": sum(
            int(item["gate_a"]["failed_count"]) for item in family_summaries
        ),
        "gate_a_max_abs_delta": max(
            (float(item["gate_a"]["max_abs_delta"]) for item in family_summaries
             if item["gate_a"]["max_abs_delta"] is not None),
            default=None,
        ),
        "gate_a_max_relative_delta": max(
            (
                float(item["gate_a"]["max_relative_delta"])
                for item in family_summaries
                if item["gate_a"]["max_relative_delta"] is not None
            ),
            default=None,
        ),
        "gate_a_by_family": {
            str(item["family"]): item["gate_a"] for item in family_summaries
        },
        "gate_a_control_by_family": {
            str(item["family"]): item["gate_a_control"] for item in family_summaries
        },
        "gate_b_by_family": {
            str(item["family"]): item["gate_b"] for item in family_summaries
        },
    }


def _run_source(cfg: dict[str, Any], source: str, run_git: Any, *, with_trial_check: bool) -> dict[str, Any]:
    """Run the full trial check before any primary unit, then score the requested source."""
    if source == "primary":
        if not with_trial_check:
            raise ValueError("--source primary requires --with-trial-check")
        trial_summary = _run_source_once(cfg, "trial", run_git)
        _LOGGER.info(
            "trial consistency gates passed before primary: %s",
            json.dumps(trial_summary["families"], sort_keys=True),
        )
        primary_summary = _run_source_once(cfg, "primary", run_git)
        _LOGGER.info(
            "primary consistency metrics (not gates): %s",
            json.dumps(primary_summary["families"], sort_keys=True),
        )
        return {"trial_check": trial_summary, "primary": primary_summary}
    return _run_source_once(cfg, source, run_git)


def _write_job_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Atomically persist a JSON run summary for Checkpoint-H collection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Parse job options, preflight without weights, or run M1 on a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("trial", "primary"), required=True)
    parser.add_argument("--with-trial-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.source == "primary" and not args.with_trial_check:
        parser.error("--source primary requires --with-trial-check")
    global _LOGGER
    _LOGGER = setup_logging("job_e2_m1", REPO / "logs")
    cfg = load_config(args.config)
    run_git = capture_run_git(REPO, cfg)
    checks = _preflight(cfg, args.source, gpu=not args.preflight_only)
    _LOGGER.info("preflight checks:\n%s", format_checks(checks))
    if args.preflight_only:
        return 0 if all(check.status not in {"FAIL", "BLOCKED"} for check in checks) else 2
    if not torch.cuda.is_available():
        parser.error("E2 M1 requires a GPU; use --preflight-only for a login-node check")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with StageTimer("E2 M1", _LOGGER):
        result = _run_source(cfg, args.source, run_git, with_trial_check=args.with_trial_check)
    wall_time = time.perf_counter() - started
    run_summary = {
        "source": args.source,
        "with_trial_check": bool(args.with_trial_check),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_time_seconds": wall_time,
        "gpu_hours": wall_time / 3600.0,
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "code_tree": dict(run_git.code_tree),
        "code_tree_sha256": run_git.code_tree_sha256,
        "result": result,
    }
    if args.source == "primary":
        trial_summary = dict(run_summary)
        trial_summary.update(
            source="trial",
            result=result["trial_check"],
        )
        _write_job_summary(
            REPO / "results/analysis/m1/trial/e2_job_summary.json",
            trial_summary,
        )
    _write_job_summary(
        REPO / "results/analysis/m1" / args.source / "e2_job_summary.json",
        run_summary,
    )
    _LOGGER.info(
        "final summary: job_id=%s gpu=%s peak_gpu_memory_bytes=%s wall_time_seconds=%.3f counts_and_checks=%s",
        os.environ.get("SLURM_JOB_ID", "login-node"),
        torch.cuda.get_device_name(0),
        torch.cuda.max_memory_allocated(),
        wall_time,
        json.dumps(result, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
