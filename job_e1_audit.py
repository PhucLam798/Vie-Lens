"""E1 model-side audit gate on the directions split only."""

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
from collections import Counter, defaultdict
import gc
import json
import logging
import math
import numbers
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from vilens import hooks, io, locks, models, prompts, score
from vilens.cli import base_parser, setup_logging
from vilens.config import load_config, study_models
from vilens.data import load_concepts_jsonl, load_prompt_file
from vilens.lens import LensHead, TFResult, teacher_forced
from vilens.manifest import verify_manifest
from vilens.passes import _pad_id, _truncate_generation
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import (
    RunGit,
    build as build_provenance,
    capture_run_git,
    e1_model_loads,
)
from vilens.tokenize import JointEncoding, encode_joint_batch

E1_CHECK_COLUMNS = ["family", "check_id", "metric", "layer", "value"]
COMPETENCE_COLUMNS = [
    "family", "format", "lang", "concept_id", "split", "answer", "class"
]
COMPETENCE_CLASSES = ("VI", "EN", "OTHER-LANG", "DAMAGE")


def _utc_now() -> str:
    """Return a compact UTC timestamp for run metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON-lines records in source order."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object atomically beside its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _measure_stage(
    name: str,
    logger: logging.Logger,
    timings: dict[str, float],
    peaks: dict[str, int],
    function: Callable[[], Any],
) -> Any:
    """Measure one synchronized CUDA stage and record allocated peak memory."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        return function()
    finally:
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        peaks[name] = int(torch.cuda.max_memory_allocated())
        logger.info(
            "stage=%s wall_seconds=%.3f peak_memory_bytes=%d",
            name,
            timings[name],
            peaks[name],
        )


def _pad_sequences(
    sequences: Sequence[Sequence[int]], pad_id: int, device: Any
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad encoded prompt sequences and return their attention mask."""
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths, default=0)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = lengths[row]
        input_ids[row, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask, lengths


def _encode_prompts(tokenizer: Any, text: Sequence[str]) -> list[tuple[int, ...]]:
    """Encode prompts with tokenizer-default special tokens, without text changes."""
    if not text:
        return []
    encoded = tokenizer(list(text), add_special_tokens=True)["input_ids"]
    rows = encoded if encoded and isinstance(encoded[0], list) else [encoded]
    result = [tuple(int(token_id) for token_id in row) for row in rows]
    if len(result) != len(text) or any(not row for row in result):
        raise ValueError("a prompt encoded to an empty token sequence or batch size changed")
    return result


def _expected_early_blocks(cfg: Mapping[str, Any], family: str) -> list[int]:
    """Expand the configured inclusive first/last early-block range."""
    bounds = cfg["e1"]["expected_early_blocks"][family]
    if len(bounds) != 2:
        raise ValueError(f"expected_early_blocks[{family}] must contain [first, last]")
    first, last = (int(value) for value in bounds)
    return list(range(first, last + 1))


def _expected_check_rows(cfg: Mapping[str, Any], family: str) -> int:
    """Compute row count by asking the same builder used for the result table."""
    reference = cfg["reference_facts"]["x0c"]
    dims = list(reference["model_dims"][family])
    precision_values = np.zeros((int(dims[0]) + 1, 1), dtype=np.float32)
    precision_rows, _ = _compare_precision(family, precision_values, precision_values)
    competence = {
        "by_format": {
            str(format_name): {
                "records": 0,
                "counts": {label: 0 for label in COMPETENCE_CLASSES},
                "rates": {label: 0.0 for label in COMPETENCE_CLASSES},
                "translation_vi_flag": False,
            }
            for format_name in cfg["e1"]["competence_formats"]
        }
    }
    rows = _build_check_rows(
        family,
        dims,
        list(reference["tokenizer_x_colon_ids"][family]),
        True,
        _expected_early_blocks(cfg, family),
        {
            "model_max_abs_logprob_delta": 0.0,
            "model_top1_agreement": 1.0,
            "fp32_max_abs_logprob_delta": 0.0,
            "fp32_top1_agreement": 1.0,
            "residual_bf16_model_dtype_max_rel_error": 0.0,
            "residual_bf16_fp32_resummation_reported_only_max_rel_error": 0.0,
            "residual_fp32_weights_max_rel_error": 0.0,
        },
        False,
        0,
        True,
        precision_rows,
        competence,
        cfg,
    )
    return len(rows)


def _current_unit_provenance(
    cfg: Mapping[str, Any], family: str, unit: str, expected_rows: int, run_git: RunGit
) -> dict[str, Any]:
    """Build pre-forward provenance for one E1 result unit."""
    selected = study_models(dict(cfg))[family]
    return build_provenance(
        cfg,
        repo_root=REPO,
        unit_key={"family": family, "pass": "e1", "unit": unit},
        entrypoint=Path(__file__),
        model_id=selected["hf_id"],
        model_revision=selected["revision"],
        dtype="torch.bfloat16",
        batch_size=int(cfg["inference"]["batch_size"][family]),
        expected_row_count=expected_rows,
        run_git=run_git,
        model_loads=e1_model_loads(selected["hf_id"], selected["revision"], unit),
    )


def _numeric_deltas(current: Any, previous: Any) -> list[float]:
    """Collect absolute differences from corresponding numeric summary leaves."""
    if isinstance(current, bool) or isinstance(previous, bool):
        return []
    if isinstance(current, numbers.Real) and isinstance(previous, numbers.Real):
        return [abs(float(current) - float(previous))]
    if isinstance(current, Mapping) and isinstance(previous, Mapping):
        return [
            difference
            for key in current.keys() & previous.keys()
            for difference in _numeric_deltas(current[key], previous[key])
        ]
    if isinstance(current, list) and isinstance(previous, list):
        return [
            difference
            for left, right in zip(current, previous)
            for difference in _numeric_deltas(left, right)
        ]
    return []


def _job_1512_reproduction(
    summary: Mapping[str, Any], previous_summary_path: Path
) -> dict[str, Any]:
    """Compare E1-2, E1-4, and E1-6 summary values with superseded job 1512."""
    previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    current_families = summary.get("families", {})
    previous_families = previous.get("families", {})
    for family, current_family in current_families.items():
        old_family = previous_families.get(family, {})
        family_comparisons: dict[str, Any] = {}
        for check_name in ("E1-2", "E1-4", "E1-6"):
            current_value = current_family.get(check_name)
            previous_value = old_family.get(check_name)
            deltas = _numeric_deltas(current_value, previous_value)
            family_comparisons[check_name] = {
                "equal": current_value == previous_value,
                "max_abs_delta": max(deltas) if deltas else None,
            }
        comparisons[str(family)] = family_comparisons
    return comparisons


def _unit_complete(
    family_dir: Path,
    unit_key: Mapping[str, str | int],
    current: Mapping[str, Any],
    cfg: Mapping[str, Any],
    filename: str,
) -> tuple[bool, str]:
    """Check a named E1 parquet unit and sidecar against current provenance."""
    return io.unit_is_complete(
        family_dir,
        unit_key,
        current,
        cfg["runtime"]["provenance_compare"],
        filename=filename,
    )


def _metric_row(
    family: str,
    check_id: str,
    metric: str,
    value: float,
    *,
    layer: int = 0,
) -> dict[str, Any]:
    """Create one numeric E1 check-table row."""
    return {
        "family": family,
        "check_id": check_id,
        "metric": metric,
        "layer": layer,
        "value": float(value),
    }


def _prompt_gate(
    loaded: models.LoadedModel,
    records: Sequence[Mapping[str, Any]],
    family: str,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[dict[str, float], int]:
    """Compute model/fp32 lens validity and residual identity on all prompts."""
    if not records:
        raise ValueError("E1 prompt gate requires directions records")
    model_lens = LensHead.from_loaded(loaded, "model")
    fp32_lens = LensHead.from_loaded(loaded, "fp32")
    prompts_text = [str(record["prompt"]) for record in records]
    sequences = _encode_prompts(loaded.tokenizer, prompts_text)
    device = next(loaded.model.parameters()).device
    pad_id = _pad_id(loaded)
    model_max_delta = 0.0
    fp32_max_delta = 0.0
    model_top1_equal = 0
    fp32_top1_equal = 0
    total = 0
    residual_model_dtype_max = 0.0
    residual_fp32_resummation_max = 0.0
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        batch_sequences = sequences[start:stop]
        input_ids, attention_mask, lengths = _pad_sequences(batch_sequences, pad_id, device)
        positions = torch.as_tensor(
            [[length - 1] for length in lengths], dtype=torch.long, device=device
        )
        recording = hooks.record(
            loaded,
            input_ids,
            attention_mask,
            positions=positions,
            components=True,
            keep_on="cpu",
        )
        final_hidden = recording.resid[-1, :, 0, :]
        model_logits = recording.logits[:, 0, :]
        lens_logits = model_lens.logits(final_hidden).detach().cpu()
        fp32_logits = fp32_lens.logits(final_hidden).detach().cpu()
        model_lps = functional.log_softmax(model_logits.float(), dim=-1)
        lens_lps = functional.log_softmax(lens_logits.float(), dim=-1)
        fp32_lps = functional.log_softmax(fp32_logits.float(), dim=-1)
        model_max_delta = max(model_max_delta, float((lens_lps - model_lps).abs().max().item()))
        fp32_max_delta = max(fp32_max_delta, float((fp32_lps - model_lps).abs().max().item()))
        model_top1_equal += int((lens_logits.argmax(-1) == model_logits.argmax(-1)).sum().item())
        fp32_top1_equal += int((fp32_logits.argmax(-1) == model_logits.argmax(-1)).sum().item())
        total += len(batch_sequences)
        if recording.attn is None or recording.mlp is None:
            raise RuntimeError("component hooks did not return attention and MLP activations")
        residual_model_dtype_max = max(
            residual_model_dtype_max,
            _residual_relative_error(recording, compute_dtype=None),
        )
        residual_fp32_resummation_max = max(
            residual_fp32_resummation_max,
            _residual_relative_error(recording, compute_dtype=torch.float32),
        )
    return (
        {
            "model_max_abs_logprob_delta": model_max_delta,
            "model_top1_agreement": model_top1_equal / total,
            "fp32_max_abs_logprob_delta": fp32_max_delta,
            "fp32_top1_agreement": fp32_top1_equal / total,
            "residual_bf16_model_dtype_max_rel_error": residual_model_dtype_max,
            "residual_bf16_fp32_resummation_reported_only_max_rel_error": residual_fp32_resummation_max,
            "records": float(total),
        },
        total,
    )


def _residual_relative_error(recording: hooks.Recording, *, compute_dtype: Any | None) -> float:
    """Return the maximum block residual error with additions in the requested dtype."""
    if recording.attn is None or recording.mlp is None:
        raise RuntimeError("component hooks did not return attention and MLP activations")
    previous = recording.resid[:-1, :, 0, :]
    following = recording.resid[1:, :, 0, :]
    attention = recording.attn[:, :, 0, :]
    mlp = recording.mlp[:, :, 0, :]
    if compute_dtype is not None:
        previous = previous.to(dtype=compute_dtype)
        following = following.to(dtype=compute_dtype)
        attention = attention.to(dtype=compute_dtype)
        mlp = mlp.to(dtype=compute_dtype)
    # Preserve the decoder's residual order: first attention, then MLP.
    reconstructed = (previous + attention) + mlp
    following_fp32 = following.float()
    relative = (following_fp32 - reconstructed.float()).norm(dim=-1) / following_fp32.norm(
        dim=-1
    ).clamp_min(torch.finfo(torch.float32).tiny)
    return float(relative.max().item())


def _prompt_residual_gate(
    loaded: models.LoadedModel,
    records: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> float:
    """Measure residual identity on every prompt for the float32-weights run."""
    if not records:
        raise ValueError("E1 residual gate requires directions records")
    sequences = _encode_prompts(loaded.tokenizer, [str(record["prompt"]) for record in records])
    device = next(loaded.model.parameters()).device
    pad_id = _pad_id(loaded)
    maximum = 0.0
    for start in range(0, len(records), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        input_ids, attention_mask, lengths = _pad_sequences(batch_sequences, pad_id, device)
        positions = torch.as_tensor(
            [[length - 1] for length in lengths], dtype=torch.long, device=device
        )
        recording = hooks.record(
            loaded,
            input_ids,
            attention_mask,
            positions=positions,
            components=True,
            keep_on="cpu",
        )
        maximum = max(
            maximum,
            _residual_relative_error(recording, compute_dtype=torch.float32),
        )
    return maximum


def _precision_encodings(
    loaded: models.LoadedModel, records: Sequence[Mapping[str, Any]]
) -> tuple[list[JointEncoding], np.ndarray]:
    """Teacher-force the correct Vietnamese answer for the frozen E0 sample."""
    encodings = encode_joint_batch(
        loaded.tokenizer,
        [str(record["prompt"]) for record in records],
        [str(record["target"]) for record in records],
        strict=True,
        prompt_cache={},
    )
    results = teacher_forced(
        loaded,
        LensHead.from_loaded(loaded, "fp32"),
        encodings,
        pad_id=_pad_id(loaded),
    )
    values = np.concatenate([result.lens_logprobs for result in results], axis=1)
    return encodings, values


def _compare_precision(
    family: str,
    bf16_values: np.ndarray,
    fp32_values: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize per-layer absolute fp32-weight versus bf16-weight log-probability deltas."""
    if bf16_values.shape != fp32_values.shape:
        raise ValueError(
            f"precision-check shapes differ: bf16={bf16_values.shape}, fp32={fp32_values.shape}"
        )
    deltas = np.abs(bf16_values - fp32_values)
    rows: list[dict[str, Any]] = []
    layers: list[dict[str, float | int]] = []
    for layer in range(deltas.shape[0]):
        maximum = float(deltas[layer].max(initial=0.0))
        mean = float(deltas[layer].mean()) if deltas.shape[1] else 0.0
        rows.extend(
            [
                _metric_row(family, "E1-5", "max_abs_logprob_delta", maximum, layer=layer),
                _metric_row(family, "E1-5", "mean_abs_logprob_delta", mean, layer=layer),
            ]
        )
        layers.append({"layer": layer, "max_abs_logprob_delta": maximum, "mean_abs_logprob_delta": mean})
    return rows, {"layers": layers, "token_count": int(deltas.shape[1])}


def _competence_preview(
    loaded: models.LoadedModel,
    records: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily generate and A10-classify all kept Vietnamese directions previews."""
    rows: list[dict[str, Any]] = []
    tokenizer = loaded.tokenizer
    device = next(loaded.model.parameters()).device
    maximum = int(cfg["e0"]["gen_max_new_tokens"])
    for record in records:
        concept_id = str(record["concept_id"])
        concept = concepts.get(concept_id)
        if concept is None:
            raise KeyError(f"No concept-table row for directions concept {concept_id}")
        prompt_ids = tokenizer(str(record["prompt"]), add_special_tokens=True)["input_ids"]
        input_ids = torch.as_tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = loaded.model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=maximum)
        generated_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
        raw_answer = tokenizer.decode(generated_ids)
        answer = _truncate_generation(raw_answer)
        format_name = str(record["format"])
        answer_class = score.classify(
            answer,
            concept,
            condition="diac",
            format=format_name,
            groups=groups,
            concepts=concepts,
        )
        rows.append(
            {
                "family": loaded.family,
                "format": format_name,
                "lang": str(record["lang"]),
                "concept_id": concept_id,
                "split": str(record["split"]),
                "answer": answer,
                "class": answer_class,
                "vi": str(concept["vi"]),
            }
        )
    by_format: dict[str, Any] = {}
    examples: dict[str, list[dict[str, str]]] = {}
    for format_name in cfg["e1"]["competence_formats"]:
        selected = [row for row in rows if row["format"] == format_name]
        counts = Counter(row["class"] for row in selected)
        number = len(selected)
        rates = {label: (counts[label] / number if number else 0.0) for label in COMPETENCE_CLASSES}
        by_format[format_name] = {
            "records": number,
            "counts": {label: int(counts[label]) for label in COMPETENCE_CLASSES},
            "rates": rates,
            "translation_vi_flag": bool(
                format_name == "translation"
                and rates["VI"] < float(cfg["e1"]["competence_flag_min_translation_vi"])
            ),
        }
        examples[format_name] = [
            {"vi": row["vi"], "answer": row["answer"], "class": row["class"]}
            for row in selected[:10]
        ]
    return rows, {"by_format": by_format, "examples": examples}


def _source_group_counts(
    concepts: Mapping[str, Mapping[str, Any]],
    translation_set1: Sequence[Mapping[str, Any]],
    expected_prompt_groups: int,
    expected_prompt_records: int,
) -> dict[str, int]:
    """Count Russian-source groups and their main-test set-1 translation records."""
    groups = score.source_groups(concepts)
    test_concepts: dict[str, Mapping[str, Any]] = {
        concept_id: concept
        for concept_id, concept in concepts.items()
        if concept.get("split") == "test"
    }
    test_groups: dict[str, set[str]] = defaultdict(set)
    for concept_id, concept in test_concepts.items():
        ru = concept.get("ru")
        if isinstance(ru, str) and ru.strip():
            key = unicodedata.normalize("NFC", ru).casefold().replace("ё", "е")
            test_groups[key].add(concept_id)
    eligible_test_ids = {
        concept_id
        for ids in test_groups.values()
        if len(ids) >= 2
        for concept_id in ids
    }
    grouped_records = sum(
        1
        for record in translation_set1
        if record.get("split") == "test"
        and record.get("m1_extension") is False
        and str(record.get("concept_id")) in eligible_test_ids
    )
    grouped_extension_records = sum(
        1
        for record in translation_set1
        if record.get("split") == "test"
        and record.get("m1_extension") is True
        and str(record.get("concept_id")) in eligible_test_ids
    )
    return {
        "source_group_count": len(groups),
        "concepts_in_source_groups": sum(len(ids) for ids in groups.values()),
        "translation_diac_set1_main_records_in_test_source_groups": grouped_records,
        "translation_diac_set1_extension_records_in_test_source_groups": grouped_extension_records,
        "x0c_prompt_duplicate_groups": expected_prompt_groups,
        "x0c_prompt_duplicate_records": expected_prompt_records,
        "difference_from_x0c_duplicate_records": grouped_records - expected_prompt_records,
        "difference_after_extension_adjustment": (
            grouped_records + grouped_extension_records - expected_prompt_records
        ),
    }


def _build_check_rows(
    family: str,
    dims: Sequence[int],
    x_colon_ids: Sequence[int],
    softcap_none: bool,
    early_blocks: Sequence[int],
    lens_metrics: Mapping[str, float],
    residual_pass: bool,
    vision_calls: int,
    paths_valid: bool,
    precision_rows: Sequence[Mapping[str, Any]],
    competence: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Assemble the numeric long E1 check table with stable row keys."""
    reference = cfg["reference_facts"]["x0c"]
    expected_dims = reference["model_dims"][family]
    expected_tokens = reference["tokenizer_x_colon_ids"][family]
    expected_early = _expected_early_blocks(cfg, family)
    e1_1_pass = (
        list(dims) == list(expected_dims)
        and list(x_colon_ids) == list(expected_tokens)
        and softcap_none
        and list(early_blocks) == list(expected_early)
    )
    e1_2_pass = (
        lens_metrics["model_max_abs_logprob_delta"]
        < float(cfg["e1"]["lens_gate_max_abs_logprob"])
        and lens_metrics["model_top1_agreement"]
        >= float(cfg["e1"]["lens_gate_top1_agreement"])
    )
    e1_4_pass = residual_pass
    rows: list[dict[str, Any]] = []
    for name, value in zip(("num_hidden_layers", "hidden_size", "vocab_size"), dims, strict=True):
        rows.append(_metric_row(family, "E1-1", name, value))
    for index, token_id in enumerate(x_colon_ids):
        rows.append(_metric_row(family, "E1-1", "tokenizer_x_colon_id", token_id, layer=index))
    rows.append(_metric_row(family, "E1-1", "softcap_is_none", int(softcap_none)))
    for index, block in enumerate(early_blocks):
        rows.append(_metric_row(family, "E1-1", "early_block", block, layer=index))
    rows.append(_metric_row(family, "E1-1", "gate_pass", int(e1_1_pass)))

    rows.extend(
        [
            _metric_row(family, "E1-2", "model_max_abs_logprob_delta", lens_metrics["model_max_abs_logprob_delta"], layer=dims[0]),
            _metric_row(family, "E1-2", "model_top1_agreement", lens_metrics["model_top1_agreement"], layer=dims[0]),
            _metric_row(family, "E1-2", "fp32_max_abs_logprob_delta", lens_metrics["fp32_max_abs_logprob_delta"], layer=dims[0]),
            _metric_row(family, "E1-2", "fp32_top1_agreement", lens_metrics["fp32_top1_agreement"], layer=dims[0]),
            _metric_row(family, "E1-2", "gate_pass", int(e1_2_pass), layer=dims[0]),
        ]
    )
    rows.extend(
        [
            _metric_row(family, "E1-3", "module_paths_resolved", int(paths_valid)),
            _metric_row(family, "E1-3", "text_layer_count", dims[0]),
            _metric_row(family, "E1-3", "vision_tower_calls", vision_calls),
        ]
    )
    rows.extend(
        [
            _metric_row(family, "E1-4", "bf16_model_dtype_max_residual_relative_error", lens_metrics["residual_bf16_model_dtype_max_rel_error"], layer=dims[0]),
            _metric_row(family, "E1-4", "fp32_weights_max_residual_relative_error", lens_metrics["residual_fp32_weights_max_rel_error"], layer=dims[0]),
            _metric_row(family, "E1-4", "bf16_fp32_resummation_reported_only_max_residual_relative_error", lens_metrics["residual_bf16_fp32_resummation_reported_only_max_rel_error"], layer=dims[0]),
            _metric_row(family, "E1-4", "gate_pass", int(e1_4_pass), layer=dims[0]),
        ]
    )
    rows.extend(dict(row) for row in precision_rows)
    for format_name in cfg["e1"]["competence_formats"]:
        format_stats = competence["by_format"][format_name]
        rows.append(
            _metric_row(family, "E1-6", f"{format_name}_records", format_stats["records"])
        )
        for label in COMPETENCE_CLASSES:
            rows.append(
                _metric_row(
                    family,
                    "E1-6",
                    f"{format_name}_{label}_count",
                    format_stats["counts"][label],
                )
            )
            rows.append(
                _metric_row(
                    family,
                    "E1-6",
                    f"{format_name}_{label}_rate",
                    format_stats["rates"][label],
                )
            )
        rows.append(
            _metric_row(
                family,
                "E1-6",
                f"{format_name}_translation_vi_flag",
                int(format_stats["translation_vi_flag"]),
            )
        )
    return rows


def _persist_family_units(
    family: str,
    family_dir: Path,
    check_rows: Sequence[Mapping[str, Any]],
    competence_rows: Sequence[Mapping[str, Any]],
    check_provenance: Mapping[str, Any],
    competence_provenance: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    write_checks: bool,
    write_competence: bool,
) -> None:
    """Write E1 check and competence units with provenance sidecars last."""
    family_dir.mkdir(parents=True, exist_ok=True)
    if write_checks:
        checks = pd.DataFrame(check_rows, columns=E1_CHECK_COLUMNS)
        io.write_unit(
            checks,
            family_dir,
            check_provenance,
            columns=E1_CHECK_COLUMNS,
            sort_by=["family", "check_id", "metric", "layer"],
            compression=str(cfg["runtime"]["parquet_compression"]),
            filename="e1_checks.parquet",
        )
    if write_competence:
        competence = pd.DataFrame(competence_rows, columns=COMPETENCE_COLUMNS)
        io.write_unit(
            competence,
            family_dir,
            competence_provenance,
            columns=COMPETENCE_COLUMNS,
            sort_by=["family", "format", "lang", "concept_id"],
            compression=str(cfg["runtime"]["parquet_compression"]),
            filename="competence.parquet",
        )


def _run(
    cfg: dict[str, Any], logger: logging.Logger, *, preflight_only: bool, run_git: RunGit
) -> int:
    """Run preflight, directions-only gates, precision comparison, and competence preview."""
    task_started = time.perf_counter()
    results_dir = REPO / str(cfg["e1"]["results_dir"])
    prompt_path = REPO / cfg["paths"]["prompts_dir"] / "directions_matched.jsonl"
    concepts_path = REPO / cfg["paths"]["concepts_jsonl"]
    prompt_records = load_prompt_file(prompt_path)
    kept, excluded = locks.pretag_filter(cfg, prompt_records)
    if len(kept) != 1315 or len(excluded) != 5:
        raise AssertionError(f"D12 counts differ: kept={len(kept)}, excluded={len(excluded)}")
    model_records = prompts.records_for_model(cfg, kept)
    competence_records = [
        record
        for record in model_records
        if record.get("lang") == "vi" and record.get("format") in cfg["e1"]["competence_formats"]
    ]
    expected_competence = {"repetition": 132, "translation": 127}
    observed_competence = dict(Counter(str(row["format"]) for row in competence_records))
    if observed_competence != expected_competence or len(competence_records) != 259:
        raise AssertionError(f"competence record counts differ: {observed_competence}")
    logger.info(
        "D12 record counts: kept=%d excluded=%d; competence=%d by_format=%s",
        len(model_records), len(excluded), len(competence_records), observed_competence,
    )

    input_paths = [
        str((Path(cfg["paths"]["prompts_dir"]) / "directions_matched.jsonl").as_posix()),
        str(Path(cfg["paths"]["concepts_jsonl"]).as_posix()),
    ]
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E1",
        gpu=not preflight_only,
        inputs=input_paths,
        families=list(cfg["e1"]["families"]),
        records=model_records,
        output_dirs=[Path(cfg["e1"]["results_dir"])],
        entrypoint=Path(__file__),
    )
    formatted = format_checks(checks)
    print(formatted)
    logger.info("preflight checks:\n%s", formatted)
    if not preflight_ok(checks):
        logger.error("E1 preflight failed; no model work was started")
        return 2
    if preflight_only:
        logger.info(
            "preflight-only counts: kept=%d; competence=%d (repetition=%d, translation=%d)",
            len(model_records), len(competence_records),
            observed_competence["repetition"], observed_competence["translation"],
        )
        return 0

    concepts_frame = load_concepts_jsonl(concepts_path)
    concepts: dict[str, dict[str, Any]] = {
        str(row["concept_id"]): row
        for row in concepts_frame.to_dict(orient="records")
    }
    source_groups = score.source_groups(concepts)
    translation_set1 = load_prompt_file(
        REPO / cfg["paths"]["prompts_dir"] / f"{cfg['data']['primary_prompt_files']['translation_diac']}.jsonl"
    )
    duplicate_reference = cfg["reference_facts"]["x0c"]["prompt_duplicates"][
        "translation_diac_per_set"
    ]
    group_counts = _source_group_counts(
        concepts,
        translation_set1,
        int(duplicate_reference["groups"]),
        int(duplicate_reference["records"]),
    )
    logger.info("A10 source-group report (not a gate): %s", json.dumps(group_counts, sort_keys=True))

    sample_manifest = _read_jsonl(REPO / cfg["paths"]["derived"] / "e0_pilot.jsonl")
    sample_manifest.sort(key=lambda row: int(row["rank"]))
    sample_ids = {str(row["concept_id"]) for row in sample_manifest[: int(cfg["e1"]["precision_check_concepts"])]}
    precision_records = [
        record
        for record in model_records
        if str(record["concept_id"]) in sample_ids
        and record.get("format") == "repetition"
        and record.get("lang") == "vi"
    ]
    if len(sample_ids) != int(cfg["e1"]["precision_check_concepts"]) or len(precision_records) != len(sample_ids):
        raise AssertionError(
            f"E0 precision sample expected {len(sample_ids)} repetition/vi records, got {len(precision_records)}"
        )
    if any(record.get("split") not in {"fit", "validate"} for record in precision_records):
        raise AssertionError("E1 precision-check sample contains a non-directions record")

    run_start = _utc_now()
    summary: dict[str, Any] = {
        "task": "T08 E1 directions-only audit",
        "utc_start": run_start,
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "record_counts": {
            "directions_kept": len(model_records),
            "directions_excluded": len(excluded),
            "competence_total": len(competence_records),
            "competence_by_format": observed_competence,
            "precision_sample": len(precision_records),
        },
        "source_group_report": group_counts,
        "families": {},
        "errors": [],
        "exit_code": 0,
    }
    exit_code = 0
    for family in cfg["e1"]["families"]:
        family = str(family)
        family_dir = results_dir / family
        n_layers = int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
        expected_check_rows = _expected_check_rows(cfg, family)
        check_key = {"family": family, "pass": "e1", "unit": "checks"}
        competence_key = {"family": family, "pass": "e1", "unit": "competence"}
        check_provenance = _current_unit_provenance(
            cfg, family, "checks", expected_check_rows, run_git
        )
        competence_provenance = _current_unit_provenance(
            cfg, family, "competence", len(competence_records), run_git
        )
        checks_complete, checks_reason = _unit_complete(
            family_dir, check_key, check_provenance, cfg, "e1_checks.parquet"
        )
        competence_complete, competence_reason = _unit_complete(
            family_dir, competence_key, competence_provenance, cfg, "competence.parquet"
        )
        if checks_complete and competence_complete:
            logger.info("family=%s skipped complete E1 units: checks and competence", family)
            try:
                cached = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
                cached_family = cached.get("families", {}).get(family)
                if not isinstance(cached_family, dict):
                    raise ValueError("cached summary has no family entry")
                summary["families"][family] = cached_family
                if not cached_family.get("gates_pass", False):
                    exit_code = 2
            except Exception as exc:
                summary["errors"].append({"family": family, "error": f"cached summary: {exc}"})
                exit_code = 2
            continue

        family_summary: dict[str, Any] = {
            "model_id": study_models(cfg)[family]["hf_id"],
            "model_revision": study_models(cfg)[family]["revision"],
            "batch_size": int(cfg["inference"]["batch_size"][family]),
            "timings_seconds": {},
            "peak_memory_bytes": {},
            "unit_resume": {
                "checks_complete": checks_complete,
                "checks_reason": checks_reason,
                "competence_complete": competence_complete,
                "competence_reason": competence_reason,
            },
        }
        summary["families"][family] = family_summary
        loaded: models.LoadedModel | None = None
        try:
            loaded = _measure_stage(
                "load_bf16",
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: models.load(cfg, family, device="cuda"),
            )
            dims = models.text_dims(loaded.model.config)
            x_colon_ids = loaded.tokenizer("x:")["input_ids"]
            softcap_none = loaded.softcap is None
            early_blocks = [
                block
                for block in range(1, loaded.n_layers + 1)
                if block / loaded.n_layers <= float(cfg["e1"]["early_depth_max"])
            ]
            expected_dims = list(cfg["reference_facts"]["x0c"]["model_dims"][family])
            expected_x_colon = list(cfg["reference_facts"]["x0c"]["tokenizer_x_colon_ids"][family])
            expected_early = _expected_early_blocks(cfg, family)
            e1_1_pass = (
                list(dims) == expected_dims
                and list(x_colon_ids) == expected_x_colon
                and softcap_none
                and early_blocks == expected_early
            )
            family_summary["E1-1"] = {
                "text_dims": list(dims),
                "expected_text_dims": expected_dims,
                "tokenizer_x_colon_ids": list(x_colon_ids),
                "expected_tokenizer_x_colon_ids": expected_x_colon,
                "softcap": loaded.softcap,
                "early_blocks": early_blocks,
                "expected_early_blocks": expected_early,
                "pass": e1_1_pass,
            }

            vision_calls = -1
            vision_handle = None
            if family == "gemma":
                vision_tower = loaded.model.model.vision_tower
                calls = {"count": 0}

                def count_vision_call(_module: Any, _args: Any, _output: Any) -> None:
                    """Count forbidden vision-tower invocations during the prompt-only gate."""
                    calls["count"] += 1

                vision_handle = vision_tower.register_forward_hook(count_vision_call)
            try:
                lens_metrics, prompt_count = _measure_stage(
                    "prompt_lens_and_residual_gates",
                    logger,
                    family_summary["timings_seconds"],
                    family_summary["peak_memory_bytes"],
                    lambda: _prompt_gate(
                        loaded,
                        model_records,
                        family,
                        int(cfg["inference"]["batch_size"][family]),
                        logger,
                    ),
                )
            finally:
                if vision_handle is not None:
                    vision_handle.remove()
                    vision_calls = int(calls["count"])
            e1_2_pass = (
                lens_metrics["model_max_abs_logprob_delta"]
                < float(cfg["e1"]["lens_gate_max_abs_logprob"])
                and lens_metrics["model_top1_agreement"]
                >= float(cfg["e1"]["lens_gate_top1_agreement"])
            )
            module_paths_expected = cfg["inference"]["module_paths"][family]
            paths_valid = loaded.module_paths == module_paths_expected and len(loaded.layers) == dims[0]
            e1_3_pass = paths_valid and (family != "gemma" or (len(loaded.layers) == 34 and vision_calls == 0))
            family_summary["E1-2"] = {**lens_metrics, "records": prompt_count, "pass": e1_2_pass}
            family_summary["E1-3"] = {
                "module_paths": loaded.module_paths,
                "text_layer_count": len(loaded.layers),
                "vision_tower_calls": vision_calls,
                "pass": e1_3_pass,
            }
            bf16_encodings, bf16_precision = _measure_stage(
                "precision_teacher_forced_bf16",
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: _precision_encodings(loaded, precision_records),
            )
            bf16_dtype = str(next(loaded.model.parameters()).dtype)
            competence_rows, competence_summary = _measure_stage(
                "competence_generation",
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: _competence_preview(
                    loaded, competence_records, concepts, source_groups, cfg
                ),
            )
            family_summary["E1-6"] = competence_summary
            for format_name, details in competence_summary["by_format"].items():
                logger.info(
                    "competence family=%s format=%s stats=%s examples=%s",
                    family,
                    format_name,
                    json.dumps(details, sort_keys=True),
                    json.dumps(competence_summary["examples"][format_name], ensure_ascii=False),
                )

            del loaded
            loaded = None
            gc.collect()
            torch.cuda.empty_cache()
            float_model = _measure_stage(
                "load_fp32_weights",
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: models.load(cfg, family, device="cuda", dtype_override=torch.float32),
            )
            fp32_dtype = str(next(float_model.model.parameters()).dtype)
            try:
                fp32_residual_error = _measure_stage(
                    "prompt_residual_gate_fp32_weights",
                    logger,
                    family_summary["timings_seconds"],
                    family_summary["peak_memory_bytes"],
                    lambda: _prompt_residual_gate(
                        float_model,
                        model_records,
                        int(cfg["inference"]["batch_size"][family]),
                    ),
                )
                lens_metrics["residual_fp32_weights_max_rel_error"] = fp32_residual_error
                residual_threshold = float(cfg["e1"]["residual_gate_max_rel_error"])
                e1_4_pass = (
                    lens_metrics["residual_bf16_model_dtype_max_rel_error"] < residual_threshold
                    and fp32_residual_error < residual_threshold
                )
                family_summary["E1-4"] = {
                    "bf16_model_dtype_max_residual_relative_error": lens_metrics[
                        "residual_bf16_model_dtype_max_rel_error"
                    ],
                    "fp32_weights_max_residual_relative_error": fp32_residual_error,
                    "bf16_fp32_resummation_reported_only_max_residual_relative_error": lens_metrics[
                        "residual_bf16_fp32_resummation_reported_only_max_rel_error"
                    ],
                    "threshold": residual_threshold,
                    "pass": e1_4_pass,
                }
                fp32_precision = _measure_stage(
                    "precision_teacher_forced_fp32",
                    logger,
                    family_summary["timings_seconds"],
                    family_summary["peak_memory_bytes"],
                    lambda: np.concatenate(
                        [
                            result.lens_logprobs
                            for result in teacher_forced(
                                float_model,
                                LensHead.from_loaded(float_model, "fp32"),
                                bf16_encodings,
                                pad_id=_pad_id(float_model),
                            )
                        ],
                        axis=1,
                    ),
                )
            finally:
                del float_model
                gc.collect()
                torch.cuda.empty_cache()
            precision_rows, precision_summary = _compare_precision(
                family, bf16_precision, fp32_precision
            )
            precision_summary["bf16_dtype"] = bf16_dtype
            precision_summary["fp32_dtype"] = fp32_dtype
            precision_summary["fp32_dtype_override"] = True
            family_summary["E1-5"] = precision_summary

            check_rows = _build_check_rows(
                family,
                dims,
                x_colon_ids,
                softcap_none,
                early_blocks,
                lens_metrics,
                e1_4_pass,
                vision_calls,
                paths_valid,
                precision_rows,
                competence_summary,
                cfg,
            )
            if len(check_rows) != expected_check_rows:
                raise AssertionError(
                    f"E1 check table has {len(check_rows)} rows; expected {expected_check_rows}"
                )
            _persist_family_units(
                family,
                family_dir,
                check_rows,
                competence_rows,
                check_provenance,
                competence_provenance,
                cfg,
                write_checks=not checks_complete,
                write_competence=not competence_complete,
            )
            family_summary["gates_pass"] = bool(e1_1_pass and e1_2_pass and e1_3_pass and e1_4_pass)
            if not family_summary["gates_pass"]:
                exit_code = 2
        except Exception as exc:
            logger.error("family=%s failed: %s: %s", family, type(exc).__name__, exc)
            logger.error("%s", traceback.format_exc())
            family_summary["error"] = f"{type(exc).__name__}: {exc}"
            family_summary["gates_pass"] = False
            summary["errors"].append({"family": family, "error": family_summary["error"]})
            exit_code = 2
        finally:
            if loaded is not None:
                del loaded
                gc.collect()
                torch.cuda.empty_cache()

    try:
        summary["reproduces_job_1512"] = _job_1512_reproduction(
            summary, REPO / "results/e1/summary.json"
        )
    except Exception as exc:
        logger.warning("could not compare with job 1512 summary: %s: %s", type(exc).__name__, exc)
        summary["reproduces_job_1512"] = {
            family: {"error": f"{type(exc).__name__}: {exc}"}
            for family in cfg["e1"]["families"]
        }
    summary["utc_end"] = _utc_now()
    summary["total_runtime_seconds"] = time.perf_counter() - task_started
    summary["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary["peak_memory_bytes"] = max(
        (
            peak
            for family_summary in summary["families"].values()
            for peak in family_summary.get("peak_memory_bytes", {}).values()
        ),
        default=0,
    )
    summary["exit_code"] = exit_code
    _atomic_json(results_dir / "summary.json", summary)
    logger.info(
        "E1 final summary: exit_code=%d total_runtime_seconds=%.3f gpu=%s peak_memory_bytes=%d",
        exit_code,
        summary["total_runtime_seconds"],
        summary["gpu_name"],
        summary["peak_memory_bytes"],
    )
    logger.info("E1 summary: %s", json.dumps(summary, sort_keys=True, ensure_ascii=False, default=str))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the shared CLI flags and run E1 or preflight-only mode."""
    parser = base_parser("E1 model-side audit gate on directions prompts")
    args = parser.parse_args(argv)
    cfg = load_config(REPO / args.config)
    run_git = capture_run_git(REPO, cfg)
    logger = setup_logging("e1_audit", REPO / "logs")
    return _run(cfg, logger, preflight_only=bool(args.preflight_only), run_git=run_git)


if __name__ == "__main__":
    raise SystemExit(main())
