"""E0 directions-only pilot for reusable P-D and P-L passes."""

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
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from vilens import hooks, io, locks, models, prompts
from vilens.cli import StageTimer, base_parser, setup_logging
from vilens.config import load_config
from vilens.lens import LensHead, teacher_forced
from vilens.manifest import verify_manifest
from vilens.passes import (
    Item,
    _pad_id,
    _pad_sequences,
    build_items,
    run_pd_resume_safe,
    run_pl_resume_safe,
)
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import RunGit, capture_run_git
from vilens.tokenize import encode_joint_batch


def _utc_now() -> str:
    """Return a compact UTC timestamp for run metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file into ordered mapping records."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sample_records(
    records: Sequence[Mapping[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep directions records whose concepts belong to the frozen pilot sample."""
    return [dict(record) for record in records if str(record.get("concept_id")) in selected_ids]


def _sample_ids(repo_root: Path, cfg: Mapping[str, Any], count: int) -> list[str]:
    """Return the first ranked E0 concepts from the frozen pilot manifest."""
    rows = _read_jsonl(repo_root / cfg["paths"]["derived"] / "e0_pilot.jsonl")
    return [str(row["concept_id"]) for row in sorted(rows, key=lambda row: int(row["rank"]))[:count]]


def _correct_encodings(loaded: models.LoadedModel, items: Sequence[Item]) -> list[Any]:
    """Jointly encode each item's correct prompt-target sequence."""
    return encode_joint_batch(
        loaded.tokenizer,
        [item.prompt for item in items],
        [item.target for item in items],
        strict=True,
        prompt_cache={},
    )


def _path_checks(
    loaded: models.LoadedModel,
    model_lens: LensHead,
    fp32_lens: LensHead,
    items: Sequence[Item],
) -> dict[str, Any]:
    """Measure model-lens agreement, residual identity, and analysis-lens difference."""
    encodings = _correct_encodings(loaded, items)
    if not encodings:
        raise ValueError("No P-L items are available for path checks")
    device = next(loaded.model.parameters()).device
    sequences = [encoding.joint_ids for encoding in encodings]
    pad_id = _pad_id(loaded)
    # Each row's retained positions are exactly the autoregressive target-prediction positions.
    target_lengths = [len(encoding.target_ids) for encoding in encodings]
    max_targets = max(target_lengths)
    positions = torch.empty((len(encodings), max_targets), dtype=torch.long)
    for row, (encoding, count) in enumerate(zip(encodings, target_lengths, strict=True)):
        row_positions = [len(encoding.prompt_ids) - 1 + offset for offset in range(count)]
        positions[row, :count] = torch.tensor(row_positions, dtype=torch.long)
        positions[row, count:] = row_positions[0]
    input_ids, attention_mask, _ = _pad_sequences(sequences, pad_id, device)
    recording = hooks.record(
        loaded,
        input_ids,
        attention_mask,
        positions=positions.to(device),
        components=True,
        keep_on="cpu",
    )
    if recording.attn is None or recording.mlp is None:
        raise RuntimeError("Path check did not capture residual components")
    max_model_delta = 0.0
    max_fp32_delta = 0.0
    top1_agreement = 1.0
    max_residual_error = 0.0
    for row, count in enumerate(target_lengths):
        expected_logits = recording.logits[row, :count].float()
        block_n = recording.resid[loaded.n_layers, row, :count]
        model_logits = model_lens.logits(block_n).float().cpu()
        fp32_logits = fp32_lens.logits(block_n).float().cpu()
        expected_logprobs = torch.log_softmax(expected_logits, dim=-1)
        model_logprobs = torch.log_softmax(model_logits, dim=-1)
        fp32_logprobs = torch.log_softmax(fp32_logits, dim=-1)
        max_model_delta = max(max_model_delta, float((model_logprobs - expected_logprobs).abs().max()))
        max_fp32_delta = max(max_fp32_delta, float((fp32_logprobs - expected_logprobs).abs().max()))
        top1_agreement = min(
            top1_agreement,
            float((model_logits.argmax(-1) == expected_logits.argmax(-1)).float().mean()),
        )
        for layer in range(loaded.n_layers):
            resid = recording.resid[layer + 1, row, :count]
            reconstructed = (
                recording.resid[layer, row, :count]
                + recording.attn[layer, row, :count]
                + recording.mlp[layer, row, :count]
            )
            denominator = torch.linalg.vector_norm(resid).clamp_min(torch.finfo(torch.float32).tiny)
            error = float(torch.linalg.vector_norm(resid - reconstructed) / denominator)
            max_residual_error = max(max_residual_error, error)
    return {
        "model_lens_max_abs_logprob_delta": max_model_delta,
        "model_lens_top1_agreement": top1_agreement,
        "residual_identity_max_relative_error": max_residual_error,
        "fp32_lens_max_abs_logprob_delta": max_fp32_delta,
    }


def _measure_stage(
    name: str,
    logger: logging.Logger,
    timings: dict[str, float],
    memory: dict[str, int | None],
    function: Callable[[], Any],
) -> Any:
    """Run one stage while recording wall time and peak CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timer = StageTimer(name, logger, timings)
    try:
        with timer:
            return function()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory[name] = int(torch.cuda.max_memory_allocated())
        else:
            memory[name] = None


def _throughput_sweep(
    loaded: models.LoadedModel,
    lens: LensHead,
    items: Sequence[Item],
    batch_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    """Teacher-force the first 64 correct sequences at increasing batch sizes."""
    encodings = _correct_encodings(loaded, items[:64])
    pad_id = _pad_id(loaded)
    sweep: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        started = time.perf_counter()
        sequences = 0
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for start in range(0, len(encodings), int(batch_size)):
                teacher_forced(
                    loaded,
                    lens,
                    encodings[start : start + int(batch_size)],
                    pad_id=pad_id,
                )
                sequences += len(encodings[start : start + int(batch_size)])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - started, 1e-9)
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            sweep.append(
                {
                    "batch_size": int(batch_size),
                    "status": "OK",
                    "sequences": sequences,
                    "sequences_per_second": sequences / elapsed,
                    "wall_seconds": elapsed,
                    "peak_memory_bytes": peak,
                }
            )
        except Exception as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
            )
            if not is_oom:
                raise
            sweep.append(
                {
                    "batch_size": int(batch_size),
                    "status": "OOM",
                    "sequences": sequences,
                    "sequences_per_second": None,
                    "wall_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            torch.cuda.empty_cache()
            break
    return sweep


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _compare_table(left_path: Path, right_path: Path) -> float:
    """Compare two long parquet units by all keys and return maximum value delta."""
    left = pd.read_parquet(left_path)
    right = pd.read_parquet(right_path)
    keys = [column for column in io.LENS_COLUMNS if column != "value"]
    if len(left) != len(right):
        raise AssertionError(f"unit row counts differ: {len(left)} != {len(right)}")
    joined = left.merge(right, on=keys, how="outer", suffixes=("_left", "_right"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise AssertionError("result unit keys differ between repeated runs")
    if joined.empty:
        return 0.0
    return float((joined["value_left"] - joined["value_right"]).abs().max())


def _compare_family_outputs(
    loaded: models.LoadedModel,
    cfg: Mapping[str, Any],
    first_dir: Path,
    rerun_dir: Path,
) -> dict[str, Any]:
    """Compare persisted P-D/P-L tables, arrays, and generation records across runs."""
    from vilens.io import unit_paths

    family = loaded.family
    values_delta = 0.0
    arrays_delta = 0.0
    generations_identical = True
    for pass_name in ("pd", "pl"):
        formats = {"repetition", "translation"}
        for format_name in sorted(formats):
            key = {"family": family, "pass": pass_name, "format": format_name}
            left_path, _ = unit_paths(first_dir, key)
            right_path, _ = unit_paths(rerun_dir, key)
            if not left_path.is_file() or not right_path.is_file():
                continue
            values_delta = max(values_delta, _compare_table(left_path, right_path))
            extra_suffixes = (".resid.npy",) if pass_name == "pd" else (
                ".resid.npy", ".attn.npy", ".mlp.npy"
            )
            for suffix in extra_suffixes:
                left_array = np.load(left_path.with_suffix(suffix), allow_pickle=False)
                right_array = np.load(right_path.with_suffix(suffix), allow_pickle=False)
                if left_array.shape != right_array.shape:
                    raise AssertionError(f"array shape differs for {left_path.name}{suffix}")
                if left_array.size:
                    arrays_delta = max(arrays_delta, float(np.max(np.abs(left_array - right_array))))
            if pass_name == "pl":
                left_gen = left_path.with_suffix(".generations.jsonl")
                right_gen = right_path.with_suffix(".generations.jsonl")
                generations_identical = generations_identical and left_gen.read_bytes() == right_gen.read_bytes()
    return {
        "max_logprob_value_delta": values_delta,
        "max_residual_array_delta": arrays_delta,
        "generations_identical": generations_identical,
        "rerun_pass": values_delta <= float(cfg["e0"]["rerun_tolerance"])
        and arrays_delta <= float(cfg["e0"]["rerun_tolerance"])
        and generations_identical,
    }


def _family_record_sets(
    cfg: Mapping[str, Any],
    all_records: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Prepare locked P-D/P-L direction records for each pilot family."""
    result: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    ordered_families = [str(cfg["e0"]["main_family"]), *map(str, cfg["e0"]["smoke_families"])]
    for family in ordered_families:
        kept, _excluded = locks.pretag_filter(cfg, all_records)
        pd_records = [dict(row) for row in kept]
        pl_records = _sample_records(kept, set(sample_ids))
        if family != cfg["e0"]["main_family"]:
            smoke_ids = _sample_ids(REPO, cfg, int(cfg["e0"]["smoke_concepts"]))
            pd_records = _sample_records(kept, set(smoke_ids))
            pl_records = _sample_records(kept, set(smoke_ids))
        result[family] = (
            prompts.records_for_model(cfg, pd_records),
            prompts.records_for_model(cfg, pl_records),
        )
    return result


def _run_job(cfg: dict[str, Any], logger: logging.Logger, run_git: RunGit) -> int:
    """Run the real-model E0 pilot and always save a final summary."""
    overall_start = time.perf_counter()
    prompt_path = REPO / cfg["paths"]["prompts_dir"] / "directions_matched.jsonl"
    direction_records = _read_jsonl(prompt_path)
    kept, excluded = locks.pretag_filter(cfg, direction_records)
    if len(kept) != 1315 or len(excluded) != 5:
        logger.error("D12 record counts differ: kept=%d excluded=%d", len(kept), len(excluded))
        return 2
    # This accessor is mandatory after D12 filtering; it is the model-facing lock boundary.
    kept = prompts.records_for_model(cfg, kept)
    sample_ids = _sample_ids(REPO, cfg, int(cfg["manifests"]["e0_sample_size"]))
    if len(sample_ids) != 50:
        logger.error("E0 sample contains %d concepts, expected 50", len(sample_ids))
        return 2
    family_sets = _family_record_sets(cfg, kept, sample_ids)
    expected_counts: dict[str, dict[str, int]] = {}
    for family, (pd_records, pl_records) in family_sets.items():
        expected_pd = 1315 if family == cfg["e0"]["main_family"] else 100
        expected_pl = 497 if family == cfg["e0"]["main_family"] else 100
        expected_counts[family] = {"pd": len(pd_records), "pl": len(pl_records)}
        if len(pd_records) != expected_pd or len(pl_records) != expected_pl:
            logger.error(
                "record count mismatch %s: P-D=%d/%d P-L=%d/%d",
                family, len(pd_records), expected_pd, len(pl_records), expected_pl,
            )
            return 2

    targets = _read_jsonl(REPO / cfg["paths"]["derived"] / "target_forms.jsonl")
    distractors = _read_jsonl(REPO / cfg["paths"]["derived"] / "e0_distractors.jsonl")
    results_dir = REPO / cfg["e0"]["results_dir"]
    rerun_dir = REPO / cfg["e0"]["rerun_dir"]
    rerun_dir.mkdir(parents=True, exist_ok=True)
    if any(rerun_dir.iterdir()):
        logger.error("rerun directory is not empty; refusing to overwrite fresh rerun units: %s", rerun_dir)
        return 2

    summary: dict[str, Any] = {
        "utc_start": _utc_now(),
        "gpu_name": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "record_counts": expected_counts,
        "families": {},
        "errors": [],
    }
    exit_code = 0
    for family, (pd_records, pl_records) in family_sets.items():
        family_summary: dict[str, Any] = {
            "record_counts": {"pd": len(pd_records), "pl": len(pl_records)},
            "unit_counts": {},
            "timings_seconds": {},
            "peak_memory_bytes": {},
        }
        summary["families"][family] = family_summary
        loaded: models.LoadedModel | None = None
        try:
            loaded = _measure_stage(
                "load_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: models.load(cfg, family, device="cuda"),
            )
            fp32_lens = LensHead.from_loaded(loaded, "fp32")
            model_lens = LensHead.from_loaded(loaded, "model")
            pl_items = build_items(cfg, pl_records, targets, distractors, family)
            if len(pl_items) != len(pl_records):
                raise AssertionError(f"{family}: expected one directions item per record")

            path_stats = _measure_stage(
                "path_checks_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: _path_checks(loaded, model_lens, fp32_lens, pl_items[: int(cfg["e0"]["batch_size"])]),
            )
            family_summary["path_checks"] = path_stats
            sweep = _measure_stage(
                "throughput_sweep_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: _throughput_sweep(loaded, fp32_lens, pl_items, cfg["e0"]["sweep_batch_sizes"]),
            )
            family_summary["sweep"] = sweep

            pd_index, pd_resid, pd_skipped, pd_total, pd_expected_rows = _measure_stage(
                "pd_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: run_pd_resume_safe(
                    loaded,
                    pd_records,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["e0"]["batch_size"]),
                    run_git=run_git,
                ),
            )
            family_summary["unit_counts"]["pd"] = {
                "records": len(pd_index) if not pd_index.empty else len(pd_records),
                "expected_records": len(pd_records),
                "expected_rows": pd_expected_rows,
                "units_skipped": pd_skipped,
                "units_total": pd_total,
                "residual_shape": list(pd_resid.shape),
            }
            pl_output, pl_skipped, pl_total, pl_expected_rows = _measure_stage(
                "pl_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    pl_items,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["e0"]["batch_size"]),
                    top_k=int(cfg["e0"]["top_k"]),
                    gen_max_new_tokens=int(cfg["e0"]["gen_max_new_tokens"]),
                    run_git=run_git,
                ),
            )
            family_summary["unit_counts"]["pl"] = {
                "records": len(pl_items),
                "expected_records": len(pl_records),
                "long_rows": len(pl_output.long) if pl_output is not None else 0,
                "expected_rows": pl_expected_rows,
                "units_skipped": pl_skipped,
                "units_total": pl_total,
                "resid_shape": list(pl_output.resid.shape) if pl_output is not None else [],
                "attn_shape": list(pl_output.attn.shape) if pl_output is not None else [],
                "mlp_shape": list(pl_output.mlp.shape) if pl_output is not None else [],
            }

            def resume_pl() -> tuple[int, int]:
                _repeated, skipped_units, total_units, _expected = run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    pl_items,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["e0"]["batch_size"]),
                    top_k=int(cfg["e0"]["top_k"]),
                    gen_max_new_tokens=int(cfg["e0"]["gen_max_new_tokens"]),
                    run_git=run_git,
                )
                return skipped_units, total_units

            resume_skipped, resume_total = _measure_stage(
                "resume_check_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                resume_pl,
            )
            family_summary["resume"] = {"skipped": resume_skipped, "total": resume_total}
            if resume_skipped != resume_total:
                exit_code = 2

            rerun_pd_index, rerun_pd_resid, _rerun_pd_skipped, _rerun_pd_total, _rerun_pd_expected = _measure_stage(
                "pd_rerun_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: run_pd_resume_safe(
                    loaded,
                    pd_records,
                    rerun_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["e0"]["batch_size"]),
                    run_git=run_git,
                ),
            )
            _rerun_pl_output, _rerun_pl_skipped, _rerun_pl_total, _rerun_pl_expected = _measure_stage(
                "pl_rerun_" + family,
                logger,
                family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    pl_items,
                    rerun_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["e0"]["batch_size"]),
                    top_k=int(cfg["e0"]["top_k"]),
                    gen_max_new_tokens=int(cfg["e0"]["gen_max_new_tokens"]),
                    run_git=run_git,
                ),
            )
            reproduction = _compare_family_outputs(loaded, cfg, results_dir, rerun_dir)
            family_summary["rerun"] = reproduction
            if not reproduction["rerun_pass"]:
                exit_code = 2
        except Exception as exc:
            logger.error("family %s failed: %s: %s", family, type(exc).__name__, exc)
            logger.error("%s", traceback.format_exc())
            family_summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["errors"].append({"family": family, "error": family_summary["error"]})
            exit_code = 2
        finally:
            if loaded is not None:
                del loaded
                gc.collect()
                torch.cuda.empty_cache()

    summary["utc_end"] = _utc_now()
    summary["wall_seconds"] = time.perf_counter() - overall_start
    summary["exit_code"] = exit_code
    _atomic_json(results_dir / "summary.json", summary)
    logger.info("E0 summary: %s", json.dumps(summary, sort_keys=True, default=str))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run fail-fast preflight, then launch E0 unless preflight-only."""
    parser = base_parser("Run the E0 directions-only pilot")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_git = capture_run_git(REPO, cfg)
    logger = setup_logging("job_e0_pilot", cfg["paths"]["logs"])
    prompt_path = REPO / cfg["paths"]["prompts_dir"] / "directions_matched.jsonl"
    raw_records = _read_jsonl(prompt_path)
    kept, excluded = locks.pretag_filter(cfg, raw_records)
    if len(kept) != 1315 or len(excluded) != 5:
        logger.error("D12 filtering count mismatch: kept=%d excluded=%d expected=1315/5", len(kept), len(excluded))
        return 2
    kept = prompts.records_for_model(cfg, kept)
    full_sample_ids = _sample_ids(REPO, cfg, int(cfg["manifests"]["e0_sample_size"]))
    family_sets = _family_record_sets(cfg, kept, full_sample_ids)
    main_family = str(cfg["e0"]["main_family"])
    counts: dict[str, dict[str, int]] = {}
    for family, (pd_records, pl_records) in family_sets.items():
        counts[family] = {"pd": len(pd_records), "pl": len(pl_records)}
    expected = {
        main_family: {"pd": 1315, "pl": 497},
        **{str(family): {"pd": 100, "pl": 100} for family in cfg["e0"]["smoke_families"]},
    }
    if counts != expected:
        logger.error("E0 record counts mismatch: actual=%s expected=%s", counts, expected)
        return 2
    print("E0 P-D / P-L record counts:")
    for family, values in counts.items():
        print(f"  {family}: P-D {values['pd']:,} / P-L {values['pl']:,}")
    access_records = [record for pair in family_sets.values() for record in pair[0] + pair[1]]
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E0",
        gpu=not args.preflight_only,
        inputs=[
            "data/external/viLens-concepts/prompts/directions_matched.jsonl",
            "data/external/viLens-concepts/concepts.jsonl",
        ],
        families=["gemma", "qwen", "llama"],
        records=access_records,
        output_dirs=[Path(cfg["e0"]["results_dir"]), Path(cfg["e0"]["rerun_dir"])],
        entrypoint=Path(__file__),
    )
    print(format_checks(checks))
    if not preflight_ok(checks):
        logger.error("preflight failed")
        return 2
    if args.preflight_only:
        logger.info("preflight-only complete; no weights were loaded")
        return 0
    return _run_job(cfg, logger, run_git)


if __name__ == "__main__":
    raise SystemExit(main())
