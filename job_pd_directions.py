"""P-D directions pass: matched prompts and FLORES residual caches."""

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
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from vilens import io, locks, models, passes, prompts
from vilens.cli import StageTimer, base_parser, setup_logging
from vilens.config import load_config, study_models
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import RunGit, build as build_provenance, capture_run_git


def _utc_now() -> str:
    """Return a compact ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read ordered JSON-lines records without displaying their string fields."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _input_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Resolve the directions prompt, concept table, and FLORES input paths."""
    root = Path(cfg["paths"]["repo_root"])
    prompt = prompt_file_paths(dict(cfg))["directions_matched"]
    if not prompt.is_absolute():
        prompt = root / prompt
    concepts = root / cfg["paths"]["concepts_jsonl"]
    flores = root / cfg["flores"]["output"]
    return prompt, concepts, flores


def _validated_inputs(
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load and count only the frozen directions and FLORES records used by P-D."""
    prompt_path, concepts_path, flores_path = _input_paths(cfg)
    matched_all = load_prompt_file(prompt_path)
    matched, excluded = locks.pretag_filter(dict(cfg), matched_all)
    if len(matched) != int(cfg["pd"]["expected_matched_records"]) or excluded:
        raise AssertionError(
            f"matched records kept/excluded={len(matched)}/{len(excluded)}; expected 1320/0"
        )
    if any(record.get("split") not in {"fit", "validate"} for record in matched):
        raise AssertionError("P-D directions input contains a non-directions split record")
    matched = prompts.records_for_model(dict(cfg), matched)
    split_by_concept: dict[str, str] = {}
    for record in matched:
        concept_id = str(record["concept_id"])
        split = str(record["split"])
        previous = split_by_concept.setdefault(concept_id, split)
        if previous != split:
            raise AssertionError(f"Directions split varies within concept {concept_id}")
    split_counts = {
        split: sum(value == split for value in split_by_concept.values())
        for split in ("fit", "validate")
    }
    if len(split_by_concept) != 132 or split_counts != {"fit": 106, "validate": 26}:
        raise AssertionError(f"directions concept split counts differ: {split_counts}")
    counts_by_concept: dict[str, set[tuple[str, str]]] = {}
    for record in matched:
        counts_by_concept.setdefault(str(record["concept_id"]), set()).add(
            (str(record["lang"]), str(record["format"]))
        )
    expected_grid = {(lang, fmt) for lang in ("vi", "en", "zh", "fr", "id") for fmt in ("repetition", "translation")}
    if any(value != expected_grid for value in counts_by_concept.values()):
        raise AssertionError("directions records do not form the expected concept × language × format grid")

    flores = _read_jsonl(flores_path)
    expected_flores = int(cfg["pd"]["expected_flores_records"])
    if len(flores) != expected_flores:
        raise AssertionError(f"FLORES records={len(flores)}; expected {expected_flores}")
    output_language_codes = {
        "vie_Latn": "vi",
        "eng_Latn": "en",
        "cmn_Hans": "zh",
        "fra_Latn": "fr",
        "ind_Latn": "id",
    }
    expected_languages = {output_language_codes[code] for code in cfg["flores"]["languages"]}
    by_language: dict[str, dict[str, str]] = {}
    for record in flores:
        if set(record) != {"lang", "sentence", "sentence_id", "split"}:
            raise AssertionError("FLORES record keys do not match the frozen schema")
        lang = str(record["lang"])
        sentence_id = str(record["sentence_id"])
        split = str(record["split"])
        if lang not in expected_languages or split not in {"fit", "validate"}:
            raise AssertionError("FLORES contains an unexpected language or split")
        if sentence_id in by_language.setdefault(lang, {}):
            raise AssertionError("FLORES contains a duplicate language/sentence id")
        by_language[lang][sentence_id] = split
    if set(by_language) != expected_languages:
        raise AssertionError("FLORES language set differs from config")
    id_sets = [set(rows) for rows in by_language.values()]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise AssertionError("FLORES sentence id sets differ across languages")
    validate_ids = {
        sentence_id
        for sentence_id, split in by_language[output_language_codes[cfg["flores"]["languages"][0]]].items()
        if split == "validate"
    }
    expected_validate_ids = int(cfg["pd"]["expected_flores_validate_ids"])
    if len(id_sets[0]) != 997 or len(validate_ids) != expected_validate_ids:
        raise AssertionError(
            f"FLORES unique/validate IDs={len(id_sets[0])}/{len(validate_ids)}; "
            f"expected 997/{expected_validate_ids}"
        )
    if sum(record["split"] == "validate" for record in flores) != expected_validate_ids * 5:
        raise AssertionError("FLORES validate record count is not 199 IDs × 5 languages")
    shared_splits = {
        sentence_id: split
        for sentence_id, split in by_language[
            output_language_codes[cfg["flores"]["languages"][0]]
        ].items()
    }
    if any(
        rows[sentence_id] != shared_splits[sentence_id]
        for rows in by_language.values()
        for sentence_id in rows
    ):
        raise AssertionError("FLORES split assignment differs across languages")
    return matched, flores, {
        "matched_kept": len(matched),
        "matched_excluded": len(excluded),
        "directions_concepts": len(split_by_concept),
        "directions_fit_concepts": split_counts["fit"],
        "directions_validate_concepts": split_counts["validate"],
        "flores_records": len(flores),
        "flores_ids": len(id_sets[0]),
        "flores_validate_ids": len(validate_ids),
    }


def _unit_key(family: str, source: str, format_name: str | None) -> dict[str, str | int]:
    """Build the fixed provenance key for one P-D source unit."""
    key: dict[str, str | int] = {"family": family, "pass": "pd", "source": source}
    if format_name is not None:
        key["format"] = format_name
    return key


def _array_paths(results_dir: Path, key: Mapping[str, str | int], source: str) -> tuple[Path, Path, list[Path]]:
    """Resolve one unit's parquet, sidecar, and NumPy activation files."""
    parquet, sidecar = io.unit_paths(results_dir, key)
    suffixes = (".resid.npy",) if source == "matched" else (".last.npy", ".mean.npy")
    return parquet, sidecar, [parquet.with_suffix(suffix) for suffix in suffixes]


def _plan_unit(
    cfg: Mapping[str, Any],
    family: str,
    source: str,
    format_name: str | None,
    records: Sequence[Mapping[str, Any]],
    run_git: RunGit,
) -> dict[str, Any]:
    """Check one result unit's provenance and auxiliary files before loading weights."""
    spec_data = study_models(dict(cfg))[family]
    key = _unit_key(family, source, format_name)
    batch_size = int(cfg["inference"]["batch_size"][family])
    provenance = build_provenance(
        cfg,
        repo_root=REPO,
        unit_key=key,
        entrypoint=Path(__file__),
        model_id=str(spec_data["hf_id"]),
        model_revision=str(spec_data["revision"]),
        dtype="torch.bfloat16",
        batch_size=batch_size,
        expected_row_count=len(records),
        run_git=run_git,
    )
    results_dir = REPO / cfg["pd"]["results_dir"]
    parquet_path, sidecar_path, arrays = _array_paths(results_dir, key, source)
    complete, reason = io.unit_is_complete(
        results_dir, key, provenance, cfg["runtime"]["provenance_compare"]
    )
    complete = complete and all(path.is_file() for path in arrays)
    if complete and source == "matched":
        complete = np.load(arrays[0], mmap_mode="r", allow_pickle=False).shape[0] == len(records)
    if complete and source == "flores":
        complete = all(
            np.load(path, mmap_mode="r", allow_pickle=False).shape[0] == len(records)
            for path in arrays
        )
    return {
        "unit_key": key,
        "source": source,
        "format": format_name,
        "records": list(records),
        "provenance": provenance,
        "complete": complete,
        "reason": reason if not complete else "complete",
        "parquet_path": parquet_path,
        "sidecar_path": sidecar_path,
        "array_paths": arrays,
        "batch_size": batch_size,
    }


def _projection(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Calculate float32 activation cache bytes from the frozen text dimensions."""
    facts = cfg["reference_facts"]["x0c"]["model_dims"]
    matched_records = int(cfg["pd"]["expected_matched_records"])
    flores_records = int(cfg["pd"]["expected_flores_records"])
    rows: list[dict[str, Any]] = []
    total = 0
    for family in cfg["pd"]["families"]:
        n_layers, d_model, _vocab = map(int, facts[family])
        matched_bytes = matched_records * (n_layers + 1) * d_model * 4
        flores_bytes = flores_records * (n_layers + 1) * d_model * 4 * 2
        family_total = matched_bytes + flores_bytes
        total += family_total
        rows.append(
            {
                "family": family,
                "matched_bytes": matched_bytes,
                "flores_bytes": flores_bytes,
                "total_bytes": family_total,
            }
        )
    free_bytes = shutil.disk_usage(REPO).free
    logger = logging.getLogger("pd_directions")
    logger.info("projection fits available datastore space: %s", total <= free_bytes)
    return rows, total, free_bytes


def _log_projection(logger: logging.Logger, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Log projected activation bytes by family and compare the total with free disk."""
    rows, total, free_bytes = _projection(cfg)
    logger.info("Projection bytes (float32 arrays only):")
    logger.info("family | matched | flores last+mean | total")
    for row in rows:
        logger.info(
            "%s | %d | %d | %d",
            row["family"], row["matched_bytes"], row["flores_bytes"], row["total_bytes"],
        )
    logger.info("TOTAL | %d; datastore free bytes=%d", total, free_bytes)
    return rows


def _measure_stage(
    name: str,
    logger: logging.Logger,
    timings: dict[str, float],
    peak_memory: dict[str, int | None],
    function: Callable[[], Any],
) -> Any:
    """Run and time one pass stage, tracking synchronized CUDA peak allocation."""
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
            peak_memory[name] = int(torch.cuda.max_memory_allocated())
        else:
            peak_memory[name] = None


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Atomically write the ID/count-only job summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(summary), stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _unit_summary_from_disk(plan: Mapping[str, Any], status: str) -> dict[str, Any]:
    """Read only NumPy array metadata and norms for a completed unit summary."""
    arrays = {
        path.name.rsplit(".", 2)[-2]: np.load(path, mmap_mode="r", allow_pickle=False)
        for path in plan["array_paths"]
    }
    n_layers = next(iter(arrays.values())).shape[1] - 1
    layers = sorted({0, n_layers // 2, n_layers})
    norms = {
        name: {str(layer): float(np.linalg.norm(array[:, layer, :], axis=-1).mean()) for layer in layers}
        for name, array in arrays.items()
    }
    files = [plan["parquet_path"], plan["sidecar_path"], *plan["array_paths"]]
    return {
        "unit_key": dict(plan["unit_key"]),
        "status": status,
        "rows": len(plan["records"]),
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "bytes_on_disk": sum(path.stat().st_size for path in files if path.is_file()),
        "nonfinite_count": sum(int((~np.isfinite(array)).sum()) for array in arrays.values()),
        "mean_residual_norms": norms,
        "reason": str(plan["reason"]),
    }


def _main(args: argparse.Namespace) -> int:
    """Run preflight or the three-family matched and FLORES P-D units."""
    run_started = time.perf_counter()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = load_config(cfg_path)
    # Capture once before any preflight or model work (OPS-11).
    run_git = capture_run_git(REPO, cfg)
    logger = setup_logging("pd_directions", REPO / "logs")
    utc_start = _utc_now()
    logger.info("P-D start: utc=%s git_commit=%s git_dirty=%s", utc_start, run_git.git_commit, run_git.git_dirty)
    matched, flores, counts = _validated_inputs(cfg)
    prereg = cfg["prereg"]
    lock_open = bool(prereg.get("analysis_tag")) and prereg.get("test_items_unlocked") is True
    logger.info(
        "D12 filter: kept=%d excluded=%d reason=%s",
        counts["matched_kept"], counts["matched_excluded"],
        "D12 exclusion lapsed: lock open" if lock_open else "D12 exclusion active: lock closed",
    )
    logger.info("Input counts: %s", json.dumps(counts, sort_keys=True))
    _log_projection(logger, cfg)

    prompt_path, concepts_path, flores_path = _input_paths(cfg)
    input_paths = [prompt_path, concepts_path, flores_path]
    input_names = [path.relative_to(REPO).as_posix() for path in input_paths]
    lock_records = matched + flores
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="P-D",
        gpu=not args.preflight_only,
        inputs=input_names,
        families=list(cfg["pd"]["families"]),
        records=lock_records,
        output_dirs=[Path(cfg["pd"]["results_dir"])],
        entrypoint=Path(__file__),
    )
    print(format_checks(checks))
    if not preflight_ok(checks):
        logger.error("P-D preflight failed")
        return 2
    if args.preflight_only:
        print(
            f"P-D counts: matched {counts['matched_kept']} kept / {counts['matched_excluded']} excluded; "
            f"FLORES {counts['flores_records']} records, {counts['flores_validate_ids']} validate ids"
        )
        return 0

    results_dir = REPO / cfg["pd"]["results_dir"]
    selected = study_models(cfg)
    batch_sizes = cfg["inference"]["batch_size"]
    plans: dict[str, list[dict[str, Any]]] = {}
    for family in cfg["pd"]["families"]:
        family_records = [
            [record for record in matched if str(record["format"]) == fmt]
            for fmt in cfg["pd"]["matched_formats"]
        ]
        family_plans = [
            _plan_unit(cfg, family, "matched", fmt, records, run_git)
            for fmt, records in zip(cfg["pd"]["matched_formats"], family_records, strict=True)
        ]
        family_plans.append(_plan_unit(cfg, family, "flores", None, flores, run_git))
        plans[family] = family_plans

    summary: dict[str, Any] = {
        "utc_start": utc_start,
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "input_counts": counts,
        "families": {},
        "errors": [],
    }
    exit_code = 0
    for family in cfg["pd"]["families"]:
        family_plans = plans[family]
        family_summary: dict[str, Any] = {
            "timings_seconds": {},
            "peak_memory_bytes": {},
            "units": [],
        }
        summary["families"][family] = family_summary
        pending = [plan for plan in family_plans if not plan["complete"]]
        for plan in family_plans:
            if plan["complete"]:
                unit_result = _unit_summary_from_disk(plan, "skipped")
                family_summary["units"].append(unit_result)
                logger.info("unit skipped: key=%s rows=%d", plan["unit_key"], len(plan["records"]))
        if not pending:
            continue

        loaded: models.LoadedModel | None = None
        try:
            loaded = _measure_stage(
                f"load_{family}", logger, family_summary["timings_seconds"],
                family_summary["peak_memory_bytes"],
                lambda: models.load(cfg, family, device="cuda"),
            )
            for plan in pending:
                plan["sidecar_path"].unlink(missing_ok=True)
                source = str(plan["source"])
                format_name = plan["format"]
                stage_name = f"{family}_{source}_{format_name or 'all'}"
                try:
                    result = _measure_stage(
                        stage_name, logger, family_summary["timings_seconds"],
                        family_summary["peak_memory_bytes"],
                        lambda plan=plan: passes.run_pd_storage_unit(
                            loaded,
                            plan["records"],
                            results_dir,
                            cfg,
                            source=source,
                            format_name=format_name,
                            repo_root=REPO,
                            entrypoint=Path(__file__),
                            batch_size=int(batch_sizes[family]),
                            run_git=run_git,
                        ),
                    )
                    result["unit_key"] = dict(plan["unit_key"])
                    result["reason"] = "completed and verified"
                    if source == "flores":
                        result["max_n_tokens"] = int(
                            pd_read_max_tokens(results_dir, plan["unit_key"])
                        )
                    family_summary["units"].append(result)
                    logger.info(
                        "unit %s: key=%s rows=%s shapes=%s bytes=%s nonfinite=%s status=%s",
                        stage_name, plan["unit_key"], result["rows"], result["shapes"],
                        result["bytes_on_disk"], result["nonfinite_count"], result["status"],
                    )
                except Exception as exc:
                    exit_code = 2
                    summary["errors"].append(
                        {"unit_key": dict(plan["unit_key"]), "exception_type": type(exc).__name__}
                    )
                    logger.error("unit failed: key=%s exception_type=%s", plan["unit_key"], type(exc).__name__)
        except Exception as exc:
            exit_code = 2
            summary["errors"].append({"family": family, "exception_type": type(exc).__name__})
            logger.error("family failed: family=%s exception_type=%s", family, type(exc).__name__)
        finally:
            if loaded is not None:
                del loaded
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary["utc_end"] = _utc_now()
    summary["total_runtime_seconds"] = time.perf_counter() - run_started
    summary["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary["peak_gpu_memory_bytes"] = max(
        (
            memory
            for family_data in summary["families"].values()
            for memory in family_data["peak_memory_bytes"].values()
            if memory is not None
        ),
        default=0,
    )
    summary_path = results_dir / "summary.json"
    _write_summary(summary_path, summary)
    logger.info(
        "P-D summary: exit_code=%d runtime_seconds=%.3f gpu=%s peak_memory_bytes=%d summary=%s",
        exit_code, summary["total_runtime_seconds"], summary["gpu_name"],
        summary["peak_gpu_memory_bytes"], summary_path,
    )
    return exit_code


def pd_read_max_tokens(results_dir: Path, unit_key: Mapping[str, str | int]) -> int:
    """Read the maximum token count from the ID/count-only FLORES parquet index."""
    import pyarrow.parquet as pq

    parquet_path, _sidecar = io.unit_paths(results_dir, unit_key)
    frame = pq.read_table(parquet_path, columns=["n_tokens"]).to_pandas()
    return int(frame["n_tokens"].max())


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the shared job CLI and run P-D."""
    parser: argparse.ArgumentParser = base_parser("P-D matched directions and FLORES residual capture")
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except Exception as exc:
        logger = logging.getLogger("pd_directions")
        logger.error("P-D stopped: exception_type=%s", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
