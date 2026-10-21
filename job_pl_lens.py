"""Resume-safe P-L trial and primary job with tokenizer-only preflight."""

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
import hashlib
import json
import logging
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from vilens import io, locks, models, prompts
from vilens.cli import StageTimer, base_parser, setup_logging
from vilens.config import load_config
from vilens.derived import target_string
from vilens.cognates import id_list_sha256
from vilens.lens import LensHead
from vilens.manifest import verify_manifest
from vilens.passes import (
    Item,
    build_items,
    build_pl_items,
    plan_pl_units,
    run_pl,
    run_pl_resume_safe,
    with_pl_shards,
)
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import RunGit, capture_run_git


def _utc_now() -> str:
    """Return an ISO UTC timestamp for job metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON-lines file in its stored row order."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one JSON object without storing input prompt text."""
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


def _is_true(value: Any) -> bool:
    """Interpret the released JSONL boolean representation strictly."""
    return value is True or (isinstance(value, str) and value.lower() == "true") or (
        type(value) is int and value == 1
    )


def _with_shards(records: Sequence[Mapping[str, Any]], shard_prompts: int) -> list[dict[str, Any]]:
    """Assign stable shard ids independently within each format/set group."""
    return with_pl_shards(records, shard_prompts)


def _primary_records(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load and model-lock the three primary test prompt files for preflight only."""
    records: list[dict[str, Any]] = []
    for stem in cfg["pl"]["primary"]["files"]:
        path = REPO / cfg["paths"]["prompts_dir"] / f"{stem}.jsonl"
        records.extend(_read_jsonl(path))
    records = _with_shards(records, int(cfg["pl"]["shard_prompts"]))
    return _guard_model_records(cfg, records)


def _r1_records(
    cfg: Mapping[str, Any], concepts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Select cognate-free main prompts from R1's registered few-shot files."""
    r1 = cfg["pl"]["r1"]
    flag_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "cognate_flags.jsonl")
    cognate_ids = {
        str(row["concept_id"])
        for row in flag_rows
        if _is_true(row.get("cognate_free", False))
    }
    expected_n = int(cfg["reference_facts"]["x0d"]["cognate_free"])
    expected_sha = str(cfg["reference_facts"]["x0c"]["cognate_free_sha256"])
    if len(cognate_ids) != expected_n or id_list_sha256(cognate_ids) != expected_sha:
        raise AssertionError(
            f"R1 cognate-free manifest count/hash differs: {len(cognate_ids)} rows"
        )

    records: list[dict[str, Any]] = []
    expected_keys = {
        (int(fewshot_set), format_name)
        for fewshot_set in r1["fewshot_sets"]
        for format_name in ("repetition", "translation")
    }
    for stem in r1["files"]:
        format_from_stem, set_suffix = stem.rsplit("_set", maxsplit=1)
        expected_format = format_from_stem.removesuffix("_diac")
        expected_set = int(set_suffix)
        if (expected_set, expected_format) not in expected_keys:
            raise AssertionError(f"Unexpected R1 prompt file configured: {stem}")
        path = REPO / cfg["paths"]["prompts_dir"] / f"{stem}.jsonl"
        for raw in _read_jsonl(path):
            if (
                raw.get("split") != "test"
                or raw.get("condition") != r1["condition"]
                or str(raw.get("format")) != expected_format
                or int(raw.get("fewshot_set", -1)) != expected_set
            ):
                raise AssertionError(f"R1 prompt metadata mismatch in {stem}")
            concept_id = str(raw["concept_id"])
            if concept_id not in cognate_ids:
                continue
            concept = concepts.get(concept_id)
            if concept is None:
                raise KeyError(f"Missing R1 concept metadata for {concept_id}")
            if _is_true(concept.get("m1_extension", raw.get("m1_extension", False))):
                raise AssertionError("R1 cognate-free set unexpectedly includes an M1 extension row")
            records.append(dict(raw))

    expected_by_format = {
        "repetition": int(r1["expected"]["repetition"]),
        "translation": int(r1["expected"]["translation"]),
    }
    observed: Counter[tuple[int, str]] = Counter(
        (int(record["fewshot_set"]), str(record["format"])) for record in records
    )
    if set(observed) != expected_keys:
        raise AssertionError("R1 prompt files do not cover exactly the configured set/format cells")
    for fewshot_set in map(int, r1["fewshot_sets"]):
        for format_name, expected in expected_by_format.items():
            actual = int(observed[(fewshot_set, format_name)])
            if actual != expected:
                raise AssertionError(
                    f"R1 set {fewshot_set}/{format_name} prompts={actual}, expected={expected}"
                )
    records = _with_shards(records, int(cfg["pl"]["shard_prompts"]))
    return _guard_model_records(cfg, records)


def _r1_counts(cfg: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return prompt and target-sequence counts for the fixed R1 capture inputs."""
    r1 = cfg["pl"]["r1"]
    by_set_format: Counter[tuple[int, str]] = Counter(
        (int(record["fewshot_set"]), str(record["format"])) for record in records
    )
    per_set: dict[str, Any] = {}
    sequences_per_prompt = int(r1["expected"]["sequences_per_prompt"])
    for fewshot_set in map(int, r1["fewshot_sets"]):
        by_format = {
            format_name: int(by_set_format[(fewshot_set, format_name)])
            for format_name in ("repetition", "translation")
        }
        expected_by_format = {
            format_name: int(r1["expected"][format_name]) for format_name in by_format
        }
        if by_format != expected_by_format:
            raise AssertionError(f"R1 set {fewshot_set} counts differ: {by_format}")
        prompts = sum(by_format.values())
        per_set[str(fewshot_set)] = {
            "by_format": by_format,
            "prompts": prompts,
            "sequences": prompts * sequences_per_prompt,
            "units_per_family": sum(
                (count + int(cfg["pl"]["shard_prompts"]) - 1)
                // int(cfg["pl"]["shard_prompts"])
                for count in by_format.values()
            ),
        }
    return {
        "sets": per_set,
        "prompts": sum(int(value["prompts"]) for value in per_set.values()),
        "sequences": sum(int(value["sequences"]) for value in per_set.values()),
        "units_per_family": sum(int(value["units_per_family"]) for value in per_set.values()),
        "greedy_generation": bool(r1["generation"]),
        "generation_rows": sum(int(value["prompts"]) for value in per_set.values()),
        "gen_max_new_tokens": int(cfg["pl"]["gen_max_new_tokens"]),
    }


def _e9_records(
    cfg: Mapping[str, Any], concepts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Select OPS-45 nodiac main prompts, excluding collapsed and extension items."""
    e9 = cfg["pl"]["e9"]
    records: list[dict[str, Any]] = []
    expected_format_by_file = {
        "repetition_nodiac_set1": "repetition",
        "translation_nodiac_set1": "translation",
        "cloze_nodiac_set1": "cloze",
    }
    for stem in e9["files"]:
        expected_format = expected_format_by_file[stem]
        path = REPO / cfg["paths"]["prompts_dir"] / f"{stem}.jsonl"
        for raw in _read_jsonl(path):
            if (
                raw.get("split") != "test"
                or raw.get("condition") != e9["condition"]
                or str(raw.get("format")) != expected_format
            ):
                raise AssertionError(f"E9 prompt metadata mismatch in {stem}")
            expected_set = None if expected_format == "cloze" else int(e9["fewshot_set"])
            if raw.get("fewshot_set") != expected_set:
                raise AssertionError(f"E9 few-shot assignment mismatch in {stem}")
            concept_id = str(raw["concept_id"])
            concept = concepts.get(concept_id)
            if concept is None:
                raise KeyError(f"Missing E9 concept metadata for {concept_id}")
            extension = _is_true(concept.get("m1_extension", raw.get("m1_extension", False)))
            if extension:
                continue
            if bool(e9["exclude_collapsed"]) and _is_true(concept.get("collapsed", False)):
                continue
            selected = dict(raw)
            selected["m1_extension"] = False
            records.append(selected)

    expected = {key: int(value) for key, value in e9["expected"].items() if key in {"repetition", "translation", "cloze"}}
    observed = Counter(str(record["format"]) for record in records)
    for fmt, count in expected.items():
        if int(observed[fmt]) != count:
            raise AssertionError(f"E9 {fmt} prompts={observed[fmt]}, expected={count}")
    if len({(str(row["format"]), str(row["concept_id"])) for row in records}) != len(records):
        raise AssertionError("E9 prompt records contain duplicate format/concept rows")
    return _with_shards(_guard_model_records(cfg, records), int(cfg["pl"]["shard_prompts"]))


def _e9_counts(cfg: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count E9 prompts, correct/distractor sequences and per-family unit plan."""
    e9 = cfg["pl"]["e9"]
    by_format = Counter(str(record["format"]) for record in records)
    correct_per_prompt = len(e9["correct_readouts"])
    if e9["translation_extra_readouts"]:
        by_format_sequences = {
            fmt: int(by_format[fmt]) * (2 * correct_per_prompt + (
                len(e9["translation_extra_readouts"]) if fmt == "translation" else 0
            ))
            for fmt in ("repetition", "translation", "cloze")
        }
    else:
        by_format_sequences = {fmt: int(by_format[fmt]) * 2 * correct_per_prompt for fmt in ("repetition", "translation", "cloze")}
    units_by_format = {
        fmt: (int(by_format[fmt]) + int(cfg["pl"]["shard_prompts"]) - 1)
        // int(cfg["pl"]["shard_prompts"])
        for fmt in ("repetition", "translation", "cloze")
    }
    expected = e9["expected"]
    observed = {fmt: int(by_format[fmt]) for fmt in ("repetition", "translation", "cloze")}
    if observed != {fmt: int(expected[fmt]) for fmt in observed}:
        raise AssertionError(f"E9 format counts differ: {observed}")
    sequences = sum(by_format_sequences.values())
    units_per_family = sum(units_by_format.values())
    if sequences != int(expected["sequences"]) or units_per_family != int(expected["units_per_family"]):
        raise AssertionError(
            f"E9 sequence/unit counts differ: sequences={sequences}, units={units_per_family}"
        )
    return {
        "prompts_by_format": observed,
        "prompts": sum(observed.values()),
        "sequences_by_format": by_format_sequences,
        "sequences": sequences,
        "units_by_format_per_family": units_by_format,
        "units_per_family": units_per_family,
        "units_total": units_per_family * len(cfg["pl"]["families"]),
        "sequences_total": sequences * len(cfg["pl"]["families"]),
        "generation_rows": sum(observed.values()) if bool(e9["generation"]) else 0,
        "greedy_generation": bool(e9["generation"]),
        "gen_max_new_tokens": int(cfg["pl"]["gen_max_new_tokens"]),
    }


def _e9_prompt_manifest_hashes(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Return the manifest SHA-256 values for the three selected E9 prompt files."""
    prompt_paths = set(_selected_inputs("e9", cfg)[1:])
    manifest_path = REPO / cfg["paths"]["manifest"]
    hashes: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            expected, relative = line.split("  ", maxsplit=1)
            if relative in prompt_paths:
                hashes[relative.rsplit("/", maxsplit=1)[-1]] = expected
    if set(hashes) != {path.rsplit("/", maxsplit=1)[-1] for path in prompt_paths}:
        raise AssertionError("E9 prompt files are not all present in data/MANIFEST.sha256")
    return hashes


def _e9_disk_projection(
    cfg: Mapping[str, Any], counts: Mapping[str, Any]
) -> dict[str, Any]:
    """Scale completed R1 unit bytes by per-family sequence counts for E9 planning."""
    r1_root = REPO / cfg["pl"]["r1"]["results_dir"]
    source_bytes: dict[str, int] = {}
    for family_value in cfg["pl"]["families"]:
        family = str(family_value)
        family_token = f"family={family}"
        family_files = [
            path for path in r1_root.rglob("*")
            if path.is_file() and family_token in path.name
        ]
        if not family_files:
            raise FileNotFoundError(f"No R1 output files available for E9 disk projection: {family}")
        source_bytes[family] = sum(path.stat().st_size for path in family_files)

    r1 = cfg["pl"]["r1"]
    expected = r1["expected"]
    r1_sequences_per_family = (
        (int(expected["repetition"]) + int(expected["translation"]))
        * len(r1["fewshot_sets"])
        * int(expected["sequences_per_prompt"])
    )
    e9_sequences_per_family = int(counts["sequences"])
    projected_by_family = {
        family: int(round(size * e9_sequences_per_family / r1_sequences_per_family))
        for family, size in source_bytes.items()
    }
    return {
        "source": "results/pl_r1 completed unit files (job 4302 scale reference)",
        "r1_source_bytes_by_family": source_bytes,
        "r1_sequences_per_family": r1_sequences_per_family,
        "e9_sequences_per_family": e9_sequences_per_family,
        "projected_bytes_by_family": projected_by_family,
        "projected_total_bytes": sum(projected_by_family.values()),
    }


def _check_e9_distractors(
    cfg: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Verify every E9 correct and matched-distractor target form exists."""
    target_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "target_forms.jsonl")
    distractor_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "distractors.jsonl")
    targets = {(str(row["concept_id"]), str(row["target_lang"]), str(row["target_variant"])) for row in target_rows}
    distractors = {
        (str(row["family"]), str(row["target_lang"]), str(row["target_variant"]), str(row["concept_id"])): str(row["distractor_id"])
        for row in distractor_rows
    }
    checked: Counter[str] = Counter()
    for family_value in cfg["pl"]["families"]:
        family = str(family_value)
        for record in records:
            concept_id = str(record["concept_id"])
            concept = concepts[concept_id]
            if _is_true(concept.get("m1_extension", False)) or _is_true(concept.get("collapsed", False)):
                raise AssertionError("E9 selection includes an extension or collapsed item")
            for lang, variant in cfg["pl"]["e9"]["correct_readouts"]:
                target_key = (concept_id, str(lang), str(variant))
                if target_key not in targets:
                    raise KeyError(f"Missing E9 target form {target_key}")
                distractor_key = (family, str(lang), str(variant), concept_id)
                distractor_id = distractors.get(distractor_key)
                if distractor_id is None:
                    raise KeyError(f"Missing E9 distractor {family}/{lang}/{variant}")
                if (distractor_id, str(lang), str(variant)) not in targets:
                    raise KeyError(f"Missing E9 distractor target form {family}/{lang}/{variant}")
                checked[family] += 1
    return dict(checked)


def _guard_model_records(cfg: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Route model-bound records through the integrity lock boundary."""
    return prompts.records_for_model(dict(cfg), [dict(record) for record in records])


def _trial_records(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select the 50 E0 concepts' Vietnamese directions records for the trial."""
    sample = _read_jsonl(REPO / cfg["paths"]["derived"] / "e0_pilot.jsonl")
    sample_ids = {
        str(row["concept_id"])
        for row in sorted(sample, key=lambda row: int(row["rank"]))[: int(cfg["manifests"]["e0_sample_size"])]
    }
    raw = _read_jsonl(REPO / cfg["paths"]["prompts_dir"] / "directions_matched.jsonl")
    selected = [
        dict(row)
        for row in raw
        if str(row["concept_id"]) in sample_ids and str(row.get("lang")) == "vi"
    ]
    if len(selected) != 100:
        raise AssertionError(f"Trial directions records={len(selected)}, expected 100")
    return _with_shards(_guard_model_records(cfg, selected), int(cfg["pl"]["shard_prompts"]))


def _primary_counts(
    cfg: Mapping[str, Any], records: Sequence[Mapping[str, Any]], concepts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Count correct and distractor sequences from primary record metadata."""
    prompts_by_format = Counter(str(record["format"]) for record in records)
    sequences: Counter[str] = Counter()
    for record in records:
        concept = concepts[str(record["concept_id"])]
        extension = _is_true(concept.get("m1_extension", record.get("m1_extension", False)))
        fmt = str(record["format"])
        correct = len(cfg["pl"]["correct_readouts"])
        if fmt == "translation":
            correct += len(cfg["pl"]["translation_extra_readouts"])
        distractor_count = 0 if extension else len(cfg["pl"]["correct_readouts"])
        sequences[fmt] += correct + distractor_count
    actual = {
        "prompts": len(records),
        "sequences": sum(sequences.values()),
        "repetition": sequences["repetition"],
        "translation": sequences["translation"],
        "cloze": sequences["cloze"],
        "prompts_by_format": dict(prompts_by_format),
    }
    expected = dict(cfg["pl"]["expected"])
    for key in expected:
        if actual[key] != expected[key]:
            raise AssertionError(f"Primary P-L {key}={actual[key]}, expected={expected[key]}")
    return actual


def _selected_inputs(mode: str, cfg: Mapping[str, Any]) -> list[str]:
    """Return manifest-tracked input paths for one preflight mode."""
    concepts = "data/external/viLens-concepts/concepts.jsonl"
    if mode == "primary":
        return [
            concepts,
            *[
                f"data/external/viLens-concepts/prompts/{stem}.jsonl"
                for stem in ("repetition_diac_set1", "translation_diac_set1", "cloze_diac_set1")
            ],
        ]
    if mode == "r1":
        return [
            concepts,
            *[
                f"data/external/viLens-concepts/prompts/{stem}.jsonl"
                for stem in cfg["pl"]["r1"]["files"]
            ],
        ]
    if mode == "e9":
        return [
            concepts,
            *[
                f"data/external/viLens-concepts/prompts/{stem}.jsonl"
                for stem in cfg["pl"]["e9"]["files"]
            ],
        ]
    return [concepts, "data/external/viLens-concepts/prompts/directions_matched.jsonl"]


def _preflight(
    cfg: dict[str, Any],
    mode: str,
    *,
    gpu: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[Any]]:
    """Prepare guarded records and run the shared fail-fast preflight."""
    concepts_rows = _read_jsonl(REPO / cfg["paths"]["concepts_jsonl"])
    concepts = {str(row["concept_id"]): row for row in concepts_rows}
    if mode == "primary":
        records = _primary_records(cfg)
        counts = _primary_counts(cfg, records, concepts)
        output_dirs = [Path(cfg["pl"]["results_dir"])]
    elif mode == "trial":
        records = _trial_records(cfg)
        counts = {"prompts": len(records), "sequences": None}
        if len(records) != 100:
            raise AssertionError(f"Trial prompt count={len(records)}, expected 100")
        output_dirs = [
            Path(cfg["pl"]["trial"]["results_dir"]),
            Path(cfg["pl"]["trial"]["rerun_dir"]),
        ]
    elif mode == "r1":
        records = _r1_records(cfg, concepts)
        counts = _r1_counts(cfg, records)
        output_dirs = [Path(cfg["pl"]["r1"]["results_dir"])]
    else:
        records = _e9_records(cfg, concepts)
        counts = _e9_counts(cfg, records)
        counts["prompt_manifest_sha256"] = _e9_prompt_manifest_hashes(cfg)
        counts["distractor_pairs_checked_per_family"] = _check_e9_distractors(cfg, records, concepts)
        counts["disk_projection"] = _e9_disk_projection(cfg, counts)
        output_dir = REPO / cfg["pl"]["e9"]["results_dir"]
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("E9 output directory already contains files; refusing to overwrite")
        output_dirs = [Path(cfg["pl"]["e9"]["results_dir"])]
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="P-L",
        gpu=gpu,
        inputs=_selected_inputs(mode, cfg),
        families=list(cfg["pl"]["families"]),
        records=records,
        output_dirs=output_dirs,
        entrypoint=Path(__file__),
    )
    return records, counts, checks


def _e0_record_sets(cfg: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct the exact locked E0 P-L record lists without the now-open D12 filter."""
    rows = _read_jsonl(REPO / cfg["paths"]["prompts_dir"] / "directions_matched.jsonl")
    excluded = {
        (str(item["concept_id"]), str(item["format"]), str(item["lang"]))
        for item in cfg["data"]["pretag_excluded_directions"]
    }
    locked_rows = [
        row for row in rows
        if (str(row["concept_id"]), str(row["format"]), str(row["lang"])) not in excluded
    ]
    sample = _read_jsonl(REPO / cfg["paths"]["derived"] / "e0_pilot.jsonl")
    result: dict[str, list[dict[str, Any]]] = {}
    main_family = str(cfg["e0"]["main_family"])
    families = [main_family, *map(str, cfg["e0"]["smoke_families"])]
    for family in families:
        count = int(cfg["manifests"]["e0_sample_size"])
        if family != main_family:
            count = int(cfg["e0"]["smoke_concepts"])
        ids = {
            str(row["concept_id"])
            for row in sorted(sample, key=lambda row: int(row["rank"]))[:count]
        }
        result[family] = [dict(row) for row in locked_rows if str(row["concept_id"]) in ids]
    expected = {main_family: 497}
    expected.update({str(family): 100 for family in cfg["e0"]["smoke_families"]})
    for family, records in result.items():
        if len(records) != expected[family]:
            raise AssertionError(
                f"E0-equivalence input count for {family} is {len(records)}, expected {expected[family]}"
            )
        _guard_model_records(cfg, records)
    return result


def _run_primary(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_git: RunGit,
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str = "primary",
) -> tuple[int, dict[str, Any]]:
    """Run primary or fixed R1 P-L inputs one resume-safe unit at a time."""
    if mode not in {"primary", "r1", "e9"}:
        raise ValueError("P-L capture mode must be primary, r1 or e9")
    if mode == "r1" and not bool(cfg["pl"]["r1"]["generation"]):
        raise AssertionError("R1 P-L requires OPS-42 greedy generation")
    if mode == "e9" and not bool(cfg["pl"]["e9"]["generation"]):
        raise AssertionError("E9 P-L requires OPS-45 greedy generation")
    target_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "target_forms.jsonl")
    distractor_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "distractors.jsonl")
    concepts_rows = _read_jsonl(REPO / cfg["paths"]["concepts_jsonl"])
    concepts = {str(row["concept_id"]): row for row in concepts_rows}
    results_dir = REPO / (
        cfg["pl"]["results_dir"] if mode == "primary"
        else cfg["pl"]["r1"]["results_dir"] if mode == "r1"
        else cfg["pl"]["e9"]["results_dir"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "utc_start": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "code_tree_sha256": run_git.code_tree_sha256,
        "mode": mode,
        "input_counts": (
            _primary_counts(cfg, records, concepts)
            if mode == "primary" else _r1_counts(cfg, records) if mode == "r1"
            else _e9_counts(cfg, records)
        ),
        "families": {},
        "errors": [],
    }
    exit_code = 0

    for family_value in cfg["pl"]["families"]:
        family = str(family_value)
        family_result: dict[str, Any] = {
            "stage_seconds": {},
            "peak_memory_bytes": {},
            "units": [],
        }
        summary["families"][family] = family_result
        loaded: models.LoadedModel | None = None

        def stage(name: str, function: Any) -> Any:
            """Time one primary stage and capture its peak device allocation."""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timer = StageTimer(name, logger, family_result["stage_seconds"])
            try:
                with timer:
                    return function()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    family_result["peak_memory_bytes"][name] = int(torch.cuda.max_memory_allocated())

        try:
            loaded = stage(f"load_{family}", lambda: models.load(cfg, family, device="cuda"))
            fp32_lens = LensHead.from_loaded(loaded, "fp32")
            family_records = [dict(record) for record in records]
            items = stage(
                f"build_{mode}_items_{family}",
                lambda: build_pl_items(
                    cfg, family_records, concepts, target_rows, distractor_rows, family, mode
                ),
            )
            prompt_count = len(
                {(str(item.record["format"]), item.concept_id, item.prompt) for item in items}
            )
            sequence_count = sum(1 + (item.distractor_encoding is not None) for item in items)
            family_result["counts"] = {
                "prompts": prompt_count,
                "items": len(items),
                "sequences": sequence_count,
            }
            expected_counts = (
                cfg["pl"]["expected"] if mode == "primary"
                else _r1_counts(cfg, records) if mode == "r1"
                else _e9_counts(cfg, records)
            )
            for key in ("prompts", "sequences"):
                if family_result["counts"][key] != int(expected_counts[key]):
                    raise AssertionError(
                        f"P-L {mode} {family} {key}={family_result['counts'][key]}, "
                        f"expected {expected_counts[key]}"
                    )
            if mode in {"r1", "e9"}:
                readout_config = cfg["pl"][mode]
                expected_items = int(expected_counts["prompts"]) * len(readout_config["correct_readouts"])
                if family_result["counts"]["items"] != expected_items:
                    raise AssertionError(
                        f"P-L {mode} {family} items={family_result['counts']['items']}, "
                        f"expected {expected_items}"
                    )

            plans = stage(
                f"plan_{mode}_units_{family}",
                lambda: plan_pl_units(
                    loaded,
                    items,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["pl"]["batch_size"][family]),
                    run_git=run_git,
                ),
            )
            if not plans:
                raise AssertionError(f"P-L {mode} {family} has no units")
            family_result["planned_units"] = len(plans)
            expected_unit_count = (
                int(expected_counts["units_per_family"])
                if mode == "r1" else int(expected_counts["units_per_family"])
                if mode == "e9" else len(plans)
            )
            if mode in {"r1", "e9"} and len(plans) != expected_unit_count:
                raise AssertionError(
                    f"P-L {mode} {family} units={len(plans)}, expected={expected_unit_count}"
                )

            for plan in plans:
                unit_sequences = sum(
                    1 + (item.distractor_encoding is not None) for item in plan.items
                )
                stage_keys = ("family", "format", "shard")
                if mode in {"r1", "e9"}:
                    stage_keys = ("family", "fewshot_set", "format", "shard")
                unit_stage = "pl_" + "_".join(
                    str(plan.unit_key[key]) for key in stage_keys if key in plan.unit_key
                )

                def run_unit() -> tuple[Any, int, int, int]:
                    """Run or provenance-skip exactly one planned primary unit."""
                    return run_pl_resume_safe(
                        loaded,
                        fp32_lens,
                        plan.items,
                        results_dir,
                        cfg,
                        repo_root=REPO,
                        entrypoint=Path(__file__),
                        batch_size=int(cfg["pl"]["batch_size"][family]),
                        top_k=int(cfg["pl"]["top_k"]),
                        gen_max_new_tokens=int(cfg["pl"]["gen_max_new_tokens"]),
                        run_git=run_git,
                    )

                output, skipped, total, expected_rows = stage(unit_stage, run_unit)
                if total != 1 or expected_rows != plan.expected_row_count:
                    raise AssertionError(
                        f"Primary P-L unit {plan.unit_key} planned {plan.expected_row_count} rows, "
                        f"resume runner returned units={total}, rows={expected_rows}"
                    )
                actual_rows = expected_rows if skipped == 1 else (
                    len(output.long) if output is not None else -1
                )
                if skipped not in {0, 1} or actual_rows != expected_rows:
                    raise AssertionError(
                        f"Primary P-L unit {plan.unit_key} rows={actual_rows}, "
                        f"expected={expected_rows}, skipped={skipped}"
                    )
                seconds = float(family_result["stage_seconds"][unit_stage])
                peak_bytes = int(family_result["peak_memory_bytes"].get(unit_stage, 0))
                unit_result = {
                    "unit_key": dict(plan.unit_key),
                    "rows": actual_rows,
                    "expected_rows": expected_rows,
                    "sequences": unit_sequences,
                    "skipped": bool(skipped),
                    "wall_seconds": seconds,
                    "sequences_per_second": unit_sequences / max(seconds, 1e-9),
                    "peak_memory_bytes": peak_bytes,
                }
                family_result["units"].append(unit_result)
                logger.info("P-L primary unit %s", json.dumps(unit_result, sort_keys=True))
                del output

            if len(family_result["units"]) != len(plans):
                raise AssertionError(
                    f"Primary P-L {family} completed {len(family_result['units'])} units, "
                    f"planned {len(plans)}"
                )
            logger.info(
                "P-L primary family summary %s: %s",
                family,
                json.dumps(family_result, sort_keys=True),
            )
        except Exception as exc:
            exit_code = 2
            error = f"{type(exc).__name__}: {exc}"
            family_result["error"] = error
            family_result["traceback"] = traceback.format_exc()
            summary["errors"].append({"family": family, "error": error})
            logger.error("P-L primary family %s failed: %s", family, error)
            logger.error("%s", traceback.format_exc())
            break
        finally:
            if loaded is not None:
                del loaded
                gc.collect()
                torch.cuda.empty_cache()

    summary["utc_end"] = _utc_now()
    summary["exit_code"] = exit_code
    return exit_code, summary


def _unit_key(family: str, items: Sequence[Item], shard: int | None = None) -> dict[str, str | int]:
    """Return the P-L unit key for one format/condition/few-shot/shard group."""
    if not items:
        raise ValueError("Cannot make a P-L unit key from no items")
    record = items[0].record
    key: dict[str, str | int] = {
        "family": family,
        "pass": "pl",
        "condition": str(record.get("condition", "diac")),
        "fewshot_set": "none" if record.get("fewshot_set") is None else str(record["fewshot_set"]),
        "format": str(record["format"]),
    }
    if shard is not None:
        key["shard"] = int(shard)
    return key


def _compare_long_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    """Compare long tables by exact rule-38 keys; return lens and model max deltas."""
    io.validate_long(left.loc[:, io.LENS_COLUMNS], io.LENS_COLUMNS)
    io.validate_long(right.loc[:, io.LENS_COLUMNS], io.LENS_COLUMNS)
    keys = [column for column in io.LENS_COLUMNS if column != "value"]
    merged = left.merge(right, on=keys, how="outer", suffixes=("_new", "_stored"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError("P-L result keys differ from the comparison table")
    if merged.empty:
        return 0.0, 0.0
    merged["delta"] = (merged["value_new"] - merged["value_stored"]).abs()
    max_delta = float(merged["delta"].max())
    lens = float(merged.loc[merged["metric"] == "lens_logprob", "delta"].max())
    model = float(merged.loc[merged["metric"] == "model_logprob", "delta"].max())
    return lens, model


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Read a P-L auxiliary JSONL file without logging its generated text."""
    if not path.is_file():
        return []
    return _read_jsonl(path)


def _compare_e0_outputs(
    family: str,
    output: Any,
    e0_results: Path,
) -> dict[str, Any]:
    """Compare new in-memory P-L outputs with the stored E0 P-L units."""
    lens_delta = 0.0
    model_delta = 0.0
    topk_ids_equal = True
    generations_equal = True
    format_counts: dict[str, int] = {}
    for format_name in sorted(set(output.long["format"].astype(str))):
        new_frame = output.long.loc[output.long["format"].astype(str) == format_name].copy()
        key = {"family": family, "pass": "pl", "format": format_name}
        parquet, _sidecar = io.unit_paths(e0_results, key)
        if not parquet.is_file():
            raise FileNotFoundError(f"Missing E0 comparison unit {parquet}")
        old_frame = pd.read_parquet(parquet)
        current_lens, current_model = _compare_long_frames(new_frame, old_frame)
        lens_delta = max(lens_delta, current_lens)
        model_delta = max(model_delta, current_model)
        format_counts[format_name] = len(new_frame)

        old_topk = _read_json_rows(parquet.with_suffix(".answer_topk.jsonl"))
        new_topk = output.answer_topk.loc[output.answer_topk["format"].astype(str) == format_name]
        top_key = ("format", "concept_id", "target_lang", "target_variant", "layer", "rank")
        old_top = {tuple(str(row[k]) for k in top_key): int(row["token_id"]) for row in old_topk}
        new_top = {
            tuple(str(row[k]) for k in top_key): int(row["token_id"])
            for row in new_topk.to_dict(orient="records")
        }
        topk_ids_equal = topk_ids_equal and old_top == new_top

        old_generations = _read_json_rows(parquet.with_suffix(".generations.jsonl"))
        new_generations = output.generations.loc[
            output.generations["format"].astype(str) == format_name
        ].to_dict(orient="records")
        gen_key = lambda row: (str(row["format"]), str(row.get("lang", "")), str(row["concept_id"]))
        old_gen = {gen_key(row): (row.get("token_ids"), row.get("text")) for row in old_generations}
        new_gen = {gen_key(row): (row.get("token_ids"), row.get("text")) for row in new_generations}
        generations_equal = generations_equal and old_gen == new_gen
    return {
        "lens_logprob_max_abs_delta": lens_delta,
        "model_logprob_max_abs_delta": model_delta,
        "topk_ids_identical": topk_ids_equal,
        "generations_identical": generations_equal,
        "formats_row_counts": format_counts,
    }


def _compare_rerun_unit(
    results_dir: Path,
    rerun_dir: Path,
    unit_key: Mapping[str, str | int],
    tolerance: float,
) -> dict[str, Any]:
    """Compare a first trial shard with its fresh rerun, including activation arrays."""
    left_path, _ = io.unit_paths(results_dir, unit_key)
    right_path, _ = io.unit_paths(rerun_dir, unit_key)
    left = pd.read_parquet(left_path)
    right = pd.read_parquet(right_path)
    lens_delta, model_delta = _compare_long_frames(left, right)
    array_deltas: dict[str, float] = {}
    for suffix in (".resid.npy", ".attn.npy", ".mlp.npy"):
        a = np.load(left_path.with_suffix(suffix), allow_pickle=False)
        b = np.load(right_path.with_suffix(suffix), allow_pickle=False)
        if a.shape != b.shape:
            raise AssertionError(f"Rerun shape differs for {suffix}: {a.shape} != {b.shape}")
        array_deltas[suffix] = float(np.max(np.abs(a - b))) if a.size else 0.0
    left_index = _read_json_rows(left_path.with_suffix(".answer_topk.jsonl"))
    right_index = _read_json_rows(right_path.with_suffix(".answer_topk.jsonl"))
    topk_identical = [row.get("token_id") for row in left_index] == [row.get("token_id") for row in right_index]
    left_gen = _read_json_rows(left_path.with_suffix(".generations.jsonl"))
    right_gen = _read_json_rows(right_path.with_suffix(".generations.jsonl"))
    generations_identical = [row.get("token_ids") for row in left_gen] == [
        row.get("token_ids") for row in right_gen
    ]
    max_delta = max([lens_delta, model_delta, *array_deltas.values()])
    return {
        "lens_logprob_max_abs_delta": lens_delta,
        "model_logprob_max_abs_delta": model_delta,
        "activation_max_abs_deltas": array_deltas,
        "topk_ids_identical": topk_identical,
        "generations_identical": generations_identical,
        "max_abs_delta": max_delta,
        "pass": max_delta <= tolerance and topk_identical and generations_identical,
    }


def _compare_inprocess_outputs(left: Any, right: Any) -> dict[str, Any]:
    """Compare two in-memory P-L outputs exactly for the D1 determinism gate."""
    lens_delta, model_delta = _compare_long_frames(left.long, right.long)
    topk_keys = [column for column in left.answer_topk.columns if column != "value"]
    topk = left.answer_topk.merge(
        right.answer_topk, on=topk_keys, how="outer", suffixes=("_left", "_right"), indicator=True
    )
    if not topk["_merge"].eq("both").all():
        raise AssertionError("In-process determinism top-k keys differ")
    topk_delta = float((topk["value_left"] - topk["value_right"]).abs().max()) if not topk.empty else 0.0
    array_deltas: dict[str, float] = {}
    for name in ("resid", "attn", "mlp"):
        left_array = getattr(left, name)
        right_array = getattr(right, name)
        if left_array.shape != right_array.shape:
            raise AssertionError(f"In-process determinism {name} shapes differ")
        array_deltas[name] = (
            float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0
        )
    topk_ids_identical = left.answer_topk["token_id"].tolist() == right.answer_topk["token_id"].tolist()
    generations_identical = left.generations["token_ids"].tolist() == right.generations["token_ids"].tolist()
    index_identical = left.resid_index.to_dict(orient="records") == right.resid_index.to_dict(orient="records")
    max_delta = max(lens_delta, model_delta, topk_delta, *array_deltas.values())
    return {
        "lens_logprob_max_abs_delta": lens_delta,
        "model_logprob_max_abs_delta": model_delta,
        "topk_value_max_abs_delta": topk_delta,
        "activation_max_abs_deltas": array_deltas,
        "topk_ids_identical": topk_ids_identical,
        "generations_identical": generations_identical,
        "residual_index_identical": index_identical,
        "max_abs_delta": max_delta,
        "pass": max_delta == 0.0 and topk_ids_identical and generations_identical and index_identical,
    }


def _audit_trial_units(
    cfg: Mapping[str, Any],
    loaded: models.LoadedModel,
    items: Sequence[Item],
    results_dir: Path,
    run_git: RunGit,
    entrypoint: Path,
) -> list[dict[str, Any]]:
    """Verify every persisted trial unit and its pinned model/job provenance."""
    plans = plan_pl_units(
        loaded,
        items,
        results_dir,
        cfg,
        repo_root=REPO,
        entrypoint=entrypoint,
        batch_size=int(cfg["pl"]["batch_size"][loaded.family]),
        run_git=run_git,
    )
    audited: list[dict[str, Any]] = []
    for plan in plans:
        parquet, sidecar = io.unit_paths(results_dir, plan.unit_key)
        complete, reason = io.unit_is_complete(
            results_dir, plan.unit_key, plan.provenance, cfg["runtime"]["provenance_compare"]
        )
        if not complete:
            raise AssertionError(f"Trial unit incomplete {plan.unit_key}: {reason}")
        frame = pd.read_parquet(parquet)
        io.validate_long(frame, io.LENS_COLUMNS)
        with sidecar.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
        if meta.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID"):
            raise AssertionError(f"Unit job id differs in {sidecar}")
        if meta.get("git_commit") != run_git.git_commit or meta.get("git_dirty") is not False:
            raise AssertionError(f"Unit Git provenance differs in {sidecar}")
        if meta.get("code_tree_sha256") != run_git.code_tree_sha256:
            raise AssertionError(f"Unit code-tree provenance differs in {sidecar}")
        if meta.get("model_id") != loaded.spec.hf_id or meta.get("model_revision") != loaded.spec.revision:
            raise AssertionError(f"Unit model pin differs in {sidecar}")
        if meta.get("dtype") != "torch.bfloat16" or int(meta.get("batch_size", -1)) != int(
            cfg["pl"]["batch_size"][loaded.family]
        ):
            raise AssertionError(f"Unit dtype/batch differs in {sidecar}")
        audited.append(
            {
                "unit_key": plan.unit_key,
                "rows": len(frame),
                "row_count_matches": len(frame) == plan.expected_row_count,
                "sidecar_job_id": meta["slurm_job_id"],
                "code_tree_sha256": meta["code_tree_sha256"],
                "model_loads": meta.get("model_loads"),
                "gpu_name": meta.get("gpu_name"),
            }
        )
    return audited


def _run_trial(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_git: RunGit,
    e0_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[int, dict[str, Any]]:
    """Run the directions-only trial, determinism and rerun gates, and resume check."""
    target_rows = _read_jsonl(REPO / cfg["paths"]["derived"] / "target_forms.jsonl")
    e0_distractors = _read_jsonl(REPO / cfg["paths"]["derived"] / "e0_distractors.jsonl")
    concepts_rows = _read_jsonl(REPO / cfg["paths"]["concepts_jsonl"])
    concepts = {str(row["concept_id"]): row for row in concepts_rows}
    all_trial_records = _trial_records(cfg)
    results_dir = REPO / cfg["pl"]["trial"]["results_dir"]
    rerun_dir = REPO / cfg["pl"]["trial"]["rerun_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    rerun_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "utc_start": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "code_tree_sha256": run_git.code_tree_sha256,
        "e0_record_counts": {family: len(records) for family, records in e0_records.items()},
        "families": {},
        "errors": [],
    }
    exit_code = 0
    total_trial_seconds = 0.0

    for family in cfg["pl"]["families"]:
        family = str(family)
        family_result: dict[str, Any] = {
            "gate_results": {},
            "stage_seconds": {},
            "peak_memory_bytes": {},
        }
        summary["families"][family] = family_result
        loaded: models.LoadedModel | None = None

        def stage(name: str, function: Any) -> Any:
            """Run and time one family stage while capturing peak GPU allocation."""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timer = StageTimer(name, logger, family_result["stage_seconds"])
            try:
                with timer:
                    return function()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    family_result["peak_memory_bytes"][name] = int(torch.cuda.max_memory_allocated())

        try:
            loaded = stage(f"load_{family}", lambda: models.load(cfg, family, device="cuda"))
            fp32_lens = LensHead.from_loaded(loaded, "fp32")

            def run_e0_equivalence() -> dict[str, Any]:
                """Run the exact pre-D12 E0 record list at matching legacy batch composition."""
                records = list(e0_records[family])
                e0_items = build_items(cfg, records, target_rows, e0_distractors, family)
                expected_items = 497 if family == cfg["e0"]["main_family"] else 100
                if len(e0_items) != expected_items:
                    raise AssertionError(f"E0 P-L items={len(e0_items)}, expected={expected_items}")
                output = run_pl(
                    loaded,
                    fp32_lens,
                    e0_items,
                    # The legacy API batches 16 items, i.e. 32 correct/distractor
                    # sequences. Preserve that exact forward composition here.
                    batch_size=2 * int(cfg["e0"]["batch_size"]),
                    top_k=int(cfg["e0"]["top_k"]),
                    gen_max_new_tokens=int(cfg["e0"]["gen_max_new_tokens"]),
                )
                measured = _compare_e0_outputs(
                    family, output, REPO / cfg["e0"]["results_dir"]
                )
                measured["record_count"] = len(records)
                measured["item_count"] = len(e0_items)
                measured["batch_size_items"] = int(cfg["e0"]["batch_size"])
                measured["batch_size_sequences"] = 2 * int(cfg["e0"]["batch_size"])
                measured["pass"] = (
                    measured["lens_logprob_max_abs_delta"] <= 1e-4
                    and measured["model_logprob_max_abs_delta"] <= 1e-4
                    and measured["topk_ids_identical"]
                    and measured["generations_identical"]
                )
                del output
                return measured

            e0_equivalence = stage(f"e0_equivalence_{family}", run_e0_equivalence)
            family_result["gate_results"]["e0_equivalence"] = e0_equivalence
            if not e0_equivalence["pass"]:
                exit_code = 2

            trial_records = [dict(row) for row in all_trial_records]
            trial_items = stage(
                f"build_trial_items_{family}",
                lambda: build_pl_items(
                    cfg, trial_records, concepts, target_rows, e0_distractors, family, "trial"
                ),
            )
            trial_prompts = len({
                (str(item.record["format"]), item.concept_id, item.prompt) for item in trial_items
            })
            trial_sequences = sum(
                1 + (item.distractor_encoding is not None) for item in trial_items
            )
            no_distractor = Counter(
                f"{item.target_lang}:{item.target_variant}"
                for item in trial_items
                if item.distractor_encoding is None
            )
            test_ids = {
                concept_id for concept_id, concept in concepts.items()
                if str(concept.get("split")) == "test"
            }
            trial_ids = {item.concept_id for item in trial_items}
            no_test_items = not (trial_ids & test_ids)
            directions_only = all(
                str(concepts[concept_id].get("split")) == "directions" for concept_id in trial_ids
            )
            family_result["trial_counts"] = {
                "prompts": trial_prompts,
                "items": len(trial_items),
                "sequences": trial_sequences,
                "no_distractor_readouts": dict(no_distractor),
            }
            family_result["gate_results"]["directions_only_no_test_ids"] = {
                "directions_only": directions_only,
                "test_id_count": len(trial_ids & test_ids),
                "pass": directions_only and no_test_items,
            }
            if not directions_only or not no_test_items:
                exit_code = 2

            unit_plans = plan_pl_units(
                loaded,
                trial_items,
                results_dir,
                cfg,
                repo_root=REPO,
                entrypoint=Path(__file__),
                batch_size=int(cfg["pl"]["batch_size"][family]),
                run_git=run_git,
            )
            if not unit_plans:
                raise AssertionError("The P-L trial has no units")
            first_plan = unit_plans[0]

            def run_d1_determinism() -> dict[str, Any]:
                """Compute the first unit twice back-to-back without persisting it."""
                run_args = {
                    "batch_size": int(cfg["pl"]["batch_size"][family]),
                    "top_k": int(cfg["pl"]["top_k"]),
                    "gen_max_new_tokens": int(cfg["pl"]["gen_max_new_tokens"]),
                    "encodings": first_plan.encodings,
                    "keep_prompt_groups": True,
                }
                first = run_pl(loaded, fp32_lens, first_plan.items, **run_args)
                second = run_pl(loaded, fp32_lens, first_plan.items, **run_args)
                result = _compare_inprocess_outputs(first, second)
                result["first_unit_key"] = first_plan.unit_key
                del first, second
                return result

            d1 = stage(f"d1_determinism_{family}", run_d1_determinism)
            family_result["gate_results"]["d1_inprocess_determinism"] = d1
            if not d1["pass"]:
                exit_code = 2

            def run_full_trial() -> dict[str, Any]:
                """Run the trial item set resume-safely and capture its unit summary."""
                output, skipped, total, expected_rows = run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    trial_items,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["pl"]["batch_size"][family]),
                    top_k=int(cfg["pl"]["top_k"]),
                    gen_max_new_tokens=int(cfg["pl"]["gen_max_new_tokens"]),
                    run_git=run_git,
                )
                return {
                    "output": output,
                    "units_skipped": skipped,
                    "units_total": total,
                    "expected_row_count": expected_rows,
                    "pending_expected_row_count": 0 if output is None else output.expected_row_count,
                }

            trial_started = time.perf_counter()
            trial_run = stage(f"trial_{family}", run_full_trial)
            trial_seconds = time.perf_counter() - trial_started
            total_trial_seconds += trial_seconds
            output = trial_run.pop("output")
            family_result["trial"] = {
                **trial_run,
                "wall_seconds": trial_seconds,
                "actual_row_count": len(output.long) if output is not None else None,
            }
            if output is not None:
                if len(output.long) != output.expected_row_count:
                    raise AssertionError("Pending P-L unit rows differ from their expected count")
                family_result["trial"]["run_time_seconds"] = family_result["stage_seconds"][f"trial_{family}"]
                if not output.resid_index.empty and not output.resid_index["split"].isin(["fit", "validate"]).all():
                    raise AssertionError("Non-directions split appeared in trial residual index")
            elif trial_run["units_skipped"] != trial_run["units_total"]:
                raise AssertionError("Trial output was skipped without all units being complete")

            audited_units = stage(
                f"audit_units_{family}",
                lambda: _audit_trial_units(
                    cfg, loaded, trial_items, results_dir, run_git, Path(__file__)
                ),
            )
            family_result["trial"]["units"] = audited_units
            stored_row_count = sum(int(unit["rows"]) for unit in audited_units)
            rows_pass = stored_row_count == trial_run["expected_row_count"]
            family_result["trial"]["actual_row_count"] = stored_row_count
            family_result["gate_results"]["trial_rows"] = {
                "actual": stored_row_count,
                "expected": trial_run["expected_row_count"],
                "pass": rows_pass,
            }
            if not rows_pass:
                exit_code = 2

            first_shard_items = list(first_plan.items)
            first_unit_key = first_plan.unit_key
            rerun_parquet, rerun_sidecar = io.unit_paths(rerun_dir, first_unit_key)
            rerun_artifacts = [
                rerun_parquet,
                rerun_sidecar,
                *[
                    rerun_parquet.with_suffix(suffix)
                    for suffix in (
                        ".resid.npy",
                        ".attn.npy",
                        ".mlp.npy",
                        ".resid_index.jsonl",
                        ".answer_topk.jsonl",
                        ".generations.jsonl",
                    )
                ],
            ]
            if any(path.exists() for path in rerun_artifacts):
                raise FileExistsError(f"Rerun artifacts already exist for {first_unit_key}")

            def rerun_first_shard() -> dict[str, Any]:
                """Write the first prompt shard to a fresh directory for reproducibility."""
                output_rerun, skipped, total, rows = run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    first_shard_items,
                    rerun_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["pl"]["batch_size"][family]),
                    top_k=int(cfg["pl"]["top_k"]),
                    gen_max_new_tokens=int(cfg["pl"]["gen_max_new_tokens"]),
                    run_git=run_git,
                )
                if output_rerun is None or skipped != 0 or total != 1:
                    raise AssertionError("Fresh first-shard rerun did not produce exactly one unit")
                return {"rows": rows, "skipped": skipped, "units": total}

            rerun_run = stage(f"rerun_{family}", rerun_first_shard)
            family_result["rerun"] = {
                **rerun_run,
                **_compare_rerun_unit(
                    results_dir,
                    rerun_dir,
                    first_unit_key,
                    float(cfg["pl"]["trial"]["rerun_tolerance"]),
                ),
                "first_unit_key": first_unit_key,
            }
            if not family_result["rerun"]["pass"]:
                exit_code = 2

            def resume_trial() -> dict[str, Any]:
                """Run resume planning again and record the all-units skip result."""
                _output, skipped, total, expected_rows = run_pl_resume_safe(
                    loaded,
                    fp32_lens,
                    trial_items,
                    results_dir,
                    cfg,
                    repo_root=REPO,
                    entrypoint=Path(__file__),
                    batch_size=int(cfg["pl"]["batch_size"][family]),
                    top_k=int(cfg["pl"]["top_k"]),
                    gen_max_new_tokens=int(cfg["pl"]["gen_max_new_tokens"]),
                    run_git=run_git,
                )
                return {"skipped": skipped, "total": total, "expected_rows": expected_rows}

            resume_started = time.perf_counter()
            resume_data = stage(f"resume_{family}", resume_trial)
            resume_seconds = time.perf_counter() - resume_started
            family_result["resume"] = {**resume_data, "wall_seconds": resume_seconds}
            family_result["resume"]["pass"] = (
                resume_data["skipped"] == resume_data["total"]
                and resume_seconds < 0.1 * max(trial_seconds, 1e-9)
            )
            if not family_result["resume"]["pass"]:
                exit_code = 2

            rate = float(trial_sequences) / max(trial_seconds, 1e-9)
            family_result["trial_stage_sequences_per_second"] = rate
            logger.info("P-L trial summary %s: %s", family, json.dumps(family_result, sort_keys=True, default=str))
        except Exception as exc:
            exit_code = 2
            error = f"{type(exc).__name__}: {exc}"
            family_result["error"] = error
            family_result["traceback"] = traceback.format_exc()
            summary["errors"].append({"family": family, "error": error})
            logger.error("P-L family %s failed: %s", family, error)
            logger.error("%s", traceback.format_exc())
        finally:
            if loaded is not None:
                del loaded
                gc.collect()
                torch.cuda.empty_cache()

    projection_by_family: dict[str, Any] = {}
    total_projected_seconds = 0.0
    for family, family_result in summary["families"].items():
        rate = family_result.get("trial_stage_sequences_per_second")
        projected_tf = 46720.0 / float(rate) if rate and float(rate) > 0 else None
        projected_total = projected_tf
        if projected_total is not None:
            total_projected_seconds += projected_total
        projection_by_family[family] = {
            "primary_sequences": 46720,
            "trial_stage_sequences_per_second": rate,
            "projected_seconds": projected_tf,
            "total_projected_seconds": projected_total,
        }
    summary["primary_projection"] = {
        "per_family": projection_by_family,
        "gpu_hours": total_projected_seconds / 3600.0 if total_projected_seconds else None,
        "wall_hours_sequential_families": total_projected_seconds / 3600.0 if total_projected_seconds else None,
        "basis": "trial-stage sequence throughput (trial sequences divided by trial_<family> stage seconds), scaled to 46,720 primary sequences per family",
    }
    summary["total_trial_wall_seconds"] = total_trial_seconds
    summary["utc_end"] = _utc_now()
    summary["exit_code"] = exit_code
    return exit_code, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight and run the trial, primary, or fixed R1 P-L capture path."""
    parser = base_parser("Run the directions-only P-L trial or primary P-L pass")
    parser.add_argument("--items", choices=("trial", "primary", "r1", "e9"), required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logging("job_pl_lens", cfg["paths"]["logs"])
    run_git = capture_run_git(REPO, cfg)
    results_name = (
        cfg["pl"]["results_dir"]
        if args.items == "primary"
        else cfg["pl"]["r1"]["results_dir"]
        if args.items == "r1"
        else cfg["pl"]["e9"]["results_dir"]
        if args.items == "e9"
        else cfg["pl"]["trial"]["results_dir"]
    )
    summary_path = REPO / results_name / "summary.json"
    if args.items == "trial":
        e0_records = _e0_record_sets(cfg)
    else:
        e0_records = {}
    try:
        records, counts, checks = _preflight(cfg, args.items, gpu=not args.preflight_only)
        print("P-L input counts:", json.dumps(counts, sort_keys=True))
        print(format_checks(checks))
        if not preflight_ok(checks):
            logger.error("P-L preflight failed")
            if not args.preflight_only:
                _atomic_json(
                    summary_path,
                    {
                        "utc_start": _utc_now(),
                        "utc_end": _utc_now(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "preflight": format_checks(checks),
                        "exit_code": 2,
                    },
                )
            return 2
        if args.preflight_only:
            logger.info("preflight-only completed; no model weights were loaded")
            return 0
        if args.items == "primary":
            exit_code, summary = _run_primary(cfg, logger, run_git, records)
        elif args.items in {"r1", "e9"}:
            exit_code, summary = _run_primary(cfg, logger, run_git, records, mode=args.items)
        else:
            exit_code, summary = _run_trial(cfg, logger, run_git, e0_records)
        _atomic_json(summary_path, summary)
        logger.info("P-L summary saved: %s", summary_path)
        return exit_code
    except Exception as exc:
        logger.error("P-L entrypoint failed: %s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        if not args.preflight_only:
            _atomic_json(
                summary_path,
                {
                    "utc_start": _utc_now(),
                    "utc_end": _utc_now(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "exit_code": 2,
                },
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
