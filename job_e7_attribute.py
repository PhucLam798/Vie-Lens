"""P-A E7 component and attention-head attribution pass."""

from __future__ import annotations

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

if __name__ == "__main__":
    env_guard.guard(offline=True)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/datastore/luannt/.cache/matplotlib")
os.environ.setdefault("CUDA_CACHE_PATH", "/datastore/luannt/.cache/nv")

import argparse
import gc
import json
import logging
import math
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from vilens import attribute, hooks, io, locks, models
from vilens.cli import setup_logging
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.manifest import sha256_file
from vilens.models import LoadedModel
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import RunGit, build as build_provenance, capture_run_git

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
PA_COLUMNS = [
    "model", "model_revision", "pass", "format", "item_set", "concept_id", "layer",
    "layer_type", "kind", "head", "direction", "projection", "update_norm",
]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _exception_details(exc: BaseException) -> dict[str, str]:
    """Return the exception message and its innermost source frame for summaries."""
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        frame = frames[-1]
        filename = Path(frame.filename)
        try:
            filename = filename.relative_to(REPO)
        except ValueError:
            pass
        innermost = f"{filename}:{frame.lineno}, {frame.name}"
    else:
        innermost = "unavailable"
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "innermost_traceback_frame": innermost,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL metadata file without echoing records."""
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _is_extension(value: Any) -> bool:
    """Interpret the native boolean or serialized boolean extension flag."""
    return value is True or str(value).strip().lower() in {"true", "1"}


def _prompt_records(cfg: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    """Load registered primary prompt metadata and the frozen cognate-free IDs."""
    paths = prompt_file_paths(dict(cfg))
    output: dict[str, list[dict[str, Any]]] = {}
    for format_name in FORMATS:
        stem = str(cfg["data"]["primary_prompt_files"][f"{format_name}_diac"])
        records = load_prompt_file(REPO / paths[stem])
        selected = [
            dict(row) for row in records
            if str(row.get("format")) == format_name
            and str(row.get("condition", "diac")) == "diac"
            and str(row.get("fewshot_set")) == str(cfg["e7"]["fewshot_set"])
            and str(row.get("split")) == "test"
            and not _is_extension(row.get("m1_extension", False))
        ]
        selected.sort(key=lambda row: str(row["concept_id"]))
        expected = int(cfg["e7"]["expected_prompt_counts"]["main"][format_name])
        if len(selected) != expected or len({str(row["concept_id"]) for row in selected}) != expected:
            raise AssertionError(f"{format_name} main prompt rows do not match the frozen E7 count")
        output[format_name] = selected
    flags = _read_jsonl(REPO / "derived/cognate_flags.jsonl")
    cognate_free = {
        str(row["concept_id"])
        for row in flags
        if row.get("cognate_free") is True or str(row.get("cognate_free", "")).lower() == "true"
    }
    primary_ids = {
        format_name: cognate_free & {str(row["concept_id"]) for row in output[format_name]}
        for format_name in FORMATS
    }
    for format_name in FORMATS:
        expected = int(cfg["e7"]["expected_prompt_counts"]["cf729"][format_name])
        if len(primary_ids[format_name]) != expected:
            raise AssertionError(f"{format_name} cf729 prompt count differs from the frozen E7 count")
    return output, primary_ids


def _pd_stable_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return P-D's stable matched-record order: format, language, concept ID."""
    return str(row["format"]), str(row.get("lang", "")), str(row["concept_id"])


def _pl_prompt_stable_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the prompt-level prefix of P-L's stable item sort key."""
    return (
        str(row.get("format", "")),
        str(row.get("condition", "diac")),
        "" if row.get("fewshot_set") is None else str(row.get("fewshot_set")),
        str(row.get("pl_shard", "")),
        str(row.get("lang", "")),
        str(row.get("concept_id", "")),
        str(row.get("prompt", "")),
    )


def _e7_prompt_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a private identity key for E7 main prompts, including the prompt bytes."""
    return (
        str(row.get("format", "")),
        str(row.get("condition", "diac")),
        "" if row.get("fewshot_set") is None else str(row.get("fewshot_set")),
        str(row.get("lang", "")),
        str(row.get("concept_id", "")),
        str(row.get("prompt", "")),
    )


def _replay_source_and_batch_size(
    cfg: Mapping[str, Any], family: str, purpose: str
) -> tuple[str, int]:
    """Resolve a source pass and its configured prompt-capture batch size."""
    replay = cfg["e7"]["replay"][purpose]
    source = str(replay["source"])
    path = str(replay["batch_size_config"])
    value: Any = cfg
    for part in path.split("."):
        value = value[part]
    base_size = int(value[family])
    divisor = int(replay["divisor"])
    if divisor <= 0 or base_size <= 0:
        raise ValueError(f"invalid configured replay batch size for {source}/{family}")
    if base_size % divisor:
        raise ValueError(f"configured replay batch size is not divisible for {source}/{family}")
    return source, base_size // divisor


def _runtime_profile() -> str:
    """Return the scheduler profile from Slurm environment metadata."""
    partition = os.environ.get("SLURM_JOB_PARTITION", "").strip().lower()
    if partition in {"small", "normal", "large", "full"}:
        return partition
    qos_profiles = {
        "mig1-18": "small",
        "mig2-35": "normal",
        "mig3-71": "large",
        "mig7-141": "full",
    }
    return qos_profiles.get(os.environ.get("SLURM_JOB_QOS", "").strip().lower(), "unknown")


def _pl_source_prompt_records(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load P-L's primary source rows and assign the original per-format shards."""
    from vilens.passes import with_pl_shards

    paths = prompt_file_paths(dict(cfg))
    records: list[dict[str, Any]] = []
    for stem in cfg["pl"]["primary"]["files"]:
        path = Path(paths[str(stem)])
        if not path.is_absolute():
            path = REPO / path
        records.extend(_read_jsonl(path))
    return with_pl_shards(records, int(cfg["pl"]["shard_prompts"]))


def _pl_source_prompt_dedup(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reproduce P-L's per-unit prompt de-duplication before batch replay."""
    ordered = sorted((dict(row) for row in records), key=_pl_prompt_stable_key)
    seen: set[tuple[str, str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in ordered:
        key = (
            str(row.get("format", "")),
            str(row.get("condition", "diac")),
            "" if row.get("fewshot_set") is None else str(row.get("fewshot_set")),
            str(row.get("concept_id", "")),
            str(row.get("prompt", "")),
        )
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _pl_source_cell_records(
    cfg: Mapping[str, Any], source_records: Sequence[Mapping[str, Any]], format_name: str
) -> list[dict[str, Any]]:
    """Return de-duplicated P-L prompt records for one E7 format and source unit set."""
    rows = [
        dict(row)
        for row in source_records
        if str(row.get("format")) == format_name
        and str(row.get("condition", "diac")) == "diac"
        and str(row.get("fewshot_set")) == str(cfg["e7"]["fewshot_set"])
        and str(row.get("split")) == "test"
    ]
    return _pl_source_prompt_dedup(rows)


def _pl_replay_batches(
    cfg: Mapping[str, Any], family: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Build P-L's per-shard exact prompt capture batches through the shared helper."""
    source, batch_size = _replay_source_and_batch_size(cfg, family, "p_a")
    if source != "pl":
        raise ValueError("P-A replay source must be P-L")
    return attribute.replay_capture_batches(
        records,
        stable_key=_pl_prompt_stable_key,
        shard_key=lambda row: int(row["pl_shard"]),
        batch_size=batch_size,
        shard_order_key=int,
    )


def _pd_replay_batches(
    cfg: Mapping[str, Any], family: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Build P-D's exact per-cell capture batches through the shared helper."""
    source, batch_size = _replay_source_and_batch_size(cfg, family, "smoke")
    if source != "pd":
        raise ValueError("smoke replay source must be P-D")
    return attribute.replay_capture_batches(
        records,
        stable_key=lambda row: int(row["pd_array_row"]),
        shard_key=lambda row: str(row["format"]),
        batch_size=batch_size,
        shard_order_key=str,
    )


def _select_smoke_records(
    prompt_records: Sequence[Mapping[str, Any]],
    pd_index_rows: Sequence[Mapping[str, Any]],
    *,
    format_name: str,
    prompt_count: int,
) -> tuple[list[dict[str, Any]], str]:
    """Select the first configured P-D ordered directions prompts without consulting test locks."""
    prompt_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in prompt_records:
        row = dict(raw)
        key = _pd_stable_sort_key(row)
        if key[0] == format_name:
            if key in prompt_by_key:
                raise ValueError(f"duplicate directions prompt metadata for {format_name}")
            prompt_by_key[key] = row

    candidates = [
        dict(row) for row in pd_index_rows
        if str(row.get("format")) == format_name
        and str(row.get("split")) in {"fit", "validate"}
    ]
    validate_rows = sorted(
        (row for row in candidates if str(row["split"]) == "validate"),
        key=_pd_stable_sort_key,
    )
    selected_split = "validate" if len(validate_rows) >= prompt_count else "fit"
    selected_rows = sorted(
        (row for row in candidates if str(row["split"]) == selected_split),
        key=_pd_stable_sort_key,
    )[:prompt_count]
    if len(selected_rows) != prompt_count:
        raise ValueError(
            f"P-D {selected_split} split has {len(selected_rows)} {format_name} rows; "
            f"smoke requires {prompt_count}"
        )

    selected: list[dict[str, Any]] = []
    for pd_row in selected_rows:
        key = _pd_stable_sort_key(pd_row)
        if key not in prompt_by_key:
            raise ValueError(f"P-D smoke identity has no matching directions prompt for {format_name}")
        record = dict(prompt_by_key[key])
        if str(record.get("split")) != selected_split:
            raise ValueError("P-D smoke split differs from the directions prompt metadata")
        if str(record.get("split")) == "test":
            raise AssertionError("smoke selection contains a test-split record")
        record["pd_array_row"] = int(pd_row["row"])
        selected.append(record)
    if any(str(row.get("split")) == "test" for row in selected):
        raise AssertionError("smoke selection contains a test-split record")
    return selected, selected_split


def _smoke_prompt_records(
    cfg: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    """Select 32 P-D directions records per family and format using P-D's stable order."""
    import pyarrow.parquet as pq

    prompt_path = prompt_file_paths(dict(cfg))["directions_matched"]
    if not prompt_path.is_absolute():
        prompt_path = REPO / prompt_path
    all_prompts = load_prompt_file(prompt_path)
    prompt_count = int(cfg["e7"]["trial_items_per_cell"])
    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    splits: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        selected[family] = {}
        splits[family] = {}
        for format_name in FORMATS:
            index_path, _sidecar_path, _residual_path = _pd_paths(cfg, family, format_name)
            if not index_path.is_file():
                raise FileNotFoundError(f"P-D matched index missing for {family}/{format_name}")
            pd_rows = pq.read_table(
                index_path, columns=["row", "concept_id", "lang", "format", "split"]
            ).to_pylist()
            rows, split = _select_smoke_records(
                all_prompts, pd_rows, format_name=format_name, prompt_count=prompt_count
            )
            selected[family][format_name] = rows
            splits[family][format_name] = split
    return selected, splits, all_prompts


def _pd_paths(cfg: Mapping[str, Any], family: str, format_name: str) -> tuple[Path, Path, Path]:
    """Return a P-D matched index, provenance sidecar, and residual array."""
    stem = f"family={family}__format={format_name}__pass=pd__source=matched"
    index = REPO / str(cfg["e7"]["p_d_results_dir"]) / f"{stem}.parquet"
    return index, index.with_suffix(".provenance.json"), index.with_suffix(".resid.npy")


def _pd_ordered_replay_records(
    format_name: str,
    all_prompt_records: Sequence[Mapping[str, Any]],
    pd_index_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match P-D's stable input order against its stored row metadata."""
    candidates = [
        dict(row)
        for row in all_prompt_records
        if str(row.get("format")) == format_name
        and str(row.get("split")) in {"fit", "validate"}
    ]
    candidates.sort(key=_pd_stable_sort_key)
    indexed = sorted(pd_index_rows, key=lambda row: int(row["row"]))
    if len(candidates) != len(indexed):
        raise ValueError("P-D replay source row count differs from the stored index")
    ordered: list[dict[str, Any]] = []
    for array_row, (record, stored) in enumerate(zip(candidates, indexed, strict=True)):
        if int(stored["row"]) != array_row:
            raise ValueError("P-D replay index rows are not contiguous")
        if (
            _pd_stable_sort_key(record) != _pd_stable_sort_key(stored)
            or str(record.get("split")) != str(stored.get("split"))
        ):
            raise ValueError("P-D replay source order differs from the stored index")
        record["pd_array_row"] = array_row
        ordered.append(record)
    return ordered


def _pl_row_identity(row: Mapping[str, Any], *, stored: bool) -> tuple[Any, ...]:
    """Return P-L prompt record identity without exposing prompt text."""
    shard_field = "shard" if stored else "pl_shard"
    return (
        str(row.get("format", "")),
        str(row.get("condition", "diac")),
        "" if row.get("fewshot_set") is None else str(row.get("fewshot_set")),
        int(row[shard_field]),
        str(row.get("lang", "")),
        str(row.get("concept_id", "")),
        str(row.get("split", "")),
        _is_extension(row.get("m1_extension", False)),
    )


def _verify_pl_replay_cell(
    cfg: Mapping[str, Any],
    family: str,
    format_name: str,
    all_pl_source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate P-L source membership/order and report its replay batches."""
    source_records = _pl_source_cell_records(cfg, all_pl_source_records, format_name)
    _source, batch_size = _replay_source_and_batch_size(cfg, family, "p_a")
    batches = _pl_replay_batches(cfg, family, source_records)
    source_by_shard: dict[int, list[Mapping[str, Any]]] = {}
    for batch in batches:
        source_by_shard.setdefault(int(batch["shard"]), []).extend(batch["records"])

    root = REPO / str(cfg["e7"]["p_l_results_dir"])
    prefix = (
        f"condition=diac__family={family}__fewshot_set={cfg['e7']['fewshot_set']}"
        f"__format={format_name}__pass=pl__shard="
    )
    array_paths = sorted(root.glob(prefix + "*.resid.npy"))
    stored_by_shard: dict[int, list[dict[str, Any]]] = {}
    expected_shape = tuple(int(value) for value in cfg["reference_facts"]["x0c"]["model_dims"][family][:2])
    expected_residual_shape = (expected_shape[0] + 1, expected_shape[1])
    for array_path in array_paths:
        index_path = array_path.with_name(array_path.name.removesuffix(".resid.npy") + ".resid_index.jsonl")
        sidecar_path = array_path.with_name(array_path.name.removesuffix(".resid.npy") + ".provenance.json")
        if not index_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError("P-L replay shard lacks its stored index or provenance")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if int(sidecar["batch_size"]) != int(cfg["pl"]["batch_size"][family]):
            raise ValueError("P-L replay configuration batch size differs from stored provenance")
        stored_rows = _read_jsonl(index_path)
        if not stored_rows:
            raise ValueError("P-L replay shard has an empty stored index")
        shard_values = {int(row["shard"]) for row in stored_rows}
        if len(shard_values) != 1:
            raise ValueError("P-L replay shard mixes shard identifiers")
        shard = next(iter(shard_values))
        if sorted(int(row["array_row"]) for row in stored_rows) != list(range(len(stored_rows))):
            raise ValueError("P-L replay stored rows are not contiguous within a shard")
        residuals = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if (
            residuals.ndim != 3
            or residuals.dtype != np.float32
            or residuals.shape[0] != len(stored_rows)
            or tuple(residuals.shape[1:]) != expected_residual_shape
        ):
            raise ValueError("P-L replay residual shape/dtype differs from its stored index or model")
        stored_by_shard[shard] = sorted(stored_rows, key=lambda row: int(row["array_row"]))

    source_shards = set(source_by_shard)
    stored_shards = set(stored_by_shard)
    if source_shards != stored_shards:
        raise ValueError("P-L replay source and stored shard sets differ")
    row_order_matches = True
    for shard in sorted(source_shards):
        source_rows = source_by_shard[shard]
        stored_rows = stored_by_shard[shard]
        if len(source_rows) != len(stored_rows):
            row_order_matches = False
            break
        if any(
            _pl_row_identity(source, stored=False) != _pl_row_identity(stored, stored=True)
            for source, stored in zip(source_rows, stored_rows, strict=True)
        ):
            row_order_matches = False
            break
    main_count = sum(not _is_extension(row.get("m1_extension", False)) for row in source_records)
    discarded_count = sum(_is_extension(row.get("m1_extension", False)) for row in source_records)
    expected_main = int(cfg["e7"]["expected_prompt_counts"]["main"][format_name])
    if main_count != expected_main or main_count + discarded_count != sum(map(len, stored_by_shard.values())):
        row_order_matches = False
    if sum(len(batch["records"]) for batch in batches) != len(source_records):
        row_order_matches = False
    return {
        "source": "pl",
        "batch_size": batch_size,
        "batch_count": len(batches),
        "row_count": len(source_records),
        "e7_rows": main_count,
        "discarded_rows": discarded_count,
        "row_order_matches": row_order_matches,
    }


def _verify_pd_replay_cell(
    cfg: Mapping[str, Any],
    family: str,
    format_name: str,
    all_direction_records: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate P-D's source order and return selected exact replay batches."""
    import pyarrow.parquet as pq

    index_path, sidecar_path, _residual_path = _pd_paths(cfg, family, format_name)
    stored_rows = pq.read_table(index_path).to_pylist()
    ordered_records = _pd_ordered_replay_records(format_name, all_direction_records, stored_rows)
    source, batch_size = _replay_source_and_batch_size(cfg, family, "smoke")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if source != "pd" or int(sidecar["batch_size"]) != batch_size:
        raise ValueError("configured P-D replay batch size differs from its provenance")
    selected_rows = {int(row["pd_array_row"]) for row in selected_records}
    batches = _pd_replay_batches(cfg, family, ordered_records)
    selected_batches: list[dict[str, Any]] = []
    for batch in batches:
        selected_local = [
            index for index, row in enumerate(batch["records"])
            if int(row["pd_array_row"]) in selected_rows
        ]
        if selected_local:
            selected_batches.append({**batch, "selected_local_rows": selected_local})
    return {
        "source": "pd",
        "batch_size": batch_size,
        "batch_count": len(selected_batches),
        "row_count": sum(len(batch["records"]) for batch in selected_batches),
        "row_order_matches": True,
    }, selected_batches


def _replay_preflight_table(
    cfg: Mapping[str, Any],
    smoke_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    all_direction_records: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, bool, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Verify stored source row order and report configured replay batch layouts."""
    pl_records = _pl_source_prompt_records(cfg)
    rows: list[tuple[str, ...]] = []
    all_ok = True
    trial_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        family: {} for family in FAMILIES
    }
    for family in FAMILIES:
        for format_name in FORMATS:
            cell = f"{family}/{format_name}"
            try:
                pl_layout = _verify_pl_replay_cell(cfg, family, format_name, pl_records)
                pl_source = _pl_source_cell_records(cfg, pl_records, format_name)
                pl_batches = _pl_replay_batches(cfg, family, pl_source)
                first_batch = pl_batches[0]["records"] if pl_batches else []
                trial = [
                    dict(record) for record in first_batch
                    if not _is_extension(record.get("m1_extension", False))
                ]
                if not trial:
                    raise ValueError("first P-L capture batch has no E7 rows")
                trial_rows[family][format_name] = trial
                pl_status = "PASS" if pl_layout["row_order_matches"] else "FAIL"
                cell_ok = bool(pl_layout["row_order_matches"])
                pd_text = "not required"
                if all_direction_records is not None:
                    pd_layout, _pd_batches = _verify_pd_replay_cell(
                        cfg, family, format_name, all_direction_records,
                        smoke_rows[family][format_name],
                    )
                    pd_status = "PASS" if pd_layout["row_order_matches"] else "FAIL"
                    pd_text = (
                        f"{pd_layout['batch_count']} batches/{pd_layout['row_count']} rows; "
                        f"B={pd_layout['batch_size']}; order={pd_status}"
                    )
                    cell_ok &= bool(pd_layout["row_order_matches"])
                all_ok &= cell_ok
                rows.append((
                    cell,
                    f"{pl_layout['batch_count']} batches/{pl_layout['row_count']} rows; "
                    f"E7={pl_layout['e7_rows']}; discarded={pl_layout['discarded_rows']}; "
                    f"B={pl_layout['batch_size']}; order={pl_status}",
                    str(len(trial)),
                    pd_text,
                    "PASS" if cell_ok else "FAIL",
                ))
            except Exception as exc:
                all_ok = False
                trial_rows[family][format_name] = []
                details = _exception_details(exc)
                rows.append((
                    cell,
                    f"ERROR {details['error_type']}: {details['error_message']}",
                    "0",
                    details["innermost_traceback_frame"] if all_direction_records is not None else "not required",
                    "FAIL",
                ))
    headers = ("CELL", "P-L replay layout and stored row order", "Trial E7 rows", "P-D replay layout and row order", "STATUS")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    table = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    table += "\n" + "-+-".join("-" * widths[i] for i in range(len(headers)))
    table += "\n" + "\n".join(
        " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows
    )
    return table, all_ok, trial_rows


def _reconstruct_selected_batches(
    ordered_records: Sequence[Mapping[str, Any]],
    selected_rows: set[int],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Rebuild exact consecutive source batches that contain selected array rows."""
    if batch_size <= 0:
        raise ValueError("P-D batch size must be positive")
    selected = {int(value) for value in selected_rows}
    if any(int(row.get("pd_array_row", -1)) != index for index, row in enumerate(ordered_records)):
        raise ValueError("P-D replay records are not in exact array-row order")
    if any(value < 0 or value >= len(ordered_records) for value in selected):
        raise ValueError("selected P-D array row is outside the replay records")
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(ordered_records), batch_size)):
        batch = list(ordered_records[start : start + batch_size])
        selected_local = [
            local for local, row in enumerate(batch)
            if int(row["pd_array_row"]) in selected
        ]
        if selected_local:
            batches.append({
                "batch_index": batch_index,
                "start_row": start,
                "records": batch,
                "selected_local_rows": selected_local,
            })
    return batches


def _diagnostic_cell_layout(
    cfg: Mapping[str, Any],
    family: str,
    format_name: str,
    selected_records: Sequence[Mapping[str, Any]],
    all_prompt_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve prompt lengths, P-D batches, and P-A batches from stored row metadata."""
    import pyarrow.parquet as pq

    index_path, sidecar_path, _residual_path = _pd_paths(cfg, family, format_name)
    index_rows = pq.read_table(
        index_path,
        columns=["row", "concept_id", "lang", "format", "split", "n_tokens", "n_special"],
    ).to_pylist()
    index_rows.sort(key=lambda row: int(row["row"]))
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    pd_batch_size = int(provenance["batch_size"])
    if not index_rows or pd_batch_size <= 0:
        raise ValueError("P-D replay metadata has no rows or an invalid batch size")

    prompt_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in all_prompt_records:
        row = dict(raw)
        key = _pd_stable_sort_key(row)
        if key[0] != format_name:
            continue
        if key in prompt_by_key:
            raise ValueError(f"duplicate directions metadata for {family}/{format_name}")
        prompt_by_key[key] = row

    ordered_records: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    for expected_row, pd_row in enumerate(index_rows):
        array_row = int(pd_row["row"])
        if array_row != expected_row or str(pd_row["format"]) != format_name:
            raise ValueError("P-D index rows are not a contiguous format-specific array order")
        key = _pd_stable_sort_key(pd_row)
        if key not in prompt_by_key:
            raise ValueError("P-D replay row has no matching directions prompt metadata")
        record = dict(prompt_by_key[key])
        if str(record.get("split")) != str(pd_row["split"]):
            raise ValueError("P-D replay split differs from directions prompt metadata")
        if str(record.get("split")) not in {"fit", "validate"}:
            raise ValueError("P-D replay includes a non-directions split row")
        token_length = int(pd_row["n_tokens"]) + int(pd_row["n_special"])
        if token_length <= 0:
            raise ValueError("P-D index contains a non-positive prompt token length")
        record["pd_array_row"] = array_row
        record["_diagnostic_token_length"] = token_length
        ordered_records.append(record)
        token_lengths.append(token_length)

    selected = [dict(row) for row in selected_records]
    if any(str(row.get("split")) == "test" for row in selected):
        raise AssertionError("diagnostic selection contains a test-split record")
    selected_rows = [int(row["pd_array_row"]) for row in selected]
    if len(selected_rows) != len(set(selected_rows)):
        raise ValueError("diagnostic selection contains duplicate P-D rows")
    selected_set = set(selected_rows)
    selected_position = {row: index for index, row in enumerate(selected_rows)}
    selected_lengths = [token_lengths[row] for row in selected_rows]
    pa_batch_size = int(cfg["e7"]["batch_size"][family])
    prompt_facts: list[dict[str, Any]] = []
    for prompt_index, (record, array_row, token_length) in enumerate(
        zip(selected, selected_rows, selected_lengths, strict=True)
    ):
        pd_batch_index = array_row // pd_batch_size
        pd_start = pd_batch_index * pd_batch_size
        pd_max_length = max(token_lengths[pd_start : pd_start + pd_batch_size])
        pa_batch_index = prompt_index // pa_batch_size
        pa_start = pa_batch_index * pa_batch_size
        pa_max_length = max(selected_lengths[pa_start : pa_start + pa_batch_size])
        prompt_facts.append({
            "prompt_index": prompt_index,
            "pd_array_row": array_row,
            "token_length": token_length,
            "pd_batch_index": pd_batch_index,
            "pd_batch_max_length": pd_max_length,
            "pd_padded": token_length < pd_max_length,
            "pa_batch_index": pa_batch_index,
            "pa_batch_max_length": pa_max_length,
            "pa_padded": token_length < pa_max_length,
        })
        record["_diagnostic_prompt_index"] = prompt_index
        record["_diagnostic_token_length"] = token_length

    for record in ordered_records:
        array_row = int(record["pd_array_row"])
        if array_row in selected_position:
            record["_diagnostic_prompt_index"] = selected_position[array_row]

    replay_batches = _reconstruct_selected_batches(
        ordered_records, selected_set, pd_batch_size
    )
    for batch in replay_batches:
        batch_records = batch["records"]
        batch_lengths = [int(row["_diagnostic_token_length"]) for row in batch_records]
        batch["max_length"] = max(batch_lengths)
        batch["padded_by_row"] = [length < batch["max_length"] for length in batch_lengths]

    padding_counts = {"pd_only": 0, "pa_only": 0, "both": 0, "neither": 0}
    for fact in prompt_facts:
        if fact["pd_padded"] and fact["pa_padded"]:
            padding_counts["both"] += 1
        elif fact["pd_padded"]:
            padding_counts["pd_only"] += 1
        elif fact["pa_padded"]:
            padding_counts["pa_only"] += 1
        else:
            padding_counts["neither"] += 1

    return {
        "pd_batch_size": pd_batch_size,
        "pa_batch_size": pa_batch_size,
        "ordered_records": ordered_records,
        "replay_batches": replay_batches,
        "prompt_facts": prompt_facts,
        "selected_records": selected,
        "padding_counts": padding_counts,
        "replay_batch_count": len(replay_batches),
        "replay_prompt_count": sum(len(batch["records"]) for batch in replay_batches),
        "selected_pd_batch_indices": [int(batch["batch_index"]) for batch in replay_batches],
    }


def _smoke_diagnose_preflight(
    cfg: Mapping[str, Any],
    smoke_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[str, bool]:
    """Report numeric prompt and replay-batch layouts without loading model weights."""
    from vilens.data import load_prompt_file, prompt_file_paths

    prompt_path = prompt_file_paths(dict(cfg))["directions_matched"]
    if not prompt_path.is_absolute():
        prompt_path = REPO / prompt_path
    all_prompts = load_prompt_file(prompt_path)
    cell_lines = [
        "CELL | split | rows | P-D batch size | S2 batches | S2 rows | pad P-D only / P-A only / both / neither"
    ]
    all_ok = True
    for family in FAMILIES:
        for format_name in FORMATS:
            try:
                layout = _diagnostic_cell_layout(
                    cfg, family, format_name, smoke_rows[family][format_name], all_prompts
                )
                split_values = sorted({str(row.get("split")) for row in smoke_rows[family][format_name]})
                split = ",".join(split_values)
                if split != "validate":
                    raise ValueError(f"diagnostic requires validate rows; found {split}")
                counts = layout["padding_counts"]
                cell_lines.append(
                    f"{family}/{format_name} | {split} | {len(layout['prompt_facts'])} | "
                    f"{layout['pd_batch_size']} | {layout['replay_batch_count']} | "
                    f"{layout['replay_prompt_count']} | {counts['pd_only']} / {counts['pa_only']} / "
                    f"{counts['both']} / {counts['neither']}"
                )
                for fact in layout["prompt_facts"]:
                    cell_lines.append(
                        f"{family}/{format_name} | {fact['prompt_index']} | {fact['token_length']} | "
                        f"{fact['pd_array_row']} | {fact['pd_batch_index']} | {fact['pd_batch_max_length']} | "
                        f"{int(fact['pd_padded'])} | {fact['pa_batch_index']} | "
                        f"{fact['pa_batch_max_length']} | {int(fact['pa_padded'])}"
                    )
            except Exception as exc:
                all_ok = False
                details = _exception_details(exc)
                cell_lines.append(
                    f"{family}/{format_name} | ERROR {details['error_type']}: {details['error_message']} "
                    f"[{details['innermost_traceback_frame']}]"
                )
    return "\n".join(cell_lines), all_ok


def _e8_vector_path(cfg: Mapping[str, Any], family: str, format_name: str) -> Path:
    """Resolve the committed E8 vector array for its matching family and format."""
    return REPO / str(cfg["e7"]["e8_results_dir"]) / f"family={family}__format={format_name}__vectors.npy"


def _pl_residual_references(
    cfg: Mapping[str, Any], family: str, format_name: str
) -> dict[str, tuple[np.ndarray, int, Path]]:
    """Memory-map the matching P-L residual shards and index them by concept ID."""
    import pyarrow.parquet as pq

    root = REPO / str(cfg["e7"]["p_l_results_dir"])
    prefix = f"condition=diac__family={family}__fewshot_set=1__format={format_name}__pass=pl__shard="
    references: dict[str, tuple[np.ndarray, int, Path]] = {}
    for array_path in sorted(root.glob(prefix + "*.resid.npy")):
        index_path = array_path.with_name(array_path.name.removesuffix(".resid.npy") + ".resid_index.jsonl")
        sidecar_path = array_path.with_name(array_path.name.removesuffix(".resid.npy") + ".provenance.json")
        if not index_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"P-L residual shard lacks index/sidecar: {array_path.name}")
        residual = np.load(array_path, mmap_mode="r", allow_pickle=False)
        rows = _read_jsonl(index_path)
        if residual.ndim != 3 or residual.dtype != np.float32 or residual.shape[0] != len(rows):
            raise ValueError(f"P-L residual shape/dtype differs from its index: {array_path.name}")
        for row in rows:
            if str(row.get("format")) != format_name or str(row.get("condition", "diac")) != "diac":
                continue
            if str(row.get("fewshot_set")) != "1" or str(row.get("split")) != "test":
                continue
            if _is_extension(row.get("m1_extension", False)):
                continue
            concept = str(row["concept_id"])
            if concept in references:
                raise ValueError(f"duplicate primary P-L residual concept for {family}/{format_name}")
            array_row = int(row["array_row"])
            if not 0 <= array_row < residual.shape[0]:
                raise ValueError(f"invalid P-L array_row for concept {concept}")
            references[concept] = (residual, array_row, sidecar_path)
    if not references:
        raise FileNotFoundError(f"No primary P-L residual rows for {family}/{format_name}")
    return references


def _head_config(loaded: LoadedModel) -> tuple[int, int, list[str]]:
    """Return the pinned attention geometry and one type label per block."""
    text_config = loaded.model.config.get_text_config()
    heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    if getattr(text_config, "layer_types", None):
        layer_types = [
            "local" if value == "sliding_attention" else "global"
            for value in text_config.layer_types
        ]
    else:
        layer_types = ["standard"] * loaded.n_layers
    if len(layer_types) != loaded.n_layers:
        raise ValueError("pinned model layer-type metadata does not match the decoder depth")
    return heads, head_dim, layer_types


def _first_tensor(output: Any) -> torch.Tensor:
    """Return a decoder module's hidden-state tensor."""
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError("decoder activation must be a [batch, sequence, hidden] tensor")
    return value


def _last(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather each right-padded prompt's final real-token activation."""
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, positions, :].detach()


def _capture_batch(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    *,
    v_vi: np.ndarray,
    d_vi_en: np.ndarray,
    heads: int,
    head_dim: int,
    layer_types: Sequence[str],
    retain_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    """Capture prompt-last-token components and per-head projections in one forward."""
    from vilens.passes import _encode_prompts, _pad_id, _pad_sequences

    device = next(loaded.model.parameters()).device
    sequences = _encode_prompts(loaded, [str(row["prompt"]) for row in records])
    input_ids, attention_mask, lengths = _pad_sequences(sequences, _pad_id(loaded), device)
    positions = torch.as_tensor([length - 1 for length in lengths], device=device, dtype=torch.long)
    direction_tensor = torch.as_tensor(np.stack([v_vi, d_vi_en], axis=1), device=device, dtype=torch.float32)
    batch_size = len(records)
    kept_rows = list(range(batch_size)) if retain_indices is None else [int(value) for value in retain_indices]
    if len(kept_rows) != len(set(kept_rows)) or any(value < 0 or value >= batch_size for value in kept_rows):
        raise ValueError("retained E7 row indices must be unique rows in the captured batch")
    kept_tensor = torch.as_tensor(kept_rows, dtype=torch.long, device=device)
    layer_states: list[dict[str, torch.Tensor]] = [{} for _ in loaded.layers]
    residual_values: list[torch.Tensor | None] = [None] * (loaded.n_layers + 1)
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[np.ndarray]] = {
        "head_sum": [],
        "residual_model_order": [],
        "residual_update_relative": [],
    }
    handles: list[Any] = []
    gemma = loaded.family == "gemma"

    def pre_hook(index: int):
        def capture(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise TypeError("decoder block input does not expose hidden_states")
            selected = _last(hidden, positions)
            layer_states[index]["resid_in"] = selected
            residual_values[index] = selected
        return capture

    def o_proj_pre_hook(index: int):
        def capture(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise TypeError("o_proj received no input tensor")
            layer_states[index]["z"] = _last(args[0], positions)
        return capture

    def module_hook(index: int, key: str):
        def capture(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            layer_states[index][key] = _last(_first_tensor(output), positions)
        return capture

    def block_hook(index: int):
        def capture(module: Any, _args: tuple[Any, ...], output: Any) -> None:
            state = layer_states[index]
            block_output = _last(_first_tensor(output), positions)
            residual_values[index + 1] = block_output
            attn = state.get("attn")
            mlp = state.get("mlp")
            z = state.get("z")
            raw_attention = state.get("raw_attention")
            needed = (attn, mlp, z)
            if any(value is None for value in needed):
                raise RuntimeError(f"component hooks incomplete at block {index + 1}")
            layer = index + 1
            directions = direction_tensor[layer]
            attn_kept = attn.index_select(0, kept_tensor)
            mlp_kept = mlp.index_select(0, kept_tensor)
            z_kept = z.index_select(0, kept_tensor)
            raw_attention_kept = raw_attention.index_select(0, kept_tensor) if gemma else None
            block_output_kept = block_output.index_select(0, kept_tensor)
            resid_in_kept = state["resid_in"].index_select(0, kept_tensor)
            attn_projection = attribute.project_vectors(attn_kept, directions)
            mlp_projection = attribute.project_vectors(mlp_kept, directions)
            attn_norm = attribute.vector_norms(attn_kept)
            mlp_norm = attribute.vector_norms(mlp_kept)
            o_proj = module.self_attn.o_proj
            if o_proj.bias is not None:
                raise ValueError("OPS-37 per-head o_proj slicing requires the pinned no-bias projection")
            norm_weight = module.post_attention_layernorm.weight if gemma else None
            eps = float(getattr(module.post_attention_layernorm, "eps", 1e-6)) if gemma else 1e-6
            head_contribs = attribute.attention_head_contributions(
                z_kept,
                o_proj.weight,
                n_heads=heads,
                head_dim=head_dim,
                total_attention=raw_attention_kept,
                gemma_post_norm_weight=norm_weight,
                gemma_post_norm_eps=eps,
            )
            head_projection = attribute.project_vectors(head_contribs, directions)
            head_norm = attribute.vector_norms(head_contribs)
            head_sum = head_contribs.sum(dim=-2)
            component_errors = attribute.captured_component_errors(
                resid_in_kept, attn_kept, mlp_kept, block_output_kept, head_sum
            )
            for error_name in errors:
                errors[error_name].append(component_errors[error_name])
            attn_projection_np = attn_projection.detach().float().cpu().numpy()
            mlp_projection_np = mlp_projection.detach().float().cpu().numpy()
            attn_norm_np = attn_norm.detach().float().cpu().numpy()
            mlp_norm_np = mlp_norm.detach().float().cpu().numpy()
            head_projection_np = head_projection.detach().float().cpu().numpy()
            head_norm_np = head_norm.detach().float().cpu().numpy()
            for output_row, batch_row in enumerate(kept_rows):
                record = records[batch_row]
                base = {
                    "model": loaded.spec.hf_id if loaded.spec else loaded.family,
                    "model_revision": loaded.spec.revision if loaded.spec else "synthetic",
                    "pass": "pa",
                    "format": str(record["format"]),
                    "item_set": "main",
                    "concept_id": str(record["concept_id"]),
                    "layer": layer,
                    "layer_type": str(layer_types[index]),
                }
                for direction_index, direction_name in enumerate(attribute.DIRECTIONS):
                    for kind, projection_values, norm_value in (
                        ("attn", attn_projection_np, float(attn_norm_np[output_row])),
                        ("mlp", mlp_projection_np, float(mlp_norm_np[output_row])),
                    ):
                        rows.append({
                            **base, "kind": kind, "head": -1, "direction": direction_name,
                            "projection": float(projection_values[output_row, direction_index]),
                            "update_norm": norm_value,
                        })
                    for head_index in range(heads):
                        rows.append({
                            **base, "kind": "head", "head": head_index, "direction": direction_name,
                            "projection": float(head_projection_np[output_row, head_index, direction_index]),
                            "update_norm": float(head_norm_np[output_row, head_index]),
                        })
        return capture

    try:
        for index, layer in enumerate(loaded.layers):
            handles.append(layer.register_forward_pre_hook(pre_hook(index), with_kwargs=True))
            handles.append(layer.register_forward_hook(block_hook(index)))
            attn_module = layer.self_attn
            handles.append(attn_module.register_forward_hook(module_hook(index, "raw_attention" if gemma else "attn")))
            handles.append(attn_module.o_proj.register_forward_pre_hook(o_proj_pre_hook(index)))
            if gemma:
                handles.append(layer.post_attention_layernorm.register_forward_hook(module_hook(index, "attn")))
                handles.append(layer.post_feedforward_layernorm.register_forward_hook(module_hook(index, "mlp")))
            else:
                handles.append(layer.mlp.register_forward_hook(module_hook(index, "mlp")))
        with torch.inference_mode():
            loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in residual_values):
        raise RuntimeError("one or more residual-layer hooks did not run")
    residual_stack = torch.stack([value for value in residual_values if value is not None], dim=1)
    error_arrays = {
        name: np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        for name, values in errors.items()
    }
    error_arrays["residual_update_relative_by_block"] = (
        np.stack(errors["residual_update_relative"], axis=1)
        if errors["residual_update_relative"]
        else np.empty((len(kept_rows), 0), dtype=np.float64)
    )
    return rows, residual_stack.detach().float().cpu().numpy(), error_arrays


def _check_summary(
    batch_results: Sequence[Mapping[str, np.ndarray]],
    cfg: Mapping[str, Any],
    *,
    p_l_errors: Sequence[float] | None = None,
    p_d_errors: Sequence[float] | None = None,
    p_l_replay_errors: Sequence[float] | None = None,
    p_d_replay_errors: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize OPS-37 checks with the OPS-38 model-order 10(b) gate."""
    settings = cfg["e7"]
    head_values = np.concatenate([np.asarray(value["head_sum"]) for value in batch_results])
    model_order_values = np.concatenate(
        [np.asarray(value["residual_model_order"]) for value in batch_results]
    )
    update_values = np.concatenate(
        [np.asarray(value["residual_update_relative"]) for value in batch_results]
    )
    update_by_block = np.concatenate(
        [np.asarray(value["residual_update_relative_by_block"]) for value in batch_results], axis=0
    )
    head_summary = attribute.error_summary(
        head_values,
        median_limit=float(settings["check_head_median_relative"]),
        p99_limit=float(settings["check_head_p99_relative"]),
    )
    head_summary["passed"] = bool(head_summary["passed"] and np.isfinite(head_values).all())
    model_limit = float(settings["check_residual_model_order_max"])
    finite_model_order = np.isfinite(model_order_values)
    model_max = float(np.max(model_order_values)) if model_order_values.size else float("nan")
    model_summary = {
        "maximum": model_max,
        "limit": model_limit,
        "passed": bool(finite_model_order.all() and model_order_values.size and model_max < model_limit),
    }
    finite_update = update_values[np.isfinite(update_values)]
    per_block_medians = {
        str(block + 1): float(np.median(update_by_block[:, block]))
        for block in range(update_by_block.shape[1])
    }
    block_median_values = np.asarray(list(per_block_medians.values()), dtype=np.float64)
    out: dict[str, dict[str, Any]] = {
        "head_contributions": head_summary,
        "residual_model_order": model_summary,
        "update_relative_diagnostic": {
            "median": float(np.median(finite_update)) if finite_update.size else float("nan"),
            "p99": float(np.quantile(finite_update, 0.99)) if finite_update.size else float("nan"),
            "maximum": float(np.max(finite_update)) if finite_update.size else float("nan"),
            "per_block_median": per_block_medians,
            "per_block_median_min": float(np.min(block_median_values)) if block_median_values.size else float("nan"),
            "per_block_median_median": float(np.median(block_median_values)) if block_median_values.size else float("nan"),
            "per_block_median_max": float(np.max(block_median_values)) if block_median_values.size else float("nan"),
            "gated": False,
            "nonfinite_count": int((~np.isfinite(update_values)).sum()),
        },
    }
    if p_l_errors is not None:
        out["pl_residual_match"] = attribute.error_summary(
            p_l_errors, median_limit=float(settings["check_pl_median_relative"])
        )
        out["pl_residual_match"]["passed"] = bool(
            out["pl_residual_match"]["passed"] and np.isfinite(np.asarray(p_l_errors, dtype=np.float64)).all()
        )
    if p_l_replay_errors is not None:
        out["pl_residual_replay"] = attribute.exact_replay_gate_summary(
            p_l_replay_errors, float(settings["check_residual_replay_max_relative"])
        )
    if p_d_errors is not None:
        out["pd_residual_match"] = attribute.error_summary(
            p_d_errors, median_limit=float(settings["check_pl_median_relative"])
        )
        out["pd_residual_match"]["passed"] = bool(
            out["pd_residual_match"]["passed"] and np.isfinite(np.asarray(p_d_errors, dtype=np.float64)).all()
        )
    if p_d_replay_errors is not None:
        out["pd_residual_replay"] = attribute.exact_replay_gate_summary(
            p_d_replay_errors, float(settings["check_residual_replay_max_relative"])
        )
    return out


def _p_l_error_for_batch(
    residuals: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    references: Mapping[str, tuple[np.ndarray, int, Path]],
) -> list[float]:
    """Compare each P-A last-token residual stack with its P-L stored counterpart."""
    errors: list[float] = []
    for row, record in enumerate(records):
        concept = str(record["concept_id"])
        if concept not in references:
            raise FileNotFoundError(f"P-L residual reference missing concept {concept}")
        reference_array, array_row, _sidecar = references[concept]
        reference = np.asarray(reference_array[array_row], dtype=np.float32)
        if reference.shape != residuals[row].shape:
            raise ValueError("P-A and P-L residual layer shapes differ")
        errors.extend(attribute.relative_errors(reference, residuals[row]).tolist())
    return errors


def _e7_replay_indices(
    records: Sequence[Mapping[str, Any]], e7_prompt_keys: set[tuple[str, ...]]
) -> list[int]:
    """Select E7 main rows for projection while retaining source rows for the forward."""
    return [
        index for index, record in enumerate(records)
        if not _is_extension(record.get("m1_extension", False))
        and _e7_prompt_key(record) in e7_prompt_keys
    ]


def _p_d_error_for_batch(
    residuals: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    pd_residuals: np.ndarray,
) -> list[float]:
    """Compare smoke P-A residual stacks with their exact stored P-D array rows."""
    errors: list[float] = []
    for row, record in enumerate(records):
        array_row = int(record["pd_array_row"])
        if not 0 <= array_row < pd_residuals.shape[0]:
            raise ValueError("selected smoke prompt has an invalid P-D residual row")
        reference = np.asarray(pd_residuals[array_row], dtype=np.float32)
        if reference.shape != residuals[row].shape:
            raise ValueError("P-A and P-D residual layer shapes differ")
        errors.extend(attribute.relative_errors(reference, residuals[row]).tolist())
    return errors


def _load_a6_directions(
    cfg: Mapping[str, Any], family: str, format_name: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int], float, str, float, float]:
    """Load P-D fit rows and rebuild the A6 directions for every residual layer."""
    import pyarrow.parquet as pq

    index_path, sidecar_path, residual_path = _pd_paths(cfg, family, format_name)
    if not all(path.is_file() for path in (index_path, sidecar_path, residual_path)):
        raise FileNotFoundError(f"P-D fit inputs missing for {family}/{format_name}")
    index_rows = pq.read_table(index_path, columns=["row", "concept_id", "lang", "format", "split"]).to_pylist()
    residuals = np.load(residual_path, mmap_mode="r", allow_pickle=False)
    v_vi, d_vi_en, counts = attribute.rebuild_a6_directions(
        residuals, index_rows, format_name=format_name
    )
    e8_path = _e8_vector_path(cfg, family, format_name)
    if not e8_path.is_file():
        raise FileNotFoundError(f"E8 vector is missing: {e8_path}")
    e8_u = np.load(e8_path, mmap_mode="r", allow_pickle=False)
    # OPS-37 compares the unnormalized class-mean contrast to E8's unnormalized -u.
    # Rebuild the raw contrast through the same fit path for this exact check.
    from vilens.analysis.la import fit_layer_directions
    fit_mask = [row for row in index_rows if str(row.get("format")) == format_name and str(row.get("split")) == "fit"]
    fit_rows = [int(row["row"]) for row in fit_mask]
    labels = [str(row["lang"]) for row in fit_mask]
    raw_d = np.empty_like(d_vi_en)
    for layer in range(residuals.shape[1]):
        _center, _directions, means = fit_layer_directions(
            np.asarray(residuals[fit_rows, layer, :], dtype=np.float32), labels
        )
        raw_d[layer] = means["vi"] - means["en"]
    relative = attribute.direction_match_max_relative(raw_d, e8_u)
    max_abs_diff = float(np.max(np.abs(raw_d + e8_u)))
    max_abs_d = float(np.max(np.abs(raw_d)))
    return v_vi, d_vi_en, counts, relative, sha256_file(e8_path), max_abs_diff, max_abs_d


def _fit_overlap_count(
    cfg: Mapping[str, Any], family: str, format_name: str, e7_ids: set[str]
) -> tuple[int, str]:
    """Count concepts shared with the P-D fit split and return the D12 documentation note."""
    import pyarrow.parquet as pq

    index_path = REPO / str(cfg["e7"]["p_d_results_dir"]) / f"family={family}__format={format_name}__pass=pd__source=matched.parquet"
    rows = pq.read_table(index_path, columns=["concept_id", "format", "split"]).to_pylist()
    fit_ids = {
        str(row["concept_id"]) for row in rows
        if str(row.get("format")) == format_name and str(row.get("split")) == "fit"
    }
    overlap = len(fit_ids & e7_ids)
    note = (
        "D12 note: five translation-format Vietnamese directions records have byte-identical prompts to test prompts; "
        "the fit-split concept overlap check is by concept_id."
    )
    return overlap, note


def _input_table(
    cfg: Mapping[str, Any], prompt_rows: Mapping[str, Any], *, smoke: bool = False
) -> tuple[str, bool]:
    """Check mode-specific P-D/P-L, E8, manifest, and output inputs without weights."""
    rows: list[tuple[str, str, str]] = []
    if not smoke:
        flags_path = REPO / "derived/cognate_flags.jsonl"
        rows.append(("cf729 item manifest", str(flags_path.relative_to(REPO)), "PASS" if flags_path.is_file() else "FAIL"))
    for family in FAMILIES:
        for fmt in FORMATS:
            pd_index, pd_sidecar, pd_resid = _pd_paths(cfg, family, fmt)
            pd_ok = all(path.is_file() for path in (pd_index, pd_sidecar, pd_resid))
            pd_label = "P-D smoke residuals" if smoke else "P-D fit data"
            rows.append((pd_label, f"{family}/{fmt}: " + ("present" if pd_ok else "missing index/sidecar/residual"), "PASS" if pd_ok else "FAIL"))
            e8 = _e8_vector_path(cfg, family, fmt)
            rows.append(("E8 contrast vector", f"{family}/{fmt}: {e8.relative_to(REPO)}", "PASS" if e8.is_file() else "FAIL"))
            if smoke:
                count = len(prompt_rows.get(family, {}).get(fmt, []))
                needed = int(cfg["e7"]["trial_items_per_cell"])
                rows.append(("Directions smoke prompts", f"{family}/{fmt}: {count}/{needed}", "PASS" if count == needed else "FAIL"))
            else:
                root = REPO / str(cfg["e7"]["p_l_results_dir"])
                prefix = f"condition=diac__family={family}__fewshot_set=1__format={fmt}__pass=pl__shard="
                shards = sorted(root.glob(prefix + "*.resid.npy"))
                shard_inputs_ok = bool(shards) and all(
                    path.with_name(path.name.removesuffix(".resid.npy") + ".resid_index.jsonl").is_file()
                    and path.with_name(path.name.removesuffix(".resid.npy") + ".provenance.json").is_file()
                    and path.with_name(path.name.removesuffix(".resid.npy") + ".attn.npy").is_file()
                    and path.with_name(path.name.removesuffix(".resid.npy") + ".mlp.npy").is_file()
                    for path in shards
                )
                rows.append(("P-L component shards", f"{family}/{fmt}: {len(shards)} residual shards", "PASS" if shard_inputs_ok else "FAIL"))
                rows.append(("P-L prompt rows", f"{family}/{fmt}: {len(prompt_rows[fmt])}", "PASS" if len(prompt_rows[fmt]) == int(cfg["e7"]["expected_prompt_counts"]["main"][fmt]) else "FAIL"))
    rows.append(("Checkpoint H lock", "reports/adjudication.md", "PASS" if (REPO / "reports/adjudication.md").is_file() else "FAIL"))
    headers = ("CHECK", "INPUT / COUNT", "STATUS")
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(3)]
    table = " | ".join(headers[index].ljust(widths[index]) for index in range(3))
    table += "\n" + "-+-".join("-" * widths[index] for index in range(3))
    table += "\n" + "\n".join(" | ".join(value[index].ljust(widths[index]) for index in range(3)) for value in rows)
    return table, all(row[2] == "PASS" for row in rows)


def _weight_free_pipeline_checks(
    cfg: Mapping[str, Any],
    smoke_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    trial_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    primary_ids: Mapping[str, set[str]] | None = None,
) -> tuple[str, bool]:
    """Validate the real direction, P-D, E8, and optional P-L inputs without weights."""
    import pyarrow.parquet as pq

    headers = (
        "CELL", "SMOKE SPLIT", "ARRAY SHAPES", "P-D smoke lookup", "10(d)",
        "P-L trial lookup", "10(e)", "STATUS",
    )
    rows: list[tuple[str, ...]] = []
    all_ok = True
    expected_smoke_count = int(cfg["e7"]["trial_items_per_cell"])
    direction_limit = float(cfg["e7"]["check_direction_relative"])
    for family in FAMILIES:
        for format_name in FORMATS:
            cell = f"{family}/{format_name}"
            smoke_split = "unknown"
            shape_text = "not checked"
            pd_lookup = "not checked"
            direction = "not checked"
            pl_lookup = "not required"
            overlap_text = "not required"
            cell_ok = True
            try:
                n_layers, d_model, _vocab = map(
                    int, cfg["reference_facts"]["x0c"]["model_dims"][family]
                )
                expected_residual_shape = (n_layers + 1, d_model)
                pd_index_path, _pd_sidecar_path, pd_residual_path = _pd_paths(
                    cfg, family, format_name
                )
                pd_index_rows = pq.read_table(
                    pd_index_path,
                    columns=["row", "concept_id", "lang", "format", "split"],
                ).to_pylist()
                pd_residuals = np.load(pd_residual_path, mmap_mode="r", allow_pickle=False)
                e8_vectors = np.load(
                    _e8_vector_path(cfg, family, format_name), mmap_mode="r", allow_pickle=False
                )
                expected_pd_shape = (len(pd_index_rows), n_layers + 1, d_model)
                if pd_residuals.shape != expected_pd_shape or pd_residuals.dtype != np.float32:
                    raise ValueError(
                        f"P-D residual shape/dtype {pd_residuals.shape}/{pd_residuals.dtype} "
                        f"does not match {expected_pd_shape}/float32"
                    )
                if sorted(int(row["row"]) for row in pd_index_rows) != list(range(len(pd_index_rows))):
                    raise ValueError("P-D row coordinates are not a complete zero-based array index")
                expected_e8_shape = expected_residual_shape
                if e8_vectors.shape != expected_e8_shape or e8_vectors.dtype != np.float32:
                    raise ValueError(
                        f"E8 vector shape/dtype {e8_vectors.shape}/{e8_vectors.dtype} "
                        f"does not match {expected_e8_shape}/float32"
                    )
                v_vi, d_vi_en, _counts, d_relative, _sha, max_abs_diff, max_abs_d = (
                    _load_a6_directions(cfg, family, format_name)
                )
                if v_vi.shape != expected_residual_shape or d_vi_en.shape != expected_residual_shape:
                    raise ValueError(
                        f"rebuilt direction shapes {v_vi.shape}/{d_vi_en.shape} "
                        f"do not match {expected_residual_shape}"
                    )
                smoke_cell_rows = list(smoke_rows[family][format_name])
                if len(smoke_cell_rows) != expected_smoke_count:
                    raise ValueError(
                        f"directions smoke lookup has {len(smoke_cell_rows)} rows; "
                        f"expected {expected_smoke_count}"
                    )
                smoke_splits = sorted({str(record.get("split")) for record in smoke_cell_rows})
                smoke_split = ",".join(smoke_splits)
                if len(smoke_splits) != 1 or smoke_splits[0] not in {"fit", "validate"}:
                    raise ValueError("directions smoke rows do not share one non-test split")
                smoke_indices: list[int] = []
                for record in smoke_cell_rows:
                    if str(record.get("split")) == "test":
                        raise ValueError("directions smoke lookup contains a test-split record")
                    array_row = int(record["pd_array_row"])
                    if not 0 <= array_row < len(pd_index_rows):
                        raise ValueError("directions smoke P-D row is outside the residual array")
                    source_row = pd_index_rows[array_row]
                    if (
                        int(source_row["row"]) != array_row
                        or str(source_row["concept_id"]) != str(record["concept_id"])
                        or str(source_row["lang"]) != str(record["lang"])
                        or str(source_row["format"]) != format_name
                        or str(source_row["split"]) != str(record["split"])
                    ):
                        raise ValueError("directions smoke P-D row metadata does not match its selected record")
                    smoke_indices.append(array_row)
                pd_lookup = f"{len(smoke_indices)}/{expected_smoke_count}; rows={smoke_indices}"
                passed_direction = bool(
                    np.isfinite(d_relative) and d_relative <= direction_limit
                )
                direction = (
                    f"max_abs_diff={max_abs_diff:.8g}; max|d|={max_abs_d:.8g}; "
                    f"relative={d_relative:.8g}; limit={direction_limit:g}; "
                    f"{'PASS' if passed_direction else 'FAIL'}"
                )
                cell_ok &= passed_direction
                shape_text = (
                    f"P-D={pd_residuals.shape}; E8={e8_vectors.shape}; "
                    f"directions={v_vi.shape}/{d_vi_en.shape}; dtype=float32"
                )
                del pd_residuals, e8_vectors, v_vi, d_vi_en

                if trial_rows is not None:
                    if primary_ids is None:
                        raise ValueError("P-A preflight requires the frozen primary ID set")
                    cell_trial_rows = list(trial_rows[family][format_name])
                    if not cell_trial_rows:
                        raise ValueError(
                            "P-A trial lookup from the first P-L capture batch is empty"
                        )
                    references = _pl_residual_references(cfg, family, format_name)
                    paths: dict[Path, tuple[int, ...]] = {}
                    for residual, _array_row, path in references.values():
                        paths[path] = tuple(int(value) for value in residual.shape)
                        if (
                            residual.ndim != 3
                            or residual.dtype != np.float32
                            or tuple(residual.shape[1:]) != expected_residual_shape
                        ):
                            raise ValueError(
                                f"P-L residual shape/dtype {residual.shape}/{residual.dtype} "
                                f"does not match [rows, {n_layers + 1}, {d_model}]/float32"
                            )
                    path_indices = {path: index for index, path in enumerate(sorted(paths))}
                    locations: list[str] = []
                    for record in cell_trial_rows:
                        concept_id = str(record["concept_id"])
                        if concept_id not in references:
                            raise ValueError("P-A trial prompt has no matching P-L residual reference")
                        _array, array_row, path = references[concept_id]
                        locations.append(f"{path_indices[path]}:{int(array_row)}")
                    pl_lookup = (
                        f"{len(locations)}/{len(cell_trial_rows)}; array_rows={locations}; "
                        f"shapes={sorted(set(paths.values()))}"
                    )
                    overlap, _d12_note = _fit_overlap_count(
                        cfg, family, format_name, set(primary_ids[format_name])
                    )
                    overlap_text = f"fit/E7 overlap={overlap}; limit=0; {'PASS' if overlap == 0 else 'FAIL'}"
                    cell_ok &= overlap == 0
                    del references
            except Exception as exc:
                cell_ok = False
                details = _exception_details(exc)
                shape_text = f"ERROR {details['error_type']}: {details['error_message']}"
                if trial_rows is not None and pl_lookup == "not required":
                    pl_lookup = details["innermost_traceback_frame"]
                if direction == "not checked":
                    direction = details["innermost_traceback_frame"]
                if trial_rows is not None and overlap_text == "not required":
                    overlap_text = details["innermost_traceback_frame"]
            status = "PASS" if cell_ok else "FAIL"
            all_ok &= cell_ok
            rows.append((cell, smoke_split, shape_text, pd_lookup, direction, pl_lookup, overlap_text, status))

    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    table = " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    table += "\n" + "-+-".join("-" * widths[index] for index in range(len(headers)))
    table += "\n" + "\n".join(
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows
    )
    return table, all_ok


def _preflight(
    cfg: dict[str, Any], prompt_rows: Mapping[str, Any], *, gpu: bool,
    smoke: bool = False,
    smoke_diagnose: bool = False,
    smoke_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    all_direction_records: Sequence[Mapping[str, Any]] | None = None,
    primary_ids: Mapping[str, set[str]] | None = None,
) -> bool:
    """Run shared job preflight and E7-specific input checks before any model load."""
    all_records = None if smoke else [dict(row) for fmt in FORMATS for row in prompt_rows[fmt]]
    prompt_file_paths_by_stem = prompt_file_paths(cfg)
    required_inputs = [
        str(prompt_file_paths_by_stem[str(cfg["data"]["primary_prompt_files"][f"{fmt}_diac"])])
        for fmt in FORMATS
    ]
    required_inputs.extend(
        str(prompt_file_paths_by_stem[str(stem)])
        for stem in cfg["pl"]["primary"]["files"]
    )
    output_dirs = [
        REPO / str(cfg["e7"]["results_dir"]),
        REPO / str(cfg["e7"]["trial_results_dir"]),
    ]
    if smoke:
        required_inputs = [
            str(prompt_file_paths_by_stem["directions_matched"]),
            *[
                str(prompt_file_paths_by_stem[str(stem)])
                for stem in cfg["pl"]["primary"]["files"]
            ],
        ]
        output_dirs = [REPO / str(cfg["e7"]["smoke_results_dir"])]
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E7",
        gpu=gpu,
        inputs=required_inputs,
        families=list(FAMILIES),
        records=all_records,
        output_dirs=output_dirs,
        entrypoint=Path(__file__),
    )
    print(format_checks(checks))
    custom, custom_ok = _input_table(cfg, prompt_rows, smoke=smoke)
    print("E7 P-A smoke input checks:" if smoke else "E7 P-A input checks:")
    print(custom)
    if smoke_rows is None:
        raise ValueError("E7 preflight requires the directions-split smoke prompt selection")
    replay_table, replay_ok, replay_trial_rows = _replay_preflight_table(
        cfg,
        smoke_rows,
        all_direction_records=all_direction_records if smoke else None,
    )
    print("E7 configured source replay validation (no prompt text):")
    print(replay_table)
    trial_rows = None
    if not smoke:
        if primary_ids is None:
            raise ValueError("P-A preflight requires the primary prompt ID set")
        trial_rows = replay_trial_rows
    weight_free, weight_free_ok = _weight_free_pipeline_checks(
        cfg, smoke_rows, trial_rows=trial_rows, primary_ids=primary_ids
    )
    print("E7 weight-free smoke checks:")
    print(weight_free)
    diagnose_ok = True
    if smoke_diagnose:
        if not smoke:
            raise ValueError("smoke-diagnose preflight requires smoke mode")
        diagnose_table, diagnose_ok = _smoke_diagnose_preflight(cfg, smoke_rows)
        print("E7 smoke-diagnose reconstructed batch layout (no prompt text):")
        print(diagnose_table)
    return preflight_ok(checks) and custom_ok and weight_free_ok and replay_ok and diagnose_ok


def _unit_key(family: str, format_name: str, item_set: str, chunk: int) -> dict[str, str | int]:
    """Build the stable resume key for one P-A prompt chunk."""
    return {
        "family": family,
        "pass": "pa",
        "format": format_name,
        "item_set": item_set,
        "replay_source": "pl",
        "chunk": chunk,
    }


def _unit_provenance(
    cfg: Mapping[str, Any], run_git: RunGit, loaded: LoadedModel, format_name: str,
    item_set: str, chunk: int, expected_rows: int, input_hashes: Mapping[str, str],
    check_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a rule-37 sidecar with the exact E7 direction and input identity."""
    if loaded.spec is None:
        raise RuntimeError("loaded study model has no pinned ModelSpec")
    payload = build_provenance(
        cfg,
        repo_root=REPO,
        unit_key=_unit_key(loaded.family, format_name, item_set, chunk),
        entrypoint=Path(__file__),
        model_id=loaded.spec.hf_id,
        model_revision=loaded.spec.revision,
        dtype="torch.bfloat16",
        batch_size=_replay_source_and_batch_size(cfg, loaded.family, "p_a")[1],
        expected_row_count=expected_rows,
        run_git=run_git,
    )
    payload.update({
        "e7_input_sha256": dict(input_hashes),
        "layer_types": list(_head_config(loaded)[2]),
        "replay_source": cfg["e7"]["replay"]["p_a"]["source"],
    })
    if check_summary is not None:
        payload["e7_check_summary"] = dict(check_summary)
        payload["ops37_update_relative_diagnostic"] = dict(check_summary["update_relative_diagnostic"])
    return payload


def _chunk_expected_rows(prompt_count: int, n_layers: int, n_heads: int) -> int:
    """Return scalar projection rows for one prompt chunk."""
    return int(prompt_count) * int(n_layers) * (4 + 2 * int(n_heads))


def _unit_complete(
    cfg: Mapping[str, Any], output_dir: Path, provenance: Mapping[str, Any], input_hashes: Mapping[str, str]
) -> bool:
    """Return true only when the sidecar, input digests, and expected rows match."""
    complete, _reason = io.unit_is_complete(
        output_dir, provenance["unit_key"], provenance, cfg["runtime"]["provenance_compare"]
    )
    if not complete:
        return False
    parquet_path, sidecar_path = io.unit_paths(output_dir, provenance["unit_key"])
    if not parquet_path.is_file() or not sidecar_path.is_file():
        return False
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return saved.get("e7_input_sha256") == dict(input_hashes)


def _write_chunk(
    frame: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
    run_git: RunGit,
    loaded: LoadedModel,
    format_name: str,
    item_set: str,
    chunk: int,
    input_hashes: Mapping[str, str],
    check_summary: Mapping[str, Any],
) -> str:
    """Write a sorted resumable P-A scalar-projection chunk."""
    if frame.empty:
        return "empty"
    expected = _chunk_expected_rows(
        frame["concept_id"].nunique(), loaded.n_layers, _head_config(loaded)[0]
    )
    if len(frame) != expected:
        raise ValueError(f"P-A unit row count {len(frame)} differs from expected {expected}")
    provenance = _unit_provenance(
        cfg, run_git, loaded, format_name, item_set, chunk, expected, input_hashes, check_summary
    )
    output_dir = REPO / str(cfg["e7"]["results_dir"])
    if _unit_complete(cfg, output_dir, provenance, input_hashes):
        return "skipped"
    io.write_unit(
        frame,
        output_dir,
        provenance,
        columns=PA_COLUMNS,
        sort_by=["model", "format", "item_set", "concept_id", "layer", "kind", "head", "direction"],
        compression=str(cfg["runtime"]["parquet_compression"]),
    )
    return "written"


def _capture_prompts(
    loaded: LoadedModel,
    format_name: str,
    replay_batches: Sequence[Mapping[str, Any]],
    e7_prompt_keys: set[tuple[str, ...]],
    v_vi: np.ndarray,
    d_vi_en: np.ndarray,
    layer_types: Sequence[str],
    references: Mapping[str, tuple[np.ndarray, int, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[float]]:
    """Capture source-pass batches and retain projections/checks only for E7 rows."""
    all_rows: list[dict[str, Any]] = []
    checks: list[dict[str, np.ndarray]] = []
    p_l_errors: list[float] = []
    p_l_replay_errors: list[float] = []
    heads, head_dim, _ = _head_config(loaded)
    for replay_batch in replay_batches:
        batch = list(replay_batch["records"])
        retained = _e7_replay_indices(batch, e7_prompt_keys)
        rows, residuals, errors = _capture_batch(
            loaded,
            batch,
            v_vi=v_vi,
            d_vi_en=d_vi_en,
            heads=heads,
            head_dim=head_dim,
            layer_types=layer_types,
            retain_indices=retained,
        )
        if retained:
            all_rows.extend(rows)
            checks.append(errors)
            kept_records = [batch[index] for index in retained]
            kept_residuals = residuals[retained]
            batch_errors = _p_l_error_for_batch(kept_residuals, kept_records, references)
            p_l_errors.extend(batch_errors)
            p_l_replay_errors.extend(batch_errors)
    return all_rows, checks, p_l_errors, p_l_replay_errors


def _capture_smoke_cell(
    loaded: LoadedModel,
    cfg: Mapping[str, Any],
    format_name: str,
    replay_batches: Sequence[Mapping[str, Any]],
    selected_rows: set[int],
    v_vi: np.ndarray,
    d_vi_en: np.ndarray,
    layer_types: Sequence[str],
    pd_residuals: np.ndarray,
) -> tuple[list[dict[str, np.ndarray]], list[float], list[float]]:
    """Capture selected rows in exact P-D batches and compare their residual stacks."""
    checks: list[dict[str, np.ndarray]] = []
    pd_errors: list[float] = []
    pd_replay_errors: list[float] = []
    heads, head_dim, _layer_types = _head_config(loaded)
    for replay_batch in replay_batches:
        batch = list(replay_batch["records"])
        retained = [
            index for index, record in enumerate(batch)
            if int(record["pd_array_row"]) in selected_rows
        ]
        if not retained:
            continue
        _rows, residuals, errors = _capture_batch(
            loaded,
            batch,
            v_vi=v_vi,
            d_vi_en=d_vi_en,
            heads=heads,
            head_dim=head_dim,
            layer_types=layer_types,
            retain_indices=retained,
        )
        checks.append(errors)
        kept_records = [batch[index] for index in retained]
        batch_errors = _p_d_error_for_batch(residuals[retained], kept_records, pd_residuals)
        pd_errors.extend(batch_errors)
        pd_replay_errors.extend(batch_errors)
    return checks, pd_errors, pd_replay_errors


def _capture_residual_batch(
    loaded: LoadedModel, records: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, list[int]]:
    """Capture prompt-last-token residuals only for one explicitly ordered batch."""
    from vilens.passes import _encode_prompts, _pad_id, _pad_sequences

    if not records:
        raise ValueError("residual capture batch cannot be empty")
    device = next(loaded.model.parameters()).device
    sequences = _encode_prompts(loaded, [str(record["prompt"]) for record in records])
    input_ids, attention_mask, lengths = _pad_sequences(sequences, _pad_id(loaded), device)
    positions = torch.as_tensor(
        [[length - 1] for length in lengths], dtype=torch.long, device=device
    )
    recording = hooks.record(
        loaded,
        input_ids,
        attention_mask,
        positions=positions,
        components=False,
        keep_on="cpu",
    )
    residuals = recording.resid[:, :, 0, :].permute(1, 0, 2).float().cpu().numpy()
    return residuals, [int(length) for length in lengths]


def _diagnostic_relative_matrix(
    pd_residuals: np.ndarray,
    selected_records: Sequence[Mapping[str, Any]],
    residuals: np.ndarray,
) -> np.ndarray:
    """Return 10(c) relative errors as [selected prompt, residual layer]."""
    if residuals.shape[0] != len(selected_records):
        raise ValueError("captured residual prompt count differs from diagnostic selection")
    errors: list[np.ndarray] = []
    for prompt_index, record in enumerate(selected_records):
        array_row = int(record["pd_array_row"])
        if not 0 <= array_row < pd_residuals.shape[0]:
            raise ValueError("diagnostic prompt has an invalid P-D residual row")
        reference = np.asarray(pd_residuals[array_row], dtype=np.float32)
        estimate = np.asarray(residuals[prompt_index], dtype=np.float32)
        if reference.shape != estimate.shape:
            raise ValueError("diagnostic P-D and captured residual layer shapes differ")
        errors.append(attribute.relative_errors(reference, estimate))
    return np.stack(errors, axis=0)


def _diagnostic_10c_summary(
    error_matrix: np.ndarray, n_layers: int, gate_median: float
) -> dict[str, Any]:
    """Summarize diagnostic 10(c) values without changing or enforcing its gate."""
    flat = np.asarray(error_matrix, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    layers = sorted({
        0, 1, 2,
        math.ceil(n_layers / 4),
        math.ceil(n_layers / 2),
        math.ceil(3 * n_layers / 4),
        n_layers,
    })
    return {
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p99": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
        "maximum": float(np.max(finite)) if finite.size else float("nan"),
        "exact_zero_count": int(np.count_nonzero(flat == 0.0)),
        "value_count": int(flat.size),
        "exact_zero_fraction": float(np.count_nonzero(flat == 0.0) / flat.size) if flat.size else float("nan"),
        "finite_count": int(finite.size),
        "gate_median_limit": float(gate_median),
        "median_within_gate": bool(finite.size and np.median(finite) <= gate_median),
        "gate_enforced": False,
        "per_layer_median": {
            str(layer): float(np.median(error_matrix[:, layer])) for layer in layers
        },
    }


def _diagnostic_padding_split(
    error_matrix: np.ndarray, prompt_facts: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Summarize S1 10(c) by the P-D/P-A padding-status categories."""
    groups: dict[str, list[int]] = {name: [] for name in ("pd_only", "pa_only", "both", "neither")}
    for prompt_index, fact in enumerate(prompt_facts):
        pd_padded = bool(fact["pd_padded"])
        pa_padded = bool(fact["pa_padded"])
        category = "both" if pd_padded and pa_padded else (
            "pd_only" if pd_padded else "pa_only" if pa_padded else "neither"
        )
        groups[category].append(prompt_index)
    result: dict[str, dict[str, Any]] = {}
    for category, prompt_indices in groups.items():
        values = error_matrix[prompt_indices, :].reshape(-1) if prompt_indices else np.empty(0)
        result[category] = {
            "prompt_count": len(prompt_indices),
            "relative_error_count": int(values.size),
            "median": float(np.median(values)) if values.size else None,
        }
    return result


def _diagnostic_long_rows(
    loaded: LoadedModel,
    format_name: str,
    records: Sequence[Mapping[str, Any]],
    prompt_facts: Sequence[Mapping[str, Any]],
    scheme: str,
    scheme_batches: Sequence[Mapping[str, Any]],
    error_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """Build rule-38 long rows without prompt text or other prompt content."""
    if loaded.spec is None:
        model_name = loaded.family
        model_revision = "unknown"
    else:
        model_name = loaded.spec.hf_id
        model_revision = loaded.spec.revision
    batch_by_prompt: dict[int, Mapping[str, Any]] = {}
    for batch in scheme_batches:
        for local_index in batch["selected_local_rows"]:
            record = batch["records"][int(local_index)]
            prompt_index = int(record["_diagnostic_prompt_index"])
            batch_by_prompt[prompt_index] = {
                "scheme_batch_index": int(batch["batch_index"]),
                "scheme_batch_max_length": int(batch["max_length"]),
                "scheme_padded": bool(batch["padded_by_row"][int(local_index)]),
            }
    rows: list[dict[str, Any]] = []
    for prompt_index, (record, fact) in enumerate(zip(records, prompt_facts, strict=True)):
        if prompt_index not in batch_by_prompt:
            # S1 and S3 pass compact per-prompt layout records directly.
            batch_by_prompt[prompt_index] = {
                key: int(fact[key]) if key.endswith("_index") or key.endswith("_length") else bool(fact[key])
                for key in ("scheme_batch_index", "scheme_batch_max_length", "scheme_padded")
            }
        scheme_fact = batch_by_prompt[prompt_index]
        base = {
            "model": model_name,
            "model_revision": model_revision,
            "pass": "e7_smoke_diagnose",
            "format": format_name,
            "concept_id": str(record["concept_id"]),
            "lang": str(record.get("lang", "")),
            "split": str(record.get("split", "")),
            "scheme": scheme,
            "prompt_index": prompt_index,
            "pd_array_row": int(fact["pd_array_row"]),
            "token_length": int(fact["token_length"]),
            "pd_batch_index": int(fact["pd_batch_index"]),
            "pd_batch_max_length": int(fact["pd_batch_max_length"]),
            "pd_padded": bool(fact["pd_padded"]),
            "pa_batch_index": int(fact["pa_batch_index"]),
            "pa_batch_max_length": int(fact["pa_batch_max_length"]),
            "pa_padded": bool(fact["pa_padded"]),
            **scheme_fact,
        }
        for layer, value in enumerate(error_matrix[prompt_index]):
            rows.append({
                **base,
                "layer": int(layer),
                "metric": "residual_match_relative",
                "value": float(value),
            })
    return rows


def _verify_checks(
    checks: Sequence[Mapping[str, np.ndarray]],
    p_l_errors: Sequence[float],
    cfg: Mapping[str, Any],
    *,
    p_l_replay_errors: Sequence[float] | None = None,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Fail the current model/format unit if an OPS-37/OPS-38 validity gate fails."""
    summary = _check_summary(
        checks, cfg, p_l_errors=p_l_errors, p_l_replay_errors=p_l_replay_errors
    )
    failed = [name for name, row in summary.items() if "passed" in row and not bool(row["passed"])]
    if failed:
        raise RuntimeError(f"E7 validity check failed for {label}: {failed}; checks={summary}")
    return summary


def _run_format(
    cfg: Mapping[str, Any],
    loaded: LoadedModel,
    run_git: RunGit,
    format_name: str,
    records: Sequence[Mapping[str, Any]],
    primary_ids: set[str],
    pl_source_records: Sequence[Mapping[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run the first P-L capture batch as a trial, then exact P-L replay batches."""
    v_vi, d_unit, fit_counts, d_check, e8_sha, d_max_abs_diff, d_max_abs = _load_a6_directions(
        cfg, loaded.family, format_name
    )
    if d_check > float(cfg["e7"]["check_direction_relative"]):
        raise RuntimeError(f"OPS-37 direction check failed for {loaded.family}/{format_name}: {d_check}")
    references = _pl_residual_references(cfg, loaded.family, format_name)
    heads, head_dim, layer_types = _head_config(loaded)
    expected_main = int(cfg["e7"]["expected_prompt_counts"]["main"][format_name])
    expected_cf = int(cfg["e7"]["expected_prompt_counts"]["cf729"][format_name])
    if len(records) != expected_main:
        raise AssertionError(f"E7 main prompt rows differ from the frozen count for {format_name}")
    primary_records = [row for row in records if str(row["concept_id"]) in primary_ids]
    if len(primary_records) != expected_cf:
        raise AssertionError(f"E7 cf729 prompt rows differ from the frozen count for {format_name}")
    p_d_overlap, d12_note = _fit_overlap_count(
        cfg, loaded.family, format_name, {str(row["concept_id"]) for row in primary_records}
    )
    if p_d_overlap > int(cfg["e7"]["max_pd_e7_overlap"]):
        raise RuntimeError(f"OPS-37 P-D fit overlap is {p_d_overlap} for {loaded.family}/{format_name}")
    e7_prompt_keys = {_e7_prompt_key(row) for row in records}
    replay_records = _pl_source_cell_records(cfg, pl_source_records, format_name)
    replay_batches = _pl_replay_batches(cfg, loaded.family, replay_records)
    replayed_e7_rows = [
        row for row in replay_records
        if not _is_extension(row.get("m1_extension", False))
        and _e7_prompt_key(row) in e7_prompt_keys
    ]
    replayed_e7_keys = {_e7_prompt_key(row) for row in replayed_e7_rows}
    if len(replayed_e7_rows) != expected_main or replayed_e7_keys != e7_prompt_keys:
        raise ValueError("P-L replay source does not contain the complete E7 main row set")
    if not replay_batches:
        raise ValueError("P-L replay source has no capture batches")
    trial_source_rows = [
        row for row in replay_batches[0]["records"]
        if not _is_extension(row.get("m1_extension", False))
        and _e7_prompt_key(row) in e7_prompt_keys
    ]
    trial_rows, trial_checks, trial_pl_errors, trial_replay_errors = _capture_prompts(
        loaded, format_name, [replay_batches[0]], e7_prompt_keys,
        v_vi, d_unit, layer_types, references,
    )
    if len({str(row["concept_id"]) for row in trial_rows}) != len(trial_source_rows):
        raise ValueError("P-A first-batch trial did not retain all E7 rows in the P-L batch")
    del trial_rows
    trial_summary = _verify_checks(
        trial_checks, trial_pl_errors, cfg,
        p_l_replay_errors=trial_replay_errors,
        label=f"trial {loaded.family}/{format_name}",
    )
    trial_summary["trial_prompt_count"] = len(trial_source_rows)
    unit_summary: dict[str, int] = {"written": 0, "skipped": 0}
    input_hashes = {
        "e8_vector": e8_sha,
        "pd_index": sha256_file(_pd_paths(cfg, loaded.family, format_name)[0]),
        "pd_sidecar": sha256_file(_pd_paths(cfg, loaded.family, format_name)[1]),
        "pd_residual": sha256_file(_pd_paths(cfg, loaded.family, format_name)[2]),
    }
    p_l_sidecar_hashes = sorted({sha256_file(value[2]) for value in references.values()})
    input_hashes["p_l_sidecars_sha256"] = __import__("hashlib").sha256(
        "\n".join(p_l_sidecar_hashes).encode("ascii")
    ).hexdigest()
    p_l_residual_hashes = sorted({
        sha256_file(Path(str(getattr(value[0], "filename"))))
        for value in references.values()
    })
    input_hashes["p_l_residual_arrays_sha256"] = __import__("hashlib").sha256(
        "\n".join(p_l_residual_hashes).encode("ascii")
    ).hexdigest()
    p_l_root = REPO / str(cfg["e7"]["p_l_results_dir"])
    p_l_index_paths = sorted(p_l_root.glob(
        f"condition=diac__family={loaded.family}__fewshot_set={cfg['e7']['fewshot_set']}"
        f"__format={format_name}__pass=pl__shard=*.resid_index.jsonl"
    ))
    input_hashes["p_l_all_index_sha256"] = __import__("hashlib").sha256(
        "\n".join(sha256_file(path) for path in p_l_index_paths).encode("ascii")
    ).hexdigest()
    pl_paths = prompt_file_paths(dict(cfg))
    input_hashes["pl_prompt_files_sha256"] = __import__("hashlib").sha256(
        "\n".join(
            sha256_file(
                Path(pl_paths[str(stem)])
                if Path(pl_paths[str(stem)]).is_absolute()
                else REPO / Path(pl_paths[str(stem)])
            )
            for stem in cfg["pl"]["primary"]["files"]
        ).encode("ascii")
    ).hexdigest()
    full_checks: list[dict[str, np.ndarray]] = []
    full_p_l_errors: list[float] = []
    full_replay_errors: list[float] = []
    for chunk_index, replay_batch in enumerate(replay_batches):
        batch_records = list(replay_batch["records"])
        batch_e7 = [
            row for row in batch_records
            if not _is_extension(row.get("m1_extension", False))
            and _e7_prompt_key(row) in e7_prompt_keys
        ]
        if not batch_e7:
            _discarded_rows, _discarded_checks, _discarded_pl, _discarded_replay = _capture_prompts(
                loaded, format_name, [replay_batch], e7_prompt_keys,
                v_vi, d_unit, layer_types, references,
            )
            unit_summary["discard_only_batches"] = unit_summary.get("discard_only_batches", 0) + 1
            continue
        expected_main_rows = _chunk_expected_rows(len(batch_e7), loaded.n_layers, heads)
        main_provenance = _unit_provenance(
            cfg, run_git, loaded, format_name, "main", chunk_index, expected_main_rows, input_hashes
        )
        cf_chunk = [row for row in batch_e7 if str(row["concept_id"]) in primary_ids]
        expected_cf_rows = _chunk_expected_rows(len(cf_chunk), loaded.n_layers, heads)
        cf_provenance = _unit_provenance(
            cfg, run_git, loaded, format_name, "cf729", chunk_index, expected_cf_rows, input_hashes
        )
        out_dir = REPO / str(cfg["e7"]["results_dir"])
        main_done = _unit_complete(cfg, out_dir, main_provenance, input_hashes)
        cf_done = not cf_chunk or _unit_complete(cfg, out_dir, cf_provenance, input_hashes)
        if main_done and cf_done:
            unit_summary["skipped"] += 1
            logger.info(
                "P-A replay batch skipped family=%s format=%s shard=%s batch=%d",
                loaded.family, format_name, replay_batch["shard"], replay_batch["batch_index"],
            )
            continue
        raw_rows, chunk_checks, chunk_p_l_errors, chunk_replay_errors = _capture_prompts(
            loaded, format_name, [replay_batch], e7_prompt_keys,
            v_vi, d_unit, layer_types, references,
        )
        full_checks.extend(chunk_checks)
        full_p_l_errors.extend(chunk_p_l_errors)
        full_replay_errors.extend(chunk_replay_errors)
        if len(raw_rows) != expected_main_rows:
            raise ValueError("P-A replay projection rows differ from the E7 main count")
        chunk_check_summary = _check_summary(
            chunk_checks, cfg,
            p_l_errors=chunk_p_l_errors,
            p_l_replay_errors=chunk_replay_errors,
        )
        frame = pd.DataFrame(raw_rows, columns=PA_COLUMNS)
        main_status = _write_chunk(
            frame, cfg=cfg, run_git=run_git, loaded=loaded, format_name=format_name,
            item_set="main", chunk=chunk_index, input_hashes=input_hashes,
            check_summary=chunk_check_summary,
        )
        if main_status == "written":
            unit_summary["written"] += 1
        if cf_chunk:
            cf_frame = frame[frame["concept_id"].astype(str).isin(primary_ids)].copy()
            cf_frame["item_set"] = "cf729"
            cf_status = _write_chunk(
                cf_frame, cfg=cfg, run_git=run_git, loaded=loaded, format_name=format_name,
                item_set="cf729", chunk=chunk_index, input_hashes=input_hashes,
                check_summary=chunk_check_summary,
            )
            if cf_status == "written":
                unit_summary["written"] += 1
        logger.info(
            "P-A progress family=%s format=%s shard=%s replay_batch=%d/%d source_rows=%d rows_main=%d rows_cf729=%d",
            loaded.family, format_name, replay_batch["shard"], replay_batch["batch_index"] + 1,
            sum(int(batch["shard"] == replay_batch["shard"]) for batch in replay_batches),
            len(batch_records), len(frame), len(cf_chunk),
        )
        del frame, raw_rows
        gc.collect()
    full_summary = _verify_checks(
        full_checks,
        full_p_l_errors,
        cfg,
        p_l_replay_errors=full_replay_errors,
        label=f"full {loaded.family}/{format_name}",
    ) if full_checks else {"resume_reuse": {"passed": True, "detail": "all completed units reused"}}
    return {
        "family": loaded.family,
        "format": format_name,
        "main_prompts": len(records),
        "cf729_prompts": len(primary_records),
        "fit_counts": fit_counts,
        "p_d_fit_e7_overlap": p_d_overlap,
        "d12_note": d12_note,
        "d_vs_negative_e8_max_relative": d_check,
        "d_vs_negative_e8_max_abs_diff": d_max_abs_diff,
        "d_vs_negative_e8_max_abs_d": d_max_abs,
        "e8_vector_sha256": e8_sha,
        "layer_types": layer_types,
        "trial_checks": trial_summary,
        "replay_batch_count": len(replay_batches),
        "trial_prompt_count": len(trial_source_rows),
        "full_checks": full_summary,
        "units": unit_summary,
    }


def _smoke_input_hashes(
    cfg: Mapping[str, Any], prompt_path: Path
) -> dict[str, str]:
    """Hash only direction prompts and stored P-D/E8 inputs used by the smoke."""
    hashes = {"directions_prompts_sha256": sha256_file(prompt_path)}
    prompt_paths = prompt_file_paths(dict(cfg))
    hashes["pl_prompt_files_sha256"] = __import__("hashlib").sha256(
        "\n".join(
            sha256_file(
                Path(prompt_paths[str(stem)])
                if Path(prompt_paths[str(stem)]).is_absolute()
                else REPO / Path(prompt_paths[str(stem)])
            )
            for stem in cfg["pl"]["primary"]["files"]
        ).encode("ascii")
    ).hexdigest()
    for family in FAMILIES:
        for format_name in FORMATS:
            index_path, sidecar_path, residual_path = _pd_paths(cfg, family, format_name)
            hashes[f"pd_index_{family}_{format_name}_sha256"] = sha256_file(index_path)
            hashes[f"pd_sidecar_{family}_{format_name}_sha256"] = sha256_file(sidecar_path)
            hashes[f"pd_residual_{family}_{format_name}_sha256"] = sha256_file(residual_path)
            hashes[f"e8_vector_{family}_{format_name}_sha256"] = sha256_file(
                _e8_vector_path(cfg, family, format_name)
            )
            root = REPO / str(cfg["e7"]["p_l_results_dir"])
            prefix = (
                f"condition=diac__family={family}__fewshot_set={cfg['e7']['fewshot_set']}"
                f"__format={format_name}__pass=pl__shard="
            )
            for shard_index, index_path in enumerate(sorted(root.glob(prefix + "*.resid_index.jsonl"))):
                hashes[f"pl_index_{family}_{format_name}_{shard_index}_sha256"] = sha256_file(index_path)
    return hashes


def _run_smoke(
    cfg: dict[str, Any],
    prompt_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    split_used: Mapping[str, Mapping[str, str]],
    all_direction_records: Sequence[Mapping[str, Any]],
    logger: logging.Logger,
) -> int:
    """Run all six directions-split P-A smoke cells and write only summary plus provenance."""
    import pyarrow.parquet as pq

    locks.ensure_adjudication_committed(REPO)
    run_git = capture_run_git(REPO, cfg)
    started_all = time.perf_counter()
    direction_path = prompt_file_paths(dict(cfg))["directions_matched"]
    if not direction_path.is_absolute():
        direction_path = REPO / direction_path
    summary: dict[str, Any] = {
        "utc_start": _utc_now(),
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "gpu_name": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "profile": _runtime_profile(),
        "batch_size": {
            family: _replay_source_and_batch_size(cfg, family, "smoke")[1]
            for family in FAMILIES
        },
        "replay_source": cfg["e7"]["replay"]["smoke"]["source"],
        "directions_split_only": True,
        "analysis_tag_lock_consulted": False,
        "families": {},
        "exit_code": 0,
    }
    for family in FAMILIES:
        family_started = time.perf_counter()
        family_summary: dict[str, Any] = {"formats": {}}
        loaded: LoadedModel | None = None
        try:
            torch.cuda.reset_peak_memory_stats()
            loaded = models.load(cfg, family, device=torch.device("cuda"))
        except Exception as exc:
            details = _exception_details(exc)
            for format_name in FORMATS:
                family_summary["formats"][format_name] = {
                    "split_used": split_used[family][format_name],
                    "prompt_count": len(prompt_rows[family][format_name]),
                    "passed": False,
                    **details,
                }
            summary["exit_code"] = 2
            logger.error(
                "E7 smoke model load failed family=%s type=%s message=%s frame=%s",
                family, details["error_type"], details["error_message"],
                details["innermost_traceback_frame"],
            )
        if loaded is not None:
            try:
                for format_name in FORMATS:
                    cell_rows = prompt_rows[family][format_name]
                    cell_summary: dict[str, Any] = {
                        "split_used": split_used[family][format_name],
                        "prompt_count": len(cell_rows),
                        "checks": {},
                        "check_10e": "skipped",
                    }
                    try:
                        (
                            v_vi, d_unit, fit_counts, direction_error, e8_sha,
                            direction_max_abs_diff, direction_max_abs_d,
                        ) = _load_a6_directions(
                            cfg, family, format_name
                        )
                        pd_index_path, _pd_sidecar, pd_residual_path = _pd_paths(
                            cfg, family, format_name
                        )
                        pd_rows = pq.read_table(
                            pd_index_path,
                            columns=["row", "concept_id", "lang", "format", "split"],
                        ).to_pylist()
                        pd_residuals = np.load(pd_residual_path, mmap_mode="r", allow_pickle=False)
                        if (
                            pd_residuals.ndim != 3
                            or pd_residuals.dtype != np.float32
                            or pd_residuals.shape[0] != len(pd_rows)
                            or pd_residuals.shape[1] != loaded.n_layers + 1
                        ):
                            raise ValueError("P-D residual array shape/dtype differs from its index or model")
                        pd_replay_layout, replay_batches = _verify_pd_replay_cell(
                            cfg, family, format_name, all_direction_records, cell_rows
                        )
                        selected_rows = {int(row["pd_array_row"]) for row in cell_rows}
                        checks, pd_errors, pd_replay_errors = _capture_smoke_cell(
                            loaded, cfg, format_name, replay_batches, selected_rows,
                            v_vi, d_unit, _head_config(loaded)[2], pd_residuals,
                        )
                        check_values = _check_summary(
                            checks,
                            cfg,
                            p_d_errors=pd_errors,
                            p_d_replay_errors=pd_replay_errors,
                        )
                        direction_limit = float(cfg["e7"]["check_direction_relative"])
                        check_values["direction_vs_negative_e8"] = {
                            "max_abs_diff": direction_max_abs_diff,
                            "max_abs_d": direction_max_abs_d,
                            "max_relative": float(direction_error),
                            "limit": direction_limit,
                            "passed": bool(np.isfinite(direction_error) and direction_error <= direction_limit),
                            "e8_vector_sha256": e8_sha,
                        }
                        check_values["fit_counts"] = fit_counts
                        cell_summary["checks"] = check_values
                        cell_summary["replay_batch_count"] = pd_replay_layout["batch_count"]
                        cell_summary["replay_source_row_count"] = pd_replay_layout["row_count"]
                        cell_summary["replay_batch_size"] = pd_replay_layout["batch_size"]
                        cell_summary["passed"] = all(
                            bool(value["passed"])
                            for value in check_values.values()
                            if isinstance(value, Mapping) and "passed" in value
                        )
                        if not cell_summary["passed"]:
                            summary["exit_code"] = 2
                        logger.info(
                            "E7 smoke cell family=%s format=%s split=%s passed=%s",
                            family, format_name, split_used[family][format_name], cell_summary["passed"],
                        )
                        del pd_residuals
                    except Exception as exc:
                        cell_summary["passed"] = False
                        details = _exception_details(exc)
                        cell_summary.update(details)
                        summary["exit_code"] = 2
                        logger.error(
                            "E7 smoke cell failed family=%s format=%s type=%s message=%s frame=%s",
                            family, format_name, details["error_type"], details["error_message"],
                            details["innermost_traceback_frame"],
                        )
                    family_summary["formats"][format_name] = cell_summary
            finally:
                del loaded
                gc.collect()
                torch.cuda.empty_cache()
        family_summary["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        family_summary["wall_seconds"] = time.perf_counter() - family_started
        summary["families"][family] = family_summary
    summary["utc_end"] = _utc_now()
    summary["wall_seconds"] = time.perf_counter() - started_all
    job_id = os.environ.get("SLURM_JOB_ID")
    summary_name = f"summary.job-{job_id}.json" if job_id else "summary.json"
    summary_path = REPO / str(cfg["e7"]["smoke_results_dir"]) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = summary_path.with_name(summary_path.stem + ".provenance.json")
    if summary_path.exists() or provenance_path.exists():
        raise FileExistsError("E7 smoke summary output already exists; refusing to overwrite it")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    study_set = str(cfg["models"]["study_set"])
    model_loads = [
        {
            "model_id": str(cfg["models"][study_set][family]["hf_id"]),
            "model_revision": str(cfg["models"][study_set][family]["revision"]),
            "dtype": "torch.bfloat16",
            "dtype_override": None,
        }
        for family in FAMILIES
    ]
    representative = model_loads[0]
    run_provenance = build_provenance(
        cfg,
        repo_root=REPO,
        unit_key={"pass": "pa_smoke", "cells": 6},
        entrypoint=Path(__file__),
        model_id="multi-family E7 smoke",
        model_revision="per-family model_loads",
        dtype="torch.bfloat16",
        batch_size=max(
            _replay_source_and_batch_size(cfg, family, "smoke")[1]
            for family in FAMILIES
        ),
        expected_row_count=sum(
            len(prompt_rows[family][format_name]) for family in FAMILIES for format_name in FORMATS
        ),
        run_git=run_git,
        model_loads=model_loads,
    )
    run_provenance["row_count"] = run_provenance["expected_row_count"]
    run_provenance["utc_end"] = _utc_now()
    run_provenance["model_id"] = representative["model_id"]
    run_provenance["model_revision"] = representative["model_revision"]
    run_provenance["e7_input_sha256"] = _smoke_input_hashes(cfg, direction_path)
    run_provenance["e7_smoke"] = {
        "directions_split_only": True,
        "analysis_tag_lock_consulted": False,
        "split_used": {family: dict(split_used[family]) for family in FAMILIES},
        "cell_summaries": {
            family: dict(summary["families"][family]["formats"]) for family in FAMILIES
        },
    }
    provenance_path.write_text(
        json.dumps(run_provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    logger.info("E7 smoke final summary %s", json.dumps(summary, sort_keys=True, default=str))
    return int(summary["exit_code"])


def _run_smoke_diagnose(
    cfg: dict[str, Any],
    prompt_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    split_used: Mapping[str, Mapping[str, str]],
    logger: logging.Logger,
) -> int:
    """Compare smoke residuals under P-A, exact P-D, and unpadded singleton batches."""
    from vilens.data import load_prompt_file, prompt_file_paths

    locks.ensure_adjudication_committed(REPO)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("smoke-diagnose output requires SLURM_JOB_ID")
    output_dir = REPO / str(cfg["e7"]["smoke_results_dir"])
    parquet_path = output_dir / f"diag.job-{job_id}.parquet"
    summary_path = output_dir / f"diag.job-{job_id}.json"
    provenance_path = output_dir / f"diag.job-{job_id}.provenance.json"
    if any(path.exists() for path in (parquet_path, summary_path, provenance_path)):
        raise FileExistsError(f"diagnostic outputs already exist for job {job_id}")

    run_git = capture_run_git(REPO, cfg)
    started_all = time.perf_counter()
    direction_path = prompt_file_paths(dict(cfg))["directions_matched"]
    if not direction_path.is_absolute():
        direction_path = REPO / direction_path
    all_prompts = load_prompt_file(direction_path)
    summary: dict[str, Any] = {
        "utc_start": _utc_now(),
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "gpu_name": torch.cuda.get_device_name(0),
        "slurm_job_id": job_id,
        "profile": _runtime_profile(),
        "directions_split_only": True,
        "analysis_tag_lock_consulted": False,
        "schemes": {
            "S1": {"description": "P-A batches of 4"},
            "S2": {"description": "exact P-D membership, order, and right-padding"},
            "S3": {"description": "batch size 1, no padding"},
        },
        "families": {},
        "exit_code": 0,
    }
    long_rows: list[dict[str, Any]] = []
    model_loads: list[dict[str, Any]] = []
    pd_batch_sizes: set[int] = set()
    output_complete = True

    for family in FAMILIES:
        family_started = time.perf_counter()
        family_summary: dict[str, Any] = {"formats": {}}
        loaded: LoadedModel | None = None
        try:
            torch.cuda.reset_peak_memory_stats()
            loaded = models.load(cfg, family, device=torch.device("cuda"))
            if loaded.spec is not None:
                model_loads.append({
                    "model_id": loaded.spec.hf_id,
                    "model_revision": loaded.spec.revision,
                    "dtype": "torch.bfloat16",
                    "dtype_override": None,
                })
        except Exception as exc:
            details = _exception_details(exc)
            for format_name in FORMATS:
                family_summary["formats"][format_name] = {
                    "split_used": split_used[family][format_name],
                    "prompt_count": len(prompt_rows[family][format_name]),
                    "complete": False,
                    "error": details,
                }
            output_complete = False
            logger.error(
                "E7 smoke-diagnose model load failed family=%s type=%s message=%s frame=%s",
                family, details["error_type"], details["error_message"],
                details["innermost_traceback_frame"],
            )
            logger.error("%s", traceback.format_exc())

        if loaded is not None:
            try:
                for format_name in FORMATS:
                    cell_started = time.perf_counter()
                    cell_records = list(prompt_rows[family][format_name])
                    cell_summary: dict[str, Any] = {
                        "split_used": split_used[family][format_name],
                        "prompt_count": len(cell_records),
                        "complete": False,
                        "schemes": {},
                    }
                    family_summary["formats"][format_name] = cell_summary
                    if any(str(record.get("split")) == "test" for record in cell_records):
                        details = _exception_details(
                            AssertionError("diagnostic selection contains a test-split record")
                        )
                        cell_summary["error"] = details
                        output_complete = False
                        continue
                    try:
                        layout = _diagnostic_cell_layout(
                            cfg, family, format_name, cell_records, all_prompts
                        )
                        pd_batch_sizes.add(int(layout["pd_batch_size"]))
                        if len(layout["prompt_facts"]) != len(cell_records):
                            raise ValueError("diagnostic batch layout differs from selected row count")
                        pd_index_path, _pd_sidecar, pd_residual_path = _pd_paths(
                            cfg, family, format_name
                        )
                        del pd_index_path
                        pd_residuals = np.load(pd_residual_path, mmap_mode="r", allow_pickle=False)
                        n_layers, d_model, _vocab = map(
                            int, cfg["reference_facts"]["x0c"]["model_dims"][family]
                        )
                        expected_shape = (len(layout["ordered_records"]), n_layers + 1, d_model)
                        if pd_residuals.shape != expected_shape or pd_residuals.dtype != np.float32:
                            raise ValueError(
                                f"P-D residual shape/dtype {pd_residuals.shape}/{pd_residuals.dtype} "
                                f"does not match {expected_shape}/float32"
                            )
                    except Exception as exc:
                        details = _exception_details(exc)
                        cell_summary["error"] = details
                        output_complete = False
                        logger.error(
                            "E7 smoke-diagnose layout failed family=%s format=%s type=%s message=%s frame=%s",
                            family, format_name, details["error_type"], details["error_message"],
                            details["innermost_traceback_frame"],
                        )
                        logger.error("%s", traceback.format_exc())
                        continue

                    selected_records = layout["selected_records"]
                    prompt_facts = layout["prompt_facts"]
                    selected_rows = [int(row["pd_array_row"]) for row in selected_records]
                    selected_position = {row: index for index, row in enumerate(selected_rows)}
                    residual_shape = (len(selected_records), n_layers + 1, d_model)

                    # S1 is the same capture and check path as the gate smoke, with batch size 4.
                    try:
                        v_vi, d_unit, _fit_counts, d_check, e8_sha, d_abs_diff, d_abs = (
                            _load_a6_directions(cfg, family, format_name)
                        )
                        s1_batch_size = int(cfg["e7"]["batch_size"][family])
                        s1_residuals = np.empty(residual_shape, dtype=np.float32)
                        s1_checks: list[dict[str, np.ndarray]] = []
                        s1_batches: list[dict[str, Any]] = []
                        heads, head_dim, layer_types = _head_config(loaded)
                        for batch_index, start in enumerate(range(0, len(selected_records), s1_batch_size)):
                            batch = selected_records[start : start + s1_batch_size]
                            _projection_rows, residuals, captured_errors = _capture_batch(
                                loaded,
                                batch,
                                v_vi=v_vi,
                                d_vi_en=d_unit,
                                heads=heads,
                                head_dim=head_dim,
                                layer_types=layer_types,
                            )
                            s1_residuals[start : start + len(batch)] = residuals
                            s1_checks.append(captured_errors)
                            batch_facts = prompt_facts[start : start + len(batch)]
                            s1_batches.append({
                                "batch_index": batch_index,
                                "records": batch,
                                "selected_local_rows": list(range(len(batch))),
                                "max_length": int(batch_facts[0]["pa_batch_max_length"]),
                                "padded_by_row": [bool(fact["pa_padded"]) for fact in batch_facts],
                            })
                        s1_errors = _diagnostic_relative_matrix(
                            pd_residuals, selected_records, s1_residuals
                        )
                        s1_check_values = _check_summary(
                            s1_checks, cfg
                        )
                        direction_limit = float(cfg["e7"]["check_direction_relative"])
                        s1_check_values["direction_vs_negative_e8"] = {
                            "max_abs_diff": float(d_abs_diff),
                            "max_abs_d": float(d_abs),
                            "max_relative": float(d_check),
                            "limit": direction_limit,
                            "passed": bool(np.isfinite(d_check) and d_check <= direction_limit),
                            "e8_vector_sha256": e8_sha,
                        }
                        cell_summary["checks_s1"] = s1_check_values
                        s1_summary = _diagnostic_10c_summary(
                            s1_errors, n_layers, float(cfg["e7"]["check_pl_median_relative"])
                        )
                        cell_summary["schemes"]["S1"] = {
                            "complete": True,
                            "capture_batch_count": len(s1_batches),
                            "captured_prompt_count": len(selected_records),
                            "padded_prompt_count": sum(bool(fact["pa_padded"]) for fact in prompt_facts),
                            "ten_c": s1_summary,
                            "ten_c_by_padding_status": _diagnostic_padding_split(s1_errors, prompt_facts),
                        }
                        long_rows.extend(_diagnostic_long_rows(
                            loaded, format_name, selected_records, prompt_facts,
                            "S1", s1_batches, s1_errors,
                        ))
                    except Exception as exc:
                        details = _exception_details(exc)
                        cell_summary["schemes"]["S1"] = {"complete": False, "error": details}
                        output_complete = False
                        logger.error(
                            "E7 smoke-diagnose S1 failed family=%s format=%s type=%s message=%s frame=%s",
                            family, format_name, details["error_type"], details["error_message"],
                            details["innermost_traceback_frame"],
                        )
                        logger.error("%s", traceback.format_exc())

                    # S2 replays complete P-D batches containing a selected prompt, then retains selected rows.
                    try:
                        s2_residuals = np.empty(residual_shape, dtype=np.float32)
                        s2_batches = layout["replay_batches"]
                        for batch in s2_batches:
                            batch_records = batch["records"]
                            captured, lengths = _capture_residual_batch(loaded, batch_records)
                            stored_lengths = [int(row["_diagnostic_token_length"]) for row in batch_records]
                            if lengths != stored_lengths or max(lengths) != int(batch["max_length"]):
                                raise ValueError("S2 tokenizer lengths differ from the stored P-D batch metadata")
                            for local_index in batch["selected_local_rows"]:
                                record = batch_records[int(local_index)]
                                prompt_index = selected_position[int(record["pd_array_row"])]
                                s2_residuals[prompt_index] = captured[int(local_index)]
                        s2_errors = _diagnostic_relative_matrix(
                            pd_residuals, selected_records, s2_residuals
                        )
                        cell_summary["schemes"]["S2"] = {
                            "complete": True,
                            "capture_batch_count": len(s2_batches),
                            "captured_prompt_count": len(selected_records),
                            "replayed_prompt_count": layout["replay_prompt_count"],
                            "padded_prompt_count": sum(
                                bool(batch["padded_by_row"][local])
                                for batch in s2_batches
                                for local in batch["selected_local_rows"]
                            ),
                            "replayed_batch_indices": layout["selected_pd_batch_indices"],
                            "ten_c": _diagnostic_10c_summary(
                                s2_errors, n_layers, float(cfg["e7"]["check_pl_median_relative"])
                            ),
                        }
                        long_rows.extend(_diagnostic_long_rows(
                            loaded, format_name, selected_records, prompt_facts,
                            "S2", s2_batches, s2_errors,
                        ))
                    except Exception as exc:
                        details = _exception_details(exc)
                        cell_summary["schemes"]["S2"] = {"complete": False, "error": details}
                        output_complete = False
                        logger.error(
                            "E7 smoke-diagnose S2 failed family=%s format=%s type=%s message=%s frame=%s",
                            family, format_name, details["error_type"], details["error_message"],
                            details["innermost_traceback_frame"],
                        )
                        logger.error("%s", traceback.format_exc())

                    # S3 uses one unpadded prompt per forward.
                    try:
                        s3_residuals = np.empty(residual_shape, dtype=np.float32)
                        s3_batches: list[dict[str, Any]] = []
                        for prompt_index, record in enumerate(selected_records):
                            captured, lengths = _capture_residual_batch(loaded, [record])
                            expected_length = int(prompt_facts[prompt_index]["token_length"])
                            if lengths != [expected_length]:
                                raise ValueError("S3 tokenizer length differs from the stored P-D prompt length")
                            s3_residuals[prompt_index] = captured[0]
                            s3_batches.append({
                                "batch_index": prompt_index,
                                "records": [record],
                                "selected_local_rows": [0],
                                "max_length": expected_length,
                                "padded_by_row": [False],
                            })
                        s3_errors = _diagnostic_relative_matrix(
                            pd_residuals, selected_records, s3_residuals
                        )
                        cell_summary["schemes"]["S3"] = {
                            "complete": True,
                            "capture_batch_count": len(s3_batches),
                            "captured_prompt_count": len(selected_records),
                            "padded_prompt_count": 0,
                            "ten_c": _diagnostic_10c_summary(
                                s3_errors, n_layers, float(cfg["e7"]["check_pl_median_relative"])
                            ),
                        }
                        long_rows.extend(_diagnostic_long_rows(
                            loaded, format_name, selected_records, prompt_facts,
                            "S3", s3_batches, s3_errors,
                        ))
                    except Exception as exc:
                        details = _exception_details(exc)
                        cell_summary["schemes"]["S3"] = {"complete": False, "error": details}
                        output_complete = False
                        logger.error(
                            "E7 smoke-diagnose S3 failed family=%s format=%s type=%s message=%s frame=%s",
                            family, format_name, details["error_type"], details["error_message"],
                            details["innermost_traceback_frame"],
                        )
                        logger.error("%s", traceback.format_exc())

                    cell_summary["p_d_batch_size"] = layout["pd_batch_size"]
                    cell_summary["p_a_batch_size"] = layout["pa_batch_size"]
                    cell_summary["selected_p_d_batch_indices"] = layout["selected_pd_batch_indices"]
                    cell_summary["padded_status_counts"] = layout["padding_counts"]
                    cell_summary["wall_seconds"] = time.perf_counter() - cell_started
                    required_schemes = {"S1", "S2", "S3"}
                    cell_summary["complete"] = required_schemes.issubset(cell_summary["schemes"]) and all(
                        bool(cell_summary["schemes"][scheme].get("complete"))
                        for scheme in required_schemes
                    )
                    output_complete &= bool(cell_summary["complete"])
                    del pd_residuals
            finally:
                if loaded is not None:
                    del loaded
                gc.collect()
                torch.cuda.empty_cache()
        family_summary["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        family_summary["wall_seconds"] = time.perf_counter() - family_started
        summary["families"][family] = family_summary

    summary["completed_all_cells"] = bool(output_complete)
    summary["exit_code"] = 0 if output_complete else 2
    summary["utc_end"] = _utc_now()
    summary["wall_seconds"] = time.perf_counter() - started_all
    if not long_rows:
        logger.error("E7 smoke-diagnose completed without any long-format measurements")
        summary["exit_code"] = 2
        summary["completed_all_cells"] = False
    frame = pd.DataFrame(long_rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["model", "format", "scheme", "prompt_index", "layer"],
            kind="stable",
        ).reset_index(drop=True)
        frame.to_parquet(parquet_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    study_set = str(cfg["models"]["study_set"])
    if len(model_loads) != len(FAMILIES):
        model_loads = [
            {
                "model_id": str(cfg["models"][study_set][family]["hf_id"]),
                "model_revision": str(cfg["models"][study_set][family]["revision"]),
                "dtype": "torch.bfloat16",
                "dtype_override": None,
            }
            for family in FAMILIES
        ]
    run_provenance = build_provenance(
        cfg,
        repo_root=REPO,
        unit_key={"pass": "pa_smoke_diagnose", "cells": len(FAMILIES) * len(FORMATS)},
        entrypoint=Path(__file__),
        model_id="multi-family E7 smoke diagnostic",
        model_revision="per-family model_loads",
        dtype="torch.bfloat16",
        batch_size=max(
            [int(cfg["e7"]["batch_size"][family]) for family in FAMILIES]
            + list(pd_batch_sizes)
        ),
        expected_row_count=len(frame),
        run_git=run_git,
        model_loads=model_loads,
    )
    run_provenance["row_count"] = len(frame)
    run_provenance["utc_end"] = _utc_now()
    run_provenance["model_id"] = model_loads[0]["model_id"]
    run_provenance["model_revision"] = model_loads[0]["model_revision"]
    run_provenance["e7_input_sha256"] = _smoke_input_hashes(cfg, direction_path)
    run_provenance["e7_smoke_diagnose"] = {
        "directions_split_only": True,
        "analysis_tag_lock_consulted": False,
        "split_used": {family: dict(split_used[family]) for family in FAMILIES},
        "cell_summaries": {
            family: dict(summary["families"][family]["formats"]) for family in FAMILIES
        },
    }
    provenance_path.write_text(
        json.dumps(run_provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    logger.info(
        "E7 smoke-diagnose final summary %s",
        json.dumps(summary, sort_keys=True, default=str),
    )
    return int(summary["exit_code"])


def _run(
    cfg: dict[str, Any],
    prompt_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    primary_ids: Mapping[str, set[str]],
    pl_source_records: Sequence[Mapping[str, Any]],
    logger: logging.Logger,
) -> int:
    """Load each pinned model once and run all format-matched P-A units."""
    run_git = capture_run_git(REPO, cfg)
    job_started = time.perf_counter()
    summary: dict[str, Any] = {
        "utc_start": _utc_now(), "git_commit": run_git.git_commit, "git_dirty": run_git.git_dirty,
        "gpu_name": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "profile": _runtime_profile(), "families": {}, "exit_code": 0,
    }
    for family in FAMILIES:
        started = time.perf_counter()
        loaded = None
        family_summary: dict[str, Any] = {}
        format_name = "model_load"
        try:
            torch.cuda.reset_peak_memory_stats()
            loaded = models.load(cfg, family, device=torch.device("cuda"))
            for format_name in FORMATS:
                family_summary[format_name] = _run_format(
                    cfg, loaded, run_git, format_name, prompt_rows[format_name],
                    primary_ids[format_name], pl_source_records, logger,
                )
            family_summary["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            family_summary["wall_seconds"] = time.perf_counter() - started
            summary["families"][family] = family_summary
            logger.info("P-A family summary %s", json.dumps(family_summary, sort_keys=True, default=str))
        except Exception as exc:
            summary["exit_code"] = 2
            details = _exception_details(exc)
            failed_formats = FORMATS if format_name == "model_load" else (format_name,)
            for failed_format in failed_formats:
                family_summary[failed_format] = {"passed": False, **details}
                summary.setdefault("errors", []).append(
                    {"family": family, "format": failed_format, **details}
                )
            if loaded is not None:
                family_summary["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            family_summary["wall_seconds"] = time.perf_counter() - started
            summary["families"][family] = family_summary
            logger.error(
                "P-A failed family=%s format=%s type=%s message=%s frame=%s",
                family, format_name, details["error_type"], details["error_message"],
                details["innermost_traceback_frame"],
            )
            break
        finally:
            if loaded is not None:
                del loaded
            gc.collect()
            torch.cuda.empty_cache()
    summary["utc_end"] = _utc_now()
    summary["wall_seconds"] = time.perf_counter() - job_started
    summary_path = REPO / str(cfg["e7"]["results_dir"]) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    logger.info("P-A final summary %s", json.dumps(summary, sort_keys=True, default=str))
    return int(summary["exit_code"])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the directions smoke or primary P-A pass after its required preflight."""
    parser = argparse.ArgumentParser(description="Run E7 P-A component/head attribution")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-e7", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-diagnose", action="store_true")
    args = parser.parse_args(argv)
    from vilens.config import load_config

    cfg = load_config(args.config)
    logger = setup_logging("job_e7_attribute", cfg["paths"]["logs"])
    try:
        if args.smoke and args.smoke_diagnose:
            raise ValueError("--smoke and --smoke-diagnose are mutually exclusive")
        if args.smoke or args.smoke_diagnose:
            smoke_rows, split_used, all_direction_records = _smoke_prompt_records(cfg)
            if any(
                str(row.get("split")) == "test"
                for family in FAMILIES
                for fmt in FORMATS
                for row in smoke_rows[family][fmt]
            ):
                raise AssertionError("smoke selection contains a test-split record")
            if not _preflight(
                cfg, smoke_rows, gpu=not args.preflight_only, smoke=True,
                smoke_diagnose=args.smoke_diagnose,
                smoke_rows=smoke_rows,
                all_direction_records=all_direction_records,
            ):
                logger.error("E7 smoke preflight failed; no study weights loaded")
                return 2
            if args.preflight_only:
                mode = "smoke-diagnose" if args.smoke_diagnose else "smoke"
                logger.info("E7 %s preflight-only passed; no model weights loaded", mode)
                return 0
            if args.smoke_diagnose:
                return _run_smoke_diagnose(cfg, smoke_rows, split_used, logger)
            return _run_smoke(cfg, smoke_rows, split_used, all_direction_records, logger)
        prompt_rows, primary_ids = _prompt_records(cfg)
        smoke_rows, _smoke_splits, _all_direction_records = _smoke_prompt_records(cfg)
        if not _preflight(
            cfg, prompt_rows, gpu=not args.preflight_only,
            smoke_rows=smoke_rows, primary_ids=primary_ids,
        ):
            logger.error("P-A preflight failed; no study weights loaded")
            return 2
        if args.preflight_only:
            logger.info("P-A preflight-only passed; no model weights loaded")
            return 0
        if not args.confirm_e7:
            logger.error("Primary E7 requires --confirm-e7")
            return 2
        lock_records = [dict(row) for fmt in FORMATS for row in prompt_rows[fmt]]
        locks.ensure_adjudication_committed(REPO)
        locks.ensure_model_access_allowed(cfg, lock_records)
        pl_source_records = _pl_source_prompt_records(cfg)
        return _run(cfg, prompt_rows, primary_ids, pl_source_records, logger)
    except Exception as exc:
        logger.error("P-A preflight/run failed: %s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
