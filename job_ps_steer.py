"""E8 class-mean steering job; trial is directions-split only."""

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
import math
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from vilens import io, locks, models, steer
from vilens.data import load_concepts_jsonl, load_prompt_file, prompt_file_paths
from vilens.derived import stream
from vilens.manifest import sha256_file
from vilens.preflight import format_checks, preflight_ok, run_preflight
from vilens.provenance import RunGit, build as build_provenance, capture_run_git


FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("translation", "repetition")
GENERATION_COLUMNS = [
    "model", "model_revision", "pass", "source", "format", "concept_id", "layer",
    "setting", "k", "answer_text", "n_new_tokens", "competence_class", "shared_ru_source",
]
DELTA_COLUMNS = [
    "model", "model_revision", "pass", "source", "format", "concept_id", "layer",
    "setting", "k", "metric", "en_token_id", "vi_token_id", "value",
]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a small committed JSONL manifest or released prompt file."""
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _concepts(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the typed released concepts used for target forms and A10 scoring."""
    frame = load_concepts_jsonl(REPO / str(cfg["paths"]["concepts_jsonl"]))
    return {str(row["concept_id"]): dict(row) for row in frame.to_dict(orient="records")}


def _load_competence_rows(path: Path) -> list[dict[str, Any]]:
    """Read only the four A10 columns needed for E8 selection."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["format", "concept_id", "competence_class"])
    return table.to_pylist()


def _primary_selection(cfg: Mapping[str, Any]) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Select competent E8 concepts with the frozen order and e8_select helper."""
    order_path = REPO / "derived/e8_order.jsonl"
    order_rows = _read_jsonl(order_path)
    selected: dict[str, list[str]] = {}
    quotas: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        competence_path = REPO / "results/analysis/primary" / f"family={family}" / "competence.parquet"
        rows = _load_competence_rows(competence_path)
        ids, family_quotas = steer.selected_competent_ids(
            order_rows, rows, family, int(cfg["manifests"]["e8_max_items"])
        )
        selected[family] = ids
        quotas[family] = family_quotas
    return selected, quotas


def _prepare_records(
    cfg: dict[str, Any], source: str
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Build locked per-family E8 records; the trial excludes D12 prompt overlaps."""
    concepts = _concepts(cfg)
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = {
        family: {format_name: [] for format_name in FORMATS} for family in FAMILIES
    }
    selected_counts: dict[str, dict[str, int]] = {}
    if source == "trial":
        prompt_path = prompt_file_paths(cfg)["directions_matched"]
        records = load_prompt_file(REPO / prompt_path)
        excluded = {
            (str(row["concept_id"]), str(row["format"]), str(row["lang"]))
            for row in cfg["data"]["pretag_excluded_directions"]
        }
        selected = [
            dict(row)
            for row in records
            if row.get("split") == "validate"
            and row.get("lang") == "vi"
            and (str(row.get("concept_id")), str(row.get("format")), str(row.get("lang"))) not in excluded
        ]
        if any(row.get("split") == "test" for row in selected):
            raise AssertionError("trial selection includes a test-split prompt")
        for row in selected:
            row["condition"] = "diac"
            by_family_records = {family: row for family in FAMILIES}
            for family in by_family_records:
                by_family[family][str(row["format"])].append(dict(row))
        selected_counts = {
            family: {format_name: len(by_family[family][format_name]) for format_name in FORMATS}
            for family in FAMILIES
        }
        return by_family, concepts, selected_counts

    if source != "primary":
        raise ValueError("source must be trial or primary")
    selected_ids, quotas = _primary_selection(cfg)
    file_paths = prompt_file_paths(cfg)
    prompt_rows = {
        "translation": load_prompt_file(REPO / file_paths[str(cfg["data"]["primary_prompt_files"]["translation_diac"])]),
        "repetition": load_prompt_file(REPO / file_paths[str(cfg["data"]["primary_prompt_files"]["repetition_diac"])]),
    }
    for family in FAMILIES:
        chosen = set(selected_ids[family])
        for format_name in FORMATS:
            rows = [dict(row) for row in prompt_rows[format_name] if str(row["concept_id"]) in chosen]
            if len(rows) != len(chosen):
                raise AssertionError(f"{family}/{format_name} prompts={len(rows)}; selected concepts={len(chosen)}")
            if any(row.get("split") != "test" for row in rows):
                raise AssertionError("primary E8 records must be main test prompts")
            by_family[family][format_name] = rows
        selected_counts[family] = {
            "translation": len(by_family[family]["translation"]),
            "repetition": len(by_family[family]["repetition"]),
            **{f"quota_{key}": value for key, value in quotas[family].items()},
        }
    return by_family, concepts, selected_counts


def _pd_input_paths(cfg: Mapping[str, Any], family: str, format_name: str) -> list[Path]:
    """Resolve the P-D matched parquet index, sidecar, and residual-array paths."""
    key = {"family": family, "pass": "pd", "source": "matched", "format": format_name}
    parquet, sidecar = io.unit_paths(REPO / str(cfg["pd"]["results_dir"]), key)
    return [parquet, sidecar, parquet.with_suffix(".resid.npy")]


def _input_table(
    cfg: Mapping[str, Any], source: str, records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    selected_counts: Mapping[str, Mapping[str, int]],
) -> tuple[str, list[str]]:
    """Return custom E8 file checks and readable input requirements."""
    rows: list[tuple[str, str, str]] = []
    order = REPO / "derived/e8_order.jsonl"
    seeds = REPO / "derived/seeds.json"
    order_ok = order.is_file()
    rows.append(("E8 item order", str(order.relative_to(REPO)), "PASS" if order_ok else "FAIL"))
    seed_ok = seeds.is_file()
    if seed_ok:
        try:
            seed_doc = json.loads(seeds.read_text(encoding="utf-8"))
            recorded_key = seed_doc.get("steer_random_direction", {}).get("key", "")
            normalized_key = str(recorded_key).replace("<family>", "{family}").replace("<layer>", "{layer}")
            seed_ok = normalized_key == cfg["steering"]["random_key_template"]
        except (OSError, json.JSONDecodeError):
            seed_ok = False
    rows.append(("seed streams", str(seeds.relative_to(REPO)), "PASS" if seed_ok else "FAIL"))
    for family in FAMILIES:
        for format_name in FORMATS:
            paths = _pd_input_paths(cfg, family, format_name)
            missing = [str(path.relative_to(REPO)) for path in paths if not path.is_file()]
            status = "FAIL" if missing else "PASS"
            rows.append(("P-D fit inputs", f"{family}/{format_name}: " + (", ".join(missing) if missing else "all present"), status))
            if not missing:
                parquet_path, array_path = paths[0], paths[2]
                try:
                    import pyarrow.parquet as pq

                    index_rows = pq.read_table(
                        parquet_path, columns=["row", "format", "split", "lang"]
                    ).to_pylist()
                    fit_counts = {
                        language: sum(
                            str(row.get("split")) == "fit"
                            and str(row.get("lang")) == language
                            and str(row.get("format")) == format_name
                            for row in index_rows
                        )
                        for language in ("en", "vi")
                    }
                    residual_shape = np.load(array_path, mmap_mode="r", allow_pickle=False).shape
                    expected_fit = int(cfg["reference_facts"]["prompt_records"]["directions"]["fit_concepts"])
                    fit_ok = all(value == expected_fit for value in fit_counts.values())
                    if len(residual_shape) != 3 or residual_shape[0] != len(index_rows):
                        fit_ok = False
                    rows.append(("P-D fit counts", f"{family}/{format_name}: en={fit_counts['en']} vi={fit_counts['vi']}; residual_shape={residual_shape}", "PASS" if fit_ok else "FAIL"))
                except Exception as exc:
                    rows.append(("P-D fit counts", f"{family}/{format_name}: {type(exc).__name__}", "FAIL"))
            if source == "primary":
                count = len(records[family][format_name])
                rows.append(("selected prompts", f"{family}/{format_name}: {count}", "PASS" if count == int(cfg["steering"]["max_items"]) else "FAIL"))
        if source == "primary":
            competence_path = REPO / "results/analysis/primary" / f"family={family}" / "competence.parquet"
            rows.append(("A10 competence", str(competence_path.relative_to(REPO)), "PASS" if competence_path.is_file() else "FAIL"))
    headers = ("CHECK", "INPUT / COUNT", "STATUS")
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(3)]
    line = " | ".join(headers[index].ljust(widths[index]) for index in range(3))
    divider = "-+-".join("-" * widths[index] for index in range(3))
    body = [" | ".join(value[index].ljust(widths[index]) for index in range(3)) for value in rows]
    return "\n".join([line, divider, *body]), [row[1] for row in rows if row[2] == "FAIL"]


def _preflight(
    cfg: dict[str, Any], source: str,
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    selected_counts: Mapping[str, Mapping[str, int]],
    logger: logging.Logger,
    *, gpu: bool,
) -> bool:
    """Run shared model/input/lock preflight plus E8-specific input checks."""
    flat_records = [dict(row) for family in FAMILIES for fmt in FORMATS for row in records[family][fmt]]
    if source == "primary":
        steer.ensure_primary_locks(cfg, REPO, flat_records)
    else:
        locks.ensure_adjudication_committed(REPO)
        locks.ensure_model_access_allowed(cfg, flat_records)
    if source == "trial" and any(row.get("split") == "test" for row in flat_records):
        raise AssertionError("trial E8 records contain a test-split prompt")
    prompt_files = (
        [
            "data/external/viLens-concepts/prompts/directions_matched.jsonl",
            "data/external/viLens-concepts/concepts.jsonl",
        ]
        if source == "trial"
        else [
            "data/external/viLens-concepts/prompts/translation_diac_set1.jsonl",
            "data/external/viLens-concepts/prompts/repetition_diac_set1.jsonl",
            "data/external/viLens-concepts/concepts.jsonl",
        ]
    )
    out_dir = cfg["steering"]["trial_results_dir"] if source == "trial" else cfg["steering"]["results_dir"]
    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E8",
        gpu=gpu,
        inputs=prompt_files,
        families=list(FAMILIES),
        records=flat_records,
        output_dirs=[Path(out_dir)],
        entrypoint=Path(__file__),
    )
    print(format_checks(checks))
    custom, failures = _input_table(cfg, source, records, selected_counts)
    print("E8 input checks:")
    print(custom)
    if source == "primary":
        print("E8 primary selection counts and stratum quotas:")
        for family in FAMILIES:
            values = selected_counts[family]
            quota_text = ", ".join(f"{key.removeprefix('quota_')}={value}" for key, value in values.items() if key.startswith("quota_"))
            print(f"  {family}: translation={values['translation']}; repetition={values['repetition']}; quotas: {quota_text}")
    shared_ok = preflight_ok(checks)
    if failures:
        logger.error("E8 required inputs missing or selected count mismatch: %s", failures)
    return shared_ok and not failures


def _vector_file(output_dir: Path, family: str, format_name: str) -> tuple[Path, str]:
    """Write the small fitted-vector file once and return its path and digest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"family={family}__format={format_name}__vectors.npy"
    vectors, _norms, _counts = steer.steering_vectors(family, format_name, results_dir=REPO / "results/pd")
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream_out:
        np.save(stream_out, vectors.astype(np.float32, copy=False), allow_pickle=False)
    temporary.replace(path)
    return path, sha256_file(path)


def _provenance(
    cfg: Mapping[str, Any], run_git: RunGit, loaded: Any, source: str,
    unit_key: Mapping[str, str | int], expected_rows: int,
) -> dict[str, Any]:
    """Build rule-37 provenance for one E8 kind/layer unit."""
    spec = loaded.spec
    if spec is None:
        raise RuntimeError("loaded model lacks its registered spec")
    return build_provenance(
        cfg,
        repo_root=REPO,
        unit_key=unit_key,
        entrypoint=Path(__file__),
        model_id=spec.hf_id,
        model_revision=spec.revision,
        dtype="torch.bfloat16",
        batch_size=int(cfg["steering"]["batch_size"][loaded.family]),
        expected_row_count=expected_rows,
        run_git=run_git,
    )


def _unit_key(family: str, source: str, format_name: str, layer: str | int, kind: str) -> dict[str, str | int]:
    """Create stable family/format/layer/kind result-unit identity."""
    return {"family": family, "pass": "ps", "source": source, "format": format_name, "layer": layer, "kind": kind}


def _write_unit(
    frame: pd.DataFrame, output_dir: Path, cfg: Mapping[str, Any], run_git: RunGit,
    loaded: Any, source: str, format_name: str, layer: str | int, kind: str,
    columns: Sequence[str], vector_path: Path, vector_sha: str,
    fit_counts: Mapping[str, int], vector_norm: float,
) -> str:
    """Write one sorted parquet unit and provenance sidecar, respecting resume checks."""
    unit_key = _unit_key(loaded.family, source, format_name, layer, kind)
    provenance = _provenance(cfg, run_git, loaded, source, unit_key, len(frame))
    provenance.update(
        {
            "vectors_file": str(vector_path.relative_to(REPO)),
            "vectors_sha256": vector_sha,
            "fit_counts": dict(fit_counts),
            "vector_norm": float(vector_norm),
        }
    )
    complete, reason = io.unit_is_complete(
        output_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"]
    )
    parquet, sidecar = io.unit_paths(output_dir, unit_key)
    if complete and sidecar.is_file():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        complete = (
            saved.get("vectors_sha256") == vector_sha
            and saved.get("fit_counts") == dict(fit_counts)
            and math.isclose(float(saved.get("vector_norm", float("nan"))), float(vector_norm), rel_tol=0, abs_tol=0)
        )
    if complete:
        return "skipped"
    io.write_unit(
        frame,
        output_dir,
        provenance,
        columns=columns,
        sort_by=[column for column in ("model", "format", "concept_id", "layer", "setting") if column in columns],
        compression=str(cfg["runtime"]["parquet_compression"]),
    )
    return "written"


def _generation_rows(
    loaded: Any, source: str, format_name: str, records: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]], concepts: Mapping[str, Mapping[str, Any]],
    *, layer: int, setting: str, k_value: float,
) -> list[dict[str, Any]]:
    """Score generated outputs through the frozen A10 competence_rows function."""
    from vilens.analysis.competence import competence_rows

    model = str(loaded.spec.hf_id)
    revision = str(loaded.spec.revision)
    generated = [
        {
            "model": model,
            "model_revision": revision,
            "pass": "ps",
            "source": source,
            "format": format_name,
            "condition": str(record.get("condition", "diac")),
            "concept_id": str(record["concept_id"]),
            "answer": str(result["answer_text"]),
            "n_new_tokens": int(result["n_new_tokens"]),
            "layer": int(layer),
            "setting": setting,
            "k": float(k_value),
        }
        for record, result in zip(records, results, strict=True)
    ]
    scored = competence_rows(generated, concepts)
    return [
        {
            "model": row["model"],
            "model_revision": row["model_revision"],
            "pass": "ps",
            "source": source,
            "format": format_name,
            "concept_id": str(row["concept_id"]),
            "layer": int(layer),
            "setting": setting,
            "k": float(k_value),
            "answer_text": str(row["answer"]),
            "n_new_tokens": int(row["n_new_tokens"]),
            "competence_class": str(row["competence_class"]),
            "shared_ru_source": bool(row["shared_ru_source"]),
        }
        for row in scored.to_dict(orient="records")
    ]


def _delta_rows(
    loaded: Any, source: str, format_name: str, records: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]], *, layer: int, setting: str, k_value: float,
    vector: np.ndarray | None, batch_size: int,
) -> list[dict[str, Any]]:
    """Build long-format E8 delta rows from fp32 first-content-token log probabilities."""
    values = steer.delta_first_batch(
        loaded,
        records,
        concepts,
        family=loaded.family,
        format=format_name,
        vector=vector,
        layer=layer if vector is not None else None,
        k=k_value,
        batch_size=batch_size,
    )
    rows = []
    for row in values:
        rows.append(
            {
                "model": str(loaded.spec.hf_id),
                "model_revision": str(loaded.spec.revision),
                "pass": "ps",
                "source": source,
                "format": format_name,
                "concept_id": str(row["concept_id"]),
                "layer": int(layer),
                "setting": setting,
                "k": float(k_value),
                "metric": "delta_first",
                "en_token_id": int(row["en_token_id"]),
                "vi_token_id": int(row["vi_token_id"]),
                "value": float(row["value"]),
            }
        )
    return rows


def _generation(
    loaded: Any, records: Sequence[Mapping[str, Any]], source: str,
    cfg: Mapping[str, Any], *, vector: np.ndarray | None = None,
    layer: int | None = None, k: float = 0.0, batched: bool = True,
) -> list[dict[str, Any]]:
    """Generate in configured left-padded batches or one prompt at a time."""
    settings = cfg["steering"]
    batch_size = int(settings["batch_size"][loaded.family]) if batched else 1
    results: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        results.extend(
            steer.greedy_generate_batch(
                loaded,
                [str(row["prompt"]) for row in batch],
                max_new_tokens=int(settings["gen_max_new_tokens"]),
                stop_strings=settings["stop_strings"],
                vector=vector,
                layer=layer,
                k=k,
            )
        )
    return results


def _prompt_logits_matches(
    loaded: Any, records: Sequence[Mapping[str, Any]], batch_size: int, layer: int,
    vector: np.ndarray,
) -> tuple[int, int]:
    """Compare prompt-only logits with a zero-strength hook for each trial item."""
    from vilens.steer import _left_pad, _pad_id, steer_block

    device = next(loaded.model.parameters()).device
    matches = 0
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        sequences = [loaded.tokenizer(str(row["prompt"]), add_special_tokens=True)["input_ids"] for row in batch]
        ids, mask, position_ids = _left_pad(sequences, _pad_id(loaded.tokenizer), device)
        with torch.inference_mode():
            baseline = loaded.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits
            with steer_block(loaded, layer, vector, 0.0):
                hooked = loaded.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits
        for row in range(len(batch)):
            total += 1
            matches += int(torch.equal(baseline[row], hooked[row]))
    return matches, total


def _bootstrap_flip_peak(
    rows_by_layer: Mapping[int, Sequence[str]], family: str, format_name: str, seed: int,
    resamples: int, key_template: str,
) -> dict[str, Any]:
    """Bootstrap the earliest maximal k=1 flip layer over concepts."""
    layers = sorted(rows_by_layer)
    if not layers:
        return {"layer": None, "ci_low": None, "ci_high": None, "valid_draws": 0}
    concept_order = list(rows_by_layer[layers[0]])
    n_items = len(concept_order)
    flip = np.asarray(
        [[rows_by_layer[layer][index] == "EN" for index in range(n_items)] for layer in layers], dtype=np.float64
    )
    point_rates = flip.mean(axis=1) if n_items else np.zeros(len(layers))
    max_index = int(np.flatnonzero(point_rates == point_rates.max())[0])
    rng = stream(int(seed), key_template.format(family=family, format=format_name))
    peaks: list[int] = []
    for _ in range(int(resamples)):
        indices = rng.integers(0, n_items, size=n_items) if n_items else np.array([], dtype=int)
        rates = flip[:, indices].mean(axis=1) if n_items else np.zeros(len(layers))
        peaks.append(layers[int(np.flatnonzero(rates == rates.max())[0])])
    return {
        "layer": layers[max_index],
        "flip_rate": float(point_rates[max_index]),
        "ci_low": float(np.quantile(peaks, 0.025)),
        "ci_high": float(np.quantile(peaks, 0.975)),
        "valid_draws": len(peaks),
        "depth": float(layers[max_index] / int(getattr(rows_by_layer, "n_layers", max(layers)))),
    }


def _save_unit(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *, source: str,
    format_name: str, layer: int, setting: str, kind: str, loaded: Any,
    output_dir: Path, cfg: Mapping[str, Any], run_git: RunGit,
    vectors: tuple[Path, str, Mapping[str, int], float],
) -> str:
    """Persist one resume-safe E8 family × format × layer × output-kind table."""
    vector_path, vector_sha, fit_counts, vector_norm = vectors
    frame = pd.DataFrame.from_records(rows, columns=list(columns))
    if len(frame) and "value" in frame and not np.isfinite(pd.to_numeric(frame["value"], errors="coerce")).all():
        raise FloatingPointError(f"non-finite values in {loaded.family}/{format_name}/{layer}/{setting}/{kind}")
    unit_layer: str | int = "baseline" if setting == "baseline" else int(layer)
    unit_key = _unit_key(loaded.family, source, format_name, unit_layer, f"{kind}-{setting}")
    provenance = _provenance(cfg, run_git, loaded, source, unit_key, len(frame))
    provenance.update(
        {
            "vectors_file": str(vector_path.relative_to(REPO)),
            "vectors_sha256": vector_sha,
            "fit_counts": dict(fit_counts),
            "vector_norm": float(vector_norm),
        }
    )
    complete, _reason = io.unit_is_complete(output_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"])
    parquet_path, sidecar_path = io.unit_paths(output_dir, unit_key)
    if complete and sidecar_path.is_file():
        saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
        complete = (
            saved.get("vectors_sha256") == vector_sha
            and saved.get("fit_counts") == dict(fit_counts)
            and math.isclose(float(saved.get("vector_norm", float("nan"))), float(vector_norm), rel_tol=0, abs_tol=0)
        )
    if complete:
        return "skipped"
    io.write_unit(
        frame,
        output_dir,
        provenance,
        columns=columns,
        sort_by=[column for column in ("model", "format", "concept_id", "layer", "setting") if column in columns],
        compression=str(cfg["runtime"]["parquet_compression"]),
    )
    return "written"


def _run_family(
    cfg: dict[str, Any], source: str, family: str,
    records_by_format: Mapping[str, Sequence[Mapping[str, Any]]],
    concepts: Mapping[str, Mapping[str, Any]], logger: logging.Logger, run_git: RunGit,
) -> dict[str, Any]:
    """Run all E8 settings for one model family and persist each unit as it completes."""
    start = time.perf_counter()
    loaded = models.load(cfg, family, device="cuda")
    output_dir = REPO / (cfg["steering"]["trial_results_dir"] if source == "trial" else cfg["steering"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg["steering"]["batch_size"][family])
    vectors: dict[str, tuple[np.ndarray, np.ndarray, dict[str, int], Path, str, np.ndarray]] = {}
    for format_name in FORMATS:
        u, norms, fit_counts = steer.steering_vectors(
            family, format_name, results_dir=REPO / str(cfg["pd"]["results_dir"])
        )
        vector_path = output_dir / f"family={family}__format={format_name}__vectors.npy"
        with vector_path.open("wb") as stream_out:
            np.save(stream_out, u.astype(np.float32, copy=False), allow_pickle=False)
        vector_sha = sha256_file(vector_path)
        random_vectors = steer.random_control(
            u, family, int(cfg["seed"]), key_template=str(cfg["steering"]["random_key_template"])
        )
        vectors[format_name] = (u, norms, fit_counts, vector_path, vector_sha, random_vectors)

    translation_records = list(records_by_format["translation"])
    repetition_records = list(records_by_format["repetition"])
    generation_seconds = 0.0
    delta_seconds = 0.0
    batched_start = time.perf_counter()
    batched_baseline = _generation(loaded, translation_records, source, cfg, batched=True)
    batched_elapsed = time.perf_counter() - batched_start
    unbatched_start = time.perf_counter()
    unbatched_baseline = _generation(loaded, translation_records, source, cfg, batched=False)
    unbatched_elapsed = time.perf_counter() - unbatched_start
    batch_matches = sum(
        left["token_ids"] == right["token_ids"]
        for left, right in zip(batched_baseline, unbatched_baseline, strict=True)
    )
    batch_equal = batch_matches == len(translation_records)
    use_batched_generation = batch_equal
    baseline_generation = batched_baseline if use_batched_generation else unbatched_baseline
    generation_seconds += batched_elapsed if use_batched_generation else unbatched_elapsed

    family_summary: dict[str, Any] = {
        "family": family,
        "record_counts": {fmt: len(records_by_format[fmt]) for fmt in FORMATS},
        "vector_fit_counts": {fmt: vectors[fmt][2] for fmt in FORMATS},
        "u_norm_min_max": {
            fmt: [float(vectors[fmt][1][1:].min()), float(vectors[fmt][1][1:].max())]
            for fmt in FORMATS
        },
        "batched_unbatched": {"matches": batch_matches, "total": len(translation_records), "equal": batch_equal},
        "generation_rows": 0,
        "delta_rows": 0,
        "k0_exact": {"logit_matches": 0, "logit_total": 0, "generation_matches": 0, "generation_total": 0},
        "rates_k1": {},
        "rates_by_setting": {"baseline": {"0": {}}},
        "units": [],
        "errors": [],
    }
    baseline_class_rows = _generation_rows(
        loaded, source, "translation", translation_records, baseline_generation, concepts,
        layer=0, setting="baseline", k_value=0.0,
    )
    fit_info = vectors["translation"]
    baseline_vector_info = (fit_info[3], fit_info[4], fit_info[2], float(fit_info[1][0]))
    if baseline_class_rows:
        family_summary["rates_by_setting"]["baseline"]["0"] = {
            "flip_rate": sum(row["competence_class"] == "EN" for row in baseline_class_rows) / len(baseline_class_rows),
            "damage_rate": sum(row["competence_class"] == "DAMAGE" for row in baseline_class_rows) / len(baseline_class_rows),
            "other_lang_rate": sum(row["competence_class"] == "OTHER-LANG" for row in baseline_class_rows) / len(baseline_class_rows),
            "n": len(baseline_class_rows),
        }
        status = _save_unit(
            baseline_class_rows, GENERATION_COLUMNS, source=source, format_name="translation", layer=0,
            setting="baseline", kind="generation", loaded=loaded, output_dir=output_dir,
            cfg=cfg, run_git=run_git, vectors=baseline_vector_info,
        )
        family_summary["units"].append({"format": "translation", "layer": "baseline", "kind": "generation", "rows": len(baseline_class_rows), "status": status})
        family_summary["generation_rows"] += len(baseline_class_rows)

    for format_name, format_records in (("translation", translation_records), ("repetition", repetition_records)):
        delta_start = time.perf_counter()
        values = steer.delta_first_batch(
            loaded, format_records, concepts, family=family, format=format_name,
            batch_size=batch_size,
        )
        delta_seconds += time.perf_counter() - delta_start
        rows = [
            {
                "model": str(loaded.spec.hf_id), "model_revision": str(loaded.spec.revision),
                "pass": "ps", "source": source, "format": format_name,
                "concept_id": str(item["concept_id"]), "layer": 0, "setting": "baseline", "k": 0.0,
                "metric": "delta_first", "en_token_id": int(item["en_token_id"]),
                "vi_token_id": int(item["vi_token_id"]), "value": float(item["value"]),
            }
            for item in values
        ]
        info = vectors[format_name]
        status = _save_unit(
            rows, DELTA_COLUMNS, source=source, format_name=format_name, layer=0,
            setting="baseline", kind="delta", loaded=loaded, output_dir=output_dir,
            cfg=cfg, run_git=run_git,
            vectors=(info[3], info[4], info[2], float(info[1][0])),
        )
        family_summary["units"].append({"format": format_name, "layer": "baseline", "kind": "delta", "rows": len(rows), "status": status})
        family_summary["delta_rows"] += len(rows)

    # Outputs under k=1 are collected by layer for flip-rate and concept bootstrap summaries.
    k1_classes: dict[int, list[str]] = {}
    for layer in range(1, int(loaded.n_layers) + 1):
        for format_name, format_records in (("translation", translation_records), ("repetition", repetition_records)):
            info = vectors[format_name]
            u, norms, fit_counts, vector_path, vector_sha, random_vectors = info
            logits_matches, logits_total = _prompt_logits_matches(
                loaded, format_records, batch_size, layer, u[layer]
            ) if source == "trial" else (0, 0)
            family_summary["k0_exact"]["logit_matches"] += logits_matches
            family_summary["k0_exact"]["logit_total"] += logits_total

            layer_settings = [
                (f"k{float(k_value):g}", float(k_value), u[layer])
                for k_value in cfg["steering"]["k_grid"]
            ]
            layer_settings.append(("random", float(cfg["steering"]["random_control_k"]), random_vectors[layer]))
            for setting, k_value, vector in layer_settings:
                delta_start = time.perf_counter()
                delta_values = steer.delta_first_batch(
                    loaded, format_records, concepts, family=family, format=format_name,
                    vector=vector, layer=layer, k=k_value, batch_size=batch_size,
                )
                delta_seconds += time.perf_counter() - delta_start
                delta_rows = [
                    {
                        "model": str(loaded.spec.hf_id), "model_revision": str(loaded.spec.revision),
                        "pass": "ps", "source": source, "format": format_name,
                        "concept_id": str(item["concept_id"]), "layer": layer, "setting": setting,
                        "k": float(k_value), "metric": "delta_first",
                        "en_token_id": int(item["en_token_id"]), "vi_token_id": int(item["vi_token_id"]),
                        "value": float(item["value"]),
                    }
                    for item in delta_values
                ]
                status = _save_unit(
                    delta_rows, DELTA_COLUMNS, source=source, format_name=format_name, layer=layer,
                    setting=setting, kind="delta", loaded=loaded, output_dir=output_dir,
                    cfg=cfg, run_git=run_git,
                    vectors=(vector_path, vector_sha, fit_counts, float(norms[layer])),
                )
                family_summary["units"].append({"format": format_name, "layer": layer, "setting": setting, "kind": "delta", "rows": len(delta_rows), "status": status})
                family_summary["delta_rows"] += len(delta_rows)
                if format_name == "translation":
                    generation_start = time.perf_counter()
                    generated = _generation(
                        loaded, format_records, source, cfg, vector=vector, layer=layer, k=k_value,
                        batched=use_batched_generation,
                    )
                    generation_seconds += time.perf_counter() - generation_start
                    generation_rows = _generation_rows(
                        loaded, source, format_name, format_records, generated, concepts,
                        layer=layer, setting=setting, k_value=k_value,
                    )
                    family_summary["rates_by_setting"].setdefault(setting, {})[str(layer)] = {
                        "flip_rate": sum(row["competence_class"] == "EN" for row in generation_rows) / max(len(generation_rows), 1),
                        "damage_rate": sum(row["competence_class"] == "DAMAGE" for row in generation_rows) / max(len(generation_rows), 1),
                        "other_lang_rate": sum(row["competence_class"] == "OTHER-LANG" for row in generation_rows) / max(len(generation_rows), 1),
                        "n": len(generation_rows),
                    }
                    if setting == "k1":
                        k1_classes[layer] = [str(row["competence_class"]) for row in generation_rows]
                        family_summary["rates_k1"][str(layer)] = family_summary["rates_by_setting"]["k1"][str(layer)]
                    status = _save_unit(
                        generation_rows, GENERATION_COLUMNS, source=source, format_name=format_name,
                        layer=layer, setting=setting, kind="generation", loaded=loaded,
                        output_dir=output_dir, cfg=cfg, run_git=run_git,
                        vectors=(vector_path, vector_sha, fit_counts, float(norms[layer])),
                    )
                    family_summary["units"].append({"format": format_name, "layer": layer, "setting": setting, "kind": "generation", "rows": len(generation_rows), "status": status})
                    family_summary["generation_rows"] += len(generation_rows)

            if source == "trial":
                # The zero-strength hook must preserve full prompt logits and greedy generation exactly.
                k0_gen = _generation(
                    loaded, format_records, source, cfg, vector=u[layer], layer=layer, k=0.0,
                    batched=use_batched_generation,
                ) if format_name == "translation" else []
                if format_name == "translation":
                    matches = sum(left["token_ids"] == right["token_ids"] for left, right in zip(k0_gen, baseline_generation, strict=True))
                    family_summary["k0_exact"]["generation_matches"] += matches
                    family_summary["k0_exact"]["generation_total"] += len(k0_gen)

    if source == "trial":
        n_layers = int(loaded.n_layers)
        flip_peak = _bootstrap_flip_peak(
            k1_classes,
            family,
            "translation",
            int(cfg["seed"]),
            int(cfg["steering"]["bootstrap_resamples"]),
            str(cfg["steering"]["bootstrap_key_template"]),
        )
        flip_peak["depth"] = float(flip_peak["layer"] / n_layers) if flip_peak["layer"] is not None else None
        family_summary["flip_peak_k1"] = flip_peak
        example_layers = sorted({1, max(1, n_layers // 2), n_layers})
        family_summary["example_layer_rates_k1"] = {
            str(layer): family_summary["rates_k1"].get(str(layer)) for layer in example_layers
        }
        random_ok = True
        random_checks = []
        for format_name in FORMATS:
            u, norms, _counts, _path, _sha, random_vectors = vectors[format_name]
            for layer in range(1, int(loaded.n_layers) + 1):
                observed = float(np.linalg.norm(random_vectors[layer]))
                expected = float(norms[layer])
                passed = math.isclose(observed, expected, rel_tol=1e-5, abs_tol=1e-6)
                random_ok = random_ok and passed
                random_checks.append(passed)
        # _save_unit validates every baseline and steered delta value before writing.
        finite_ok = True
        unit_expected = {
            "delta_baseline": sum(len(records_by_format[fmt]) for fmt in FORMATS),
            "delta_steered": sum(len(records_by_format[fmt]) for fmt in FORMATS) * int(loaded.n_layers) * (len(cfg["steering"]["k_grid"]) + 1),
            "generation_baseline": len(translation_records),
            "generation_steered": len(translation_records) * int(loaded.n_layers) * (len(cfg["steering"]["k_grid"]) + 1),
        }
        expected_ok = (
            family_summary["delta_rows"] == unit_expected["delta_baseline"] + unit_expected["delta_steered"]
            and family_summary["generation_rows"] == unit_expected["generation_baseline"] + unit_expected["generation_steered"]
        )
        k0_ok = (
            family_summary["k0_exact"]["logit_matches"] == family_summary["k0_exact"]["logit_total"]
            and family_summary["k0_exact"]["generation_matches"] == family_summary["k0_exact"]["generation_total"]
        )
        family_summary["gates"] = {
            "k0_hooked_logits_and_generation_exact": {"pass": k0_ok, **family_summary["k0_exact"]},
            "random_vector_norm_matches_u": {"pass": random_ok, "checks": len(random_checks)},
            "expected_rows": {"pass": expected_ok, **unit_expected, "observed_delta": family_summary["delta_rows"], "observed_generation": family_summary["generation_rows"]},
            "finite_values": {"pass": finite_ok},
            "batched_unbatched_equality": family_summary["batched_unbatched"],
        }
        if not all(
            family_summary["gates"][name]["pass"]
            for name in ("k0_hooked_logits_and_generation_exact", "random_vector_norm_matches_u", "expected_rows", "finite_values")
        ):
            raise AssertionError(f"OPS-27 trial gate failed for {family}: {family_summary['gates']}")

    elapsed = time.perf_counter() - start
    family_summary["wall_seconds"] = elapsed
    family_summary["generation_seconds"] = generation_seconds
    family_summary["delta_seconds"] = delta_seconds
    family_summary["generations_per_second"] = family_summary["generation_rows"] / generation_seconds if generation_seconds else 0.0
    family_summary["delta_sequences_per_second"] = (
        2 * family_summary["delta_rows"] / delta_seconds if delta_seconds else 0.0
    )
    family_summary["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return family_summary


def _run_job(
    cfg: dict[str, Any], source: str, records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    concepts: Mapping[str, Mapping[str, Any]], logger: logging.Logger, run_git: RunGit,
) -> int:
    """Run families serially, stop on first failed trial gate, and log a final summary."""
    start = time.perf_counter()
    summary: dict[str, Any] = {
        "utc_start": _utc_now(), "source": source,
        "git_commit": run_git.git_commit, "git_dirty": run_git.git_dirty,
        "gpu_name": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "families": {}, "errors": [],
    }
    exit_code = 0
    for family in FAMILIES:
        try:
            torch.cuda.reset_peak_memory_stats()
            result = _run_family(cfg, source, family, records[family], concepts, logger, run_git)
            summary["families"][family] = result
            logger.info("E8 family summary %s", json.dumps(result, sort_keys=True, default=str))
        except Exception as exc:
            exit_code = 2
            summary["errors"].append({"family": family, "error": f"{type(exc).__name__}: {exc}"})
            logger.error("E8 family failed: family=%s error=%s: %s", family, type(exc).__name__, exc)
            logger.error("%s", traceback.format_exc())
            break
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary["utc_end"] = _utc_now()
    summary["wall_seconds"] = time.perf_counter() - start
    summary["exit_code"] = exit_code
    if source == "trial" and exit_code == 0:
        primary_factor = 300 / max(min(len(records[family]["translation"]) for family in FAMILIES), 1)
        projected = sum(float(value["wall_seconds"]) * primary_factor for value in summary["families"].values())
        summary["projected_primary_wall_seconds"] = projected
        summary["projected_primary_gpu_hours"] = projected / 3600.0
    summary_path = REPO / (cfg["steering"]["trial_results_dir"] if source == "trial" else cfg["steering"]["results_dir"]) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    logger.info("E8 final summary %s", json.dumps(summary, sort_keys=True, default=str))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse flags, preflight all inputs, then run E8 unless preflight-only was requested."""
    parser = argparse.ArgumentParser(description="Run the E8 steering pass")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("trial", "primary"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    from vilens.config import load_config

    cfg = load_config(args.config)
    logger = logging.getLogger("job_ps_steer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    from vilens.cli import setup_logging

    logger = setup_logging("job_ps_steer", cfg["paths"]["logs"])
    try:
        records, concepts, selected_counts = _prepare_records(cfg, args.source)
        run_git = capture_run_git(REPO, cfg)
        if not _preflight(cfg, args.source, records, selected_counts, logger, gpu=not args.preflight_only):
            logger.error("E8 preflight failed; no model weights loaded")
            return 2
        result = steer.run_if_not_preflight(
            args.preflight_only,
            lambda: _run_job(cfg, args.source, records, concepts, logger, run_git),
        )
        if args.preflight_only:
            logger.info("E8 preflight-only complete; no model weights loaded")
            return 0 if result is None else int(result)
        return int(result)
    except Exception as exc:
        logger.error("E8 preflight/run failed before completion: %s: %s", type(exc).__name__, exc)
        logger.error("%s", traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
