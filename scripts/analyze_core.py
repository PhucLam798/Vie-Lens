#!/datastore/luannt/.cache/sb/bin/python
"""Build preregistered E2–E5 analysis tables from synthetic or directions data."""

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard; env_guard.guard(offline=True)

import argparse
import json
import logging
import re
import resource
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens.analysis.bootstrap import (
    bootstrap_means,
    bootstrap_key,
    concept_indices,
    percentile_interval,
    percentile_interval_with_count,
)
from vilens.analysis.c1 import paired_selection_draws
from vilens.analysis.competence import competence_rows, competence_summary
from vilens.analysis.freeze import (
    FROZEN_ANALYSIS_COMMIT,
    AnalysisCommitIdentity,
    assert_frozen_analysis_tree,
)
from vilens.analysis.holm import holm_adjust, window_positive_test
from vilens.analysis.landmarks import concept_layer, landmarks, probability_concept_layer
from vilens.analysis.rise import concept_rise_curves, decision_rule_quantities, mean_rise_curves
from vilens.analysis.scores import SCORE_KEYS, content_token_scores, matched_distractor_deltas
from vilens.config import load_config, study_models
from vilens.cognates import id_list_sha256
from vilens.cli import StageTimer, setup_logging
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.io import unit_paths
from vilens.passes import pl_unit_key, with_pl_shards
from vilens.analysis.la import (
    classification_accuracy,
    d_en_curve,
    fit_layer_directions,
    la_rule_sensitivity_changed,
    la_layer_selections,
    native_validation_cosines,
    normalized_alignment,
    random_cosine_baseline,
)
from vilens.derived import stream

LONG_COLUMNS = [
    "model", "model_revision", "pass", "format", "condition", "fewshot_set",
    "concept_id", "target_lang", "target_variant", "role", "ref_concept_id",
    "tok_idx", "token_id", "is_content", "layer", "metric", "value",
]
PRIMARY_VARIANTS = {"vi": "released", "en": "released", "zh": "nospace", "fr": "released", "id": "released"}
SENSITIVITY_VARIANTS = {("zh", "released"), ("vi", "nodiac"), ("ru", "released")}
TRIAL_FAMILIES = ("gemma", "qwen", "llama")
TRIAL_FORMATS = ("repetition", "translation")
_LOGGER = logging.getLogger("analyze_core")


def refuse_primary_without_checkpoint_h(source: str, confirmed: bool) -> None:
    """Reject a primary-source invocation before any result path is opened."""
    if source == "primary" and not confirmed:
        raise PermissionError(
            "--source primary reads test-item results and requires --confirm-checkpoint-h"
        )


def _synthetic_rows() -> pd.DataFrame:
    """Build a small deterministic five-language long table for smoke execution."""
    rows: list[dict[str, Any]] = []
    slopes = {"vi": 0.42, "en": 0.36, "zh": 0.26, "fr": 0.18, "id": 0.17}
    base = {"vi": -5.0, "en": -5.4, "zh": -4.9, "fr": -5.6, "id": -5.3}
    for concept_index in range(12):
        concept_id = f"synthetic-{concept_index:03d}"
        distractor_id = f"synthetic-d-{concept_index:03d}"
        for language, slope in slopes.items():
            variant = "nospace" if language == "zh" else "released"
            for layer in range(11):
                correct_value = base[language] + slope * layer / 10.0 + concept_index * 0.001
                for role, ref_id, value in (
                    ("correct", concept_id, correct_value),
                    ("distractor", distractor_id, base[language] - 0.03 * layer),
                ):
                    rows.append(
                        {
                            "model": "synthetic",
                            "model_revision": "synthetic",
                            "pass": "pl",
                            "format": "repetition",
                            "condition": "diac",
                            "fewshot_set": 1,
                            "concept_id": concept_id,
                            "target_lang": language,
                            "target_variant": variant,
                            "role": role,
                            "ref_concept_id": ref_id,
                            "tok_idx": 0,
                            "token_id": 0,
                            "is_content": True,
                            "layer": layer,
                            "metric": "lens_logprob",
                            "value": value,
                        }
                    )
    return pd.DataFrame.from_records(rows, columns=LONG_COLUMNS)


def _trial_paths(repo: Path) -> Iterable[Path]:
    """Yield only the P-L trial parquet paths for the frozen directions trial."""
    for family in TRIAL_FAMILIES:
        for fmt in TRIAL_FORMATS:
            yield repo / (
                "results/pl_trial2/condition=diac__family={family}__fewshot_set=none__"
                "format={fmt}__pass=pl__shard=0.parquet".format(family=family, fmt=fmt)
            )


def _primary_prompt_records(
    cfg: Mapping[str, Any], repo: Path
) -> list[dict[str, Any]]:
    """Load and shard the registered Vietnamese primary prompts as P-L does."""
    prompt_paths = prompt_file_paths(dict(cfg))
    records: list[dict[str, Any]] = []
    for stem in cfg["pl"]["primary"]["files"]:
        path = Path(prompt_paths[str(stem)])
        if not path.is_absolute():
            path = repo / path
        records.extend(load_prompt_file(path))
    return with_pl_shards(records, int(cfg["pl"]["shard_prompts"]))


def _planned_primary_units(
    records: Iterable[Mapping[str, Any]],
    family: str,
    results_dir: Path,
) -> list[tuple[str, Path]]:
    """Resolve primary P-L parquet paths from its shared unit-key function."""
    units: dict[tuple[tuple[str, str], ...], tuple[str, Path]] = {}
    for record in records:
        key = pl_unit_key(family, record)
        parquet_path, _sidecar = unit_paths(results_dir, key)
        signature = tuple((name, str(value)) for name, value in sorted(key.items()))
        units[signature] = (str(record["format"]), parquet_path)
    return sorted(units.values(), key=lambda item: item[1].name)


def _primary_paths(
    repo: Path,
    cfg: Mapping[str, Any],
    *,
    family: str | None = None,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> Iterable[Path]:
    """Yield only P-L units named by its primary prompt plan."""
    planned_records = list(records) if records is not None else _primary_prompt_records(cfg, repo)
    families = [family] if family is not None else [str(value) for value in cfg["pd"]["families"]]
    results_dir = repo / str(cfg["pl"]["results_dir"])
    paths = [
        path
        for selected_family in families
        for _fmt, path in _planned_primary_units(planned_records, selected_family, results_dir)
    ]
    yield from sorted(paths)


def _load_input(
    source: str,
    repo: Path,
    cfg: Mapping[str, Any],
    family: str | None = None,
) -> pd.DataFrame:
    """Read the requested analysis source, after the primary access guard."""
    if source == "synthetic":
        return _synthetic_rows()
    paths = list(_trial_paths(repo) if source == "trial" else _primary_paths(repo, cfg, family=family))
    if family is not None:
        paths = [path for path in paths if f"family={family}__" in path.name]
    if not paths:
        raise FileNotFoundError(f"No parquet inputs for source={source}, family={family}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing analysis inputs: " + ", ".join(missing))
    tables = [
        batch.to_pandas()
        for path in paths
        for batch in pq.ParquetFile(path).iter_batches(columns=LONG_COLUMNS, batch_size=65_536)
    ]
    return pd.concat(tables, ignore_index=True)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read structured rows from a JSON-lines metadata sidecar."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def r1_query_index_row_selected(row: Mapping[str, Any], fewshot_set: int) -> bool:
    """Identify a P-L Vietnamese prompt row for one R1 set without a lang field."""
    return (
        int(row.get("fewshot_set", -1)) == int(fewshot_set)
        and str(row.get("split", "")) == "test"
    )


def _source_residual_paths(
    source: str,
    repo: Path,
    *,
    family: str,
    fmt: str,
    config: Mapping[str, Any],
    primary_records: Iterable[Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Resolve residual-unit parquet paths for one guarded source/model/format."""
    if source == "trial":
        candidates = list(_trial_paths(repo))
    elif source == "primary":
        candidates = list(
            _primary_paths(repo, config, family=family, records=primary_records)
        )
    elif source == "r1":
        candidates = list((repo / str(config["pl"]["r1"]["results_dir"])).glob("*.parquet"))
    else:
        return []
    return [
        path for path in candidates
        if f"family={family}__" in path.name and f"format={fmt}__" in path.name
    ]


def _check_primary_pl_inputs(
    records: Iterable[Mapping[str, Any]],
    family: str,
    results_dir: Path,
    *,
    main_ids: set[str],
    expected_unit_counts: Mapping[str, int] | None = None,
    expected_query_counts: Mapping[str, int] | None = None,
) -> list[dict[str, str]]:
    """Check planned P-L units and Vietnamese query-index rows without loading values."""
    planned_records = list(records)
    planned = _planned_primary_units(planned_records, family, results_dir)
    formats = ("repetition", "translation", "cloze")
    unit_counts = {fmt: sum(unit_fmt == fmt for unit_fmt, _path in planned) for fmt in formats}
    expected_queries = {
        fmt: {
            str(record["concept_id"])
            for record in planned_records
            if str(record.get("format")) == fmt
            and str(record["concept_id"]) in main_ids
            and not (
                record.get("m1_extension") is True
                or str(record.get("m1_extension", "false")).lower() == "true"
            )
        }
        for fmt in formats
    }
    rows: list[dict[str, str]] = []
    by_format: dict[str, list[Path]] = {fmt: [] for fmt in formats}
    for fmt, path in planned:
        by_format.setdefault(fmt, []).append(path)

    for fmt in formats:
        unit_paths_for_format = by_format.get(fmt, [])
        missing: list[str] = []
        index_rows: list[dict[str, Any]] = []
        for parquet_path in unit_paths_for_format:
            sidecar_path = parquet_path.with_name(parquet_path.stem + ".provenance.json")
            index_path = parquet_path.with_suffix(".resid_index.jsonl")
            required = (
                parquet_path,
                sidecar_path,
                parquet_path.with_suffix(".resid.npy"),
                index_path,
                parquet_path.with_suffix(".generations.jsonl"),
            )
            missing.extend(str(path) for path in required if not path.is_file())
            if index_path.is_file():
                try:
                    index_rows.extend(_read_jsonl_rows(index_path))
                except (OSError, json.JSONDecodeError) as exc:
                    missing.append(f"{index_path} (unreadable: {type(exc).__name__})")

        planned_count = unit_counts[fmt]
        expected_units = (
            int(expected_unit_counts[fmt])
            if expected_unit_counts is not None and fmt in expected_unit_counts
            else planned_count
        )
        unit_ok = planned_count == expected_units and not missing
        detail = f"planned_units={planned_count}; missing={missing or 'none'}"
        rows.append(
            {
                "check": "P-L units",
                "scope": f"{family}/{fmt}",
                "observed": detail,
                "expected": f"units={expected_units}; parquet, sidecar, residual, index, generations present",
                "status": "MATCH" if unit_ok else "MISMATCH",
            }
        )

        observed_ids: list[str] = []
        malformed = 0
        for row in index_rows:
            if str(row.get("format")) != fmt:
                malformed += 1
                continue
            extension_value = row.get("m1_extension", False)
            is_extension = extension_value is True or str(extension_value).lower() == "true"
            if str(row.get("split")) == "test" and not is_extension:
                observed_ids.append(str(row.get("concept_id", "")))
        unique_observed = set(observed_ids)
        expected_ids = expected_queries[fmt]
        missing_ids = expected_ids - unique_observed
        extra_ids = unique_observed - expected_ids
        count_expected = (
            int(expected_query_counts[fmt])
            if expected_query_counts is not None and fmt in expected_query_counts
            else len(expected_ids)
        )
        query_ok = (
            len(observed_ids) == count_expected
            and unique_observed == expected_ids
            and len(observed_ids) == len(unique_observed)
            and malformed == 0
            and not missing
        )
        query_detail = (
            f"main_vi_query_rows={len(observed_ids)}; missing_ids={len(missing_ids)}; "
            f"extra_ids={len(extra_ids)}; duplicate_ids={len(observed_ids) - len(unique_observed)}; "
            f"bad_format_rows={malformed}; index_lang_field=ignored (primary prompts are Vietnamese)"
        )
        rows.append(
            {
                "check": "LA query index",
                "scope": f"{family}/{fmt}",
                "observed": query_detail,
                "expected": f"main Vietnamese prompt rows={count_expected}; exact concept-id set",
                "status": "MATCH" if query_ok else "MISMATCH",
            }
        )
    return rows


def _compute_la_source(
    source: str,
    *,
    config: Mapping[str, Any],
    repo: Path,
    main_ids: set[str] | None,
    family_filter: str | None = None,
    fewshot_set: int | None = None,
    direction_source: str = "matched",
    pooling: str = "last",
    bootstrap_key_template: str | None = None,
    formats_override: Sequence[str] | None = None,
    query_scope_ids: set[str] | None = None,
    expected_query_counts: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit A6 directions and score query activations in bounded chunks."""
    if source not in {"trial", "primary", "r1"}:
        return (
            pd.DataFrame(columns=LONG_COLUMNS),
            pd.DataFrame(columns=[*LONG_COLUMNS, "ci_low", "ci_high", "n_concepts", "valid_draws"]),
        )
    if source == "primary" and main_ids is None:
        raise ValueError("Primary LA requires the frozen main-test concept set")
    if source == "r1" and fewshot_set not in {int(value) for value in config["pl"]["r1"]["fewshot_sets"]}:
        raise ValueError("R1 LA requires a configured few-shot set")
    if direction_source not in {"matched", "flores"}:
        raise ValueError("direction_source must be matched or flores")
    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be last or mean")
    from vilens.analysis.la import alignment_layer_status, random_cosine_values, random_unit_directions

    languages = ("en", "zh", "fr", "id")
    class_order = ("vi", *languages)
    formats = (
        ("repetition", "translation", "cloze")
        if source == "primary"
        else ("repetition", "translation")
    )
    if formats_override is not None:
        requested_formats = tuple(map(str, formats_override))
        if not requested_formats or any(fmt not in formats for fmt in requested_formats):
            raise ValueError("formats_override must be a non-empty subset of the source formats")
        formats = requested_formats
    primary_records = _primary_prompt_records(config, repo) if source == "primary" else None
    primary_query_ids: dict[str, set[str]] = {}
    if primary_records is not None:
        for fmt in formats:
            primary_query_ids[fmt] = {
                str(record["concept_id"])
                for record in primary_records
                if str(record.get("format")) == fmt
                and str(record["concept_id"]) in (main_ids or set())
                and (query_scope_ids is None or str(record["concept_id"]) in query_scope_ids)
                and not (
                    record.get("m1_extension") is True
                    or str(record.get("m1_extension", "false")).lower() == "true"
                )
            }
    model_specs = study_models(dict(config))
    families = [str(value) for value in config["pd"]["families"]]
    if family_filter is not None:
        families = [family for family in families if family == family_filter]
        if not families:
            raise ValueError(f"Unknown analysis family {family_filter!r}")
    analysis_cfg = config["analysis"]
    key_template = bootstrap_key_template or str(analysis_cfg["bootstrap_key_template"])
    random_template = str(analysis_cfg["la_random_key_template"])
    seed = int(config["seed"])
    chunk_size = 128
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for family in families:
        model_spec = model_specs[family]
        model_id = str(model_spec["hf_id"])
        model_revision = str(model_spec["revision"])
        for fmt in formats:
            direction_format = "repetition" if fmt == "cloze" else fmt
            pd_root = repo / str(config["pd"]["results_dir"])
            pd_parquet = pd_root / (
                f"family={family}__format={direction_format}__pass=pd__source=matched.parquet"
            )
            pd_index = pq.read_table(pd_parquet, columns=["row", "lang", "split"]).to_pandas()
            expected_pd_rows = int(config["pd"]["expected_matched_records"]) // 2
            if len(pd_index) != expected_pd_rows or not np.array_equal(
                pd_index["row"].to_numpy(dtype=np.int64), np.arange(expected_pd_rows)
            ):
                raise AssertionError(f"P-D matched row coordinates are invalid for {family}/{direction_format}")
            pd_resid = np.load(
                pd_parquet.with_suffix(".resid.npy"), mmap_mode="r", allow_pickle=False
            )
            if pd_resid.ndim != 3 or pd_resid.shape[0] != len(pd_index) or pd_resid.dtype != np.float32:
                raise AssertionError(f"P-D residual shape/dtype differs from its index for {family}/{direction_format}")
            fit_mask = pd_index["split"].astype(str).eq("fit").to_numpy()
            valid_mask = pd_index["split"].astype(str).eq("validate").to_numpy()
            fit_rows = pd_index.loc[fit_mask, "row"].to_numpy(dtype=np.int64)
            valid_rows = pd_index.loc[valid_mask, "row"].to_numpy(dtype=np.int64)
            fit_labels = pd_index.loc[fit_mask, "lang"].astype(str).to_numpy()
            valid_labels = pd_index.loc[valid_mask, "lang"].astype(str).to_numpy()
            if len(fit_rows) != 530 or len(valid_rows) != 130:
                raise AssertionError(f"P-D fit/validate counts differ from 530/130 for {family}/{direction_format}")
            fit_counts = pd.Series(fit_labels).value_counts().to_dict()
            valid_counts = pd.Series(valid_labels).value_counts().to_dict()
            if any(int(fit_counts.get(language, 0)) != 106 for language in class_order):
                raise AssertionError(f"P-D fit labels are not 106 per language: {fit_counts}")
            if any(int(valid_counts.get(language, 0)) != 26 for language in class_order):
                raise AssertionError(f"P-D validate labels are not 26 per language: {valid_counts}")

            flores_index: pd.DataFrame | None = None
            flores_resid: np.ndarray | None = None
            flores_fit_rows = np.array([], dtype=np.int64)
            flores_fit_labels = np.array([], dtype=object)
            primary_gate_map: dict[int, bool] = {}
            if direction_source == "flores":
                flores_parquet = pd_root / f"family={family}__pass=pd__source=flores.parquet"
                flores_index = pq.read_table(
                    flores_parquet, columns=["row", "lang", "split"]
                ).to_pandas()
                expected_flores = int(config["pd"]["expected_flores_records"])
                if len(flores_index) != expected_flores or not np.array_equal(
                    flores_index["row"].to_numpy(dtype=np.int64), np.arange(expected_flores)
                ):
                    raise AssertionError(f"FLORES row coordinates are invalid for {family}")
                flores_array_path = pd_root / f"family={family}__pass=pd__source=flores.{pooling}.npy"
                flores_resid = np.load(flores_array_path, mmap_mode="r", allow_pickle=False)
                if (
                    flores_resid.ndim != 3
                    or flores_resid.shape[0] != len(flores_index)
                    or flores_resid.shape[1:] != pd_resid.shape[1:]
                    or flores_resid.dtype != np.float32
                ):
                    raise AssertionError(f"FLORES {pooling} shape/dtype differs from matched P-D for {family}")
                flores_fit_mask = flores_index["split"].astype(str).eq("fit").to_numpy()
                flores_fit_rows = flores_index.loc[flores_fit_mask, "row"].to_numpy(dtype=np.int64)
                flores_fit_labels = flores_index.loc[flores_fit_mask, "lang"].astype(str).to_numpy()
                if len(flores_fit_rows) != 3990 or any(
                    int(pd.Series(flores_fit_labels).value_counts().get(language, 0)) != 798
                    for language in class_order
                ):
                    raise AssertionError(f"FLORES fit rows are not 798 per language for {family}")
                gate_table = pd.read_csv(repo / "reports/E5_gate.csv")
                gate_rows = gate_table.loc[
                    gate_table["family"].astype(str).eq(family)
                    & gate_table["fit_format"].astype(str).eq(fmt)
                ]
                primary_gate_map = {
                    int(row.layer): bool(row.gate_pass) for row in gate_rows.itertuples(index=False)
                    if 1 <= int(row.layer) <= int(pd_resid.shape[1] - 1)
                }
                if set(primary_gate_map) != set(range(1, int(pd_resid.shape[1]))):
                    raise AssertionError(f"Primary A6 gate does not cover 1..N for {family}/{fmt}")

            query_refs: list[dict[str, Any]] = []
            for parquet_path in _source_residual_paths(
                source,
                repo,
                family=family,
                fmt=fmt,
                config=config,
                primary_records=primary_records,
            ):
                index_path = parquet_path.with_suffix(".resid_index.jsonl")
                resid_path = parquet_path.with_suffix(".resid.npy")
                index_rows = _read_jsonl_rows(index_path)
                query_resid = np.load(resid_path, mmap_mode="r", allow_pickle=False)
                if (
                    query_resid.ndim != 3
                    or query_resid.shape[0] != len(index_rows)
                    or query_resid.shape[1] != pd_resid.shape[1]
                    or query_resid.dtype != np.float32
                ):
                    raise AssertionError(f"P-L residual shape/dtype differs from P-D for {parquet_path.name}")
                for row in index_rows:
                    if str(row.get("format")) != fmt:
                        continue
                    concept_id = str(row["concept_id"])
                    extension_value = row.get("m1_extension", False)
                    is_extension = extension_value is True or str(extension_value).lower() == "true"
                    if is_extension:
                        continue
                    if source == "primary":
                        if (
                            concept_id not in primary_query_ids.get(fmt, set())
                            or str(row.get("split")) != "test"
                        ):
                            continue
                    elif source == "r1":
                        if not r1_query_index_row_selected(row, int(fewshot_set)):
                            continue
                    else:
                        if str(row.get("lang")) != "vi":
                            continue
                        if str(row.get("split")) not in {"fit", "validate"}:
                            continue
                    query_refs.append(
                        {
                            "concept_id": concept_id,
                            "array_row": int(row["array_row"]),
                            "array": query_resid,
                            "condition": str(row.get("condition", "diac")),
                            "fewshot_set": row.get("fewshot_set"),
                        }
                    )
            query_refs.sort(key=lambda row: row["concept_id"])
            query_ids = [str(row["concept_id"]) for row in query_refs]
            if len(query_ids) != len(set(query_ids)):
                raise AssertionError(f"Duplicate Vietnamese query residuals for {family}/{fmt}")
            if not query_refs:
                raise FileNotFoundError(f"No Vietnamese P-L residual queries for {source}/{family}/{fmt}")
            if source == "primary":
                default_counts = {"repetition": 1535, "translation": 1524, "cloze": 34}
                expected_queries = int((expected_query_counts or default_counts)[fmt])
                observed_ids = set(query_ids)
                expected_ids = primary_query_ids.get(fmt, set())
                if len(query_refs) != expected_queries or observed_ids != expected_ids:
                    raise AssertionError(
                        f"Primary {family}/{fmt} Vietnamese prompt query mismatch: "
                        f"rows={len(query_refs)}/{expected_queries}, "
                        f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}"
                    )
            if source == "r1":
                expected_queries = int(config["pl"]["r1"]["expected"][fmt])
                if len(query_refs) != expected_queries:
                    raise AssertionError(
                        f"R1 {family}/{fmt}/set{fewshot_set} residual queries="
                        f"{len(query_refs)}, expected {expected_queries}"
                    )

            n_layers = int(pd_resid.shape[1] - 1)
            hidden_width = int(pd_resid.shape[2])
            all_layer_values = np.full((len(query_refs), len(languages), n_layers), np.nan, dtype=np.float64)
            passing_values = np.full_like(all_layer_values, np.nan)
            passing_layers: list[int] = []
            defined_layers: list[int] = []
            layer_random: dict[int, float] = {}
            layer_gate: dict[int, tuple[int, int, bool, bool, bool]] = {}
            for layer in range(1, n_layers + 1):
                if direction_source == "flores":
                    assert flores_resid is not None
                    fit_acts = np.asarray(flores_resid[flores_fit_rows, layer, :], dtype=np.float32)
                    fit_labels_for_layer = flores_fit_labels
                else:
                    fit_acts = np.asarray(pd_resid[fit_rows, layer, :], dtype=np.float32)
                    fit_labels_for_layer = fit_labels
                valid_acts = np.asarray(pd_resid[valid_rows, layer, :], dtype=np.float32)
                if not np.isfinite(fit_acts).all() or not np.isfinite(valid_acts).all():
                    raise AssertionError(f"Non-finite P-D activation in {family}/{direction_format}/layer {layer}")
                center, directions, _means = fit_layer_directions(
                    fit_acts, fit_labels_for_layer, class_order=class_order
                )
                gate_accuracy, gate_correct, gate_n = classification_accuracy(
                    valid_acts, valid_labels, center, directions
                )
                native = native_validation_cosines(valid_acts, valid_labels, center, directions)
                random_key = random_template.format(family=family, format=fmt, layer=layer)
                random_units = random_unit_directions(
                    hidden_width,
                    rng=stream(seed, random_key),
                    n_directions=1_000,
                )
                random_sum = 0.0
                for start in range(0, len(query_refs), chunk_size):
                    stop = min(start + chunk_size, len(query_refs))
                    query_chunk = np.stack(
                        [
                            np.asarray(ref["array"][ref["array_row"], layer, :], dtype=np.float32)
                            for ref in query_refs[start:stop]
                        ]
                    )
                    if not np.isfinite(query_chunk).all():
                        raise AssertionError(f"Non-finite P-L query activation in {source}/{family}/{fmt}/layer {layer}")
                    random_sum += float(random_cosine_values(query_chunk - center, random_units).sum())
                c_random = random_sum / len(query_refs)
                layer_random[layer] = c_random
                gate_pass = (
                    primary_gate_map[layer]
                    if direction_source == "flores"
                    else gate_correct >= int(analysis_cfg["gate_min_correct"])
                )
                layer_pass, undefined = alignment_layer_status(gate_pass, native, c_random)
                layer_gate[layer] = (gate_correct, gate_n, gate_pass, undefined, layer_pass)
                if not undefined:
                    defined_layers.append(layer)
                layer_rows_common = {
                    "model": model_id,
                    "model_revision": model_revision,
                    "pass": "analysis",
                    "format": fmt,
                    "condition": "diac",
                    "fewshot_set": (
                        1 if source == "primary" and fmt != "cloze"
                        else fewshot_set if source == "r1"
                        else None
                    ),
                    "target_variant": None,
                    "ref_concept_id": None,
                    "tok_idx": None,
                    "token_id": None,
                    "is_content": None,
                    "layer": layer,
                    "role": "path",
                }
                for metric, value in (
                    ("gate_accuracy", gate_accuracy),
                    ("gate_pass", float(gate_pass)),
                    ("undefined_denominator", float(undefined)),
                    ("layer_pass", float(layer_pass)),
                    ("c_random", c_random),
                ):
                    long_rows.append(
                        {
                            **layer_rows_common,
                            "concept_id": None,
                            "target_lang": None,
                            "metric": metric,
                            "value": value,
                        }
                    )
                for language, native_cosine in native.items():
                    long_rows.append(
                        {
                            **layer_rows_common,
                            "concept_id": None,
                            "target_lang": language,
                            "metric": "c_nat",
                            "value": native_cosine,
                        }
                    )
                if undefined:
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "undefined_denominator_layer",
                            "target_lang": None,
                            "layer": layer,
                            "value": 1.0,
                        }
                    )
                if not gate_pass:
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "gate_failed_layer",
                            "target_lang": None,
                            "layer": layer,
                            "value": 1.0,
                        }
                    )
                if not layer_pass:
                    continue
                passing_layers.append(layer)

                for start in range(0, len(query_refs), chunk_size):
                    stop = min(start + chunk_size, len(query_refs))
                    query_chunk = np.stack(
                        [
                            np.asarray(ref["array"][ref["array_row"], layer, :], dtype=np.float32)
                            for ref in query_refs[start:stop]
                        ]
                    )
                    alignments = normalized_alignment(
                        query_chunk,
                        center,
                        directions,
                        native_cosines=native,
                        c_random=c_random,
                    )
                    for lang_index, language in enumerate(languages):
                        values = np.asarray(alignments[language], dtype=np.float64)
                        if not np.isfinite(values).all():
                            raise AssertionError(
                                f"Non-finite LA for a passing layer {family}/{fmt}/{layer}/{language}"
                            )
                        all_layer_values[start:stop, lang_index, layer - 1] = values
                        passing_values[start:stop, lang_index, layer - 1] = values
                        for query_offset, value in enumerate(values):
                            ref = query_refs[start + query_offset]
                            long_rows.append(
                                {
                                    **layer_rows_common,
                                    "concept_id": ref["concept_id"],
                                    "target_lang": language,
                                    "metric": "LA",
                                    "value": float(value),
                                }
                            )
                    den_values = d_en_curve(alignments)
                    for query_offset, value in enumerate(den_values):
                        ref = query_refs[start + query_offset]
                        long_rows.append(
                            {
                                **layer_rows_common,
                                "concept_id": ref["concept_id"],
                                "target_lang": None,
                                "metric": "D_en",
                                "value": float(value),
                            }
                        )
            for layer in defined_layers:
                if layer in passing_layers:
                    continue
                # Defined but gate-failed layers are used only by the all-layer sensitivity.
                center, directions, _means = fit_layer_directions(
                    np.asarray(
                        flores_resid[flores_fit_rows, layer, :]
                        if direction_source == "flores" and flores_resid is not None
                        else pd_resid[fit_rows, layer, :],
                        dtype=np.float32,
                    ),
                    flores_fit_labels if direction_source == "flores" else fit_labels,
                    class_order=class_order,
                )
                valid_acts = np.asarray(pd_resid[valid_rows, layer, :], dtype=np.float32)
                native = native_validation_cosines(valid_acts, valid_labels, center, directions)
                c_random = layer_random[layer]
                for start in range(0, len(query_refs), chunk_size):
                    stop = min(start + chunk_size, len(query_refs))
                    query_chunk = np.stack(
                        [
                            np.asarray(ref["array"][ref["array_row"], layer, :], dtype=np.float32)
                            for ref in query_refs[start:stop]
                        ]
                    )
                    alignments = normalized_alignment(
                        query_chunk,
                        center,
                        directions,
                        native_cosines=native,
                        c_random=c_random,
                    )
                    for lang_index, language in enumerate(languages):
                        all_layer_values[start:stop, lang_index, layer - 1] = alignments[language]

            if source == "trial":
                for lang_index, language in enumerate(languages):
                    for layer in range(1, n_layers + 1):
                        values = passing_values[:, lang_index, layer - 1]
                        summary_rows.append(
                            {
                                "source": source,
                                "model": model_id,
                                "format": fmt,
                                "metric": "mean_LA",
                                "target_lang": language,
                                "layer": layer,
                                "value": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                                "n_concepts": len(query_refs),
                                "gate_correct": layer_gate[layer][0],
                                "gate_pass": layer_gate[layer][2],
                                "undefined_denominator": layer_gate[layer][3],
                                "c_random": layer_random[layer],
                            }
                        )
                continue

            indices = concept_indices(
                query_ids,
                resamples=int(analysis_cfg["bootstrap_resamples"]),
                seed=seed,
                key=bootstrap_key(
                    "main" if source != "r1" else f"r1_set{fewshot_set}",
                    model_id,
                    fmt,
                    template=key_template,
                ),
            )
            gate_flags = [layer_gate[layer][2] for layer in range(1, n_layers + 1)]
            undefined_flags = [layer_gate[layer][3] for layer in range(1, n_layers + 1)]
            passing_mask, sensitivity_mask = la_layer_selections(gate_flags, undefined_flags)
            passing_positions = np.flatnonzero(passing_mask)
            defined_positions = np.flatnonzero(sensitivity_mask)
            boot_pass = (
                bootstrap_means(passing_values[:, :, passing_positions], indices)
                if passing_positions.size
                else np.empty((len(indices), len(languages), 0), dtype=np.float64)
            )
            boot_all = (
                bootstrap_means(all_layer_values[:, :, defined_positions], indices)
                if defined_positions.size
                else np.empty((len(indices), len(languages), 0), dtype=np.float64)
            )
            for lang_index, language in enumerate(languages):
                threshold = 0.50 if language == "en" else 0.20
                if passing_positions.size:
                    pass_layer_means = np.nanmean(passing_values[:, lang_index, passing_positions], axis=0)
                    pass_position = int(np.argmax(pass_layer_means))
                    point_max = float(pass_layer_means[pass_position])
                    pass_samples = np.nanmax(boot_pass[:, lang_index, :], axis=1)
                    low, high = percentile_interval(pass_samples)
                    rule_at_point = point_max >= threshold if language == "en" else point_max < threshold
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "max_LA",
                            "target_lang": language,
                            "layer": int(passing_layers[pass_position]),
                            "value": point_max,
                            "ci_low": low,
                            "ci_high": high,
                            "valid_draws": int(np.isfinite(pass_samples).sum()),
                            "straddles_threshold": bool(low <= threshold <= high),
                            "threshold": threshold,
                            "n_concepts": len(query_refs),
                            "passing_layers": len(passing_layers),
                            "la_evaluable": True,
                        }
                    )
                else:
                    point_max = np.nan
                    rule_at_point = None
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "max_LA",
                            "target_lang": language,
                            "layer": None,
                            "value": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "valid_draws": 0,
                            "straddles_threshold": None,
                            "threshold": threshold,
                            "n_concepts": len(query_refs),
                            "passing_layers": 0,
                            "la_evaluable": False,
                        }
                    )
                if defined_positions.size:
                    all_layer_means = np.nanmean(all_layer_values[:, lang_index, defined_positions], axis=0)
                    all_position = int(np.argmax(all_layer_means))
                    all_max = float(all_layer_means[all_position])
                    all_samples = np.nanmax(boot_all[:, lang_index, :], axis=1)
                    all_low, all_high = percentile_interval(all_samples)
                    change = la_rule_sensitivity_changed(point_max, all_max, language)
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "max_LA_all_layer_sensitivity",
                            "target_lang": language,
                            "layer": int(defined_layers[all_position]),
                            "value": all_max,
                            "ci_low": all_low,
                            "ci_high": all_high,
                            "valid_draws": int(np.isfinite(all_samples).sum()),
                            "threshold": threshold,
                            "would_change_rule_outcome": change,
                            "n_concepts": len(query_refs),
                            "passing_layers": len(passing_layers),
                            "defined_layers": len(defined_layers),
                        }
                    )
                if fmt == "cloze":
                    for position, layer in enumerate(passing_layers):
                        samples = boot_pass[:, lang_index, position]
                        low, high = percentile_interval(samples)
                        summary_rows.append(
                            {
                                "source": source,
                                "model": model_id,
                                "format": fmt,
                                "metric": "mean_LA",
                                "target_lang": language,
                                "layer": layer,
                                "value": float(np.nanmean(passing_values[:, lang_index, layer - 1])),
                                "ci_low": low,
                                "ci_high": high,
                                "valid_draws": int(np.isfinite(samples).sum()),
                                "n_concepts": len(query_refs),
                                "passing_layers": len(passing_layers),
                            }
                        )

            if passing_positions.size:
                den_pass = d_en_curve(
                    {language: passing_values[:, index, :] for index, language in enumerate(languages)}
                )
                den_boot = boot_pass[:, 0, :] - (boot_pass[:, 2, :] + boot_pass[:, 3, :]) / 2.0
                for position, layer in enumerate(passing_layers):
                    low, high = percentile_interval(den_boot[:, position])
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "D_en",
                            "target_lang": None,
                            "layer": layer,
                            "value": float(den_pass[:, layer - 1].mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "valid_draws": int(np.isfinite(den_boot[:, position]).sum()),
                            "n_concepts": len(query_refs),
                            "passing_layers": len(passing_layers),
                        }
                    )
                window_positions = [
                    i for i, layer in enumerate(passing_layers)
                    if float(analysis_cfg["window"][0]) <= layer / n_layers <= float(analysis_cfg["window"][1])
                ]
                if window_positions:
                    scalar_samples = den_boot[:, window_positions].mean(axis=1)
                    low, high = percentile_interval(scalar_samples)
                    selected_layers = [passing_layers[i] for i in window_positions]
                    summary_rows.append(
                        {
                            "source": source,
                            "model": model_id,
                            "format": fmt,
                            "metric": "D_en_window_mean",
                            "target_lang": None,
                            "layer": None,
                            "value": float(den_pass[:, np.asarray(selected_layers) - 1].mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "valid_draws": int(np.isfinite(scalar_samples).sum()),
                            "n_concepts": len(query_refs),
                            "passing_layers": len(passing_layers),
                        }
                    )
    summary_frame = pd.DataFrame.from_records(summary_rows)
    if not summary_frame.empty:
        revision_by_id = {str(spec["hf_id"]): str(spec["revision"]) for spec in model_specs.values()}
        defaults: dict[str, Any] = {
            "model_revision": summary_frame["model"].astype(str).map(revision_by_id),
            "pass": "analysis",
            "condition": "diac",
            "fewshot_set": None,
            "concept_id": None,
            "target_variant": None,
            "ref_concept_id": None,
            "tok_idx": None,
            "token_id": None,
            "is_content": None,
            "role": "path",
        }
        for column, value in defaults.items():
            if column not in summary_frame:
                summary_frame[column] = value
        if source == "primary":
            summary_frame["fewshot_set"] = summary_frame["format"].astype(str).map(
                lambda fmt: 1 if fmt != "cloze" else None
            )
        summary_frame["metric"] = summary_frame["metric"].astype(str)
    else:
        summary_frame = pd.DataFrame(columns=[*LONG_COLUMNS, "ci_low", "ci_high", "n_concepts", "valid_draws"])
    return (
        pd.DataFrame.from_records(long_rows, columns=LONG_COLUMNS),
        summary_frame,
    )


def _load_cognate_free_ids(repo: Path) -> list[str]:
    """Read and hash the frozen cognate-free IDs before applying A5."""
    path = repo / "derived/cognate_flags.jsonl"
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row.get("cognate_free", "false")).lower() == "true":
                ids.append(str(row["concept_id"]))
    digest = id_list_sha256(ids)
    expected = "f1e7a6b5a0c197b1769215224503ee33a8f2e216943ae0d8088d55219042a58f"
    if digest != expected or len(ids) != 729:
        raise AssertionError(f"A5 cognate-free ids count/hash={len(ids)}/{digest}, expected 729/{expected}")
    return sorted(ids)


def _primary_scope_ids(repo: Path, config: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return main and cognate-free ids for post-confirmation primary analysis."""
    concept_path = repo / str(config["paths"]["concepts_jsonl"])
    main_ids: set[str] = set()
    with concept_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            concept = json.loads(line)
            concept_id = str(concept["concept_id"])
            is_main_test = str(concept.get("split", "")) == "test"
            extension = str(concept.get("m1_extension", "false")).lower() == "true"
            if is_main_test and not extension:
                main_ids.add(concept_id)
    if len(main_ids) != 1_535:
        raise AssertionError(f"Main test set has {len(main_ids)} concepts; expected 1,535")
    return main_ids, set(_load_cognate_free_ids(repo))


def _load_primary_concepts(repo: Path, main_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load only main-test forms required by the guarded primary A10 scorer."""
    path = repo / "data/external/viLens-concepts/concepts.jsonl"
    concepts: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            concept_id = str(row["concept_id"])
            if concept_id in main_ids:
                concepts[concept_id] = row
    if set(concepts) != main_ids:
        raise AssertionError("A10 concept forms do not cover the complete main-test set")
    return concepts


def _load_primary_generations(
    repo: Path,
    cfg: Mapping[str, Any],
    main_ids: set[str],
    family: str | None = None,
) -> list[dict[str, Any]]:
    """Read primary generation JSONL only after the Checkpoint-H access guard."""
    generations: list[dict[str, Any]] = []
    paths = list(_primary_paths(repo, cfg, family=family))
    if family is not None:
        paths = [path for path in paths if f"family={family}__" in path.name]
    for parquet_path in paths:
        family_match = re.search(r"(?:^|__)family=([^_]+)(?:__|$)", parquet_path.name)
        if family_match is None:
            raise ValueError(f"Cannot identify model family in {parquet_path.name}")
        family = family_match.group(1)
        generation_path = parquet_path.with_suffix(".generations.jsonl")
        if not generation_path.is_file():
            raise FileNotFoundError(f"Missing generation table for {parquet_path.name}")
        with generation_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                concept_id = str(row["concept_id"])
                if concept_id not in main_ids or str(row.get("m1_extension", "false")).lower() == "true":
                    continue
                generations.append(
                    {
                        "model": family,
                        "format": str(row["format"]),
                        "condition": "diac",
                        "concept_id": concept_id,
                        "answer": str(row.get("text", "")),
                    }
                )
    return generations


def _score_long(scores: pd.DataFrame) -> pd.DataFrame:
    """Convert per-token sequence scores to rule-38 long-format rows."""
    measures = ("s", "s1", "s_plus", "sequence_sum")
    row_indices = np.repeat(np.arange(len(scores), dtype=np.int64), len(measures))
    key_columns = [column for column in SCORE_KEYS if column != "metric"]
    output = scores.iloc[row_indices].loc[:, key_columns].reset_index(drop=True)
    output["tok_idx"] = None
    output["token_id"] = None
    output["is_content"] = None
    output["layer"] = pd.to_numeric(output["layer"], errors="raise").astype("int64")
    output["metric"] = np.tile(np.asarray(measures, dtype=object), len(scores))
    output["value"] = scores.loc[:, measures].to_numpy(dtype=np.float64).reshape(-1)
    return output.loc[:, LONG_COLUMNS]


def _delta_long(deltas: pd.DataFrame, *, metric_column: str, metric_name: str) -> pd.DataFrame:
    """Convert a concept/language contrast curve to rule-38 long format."""
    if deltas.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)
    key_columns = (
        "model", "model_revision", "pass", "format", "condition", "fewshot_set",
        "concept_id", "target_lang", "target_variant",
    )
    output = deltas.loc[:, key_columns].reset_index(drop=True).copy()
    output["role"] = "path"
    ref_column = "distractor_id" if "distractor_id" in deltas.columns else "ref_concept_id"
    output["ref_concept_id"] = deltas[ref_column].reset_index(drop=True)
    output["tok_idx"] = None
    output["token_id"] = None
    output["is_content"] = None
    output["layer"] = pd.to_numeric(deltas["layer"], errors="raise").astype("int64").to_numpy()
    output["metric"] = metric_name
    output["value"] = pd.to_numeric(deltas[metric_column], errors="raise").to_numpy(dtype=np.float64)
    return output.loc[:, LONG_COLUMNS]


def _summary_metric_long(
    frame: pd.DataFrame,
    *,
    source_rows: pd.DataFrame,
    kind: str,
) -> pd.DataFrame:
    """Render aggregate rule/landmark/bootstrap summaries as rule-38 long rows."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *LONG_COLUMNS, "ci_low", "ci_high", "n_concepts", "valid_draws",
                "item_set", "metric_variant", "category_value",
            ]
        )
    revision_by_model = source_rows.groupby("model", sort=True)["model_revision"].first().to_dict()
    output: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        model = row.get("model")
        fmt = row.get("format")
        common = {
            "model": model,
            "model_revision": revision_by_model.get(model),
            "pass": "analysis",
            "format": fmt,
            "condition": "diac",
            "fewshot_set": 1,
            "concept_id": None,
            "target_lang": row.get("target_lang"),
            "target_variant": None,
            "role": "path",
            "ref_concept_id": None,
            "tok_idx": None,
            "token_id": None,
            "is_content": None,
            "layer": None,
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "n_concepts": row.get("n_concepts"),
            "valid_draws": row.get("valid_draws"),
            "n_layers": row.get("n_layers"),
            "item_set": row.get("item_set"),
            "metric_variant": row.get("metric_variant", "s"),
            "category_value": None,
        }
        numeric_metrics: list[tuple[str, Any, Any, Any, Any]] = []
        if kind == "landmark":
            metric = str(row.get("metric"))
            if metric == "pivot_language":
                record = dict(common)
                record.update(
                    metric=metric,
                    value=np.nan,
                    category_value=row.get("value"),
                    target_lang=row.get("value"),
                    ci_low=None,
                    ci_high=None,
                )
                output.append(record)
            else:
                numeric_metrics.append(
                    (
                        metric,
                        row.get("value"),
                        row.get("ci_low"),
                        row.get("ci_high"),
                        row.get("valid_draws"),
                    )
                )
        elif kind == "holm":
            for metric in ("p_value", "p_holm", "reject_0_05"):
                value = row.get(metric)
                if value is not None and pd.notna(value):
                    numeric_metrics.append((metric, float(value), None, None, row.get("valid_draws")))
        elif kind == "rule":
            if row.get("target_lang") != "__H1_H2__":
                if pd.notna(row.get("M_L")):
                    numeric_metrics.append(
                        ("M_L", row["M_L"], row.get("ci_low"), row.get("ci_high"), row.get("valid_draws"))
                    )
            else:
                for metric, value, low, high, valid in (
                    ("h1_q", row.get("h1_q"), row.get("h1_q_ci_low"), row.get("h1_q_ci_high"), row.get("h1_q_valid_resamples")),
                    ("h1_ratio", row.get("h1_ratio"), row.get("h1_ratio_ci_low"), row.get("h1_ratio_ci_high"), row.get("h1_ratio_valid_resamples")),
                    ("control_mean", row.get("control_mean"), row.get("control_mean_ci_low"), row.get("control_mean_ci_high"), row.get("bootstrap_resamples")),
                    ("h2_band_ratio", row.get("h2_band_ratio"), row.get("h2_band_ci_low"), row.get("h2_band_ci_high"), row.get("h2_band_valid_resamples")),
                ):
                    if (value is not None and pd.notna(value)) or metric in {"h1_ratio", "h2_band_ratio"}:
                        numeric_metrics.append((metric, value, low, high, valid))
        for metric, value, low, high, valid_draws in numeric_metrics:
            record = dict(common)
            record["metric"] = metric
            record["value"] = float(value) if value is not None and pd.notna(value) else np.nan
            record["ci_low"] = low
            record["ci_high"] = high
            record["valid_draws"] = valid_draws
            if metric.startswith("L_concept") or metric in {"L_peak", "L_cross"}:
                record["layer"] = int(value) if value is not None and pd.notna(value) else None
            output.append(record)
    return pd.DataFrame.from_records(output)


def _summary_text(
    source: str,
    input_rows: int,
    scores: pd.DataFrame,
    deltas: pd.DataFrame,
    rises: pd.DataFrame,
    score_long: pd.DataFrame,
    delta_long: pd.DataFrame,
    rise_long: pd.DataFrame,
    la_long: pd.DataFrame,
    rules: pd.DataFrame,
    landmarks_table: pd.DataFrame,
    rule_long: pd.DataFrame,
    landmark_long: pd.DataFrame,
    holm_long: pd.DataFrame,
) -> str:
    """Build a concise run-output inventory with no result interpretation."""
    return (
        f"# Core analysis smoke run — {source}\n\n"
        "The run reports table production only; no scientific interpretation is made. "
        "The trial source contains directions-split pilot records.\n\n"
        "| Output | Rows |\n|---|---:|\n"
        f"| Input P-L rows | {input_rows} |\n"
        f"| Per-sequence score groups | {len(scores)} |\n"
        f"| Matched correct/distractor pairs | {len(deltas)} |\n"
        f"| Concept-level rise rows | {len(rises)} |\n"
        f"| `scores.parquet` long rows | {len(score_long)} |\n"
        f"| `deltas.parquet` long rows | {len(delta_long)} |\n"
        f"| `rise.parquet` long rows | {len(rise_long)} |\n"
        f"| `la.parquet` long rows | {len(la_long)} |\n"
        f"| Rule summary rows | {len(rules)} |\n"
        f"| Landmark summary rows | {len(landmarks_table)} |\n"
        f"| `rule_quantities.parquet` long rows | {len(rule_long)} |\n"
        f"| `landmarks.parquet` long rows | {len(landmark_long)} |\n"
        f"| `holm_family_a.parquet` long rows | {len(holm_long)} |\n"
        "\nOutputs are in `results/analysis/" + source + "/` and the companion summary CSVs are under `reports/`.\n"
    )


def _readout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the registered five primary language/variant pairs."""
    language = frame["target_lang"].astype(str)
    variant = frame["target_variant"].astype(str)
    expected_variant = language.map(PRIMARY_VARIANTS)
    return frame.loc[expected_variant.notna() & expected_variant.eq(variant)].copy()


def _sensitivity_readout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the frozen zh-space, vi-nodiac, and translation ru sensitivities."""
    language = frame["target_lang"].astype(str)
    variant = frame["target_variant"].astype(str)
    selected = pd.Series(False, index=frame.index)
    for target_language, target_variant in SENSITIVITY_VARIANTS:
        selected |= language.eq(target_language) & variant.eq(target_variant)
    selected &= ~language.eq("ru") | frame["format"].astype(str).eq("translation")
    return frame.loc[selected].copy()


def _complete_matrix(
    frame: pd.DataFrame,
    *,
    languages: tuple[str, ...],
    value: str,
    allow_missing: bool = False,
) -> tuple[list[str], list[int], np.ndarray]:
    """Build a complete sorted concept×language×layer tensor."""
    # Layer 0 (the embedding output) is intentionally retained in the raw
    # score table, but every E2–E5 curve and bootstrap matrix uses block
    # outputs only. Drop it here so descriptive score bootstraps follow the
    # same layer convention as rise curves.
    block_frame = frame.loc[pd.to_numeric(frame["layer"], errors="raise") >= 1].copy()
    layers = sorted(map(int, block_frame["layer"].unique()))
    if not layers:
        raise ValueError("Analysis matrices require block-output layers only")
    ids_by_language = []
    for language in languages:
        selected = block_frame.loc[block_frame["target_lang"].astype(str).eq(language)]
        ids_by_language.append(set(selected["concept_id"].astype(str).unique()))
    concept_ids = sorted(set.intersection(*ids_by_language)) if ids_by_language else []
    arrays = []
    for language in languages:
        selected = block_frame.loc[block_frame["target_lang"].astype(str).eq(language)]
        pivot = selected.pivot(index="concept_id", columns="layer", values=value)
        arrays.append(pivot.reindex(index=concept_ids, columns=layers).to_numpy(dtype=np.float64))
    values = np.stack(arrays, axis=1) if arrays else np.empty((0, len(languages), len(layers)))
    if not allow_missing and not np.isfinite(values).all():
        raise ValueError("Analysis tensor has missing or non-finite concept/language/layer cells")
    return concept_ids, layers, values


def _landmark_bootstrap_samples(
    score_boot: np.ndarray,
    rise_boot: np.ndarray,
    *,
    layers: list[int],
    n_layers: int,
    thresholds: list[float] | tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Compute each frozen landmark for all bootstrap draws in array form."""
    scores = np.asarray(score_boot, dtype=np.float64)
    rises = np.asarray(rise_boot, dtype=np.float64)
    layer_values = np.asarray(layers, dtype=np.int64)
    if (
        scores.ndim != 3
        or rises.shape != scores.shape
        or scores.shape[1] != 5
        or scores.shape[2] != len(layer_values)
        or not len(layer_values)
        or int(layer_values.max()) != int(n_layers)
    ):
        raise ValueError("bootstrap score/rise arrays must align as draws×five languages×layers")

    output: dict[str, np.ndarray] = {}
    for threshold in thresholds:
        cutoff = scores[:, 0, -1] + float(np.log(float(threshold)))
        reached = (scores >= cutoff[:, None, None]).any(axis=1)
        valid = reached.any(axis=1)
        first = reached.argmax(axis=1)
        values = layer_values[first].astype(np.float64)
        values[~valid] = np.nan
        output[f"L_concept_{int(float(threshold) * 100)}"] = values

    non_vi_maxima = rises[:, 1:, :].max(axis=2)
    pivot_positions = non_vi_maxima.argmax(axis=1) + 1
    draw_positions = np.arange(scores.shape[0], dtype=np.int64)
    pivot_curves = rises[draw_positions, pivot_positions, :]
    peak_positions = pivot_curves.argmax(axis=1)
    peak_layers = layer_values[peak_positions]
    output["L_peak"] = peak_layers.astype(np.float64)

    after_peak = layer_values[None, :] > peak_layers[:, None]
    crossings = after_peak & (rises[:, 0, :] > pivot_curves)
    has_crossing = crossings.any(axis=1)
    crossing_positions = crossings.argmax(axis=1)
    crossing_layers = layer_values[crossing_positions].astype(np.float64)
    crossing_layers[~has_crossing] = np.nan
    output["L_cross"] = crossing_layers

    concept_10 = output.get("L_concept_10")
    if concept_10 is None:
        raise ValueError("landmark thresholds must include 0.10 for pivot width")
    width = (crossing_layers - concept_10) / float(n_layers)
    width[~(has_crossing & np.isfinite(concept_10))] = np.nan
    output["pivot_width"] = width
    output["pivot_magnitude"] = non_vi_maxima[draw_positions, pivot_positions - 1]
    return output


def _bootstrap_core(
    *,
    per_concept_rise: pd.DataFrame,
    per_concept_scores: pd.DataFrame,
    source: str,
    item_set: str,
    seed: int,
    analysis_config: Mapping[str, Any],
    score_column: str = "s",
    metric_variant: str = "s",
    include_decision_rules: bool = True,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute OPS-14 rule, W_L, and landmark summaries for one item set."""
    language_order = ("vi", "en", "zh", "fr", "id")
    rule_rows: list[dict[str, Any]] = []
    landmark_rows: list[dict[str, Any]] = []
    boot_p_rows: list[dict[str, Any]] = []
    key_columns = ["model", "format"]
    groups = sorted(
        set(tuple(map(str, row)) for row in per_concept_rise[key_columns].drop_duplicates().itertuples(index=False, name=None))
    )
    for model, fmt in groups:
        rise_group = per_concept_rise.loc[
            per_concept_rise["model"].astype(str).eq(model)
            & per_concept_rise["format"].astype(str).eq(fmt)
        ]
        score_group = per_concept_scores.loc[
            per_concept_scores["model"].astype(str).eq(model)
            & per_concept_scores["format"].astype(str).eq(fmt)
        ]
        concept_ids, layers, rise_values = _complete_matrix(
            rise_group, languages=language_order, value="rise", allow_missing=allow_missing
        )
        score_ids, score_layers, score_values = _complete_matrix(
            score_group, languages=language_order, value=score_column, allow_missing=allow_missing
        )
        common = sorted(set(concept_ids) & set(score_ids))
        if common != concept_ids or common != score_ids or layers != score_layers:
            rise_positions = {concept_id: i for i, concept_id in enumerate(concept_ids)}
            score_positions = {concept_id: i for i, concept_id in enumerate(score_ids)}
            rise_values = rise_values[[rise_positions[item] for item in common]]
            score_values = score_values[[score_positions[item] for item in common]]
            concept_ids = common
        finite_concepts = np.isfinite(rise_values).all(axis=(1, 2)) & np.isfinite(
            score_values
        ).all(axis=(1, 2))
        if not finite_concepts.all():
            if not allow_missing:
                raise ValueError("Analysis tensor has missing or non-finite concept/language/layer cells")
            rise_values = rise_values[finite_concepts]
            score_values = score_values[finite_concepts]
            concept_ids = [item for item, keep in zip(concept_ids, finite_concepts, strict=True) if keep]
            if not concept_ids:
                continue
        indices = concept_indices(
            concept_ids,
            resamples=int(analysis_config["bootstrap_resamples"]),
            seed=seed,
            key=bootstrap_key(
                item_set,
                model,
                fmt,
                template=str(analysis_config["bootstrap_key_template"]),
            ),
        )
        observed_rise = rise_values.mean(axis=0)
        bootstrap_rise = bootstrap_means(rise_values, indices)
        window = analysis_config["window"]
        if include_decision_rules:
            maxima = observed_rise.max(axis=1)
            maxima_boot = bootstrap_rise.max(axis=2)
            max_layers = np.asarray(layers, dtype=np.int64)[np.argmax(observed_rise, axis=1)]
            quantities = decision_rule_quantities(
                {language: float(maxima[index]) for index, language in enumerate(language_order)}
            )
            boot_h1 = maxima_boot[:, 1] - (maxima_boot[:, 3] + maxima_boot[:, 4])
            boot_h1_ratio = np.divide(
                maxima_boot[:, 1],
                (maxima_boot[:, 3] + maxima_boot[:, 4]) / 2.0,
                out=np.full(maxima_boot.shape[0], np.nan),
                where=(maxima_boot[:, 3] + maxima_boot[:, 4]) > 0,
            )
            minimum_boot = maxima_boot.min(axis=1)
            maximum_boot = maxima_boot.max(axis=1)
            boot_band = np.divide(
                maximum_boot,
                minimum_boot,
                out=np.full(maxima_boot.shape[0], np.nan),
                where=minimum_boot > 0,
            )
            for index, language in enumerate(language_order):
                low, high = percentile_interval(maxima_boot[:, index])
                rule_rows.append(
                    {
                        "source": source,
                        "model": model,
                        "format": fmt,
                        "item_set": item_set,
                        "metric_variant": metric_variant,
                        "target_lang": language,
                        "M_L": float(maxima[index]),
                        "layer": int(max_layers[index]),
                        "ci_low": low,
                        "ci_high": high,
                        "straddles_zero": bool(low <= 0.0 <= high),
                        "n_concepts": len(concept_ids),
                        "bootstrap_resamples": int(len(indices)),
                        "valid_draws": int(np.isfinite(maxima_boot[:, index]).sum()),
                    }
                )
                window_test = window_positive_test(
                    rise_values[:, index, :],
                    layers=layers,
                    window=window,
                    indices=indices,
                )
                boot_p_rows.append(
                    {
                        "source": source,
                        "model": model,
                        "format": fmt,
                        "item_set": item_set,
                        "metric_variant": metric_variant,
                        "target_lang": language,
                        "W_L": float(window_test["W_L"]),
                        "ci_low": float(window_test["ci_low"]),
                        "ci_high": float(window_test["ci_high"]),
                        "p_value": float(window_test["p_value"]),
                        "valid_draws": int(window_test["valid_draws"]),
                        "n_concepts": len(concept_ids),
                        "bootstrap_resamples": int(len(indices)),
                    }
                )
            h1_low, h1_high = percentile_interval(boot_h1)
            ratio_low, ratio_high = percentile_interval(boot_h1_ratio)
            band_low, band_high = percentile_interval(boot_band)
            control_boot = (maxima_boot[:, 3] + maxima_boot[:, 4]) / 2.0
            control_low, control_high = percentile_interval(control_boot)
            rule_rows.append(
                {
                    "source": source,
                    "model": model,
                    "format": fmt,
                    "item_set": item_set,
                    "metric_variant": metric_variant,
                    "target_lang": "__H1_H2__",
                    "M_L": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "straddles_zero": False,
                    "p_value": np.nan,
                    "n_concepts": len(concept_ids),
                    "bootstrap_resamples": int(len(indices)),
                    **quantities,
                    "control_mean": (float(maxima[3]) + float(maxima[4])) / 2.0,
                    "control_mean_ci_low": control_low,
                    "control_mean_ci_high": control_high,
                    "h1_q_ci_low": h1_low,
                    "h1_q_ci_high": h1_high,
                    "h1_q_straddles_zero": bool(h1_low <= 0.0 <= h1_high),
                    "h1_q_valid_resamples": int(np.isfinite(boot_h1).sum()),
                    "h1_ratio_ci_low": ratio_low,
                    "h1_ratio_ci_high": ratio_high,
                    "h1_ratio_straddles_2": bool(ratio_low <= 2.0 <= ratio_high),
                    "h1_ratio_valid_resamples": int(np.isfinite(boot_h1_ratio).sum()),
                    "h2_band_ci_low": band_low,
                    "h2_band_ci_high": band_high,
                    "h2_band_valid_resamples": int(np.isfinite(boot_band).sum()),
                    "h2_band_straddles_1_5": bool(band_low <= 1.5 <= band_high),
                }
            )

        observed_scores = score_values.mean(axis=0)
        score_boot = bootstrap_means(score_values, indices)
        model_n_layers = int(max(layers))
        observed_landmarks = landmarks(
            {language: observed_scores[i] for i, language in enumerate(language_order)},
            {language: observed_rise[i] for i, language in enumerate(language_order)},
            layers=layers,
            n_layers=model_n_layers,
            thresholds=analysis_config["l_concept_fractions"],
        )
        landmark_samples = _landmark_bootstrap_samples(
            score_boot,
            bootstrap_rise,
            layers=layers,
            n_layers=model_n_layers,
            thresholds=analysis_config["l_concept_fractions"],
        )
        for metric, value in observed_landmarks.items():
            samples = np.asarray(landmark_samples.get(metric, []), dtype=np.float64)
            lower, upper = percentile_interval(samples)
            landmark_rows.append(
                {
                    "source": source,
                    "model": model,
                    "format": fmt,
                    "item_set": item_set,
                    "metric_variant": metric_variant,
                    "metric": metric,
                    "value": value,
                    "ci_low": lower,
                    "ci_high": upper,
                    "valid_draws": int(np.isfinite(samples).sum()),
                    "bootstrap_resamples": int(len(indices)),
                    "n_concepts": len(concept_ids),
                    "n_layers": model_n_layers,
                }
            )
    return (
        pd.DataFrame.from_records(rule_rows),
        pd.DataFrame.from_records(landmark_rows),
        pd.DataFrame.from_records(boot_p_rows),
    )


def _bootstrap_readout_sensitivities(
    per_concept_rise: pd.DataFrame,
    *,
    source: str,
    item_set: str,
    seed: int,
    analysis_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Bootstrap individual alternate-readout maxima on the frozen item set."""
    rows: list[dict[str, Any]] = []
    if per_concept_rise.empty:
        return pd.DataFrame()
    grouping = ["model", "format", "target_lang", "target_variant"]
    for (model, fmt, language, variant), group in per_concept_rise.groupby(
        grouping, sort=True, dropna=False
    ):
        concept_ids = sorted(group["concept_id"].astype(str).unique())
        layers = sorted(map(int, group["layer"].unique()))
        matrix = (
            group.assign(concept_id=group["concept_id"].astype(str))
            .pivot(index="concept_id", columns="layer", values="rise")
            .reindex(index=concept_ids, columns=layers)
            .to_numpy(dtype=np.float64)
        )
        complete = np.isfinite(matrix).all(axis=1)
        concept_ids = [concept for concept, keep in zip(concept_ids, complete, strict=True) if keep]
        matrix = matrix[complete]
        if not concept_ids:
            continue
        indices = concept_indices(
            concept_ids,
            resamples=int(analysis_config["bootstrap_resamples"]),
            seed=seed,
            key=bootstrap_key(
                item_set,
                model,
                fmt,
                template=str(analysis_config["bootstrap_key_template"]),
            ),
        )
        observed = matrix.mean(axis=0)
        samples = bootstrap_means(matrix, indices).max(axis=1)
        low, high = percentile_interval(samples)
        rows.append(
            {
                "source": source,
                "model": model,
                "format": fmt,
                "item_set": item_set,
                "metric_variant": f"{language}_{variant}",
                "target_lang": language,
                "target_variant": variant,
                "M_L": float(observed.max()),
                "layer": int(layers[int(np.argmax(observed))]),
                "ci_low": low,
                "ci_high": high,
                "valid_draws": int(np.isfinite(samples).sum()),
                "n_concepts": len(concept_ids),
                "bootstrap_resamples": int(len(indices)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_long_frame(frame: pd.DataFrame, path: Path, *, compression: str) -> None:
    """Write an analysis table atomically through a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression=compression)
    temporary.replace(path)


def _family_root(repo: Path, source: str, family: str | None) -> Path:
    """Return the isolated output directory for one source and model family."""
    root = repo / "results/analysis" / source
    return root if family is None else root / f"family={family}"


def _run_core_family(
    source: str,
    *,
    cfg: Mapping[str, Any],
    repo: Path,
    confirm_checkpoint_h: bool,
    family: str | None,
    analysis_identity: AnalysisCommitIdentity | None = None,
) -> dict[str, Any]:
    """Run E3 landmarks without C1, E4 rules, and E5 LA for one family."""
    started = time.perf_counter()
    raw = _load_input(source, repo, cfg, family)
    scores = content_token_scores(raw)
    deltas = matched_distractor_deltas(scores)
    main_ids: set[str] | None = None
    cognate_ids: set[str] | None = None
    if source == "primary":
        main_ids, cognate_ids = _primary_scope_ids(repo, cfg)
        main_deltas = deltas.loc[deltas["concept_id"].astype(str).isin(main_ids)].copy()
        main_scores = scores.loc[scores["concept_id"].astype(str).isin(main_ids)].copy()
    else:
        main_deltas = deltas.copy()
        main_scores = scores.copy()

    main_readouts = _readout_rows(main_deltas)
    early_depth_max = float(cfg["e1"]["early_depth_max"])
    main_rises = concept_rise_curves(main_readouts, early_depth_max=early_depth_max)
    main_rises_s1 = concept_rise_curves(
        main_readouts, early_depth_max=early_depth_max, value_column="delta_s1"
    )
    main_rises_splus = concept_rise_curves(
        main_readouts, early_depth_max=early_depth_max, value_column="delta_s_plus"
    )
    main_rises_sum = concept_rise_curves(
        main_readouts, early_depth_max=early_depth_max, value_column="delta_sequence_sum"
    )
    sensitivity_readouts = _sensitivity_readout_rows(main_deltas)
    sensitivity_rises = (
        concept_rise_curves(sensitivity_readouts, early_depth_max=early_depth_max)
        if not sensitivity_readouts.empty
        else pd.DataFrame()
    )
    main_score_readouts = _readout_rows(
        main_scores.loc[main_scores["role"].astype(str).eq("correct")]
    )

    if cognate_ids is not None:
        rule_rises = main_rises.loc[
            main_rises["format"].astype(str).isin({"repetition", "translation"})
            & main_rises["concept_id"].astype(str).isin(cognate_ids)
        ].copy()
        rule_rises_s1 = main_rises_s1.loc[
            main_rises_s1["format"].astype(str).isin({"repetition", "translation"})
            & main_rises_s1["concept_id"].astype(str).isin(cognate_ids)
        ].copy()
        rule_rises_splus = main_rises_splus.loc[
            main_rises_splus["format"].astype(str).isin({"repetition", "translation"})
            & main_rises_splus["concept_id"].astype(str).isin(cognate_ids)
        ].copy()
        rule_rises_sum = main_rises_sum.loc[
            main_rises_sum["format"].astype(str).isin({"repetition", "translation"})
            & main_rises_sum["concept_id"].astype(str).isin(cognate_ids)
        ].copy()
        rule_scores = main_score_readouts.loc[
            main_score_readouts["format"].astype(str).isin({"repetition", "translation"})
            & main_score_readouts["concept_id"].astype(str).isin(cognate_ids)
        ].copy()
        for fmt, expected in (("repetition", 729), ("translation", 721)):
            observed = rule_rises.loc[rule_rises["format"].astype(str).eq(fmt), "concept_id"].nunique()
            if observed != expected:
                raise AssertionError(f"cf729 {fmt} count {observed}, expected {expected}")
        item_set = "cf729"
    else:
        rule_rises = main_rises.loc[main_rises["format"].astype(str).isin({"repetition", "translation"})].copy()
        rule_rises_s1 = main_rises_s1.loc[main_rises_s1["format"].astype(str).isin({"repetition", "translation"})].copy()
        rule_rises_splus = main_rises_splus.loc[main_rises_splus["format"].astype(str).isin({"repetition", "translation"})].copy()
        rule_rises_sum = main_rises_sum.loc[main_rises_sum["format"].astype(str).isin({"repetition", "translation"})].copy()
        rule_scores = main_score_readouts.loc[main_score_readouts["format"].astype(str).isin({"repetition", "translation"})].copy()
        item_set = "main"

    analysis_cfg = cfg["analysis"]
    rules, cf_landmarks, holm_w = _bootstrap_core(
        per_concept_rise=rule_rises,
        per_concept_scores=rule_scores,
        source=source,
        item_set=item_set,
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
    )
    rules_s1, cf_landmarks_s1, _holm_s1 = _bootstrap_core(
        per_concept_rise=rule_rises_s1,
        per_concept_scores=rule_scores,
        source=source,
        item_set=item_set,
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
        score_column="s1",
        metric_variant="s1",
    )
    rules_splus, _cf_landmarks_splus, _holm_splus = _bootstrap_core(
        per_concept_rise=rule_rises_splus if cognate_ids is not None else main_rises_splus,
        per_concept_scores=rule_scores if cognate_ids is not None else main_score_readouts,
        source=source,
        item_set=item_set,
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
        score_column="s_plus",
        metric_variant="s_plus",
        allow_missing=True,
    )
    rules_sum, _cf_landmarks_sum, _holm_sum = _bootstrap_core(
        per_concept_rise=rule_rises_sum if cognate_ids is not None else main_rises_sum,
        per_concept_scores=rule_scores if cognate_ids is not None else main_score_readouts,
        source=source,
        item_set=item_set,
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
        score_column="sequence_sum",
        metric_variant="sequence_sum",
    )
    main_rules, main_landmarks, _main_holm = _bootstrap_core(
        per_concept_rise=main_rises,
        per_concept_scores=main_score_readouts,
        source=source,
        item_set="main",
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
        include_decision_rules=False,
    )
    _main_rules_s1, main_landmarks_s1, _main_holm_s1 = _bootstrap_core(
        per_concept_rise=main_rises_s1,
        per_concept_scores=main_score_readouts,
        source=source,
        item_set="main",
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
        score_column="s1",
        metric_variant="s1",
        include_decision_rules=False,
    )
    del main_rules, _main_rules_s1, _main_holm, _main_holm_s1
    landmark_summary = pd.concat(
        [cf_landmarks, cf_landmarks_s1, main_landmarks, main_landmarks_s1],
        ignore_index=True,
    )
    sensitivity_rule_rises = sensitivity_rises
    if cognate_ids is not None and not sensitivity_rule_rises.empty:
        sensitivity_rule_rises = sensitivity_rule_rises.loc[
            sensitivity_rule_rises["concept_id"].astype(str).isin(cognate_ids)
            & sensitivity_rule_rises["format"].astype(str).isin({"repetition", "translation"})
        ].copy()
    sensitivity_rules = _bootstrap_readout_sensitivities(
        sensitivity_rule_rises,
        source=source,
        item_set=item_set,
        seed=int(cfg["seed"]),
        analysis_config=analysis_cfg,
    )
    rule_summary = pd.concat(
        [rules, rules_s1, rules_splus, rules_sum, sensitivity_rules], ignore_index=True
    )

    la_table, la_summary = _compute_la_source(
        source,
        config=cfg,
        repo=repo,
        main_ids=main_ids,
        family_filter=family,
    )
    output_root = _family_root(repo, source, family)
    output_root.mkdir(parents=True, exist_ok=True)

    score_table = _score_long(scores)
    delta_parts = [
        _delta_long(deltas, metric_column=column, metric_name=name)
        for column, name in (
            ("delta_s", "delta"),
            ("delta_s1", "delta_first"),
            ("delta_s_plus", "delta_continuation"),
            ("delta_sequence_sum", "delta_sequence_sum"),
        )
    ]
    delta_table = pd.concat(delta_parts, ignore_index=True)
    rise_parts = []
    for frame, column, name in (
        (main_rises, "rise", "rise_main"),
        (main_rises_s1, "rise", "rise_s1"),
        (main_rises_splus, "rise", "rise_continuation"),
        (main_rises_sum, "rise", "rise_sequence_sum"),
    ):
        rise_parts.append(_delta_long(frame, metric_column=column, metric_name=name))
    if not sensitivity_rises.empty:
        rise_parts.append(
            _delta_long(sensitivity_rises, metric_column="rise", metric_name="rise_readout_sensitivity")
        )
    if cognate_ids is not None:
        rise_parts.extend(
            [
                _delta_long(rule_rises, metric_column="rise", metric_name="rise_cf729"),
                _delta_long(rule_rises_s1, metric_column="rise", metric_name="rise_cf729_s1"),
            ]
        )
    rise_table = pd.concat(rise_parts, ignore_index=True)

    rule_long = _summary_metric_long(rule_summary, source_rows=scores, kind="rule")
    landmark_long = _summary_metric_long(landmark_summary, source_rows=scores, kind="landmark")
    holm_rows: list[dict[str, Any]] = []
    for row in holm_w.to_dict(orient="records"):
        base = {
            "model": row["model"],
            "model_revision": str(scores.loc[scores["model"].astype(str).eq(str(row["model"])), "model_revision"].iloc[0]),
            "pass": "analysis",
            "format": row["format"],
            "condition": "diac",
            "fewshot_set": 1 if source == "primary" else None,
            "concept_id": None,
            "target_lang": row["target_lang"],
            "target_variant": None,
            "role": "path",
            "ref_concept_id": None,
            "tok_idx": None,
            "token_id": None,
            "is_content": None,
            "layer": None,
            "n_concepts": row["n_concepts"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "valid_draws": row["valid_draws"],
            "p_value": row["p_value"],
            "p_holm": np.nan,
            "item_set": row.get("item_set"),
            "metric_variant": row.get("metric_variant", "s"),
        }
        for metric, value in (("W_L", row["W_L"]), ("p_value", row["p_value"])):
            holm_rows.append({**base, "metric": metric, "value": value})
    holm_long = pd.DataFrame.from_records(holm_rows)
    _write_long_frame(score_table, output_root / "scores.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(delta_table, output_root / "deltas.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(rise_table, output_root / "rise.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(rule_long, output_root / "rule_quantities.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(landmark_long, output_root / "landmarks.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(holm_long, output_root / "holm_w_raw.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(la_table, output_root / "la.parquet", compression=cfg["runtime"]["parquet_compression"])
    _write_long_frame(la_summary, output_root / "la_summary.parquet", compression=cfg["runtime"]["parquet_compression"])
    summary = {
        "source": source,
        "family": family,
        "input_rows": int(len(raw)),
        "score_groups": int(len(scores)),
        "delta_pairs": int(len(deltas)),
        "rise_rows": int(len(main_rises)),
        "score_output_rows": int(len(score_table)),
        "delta_output_rows": int(len(delta_table)),
        "rise_output_rows": int(len(rise_table)),
        "la_output_rows": int(len(la_table)),
        "rule_output_rows": int(len(rule_long)),
        "landmark_output_rows": int(len(landmark_long)),
        "holm_w_output_rows": int(len(holm_long)),
        "outputs": [
            str((output_root / name).relative_to(repo))
            for name in (
                "scores.parquet", "deltas.parquet", "rise.parquet",
                "rule_quantities.parquet", "landmarks.parquet",
                "holm_w_raw.parquet", "la.parquet", "la_summary.parquet",
            )
        ],
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    if analysis_identity is not None:
        summary.update(
            {
                "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
                "executed_commit": analysis_identity.executed_commit,
            }
        )
    (output_root / "core_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _LOGGER.info(
        "family core complete: family=%s wall_time_seconds=%.3f peak_rss_bytes=%s rows=%s",
        family or "synthetic",
        summary["wall_time_seconds"],
        summary["peak_rss_bytes"],
        summary["input_rows"],
    )
    return summary


def _filtered_parquet_rows(
    path: Path,
    *,
    columns: list[str],
    predicate: Any,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    """Read selected parquet rows incrementally to bound peak memory."""
    parts: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        selected = predicate(frame)
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _landmark_tensor(
    frame: pd.DataFrame,
    *,
    concept_ids: list[str],
    languages: tuple[str, ...],
    value_column: str,
) -> tuple[list[int], np.ndarray]:
    """Build a finite concept × readout × block tensor from one score table."""
    selected = frame.loc[frame["concept_id"].astype(str).isin(concept_ids)].copy()
    selected["concept_id"] = selected["concept_id"].astype(str)
    selected = selected.loc[pd.to_numeric(selected["layer"], errors="raise") >= 1]
    if selected.duplicated(["concept_id", "target_lang", "layer"]).any():
        raise AssertionError("C1 score rows duplicate a concept/readout/layer cell")
    layers = sorted(map(int, selected["layer"].unique()))
    matrices = []
    for language in languages:
        language_rows = selected.loc[selected["target_lang"].astype(str).eq(language)]
        pivot = language_rows.pivot(index="concept_id", columns="layer", values=value_column)
        matrices.append(pivot.reindex(index=concept_ids, columns=layers).to_numpy(dtype=np.float64))
    tensor = np.stack(matrices, axis=1)
    if not np.isfinite(tensor).all():
        raise AssertionError("C1 requires complete finite values for every included item/readout/layer")
    return layers, tensor


def _landmark_thresholds(
    curves: np.ndarray,
    *,
    layers: list[int],
    thresholds: list[float],
    is_probability: bool,
) -> dict[float, int | None]:
    """Calculate all configured L_concept thresholds from a concept tensor."""
    means = curves.mean(axis=0, dtype=np.float64)
    by_language = {language: means[i] for i, language in enumerate(("vi", "en", "zh", "fr", "id"))}
    result: dict[float, int | None] = {}
    for threshold in thresholds:
        if is_probability:
            value = probability_concept_layer(by_language, layers=layers, threshold=threshold)
        else:
            value = concept_layer(by_language, layers=layers, threshold=threshold)
        result[float(threshold)] = value
    return result


def _c1_effect_rows(
    *,
    model: str,
    fmt: str,
    item_set: str,
    contrast: str,
    ids: list[str],
    m2_curves: np.ndarray,
    m1_curves: np.ndarray,
    layers: list[int],
    cfg: Mapping[str, Any],
    threshold_values: list[float],
) -> list[dict[str, Any]]:
    """Compute the paired M2-minus-M1 L_concept effects for one item set."""
    indices = concept_indices(
        ids,
        resamples=int(cfg["analysis"]["bootstrap_resamples"]),
        seed=int(cfg["seed"]),
        key=bootstrap_key(
            item_set,
            model,
            fmt,
            template=str(cfg["analysis"]["bootstrap_key_template"]),
        ),
    )
    point_m2 = _landmark_thresholds(
        m2_curves, layers=layers, thresholds=threshold_values, is_probability=False
    )
    point_m1 = _landmark_thresholds(
        m1_curves, layers=layers, thresholds=threshold_values, is_probability=True
    )
    m2_boot = m2_curves[indices]
    m1_boot = m1_curves[indices]
    rows: list[dict[str, Any]] = []
    for threshold in threshold_values:
        samples: list[float] = []
        for m2_draw, m1_draw in zip(m2_boot, m1_boot, strict=True):
            l2 = _landmark_thresholds(
                m2_draw, layers=layers, thresholds=[threshold], is_probability=False
            )[threshold]
            l1 = _landmark_thresholds(
                m1_draw, layers=layers, thresholds=[threshold], is_probability=True
            )[threshold]
            if l2 is not None and l1 is not None:
                samples.append(float(l2 - l1))
        low, high, valid = percentile_interval_with_count(samples)
        point = (
            float(point_m2[threshold] - point_m1[threshold])
            if point_m2[threshold] is not None and point_m1[threshold] is not None
            else float("nan")
        )
        for unit, scale in (("blocks", 1.0), ("depth", 1.0 / float(max(layers)))):
            rows.append(
                {
                    "model": model,
                    "format": fmt,
                    "item_set": item_set,
                    "contrast": contrast,
                    "threshold": threshold,
                    "metric": f"{contrast}_{int(threshold * 100)}_{unit}",
                    "value": point * scale,
                    "ci_low": low * scale,
                    "ci_high": high * scale,
                    "valid_draws": valid,
                    "bootstrap_resamples": int(len(indices)),
                    "n_concepts": len(ids),
                    "units": unit,
                }
            )
    return rows


def _run_c1_family(
    cfg: Mapping[str, Any], repo: Path, family: str, *, source: str = "primary"
) -> dict[str, Any]:
    """Compute C1 instrument and paired-selection landmarks for one family."""
    if source not in {"primary", "trial"}:
        raise ValueError(f"C1 does not support source={source!r}")
    started = time.perf_counter()
    languages = ("vi", "en", "zh", "fr", "id")
    variants = PRIMARY_VARIANTS
    thresholds = list(map(float, cfg["analysis"]["l_concept_fractions"]))
    source_root = _family_root(repo, source, family)
    score_path = source_root / "scores.parquet"
    score_rows = _filtered_parquet_rows(
        score_path,
        columns=["model", "model_revision", "format", "concept_id", "target_lang", "target_variant", "role", "layer", "metric", "value"],
        predicate=lambda frame: frame.loc[
            frame["metric"].astype(str).eq("s")
            & frame["role"].astype(str).eq("correct")
            & frame.apply(
                lambda row: variants.get(str(row["target_lang"])) == str(row["target_variant"]),
                axis=1,
            )
        ],
    )
    if score_rows.empty:
        raise FileNotFoundError(f"C1 requires frozen core scores at {score_path}")
    model = str(score_rows["model"].iloc[0])
    output_rows: list[dict[str, Any]] = []
    formats = ("repetition", "translation", "cloze") if source == "primary" else TRIAL_FORMATS
    main_scope_ids: set[str] = set()
    eligible_main: set[str] = set()
    eligible_all: set[str] = set()
    if source == "primary":
        concepts_path = repo / str(cfg["paths"]["concepts_jsonl"])
        main_scope_ids, _cognate_scope_ids = _primary_scope_ids(repo, dict(cfg))
        for concept in _read_jsonl_rows(concepts_path):
            concept_id = str(concept["concept_id"])
            is_eligible = str(concept.get(f"m1_eligible_{family}", "false")).lower() == "true"
            is_extension = str(concept.get("m1_extension", "false")).lower() == "true"
            if is_eligible:
                eligible_all.add(concept_id)
            if is_eligible and str(concept.get("split")) == "test" and not is_extension:
                eligible_main.add(concept_id)

    for fmt in formats:
        m2_fmt = score_rows.loc[score_rows["format"].astype(str).eq(fmt)].copy()
        observed_m2_ids = set(m2_fmt["concept_id"].astype(str))
        main_ids = sorted(
            observed_m2_ids & main_scope_ids if source == "primary" else observed_m2_ids
        )
        if not main_ids:
            raise AssertionError(f"C1 has no main M2 values for {family}/{fmt}")
        main_layers, m2_main = _landmark_tensor(
            m2_fmt, concept_ids=main_ids, languages=languages, value_column="value"
        )
        m1_path = repo / "results/analysis/m1" / source / (
            f"family={family}__format={fmt}__pass=m1__source={source}.parquet"
        )
        m1_rows = _filtered_parquet_rows(
            m1_path,
            columns=["model", "model_revision", "format", "concept_id", "target_lang", "target_variant", "role", "layer", "metric", "value"],
            predicate=lambda frame: frame.loc[
                frame["metric"].astype(str).eq("prefix_sum_probability")
                & frame["role"].astype(str).eq("correct")
                & frame.apply(
                    lambda row: variants.get(str(row["target_lang"])) == str(row["target_variant"]),
                    axis=1,
                )
            ],
        )
        if m1_rows.empty:
            raise FileNotFoundError(f"C1 requires E2 results at {m1_path}")
        m1_ids = sorted(m1_rows["concept_id"].astype(str).unique())
        m2_available_ids = set(m2_fmt["concept_id"].astype(str))
        if source == "primary":
            expected_main_m1 = sorted(eligible_main & m2_available_ids)
            expected_all_m1 = sorted(eligible_all & m2_available_ids)
            observed_main_m1 = sorted(set(m1_ids) & set(main_ids))
            if observed_main_m1 != expected_main_m1 or m1_ids != expected_all_m1:
                raise AssertionError(
                    f"E2 M1 IDs differ from frozen eligibility for {family}/{fmt}: "
                    f"main {len(observed_main_m1)}/{len(expected_main_m1)}, "
                    f"all {len(m1_ids)}/{len(expected_all_m1)}"
                )
            m1_main_ids = sorted(set(m1_ids) & eligible_main & set(main_ids))
        else:
            if not set(m1_ids).issubset(m2_available_ids):
                raise AssertionError(f"Trial E2 IDs are absent from core scores for {family}/{fmt}")
            # Trial records are a smoke fixture, not a registered main/extension split.
            m1_main_ids = sorted(set(m1_ids) & set(main_ids))
        m1_all_ids = sorted(set(m1_ids))
        # E2's saved rows are the authoritative format-specific family-eligible set.
        m1_layers, m1_all_tensor = _landmark_tensor(
            m1_rows,
            concept_ids=m1_all_ids,
            languages=languages,
            value_column="value",
        )
        if m1_layers != main_layers:
            raise AssertionError(f"C1 M1/M2 layer mismatch for {family}/{fmt}")
        m1_positions = {concept_id: i for i, concept_id in enumerate(m1_all_ids)}
        m1_main_tensor = m1_all_tensor[[m1_positions[concept_id] for concept_id in m1_main_ids]]
        m2_main_positions = {concept_id: i for i, concept_id in enumerate(main_ids)}
        m2_m1_main = m2_main[[m2_main_positions[concept_id] for concept_id in m1_main_ids]]

        output_rows.extend(
            _c1_effect_rows(
                model=model,
                fmt=fmt,
                item_set="m1_main",
                contrast="instrument_effect",
                ids=m1_main_ids,
                m2_curves=m2_m1_main,
                m1_curves=m1_main_tensor,
                layers=main_layers,
                cfg=cfg,
                threshold_values=thresholds,
            )
        )
        output_rows.extend(
            _c1_effect_rows(
                model=model,
                fmt=fmt,
                item_set="m1_all",
                contrast="extension_inclusive_instrument_effect",
                ids=m1_all_ids,
                m2_curves=_landmark_tensor(
                    m2_fmt,
                    concept_ids=m1_all_ids,
                    languages=languages,
                    value_column="value",
                )[1],
                m1_curves=m1_all_tensor,
                layers=main_layers,
                cfg=cfg,
                threshold_values=thresholds,
            )
        )

        selection_indices = concept_indices(
            main_ids,
            resamples=int(cfg["analysis"]["bootstrap_resamples"]),
            seed=int(cfg["seed"]),
            key=bootstrap_key(
                "main", model, fmt, template=str(cfg["analysis"]["bootstrap_key_template"])
            ),
        )
        paired = paired_selection_draws(main_ids, m1_main_ids, selection_indices)
        full_samples = m2_main[selection_indices]
        subset_samples = [m2_main[indices] for indices in paired]
        full_point = _landmark_thresholds(
            m2_main, layers=main_layers, thresholds=thresholds, is_probability=False
        )
        subset_point = _landmark_thresholds(
            m2_m1_main, layers=main_layers, thresholds=thresholds, is_probability=False
        )
        for threshold in thresholds:
            diffs: list[float] = []
            for full_draw, subset_draw in zip(full_samples, subset_samples, strict=True):
                full_value = _landmark_thresholds(
                    full_draw, layers=main_layers, thresholds=[threshold], is_probability=False
                )[threshold]
                subset_value = (
                    _landmark_thresholds(
                        subset_draw, layers=main_layers, thresholds=[threshold], is_probability=False
                    )[threshold]
                    if len(subset_draw)
                    else None
                )
                if full_value is not None and subset_value is not None:
                    diffs.append(float(subset_value - full_value))
            low, high, valid = percentile_interval_with_count(diffs)
            point = (
                float(subset_point[threshold] - full_point[threshold])
                if subset_point[threshold] is not None and full_point[threshold] is not None
                else float("nan")
            )
            for unit, scale in (("blocks", 1.0), ("depth", 1.0 / float(max(main_layers)))):
                output_rows.append(
                    {
                        "model": model,
                        "format": fmt,
                        "item_set": "main",
                        "contrast": "selection_effect",
                        "threshold": threshold,
                        "metric": f"selection_effect_{int(threshold * 100)}_{unit}",
                        "value": point * scale,
                        "ci_low": low * scale,
                        "ci_high": high * scale,
                        "valid_draws": valid,
                        "bootstrap_resamples": len(selection_indices),
                        "n_concepts": len(main_ids),
                        "units": unit,
                    }
                )

    output = pd.DataFrame.from_records(output_rows)
    revision = str(score_rows["model_revision"].iloc[0])
    for column, default in (
        ("model_revision", revision),
        ("pass", "analysis"),
        ("condition", "diac"),
        ("fewshot_set", 1 if source == "primary" else None),
        ("concept_id", None),
        ("target_lang", None),
        ("target_variant", None),
        ("role", "path"),
        ("ref_concept_id", None),
        ("tok_idx", None),
        ("token_id", None),
        ("is_content", None),
        ("layer", None),
    ):
        output[column] = default
    output_path = source_root / "c1.parquet"
    _write_long_frame(output, output_path, compression=cfg["runtime"]["parquet_compression"])
    summary = {
        "family": family,
        "model": model,
        "c1_rows": int(len(output)),
        "output": str(output_path.relative_to(repo)),
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    _LOGGER.info(
        "family C1 complete: family=%s wall_time_seconds=%.3f peak_rss_bytes=%s rows=%s",
        family, summary["wall_time_seconds"], summary["peak_rss_bytes"], summary["c1_rows"],
    )
    return summary


def _run_c1_stage(
    cfg: Mapping[str, Any], repo: Path, families: list[str], *, source: str = "primary"
) -> dict[str, Any]:
    """Run C1 serially by family from collected E2 and E3 tables."""
    summaries = [_run_c1_family(cfg, repo, family, source=source) for family in families]
    return {"families": summaries, "rows": sum(item["c1_rows"] for item in summaries)}


def _run_competence_family(
    cfg: Mapping[str, Any], repo: Path, family: str
) -> dict[str, Any]:
    """Run A10 scoring for one model family from stored primary generations."""
    started = time.perf_counter()
    main_ids, _cognate_ids = _primary_scope_ids(repo, cfg)
    concepts = _load_primary_concepts(repo, main_ids)
    generations = _load_primary_generations(repo, cfg, main_ids, family=family)
    scored = competence_rows(generations, concepts)
    output_root = _family_root(repo, "primary", family)
    output_root.mkdir(parents=True, exist_ok=True)
    scored_path = output_root / "competence.parquet"
    scored.sort_values(["model", "format", "concept_id"], kind="stable").to_parquet(
        scored_path, index=False, compression=cfg["runtime"]["parquet_compression"]
    )
    summary = {
        "family": family,
        "generation_rows": len(generations),
        "competence_rows": len(scored),
        "shared_source_rows": int(scored["shared_ru_source"].sum()) if not scored.empty else 0,
        "outputs": [str(scored_path.relative_to(repo))],
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    _LOGGER.info(
        "family competence complete: family=%s wall_time_seconds=%.3f peak_rss_bytes=%s rows=%s",
        family, summary["wall_time_seconds"], summary["peak_rss_bytes"], summary["competence_rows"],
    )
    return summary


def _optional_number(value: Any) -> float | None:
    """Convert a report cell to a finite float or None."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _first_summary_row(
    frame: pd.DataFrame,
    *,
    model: str,
    fmt: str,
    metric: str,
    target_lang: str | None = None,
    metric_variant: str = "s",
) -> pd.Series | None:
    """Select exactly one summary row by its frozen identity keys."""
    selected = frame.loc[
        frame["model"].astype(str).eq(model)
        & frame["format"].astype(str).eq(fmt)
        & frame["metric"].astype(str).eq(metric)
    ]
    if "metric_variant" in selected:
        selected = selected.loc[selected["metric_variant"].astype(str).eq(metric_variant)]
    if target_lang is not None:
        selected = selected.loc[selected["target_lang"].astype(str).eq(target_lang)]
    if len(selected) > 1:
        raise AssertionError(
            f"Summary identity is not unique: model={model} format={fmt} metric={metric} "
            f"target_lang={target_lang} variant={metric_variant}"
        )
    return None if selected.empty else selected.iloc[0]


def _straddles(row: pd.Series | None, threshold: float) -> bool | None:
    """Return whether a reported percentile interval contains a decision threshold."""
    if row is None:
        return None
    low = _optional_number(row.get("ci_low"))
    high = _optional_number(row.get("ci_high"))
    if low is None or high is None:
        return None
    return bool(low <= threshold <= high)


def _outcome_for_format(
    rules: pd.DataFrame,
    la: pd.DataFrame,
    *,
    model: str,
    fmt: str,
    metric_variant: str,
) -> str:
    """Apply the OPS-14 H1/H2 point rules to one format and score variant."""
    la_en = _first_summary_row(la, model=model, fmt=fmt, metric="max_LA", target_lang="en")
    q_h1 = _first_summary_row(
        rules, model=model, fmt=fmt, metric="h1_q", target_lang="__H1_H2__", metric_variant=metric_variant
    )
    band = _first_summary_row(
        rules, model=model, fmt=fmt, metric="h2_band_ratio", target_lang="__H1_H2__", metric_variant=metric_variant
    )
    if la_en is None or not bool(la_en.get("la_evaluable", False)) or q_h1 is None or band is None:
        return "inconclusive"
    h1_la = _optional_number(la_en.get("value"))
    q_value = _optional_number(q_h1.get("value"))
    band_ratio = _optional_number(band.get("value"))
    lang_rows = [
        _first_summary_row(
            la, model=model, fmt=fmt, metric="max_LA", target_lang=language
        )
        for language in ("en", "zh", "fr", "id")
    ]
    max_la_values = [
        _optional_number(row.get("value")) if row is not None and bool(row.get("la_evaluable", False)) else None
        for row in lang_rows
    ]
    m_values = [
        _optional_number(
            row.get("value")
        )
        for language in ("vi", "en", "zh", "fr", "id")
        if (row := _first_summary_row(
            rules, model=model, fmt=fmt, metric="M_L", target_lang=language, metric_variant=metric_variant
        )) is not None
    ]
    if h1_la is None or q_value is None or len(max_la_values) != 4 or len(m_values) != 5:
        return "inconclusive"
    h1 = h1_la >= 0.50 and q_value >= 0.0
    h2 = all(value is not None and value < 0.20 for value in max_la_values)
    h2 = h2 and min(m_values) > 0.0 and band_ratio is not None and band_ratio <= 1.5
    if h1:
        return "H1"
    if h2:
        return "H2"
    return "inconclusive"


def _build_checkpoint_rules(rules: pd.DataFrame, la: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-format and per-model OPS-14 decision tables."""
    rows: list[dict[str, Any]] = []
    models = sorted(set(rules["model"].astype(str)) & set(la["model"].astype(str)))
    for model in models:
        for fmt in ("repetition", "translation"):
            row: dict[str, Any] = {"model": model, "format": fmt}
            primary_m = {
                language: _first_summary_row(
                    rules, model=model, fmt=fmt, metric="M_L", target_lang=language, metric_variant="s"
                )
                for language in ("vi", "en", "zh", "fr", "id")
            }
            for language, source_row in primary_m.items():
                prefix = f"M_{language}"
                row[f"{prefix}_point"] = _optional_number(source_row.get("value")) if source_row is not None else None
                row[f"{prefix}_ci_low"] = _optional_number(source_row.get("ci_low")) if source_row is not None else None
                row[f"{prefix}_ci_high"] = _optional_number(source_row.get("ci_high")) if source_row is not None else None
            h1_la = _first_summary_row(la, model=model, fmt=fmt, metric="max_LA", target_lang="en")
            q = _first_summary_row(rules, model=model, fmt=fmt, metric="h1_q", target_lang="__H1_H2__")
            ratio = _first_summary_row(rules, model=model, fmt=fmt, metric="h1_ratio", target_lang="__H1_H2__")
            band = _first_summary_row(rules, model=model, fmt=fmt, metric="h2_band_ratio", target_lang="__H1_H2__")
            row.update(
                H1_LA_point=_optional_number(h1_la.get("value")) if h1_la is not None else None,
                H1_LA_ci_low=_optional_number(h1_la.get("ci_low")) if h1_la is not None else None,
                H1_LA_ci_high=_optional_number(h1_la.get("ci_high")) if h1_la is not None else None,
                H1_LA_straddles_0_50=_straddles(h1_la, 0.50),
                H1_rise_Q_H1=_optional_number(q.get("value")) if q is not None else None,
                H1_rise_ci_low=_optional_number(q.get("ci_low")) if q is not None else None,
                H1_rise_ci_high=_optional_number(q.get("ci_high")) if q is not None else None,
                H1_rise_straddles_0=_straddles(q, 0.0),
                H1_ratio=_optional_number(ratio.get("value")) if ratio is not None else None,
                H1_ratio_ci_low=_optional_number(ratio.get("ci_low")) if ratio is not None else None,
                H1_ratio_ci_high=_optional_number(ratio.get("ci_high")) if ratio is not None else None,
                H2_band_ratio=_optional_number(band.get("value")) if band is not None else None,
                H2_band_ci_low=_optional_number(band.get("ci_low")) if band is not None else None,
                H2_band_ci_high=_optional_number(band.get("ci_high")) if band is not None else None,
                H2_band_straddles_1_5=_straddles(band, 1.5),
            )
            for language in ("en", "zh", "fr", "id"):
                la_row = _first_summary_row(la, model=model, fmt=fmt, metric="max_LA", target_lang=language)
                prefix = f"H2_LA_{language}"
                row[f"{prefix}_point"] = _optional_number(la_row.get("value")) if la_row is not None else None
                row[f"{prefix}_ci_low"] = _optional_number(la_row.get("ci_low")) if la_row is not None else None
                row[f"{prefix}_ci_high"] = _optional_number(la_row.get("ci_high")) if la_row is not None else None
                row[f"{prefix}_straddles_0_20"] = _straddles(la_row, 0.20)
            primary_outcome = _outcome_for_format(rules, la, model=model, fmt=fmt, metric_variant="s")
            s1_outcome = _outcome_for_format(rules, la, model=model, fmt=fmt, metric_variant="s1")
            row["Outcome"] = primary_outcome
            row["s1_outcome"] = s1_outcome
            rows.append(row)
    format_table = pd.DataFrame.from_records(rows)
    verdict_rows: list[dict[str, Any]] = []
    for model, group in format_table.groupby("model", sort=True):
        outcomes = dict(zip(group["format"], group["Outcome"], strict=True))
        s1_outcomes = dict(zip(group["format"], group["s1_outcome"], strict=True))
        verdict = "H1" if all(outcomes.get(fmt) == "H1" for fmt in ("repetition", "translation")) else (
            "H2" if all(outcomes.get(fmt) == "H2" for fmt in ("repetition", "translation")) else "inconclusive"
        )
        s1_verdict = "H1" if all(s1_outcomes.get(fmt) == "H1" for fmt in ("repetition", "translation")) else (
            "H2" if all(s1_outcomes.get(fmt) == "H2" for fmt in ("repetition", "translation")) else "inconclusive"
        )
        verdict_rows.append(
            {
                "model": model,
                "repetition": outcomes.get("repetition", "inconclusive"),
                "translation": outcomes.get("translation", "inconclusive"),
                "verdict": verdict,
                "s1_repetition": s1_outcomes.get("repetition", "inconclusive"),
                "s1_translation": s1_outcomes.get("translation", "inconclusive"),
                "s1_verdict": s1_verdict,
            }
        )
    verdict_table = pd.DataFrame.from_records(verdict_rows)
    if not format_table.empty:
        format_table = format_table.merge(verdict_table[["model", "verdict", "s1_verdict"]], on="model", how="left")
    return format_table, verdict_table


def _checkpoint_layer_gate_statuses(
    gate_rows: pd.DataFrame, *, fmt: str
) -> tuple[dict[int, str], int]:
    """Map each block layer to its Checkpoint-H gate status for one format."""
    selected = gate_rows.loc[
        gate_rows["format"].astype(str).eq(fmt)
        & gate_rows["metric"].astype(str).isin({"layer_pass", "undefined_denominator"})
        & gate_rows["target_lang"].isna()
    ]
    passed = selected.loc[selected["metric"].astype(str).eq("layer_pass")]
    undefined = selected.loc[selected["metric"].astype(str).eq("undefined_denominator")]
    pass_by_layer = {
        int(row.layer): float(row.value)
        for row in passed[["layer", "value"]].itertuples(index=False)
    }
    undefined_by_layer = {
        int(row.layer): float(row.value)
        for row in undefined[["layer", "value"]].itertuples(index=False)
    }
    if not pass_by_layer or set(pass_by_layer) != set(undefined_by_layer):
        raise AssertionError(f"Incomplete layer gate rows for format={fmt}")
    n_layers = max(pass_by_layer)
    expected_layers = set(range(1, n_layers + 1))
    if set(pass_by_layer) != expected_layers:
        raise AssertionError(f"Non-contiguous layer gate rows for format={fmt}")
    statuses: dict[int, str] = {}
    for layer in range(1, n_layers + 1):
        if undefined_by_layer[layer] == 1.0:
            statuses[layer] = "undefined"
        elif pass_by_layer[layer] == 1.0:
            statuses[layer] = "pass"
        elif pass_by_layer[layer] == 0.0:
            statuses[layer] = "failed"
        else:
            raise AssertionError(f"Invalid layer gate value for format={fmt}, layer={layer}")
    return statuses, n_layers


def _build_checkpoint_argmax_layers(
    rules: pd.DataFrame,
    la_summary: pd.DataFrame,
    la_gate_by_family: Mapping[str, pd.DataFrame],
    family_by_model: Mapping[str, str],
) -> pd.DataFrame:
    """Report each maximum's layer and the gate status at it and adjacent layers."""
    m_maxima = rules.loc[
        rules["metric"].astype(str).eq("M_L")
        & rules["target_lang"].astype(str).isin(("vi", "en", "zh", "fr", "id"))
    ].copy()
    la_maxima = la_summary.loc[
        la_summary["metric"].astype(str).isin(("max_LA", "max_LA_all_layer_sensitivity"))
        & la_summary["target_lang"].astype(str).isin(("en", "zh", "fr", "id"))
    ].copy()
    maxima = pd.concat([m_maxima, la_maxima], ignore_index=True, sort=False)
    output: list[dict[str, Any]] = []
    statuses_by_family_format: dict[tuple[str, str], tuple[dict[int, str], int]] = {}
    for row in maxima.to_dict(orient="records"):
        model = str(row["model"])
        family = family_by_model.get(model)
        if family is None or family not in la_gate_by_family:
            raise AssertionError(f"No family gate rows resolve model={model}")
        fmt = str(row["format"])
        status_key = (family, fmt)
        if status_key not in statuses_by_family_format:
            statuses_by_family_format[status_key] = _checkpoint_layer_gate_statuses(
                la_gate_by_family[family], fmt=fmt
            )
        layer_status, n_layers = statuses_by_family_format[status_key]
        raw_layer = row.get("layer")
        argmax_layer = None if raw_layer is None or pd.isna(raw_layer) else int(raw_layer)

        def status_at(layer: int | None) -> str:
            if layer is None:
                return "not_evaluable"
            if layer < 1 or layer > n_layers:
                return "outside_layer_range"
            try:
                return layer_status[layer]
            except KeyError as exc:
                raise AssertionError(
                    f"Missing gate status for {family}/{fmt}/layer={layer}"
                ) from exc

        previous_layer = None if argmax_layer is None else argmax_layer - 1
        next_layer = None if argmax_layer is None else argmax_layer + 1
        output.append(
            {
                "family": family,
                "model": model,
                "format": fmt,
                "max_quantity": str(row["metric"]),
                "target_lang": str(row["target_lang"]),
                "item_set": row.get("item_set"),
                "metric_variant": row.get("metric_variant"),
                "max_value": row.get("value"),
                "argmax_layer": argmax_layer,
                "argmax_gate_status": status_at(argmax_layer),
                "previous_layer": previous_layer,
                "previous_gate_status": status_at(previous_layer),
                "next_layer": next_layer,
                "next_gate_status": status_at(next_layer),
            }
        )
    result = pd.DataFrame.from_records(output)
    if not result.empty:
        result = result.sort_values(
            ["family", "format", "max_quantity", "target_lang", "item_set", "metric_variant"],
            kind="stable",
            na_position="first",
        ).reset_index(drop=True)
    return result


def _checkpoint_la_sensitivity_table(la_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the frozen all-defined-layer LA sensitivity and its rule-change flag."""
    sensitivity = la_summary.loc[
        la_summary["metric"].astype(str).eq("max_LA_all_layer_sensitivity")
        & la_summary["target_lang"].astype(str).isin(("en", "zh", "fr", "id"))
    ].copy()
    if not sensitivity.empty and "would_change_rule_outcome" not in sensitivity.columns:
        raise AssertionError("All-layer LA sensitivity is missing its rule-change flag")
    return sensitivity


def _assert_checkpoint_reference_facts(
    cfg: Mapping[str, Any],
    repo: Path,
    *,
    la_summary: pd.DataFrame,
    la_by_family: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Assert every T13b Checkpoint-H reference count and the frozen A6 gate."""
    main_ids, cognate_ids = _primary_scope_ids(repo, dict(cfg))
    concepts = _read_jsonl_rows(repo / str(cfg["paths"]["concepts_jsonl"]))
    by_id = {str(row["concept_id"]): row for row in concepts}
    strata = {name: 0 for name in ("sino", "nonsino", "ambiguous", "other_loan")}
    for concept_id in main_ids:
        stratum = str(by_id[concept_id].get("stratum", ""))
        if stratum in strata:
            strata[stratum] += 1
    expected_strata = {"sino": 481, "nonsino": 706, "ambiguous": 347, "other_loan": 1}
    if len(main_ids) != 1_535 or strata != expected_strata:
        raise AssertionError(f"Main/stratum reference mismatch: main={len(main_ids)} strata={strata}")
    if len(cognate_ids) != 729 or id_list_sha256(cognate_ids) != "f1e7a6b5a0c197b1769215224503ee33a8f2e216943ae0d8088d55219042a58f":
        raise AssertionError("cf729 count/hash reference mismatch")

    m1_total_expected = {"gemma": 186, "qwen": 166, "llama": 162}
    m1_main_expected = {"gemma": 108, "qwen": 96, "llama": 96}
    m1_counts: dict[str, dict[str, int]] = {}
    for family in ("gemma", "qwen", "llama"):
        eligible = {
            str(row["concept_id"])
            for row in concepts
            if str(row.get(f"m1_eligible_{family}", "false")).lower() == "true"
        }
        total = len(eligible)
        main = len(eligible & main_ids)
        m1_counts[family] = {"total": total, "main": main}
        if total != m1_total_expected[family] or main != m1_main_expected[family]:
            raise AssertionError(
                f"M1 eligibility reference mismatch for {family}: total={total} main={main}"
            )

    generation_counts: dict[str, int] = {}
    for family in ("gemma", "qwen", "llama"):
        count = 0
        paths = list(_primary_paths(repo, cfg, family=family))
        for parquet_path in paths:
            generation_path = parquet_path.with_suffix(".generations.jsonl")
            if not generation_path.is_file():
                raise FileNotFoundError(f"Missing generation rows for {parquet_path.name}")
            with generation_path.open("r", encoding="utf-8") as stream:
                count += sum(1 for line in stream if line.strip())
        generation_counts[family] = count
        if count != 3_346:
            raise AssertionError(f"Generation reference mismatch for {family}: {count}, expected 3,346")

    model_specs = study_models(dict(cfg))
    expected_queries = {"repetition": 1_535, "translation": 1_524, "cloze": 34}
    la_query_counts: dict[str, dict[str, int]] = {}
    for family, spec in model_specs.items():
        model = str(spec["hf_id"])
        la_query_counts[family] = {}
        for fmt, expected in expected_queries.items():
            rows = la_summary.loc[
                la_summary["model"].astype(str).eq(model)
                & la_summary["format"].astype(str).eq(fmt)
                & la_summary["metric"].astype(str).eq("max_LA")
            ]
            observed_values = sorted(set(pd.to_numeric(rows["n_concepts"], errors="coerce").dropna().astype(int)))
            if observed_values != [expected]:
                raise AssertionError(
                    f"LA query reference mismatch for {family}/{fmt}: {observed_values}, expected {expected}"
                )
            la_query_counts[family][fmt] = expected

    gate_path = repo / "reports/E5_gate.csv"
    gate_reference = pd.read_csv(gate_path)
    gate_comparisons: dict[str, dict[str, list[int]]] = {}
    family_by_model = {str(spec["hf_id"]): family for family, spec in model_specs.items()}
    for family, raw in la_by_family.items():
        model = next(model_id for model_id, name in family_by_model.items() if name == family)
        raw_gate = raw.loc[raw["metric"].astype(str).eq("gate_pass")]
        for fmt in ("repetition", "translation"):
            ours = raw_gate.loc[
                raw_gate["model"].astype(str).eq(model)
                & raw_gate["format"].astype(str).eq(fmt)
            ].sort_values("layer", kind="stable")
            expected = gate_reference.loc[
                gate_reference["family"].astype(str).eq(family)
                & gate_reference["fit_format"].astype(str).eq(fmt)
            ].sort_values("layer", kind="stable")
            ours_pairs = [
                (int(row.layer), bool(float(row.value) >= 0.5))
                for row in ours.itertuples(index=False)
            ]
            expected_pairs = [
                (int(row.layer), str(row.gate_pass).strip().lower() == "true")
                for row in expected.itertuples(index=False)
            ]
            if ours_pairs != expected_pairs:
                raise AssertionError(f"A6 gate mismatch for {family}/{fmt}: {ours_pairs} != {expected_pairs}")
            gate_comparisons.setdefault(family, {})[fmt] = [
                layer for layer, passed in ours_pairs if not passed
            ]

    return {
        "main": len(main_ids),
        "strata": strata,
        "cf729": {"count": len(cognate_ids), "sha256": id_list_sha256(cognate_ids)},
        "m1_eligible": m1_counts,
        "generations": generation_counts,
        "la_queries": la_query_counts,
        "gate_failed_layers": gate_comparisons,
        "matches": True,
    }


def _count_opaque_jsonl_rows(path: Path) -> int:
    """Count JSONL records by newline bytes without decoding generation text."""
    newline_count = 0
    has_bytes = False
    last_byte = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            has_bytes = True
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return newline_count + int(has_bytes and last_byte != b"\n")


def _check_inputs_row(
    check: str,
    scope: str,
    observed: object,
    expected: object,
    matches: bool,
) -> dict[str, str]:
    """Normalize one whole-run input-check table entry."""
    return {
        "check": check,
        "scope": scope,
        "observed": str(observed),
        "expected": str(expected),
        "status": "MATCH" if matches else "MISMATCH",
    }


def _reference_concept_metadata(path: Path) -> tuple[set[str], dict[str, int], dict[str, dict[str, int]]]:
    """Read only concept-id and reference-count fields needed by check_inputs."""
    main_ids: set[str] = set()
    strata = {name: 0 for name in ("sino", "nonsino", "ambiguous", "other_loan")}
    eligible_ids: dict[str, set[str]] = {family: set() for family in TRIAL_FAMILIES}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            concept_id = str(row["concept_id"])
            is_main = (
                str(row.get("split", "")) == "test"
                and str(row.get("m1_extension", "false")).lower() != "true"
            )
            if is_main:
                main_ids.add(concept_id)
                stratum = str(row.get("stratum", ""))
                if stratum in strata:
                    strata[stratum] += 1
            for family in TRIAL_FAMILIES:
                if str(row.get(f"m1_eligible_{family}", "false")).lower() == "true":
                    eligible_ids[family].add(concept_id)
    eligibility = {
        family: {
            "total": len(ids),
            "main": len(ids & main_ids),
        }
        for family, ids in eligible_ids.items()
    }
    return main_ids, strata, eligibility


def _whole_run_input_checks(cfg: Mapping[str, Any], repo: Path) -> list[dict[str, str]]:
    """Resolve all external inputs for core, competence, C1 and Checkpoint H."""
    rows: list[dict[str, str]] = []
    expected_unit_counts = {"repetition": 7, "translation": 7, "cloze": 1}
    expected_la_queries = {"repetition": 1_535, "translation": 1_524, "cloze": 34}
    expected_e2_rows = {"gemma": 66_325, "qwen": 62_715, "llama": 47_995}
    expected_m1 = {
        "gemma": {"total": 186, "main": 108},
        "qwen": {"total": 166, "main": 96},
        "llama": {"total": 162, "main": 96},
    }
    expected_strata = {"sino": 481, "nonsino": 706, "ambiguous": 347, "other_loan": 1}
    expected_cf729_sha256 = "f1e7a6b5a0c197b1769215224503ee33a8f2e216943ae0d8088d55219042a58f"
    families = [str(value) for value in cfg["pd"]["families"]]

    concept_path = repo / str(cfg["paths"]["concepts_jsonl"])
    derived_cognate_path = repo / "derived/cognate_flags.jsonl"
    try:
        main_ids, strata, m1_counts = _reference_concept_metadata(concept_path)
        concepts_ok = True
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        main_ids, strata, m1_counts = set(), {}, {}
        concepts_ok = False
        concept_error = f"{type(exc).__name__}: {exc}"
    rows.append(
        _check_inputs_row(
            "concept metadata input",
            "all",
            "readable" if concepts_ok else concept_error,
            str(cfg["paths"]["concepts_jsonl"]),
            concepts_ok,
        )
    )
    rows.append(
        _check_inputs_row(
            "main test count",
            "all",
            len(main_ids),
            1_535,
            concepts_ok and len(main_ids) == 1_535,
        )
    )
    rows.append(
        _check_inputs_row(
            "main strata",
            "all",
            strata,
            expected_strata,
            concepts_ok and strata == expected_strata,
        )
    )
    for family in TRIAL_FAMILIES:
        observed = m1_counts.get(family, {})
        rows.append(
            _check_inputs_row(
                "M1 eligibility",
                family,
                observed,
                expected_m1[family],
                concepts_ok and observed == expected_m1[family],
            )
        )

    try:
        cognate_ids: list[str] = []
        with derived_cognate_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if str(row.get("cognate_free", "false")).lower() == "true":
                    cognate_ids.append(str(row["concept_id"]))
        cognate_digest = id_list_sha256(cognate_ids)
        cognate_ok = len(cognate_ids) == 729 and cognate_digest == expected_cf729_sha256
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        cognate_ids = []
        cognate_digest = f"unavailable ({type(exc).__name__})"
        cognate_ok = False
    rows.append(
        _check_inputs_row(
            "cf729 id set",
            "all",
            f"count={len(cognate_ids)}; sha256={cognate_digest}",
            f"count=729; sha256={expected_cf729_sha256}",
            cognate_ok,
        )
    )

    primary_records: list[dict[str, Any]] = []
    try:
        primary_records = _primary_prompt_records(cfg, repo)
        prompt_plan_ok = len(primary_records) == 3_346
        plan_error = ""
    except (OSError, KeyError, ValueError, TypeError) as exc:
        prompt_plan_ok = False
        plan_error = f" ({type(exc).__name__}: {exc})"
    rows.append(
        _check_inputs_row(
            "primary P-L prompt plan",
            "all",
            f"records={len(primary_records)}{plan_error}",
            "records=3346; P-L primary files resolved and sharded",
            prompt_plan_ok,
        )
    )
    if primary_records:
        for family in families:
            try:
                rows.extend(
                    _check_primary_pl_inputs(
                        primary_records,
                        family,
                        repo / str(cfg["pl"]["results_dir"]),
                        main_ids=main_ids,
                        expected_unit_counts=expected_unit_counts,
                        expected_query_counts=expected_la_queries,
                    )
                )
            except (OSError, KeyError, ValueError, TypeError) as exc:
                rows.append(
                    _check_inputs_row(
                        "P-L/LA resolver",
                        family,
                        f"{type(exc).__name__}: {exc}",
                        "planned P-L unit files and exact main Vietnamese query indexes",
                        False,
                    )
                )
        for fmt in ("repetition", "translation"):
            subset_ids = {
                str(record["concept_id"])
                for record in primary_records
                if str(record.get("format")) == fmt
                and str(record["concept_id"]) in set(cognate_ids)
                and str(record["concept_id"]) in main_ids
                and str(record.get("m1_extension", "false")).lower() != "true"
            }
            expected_count = 729 if fmt == "repetition" else 721
            rows.append(
                _check_inputs_row(
                    "cf729 prompt coverage",
                    fmt,
                    len(subset_ids),
                    expected_count,
                    cognate_ok and len(subset_ids) == expected_count,
                )
            )
    else:
        for family in families:
            rows.append(
                _check_inputs_row(
                    "P-L/LA resolver", family, "unavailable", "planned P-L unit files and query indexes", False
                )
            )

    for family in families:
        generation_rows = 0
        generation_missing: list[str] = []
        for _fmt, parquet_path in _planned_primary_units(
            primary_records, family, repo / str(cfg["pl"]["results_dir"])
        ) if primary_records else []:
            generation_path = parquet_path.with_suffix(".generations.jsonl")
            if not generation_path.is_file():
                generation_missing.append(str(generation_path))
                continue
            try:
                generation_rows += _count_opaque_jsonl_rows(generation_path)
            except OSError:
                generation_missing.append(str(generation_path))
        rows.append(
            _check_inputs_row(
                "generation rows",
                family,
                f"rows={generation_rows}; missing={generation_missing or 'none'}",
                "rows=3346; all planned generation files present (line-count only)",
                bool(primary_records) and generation_rows == 3_346 and not generation_missing,
            )
        )

    pd_root = repo / str(cfg["pd"]["results_dir"])
    for family in families:
        required_pd: list[Path] = []
        for fmt in ("repetition", "translation"):
            stem = f"family={family}__format={fmt}__pass=pd__source=matched"
            required_pd.extend((pd_root / f"{stem}.parquet", pd_root / f"{stem}.resid.npy"))
        required_pd.extend(
            (
                pd_root / f"family={family}__pass=pd__source=flores.mean.npy",
                pd_root / f"family={family}__pass=pd__source=flores.last.npy",
            )
        )
        missing_pd = [str(path) for path in required_pd if not path.is_file()]
        rows.append(
            _check_inputs_row(
                "P-D matched/FLORES arrays",
                family,
                "present" if not missing_pd else "missing=" + ", ".join(missing_pd),
                "matched repetition/translation parquet + residuals; FLORES mean/last arrays",
                not missing_pd,
            )
        )

    e2_expected_formats = ("repetition", "translation", "cloze")
    primary_e2_root = repo / "results/analysis/m1/primary"
    if not (primary_e2_root / "e2_job_summary.json").is_file():
        e2_summary_missing = True
    else:
        e2_summary_missing = False
    rows.append(
        _check_inputs_row(
            "E2 primary summary",
            "all",
            "missing" if e2_summary_missing else "present",
            "results/analysis/m1/primary/e2_job_summary.json",
            not e2_summary_missing,
        )
    )
    for family in families:
        observed_by_format: dict[str, int | str] = {}
        sidecar_mismatches: list[str] = []
        missing_e2: list[str] = []
        total_rows = 0
        for fmt in e2_expected_formats:
            filename = f"family={family}__format={fmt}__pass=m1__source=primary.parquet"
            parquet_path = primary_e2_root / filename
            sidecar_path = parquet_path.with_name(parquet_path.stem + ".provenance.json")
            if not parquet_path.is_file() or not sidecar_path.is_file():
                missing_e2.extend(str(path) for path in (parquet_path, sidecar_path) if not path.is_file())
                observed_by_format[fmt] = "missing"
                continue
            try:
                row_count = int(pq.ParquetFile(parquet_path).metadata.num_rows)
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if (
                    int(sidecar.get("row_count", -1)) != row_count
                    or int(sidecar.get("expected_row_count", -1)) != row_count
                ):
                    sidecar_mismatches.append(fmt)
                observed_by_format[fmt] = row_count
                total_rows += row_count
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                observed_by_format[fmt] = f"unreadable:{type(exc).__name__}"
                sidecar_mismatches.append(fmt)
        rows.append(
            _check_inputs_row(
                "E2 primary rows",
                family,
                f"total={total_rows}; by_format={observed_by_format}; missing={missing_e2 or 'none'}; "
                f"sidecar_row_mismatch={sidecar_mismatches or 'none'}",
                f"total={expected_e2_rows[family]}; all 3 parquet footers/sidecars agree",
                total_rows == expected_e2_rows[family]
                and not missing_e2
                and not sidecar_mismatches,
            )
        )

    trial_e2_root = repo / "results/analysis/m1/trial"
    trial_missing: list[str] = []
    trial_mismatches: list[str] = []
    for family in families:
        for fmt in ("repetition", "translation"):
            filename = f"family={family}__format={fmt}__pass=m1__source=trial.parquet"
            parquet_path = trial_e2_root / filename
            sidecar_path = parquet_path.with_name(parquet_path.stem + ".provenance.json")
            if not parquet_path.is_file() or not sidecar_path.is_file():
                trial_missing.extend(str(path) for path in (parquet_path, sidecar_path) if not path.is_file())
                continue
            try:
                row_count = int(pq.ParquetFile(parquet_path).metadata.num_rows)
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if row_count != int(sidecar.get("row_count", -1)) or row_count != int(
                    sidecar.get("expected_row_count", -2)
                ):
                    trial_mismatches.append(f"{family}/{fmt}")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                trial_mismatches.append(f"{family}/{fmt}")
    trial_summary_missing = not (trial_e2_root / "e2_job_summary.json").is_file()
    rows.append(
        _check_inputs_row(
            "E2 trial inputs",
            "all",
            f"missing={trial_missing or 'none'}; row_mismatches={trial_mismatches or 'none'}; "
            f"summary={'missing' if trial_summary_missing else 'present'}",
            "all 6 parquet footers match their sidecar row counts; summary present",
            not trial_missing and not trial_mismatches and not trial_summary_missing,
        )
    )

    gate_paths = (repo / "reports/E5_gate.csv", repo / "reports/E5_gate.md")
    missing_gate = [str(path) for path in gate_paths if not path.is_file()]
    rows.append(
        _check_inputs_row(
            "E5 gate reference",
            "all",
            "present" if not missing_gate else "missing=" + ", ".join(missing_gate),
            "reports/E5_gate.csv and reports/E5_gate.md present",
            not missing_gate,
        )
    )
    return rows


def _input_check_markdown(rows: list[Mapping[str, str]]) -> str:
    """Render the whole-run input verification as a complete Markdown table."""
    header = "| Check | Scope | Observed | Expected | Status |\n|---|---|---|---|---|"
    body = []
    for row in rows:
        cells = [row[key].replace("|", "\\|").replace("\n", " ") for key in ("check", "scope", "observed", "expected", "status")]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join((header, *body))


def _collect_e2_summary(
    repo: Path, *, families: list[str], sources: tuple[str, ...] = ("trial", "primary")
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate E2 outputs/provenance and summarize maximum mean M1 probabilities."""
    source_summaries: dict[str, dict[str, Any]] = {}
    source_results: dict[str, dict[str, Any]] = {}
    units: list[dict[str, Any]] = []
    maxima: list[dict[str, Any]] = []
    for source in sources:
        summary_path = repo / "results/analysis/m1" / source / "e2_job_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing E2 job summary: {summary_path.relative_to(repo)}")
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_summaries[source] = source_summary
        e2_executed_commit = source_summary.get("git_commit")
        if not e2_executed_commit or source_summary.get("git_dirty"):
            raise AssertionError(f"E2 {source} run does not map to a clean executed commit")
        run_result = source_summary.get("result", {})
        source_result = (
            run_result.get("primary")
            if source == "primary"
            else run_result.get("trial_check", run_result)
        )
        if source_result is None:
            raise AssertionError(f"E2 summary lacks {source} output")
        source_results[source] = source_result
        gate_a_max = _optional_number(source_result.get("gate_a_max_abs_delta"))
        if source == "trial":
            if gate_a_max is None or gate_a_max > 1.0e-2:
                raise AssertionError(f"E2 trial Gate A maximum is invalid: {gate_a_max}")
            control_by_family = source_result.get("gate_a_control_by_family", {})
            for family in families:
                control = control_by_family.get(family, {})
                median = _optional_number(control.get("median_abs_delta"))
                if median is None or median < 0.1:
                    raise AssertionError(
                        f"E2 trial Gate A-control is not discriminative for {family}: {median}"
                    )
        expected_units = {
            (str(row["family"]), str(row["format"])): row
            for row in source_result.get("units", [])
        }
        formats = ("repetition", "translation") if source == "trial" else ("repetition", "translation", "cloze")
        for family in families:
            for fmt in formats:
                filename = f"family={family}__format={fmt}__pass=m1__source={source}.parquet"
                parquet_path = repo / "results/analysis/m1" / source / filename
                sidecar_path = parquet_path.with_name(parquet_path.stem + ".provenance.json")
                if not parquet_path.is_file() or not sidecar_path.is_file():
                    raise FileNotFoundError(f"E2 output or provenance is missing for {source}/{family}/{fmt}")
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if sidecar.get("git_commit") != e2_executed_commit:
                    raise AssertionError(f"E2 sidecar commit differs from its run summary for {source}/{family}/{fmt}")
                if sidecar.get("code_tree_sha256") != source_summary.get("code_tree_sha256"):
                    raise AssertionError(f"E2 sidecar code-tree mismatch for {source}/{family}/{fmt}")
                if sidecar.get("git_dirty"):
                    raise AssertionError(f"E2 sidecar records a dirty code tree for {source}/{family}/{fmt}")
                frame = pq.read_table(
                    parquet_path,
                    columns=["model", "model_revision", "format", "concept_id", "target_lang", "target_variant", "layer", "metric", "value"],
                ).to_pandas()
                frame = frame.loc[frame["metric"].astype(str).eq("prefix_sum_probability")].copy()
                values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=np.float64)
                row_count = len(frame)
                expected_rows = int(sidecar.get("expected_row_count", -1))
                if row_count != int(sidecar.get("row_count", -2)) or row_count != expected_rows:
                    raise AssertionError(
                        f"E2 row mismatch for {source}/{family}/{fmt}: {row_count}/{expected_rows}"
                    )
                if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                    raise AssertionError(f"E2 probabilities are invalid for {source}/{family}/{fmt}")
                unit_summary = expected_units.get((family, fmt))
                if unit_summary is None or int(unit_summary.get("rows", -1)) != row_count or int(unit_summary.get("expected_rows", -2)) != row_count:
                    raise AssertionError(f"E2 job summary row counts differ for {source}/{family}/{fmt}")
                family_gate_a = source_result.get("gate_a_by_family", {}).get(family, {})
                family_gate_b = source_result.get("gate_b_by_family", {}).get(family, {})
                units.append(
                    {
                        "source": source,
                        "family": family,
                        "format": fmt,
                        "rows": row_count,
                        "expected_rows": expected_rows,
                        "finite": bool(np.isfinite(values).all()),
                        "in_0_1": bool(np.all((values >= 0.0) & (values <= 1.0))),
                        "gate_a_max_abs_delta": family_gate_a.get("max_abs_delta"),
                        "gate_b_max_abs_delta": family_gate_b.get("max_abs_delta"),
                        "code_tree_sha256": sidecar.get("code_tree_sha256"),
                    }
                )
                if source == "primary":
                    frame = frame.loc[pd.to_numeric(frame["layer"], errors="raise") >= 1].copy()
                    layer_means = (
                        frame.groupby(["model", "format", "target_lang", "layer"], sort=True)["value"]
                        .mean()
                        .reset_index(name="concept_mean_probability")
                    )
                    for (model, out_fmt, language), group in layer_means.groupby(
                        ["model", "format", "target_lang"], sort=True
                    ):
                        group = group.sort_values("layer", kind="stable")
                        selected = group.iloc[int(np.argmax(group["concept_mean_probability"].to_numpy()))]
                        maxima.append(
                            {
                                "model": model,
                                "family": family,
                                "format": out_fmt,
                                "readout": language,
                                "max_mean_P_M1": float(selected["concept_mean_probability"]),
                                "layer": int(selected["layer"]),
                                "rows": row_count,
                            }
                        )
    return pd.DataFrame.from_records(maxima), {
        "sources": source_summaries,
        "units": units,
        "gate_a_max_abs_delta": {
            source: source_results[source].get("gate_a_max_abs_delta") for source in sources
        },
    }


def _load_family_parquet_tables(
    repo: Path, families: list[str], filename: str, *, source: str = "primary",
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Load the compact one-family summary tables produced by the frozen core stage."""
    parts = []
    for family in families:
        path = _family_root(repo, source, family) / filename
        if not path.is_file():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing frozen family output {path.relative_to(repo)}")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_holm_summary(holm_rows: pd.DataFrame, *, item_set: str = "cf729") -> pd.DataFrame:
    """Apply Holm to exactly 15 W_L hypotheses within each format."""
    if holm_rows.empty:
        raise AssertionError("Checkpoint H requires raw W_L bootstrap rows")
    index = ["model", "format", "target_lang", "item_set", "metric_variant"]
    w = holm_rows.loc[holm_rows["metric"].astype(str).eq("W_L")].copy()
    p = holm_rows.loc[holm_rows["metric"].astype(str).eq("p_value")].copy()
    merged = w[index + ["value", "ci_low", "ci_high", "valid_draws", "n_concepts"]].merge(
        p[index + ["value"]], on=index, how="outer", validate="one_to_one", suffixes=("_W_L", "_p")
    )
    merged = merged.rename(columns={"value_W_L": "W_L", "value_p": "p_raw"})
    merged = merged.loc[
        merged["item_set"].astype(str).eq(item_set)
        & merged["metric_variant"].astype(str).eq("s")
    ].copy()
    result = []
    for fmt, group in merged.groupby("format", sort=True):
        group = group.sort_values(["model", "target_lang"], kind="stable").copy()
        if len(group) != 15:
            raise AssertionError(f"Holm family for {fmt} has {len(group)} tests, expected 15")
        group["p_holm"] = holm_adjust(group["p_raw"].to_numpy(dtype=np.float64))
        group["reject_0_05"] = group["p_holm"] <= 0.05
        result.append(group)
    return pd.concat(result, ignore_index=True)


def _bootstrap_curve_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    item_set: str,
    seed: int,
    analysis_config: Mapping[str, Any],
    target_language: bool,
) -> pd.DataFrame:
    """Build pointwise concept-bootstrap CIs for rise, LA, or D_en curves."""
    selected = frame.loc[frame["metric"].astype(str).eq(metric)].copy()
    if selected.empty:
        return pd.DataFrame()
    group_keys = ["model", "format"] + (["target_lang"] if target_language else [])
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(group_keys, sort=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_keys, keys, strict=True))
        concept_ids = sorted(group["concept_id"].astype(str).unique())
        layers = sorted(map(int, group["layer"].unique()))
        pivot = (
            group.assign(concept_id=group["concept_id"].astype(str))
            .pivot(index="concept_id", columns="layer", values="value")
            .reindex(index=concept_ids, columns=layers)
        )
        values = pivot.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"Curve bootstrap input has missing values for {identity}")
        indices = concept_indices(
            concept_ids,
            resamples=int(analysis_config["bootstrap_resamples"]),
            seed=seed,
            key=bootstrap_key(
                item_set,
                str(identity["model"]),
                str(identity["format"]),
                template=str(analysis_config["bootstrap_key_template"]),
            ),
        )
        samples = bootstrap_means(values, indices)
        point = values.mean(axis=0, dtype=np.float64)
        lows = np.percentile(samples, 2.5, axis=0)
        highs = np.percentile(samples, 97.5, axis=0)
        for position, layer in enumerate(layers):
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "layer": layer,
                    "value": float(point[position]),
                    "ci_low": float(lows[position]),
                    "ci_high": float(highs[position]),
                    "valid_draws": int(np.isfinite(samples[:, position]).sum()),
                    "n_concepts": len(concept_ids),
                    "bootstrap_resamples": int(len(indices)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _markdown_table(frame: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    """Render a DataFrame as CommonMark without adding a plotting dependency."""
    selected = frame if columns is None else frame[columns]
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False, name=None):
        values = ["" if value is None or pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _write_checkpoint_figures(
    repo: Path,
    *,
    families: list[str],
    model_by_family: Mapping[str, str],
    rise_curves: pd.DataFrame,
    la_curves: pd.DataFrame,
    la_by_family: Mapping[str, pd.DataFrame],
) -> list[str]:
    """Save per-model/format five-way rise and LA curves under reports/figures."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_root = repo / "reports/figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    model_to_family = {model: family for family, model in model_by_family.items()}
    created: list[str] = []
    for model, family in sorted(model_to_family.items()):
        for fmt in ("repetition", "translation"):
            curve = rise_curves.loc[
                rise_curves["model"].astype(str).eq(model)
                & rise_curves["format"].astype(str).eq(fmt)
            ]
            if curve.empty:
                raise AssertionError(f"Missing cf729 rise curves for {family}/{fmt}")
            fig, ax = plt.subplots(figsize=(8, 5))
            for language, group in curve.groupby("target_lang", sort=True):
                group = group.sort_values("layer", kind="stable")
                x = group["layer"].to_numpy(dtype=np.int64)
                y = group["value"].to_numpy(dtype=np.float64)
                low = group["ci_low"].to_numpy(dtype=np.float64)
                high = group["ci_high"].to_numpy(dtype=np.float64)
                ax.plot(x, y, label=str(language))
                ax.fill_between(x, low, high, alpha=0.16)
            ax.set(title=f"{family} — {fmt}: cf729 rise", xlabel="Block output layer", ylabel="Rise (nats)")
            ax.legend(title="Readout")
            fig.tight_layout()
            path = figure_root / f"E3_{family}_{fmt}_rise.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path.relative_to(repo)))

        for fmt in ("repetition", "translation", "cloze"):
            curve = la_curves.loc[
                la_curves["model"].astype(str).eq(model)
                & la_curves["format"].astype(str).eq(fmt)
            ]
            raw = la_by_family[family]
            gate = raw.loc[
                raw["format"].astype(str).eq(fmt)
                & raw["metric"].astype(str).isin({"layer_pass", "undefined_denominator"})
                & raw["target_lang"].isna()
            ]
            failed = gate.loc[
                gate["metric"].astype(str).eq("layer_pass") & pd.to_numeric(gate["value"], errors="coerce").eq(0.0),
                "layer",
            ].dropna().astype(int).tolist()
            undefined = gate.loc[
                gate["metric"].astype(str).eq("undefined_denominator") & pd.to_numeric(gate["value"], errors="coerce").eq(1.0),
                "layer",
            ].dropna().astype(int).tolist()
            if curve.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for language, group in curve.groupby("target_lang", sort=True):
                group = group.sort_values("layer", kind="stable")
                x = group["layer"].to_numpy(dtype=np.int64)
                y = group["value"].to_numpy(dtype=np.float64)
                ax.plot(x, y, label=str(language))
                ax.fill_between(
                    x,
                    group["ci_low"].to_numpy(dtype=np.float64),
                    group["ci_high"].to_numpy(dtype=np.float64),
                    alpha=0.16,
                )
            for layer in failed:
                ax.axvline(layer, color="gray", alpha=0.22, linestyle="--")
                ax.scatter([layer], [0.0], marker="x", color="gray", s=28, label="gate failed" if layer == failed[0] else None)
            for layer in undefined:
                ax.axvline(layer, color="red", alpha=0.30, linestyle=":")
                ax.scatter([layer], [0.0], marker="x", color="red", s=32, label="undefined" if layer == undefined[0] else None)
            ax.set(title=f"{family} — {fmt}: language alignment", xlabel="Block output layer", ylabel="LA")
            ax.legend(title="Readout", ncol=2)
            fig.tight_layout()
            path = figure_root / f"E5_{family}_{fmt}_la.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path.relative_to(repo)))
    return created


def _build_a10_table(competence: pd.DataFrame) -> pd.DataFrame:
    """Summarize A10 class counts/rates, shared-source flags, and competent sizes."""
    if competence.empty:
        raise AssertionError("Checkpoint H requires collected A10 generations")
    rows = []
    for (model, fmt), group in competence.groupby(["model", "format"], sort=True):
        counts = group["competence_class"].astype(str).value_counts().to_dict()
        total = int(len(group))
        row: dict[str, Any] = {
            "model": model,
            "format": fmt,
            "denominator": total,
            "shared_ru_source_flagged": int(group["shared_ru_source"].astype(bool).sum()),
            "competent_subset_n": int(group["competent"].astype(bool).sum()),
        }
        for label in ("VI", "EN", "OTHER-LANG", "DAMAGE"):
            count = int(counts.get(label, 0))
            row[f"{label}_n"] = count
            row[f"{label}_rate"] = count / total if total else float("nan")
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _format_point_ci(row: pd.Series | None) -> str:
    """Format a stored estimate with its percentile interval for factual reports."""
    if row is None:
        return "NA"
    value = _optional_number(row.get("value"))
    low = _optional_number(row.get("ci_low"))
    high = _optional_number(row.get("ci_high"))
    if value is None:
        return "NA"
    if low is None or high is None:
        return f"{value:.4f} [CI unavailable]"
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def _make_adjudication_draft(
    repo: Path,
    *,
    frozen_commit: str,
    executed_commit: str,
    verdicts: pd.DataFrame,
    landmarks_table: pd.DataFrame,
    la_summary: pd.DataFrame,
    rules: pd.DataFrame,
    c1: pd.DataFrame,
) -> tuple[Path, str, list[str], list[str]]:
    """Select a pre-written abstract version and fill only computed bracket fields."""
    versions = verdicts["verdict"].astype(str).tolist()
    if versions and all(value == "H1" for value in versions):
        version = "A"
    elif versions and all(value == "H2" for value in versions):
        version = "B"
    else:
        version = "C"
    abstract_path = repo / "docs/prereg/abstracts_prewritten.md"
    source_text = abstract_path.read_text(encoding="utf-8")
    marker = f"## Version {version} —"
    start = source_text.index(marker)
    next_start = source_text.find("\n## Version ", start + len(marker))
    abstract = source_text[start: next_start if next_start >= 0 else len(source_text)].strip()
    models = sorted(verdicts["model"].astype(str).unique())
    formats = ("repetition", "translation")
    sources: list[str] = []
    unresolved: list[str] = []

    def one_row(frame: pd.DataFrame, conditions: Mapping[str, str]) -> pd.Series | None:
        """Select one exact abstract source cell."""
        selected = frame
        for column, value in conditions.items():
            if column not in selected:
                return None
            selected = selected.loc[selected[column].astype(str).eq(value)]
        if len(selected) > 1:
            raise AssertionError(f"Abstract source is not unique for {dict(conditions)}")
        return None if selected.empty else selected.iloc[0]

    def estimate(row: pd.Series | None, scale: float = 1.0) -> str:
        """Format a point estimate and 95% percentile CI."""
        if row is None:
            return "NA"
        value = _optional_number(row.get("value"))
        if value is None:
            return "NA"
        low = _optional_number(row.get("ci_low"))
        high = _optional_number(row.get("ci_high"))
        if low is None or high is None:
            return f"{value * scale:.4f} [CI unavailable]"
        return f"{value * scale:.4f} [{low * scale:.4f}, {high * scale:.4f}]"

    def source_ref(filename: str, conditions: Mapping[str, str]) -> None:
        """Record an abstract field's exact identifying table keys."""
        keys = ", ".join(f"{key}={value}" for key, value in conditions.items())
        columns = conditions.get("field", "value, ci_low, ci_high")
        sources.append(f"reports/E2-E5.md → {filename}[{keys}] → {columns}")

    def scalar_entries(
        frame: pd.DataFrame,
        *,
        metric: str,
        filename: str,
        target_lang: str | None = None,
        item_set: str | None = None,
        metric_variant: str | None = None,
        contrast: str | None = None,
        normalized: bool = False,
    ) -> str:
        """Format one metric for every model × study format."""
        values = []
        for model in models:
            for fmt in formats:
                conditions = {"model": model, "format": fmt, "metric": metric}
                if target_lang is not None:
                    conditions["target_lang"] = target_lang
                if item_set is not None:
                    conditions["item_set"] = item_set
                if metric_variant is not None:
                    conditions["metric_variant"] = metric_variant
                if contrast is not None:
                    conditions["contrast"] = contrast
                row = one_row(frame, conditions)
                if normalized and row is not None and _optional_number(row.get("value")) is not None:
                    n_layers = _optional_number(row.get("n_layers"))
                    if n_layers is None or n_layers <= 0:
                        raise AssertionError(f"Missing N for normalized landmark {conditions}")
                    text = estimate(row, 1.0 / n_layers)
                else:
                    text = estimate(row)
                values.append(f"{model}/{fmt}: {text}")
                source_ref(filename, conditions)
        return "; ".join(values)

    def language_entries(
        frame: pd.DataFrame,
        *,
        metric: str,
        languages: tuple[str, ...],
        filename: str,
        item_set: str | None = None,
        metric_variant: str | None = None,
    ) -> str:
        """Format each language's metric for every model × study format."""
        values = []
        for model in models:
            for fmt in formats:
                language_values = []
                for language in languages:
                    conditions = {
                        "model": model,
                        "format": fmt,
                        "metric": metric,
                        "target_lang": language,
                    }
                    if item_set is not None:
                        conditions["item_set"] = item_set
                    if metric_variant is not None:
                        conditions["metric_variant"] = metric_variant
                    row = one_row(frame, conditions)
                    language_values.append(f"{language}={estimate(row)}")
                    source_ref(filename, conditions)
                values.append(f"{model}/{fmt}: " + ", ".join(language_values))
        return "; ".join(values)

    if version == "C":
        all_inconclusive = all(value == "inconclusive" for value in versions)
        if all_inconclusive:
            decision = "support neither a high-resource pivot (H1) nor a language-neutral concept space (H2)"
        else:
            decision = "give different verdicts across models: " + "; ".join(
                f"{row.model}: {row.verdict}"
                for row in verdicts.sort_values("model", kind="stable").itertuples(index=False)
            )
        abstract = abstract.replace("[support neither a high-resource pivot (H1) nor a language-neutral concept space (H2) / give different verdicts across models: [model: verdict; …]]", f"[{decision}]")
        for verdict_row in verdicts.sort_values("model", kind="stable").itertuples(index=False):
            source_ref(
                "reports/checkpoint_h_rules.csv",
                {"model": str(verdict_row.model), "field": "verdict"},
            )
        alignment_text = scalar_entries(
            la_summary, metric="max_LA", filename="reports/E2-E5_e5.csv", target_lang="en"
        )
        ratio_text = scalar_entries(
            rules, metric="h1_ratio", filename="reports/E2-E5_rules.csv",
            target_lang="__H1_H2__", item_set="cf729", metric_variant="s",
        )
        switch_text = scalar_entries(
            landmarks_table, metric="L_cross", filename="reports/E2-E5_landmarks.csv",
            item_set="cf729", metric_variant="s", normalized=True,
        )
        abstract = abstract.replace("`max LA_en` = [ ]", f"`max LA_en` = {alignment_text}")
        abstract = abstract.replace("[ ] times as much", f"{ratio_text} times as much")
        abstract = abstract.replace("normalized depth [ ]", f"normalized depth {switch_text}")
        unresolved.extend([
            "The `[intermediate alignment with English / …]` wording is retained because OPS-14 defines no textual cutoff for `intermediate`.",
            "The `[no / a]` etymology alternative awaits E6.",
            "The `[within / outside]` steering-window alternative awaits E8.",
        ])
        abstract = abstract.replace("steering flips answers at depth [ ]", "steering flips answers at depth [pending E8]")
        abstract = abstract.replace("(`D_zh` = [ ])", "(`D_zh` = [pending E6])")
        sources.extend(["[pending E8] is owned by E8, outside T13b", "[pending E6] is owned by E6, outside T13b"])
    elif version == "A":
        alignment_text = scalar_entries(
            la_summary, metric="max_LA", filename="reports/E2-E5_e5.csv", target_lang="en"
        )
        en_text = scalar_entries(
            rules, metric="M_L", filename="reports/E2-E5_rules.csv", target_lang="en",
            item_set="cf729", metric_variant="s",
        )
        control_text = scalar_entries(
            rules, metric="control_mean", filename="reports/E2-E5_rules.csv",
            target_lang="__H1_H2__", item_set="cf729", metric_variant="s",
        )
        switch_text = scalar_entries(
            landmarks_table, metric="L_cross", filename="reports/E2-E5_landmarks.csv",
            item_set="cf729", metric_variant="s", normalized=True,
        )
        abstract = abstract.replace("(`max LA_en` = [ ])", f"(`max LA_en` = {alignment_text})")
        abstract = abstract.replace("([ ] vs [ ] nats)", f"({en_text} vs {control_text} nats)")
        abstract = abstract.replace("normalized depth [ ]", f"normalized depth {switch_text}")
        abstract = abstract.replace("depth [ ], [within / outside]", "depth [pending E8], [within / outside]")
        abstract = abstract.replace("(`D_zh` = [ ])", "(`D_zh` = [pending E6])")
        unresolved.extend(["The `[within / outside]` steering-window alternative awaits E8.", "The `[no / a]` etymology alternative awaits E6."])
        sources.extend(["[pending E8] is owned by E8, outside T13b", "[pending E6] is owned by E6, outside T13b"])
    else:
        alignment_text = language_entries(
            la_summary, metric="max_LA", languages=("en", "zh", "fr", "id"),
            filename="reports/E2-E5_e5.csv",
        )
        maxima_text = language_entries(
            rules, metric="M_L", languages=("vi", "en", "zh", "fr", "id"),
            filename="reports/E2-E5_rules.csv", item_set="cf729", metric_variant="s",
        )
        l_concept_text = scalar_entries(
            landmarks_table, metric="L_concept_10", filename="reports/E2-E5_landmarks.csv",
            item_set="cf729", metric_variant="s", normalized=True,
        )
        c1_text = scalar_entries(
            c1, metric="instrument_effect_10_blocks", filename="reports/E2-E5_c1.csv",
            item_set="m1_main", contrast="instrument_effect",
        )
        abstract = abstract.replace("(`max LA` = [ ])", f"(`max LA` = {alignment_text})")
        abstract = abstract.replace("([ ] nats)", f"({maxima_text} nats)")
        abstract = abstract.replace("normalized depth [ ]", f"normalized depth {l_concept_text}")
        abstract = abstract.replace("tuned lens ([ ])", f"tuned lens ({c1_text})")
        abstract = abstract.replace("(`D_zh` = [ ])", "(`D_zh` = [pending E6])")
        unresolved.extend([
            "The `[shrinks / persists]` alternative is retained: C1 reports L_concept timing effects, and OPS-14 does not define that metric as an English-dominance classification.",
            "The `[no / a]` etymology alternative awaits E6.",
        ])
        sources.append("[pending E6] is owned by E6, outside T13b")

    if "[ ]" in abstract or "[computed" in abstract:
        raise AssertionError("The selected abstract still has an unfilled computed bracket")

    draft_path = repo / "reports/adjudication_draft.md"
    body = [
        "DRAFT — not final. The human finalises this as reports/adjudication.md and commits it (AGENTS.md rule 30).",
        "",
        f"Frozen analysis commit: `{frozen_commit}`",
        f"Executed analysis commit: `{executed_commit}`",
        "",
        "## Per-model verdicts",
        "",
        _markdown_table(verdicts),
        "",
        f"## Selected abstract — Version {version}",
        "",
        abstract,
        "",
        "## Filled bracket source cells",
        "",
        *[f"- {item}" for item in sources],
        "",
        "## Unresolved bracket alternatives",
        "",
        *([f"- {item}" for item in unresolved] if unresolved else ["- none"]),
        "",
    ]
    draft_path.write_text("\n".join(body), encoding="utf-8")
    return draft_path, version, sources, unresolved


def _run_checkpoint_h_trial_smoke(
    cfg: Mapping[str, Any], repo: Path
) -> dict[str, Any]:
    """Run Checkpoint-H table calculations on stored trial outputs in isolation."""
    families = [str(value) for value in cfg["pd"]["families"]]
    rules = _load_family_parquet_tables(repo, families, "rule_quantities.parquet", source="trial")
    landmarks_table = _load_family_parquet_tables(repo, families, "landmarks.parquet", source="trial")
    holm_raw = _load_family_parquet_tables(repo, families, "holm_w_raw.parquet", source="trial")
    la_summary = _load_family_parquet_tables(repo, families, "la_summary.parquet", source="trial")
    c1 = _load_family_parquet_tables(repo, families, "c1.parquet", source="trial")
    competence = _load_family_parquet_tables(
        repo, families, "competence.parquet", source="trial", allow_missing=True
    )

    la_curve_parts: list[pd.DataFrame] = []
    den_curve_parts: list[pd.DataFrame] = []
    rise_curve_parts: list[pd.DataFrame] = []
    undefined_layers: dict[str, dict[str, list[int]]] = {}
    la_gate_by_family: dict[str, pd.DataFrame] = {}
    for family in families:
        family_root = _family_root(repo, "trial", family)
        la_raw = _filtered_parquet_rows(
            family_root / "la.parquet",
            columns=LONG_COLUMNS,
            predicate=lambda frame: frame.loc[
                frame["metric"].astype(str).isin(
                    {"LA", "D_en", "gate_pass", "layer_pass", "undefined_denominator"}
                )
            ],
        )
        la_by_family = la_raw.loc[
            la_raw["metric"].astype(str).isin({"gate_pass", "layer_pass", "undefined_denominator"})
        ].copy()
        la_gate_by_family[family] = la_by_family
        undefined_layers[family] = {}
        for fmt in ("repetition", "translation", "cloze"):
            selected_undefined = la_by_family.loc[
                la_by_family["format"].astype(str).eq(fmt)
                & la_by_family["metric"].astype(str).eq("undefined_denominator")
                & pd.to_numeric(la_by_family["value"], errors="coerce").eq(1.0)
            ]
            undefined_layers[family][fmt] = sorted(
                pd.to_numeric(selected_undefined["layer"], errors="raise").astype(int).unique().tolist()
            )
        la_curve_parts.append(
            _bootstrap_curve_summary(
                la_raw,
                metric="LA",
                item_set="main",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=True,
            )
        )
        den_curve_parts.append(
            _bootstrap_curve_summary(
                la_raw,
                metric="D_en",
                item_set="main",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=False,
            )
        )
        rise_raw = _filtered_parquet_rows(
            family_root / "rise.parquet",
            columns=LONG_COLUMNS,
            predicate=lambda frame: frame.loc[
                frame["metric"].astype(str).eq("rise_cf729")
            ],
        )
        rise_curve_parts.append(
            _bootstrap_curve_summary(
                rise_raw,
                metric="rise_cf729",
                item_set="cf729",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=True,
            )
        )

    rise_curves = pd.concat(rise_curve_parts, ignore_index=True)
    la_curves = pd.concat(la_curve_parts, ignore_index=True)
    den_curves = pd.concat(den_curve_parts, ignore_index=True)
    holm = _build_holm_summary(holm_raw, item_set="main")
    decision_table, verdict_table = _build_checkpoint_rules(rules, la_summary)
    if competence.empty:
        a10 = pd.DataFrame(
            columns=[
                "model", "format", "denominator", "shared_ru_source_flagged", "competent_subset_n",
                "VI_n", "VI_rate", "EN_n", "EN_rate", "OTHER-LANG_n", "OTHER-LANG_rate",
                "DAMAGE_n", "DAMAGE_rate",
            ]
        )
    else:
        a10 = _build_a10_table(competence)
    e2_summary, e2_info = _collect_e2_summary(repo, families=families, sources=("trial",))
    e5_rows = la_summary.loc[
        la_summary["metric"].astype(str).isin(
            {"max_LA", "max_LA_all_layer_sensitivity", "D_en", "D_en_window_mean", "gate_failed_layer", "undefined_denominator_layer"}
        )
    ].copy()
    la_sensitivity = _checkpoint_la_sensitivity_table(la_summary)
    family_by_model = {f"fixture/{family}": family for family in families}
    argmax_layers = _build_checkpoint_argmax_layers(
        rules, la_summary, la_gate_by_family, family_by_model
    )
    curve_report = pd.concat([rise_curves, la_curves, den_curves], ignore_index=True, sort=False)
    undefined_table = pd.DataFrame(
        [
            {
                "family": family,
                "format": fmt,
                "undefined_layers": ",".join(map(str, layers)) or "none",
            }
            for family, formats in undefined_layers.items()
            for fmt, layers in formats.items()
        ]
    )

    output_root = repo / "results/analysis/trial/checkpoint_h_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    e2_units = pd.DataFrame.from_records(e2_info["units"])
    outputs: dict[str, int] = {}
    output_tables: dict[str, pd.DataFrame] = {
        "E2-E5_e2.csv": e2_summary,
        "E2-E5_e2_units.csv": e2_units,
        "E2-E5_landmarks.csv": landmarks_table,
        "E2-E5_c1.csv": c1,
        "E2-E5_rules.csv": rules,
        "E2-E5_holm.csv": holm,
        "checkpoint_h_rules.csv": decision_table,
        "checkpoint_h_verdicts.csv": verdict_table,
        "E2-E5_a10.csv": a10,
        "E2-E5_la_curves.csv": curve_report,
        "E2-E5_e5.csv": e5_rows,
        "E2-E5_undefined_layers.csv": undefined_table,
        "checkpoint_h_la_sensitivity.csv": la_sensitivity,
        "checkpoint_h_argmax_layers.csv": argmax_layers,
    }
    if e2_summary.empty and not len(e2_summary.columns):
        output_tables["E2-E5_e2.csv"] = pd.DataFrame(
            columns=["model", "family", "format", "readout", "max_mean_P_M1", "layer", "rows"]
        )
    for filename, frame in output_tables.items():
        path = output_root / filename
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(repo))] = int(len(frame))

    summary_path = output_root / "checkpoint_h_trial_summary.json"
    summary_path.write_text(
        json.dumps({"source": "trial", "files_and_rows": outputs}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info("trial Checkpoint H smoke complete: files_and_rows=%s", json.dumps(outputs, sort_keys=True))
    return {"source": "trial", "stage": "checkpoint_h", "files_and_rows": outputs}


def _run_checkpoint_h_stage(
    cfg: Mapping[str, Any],
    repo: Path,
    *,
    analysis_identity: AnalysisCommitIdentity,
) -> dict[str, Any]:
    """Collect E2–E5 outputs, assert reference facts, and write the Checkpoint-H draft."""
    draft_final = repo / "reports/adjudication.md"
    if draft_final.exists():
        raise FileExistsError("reports/adjudication.md already exists; the agent never writes or overwrites it")
    frozen_commit = analysis_identity.frozen_analysis_commit
    executed_commit = analysis_identity.executed_commit
    families = [str(value) for value in cfg["pd"]["families"]]
    rules = _load_family_parquet_tables(repo, families, "rule_quantities.parquet")
    landmarks_table = _load_family_parquet_tables(repo, families, "landmarks.parquet")
    holm_raw = _load_family_parquet_tables(repo, families, "holm_w_raw.parquet")
    la_summary = _load_family_parquet_tables(repo, families, "la_summary.parquet")
    c1 = _load_family_parquet_tables(repo, families, "c1.parquet")
    competence = _load_family_parquet_tables(repo, families, "competence.parquet")

    la_by_family: dict[str, pd.DataFrame] = {}
    rise_curve_parts: list[pd.DataFrame] = []
    la_curve_parts: list[pd.DataFrame] = []
    den_curve_parts: list[pd.DataFrame] = []
    undefined_layers: dict[str, dict[str, list[int]]] = {}
    for family in families:
        family_root = _family_root(repo, "primary", family)
        la_path = family_root / "la.parquet"
        la_raw = _filtered_parquet_rows(
            la_path,
            columns=LONG_COLUMNS,
            predicate=lambda frame: frame.loc[
                frame["metric"].astype(str).isin(
                    {"LA", "D_en", "gate_pass", "layer_pass", "undefined_denominator"}
                )
            ],
        )
        la_by_family[family] = la_raw.loc[
            la_raw["metric"].astype(str).isin({"gate_pass", "layer_pass", "undefined_denominator"})
        ].copy()
        la_curve_parts.append(
            _bootstrap_curve_summary(
                la_raw,
                metric="LA",
                item_set="main",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=True,
            )
        )
        den_curve_parts.append(
            _bootstrap_curve_summary(
                la_raw,
                metric="D_en",
                item_set="main",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=False,
            )
        )
        rise_path = family_root / "rise.parquet"
        rise_raw = _filtered_parquet_rows(
            rise_path,
            columns=LONG_COLUMNS,
            predicate=lambda frame: frame.loc[
                frame["metric"].astype(str).eq("rise_cf729")
            ],
        )
        rise_curve_parts.append(
            _bootstrap_curve_summary(
                rise_raw,
                metric="rise_cf729",
                item_set="cf729",
                seed=int(cfg["seed"]),
                analysis_config=cfg["analysis"],
                target_language=True,
            )
        )
        undefined_layers[family] = {}
        for fmt in ("repetition", "translation", "cloze"):
            frame = la_raw.loc[
                la_raw["format"].astype(str).eq(fmt)
                & la_raw["metric"].astype(str).eq("undefined_denominator")
                & pd.to_numeric(la_raw["value"], errors="coerce").eq(1.0)
            ]
            undefined_layers[family][fmt] = sorted(pd.to_numeric(frame["layer"], errors="raise").astype(int).unique().tolist())
        del la_raw, rise_raw

    la_by_family = {family: frame for family, frame in la_by_family.items()}
    reference_facts = _assert_checkpoint_reference_facts(
        cfg, repo, la_summary=la_summary, la_by_family=la_by_family
    )
    rise_curves = pd.concat(rise_curve_parts, ignore_index=True)
    la_curves = pd.concat(la_curve_parts, ignore_index=True)
    den_curves = pd.concat(den_curve_parts, ignore_index=True)
    holm = _build_holm_summary(holm_raw)
    decision_table, verdict_table = _build_checkpoint_rules(rules, la_summary)
    a10 = _build_a10_table(competence)
    e2_summary, e2_info = _collect_e2_summary(repo, families=families)

    report_root = repo / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    e2_summary.to_csv(report_root / "E2-E5_e2.csv", index=False)
    e2_units = pd.DataFrame.from_records(e2_info["units"])
    e2_units.to_csv(report_root / "E2-E5_e2_units.csv", index=False)
    landmarks_table.to_csv(report_root / "E2-E5_landmarks.csv", index=False)
    c1.to_csv(report_root / "E2-E5_c1.csv", index=False)
    rules.to_csv(report_root / "E2-E5_rules.csv", index=False)
    holm.to_csv(report_root / "E2-E5_holm.csv", index=False)
    decision_table.to_csv(report_root / "checkpoint_h_rules.csv", index=False)
    verdict_table.to_csv(report_root / "checkpoint_h_verdicts.csv", index=False)
    a10.to_csv(report_root / "E2-E5_a10.csv", index=False)
    curve_report = pd.concat([rise_curves, la_curves, den_curves], ignore_index=True, sort=False)
    curve_report.to_csv(report_root / "E2-E5_la_curves.csv", index=False)

    e5_rows = la_summary.loc[
        la_summary["metric"].astype(str).isin(
            {"max_LA", "max_LA_all_layer_sensitivity", "D_en", "D_en_window_mean", "gate_failed_layer", "undefined_denominator_layer"}
        )
    ].copy()
    e5_rows.to_csv(report_root / "E2-E5_e5.csv", index=False)
    la_sensitivity = _checkpoint_la_sensitivity_table(la_summary)
    la_sensitivity.to_csv(report_root / "checkpoint_h_la_sensitivity.csv", index=False)
    family_by_model = {
        str(spec["hf_id"]): family for family, spec in study_models(dict(cfg)).items()
    }
    argmax_layers = _build_checkpoint_argmax_layers(
        rules, la_summary, la_by_family, family_by_model
    )
    argmax_layers.to_csv(report_root / "checkpoint_h_argmax_layers.csv", index=False)
    undefined_table = pd.DataFrame(
        [
            {
                "family": family,
                "format": fmt,
                "undefined_layers": ",".join(map(str, layers)) or "none",
            }
            for family, formats in undefined_layers.items()
            for fmt, layers in formats.items()
        ]
    )
    undefined_table.to_csv(report_root / "E2-E5_undefined_layers.csv", index=False)
    e3_rows = landmarks_table.loc[
        landmarks_table["item_set"].astype(str).isin({"cf729", "main"})
        & landmarks_table["metric_variant"].astype(str).isin({"s", "s1"})
    ].copy()
    e4_rows = rules.loc[
        rules["item_set"].astype(str).eq("cf729")
        & rules["metric_variant"].astype(str).isin({"s", "s1", "s_plus", "sequence_sum", "zh_released", "vi_nodiac", "ru_released"})
    ].copy()
    report_sections = [
        "# E2–E5 — Checkpoint H results",
        "",
        f"Frozen analysis commit: `{frozen_commit}`.",
        f"Executed analysis commit: `{executed_commit}`.",
        "All output tables are descriptive summaries of the frozen runs; each wide CSV is in `reports/`. E2 stores layers 0..N; its maxima exclude layer 0 and use block outputs 1..N.",
        "",
        "## Reference facts",
        "",
        "All required reference facts matched.",
        "",
        _markdown_table(pd.DataFrame([{"fact": "main", "observed": reference_facts["main"], "status": "MATCH"},
                                      {"fact": "strata", "observed": json.dumps(reference_facts["strata"], sort_keys=True), "status": "MATCH"},
                                      {"fact": "cf729 count/hash", "observed": json.dumps(reference_facts["cf729"], sort_keys=True), "status": "MATCH"},
                                      {"fact": "M1 eligibility", "observed": json.dumps(reference_facts["m1_eligible"], sort_keys=True), "status": "MATCH"},
                                      {"fact": "generations", "observed": json.dumps(reference_facts["generations"], sort_keys=True), "status": "MATCH"},
                                      {"fact": "LA queries", "observed": json.dumps(reference_facts["la_queries"], sort_keys=True), "status": "MATCH"},
                                      {"fact": "E5 gate lists", "observed": json.dumps(reference_facts["gate_failed_layers"], sort_keys=True), "status": "MATCH"}])),
        "",
        "## E2",
        "",
        f"Trial Gate A maximum |Δ|: {e2_info['gate_a_max_abs_delta']['trial']:.8g}. Primary Gate A maximum |Δ|: {e2_info['gate_a_max_abs_delta']['primary']:.8g}.",
        "Per-unit row counts, probability checks, Gate A/Gate B summaries and frozen-code provenance:",
        "",
        _markdown_table(e2_units),
        "",
        "Per-model × format × readout E2 maxima over block outputs 1..N:",
        "",
        _markdown_table(e2_summary),
        "",
        "## E3 landmarks and C1",
        "",
        "`cf729` is primary; `main` is the R9 sensitivity. `L_cross` no-crossing rows retain NA and their valid-draw count.",
        "",
        _markdown_table(e3_rows),
        "",
        "C1 instrument and paired-selection effects, reported in blocks and normalized depth:",
        "",
        _markdown_table(c1),
        "",
        "## E4 rules and Holm family (a)",
        "",
        _markdown_table(e4_rows),
        "",
        "Holm family (a): `W_L` is the 0.25–0.75 normalized-depth window mean. There are 15 tests in each format.",
        "",
        _markdown_table(holm),
        "",
        "## E5 language alignment",
        "",
        "Failed and undefined layers are listed separately. Primary maxima use only passing layers; all-defined-layer maxima are the sensitivity.",
        "",
        _markdown_table(e5_rows),
        "",
        "Undefined denominator layers by family and format:",
        "",
        _markdown_table(pd.DataFrame([
            {"family": family, "format": fmt, "undefined_layers": ",".join(map(str, values)) or "none"}
            for family, formats in undefined_layers.items() for fmt, values in formats.items()
        ])),
        "",
        "Pointwise LA and D_en curves with concept-bootstrap intervals are in `reports/E2-E5_la_curves.csv`.",
        "",
        "## A10 competence",
        "",
        _markdown_table(a10),
        "",
        "## Decision-rule table",
        "",
        _markdown_table(decision_table),
        "",
        "Per-model verdicts (OPS-14-9):",
        "",
        _markdown_table(verdict_table),
        "",
        "s vs s¹ model verdicts are shown together in the table above. E3 `L_cross` comparisons are in `reports/E2-E5_landmarks.csv`.",
        "",
        "## Figures",
        "",
        "Figure files are generated separately by the `figures` stage.",
        "",
        "## LA sensitivity and maximum layers",
        "",
        "All-defined-layer LA sensitivity, including gate-failed layers, with the rule-change flag is in `reports/checkpoint_h_la_sensitivity.csv`.",
        "Maximum layers and gate status at each maximum and its adjacent layers are in `reports/checkpoint_h_argmax_layers.csv`.",
        "",
        _markdown_table(argmax_layers),
        "",
        "",
        "## Output inventory",
        "",
        _markdown_table(pd.DataFrame([
            {"file": "reports/E2-E5_e2.csv", "rows": len(e2_summary)},
            {"file": "reports/E2-E5_e2_units.csv", "rows": len(e2_units)},
            {"file": "reports/E2-E5_landmarks.csv", "rows": len(landmarks_table)},
            {"file": "reports/E2-E5_c1.csv", "rows": len(c1)},
            {"file": "reports/E2-E5_rules.csv", "rows": len(rules)},
            {"file": "reports/E2-E5_holm.csv", "rows": len(holm)},
            {"file": "reports/checkpoint_h_rules.csv", "rows": len(decision_table)},
            {"file": "reports/checkpoint_h_verdicts.csv", "rows": len(verdict_table)},
            {"file": "reports/E2-E5_a10.csv", "rows": len(a10)},
            {"file": "reports/E2-E5_la_curves.csv", "rows": len(curve_report)},
            {"file": "reports/E2-E5_e5.csv", "rows": len(e5_rows)},
            {"file": "reports/E2-E5_undefined_layers.csv", "rows": len(undefined_table)},
            {"file": "reports/checkpoint_h_la_sensitivity.csv", "rows": len(la_sensitivity)},
            {"file": "reports/checkpoint_h_argmax_layers.csv", "rows": len(argmax_layers)},
        ])),
        "",
    ]
    draft_path, abstract_version, bracket_sources, unresolved_brackets = _make_adjudication_draft(
        repo,
        frozen_commit=frozen_commit,
        executed_commit=executed_commit,
        verdicts=verdict_table,
        landmarks_table=landmarks_table,
        la_summary=la_summary,
        rules=rules,
        c1=c1,
    )
    report_sections.extend(
        [
            "## Abstract bracket source cells",
            "",
            *[f"- {item}" for item in bracket_sources],
            "",
            "## Unresolved abstract alternatives",
            "",
            *([f"- {item}" for item in unresolved_brackets] if unresolved_brackets else ["- none"]),
            "",
        ]
    )
    (report_root / "E2-E5.md").write_text("\n".join(report_sections), encoding="utf-8")
    return {
        "frozen_analysis_commit": frozen_commit,
        "executed_commit": executed_commit,
        "reference_facts": reference_facts,
        "e2_units": len(e2_info["units"]),
        "e2_rows": int(sum(row["rows"] for row in e2_info["units"] if row["source"] == "primary")),
        "rule_rows": int(len(rules)),
        "landmark_rows": int(len(landmarks_table)),
        "c1_rows": int(len(c1)),
        "holm_rows": int(len(holm)),
        "a10_rows": int(len(a10)),
        "verdicts": verdict_table.to_dict(orient="records"),
        "abstract_version": abstract_version,
        "abstract_bracket_sources": bracket_sources,
        "unresolved_brackets": unresolved_brackets,
        "report": "reports/E2-E5.md",
        "rules": "reports/checkpoint_h_rules.csv",
        "draft": str(draft_path.relative_to(repo)),
        "figures_stage": "figures",
        "la_sensitivity_rows": int(len(la_sensitivity)),
        "argmax_layer_rows": int(len(argmax_layers)),
    }


def _run_checkpoint_figures_stage(
    cfg: Mapping[str, Any],
    repo: Path,
    *,
    analysis_identity: AnalysisCommitIdentity,
) -> dict[str, Any]:
    """Render plots from Checkpoint-H CSVs without recomputing analysis quantities."""
    report_root = repo / "reports"
    curve_path = report_root / "E2-E5_la_curves.csv"
    e5_path = report_root / "E2-E5_e5.csv"
    if not curve_path.is_file() or not e5_path.is_file():
        raise FileNotFoundError("The figures stage requires completed Checkpoint-H curve and E5 tables")
    curve_report = pd.read_csv(curve_path)
    e5_rows = pd.read_csv(e5_path)
    model_specs = study_models(dict(cfg))
    model_by_family = {family: str(spec["hf_id"]) for family, spec in model_specs.items()}
    family_by_model = {model: family for family, model in model_by_family.items()}
    la_by_family: dict[str, pd.DataFrame] = {}
    for model, family in family_by_model.items():
        selected = e5_rows.loc[
            e5_rows["model"].astype(str).eq(model)
            & e5_rows["metric"].astype(str).isin(("gate_failed_layer", "undefined_denominator_layer"))
        ]
        gate_rows: list[dict[str, Any]] = []
        for row in selected.to_dict(orient="records"):
            if row["metric"] == "gate_failed_layer":
                gate_rows.append(
                    {
                        "format": row["format"], "metric": "layer_pass", "target_lang": None,
                        "layer": row["layer"], "value": 0.0,
                    }
                )
            elif row["metric"] == "undefined_denominator_layer":
                gate_rows.append(
                    {
                        "format": row["format"], "metric": "undefined_denominator", "target_lang": None,
                        "layer": row["layer"], "value": 1.0,
                    }
                )
        la_by_family[family] = pd.DataFrame.from_records(gate_rows)
    figures = _write_checkpoint_figures(
        repo,
        families=list(model_by_family),
        model_by_family=model_by_family,
        rise_curves=curve_report.loc[curve_report["metric"].astype(str).eq("rise_cf729")],
        la_curves=curve_report.loc[curve_report["metric"].astype(str).eq("LA")],
        la_by_family=la_by_family,
    )
    _LOGGER.info(
        "figures stage provenance frozen_analysis_commit=%s executed_commit=%s files=%s",
        analysis_identity.frozen_analysis_commit,
        analysis_identity.executed_commit,
        json.dumps(figures),
    )
    return {
        "source": "primary",
        "stage": "figures",
        "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
        "executed_commit": analysis_identity.executed_commit,
        "files": figures,
    }


def run_analysis(
    source: str,
    *,
    config_path: str = "configs/experiments.yaml",
    repo: Path = REPO,
    confirm_checkpoint_h: bool = False,
    stages: str = "all",
    family: str | None = None,
) -> dict[str, Any]:
    """Run selected frozen analysis stages, never opening primary data without confirmation."""
    refuse_primary_without_checkpoint_h(source, confirm_checkpoint_h)
    cfg = load_config(config_path)
    if stages == "check_inputs":
        if source != "primary":
            raise ValueError("check_inputs is defined only for the primary source")
        if family is not None:
            raise ValueError("check_inputs always checks all configured families; omit --family")
        executed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _LOGGER.info(
            "analysis provenance frozen_analysis_commit=%s executed_commit=%s stage=check_inputs metadata_only=true",
            FROZEN_ANALYSIS_COMMIT,
            executed_commit,
        )
        checks = _whole_run_input_checks(cfg, repo)
        table = _input_check_markdown(checks)
        print(table, flush=True)
        passed = all(row["status"] == "MATCH" for row in checks)
        return {
            "source": source,
            "stages": ["check_inputs"],
            "input_checks": checks,
            "input_checks_passed": passed,
        }
    analysis_identity = assert_frozen_analysis_tree(repo) if source == "primary" else None
    if analysis_identity is not None:
        _LOGGER.info(
            "analysis provenance frozen_analysis_commit=%s executed_commit=%s",
            analysis_identity.frozen_analysis_commit,
            analysis_identity.executed_commit,
        )
    if stages not in {"core", "competence", "c1", "checkpoint_h", "figures", "all", "check_inputs"}:
        raise ValueError(f"Unknown analysis stage selection {stages!r}")
    if source != "primary":
        if source == "trial" and stages == "checkpoint_h":
            if family is not None:
                raise ValueError("trial checkpoint_h always processes all configured families; omit --family")
            return _run_checkpoint_h_trial_smoke(cfg, repo)
        if stages not in {"core", "all"}:
            raise ValueError(f"stage {stages!r} requires --source primary")
        if source == "trial" and family is None:
            families = [str(value) for value in cfg["pd"]["families"]]
            summaries = [
                _run_core_family(source, cfg=cfg, repo=repo, confirm_checkpoint_h=confirm_checkpoint_h, family=name)
                for name in families
            ]
            return {"source": source, "stages": ["core"], "families": summaries}
        return _run_core_family(
            source, cfg=cfg, repo=repo, confirm_checkpoint_h=confirm_checkpoint_h, family=family
        )

    families = [str(value) for value in cfg["pd"]["families"]]
    selected_families = [family] if family is not None else families
    unknown = sorted(set(selected_families) - set(families))
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}")
    requested = (
        ["core", "competence", "c1", "checkpoint_h", "figures"] if stages == "all" else [stages]
    )
    if requested == ["core"]:
        outputs = []
        for selected in selected_families:
            outputs.append(
                _run_core_family(
                    source, cfg=cfg, repo=repo,
                    confirm_checkpoint_h=confirm_checkpoint_h, family=selected,
                    analysis_identity=analysis_identity,
                )
            )
        return {
            "source": source,
            "stages": requested,
            "families": outputs,
            "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
            "executed_commit": analysis_identity.executed_commit,
        }
    if requested == ["competence"]:
        outputs = [_run_competence_family(cfg, repo, selected) for selected in selected_families]
        return {
            "source": source,
            "stages": requested,
            "families": outputs,
            "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
            "executed_commit": analysis_identity.executed_commit,
        }
    if requested == ["c1"]:
        return {
            **_run_c1_stage(cfg, repo, selected_families),
            "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
            "executed_commit": analysis_identity.executed_commit,
        }
    if requested == ["checkpoint_h"]:
        return _run_checkpoint_h_stage(cfg, repo, analysis_identity=analysis_identity)
    if requested == ["figures"]:
        return _run_checkpoint_figures_stage(cfg, repo, analysis_identity=analysis_identity)
    # 'all' uses explicit sub-stages but requires their outputs to exist in order.
    core = [
        _run_core_family(
            source, cfg=cfg, repo=repo,
            confirm_checkpoint_h=confirm_checkpoint_h, family=selected,
            analysis_identity=analysis_identity,
        )
        for selected in selected_families
    ]
    competence = [_run_competence_family(cfg, repo, selected) for selected in selected_families]
    c1 = _run_c1_stage(cfg, repo, selected_families)
    checkpoint = _run_checkpoint_h_stage(cfg, repo, analysis_identity=analysis_identity)
    figures = _run_checkpoint_figures_stage(cfg, repo, analysis_identity=analysis_identity)
    return {
        "source": source,
        "stages": requested,
        "core": core,
        "competence": competence,
        "c1": c1,
        "checkpoint_h": checkpoint,
        "figures": figures,
        "frozen_analysis_commit": analysis_identity.frozen_analysis_commit,
        "executed_commit": analysis_identity.executed_commit,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse guarded source, stage, and family selection and run analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("synthetic", "trial", "primary"), required=True)
    parser.add_argument("--confirm-checkpoint-h", action="store_true")
    parser.add_argument(
        "--stages",
        choices=("core", "competence", "c1", "checkpoint_h", "figures", "all", "check_inputs"),
        default="all",
    )
    parser.add_argument("--family", choices=TRIAL_FAMILIES)
    args = parser.parse_args(argv)
    try:
        refuse_primary_without_checkpoint_h(args.source, args.confirm_checkpoint_h)
    except PermissionError as exc:
        parser.error(str(exc))
    global _LOGGER
    _LOGGER = setup_logging("analyze_core", REPO / "logs")
    started = time.perf_counter()
    with StageTimer(f"core {args.source}", _LOGGER):
        result = run_analysis(
            args.source,
            config_path=args.config,
            confirm_checkpoint_h=args.confirm_checkpoint_h,
            stages=args.stages,
            family=args.family,
        )
    _LOGGER.info("output summary: %s", json.dumps(result, sort_keys=True))
    _LOGGER.info("final summary: source=%s wall_time_seconds=%.3f", args.source, time.perf_counter() - started)
    if args.stages == "check_inputs" and not result["input_checks_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
