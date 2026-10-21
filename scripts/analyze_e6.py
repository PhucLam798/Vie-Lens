#!/datastore/luannt/.cache/sb/bin/python
"""Run the OPS-28 E6 analysis, or the outcome-free covariate match alone."""

from __future__ import annotations

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

if __name__ == "__main__":
    env_guard.guard(offline=True)

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import resource
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens.analysis import e6
from vilens.config import load_config, study_models
from vilens.data import load_prompt_file, prompt_file_paths
from vilens.derived import stream
from vilens.io import unit_paths
from vilens.locks import ensure_adjudication_committed, ensure_model_access_allowed
from vilens.manifest import verify_manifest
from vilens.passes import pl_unit_key, with_pl_shards
from vilens.provenance import RunGit, build as build_provenance, capture_run_git, differences

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
LANGUAGES = ("vi", "en", "zh", "fr", "id")
DELTA_COLUMNS = [
    "model", "format", "concept_id", "target_lang", "target_variant", "layer", "metric", "value",
]
LA_COLUMNS = [
    "model", "format", "concept_id", "target_lang", "target_variant", "layer", "metric", "value",
]
MATCH_COLUMNS = ["pair_id", "sino_concept_id", "nonsino_concept_id"]
SENSITIVITY_MASKS = {
    "bt_pass_strict_v1": "bt_pass_strict_v1",
    "external_attested": "external_attested",
    "all_three_nllb": "all_three_nllb",
    "min_surface_dist_ge_060": "min_surface_dist_ge_060",
}
PREFLIGHT_LA_COLUMNS = ["format", "concept_id", "target_lang", "layer", "metric"]
PREFLIGHT_RISE_COLUMNS = ["concept_id", "format", "target_lang", "layer", "metric"]
EXPECTED_LA_QUERIES = {"repetition": 1535, "translation": 1524}


def construct_provenance_preflight_records(
    cfg: Mapping[str, Any], repo: Path
) -> list[tuple[str, dict[str, Any]]]:
    """Construct, but do not write, provenance for every E6 persisted unit shape."""
    run_git = capture_run_git(repo, cfg)
    if run_git.git_dirty:
        raise RuntimeError("E6 provenance capture requires a clean code tree")
    models = study_models(dict(cfg))
    chunk_rows = min(
        int(cfg["e6"]["bootstrap_chunk"]), int(cfg["e6"]["bootstrap_resamples"])
    )
    records: list[tuple[str, dict[str, Any]]] = []

    def add_record(
        family: str,
        label: str,
        unit_key: Mapping[str, Any],
        expected_rows: int,
    ) -> None:
        """Build one rule-37 payload without creating a result file."""
        record = build_provenance(
            cfg,
            repo_root=repo,
            unit_key=unit_key,
            entrypoint=repo / "scripts/analyze_e6.py",
            model_id=str(models[family]["hf_id"]),
            model_revision=str(models[family]["revision"]),
            dtype="analysis_float64",
            batch_size=None,
            expected_row_count=int(expected_rows),
            run_git=run_git,
            model_loads=[],
        )
        records.append((f"{family}:{label}", record))

    def chunk_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    variants = tuple(map(str, cfg["e6"]["sensitivity_variants"]))
    for family in FAMILIES:
        dzh_key = str(cfg["e6"]["dzh_key_template"]).format(model=family)
        model_key = str(cfg["e6"]["model_key_template"]).format(model=family)
        matched_key = str(cfg["e6"]["matched_key_template"]).format(model=family)
        add_record(
            family, "dzh_chunk",
            {"family": family, "stage": "dzh", "bootstrap_key_sha256": chunk_key(dzh_key), "chunk_index": 0},
            chunk_rows,
        )
        add_record(
            family, "model_full_fit",
            {"family": family, "stage": "model_full_fit"}, 1,
        )
        add_record(
            family, "bootstrap_chunk",
            {"family": family, "stage": "model", "bootstrap_key_sha256": chunk_key(model_key), "chunk_index": 0},
            chunk_rows,
        )
        for stage in ("matched_dzh", "matched_model"):
            add_record(
                family, f"{stage}_chunk",
                {"family": family, "stage": stage, "bootstrap_key_sha256": chunk_key(matched_key), "chunk_index": 0},
                chunk_rows,
            )
        for variant in variants:
            add_record(
                family, f"sensitivity_fit:{variant}",
                {"family": family, "stage": "sensitivity_fit", "variant": variant}, 1,
            )
            if variant not in {"zipf_vi", "zipf_zh"}:
                sensitivity_key = f"bootstrap|h3_sensitivity_{variant}|{family}|pooled"
                add_record(
                    family, f"sensitivity_chunk:{variant}",
                    {"family": family, "stage": f"sensitivity_{variant}",
                     "bootstrap_key_sha256": chunk_key(sensitivity_key), "chunk_index": 0},
                    chunk_rows,
                )
        add_record(
            family, "report",
            {"family": family, "stage": "report"}, 1,
        )
    return records


class BootstrapChunkStore:
    """Persist bootstrap chunks with rule-37 provenance and row-count checks."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        repo: Path,
        run_git: RunGit,
        family: str,
        stage: str,
        key: str,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.run_git = run_git
        self.family = family
        self.stage = stage
        self.key = key
        self.key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        self.root = repo / str(cfg["e6"]["output_dir"]) / "checkpoints" / f"family={family}"
        self.root.mkdir(parents=True, exist_ok=True)
        models = study_models(dict(cfg))
        self.model_id = str(models[family]["hf_id"])
        self.revision = str(models[family]["revision"])

    def _paths(self, chunk_index: int) -> tuple[Path, Path]:
        """Return the deterministic archive and sidecar paths for one chunk."""
        stem = f"{self.stage}__key={self.key_hash}__chunk={int(chunk_index):04d}"
        return self.root / f"{stem}.npz", self.root / f"{stem}.provenance.json"

    def _provenance(self, unit_key: Mapping[str, Any], row_count: int) -> dict[str, Any]:
        """Build the shared rule-37 sidecar fields for an analysis-only unit."""
        return build_provenance(
            self.cfg,
            repo_root=self.repo,
            unit_key=unit_key,
            entrypoint=self.repo / "scripts/analyze_e6.py",
            model_id=self.model_id,
            model_revision=self.revision,
            dtype="analysis_float64",
            batch_size=None,
            expected_row_count=int(row_count),
            run_git=self.run_git,
            model_loads=[],
        )

    def load(self, total: int, chunk_size: int) -> dict[int, tuple[np.ndarray, ...]]:
        """Return complete model-fit chunks with per-draw OPS-29 diagnostics."""
        from vilens.manifest import sha256_file

        completed: dict[int, tuple[np.ndarray, ...]] = {}
        count = (int(total) + int(chunk_size) - 1) // int(chunk_size)
        for index in range(count):
            begin = index * int(chunk_size)
            expected_rows = min(int(chunk_size), int(total) - begin)
            archive, sidecar_path = self._paths(index)
            if not archive.is_file() or not sidecar_path.is_file():
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                unit_key = {
                    "family": self.family,
                    "stage": self.stage,
                    "bootstrap_key_sha256": self.key_hash,
                    "chunk_index": index,
                }
                current = self._provenance(unit_key, expected_rows)
                changed = differences(sidecar, current, list(self.cfg["runtime"]["provenance_compare"]))
                if changed:
                    continue
                if sidecar.get("row_count") != expected_rows or sidecar.get("expected_row_count") != expected_rows:
                    continue
                if sidecar.get("output_sha256") != sha256_file(archive):
                    continue
                with np.load(archive, allow_pickle=False) as stored:
                    estimates = np.asarray(stored["estimates"], dtype=np.float64)
                    valid = np.asarray(stored["valid"], dtype=bool)
                    methods = np.asarray(stored["methods"], dtype=str)
                    tries = np.asarray(stored["tries"], dtype=np.int16)
                    boundaries = np.asarray(stored["boundaries"], dtype=bool)
                    warning_json = np.asarray(stored["warning_json"], dtype=str)
                    failure_json = np.asarray(stored["failure_json"], dtype=str)
                    fit_seconds = np.asarray(stored["fit_seconds"], dtype=np.float64)
                    boundary_rule_ran = np.asarray(stored["boundary_rule_ran"], dtype=bool)
                    boundary_rule_changed = np.asarray(stored["boundary_rule_changed"], dtype=bool)
                    llf_kept = np.asarray(stored["llf_kept"], dtype=np.float64)
                    llf_first_valid = np.asarray(stored["llf_first_valid"], dtype=np.float64)
                    random_intercept_variance = np.asarray(stored["random_intercept_variance"], dtype=np.float64)
                arrays = (
                    estimates, valid, methods, tries, boundaries, warning_json, failure_json, fit_seconds,
                    boundary_rule_ran, boundary_rule_changed, llf_kept, llf_first_valid,
                    random_intercept_variance,
                )
                if any(array.shape != (expected_rows,) for array in arrays):
                    continue
                completed[index] = arrays
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return completed

    def load_values(self, total: int, chunk_size: int) -> dict[int, np.ndarray]:
        """Load cached scalar/vector bootstrap output arrays."""
        from vilens.manifest import sha256_file

        completed: dict[int, np.ndarray] = {}
        count = (int(total) + int(chunk_size) - 1) // int(chunk_size)
        for index in range(count):
            begin = index * int(chunk_size)
            expected_rows = min(int(chunk_size), int(total) - begin)
            archive, sidecar_path = self._paths(index)
            if not archive.is_file() or not sidecar_path.is_file():
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                unit_key = {
                    "family": self.family,
                    "stage": self.stage,
                    "bootstrap_key_sha256": self.key_hash,
                    "chunk_index": index,
                }
                current = self._provenance(unit_key, expected_rows)
                if differences(sidecar, current, list(self.cfg["runtime"]["provenance_compare"])):
                    continue
                if sidecar.get("row_count") != expected_rows or sidecar.get("expected_row_count") != expected_rows:
                    continue
                if sidecar.get("output_sha256") != sha256_file(archive):
                    continue
                with np.load(archive, allow_pickle=False) as stored:
                    values = np.asarray(stored["values"], dtype=np.float64)
                if values.ndim == 0 or values.shape[0] != expected_rows:
                    continue
                completed[index] = values
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return completed

    def save(
        self,
        chunk_index: int,
        _begin: int,
        estimates: np.ndarray,
        valid: np.ndarray,
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically save estimates and per-draw OPS-29 diagnostics."""
        from vilens.manifest import sha256_file

        archive, sidecar_path = self._paths(chunk_index)
        unit_key = {
            "family": self.family,
            "stage": self.stage,
            "bootstrap_key_sha256": self.key_hash,
            "chunk_index": int(chunk_index),
        }
        payload = self._provenance(unit_key, len(estimates))
        temp_archive = archive.with_name(archive.name + f".{os.getpid()}.tmp")
        temp_sidecar = sidecar_path.with_name(sidecar_path.name + f".{os.getpid()}.tmp")
        sidecar_path.unlink(missing_ok=True)
        try:
            with temp_archive.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    estimates=np.asarray(estimates, dtype=np.float64),
                    valid=np.asarray(valid, dtype=bool),
                    methods=np.asarray([str(item.get("method_used", "not_estimable")) for item in diagnostics], dtype="U32"),
                    tries=np.asarray([int(item.get("tries", 0)) for item in diagnostics], dtype=np.int16),
                    boundaries=np.asarray([bool(item.get("variance_at_boundary", False)) for item in diagnostics], dtype=bool),
                    warning_json=np.asarray([json.dumps(item.get("warning_categories", {}), sort_keys=True) for item in diagnostics], dtype="U4096"),
                    failure_json=np.asarray([json.dumps(item.get("failure_reasons", []), sort_keys=True) for item in diagnostics], dtype="U4096"),
                    fit_seconds=np.asarray([float(item.get("fit_seconds", 0.0)) for item in diagnostics], dtype=np.float64),
                    boundary_rule_ran=np.asarray([bool(item.get("boundary_rule_ran", False)) for item in diagnostics], dtype=bool),
                    boundary_rule_changed=np.asarray([bool(item.get("boundary_rule_changed_fit", False)) for item in diagnostics], dtype=bool),
                    llf_kept=np.asarray([float(item["llf_kept"]) if item.get("llf_kept") is not None else np.nan for item in diagnostics], dtype=np.float64),
                    llf_first_valid=np.asarray([float(item["llf_first_valid"]) if item.get("llf_first_valid") is not None else np.nan for item in diagnostics], dtype=np.float64),
                    random_intercept_variance=np.asarray([float(item["random_intercept_variance"]) if item.get("random_intercept_variance") is not None else np.nan for item in diagnostics], dtype=np.float64),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_archive, archive)
            payload["row_count"] = int(len(estimates))
            payload["output_sha256"] = sha256_file(archive)
            payload["utc_end"] = utc_now()
            with temp_sidecar.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_sidecar, sidecar_path)
        finally:
            temp_archive.unlink(missing_ok=True)
            temp_sidecar.unlink(missing_ok=True)

    def save_values(self, chunk_index: int, _begin: int, values: np.ndarray) -> None:
        """Atomically save bootstrap values with the same sidecar contract."""
        from vilens.manifest import sha256_file

        archive, sidecar_path = self._paths(chunk_index)
        unit_key = {
            "family": self.family,
            "stage": self.stage,
            "bootstrap_key_sha256": self.key_hash,
            "chunk_index": int(chunk_index),
        }
        row_count = int(np.asarray(values).shape[0])
        payload = self._provenance(unit_key, row_count)
        temp_archive = archive.with_name(archive.name + f".{os.getpid()}.tmp")
        temp_sidecar = sidecar_path.with_name(sidecar_path.name + f".{os.getpid()}.tmp")
        sidecar_path.unlink(missing_ok=True)
        try:
            with temp_archive.open("wb") as stream:
                np.savez_compressed(stream, values=np.asarray(values))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_archive, archive)
            payload["row_count"] = row_count
            payload["output_sha256"] = sha256_file(archive)
            payload["utc_end"] = utc_now()
            with temp_sidecar.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_sidecar, sidecar_path)
        finally:
            temp_archive.unlink(missing_ok=True)
            temp_sidecar.unlink(missing_ok=True)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def setup_logger() -> logging.Logger:
    """Write stage records to stdout and a unique logs file."""
    logs = REPO / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = logging.getLogger("analyze_e6")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logs / f"analyze_e6_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def read_parquet_columns(path: Path, columns: Sequence[str], *, batch_size: int = 65_536) -> pd.DataFrame:
    """Project and stream only named columns from one parquet file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    pieces = [batch.to_pandas() for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size)]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(columns))


def read_e5_gate_layers(path: Path, model_id: str) -> dict[str, set[int]]:
    """Read only family/format/layer gate metadata, never LA value columns."""
    table = pd.read_csv(path, usecols=["model", "format", "metric", "layer"])
    family_rows = table.loc[table["model"].astype(str).eq(str(model_id))]
    failed: dict[str, set[int]] = {}
    for fmt in FORMATS:
        selected = family_rows.loc[
            family_rows["format"].astype(str).eq(fmt)
            & family_rows["metric"].astype(str).isin(("gate_failed_layer", "undefined_denominator_layer"))
        ]
        failed[fmt] = set(pd.to_numeric(selected["layer"], errors="raise").dropna().astype(int))
    return failed


def validate_la_gate_layers(
    la_frame: pd.DataFrame,
    failed_layers: Mapping[str, set[int]],
    *,
    n_layers: int,
) -> None:
    """Cross-check that stored LA rows are exactly the E5 passing block layers."""
    for fmt in FORMATS:
        present = set(
            pd.to_numeric(
                la_frame.loc[la_frame["format"].astype(str).eq(fmt), "layer"], errors="raise"
            ).astype(int)
        )
        expected = set(range(1, int(n_layers) + 1)) - set(failed_layers.get(fmt, set()))
        if present != expected:
            raise AssertionError(
                f"{fmt} LA layer set differs from E5 gate metadata: "
                f"missing={sorted(expected - present)} extra={sorted(present - expected)}"
            )


def _select_zh_la_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the E2/E5 LA_zh rows, whose aggregate target_variant is null."""
    return frame.loc[
        frame["metric"].astype(str).eq("LA")
        & frame["target_lang"].astype(str).eq("zh")
        & frame["format"].astype(str).isin(FORMATS)
    ].copy()


def _load_tsv_metadata(path: Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Read typed covariate/stratum metadata after the frozen first comment line."""
    return pd.read_csv(path, sep="\t", skiprows=1, usecols=usecols, low_memory=False)


def _main_strata_metadata(path: Path) -> pd.DataFrame:
    """Return main non-extension test concepts and E6 metadata fields."""
    columns = [
        "concept_id", "split", "m1_extension", "stratum", "native_strict",
        "zipf_en", "zipf_vi", "zipf_zh", "n_syllables", "concreteness",
        "bt_pass_strict_v1", "external_attested", "zh_nllb_agree", "fr_nllb_agree",
        "id_nllb_agree", "min_surface_dist", "ntok_gemma_vi", "ntok_qwen_vi", "ntok_llama_vi",
    ]
    frame = _load_tsv_metadata(path, usecols=columns)
    frame["concept_id"] = frame["concept_id"].astype(str)
    frame["all_three_nllb"] = _all_true_bool_flags(
        frame, ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree")
    )
    frame["min_surface_dist_ge_060"] = pd.to_numeric(frame["min_surface_dist"], errors="coerce").ge(0.60)
    extension = e6._bool_values(frame["m1_extension"])
    return frame.loc[
        frame["split"].astype(str).eq("test")
        & ~extension
    ].copy()


def _all_true_bool_flags(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Parse TSV boolean strings before combining the three NLLB flags."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"boolean metadata lacks {sorted(missing)}")
    parsed = pd.concat(
        [e6._bool_values(frame[column]).rename(column) for column in columns], axis=1
    )
    return parsed.all(axis=1)


def _analysis_main_ids(metadata: pd.DataFrame) -> set[str]:
    """Return all main sino/nonsino test IDs, excluding extensions."""
    return set(metadata["concept_id"].astype(str))


def _complete_covariates(
    metadata: pd.DataFrame, expected_counts: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """Select complete-case model/matching covariates without exposing their values."""
    selected = metadata.loc[:, [
        "concept_id", "stratum", "native_strict", "zipf_en", "zipf_vi", "zipf_zh",
        "n_syllables", "concreteness", "bt_pass_strict_v1", "external_attested",
        "all_three_nllb", "min_surface_dist_ge_060",
    ]].copy()
    for column in ("zipf_en", "zipf_vi", "zipf_zh", "n_syllables", "concreteness"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        selected["stratum"].astype(str).isin(("sino", "nonsino"))
        & selected[["zipf_en", "n_syllables", "concreteness"]].notna().all(axis=1)
    ].copy()
    counts = selected["stratum"].value_counts().to_dict()
    expected = {"nonsino": 688, "sino": 469} if expected_counts is None else dict(expected_counts)
    if counts != expected:
        raise AssertionError(f"complete cases differ from OPS-28: {counts}")
    return selected


def _primary_format_prompt_ids(cfg: Mapping[str, Any], repo: Path) -> dict[str, set[str]]:
    """Read primary repetition/translation prompt availability from released metadata."""
    paths = prompt_file_paths(dict(cfg))
    selected_files = tuple(map(str, cfg["pl"]["primary"]["files"]))
    output: dict[str, set[str]] = {}
    for fmt in FORMATS:
        matching = [name for name in selected_files if name.startswith(f"{fmt}_")]
        if len(matching) != 1:
            raise ValueError(f"primary P-L plan must name exactly one {fmt} prompt file")
        path = Path(paths[matching[0]])
        if not path.is_absolute():
            path = repo / path
        records = load_prompt_file(path)
        output[fmt] = {
            str(record["concept_id"])
            for record in records
            if str(record.get("split")) == "test" and not bool(record.get("m1_extension", False))
        }
    return output


def _format_concept_counts(
    covariates: pd.DataFrame, prompt_ids_by_format: Mapping[str, set[str]]
) -> dict[str, dict[str, int]]:
    """Count complete-case concepts with a primary prompt, separately by format."""
    if set(prompt_ids_by_format) != set(FORMATS):
        raise ValueError(f"prompt availability must contain exactly {FORMATS}")
    concepts = covariates.loc[:, ["concept_id", "stratum"]].copy()
    concepts["concept_id"] = concepts["concept_id"].astype(str)
    concepts["stratum"] = concepts["stratum"].astype(str)
    return {
        fmt: {
            stratum: int(
                concepts["concept_id"].isin(prompt_ids_by_format[fmt])
                .where(concepts["stratum"].eq(stratum), False)
                .sum()
            )
            for stratum in ("sino", "nonsino")
        }
        for fmt in FORMATS
    }


def primary_complete_case_format_counts(
    cfg: Mapping[str, Any], repo: Path, *, selected_concept_ids: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """Return covariate-complete prompt counts for production-shape synthetic timing."""
    metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
    covariates = _complete_covariates(metadata, cfg["e6"]["complete_case_counts"])
    if selected_concept_ids is not None:
        covariates = covariates.loc[
            covariates["concept_id"].astype(str).isin(set(map(str, selected_concept_ids)))
        ].copy()
    return _format_concept_counts(covariates, _primary_format_prompt_ids(cfg, repo))


def primary_dzh_format_counts(cfg: Mapping[str, Any], repo: Path) -> dict[str, dict[str, int]]:
    """Return main sino/nonsino prompt counts used by the format-specific D_zh inputs."""
    metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
    eligible = metadata.loc[metadata["stratum"].astype(str).isin(("sino", "nonsino"))]
    return _format_concept_counts(eligible, _primary_format_prompt_ids(cfg, repo))


def primary_matched_pair_format_ids(
    cfg: Mapping[str, Any], repo: Path, pairs: pd.DataFrame
) -> dict[str, set[str]]:
    """Return matched pair IDs with both concepts prompted in each format."""
    prompt_ids = _primary_format_prompt_ids(cfg, repo)
    required = {"pair_id", "sino_concept_id", "nonsino_concept_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"matched pair table lacks {sorted(required - set(pairs.columns))}")
    output = {fmt: set() for fmt in FORMATS}
    for row in pairs.loc[:, ["pair_id", "sino_concept_id", "nonsino_concept_id"]].itertuples(index=False):
        for fmt in FORMATS:
            if str(row.sino_concept_id) in prompt_ids[fmt] and str(row.nonsino_concept_id) in prompt_ids[fmt]:
                output[fmt].add(str(row.pair_id))
    return output


def preflight_primary_inputs(cfg: Mapping[str, Any], repo: Path) -> list[tuple[str, str, str]]:
    """Check E6 primary input metadata without projecting any outcome value columns."""
    rows: list[tuple[str, str, str]] = []
    concepts_path = repo / str(cfg["paths"]["concepts_tsv"])
    metadata = _main_strata_metadata(concepts_path)
    strata = metadata["stratum"].astype(str).value_counts().to_dict()
    expected_strata = {"sino": 481, "nonsino": 706, "ambiguous": 347, "other_loan": 1}
    rows.append(("main_strata", "MATCH" if strata == expected_strata else "MISMATCH", str(strata)))

    covariates = _complete_covariates(metadata, cfg["e6"]["complete_case_counts"])
    complete_counts = covariates["stratum"].astype(str).value_counts().to_dict()
    complete_expected = {"sino": 469, "nonsino": 688}
    rows.append(("complete_cases", "MATCH" if complete_counts == complete_expected else "MISMATCH", str(complete_counts)))
    prompt_ids_by_format = _primary_format_prompt_ids(cfg, repo)
    format_concept_counts = _format_concept_counts(covariates, prompt_ids_by_format)

    native = e6._bool_values(metadata["native_strict"])
    native_count = int(native.sum())
    rows.append(("native_strict", "MATCH" if native_count == 51 else "MISMATCH", f"count={native_count}"))
    flags = {
        "bt_pass_strict_v1": 1111,
        "external_attested": 1086,
        "all_three_nllb": 964,
        "min_surface_dist_ge_060": 1485,
    }
    actual_flags = {
        "bt_pass_strict_v1": int(e6._bool_values(metadata["bt_pass_strict_v1"]).sum()),
        "external_attested": int(e6._bool_values(metadata["external_attested"]).sum()),
        "all_three_nllb": int(_all_true_bool_flags(
            metadata, ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree")
        ).sum()),
        "min_surface_dist_ge_060": int(pd.to_numeric(metadata["min_surface_dist"], errors="coerce").ge(0.60).sum()),
    }
    for name, expected in flags.items():
        actual = actual_flags[name]
        rows.append((f"v1.2_{name}", "MATCH" if actual == expected else "MISMATCH", f"count={actual}; expected={expected}"))

    main_ids = set(metadata["concept_id"].astype(str))
    r11_present = sorted(set(map(str, cfg["e6"]["r11_concept_ids"])) & main_ids)
    rows.append(("R11_ids", "MATCH" if len(r11_present) == 6 else "MISMATCH", f"present={len(r11_present)}/6"))

    models = study_models(dict(cfg))
    e5_path = repo / "reports/E2-E5_e5.csv"
    if not e5_path.is_file():
        rows.append(("E5_gate_layers", "MISMATCH", "reports/E2-E5_e5.csv missing"))
        return rows
    for family in FAMILIES:
        core_dir = repo / "results/analysis/primary" / f"family={family}"
        missing_core = [name for name in ("la.parquet", "deltas.parquet", "rise.parquet") if not (core_dir / name).is_file()]
        rows.append((f"core_inputs_{family}", "MATCH" if not missing_core else "MISMATCH", "all present" if not missing_core else ",".join(missing_core)))
        try:
            rise_meta = read_parquet_columns(core_dir / "rise.parquet", PREFLIGHT_RISE_COLUMNS)
            complete_ids = set(covariates["concept_id"].astype(str))
            selected_rise = rise_meta.loc[
                rise_meta["metric"].astype(str).eq("rise_main")
                & rise_meta["format"].astype(str).isin(FORMATS)
                & rise_meta["target_lang"].astype(str).isin(LANGUAGES)
                & rise_meta["concept_id"].astype(str).isin(complete_ids)
            ].copy()
            observed_by_format = {
                fmt: int(selected_rise["format"].astype(str).eq(fmt).sum()) for fmt in FORMATS
            }
            expected_by_format, expected_total = e6.expected_s71_row_grid(
                format_concept_counts, n_layers=_family_n_layers(cfg, family)
            )
            key_columns = ["concept_id", "format", "target_lang", "layer", "metric"]
            duplicate_keys = bool(selected_rise.duplicated(key_columns).any())
            observed_total = int(len(selected_rise))
            grid_ok = observed_by_format == expected_by_format and observed_total == expected_total and not duplicate_keys
            count_text = ", ".join(
                f"{fmt}={observed_by_format[fmt]}/{expected_by_format[fmt]}"
                for fmt in FORMATS
            )
            rows.append((
                f"row_grid_{family}", "MATCH" if grid_ok else "MISMATCH",
                f"observed={observed_total}; expected={expected_total}; {count_text}; duplicate_keys={duplicate_keys}",
            ))
        except Exception as exc:
            rows.append((f"row_grid_{family}", "MISMATCH", f"{type(exc).__name__}: {exc}"))
        try:
            e5_failed = read_e5_gate_layers(e5_path, models[family]["hf_id"])
            la_meta = read_parquet_columns(core_dir / "la.parquet", PREFLIGHT_LA_COLUMNS)
            la_meta = _select_zh_la_rows(la_meta)
            layer_ok = True
            count_ok = True
            count_details: list[str] = []
            for fmt in FORMATS:
                selected = la_meta.loc[la_meta["format"].astype(str).eq(fmt)]
                present_layers = set(pd.to_numeric(selected["layer"], errors="raise").astype(int))
                expected_layers = set(range(1, _family_n_layers(cfg, family) + 1)) - e5_failed.get(fmt, set())
                layer_ok &= present_layers == expected_layers
                per_layer = selected.groupby("layer", sort=True)["concept_id"].nunique()
                counts_match = bool(per_layer.eq(EXPECTED_LA_QUERIES[fmt]).all()) and bool(len(per_layer))
                count_ok &= counts_match
                count_details.append(f"{fmt}={len(per_layer)} layers; {EXPECTED_LA_QUERIES[fmt]} queries/layer")
            rows.append((f"E5_gate_layers_{family}", "MATCH" if layer_ok else "MISMATCH", "LA layers equal E5 passing layers"))
            rows.append((f"LA_zh_queries_{family}", "MATCH" if count_ok else "MISMATCH", "; ".join(count_details)))
        except Exception as exc:
            rows.append((f"E5_LA_metadata_{family}", "MISMATCH", f"{type(exc).__name__}: {exc}"))
        try:
            la_units = read_parquet_columns(core_dir / "la.parquet", PREFLIGHT_LA_COLUMNS)
            la_units = _select_zh_la_rows(la_units)
            la_units = la_units.loc[la_units["concept_id"].astype(str).isin(main_ids)].copy()
            strata_lookup = metadata.set_index("concept_id")["stratum"].astype(str)
            la_units["stratum"] = la_units["concept_id"].astype(str).map(strata_lookup)
            la_units = la_units.loc[la_units["stratum"].isin(("sino", "nonsino"))].copy()
            layers = pd.to_numeric(la_units["layer"], errors="raise").astype(int)
            depth = layers / _family_n_layers(cfg, family)
            low, high = map(float, cfg["e6"]["window"])
            la_units = la_units.loc[layers.ge(1) & depth.between(low, high, inclusive="both")].copy()
            unit_details: list[str] = []
            unit_ok = True
            for fmt in FORMATS:
                selected = la_units.loc[la_units["format"].astype(str).eq(fmt)]
                per_layer = selected.groupby("layer", sort=True)["concept_id"].nunique()
                strata_per_layer = selected.groupby(["layer", "stratum"], sort=True)["concept_id"].nunique()
                both_strata = all(
                    bool(strata_per_layer.xs(layer, level="layer").reindex(["sino", "nonsino"]).notna().all())
                    if layer in strata_per_layer.index.get_level_values("layer") else False
                    for layer in per_layer.index
                )
                unit_ok &= bool(len(per_layer)) and both_strata
                unit_details.append(
                    f"{fmt}={len(selected)} rows/{selected['concept_id'].nunique()} concepts/{len(per_layer)} layers"
                )
            rows.append((
                f"D_zh_units_{family}", "MATCH" if unit_ok else "MISMATCH", "; ".join(unit_details)
            ))
        except Exception as exc:
            rows.append((f"D_zh_units_{family}", "MISMATCH", f"{type(exc).__name__}: {exc}"))
        try:
            expected_paths = _pl_unit_paths(cfg, family, repo)
            missing_pl = [path.name for path in expected_paths if not path.is_file()]
            rows.append((f"PL_units_{family}", "MATCH" if not missing_pl else "MISMATCH", f"{len(expected_paths) - len(missing_pl)}/{len(expected_paths)} present"))
        except Exception as exc:
            rows.append((f"PL_units_{family}", "MISMATCH", f"{type(exc).__name__}: {exc}"))
    return rows


def _family_n_layers(cfg: Mapping[str, Any], family: str) -> int:
    """Read the frozen model block count from X0 configuration facts."""
    return int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])


def _load_family_table(repo: Path, family: str, filename: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read one family-scoped core output in bounded parquet batches."""
    path = repo / "results/analysis/primary" / f"family={family}" / filename
    return read_parquet_columns(path, columns)


def _pl_unit_paths(cfg: Mapping[str, Any], family: str, repo: Path) -> list[Path]:
    """Resolve primary P-L parquet units from the same prompt unit-key plan."""
    paths = prompt_file_paths(dict(cfg))
    records: list[dict[str, Any]] = []
    for stem in cfg["pl"]["primary"]["files"]:
        path = Path(paths[str(stem)])
        records.extend(load_prompt_file(path if path.is_absolute() else repo / path))
    planned = with_pl_shards(records, int(cfg["pl"]["shard_prompts"]))
    output: dict[str, Path] = {}
    result_root = repo / str(cfg["pl"]["results_dir"])
    for record in planned:
        key = pl_unit_key(family, record)
        parquet, _sidecar = unit_paths(result_root, key)
        output[parquet.name] = parquet
    return [output[key] for key in sorted(output)]


def _load_pl_token_counts(cfg: Mapping[str, Any], family: str, repo: Path, main_ids: set[str]) -> pd.DataFrame:
    """Count target content-token indices from P-L metadata columns only."""
    selected: list[pd.DataFrame] = []
    columns = ["concept_id", "format", "target_lang", "target_variant", "role", "tok_idx", "is_content"]
    for path in _pl_unit_paths(cfg, family, repo):
        if not path.is_file():
            raise FileNotFoundError(f"Missing P-L unit for {family}: {path}")
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=65_536):
            frame = batch.to_pandas()
            frame = frame.loc[
                frame["concept_id"].astype(str).isin(main_ids)
                & frame["format"].astype(str).isin(FORMATS)
                & frame["target_lang"].astype(str).isin(LANGUAGES)
                & frame["target_variant"].astype(str).eq(frame["target_lang"].astype(str).map(e6.PRIMARY_VARIANTS))
                & frame["role"].astype(str).eq("correct")
                & frame["is_content"].fillna(False).astype(bool)
            ]
            if not frame.empty:
                selected.append(frame.loc[:, ["concept_id", "format", "target_lang", "tok_idx"]])
    if not selected:
        raise FileNotFoundError(f"No P-L correct-target token rows for {family}")
    tokens = pd.concat(selected, ignore_index=True).drop_duplicates(
        ["concept_id", "format", "target_lang", "tok_idx"]
    )
    tokens["n_tokens"] = 1
    return tokens.groupby(["concept_id", "format", "target_lang"], as_index=False, sort=True)["n_tokens"].sum()


def _load_response_inputs(
    cfg: Mapping[str, Any], repo: Path, family: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Load delta/rise and allowed metadata projections for one family."""
    metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
    main_ids = _analysis_main_ids(metadata)
    covars = _complete_covariates(metadata, cfg["e6"]["complete_case_counts"])
    delta = _load_family_table(repo, family, "deltas.parquet", DELTA_COLUMNS)
    delta = delta.loc[
        delta["metric"].astype(str).eq("delta")
        & delta["format"].astype(str).isin(FORMATS)
        & delta["concept_id"].astype(str).isin(main_ids)
        & delta["target_lang"].astype(str).isin(LANGUAGES)
        & delta["target_variant"].astype(str).eq(delta["target_lang"].astype(str).map(e6.PRIMARY_VARIANTS))
    ].copy()
    stored = _load_family_table(repo, family, "rise.parquet", DELTA_COLUMNS)
    stored = stored.loc[
        stored["metric"].astype(str).eq("rise_main")
        & stored["format"].astype(str).isin(FORMATS)
        & stored["concept_id"].astype(str).isin(main_ids)
        & stored["target_lang"].astype(str).isin(LANGUAGES)
        & stored["target_variant"].astype(str).eq(stored["target_lang"].astype(str).map(e6.PRIMARY_VARIANTS))
    ].copy()
    delta = delta.merge(metadata.loc[:, ["concept_id", "stratum"]], on="concept_id", how="inner", validate="many_to_one")
    response = e6.per_concept_response(delta, early_depth_max=float(cfg["e1"]["early_depth_max"]))
    response["stratum"] = response["concept_id"].map(metadata.set_index("concept_id")["stratum"])
    stored = stored.merge(metadata.loc[:, ["concept_id", "stratum"]], on="concept_id", how="inner", validate="many_to_one")
    max_diff = e6.compare_with_stored_rise(response, stored)
    aggregate_diff = e6.assert_concept_mean_matches_aggregate(response, stored, atol=1e-9)
    if aggregate_diff > 1e-9:
        raise AssertionError(f"OPS-28-2 aggregate rise check failed for {family}: {aggregate_diff}")
    token_counts = _load_pl_token_counts(cfg, family, repo, main_ids)
    response = response.loc[
        response["concept_id"].astype(str).isin(set(covars["concept_id"].astype(str)))
    ].copy()
    response = response.merge(
        covars.drop(columns="stratum"), on="concept_id", how="inner", validate="many_to_one"
    ).merge(
        token_counts, left_on=["concept_id", "format", "target_lang"],
        right_on=["concept_id", "format", "target_lang"], how="left", validate="many_to_one",
    )
    response["language"] = response["target_lang"].astype(str)
    response["task_format"] = response["format"].astype(str)
    response["depth"] = response["layer"].astype(float) / _family_n_layers(cfg, family)
    response["concept_group"] = response["concept_id"].astype(str)
    model_keys = ["concept_id", "format", "target_lang", "layer"]
    format_counts = _format_concept_counts(covars, _primary_format_prompt_ids(cfg, repo))
    expected_by_format, expected_rows = e6.expected_s71_row_grid(
        format_counts, n_layers=_family_n_layers(cfg, family)
    )
    observed_by_format = {
        fmt: int(response["format"].astype(str).eq(fmt).sum()) for fmt in FORMATS
    }
    if response.duplicated(model_keys).any() or len(response) != expected_rows or observed_by_format != expected_by_format:
        raise ValueError(
            f"§7.1 complete-case row grid differs for {family}: "
            f"observed={len(response)} expected={expected_rows}; "
            f"observed_by_format={observed_by_format}; expected_by_format={expected_by_format}"
        )
    if response[["n_tokens", "zipf_en", "n_syllables", "concreteness"]].isna().any().any():
        raise ValueError(f"§7.1 required covariates/token counts are missing for {family}")
    return response, stored, covars, token_counts, max_diff


def _load_la_zh(cfg: Mapping[str, Any], repo: Path, family: str) -> pd.DataFrame:
    """Read primary per-concept LA_zh values for D_zh, projected to required columns."""
    metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
    ids = _analysis_main_ids(metadata)
    frame = _load_family_table(repo, family, "la.parquet", LA_COLUMNS)
    frame = _select_zh_la_rows(frame)
    frame = frame.loc[frame["concept_id"].astype(str).isin(ids)].copy()
    frame = frame.merge(metadata.loc[:, ["concept_id", "stratum"]], on="concept_id", how="inner", validate="many_to_one")
    frame = frame.loc[frame["stratum"].astype(str).isin(("sino", "nonsino"))].copy()
    return frame


def _load_passing_la_zh(cfg: Mapping[str, Any], repo: Path, family: str) -> pd.DataFrame:
    """Load LA_zh and verify its layer set against the committed E5 gate table."""
    models = study_models(dict(cfg))
    la = _load_la_zh(cfg, repo, family)
    failed = read_e5_gate_layers(repo / "reports/E2-E5_e5.csv", models[family]["hf_id"])
    validate_la_gate_layers(la, failed, n_layers=_family_n_layers(cfg, family))
    return la


def _assert_matching_committed(repo: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Require the covariate-only match files to be tracked and clean before outcomes."""
    import subprocess

    relative_paths = [str(cfg["e6"]["outputs"]["matched_pairs"]), str(cfg["e6"]["outputs"]["balance"])]
    for relative in relative_paths:
        path = repo / relative
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=repo, capture_output=True, text=True)
        if tracked.returncode != 0:
            raise RuntimeError("E6 matched IDs and balance must be committed before reading outcome tables")
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=repo, check=False)
        if clean.returncode != 0:
            raise RuntimeError("E6 matched ID/balance files have uncommitted changes")
    return pd.read_csv(repo / relative_paths[0], dtype=str)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a summary CSV after enforcing the no-raw-concreteness rule."""
    forbidden = {"concreteness", "concreteness_mean", "mean_concreteness"}
    if forbidden & set(frame.columns):
        raise ValueError("E6 report tables may contain only counts/SMDs, not concreteness values")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_match(cfg: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Build and write the covariate-only matched pair list; open only concepts.tsv."""
    tsv = repo / str(cfg["paths"]["concepts_tsv"])
    concepts = e6.read_matching_covariates(str(tsv), repo_root=str(repo))
    pairs, balance, unmatched = e6.match_without_replacement(
        concepts, seed=int(cfg["seed"]), stream_key=str(cfg["e6"]["match_stream"]),
        caliper=float(cfg["e6"]["caliper"]), expected_counts=cfg["e6"]["complete_case_counts"],
    )
    out = cfg["e6"]["outputs"]
    pair_path = repo / str(out["matched_pairs"])
    balance_path = repo / str(out["balance"])
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(pair_path, index=False)
    balance.to_csv(balance_path, index=False)
    return {"pairs": int(len(pairs)), "unmatched_sino": int(unmatched), "balance": balance, "pair_path": pair_path, "balance_path": balance_path}


def _strata_ids(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Build stable concept-ID lists for the two frozen strata."""
    return {
        stratum: sorted(frame.loc[frame["stratum"].astype(str).eq(stratum), "concept_id"].astype(str).unique())
        for stratum in ("sino", "nonsino")
    }


def run_dzh(
    cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-set D_zh point/CI rows and Holm-family bootstrap p inputs."""
    out_rows: list[pd.DataFrame] = []
    p_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        began = time.perf_counter()
        la = _load_passing_la_zh(cfg, repo, family)
        n_layers = _family_n_layers(cfg, family)
        ids = _strata_ids(la.drop_duplicates("concept_id"))
        key = str(cfg["e6"]["dzh_key_template"]).format(model=family)
        draws = e6.stratified_draws(ids, resamples=int(cfg["e6"]["bootstrap_resamples"]), seed=int(cfg["seed"]), key=key)
        chunk_size = int(cfg["e6"]["bootstrap_chunk"])
        store = BootstrapChunkStore(cfg, repo, run_git, family, "dzh", key)
        table, _samples = e6.dzh_from_frame(
            la, n_layers=n_layers, window=cfg["e6"]["window"], draws=draws,
            chunk_size=chunk_size, completed=store.load_values(len(draws), chunk_size),
            on_chunk=store.save_values,
        )
        table["model"] = family
        table["bootstrap_key"] = key
        table["result_class"] = [
            e6.interval_class(float(row.ci_low), float(row.ci_high))
            for row in table.itertuples(index=False)
        ]
        out_rows.append(table)
        pooled = table.loc[table["format"].astype(str).eq("pooled")].iloc[0]
        p_rows.append({"model": family, "instrument": "D_zh", "p_value": float(pooled["p_value"])})
        logger.info("stage=dzh family=%s wall_seconds=%.3f peak_rss_bytes=%d", family, time.perf_counter() - began, peak_rss_bytes())
    return pd.concat(out_rows, ignore_index=True), pd.DataFrame.from_records(p_rows)


def run_model(
    cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fit full §7.1 models, bootstrap the interaction, and return Holm input rows."""
    term_rows: list[dict[str, Any]] = []
    holm_rows: list[dict[str, Any]] = []
    diffs: dict[str, float] = {}
    models = study_models(dict(cfg))
    for family in FAMILIES:
        began = time.perf_counter()
        frame, _stored, covars, _tokens, max_diff = _load_response_inputs(cfg, repo, family)
        diffs[family] = max_diff
        full_fit = e6.fit_mixedlm(
            frame,
            spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]),
            fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
        )
        instrument = e6.model_instrument(full_fit)
        common = {
            "model": family, "instrument": "rise_zh_interaction", "term": instrument["term"],
            "estimate": instrument["estimate"], "wald_ci_low": instrument["wald_ci_low"],
            "wald_ci_high": instrument["wald_ci_high"], "standard_error": instrument["standard_error"],
            "converged": instrument["converged"], "fit_valid": instrument["valid"],
            "fit_method": instrument["method_used"], "fit_tries": instrument["tries"],
            "variance_at_boundary": instrument["variance_at_boundary"],
            "boundary_rule_ran": instrument["boundary_rule_ran"],
            "boundary_rule_changed_fit": instrument["boundary_rule_changed_fit"],
            "llf_kept": instrument["llf_kept"], "llf_first_valid": instrument["llf_first_valid"],
            "random_intercept_variance": instrument["random_intercept_variance"],
            "fit_warning_categories": json.dumps(instrument["warning_categories"], sort_keys=True),
            "fit_failure_reasons": json.dumps(instrument["failure_reasons"], sort_keys=True),
            "full_fit_seconds": instrument["fit_seconds"],
            "zh_specific_stratum_effect": instrument["zh_specific_stratum_effect"],
            "n_sino": int(covars["stratum"].astype(str).eq("sino").sum()),
            "n_nonsino": int(covars["stratum"].astype(str).eq("nonsino").sum()),
            "bootstrap_key": str(cfg["e6"]["model_key_template"]).format(model=family),
            "rise_main_max_abs_diff": max_diff,
        }
        if not instrument["valid"]:
            common.update({
                "bootstrap_ci_low": float("nan"), "bootstrap_ci_high": float("nan"),
                "bootstrap_p": float("nan"), "result_class": "not estimable",
                "valid_draws": 0, "nonconverged_draws": 0,
                "valid_draw_gate_passed": False, "bootstrap_method_mix": "{}",
                "bootstrap_boundary_count": 0, "bootstrap_warning_categories": "{}",
                "bootstrap_boundary_rule_trigger_count": 0,
                "bootstrap_boundary_rule_changed_fit_count": 0,
            })
            term_rows.append(common)
            holm_rows.append({"model": family, "instrument": "rise_zh_interaction", "p_value": float("nan")})
            logger.warning(
                "stage=model family=%s status=not_estimable full_fit_tries=%d wall_seconds=%.3f peak_rss_bytes=%d",
                family, instrument["tries"], time.perf_counter() - began, peak_rss_bytes(),
            )
            continue
        ids = _strata_ids(frame.drop_duplicates("concept_id"))
        key = common["bootstrap_key"]
        draws = e6.stratified_draws(ids, resamples=int(cfg["e6"]["bootstrap_resamples"]), seed=int(cfg["seed"]), key=key)
        chunk_size = int(cfg["e6"]["bootstrap_chunk"])
        store = BootstrapChunkStore(cfg, repo, run_git, family, "model", key)
        samples, valid, fit_records = e6.model_bootstrap(
            frame, draws, workers=int(cfg["e6"]["workers"]), start_params=full_fit.params.to_numpy(dtype=np.float64),
            spline_df=int(cfg["e6"]["spline_df"]), maxiter=int(cfg["e6"]["maxiter"]),
            fit_methods=cfg["e6"]["fit_methods"], chunk_size=chunk_size,
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            completed=store.load(len(draws), chunk_size), on_chunk=store.save,
        )
        summary = e6.bootstrap_summary(samples, min_valid_draws=int(cfg["e6"]["min_valid_draws"]))
        fit_summary = e6.fit_diagnostics_summary(fit_records)
        common.update({
            "bootstrap_ci_low": summary["ci_low"], "bootstrap_ci_high": summary["ci_high"],
            "bootstrap_p": summary["p_value"], "result_class": summary["result_class"],
            "valid_draws": summary["valid_draws"], "nonconverged_draws": summary["nonconverged_draws"],
            "valid_draw_gate_passed": summary["valid_draw_gate_passed"],
            "bootstrap_method_mix": json.dumps(fit_summary["method_mix"], sort_keys=True),
            "bootstrap_boundary_count": fit_summary["boundary_count"],
            "bootstrap_boundary_rule_trigger_count": fit_summary["boundary_rule_trigger_count"],
            "bootstrap_boundary_rule_changed_fit_count": fit_summary["boundary_rule_changed_fit_count"],
            "bootstrap_warning_categories": json.dumps(fit_summary["warning_categories"], sort_keys=True),
            "bootstrap_mean_fit_seconds": fit_summary["mean_fit_seconds"],
            "bootstrap_mean_tries": fit_summary["mean_tries"],
        })
        term_rows.append(common)
        holm_rows.append({"model": family, "instrument": "rise_zh_interaction", "p_value": summary["p_value"]})
        logger.info(
            "stage=model family=%s fit_valid=%s method=%s tries=%d boundary=%s bootstrap_valid=%d/%d "
            "gate=%s method_mix=%s boundary_count=%d boundary_rule_triggers=%d boundary_rule_changes=%d "
            "warnings=%s wall_seconds=%.3f peak_rss_bytes=%d",
            family, instrument["valid"], instrument["method_used"], instrument["tries"],
            instrument["variance_at_boundary"], summary["valid_draws"], int(cfg["e6"]["bootstrap_resamples"]),
            summary["valid_draw_gate_passed"], fit_summary["method_mix"], fit_summary["boundary_count"],
            fit_summary["boundary_rule_trigger_count"], fit_summary["boundary_rule_changed_fit_count"],
            fit_summary["warning_categories"], time.perf_counter() - began, peak_rss_bytes(),
        )
    return pd.DataFrame.from_records(term_rows), pd.DataFrame.from_records(holm_rows), diffs


def run_matched(cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit) -> pd.DataFrame:
    """Recompute D_zh and the §7.1 term on the committed matched pairs."""
    pairs = _assert_matching_committed(repo, cfg)
    full_dzh_path = repo / cfg["e6"]["outputs"]["dzh"]
    full_terms_path = repo / cfg["e6"]["outputs"]["model_terms"]
    if not full_dzh_path.is_file() or not full_terms_path.is_file():
        raise FileNotFoundError("run full D_zh and model stages before matched estimates")
    full_dzh = pd.read_csv(full_dzh_path)
    full_terms = pd.read_csv(full_terms_path)
    full_classes: dict[tuple[str, str], str] = {}
    for row in full_dzh.loc[full_dzh["format"].astype(str).eq("pooled")].itertuples(index=False):
        full_classes[(str(row.model), "D_zh")] = str(row.result_class)
    for row in full_terms.itertuples(index=False):
        full_classes[(str(row.model), "rise_zh_interaction")] = (
            "not estimable" if getattr(row, "result_class", "") == "not estimable"
            else e6.interval_class(float(row.bootstrap_ci_low), float(row.bootstrap_ci_high))
        )
    models = study_models(dict(cfg))
    output: list[dict[str, Any]] = []
    for family in FAMILIES:
        began = time.perf_counter()
        la = _load_passing_la_zh(cfg, repo, family)
        pair_map = pairs.copy()
        sino = pair_map.loc[:, ["pair_id", "sino_concept_id"]].rename(columns={"sino_concept_id": "concept_id"}).assign(stratum="sino")
        nonsino = pair_map.loc[:, ["pair_id", "nonsino_concept_id"]].rename(columns={"nonsino_concept_id": "concept_id"}).assign(stratum="nonsino")
        pair_ids = pd.concat([sino, nonsino], ignore_index=True)
        pair_values = _matched_dzh_pair_differences(
            la, pair_ids, n_layers=_family_n_layers(cfg, family), window=cfg["e6"]["window"]
        )
        format_points = pair_values.loc[np.isfinite(pd.to_numeric(pair_values["difference"], errors="coerce"))].groupby(
            "format", sort=True
        )["difference"].mean()
        if not set(FORMATS).issubset(format_points.index.astype(str)):
            raise ValueError(f"matched D_zh has no complete pairs for every format in {family}")
        key = str(cfg["e6"]["matched_key_template"]).format(model=family)
        chunk_size = int(cfg["e6"]["bootstrap_chunk"])
        dzh_store = BootstrapChunkStore(cfg, repo, run_git, family, "matched_dzh", key)
        dzh_draws = e6.dzh_pair_bootstrap(
            pair_values, resamples=int(cfg["e6"]["bootstrap_resamples"]), seed=int(cfg["seed"]), key=key,
            chunk_size=chunk_size, completed=dzh_store.load_values(int(cfg["e6"]["bootstrap_resamples"]), chunk_size),
            on_chunk=dzh_store.save_values,
        )
        point = float(format_points.reindex(FORMATS).mean())
        low, high = e6.percentile_interval(dzh_draws)
        output.append({"model": family, "instrument": "D_zh", "estimate": point, "ci_low": low, "ci_high": high, "valid_draws": int(np.isfinite(dzh_draws).sum()), "bootstrap_key": key, "result_class": e6.interval_class(low, high)})

        response, _stored, _covars, _tokens, _diff = _load_response_inputs(cfg, repo, family)
        matched_response = response.merge(pair_ids, on=["concept_id", "stratum"], how="inner", validate="many_to_one")
        matched_response["concept_group"] = matched_response["concept_id"].astype(str)
        full_fit = e6.fit_mixedlm(
            matched_response,
            spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]),
            fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
        )
        instrument = e6.model_instrument(full_fit)
        if not instrument["valid"]:
            output.append({
                "model": family, "instrument": "rise_zh_interaction",
                "estimate": float("nan"), "standard_error": float("nan"),
                "wald_ci_low": float("nan"), "wald_ci_high": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "valid_draws": 0, "nonconverged_draws": 0, "valid_draw_gate_passed": False,
                "bootstrap_method_mix": "{}", "bootstrap_boundary_count": 0,
                "bootstrap_warning_categories": "{}", "fit_valid": False,
                "fit_method": instrument["method_used"], "fit_tries": instrument["tries"],
                "variance_at_boundary": instrument["variance_at_boundary"],
                "random_intercept_variance": instrument["random_intercept_variance"],
                "boundary_rule_ran": instrument["boundary_rule_ran"],
                "boundary_rule_changed_fit": instrument["boundary_rule_changed_fit"],
                "llf_kept": instrument["llf_kept"], "llf_first_valid": instrument["llf_first_valid"],
                "bootstrap_boundary_rule_trigger_count": 0,
                "bootstrap_boundary_rule_changed_fit_count": 0,
                "fit_warning_categories": json.dumps(instrument["warning_categories"], sort_keys=True),
                "fit_failure_reasons": json.dumps(instrument["failure_reasons"], sort_keys=True),
                "bootstrap_key": key, "result_class": "not estimable",
            })
            logger.warning(
                "stage=matched family=%s status=not_estimable full_fit_tries=%d",
                family, instrument["tries"],
            )
            continue
        pair_draws = e6.match_bootstrap_draws(pairs, resamples=int(cfg["e6"]["bootstrap_resamples"]), seed=int(cfg["seed"]), key=key)
        model_store = BootstrapChunkStore(cfg, repo, run_git, family, "matched_model", key)
        model_samples, valid, fit_records = _matched_model_bootstrap(
            matched_response, pairs, pair_draws, workers=int(cfg["e6"]["workers"]),
            start_params=full_fit.params.to_numpy(dtype=np.float64), spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]), fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            chunk_size=chunk_size, completed=model_store.load(len(pair_draws), chunk_size),
            on_chunk=model_store.save,
        )
        summary = e6.bootstrap_summary(model_samples, min_valid_draws=int(cfg["e6"]["min_valid_draws"]))
        fit_summary = e6.fit_diagnostics_summary(fit_records)
        output.append({
            "model": family, "instrument": "rise_zh_interaction", "estimate": instrument["estimate"],
            "standard_error": instrument["standard_error"],
            "wald_ci_low": instrument["wald_ci_low"], "wald_ci_high": instrument["wald_ci_high"],
            "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            "valid_draws": summary["valid_draws"], "nonconverged_draws": summary["nonconverged_draws"],
            "valid_draw_gate_passed": summary["valid_draw_gate_passed"],
            "bootstrap_method_mix": json.dumps(fit_summary["method_mix"], sort_keys=True),
            "bootstrap_boundary_count": fit_summary["boundary_count"],
            "bootstrap_boundary_rule_trigger_count": fit_summary["boundary_rule_trigger_count"],
            "bootstrap_boundary_rule_changed_fit_count": fit_summary["boundary_rule_changed_fit_count"],
            "bootstrap_warning_categories": json.dumps(fit_summary["warning_categories"], sort_keys=True),
            "fit_valid": instrument["valid"], "fit_method": instrument["method_used"],
            "fit_tries": instrument["tries"], "variance_at_boundary": instrument["variance_at_boundary"],
            "random_intercept_variance": instrument["random_intercept_variance"],
            "boundary_rule_ran": instrument["boundary_rule_ran"],
            "boundary_rule_changed_fit": instrument["boundary_rule_changed_fit"],
            "llf_kept": instrument["llf_kept"], "llf_first_valid": instrument["llf_first_valid"],
            "fit_warning_categories": json.dumps(instrument["warning_categories"], sort_keys=True),
            "fit_failure_reasons": json.dumps(instrument["failure_reasons"], sort_keys=True),
            "bootstrap_key": key, "result_class": summary["result_class"],
        })
        logger.info(
            "stage=matched family=%s fit_valid=%s method=%s tries=%d bootstrap_valid=%d/%d gate=%s "
            "method_mix=%s boundary_count=%d boundary_rule_triggers=%d boundary_rule_changes=%d "
            "warnings=%s wall_seconds=%.3f peak_rss_bytes=%d",
            family, instrument["valid"], instrument["method_used"], instrument["tries"],
            summary["valid_draws"], int(cfg["e6"]["bootstrap_resamples"]), summary["valid_draw_gate_passed"],
            fit_summary["method_mix"], fit_summary["boundary_count"],
            fit_summary["boundary_rule_trigger_count"], fit_summary["boundary_rule_changed_fit_count"],
            fit_summary["warning_categories"],
            time.perf_counter() - began, peak_rss_bytes(),
        )
    result = pd.DataFrame.from_records(output)
    result["full_result_class"] = [full_classes[(str(row.model), str(row.instrument))] for row in result.itertuples(index=False)]
    result["class_changed"] = result["result_class"].astype(str).ne(result["full_result_class"].astype(str))
    result["H3_unresolved"] = bool(result["class_changed"].any())
    return result


def _matched_dzh_pair_differences(
    la: pd.DataFrame,
    pair_ids: pd.DataFrame,
    *,
    n_layers: int,
    window: Sequence[float],
) -> pd.DataFrame:
    """Build per-pair per-format/per-layer differences for matched bootstrap."""
    layer = pd.to_numeric(la["layer"], errors="raise").astype(int)
    depth = layer / int(n_layers)
    selected = la.loc[layer.ge(1) & depth.between(float(window[0]), float(window[1]), inclusive="both")].copy()
    joined = selected.merge(pair_ids, on=["concept_id", "stratum"], how="inner", validate="many_to_one")
    pivot = joined.pivot(index=["pair_id", "format", "layer"], columns="stratum", values="value").reset_index()
    if not {"sino", "nonsino"}.issubset(pivot.columns):
        raise ValueError("matched D_zh requires both members of each pair")
    pivot["difference"] = pivot["sino"] - pivot["nonsino"]
    return pivot.loc[:, ["pair_id", "format", "layer", "difference"]]


def _matched_model_bootstrap(
    frame: pd.DataFrame, pairs: pd.DataFrame, draws: Sequence[Sequence[str]], *, workers: int,
    start_params: Sequence[float], spline_df: int, maxiter: int, fit_methods: Sequence[str], chunk_size: int,
    boundary_rule: bool = True, llf_tie_tol: float = 1.0e-6,
    completed: Mapping[int, tuple[np.ndarray, ...]] | None = None,
    on_chunk: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Warm-start matched refits with per-draw diagnostics and distinct pair groups."""
    from concurrent.futures import ProcessPoolExecutor
    samples = np.full(len(draws), np.nan, dtype=np.float64)
    valid = np.zeros(len(draws), dtype=bool)
    diagnostics = [e6.empty_fit_diagnostics() for _ in draws]
    completed = completed or {}
    chunk_size = max(1, int(chunk_size))
    executor = ProcessPoolExecutor(max_workers=int(workers)) if int(workers) > 1 else None
    pair_index = pairs.set_index("pair_id")
    frame_without_group = frame.drop(columns="concept_group", errors="ignore")
    try:
        for chunk_index, begin in enumerate(range(0, len(draws), chunk_size)):
            stop = min(begin + chunk_size, len(draws))
            cached = completed.get(chunk_index)
            if cached is not None:
                values = np.asarray(cached[0], dtype=np.float64)
                flags = np.asarray(cached[1], dtype=bool)
                arrays = [np.asarray(value) for value in cached[2:]]
                if values.shape != (stop - begin,) or flags.shape != (stop - begin,) or any(array.shape != (stop - begin,) for array in arrays):
                    raise ValueError(f"cached matched model chunk {chunk_index} has the wrong row count")
                samples[begin:stop] = values
                valid[begin:stop] = flags
                for offset in range(stop - begin):
                    diagnostics[begin + offset] = {
                        "valid": bool(flags[offset]),
                        "converged": bool(flags[offset]),
                        "method_used": str(arrays[0][offset]),
                        "tries": int(arrays[1][offset]),
                        "variance_at_boundary": bool(arrays[2][offset]),
                        "warning_categories": json.loads(str(arrays[3][offset])),
                        "failure_reasons": json.loads(str(arrays[4][offset])),
                        "fit_seconds": float(arrays[5][offset]),
                        "boundary_rule_ran": bool(arrays[6][offset]),
                        "boundary_rule_changed_fit": bool(arrays[7][offset]),
                        "llf_kept": e6._optional_float(float(arrays[8][offset])),
                        "llf_first_valid": e6._optional_float(float(arrays[9][offset])),
                        "random_intercept_variance": e6._optional_float(float(arrays[10][offset])),
                    }
                continue
            for micro_begin in range(begin, stop, max(1, int(workers))):
                micro_stop = min(micro_begin + max(1, int(workers)), stop)
                payloads = []
                for draw_index in range(micro_begin, micro_stop):
                    sample_rows = []
                    for index, pair_id in enumerate(draws[draw_index]):
                        pair = pair_index.loc[str(pair_id)]
                        sample_rows.extend([
                            {"concept_id": str(pair["sino_concept_id"]), "stratum": "sino", "concept_group": f"p{index}s"},
                            {"concept_id": str(pair["nonsino_concept_id"]), "stratum": "nonsino", "concept_group": f"p{index}n"},
                        ])
                    expanded = pd.DataFrame(sample_rows).merge(
                        frame_without_group, on=["concept_id", "stratum"], how="inner", validate="many_to_many",
                    )
                    payloads.append((
                        expanded, np.asarray(start_params), int(spline_df), "zipf_en", int(maxiter),
                        tuple(fit_methods), bool(boundary_rule), float(llf_tie_tol),
                    ))
                if executor is None:
                    current = [e6._fit_model_payload(payload) for payload in payloads]
                else:
                    current = list(executor.map(e6._fit_model_payload, payloads, chunksize=1))
                for index, (estimate, fit_diag) in enumerate(current, start=micro_begin):
                    valid[index] = bool(fit_diag["valid"])
                    samples[index] = float(estimate) if valid[index] else np.nan
                    diagnostics[index] = fit_diag
            if on_chunk is not None:
                on_chunk(chunk_index, begin, samples[begin:stop].copy(), valid[begin:stop].copy(), diagnostics[begin:stop])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    samples[~valid] = np.nan
    return samples, valid, diagnostics


def run_sensitivity(
    cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit
) -> pd.DataFrame:
    """Run the eight named OPS-28 sensitivity variants without Holm adjustment."""
    rows: list[dict[str, Any]] = []
    models = study_models(dict(cfg))
    sensitivity_names = tuple(map(str, cfg["e6"]["sensitivity_variants"]))
    for family in FAMILIES:
        started = time.perf_counter()
        response, _stored, covars, _tokens, _diff = _load_response_inputs(cfg, repo, family)
        la = _load_passing_la_zh(cfg, repo, family)
        metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
        for name in sensitivity_names:
            frequency = name if name in {"zipf_vi", "zipf_zh"} else "zipf_en"
            selected = response.copy()
            selected_la = la.copy()
            selected = selected.dropna(subset=[frequency]).copy()
            if selected.empty:
                raise ValueError(f"empty complete-case §7.1 sensitivity for {family}/{name}")
            if name == "native_strict":
                native_ids = set(metadata.loc[
                    e6._bool_values(metadata["native_strict"])
                    & metadata["stratum"].astype(str).eq("nonsino"),
                    "concept_id",
                ].astype(str))
                selected = selected.loc[selected["stratum"].eq("sino") | selected["concept_id"].astype(str).isin(native_ids)].copy()
                selected_la = selected_la.loc[selected_la["stratum"].eq("sino") | selected_la["concept_id"].astype(str).isin(native_ids)].copy()
                selected.loc[selected["concept_id"].astype(str).isin(native_ids), "stratum"] = "nonsino"
                selected_la.loc[selected_la["concept_id"].astype(str).isin(native_ids), "stratum"] = "nonsino"
            elif name in SENSITIVITY_MASKS and name != "R11_excluded":
                mask_col = SENSITIVITY_MASKS[name]
                selected_ids = set(metadata.loc[e6._bool_values(metadata[mask_col]), "concept_id"].astype(str))
                selected = selected.loc[selected["concept_id"].astype(str).isin(selected_ids)].copy()
                selected_la = selected_la.loc[selected_la["concept_id"].astype(str).isin(selected_ids)].copy()
            elif name == "R11_excluded":
                excluded = set(map(str, cfg["e6"]["r11_concept_ids"]))
                selected = selected.loc[~selected["concept_id"].astype(str).isin(excluded)].copy()
                selected_la = selected_la.loc[~selected_la["concept_id"].astype(str).isin(excluded)].copy()
            fitted = e6.fit_mixedlm(
                selected, spline_df=int(cfg["e6"]["spline_df"]), frequency_covariate=frequency,
                maxiter=int(cfg["e6"]["maxiter"]), fit_methods=cfg["e6"]["fit_methods"],
                boundary_rule=cfg["e6"]["boundary_rule"],
                llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            )
            term = e6.model_instrument(fitted)
            rows.append({
                "model": family, "sensitivity": name, "instrument": "rise_zh_interaction",
                "estimate": term["estimate"], "ci_low": term["wald_ci_low"], "ci_high": term["wald_ci_high"],
                "standard_error": term["standard_error"], "converged": term["converged"],
                "fit_valid": term["valid"], "fit_method": term["method_used"], "fit_tries": term["tries"],
                "variance_at_boundary": term["variance_at_boundary"],
                "random_intercept_variance": term["random_intercept_variance"],
                "boundary_rule_ran": term["boundary_rule_ran"],
                "boundary_rule_changed_fit": term["boundary_rule_changed_fit"],
                "llf_kept": term["llf_kept"], "llf_first_valid": term["llf_first_valid"],
                "fit_warning_categories": json.dumps(term["warning_categories"], sort_keys=True),
                "fit_failure_reasons": json.dumps(term["failure_reasons"], sort_keys=True),
                "n_rows": len(selected),
            })
            if name not in {"zipf_vi", "zipf_zh"}:
                ids = _strata_ids(selected_la.drop_duplicates("concept_id"))
                key = f"bootstrap|h3_sensitivity_{name}|{family}|pooled"
                draws = e6.stratified_draws(ids, resamples=int(cfg["e6"]["bootstrap_resamples"]), seed=int(cfg["seed"]), key=key)
                chunk_size = int(cfg["e6"]["bootstrap_chunk"])
                store = BootstrapChunkStore(cfg, repo, run_git, family, f"sensitivity_{name}", key)
                dzh, _samples = e6.dzh_from_frame(
                    selected_la, n_layers=_family_n_layers(cfg, family), window=cfg["e6"]["window"], draws=draws,
                    chunk_size=chunk_size, completed=store.load_values(len(draws), chunk_size),
                    on_chunk=store.save_values,
                )
                dzh_row = dzh.loc[dzh["format"].eq("pooled")].iloc[0]
                rows.append({"model": family, "sensitivity": name, "instrument": "D_zh", "estimate": dzh_row["estimate"], "ci_low": dzh_row["ci_low"], "ci_high": dzh_row["ci_high"], "valid_draws": dzh_row["valid_draws"], "p_value": dzh_row["p_value"], "bootstrap_key": key})
        logger.info("stage=sensitivity family=%s fits=%d wall_seconds=%.3f peak_rss_bytes=%d", family, len(sensitivity_names), time.perf_counter()-started, peak_rss_bytes())
    return pd.DataFrame.from_records(rows)


def ensure_closeout_outputs_absent(closeout_dir: Path, report_path: Path) -> None:
    """Refuse to replace any result from an earlier closeout attempt."""
    if closeout_dir.exists() and any(closeout_dir.iterdir()):
        raise FileExistsError(f"E6 closeout output directory is not empty: {closeout_dir}")
    if report_path.exists():
        raise FileExistsError(f"E6 closeout report already exists: {report_path}")


def _sensitivity_model_frames(
    cfg: Mapping[str, Any], repo: Path, family: str, response: pd.DataFrame
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield one §7.1 sensitivity row set at a time, without bootstraps."""
    metadata = _main_strata_metadata(repo / str(cfg["paths"]["concepts_tsv"]))
    for name in map(str, cfg["e6"]["sensitivity_variants"]):
        frequency = name if name in {"zipf_vi", "zipf_zh"} else "zipf_en"
        selected = response.copy().dropna(subset=[frequency]).copy()
        if name == "native_strict":
            native_ids = set(metadata.loc[
                e6._bool_values(metadata["native_strict"])
                & metadata["stratum"].astype(str).eq("nonsino"), "concept_id",
            ].astype(str))
            selected = selected.loc[
                selected["stratum"].eq("sino")
                | selected["concept_id"].astype(str).isin(native_ids)
            ].copy()
            selected.loc[selected["concept_id"].astype(str).isin(native_ids), "stratum"] = "nonsino"
        elif name in SENSITIVITY_MASKS and name != "R11_excluded":
            mask_col = SENSITIVITY_MASKS[name]
            selected_ids = set(metadata.loc[e6._bool_values(metadata[mask_col]), "concept_id"].astype(str))
            selected = selected.loc[selected["concept_id"].astype(str).isin(selected_ids)].copy()
        elif name == "R11_excluded":
            excluded = set(map(str, cfg["e6"]["r11_concept_ids"]))
            selected = selected.loc[~selected["concept_id"].astype(str).isin(excluded)].copy()
        if selected.empty:
            raise ValueError(f"empty complete-case §7.1 sensitivity for {family}/{name}")
        selected["concept_group"] = selected["concept_id"].astype(str)
        yield name, frequency, selected
        # The consumer has completed this fit before iteration resumes.  Drop
        # the yielded frame before constructing the next full-sized variant.
        del selected


def _iter_closeout_fit_sets(
    cfg: Mapping[str, Any], repo: Path, family: str, response: pd.DataFrame,
    pair_ids: pd.DataFrame, family_sensitivity: pd.DataFrame,
    full_kept: pd.Series, matched_kept: pd.Series,
) -> Iterator[tuple[str, str, pd.DataFrame, Any]]:
    """Yield the full, matched and sensitivity data frames one at a time."""
    yield "full", "zipf_en", response, full_kept
    matched = response.merge(pair_ids, on=["concept_id", "stratum"], how="inner", validate="many_to_one")
    matched["concept_group"] = matched["concept_id"].astype(str)
    yield "matched", "zipf_en", matched, matched_kept
    del matched
    for name, frequency, selected in _sensitivity_model_frames(cfg, repo, family, response):
        yield f"sensitivity:{name}", frequency, selected, family_sensitivity.loc[name]


def _closeout_serial_map(
    items: Iterable[Any], process_one: Callable[[Any], Any], *, workers: int
) -> list[Any]:
    """Process closeout units in order, releasing each input before advancing."""
    if workers != 1:
        raise ValueError("closeout currently requires closeout_workers=1")
    results: list[Any] = []
    for item in items:
        results.append(process_one(item))
        del item
        gc.collect()
    return results


def _fit_generalized_moment_candidate(
    frame: pd.DataFrame, *, frequency_covariate: str, maxiter: int
) -> dict[str, Any]:
    """Run the OPS-34 unequal-group moment-start candidate with lbfgs only."""
    import statsmodels.formula.api as smf

    formula = e6.mixedlm_formula(frequency_covariate=frequency_covariate)
    model = smf.mixedlm(formula, data=frame, groups=frame["concept_group"], re_formula="1")
    started = time.perf_counter()
    warning_categories: dict[str, int] = {}
    try:
        start = e6.generalized_moment_start(model, frame, formula)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(
                reml=True, method="lbfgs", start_params=start,
                maxiter=int(maxiter), disp=False,
            )
        for item in caught:
            category = e6._warning_category(item.category.__name__, str(item.message))
            warning_categories[category] = warning_categories.get(category, 0) + 1
        valid, converged, variance, boundary, reasons = e6._mixedlm_candidate_validity(result)
        llf = e6._finite_llf(result)
        valid = bool(valid and llf is not None)
        return {
            "valid": valid, "converged": converged, "method": "lbfgs_mom_unequal",
            "variance_at_boundary": boundary, "random_intercept_variance": variance,
            "llf_mom": llf, "failure_reasons": reasons,
            "warning_categories": dict(sorted(warning_categories.items())),
            "seconds": time.perf_counter() - started, "result": result,
        }
    except Exception as exc:
        return {
            "valid": False, "converged": False, "method": "lbfgs_mom_unequal",
            "variance_at_boundary": False, "random_intercept_variance": None,
            "llf_mom": None, "failure_reasons": [f"{type(exc).__name__}: {exc}"],
            "warning_categories": dict(sorted(warning_categories.items())),
            "seconds": time.perf_counter() - started, "result": None,
        }


def ensure_moment_refit_outputs_absent(
    result_path: Path, report_path: Path, sidecar_path: Path
) -> None:
    """Refuse to overwrite any prior OPS-36 moment-refit output."""
    existing = [path for path in (result_path, report_path, sidecar_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "E6 moment-refit outputs already exist: " + ", ".join(map(str, existing))
        )


def _moment_refit_llf_matches(actual: float | None, expected: float, *, tolerance: float = 1.0e-6) -> bool:
    """Return whether the reproduced OPS-34 moment-start likelihood is within tolerance."""
    return actual is not None and np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= float(tolerance)


def _moment_refit_term_estimates(result: Any) -> dict[str, float | None]:
    """Extract the OPS-36 interaction and sum-effect estimates with Wald CIs."""
    if result is None:
        return {
            "interaction_estimate": None, "interaction_se": None,
            "interaction_wald_ci_low": None, "interaction_wald_ci_high": None,
            "sum_effect_estimate": None, "sum_effect_se": None,
            "sum_effect_wald_ci_low": None, "sum_effect_wald_ci_high": None,
        }
    params = result.fe_params
    names = list(map(str, params.index))
    interaction, sino = e6._interaction_terms(names)
    bse = result.bse_fe
    interaction_estimate = e6._term_value(params, names, interaction)
    interaction_se = e6._term_value(bse, names, interaction)
    z975 = 1.959963984540054
    if not np.isfinite(interaction_estimate) or not np.isfinite(interaction_se):
        interaction_estimate_value = interaction_se_value = None
        interaction_low = interaction_high = None
    else:
        interaction_estimate_value = float(interaction_estimate)
        interaction_se_value = float(interaction_se)
        interaction_low = interaction_estimate_value - z975 * interaction_se_value
        interaction_high = interaction_estimate_value + z975 * interaction_se_value
    sum_estimate: float | None = None
    sum_se: float | None = None
    sum_low: float | None = None
    sum_high: float | None = None
    if interaction is not None and sino is not None:
        try:
            sum_value = float(params.loc[sino] + params.loc[interaction])
            covariance = result.cov_params().loc[[sino, interaction], [sino, interaction]].to_numpy(dtype=np.float64)
            variance = float(np.ones(2, dtype=np.float64) @ covariance @ np.ones(2, dtype=np.float64))
            if np.isfinite(sum_value) and np.isfinite(variance) and variance >= 0:
                sum_estimate = sum_value
                sum_se = float(np.sqrt(variance))
                sum_low = sum_estimate - z975 * sum_se
                sum_high = sum_estimate + z975 * sum_se
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "interaction_estimate": interaction_estimate_value,
        "interaction_se": interaction_se_value,
        "interaction_wald_ci_low": interaction_low,
        "interaction_wald_ci_high": interaction_high,
        "sum_effect_estimate": sum_estimate, "sum_effect_se": sum_se,
        "sum_effect_wald_ci_low": sum_low, "sum_effect_wald_ci_high": sum_high,
    }


def _parse_moment_refit_specs(
    spec: str, *, sensitivity_variants: Sequence[str]
) -> list[tuple[str, str]]:
    """Parse and validate the family:fit-set list for OPS-36."""
    selected: list[tuple[str, str]] = []
    allowed_families = set(FAMILIES)
    for raw in str(spec).split(","):
        parts = raw.strip().split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid --moment-refit item {raw!r}; expected FAMILY:FIT_SET")
        family, fit_set = parts[0], parts[1]
        is_sensitivity = fit_set.startswith("sensitivity:")
        sensitivity_name = fit_set.split(":", 1)[1] if is_sensitivity else ""
        valid_set = fit_set in {"full", "matched"} or (
            is_sensitivity and sensitivity_name in set(map(str, sensitivity_variants))
        )
        if family not in allowed_families or not valid_set:
            raise ValueError(f"unsupported --moment-refit item {raw!r}")
        if (family, fit_set) in selected:
            raise ValueError(f"duplicate --moment-refit item {raw!r}")
        selected.append((family, fit_set))
    if not selected:
        raise ValueError("--moment-refit needs at least one FAMILY:FIT_SET item")
    return selected


def _stored_sum_effect_draws(repo: Path, cfg: Mapping[str, Any], family: str) -> np.ndarray | None:
    """Read a stored bootstrap sum-effect parameter only when job 3527 emitted it."""
    key = str(cfg["e6"]["model_key_template"]).format(model=family)
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    root = repo / str(cfg["e6"]["output_dir"]) / "checkpoints" / f"family={family}"
    candidates: list[np.ndarray] = []
    for archive in sorted(root.glob(f"model__key={key_hash}__chunk=*.npz")):
        with np.load(archive, allow_pickle=False) as data:
            for name in ("sum_effect_estimates", "zh_specific_stratum_effect"):
                if name in data.files:
                    candidates.append(np.asarray(data[name], dtype=np.float64).reshape(-1))
                    break
    if not candidates:
        return None
    joined = np.concatenate(candidates)
    return joined[np.isfinite(joined)]


def _sum_effect_wald_ci(fitted: e6.MixedLMFit) -> dict[str, Any]:
    """Return the fitted β_sino + β_sino:zh contrast and its Wald interval."""
    result = fitted.result
    if result is None or not fitted.diagnostics.valid:
        return {"estimable": False, "estimate": None, "se": None, "ci_low": None, "ci_high": None}
    names = list(map(str, result.fe_params.index))
    interaction, sino = e6._interaction_terms(names)
    if not interaction or not sino:
        return {"estimable": False, "estimate": None, "se": None, "ci_low": None, "ci_high": None}
    estimate = float(result.fe_params.loc[sino] + result.fe_params.loc[interaction])
    covariance = result.cov_params().loc[[sino, interaction], [sino, interaction]].to_numpy(dtype=np.float64)
    vector = np.ones(2, dtype=np.float64)
    variance = float(vector @ covariance @ vector)
    if not np.isfinite(estimate) or not np.isfinite(variance) or variance < 0.0:
        return {"estimable": False, "estimate": None, "se": None, "ci_low": None, "ci_high": None}
    standard_error = float(np.sqrt(variance))
    z975 = 1.959963984540054
    return {
        "estimable": True, "estimate": estimate, "se": standard_error,
        "ci_low": estimate - z975 * standard_error,
        "ci_high": estimate + z975 * standard_error,
    }


def run_closeout(
    cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit
) -> pd.DataFrame:
    """Run the outcome-preserving OPS-34 estimability and likelihood checks."""
    from vilens.manifest import sha256_file

    closeout_dir = repo / str(cfg["e6"]["closeout_dir"])
    closeout_report = repo / str(cfg["e6"]["outputs"]["closeout"])
    ensure_closeout_outputs_absent(closeout_dir, closeout_report)
    models = study_models(dict(cfg))
    model_terms_path = repo / str(cfg["e6"]["outputs"]["model_terms"])
    matched_path = repo / str(cfg["e6"]["outputs"]["matched"])
    sensitivity_path = repo / str(cfg["e6"]["outputs"]["sensitivity"])
    for path in (model_terms_path, matched_path, sensitivity_path):
        if not path.is_file():
            raise FileNotFoundError(f"required job 3527 summary is missing: {path}")
    model_terms = pd.read_csv(model_terms_path).set_index("model")
    matched_terms = pd.read_csv(matched_path)
    matched_terms = matched_terms.loc[matched_terms["instrument"].astype(str).eq("rise_zh_interaction")].set_index("model")
    sensitivities = pd.read_csv(sensitivity_path)
    sensitivities = sensitivities.loc[sensitivities["instrument"].astype(str).eq("rise_zh_interaction")]
    pairs = _assert_matching_committed(repo, cfg)
    pair_map = pairs.copy()
    pair_ids = pd.concat([
        pair_map.loc[:, ["pair_id", "sino_concept_id"]].rename(columns={"sino_concept_id": "concept_id"}).assign(stratum="sino"),
        pair_map.loc[:, ["pair_id", "nonsino_concept_id"]].rename(columns={"nonsino_concept_id": "concept_id"}).assign(stratum="nonsino"),
    ], ignore_index=True)
    records: list[dict[str, Any]] = []
    family_payloads: dict[str, dict[str, Any]] = {}
    outputs_moment_improved: list[str] = []
    non_estimable: list[str] = []
    ratio_by_family: dict[str, float | None] = {}
    closeout_workers = int(cfg["e6"]["closeout_workers"])
    for family in FAMILIES:
        family_sensitivity = sensitivities.loc[sensitivities["model"].astype(str).eq(family)].set_index("sensitivity")
        response, _stored, _covars, _tokens, _diff = _load_response_inputs(cfg, repo, family)
        del _stored, _covars, _tokens
        gc.collect()

        def process_fit(item: tuple[str, str, pd.DataFrame, Any]) -> dict[str, Any]:
            set_name, frequency, frame, kept = item
            logger.info(
                "closeout_start family=%s fit=%s rows=%d workers=%d",
                family, set_name, len(frame), closeout_workers,
            )
            design = e6.fixed_design_estimability(frame, frequency_covariate=frequency)
            if not design["interaction_estimable"] or not design["sum_effect_estimable"]:
                non_estimable.append(f"{family}/{set_name}")
            moment = _fit_generalized_moment_candidate(
                frame, frequency_covariate=frequency, maxiter=int(cfg["e6"]["maxiter"])
            )
            try:
                kept_llf = float(kept["llf_kept"])
                if not np.isfinite(kept_llf):
                    kept_llf = None
            except (TypeError, ValueError, KeyError):
                kept_llf = None
            llf_diff = (
                float(moment["llf_mom"] - kept_llf)
                if moment["valid"] and moment["llf_mom"] is not None and kept_llf is not None
                else None
            )
            if llf_diff is not None and llf_diff > 1.0e-6:
                outputs_moment_improved.append(f"{family}/{set_name}")
            record = {
                "model": family, "fit_set": set_name, "frequency_covariate": frequency,
                "n_rows": int(len(frame)), **design,
                "llf_kept": kept_llf, "llf_mom": moment["llf_mom"], "llf_mom_minus_kept": llf_diff,
                "moment_valid": bool(moment["valid"]), "moment_method": moment["method"],
                "moment_variance_at_boundary": bool(moment["variance_at_boundary"]),
                "moment_random_intercept_variance": moment["random_intercept_variance"],
                "moment_failure_reasons": list(moment["failure_reasons"]),
                "moment_warning_categories": moment["warning_categories"],
                "moment_seconds": float(moment["seconds"]),
            }
            records.append(record)
            logger.info(
                "closeout family=%s fit=%s rank=%d/%d null_space_columns=%s interaction_estimable=%s "
                "interaction_residual=%.3g sum_estimable=%s sum_residual=%.3g llf_kept=%s llf_mom=%s "
                "llf_diff=%s moment_valid=%s seconds=%.2f",
                family, set_name, design["rank"], design["column_count"], design["null_space_columns"],
                design["interaction_estimable"], design["interaction_estimability_residual"],
                design["sum_effect_estimable"], design["sum_effect_estimability_residual"],
                kept_llf, moment["llf_mom"], llf_diff, moment["valid"], moment["seconds"],
            )
            del design, moment, frame, kept
            gc.collect()
            return record

        fit_records = _closeout_serial_map(
            _iter_closeout_fit_sets(
                cfg, repo, family, response, pair_ids, family_sensitivity,
                model_terms.loc[family], matched_terms.loc[family],
            ),
            process_fit,
            workers=closeout_workers,
        )
        full_row = model_terms.loc[family]
        term_width = float(full_row["bootstrap_ci_high"] - full_row["bootstrap_ci_low"])
        wald_width = float(full_row["wald_ci_high"] - full_row["wald_ci_low"])
        ratio_by_family[family] = term_width / wald_width if np.isfinite(term_width) and np.isfinite(wald_width) and wald_width != 0 else None
        full_rank = next(item for item in fit_records if item["fit_set"] == "full")
        if not full_rank["sum_effect_estimable"]:
            sum_ci: dict[str, Any] = {"source": "not identified", "estimable": False, "estimate": None, "se": None, "ci_low": None, "ci_high": None}
        else:
            fitted = e6.fit_mixedlm(
                response, spline_df=int(cfg["e6"]["spline_df"]), maxiter=int(cfg["e6"]["maxiter"]),
                fit_methods=cfg["e6"]["fit_methods"], boundary_rule=cfg["e6"]["boundary_rule"],
                llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            )
            stored_sum_draws = _stored_sum_effect_draws(repo, cfg, family)
            if stored_sum_draws is not None and stored_sum_draws.size:
                sum_ci = {
                    "source": "bootstrap", "estimable": True,
                    "estimate": float(full_row["zh_specific_stratum_effect"]), "se": None,
                    "ci_low": float(np.quantile(stored_sum_draws, 0.025)),
                    "ci_high": float(np.quantile(stored_sum_draws, 0.975)),
                    "valid_draws": int(stored_sum_draws.size),
                }
            else:
                sum_ci = {"source": "Wald", **_sum_effect_wald_ci(fitted)}
                sum_ci["valid_draws"] = 0
            del fitted
            gc.collect()
        family_payloads[family] = {
            "family": family, "fit_rows": fit_records, "sum_effect_ci": sum_ci,
            "bootstrap_to_wald_ci_width_ratio": ratio_by_family[family],
        }
        del response, fit_records, family_sensitivity
        gc.collect()
    # Dedicated closeout destinations are write-once; all previous E6 tables remain untouched.
    closeout_dir.mkdir(parents=True, exist_ok=True)
    for family, payload in family_payloads.items():
        data_path = closeout_dir / f"family={family}.json"
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        data_path.write_text(serialized, encoding="utf-8")
        model = models[family]
        sidecar = build_provenance(
            cfg, repo_root=repo,
            unit_key={"family": family, "stage": "closeout"},
            entrypoint=repo / "scripts/analyze_e6.py",
            model_id=str(model["hf_id"]), model_revision=str(model["revision"]),
            dtype="analysis_float64", batch_size=None,
            expected_row_count=len(payload["fit_rows"]), run_git=run_git, model_loads=[],
        )
        sidecar["row_count"] = len(payload["fit_rows"])
        sidecar["output_sha256"] = sha256_file(data_path)
        sidecar["utc_end"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        (closeout_dir / f"family={family}.provenance.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = pd.DataFrame.from_records(records)
    for family, payload in family_payloads.items():
        sum_ci = payload["sum_effect_ci"]
        mask = summary["model"].astype(str).eq(family) & summary["fit_set"].eq("full")
        summary.loc[mask, "sum_effect_ci_source"] = sum_ci["source"]
        summary.loc[mask, "sum_effect_estimate"] = sum_ci["estimate"]
        summary.loc[mask, "sum_effect_ci_low"] = sum_ci["ci_low"]
        summary.loc[mask, "sum_effect_ci_high"] = sum_ci["ci_high"]
        summary.loc[mask, "sum_effect_valid_draws"] = sum_ci["valid_draws"]
        summary.loc[mask, "bootstrap_to_wald_ci_width_ratio"] = payload["bootstrap_to_wald_ci_width_ratio"]
    _write_csv(summary, closeout_report)
    logger.info(
        "closeout_summary=%s rows=%d non_estimable=%s moment_llf_improved=%s",
        closeout_report.relative_to(repo), len(summary), non_estimable, outputs_moment_improved,
    )
    return summary


def _kept_term_row(
    cfg: Mapping[str, Any], repo: Path, family: str, fit_set: str
) -> pd.Series:
    """Read job 3527's retained term row for an OPS-36 moment comparison."""
    if fit_set == "full":
        table = pd.read_csv(repo / str(cfg["e6"]["outputs"]["model_terms"]))
        selected = table.loc[
            table["model"].astype(str).eq(family)
            & table["instrument"].astype(str).eq("rise_zh_interaction")
        ]
    elif fit_set == "matched":
        table = pd.read_csv(repo / str(cfg["e6"]["outputs"]["matched"]))
        selected = table.loc[
            table["model"].astype(str).eq(family)
            & table["instrument"].astype(str).eq("rise_zh_interaction")
        ]
    else:
        sensitivity = fit_set.split(":", 1)[1]
        table = pd.read_csv(repo / str(cfg["e6"]["outputs"]["sensitivity"]))
        selected = table.loc[
            table["model"].astype(str).eq(family)
            & table["instrument"].astype(str).eq("rise_zh_interaction")
            & table["sensitivity"].astype(str).eq(sensitivity)
        ]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one job 3527 interaction row for {family}/{fit_set}")
    return selected.iloc[0]


def run_moment_refits(
    cfg: Mapping[str, Any], repo: Path, logger: logging.Logger, run_git: RunGit,
    spec: str,
) -> tuple[pd.DataFrame, bool]:
    """Reproduce selected OPS-34 generalized moment fits and write new-only outputs."""
    from vilens.manifest import sha256_file

    selected_specs = _parse_moment_refit_specs(
        spec, sensitivity_variants=cfg["e6"]["sensitivity_variants"]
    )
    result_path = repo / str(cfg["e6"]["moment_refit_json"])
    report_path = repo / str(cfg["e6"]["outputs"]["moment_refit"])
    sidecar_path = result_path.with_suffix(".provenance.json")
    ensure_moment_refit_outputs_absent(result_path, report_path, sidecar_path)

    closeout_path = repo / str(cfg["e6"]["outputs"]["closeout"])
    closeout = pd.read_csv(closeout_path)
    rows: list[dict[str, Any]] = []
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    match_pairs: pd.DataFrame | None = None
    for family, fit_set in selected_specs:
        prior = closeout.loc[
            closeout["model"].astype(str).eq(family)
            & closeout["fit_set"].astype(str).eq(fit_set)
        ]
        if len(prior) != 1 or not bool(prior.iloc[0]["moment_valid"]):
            raise ValueError(f"job 3891 has no unique valid moment candidate for {family}/{fit_set}")
        expected_llf = float(prior.iloc[0]["llf_mom"])
        kept = _kept_term_row(cfg, repo, family, fit_set)
        response, _stored, _covars, _tokens, _diff = _load_response_inputs(cfg, repo, family)
        frequency = "zipf_en"
        if fit_set == "full":
            frame = response
        elif fit_set == "matched":
            if match_pairs is None:
                match_pairs = _assert_matching_committed(repo, cfg)
            pair_ids = pd.concat([
                match_pairs.loc[:, ["pair_id", "sino_concept_id"]]
                .rename(columns={"sino_concept_id": "concept_id"}).assign(stratum="sino"),
                match_pairs.loc[:, ["pair_id", "nonsino_concept_id"]]
                .rename(columns={"nonsino_concept_id": "concept_id"}).assign(stratum="nonsino"),
            ], ignore_index=True)
            frame = response.merge(
                pair_ids, on=["concept_id", "stratum"], how="inner", validate="many_to_one"
            )
            frame["concept_group"] = frame["concept_id"].astype(str)
        else:
            wanted = fit_set.split(":", 1)[1]
            frame = None
            for name, frequency, candidate_frame in _sensitivity_model_frames(
                cfg, repo, family, response
            ):
                if name == wanted:
                    frame = candidate_frame
                    break
                del candidate_frame
            if frame is None:
                raise ValueError(f"could not resolve OPS-34 sensitivity input {family}/{fit_set}")
        logger.info(
            "moment_refit_start family=%s fit=%s rows=%d method=lbfgs_mom_unequal",
            family, fit_set, len(frame),
        )
        candidate = _fit_generalized_moment_candidate(
            frame, frequency_covariate=frequency, maxiter=int(cfg["e6"]["maxiter"])
        )
        result = candidate.pop("result")
        llf = candidate["llf_mom"]
        reproduction_diff = abs(float(llf) - expected_llf) if llf is not None else None
        reproduces = _moment_refit_llf_matches(llf, expected_llf)
        estimates = _moment_refit_term_estimates(result)
        record: dict[str, Any] = {
            "model": family, "fit_set": fit_set,
            "method": candidate["method"], "valid": bool(candidate["valid"]),
            "converged": bool(candidate["converged"]),
            "concept_variance": candidate["random_intercept_variance"],
            "llf_job3891": expected_llf, "llf_refit": llf,
            "llf_abs_diff": reproduction_diff, "llf_reproduction_match": reproduces,
            "job3527_kept_interaction_estimate": float(kept["estimate"]),
            "job3527_kept_interaction_ci_low": float(kept.get("wald_ci_low", kept.get("ci_low"))),
            "job3527_kept_interaction_ci_high": float(kept.get("wald_ci_high", kept.get("ci_high"))),
            "job3527_kept_llf": float(kept["llf_kept"]),
            **estimates,
            "failure_reasons": candidate["failure_reasons"],
            "warning_categories": candidate["warning_categories"],
            "fit_seconds": float(candidate["seconds"]),
        }
        rows.append(record)
        logger.info(
            "moment_refit family=%s fit=%s llf_match=%s abs_diff=%.9g valid=%s "
            "interaction=%s interaction_ci=[%s,%s] sum=%s sum_ci=[%s,%s]",
            family, fit_set, reproduces,
            reproduction_diff if reproduction_diff is not None else float("inf"), record["valid"],
            estimates["interaction_estimate"], estimates["interaction_wald_ci_low"],
            estimates["interaction_wald_ci_high"], estimates["sum_effect_estimate"],
            estimates["sum_effect_wald_ci_low"], estimates["sum_effect_wald_ci_high"],
        )
        del response, frame, candidate, result
        gc.collect()

    table = pd.DataFrame.from_records(rows)
    payload = {
        "source": "OPS-36 moment refit of job 3891 candidates",
        "fit_rows": rows,
        "llf_tolerance": 1.0e-6,
        "utc_start": started_utc,
        "utc_end": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(table, report_path)
    model = study_models(dict(cfg))[selected_specs[0][0]]
    sidecar = build_provenance(
        cfg, repo_root=repo,
        unit_key={"family": selected_specs[0][0], "stage": "moment_refit", "fit_sets": [fit_set for _, fit_set in selected_specs]},
        entrypoint=repo / "scripts/analyze_e6.py",
        model_id=str(model["hf_id"]), model_revision=str(model["revision"]),
        dtype="analysis_float64", batch_size=None, expected_row_count=len(rows),
        run_git=run_git, model_loads=[],
    )
    sidecar["row_count"] = len(rows)
    sidecar["output_sha256"] = sha256_file(result_path)
    sidecar["utc_end"] = payload["utc_end"]
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "moment_refit_summary json=%s csv=%s rows=%d all_valid=%s all_llf_reproduced=%s",
        result_path.relative_to(repo), report_path.relative_to(repo), len(table),
        bool(table["valid"].all()), bool(table["llf_reproduction_match"].all()),
    )
    passed = bool(table["valid"].all() and table["llf_reproduction_match"].all())
    return table, passed


def build_report(cfg: Mapping[str, Any], repo: Path) -> Path:
    """Write a factual E6 report with estimates and source-table pointers."""
    output = cfg["e6"]["outputs"]
    dzh = pd.read_csv(repo / output["dzh"])
    terms = pd.read_csv(repo / output["model_terms"])
    holm = pd.read_csv(repo / output["holm"])
    matched = pd.read_csv(repo / output["matched"])
    sensitivity = pd.read_csv(repo / output["sensitivity"])
    unresolved = bool(matched["H3_unresolved"].astype(bool).any()) if "H3_unresolved" in matched else True
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a small report table without an optional tabulate dependency."""
        columns = [str(column) for column in frame.columns]
        values = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        widths = [max(len(columns[index]), *(len(row[index]) for row in values)) for index in range(len(columns))]
        header = "| " + " | ".join(columns[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        rule = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
        body_rows = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |" for row in values]
        return "\n".join([header, rule, *body_rows])

    body = [
        "# E6 — Etymology (H3, exploratory)", "",
        "Specification: OPS-28. Main items exclude ambiguous, other_loan, M1-extension and cloze; §7.1 uses complete cases.",
        "All intervals are concept-bootstrap 95% percentile intervals except the stated MixedLM Wald intervals.", "",
        "## D_zh", "", markdown_table(dzh), "",
        "## §7.1 model instrument", "", markdown_table(terms), "",
        "## Holm family (b)", "", markdown_table(holm), "",
        "## Matched estimates", "", markdown_table(matched), "",
        f"H3 matched/full class comparison: {'unresolved' if unresolved else 'classes agree'}.", "",
        "## Sensitivities", "", markdown_table(sensitivity), "",
        "Summary tables: `reports/E6_dzh.csv`, `reports/E6_model_terms.csv`, `reports/E6_holm.csv`, `reports/E6_matched.csv`, `reports/E6_sensitivity.csv`, `reports/E6_matched_pairs.csv`, and `reports/E6_balance.csv`.",
    ]
    target = repo / str(cfg["e6"]["report"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(body) + "\n", encoding="utf-8")
    return target


def _ensure_primary_locks(repo: Path, cfg: Mapping[str, Any]) -> None:
    """Apply both the adjudication and test-item locks before E6 primary access."""
    ensure_adjudication_committed(repo)
    ensure_model_access_allowed(cfg, [{"split": "test"}])


def _ensure_match_files_committed(repo: Path, cfg: Mapping[str, Any]) -> None:
    """Fail before reading outcome values unless the match files are committed/clean."""
    _assert_matching_committed(repo, cfg)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse E6 stages, enforce locks, and run requested source-scoped work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("synthetic", "primary"), required=True)
    parser.add_argument("--confirm-e6", action="store_true")
    parser.add_argument("--closeout", action="store_true")
    parser.add_argument("--moment-refit")
    parser.add_argument("--stages", nargs="+", choices=("match", "dzh", "model", "matched", "sensitivity", "report", "all"))
    args = parser.parse_args(argv)
    if args.closeout and (args.source != "primary" or not args.confirm_e6 or args.stages or args.moment_refit):
        parser.error("--closeout requires --source primary --confirm-e6 and no --stages")
    if args.moment_refit and (args.source != "primary" or not args.confirm_e6 or args.stages or args.closeout):
        parser.error("--moment-refit requires --source primary --confirm-e6 and no --stages")
    if not args.closeout and not args.moment_refit and not args.stages:
        parser.error("--stages is required unless --closeout is selected")
    cfg = load_config(REPO / args.config)
    logger = setup_logger()
    stages = [] if args.closeout or args.moment_refit else (["dzh", "model", "matched", "sensitivity", "report"] if args.stages == ["all"] else args.stages)
    if args.source == "primary":
        if not args.confirm_e6:
            parser.error("--source primary requires --confirm-e6")
        verified = verify_manifest(
            REPO / str(cfg["paths"]["manifest"]), REPO,
            only=[str(cfg["paths"]["concepts_tsv"])],
        )
        logger.info("concepts_tsv manifest verified: %d input(s)", len(verified))
        _ensure_primary_locks(REPO, cfg)
        if any(stage != "match" for stage in stages):
            _ensure_match_files_committed(REPO, cfg)
    if args.source == "synthetic":
        logger.info("synthetic source selected; no primary outcomes are opened")
        if "model" in stages:
            frame = e6.make_synthetic_model_frame(
                n_sino=24, n_nonsino=32, n_layers=8, seed=int(cfg["seed"]),
                synthetic_sd=cfg["e6"]["synthetic_sd"],
            )
            fitted = e6.fit_mixedlm(
                frame, spline_df=int(cfg["e6"]["spline_df"]),
                maxiter=int(cfg["e6"]["maxiter"]), fit_methods=cfg["e6"]["fit_methods"],
                boundary_rule=cfg["e6"]["boundary_rule"],
                llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            )
            detail = e6.model_instrument(fitted)
            logger.info(
                "synthetic fit valid=%s method=%s tries=%d boundary=%s rows=%d",
                detail["valid"], detail["method_used"], detail["tries"],
                detail["variance_at_boundary"], len(frame),
            )
        return 0

    logger.info("source=primary confirm_e6=true stages=%s", ",".join(stages))
    run_git = capture_run_git(REPO, cfg) if args.closeout or args.moment_refit or any(stage != "match" for stage in stages) else None
    if args.closeout:
        if run_git is None:
            raise RuntimeError("closeout requires captured provenance")
        run_closeout(cfg, REPO, logger, run_git)
        return 0
    if args.moment_refit:
        if run_git is None:
            raise RuntimeError("moment refit requires captured provenance")
        _table, passed = run_moment_refits(cfg, REPO, logger, run_git, args.moment_refit)
        return 0 if passed else 4
    for stage in stages:
        started = time.perf_counter()
        if stage == "match":
            result = run_match(cfg, REPO)
            logger.info("match pairs=%d unmatched_sino=%d pair_file=%s balance_file=%s", result["pairs"], result["unmatched_sino"], result["pair_path"], result["balance_path"])
            logger.info("match SMD table:\n%s", result["balance"].to_string(index=False))
        elif stage == "dzh":
            if run_git is None:
                raise RuntimeError("primary D_zh stages require captured run provenance")
            dzh_table, dzh_p = run_dzh(cfg, REPO, logger, run_git)
            _write_csv(dzh_table, REPO / cfg["e6"]["outputs"]["dzh"])
            # Save the D_zh instrument p-values; model p-values are added in the model stage.
            _write_csv(dzh_p, REPO / "results/analysis/primary/e6/dzh_p_values.csv")
        elif stage == "model":
            if run_git is None:
                raise RuntimeError("primary model stages require captured run provenance")
            terms, model_p, diffs = run_model(cfg, REPO, logger, run_git)
            _write_csv(terms, REPO / cfg["e6"]["outputs"]["model_terms"])
            _write_csv(model_p, REPO / "results/analysis/primary/e6/model_p_values.csv")
            logger.info("rise_main_max_abs_diff_by_family=%s", json.dumps(diffs, sort_keys=True))
        elif stage == "matched":
            if run_git is None:
                raise RuntimeError("primary matched stages require captured run provenance")
            matched = run_matched(cfg, REPO, logger, run_git)
            _write_csv(matched, REPO / cfg["e6"]["outputs"]["matched"])
        elif stage == "sensitivity":
            if run_git is None:
                raise RuntimeError("primary sensitivity stages require captured run provenance")
            sensitivity = run_sensitivity(cfg, REPO, logger, run_git)
            _write_csv(sensitivity, REPO / cfg["e6"]["outputs"]["sensitivity"])
        elif stage == "report":
            dzh_p = pd.read_csv(REPO / "results/analysis/primary/e6/dzh_p_values.csv")
            model_p = pd.read_csv(REPO / "results/analysis/primary/e6/model_p_values.csv")
            _write_csv(e6.holm_family_b(pd.concat([dzh_p, model_p], ignore_index=True)), REPO / cfg["e6"]["outputs"]["holm"])
            report = build_report(cfg, REPO)
            logger.info("report=%s", report)
        logger.info("stage=%s wall_seconds=%.3f peak_rss_bytes=%d", stage, time.perf_counter() - started, peak_rss_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

