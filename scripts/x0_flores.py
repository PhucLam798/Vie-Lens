#!/usr/bin/env python
"""Rebuild and verify the frozen FLORES+ development directions file."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard


EXPECTED_DIRECTIONS_SHA256 = "e768392ccbee4ddbd20fd127fa97103f64f19c54e333df285e12bb67067ea25e"
LANGUAGE_CODES = {
    "vie_Latn": "vi",
    "eng_Latn": "en",
    "cmn_Hans": "zh",
    "fra_Latn": "fr",
    "ind_Latn": "id",
}


def _utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup_logging() -> tuple[logging.Logger, Path]:
    """Configure stdout and a UTC-stamped file log without logging data text."""
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    logs_dir = REPO / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"x0_flores_{stamp}.log"
    logger = logging.getLogger("x0_flores")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = lambda timestamp: datetime.fromtimestamp(timestamp, timezone.utc).timetuple()
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger, log_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the login-node command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    return parser.parse_args(argv)


def _read_flores_section(config_path: Path) -> dict[str, Any]:
    """Read only the early cache setting needed to validate the environment."""
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    flores = raw.get("flores") if isinstance(raw, dict) else None
    if not isinstance(flores, dict):
        raise ValueError("Configuration has no flores mapping")
    return flores


def _check_download_environment(flores: dict[str, Any]) -> str:
    """Validate cache locations, online mode and the required token before data-library imports."""
    env_guard.check_no_heavy_imports()
    env_guard.check_environment(
        environ=os.environ,
        cwd=str(REPO),
        hub_cache=str(flores.get("hub_cache", "")),
    )
    expected_datasets = os.path.realpath(str(flores.get("datasets_cache", "")))
    actual_datasets = os.environ.get("HF_DATASETS_CACHE")
    if not actual_datasets or os.path.realpath(actual_datasets) != expected_datasets:
        raise env_guard.EnvGuardError(
            f"HF_DATASETS_CACHE must resolve to {expected_datasets}; got "
            f"{os.path.realpath(actual_datasets) if actual_datasets else 'unset'}"
        )
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if name in os.environ and os.environ[name] != "0":
            raise env_guard.EnvGuardError(f"{name} must be unset or 0 for the authorized FLORES download")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise env_guard.EnvGuardError("HF_TOKEN is required for the gated FLORES+ dataset")
    env_guard.set_runtime_dirs()
    return token


def _load_prep_functions(script_path: Path) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Import the pinned preparation validators without executing its pipeline."""
    module_name = "vielens_pinned_flores_download"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import preparation helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    prior_common: Any = sys.modules.get("common", missing)
    old_path = list(sys.path)
    sys.path.insert(0, str(script_path.parent))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)
        if prior_common is missing:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = prior_common
    return (
        module.map_flores_source_table,
        module.validate_flores_alignment,
        module.validate_flores_output_table,
    )


def _source_sha256(sources_path: Path, filename: str) -> str | None:
    """Find a FLORES Parquet digest in the pinned source table."""
    for line in sources_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| flores | data/raw/flores/"):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and Path(cells[1]).name == filename:
            return cells[5]
    return None


def _safe_child_output(label: str, text: str, logger: logging.Logger) -> None:
    """Print safe builder output while suppressing unexpected content-bearing lines."""
    safe_patterns = (
        re.compile(r"^Wrote [0-9]+ FLORES direction records to [A-Za-z0-9_./-]+$"),
        re.compile(r"^WARNING: no unique FLORES provenance row for dev_[A-Za-z0-9_]+\.parquet; continuing\.$"),
        re.compile(r"^WARNING: SHA-256 mismatch for dev_[A-Za-z0-9_]+\.parquet; expected [0-9a-f]{64}, got [0-9a-f]{64}\.$"),
    )
    for line in text.splitlines():
        if any(pattern.fullmatch(line) for pattern in safe_patterns):
            logger.info("prep directions %s: %s", label, line[:120])
        elif line:
            # The preparation program is expected to emit only the allowlisted
            # summary/warning forms; suppress any unexpected line under rule 48.
            logger.warning("prep directions %s: suppressed unexpected output (%d characters)", label, len(line))


def _add_check(
    checks: list[dict[str, str]], logger: logging.Logger, name: str,
    expected: str, actual: str, passed: bool,
) -> None:
    """Append and immediately log one PASS/FAIL summary row."""
    row = {
        "check": name,
        "expected": expected,
        "actual": actual,
        "result": "PASS" if passed else "FAIL",
    }
    checks.append(row)
    logger.info("CHECK | %s | %s | expected=%s | actual=%s", name, row["result"], expected, actual)


def _run_build(
    cfg: dict[str, Any], token: str, logger: logging.Logger,
    checks: list[dict[str, str]],
) -> dict[str, Any]:
    """Download or resume five validated partitions, build directions, and verify the release gate."""
    flores = cfg["flores"]
    parquet_dir = REPO / flores["parquet_dir"]
    directions_script = REPO / flores["prep_directions_script"]
    download_script = REPO / flores["prep_download_script"]
    sources_path = REPO / flores["prep_sources"]
    output_path = REPO / flores["output"]
    sentinel = Path("/datastore/shared/models/datasets--openlanguagedata--flores_plus")
    sentinel_before = sentinel.exists()
    _add_check(checks, logger, "sentinel_before", "absent", "present" if sentinel_before else "absent", not sentinel_before)

    map_table, validate_alignment, validate_output = _load_prep_functions(download_script)
    details: dict[str, Any] = {
        "sentinel_before": sentinel_before,
        "download_script_sha256": _sha256_file(download_script),
        "directions_script_sha256": _sha256_file(directions_script),
        "languages": [],
        "parquets": [],
        "prep_stdout": "",
        "prep_stderr": "",
        "output_sha256": None,
        "output_records": None,
        "checks": checks,
    }
    logger.info("Preparation download script SHA-256: %s", details["download_script_sha256"])
    logger.info("Preparation directions script SHA-256: %s", details["directions_script_sha256"])
    logger.info("Shared cache sentinel before: %s", "present" if sentinel_before else "absent")

    import pyarrow.parquet as pq
    from datasets import load_dataset
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error

    disable_progress_bar()
    set_verbosity_error()

    tables: dict[tuple[str, str], Any] = {}
    pending: list[tuple[Path, Any, str]] = []
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for language in flores["languages"]:
        if language not in LANGUAGE_CODES:
            raise ValueError(f"Unsupported configured FLORES language code: {language}")
        filename = f"dev_{language}.parquet"
        parquet_path = parquet_dir / filename
        if parquet_path.is_file():
            raw_table = pq.read_table(parquet_path)
            mapped, variants = validate_output(raw_table, split=flores["split"], language=language)
            source_kind = "resumed"
        else:
            dataset = load_dataset(
                flores["repo"],
                language,
                split=flores["split"],
                revision=flores["revision"],
                token=token,
                cache_dir=flores["datasets_cache"],
            )
            mapped, variants = map_table(dataset.data.table, split=flores["split"], language=language)
            pending.append((parquet_path, mapped, filename))
            source_kind = "downloaded"
        tables[(flores["split"], language)] = mapped
        language_detail = {
            "language": language,
            "rows": mapped.num_rows,
            "variant_counts": variants,
            "columns": list(mapped.column_names),
            "source": source_kind,
        }
        details["languages"].append(language_detail)
        logger.info(
            "Language %s | rows=%d | variants=%s | columns=%s | source=%s",
            language, mapped.num_rows, json.dumps(variants, sort_keys=True),
            json.dumps(mapped.column_names), source_kind,
        )

    validate_alignment(tables, splits=[flores["split"]], languages=flores["languages"])
    _add_check(
        checks, logger, "aligned_ids",
        f"{flores['expected_ids']} common ids across 5 languages", "alignment passed", True,
    )

    for parquet_path, table, filename in pending:
        with tempfile.NamedTemporaryFile("wb", dir=parquet_dir, prefix=".flores-", suffix=".parquet", delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            pq.write_table(table, temporary_path, compression="zstd", version="2.6")
            if parquet_path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing partition: {parquet_path}")
            os.replace(temporary_path, parquet_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    for language in flores["languages"]:
        filename = f"dev_{language}.parquet"
        parquet_path = parquet_dir / filename
        actual_hash = _sha256_file(parquet_path)
        expected_hash = _source_sha256(sources_path, filename)
        same = expected_hash is not None and actual_hash == expected_hash
        parquet_detail = {
            "language": language,
            "file": str(parquet_path.relative_to(REPO)),
            "size": parquet_path.stat().st_size,
            "sha256": actual_hash,
            "sources_sha256": expected_hash,
            "sources_comparison": "SAME" if same else "DIFFERENT",
        }
        details["parquets"].append(parquet_detail)
        logger.info(
            "Parquet %s | size=%d | sha256=%s | SOURCES.md=%s | %s",
            language, parquet_detail["size"], actual_hash, expected_hash or "not found", parquet_detail["sources_comparison"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(directions_script),
        "--flores-dir", str(parquet_dir),
        "--sources", str(sources_path),
        "--output", str(output_path),
    ]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    details["prep_stdout"] = completed.stdout
    details["prep_stderr"] = completed.stderr
    _safe_child_output("stdout", completed.stdout, logger)
    _safe_child_output("stderr", completed.stderr, logger)
    _add_check(
        checks, logger, "directions_builder", "exit code 0",
        f"exit code {completed.returncode}", completed.returncode == 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Preparation directions builder exited {completed.returncode}")

    record_count = 0
    output_languages = [LANGUAGE_CODES[language] for language in flores["languages"]]
    by_language: dict[str, dict[str, str]] = {language: {} for language in output_languages}
    exact_keys = True
    allowed_splits = {"fit", "validate"}
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            record_count += 1
            if set(record) != {"lang", "sentence", "sentence_id", "split"}:
                exact_keys = False
            language = record.get("lang")
            sentence_id = record.get("sentence_id")
            split = record.get("split")
            sentence = record.get("sentence")
            if language not in by_language or not isinstance(sentence_id, str) or not sentence_id:
                raise ValueError(f"Invalid record metadata at output line {line_number}")
            if not isinstance(sentence, str) or not sentence.strip() or split not in allowed_splits:
                raise ValueError(f"Invalid sentence or split metadata at output line {line_number}")
            if sentence_id in by_language[language]:
                raise ValueError(f"Duplicate sentence ID for language {language} at output line {line_number}")
            by_language[language][sentence_id] = split
    details["output_records"] = record_count
    output_hash = _sha256_file(output_path)
    details["output_sha256"] = output_hash

    expected_ids = int(flores["expected_ids"])
    id_counts = {language: len(records) for language, records in by_language.items()}
    id_sets = {frozenset(records) for records in by_language.values()}
    id_ok = all(count == expected_ids for count in id_counts.values()) and len(id_sets) == 1
    split_counts = {
        language: Counter(records.values()) for language, records in by_language.items()
    }
    splits_ok = all(
        count.get("validate", 0) == int(flores["expected_validate_ids"])
        and count.get("fit", 0) == expected_ids - int(flores["expected_validate_ids"])
        for count in split_counts.values()
    )
    keys_ok = exact_keys
    records_ok = record_count == int(flores["expected_records"])
    hash_ok = output_hash == EXPECTED_DIRECTIONS_SHA256

    _add_check(checks, logger, "record_count", str(flores["expected_records"]), str(record_count), records_ok)
    _add_check(checks, logger, "ids_per_language_and_alignment", f"{expected_ids} each; same id set", json.dumps(id_counts, sort_keys=True), id_ok)
    split_actual = {
        language: {name: counts.get(name, 0) for name in ("fit", "validate")}
        for language, counts in split_counts.items()
    }
    _add_check(
        checks, logger, "fit_validate_counts",
        f"{expected_ids - int(flores['expected_validate_ids'])} fit / {flores['expected_validate_ids']} validate per language",
        json.dumps(split_actual, sort_keys=True), splits_ok,
    )
    _add_check(checks, logger, "record_keys", "lang, sentence, sentence_id, split exactly", "exact" if keys_ok else "not exact", keys_ok)
    _add_check(checks, logger, "directions_sha256", EXPECTED_DIRECTIONS_SHA256, output_hash, hash_ok)

    sentinel_after = sentinel.exists()
    prep_status = subprocess.run(
        ["git", "-C", str(REPO / "data/external/viLens-concepts-preparation"), "status", "--short"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    prep_clean = prep_status.returncode == 0 and not prep_status.stdout.strip()
    details["sentinel_after"] = sentinel_after
    details["prep_status"] = prep_status.stdout
    _add_check(checks, logger, "shared_cache_untouched", "sentinel remains absent", "present" if sentinel_after else "absent", not sentinel_before and not sentinel_after)
    _add_check(checks, logger, "prep_clone_clean", "empty git status", "empty" if prep_clean else prep_status.stdout.strip() or f"git exit {prep_status.returncode}", prep_clean)
    return details


def main(argv: list[str] | None = None) -> int:
    """Run the gated-source rebuild, log checks, and return a status code."""
    args = _parse_args(argv)
    config_path = (REPO / args.config).resolve()
    try:
        raw_flores = _read_flores_section(config_path)
        token = _check_download_environment(raw_flores)
        from vilens.config import load_config

        cfg = load_config(config_path)
    except Exception as exc:
        print(f"FLORES precondition failed: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
        return 2

    logger, log_path = _setup_logging()
    logger.info("HF_TOKEN: set")
    logger.info("Log file: %s", log_path.relative_to(REPO))
    logger.info("Run start UTC: %s", _utc_now().isoformat(timespec="seconds"))
    checks: list[dict[str, str]] = []
    details: dict[str, Any] = {}
    try:
        details = _run_build(cfg, token, logger, checks)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            failure = f"HTTP {status} gated access"
        elif "gated" in type(exc).__name__.lower() or "gated" in str(exc).lower():
            failure = "gated access error"
        else:
            failure = f"{type(exc).__name__}: {str(exc)[:160]}"
        logger.error("Build stopped: %s", failure)
        if not checks:
            checks = []
        _add_check(checks, logger, "build_and_validate", "download, validate, and build succeeds", failure, False)
        sentinel = Path("/datastore/shared/models/datasets--openlanguagedata--flores_plus")
        sentinel_after = sentinel.exists()
        _add_check(checks, logger, "shared_cache_untouched", "sentinel remains absent", "present" if sentinel_after else "absent", not sentinel_after)
        prep_status = subprocess.run(
            ["git", "-C", str(REPO / "data/external/viLens-concepts-preparation"), "status", "--short"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        _add_check(checks, logger, "prep_clone_clean", "empty git status", "empty" if not prep_status.stdout.strip() else prep_status.stdout.strip(), prep_status.returncode == 0 and not prep_status.stdout.strip())
        details.setdefault("failure", failure)
        details.setdefault("sentinel_after", sentinel_after)
        details.setdefault("prep_status", prep_status.stdout)

    logger.info("Final summary | check | expected | actual | result")
    for row in checks:
        logger.info("Final summary | %s | %s | %s | %s", row["check"], row["expected"], row["actual"], row["result"])
    failures = sum(row["result"] != "PASS" for row in checks)
    logger.info("Run end UTC: %s | failures=%d | checks=%d", _utc_now().isoformat(timespec="seconds"), failures, len(checks))
    return 0 if checks and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
