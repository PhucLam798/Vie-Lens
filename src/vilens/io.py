"""Validated long-format table and atomic result-unit persistence."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .provenance import differences

LENS_COLUMNS = [
    "model",
    "model_revision",
    "pass",
    "format",
    "condition",
    "fewshot_set",
    "concept_id",
    "target_lang",
    "target_variant",
    "role",
    "ref_concept_id",
    "tok_idx",
    "token_id",
    "is_content",
    "layer",
    "metric",
    "value",
]


def _now_utc() -> str:
    """Return the current UTC time in an ISO-8601 representation."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_long(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate exact long-table columns, unique measurement keys, roles, and layers."""
    expected = list(columns)
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(f"Columns must equal the requested order: expected {expected}, got {actual}")
    key_columns = [column for column in expected if column != "value"]
    if df.duplicated(subset=key_columns).any():
        raise ValueError("Duplicate long-format rows found when ignoring value")
    if "role" in expected and not df["role"].isin({"correct", "distractor", "path"}).all():
        raise ValueError("role must contain only correct, distractor, or path")
    if "layer" in expected:
        try:
            layers = pd.to_numeric(df["layer"], errors="raise")
            invalid_layer = layers.isna().any() or (layers < 0).any()
        except (TypeError, ValueError) as exc:
            raise ValueError("layer must contain non-negative numbers") from exc
        if invalid_layer:
            raise ValueError("layer must contain non-negative numbers")


def unit_paths(
    results_dir: str | Path,
    unit_key: Mapping[str, str | int],
    *,
    filename: str | None = None,
) -> tuple[Path, Path]:
    """Return deterministic parquet and provenance paths for a result unit key."""
    stem = "__".join(f"{key}={unit_key[key]}" for key in sorted(unit_key))
    root = Path(results_dir)
    if filename is not None:
        target = Path(filename)
        if target.name != filename or target.suffix != ".parquet":
            raise ValueError("filename must be a plain .parquet filename")
        return root / target, root / f"{target.stem}.provenance.json"
    return root / f"{stem}.parquet", root / f"{stem}.provenance.json"


def write_unit(
    df: pd.DataFrame,
    results_dir: str | Path,
    provenance: Mapping[str, Any],
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
    compression: str,
    filename: str | None = None,
) -> tuple[Path, Path]:
    """Atomically write a sorted parquet unit and its completion sidecar last."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_long(df, columns)
    sorted_df = df.sort_values(list(sort_by), kind="mergesort", na_position="last").reset_index(drop=True)
    parquet_path, sidecar_path = unit_paths(
        results_dir, provenance["unit_key"], filename=filename
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_temp: Path | None = None
    sidecar_temp: Path | None = None
    sidecar_path.unlink(missing_ok=True)
    payload = dict(provenance)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parquet_path.parent, prefix=f".{parquet_path.name}.", suffix=".tmp", delete=False
        ) as stream:
            parquet_temp = Path(stream.name)
        table = pa.Table.from_pandas(sorted_df, preserve_index=False)
        pq.write_table(table, parquet_temp, compression=compression)
        os.replace(parquet_temp, parquet_path)
        parquet_temp = None
        payload["row_count"] = int(len(sorted_df))
        payload["utc_end"] = _now_utc()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=sidecar_path.parent,
            prefix=f".{sidecar_path.name}.", suffix=".tmp", delete=False
        ) as stream:
            sidecar_temp = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(sidecar_temp, sidecar_path)
        sidecar_temp = None
        return parquet_path, sidecar_path
    except Exception:
        sidecar_path.unlink(missing_ok=True)
        for temporary in (parquet_temp, sidecar_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        raise


def unit_is_complete(
    results_dir: str | Path,
    unit_key: Mapping[str, str | int],
    current_provenance: Mapping[str, Any],
    fields: Sequence[str],
    *,
    filename: str | None = None,
) -> tuple[bool, str]:
    """Check both completion files, provenance equality, and parquet metadata row counts."""
    import pyarrow.parquet as pq

    parquet_path, sidecar_path = unit_paths(results_dir, unit_key, filename=filename)
    if not sidecar_path.is_file():
        return False, f"missing sidecar: {sidecar_path.name}"
    if not parquet_path.is_file():
        return False, f"missing parquet: {parquet_path.name}"
    try:
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid sidecar: {exc}"
    if not isinstance(sidecar, dict):
        return False, "invalid sidecar: top-level JSON value must be an object"
    if "code_tree_sha256" not in sidecar:
        return False, "old sidecar missing code_tree_sha256"
    if "code_tree_sha256" not in current_provenance:
        return False, "current provenance missing code_tree_sha256"
    changed = differences(sidecar, current_provenance, fields)
    if changed:
        return False, "provenance differs: " + ", ".join(changed)
    row_count = sidecar.get("row_count")
    expected = sidecar.get("expected_row_count")
    current_expected = current_provenance.get("expected_row_count")
    if row_count != expected or row_count != current_expected:
        return False, (
            f"row count mismatch: sidecar row_count={row_count}, expected_row_count={expected}, "
            f"current expected_row_count={current_expected}"
        )
    try:
        parquet_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception as exc:
        return False, f"cannot read parquet metadata: {exc}"
    if parquet_rows != row_count:
        return False, f"parquet row count {parquet_rows} differs from sidecar row_count {row_count}"
    return True, "complete"
