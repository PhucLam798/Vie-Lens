"""Tests for strict concept schema parsing and nullable pandas dtypes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from vilens.data import load_concepts_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write fixture dictionaries as line-delimited JSON for negative tests."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_fixture_schema_and_dtypes() -> None:
    """Load fixture rows with pandas extension dtypes, explicit NA, and zero intact."""
    frame = load_concepts_jsonl(FIXTURES / "concepts.jsonl")
    assert len(frame) == 3
    assert "cloze_available" not in frame.columns
    assert str(frame["concept_id"].dtype) == "string"
    assert str(frame["collapsed"].dtype) == "boolean"
    assert str(frame["fewshot_set"].dtype) == "Int64"
    assert str(frame["zipf_vi"].dtype) == "Float64"
    assert frame.loc[0, "zipf_vi"] == 0.0
    assert pd.isna(frame.loc[0, "ru"])
    assert pd.isna(frame.loc[0, "fewshot_set"])
    assert pd.isna(frame.loc[1, "m1_eligible_gemma"])
    assert isinstance(frame.loc[0, "vi_alts"], list)


def test_bad_boolean_fails(tmp_path: Path) -> None:
    """Reject boolean strings outside the exact true/false vocabulary."""
    rows = [json.loads(line) for line in (FIXTURES / "concepts.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["collapsed"] = "yes"
    path = tmp_path / "bad_bool.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="collapsed must be 'true' or 'false'"):
        load_concepts_jsonl(path)


def test_unknown_and_missing_columns_fail(tmp_path: Path) -> None:
    """Reject both extra and omitted schema fields."""
    original = [json.loads(line) for line in (FIXTURES / "concepts.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [dict(row) for row in original]
    rows[0]["unexpected"] = "value"
    path = tmp_path / "extra.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="extra=\['unexpected'\]"):
        load_concepts_jsonl(path)
    rows = [dict(row) for row in original]
    del rows[0]["concept_id"]
    path = tmp_path / "missing.jsonl"
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="missing=\['concept_id'\]"):
        load_concepts_jsonl(path)
