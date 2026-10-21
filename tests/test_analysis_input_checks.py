"""Tests for the whole-run primary input check and P-L unit resolution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from vilens import env_guard, io
from vilens.passes import pl_unit_key, with_pl_shards


def _load_analyze_core(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the guarded analysis entrypoint for isolated metadata-only tests."""
    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_core.py"
    spec = importlib.util.spec_from_file_location("analyze_core_input_checks_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_pl_units(results_dir: Path) -> list[dict[str, Any]]:
    """Write tiny primary-style unit metadata, including the cloze null set."""
    records = with_pl_shards(
        [
            {"concept_id": "rep-main", "format": "repetition", "condition": "diac", "fewshot_set": 1, "m1_extension": False},
            {"concept_id": "tr-main", "format": "translation", "condition": "diac", "fewshot_set": 1, "m1_extension": False},
            {"concept_id": "cl-main", "format": "cloze", "condition": "diac", "fewshot_set": None, "m1_extension": False},
        ],
        shard_prompts=250,
    )
    for record in records:
        parquet_path, sidecar_path = io.unit_paths(results_dir, pl_unit_key("llama", record))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.touch()
        sidecar_path.write_text("{}\n", encoding="utf-8")
        parquet_path.with_suffix(".resid.npy").touch()
        parquet_path.with_suffix(".generations.jsonl").write_bytes(b"{}\n")
        parquet_path.with_suffix(".resid_index.jsonl").write_text(
            json.dumps(
                {
                    "array_row": 0,
                    "concept_id": record["concept_id"],
                    "format": record["format"],
                    "lang": "",
                    "fewshot_set": "none" if record["fewshot_set"] is None else 1,
                    "split": "test",
                    "m1_extension": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return records


def _fixture_check(module: Any, records: list[dict[str, Any]], results_dir: Path) -> list[dict[str, str]]:
    """Run the shared P-L and query portion of check_inputs on the fixture."""
    return module._check_primary_pl_inputs(
        records,
        "llama",
        results_dir,
        main_ids={"rep-main", "tr-main", "cl-main"},
        expected_unit_counts={"repetition": 1, "translation": 1, "cloze": 1},
        expected_query_counts={"repetition": 1, "translation": 1, "cloze": 1},
    )


def test_check_inputs_resolves_primary_style_fewshot_and_cloze_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P-L plan names repetition/translation with set 1 and cloze with none."""
    module = _load_analyze_core(monkeypatch)
    results_dir = tmp_path / "results/pl"
    records = _fixture_pl_units(results_dir)

    resolved = module._planned_primary_units(records, "llama", results_dir)
    names = {fmt: path.name for fmt, path in resolved}
    assert "fewshot_set=1" in names["repetition"]
    assert "fewshot_set=1" in names["translation"]
    assert "fewshot_set=none" in names["cloze"]
    assert all(row["status"] == "MATCH" for row in _fixture_check(module, records, results_dir))


def test_check_inputs_fails_when_a_pl_unit_file_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing planned parquet unit is reported as a mismatch."""
    module = _load_analyze_core(monkeypatch)
    results_dir = tmp_path / "results/pl"
    records = _fixture_pl_units(results_dir)
    _fmt, parquet_path = module._planned_primary_units(records, "llama", results_dir)[0]
    parquet_path.unlink()

    rows = _fixture_check(module, records, results_dir)

    assert any(row["check"] == "P-L units" and row["status"] == "MISMATCH" for row in rows)


def test_check_inputs_fails_when_a_main_query_index_row_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The query check detects a missing primary concept in a residual index."""
    module = _load_analyze_core(monkeypatch)
    results_dir = tmp_path / "results/pl"
    records = _fixture_pl_units(results_dir)
    _fmt, parquet_path = module._planned_primary_units(records, "llama", results_dir)[0]
    parquet_path.with_suffix(".resid_index.jsonl").write_text("", encoding="utf-8")

    rows = _fixture_check(module, records, results_dir)

    assert any(row["check"] == "LA query index" and row["status"] == "MISMATCH" for row in rows)
