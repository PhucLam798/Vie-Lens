"""Tests for E2's primary P-L unit resolution preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from vilens import env_guard
from vilens.passes import with_pl_shards


def _load_job(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the job entrypoint while bypassing its process-start environment guard."""
    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    script = Path(__file__).resolve().parents[1] / "job_e2_m1.py"
    spec = importlib.util.spec_from_file_location("job_e2_m1_unit_resolution_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primary_records() -> list[dict[str, Any]]:
    """Build one prompt record for each primary format, including null cloze set."""
    return [
        {"format": "repetition", "condition": "diac", "fewshot_set": 1},
        {"format": "translation", "condition": "diac", "fewshot_set": 1},
        {"format": "cloze", "condition": "diac", "fewshot_set": None},
    ]


def test_e2_cloze_unit_uses_p_l_none_fewshot_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cloze's null prompt field resolves through P-L as fewshot_set=none."""
    job = _load_job(monkeypatch)
    records = with_pl_shards([_primary_records()[2]], shard_prompts=10)

    units, counts, missing = job._expected_primary_units(records, "llama", tmp_path)

    assert len(units) == 1
    assert "fewshot_set=none" in units[0][1].name
    assert "fewshot_set=1" not in units[0][1].name
    assert counts == {"cloze": (1, 1)}
    assert missing


def test_e2_inputs_resolve_fails_when_a_pl_unit_file_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The named preflight check fails before E2 when a required unit file is absent."""
    job = _load_job(monkeypatch)
    records = with_pl_shards(_primary_records(), shard_prompts=10)
    units, counts, _missing = job._expected_primary_units(records, "gemma", tmp_path)
    assert counts == {"cloze": (1, 1), "repetition": (1, 1), "translation": (1, 1)}

    missing_file = units[0][1].with_suffix(".resid.npy")
    for _fmt, parquet_path in units:
        sidecar = parquet_path.with_name(f"{parquet_path.stem}.provenance.json")
        required = (
            parquet_path,
            sidecar,
            parquet_path.with_suffix(".resid.npy"),
            parquet_path.with_suffix(".resid_index.jsonl"),
            parquet_path.with_suffix(".answer_topk.jsonl"),
        )
        for path in required:
            if path != missing_file:
                path.touch()

    check = job._e2_inputs_resolve(records, ["gemma"], tmp_path)

    assert check.name == "e2_inputs_resolve"
    assert check.status == "FAIL"
    assert str(missing_file) in check.detail
