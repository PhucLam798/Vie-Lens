"""Tests for long-format schema validation and atomic result-unit persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.parquet as pq
import pytest

from vilens.io import LENS_COLUMNS, unit_is_complete, unit_paths, validate_long, write_unit
from vilens.provenance import build, capture_run_git


def _frame() -> pd.DataFrame:
    """Return a two-row valid lens table in unsorted order."""
    common = {
        "model": "tiny",
        "model_revision": "rev",
        "pass": "P-L",
        "format": "repetition",
        "condition": "diac",
        "fewshot_set": 1,
        "target_lang": "vi",
        "target_variant": "released",
        "role": "correct",
        "ref_concept_id": "c1",
        "token_id": 5,
        "is_content": True,
        "layer": 0,
        "metric": "logprob",
    }
    return pd.DataFrame(
        [
            {**common, "concept_id": "c2", "tok_idx": 1, "value": -2.0},
            {**common, "concept_id": "c1", "tok_idx": 0, "value": -1.0},
        ],
        columns=LENS_COLUMNS,
    )


def _provenance() -> dict[str, object]:
    """Return the small set of provenance values used by result-unit checks."""
    return {
        "unit_key": {"model": "tiny", "pass": "P-L"},
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "derived_sha256": "derived",
        "code_sha256": "code",
        "code_tree_sha256": "code-tree",
        "model_id": "model",
        "model_revision": "revision",
        "dtype": "bf16",
        "batch_size": 4,
        "torch_version": "2.8.0",
        "transformers_version": "4.57.6",
        "tokenizers_version": "0.22.2",
        "expected_row_count": 2,
    }


def _git(repo: Path, *args: str) -> str:
    """Run Git with a local test identity and return stdout."""
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid"})
    env.update({"GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"})
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_run_capture_is_stable_across_docs_commit_and_tracks_source_change(tmp_path: Path) -> None:
    """Write with the start commit, resume after docs-only change, and invalidate source edits."""
    repo = tmp_path / "repo"
    for directory in ("configs", "src", "derived", "data", "docs"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    (repo / "configs/experiments.yaml").write_text("seed: 1\n", encoding="utf-8")
    (repo / "src/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "derived/seed.json").write_text("{}\n", encoding="utf-8")
    (repo / "data/MANIFEST.sha256").write_text("frozen\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "configs", "src", "derived", "data")
    _git(repo, "commit", "-qm", "initial")
    cfg = {
        "runtime": {"git_clean_paths": ["src", "configs", "derived"]},
        "paths": {"manifest": "data/MANIFEST.sha256"},
    }
    run_git = capture_run_git(repo, cfg)
    start_commit = run_git.git_commit
    provenance = build(
        cfg,
        repo_root=repo,
        unit_key=_provenance()["unit_key"],
        entrypoint=None,
        model_id="tiny",
        model_revision="revision",
        dtype="torch.float32",
        batch_size=1,
        expected_row_count=2,
        run_git=run_git,
        model_loads=[{"model_id": "tiny", "model_revision": "revision", "dtype": "torch.float32", "dtype_override": None}],
    )
    _parquet, sidecar_path = write_unit(
        _frame(), tmp_path / "results", provenance,
        columns=LENS_COLUMNS, sort_by=["concept_id", "tok_idx"], compression="zstd",
    )
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert saved["git_commit"] == start_commit
    assert saved["model_loads"] == provenance["model_loads"]

    (repo / "docs/note.md").write_text("docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/note.md")
    _git(repo, "commit", "-qm", "docs only")
    docs_run_git = capture_run_git(repo, cfg)
    current = build(
        cfg, repo_root=repo, unit_key=_provenance()["unit_key"], entrypoint=None,
        model_id="tiny", model_revision="revision", dtype="torch.float32", batch_size=1,
        expected_row_count=2, run_git=docs_run_git,
        model_loads=provenance["model_loads"],
    )
    fields = [
        "unit_key", "config_sha256", "manifest_sha256", "derived_sha256", "code_sha256",
        "code_tree_sha256", "model_id", "model_revision", "dtype", "batch_size",
        "torch_version", "transformers_version", "tokenizers_version",
    ]
    assert docs_run_git.git_commit != start_commit
    assert unit_is_complete(
        tmp_path / "results", current["unit_key"], current, fields
    ) == (True, "complete")

    (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/module.py")
    _git(repo, "commit", "-qm", "source change")
    source_run_git = capture_run_git(repo, cfg)
    changed = build(
        cfg, repo_root=repo, unit_key=_provenance()["unit_key"], entrypoint=None,
        model_id="tiny", model_revision="revision", dtype="torch.float32", batch_size=1,
        expected_row_count=2, run_git=source_run_git,
        model_loads=provenance["model_loads"],
    )
    complete, reason = unit_is_complete(tmp_path / "results", changed["unit_key"], changed, fields)
    assert not complete
    assert "code_tree_sha256" in reason


@pytest.mark.parametrize("case", ["order", "duplicate", "role", "layer"])
def test_validate_long_rejects_invalid_rows(case: str) -> None:
    """Reject schema order, duplicate keys, invalid roles, and negative layers."""
    frame = _frame()
    columns = list(LENS_COLUMNS)
    if case == "order":
        columns.reverse()
    elif case == "duplicate":
        frame.loc[1, [column for column in LENS_COLUMNS if column != "value"]] = frame.loc[
            0, [column for column in LENS_COLUMNS if column != "value"]
        ]
    elif case == "role":
        frame.loc[0, "role"] = "other"
    else:
        frame.loc[0, "layer"] = -1
    with pytest.raises(ValueError):
        validate_long(frame, columns)


def test_write_unit_complete_and_sorted(tmp_path: Path) -> None:
    """Write a stable parquet with metadata and a final completion sidecar."""
    frame = _frame()
    parquet, sidecar = write_unit(
        frame,
        tmp_path,
        _provenance(),
        columns=LENS_COLUMNS,
        sort_by=["concept_id", "tok_idx"],
        compression="zstd",
    )
    assert parquet.exists() and sidecar.exists()
    fields = [
        "unit_key", "config_sha256", "manifest_sha256", "derived_sha256", "code_sha256", "code_tree_sha256",
        "model_id", "model_revision", "dtype", "batch_size", "torch_version",
        "transformers_version", "tokenizers_version",
    ]
    assert unit_is_complete(tmp_path, _provenance()["unit_key"], _provenance(), fields)[0]
    assert pq.ParquetFile(parquet).metadata.num_rows == 2
    assert pd.read_parquet(parquet)["concept_id"].tolist() == ["c1", "c2"]
    assert json.loads(sidecar.read_text(encoding="utf-8"))["row_count"] == 2


@pytest.mark.parametrize(
    "field",
    [
        "unit_key", "config_sha256", "manifest_sha256", "derived_sha256", "code_sha256", "code_tree_sha256",
        "model_id", "model_revision", "dtype", "batch_size", "torch_version",
        "transformers_version", "tokenizers_version",
    ],
)
def test_unit_is_incomplete_when_comparison_field_changes(tmp_path: Path, field: str) -> None:
    """Name every provenance mismatch in the incomplete-unit reason."""
    write_unit(_frame(), tmp_path, _provenance(), columns=LENS_COLUMNS, sort_by=["concept_id"], compression="zstd")
    current = _provenance()
    current[field] = {"model": "different"} if field == "unit_key" else "different"
    fields = [
        "unit_key", "config_sha256", "manifest_sha256", "derived_sha256", "code_sha256",
        "code_tree_sha256",
        "model_id", "model_revision", "dtype", "batch_size", "torch_version",
        "transformers_version", "tokenizers_version",
    ]
    complete, reason = unit_is_complete(tmp_path, _provenance()["unit_key"], current, fields)
    assert not complete
    assert field in reason


def test_unit_is_incomplete_on_row_count_mismatch(tmp_path: Path) -> None:
    """Require sidecar and current expected row counts to agree with parquet metadata."""
    write_unit(_frame(), tmp_path, _provenance(), columns=LENS_COLUMNS, sort_by=["concept_id"], compression="zstd")
    current = _provenance()
    current["expected_row_count"] = 3
    complete, reason = unit_is_complete(tmp_path, current["unit_key"], current, ["unit_key", "config_sha256"])
    assert not complete
    assert "row count mismatch" in reason


def test_unit_is_incomplete_without_sidecar(tmp_path: Path) -> None:
    """Treat parquet presence without its final sidecar as an incomplete unit."""
    parquet, sidecar = write_unit(_frame(), tmp_path, _provenance(), columns=LENS_COLUMNS, sort_by=["concept_id"], compression="zstd")
    sidecar.unlink()
    assert parquet.exists()
    complete, reason = unit_is_complete(tmp_path, _provenance()["unit_key"], _provenance(), ["unit_key"])
    assert complete is False
    assert "sidecar" in reason


def test_old_sidecar_without_code_tree_hash_is_incomplete(tmp_path: Path) -> None:
    """Reject pre-OPS-11 sidecars that lack the new resume-key hash."""
    _parquet, sidecar = write_unit(
        _frame(), tmp_path, _provenance(), columns=LENS_COLUMNS,
        sort_by=["concept_id"], compression="zstd",
    )
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    saved.pop("code_tree_sha256")
    sidecar.write_text(json.dumps(saved), encoding="utf-8")
    complete, reason = unit_is_complete(
        tmp_path, _provenance()["unit_key"], _provenance(),
        ["unit_key", "code_tree_sha256"],
    )
    assert not complete
    assert "missing code_tree_sha256" in reason


def test_parquet_write_failure_cleans_temporary_files_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all temporary artifacts and leave no completion sidecar on write failure."""
    import pyarrow.parquet

    def fail_write(*args: object, **kwargs: object) -> None:
        """Inject a parquet writer failure."""
        raise RuntimeError("injected parquet failure")

    monkeypatch.setattr(pyarrow.parquet, "write_table", fail_write)
    with pytest.raises(RuntimeError, match="injected"):
        write_unit(_frame(), tmp_path, _provenance(), columns=LENS_COLUMNS, sort_by=["concept_id"], compression="zstd")
    parquet, sidecar = unit_paths(tmp_path, _provenance()["unit_key"])
    assert not parquet.exists()
    assert not sidecar.exists()
    assert list(tmp_path.iterdir()) == []
