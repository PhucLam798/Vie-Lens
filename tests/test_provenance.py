"""Tests for content hashes, Git cleanliness, and provenance comparison."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from vilens.provenance import (
    capture_run_git,
    code_sha256,
    derived_sha256,
    differences,
    e1_model_loads,
    git_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    """Run a Git command with a deterministic test identity."""
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid"})
    env.update({"GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"})
    result = subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    """Create a committed repository with the runtime config needed by git_state."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "configs").mkdir()
    (repo / "src").mkdir()
    (repo / "derived").mkdir()
    (repo / "docs").mkdir()
    (repo / "scripts").mkdir()
    (repo / "derived/seed.json").write_text("{}\n", encoding="utf-8")
    (repo / "configs/experiments.yaml").write_text(
        (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (repo / "src/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "scripts/analyze_e6.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (repo / "entry.py").write_text("print('one')\n", encoding="utf-8")
    _git(repo, "add", "configs", "src", "derived", "scripts", "entry.py")
    _git(repo, "commit", "-qm", "initial")
    return repo


def test_derived_sha256_is_order_invariant_and_changes_on_edit(tmp_path: Path) -> None:
    """Hash records in stable order and include each visible regular file's contents."""
    repo = tmp_path / "repo"
    (repo / "derived/sub").mkdir(parents=True)
    (repo / "derived/z.txt").write_text("z", encoding="utf-8")
    (repo / "derived/sub/a.txt").write_text("a", encoding="utf-8")
    (repo / "derived/.hidden").write_text("ignored", encoding="utf-8")
    first = derived_sha256(repo)
    second = derived_sha256(repo)
    assert first == second
    (repo / "derived/sub/a.txt").write_text("changed", encoding="utf-8")
    assert derived_sha256(repo) != first


def test_derived_sha256_empty_directory_and_missing_directory(tmp_path: Path) -> None:
    """Return the empty-string digest for an empty or missing derived directory."""
    import hashlib

    assert derived_sha256(tmp_path) == hashlib.sha256(b"").hexdigest()
    (tmp_path / "derived").mkdir()
    assert derived_sha256(tmp_path) == hashlib.sha256(b"").hexdigest()


def test_code_sha256_ignores_docs_commit_but_tracks_source_and_entrypoint(tmp_path: Path) -> None:
    """Hash committed src identity plus current entrypoint object content only."""
    repo = _repo(tmp_path)
    before = code_sha256(repo, Path("entry.py"))
    (repo / "docs/note.md").write_text("docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/note.md")
    _git(repo, "commit", "-qm", "docs")
    assert code_sha256(repo, Path("entry.py")) == before
    (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/module.py")
    _git(repo, "commit", "-qm", "source")
    source_changed = code_sha256(repo, Path("entry.py"))
    assert source_changed != before
    (repo / "entry.py").write_text("print('two')\n", encoding="utf-8")
    assert code_sha256(repo, Path("entry.py")) != source_changed


@pytest.mark.parametrize("target", ["src", "configs", "derived", "job_x.py"])
def test_git_state_detects_configured_paths_and_root_jobs(tmp_path: Path, target: str) -> None:
    """Treat tracked edits and untracked root job scripts as dirty paths."""
    repo = _repo(tmp_path)
    head, dirty = git_state(repo)
    assert head
    assert not dirty
    if target == "src":
        (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif target == "configs":
        (repo / "configs/other.yaml").write_text("new: true\n", encoding="utf-8")
    elif target == "derived":
        (repo / "derived/new.json").write_text("{}\n", encoding="utf-8")
    else:
        (repo / target).write_text("# job\n", encoding="utf-8")
    assert git_state(repo)[1]


def test_git_state_ignores_untracked_docs(tmp_path: Path) -> None:
    """Ignore documentation-only untracked files in the runtime clean-path gate."""
    repo = _repo(tmp_path)
    (repo / "docs/untracked.md").write_text("note\n", encoding="utf-8")
    assert git_state(repo)[1] is False


def test_capture_run_git_freezes_commit_and_hashes_code_tree(tmp_path: Path) -> None:
    """Capture source, scripts and committed code object ids before docs commits."""
    repo = _repo(tmp_path)
    cfg = {"runtime": {"git_clean_paths": ["src", "configs", "derived"]}}
    captured = capture_run_git(repo, cfg)
    assert captured.git_commit == _git(repo, "rev-parse", "HEAD")
    assert set(captured.code_tree) == {
        "src", "configs", "derived", "scripts", "scripts/analyze_e6.py",
    }
    assert len(captured.code_tree_sha256) == 64
    (repo / "docs/note.md").write_text("docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/note.md")
    _git(repo, "commit", "-qm", "docs")
    assert captured.git_commit != _git(repo, "rev-parse", "HEAD")
    assert capture_run_git(repo, cfg).code_tree_sha256 == captured.code_tree_sha256


def test_capture_run_git_tracks_nested_script_entrypoints_and_dirty_state(tmp_path: Path) -> None:
    """Capture the nested E6 script blob and refuse a dirty script tree."""
    repo = _repo(tmp_path)
    cfg = {"runtime": {"git_clean_paths": ["src", "configs", "derived"]}}
    captured = capture_run_git(repo, cfg)
    assert "scripts/analyze_e6.py" in captured.code_tree
    assert not captured.git_dirty

    script = repo / "scripts/analyze_e6.py"
    script.write_text("def main():\n    return 1\n", encoding="utf-8")
    dirty = capture_run_git(repo, cfg)
    assert dirty.git_dirty


def test_e1_checks_sidecar_load_manifest_has_both_dtypes() -> None:
    """Describe both model loads that contribute to E1 checks, including override."""
    loads = e1_model_loads("org/model", "a" * 40, "checks")
    assert [load["dtype"] for load in loads] == ["torch.bfloat16", "torch.float32"]
    assert [load["dtype_override"] for load in loads] == [None, "float32"]
    assert all(load["model_id"] == "org/model" for load in loads)
    assert all(load["model_revision"] == "a" * 40 for load in loads)


def test_differences_lists_only_changed_fields() -> None:
    """List provenance keys whose values changed, preserving requested order."""
    saved = {"unit_key": {"model": "a"}, "dtype": "bf16", "batch_size": 4}
    current = {"unit_key": {"model": "a"}, "dtype": "fp32", "batch_size": 8}
    assert differences(saved, current, ["unit_key", "dtype", "batch_size"]) == ["dtype", "batch_size"]
