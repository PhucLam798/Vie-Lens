"""Tests for the Part A ancestry and clean-tree guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vilens.analysis.freeze import assert_frozen_analysis_tree


def _git(repo: Path, *args: str) -> str:
    """Run a git command in an isolated fixture repository."""
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _init_repo(repo: Path, *, content: str = "tracked") -> str:
    """Create a clean commit containing every path protected by the guard."""
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    for directory in ("src", "scripts", "configs", "derived"):
        path = repo / directory / "tracked.txt"
        path.parent.mkdir(parents=True)
        path.write_text(f"{content}\n", encoding="utf-8")
    (repo / "job_fixture.py").write_text("pass\n", encoding="utf-8")
    _git(repo, "add", "src", "scripts", "configs", "derived", "job_fixture.py")
    _git(repo, "commit", "-qm", "fixture freeze")
    return _git(repo, "rev-parse", "HEAD")


def test_frozen_guard_accepts_clean_descendant(tmp_path: Path) -> None:
    """A clean HEAD may descend from, rather than equal, the freeze commit."""
    repo = tmp_path / "descendant"
    frozen = _init_repo(repo)
    (repo / "docs.txt").write_text("later documentation\n", encoding="utf-8")
    _git(repo, "add", "docs.txt")
    _git(repo, "commit", "-qm", "documentation after freeze")

    identity = assert_frozen_analysis_tree(repo, frozen_commit=frozen)

    assert identity.frozen_analysis_commit == frozen
    assert identity.executed_commit == _git(repo, "rev-parse", "HEAD")
    assert identity.executed_commit != identity.frozen_analysis_commit


def test_frozen_guard_rejects_dirty_analysis_tree(tmp_path: Path) -> None:
    """An untracked file in a protected code path fails the guard."""
    repo = tmp_path / "dirty"
    frozen = _init_repo(repo)
    (repo / "scripts" / "untracked.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean analysis code tree"):
        assert_frozen_analysis_tree(repo, frozen_commit=frozen)


def test_frozen_guard_rejects_unrelated_head(tmp_path: Path) -> None:
    """A commit from another history is not accepted as a frozen ancestor."""
    frozen_repo = tmp_path / "frozen"
    frozen = _init_repo(frozen_repo)
    unrelated_repo = tmp_path / "unrelated"
    _init_repo(unrelated_repo, content="unrelated")

    with pytest.raises(RuntimeError, match="to be an ancestor"):
        assert_frozen_analysis_tree(unrelated_repo, frozen_commit=frozen)
