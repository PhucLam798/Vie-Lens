"""Guard primary analyses against changes to the frozen Part A code."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

FROZEN_ANALYSIS_COMMIT = "42c973c014bacd458e81477d38e9f3a57f61a397"
CLEAN_PATHS = ("src", "scripts", "configs", "derived", ":(glob)job_*.py")


@dataclass(frozen=True)
class AnalysisCommitIdentity:
    """Frozen Part A commit and the exact commit executing the analysis."""

    frozen_analysis_commit: str
    executed_commit: str


def assert_frozen_analysis_tree(
    repo: str | Path,
    *,
    frozen_commit: str = FROZEN_ANALYSIS_COMMIT,
) -> AnalysisCommitIdentity:
    """Require the frozen commit to be an ancestor and analysis paths to be clean."""
    root = Path(repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, head],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        detail = ancestor.stderr.strip() or f"exit code {ancestor.returncode}"
        raise RuntimeError(
            f"Part C requires frozen commit {frozen_commit} to be an ancestor of "
            f"executed HEAD {head}: {detail}"
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *CLEAN_PATHS,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Part C requires a clean analysis code tree: " + status.strip())
    return AnalysisCommitIdentity(frozen_commit, head)
