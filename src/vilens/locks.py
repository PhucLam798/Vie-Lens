"""Fail-closed guards for preregistered test items and adjudication."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable


class IntegrityLockError(RuntimeError):
    """Raised when an integrity lock does not permit the requested operation."""


def ensure_model_access_allowed(cfg: dict[str, Any], records: Iterable[dict[str, Any]]) -> None:
    """Reject locked test prompts and D12-overlap directions before preregistration unlock."""
    record_list = list(records)
    prereg = cfg.get("prereg", {})
    analysis_tag = prereg.get("analysis_tag")
    unlocked = prereg.get("test_items_unlocked")
    lock_open = isinstance(analysis_tag, str) and bool(analysis_tag.strip()) and unlocked is True
    if lock_open:
        return
    if any(record.get("split") == "test" for record in record_list):
        raise IntegrityLockError(
            "Test-item model access requires a non-empty prereg.analysis_tag "
            "and prereg.test_items_unlocked is true"
        )
    excluded = {
        (item["concept_id"], item["format"], item["lang"])
        for item in cfg.get("data", {}).get("pretag_excluded_directions", [])
        if isinstance(item, dict)
        and all(isinstance(item.get(key), str) for key in ("concept_id", "format", "lang"))
    }
    blocked = [
        record
        for record in record_list
        if record.get("split") in {"fit", "validate"}
        and (record.get("concept_id"), record.get("format"), record.get("lang")) in excluded
    ]
    if blocked:
        identities = [
            (record.get("concept_id"), record.get("format"), record.get("lang")) for record in blocked
        ]
        raise IntegrityLockError(
            "Pre-tag D12 lock forbids model access to excluded directions records: "
            + ", ".join(map(str, identities))
        )


def pretag_filter(
    cfg: dict[str, Any], records: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split D12-overlap directions only while the preregistration lock is closed."""
    prereg = cfg.get("prereg", {})
    analysis_tag = prereg.get("analysis_tag")
    lock_open = isinstance(analysis_tag, str) and bool(analysis_tag.strip()) and prereg.get(
        "test_items_unlocked"
    ) is True
    if lock_open:
        return list(records), []
    excluded_ids = {
        (item["concept_id"], item["format"], item["lang"])
        for item in cfg.get("data", {}).get("pretag_excluded_directions", [])
        if isinstance(item, dict)
        and all(isinstance(item.get(key), str) for key in ("concept_id", "format", "lang"))
    }
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        identity = (record.get("concept_id"), record.get("format"), record.get("lang"))
        if record.get("split") in {"fit", "validate"} and identity in excluded_ids:
            excluded.append(record)
        else:
            kept.append(record)
    return kept, excluded


def ensure_adjudication_committed(repo_root: str | Path) -> None:
    """Require reports/adjudication.md to be tracked and clean relative to HEAD."""
    root = Path(repo_root)
    relative = "reports/adjudication.md"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise IntegrityLockError(f"{relative} must be tracked in git")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if clean.returncode != 0:
        raise IntegrityLockError(f"{relative} has uncommitted changes")
