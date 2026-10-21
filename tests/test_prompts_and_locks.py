"""Tests for test-item gates, prompt integrity, and adjudication tracking."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from vilens.locks import (
    IntegrityLockError,
    ensure_adjudication_committed,
    ensure_model_access_allowed,
    pretag_filter,
)
from vilens.prompts import check_prompt_integrity, records_for_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _prompt_records(filename: str) -> list[dict[str, Any]]:
    """Read one of the small, hand-authored prompt fixtures."""
    import json

    return [json.loads(line) for line in (FIXTURES / filename).read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "cfg",
    [
        {"prereg": {"analysis_tag": None, "test_items_unlocked": False}},
        {"prereg": {"analysis_tag": "prereg-v1.4", "test_items_unlocked": False}},
        {"prereg": {"analysis_tag": None, "test_items_unlocked": True}},
    ],
)
def test_test_records_locked_until_both_controls_set(cfg: dict[str, Any]) -> None:
    """Require both human-controlled lock values before returning test prompts."""
    with pytest.raises(IntegrityLockError):
        ensure_model_access_allowed(cfg, _prompt_records("repetition.jsonl"))


def test_both_test_item_controls_set_passes() -> None:
    """Allow test records only when the preregistration tag and boolean are set."""
    records = _prompt_records("repetition.jsonl")
    cfg = {"prereg": {"analysis_tag": "prereg-v1.4", "test_items_unlocked": True}}
    assert records_for_model(cfg, records) is records


def test_directions_records_always_pass() -> None:
    """Allow fit/validate directions records while test locks remain closed."""
    cfg = {"prereg": {"analysis_tag": None, "test_items_unlocked": False}}
    records = _prompt_records("directions.jsonl")
    ensure_model_access_allowed(cfg, records)


def _d12_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a closed-lock config, its excluded direction record, and a safe direction record."""
    excluded = {"concept_id": "overlap", "format": "translation", "lang": "vi", "split": "fit"}
    safe = {"concept_id": "other", "format": "translation", "lang": "vi", "split": "fit"}
    cfg = {
        "prereg": {"analysis_tag": None, "test_items_unlocked": False},
        "data": {
            "pretag_excluded_directions": [
                {"concept_id": "overlap", "format": "translation", "lang": "vi"}
            ]
        },
    }
    return cfg, excluded, safe


def test_excluded_directions_fail_with_locks_closed() -> None:
    """Block D12-overlap directions records before the preregistration unlock."""
    cfg, excluded, _ = _d12_fixture()
    with pytest.raises(IntegrityLockError, match="Pre-tag D12 lock"):
        ensure_model_access_allowed(cfg, [excluded])


def test_excluded_directions_pass_after_both_locks_open() -> None:
    """Allow D12-overlap directions records once both human-controlled locks are open."""
    cfg, excluded, _ = _d12_fixture()
    cfg["prereg"] = {"analysis_tag": "prereg-v1.4", "test_items_unlocked": True}
    ensure_model_access_allowed(cfg, [excluded])


def test_nonexcluded_directions_pass_with_locks_closed() -> None:
    """Keep non-overlapping directions records available while test items remain locked."""
    cfg, _, safe = _d12_fixture()
    ensure_model_access_allowed(cfg, [safe])


def test_pretag_filter_splits_excluded_and_kept_records() -> None:
    """Return the configured overlap record separately without changing record objects."""
    cfg, excluded, safe = _d12_fixture()
    records = [excluded, safe]
    kept, removed = pretag_filter(cfg, records)
    assert kept == [safe]
    assert removed == [excluded]


def test_pretag_filter_excludes_only_while_lock_closed() -> None:
    """Exclude D12 records before unlock and keep all 1,320 after both controls open."""
    cfg, excluded, safe = _d12_fixture()
    filler = [
        {"concept_id": f"other-{index}", "format": "repetition", "lang": "vi", "split": "fit"}
        for index in range(1314)
    ]
    records = [excluded] * 5 + [safe, *filler]

    cfg["prereg"] = {"analysis_tag": "prereg-v1.4", "test_items_unlocked": False}
    kept, removed = pretag_filter(cfg, records)
    assert (len(kept), len(removed)) == (1315, 5)

    cfg["prereg"] = {"analysis_tag": "prereg-v1.4", "test_items_unlocked": True}
    kept, removed = pretag_filter(cfg, records)
    assert (len(kept), len(removed)) == (1320, 0)

    cfg["prereg"] = {"analysis_tag": None, "test_items_unlocked": True}
    kept, removed = pretag_filter(cfg, records)
    assert (len(kept), len(removed)) == (1315, 5)


def test_prompt_integrity_reports_whitespace_and_nfc_errors() -> None:
    """Detect trailing prompt whitespace, double-space target prefixes, and decomposed text."""
    record = {
        "concept_id": "fixture",
        "prompt": "Prompt: ",
        "target_vi": "  target",
        "target_en": " cafe\u0301",
    }
    problems = check_prompt_integrity([record])
    assert any("ends in whitespace" in issue for issue in problems)
    assert any("exactly one space" in issue for issue in problems)
    assert any("is not NFC" in issue for issue in problems)


def _git(cwd: Path, *args: str) -> None:
    """Run one local git command for the temporary lock test repository."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_adjudication_lock_untracked_committed_modified(tmp_path: Path) -> None:
    """Require adjudication to exist, be tracked, and match HEAD exactly."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    reports = root / "reports"
    reports.mkdir()
    adjudication = reports / "adjudication.md"
    adjudication.write_text("verdicts\n", encoding="utf-8")
    with pytest.raises(IntegrityLockError, match="must be tracked"):
        ensure_adjudication_committed(root)
    _git(root, "add", "reports/adjudication.md")
    _git(root, "commit", "-qm", "adjudication")
    ensure_adjudication_committed(root)
    adjudication.write_text("changed\n", encoding="utf-8")
    with pytest.raises(IntegrityLockError, match="uncommitted changes"):
        ensure_adjudication_committed(root)
