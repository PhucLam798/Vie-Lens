"""Tests for the guarded primary-analysis Slurm wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import job_analyze_core


def test_primary_wrapper_requires_checkpoint_confirmation() -> None:
    """A real primary analysis must be explicitly confirmed."""
    with pytest.raises(SystemExit) as exc_info:
        job_analyze_core.main(["--stages", "core"])

    assert exc_info.value.code == 2


def test_primary_wrapper_preserves_requested_stage_list_order() -> None:
    """Every requested stage reaches the child unchanged and in input order."""
    stages = ["competence", "core", "c1"]
    work = job_analyze_core._build_work_items(stages, ("gemma", "qwen"))

    expected_stages = [
        "competence", "competence", "core", "core", "c1", "c1"
    ]
    assert [stage for _, stage in work] == expected_stages

    passed_stages = []
    for family, stage in work:
        child_args = job_analyze_core._analysis_args("cfg.yaml", stage, family)
        stage_option = child_args.index("--stages")
        passed_stages.append(child_args[stage_option + 1])
    assert passed_stages == expected_stages


def test_checkpoint_stage_is_dispatched_once_as_cross_family() -> None:
    """The final report stage runs once after family-local stages."""
    assert job_analyze_core._build_work_items(
        ["core", "checkpoint_h"], ("gemma", "qwen", "llama")
    ) == [
        ("gemma", "core"),
        ("qwen", "core"),
        ("llama", "core"),
        (None, "checkpoint_h"),
    ]


def test_check_inputs_is_dispatched_once_across_all_families() -> None:
    """The whole-run input check reaches analyze_core as one cross-family stage."""
    assert job_analyze_core._build_work_items(
        ["check_inputs"], ("gemma", "qwen", "llama")
    ) == [(None, "check_inputs")]
    args = job_analyze_core._analysis_args("cfg.yaml", "check_inputs", None)
    assert args[args.index("--stages") + 1] == "check_inputs"
    assert "--family" not in args


def test_figures_is_dispatched_once_as_a_separate_cross_family_stage() -> None:
    """Plot rendering is a separate aggregate stage after Checkpoint-H tables."""
    assert job_analyze_core._build_work_items(
        ["checkpoint_h", "figures"], ("gemma", "qwen", "llama")
    ) == [(None, "checkpoint_h"), (None, "figures")]
    args = job_analyze_core._analysis_args("cfg.yaml", "figures", None)
    assert args[args.index("--stages") + 1] == "figures"


def test_stage_subprocess_receives_fixed_thread_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child gets the declared caps even when the parent inherited other values."""
    captured: dict[str, str] = {}

    class CompletedChild:
        """Minimal completed subprocess stand-in."""

        pid = -1

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command, **kwargs):
        captured.update(kwargs["env"])
        return CompletedChild()

    monkeypatch.setattr(job_analyze_core, "REPO", tmp_path)
    monkeypatch.setattr(job_analyze_core.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(job_analyze_core, "_sample_peak_rss_bytes", lambda _pid: 0)
    monkeypatch.setenv("SLURM_JOB_ID", "test-123")
    for name in job_analyze_core.THREAD_CAPS:
        monkeypatch.setenv(name, "99" if name != "TOKENIZERS_PARALLELISM" else "true")

    status = job_analyze_core._run_stage(
        "configs/experiments.yaml",
        "gemma",
        "core",
        analysis_provenance={
            "frozen_analysis_commit": "frozen-hash",
            "executed_commit": "head-hash",
            "executed_tree": "tree-hash",
            "code_tree_sha256": "tree-sha256",
        },
    )

    assert status == 0
    assert {name: captured[name] for name in job_analyze_core.THREAD_CAPS} == job_analyze_core.THREAD_CAPS
    sidecars = list((tmp_path / "results/analysis/primary/provenance").glob("*.provenance.json"))
    assert len(sidecars) == 1
    record = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert record["frozen_analysis_commit"] == "frozen-hash"
    assert record["executed_commit"] == "head-hash"
    assert record["thread_environment"] == job_analyze_core.THREAD_CAPS
