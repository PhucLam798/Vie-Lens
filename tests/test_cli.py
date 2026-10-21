"""Tests for common entrypoint parsing and stage timing."""

from __future__ import annotations

import logging

from vilens.cli import StageTimer, base_parser


def test_base_parser_defaults() -> None:
    """Provide the shared config default and preflight-only flag."""
    args = base_parser("test task").parse_args([])
    assert args.config == "configs/experiments.yaml"
    assert args.preflight_only is False
    assert base_parser("test task").parse_args(["--preflight-only"]).preflight_only is True


def test_stage_timer_records_named_stages() -> None:
    """Keep named durations in the caller-owned final summary mapping."""
    timings: dict[str, float] = {}
    logger = logging.getLogger("test-stage-timer")
    with StageTimer("fixture", logger, timings):
        pass
    assert "fixture" in timings
    assert timings["fixture"] >= 0
