"""Tests for fail-soft X0c audit helpers without loading real tokenizers."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "x0c_tokenizers.py"


def _load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the entrypoint module after preparing its required environment."""
    monkeypatch.chdir(ROOT)
    from vilens import env_guard

    # Other test modules import the model tooling before this test is collected.
    # The script import is structural here; it must not initialize an audit.
    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    env_values = {
        "HF_HOME": "/datastore/luannt/.cache/huggingface",
        "HF_HUB_CACHE": "/datastore/shared/models",
        "TORCH_HOME": "/datastore/luannt/.cache/torch",
        "PIP_CACHE_DIR": "/datastore/luannt/.cache/pip",
        "UV_CACHE_DIR": "/datastore/luannt/.cache/uv",
        "XDG_CACHE_HOME": "/datastore/luannt/.cache",
        "TRITON_CACHE_DIR": "/datastore/luannt/.cache/triton",
    }
    for name, value in env_values.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("x0c_tokenizers_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_writer_records_error_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception creates an ERROR row while a later check still executes."""
    module = _load_script(monkeypatch)
    logger = logging.getLogger("x0c-test")
    checks: dict[str, dict[str, str]] = {}

    def broken_check() -> tuple[str, str]:
        raise RuntimeError("test failure")

    module._run_check(checks, logger, "C21", "broken", "expected", broken_check)
    module._run_check(checks, logger, "C22", "following", "yes", lambda: ("yes", "MATCH"))

    assert checks["C21"]["result"] == "ERROR"
    assert checks["C21"]["actual"] == "RuntimeError: test failure"
    assert checks["C22"]["result"] == "MATCH"


def test_smoke_option_routes_under_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --smoke argument selects the isolated logs output directory."""
    module = _load_script(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--smoke", "50"])
    args = module._parse_args()
    output = module._output_directory(args.smoke, "20260929T000000Z")

    assert args.smoke == 50
    assert output == ROOT / "logs" / "x0c_smoke_20260929T000000Z"
    assert output.is_relative_to(ROOT / "logs")
    assert not output.is_relative_to(ROOT / "reports")
