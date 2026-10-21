"""Tests for safe cache paths and pre-import runtime checks."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from vilens.env_guard import EnvGuardError, REQUIRED_VARS, check_environment, check_no_heavy_imports


def good_environment() -> dict[str, str]:
    """Build an isolated environment satisfying every rule-3 path requirement."""
    return {
        "HF_HOME": "/datastore/luannt/.cache/huggingface",
        "HF_HUB_CACHE": "/datastore/shared/models",
        "TORCH_HOME": "/datastore/luannt/.cache/torch",
        "PIP_CACHE_DIR": "/datastore/luannt/.cache/pip",
        "UV_CACHE_DIR": "/datastore/luannt/.cache/uv",
        "XDG_CACHE_HOME": "/datastore/luannt/.cache",
        "TRITON_CACHE_DIR": "/datastore/luannt/.cache/triton",
    }


def test_good_environment_passes() -> None:
    """Accept the expected rule-3 paths and a repo-contained cwd."""
    check_environment(good_environment(), "/datastore/luannt/Vie-Lens/tests")


def test_missing_variable_fails() -> None:
    """Reject a missing required cache variable."""
    values = good_environment()
    del values[REQUIRED_VARS[0]]
    with pytest.raises(EnvGuardError, match="unset or empty"):
        check_environment(values, "/datastore/luannt/Vie-Lens")


def test_outside_datastore_fails() -> None:
    """Reject cache paths outside the account datastore."""
    values = good_environment()
    values["HF_HOME"] = "/tmp/cache"
    with pytest.raises(EnvGuardError, match="inside /datastore/luannt"):
        check_environment(values, "/datastore/luannt/Vie-Lens")


def test_datastore_prefix_collision_fails() -> None:
    """Reject a sibling path whose string merely starts with the datastore prefix."""
    values = good_environment()
    values["HF_HOME"] = "/datastore/luannt2/cache"
    with pytest.raises(EnvGuardError, match="inside /datastore/luannt"):
        check_environment(values, "/datastore/luannt/Vie-Lens")


def test_wrong_shared_model_cache_fails() -> None:
    """Require the shared model cache to resolve to its exact approved path."""
    values = good_environment()
    values["HF_HUB_CACHE"] = "/shared" + "/h200node01/models"
    with pytest.raises(EnvGuardError, match="must resolve to /datastore/shared/models"):
        check_environment(values, "/datastore/luannt/Vie-Lens")


def test_custom_hub_cache_accepts_datastore_cache() -> None:
    """Allow the authorized login-node data cache when explicitly selected."""
    values = good_environment()
    values["HF_HUB_CACHE"] = "/datastore/luannt/.cache/huggingface/hub"
    check_environment(
        values,
        "/datastore/luannt/Vie-Lens",
        hub_cache="/datastore/luannt/.cache/huggingface/hub",
    )


def test_custom_hub_cache_rejects_shared_cache() -> None:
    """Reject the shared model cache when a datastore cache is required."""
    values = good_environment()
    with pytest.raises(EnvGuardError, match="HF_HUB_CACHE must resolve to /datastore/luannt/.cache/huggingface/hub"):
        check_environment(
            values,
            "/datastore/luannt/Vie-Lens",
            hub_cache="/datastore/luannt/.cache/huggingface/hub",
        )


def test_cwd_outside_repo_fails() -> None:
    """Reject current directories outside the repository root."""
    with pytest.raises(EnvGuardError, match="Current directory"):
        check_environment(good_environment(), "/datastore/luannt/Other")


def test_heavy_import_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect a simulated heavy import and let pytest restore sys.modules afterward."""
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(EnvGuardError, match="torch"):
        check_no_heavy_imports()
