"""Environment checks that run before importing heavyweight libraries."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

REQUIRED_VARS: tuple[str, ...] = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
)
DATASTORE = "/datastore/luannt"
SHARED_MODELS = "/datastore/shared/models"
REPO_ROOT = "/datastore/luannt/Vie-Lens"
_HEAVY_IMPORTS = ("torch", "transformers", "tokenizers", "datasets", "matplotlib")


class EnvGuardError(RuntimeError):
    """Raised when the runtime violates a required filesystem or import rule."""


def _is_within(path: str, root: str) -> bool:
    """Return whether a resolved path is root or one of its descendants."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    try:
        return os.path.commonpath((resolved_root, resolved_path)) == resolved_root
    except ValueError:
        return False


def check_environment(
    environ: Mapping[str, str] | None = None,
    cwd: str | None = None,
    *,
    hub_cache: str = SHARED_MODELS,
) -> None:
    """Validate cache variables and ensure the current directory is in the repo.

    ``hub_cache`` defaults to the shared model cache and may be overridden by
    the one authorized login-node data-download route.
    """
    values = os.environ if environ is None else environ
    for name in REQUIRED_VARS:
        value = values.get(name)
        if not value:
            raise EnvGuardError(f"Required environment variable {name} is unset or empty")
        resolved = os.path.realpath(value)
        if name == "HF_HUB_CACHE":
            expected_hub_cache = os.path.realpath(hub_cache)
            if resolved != expected_hub_cache:
                raise EnvGuardError(f"{name} must resolve to {expected_hub_cache}; got {resolved}")
        elif not _is_within(resolved, DATASTORE):
            raise EnvGuardError(f"{name} must resolve inside {DATASTORE}; got {resolved}")
    current = os.getcwd() if cwd is None else cwd
    if not _is_within(current, REPO_ROOT):
        raise EnvGuardError(f"Current directory must resolve inside {REPO_ROOT}; got {os.path.realpath(current)}")


def check_no_heavy_imports() -> None:
    """Reject execution after heavyweight ML/data libraries have been imported."""
    present = [name for name in _HEAVY_IMPORTS if name in sys.modules]
    if present:
        raise EnvGuardError("Heavy imports already loaded: " + ", ".join(present))


def set_runtime_dirs(environ: MutableMapping[str, str] | None = None) -> None:
    """Set safe plotting/CUDA cache defaults and create their directories."""
    values = os.environ if environ is None else environ
    values.setdefault("MPLCONFIGDIR", f"{DATASTORE}/.cache/matplotlib")
    values.setdefault("CUDA_CACHE_PATH", f"{DATASTORE}/.cache/nv")
    Path(values["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(values["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)


def set_offline(environ: MutableMapping[str, str] | None = None) -> None:
    """Set Hugging Face library offline flags in the supplied environment."""
    values = os.environ if environ is None else environ
    values["HF_HUB_OFFLINE"] = "1"
    values["TRANSFORMERS_OFFLINE"] = "1"
    values["HF_DATASETS_OFFLINE"] = "1"


def guard(offline: bool) -> None:
    """Check imports and paths, configure runtime dirs, and optionally force offline mode."""
    try:
        check_no_heavy_imports()
        check_environment()
        set_runtime_dirs()
        if offline:
            set_offline()
    except EnvGuardError as exc:
        print(f"Environment guard failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
