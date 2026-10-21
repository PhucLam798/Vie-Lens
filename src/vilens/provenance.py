"""Build and compare run provenance for resume-safe result units."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import sha256_file

SIDECAR_FIELDS = frozenset(
    {
        "unit_key",
        "git_commit",
        "git_dirty",
        "code_tree",
        "code_tree_sha256",
        "config_sha256",
        "manifest_sha256",
        "derived_sha256",
        "code_sha256",
        "model_id",
        "model_revision",
        "dtype",
        "batch_size",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "python_version",
        "gpu_name",
        "slurm_job_id",
        "model_loads",
        "row_count",
        "expected_row_count",
        "utc_start",
        "utc_end",
    }
)


@dataclass(frozen=True)
class RunGit:
    """Git identity captured once at the beginning of a result-producing run."""

    git_commit: str
    git_dirty: bool
    code_tree: Mapping[str, str]
    code_tree_sha256: str


def capture_run_git(repo_root: str | Path, cfg: Mapping[str, Any]) -> RunGit:
    """Capture the commit, dirty state, and code-tree object ids once for a run."""
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    clean_paths = list(cfg["runtime"]["git_clean_paths"])
    status = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            *clean_paths, "scripts", ":(glob)job_*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    code_tree: dict[str, str] = {}
    for path in ("src", "configs", "derived"):
        code_tree[path] = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    script_tree = subprocess.run(
        ["git", "rev-parse", f"{commit}:scripts"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if script_tree.returncode == 0:
        code_tree["scripts"] = script_tree.stdout.strip()
        script_entries = subprocess.run(
            ["git", "ls-tree", "-r", "-z", commit, "--", "scripts"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    else:
        script_entries = b""
    for entry in script_entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        _mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type == "blob":
            code_tree[raw_path.decode("utf-8", errors="surrogateescape")] = object_id
    root_entries = subprocess.run(
        ["git", "ls-tree", commit], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    for line in root_entries.splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, object_id = metadata.split()
        if object_type == "blob" and path.startswith("job_") and path.endswith(".py"):
            code_tree[path] = object_id
    serialized = json.dumps(code_tree, sort_keys=True)
    tree_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return RunGit(commit, bool(status.strip()), code_tree, tree_sha256)


def _entrypoint_relative(repo_root: Path, entrypoint: Path) -> str:
    """Resolve an entrypoint argument to its repository-relative path."""
    path = Path(entrypoint)
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"entrypoint must be inside the repository: {entrypoint}") from exc
    return path.as_posix()


def _code_sha256_from_run(run_git: RunGit, repo_root: Path, entrypoint: Path | None) -> str:
    """Derive the legacy code digest from object ids already captured for this run."""
    objects = [run_git.code_tree["src"]]
    if entrypoint is not None:
        relative = _entrypoint_relative(repo_root, entrypoint)
        if relative not in run_git.code_tree:
            raise ValueError(f"entrypoint is absent from the captured code tree: {relative}")
        objects.append(run_git.code_tree[relative])
    return hashlib.sha256("\n".join(objects).encode("ascii")).hexdigest()


def e1_model_loads(model_id: str, model_revision: str, unit: str) -> list[dict[str, Any]]:
    """Describe the bf16 E1 load and, for the checks unit, its float32 override load."""
    records: list[dict[str, Any]] = [
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "dtype": "torch.bfloat16",
            "dtype_override": None,
        }
    ]
    if unit == "checks":
        records.append(
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "dtype": "torch.float32",
                "dtype_override": "float32",
            }
        )
    return records


def _utc_now() -> str:
    """Return the current UTC time in an ISO-8601 representation."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def derived_sha256(repo_root: str | Path) -> str:
    """Hash sorted path and file-digest records for visible regular derived files."""
    root = Path(repo_root)
    derived = root / "derived"
    records: list[str] = []
    if derived.is_dir():
        for path in derived.rglob("*"):
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_file() and not path.is_symlink():
                records.append(f"{relative.as_posix()}\t{sha256_file(path)}\n")
    return hashlib.sha256("".join(sorted(records)).encode("utf-8")).hexdigest()


def code_sha256(repo_root: str | Path, entrypoint: Path | None = None) -> str:
    """Hash the committed source-tree identity and, when supplied, entrypoint contents."""
    root = Path(repo_root)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD:src"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    digest = hashlib.sha256()
    digest.update(tree.encode("ascii"))
    if entrypoint is not None:
        entry = Path(entrypoint)
        if entry.is_absolute():
            try:
                entry = entry.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"entrypoint must be inside the repository: {entry}") from exc
        entry_hash = subprocess.run(
            ["git", "hash-object", str(entry)], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        digest.update(b"\n")
        digest.update(entry_hash.encode("ascii"))
    return digest.hexdigest()


def git_state(repo_root: str | Path) -> tuple[str, bool]:
    """Return HEAD and whether configured code/config/data paths or root jobs are dirty."""
    root = Path(repo_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    config_path = root / "configs/experiments.yaml"
    clean_paths: list[str] = ["src", "configs", "derived"]
    if config_path.is_file():
        from .config import load_config

        clean_paths = list(load_config(config_path)["runtime"]["git_clean_paths"])
    pathspecs = [*clean_paths, ":(glob)job_*.py"]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *pathspecs],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, bool(status.strip())


def build(
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    unit_key: Mapping[str, str | int],
    entrypoint: Path | None,
    model_id: str,
    model_revision: str,
    dtype: str,
    batch_size: int | None,
    expected_row_count: int,
    run_git: RunGit,
    model_loads: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a provenance sidecar payload before result writing begins."""
    import torch

    root = Path(repo_root)
    config_path = root / "configs/experiments.yaml"
    manifest_path = root / str(cfg["paths"]["manifest"])
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    loads = list(model_loads) if model_loads is not None else [
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "dtype": dtype,
            "dtype_override": None,
        }
    ]
    return {
        "unit_key": dict(unit_key),
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "code_tree": dict(run_git.code_tree),
        "code_tree_sha256": run_git.code_tree_sha256,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "derived_sha256": derived_sha256(root),
        "code_sha256": _code_sha256_from_run(run_git, root, entrypoint),
        "model_id": model_id,
        "model_revision": model_revision,
        "dtype": dtype,
        "batch_size": batch_size,
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "python_version": platform.python_version(),
        "gpu_name": gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_loads": loads,
        "expected_row_count": expected_row_count,
        "utc_start": _utc_now(),
    }


def differences(
    sidecar: Mapping[str, Any], current: Mapping[str, Any], fields: list[str] | tuple[str, ...]
) -> list[str]:
    """Return compared field names whose values differ between saved and current provenance."""
    return [field for field in fields if sidecar.get(field) != current.get(field)]
