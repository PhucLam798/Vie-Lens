"""Shared fail-fast preflight checks for offline model jobs."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import locks, manifest, models
from .config import study_models
from .provenance import git_state

CheckStatus = Literal["PASS", "FAIL", "SKIPPED", "BLOCKED"]


@dataclass(frozen=True)
class Check:
    """One named preflight result and its human-readable detail."""

    name: str
    status: CheckStatus
    detail: str


def _check(name: str, status: CheckStatus, detail: str) -> Check:
    """Construct a check while keeping exception handling concise."""
    return Check(name=name, status=status, detail=detail)


def run_preflight(
    cfg: dict[str, Any],
    *,
    repo_root: str | Path,
    step: str,
    gpu: bool,
    inputs: list[str],
    families: list[str],
    records: list[dict[str, Any]] | None,
    output_dirs: list[Path],
    entrypoint: Path | None,
) -> list[Check]:
    """Run environment, input, checkpoint, output, Git, and integrity checks in order."""
    root = Path(repo_root)
    checks: list[Check] = []

    if not gpu:
        checks.append(_check("gpu_visible", "SKIPPED", "GPU check skipped for login-node preflight"))
        checks.append(_check("bf16_matmul", "SKIPPED", "GPU check skipped for login-node preflight"))
    else:
        try:
            import torch

            if torch.cuda.is_available():
                checks.append(_check("gpu_visible", "PASS", torch.cuda.get_device_name(0)))
            else:
                checks.append(_check("gpu_visible", "FAIL", "torch.cuda.is_available() is false"))
        except Exception as exc:
            checks.append(_check("gpu_visible", "FAIL", f"{type(exc).__name__}: {exc}"))
        try:
            import torch

            left = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
            result = left @ left
            if torch.isfinite(result).all().item():
                checks.append(_check("bf16_matmul", "PASS", "64x64 bf16 CUDA matmul is finite"))
            else:
                checks.append(_check("bf16_matmul", "FAIL", "64x64 bf16 CUDA matmul returned non-finite values"))
        except Exception as exc:
            checks.append(_check("bf16_matmul", "FAIL", f"{type(exc).__name__}: {exc}"))

    try:
        results = manifest.verify_manifest(root / cfg["paths"]["manifest"], root, only=inputs)
        checks.append(_check("inputs_sha256", "PASS", f"{len(results)}/{len(inputs)} selected inputs verified"))
    except Exception as exc:
        checks.append(_check("inputs_sha256", "FAIL", f"{type(exc).__name__}: {exc}"))

    if cfg.get("models", {}).get("study_set") is None:
        checks.append(_check("snapshots", "BLOCKED", "D10 undecided: models.study_set is null"))
    else:
        try:
            selected = study_models(cfg)
            cache_dir = Path(cfg["paths"]["model_cache"])
            checked: list[str] = []
            for family in families:
                entry = selected[family]
                spec = models.ModelSpec(
                    cfg["models"]["study_set"], family, entry["hf_id"], entry["revision"]
                )
                snapshot = models.resolve_snapshot(spec, cache_dir)
                problems = models.snapshot_problems(snapshot)
                if problems:
                    raise models.SnapshotError(
                        f"{spec.hf_id}@{spec.revision}: " + "; ".join(problems)
                    )
                tokenizer = models.load_tokenizer(spec, cache_dir)
                hf_config = models.load_hf_config(spec, cache_dir)
                if getattr(tokenizer, "is_fast", False) is not True:
                    raise models.SnapshotError(f"Tokenizer is not fast for {spec.hf_id}@{spec.revision}")
                checked.append(f"{family}={spec.hf_id}@{spec.revision} ({type(hf_config).__name__})")
            checks.append(_check("snapshots", "PASS", "; ".join(checked)))
        except Exception as exc:
            checks.append(_check("snapshots", "FAIL", f"{type(exc).__name__}: {exc}"))

    output_errors: list[str] = []
    for directory in output_dirs:
        path = Path(directory)
        if not path.is_absolute():
            path = root / path
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, prefix=".preflight-", delete=False) as stream:
                probe = Path(stream.name)
            probe.unlink()
        except Exception as exc:
            output_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if output_errors:
        checks.append(_check("output_dirs_writable", "FAIL", "; ".join(output_errors)))
    else:
        checks.append(_check("output_dirs_writable", "PASS", f"{len(output_dirs)} output directories writable"))

    try:
        commit, dirty = git_state(root)
        if dirty:
            checks.append(_check("git_clean", "FAIL", f"working paths are dirty at {commit}"))
        else:
            checks.append(_check("git_clean", "PASS", commit))
    except Exception as exc:
        checks.append(_check("git_clean", "FAIL", f"{type(exc).__name__}: {exc}"))

    lock_errors: list[str] = []
    if records is not None:
        try:
            locks.ensure_model_access_allowed(cfg, records)
        except Exception as exc:
            lock_errors.append(f"{type(exc).__name__}: {exc}")
    if step in {"E6", "E7", "E8"}:
        try:
            locks.ensure_adjudication_committed(root)
        except Exception as exc:
            lock_errors.append(f"{type(exc).__name__}: {exc}")
    checks.append(
        _check(
            "integrity_locks",
            "FAIL" if lock_errors else "PASS",
            "; ".join(lock_errors) if lock_errors else "applicable integrity locks pass",
        )
    )
    return checks


def preflight_ok(checks: list[Check]) -> bool:
    """Return true only when no check failed or was blocked."""
    return all(check.status not in {"FAIL", "BLOCKED"} for check in checks)


def format_checks(checks: list[Check]) -> str:
    """Format preflight results as an aligned plain-text table."""
    headers = ("CHECK", "STATUS", "DETAIL")
    rows = [(check.name, check.status, check.detail.replace("\n", " ")) for check in checks]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(3)]
    line = " | ".join(headers[i].ljust(widths[i]) for i in range(3))
    divider = "-+-".join("-" * widths[i] for i in range(3))
    body = [" | ".join(row[i].ljust(widths[i]) for i in range(3)) for row in rows]
    return "\n".join([line, divider, *body])
