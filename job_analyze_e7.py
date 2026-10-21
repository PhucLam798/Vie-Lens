#!/datastore/luannt/.cache/sb/bin/python
"""Root job wrapper for the E7 CPU analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Sequence

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

THREAD_CAPS = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
    "ARROW_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}


def resolve_repo_path(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a config or script path from the repository root, independent of cwd."""
    root = (repo_root or REPO).resolve()
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    """Build the supported wrapper arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _provenance_path() -> Path:
    """Return a per-job sidecar path whose name cannot collide across Slurm jobs."""
    job_id = os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}")
    return REPO / "results/analysis/primary/e7_provenance" / f"job={job_id}.json"


def _write_provenance(path: Path, record: dict[str, object]) -> None:
    """Write a unique OPS-32 analysis provenance sidecar without replacing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the E7 analyzer after lock, code-tree, and metadata preflight checks."""
    args = _parser().parse_args(argv)
    os.environ.update(THREAD_CAPS)

    from vilens.config import load_config
    from vilens.locks import ensure_adjudication_committed
    from vilens.manifest import sha256_file
    from vilens.provenance import capture_run_git, derived_sha256, git_state

    config_path = resolve_repo_path(args.config)
    cfg = load_config(config_path)
    ensure_adjudication_committed(REPO)
    commit, dirty = git_state(REPO)
    if dirty:
        raise RuntimeError(f"analysis code/config tree is dirty at {commit}")

    run_git = capture_run_git(REPO, cfg)
    provenance = {
        "analysis": "E7",
        "git_commit": run_git.git_commit,
        "git_dirty": run_git.git_dirty,
        "code_tree": dict(run_git.code_tree),
        "code_tree_sha256": run_git.code_tree_sha256,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(resolve_repo_path(str(cfg["paths"]["manifest"]))),
        "derived_sha256": derived_sha256(REPO),
        "entrypoint": "job_analyze_e7.py",
        "analyzer": "scripts/analyze_e7.py",
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "thread_environment": {name: os.environ[name] for name in THREAD_CAPS},
        "utc_start": _utc_now(),
    }

    from scripts import analyze_e7

    forwarded = ["--config", str(config_path)]
    if args.preflight_only:
        forwarded.append("--preflight")
        print(f"Checkpoint-H lock: PASS; committed analysis tree: {commit}")
        return analyze_e7.main(forwarded)

    sidecar_path = _provenance_path()
    if sidecar_path.exists():
        raise FileExistsError(f"refusing to replace existing provenance: {sidecar_path}")
    status = 1
    try:
        status = int(analyze_e7.main(forwarded))
        provenance["exit_code"] = status
        return status
    except BaseException as exc:
        provenance["exception_type"] = type(exc).__name__
        provenance["exception_message"] = str(exc)
        traceback.print_exc()
        raise
    finally:
        provenance["exit_code"] = status
        provenance["utc_end"] = _utc_now()
        _write_provenance(sidecar_path, provenance)


if __name__ == "__main__":
    env_guard.guard(offline=True)
    raise SystemExit(main())
