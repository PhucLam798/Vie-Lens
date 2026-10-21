#!/datastore/luannt/.cache/sb/bin/python
"""Run the frozen E2–E5 primary analysis one family and stage at a time."""

from pathlib import Path
import hashlib
import json
import os
import sys

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

if __name__ == "__main__":
    env_guard.guard(offline=True)

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import signal
import subprocess
import time
from typing import TypeAlias

from vilens.config import load_config
from vilens.analysis.freeze import assert_frozen_analysis_tree

STAGES = ("core", "competence", "c1", "checkpoint_h", "figures", "check_inputs")
FAMILIES = ("gemma", "qwen", "llama")
MAX_STAGE_SECONDS = 60 * 60
ANALYSIS_SCRIPT = REPO / "scripts" / "analyze_core.py"
WorkItem: TypeAlias = tuple[str | None, str]
THREAD_CAPS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
    "ARROW_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}

_CHILD_RUNNER = r'''
import os
from pathlib import Path
import resource
import signal
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "scripts"))
report_fd = int(sys.argv[1])
analysis_args = sys.argv[2:]

class StageTimeout(Exception):
    """Raised when the parent reaches the per-stage time limit."""

def _handle_timeout(signum, frame):
    raise StageTimeout("family/stage exceeded 60 minutes")

signal.signal(signal.SIGTERM, _handle_timeout)
try:
    import analyze_core
    status = analyze_core.main(analysis_args)
finally:
    peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    try:
        os.write(report_fd, f"{peak_bytes}\n".encode("ascii"))
    finally:
        os.close(report_fd)
raise SystemExit(status)
'''


def _timestamp() -> str:
    """Return a compact UTC timestamp for stage logs."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_work_items(stages: Sequence[str], families: Sequence[str]) -> list[WorkItem]:
    """Keep the requested stage order and run each family-local stage serially."""
    items: list[WorkItem] = []
    for stage in stages:
        if stage in {"checkpoint_h", "figures", "check_inputs"}:
            # These stages combine or validate inputs across all model families.
            items.append((None, stage))
        else:
            items.extend((family, stage) for family in families)
    return items


def _analysis_args(config: str, stage: str, family: str | None) -> list[str]:
    """Build the exact single-stage arguments passed to analyze_core.main."""
    args = [
        "--config", config,
        "--source", "primary",
        "--confirm-checkpoint-h",
        "--stages", stage,
    ]
    if family is not None:
        args.extend(("--family", family))
    return args


def _job_environment() -> dict[str, str]:
    """Copy the current environment and enforce the analysis job's thread caps."""
    environment = os.environ.copy()
    environment.update(THREAD_CAPS)
    return environment


def _set_job_thread_caps() -> None:
    """Apply the fixed analysis caps to this job before its preflight imports."""
    os.environ.update(THREAD_CAPS)


def _thread_caps_log(environment: Mapping[str, str]) -> str:
    """Format the complete thread environment for wrapper logs and sidecars."""
    return ",".join(f"{name}={environment[name]}" for name in THREAD_CAPS)


def _analysis_provenance(repo: Path) -> dict[str, str]:
    """Capture both commits and a reproducible digest of the executed tree."""
    identity = assert_frozen_analysis_tree(repo)
    tree = subprocess.run(
        ["git", "rev-parse", f"{identity.executed_commit}^{{tree}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree_listing = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", identity.executed_commit],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "frozen_analysis_commit": identity.frozen_analysis_commit,
        "executed_commit": identity.executed_commit,
        "executed_tree": tree,
        "code_tree_sha256": hashlib.sha256(tree_listing).hexdigest(),
    }


def _write_stage_sidecar(record: Mapping[str, object], *, repo: Path = REPO) -> Path:
    """Atomically write the provenance record for one family × stage."""
    job_id = os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}")
    family = str(record["family"]).replace("/", "_")
    stage = str(record["stage"]).replace("/", "_")
    sidecar_dir = repo / "results/analysis/primary/provenance"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / f"job={job_id}__family={family}__stage={stage}.provenance.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(record), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _sample_peak_rss_bytes(pid: int) -> int:
    """Read the child process's Linux high-water RSS, returning zero if gone."""
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return 0
    return 0


def _run_stage(
    config: str,
    family: str | None,
    stage: str,
    *,
    analysis_provenance: Mapping[str, str],
) -> int:
    """Run one analysis stage, enforce its time limit, and report peak RSS."""
    family_label = family or "all"
    started_at = _timestamp()
    environment = _job_environment()
    thread_caps = _thread_caps_log(environment)
    stage_record: dict[str, object] = {
        **analysis_provenance,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "family": family_label,
        "stage": stage,
        "started_at": started_at,
        "ended_at": None,
        "wall_time_seconds": None,
        "exit_code": None,
        "timed_out": None,
        "peak_rss_bytes": None,
        "thread_environment": {name: environment[name] for name in THREAD_CAPS},
    }
    sidecar_path = _write_stage_sidecar(stage_record, repo=REPO)
    print(
        f"{started_at} INFO analyze_core family={family_label} stage={stage} "
        f"event=start timeout_seconds={MAX_STAGE_SECONDS} "
        f"frozen_analysis_commit={analysis_provenance['frozen_analysis_commit']} "
        f"executed_commit={analysis_provenance['executed_commit']} "
        f"code_tree_sha256={analysis_provenance['code_tree_sha256']} thread_caps={thread_caps}",
        flush=True,
    )

    read_fd, write_fd = os.pipe()
    command = [
        sys.executable,
        "-c",
        _CHILD_RUNNER,
        str(write_fd),
        *_analysis_args(config, stage, family),
    ]
    child: subprocess.Popen[bytes] | None = None
    sampled_peak = 0
    timed_out = False
    try:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
        os.close(write_fd)
        write_fd = -1
        started_clock = time.monotonic()
        while child.poll() is None:
            sampled_peak = max(sampled_peak, _sample_peak_rss_bytes(child.pid))
            elapsed = time.monotonic() - started_clock
            if elapsed >= MAX_STAGE_SECONDS:
                timed_out = True
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(min(1.0, MAX_STAGE_SECONDS - elapsed))
        return_code = child.wait()
        sampled_peak = max(sampled_peak, _sample_peak_rss_bytes(child.pid))
        try:
            report = os.read(read_fd, 128).decode("ascii").strip()
            exact_peak = int(report) if report else 0
        except (OSError, ValueError):
            exact_peak = 0
        peak_rss_bytes = exact_peak or sampled_peak
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)

    ended_at = _timestamp()
    status = 124 if timed_out else (return_code if return_code >= 0 else 128 + abs(return_code))
    stage_record.update(
        {
            "ended_at": ended_at,
            "wall_time_seconds": time.monotonic() - started_clock,
            "exit_code": status,
            "timed_out": timed_out,
            "peak_rss_bytes": peak_rss_bytes,
        }
    )
    sidecar_path = _write_stage_sidecar(stage_record, repo=REPO)
    print(
        f"{ended_at} INFO analyze_core family={family_label} stage={stage} "
        f"event=end started_at={started_at} peak_rss_bytes={peak_rss_bytes} "
        f"timed_out={str(timed_out).lower()} exit_code={status} "
        f"frozen_analysis_commit={analysis_provenance['frozen_analysis_commit']} "
        f"executed_commit={analysis_provenance['executed_commit']} "
        f"code_tree_sha256={analysis_provenance['code_tree_sha256']} thread_caps={thread_caps} "
        f"sidecar={sidecar_path.relative_to(REPO)}",
        flush=True,
    )
    return status


def _preflight(
    config: str,
    *,
    gpu: bool,
    analysis_provenance: Mapping[str, str] | None = None,
) -> int:
    """Run the shared job preflight without opening primary results or weights."""
    cfg = load_config(config)
    families = tuple(str(value) for value in cfg["pd"]["families"])
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError(f"Invalid primary analysis family list: {families!r}")
    if not ANALYSIS_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing analysis entrypoint: {ANALYSIS_SCRIPT}")
    provenance = dict(analysis_provenance or _analysis_provenance(REPO))
    print(
        "analysis provenance: "
        f"frozen_analysis_commit={provenance['frozen_analysis_commit']} "
        f"executed_commit={provenance['executed_commit']} "
        f"code_tree_sha256={provenance['code_tree_sha256']}",
        flush=True,
    )
    from vilens.preflight import format_checks, preflight_ok, run_preflight

    checks = run_preflight(
        cfg,
        repo_root=REPO,
        step="E2-E5",
        gpu=gpu,
        inputs=[str(cfg["paths"]["concepts_jsonl"])],
        families=list(families),
        records=None,
        output_dirs=[REPO / "results/analysis/primary", REPO / "reports"],
        entrypoint=Path(__file__),
    )
    print(format_checks(checks), flush=True)
    print(
        "preflight summary: primary result files not opened; model weights not loaded",
        flush=True,
    )
    return 0 if preflight_ok(checks) else 1


def _parser() -> argparse.ArgumentParser:
    """Build the job command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("primary",), default="primary")
    parser.add_argument("--confirm-checkpoint-h", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate job arguments, then dispatch requested stages serially."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.preflight_only:
        try:
            return _preflight(args.config, gpu=False)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    if not args.confirm_checkpoint_h:
        parser.error("--source primary requires --confirm-checkpoint-h")
    if not args.stages:
        parser.error("a non-empty --stages list is required for analysis")

    cfg = load_config(args.config)
    families = tuple(str(value) for value in cfg["pd"]["families"])
    if not families or any(family not in FAMILIES for family in families):
        parser.error(f"invalid primary analysis family list: {families!r}")
    _set_job_thread_caps()
    try:
        analysis_provenance = _analysis_provenance(REPO)
    except RuntimeError as exc:
        parser.error(str(exc))
    preflight_status = _preflight(args.config, gpu=True, analysis_provenance=analysis_provenance)
    if preflight_status != 0:
        return preflight_status
    for family, stage in _build_work_items(args.stages, families):
        status = _run_stage(
            args.config,
            family,
            stage,
            analysis_provenance=analysis_provenance,
        )
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
