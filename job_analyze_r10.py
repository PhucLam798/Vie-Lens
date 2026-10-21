#!/datastore/luannt/.cache/sb/bin/python
"""Root CPU entrypoint for the descriptive OPS-46 R10 analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

THREAD_CAPS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
    "ARROW_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run R10 preflight or analysis from the repository root."""
    env_guard.guard(offline=True)
    os.environ.update(THREAD_CAPS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    from vilens.config import load_config
    from vilens.locks import ensure_adjudication_committed
    from vilens.provenance import git_state
    from scripts import analyze_r10

    cfg = load_config(REPO / args.config)
    ensure_adjudication_committed(REPO)
    commit, dirty = git_state(REPO)
    if dirty:
        raise RuntimeError(f"R10 analysis code tree is dirty at {commit}")
    table, okay = analyze_r10.preflight(cfg, repo=REPO)
    print(table)
    if args.preflight_only:
        return 0 if okay else 2
    if not okay:
        raise SystemExit("R10 P-L path preflight failed; refusing analysis")
    analyze_r10.analyze(cfg, repo=REPO)
    return 0


if __name__ == "__main__":
    env_guard.guard(offline=True)
    raise SystemExit(main())
