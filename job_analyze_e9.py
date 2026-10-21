#!/datastore/luannt/.cache/sb/bin/python
"""Root CPU job wrapper for the OPS-45 paired E9 analysis."""

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
    """Run E9 input validation or the paired descriptive analysis."""
    env_guard.guard(offline=True)
    os.environ.update(THREAD_CAPS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    from vilens.config import load_config
    from vilens.locks import ensure_adjudication_committed
    from vilens.provenance import git_state
    from scripts import analyze_e9

    cfg = load_config(REPO / args.config)
    ensure_adjudication_committed(REPO)
    commit, dirty = git_state(REPO)
    if dirty:
        raise RuntimeError(f"E9 analysis code tree is dirty at {commit}")
    table, okay, _scopes = analyze_e9.preflight(cfg, repo=REPO)
    print(table)
    if args.preflight_only:
        return 0 if okay else 2
    if not okay:
        raise SystemExit("E9 analysis preflight failed; refusing analysis")
    analyze_e9.analyze(cfg, repo=REPO)
    return 0


if __name__ == "__main__":
    env_guard.guard(offline=True)
    raise SystemExit(main())
