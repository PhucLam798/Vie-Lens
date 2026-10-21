"""Shared argument parsing, UTC logging, and stage timing helpers."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    """Return a compact UTC timestamp suitable for log filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create a common entrypoint parser with config and preflight-only options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def setup_logging(name: str, logs_dir: str | Path) -> logging.Logger:
    """Configure a named logger for stdout and a timestamped file under logs_dir."""
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(directory / f"{name}_{_utc_stamp()}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


class StageTimer:
    """Measure a named block, log its wall time, and retain times for a summary."""

    def __init__(
        self,
        name: str,
        logger: logging.Logger | None = None,
        timings: dict[str, float] | None = None,
    ) -> None:
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.timings = timings if timings is not None else {}
        self.elapsed: float | None = None
        self._started: float | None = None

    def __enter__(self) -> StageTimer:
        """Start the wall-time measurement."""
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Record elapsed time and log it even when the stage raises."""
        if self._started is None:
            raise RuntimeError("StageTimer was not entered")
        self.elapsed = time.perf_counter() - self._started
        self.timings[self.name] = self.elapsed
        self.logger.info("stage %s wall_time_seconds=%.3f", self.name, self.elapsed)
        return False
