"""Streaming SHA-256 verification for pinned data inputs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

_LINE = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")


def sha256_file(path: str | Path) -> str:
    """Calculate a file's SHA-256 digest using bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    manifest_path: str | Path,
    repo_root: str | Path,
    only: Iterable[str] | None = None,
) -> list[tuple[str, bool]]:
    """Verify selected manifest entries and raise after collecting all failures."""
    root = Path(repo_root)
    selected = None if only is None else set(only)
    results: list[tuple[str, bool]] = []
    failures: list[str] = []
    seen: set[str] = set()
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            match = _LINE.fullmatch(line)
            if match is None:
                failures.append(f"line {line_no}: malformed manifest entry")
                continue
            expected, relative = match.groups()
            if relative in seen:
                failures.append(f"line {line_no}: duplicate path {relative}")
            seen.add(relative)
            if selected is not None and relative not in selected:
                continue
            path = root / relative
            if not path.is_file():
                results.append((relative, False))
                failures.append(f"{relative}: missing file")
                continue
            actual = sha256_file(path)
            ok = actual.lower() == expected.lower()
            results.append((relative, ok))
            if not ok:
                failures.append(f"{relative}: expected {expected}, got {actual}")
    if selected is not None:
        absent = sorted(selected - seen)
        failures.extend(f"{relative}: not listed in manifest" for relative in absent)
    if failures:
        raise ValueError("Manifest verification failed:\n" + "\n".join(failures))
    return results
