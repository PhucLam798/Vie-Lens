"""Tests for streaming manifest checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vilens.manifest import sha256_file, verify_manifest


def test_manifest_ok_and_mismatch(tmp_path: Path) -> None:
    """Accept a correct temporary entry, then report a changed file."""
    payload = tmp_path / "payload.txt"
    payload.write_text("pinned\n", encoding="utf-8")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{expected}  payload.txt\n", encoding="utf-8")
    assert sha256_file(payload) == expected
    assert verify_manifest(manifest, tmp_path) == [("payload.txt", True)]
    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload.txt: expected"):
        verify_manifest(manifest, tmp_path)
