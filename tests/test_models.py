"""Tests for pinned local snapshot and tokenizer loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vilens.config import load_config
from vilens.models import (
    ModelSpec,
    SnapshotError,
    candidate_specs,
    load_tokenizer,
    resolve_snapshot,
    snapshot_problems,
)

ROOT = Path(__file__).resolve().parents[1]
TINY_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _byte_level_fast_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal fast byte-level tokenizer for the fake cache."""
    backend = Tokenizer(models.BPE(
        vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
        merges=[],
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def make_fake_cache(cache_root: Path) -> tuple[Path, Path]:
    """Create a local Hugging Face cache layout and return repo/snapshot paths."""
    hf_id = "org/tiny"
    repo_dir = cache_root / "models--org--tiny"
    blobs = repo_dir / "blobs"
    snapshot = repo_dir / "snapshots" / TINY_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    tokenizer_dir = cache_root / "tokenizer-source"
    tokenizer_dir.mkdir()
    tokenizer = _byte_level_fast_tokenizer()
    tokenizer.save_pretrained(tokenizer_dir)
    file_bytes = {path.name: path.read_bytes() for path in tokenizer_dir.iterdir() if path.is_file()}
    file_bytes["config.json"] = json.dumps({"model_type": "gpt2"}).encode("utf-8")
    weight_path = tokenizer_dir / "model.safetensors"
    save_file({"w": np.zeros(4, dtype=np.float32)}, str(weight_path))
    file_bytes["model.safetensors"] = weight_path.read_bytes()
    for name, contents in file_bytes.items():
        blob_name = hashlib.sha256(contents).hexdigest()
        (blobs / blob_name).write_bytes(contents)
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob_name)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(TINY_REVISION, encoding="utf-8")
    return repo_dir, snapshot


def _make_indexed_shard(
    repo_dir: Path,
    snapshot: Path,
    *,
    indexed_tensor: str = "w",
    total_size: int | None = None,
) -> tuple[Path, int]:
    """Replace the single-file weight with a valid tiny indexed safetensors shard."""
    (snapshot / "model.safetensors").unlink()
    source = repo_dir / "shard-source.safetensors"
    save_file({"w": np.zeros(4, dtype=np.float32)}, str(source))
    contents = source.read_bytes()
    blob_name = hashlib.sha256(contents).hexdigest()
    blob_path = repo_dir / "blobs" / blob_name
    blob_path.write_bytes(contents)
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).symlink_to(Path("..") / ".." / "blobs" / blob_name)
    index: dict[str, Any] = {"weight_map": {indexed_tensor: shard_name}}
    if total_size is not None:
        index["metadata"] = {"total_size": total_size}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return blob_path, len(contents)


def _small_config() -> dict[str, Any]:
    """Load the checked-in config for the fixed candidate ordering test."""
    return load_config(ROOT / "configs/experiments.yaml")


def test_candidate_specs_order_and_labels() -> None:
    """Return both model groups in stable set and family order."""
    specs = candidate_specs(_small_config())
    assert [spec.label for spec in specs] == [
        "registered/gemma", "registered/qwen", "registered/llama",
        "base/gemma", "base/qwen", "base/llama",
    ]
    assert all(spec.set_name in {"registered", "base"} for spec in specs)


def test_resolve_snapshot_from_fake_cache(tmp_path: Path) -> None:
    """Resolve a local snapshot using its immutable revision name."""
    _, snapshot = make_fake_cache(tmp_path)
    assert resolve_snapshot(ModelSpec("registered", "llama", "org/tiny", TINY_REVISION), tmp_path) == snapshot


def test_wrong_revision_names_repo_and_revision(tmp_path: Path) -> None:
    """Include exact repo and revision details when local resolution fails."""
    spec = ModelSpec("registered", "llama", "org/tiny", "f" * 40)
    with pytest.raises(SnapshotError) as error:
        resolve_snapshot(spec, tmp_path)
    assert "org/tiny" in str(error.value)
    assert "f" * 40 in str(error.value)
    assert str(tmp_path) in str(error.value)


def test_snapshot_problems_empty_for_complete_fake_cache(tmp_path: Path) -> None:
    """Accept a complete cache snapshot after reading only its safetensors header."""
    _, snapshot = make_fake_cache(tmp_path)
    assert snapshot_problems(snapshot) == []


def test_snapshot_problems_missing_tokenizer_json(tmp_path: Path) -> None:
    """Report a missing required tokenizer file."""
    _, snapshot = make_fake_cache(tmp_path)
    (snapshot / "tokenizer.json").unlink()
    assert any("tokenizer.json" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_zero_size_file(tmp_path: Path) -> None:
    """Report an empty required file."""
    _, snapshot = make_fake_cache(tmp_path)
    (snapshot.parents[1] / "blobs" / "zero.tmp").write_bytes(b"")
    (snapshot / "tokenizer_config.json").unlink()
    (snapshot / "tokenizer_config.json").symlink_to(Path("..") / ".." / "blobs" / "zero.tmp")
    assert any("zero-size file: tokenizer_config.json" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_missing_index_shard(tmp_path: Path) -> None:
    """Report a shard referenced by an index but absent from the snapshot."""
    _, snapshot = make_fake_cache(tmp_path)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard-00001.safetensors"}}),
        encoding="utf-8",
    )
    assert any("shard-00001.safetensors" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_ignores_inconsistent_index_total_size(tmp_path: Path) -> None:
    """Treat upstream total_size disagreement as reporting information, not a gate."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    _make_indexed_shard(repo_dir, snapshot, total_size=1000)
    assert snapshot_problems(snapshot) == []


def test_snapshot_problems_truncated_shard(tmp_path: Path) -> None:
    """Report a shard truncated after its safetensors header was written."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    blob_path, _ = _make_indexed_shard(repo_dir, snapshot)
    blob_path.write_bytes(blob_path.read_bytes()[:-1])
    assert any("file size mismatch" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_index_tensor_absent_from_shard_header(tmp_path: Path) -> None:
    """Report index assignments missing from the corresponding shard header."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    _make_indexed_shard(repo_dir, snapshot, indexed_tensor="missing_tensor")
    assert any("tensors absent" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_weight_blob_name_not_sha256(tmp_path: Path) -> None:
    """Reject a weight blob whose basename is not a 64-character lowercase hash."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    model_path = snapshot / "model.safetensors"
    model_path.unlink()
    source = repo_dir / "invalid-name-source.safetensors"
    save_file({"w": np.zeros(4, dtype=np.float32)}, str(source))
    (repo_dir / "blobs" / "not-a-sha256").write_bytes(source.read_bytes())
    model_path.symlink_to(Path("..") / ".." / "blobs" / "not-a-sha256")
    assert any("not a 64-hex sha256" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_zero_header_length(tmp_path: Path) -> None:
    """Reject a safetensors file that announces an empty JSON header."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    model_path = snapshot / "model.safetensors"
    model_path.unlink()
    empty_header = b"\x00" * 8
    blob_name = hashlib.sha256(empty_header).hexdigest()
    (repo_dir / "blobs" / blob_name).write_bytes(empty_header)
    model_path.symlink_to(Path("..") / ".." / "blobs" / blob_name)
    assert any("invalid safetensors header length 0" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_dangling_symlink(tmp_path: Path) -> None:
    """Reject any dangling path inside the snapshot."""
    _, snapshot = make_fake_cache(tmp_path)
    (snapshot / "dangling").symlink_to("missing-target")
    assert any("dangling or inaccessible snapshot entry" in problem for problem in snapshot_problems(snapshot))


def test_snapshot_problems_incomplete_blob(tmp_path: Path) -> None:
    """Reject incomplete transfer marker files in the repository blob store."""
    repo_dir, snapshot = make_fake_cache(tmp_path)
    (repo_dir / "blobs" / "unfinished.incomplete").write_bytes(b"x")
    assert any(".incomplete" in problem for problem in snapshot_problems(snapshot))


def test_load_tokenizer_from_fake_cache(tmp_path: Path) -> None:
    """Load the fake local cache as a fast AutoTokenizer without network access."""
    make_fake_cache(tmp_path)
    tokenizer = load_tokenizer(ModelSpec("registered", "llama", "org/tiny", TINY_REVISION), tmp_path)
    assert tokenizer.is_fast is True
