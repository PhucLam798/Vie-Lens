"""Tests for the single weight-loader path using only a synthetic local cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from conftest import make_tiny_model
from vilens.config import ConfigError, load_config
from vilens.models import ModelSpec, SnapshotError, _load_from_spec, load, wrap_model
from vilens.passes import _unit_provenance
from vilens.provenance import capture_run_git


ROOT = Path(__file__).resolve().parents[1]
TINY_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _test_tokenizer() -> PreTrainedTokenizerFast:
    """Build a small fast tokenizer that can be serialized into a fake HF cache."""
    backend = Tokenizer(models.BPE(
        vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
        merges=[],
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


def _make_fake_llama_cache(cache_root: Path) -> tuple[Path, Path]:
    """Save a tiny Llama and tokenizer into a real sha256-linked Hub cache layout."""
    repo_dir = cache_root / "models--org--tiny"
    blobs = repo_dir / "blobs"
    snapshot = repo_dir / "snapshots" / TINY_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    source = cache_root / "serialized-source"
    source.mkdir()
    model = make_tiny_model("llama")
    model.save_pretrained(source, safe_serialization=True)
    _test_tokenizer().save_pretrained(source)
    for path in source.iterdir():
        if not path.is_file():
            continue
        contents = path.read_bytes()
        blob_name = hashlib.sha256(contents).hexdigest()
        (blobs / blob_name).write_bytes(contents)
        (snapshot / path.name).symlink_to(Path("..") / ".." / "blobs" / blob_name)
    refs = repo_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text(TINY_REVISION, encoding="utf-8")
    return repo_dir, snapshot


def test_load_refuses_null_study_set() -> None:
    """Respect the human-only D10 lock before touching any checkpoint."""
    cfg = load_config(ROOT / "configs/experiments.yaml")
    cfg["models"]["study_set"] = None
    with pytest.raises(ConfigError, match="D10 undecided: models.study_set is null"):
        load(cfg, "llama", device="cpu")


def test_load_from_spec_uses_local_cache_and_wraps_modules(tmp_path: Path) -> None:
    """Load and freeze a tiny bf16 checkpoint from a fake offline cache."""
    _make_fake_llama_cache(tmp_path)
    cfg = load_config(ROOT / "configs/experiments.yaml")
    cfg["paths"]["model_cache"] = str(tmp_path)
    spec = ModelSpec("registered", "llama", "org/tiny", TINY_REVISION)
    loaded = _load_from_spec(spec, cfg, device="cpu")
    assert loaded.spec == spec
    assert loaded.family == "llama"
    assert loaded.model.training is False
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in loaded.model.parameters())
    assert loaded.decoder is loaded.model.model
    assert loaded.final_norm is loaded.decoder.norm
    assert loaded.embed_tokens is loaded.decoder.embed_tokens
    assert loaded.lm_head is loaded.model.lm_head


def test_dtype_override_loads_fp32_and_records_dtype(tmp_path: Path) -> None:
    """Load a synthetic local checkpoint in fp32 and record that dtype in provenance."""
    _make_fake_llama_cache(tmp_path)
    cfg = load_config(ROOT / "configs/experiments.yaml")
    cfg["paths"]["model_cache"] = str(tmp_path)
    spec = ModelSpec("registered", "llama", "org/tiny", TINY_REVISION)
    loaded = _load_from_spec(spec, cfg, device="cpu", dtype_override=torch.float32)
    assert all(parameter.dtype == torch.float32 for parameter in loaded.model.parameters())
    provenance = _unit_provenance(
        loaded,
        cfg,
        ROOT,
        None,
        {"family": "llama", "pass": "e1", "check": "precision"},
        1,
        1,
        capture_run_git(ROOT, cfg),
    )
    assert provenance["dtype"] == "torch.float32"
    assert provenance["model_loads"][0]["dtype_override"] == "float32"


def test_load_from_spec_rejects_architecture_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a cache config whose declared architecture differs from the configured family."""
    _make_fake_llama_cache(tmp_path)
    cfg = load_config(ROOT / "configs/experiments.yaml")
    cfg["paths"]["model_cache"] = str(tmp_path)
    spec = ModelSpec("registered", "llama", "org/tiny", TINY_REVISION)

    class WrongArchitectureConfig:
        architectures = ["Qwen3ForCausalLM"]

    monkeypatch.setattr("vilens.models.load_hf_config", lambda *_args, **_kwargs: WrongArchitectureConfig())
    with pytest.raises(SnapshotError, match="Architecture mismatch"):
        _load_from_spec(spec, cfg, device="cpu")


def test_wrap_gemma_resolves_text_tower_not_vision_tower() -> None:
    """Resolve decoder modules from Gemma's language model submodule explicitly."""
    model = make_tiny_model("gemma")
    cfg = load_config(ROOT / "configs/experiments.yaml")
    loaded = wrap_model(model, None, "gemma", cfg)
    assert loaded.decoder is model.model.language_model
    assert loaded.decoder is not model.model.vision_tower
    assert len(loaded.layers) == model.config.get_text_config().num_hidden_layers
