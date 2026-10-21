"""Shared test setup for source-tree imports and fixture paths."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def fixture_path(name: str) -> Path:
    """Return a path under the hand-authored fixture directory."""
    return Path(__file__).resolve().parent / "fixtures" / name


def make_tiny_model(family: str) -> Any:
    """Build one seeded CPU-only tiny causal LM fixture for a model family."""
    import torch
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
        SiglipVisionConfig,
    )

    torch.manual_seed(20260925)
    if family == "llama":
        config = LlamaConfig(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            _attn_implementation="eager",
        )
        return LlamaForCausalLM(config)
    if family == "qwen":
        config = Qwen3Config(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            tie_word_embeddings=True,
            _attn_implementation="eager",
        )
        return Qwen3ForCausalLM(config)
    if family == "gemma":
        text_config = Gemma3TextConfig(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            sliding_window=4,
            _attn_implementation="eager",
        )
        vision_config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        )
        config = Gemma3Config(
            text_config=text_config,
            vision_config=vision_config,
            mm_tokens_per_image=4,
            attn_implementation="eager",
        )
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        return Gemma3ForConditionalGeneration(config)
    raise ValueError(f"Unknown tiny model family: {family}")


@pytest.fixture(params=("llama", "qwen", "gemma"))
def tiny_bundle(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Provide a tiny family model, validated loader wrapper, and repo config."""
    from vilens.config import load_config
    from vilens.models import wrap_model

    family = str(request.param)
    model = make_tiny_model(family)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/experiments.yaml")
    loaded = wrap_model(model, None, family, cfg)
    return {"family": family, "model": model, "loaded": loaded, "cfg": cfg}
