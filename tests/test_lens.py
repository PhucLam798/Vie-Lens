"""Tests for precision modes and bounded logit-lens projections."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as functional

from vilens.hooks import record
from vilens.lens import LensHead
from vilens.models import wrap_model


def test_lens_precision_modes_and_fp32_logprob_error(tiny_bundle: dict[str, object]) -> None:
    """Verify exact model-dtype paths and report fp32 projection differences."""
    bundle = tiny_bundle
    family = bundle["family"]
    model = bundle["model"]
    loaded = bundle["loaded"]
    cfg = bundle["cfg"]
    assert isinstance(model, torch.nn.Module)
    ids = torch.tensor([[4, 8, 10, 11, 19, 23, 27], [3, 5, 9, 15, 17, 21, 24]], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1], dtype=torch.long).expand(ids.shape[0], -1)
    fp32_recording = record(loaded, ids, mask, positions=positions)
    model_lens_fp32 = LensHead.from_loaded(loaded, "model")
    fp32_lens = LensHead.from_loaded(loaded, "fp32")
    full_h = fp32_recording.resid[-1].reshape(-1, loaded.d_model)
    full_model_logits = fp32_recording.logits.reshape(-1, loaded.vocab_size)
    fp32_exact = torch.equal(model_lens_fp32.logits(full_h), full_model_logits)
    assert fp32_exact

    fp32_lp = functional.log_softmax(fp32_lens.logits(full_h).float(), dim=-1)
    model_fp32_lp = functional.log_softmax(full_model_logits.float(), dim=-1)
    fp32_delta = float((fp32_lp - model_fp32_lp).abs().max())
    print(f"T04A_METRIC model_exact_fp32 {family} {fp32_exact}")
    print(f"T04A_METRIC fp32_lens_delta_fp32_model {family} {fp32_delta:.9g}")
    assert fp32_delta < 1e-5

    bf16_model = copy.deepcopy(model).to(dtype=torch.bfloat16)
    bf16_loaded = wrap_model(bf16_model, None, family, cfg)
    bf16_recording = record(bf16_loaded, ids, mask, positions=positions)
    bf16_h = bf16_recording.resid[-1].reshape(-1, bf16_loaded.d_model)
    bf16_logits = bf16_recording.logits.reshape(-1, bf16_loaded.vocab_size)
    model_lens_bf16 = LensHead.from_loaded(bf16_loaded, "model")
    bf16_exact = torch.equal(model_lens_bf16.logits(bf16_h), bf16_logits)
    assert bf16_exact
    fp32_lens_bf16 = LensHead.from_loaded(bf16_loaded, "fp32")
    fp32_bf16_lp = functional.log_softmax(fp32_lens_bf16.logits(bf16_h).float(), dim=-1)
    model_bf16_lp = functional.log_softmax(bf16_logits.float(), dim=-1)
    bf16_delta = float((fp32_bf16_lp - model_bf16_lp).abs().max())
    print(f"T04A_METRIC model_exact_bf16 {family} {bf16_exact}")
    print(f"T04A_METRIC fp32_lens_delta_bf16_model {family} {bf16_delta:.9g}")


def test_chunked_logprobs_and_softcap(tiny_bundle: dict[str, object]) -> None:
    """Keep requested-id logprob results stable across projection chunk sizes."""
    bundle = tiny_bundle
    model = bundle["model"]
    family = bundle["family"]
    cfg = bundle["cfg"]
    loaded = bundle["loaded"]
    assert isinstance(model, torch.nn.Module)
    lens = LensHead.from_loaded(loaded, "fp32")
    hidden = torch.randn(8, loaded.d_model)
    ids = torch.randint(0, loaded.vocab_size, (8, 4), dtype=torch.long)
    lens.position_chunk = 100
    unchunked = lens.logprobs_for(hidden, ids)
    lens.position_chunk = 3
    chunked = lens.logprobs_for(hidden, ids)
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=1e-6)

    text_config = model.config.get_text_config()
    text_config.final_logit_softcapping = 7.0
    capped_loaded = wrap_model(model, None, family, cfg)
    capped_lens = LensHead.from_loaded(capped_loaded, "fp32")
    projected = functional.linear(
        capped_lens.norm(hidden.float()), capped_lens.weight, capped_lens.bias
    )
    expected = torch.tanh(projected / 7.0) * 7.0
    torch.testing.assert_close(capped_lens.logits(hidden), expected, rtol=0, atol=0)
