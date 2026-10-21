"""Tiny-model tests for E1 model-dtype lens and residual gates."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from conftest import make_tiny_model
from vilens import hooks
from vilens.lens import LensHead
from vilens.models import wrap_model


def _load_e1_script(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the entrypoint helpers without running its environment guard."""
    from vilens import env_guard

    monkeypatch.setattr(env_guard, "guard", lambda **_kwargs: None)
    path = Path(__file__).resolve().parents[1] / "job_e1_audit.py"
    spec = importlib.util.spec_from_file_location("_test_job_e1_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import job_e1_audit.py for helper tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _e1_metrics(model: Any, family: str, cfg: dict[str, Any]) -> tuple[float, float, int]:
    """Run the prompt-position calculations used by E1-2 and E1-4."""
    loaded = wrap_model(model, None, family, cfg)
    input_ids = torch.tensor([[5, 13, 27, 44, 62, 81]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    positions = torch.tensor([[input_ids.shape[1] - 1]], dtype=torch.long)
    recording = hooks.record(
        loaded, input_ids, attention_mask, positions=positions, components=True
    )

    # The path-validity check is model-dtype: bf16 reproduces the full model
    # projection exactly even though an fp32 projection over one selected
    # position can differ by an ulp from the model's batched projection.
    bf16_model = copy.deepcopy(model).to(dtype=torch.bfloat16)
    bf16_loaded = wrap_model(bf16_model, None, family, cfg)
    bf16_recording = hooks.record(
        bf16_loaded, input_ids, attention_mask, positions=positions, components=False
    )
    model_lens = LensHead.from_loaded(bf16_loaded, "model")
    block_n_bf16 = bf16_recording.resid[-1, :, 0, :]
    model_logits_bf16 = bf16_recording.logits[:, 0, :]
    lens_logits = model_lens.logits(block_n_bf16)
    assert torch.equal(lens_logits, model_logits_bf16)
    lens_logprobs = functional.log_softmax(lens_logits.float(), dim=-1)
    model_logprobs = functional.log_softmax(model_logits_bf16.float(), dim=-1)
    max_delta = float((lens_logprobs - model_logprobs).abs().max().item())

    assert recording.attn is not None and recording.mlp is not None
    previous = recording.resid[:-1, :, 0, :].float()
    following = recording.resid[1:, :, 0, :].float()
    attn = recording.attn[:, :, 0, :].float()
    mlp = recording.mlp[:, :, 0, :].float()
    residual_error = (following - (previous + attn + mlp)).norm(dim=-1) / following.norm(
        dim=-1
    ).clamp_min(torch.finfo(torch.float32).tiny)
    max_residual_error = float(residual_error.max().item())
    return max_delta, max_residual_error, input_ids.shape[0]


def test_e1_lens_and_residual_gates_on_tiny_architectures(
    tiny_bundle: dict[str, Any],
) -> None:
    """The model-dtype lens is exact and residual components reconstruct each block."""
    delta, residual_error, batch_size = _e1_metrics(
        tiny_bundle["model"], tiny_bundle["family"], tiny_bundle["cfg"]
    )
    assert batch_size == 1
    assert delta == 0.0
    assert residual_error < 1e-4


def test_gemma_e1_prompt_forward_does_not_call_vision_tower(
    tiny_bundle: dict[str, Any],
) -> None:
    """The E1 prompt-only forward never invokes Gemma's vision tower."""
    if tiny_bundle["family"] != "gemma":
        return
    calls = 0
    vision = tiny_bundle["model"].model.vision_tower

    def count_call(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        nonlocal calls
        calls += 1

    handle = vision.register_forward_hook(count_call)
    try:
        _e1_metrics(tiny_bundle["model"], "gemma", tiny_bundle["cfg"])
    finally:
        handle.remove()
    assert calls == 0


def test_e1_early_block_ranges_and_check_table_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use inclusive early-block ranges and derive expected rows from the builder."""
    from vilens.config import load_config

    e1 = _load_e1_script(monkeypatch)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/experiments.yaml")
    expected_ranges = {"llama": list(range(1, 5)), "qwen": list(range(1, 6)), "gemma": list(range(1, 6))}
    for family, early_blocks in expected_ranges.items():
        assert e1._expected_early_blocks(cfg, family) == early_blocks
        reference = cfg["reference_facts"]["x0c"]
        dims = list(reference["model_dims"][family])
        assert dims[0] == {"llama": 28, "gemma": 34, "qwen": 36}[family]
        values = np.zeros((dims[0] + 1, 2), dtype=np.float32)
        precision_rows, _ = e1._compare_precision(family, values, values)
        competence = {
            "by_format": {
                format_name: {
                    "records": 1,
                    "counts": {label: 0 for label in e1.COMPETENCE_CLASSES},
                    "rates": {label: 0.0 for label in e1.COMPETENCE_CLASSES},
                    "translation_vi_flag": False,
                }
                for format_name in cfg["e1"]["competence_formats"]
            }
        }
        metrics = {
            "model_max_abs_logprob_delta": 0.0,
            "model_top1_agreement": 1.0,
            "fp32_max_abs_logprob_delta": 0.0,
            "fp32_top1_agreement": 1.0,
            "residual_bf16_model_dtype_max_rel_error": 0.0,
            "residual_bf16_fp32_resummation_reported_only_max_rel_error": 0.005,
            "residual_fp32_weights_max_rel_error": 0.0,
        }
        rows = e1._build_check_rows(
            family,
            dims,
            list(reference["tokenizer_x_colon_ids"][family]),
            True,
            early_blocks,
            metrics,
            True,
            0,
            True,
            precision_rows,
            competence,
            cfg,
        )
        assert e1._expected_check_rows(cfg, family) == len(rows)
