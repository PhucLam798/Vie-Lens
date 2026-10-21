"""Tests for selected residual and component capture on tiny decoder models."""

from __future__ import annotations

import pytest
import torch

from vilens.hooks import record


def _assert_no_hooks(model: torch.nn.Module) -> None:
    """Assert that recording left no forward hooks on any module."""
    assert all(not module._forward_hooks for module in model.modules())
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_residual_identity_selection_and_cleanup(tiny_bundle: dict[str, object]) -> None:
    """Capture embedding, block outputs and post-norm residual components."""
    bundle = tiny_bundle
    loaded = bundle["loaded"]
    model = bundle["model"]
    family = bundle["family"]
    assert not isinstance(loaded, dict)
    assert isinstance(model, torch.nn.Module)
    ids = torch.tensor([[1, 5, 8, 9, 11, 17, 20], [2, 3, 6, 7, 12, 14, 18]], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.tensor([[0, 3, 6], [1, 4, 5]], dtype=torch.long)
    recording = record(loaded, ids, mask, positions=positions, components=True)

    embedding = loaded.embed_tokens(ids).gather(
        1, positions.unsqueeze(-1).expand(-1, -1, loaded.d_model)
    )
    torch.testing.assert_close(recording.resid[0], embedding, rtol=0, atol=0)
    assert recording.resid.shape == (loaded.n_layers + 1, 2, 3, loaded.d_model)
    assert recording.attn is not None and recording.mlp is not None
    assert recording.attn.shape == recording.mlp.shape == (loaded.n_layers, 2, 3, loaded.d_model)

    max_relative_error = 0.0
    for layer_index in range(loaded.n_layers):
        residual = recording.resid[layer_index + 1]
        expected = recording.resid[layer_index] + recording.attn[layer_index] + recording.mlp[layer_index]
        relative_error = torch.linalg.vector_norm(residual - expected) / torch.linalg.vector_norm(residual)
        max_relative_error = max(max_relative_error, float(relative_error))
    print(f"T04A_METRIC residual_identity {family} {max_relative_error:.9g}")
    assert max_relative_error < 1e-4

    captured: dict[str, torch.Tensor] = {}
    last_layer = loaded.layers[-1]

    def capture_last(_module: object, _args: object, output: object) -> None:
        captured["out"] = output[0] if isinstance(output, tuple) else output

    handle = last_layer.register_forward_hook(capture_last)
    try:
        with torch.inference_mode():
            model_output = model(input_ids=ids, attention_mask=mask)
    finally:
        handle.remove()
    del model_output
    last_gathered = captured["out"].gather(
        1, positions.unsqueeze(-1).expand(-1, -1, loaded.d_model)
    )
    torch.testing.assert_close(recording.resid[-1], last_gathered, rtol=0, atol=0)
    _assert_no_hooks(model)
    if family == "gemma":
        assert loaded.decoder is model.model.language_model
        assert len(loaded.layers) == model.config.get_text_config().num_hidden_layers
        assert loaded.decoder is not model.model.vision_tower


def test_record_removes_hooks_after_forward_exception(tiny_bundle: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove registered hooks even when model forward raises."""
    loaded = tiny_bundle["loaded"]
    model = tiny_bundle["model"]
    assert isinstance(model, torch.nn.Module)
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    def fail_forward(**_kwargs: object) -> None:
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        record(loaded, ids, torch.ones_like(ids), positions=torch.tensor([[2]], dtype=torch.long), components=True)
    _assert_no_hooks(model)
