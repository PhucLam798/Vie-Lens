"""Synthetic unit tests for OPS-37 E7 attribution helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from vilens import attribute


class _TinyConfig:
    """Minimal text config used by the production E7 capture path."""

    num_attention_heads = 2
    head_dim = 4
    hidden_size = 8
    num_hidden_layers = 2
    vocab_size = 32
    layer_types = ["full_attention", "full_attention"]

    def get_text_config(self) -> _TinyConfig:
        return self


class _TinyTokenizer:
    """Synthetic prompt tokenizer for exercising the hook capture path."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 1

    def __call__(self, prompt: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        values = [2 + (ord(char) % 29) for char in prompt]
        return {"input_ids": ([1] + values) if add_special_tokens else values}


class _TinyRMSNorm(nn.Module):
    """Small Gemma-like RMS normalization that preserves the input dtype."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        source_dtype = values.dtype
        values32 = values.float()
        normalized = values32 * torch.rsqrt(values32.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(source_dtype)


class _TinyAttention(nn.Module):
    """Identity-projection attention exposing the o_proj hooks E7 captures."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(width))
            self.o_proj.weight.copy_(torch.eye(width))

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> tuple[torch.Tensor, None]:
        return self.o_proj(self.q_proj(hidden_states)), None


class _TinyMLP(nn.Module):
    """Small BF16 feed-forward update for both decoder-layer shapes."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width) * 0.125)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden_states))


class _TinyLayer(nn.Module):
    """Llama-like or Gemma-like decoder layer with the production hook names."""

    def __init__(self, family: str, width: int) -> None:
        super().__init__()
        self.family = family
        self.self_attn = _TinyAttention(width)
        self.input_layernorm = nn.LayerNorm(width)
        self.post_attention_layernorm = _TinyRMSNorm(width) if family == "gemma" else nn.LayerNorm(width)
        self.pre_feedforward_layernorm = _TinyRMSNorm(width)
        self.mlp = _TinyMLP(width)
        self.post_feedforward_layernorm = _TinyRMSNorm(width)

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention, _weights = self.self_attn(normalized)
        if self.family == "gemma":
            attention = self.post_attention_layernorm(attention)
        hidden_states = residual + attention

        residual = hidden_states
        if self.family == "gemma":
            mlp = self.mlp(self.pre_feedforward_layernorm(hidden_states))
            mlp = self.post_feedforward_layernorm(mlp)
        else:
            mlp = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + mlp


class _TinyModel(nn.Module):
    """Minimal causal model wrapper accepted by job_e7_attribute._capture_batch."""

    def __init__(self, family: str, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = _TinyConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [_TinyLayer(family, self.config.hidden_size) for _ in range(self.config.num_hidden_layers)]
        )

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def _tiny_loaded_model(family: str, *, seed: int) -> SimpleNamespace:
    """Build a CPU BF16 model bundle with the decoder paths used by production capture."""
    model = _TinyModel(family, seed=seed).to(dtype=torch.bfloat16).eval()
    return SimpleNamespace(
        model=model,
        tokenizer=_TinyTokenizer(),
        family=family,
        spec=None,
        layers=list(model.layers),
        n_layers=len(model.layers),
    )


def _e7_check_config() -> dict[str, dict[str, float]]:
    """Return the unchanged 10(a)/(c) thresholds plus the OPS-38 10(b) threshold."""
    return {
        "e7": {
            "check_head_median_relative": 1.0e-2,
            "check_head_p99_relative": 5.0e-2,
            "check_residual_model_order_max": 1.0e-4,
            "check_pl_median_relative": 1.0e-2,
        }
    }


def test_llama_like_head_contributions_and_residual_decomposition() -> None:
    """A tiny Llama-like linear attention block decomposes exactly in fp32."""
    generator = torch.Generator().manual_seed(20260925)
    batch, hidden, heads, head_dim = 3, 8, 2, 4
    z = torch.randn(batch, heads * head_dim, generator=generator)
    o_weight = torch.randn(hidden, heads * head_dim, generator=generator)
    mlp_update = torch.randn(batch, hidden, generator=generator)
    attention_update = torch.nn.functional.linear(z, o_weight)
    head_updates = attribute.attention_head_contributions(
        z, o_weight, n_heads=heads, head_dim=head_dim
    )
    torch.testing.assert_close(head_updates.sum(dim=-2), attention_update)
    residual_in = torch.randn(batch, hidden, generator=generator)
    residual_out = residual_in + attention_update + mlp_update
    torch.testing.assert_close(residual_out - residual_in, attention_update + mlp_update)


def test_gemma_like_post_norm_head_contributions_sum_to_component() -> None:
    """Gemma post-attention RMS gain distributes the exact post-norm update."""
    generator = torch.Generator().manual_seed(29)
    batch, hidden, heads, head_dim = 4, 8, 2, 4
    z = torch.randn(batch, heads * head_dim, generator=generator)
    o_weight = torch.randn(hidden, heads * head_dim, generator=generator)
    raw_attention = torch.nn.functional.linear(z, o_weight)
    norm_weight = torch.randn(hidden, generator=generator) * 0.1
    eps = 1e-6
    rms = torch.sqrt(raw_attention.square().mean(dim=-1, keepdim=True) + eps)
    post_norm_attention = raw_attention / rms * (1.0 + norm_weight)
    head_updates = attribute.attention_head_contributions(
        z,
        o_weight,
        n_heads=heads,
        head_dim=head_dim,
        total_attention=raw_attention,
        gemma_post_norm_weight=norm_weight,
        gemma_post_norm_eps=eps,
    )
    torch.testing.assert_close(head_updates.sum(dim=-2), post_norm_attention)
    mlp_update = torch.randn(batch, hidden, generator=generator)
    residual_in = torch.randn(batch, hidden, generator=generator)
    residual_out = residual_in + post_norm_attention + mlp_update
    torch.testing.assert_close(residual_out - residual_in, post_norm_attention + mlp_update)


def test_resolve_bands_crossing_and_ops43_rule_ii_primary() -> None:
    """Rule (i) stays landmark based and rule (ii) takes the eligible slope band."""
    n_layers = 10
    gate = {layer: True for layer in range(1, n_layers + 1)}
    curve = np.arange(n_layers + 1, dtype=np.float64)
    result = attribute.resolve_bands(
        n_layers=n_layers, l_cross=7, l_peak=6, mean_delta_vi=curve,
        gate_pass=gate, early_blocks=[1, 2],
    )
    assert result["primary"]["blocks"] == [5, 6, 7]
    assert result["slope_sensitivity"]["blocks"] == [8, 9, 10]
    assert result["ops37_as_written"]["blocks"] == [8, 9, 10]
    peak_missing = attribute.resolve_bands(
        n_layers=n_layers, l_cross=7, l_peak=None, mean_delta_vi=curve,
        gate_pass=gate, early_blocks=[1, 2],
    )
    assert peak_missing["primary"]["blocks"] == [5, 6, 7]
    peak_after = attribute.resolve_bands(
        n_layers=n_layers, l_cross=7, l_peak=9, mean_delta_vi=curve,
        gate_pass=gate, early_blocks=[1, 2],
    )
    assert peak_after["primary"]["blocks"] == [5, 6, 7]


def test_ops43_eligible_blocks_exclude_early_and_gate_failures() -> None:
    """Eligible slope bands exclude early blocks and A6 gate failures."""
    n_layers = 12
    early = [1, 2, 3]
    gate = {layer: layer not in {4, 6} for layer in range(1, n_layers + 1)}
    slopes = np.arange(1, n_layers + 1, dtype=np.float64)
    curve = np.concatenate(([0.0], np.cumsum(slopes)))
    result = attribute.resolve_bands(
        n_layers=n_layers, l_cross=None, l_peak=None,
        mean_delta_vi=curve, gate_pass=gate, early_blocks=early,
    )
    eligible = [5, 7, 8, 9, 10, 11, 12]
    assert result["rule"] == "ii"
    assert result["eligible_blocks"] == eligible
    assert result["primary"]["blocks"] == [10, 11, 12]
    assert result["primary"]["blocks"] == result["slope_sensitivity"]["blocks"]
    assert all(block > max(early) and gate[block] for block in result["primary"]["blocks"])
    assert result["primary"]["eligible_count"] == len(eligible)
    assert result["primary"]["requested_size"] == 3
    assert not result["primary"]["used_all_eligible"]


def test_ops43_quarter_size_for_each_model_depth() -> None:
    """Enough eligible blocks yield ceil(N/4): 9, 9, and 7 for study models."""
    for n_layers, early_end, expected_size in ((34, 5, 9), (36, 5, 9), (28, 4, 7)):
        gate = {layer: True for layer in range(1, n_layers + 1)}
        result = attribute.resolve_bands(
            n_layers=n_layers, l_cross=None, l_peak=None,
            mean_delta_vi=np.zeros(n_layers + 1), gate_pass=gate,
            early_blocks=list(range(1, early_end + 1)),
        )
        assert result["primary"]["requested_size"] == expected_size
        assert len(result["primary"]["blocks"]) == expected_size
        assert result["primary"]["eligible_count"] > expected_size


def test_ops43_uses_all_eligible_when_fewer_than_quarter() -> None:
    """A short eligible pool is used in full and marked in the band metadata."""
    n_layers = 12
    gate = {layer: True for layer in range(1, n_layers + 1)}
    curve = np.zeros(n_layers + 1, dtype=np.float64)
    result = attribute.resolve_bands(
        n_layers=n_layers, l_cross=None, l_peak=None,
        mean_delta_vi=curve, gate_pass=gate, early_blocks=list(range(1, 11)),
    )
    assert result["primary"]["blocks"] == [11, 12]
    assert result["primary"]["eligible_count"] == 2
    assert result["primary"]["requested_size"] == 3
    assert result["primary"]["used_all_eligible"]


def test_ops43_ties_choose_deeper_eligible_blocks() -> None:
    """Equal eligible slopes are ordered toward deeper blocks."""
    n_layers = 8
    gate = {layer: True for layer in range(1, n_layers + 1)}
    result = attribute.resolve_bands(
        n_layers=n_layers, l_cross=None, l_peak=None,
        mean_delta_vi=np.zeros(n_layers + 1), gate_pass=gate, early_blocks=[1],
    )
    assert result["primary"]["blocks"] == [7, 8]


def test_ops37_as_written_reproduces_t22b_sensitivity_rows() -> None:
    """The all-block sensitivity preserves the six T22b selected-block rows."""
    expected = {
        ("gemma", "repetition"): [1, 2, 3, 22, 24, 25, 26, 27, 30],
        ("gemma", "translation"): [1, 23, 24, 25, 26, 27, 28, 29, 30],
        ("qwen", "repetition"): [23, 24, 25, 27, 29, 30, 32, 33, 34],
        ("qwen", "translation"): [22, 24, 25, 27, 29, 31, 32, 33, 34],
        ("llama", "repetition"): [3, 15, 16, 21, 22, 25, 27],
        ("llama", "translation"): [15, 16, 20, 23, 24, 25, 26],
    }
    gate_failures = {
        ("gemma", "repetition"): [24],
        ("gemma", "translation"): [23, 24, 25, 26, 29, 30],
        ("qwen", "repetition"): [],
        ("qwen", "translation"): [],
        ("llama", "repetition"): [],
        ("llama", "translation"): [],
    }
    layers = {"gemma": 34, "qwen": 36, "llama": 28}
    early = {"gemma": [1, 2, 3, 4, 5], "qwen": [1, 2, 3, 4, 5], "llama": [1, 2, 3, 4]}
    for cell, selected_blocks in expected.items():
        n_layers = layers[cell[0]]
        slopes = np.arange(n_layers, dtype=np.float64)
        for rank, block in enumerate(selected_blocks):
            slopes[block - 1] = 1000.0 + len(selected_blocks) - rank
        curve = np.concatenate(([0.0], np.cumsum(slopes)))
        gate = {
            layer: layer not in set(gate_failures[cell])
            for layer in range(1, n_layers + 1)
        }
        result = attribute.resolve_bands(
            n_layers=n_layers, l_cross=None, l_peak=None,
            mean_delta_vi=curve, gate_pass=gate, early_blocks=early[cell[0]],
        )
        band = result["ops37_as_written"]
        assert band["blocks"] == selected_blocks
        assert band["excluded_gate_failed_blocks"] == gate_failures[cell]


def test_e7_landmark_lookup_selects_only_primary_s_cf729_point_estimate() -> None:
    """OPS-37-5 lookup ignores s¹, non-primary sets, and other item sets."""
    common = {
        "model": "model-x",
        "format": "repetition",
        "pass": "analysis",
        "condition": "diac",
        "fewshot_set": 1,
        "role": "path",
        "item_set": "cf729",
        "metric_variant": "s",
    }
    rows = [
        {**common, "metric": "L_peak", "value": 23.0},
        {**common, "metric": "L_peak", "metric_variant": "s1", "value": 25.0},
        {**common, "metric": "L_peak", "fewshot_set": 2, "value": 20.0},
        {**common, "metric": "L_peak", "item_set": "main", "value": 29.0},
        {**common, "metric": "L_cross", "value": np.nan},
    ]
    frame = pd.DataFrame(rows)
    assert attribute.primary_landmark_point(
        frame, model="model-x", format_name="repetition", metric="L_peak"
    ) == 23
    assert attribute.primary_landmark_point(
        frame, model="model-x", format_name="repetition", metric="L_cross"
    ) is None


def test_e7_landmark_lookup_rejects_non_unique_primary_rows() -> None:
    """Ambiguous exact OPS-37-5 rows fail instead of selecting one silently."""
    row = {
        "model": "model-x",
        "format": "translation",
        "pass": "analysis",
        "condition": "diac",
        "fewshot_set": 1,
        "role": "path",
        "item_set": "cf729",
        "metric_variant": "s",
        "metric": "L_cross",
        "value": 27.0,
    }
    frame = pd.DataFrame([row, row])
    try:
        attribute.primary_landmark_point(
            frame, model="model-x", format_name="translation", metric="L_cross"
        )
    except ValueError as exc:
        assert "expected exactly one" in str(exc)
    else:
        raise AssertionError("non-unique primary landmark rows must fail")


def test_attention_share_reports_na_for_tiny_denominator() -> None:
    """A denominator below the frozen epsilon produces an NA share."""
    assert np.isnan(attribute.attention_share(1e-10, -2e-10, epsilon=1e-8))
    assert attribute.attention_share(2.0, -1.0, epsilon=1e-8) == 2.0 / 3.0


def test_direction_rebuild_preserves_e8_sign_and_bootstrap_key() -> None:
    """The E7 contrast is the negative E8 raw direction; bootstrap keys are stable."""
    languages = ("vi", "en", "zh", "fr", "id")
    centers = {
        "vi": np.array([1.0, 0.0, 0.2, -0.1], dtype=np.float32),
        "en": np.array([0.0, 1.0, -0.2, 0.1], dtype=np.float32),
        "zh": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "fr": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "id": np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    }
    rows = []
    values = []
    for lang in languages:
        for repetition in range(2):
            rows.append({"row": len(rows), "lang": lang, "format": "translation", "split": "fit"})
            block_value = centers[lang] + repetition * 0.01
            values.append(np.stack([np.zeros_like(block_value), block_value, block_value]))
    residuals = np.asarray(values, dtype=np.float32)
    v_vi, d_vi_en, counts = attribute.rebuild_a6_directions(
        residuals, rows, format_name="translation"
    )
    means = {
        language: residuals[[i for i, row in enumerate(rows) if row["lang"] == language]].mean(axis=0)
        for language in languages
    }
    raw_d = means["vi"] - means["en"]
    raw_u = means["en"] - means["vi"]
    # Since each class gets the same two-row offset, its mean contrast is fixed.
    assert attribute.direction_match_max_relative(raw_d, raw_u) == 0.0
    np.testing.assert_array_equal(v_vi[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(d_vi_en[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(
        d_vi_en[1:], raw_d[1:] / np.linalg.norm(raw_d[1:], axis=1, keepdims=True), atol=1e-7
    )
    assert v_vi.shape == d_vi_en.shape == (3, 4)
    assert counts == {language: 2 for language in languages}
    concepts = [f"c{i}" for i in range(30)]
    first = attribute.bootstrap_indices(concepts, seed=20260925, key="bootstrap|e7|gemma|translation")
    second = attribute.bootstrap_indices(concepts, seed=20260925, key="bootstrap|e7|gemma|translation")
    other = attribute.bootstrap_indices(concepts, seed=20260925, key="bootstrap|e7|llama|translation")
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_direction_rebuild_skips_zero_embedding_row() -> None:
    """Real P-D layer-axis fixtures keep identical residual row zero unnormalized."""
    languages = ("vi", "en", "zh", "fr", "id")
    rows: list[dict[str, object]] = []
    samples: list[np.ndarray] = []
    offsets = {
        "vi": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "en": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "zh": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        "fr": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        "id": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    }
    for language in languages:
        for replicate in range(2):
            rows.append({
                "row": len(rows), "lang": language,
                "format": "repetition", "split": "fit",
            })
            samples.append(np.stack([
                np.zeros(8, dtype=np.float32),
                offsets[language] + replicate * 0.01,
                offsets[language] + replicate * 0.02,
            ]))
    residuals = np.asarray(samples, dtype=np.float32)
    assert residuals.shape == (10, 3, 8)
    v_vi, d_vi_en, _counts = attribute.rebuild_a6_directions(
        residuals, rows, format_name="repetition"
    )
    np.testing.assert_array_equal(v_vi[0], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(d_vi_en[0], np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(v_vi[1:], axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(d_vi_en[1:], axis=1), 1.0, atol=1e-7)


def test_llama_bf16_capture_runs_e7_checks_end_to_end() -> None:
    """The production capture/check path handles a tiny CPU BF16 Llama-like decoder."""
    import job_e7_attribute as job

    loaded = _tiny_loaded_model("llama", seed=31)
    assert next(loaded.model.parameters()).dtype == torch.bfloat16
    generator = np.random.default_rng(20260925)
    directions = generator.normal(size=(3, 8)).astype(np.float32)
    records = [
        {"format": "repetition", "concept_id": f"tiny-{index}", "prompt": "x" * (index + 2)}
        for index in range(3)
    ]
    rows, _residuals, errors = job._capture_batch(
        loaded,
        records,
        v_vi=directions,
        d_vi_en=directions[::-1].copy(),
        heads=2,
        head_dim=4,
        layer_types=["standard"] * loaded.n_layers,
    )
    checks = job._check_summary([errors], _e7_check_config())
    assert rows
    assert checks["head_contributions"]["passed"]
    assert errors["residual_model_order"].max() == 0.0
    assert checks["residual_model_order"]["passed"]
    diagnostic = checks["update_relative_diagnostic"]
    assert diagnostic["gated"] is False
    assert np.isfinite(diagnostic["median"])
    assert len(diagnostic["per_block_median"]) == loaded.n_layers


def test_gemma_bf16_capture_runs_e7_checks_end_to_end() -> None:
    """The production capture/check path handles a tiny CPU BF16 Gemma-like decoder."""
    import job_e7_attribute as job

    loaded = _tiny_loaded_model("gemma", seed=47)
    assert next(loaded.model.parameters()).dtype == torch.bfloat16
    generator = np.random.default_rng(20260925)
    directions = generator.normal(size=(3, 8)).astype(np.float32)
    records = [
        {"format": "translation", "concept_id": f"tiny-{index}", "prompt": "y" * (index + 3)}
        for index in range(4)
    ]
    rows, _residuals, errors = job._capture_batch(
        loaded,
        records,
        v_vi=directions,
        d_vi_en=directions[::-1].copy(),
        heads=2,
        head_dim=4,
        layer_types=["global"] * loaded.n_layers,
    )
    checks = job._check_summary([errors], _e7_check_config())
    assert rows
    assert checks["head_contributions"]["passed"]
    assert errors["residual_model_order"].max() == 0.0
    assert checks["residual_model_order"]["passed"]
    diagnostic = checks["update_relative_diagnostic"]
    assert diagnostic["gated"] is False
    assert np.isfinite(diagnostic["median"])
    assert len(diagnostic["per_block_median"]) == loaded.n_layers


def test_five_percent_mlp_perturbation_fails_ops38_model_order_gate() -> None:
    """A perturbed captured MLP contribution fails the production OPS-38 check."""
    import job_e7_attribute as job

    resid_in = torch.zeros((2, 8), dtype=torch.bfloat16)
    attn = torch.zeros_like(resid_in)
    mlp = torch.ones_like(resid_in)
    resid_out = (resid_in + attn) + mlp
    head_sum = attn.float()
    perturbed = attribute.captured_component_errors(
        resid_in, attn, mlp * 1.05, resid_out, head_sum
    )
    perturbed["residual_update_relative_by_block"] = perturbed[
        "residual_update_relative"
    ][:, None]
    checks = job._check_summary([perturbed], _e7_check_config())
    assert not checks["residual_model_order"]["passed"]
    assert checks["residual_model_order"]["maximum"] >= 1.0e-4


def test_smoke_selection_uses_directions_rows_and_never_test_rows() -> None:
    """P-D stable ordering selects validate directions and excludes test metadata."""
    import job_e7_attribute as job

    prompts = [
        {"format": "repetition", "lang": "vi", "concept_id": "a", "split": "test", "prompt": "synthetic-test"},
        {"format": "repetition", "lang": "vi", "concept_id": "b", "split": "validate", "prompt": "synthetic-b"},
        {"format": "repetition", "lang": "en", "concept_id": "a", "split": "validate", "prompt": "synthetic-a"},
        {"format": "repetition", "lang": "en", "concept_id": "b", "split": "fit", "prompt": "synthetic-fit"},
    ]
    pd_rows = [
        {"row": index, **{key: row[key] for key in ("format", "lang", "concept_id", "split")}}
        for index, row in enumerate(prompts)
    ]
    selected, split = job._select_smoke_records(
        prompts, pd_rows, format_name="repetition", prompt_count=2
    )
    assert split == "validate"
    assert [(row["lang"], row["concept_id"]) for row in selected] == [("en", "a"), ("vi", "b")]
    assert all(row["split"] != "test" for row in selected)


def test_smoke_diagnose_reconstructs_selected_p_d_batches_in_exact_order() -> None:
    """S2 restores whole consecutive P-D batches and preserves source row order."""
    import job_e7_attribute as job

    ordered = [
        {"pd_array_row": row, "record_id": f"r{row}"}
        for row in range(10)
    ]
    batches = job._reconstruct_selected_batches(ordered, {1, 4, 8}, batch_size=3)
    assert [batch["batch_index"] for batch in batches] == [0, 1, 2]
    assert [
        [record["record_id"] for record in batch["records"]]
        for batch in batches
    ] == [["r0", "r1", "r2"], ["r3", "r4", "r5"], ["r6", "r7", "r8"]]
    assert [batch["selected_local_rows"] for batch in batches] == [[1], [1], [2]]


def test_smoke_diagnose_selection_cannot_include_test_rows() -> None:
    """The diagnostic selector filters test-split rows before constructing S1/S2/S3."""
    import job_e7_attribute as job

    prompts = [
        {"format": "translation", "lang": "en", "concept_id": "test", "split": "test", "prompt": "test fixture"},
        {"format": "translation", "lang": "en", "concept_id": "validate-a", "split": "validate", "prompt": "validate fixture a"},
        {"format": "translation", "lang": "vi", "concept_id": "validate-b", "split": "validate", "prompt": "validate fixture b"},
    ]
    pd_rows = [
        {"row": index, **{key: row[key] for key in ("format", "lang", "concept_id", "split")}}
        for index, row in enumerate(prompts)
    ]
    selected, split = job._select_smoke_records(
        prompts, pd_rows, format_name="translation", prompt_count=2
    )
    assert split == "validate"
    assert len(selected) == 2
    assert all(row["split"] != "test" for row in selected)


def test_e7_replay_batches_preserve_source_order_and_discard_non_e7_projection_rows() -> None:
    """P-A replay retains exact batch membership but selects only E7 rows for output."""
    import job_e7_attribute as job

    records = [
        {
            "pl_shard": 0,
            "format": "repetition",
            "condition": "diac",
            "fewshot_set": 1,
            "lang": "vi",
            "concept_id": concept,
            "prompt": f"fixture-{concept}",
            "m1_extension": extension,
        }
        for concept, extension in (("b", False), ("x", True), ("a", False), ("y", True))
    ]
    batches = attribute.replay_capture_batches(
        records,
        stable_key=job._pl_prompt_stable_key,
        shard_key=lambda row: int(row["pl_shard"]),
        batch_size=2,
        shard_order_key=int,
    )
    assert [[row["concept_id"] for row in batch["records"]] for batch in batches] == [
        ["a", "b"], ["x", "y"]
    ]
    e7_keys = {job._e7_prompt_key(row) for row in records if not row["m1_extension"]}
    assert [
        [batch["records"][index]["concept_id"] for index in job._e7_replay_indices(batch["records"], e7_keys)]
        for batch in batches
    ] == [["a", "b"], []]


def test_p_a_and_smoke_replay_wrappers_call_the_same_generic_function(monkeypatch) -> None:
    """P-A and smoke both delegate batch construction to attribute.replay_capture_batches."""
    import job_e7_attribute as job

    config = {
        "e7": {
            "replay": {
                "p_a": {"source": "pl", "batch_size_config": "pl.batch_size", "divisor": 2},
                "smoke": {"source": "pd", "batch_size_config": "inference.batch_size", "divisor": 1},
            }
        },
        "pl": {"batch_size": {"gemma": 8}},
        "inference": {"batch_size": {"gemma": 8}},
    }
    calls: list[int] = []
    original = attribute.replay_capture_batches

    def spy(records, **kwargs):
        calls.append(int(kwargs["batch_size"]))
        return original(records, **kwargs)

    monkeypatch.setattr(attribute, "replay_capture_batches", spy)
    job._pl_replay_batches(config, "gemma", [{
        "pl_shard": 0, "format": "repetition", "condition": "diac",
        "fewshot_set": 1, "lang": "vi", "concept_id": "a", "prompt": "fixture",
    }])
    job._pd_replay_batches(config, "gemma", [{
        "pd_array_row": 0, "format": "repetition", "concept_id": "a",
    }])
    assert calls == [4, 8]


def test_ops39_exact_replay_gate_uses_inclusive_1e6_maximum() -> None:
    """10(c′) accepts its inclusive 1e-6 boundary and rejects a larger maximum."""
    assert attribute.exact_replay_gate_summary([0.0, 1.0e-6], 1.0e-6)["passed"]
    rejected = attribute.exact_replay_gate_summary([1.0001e-6], 1.0e-6)
    assert not rejected["passed"]
    assert rejected["maximum"] == 1.0001e-6


def test_p_a_analysis_unit_plan_uses_summary_replay_batches_and_source() -> None:
    """P-A unit keys follow P-L replay chunks rather than the former 32-row plan."""
    summary = {
        "families": {
            "gemma": {
                "repetition": {"replay_batch_count": 3},
            }
        }
    }
    keys = attribute.replay_unit_keys_from_summary(
        summary,
        family="gemma",
        format_name="repetition",
        pass_name="pa",
        replay_source="pl",
        item_sets=("main", "cf729"),
    )
    assert keys == [
        {
            "family": "gemma", "pass": "pa", "format": "repetition",
            "item_set": item_set, "chunk": chunk, "replay_source": "pl",
        }
        for chunk in range(3)
        for item_set in ("main", "cf729")
    ]


def test_summary_profile_comes_from_slurm_environment_not_config(monkeypatch) -> None:
    """Profile reporting follows Slurm partition/QOS rather than E7 config defaults."""
    import job_e7_attribute as job

    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)
    monkeypatch.delenv("SLURM_JOB_QOS", raising=False)
    assert job._runtime_profile() == "unknown"
    monkeypatch.setenv("SLURM_JOB_PARTITION", "normal")
    assert job._runtime_profile() == "normal"
    monkeypatch.setenv("SLURM_JOB_PARTITION", "unset-partition")
    monkeypatch.setenv("SLURM_JOB_QOS", "mig1-18")
    assert job._runtime_profile() == "small"

