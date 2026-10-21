"""E8 steering helpers on tiny random causal models and synthetic fit arrays."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vilens import io, locks, steer
from vilens.derived import e8_select


class TinyTokenizer:
    """Small deterministic tokenizer fixture with optional whitespace-only first tokens."""

    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __init__(self, whitespace_first: bool = False) -> None:
        self.whitespace_first = whitespace_first

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        if text.startswith("P"):
            ids = [10]
            suffix = text[1:]
            if suffix == " en":
                ids += [30, 20] if self.whitespace_first else [20]
            elif suffix == " vi":
                ids += [30, 21] if self.whitespace_first else [21]
            elif suffix:
                ids += [40 + (ord(char) % 80) for char in suffix]
        else:
            ids = [40 + (ord(char) % 80) for char in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        result = []
        for token_id in token_ids:
            if token_id in {0, 1, 2}:
                if skip_special_tokens:
                    continue
                if token_id == 2:
                    result.append("<eos>")
                continue
            if token_id == 10:
                result.append("P")
            elif token_id == 20:
                result.append("en")
            elif token_id == 21:
                result.append("vi")
            elif token_id == 30:
                result.append(" ")
            elif token_id == 31:
                result.append("\n")
            elif token_id == 32:
                result.append("-")
            else:
                result.append(chr((int(token_id) - 40) % 80 + 32))
        return "".join(result)


def _capture_block(loaded: Any, ids: torch.Tensor, mask: torch.Tensor, layer: int, **kwargs: Any) -> torch.Tensor:
    """Return a clone of one block's output for one model forward."""
    saved: list[torch.Tensor] = []
    module = loaded.layers[layer - 1]

    def capture(_module: Any, _args: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        saved.append(hidden.detach().clone())

    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            loaded.model(input_ids=ids, attention_mask=mask, **kwargs)
    finally:
        handle.remove()
    return saved[0]


def test_hook_changes_only_requested_positions_for_prompt_cache_and_teacher_forcing(
    tiny_bundle: dict[str, Any],
) -> None:
    """Apply the vector only at the final prompt token, cached token, or listed teacher positions."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = TinyTokenizer()
    layer = min(2, loaded.n_layers)
    ids = torch.tensor([[1, 10, 11, 12, 13], [1, 14, 15, 16, 17]], dtype=torch.long)
    mask = torch.ones_like(ids)
    vector = torch.ones(loaded.d_model, dtype=torch.float32)

    baseline = _capture_block(loaded, ids, mask, layer, use_cache=False)
    with steer.steer_block(loaded, layer, vector, 0.5):
        changed = _capture_block(loaded, ids, mask, layer, use_cache=False)
    expected = torch.zeros_like(baseline)
    expected[:, -1, :] = 0.5
    difference = changed - baseline
    torch.testing.assert_close(difference, expected, rtol=0, atol=1e-6)
    assert torch.equal(difference[:, :-1, :], torch.zeros_like(difference[:, :-1, :]))

    with torch.inference_mode():
        base_first = loaded.model(input_ids=ids, attention_mask=mask, use_cache=True)
        base_step = _capture_block(
            loaded, torch.tensor([[18], [19]]), torch.ones((2, 6), dtype=torch.long), layer,
            past_key_values=base_first.past_key_values, use_cache=True,
        )
        steered_first = loaded.model(input_ids=ids, attention_mask=mask, use_cache=True)
        with steer.steer_block(loaded, layer, vector, 0.5):
            changed_step = _capture_block(
                loaded, torch.tensor([[18], [19]]), torch.ones((2, 6), dtype=torch.long), layer,
                past_key_values=steered_first.past_key_values, use_cache=True,
            )
    torch.testing.assert_close(changed_step - base_step, torch.full_like(base_step, 0.5), rtol=0, atol=1e-6)

    positions = [[1, 3], [2]]
    base_teacher = _capture_block(loaded, ids, mask, layer, use_cache=False)
    with steer.steer_block(loaded, layer, vector, 0.5, positions=positions):
        changed_teacher = _capture_block(loaded, ids, mask, layer, use_cache=False)
    expected_teacher = torch.zeros_like(base_teacher)
    for row, selected in enumerate(positions):
        for position in selected:
            expected_teacher[row, position, :] = 0.5
    difference = changed_teacher - base_teacher
    torch.testing.assert_close(difference, expected_teacher, rtol=0, atol=1e-6)
    for row, selected in enumerate(positions):
        outside = sorted(set(range(ids.shape[1])) - set(selected))
        assert torch.equal(difference[row, outside, :], torch.zeros_like(difference[row, outside, :]))


def test_zero_strength_hook_preserves_logits_and_generation(tiny_bundle: dict[str, Any]) -> None:
    """A k=0 intervention preserves exact logits and greedy token IDs."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = TinyTokenizer()
    layer = 1
    ids = torch.tensor([[1, 10, 11]], dtype=torch.long)
    mask = torch.ones_like(ids)
    vector = np.ones(loaded.d_model, dtype=np.float32)
    with torch.inference_mode():
        base_logits = loaded.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with steer.steer_block(loaded, layer, vector, 0.0):
            hooked_logits = loaded.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    assert torch.equal(base_logits, hooked_logits)
    baseline = steer.greedy_generate_batch(loaded, ["P"], max_new_tokens=4)
    hooked = steer.greedy_generate_batch(
        loaded, ["P"], max_new_tokens=4, vector=vector, layer=layer, k=0.0
    )
    assert baseline[0]["token_ids"] == hooked[0]["token_ids"]


def test_random_control_is_deterministic_and_matches_vector_norm() -> None:
    """Use the registered family/layer stream and preserve each u-layer norm."""
    u = np.arange(24, dtype=np.float32).reshape(4, 6) + 1
    left = steer.random_control(u, "gemma", 20260925)
    right = steer.random_control(u, "gemma", 20260925)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), np.linalg.norm(u, axis=1), rtol=1e-6, atol=1e-6)


def test_steering_vectors_reads_synthetic_fit_split_residuals(tmp_path: Path) -> None:
    """Compute fp32 en-minus-vi means and retain class fit counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    key = {"family": "llama", "pass": "pd", "source": "matched", "format": "translation"}
    parquet, _sidecar = io.unit_paths(tmp_path, key)
    array_path = parquet.with_suffix(".resid.npy")
    rows = [
        {"row": 0, "format": "translation", "lang": "en", "split": "fit"},
        {"row": 1, "format": "translation", "lang": "vi", "split": "fit"},
        {"row": 2, "format": "translation", "lang": "en", "split": "validate"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    np.save(array_path, np.asarray([np.full((3, 4), 2), np.full((3, 4), 0), np.full((3, 4), 99)], dtype=np.float32))
    vectors, norms, counts = steer.steering_vectors("llama", "translation", results_dir=tmp_path)
    assert vectors.shape == (3, 4)
    np.testing.assert_array_equal(vectors, np.full((3, 4), 2, dtype=np.float32))
    np.testing.assert_allclose(norms, np.full(3, 4, dtype=np.float32))
    assert counts == {"en": 1, "vi": 1}


@pytest.mark.parametrize("whitespace_first", [False, True])
def test_delta_first_handles_normal_and_whitespace_first_targets(whitespace_first: bool) -> None:
    """Score the first content token after joint tokenization, teacher-forcing a whitespace token when needed."""
    from conftest import make_tiny_model
    from vilens.config import load_config
    from vilens.models import wrap_model

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/experiments.yaml")
    tokenizer = TinyTokenizer(whitespace_first=whitespace_first)
    loaded = wrap_model(make_tiny_model("llama"), tokenizer, "llama", cfg)
    records = [{"concept_id": "c1", "prompt": "P", "format": "repetition"}]
    concepts = {
        "c1": {"en": "en", "vi": "vi"},
        "unselected": {"en": "extra", "vi": "extra"},
    }
    rows = steer.delta_first_batch(
        loaded, records, concepts, family="llama", format="repetition", batch_size=1
    )
    assert len(rows) == 1
    assert rows[0]["en_token_id"] != rows[0]["vi_token_id"]
    assert np.isfinite(rows[0]["value"])


def test_stop_rule_uses_registered_newline_and_space_hyphen() -> None:
    """The existing truncator cuts at either registered stop string."""
    from vilens.passes import _truncate_generation

    assert _truncate_generation("answer\nmore") == "answer"
    assert _truncate_generation("answer - note") == "answer"


@pytest.mark.parametrize(("tokens", "expected_count"), [([31], 1), ([30, 32], 2)])
def test_greedy_generation_stops_on_newline_or_space_hyphen(tokens: list[int], expected_count: int) -> None:
    """The autoregressive loop stops as soon as a registered string is decoded."""
    class ScriptedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> Any:
            target = tokens[min(self.calls, len(tokens) - 1)]
            self.calls += 1
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 300), -10.0)
            logits[:, :, target] = 10.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    loaded = SimpleNamespace(model=ScriptedModel(), tokenizer=TinyTokenizer(), family="llama", n_layers=3, d_model=64)
    result = steer.greedy_generate_batch(loaded, ["P"], max_new_tokens=8, stop_strings=("\n", " -"))[0]
    assert result["n_new_tokens"] == expected_count


def test_left_padded_batched_generation_matches_unbatched(tiny_bundle: dict[str, Any]) -> None:
    """Tiny random decoders preserve greedy output for left-padded batch versus individual prompts."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = TinyTokenizer()
    prompts = ["P", "P-long"]
    batched = steer.greedy_generate_batch(loaded, prompts, max_new_tokens=5)
    individual = [steer.greedy_generate_batch(loaded, [prompt], max_new_tokens=5)[0] for prompt in prompts]
    assert [row["token_ids"] for row in batched] == [row["token_ids"] for row in individual]


def test_primary_selection_matches_e8_select_on_synthetic_competence() -> None:
    """The E8 selector preserves the frozen rank-and-quota result for VI translation rows."""
    order = [
        {"family": "gemma", "stratum": stratum, "rank": rank, "concept_id": f"{stratum}{rank}"}
        for stratum in ("a", "b") for rank in range(5)
    ]
    competent_rows = [
        {"format": "translation", "concept_id": row["concept_id"], "competence_class": "VI"}
        for row in order if int(row["rank"]) < 4
    ] + [{"format": "repetition", "concept_id": "a4", "competence_class": "VI"}]
    selected, quotas = steer.selected_competent_ids(order, competent_rows, "gemma", 5)
    expected = e8_select(order, {row["concept_id"] for row in competent_rows if row["format"] == "translation"}, "gemma", 5)
    assert selected == expected
    assert sum(quotas.values()) == len(selected) == 5


def test_primary_lock_failure_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary access propagates a failed Checkpoint-H lock before any model load."""
    def reject(_root: Any) -> None:
        raise locks.IntegrityLockError("synthetic adjudication lock failure")

    monkeypatch.setattr(locks, "ensure_adjudication_committed", reject)
    with pytest.raises(locks.IntegrityLockError, match="synthetic adjudication lock failure"):
        steer.ensure_primary_locks({}, ".", [{"split": "test"}])


def test_preflight_only_never_calls_weight_loader() -> None:
    """The preflight dispatch returns without invoking the model-loading runner."""
    calls = []
    assert steer.run_if_not_preflight(True, lambda: calls.append("weights")) is None
    assert calls == []
