"""Tests for right-padded teacher-forced token scoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from vilens.lens import LensHead, teacher_forced
from vilens.tokenize import JointEncoding


def test_teacher_forced_positions_scores_and_padding_invariance(tiny_bundle: dict[str, object]) -> None:
    """Use the previous-token positions and preserve per-sequence scores in batches."""
    bundle = tiny_bundle
    loaded = bundle["loaded"]
    model = bundle["model"]
    family = bundle["family"]
    assert isinstance(model, torch.nn.Module)
    encodings = [
        JointEncoding((1, 5, 7), (1, 5, 7, 9, 11, 13), True),
        JointEncoding((2, 4, 6, 8, 10), (2, 4, 6, 8, 10, 12, 14), True),
        JointEncoding((3, 6), (3, 6, 15, 17, 19), True),
    ]
    results = teacher_forced(loaded, LensHead.from_loaded(loaded, "fp32"), encodings, pad_id=0)
    assert results[0].positions.tolist() == [2, 3, 4]
    assert results[1].positions.tolist() == [4, 5]
    assert results[2].positions.tolist() == [1, 2, 3]

    max_padding_delta = 0.0
    for encoding, batch_result in zip(encodings, results, strict=True):
        ids = torch.tensor([encoding.joint_ids], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits[0]
        target_ids = torch.tensor(encoding.target_ids, dtype=torch.long)
        positions = torch.tensor(batch_result.positions, dtype=torch.long)
        plain = functional.log_softmax(logits.index_select(0, positions).float(), dim=-1)
        expected = plain.gather(1, target_ids[:, None]).squeeze(-1).numpy()
        np.testing.assert_allclose(batch_result.model_logprobs, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            batch_result.lens_logprobs[loaded.n_layers], batch_result.model_logprobs,
            rtol=0, atol=1e-5,
        )
        one = teacher_forced(loaded, LensHead.from_loaded(loaded, "fp32"), [encoding], pad_id=0)[0]
        max_padding_delta = max(
            max_padding_delta,
            float(np.max(np.abs(batch_result.model_logprobs - one.model_logprobs))),
            float(np.max(np.abs(batch_result.lens_logprobs - one.lens_logprobs))),
        )
    print(f"T04A_METRIC padding_invariance {family} {max_padding_delta:.9g}")
    assert max_padding_delta < 1e-5
