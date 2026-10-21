"""Logit-lens projections and right-padded teacher-forced scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

from .hooks import record
from .models import LoadedModel
from .tokenize import JointEncoding


class LensHead:
    """Apply a model's final normalization and unembedding in one precision mode."""

    def __init__(
        self,
        norm: Any,
        lm_head: Any,
        *,
        mode: Literal["fp32", "model"],
        softcap: float | None,
        position_chunk: int,
    ) -> None:
        """Store the projection modules and precision-specific parameters."""
        import torch

        self.norm = norm
        self.lm_head = lm_head
        self.mode = mode
        self.softcap = softcap
        self.position_chunk = int(position_chunk)
        self.weight = None if mode == "model" else lm_head.weight.detach().to(dtype=torch.float32)
        bias = getattr(lm_head, "bias", None)
        self.bias = None if mode == "model" or bias is None else bias.detach().to(dtype=torch.float32)

    @classmethod
    def from_loaded(cls, loaded: LoadedModel, mode: Literal["fp32", "model"]) -> LensHead:
        """Build an fp32 analysis lens or a model-dtype path-check lens."""
        import copy as copy_module

        if mode not in {"fp32", "model"}:
            raise ValueError("mode must be 'fp32' or 'model'")
        norm = loaded.final_norm
        lm_head = loaded.lm_head
        if mode == "fp32":
            norm = copy_module.deepcopy(loaded.final_norm).float()
        return cls(
            norm,
            lm_head,
            mode=mode,
            softcap=loaded.softcap,
            position_chunk=loaded.lens_position_chunk,
        )

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits, applying configured softcap."""
        import torch
        import torch.nn.functional as functional

        reference = next(self.norm.parameters(), None)
        if reference is not None:
            h = h.to(device=reference.device)
            if self.mode == "model":
                h = h.to(dtype=reference.dtype)

        if self.mode == "fp32":
            normalized = self.norm(h.to(dtype=torch.float32))
            logits = functional.linear(normalized, self.weight, self.bias)
        else:
            logits = self.lm_head(self.norm(h))
        if self.softcap is not None:
            cap = float(self.softcap)
            logits = torch.tanh(logits / cap) * cap
        return logits

    def logprobs_for(self, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Compute float32 log probabilities for requested ids in bounded row chunks."""
        import torch
        import torch.nn.functional as functional

        if h.ndim != 2:
            raise ValueError("h must have shape [positions, hidden]")
        if ids.ndim != 2 or ids.shape[0] != h.shape[0]:
            raise ValueError("ids must have shape [positions, requested_ids]")
        if self.position_chunk <= 0:
            raise ValueError("position_chunk must be positive")
        if h.shape[0] == 0:
            return torch.empty((0, ids.shape[1]), dtype=torch.float32, device=h.device)
        chunks: list[torch.Tensor] = []
        for start in range(0, h.shape[0], self.position_chunk):
            end = min(start + self.position_chunk, h.shape[0])
            logits = self.logits(h[start:end])
            log_probs = functional.log_softmax(logits.float(), dim=-1)
            wanted = ids[start:end].to(device=log_probs.device, dtype=torch.long)
            expanded = log_probs.unsqueeze(1).expand(-1, wanted.shape[1], -1)
            chunks.append(expanded.gather(2, wanted.unsqueeze(-1)).squeeze(-1))
        return torch.cat(chunks, dim=0)

    def top1(self, h: torch.Tensor) -> torch.Tensor:
        """Return the most likely vocabulary id for each hidden-state row."""
        return self.logits(h).argmax(dim=-1)


@dataclass(frozen=True)
class TFResult:
    """Teacher-forced scores and scoring positions for one joint encoding."""

    target_ids: np.ndarray
    positions: np.ndarray
    lens_logprobs: np.ndarray
    model_logprobs: np.ndarray
    lens_top1: np.ndarray


def teacher_forced(
    loaded: LoadedModel,
    lens: LensHead,
    encodings: Sequence[JointEncoding],
    *,
    pad_id: int,
) -> list[TFResult]:
    """Score jointly encoded targets in one right-padded causal-model batch."""
    import torch
    import torch.nn.functional as functional

    if not encodings:
        return []
    device = next(loaded.model.parameters()).device
    target_rows: list[tuple[int, ...]] = []
    score_rows: list[tuple[int, ...]] = []
    sequences: list[tuple[int, ...]] = []
    for encoding in encodings:
        if not encoding.prefix_ok:
            raise ValueError("teacher_forced requires a valid prompt-prefix encoding")
        target_ids = tuple(encoding.target_ids)
        prompt_length = len(encoding.prompt_ids)
        if prompt_length < 1 or not target_ids:
            raise ValueError("each encoding must contain a non-empty prompt and target")
        sequence = tuple(encoding.joint_ids)
        positions = tuple(prompt_length - 1 + index for index in range(len(target_ids)))
        if positions[-1] >= len(sequence):
            raise AssertionError("a scoring position falls outside its unpadded sequence")
        target_rows.append(target_ids)
        score_rows.append(positions)
        sequences.append(sequence)

    batch_size = len(sequences)
    max_sequence = max(map(len, sequences))
    max_targets = max(map(len, target_rows))
    input_ids = torch.full((batch_size, max_sequence), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_sequence), dtype=torch.long, device=device)
    positions = torch.empty((batch_size, max_targets), dtype=torch.long, device=device)
    for row, (sequence, score_positions) in enumerate(zip(sequences, score_rows, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        if any(position < 0 or position >= length for position in score_positions):
            raise AssertionError("a scoring position falls on padding")
        positions[row, :len(score_positions)] = torch.as_tensor(
            score_positions, dtype=torch.long, device=device
        )
        positions[row, len(score_positions):] = score_positions[0]

    recording = record(
        loaded,
        input_ids,
        attention_mask,
        positions=positions,
        components=False,
        keep_on="cpu",
    )
    lens_device = next(lens.norm.parameters()).device
    batch_lens_logprobs = torch.zeros(
        (batch_size, loaded.n_layers + 1, max_targets), dtype=torch.float32, device=lens_device
    )
    batch_lens_top1 = torch.zeros(
        (batch_size, loaded.n_layers + 1, max_targets), dtype=torch.long, device=lens_device
    )
    model_logprobs: list[np.ndarray] = []
    results: list[TFResult] = []
    for row, (target_ids_tuple, score_positions) in enumerate(zip(target_rows, score_rows, strict=True)):
        target_count = len(target_ids_tuple)
        ids = torch.as_tensor(target_ids_tuple, dtype=torch.long).reshape(-1, 1)
        layer_logprobs: list[torch.Tensor] = []
        layer_top1: list[torch.Tensor] = []
        for layer_index in range(loaded.n_layers + 1):
            hidden = recording.resid[layer_index, row, :target_count]
            layer_logprobs.append(lens.logprobs_for(hidden, ids).squeeze(-1))
            layer_top1.append(lens.top1(hidden))
        batch_lens_logprobs[row, :, :target_count] = torch.stack(layer_logprobs)
        batch_lens_top1[row, :, :target_count] = torch.stack(layer_top1)
        model_logits = recording.logits[row, :target_count].float()
        model_lp = functional.log_softmax(model_logits, dim=-1).gather(1, ids).squeeze(-1)
        model_logprobs.append(model_lp.numpy().astype(np.float32, copy=False))

    # Transfer all lens outputs from the device once per teacher-forced batch.
    lens_logprobs_np = batch_lens_logprobs.detach().cpu().numpy()
    lens_top1_np = batch_lens_top1.detach().cpu().numpy()
    for row, (target_ids_tuple, score_positions) in enumerate(zip(target_rows, score_rows, strict=True)):
        target_count = len(target_ids_tuple)
        results.append(
            TFResult(
                target_ids=np.asarray(target_ids_tuple, dtype=np.int64),
                positions=np.asarray(score_positions, dtype=np.int64),
                lens_logprobs=lens_logprobs_np[row, :, :target_count].astype(np.float32, copy=False),
                model_logprobs=model_logprobs[row],
                lens_top1=lens_top1_np[row, :, :target_count].astype(np.int64, copy=False),
            )
        )
    return results
