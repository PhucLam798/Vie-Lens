"""E8 class-mean steering, generation, and first-content-token diagnostics."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .derived import stream, target_string
from .tokenize import TokenClassifier, encode_joint


def steering_vectors(
    family: str,
    format: str,
    *,
    results_dir: str | Path = "results/pd",
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return fp32 en-minus-vi fit-split means from the registered P-D residual unit."""
    if family not in {"gemma", "qwen", "llama"} or format not in {"translation", "repetition"}:
        raise ValueError("family and format must identify a registered P-D matched unit")
    root = Path(results_dir)
    key = {"family": family, "pass": "pd", "source": "matched", "format": format}
    from .io import unit_paths

    parquet_path, _sidecar = unit_paths(root, key)
    array_path = parquet_path.with_suffix(".resid.npy")
    if not array_path.is_file() or not parquet_path.is_file():
        raise FileNotFoundError(f"Missing P-D residual/index inputs for {family}/{format}: {array_path.name}")
    import pyarrow.parquet as pq

    index = pq.read_table(parquet_path, columns=["row", "format", "lang", "split"]).to_pylist()
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or values.shape[0] != len(index):
        raise ValueError(f"P-D residual/index shape mismatch for {family}/{format}")
    rows = {int(row["row"]): row for row in index}
    fit_by_language: dict[str, list[int]] = {"en": [], "vi": []}
    for array_row, row in rows.items():
        if str(row.get("format")) == format and str(row.get("split")) == "fit":
            language = str(row.get("lang"))
            if language in fit_by_language:
                fit_by_language[language].append(array_row)
    counts = {language: len(indices) for language, indices in fit_by_language.items()}
    if not counts["en"] or not counts["vi"]:
        raise ValueError(f"P-D fit split lacks an en/vi class for {family}/{format}: {counts}")
    en = np.asarray(values[fit_by_language["en"]], dtype=np.float32).mean(axis=0, dtype=np.float32)
    vi = np.asarray(values[fit_by_language["vi"]], dtype=np.float32).mean(axis=0, dtype=np.float32)
    vectors = np.asarray(en - vi, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32, copy=False)
    return vectors, norms, counts


def random_control(
    u: np.ndarray,
    family: str,
    seed: int,
    *,
    key_template: str = "steer_random|{family}|{layer}",
) -> np.ndarray:
    """Draw deterministic family/layer Gaussian controls scaled to each u-layer norm."""
    vectors = np.asarray(u, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("u must have shape [layer, hidden]")
    output = np.zeros_like(vectors, dtype=np.float32)
    for layer, target_norm in enumerate(np.linalg.norm(vectors, axis=1)):
        key = key_template.format(family=family, layer=layer)
        direction = stream(int(seed), key).standard_normal(
            vectors.shape[1]
        ).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 0 and target_norm > 0:
            output[layer] = direction * (float(target_norm) / norm)
    return output


@contextlib.contextmanager
def steer_block(
    loaded: Any,
    layer: int,
    vector: Any,
    k: float,
    *,
    positions: Sequence[Sequence[int]] | None = None,
) -> Iterator[None]:
    """Add cast(k·vector) to one block's selected sequence positions for each forward."""
    import torch

    if not 1 <= int(layer) <= int(loaded.n_layers):
        raise ValueError(f"steered layer must be in 1..{loaded.n_layers}")
    module = loaded.layers[int(layer) - 1]
    source = torch.as_tensor(vector, dtype=torch.float32)
    if source.ndim != 1 or source.numel() != int(loaded.d_model):
        raise ValueError("steering vector width must equal the decoder hidden size")

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("decoder block output must be [batch, sequence, hidden]")
        delta = (source.to(device=hidden.device) * float(k)).to(dtype=hidden.dtype)
        changed = hidden.clone()
        if positions is None:
            changed[:, -1, :] = changed[:, -1, :] + delta
        else:
            if len(positions) != hidden.shape[0]:
                raise ValueError("positions must contain one position-list per batch row")
            for row_index, row_positions in enumerate(positions):
                for position in row_positions:
                    normalized = int(position)
                    if normalized < 0:
                        normalized += int(hidden.shape[1])
                    if not 0 <= normalized < hidden.shape[1]:
                        raise ValueError("steering position is outside the sequence")
                    changed[row_index, normalized, :] = changed[row_index, normalized, :] + delta
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _pad_id(tokenizer: Any) -> int:
    """Return a configured padding token, falling back to EOS or BOS."""
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def _left_pad(
    sequences: Sequence[Sequence[int]], pad_id: int, device: Any
) -> tuple[Any, Any, Any]:
    """Left-pad token sequences and produce masks and position IDs for decoder forwards."""
    import torch

    width = max((len(sequence) for sequence in sequences), default=0)
    if width == 0:
        raise ValueError("cannot pad empty token sequences")
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        values = torch.as_tensor(sequence, dtype=torch.long, device=device)
        ids[row, width - len(sequence) :] = values
        mask[row, width - len(sequence) :] = 1
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids


def greedy_generate_batch(
    loaded: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 8,
    stop_strings: Sequence[str] = ("\n", " -"),
    vector: Any | None = None,
    layer: int | None = None,
    k: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate greedily with left padding, optional layer hook, and registered text stops."""
    import torch
    from .passes import _truncate_generation

    if not prompts:
        return []
    tokenizer = loaded.tokenizer
    device = next(loaded.model.parameters()).device
    encoded = [
        [int(value) for value in tokenizer(str(prompt), add_special_tokens=True)["input_ids"]]
        for prompt in prompts
    ]
    if any(not sequence for sequence in encoded):
        raise ValueError("a prompt encoded to no tokens")
    ids, attention_mask, position_ids = _left_pad(encoded, _pad_id(tokenizer), device)
    past = None
    generated: list[list[int]] = [[] for _ in encoded]
    finished = [False] * len(encoded)
    eos_ids = {int(value) for value in (getattr(tokenizer, "eos_token_id", None),) if value is not None}
    if isinstance(getattr(tokenizer, "eos_token_id", None), (list, tuple)):
        eos_ids = {int(value) for value in tokenizer.eos_token_id}
    if vector is None or layer is None:
        context = contextlib.nullcontext()
    else:
        context = steer_block(loaded, int(layer), vector, k)
    with context, torch.inference_mode():
        output = loaded.model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past = getattr(output, "past_key_values", None)
        if past is None:
            raise RuntimeError("model forward did not return a cache for greedy generation")
        next_logits = output.logits[:, -1, :]
        for step in range(int(max_new_tokens)):
            next_ids = next_logits.argmax(dim=-1)
            next_values = next_ids.detach().cpu().tolist()
            for row, token_id in enumerate(next_values):
                if finished[row]:
                    continue
                generated[row].append(int(token_id))
                decoded = tokenizer.decode(
                    generated[row], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if int(token_id) in eos_ids or any(stop in decoded for stop in stop_strings):
                    finished[row] = True
            if all(finished) or step + 1 >= int(max_new_tokens):
                break
            active = torch.as_tensor([not value for value in finished], dtype=torch.bool, device=device)
            input_step = next_ids[:, None]
            attention_mask = torch.cat([attention_mask, active.to(dtype=attention_mask.dtype)[:, None]], dim=1)
            position_step = (attention_mask.sum(dim=1) - 1).clamp_min(0)[:, None]
            cache_position = torch.as_tensor([ids.shape[1] + step], dtype=torch.long, device=device)
            output = loaded.model(
                input_ids=input_step,
                attention_mask=attention_mask,
                position_ids=position_step,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
            )
            past = getattr(output, "past_key_values", None)
            if past is None:
                raise RuntimeError("cached decoder forward did not return a cache")
            next_logits = output.logits[:, -1, :]
    rows = []
    for token_ids in generated:
        raw = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = _truncate_generation(raw)
        rows.append({"answer_text": answer, "token_ids": token_ids, "n_new_tokens": len(token_ids)})
    return rows


def delta_first_batch(
    loaded: Any,
    prompts: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    format: str,
    vector: Any | None = None,
    layer: int | None = None,
    k: float = 0.0,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Compute en-minus-vi first-content-token log probability under hooked forwards."""
    import torch
    import torch.nn.functional as functional
    missing_concepts = sorted(
        {str(prompt.get("concept_id")) for prompt in prompts} - set(concepts)
    )
    if missing_concepts:
        raise KeyError(f"concept metadata is missing for {missing_concepts[:5]}")
    tokenizer = loaded.tokenizer
    classifier = TokenClassifier(tokenizer)
    specs: list[dict[str, Any]] = []
    for prompt in prompts:
        concept_id = str(prompt["concept_id"])
        concept = concepts[concept_id]
        target_data = []
        for language in ("en", "vi"):
            target = target_string(concept, language, "released")
            encoding = encode_joint(tokenizer, str(prompt["prompt"]), target, strict=True)
            content_index = next(
                (index for index, token_id in enumerate(encoding.target_ids) if classifier.is_content(token_id)),
                None,
            )
            if content_index is None:
                raise ValueError(f"{concept_id}/{language} has no content token")
            prefix_length = len(encoding.prompt_ids) + int(content_index) + 1
            sequence = list(encoding.joint_ids[:prefix_length])
            if not sequence:
                raise ValueError("teacher-forced target sequence is empty")
            target_data.append(
                {
                    "sequence": sequence,
                    "token_id": int(encoding.target_ids[content_index]),
                    "positions_start": len(encoding.prompt_ids) - 1,
                    "predict_position": len(encoding.prompt_ids) - 1 + int(content_index),
                }
            )
        specs.append({"concept_id": concept_id, "targets": target_data})

    output_rows: list[dict[str, Any]] = []
    device = next(loaded.model.parameters()).device
    for start in range(0, len(specs), int(batch_size)):
        batch = specs[start : start + int(batch_size)]
        flat = [target for item in batch for target in item["targets"]]
        sequences = [target["sequence"] for target in flat]
        ids, attention_mask, position_ids = _left_pad(sequences, _pad_id(tokenizer), device)
        width = ids.shape[1]
        hook_positions = [
            list(range(width - len(target["sequence"]) + target["positions_start"], width))
            for target in flat
        ]
        if vector is None or layer is None:
            hook = contextlib.nullcontext()
        else:
            hook = steer_block(loaded, int(layer), vector, k, positions=hook_positions)
        with hook, torch.inference_mode():
            result = loaded.model(
                input_ids=ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        prediction_positions = torch.as_tensor(
            [width - len(target["sequence"]) + target["predict_position"] for target in flat],
            dtype=torch.long,
            device=device,
        )
        row_indices = torch.arange(len(flat), dtype=torch.long, device=device)
        prediction_logits = result.logits[row_indices, prediction_positions]
        log_probs = functional.log_softmax(prediction_logits.float(), dim=-1)
        token_ids = torch.as_tensor([target["token_id"] for target in flat], dtype=torch.long, device=device)
        values = log_probs.gather(1, token_ids[:, None]).squeeze(1).detach().cpu().numpy()
        for local_index, item in enumerate(batch):
            en_value = float(values[2 * local_index])
            vi_value = float(values[2 * local_index + 1])
            output_rows.append(
                {
                    "concept_id": item["concept_id"],
                    "en_token_id": int(item["targets"][0]["token_id"]),
                    "vi_token_id": int(item["targets"][1]["token_id"]),
                    "en_logprob": en_value,
                    "vi_logprob": vi_value,
                    "value": en_value - vi_value,
                }
            )
    return output_rows


def selected_competent_ids(
    order_rows: Sequence[Mapping[str, Any]],
    competence_rows: Sequence[Mapping[str, Any]],
    family: str,
    max_items: int,
) -> tuple[list[str], dict[str, int]]:
    """Apply the frozen e8_select helper to VI-class translation competence rows."""
    from .derived import e8_select

    competent = {
        str(row["concept_id"])
        for row in competence_rows
        if str(row.get("format")) == "translation" and str(row.get("competence_class")) == "VI"
    }
    selected = e8_select(order_rows, competent, family, int(max_items))
    stratum_by_id = {
        str(row["concept_id"]): str(row["stratum"])
        for row in order_rows
        if str(row["family"]) == family
    }
    quotas: dict[str, int] = {}
    for concept_id in selected:
        stratum = stratum_by_id[concept_id]
        quotas[stratum] = quotas.get(stratum, 0) + 1
    return selected, quotas


def ensure_primary_locks(cfg: dict[str, Any], repo_root: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Enforce both Checkpoint-H and test-item locks before primary model access."""
    from .locks import ensure_adjudication_committed, ensure_model_access_allowed

    ensure_adjudication_committed(repo_root)
    ensure_model_access_allowed(cfg, records)


def run_if_not_preflight(preflight_only: bool, runner: Any) -> Any:
    """Invoke a weight-loading runner only after all non-weight preflight checks pass."""
    if preflight_only:
        return None
    return runner()
