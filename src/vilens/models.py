"""Read-only checkpoint metadata and tokenizer loaders for both candidate sets."""

from __future__ import annotations

import json
import logging
import os
from operator import attrgetter
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class SnapshotError(RuntimeError):
    """Raised when a pinned local model snapshot cannot be resolved or checked."""


@dataclass(frozen=True)
class ModelSpec:
    """Immutable identity of one candidate checkpoint."""

    set_name: Literal["registered", "base"]
    family: Literal["gemma", "qwen", "llama"]
    hf_id: str
    revision: str

    @property
    def label(self) -> str:
        """Return the stable set/family label."""
        return f"{self.set_name}/{self.family}"


@dataclass
class LoadedModel:
    """Resolved model, tokenizer, decoder modules, and text-tower dimensions."""

    model: Any
    tokenizer: Any
    family: str
    spec: ModelSpec | None
    decoder: Any
    layers: list[Any]
    final_norm: Any
    embed_tokens: Any
    lm_head: Any
    n_layers: int
    d_model: int
    vocab_size: int
    softcap: float | None
    module_paths: dict[str, str]
    lens_position_chunk: int = 256


def wrap_model(model: Any, tokenizer: Any, family: str, cfg: dict[str, Any]) -> LoadedModel:
    """Resolve the configured decoder modules and freeze a model for inference."""
    inference = cfg["inference"]
    architectures = inference["architectures"]
    if family not in architectures:
        raise ValueError(f"Unknown model family: {family}")
    expected_name = architectures[family]
    if type(model).__name__ != expected_name:
        raise ValueError(
            f"Model architecture mismatch for {family}: expected {expected_name}, "
            f"got {type(model).__name__}"
        )
    module_paths = dict(inference["module_paths"][family])
    decoder = attrgetter(module_paths["decoder"])(model)
    final_norm = decoder.norm
    embed_tokens = decoder.embed_tokens
    lm_head = attrgetter(module_paths["lm_head"])(model)
    layers = list(decoder.layers)
    text_config = model.config.get_text_config()
    if len(layers) != int(text_config.num_hidden_layers):
        raise ValueError(
            f"Decoder layer count mismatch for {family}: found {len(layers)}, "
            f"config declares {text_config.num_hidden_layers}"
        )
    model.eval()
    model.requires_grad_(False)
    softcap_value = getattr(text_config, "final_logit_softcapping", None)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        family=family,
        spec=None,
        decoder=decoder,
        layers=layers,
        final_norm=final_norm,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        n_layers=len(layers),
        d_model=int(text_config.hidden_size),
        vocab_size=int(text_config.vocab_size),
        softcap=None if softcap_value is None else float(softcap_value),
        module_paths=module_paths,
        lens_position_chunk=int(inference["lens_position_chunk"]),
    )


def _load_from_spec(
    spec: ModelSpec,
    cfg: dict[str, Any],
    *,
    device: Any,
    dtype_override: Any | None = None,
) -> LoadedModel:
    """Load one pinned local model in bf16 or an explicitly requested dtype."""
    import torch
    import transformers

    cache_dir = Path(cfg["paths"]["model_cache"])
    snapshot = resolve_snapshot(spec, cache_dir)
    problems = snapshot_problems(snapshot)
    if problems:
        raise SnapshotError(
            f"Incomplete snapshot for {spec.hf_id}@{spec.revision}: " + "; ".join(problems)
        )
    hf_config = load_hf_config(spec, cache_dir)
    architectures = getattr(hf_config, "architectures", None)
    expected_architecture = cfg["inference"]["architectures"][spec.family]
    if not architectures:
        raise SnapshotError(f"Config for {spec.hf_id}@{spec.revision} has no architecture declaration")
    architecture = architectures[0]
    model_class = getattr(transformers, architecture, None)
    if architecture != expected_architecture or model_class is None:
        raise SnapshotError(
            f"Architecture mismatch for {spec.hf_id}@{spec.revision}: "
            f"config declares {architecture!r}, expected {expected_architecture!r}"
        )
    model_dtype = torch.bfloat16 if dtype_override is None else dtype_override
    model = model_class.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=model_dtype,
        attn_implementation=cfg["inference"]["attn_implementation"],
    ).to(device)
    loaded = wrap_model(model, load_tokenizer(spec, cache_dir), spec.family, cfg)
    loaded.spec = spec
    return loaded


def load(
    cfg: dict[str, Any],
    family: str,
    *,
    device: Any,
    dtype_override: Any | None = None,
) -> LoadedModel:
    """Load the selected study checkpoint, with bf16 default and optional explicit dtype."""
    from .config import study_models

    selected = cfg["models"]["study_set"]
    selected_models = study_models(cfg)
    if family not in selected_models:
        raise ValueError(f"Unknown model family: {family}")
    entry = selected_models[family]
    spec = ModelSpec(selected, family, entry["hf_id"], entry["revision"])
    loaded = _load_from_spec(spec, cfg, device=device, dtype_override=dtype_override)
    expected_dims = cfg["reference_facts"]["x0c"]["model_dims"][family]
    actual_dims = list(text_dims(loaded.model.config))
    if actual_dims != list(expected_dims):
        raise SnapshotError(
            f"Text dimensions for {spec.hf_id}@{spec.revision} are {actual_dims}; "
            f"reference facts require {expected_dims}"
        )
    logging.getLogger(__name__).info(
        "Loaded %s@%s dtype=%s dtype_override=%s device=%s",
        spec.hf_id,
        spec.revision,
        next(loaded.model.parameters()).dtype,
        dtype_override is not None,
        device,
    )
    return loaded


def candidate_specs(cfg: dict[str, Any]) -> list[ModelSpec]:
    """Return registered then base checkpoints in Gemma, Qwen, Llama order."""
    specs: list[ModelSpec] = []
    for set_name in ("registered", "base"):
        for family in ("gemma", "qwen", "llama"):
            model = cfg["models"][set_name][family]
            specs.append(ModelSpec(set_name, family, model["hf_id"], model["revision"]))
    return specs


def resolve_snapshot(spec: ModelSpec, cache_dir: str | Path) -> Path:
    """Resolve a pinned snapshot from the given shared cache without network access."""
    cache_path = Path(cache_dir)
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            spec.hf_id,
            revision=spec.revision,
            cache_dir=str(cache_path),
            local_files_only=True,
        ))
        if snapshot.name != spec.revision:
            raise ValueError(f"resolved snapshot directory is named {snapshot.name!r}")
        return snapshot
    except Exception as exc:
        raise SnapshotError(
            f"Could not resolve {spec.hf_id}@{spec.revision} in cache {cache_path}: {exc}"
        ) from exc


def snapshot_problems(snapshot_dir: str | Path) -> list[str]:
    """Check snapshot files and safetensors headers without reading tensor data."""
    root = Path(snapshot_dir)
    problems: list[str] = []
    if not root.is_dir():
        return [f"snapshot directory is missing: {root}"]

    def stat_size(relative: str) -> int | None:
        path = root / relative
        try:
            info = os.stat(path)
        except OSError:
            problems.append(f"missing or unresolved file: {relative}")
            return None
        if not os.path.isfile(path):
            problems.append(f"not a regular file: {relative}")
            return None
        if info.st_size == 0:
            problems.append(f"zero-size file: {relative}")
            return None
        return info.st_size

    def inspect_weight_file(relative: str, assigned_tensors: set[str] | None) -> None:
        """Validate one weight blob's cache location, safetensors header, and indexed tensors."""
        weight_path = root / relative
        try:
            weight_size = os.stat(weight_path).st_size
        except OSError:
            # stat_size already records the missing or unresolved path.
            return

        blobs = root.parents[1] / "blobs"
        resolved_blobs = Path(os.path.realpath(blobs))
        resolved_weight = Path(os.path.realpath(weight_path))
        try:
            resolved_weight.relative_to(resolved_blobs)
        except ValueError:
            problems.append(f"weight file does not resolve inside blobs/: {relative}")
        if re.fullmatch(r"[0-9a-f]{64}", resolved_weight.name) is None:
            problems.append(f"weight blob name is not a 64-hex sha256: {relative} -> {resolved_weight.name}")

        try:
            with weight_path.open("rb") as stream:
                length_bytes = stream.read(8)
                if len(length_bytes) != 8:
                    problems.append(f"safetensors header is missing its 8-byte length: {relative}")
                    return
                header_length = int.from_bytes(length_bytes, "little")
                if not 0 < header_length <= 100_000_000:
                    problems.append(f"invalid safetensors header length {header_length}: {relative}")
                    return
                header_bytes = stream.read(header_length)
                if len(header_bytes) != header_length:
                    problems.append(f"truncated safetensors header: {relative}")
                    return
            header = json.loads(header_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"cannot read safetensors header for {relative}: {exc}")
            return

        if not isinstance(header, dict):
            problems.append(f"safetensors header is not an object: {relative}")
            return
        tensor_entries = {name: value for name, value in header.items() if name != "__metadata__"}
        tensor_ends: list[int] = []
        for tensor_name, tensor_info in tensor_entries.items():
            offsets = tensor_info.get("data_offsets") if isinstance(tensor_info, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                problems.append(f"invalid data_offsets for tensor {tensor_name!r} in {relative}")
                continue
            tensor_ends.append(offsets[1])
        if not tensor_entries or len(tensor_ends) != len(tensor_entries):
            problems.append(f"safetensors header has no valid tensor offsets: {relative}")
        elif weight_size != 8 + header_length + max(tensor_ends):
            problems.append(
                f"safetensors file size mismatch for {relative}: file has {weight_size} bytes, "
                f"header requires {8 + header_length + max(tensor_ends)}"
            )
        if assigned_tensors is not None:
            missing_tensors = sorted(assigned_tensors.difference(tensor_entries))
            if missing_tensors:
                problems.append(
                    f"index assigns tensors absent from {relative}: {', '.join(missing_tensors[:10])}"
                )

    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        stat_size(required)

    index_path = root / "model.safetensors.index.json"
    try:
        index_info = os.stat(index_path)
        index_exists = os.path.isfile(index_path) and index_info.st_size > 0
        if os.path.isfile(index_path) and index_info.st_size == 0:
            problems.append("zero-size file: model.safetensors.index.json")
    except FileNotFoundError:
        index_exists = False
    except OSError as exc:
        index_exists = False
        problems.append(f"cannot inspect model.safetensors.index.json: {exc}")

    if index_exists:
        try:
            with index_path.open("r", encoding="utf-8") as stream:
                index = json.load(stream)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                problems.append("model.safetensors.index.json has no non-empty weight_map")
            else:
                tensors_by_shard: dict[str, set[str]] = {}
                for tensor_name, shard_name in weight_map.items():
                    if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                        problems.append("model.safetensors.index.json weight_map keys and values must be strings")
                        continue
                    tensors_by_shard.setdefault(shard_name, set()).add(tensor_name)
                for shard_name, tensor_names in sorted(tensors_by_shard.items()):
                    if stat_size(shard_name) is not None:
                        inspect_weight_file(shard_name, tensor_names)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"cannot read model.safetensors.index.json: {exc}")
    else:
        if stat_size("model.safetensors") is not None:
            inspect_weight_file("model.safetensors", None)

    try:
        entries = list(root.rglob("*"))
    except OSError as exc:
        entries = []
        problems.append(f"cannot enumerate snapshot entries: {exc}")
    for entry in entries:
        try:
            os.stat(entry)
        except OSError:
            problems.append(f"dangling or inaccessible snapshot entry: {entry.relative_to(root)}")

    blobs = root.parents[1] / "blobs"
    if blobs.is_dir():
        try:
            for blob in blobs.iterdir():
                if blob.name.endswith(".incomplete"):
                    problems.append(f"incomplete cache blob: {blob.name}")
        except OSError as exc:
            problems.append(f"cannot inspect blobs directory: {exc}")
    return problems


def index_size_info(snapshot_dir: str | Path) -> dict[str, int | None] | None:
    """Return index-reported and on-disk shard sizes for reporting, never gating."""
    root = Path(snapshot_dir)
    index_path = root / "model.safetensors.index.json"
    try:
        if not index_path.is_file():
            return None
        with index_path.open("r", encoding="utf-8") as stream:
            index = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {"index_total_size": None, "shard_bytes_sum": None}

    metadata = index.get("metadata", {}) if isinstance(index, dict) else {}
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if type(total_size) is not int:
        total_size = None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or any(
        not isinstance(shard_name, str) for shard_name in weight_map.values()
    ):
        return {"index_total_size": total_size, "shard_bytes_sum": None}
    try:
        shard_bytes_sum = sum(
            os.stat(root / shard_name).st_size for shard_name in set(weight_map.values())
        )
    except OSError:
        shard_bytes_sum = None
    return {"index_total_size": total_size, "shard_bytes_sum": shard_bytes_sum}


def load_tokenizer(spec: ModelSpec, cache_dir: str | Path) -> Any:
    """Load a fast tokenizer for a complete pinned local snapshot."""
    snapshot = resolve_snapshot(spec, cache_dir)
    problems = snapshot_problems(snapshot)
    if problems:
        raise SnapshotError(
            f"Incomplete snapshot for {spec.hf_id}@{spec.revision} in cache {cache_dir}: "
            + "; ".join(problems)
        )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
            use_fast=True,
        )
    except Exception as exc:
        raise SnapshotError(
            f"Could not load tokenizer for {spec.hf_id}@{spec.revision} in cache {cache_dir}: {exc}"
        ) from exc
    if getattr(tokenizer, "is_fast", False) is not True:
        raise SnapshotError(f"Tokenizer for {spec.hf_id}@{spec.revision} is not fast")
    return tokenizer


def load_hf_config(spec: ModelSpec, cache_dir: str | Path) -> Any:
    """Load only the Hugging Face config from a pinned local snapshot."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except Exception as exc:
        raise SnapshotError(
            f"Could not load config for {spec.hf_id}@{spec.revision} in cache {cache_dir}: {exc}"
        ) from exc


def text_dims(hf_config: Any) -> tuple[int, int, int]:
    """Return text-tower layer count, hidden size, and vocabulary size."""
    config = hf_config.get_text_config()
    return int(config.num_hidden_layers), int(config.hidden_size), int(config.vocab_size)
