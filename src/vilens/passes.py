"""Reusable prompt-only and teacher-forced passes for P-D and P-L."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import hooks, io
from .lens import LensHead, teacher_forced
from .models import LoadedModel
from .provenance import RunGit, build as build_provenance
from .tokenize import (
    JointEncoding,
    TokenClassifier,
    encode_joint,
    encode_joint_batch,
    first_content_index,
)


@dataclass(frozen=True)
class PathForm:
    """One concept or distractor target used as an along-path readout."""

    target_lang: str
    target_variant: str
    ref_concept_id: str
    target: str


@dataclass(frozen=True)
class Item:
    """A prompt, correct answer, matched distractor, and optional path forms."""

    record: dict[str, Any]
    concept_id: str
    target_lang: str
    target_variant: str
    prompt: str
    target: str
    distractor_id: str | None
    distractor_target: str | None
    path_forms: tuple[PathForm, ...] = ()
    correct_encoding: JointEncoding | None = None
    distractor_encoding: JointEncoding | None = None


@dataclass
class PLOutput:
    """Long P-L rows, answer-position summaries, residual caches, and generations."""

    long: pd.DataFrame
    answer_topk: pd.DataFrame
    resid_index: pd.DataFrame
    resid: np.ndarray
    attn: np.ndarray
    mlp: np.ndarray
    generations: pd.DataFrame
    expected_row_count: int


@dataclass(frozen=True)
class PDUnitPlan:
    """Prepared prompt-only result unit, checked before any model forward."""

    unit_key: dict[str, str | int]
    format_name: str
    records: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    expected_row_count: int
    complete: bool
    reason: str


@dataclass(frozen=True)
class PLUnitPlan:
    """Prepared teacher-forced result unit with its cached joint tokenizations."""

    unit_key: dict[str, str | int]
    format_name: str
    items: tuple[Item, ...]
    encodings: tuple[JointEncoding, ...]
    provenance: dict[str, Any]
    expected_row_count: int
    complete: bool
    reason: str


def with_pl_shards(
    records: Sequence[Mapping[str, Any]], shard_prompts: int
) -> list[dict[str, Any]]:
    """Assign P-L shards independently within each format, condition, and few-shot set."""
    if shard_prompts <= 0:
        raise ValueError("pl.shard_prompts must be positive")
    group_positions: dict[tuple[str, str, str], int] = {}
    result: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        group = (
            str(record["format"]),
            str(record.get("condition", "diac")),
            "none" if record.get("fewshot_set") is None else str(record["fewshot_set"]),
        )
        position = group_positions.get(group, 0)
        record["pl_shard"] = position // shard_prompts
        group_positions[group] = position + 1
        result.append(record)
    return result


def pl_unit_group_key(record: Mapping[str, Any]) -> tuple[str, str, str, int | None]:
    """Return the normalized format/condition/few-shot/shard key used by P-L."""
    shard_value = record.get("pl_shard")
    shard = None if shard_value is None else int(shard_value)
    return (
        str(record["format"]),
        str(record.get("condition", "diac")),
        "none" if record.get("fewshot_set") is None else str(record["fewshot_set"]),
        shard,
    )


def pl_unit_key(family: str, record: Mapping[str, Any]) -> dict[str, str | int]:
    """Build the P-L result key from the same normalized prompt fields as its plan."""
    format_name, condition, fewshot_set, shard = pl_unit_group_key(record)
    unit_key: dict[str, str | int] = {
        "family": family,
        "pass": "pl",
        "condition": condition,
        "fewshot_set": fewshot_set,
        "format": format_name,
    }
    if shard is not None:
        unit_key["shard"] = shard
    return unit_key


def _target_index(targets: Sequence[Mapping[str, Any]] | Mapping[Any, Any]) -> dict[tuple[str, str, str], str]:
    """Normalize target-form records into a key-to-string lookup."""
    if isinstance(targets, Mapping):
        result: dict[tuple[str, str, str], str] = {}
        for key, value in targets.items():
            if isinstance(key, tuple) and len(key) == 3:
                target = value.get("target") if isinstance(value, Mapping) else value
                result[(str(key[0]), str(key[1]), str(key[2]))] = str(target)
        return result
    return {
        (str(row["concept_id"]), str(row["target_lang"]), str(row["target_variant"])): str(row["target"])
        for row in targets
    }


def _distractor_index(
    distractors: Sequence[Mapping[str, Any]] | Mapping[Any, Any],
) -> dict[tuple[str, str, str, str], str]:
    """Normalize distractor rows into family/language/variant/concept lookups."""
    if isinstance(distractors, Mapping):
        result: dict[tuple[str, str, str, str], str] = {}
        for key, value in distractors.items():
            if isinstance(key, tuple) and len(key) == 4:
                distractor_id = value.get("distractor_id") if isinstance(value, Mapping) else value
                result[tuple(map(str, key))] = str(distractor_id)
        return result
    return {
        (
            str(row["family"]),
            str(row.get("target_lang", row.get("lang"))),
            str(row.get("target_variant", row.get("variant"))),
            str(row["concept_id"]),
        ): str(row["distractor_id"])
        for row in distractors
    }


def _record_target(record: Mapping[str, Any], lang: str) -> str | None:
    """Return the raw prompt-file target for a language when one is present."""
    if "lang" in record:
        return str(record["target"])
    key = f"target_{lang}"
    value = record.get(key)
    return None if value is None else str(value)


def _variant(record: Mapping[str, Any], lang: str) -> str:
    """Choose the frozen target variant for a prompt language and condition."""
    if lang == "zh":
        return "nospace"
    if lang == "vi" and record.get("condition") == "nodiac":
        return "nodiac"
    return "released"


def build_items(
    cfg: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    distractors: Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    family: str,
) -> list[Item]:
    """Join guarded prompt records to frozen targets, distractors, and path forms."""
    target_map = _target_index(targets)
    distractor_map = _distractor_index(distractors)
    items: list[Item] = []
    for raw_record in records:
        record = dict(raw_record)
        concept_id = str(record["concept_id"])
        prompt = str(record["prompt"])
        langs = [str(record["lang"])] if "lang" in record else [
            lang for lang in ("vi", "en", "zh", "fr", "id") if f"target_{lang}" in record
        ]
        for lang in langs:
            variant = _variant(record, lang)
            target_key = (concept_id, lang, variant)
            if target_key not in target_map:
                raise KeyError(f"Missing target form {target_key}")
            target = target_map[target_key]
            raw_target = _record_target(record, lang)
            if "lang" in record and raw_target is not None:
                if lang == "zh":
                    if not raw_target.startswith(" ") or raw_target[1:].startswith(" "):
                        raise ValueError(f"Directions zh target must have one leading space: {concept_id}")
                    comparison_target = raw_target[1:]
                else:
                    comparison_target = raw_target
                if comparison_target != target:
                    raise ValueError(
                        f"Directions target mismatch for {concept_id}/{lang}: "
                        f"manifest={target!r}, record={comparison_target!r}"
                    )
            distractor_key = (family, lang, variant, concept_id)
            if distractor_key not in distractor_map:
                raise KeyError(f"Missing distractor row {distractor_key}")
            distractor_id = distractor_map[distractor_key]
            distractor_target_key = (distractor_id, lang, variant)
            if distractor_target_key not in target_map:
                raise KeyError(f"Missing distractor target form {distractor_target_key}")
            path_forms: list[PathForm] = []
            if lang == "vi":
                for path_lang, path_variant in (
                    ("en", "released"), ("zh", "nospace"), ("fr", "released"), ("id", "released")
                ):
                    path_target_key = (concept_id, path_lang, path_variant)
                    if path_target_key not in target_map:
                        raise KeyError(f"Missing path target form {path_target_key}")
                    path_distractor_key = (family, path_lang, path_variant, concept_id)
                    if path_distractor_key not in distractor_map:
                        raise KeyError(f"Missing path distractor row {path_distractor_key}")
                    path_ref = distractor_map[path_distractor_key]
                    path_distractor_target_key = (path_ref, path_lang, path_variant)
                    if path_distractor_target_key not in target_map:
                        raise KeyError(f"Missing path distractor target form {path_distractor_target_key}")
                    path_forms.extend(
                        (
                            PathForm(path_lang, path_variant, concept_id, target_map[path_target_key]),
                            PathForm(
                                path_lang,
                                path_variant,
                                path_ref,
                                target_map[path_distractor_target_key],
                            ),
                        )
                    )
            items.append(
                Item(
                    record=record,
                    concept_id=concept_id,
                    target_lang=lang,
                    target_variant=variant,
                    prompt=prompt,
                    target=target,
                    distractor_id=distractor_id,
                    distractor_target=target_map[distractor_target_key],
                    path_forms=tuple(path_forms),
                )
            )
    return items


def build_pl_items(
    cfg: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    distractors: Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    family: str,
    mode: str,
) -> list[Item]:
    """Expand prompt records into validated correct, distractor, and path readouts."""
    from .derived import target_string
    from .config import study_models
    from .models import ModelSpec, load_tokenizer

    if mode not in {"primary", "trial", "r1", "e9"}:
        raise ValueError("mode must be primary, trial, r1, or e9")
    if family not in {"gemma", "qwen", "llama"}:
        raise ValueError(f"Unknown model family: {family}")
    concept_map = (
        {str(key): value for key, value in concepts.items()}
        if isinstance(concepts, Mapping)
        else {str(row["concept_id"]): row for row in concepts}
    )
    target_map = _target_index(targets)
    distractor_map = _distractor_index(distractors)
    model = study_models(cfg)[family]
    spec = ModelSpec(str(cfg["models"]["study_set"]), family, str(model["hf_id"]), str(model["revision"]))
    tokenizer = load_tokenizer(spec, Path(cfg["paths"]["model_cache"]))
    classifier = TokenClassifier(tokenizer)
    readout_cfg = cfg["pl"]["r1"] if mode == "r1" else cfg["pl"]["e9"] if mode == "e9" else cfg["pl"]
    correct_readouts = [tuple(map(str, pair)) for pair in readout_cfg["correct_readouts"]]
    translation_extra = [tuple(map(str, pair)) for pair in readout_cfg["translation_extra_readouts"]]
    path_langs = [tuple(map(str, pair)) for pair in readout_cfg["path_langs"]]
    trial_distractor_combos = {tuple(map(str, pair)) for pair in cfg["manifests"]["e0_combos"]}
    pending: list[tuple[Item, str, str]] = []
    items: list[Item] = []

    for raw in records:
        record = dict(raw)
        concept_id = str(record["concept_id"])
        if concept_id not in concept_map:
            raise KeyError(f"Missing concept row {concept_id}")
        concept = concept_map[concept_id]
        prompt = str(record["prompt"])
        if prompt != prompt.rstrip():
            raise ValueError(f"Prompt ends in whitespace: {concept_id}")
        if mode == "primary":
            if record.get("split") != "test":
                raise ValueError(f"Primary P-L requires test records: {concept_id}")
            if record.get("condition") != cfg["pl"]["primary"]["condition"]:
                raise ValueError(f"Primary P-L condition differs for {concept_id}")
            if str(record.get("format")) == "cloze":
                if record.get("fewshot_set") is not None:
                    raise ValueError(f"Primary P-L cloze few-shot set must be null: {concept_id}")
            elif int(record.get("fewshot_set", -1)) != int(cfg["pl"]["primary"]["fewshot_set"]):
                raise ValueError(f"Primary P-L few-shot set differs for {concept_id}")
        elif mode == "trial":
            if record.get("split") not in {"fit", "validate"} or str(record.get("lang")) != "vi":
                raise ValueError("Trial P-L accepts only directions-split Vietnamese records")
        elif mode == "r1":
            r1 = cfg["pl"]["r1"]
            if record.get("split") != "test":
                raise ValueError(f"R1 P-L requires test records: {concept_id}")
            if record.get("condition") != r1["condition"]:
                raise ValueError(f"R1 P-L condition differs for {concept_id}")
            if int(record.get("fewshot_set", -1)) not in {int(value) for value in r1["fewshot_sets"]}:
                raise ValueError(f"R1 P-L few-shot set differs for {concept_id}")
        else:
            e9 = cfg["pl"]["e9"]
            if record.get("split") != "test":
                raise ValueError(f"E9 P-L requires test records: {concept_id}")
            if record.get("condition") != e9["condition"]:
                raise ValueError(f"E9 P-L condition differs for {concept_id}")
            if str(record.get("format")) == "cloze":
                if record.get("fewshot_set") is not None:
                    raise ValueError(f"E9 P-L cloze few-shot set must be null: {concept_id}")
            elif int(record.get("fewshot_set", -1)) != int(e9["fewshot_set"]):
                raise ValueError(f"E9 P-L few-shot set differs for {concept_id}")

        raw_extension = concept.get("m1_extension", record.get("m1_extension", False))
        m1_extension = (
            raw_extension is True
            or (isinstance(raw_extension, str) and raw_extension.lower() == "true")
            or (type(raw_extension) in (int, np.integer) and int(raw_extension) == 1)
        )
        record["m1_extension"] = m1_extension
        if mode in {"r1", "e9"} and m1_extension:
            raise ValueError(f"{mode.upper()} P-L excludes M1 extension records")
        readouts = list(correct_readouts)
        if str(record.get("format")) == "translation":
            readouts.extend(translation_extra)

        for lang, variant in readouts:
            if lang == "ru":
                target = target_string(concept, lang, variant)
            else:
                target_key = (concept_id, lang, variant)
                if target_key not in target_map:
                    raise KeyError(f"Missing target form {target_key}")
                target = target_map[target_key]
                expected_target = target_string(concept, lang, variant)
                if target != expected_target:
                    raise ValueError(f"Target form differs from frozen helper for {target_key}")

            dist_id: str | None = None
            dist_target: str | None = None
            combo = (lang, variant)
            if not m1_extension and lang != "ru":
                dist_key = (family, lang, variant, concept_id)
                needs_dist = mode in {"primary", "r1", "e9"} or combo in trial_distractor_combos
                if needs_dist:
                    if dist_key not in distractor_map:
                        raise KeyError(f"Missing main-concept distractor {dist_key}")
                    dist_id = distractor_map[dist_key]
                    dist_target_key = (dist_id, lang, variant)
                    if dist_target_key not in target_map:
                        raise KeyError(f"Missing distractor target form {dist_target_key}")
                    dist_target = target_map[dist_target_key]

            forms: list[PathForm] = []
            if (lang, variant) == ("vi", "released"):
                for path_lang, path_variant in path_langs:
                    path_key = (concept_id, path_lang, path_variant)
                    if path_key not in target_map:
                        raise KeyError(f"Missing path target form {path_key}")
                    forms.append(PathForm(path_lang, path_variant, concept_id, target_map[path_key]))
                    path_dist_key = (family, path_lang, path_variant, concept_id)
                    path_dist_id = distractor_map.get(path_dist_key)
                    if path_dist_id is None and mode == "primary" and not m1_extension:
                        raise KeyError(f"Missing path distractor row {path_dist_key}")
                    if path_dist_id is not None:
                        path_dist_form = (path_dist_id, path_lang, path_variant)
                        if path_dist_form not in target_map:
                            raise KeyError(f"Missing path distractor target form {path_dist_form}")
                        forms.append(
                            PathForm(path_lang, path_variant, path_dist_id, target_map[path_dist_form])
                        )

            item = Item(
                record=record,
                concept_id=concept_id,
                target_lang=lang,
                target_variant=variant,
                prompt=prompt,
                target=target,
                distractor_id=dist_id,
                distractor_target=dist_target,
                path_forms=tuple(forms),
            )
            items.append(item)
            pending.append((item, prompt, target))
            if dist_target is not None:
                pending.append((item, prompt, dist_target))

    encodings = encode_joint_batch(
        tokenizer,
        [prompt for _item, prompt, _target in pending],
        [target for _item, _prompt, target in pending],
        strict=True,
        prompt_cache={},
    )
    per_item: dict[int, list[JointEncoding]] = {}
    for (item, _prompt, _target), encoding in zip(pending, encodings, strict=True):
        if not any(classifier.is_content(token_id) for token_id in encoding.target_ids):
            raise ValueError(f"Target has no content token: {item.concept_id}/{item.target_lang}")
        per_item.setdefault(id(item), []).append(encoding)
    return [
        replace(
            item,
            correct_encoding=per_item[id(item)][0],
            distractor_encoding=(per_item[id(item)][1] if len(per_item[id(item)]) > 1 else None),
        )
        for item in items
    ]


def _pad_id(loaded: LoadedModel) -> int:
    """Choose an existing tokenizer pad/eos id for masked right padding."""
    tokenizer = loaded.tokenizer
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    return 0


def _encode_prompts(loaded: LoadedModel, prompts: Sequence[str]) -> list[tuple[int, ...]]:
    """Encode prompt-only sequences with the tokenizer's default special tokens."""
    encoded: list[tuple[int, ...]] = []
    for prompt in prompts:
        ids = loaded.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not ids:
            raise ValueError("A prompt encoded to an empty token sequence")
        encoded.append(tuple(int(token_id) for token_id in ids))
    return encoded


def _pad_sequences(
    sequences: Sequence[Sequence[int]], pad_id: int, device: Any
) -> tuple[Any, Any, list[int]]:
    """Right-pad token id sequences and return ids, attention mask, and lengths."""
    import torch

    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths, default=0)
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        length = lengths[index]
        input_ids[index, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
    return input_ids, attention_mask, lengths


def run_pd(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Forward prompt-only batches and retain final-prompt-position residuals."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(records, key=lambda row: (str(row["format"]), str(row.get("lang", "")), str(row["concept_id"])))
    device = next(loaded.model.parameters()).device
    index_rows: list[dict[str, Any]] = []
    residual_rows: list[np.ndarray] = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        sequences = _encode_prompts(loaded, [str(record["prompt"]) for record in batch])
        input_ids, attention_mask, lengths = _pad_sequences(sequences, _pad_id(loaded), device)
        positions = torch.as_tensor([[length - 1] for length in lengths], dtype=torch.long, device=device)
        recording = hooks.record(
            loaded,
            input_ids,
            attention_mask,
            positions=positions,
            components=False,
            keep_on="cpu",
        )
        values = recording.resid[:, :, 0, :].permute(1, 0, 2).float().numpy()
        for record, residual, sequence in zip(batch, values, sequences, strict=True):
            special_mask = loaded.tokenizer.get_special_tokens_mask(
                list(sequence),
                already_has_special_tokens=True,
            )
            n_special = int(sum(bool(value) for value in special_mask))
            index_rows.append(
                {
                    "format": str(record["format"]),
                    "lang": str(record.get("lang", "")),
                    "concept_id": str(record["concept_id"]),
                    "split": str(record.get("split", "")),
                    "condition": str(record.get("condition", "diac")),
                    "fewshot_set": record.get("fewshot_set"),
                    "n_tokens": len(sequence) - n_special,
                    "n_special": n_special,
                }
            )
            residual_rows.append(residual)
    index = pd.DataFrame(index_rows)
    if not index.empty:
        index["array_row"] = np.arange(len(index), dtype=np.int64)
    resid = np.stack(residual_rows).astype(np.float32, copy=False) if residual_rows else np.empty(
        (0, loaded.n_layers + 1, loaded.d_model), dtype=np.float32
    )
    return index, resid


def run_pd_flores(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Capture final-sentence-token and non-special-token-mean residuals for FLORES records."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(records, key=lambda row: (str(row["lang"]), str(row["sentence_id"])))
    expected_specials = 0 if loaded.family == "qwen" else 1
    device = next(loaded.model.parameters()).device
    index_rows: list[dict[str, Any]] = []
    last_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []

    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        encoded_batch = loaded.tokenizer(
            [str(record["sentence"]) for record in batch],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        sequences = encoded_batch["input_ids"]
        if len(sequences) != len(batch) or any(not sequence for sequence in sequences):
            raise ValueError("FLORES tokenizer returned an empty or misaligned batch")
        content_positions: list[list[int]] = []
        counts: list[tuple[int, int]] = []
        for record, raw_ids in zip(batch, sequences, strict=True):
            ids = [int(token_id) for token_id in raw_ids]
            special_mask = loaded.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            if len(special_mask) != len(ids):
                raise ValueError("Tokenizer special-token mask length differs from encoded FLORES input")
            special_positions = [index for index, value in enumerate(special_mask) if bool(value)]
            token_positions = [index for index, value in enumerate(special_mask) if not bool(value)]
            n_special = len(special_positions)
            n_tokens = len(token_positions)
            if n_special != expected_specials:
                raise AssertionError(
                    f"{loaded.family} tokenizer added {n_special} specials; expected {expected_specials}"
                )
            if n_tokens < 1:
                raise ValueError(f"FLORES record {record['sentence_id']} has no non-special tokens")
            content_positions.append(token_positions)
            counts.append((n_tokens, n_special))

        input_ids, attention_mask, _lengths = _pad_sequences(
            sequences, _pad_id(loaded), device
        )
        max_tokens = max(len(positions) for positions in content_positions)
        selected_positions = torch.empty((len(batch), max_tokens), dtype=torch.long, device=device)
        for row, positions in enumerate(content_positions):
            selected_positions[row, : len(positions)] = torch.as_tensor(
                positions, dtype=torch.long, device=device
            )
            selected_positions[row, len(positions) :] = positions[-1]
        recording = hooks.record(
            loaded,
            input_ids,
            attention_mask,
            positions=selected_positions,
            components=False,
            keep_on="cpu",
        )
        # [layers, batch, selected positions, width] -> [batch, layers, positions, width]
        residuals = recording.resid.permute(1, 0, 2, 3).float().numpy()
        for local_row, (record, positions, (n_tokens, n_special)) in enumerate(
            zip(batch, content_positions, counts, strict=True)
        ):
            last_rows.append(residuals[local_row, :, len(positions) - 1, :].copy())
            mean_rows.append(
                residuals[local_row, :, : len(positions), :].mean(axis=1, dtype=np.float32)
            )
            index_rows.append(
                {
                    "row": start + local_row,
                    "sentence_id": str(record["sentence_id"]),
                    "lang": str(record["lang"]),
                    "format": "flores",
                    "split": str(record["split"]),
                    "n_tokens": n_tokens,
                    "n_special": n_special,
                }
            )

    index = pd.DataFrame(
        index_rows,
        columns=["row", "sentence_id", "lang", "format", "split", "n_tokens", "n_special"],
    )
    empty = np.empty((0, loaded.n_layers + 1, loaded.d_model), dtype=np.float32)
    last = np.stack(last_rows).astype(np.float32, copy=False) if last_rows else empty.copy()
    mean = np.stack(mean_rows).astype(np.float32, copy=False) if mean_rows else empty.copy()
    return index, last, mean


def run_pd_storage_unit(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    source: str,
    format_name: str | None,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> dict[str, Any]:
    """Resume-check, run, and atomically persist one matched or FLORES P-D unit."""
    if source not in {"matched", "flores"}:
        raise ValueError("source must be matched or flores")
    if (source == "matched") != (format_name is not None):
        raise ValueError("matched units require a format; FLORES units must not have one")
    unit_key: dict[str, str | int] = {
        "family": loaded.family,
        "pass": "pd",
        "source": source,
    }
    if format_name is not None:
        unit_key["format"] = format_name
    expected_rows = len(records)
    provenance = _unit_provenance(
        loaded, cfg, repo_root, entrypoint, unit_key, batch_size, expected_rows, run_git
    )
    parquet_path, sidecar_path = io.unit_paths(results_dir, unit_key)
    suffixes = (".resid.npy",) if source == "matched" else (".last.npy", ".mean.npy")
    array_paths = [parquet_path.with_suffix(suffix) for suffix in suffixes]
    complete, reason = io.unit_is_complete(
        results_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"]
    )
    if complete and all(path.is_file() for path in array_paths):
        return _summarize_pd_storage_unit(
            "skipped", parquet_path, sidecar_path, array_paths, expected_rows
        )

    # A stale marker cannot certify arrays that this invocation is about to replace.
    sidecar_path.unlink(missing_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source == "matched":
            index, residual = run_pd(loaded, records, batch_size=batch_size)
            arrays = {"resid": residual}
            if index.empty:
                raise ValueError("Matched P-D unit has no records")
            index = index.copy()
            index["row"] = index["array_row"].astype(int)
            columns = ["row", "concept_id", "lang", "format", "split", "n_tokens", "n_special"]
            sort_by = ["format", "lang", "concept_id"]
        else:
            index, last, mean = run_pd_flores(loaded, records, batch_size=batch_size)
            arrays = {"last": last, "mean": mean}
            columns = ["row", "sentence_id", "lang", "format", "split", "n_tokens", "n_special"]
            sort_by = ["lang", "sentence_id"]
        if len(index) != expected_rows:
            raise AssertionError(f"index rows={len(index)} differs from expected record rows={expected_rows}")
        nonfinite_count = sum(int((~np.isfinite(array)).sum()) for array in arrays.values())
        if nonfinite_count:
            raise FloatingPointError(f"P-D unit contains {nonfinite_count} non-finite activation values")
        for path, array_name in zip(array_paths, arrays, strict=True):
            _atomic_npy(path, arrays[array_name])
        io.write_unit(
            index.loc[:, columns],
            results_dir,
            provenance,
            columns=columns,
            sort_by=sort_by,
            compression=str(cfg["runtime"]["parquet_compression"]),
        )
        return _summarize_pd_storage_unit(
            "ran", parquet_path, sidecar_path, array_paths, expected_rows
        )
    except Exception:
        sidecar_path.unlink(missing_ok=True)
        parquet_path.unlink(missing_ok=True)
        for path in array_paths:
            path.unlink(missing_ok=True)
        raise


def _summarize_pd_storage_unit(
    status: str,
    parquet_path: Path,
    sidecar_path: Path,
    array_paths: Sequence[Path],
    rows: int,
) -> dict[str, Any]:
    """Summarize one persisted unit without exposing any input text."""
    arrays: dict[str, np.ndarray] = {
        path.name.rsplit(".", 2)[-2]: np.load(path, mmap_mode="r", allow_pickle=False)
        for path in array_paths
    }
    shapes = {name: list(array.shape) for name, array in arrays.items()}
    norm_layers = sorted({0, *(array.shape[1] // 2 for array in arrays.values()), *(array.shape[1] - 1 for array in arrays.values())})
    mean_norms = {
        name: {
            str(layer): float(np.linalg.norm(array[:, layer, :], axis=-1).mean())
            for layer in norm_layers
            if array.shape[1] > layer
        }
        for name, array in arrays.items()
    }
    paths = [parquet_path, sidecar_path, *array_paths]
    nonfinite_count = sum(int((~np.isfinite(array)).sum()) for array in arrays.values())
    return {
        "status": status,
        "rows": rows,
        "shapes": shapes,
        "bytes_on_disk": sum(path.stat().st_size for path in paths if path.is_file()),
        "nonfinite_count": nonfinite_count,
        "mean_residual_norms": mean_norms,
    }


def _long_row(
    loaded: LoadedModel,
    item: Item,
    *,
    role: str,
    ref_concept_id: str,
    target_lang: str,
    target_variant: str,
    tok_idx: int,
    token_id: int,
    is_content: bool,
    layer: int,
    metric: str,
    value: float,
) -> dict[str, Any]:
    """Construct one canonical rule-38 measurement row."""
    spec = loaded.spec
    return {
        "model": spec.hf_id if spec is not None else loaded.family,
        "model_revision": spec.revision if spec is not None else "synthetic",
        "pass": "pl",
        "format": str(item.record["format"]),
        "condition": str(item.record.get("condition", "diac")),
        "fewshot_set": item.record.get("fewshot_set"),
        "concept_id": item.concept_id,
        "target_lang": target_lang,
        "target_variant": target_variant,
        "role": role,
        "ref_concept_id": ref_concept_id,
        "tok_idx": int(tok_idx),
        "token_id": int(token_id),
        "is_content": bool(is_content),
        "layer": int(layer),
        "metric": metric,
        "value": float(value),
    }


def _truncate_generation(text: str) -> str:
    """Truncate decoded output at the first registered newline or space-hyphen stop."""
    stops = [position for marker in ("\n", " -") if (position := text.find(marker)) >= 0]
    return text[: min(stops)] if stops else text


def _score_path_forms(
    loaded: LoadedModel,
    lens: LensHead,
    items: Sequence[Item],
    correct_encodings: Sequence[JointEncoding],
    classifier: TokenClassifier,
    *,
    pad_id: int,
) -> list[dict[str, Any]]:
    """Score path-form first tokens at every correct Vietnamese target position."""
    import torch

    vi_indices = [index for index, item in enumerate(items) if item.target_lang == "vi" and item.path_forms]
    if not vi_indices:
        return []
    vi_items = [items[index] for index in vi_indices]
    vi_encodings = [correct_encodings[index] for index in vi_indices]
    token_counts = [len(encoding.target_ids) for encoding in vi_encodings]
    score_positions = [
        tuple(len(encoding.prompt_ids) - 1 + offset for offset in range(count))
        for encoding, count in zip(vi_encodings, token_counts, strict=True)
    ]
    max_targets = max(token_counts)
    positions = torch.empty((len(vi_items), max_targets), dtype=torch.long)
    for row, row_positions in enumerate(score_positions):
        positions[row, : len(row_positions)] = torch.as_tensor(row_positions, dtype=torch.long)
        positions[row, len(row_positions) :] = row_positions[0]
    sequences = [encoding.joint_ids for encoding in vi_encodings]
    device = next(loaded.model.parameters()).device
    input_ids, attention_mask, _ = _pad_sequences(sequences, pad_id, device)
    recording = hooks.record(
        loaded,
        input_ids,
        attention_mask,
        positions=positions.to(device),
        components=False,
        keep_on="cpu",
    )

    flat_path_pairs: list[tuple[int, PathForm, int]] = []
    flat_prompts: list[str] = []
    flat_targets: list[str] = []
    for item_index, item in enumerate(vi_items):
        for path_form in item.path_forms:
            flat_path_pairs.append((item_index, path_form, -1))
            flat_prompts.append(item.prompt)
            flat_targets.append(path_form.target)
    path_encodings = encode_joint_batch(
        loaded.tokenizer, flat_prompts, flat_targets, strict=True, prompt_cache={}
    )
    path_ids: list[int] = []
    for encoding in path_encodings:
        first = first_content_index(encoding.target_ids, classifier)
        if first is None:
            raise ValueError("A path form has no content token")
        path_ids.append(int(encoding.target_ids[first]))

    layer_count = loaded.n_layers + 1
    layer_logprobs: list[Any] = []
    for layer in range(layer_count):
        hidden_parts: list[Any] = []
        id_parts: list[Any] = []
        for item_index, item in enumerate(vi_items):
            count = token_counts[item_index]
            hidden = recording.resid[layer, item_index, :count]
            hidden_parts.append(hidden.repeat_interleave(len(item.path_forms), dim=0))
            item_ids = [
                path_ids[index]
                for index, pair in enumerate(flat_path_pairs)
                if pair[0] == item_index
            ]
            id_parts.append(
                torch.as_tensor(item_ids, dtype=torch.long).repeat(count, 1).reshape(count * len(item_ids), 1)
            )
        if not hidden_parts:
            continue
        hidden_all = torch.cat(hidden_parts, dim=0)
        ids_all = torch.cat(id_parts, dim=0)
        layer_logprobs.append(lens.logprobs_for(hidden_all, ids_all).squeeze(-1))
    logprobs_by_layer = torch.stack(layer_logprobs, dim=0).detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        logprobs = logprobs_by_layer[layer]
        cursor = 0
        for item_index, item in enumerate(vi_items):
            count = token_counts[item_index]
            width = len(item.path_forms)
            values = logprobs[cursor : cursor + count * width].reshape(count, width)
            cursor += count * width
            for token_position in range(count):
                for path_index, path_form in enumerate(item.path_forms):
                    token_id = path_ids[
                        next(
                            index
                            for index, pair in enumerate(flat_path_pairs)
                            if pair[0] == item_index and pair[1] == path_form
                        )
                    ]
                    rows.append(
                        _long_row(
                            loaded,
                            item,
                            role="path",
                            ref_concept_id=path_form.ref_concept_id,
                            target_lang=path_form.target_lang,
                            target_variant=path_form.target_variant,
                            tok_idx=token_position,
                            token_id=token_id,
                            is_content=True,
                            layer=layer,
                            metric="lens_logprob",
                            value=float(values[token_position, path_index]),
                        )
                    )
    return rows


def _run_pl_reference(
    loaded: LoadedModel,
    lens: LensHead,
    items: Sequence[Item],
    *,
    batch_size: int,
    top_k: int,
    gen_max_new_tokens: int,
    encodings: Sequence[JointEncoding] | None = None,
) -> PLOutput:
    """Retain the original P-L implementation as an exactness reference for CPU tests."""
    import torch
    import torch.nn.functional as functional

    if batch_size <= 0 or top_k <= 0 or gen_max_new_tokens <= 0:
        raise ValueError("batch_size, top_k, and gen_max_new_tokens must be positive")
    classifier = TokenClassifier(loaded.tokenizer)
    prompt_cache: dict[str, tuple[int, ...]] = {}
    lens_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    residual_rows: list[np.ndarray] = []
    attn_rows: list[np.ndarray] = []
    mlp_rows: list[np.ndarray] = []
    generation_rows: list[dict[str, Any]] = []
    device = next(loaded.model.parameters()).device
    pad_id = _pad_id(loaded)
    all_prompts = [item.prompt for item in items for _ in (0, 1)]
    all_targets = [target for item in items for target in (item.target, item.distractor_target)]
    if encodings is None:
        all_encodings = encode_joint_batch(
            loaded.tokenizer,
            all_prompts,
            all_targets,
            strict=True,
            prompt_cache=prompt_cache,
        )
    else:
        all_encodings = list(encodings)
        if len(all_encodings) != len(all_prompts):
            raise ValueError(
                f"Expected {len(all_prompts)} precomputed joint encodings, got {len(all_encodings)}"
            )
    expected_row_count = sum(
        (len(all_encodings[2 * index].target_ids) + len(all_encodings[2 * index + 1].target_ids))
        * (loaded.n_layers + 2)
        for index in range(len(items))
    )
    expected_row_count += sum(
        len(all_encodings[2 * index].target_ids) * (loaded.n_layers + 1) * len(item.path_forms)
        for index, item in enumerate(items)
        if item.target_lang == "vi" and item.path_forms
    )
    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        encodings = all_encodings[2 * start : 2 * (start + len(batch))]
        tf_results = teacher_forced(loaded, lens, encodings, pad_id=pad_id)
        correct_encodings = encodings[::2]
        for item_index, item in enumerate(batch):
            for role_index, role, ref_id in (
                (2 * item_index, "correct", item.concept_id),
                (2 * item_index + 1, "distractor", item.distractor_id),
            ):
                result = tf_results[role_index]
                encoding = encodings[role_index]
                content = [classifier.is_content(token_id) for token_id in result.target_ids]
                for token_index, token_id in enumerate(result.target_ids):
                    for layer in range(loaded.n_layers + 1):
                        lens_rows.append(
                            _long_row(
                                loaded,
                                item,
                                role=role,
                                ref_concept_id=ref_id,
                                target_lang=item.target_lang,
                                target_variant=item.target_variant,
                                tok_idx=token_index,
                                token_id=int(token_id),
                                is_content=content[token_index],
                                layer=layer,
                                metric="lens_logprob",
                                value=float(result.lens_logprobs[layer, token_index]),
                            )
                        )
                    lens_rows.append(
                        _long_row(
                            loaded,
                            item,
                            role=role,
                            ref_concept_id=ref_id,
                            target_lang=item.target_lang,
                            target_variant=item.target_variant,
                            tok_idx=token_index,
                            token_id=int(token_id),
                            is_content=content[token_index],
                            layer=loaded.n_layers,
                            metric="model_logprob",
                            value=float(result.model_logprobs[token_index]),
                        )
                    )

        prompt_sequences = _encode_prompts(loaded, [item.prompt for item in batch])
        prompt_ids, prompt_mask, prompt_lengths = _pad_sequences(prompt_sequences, pad_id, device)
        prompt_positions = torch.as_tensor(
            [[length - 1] for length in prompt_lengths], dtype=torch.long, device=device
        )
        prompt_recording = hooks.record(
            loaded,
            prompt_ids,
            prompt_mask,
            positions=prompt_positions,
            components=True,
            keep_on="cpu",
        )
        if prompt_recording.attn is None or prompt_recording.mlp is None:
            raise RuntimeError("component recording did not return attention and MLP activations")
        resid_batch = prompt_recording.resid[:, :, 0, :].permute(1, 0, 2).float().numpy()
        attn_batch = prompt_recording.attn[:, :, 0, :].permute(1, 0, 2).float().numpy()
        mlp_batch = prompt_recording.mlp[:, :, 0, :].permute(1, 0, 2).float().numpy()
        for item_index, item in enumerate(batch):
            array_row = len(index_rows)
            index_rows.append(
                {
                    "format": str(item.record["format"]),
                    "lang": item.target_lang,
                    "concept_id": item.concept_id,
                    "split": str(item.record.get("split", "")),
                    "array_row": array_row,
                }
            )
            residual_rows.append(resid_batch[item_index])
            attn_rows.append(attn_batch[item_index])
            mlp_rows.append(mlp_batch[item_index])
            for layer in range(loaded.n_layers + 1):
                hidden = torch.as_tensor(resid_batch[item_index, layer], dtype=torch.float32)
                logits = lens.logits(hidden.unsqueeze(0)).float()
                log_probs = functional.log_softmax(logits, dim=-1)[0]
                values, token_ids = torch.topk(log_probs, min(top_k, loaded.vocab_size))
                for rank, (token_id, value) in enumerate(zip(token_ids.tolist(), values.tolist(), strict=True), start=1):
                    topk_rows.append(
                        {
                            "model": loaded.spec.hf_id if loaded.spec else loaded.family,
                            "model_revision": loaded.spec.revision if loaded.spec else "synthetic",
                            "pass": "pl",
                            "format": str(item.record["format"]),
                            "condition": str(item.record.get("condition", "diac")),
                            "fewshot_set": item.record.get("fewshot_set"),
                            "concept_id": item.concept_id,
                            "target_lang": item.target_lang,
                            "target_variant": item.target_variant,
                            "layer": layer,
                            "rank": rank,
                            "token_id": int(token_id),
                            "value": float(value),
                        }
                    )

        path_rows = _score_path_forms(
            loaded,
            lens,
            batch,
            correct_encodings,
            classifier,
            pad_id=pad_id,
        )
        lens_rows.extend(path_rows)

        for item, sequence in zip(batch, prompt_sequences, strict=True):
            input_ids = torch.as_tensor([sequence], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = loaded.model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=gen_max_new_tokens,
                )
            generated_ids = generated[0, len(sequence) :].detach().cpu().tolist()
            raw_text = loaded.tokenizer.decode(generated_ids)
            generation_rows.append(
                {
                    "format": str(item.record["format"]),
                    "lang": item.target_lang,
                    "concept_id": item.concept_id,
                    "token_ids": [int(token_id) for token_id in generated_ids],
                    "text_raw": raw_text,
                    "text": _truncate_generation(raw_text),
                }
            )

    long_df = pd.DataFrame(lens_rows, columns=io.LENS_COLUMNS)
    io.validate_long(long_df, io.LENS_COLUMNS)
    actual_row_count = len(long_df)
    if actual_row_count != expected_row_count:
        raise AssertionError(f"P-L row count {actual_row_count} differs from expected {expected_row_count}")
    empty_resid = np.empty((0, loaded.n_layers + 1, loaded.d_model), dtype=np.float32)
    empty_components = np.empty((0, loaded.n_layers, loaded.d_model), dtype=np.float32)
    return PLOutput(
        long=long_df,
        answer_topk=pd.DataFrame(topk_rows),
        resid_index=pd.DataFrame(index_rows),
        resid=np.stack(residual_rows).astype(np.float32, copy=False) if residual_rows else empty_resid,
        attn=np.stack(attn_rows).astype(np.float32, copy=False) if attn_rows else empty_components,
        mlp=np.stack(mlp_rows).astype(np.float32, copy=False) if mlp_rows else empty_components.copy(),
        generations=pd.DataFrame(generation_rows),
        expected_row_count=expected_row_count,
    )


def _sequence_specs(
    items: Sequence[Item],
    tokenizer: Any,
    supplied: Sequence[JointEncoding] | None,
) -> list[tuple[int, Item, str, str, JointEncoding]]:
    """Return each correct/distractor sequence in stable item order."""
    raw: list[tuple[int, Item, str, str, str, JointEncoding | None]] = []
    for index, item in enumerate(items):
        raw.append((index, item, "correct", item.concept_id, item.target, item.correct_encoding))
        if item.distractor_target is not None:
            if item.distractor_id is None:
                raise ValueError("A distractor target requires a distractor concept id")
            raw.append(
                (index, item, "distractor", item.distractor_id, item.distractor_target,
                 item.distractor_encoding)
            )
        elif item.distractor_id is not None:
            raise ValueError("A distractor concept id requires a distractor target")
    if supplied is not None:
        if len(supplied) != len(raw):
            raise ValueError(f"Expected {len(raw)} precomputed encodings, got {len(supplied)}")
        encodings = list(supplied)
    elif all(entry[-1] is not None for entry in raw):
        encodings = [entry[-1] for entry in raw]
    else:
        missing_indices = [index for index, entry in enumerate(raw) if entry[-1] is None]
        encoded = encode_joint_batch(
            tokenizer,
            [raw[index][1].prompt for index in missing_indices],
            [raw[index][4] for index in missing_indices],
            strict=True,
            prompt_cache={},
        )
        encodings = [entry[-1] for entry in raw]
        for index, encoding in zip(missing_indices, encoded, strict=True):
            encodings[index] = encoding
    return [
        (index, item, role, ref_id, encoding)
        for (index, item, role, ref_id, _target, _cached), encoding in zip(raw, encodings, strict=True)
        if encoding is not None
    ]


def _prompt_record_key(item: Item) -> tuple[str, ...]:
    """Return the stable identity of one prompt record within a P-L unit."""
    record = item.record
    return (
        str(record.get("format", "")),
        str(record.get("condition", "diac")),
        "" if record.get("fewshot_set") is None else str(record["fewshot_set"]),
        "" if record.get("pl_shard") is None else str(record["pl_shard"]),
        str(record.get("lang", "")),
        item.concept_id,
        item.prompt,
    )


def _pl_item_stable_key(item: Item) -> tuple[str, ...]:
    """Return a stable sort key that keeps each record's readouts adjacent."""
    return (
        *_prompt_record_key(item),
        item.target_lang,
        item.target_variant,
        item.target,
        "" if item.distractor_id is None else item.distractor_id,
    )


def _sequence_batches(
    specs: Sequence[tuple[int, Item, str, str, JointEncoding]],
    batch_size: int,
    *,
    keep_prompt_groups: bool,
) -> list[list[tuple[int, Item, str, str, JointEncoding]]]:
    """Partition sequences, optionally packing whole prompt-record groups."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not keep_prompt_groups:
        return [list(specs[start : start + batch_size]) for start in range(0, len(specs), batch_size)]

    groups: list[list[tuple[int, Item, str, str, JointEncoding]]] = []
    group_keys: list[tuple[str, ...]] = []
    for spec in specs:
        key = _prompt_record_key(spec[1])
        if not group_keys or group_keys[-1] != key:
            group_keys.append(key)
            groups.append([])
        groups[-1].append(spec)

    batches: list[list[tuple[int, Item, str, str, JointEncoding]]] = []
    current: list[tuple[int, Item, str, str, JointEncoding]] = []
    for group in groups:
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _append_sequence_columns(
    column_parts: dict[str, list[np.ndarray]],
    loaded: LoadedModel,
    item: Item,
    role: str,
    ref_id: str,
    result: Any,
    classifier: TokenClassifier,
) -> None:
    """Append one sequence's long measurements using column arrays, not row dictionaries."""
    target_ids = np.asarray(result.target_ids, dtype=np.int64)
    count = len(target_ids)
    layers = loaded.n_layers + 1
    rows_per_token = layers + 1
    row_count = count * rows_per_token
    token_rows = np.repeat(target_ids, rows_per_token)
    content = np.fromiter(
        (classifier.is_content(int(token_id)) for token_id in target_ids), dtype=np.bool_, count=count
    )
    metrics_per_token = np.asarray(["lens_logprob"] * layers + ["model_logprob"], dtype=object)
    layer_per_token = np.asarray([*range(layers), loaded.n_layers], dtype=np.int64)
    values = np.concatenate(
        (np.asarray(result.lens_logprobs, dtype=np.float32).T,
         np.asarray(result.model_logprobs, dtype=np.float32)[:, None]),
        axis=1,
    ).reshape(-1)
    record = item.record
    static: dict[str, Any] = {
        "model": loaded.spec.hf_id if loaded.spec is not None else loaded.family,
        "model_revision": loaded.spec.revision if loaded.spec is not None else "synthetic",
        "pass": "pl",
        "format": str(record["format"]),
        "condition": str(record.get("condition", "diac")),
        "fewshot_set": record.get("fewshot_set"),
        "concept_id": item.concept_id,
        "target_lang": item.target_lang,
        "target_variant": item.target_variant,
        "role": role,
        "ref_concept_id": ref_id,
    }
    for key, value in static.items():
        column_parts[key].append(np.full(row_count, value, dtype=object))
    column_parts["tok_idx"].append(np.repeat(np.arange(count, dtype=np.int64), rows_per_token))
    column_parts["token_id"].append(token_rows)
    column_parts["is_content"].append(np.repeat(content, rows_per_token))
    column_parts["layer"].append(np.tile(layer_per_token, count))
    column_parts["metric"].append(np.tile(metrics_per_token, count))
    column_parts["value"].append(values)


def run_pl(
    loaded: LoadedModel,
    lens: LensHead,
    items: Sequence[Item],
    *,
    batch_size: int,
    top_k: int,
    gen_max_new_tokens: int,
    encodings: Sequence[JointEncoding] | None = None,
    keep_prompt_groups: bool = False,
) -> PLOutput:
    """Run P-L with sequence-sized batches and output only stored target positions.

    When ``keep_prompt_groups`` is true, a prompt record's target readouts stay in
    the same batch. Resume-safe P-L units use this mode so batch membership is
    independent of other units.
    """
    import torch
    import torch.nn.functional as functional

    if batch_size <= 0 or top_k <= 0 or gen_max_new_tokens <= 0:
        raise ValueError("batch_size, top_k, and gen_max_new_tokens must be positive")
    classifier = TokenClassifier(loaded.tokenizer)
    specs = _sequence_specs(items, loaded.tokenizer, encodings)
    column_parts = {column: [] for column in io.LENS_COLUMNS}
    topk_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    residual_rows: list[np.ndarray] = []
    attn_rows: list[np.ndarray] = []
    mlp_rows: list[np.ndarray] = []
    generation_rows: list[dict[str, Any]] = []
    device = next(loaded.model.parameters()).device
    pad_id = _pad_id(loaded)

    expected_row_count = 0
    for _item_index, item, _role, _ref_id, encoding in specs:
        expected_row_count += len(encoding.target_ids) * (loaded.n_layers + 2)
    expected_row_count += sum(
        len(encoding.target_ids) * (loaded.n_layers + 1) * len(item.path_forms)
        for _item_index, item, role, _ref_id, encoding in specs
        if role == "correct" and item.target_lang == "vi" and item.path_forms
    )

    # Prompt metadata and activations are recorded once per prompt record, even when
    # that prompt has several language readouts.
    prompt_items: list[Item] = []
    seen_prompt_keys: set[tuple[str, str, str, str, str]] = set()
    for item in items:
        record = item.record
        key = (
            str(record["format"]), str(record.get("condition", "diac")),
            str(record.get("fewshot_set", "")), item.concept_id, item.prompt,
        )
        if key not in seen_prompt_keys:
            seen_prompt_keys.add(key)
            prompt_items.append(item)

    for chunk in _sequence_batches(specs, batch_size, keep_prompt_groups=keep_prompt_groups):
        chunk_encodings = [entry[4] for entry in chunk]
        tf_results = teacher_forced(loaded, lens, chunk_encodings, pad_id=pad_id)
        for (_item_index, item, role, ref_id, encoding), result in zip(chunk, tf_results, strict=True):
            if tuple(result.target_ids) != tuple(encoding.target_ids):
                raise AssertionError("teacher-forced result target ids differ from their joint encodings")
            _append_sequence_columns(column_parts, loaded, item, role, ref_id, result, classifier)
        vi_specs = [entry for entry in chunk if entry[2] == "correct" and entry[1].target_lang == "vi" and entry[1].path_forms]
        if vi_specs:
            lens_rows = _score_path_forms(
                loaded,
                lens,
                [entry[1] for entry in vi_specs],
                [entry[4] for entry in vi_specs],
                classifier,
                pad_id=pad_id,
            )
            # Path scoring is the only remaining row-oriented part; these are a
            # small fraction of all long rows and retain the established rule-38 order.
            for row in lens_rows:
                for column in io.LENS_COLUMNS:
                    column_parts[column].append(np.asarray([row[column]], dtype=object if column not in {"tok_idx", "token_id", "is_content", "layer", "value"} else None))

    prompt_batch_size = max(1, batch_size // 2)
    for start in range(0, len(prompt_items), prompt_batch_size):
        prompt_batch = prompt_items[start : start + prompt_batch_size]
        prompt_sequences = _encode_prompts(loaded, [item.prompt for item in prompt_batch])
        prompt_ids, prompt_mask, prompt_lengths = _pad_sequences(prompt_sequences, pad_id, device)
        positions = torch.as_tensor([[length - 1] for length in prompt_lengths], dtype=torch.long, device=device)
        prompt_recording = hooks.record(
            loaded, prompt_ids, prompt_mask, positions=positions, components=True, keep_on="cpu"
        )
        if prompt_recording.attn is None or prompt_recording.mlp is None:
            raise RuntimeError("component recording did not return attention and MLP activations")
        resid = prompt_recording.resid[:, :, 0, :].permute(1, 0, 2).float().numpy()
        attn = prompt_recording.attn[:, :, 0, :].permute(1, 0, 2).float().numpy()
        mlp = prompt_recording.mlp[:, :, 0, :].permute(1, 0, 2).float().numpy()
        for local_index, item in enumerate(prompt_batch):
            row_index = len(index_rows)
            record = item.record
            index_rows.append(
                {
                    "format": str(record["format"]),
                    "lang": str(record.get("lang", "")),
                    "concept_id": item.concept_id,
                    "split": str(record.get("split", "")),
                    "condition": str(record.get("condition", "diac")),
                    "fewshot_set": record.get("fewshot_set"),
                    "shard": record.get("pl_shard"),
                    "m1_extension": bool(record.get("m1_extension", False)),
                    "array_row": row_index,
                }
            )
            residual_rows.append(resid[local_index])
            attn_rows.append(attn[local_index])
            mlp_rows.append(mlp[local_index])
        top_values_by_layer: list[Any] = []
        top_ids_by_layer: list[Any] = []
        for layer in range(loaded.n_layers + 1):
            hidden = torch.as_tensor(resid[:, layer, :], dtype=torch.float32)
            per_prompt_values: list[Any] = []
            per_prompt_ids: list[Any] = []
            for prompt_index in range(len(prompt_batch)):
                # Preserve the original single-position projection arithmetic; stack
                # before transferring to CPU so there is no per-rank/device sync.
                row_log_probs = functional.log_softmax(
                    lens.logits(hidden[prompt_index : prompt_index + 1]), dim=-1
                )
                row_values, row_ids = torch.topk(
                    row_log_probs, min(top_k, loaded.vocab_size), dim=-1
                )
                per_prompt_values.append(row_values[0])
                per_prompt_ids.append(row_ids[0])
            values = torch.stack(per_prompt_values, dim=0)
            ids = torch.stack(per_prompt_ids, dim=0)
            top_values_by_layer.append(values)
            top_ids_by_layer.append(ids)
        all_values_np = torch.stack(top_values_by_layer, dim=0).detach().cpu().numpy()
        all_ids_np = torch.stack(top_ids_by_layer, dim=0).detach().cpu().numpy()
        for layer in range(loaded.n_layers + 1):
            values_np = all_values_np[layer]
            ids_np = all_ids_np[layer]
            for local_index, item in enumerate(prompt_batch):
                record = item.record
                for rank in range(ids_np.shape[1]):
                    topk_rows.append(
                        {
                            "model": loaded.spec.hf_id if loaded.spec else loaded.family,
                            "model_revision": loaded.spec.revision if loaded.spec else "synthetic",
                            "pass": "pl",
                            "format": str(record["format"]),
                            "condition": str(record.get("condition", "diac")),
                            "fewshot_set": record.get("fewshot_set"),
                            "concept_id": item.concept_id,
                            "target_lang": str(record.get("lang", item.target_lang)),
                            "target_variant": item.target_variant if "lang" in record else "prompt",
                            "layer": layer,
                            "rank": rank + 1,
                            "token_id": int(ids_np[local_index, rank]),
                            "value": float(values_np[local_index, rank]),
                        }
                    )
        for item, sequence in zip(prompt_batch, prompt_sequences, strict=True):
            input_ids = torch.as_tensor([sequence], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = loaded.model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=gen_max_new_tokens,
                )
            generated_ids = generated[0, len(sequence) :].detach().cpu().tolist()
            raw_text = loaded.tokenizer.decode(generated_ids)
            generation_rows.append(
                {
                    "format": str(item.record["format"]),
                    "lang": str(item.record.get("lang", "")),
                    "concept_id": item.concept_id,
                    "shard": item.record.get("pl_shard"),
                    "m1_extension": bool(item.record.get("m1_extension", False)),
                    "token_ids": [int(token_id) for token_id in generated_ids],
                    "text_raw": raw_text,
                    "text": _truncate_generation(raw_text),
                }
            )

    def _join_columns(parts: list[np.ndarray]) -> np.ndarray:
        """Join arrays for one output column without per-token Python rows."""
        if not parts:
            return np.asarray([], dtype=object)
        return np.concatenate(parts)

    long_df = pd.DataFrame(
        {column: _join_columns(column_parts[column]) for column in io.LENS_COLUMNS},
        columns=io.LENS_COLUMNS,
    )
    io.validate_long(long_df, io.LENS_COLUMNS)
    if len(long_df) != expected_row_count:
        raise AssertionError(f"P-L row count {len(long_df)} differs from expected {expected_row_count}")
    empty_resid = np.empty((0, loaded.n_layers + 1, loaded.d_model), dtype=np.float32)
    empty_components = np.empty((0, loaded.n_layers, loaded.d_model), dtype=np.float32)
    return PLOutput(
        long=long_df,
        answer_topk=pd.DataFrame(topk_rows),
        resid_index=pd.DataFrame(index_rows),
        resid=np.stack(residual_rows).astype(np.float32, copy=False) if residual_rows else empty_resid,
        attn=np.stack(attn_rows).astype(np.float32, copy=False) if attn_rows else empty_components,
        mlp=np.stack(mlp_rows).astype(np.float32, copy=False) if mlp_rows else empty_components.copy(),
        generations=pd.DataFrame(generation_rows),
        expected_row_count=expected_row_count,
    )


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    """Write one NumPy array atomically beside its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a JSON-lines auxiliary table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _unit_provenance(
    loaded: LoadedModel,
    cfg: Mapping[str, Any],
    repo_root: str | Path,
    entrypoint: Path | None,
    unit_key: Mapping[str, str | int],
    batch_size: int,
    expected_rows: int,
    run_git: RunGit,
) -> dict[str, Any]:
    """Build standard sidecar metadata for one family/pass/format result unit."""
    if loaded.spec is None:
        model_id, revision = loaded.family, "synthetic"
    else:
        model_id, revision = loaded.spec.hf_id, loaded.spec.revision
    loaded_dtype = str(next(loaded.model.parameters()).dtype)
    dtype_override = (
        "float32"
        if loaded_dtype == "torch.float32"
        and cfg.get("inference", {}).get("weights_dtype") == "bfloat16"
        else None
    )
    return build_provenance(
        cfg,
        repo_root=repo_root,
        unit_key=unit_key,
        entrypoint=entrypoint,
        model_id=model_id,
        model_revision=revision,
        dtype=loaded_dtype,
        batch_size=batch_size,
        expected_row_count=expected_rows,
        run_git=run_git,
        model_loads=[
            {
                "model_id": model_id,
                "model_revision": revision,
                "dtype": loaded_dtype,
                "dtype_override": dtype_override,
            }
        ],
    )


def _unit_completion(
    results_dir: str | Path,
    unit_key: Mapping[str, str | int],
    provenance: Mapping[str, Any],
    cfg: Mapping[str, Any],
    extra_suffixes: Sequence[str],
) -> tuple[bool, str]:
    """Check the parquet unit and its required auxiliary files before model work."""
    complete, reason = io.unit_is_complete(
        results_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"]
    )
    if not complete:
        return False, reason
    parquet_path, _sidecar_path = io.unit_paths(results_dir, unit_key)
    missing = [
        parquet_path.with_suffix(suffix).name
        for suffix in extra_suffixes
        if not parquet_path.with_suffix(suffix).is_file()
    ]
    if missing:
        return False, "missing auxiliary files: " + ", ".join(missing)
    return True, "complete"


def plan_pd_units(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> list[PDUnitPlan]:
    """Build P-D unit row counts and provenance, then check completion before forwards."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["format"]), []).append(dict(record))
    plans: list[PDUnitPlan] = []
    for format_name in sorted(grouped):
        unit_records = tuple(grouped[format_name])
        unit_key: dict[str, str | int] = {
            "family": loaded.family, "pass": "pd", "format": format_name
        }
        expected = len(unit_records) * (loaded.n_layers + 1)
        provenance = _unit_provenance(
            loaded, cfg, repo_root, entrypoint, unit_key, batch_size, expected, run_git
        )
        complete, reason = _unit_completion(
            results_dir,
            unit_key,
            provenance,
            cfg,
            (".resid.npy", ".resid_index.jsonl"),
        )
        plans.append(
            PDUnitPlan(
                unit_key=unit_key,
                format_name=format_name,
                records=unit_records,
                provenance=provenance,
                expected_row_count=expected,
                complete=complete,
                reason=reason,
            )
        )
    return plans


def plan_pl_units(
    loaded: LoadedModel,
    items: Sequence[Item],
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> list[PLUnitPlan]:
    """Tokenize P-L units and compute exact row counts before checking completion."""
    grouped: dict[tuple[str, str, str, int | None], list[Item]] = {}
    for item in items:
        key = pl_unit_group_key(item.record)
        grouped.setdefault(key, []).append(item)
    plans: list[PLUnitPlan] = []
    extra_suffixes = (
        ".resid.npy",
        ".attn.npy",
        ".mlp.npy",
        ".resid_index.jsonl",
        ".answer_topk.jsonl",
        ".generations.jsonl",
    )
    for group_key in sorted(grouped, key=lambda key: (key[0], key[1], key[2], -1 if key[3] is None else key[3])):
        format_name, condition, fewshot_set, shard = group_key
        unit_items = tuple(sorted(grouped[group_key], key=_pl_item_stable_key))
        sequence_specs = _sequence_specs(unit_items, loaded.tokenizer, None)
        encodings = tuple(entry[4] for entry in sequence_specs)
        expected = sum(len(entry[4].target_ids) * (loaded.n_layers + 2) for entry in sequence_specs)
        expected += sum(
            len(entry[4].target_ids) * (loaded.n_layers + 1) * len(entry[1].path_forms)
            for entry in sequence_specs
            if entry[2] == "correct" and entry[1].target_lang == "vi" and entry[1].path_forms
        )
        unit_key = pl_unit_key(loaded.family, unit_items[0].record)
        provenance = _unit_provenance(
            loaded, cfg, repo_root, entrypoint, unit_key, batch_size, expected, run_git
        )
        complete, reason = _unit_completion(
            results_dir, unit_key, provenance, cfg, extra_suffixes
        )
        plans.append(
            PLUnitPlan(
                unit_key=unit_key,
                format_name=format_name,
                items=unit_items,
                encodings=encodings,
                provenance=provenance,
                expected_row_count=expected,
                complete=complete,
                reason=reason,
            )
        )
    return plans


def run_pd_resume_safe(
    loaded: LoadedModel,
    records: Sequence[Mapping[str, Any]],
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> tuple[pd.DataFrame, np.ndarray, int, int, int]:
    """Skip complete P-D units before forwarding; otherwise run and persist missing units."""
    plans = plan_pd_units(
        loaded,
        records,
        results_dir,
        cfg,
        repo_root=repo_root,
        entrypoint=entrypoint,
        batch_size=batch_size,
        run_git=run_git,
    )
    incomplete = [plan for plan in plans if not plan.complete]
    skipped = len(plans) - len(incomplete)
    expected_total = sum(plan.expected_row_count for plan in plans)
    if not incomplete:
        return (
            pd.DataFrame(),
            np.empty((0, loaded.n_layers + 1, loaded.d_model), dtype=np.float32),
            skipped,
            len(plans),
            expected_total,
        )
    pending_records = [record for plan in incomplete for record in plan.records]
    index, residual = run_pd(loaded, pending_records, batch_size=batch_size)
    for plan in incomplete:
        actual = int((index["format"].astype(str) == plan.format_name).sum()) * (loaded.n_layers + 1)
        if actual != plan.expected_row_count:
            raise AssertionError(
                f"P-D {plan.format_name} row count {actual} differs from expected "
                f"{plan.expected_row_count}"
            )
    persist_pd_outputs(
        loaded,
        index,
        residual,
        results_dir,
        cfg,
        repo_root=repo_root,
        entrypoint=entrypoint,
        batch_size=batch_size,
        run_git=run_git,
    )
    return index, residual, skipped, len(plans), expected_total


def run_pl_resume_safe(
    loaded: LoadedModel,
    lens: LensHead,
    items: Sequence[Item],
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    top_k: int,
    gen_max_new_tokens: int,
    run_git: RunGit,
) -> tuple[PLOutput | None, int, int, int]:
    """Skip complete P-L units before forwards, otherwise run and persist missing units."""
    plans = plan_pl_units(
        loaded,
        items,
        results_dir,
        cfg,
        repo_root=repo_root,
        entrypoint=entrypoint,
        batch_size=batch_size,
        run_git=run_git,
    )
    incomplete = [plan for plan in plans if not plan.complete]
    skipped = len(plans) - len(incomplete)
    expected_total = sum(plan.expected_row_count for plan in plans)
    if not incomplete:
        return None, skipped, len(plans), expected_total
    expected_pending = sum(plan.expected_row_count for plan in incomplete)
    unit_outputs: list[PLOutput] = []
    for plan in incomplete:
        output = run_pl(
            loaded,
            lens,
            plan.items,
            batch_size=batch_size,
            top_k=top_k,
            gen_max_new_tokens=gen_max_new_tokens,
            encodings=plan.encodings,
            keep_prompt_groups=True,
        )
        if len(output.long) != plan.expected_row_count or output.expected_row_count != plan.expected_row_count:
            raise AssertionError(
                f"P-L unit {plan.unit_key} row count {len(output.long)} differs from "
                f"expected {plan.expected_row_count}"
            )
        persist_pl_outputs(
            loaded,
            output,
            results_dir,
            cfg,
            repo_root=repo_root,
            entrypoint=entrypoint,
            batch_size=batch_size,
            run_git=run_git,
        )
        unit_outputs.append(output)
    output = _combine_pl_outputs(unit_outputs)
    if len(output.long) != expected_pending or output.expected_row_count != expected_pending:
        raise AssertionError(
            f"P-L row count {len(output.long)} differs from pending-unit expected {expected_pending}"
        )
    return output, skipped, len(plans), expected_total


def _combine_pl_outputs(outputs: Sequence[PLOutput]) -> PLOutput:
    """Combine outputs from independently computed units for the caller."""
    if not outputs:
        raise ValueError("At least one P-L unit output is required")
    indices: list[pd.DataFrame] = []
    row_offset = 0
    for output in outputs:
        index = output.resid_index.copy()
        if not index.empty:
            index["array_row"] = index["array_row"].astype(int) + row_offset
        row_offset += len(output.resid_index)
        indices.append(index)
    return PLOutput(
        long=pd.concat([output.long for output in outputs], ignore_index=True),
        answer_topk=pd.concat([output.answer_topk for output in outputs], ignore_index=True),
        resid_index=pd.concat(indices, ignore_index=True),
        resid=np.concatenate([output.resid for output in outputs], axis=0),
        attn=np.concatenate([output.attn for output in outputs], axis=0),
        mlp=np.concatenate([output.mlp for output in outputs], axis=0),
        generations=pd.concat([output.generations for output in outputs], ignore_index=True),
        expected_row_count=sum(output.expected_row_count for output in outputs),
    )


def persist_pl_outputs(
    loaded: LoadedModel,
    output: PLOutput,
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> tuple[int, int]:
    """Atomically persist each P-L format/shard unit and return unit counts."""
    skipped = 0
    groups: dict[tuple[str, str, str, int | None], list[int]] = {}
    for row_index, row in output.resid_index.iterrows():
        raw_fewshot = row.get("fewshot_set")
        fewshot = "none" if raw_fewshot is None or pd.isna(raw_fewshot) else str(int(raw_fewshot))
        raw_shard = row.get("shard")
        shard = None if raw_shard is None or pd.isna(raw_shard) else int(raw_shard)
        key = (
            str(row["format"]),
            str(row.get("condition", "diac")),
            fewshot,
            shard,
        )
        groups.setdefault(key, []).append(int(row_index))
    ordered_groups = sorted(
        groups,
        key=lambda key: (key[0], key[1], key[2], -1 if key[3] is None else key[3]),
    )
    for group_key in ordered_groups:
        format_name, condition, fewshot, shard = group_key
        index = output.resid_index.loc[groups[group_key]].copy()
        concept_ids = set(index["concept_id"].astype(str))
        frame = output.long.loc[
            output.long["format"].astype(str).eq(format_name)
            & output.long["concept_id"].astype(str).isin(concept_ids)
        ].copy()
        array_rows = index["array_row"].astype(int).to_numpy() if not index.empty else np.array([], dtype=int)
        unit_key: dict[str, str | int] = {
            "family": loaded.family,
            "pass": "pl",
            "condition": condition,
            "fewshot_set": fewshot,
            "format": format_name,
        }
        if shard is not None:
            unit_key["shard"] = shard
        expected = int(len(frame))
        provenance = _unit_provenance(
            loaded, cfg, repo_root, entrypoint, unit_key, batch_size, expected, run_git
        )
        parquet_path, sidecar_path = io.unit_paths(results_dir, unit_key)
        extras = [
            parquet_path.with_suffix(".resid.npy"),
            parquet_path.with_suffix(".attn.npy"),
            parquet_path.with_suffix(".mlp.npy"),
            parquet_path.with_suffix(".resid_index.jsonl"),
            parquet_path.with_suffix(".answer_topk.jsonl"),
            parquet_path.with_suffix(".generations.jsonl"),
        ]
        complete, _reason = io.unit_is_complete(
            results_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"]
        )
        if complete and all(path.is_file() for path in extras):
            skipped += 1
            continue
        sidecar_path.unlink(missing_ok=True)
        persisted_index = index.copy()
        persisted_index["array_row"] = np.arange(len(persisted_index), dtype=np.int64)
        _atomic_npy(extras[0], output.resid[array_rows])
        _atomic_npy(extras[1], output.attn[array_rows])
        _atomic_npy(extras[2], output.mlp[array_rows])
        _atomic_jsonl(extras[3], persisted_index.to_dict(orient="records"))
        topk = output.answer_topk.loc[
            output.answer_topk["format"].astype(str).eq(format_name)
            & output.answer_topk["concept_id"].astype(str).isin(concept_ids)
        ]
        _atomic_jsonl(extras[4], topk.to_dict(orient="records"))
        generations = output.generations.loc[
            output.generations["format"].astype(str).eq(format_name)
            & output.generations["concept_id"].astype(str).isin(concept_ids)
        ]
        _atomic_jsonl(extras[5], generations.to_dict(orient="records"))
        io.write_unit(
            frame,
            results_dir,
            provenance,
            columns=io.LENS_COLUMNS,
            sort_by=[column for column in io.LENS_COLUMNS if column != "value"],
            compression=str(cfg["runtime"]["parquet_compression"]),
        )
    return skipped, len(groups)


def persist_pd_outputs(
    loaded: LoadedModel,
    index_df: pd.DataFrame,
    resid: np.ndarray,
    results_dir: str | Path,
    cfg: Mapping[str, Any],
    *,
    repo_root: str | Path,
    entrypoint: Path | None,
    batch_size: int,
    run_git: RunGit,
) -> tuple[int, int]:
    """Persist prompt-only residual arrays and their long-form index units."""
    skipped = 0
    formats = sorted(set(index_df["format"].astype(str))) if not index_df.empty else []
    for format_name in formats:
        index = index_df.loc[index_df["format"].astype(str) == format_name].copy().reset_index(drop=True)
        row_indices = index["array_row"].astype(int).to_numpy()
        unit_key = {"family": loaded.family, "pass": "pd", "format": format_name}
        expected = len(index) * (loaded.n_layers + 1)
        provenance = _unit_provenance(
            loaded, cfg, repo_root, entrypoint, unit_key, batch_size, expected, run_git
        )
        parquet_path, sidecar_path = io.unit_paths(results_dir, unit_key)
        array_path = parquet_path.with_suffix(".resid.npy")
        index_path = parquet_path.with_suffix(".resid_index.jsonl")
        complete, _reason = io.unit_is_complete(
            results_dir, unit_key, provenance, cfg["runtime"]["provenance_compare"]
        )
        if complete and array_path.is_file() and index_path.is_file():
            skipped += 1
            continue
        sidecar_path.unlink(missing_ok=True)
        _atomic_npy(array_path, resid[row_indices])
        persisted_index = index.copy()
        persisted_index["array_row"] = np.arange(len(persisted_index), dtype=np.int64)
        _atomic_jsonl(index_path, persisted_index.to_dict(orient="records"))
        spec = loaded.spec
        model_name = spec.hf_id if spec else loaded.family
        revision = spec.revision if spec else "synthetic"
        rows: list[dict[str, Any]] = []
        for local_row, record in enumerate(persisted_index.to_dict(orient="records")):
            lang = str(record.get("lang", ""))
            variant = "nospace" if lang == "zh" else "released"
            for layer in range(loaded.n_layers + 1):
                rows.append(
                    {
                        "model": model_name,
                        "model_revision": revision,
                        "pass": "pd",
                        "format": format_name,
                        "condition": str(record.get("condition", "diac")),
                        "fewshot_set": record.get("fewshot_set"),
                        "concept_id": str(record["concept_id"]),
                        "target_lang": lang,
                        "target_variant": variant,
                        "role": "correct",
                        "ref_concept_id": str(record["concept_id"]),
                        "tok_idx": -1,
                        "token_id": -1,
                        "is_content": False,
                        "layer": layer,
                        "metric": "residual_cache_index",
                        "value": float(local_row),
                    }
                )
        table = pd.DataFrame(rows, columns=io.LENS_COLUMNS)
        io.write_unit(
            table,
            results_dir,
            provenance,
            columns=io.LENS_COLUMNS,
            sort_by=[column for column in io.LENS_COLUMNS if column != "value"],
            compression=str(cfg["runtime"]["parquet_compression"]),
        )
    return skipped, len(formats)
