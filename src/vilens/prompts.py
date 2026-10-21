"""Prompt-integrity checks and the gated model-facing prompt accessor."""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable

from .locks import ensure_model_access_allowed


def check_prompt_integrity(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return NFC, trailing-whitespace, or target-prefix problems without mutating records."""
    problems: list[str] = []
    for record_index, record in enumerate(records):
        prompt = record.get("prompt")
        label = str(record.get("concept_id", record_index))
        if not isinstance(prompt, str):
            problems.append(f"{label}: prompt is not a string")
        else:
            if unicodedata.normalize("NFC", prompt) != prompt:
                problems.append(f"{label}: prompt is not NFC")
            if prompt and prompt[-1].isspace():
                problems.append(f"{label}: prompt ends in whitespace")
        target_keys = [key for key in record if key.startswith("target_")] + (["target"] if "target" in record else [])
        for key in target_keys:
            target = record[key]
            if not isinstance(target, str):
                problems.append(f"{label}: {key} is not a string")
                continue
            if unicodedata.normalize("NFC", target) != target:
                problems.append(f"{label}: {key} is not NFC")
            if len(target) < 2 or target[0] != " " or target[1].isspace():
                problems.append(f"{label}: {key} must start with exactly one space then non-whitespace")
    return problems


def records_for_model(cfg: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records unchanged only after the model-access integrity lock passes."""
    ensure_model_access_allowed(cfg, records)
    return records
