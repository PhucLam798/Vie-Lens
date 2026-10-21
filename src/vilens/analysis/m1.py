"""Wendler-style vocabulary-prefix token sets and probability aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

SPACE_MARKERS = {"gemma": "▁", "llama": "Ġ", "qwen": "Ġ"}


def vocabulary_prefix_index(tokenizer: Any, *, family: str) -> dict[str, tuple[tuple[str, int], ...]]:
    """Index vocabulary pieces by their first character for repeated prefix lookup."""
    if family not in SPACE_MARKERS:
        raise ValueError(f"Unknown tokenizer family: {family}")
    marker = SPACE_MARKERS[family]
    values: dict[str, list[tuple[str, int]]] = {}
    for token, token_id in tokenizer.get_vocab().items():
        piece = str(token)
        if piece.startswith(marker):
            piece = piece[len(marker) :]
        if piece:
            values.setdefault(piece[0], []).append((piece, int(token_id)))
    return {first: tuple(sorted(items)) for first, items in values.items()}


def prefix_token_ids(
    tokenizer: Any,
    word: str,
    *,
    family: str,
    language: str,
    unicode_prefix_tokid: int | None = None,
    prefix_index: Mapping[str, tuple[tuple[str, int], ...]] | None = None,
) -> tuple[int, ...]:
    """Collect vocab tokens spelling a non-empty prefix, with/without word marker.

    A single leading literal space is removed from ``word`` because the token
    variants are indexed both with and without the tokenizer's marker. Byte-level
    BPE vocabularies store UTF-8 bytes in the byte-to-Unicode alphabet, so the
    lookup spelling is converted before matching. For zh/ru, callers may supply
    the first Unicode-prefix token id obtained by the tokenizer; it is unioned
    with the spelling-derived ids.
    """
    if family not in SPACE_MARKERS:
        raise ValueError(f"Unknown tokenizer family: {family}")
    text = str(word)
    if text.startswith(" "):
        text = text[1:]
    if not text:
        raise ValueError("word must contain at least one non-space character")
    lookup_text = _prefix_lookup_text(text, family=family)
    index = vocabulary_prefix_index(tokenizer, family=family) if prefix_index is None else prefix_index
    token_ids: set[int] = set()
    for piece, token_id in index.get(lookup_text[0], ()):
        if piece and lookup_text.startswith(piece):
            token_ids.add(int(token_id))
    if language in {"zh", "ru"} and unicode_prefix_tokid is not None:
        token_ids.add(int(unicode_prefix_tokid))
    return tuple(sorted(token_ids))


def require_nonempty_prefix_token_ids(
    tokenizer: Any,
    word: str,
    *,
    family: str,
    language: str,
    item_description: str,
    unicode_prefix_tokid: int | None = None,
    prefix_index: Mapping[str, tuple[tuple[str, int], ...]] | None = None,
) -> tuple[int, ...]:
    """Return a prefix set or raise an item-specific error when it is empty."""
    token_ids = prefix_token_ids(
        tokenizer,
        word,
        family=family,
        language=language,
        unicode_prefix_tokid=unicode_prefix_tokid,
        prefix_index=prefix_index,
    )
    if not token_ids:
        raise ValueError(
            f"Empty vocabulary-prefix token set for {item_description}; "
            f"family={family}, language={language}, target={word!r}"
        )
    return token_ids


def _prefix_lookup_text(text: str, *, family: str) -> str:
    """Encode a target prefix in the byte alphabet used by byte-level BPE."""
    if family in {"llama", "qwen"}:
        byte_map = _byte_to_unicode()
        return "".join(byte_map[value] for value in text.encode("utf-8"))
    return text


def unicode_prefix_token_id(tokenizer: Any, word: str, *, family: str) -> int | None:
    """Return the first-byte token id used by the registered tokenizer family.

    SentencePiece uses the literal ``<0xHH>`` byte-fallback spelling. The
    byte-level BPE tokenizers use GPT-2's byte-to-Unicode mapping. Some
    tokenizers omit the mapped piece, in which case no extra id is added.
    """
    return unicode_prefix_token_id_for_family(tokenizer, word, family=family)


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-Unicode map used by byte-level BPE."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(visible, codepoints, strict=True)}


def unicode_prefix_token_id_for_family(tokenizer: Any, word: str, *, family: str) -> int | None:
    """Resolve the first UTF-8 byte through the family tokenizer vocabulary."""
    if family not in SPACE_MARKERS:
        raise ValueError(f"Unknown tokenizer family: {family}")
    text = str(word)
    if text.startswith(" "):
        text = text[1:]
    if not text:
        raise ValueError("word must contain at least one non-space character")
    first_byte = text[0].encode("utf-8")[0]
    byte_token = (
        f"<0x{first_byte:02X}>"
        if family == "gemma"
        else _byte_to_unicode()[first_byte]
    )
    token_id = tokenizer.get_vocab().get(byte_token)
    return int(token_id) if token_id is not None else None


def validate_probability_values(
    values: np.ndarray,
    *,
    expected_rows: int,
    observed_rows: int,
) -> None:
    """Fail unless M1 rows have the expected count and finite probabilities."""
    probabilities = np.asarray(values, dtype=np.float64)
    if int(observed_rows) != int(expected_rows) or probabilities.size != int(expected_rows):
        raise AssertionError(
            f"M1 row count {observed_rows} differs from expected {expected_rows}"
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("M1 probabilities contain non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise AssertionError("M1 probabilities fall outside [0, 1]")


def lens_consistency_deltas(
    computed_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
) -> np.ndarray:
    """Return aligned absolute log-probability differences in float64."""
    computed = np.asarray(computed_log_probs, dtype=np.float64)
    reference = np.asarray(reference_log_probs, dtype=np.float64)
    if computed.shape != reference.shape or computed.size == 0:
        raise AssertionError("E2 and P-L lens consistency vectors must be non-empty and aligned")
    if not np.isfinite(computed).all() or not np.isfinite(reference).all():
        raise AssertionError("Lens consistency values contain non-finite entries")
    return np.abs(computed - reference)


def absolute_delta_gate_passes(
    max_abs_delta: float | None,
    *,
    tolerance: float,
) -> bool:
    """Return whether an observed maximum absolute delta meets its gate."""
    if max_abs_delta is None:
        return False
    maximum = float(max_abs_delta)
    return bool(np.isfinite(maximum) and maximum <= float(tolerance))


def cyclic_shift_lens_control_deltas(
    computed_log_probs: np.ndarray,
    prompt_ids: Sequence[str],
    next_prompt_by_id: Mapping[str, str],
    answer_topk_by_prompt_layer: Mapping[
        tuple[str, int], Sequence[Mapping[str, Any]]
    ],
    *,
    layer: int,
) -> np.ndarray:
    """Compare prompt log-probs to the next prompt's same-layer top-k values.

    ``prompt_ids`` describes the current rows in ``computed_log_probs`` and
    ``next_prompt_by_id`` supplies a cyclic one-row shift within one unit.
    Layer 0 is intentionally excluded from the discrimination control.
    Missing or invalid top-k entries are skipped here; the aligned Gate A
    coverage check reports those separately.
    """
    log_probs = np.asarray(computed_log_probs, dtype=np.float64)
    ids = [str(prompt_id) for prompt_id in prompt_ids]
    if log_probs.ndim != 2 or log_probs.shape[0] != len(ids) or not ids:
        raise AssertionError(
            "Cyclic lens control requires a non-empty prompt-by-vocabulary matrix"
        )
    if len(ids) != len(set(ids)):
        raise AssertionError("Cyclic lens control prompt ids must be unique within a batch")
    if int(layer) < 0:
        raise ValueError("Cyclic lens control layer must be non-negative")
    if int(layer) == 0:
        return np.asarray([], dtype=np.float64)

    deltas: list[float] = []
    for row_index, prompt_id in enumerate(ids):
        next_prompt_id = next_prompt_by_id.get(prompt_id)
        if next_prompt_id is None:
            raise AssertionError(f"Cyclic lens control has no next prompt for {prompt_id}")
        reference_rows = answer_topk_by_prompt_layer.get((str(next_prompt_id), int(layer)), ())
        for reference_row in reference_rows:
            token_id = int(reference_row.get("token_id", -1))
            reference_value = float(reference_row.get("value", np.nan))
            if (
                not 0 <= token_id < log_probs.shape[1]
                or not np.isfinite(reference_value)
            ):
                continue
            computed_value = float(log_probs[row_index, token_id])
            if np.isfinite(computed_value):
                deltas.append(abs(computed_value - reference_value))
    return np.asarray(deltas, dtype=np.float64)


def summarize_lens_control_deltas(deltas: np.ndarray) -> dict[str, int | float | None]:
    """Summarize the negative-control absolute deltas for one family."""
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "matched_count": 0,
            "median_abs_delta": None,
            "p05_abs_delta": None,
            "min_abs_delta": None,
        }
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise AssertionError("Lens control deltas must be finite and non-negative")
    return {
        "matched_count": int(values.size),
        "median_abs_delta": float(np.median(values)),
        "p05_abs_delta": float(np.percentile(values, 5)),
        "min_abs_delta": float(np.min(values)),
    }


def lens_control_passes(
    summary: Mapping[str, Any],
    *,
    minimum_median: float,
) -> bool:
    """Return whether a negative control is discriminative at its median gate."""
    median = summary.get("median_abs_delta")
    return bool(
        median is not None
        and np.isfinite(float(median))
        and float(median) >= float(minimum_median)
    )


def summarize_report_only_lens_deltas(deltas: np.ndarray) -> dict[str, int | float | None]:
    """Summarize a report-only comparison without applying a pass/fail tolerance."""
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "matched_count": 0,
            "median_abs_delta": None,
            "p95_abs_delta": None,
            "p99_abs_delta": None,
            "max_abs_delta": None,
            "argmax_index": None,
        }
    values = np.where(np.isfinite(values) & (values >= 0.0), values, np.inf)
    return {
        "matched_count": int(values.size),
        "median_abs_delta": float(np.median(values)),
        "p95_abs_delta": float(np.percentile(values, 95)),
        "p99_abs_delta": float(np.percentile(values, 99)),
        "max_abs_delta": float(np.max(values)),
        "argmax_index": int(np.argmax(values)),
    }


def summarize_lens_consistency_deltas(deltas: np.ndarray) -> dict[str, int | float]:
    """Summarize absolute consistency differences and identify their argmax index."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise AssertionError("Lens consistency deltas must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise AssertionError("Lens consistency deltas must be finite and non-negative")
    return {
        "matched_count": int(values.size),
        "median_abs_delta": float(np.median(values)),
        "p95_abs_delta": float(np.percentile(values, 95)),
        "p99_abs_delta": float(np.percentile(values, 99)),
        "max_abs_delta": float(np.max(values)),
        "argmax_index": int(np.argmax(values)),
    }


def lens_consistency_max_abs(
    computed_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    *,
    tolerance: float = 1.0e-2,
) -> float:
    """Apply Gate A's absolute maximum-difference tolerance to aligned values."""
    deltas = lens_consistency_deltas(computed_log_probs, reference_log_probs)
    maximum = float(np.max(deltas))
    if maximum > float(tolerance):
        raise AssertionError(
            f"E2/P-L lens consistency max |Δ|={maximum:.8g} exceeds {float(tolerance):.8g}"
        )
    return maximum


def first_token_lens_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    concept_ids: set[str],
    readouts: Sequence[tuple[str, str]],
    n_layers: int,
) -> dict[tuple[str, str, str, int], tuple[int, float]]:
    """Index P-L first-token rows for every selected prompt/readout/block.

    Correct-target rows take precedence over path rows when both represent the
    same prompt and readout. Path rows fill the other four readouts for the
    Vietnamese directions trial, whose P-L unit has one correct readout.
    """
    readout_set = set(readouts)
    indexed: dict[tuple[str, str, str, int], tuple[int, float]] = {}
    roles: dict[tuple[str, str, str, int], str] = {}
    for row in rows:
        concept_id = str(row.get("concept_id", ""))
        language = str(row.get("target_lang", ""))
        variant = str(row.get("target_variant", ""))
        role = str(row.get("role", ""))
        layer = int(row.get("layer", -1))
        if (
            concept_id not in concept_ids
            or (language, variant) not in readout_set
            or role not in {"correct", "path"}
            or not 0 <= layer <= int(n_layers)
        ):
            continue
        key = (concept_id, language, variant, layer)
        old_role = roles.get(key)
        if old_role == role:
            raise AssertionError(f"Duplicate P-L first-token lens row for {key}")
        if old_role is None or role == "correct":
            indexed[key] = (int(row["token_id"]), float(row["value"]))
            roles[key] = role
    expected = {
        (concept_id, language, variant, layer)
        for concept_id in concept_ids
        for language, variant in readouts
        for layer in range(int(n_layers) + 1)
    }
    missing = expected - indexed.keys()
    if missing:
        example = sorted(missing)[:3]
        raise AssertionError(
            f"P-L first-token lens rows missing for {len(missing)} prompt/readout/layer keys: {example}"
        )
    return indexed


def sum_token_probabilities(logits: np.ndarray, token_ids: tuple[int, ...]) -> float:
    """Softmax logits in float32 and sum probability over a deduplicated token set."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("answer-position logits must be one-dimensional")
    ids = np.asarray(sorted(set(map(int, token_ids))), dtype=np.int64)
    if ids.size == 0 or ids.min() < 0 or ids.max() >= values.size:
        raise ValueError("token id set is empty or outside the vocabulary")
    shifted = values - np.max(values)
    probs = np.exp(shifted, dtype=np.float32)
    probs /= np.sum(probs, dtype=np.float64)
    return float(np.sum(probs[ids], dtype=np.float64))
