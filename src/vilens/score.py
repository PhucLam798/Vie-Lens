"""Answer normalization and v1.4 A10 competence classification."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .text import strip_diacritics, vi_orth_key


def normalize_answer(text: str) -> str:
    """Normalize an answer with NFC, outer whitespace removal, and case folding."""
    return unicodedata.normalize("NFC", text).strip().casefold()


def _contains_cjk(text: str) -> bool:
    """Return whether text contains an ideograph or Japanese syllabary character."""
    for character in text:
        codepoint = ord(character)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x323AF
            or 0x3040 <= codepoint <= 0x30FF
            or 0x31F0 <= codepoint <= 0x31FF
        ):
            return True
    return False


def matches(answer: str, form: str) -> bool:
    """Match a normalized form exactly or at a word boundary, with CJK prefix support."""
    normalized_answer = normalize_answer(answer)
    normalized_form = normalize_answer(form)
    if not normalized_form:
        return False
    if normalized_answer == normalized_form:
        return True
    if not normalized_answer.startswith(normalized_form):
        return False
    if _contains_cjk(normalized_form):
        return True
    return len(normalized_answer) > len(normalized_form) and not normalized_answer[
        len(normalized_form)
    ].isalnum()


def _nonempty_text(value: Any) -> str | None:
    """Return a non-empty string cell, treating pandas nullable values as absent."""
    return value if isinstance(value, str) and value.strip() else None


def source_groups(concepts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group all concepts sharing a normalized non-empty Russian source string."""
    if isinstance(concepts, Mapping):
        rows = concepts.items()
    else:
        rows = ((str(row["concept_id"]), row) for row in concepts)
    grouped: dict[str, list[str]] = defaultdict(list)
    for concept_id, concept in rows:
        ru = _nonempty_text(concept.get("ru"))
        if ru is None:
            continue
        key = unicodedata.normalize("NFC", ru).casefold().replace("ё", "е")
        grouped[key].append(str(concept_id))
    return {
        key: sorted(concept_ids)
        for key, concept_ids in sorted(grouped.items())
        if len(concept_ids) >= 2
    }


def _vi_forms(concept: Mapping[str, Any], *, condition: str) -> list[str]:
    """Collect the Vietnamese form and alternatives for the selected condition."""
    values: list[str] = []
    vi = _nonempty_text(concept.get("vi"))
    if vi is not None:
        values.append(vi)
    alternatives = concept.get("vi_alts", [])
    if isinstance(alternatives, (list, tuple)):
        values.extend(value for value in alternatives if isinstance(value, str) and value.strip())
    if condition == "nodiac":
        return [strip_diacritics(value, preserve_case=True) for value in values]
    return values


def _source_key(concept: Mapping[str, Any]) -> str | None:
    """Return the source-group key for one concept when its Russian source exists."""
    ru = _nonempty_text(concept.get("ru"))
    if ru is None:
        return None
    return unicodedata.normalize("NFC", ru).casefold().replace("ё", "е")


def classify(
    answer: str,
    concept: Mapping[str, Any],
    *,
    condition: str,
    format: str,
    groups: Mapping[str, Sequence[str]],
    concepts: Mapping[str, Mapping[str, Any]],
) -> str:
    """Classify a generated answer as VI, EN, OTHER-LANG, or DAMAGE in A10 order."""
    answer_for_vi = (
        strip_diacritics(answer, preserve_case=True) if condition == "nodiac" else answer
    )
    vi_forms = _vi_forms(concept, condition=condition)
    if format == "translation":
        key = _source_key(concept)
        if key is not None:
            for grouped_id in groups.get(key, ()):
                if str(grouped_id) == str(concept.get("concept_id", "")):
                    continue
                grouped_concept = concepts.get(str(grouped_id))
                if grouped_concept is not None:
                    vi_forms.extend(_vi_forms(grouped_concept, condition=condition))
    normalized_vi_answer = vi_orth_key(normalize_answer(answer_for_vi))
    if any(
        matches(normalized_vi_answer, vi_orth_key(normalize_answer(form)))
        for form in vi_forms
    ):
        return "VI"

    en = _nonempty_text(concept.get("en"))
    if en is not None and matches(answer, en):
        return "EN"

    for language in ("zh", "fr", "id", "ru"):
        form = _nonempty_text(concept.get(language))
        if form is not None and matches(answer, form):
            return "OTHER-LANG"
    return "DAMAGE"
