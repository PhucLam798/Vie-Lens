"""Text normalization helpers ported from the frozen preparation pipeline."""

from __future__ import annotations

import unicodedata

# Verbatim helpers from data/build/common.py and data/build/06_filter.py
# at tarudesu/viLens-concepts-preparation commit 873def8.


def normalize_nfc(value: str) -> str:
    """Return *value* normalized to Unicode NFC."""
    if not isinstance(value, str):
        raise TypeError(f"normalize_nfc expects str, got {type(value).__name__}")
    return unicodedata.normalize("NFC", value)


def strip_diacritics(value: str, *, preserve_case: bool = False) -> str:
    """Remove combining marks and Vietnamese đ/Đ; optionally preserve letter case."""
    if not isinstance(value, str):
        raise TypeError(f"strip_diacritics expects str, got {type(value).__name__}")
    decomposed = unicodedata.normalize("NFD", normalize_nfc(value))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return stripped if preserve_case else stripped.lower()


def normalized_levenshtein(left: str, right: str) -> float:
    """Return Levenshtein distance divided by the longer string length."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("normalized_levenshtein expects two strings")
    left, right = normalize_nfc(left), normalize_nfc(right)
    if left == right:
        return 0.0
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    # Two-row dynamic program keeps memory linear in the shorter input.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1] / denominator


def surface_norm(
    value: str,
    *,
    vietnamese: bool,
    remove_chars: list[str],
    vi_replace: dict[str, str],
) -> str:
    """Normalize a surface form as configured for the surface-distance filter."""
    result = strip_diacritics(normalize_nfc(value)).lower()
    for char in remove_chars:
        result = result.replace(char, "")
    if vietnamese:
        for source, target in sorted(vi_replace.items(), key=lambda item: (-len(item[0]), item[0])):
            result = result.replace(source, target)
    return result


# Verbatim port from data/build/common.py in tarudesu/viLens-concepts-preparation,
# commit 873def8.
def vi_orth_key(value: str) -> str:
    """Return a normalized Vietnamese orthographic-equivalence key.

    Older tone placement in open Vietnamese data is mapped to the modern
    first-vowel placement for open ``oa``, ``oe``, and ``uy`` rimes. A final
    ``y`` after the specified consonants is folded to ``i``; standalone ``y``
    and ``qu`` spellings are left intact.
    """
    if not isinstance(value, str):
        raise TypeError(f"vi_orth_key expects str, got {type(value).__name__}")

    tone_marks = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
    modern_rimes = {"oa", "oe", "uy"}
    final_y_onsets = set("bcđdhklmnstvx")
    modern_syllables: list[str] = []

    for syllable in normalize_nfc(value).casefold().split():
        graphemes: list[list[str]] = []
        for char in unicodedata.normalize("NFD", syllable):
            if unicodedata.combining(char):
                if not graphemes:
                    graphemes.append([char])
                else:
                    graphemes[-1].append(char)
            else:
                graphemes.append([char])

        bases = "".join(group[0] for group in graphemes)
        if len(graphemes) >= 2 and bases[-2:] in modern_rimes and not bases.startswith("qu"):
            first, second = graphemes[-2], graphemes[-1]
            old_tone = [mark for mark in second[1:] if mark in tone_marks]
            if old_tone:
                second[:] = [second[0], *(mark for mark in second[1:] if mark not in tone_marks)]
                first.extend(old_tone)

        bases = "".join(group[0] for group in graphemes)
        if bases.endswith("y") and bases[0] in final_y_onsets and not bases.startswith("qu"):
            graphemes[-1][0] = "i"

        modern_syllables.append(normalize_nfc("".join("".join(group) for group in graphemes)))

    return " ".join(modern_syllables)
