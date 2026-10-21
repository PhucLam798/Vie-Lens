"""Pure deterministic helpers for frozen X0 derived manifests."""

from __future__ import annotations

import hashlib
import os
import zlib
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .text import strip_diacritics
from .tokenize import JointEncoding, TokenClassifier, encode_joint

StatKey = tuple[str, str, str, str]
FormStat = tuple[int, int | None]


def stream(seed: int, key: str) -> np.random.Generator:
    """Return the registered independent NumPy RNG stream for one exact key."""
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def target_string(concept: Mapping[str, Any], lang: str, variant: str) -> str:
    """Return a frozen target form or the runtime Russian translation readout."""
    if variant == "released" and lang in {"vi", "en", "fr", "id", "zh"}:
        return " " + str(concept[lang])
    if (lang, variant) == ("zh", "nospace"):
        return str(concept["zh"])
    if (lang, variant) == ("vi", "nodiac"):
        return " " + strip_diacritics(str(concept["vi"]), preserve_case=True)
    if (lang, variant) == ("ru", "released"):
        value = concept.get("ru")
        if value is None or not str(value).strip():
            raise ValueError(f"Russian target is missing for concept {concept.get('concept_id', '<unknown>')}")
        return " " + str(value)
    raise ValueError(f"Unsupported target combination: {lang}/{variant}")


def form_stats(tokenizer: Any, classifier: TokenClassifier, prompt: str, target: str) -> FormStat:
    """Strictly encode one prompt and target, returning content count and first ID."""
    encoding = encode_joint(tokenizer, prompt, target, strict=True)
    return form_stats_from_encoding(encoding, classifier)


def form_stats_from_encoding(encoding: JointEncoding, classifier: TokenClassifier) -> FormStat:
    """Summarize an already jointly encoded form without additional tokenizer calls."""
    content_ids = [token_id for token_id in encoding.target_ids if classifier.is_content(token_id)]
    return len(content_ids), int(content_ids[0]) if content_ids else None


def _candidate_counts(
    candidates: Sequence[str],
    concepts: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, str],
    stats: Mapping[StatKey, FormStat],
    family: str,
    lang: str,
    variant: str,
) -> dict[int, list[str]]:
    """Index candidate distractors by content-token count."""
    by_count: dict[int, list[str]] = defaultdict(list)
    for concept_id in candidates:
        by_count[stats[(family, lang, variant, concept_id)][0]].append(concept_id)
    for values in by_count.values():
        values.sort()
    return dict(by_count)


def _match_distractors(
    concepts: Mapping[str, Mapping[str, Any]],
    stats: Mapping[StatKey, FormStat],
    *,
    family: str,
    lang: str,
    variant: str,
    pool_ids: Sequence[str],
    seed: int,
    key_prefix: str,
) -> list[dict[str, Any]]:
    """Match from cached form statistics; this loop performs no tokenization."""
    ids = sorted(pool_ids)
    targets = {concept_id: target_string(concepts[concept_id], lang, variant) for concept_id in ids}
    candidate_ids = [
        concept_id
        for concept_id in ids
        if not (lang, variant) == ("vi", "nodiac") or not bool(concepts[concept_id]["collapsed"])
    ]
    by_count = _candidate_counts(candidate_ids, concepts, targets, stats, family, lang, variant)
    # For each count, retain zipf levels and frequencies needed for exact nearest-neighbour matching.
    by_count_zipf: dict[int, tuple[list[float], dict[float, list[str]]]] = {}
    for count, bucket in by_count.items():
        zipf_ids: dict[float, list[str]] = defaultdict(list)
        for concept_id in bucket:
            zipf_ids[float(concepts[concept_id]["zipf_en"])].append(concept_id)
        levels = sorted(zipf_ids)
        by_count_zipf[count] = (levels, dict(zipf_ids))

    target_count: dict[tuple[int, str], int] = Counter()
    first_count: dict[tuple[int, int | None], int] = Counter()
    pair_count: dict[tuple[int, str, int | None], int] = Counter()
    for count, bucket in by_count.items():
        for concept_id in bucket:
            first_id = stats[(family, lang, variant, concept_id)][1]
            target = targets[concept_id]
            target_count[(count, target)] += 1
            first_count[(count, first_id)] += 1
            pair_count[(count, target, first_id)] += 1

    def valid_count(count: int, target: str, first_id: int | None) -> int:
        """Count candidates remaining after target and first-ID exclusions."""
        size = len(by_count.get(count, ()))
        invalid = target_count[(count, target)] + first_count[(count, first_id)]
        both = pair_count[(count, target, first_id)]
        return size - invalid + both

    rows: list[dict[str, Any]] = []
    available_counts = sorted(by_count)
    for concept_id in ids:
        content_count, first_id = stats[(family, lang, variant, concept_id)]
        target = targets[concept_id]
        zipf = float(concepts[concept_id]["zipf_en"])
        exact_available = valid_count(content_count, target, first_id) > 0
        if exact_available:
            chosen_counts = [content_count]
            kind = "exact"
        else:
            viable = [count for count in available_counts if valid_count(count, target, first_id) > 0]
            if not viable:
                raise ValueError(f"No eligible distractor for {family}/{lang}/{variant}/{concept_id}")
            min_delta = min(abs(count - content_count) for count in viable)
            chosen_counts = [count for count in viable if abs(count - content_count) == min_delta]
            kind = "count_fallback"

        level_ids_mutable: dict[float, list[str]] = defaultdict(list)
        for chosen_count in chosen_counts:
            for level, level_concepts in by_count_zipf[chosen_count][1].items():
                level_ids_mutable[level].extend(level_concepts)
        levels = sorted(level_ids_mutable)
        level_ids = dict(level_ids_mutable)
        insertion = bisect_left(levels, zipf)
        left, right = insertion - 1, insertion
        best_delta: float | None = None
        tied: list[str] = []
        while left >= 0 or right < len(levels):
            next_left = abs(levels[left] - zipf) if left >= 0 else float("inf")
            next_right = abs(levels[right] - zipf) if right < len(levels) else float("inf")
            next_delta = min(next_left, next_right)
            if best_delta is not None and next_delta > best_delta:
                break
            indices: list[int] = []
            if left >= 0 and next_left == next_delta:
                indices.append(left)
                left -= 1
            if right < len(levels) and next_right == next_delta:
                indices.append(right)
                right += 1
            found_at_level: list[str] = []
            for level_index in indices:
                found_at_level.extend(
                    candidate
                    for candidate in level_ids[levels[level_index]]
                    if candidate != concept_id
                    and targets[candidate] != target
                    and stats[(family, lang, variant, candidate)][1] != first_id
                )
            if found_at_level:
                if best_delta is None:
                    best_delta = next_delta
                if next_delta == best_delta:
                    tied.extend(found_at_level)
        if not tied:
            raise ValueError(f"No eligible nearest-zipf distractor for {family}/{lang}/{variant}/{concept_id}")
        tied.sort()
        key = f"{key_prefix}|{family}|{lang}|{variant}|{concept_id}"
        distractor_id = tied[int(stream(seed, key).integers(len(tied)))]
        distractor_count, distractor_first = stats[(family, lang, variant, distractor_id)]
        rows.append(
            {
                "family": family,
                "target_lang": lang,
                "target_variant": variant,
                "concept_id": concept_id,
                "distractor_id": distractor_id,
                "kind": kind,
                "n_tied": len(tied),
                "c_content_count": content_count,
                "d_content_count": distractor_count,
                "c_first_id": first_id,
                "d_first_id": distractor_first,
                "abs_dzipf_en": abs(float(concepts[distractor_id]["zipf_en"]) - zipf),
            }
        )
    return rows


def match_distractors(
    concepts: Mapping[str, Mapping[str, Any]],
    stats: Mapping[StatKey, FormStat],
    *,
    family: str,
    lang: str,
    variant: str,
    pool_ids: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Match one distractor per concept using only precomputed tokenizer statistics."""
    return _match_distractors(
        concepts,
        stats,
        family=family,
        lang=lang,
        variant=variant,
        pool_ids=pool_ids,
        seed=seed,
        key_prefix="distractor",
    )


def e8_order_rows(concepts: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Create stratified seeded order rows for the E8 main translation concepts."""
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for concept in concepts:
        by_stratum[str(concept["stratum"])].append(str(concept["concept_id"]))
    rows: list[dict[str, Any]] = []
    for family in ("gemma", "qwen", "llama"):
        for stratum in sorted(by_stratum):
            ordered_ids = sorted(by_stratum[stratum])
            permuted = stream(seed, f"e8|{family}|{stratum}").permutation(len(ordered_ids))
            rows.extend(
                {
                    "family": family,
                    "stratum": stratum,
                    "rank": rank,
                    "concept_id": ordered_ids[int(index)],
                }
                for rank, index in enumerate(permuted)
            )
    return rows


def e8_select(
    order_rows: Sequence[Mapping[str, Any]],
    competent_ids: set[str],
    family: str,
    max_items: int,
) -> list[str]:
    """Select competent concepts by proportional largest-remainder stratum quotas."""
    family_rows = [row for row in order_rows if row["family"] == family]
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for row in sorted(family_rows, key=lambda item: (str(item["stratum"]), int(item["rank"]))):
        concept_id = str(row["concept_id"])
        if concept_id in competent_ids:
            by_stratum[str(row["stratum"])].append(concept_id)
    total = sum(len(ids) for ids in by_stratum.values())
    selected_n = min(max_items, total)
    if selected_n == 0:
        return []
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, ids in by_stratum.items():
        ideal = selected_n * len(ids) / total
        quotas[stratum] = int(ideal)
        remainders.append((ideal - quotas[stratum], stratum))
    remaining = selected_n - sum(quotas.values())
    for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        quotas[stratum] += 1
    selected: list[str] = []
    for stratum in sorted(by_stratum):
        selected.extend(by_stratum[stratum][:quotas[stratum]])
    return selected


def e0_sample(direction_ids: Sequence[str], seed: int, sample_size: int) -> list[dict[str, Any]]:
    """Return a rank-preserving seeded sample of sorted directions concept IDs."""
    ids = sorted(direction_ids)
    if sample_size > len(ids):
        raise ValueError(f"Requested E0 sample of {sample_size} from only {len(ids)} directions IDs")
    permuted = stream(seed, "e0_sample").permutation(len(ids))
    return [
        {"rank": rank, "concept_id": ids[int(index)], "split": "directions"}
        for rank, index in enumerate(permuted[:sample_size])
    ]


def cognate_flag_rows(
    concepts: Any,
    text_cfg: Mapping[str, Any],
    distinct_by_family: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build sorted per-main-concept cognate flags and return the common subset IDs."""
    from .cognates import cognate_free_ids, distance_ok_ids

    main = concepts[(concepts["split"] == "test") & ~concepts["m1_extension"]]
    d_en_fr, d_en_id, distance_ok = distance_ok_ids(main, text_cfg)
    ids = cognate_free_ids(distance_ok, distinct_by_family)
    selected = set(ids)
    distinct_union_ids = {family: set(values) for family, values in distinct_by_family.items()}
    rows = [
        {
            "concept_id": str(row.concept_id),
            "d_en_fr": float(d_en_fr[str(row.concept_id)]),
            "d_en_id": float(d_en_id[str(row.concept_id)]),
            "distance_ok": str(row.concept_id) in distance_ok,
            "distinct_first_gemma": str(row.concept_id) in distinct_union_ids["gemma"],
            "distinct_first_qwen": str(row.concept_id) in distinct_union_ids["qwen"],
            "distinct_first_llama": str(row.concept_id) in distinct_union_ids["llama"],
            "cognate_free": str(row.concept_id) in selected,
        }
        for row in main.itertuples(index=False)
    ]
    rows.sort(key=lambda row: row["concept_id"])
    return rows, ids


def sha256_lines(lines: Sequence[str]) -> str:
    """Hash exact UTF-8 lines already including their intended newlines."""
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def refuse_nonempty_derived(derived_dir: str | Path) -> None:
    """Refuse default generation when the frozen output directory is non-empty."""
    directory = Path(derived_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty derived directory: {directory}")


def compare_directories(left: str | Path, right: str | Path) -> tuple[bool, list[str]]:
    """Compare two generated directory trees byte-for-byte and return differing paths."""
    first, second = Path(left), Path(right)
    first_files = {path.relative_to(first).as_posix(): path for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second).as_posix(): path for path in second.rglob("*") if path.is_file()}
    differences = sorted(set(first_files) ^ set(second_files))
    for relative in sorted(set(first_files) & set(second_files)):
        if first_files[relative].read_bytes() != second_files[relative].read_bytes():
            differences.append(relative)
    differences.sort()
    return not differences, differences
