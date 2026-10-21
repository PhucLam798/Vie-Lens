#!/usr/bin/env python3
"""Audit both pinned tokenizer candidate sets without loading model weights."""

from __future__ import annotations

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

env_guard.guard(offline=True)

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import resource
import struct
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from vilens.cognates import cognate_free_ids, distance_ok_ids, distinct_first_token_ids, id_list_sha256
from vilens.config import config_sha256, load_config
from vilens.data import load_concepts_jsonl, load_prompt_file, prompt_file_paths
from vilens.manifest import sha256_file, verify_manifest
from vilens.models import ModelSpec, SnapshotError, candidate_specs, index_size_info, load_hf_config, load_tokenizer, resolve_snapshot, snapshot_problems, text_dims
from vilens.tokenize import TokenClassifier, encode_joint_batch, ntok_build_rule


LANGUAGES = ("vi", "en", "zh", "fr", "id")
VARIANTS = ("released", "zh_nospace", "vi_nodiac_in_diac")
CHECK_COLUMNS = ("check_id", "description", "expected", "actual", "result")
CHECK_DESCRIPTIONS = {
    "C25": "Registered/base tokenizer vocabulary and size identity; report special-token differences",
    "C26": "All 15 frozen contextual-token columns match on every concept and checkpoint",
    "C27": "Recomputed ntok_vi==1 matches frozen flags and eligibility counts",
    "C28": "Joint pair totals, unique prompts, and duplicate prompt groups",
    "C29": "Released prompt-target prefix property and non-empty content suffix",
    "C30": "Diagnostic zh-no-space and Vietnamese-no-diacritic prompt-target encodings",
    "C31": "Registered/base prompt and joint sequence digest identity",
    "C32": "Token IDs below text vocabulary size and joint sequences within length bound",
    "C33": "Plan section 2.5 usable first-token counts over eligible concepts",
    "C34": "Main-test French/Indonesian first-content token equals English counts",
    "C35": "Plan section 2.6 distance filter, distinct first tokens, shared subset, and composition",
}

_ACTIVE_LOGGER: logging.Logger | None = None
_ACTIVE_RUN: dict[str, Any] = {}


def _json(value: Any) -> str:
    """Serialize values deterministically for CSV output."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest_ids(ids: Sequence[int]) -> str:
    """Return a 16-byte BLAKE2b digest over little-endian uint32 token ids."""
    packed = struct.pack(f"<{len(ids)}I", *(int(token_id) for token_id in ids))
    return hashlib.blake2b(packed, digest_size=16).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write a stable UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(str(item) for item in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _add_check(
    checks: dict[str, dict[str, str]],
    check_id: str,
    description: str,
    expected: Any,
    actual: Any,
    result: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Store a check row and compare serialized expected and actual values."""
    expected_text = expected if isinstance(expected, str) else _json(expected)
    actual_text = actual if isinstance(actual, str) else _json(actual)
    checks[check_id] = {
        "check_id": check_id,
        "description": description,
        "expected": expected_text,
        "actual": actual_text,
        "result": result or ("MATCH" if expected_text == actual_text else "MISMATCH"),
    }
    active_logger = logger or _ACTIVE_LOGGER
    if active_logger is not None:
        row = checks[check_id]
        active_logger.info(
            "%s | %s | expected=%s | actual=%s | %s",
            row["check_id"], row["description"], row["expected"], row["actual"], row["result"],
        )


def _run_check(
    checks: dict[str, dict[str, str]],
    logger: logging.Logger,
    check_id: str,
    description: str,
    expected: Any,
    check: Callable[[], tuple[Any, str]],
) -> dict[str, str]:
    """Run one check independently, recording an ERROR row and traceback on exception."""
    try:
        actual, result = check()
    except Exception as exc:
        logger.exception("Check %s raised an exception", check_id)
        _add_check(
            checks, check_id, description, expected,
            f"{type(exc).__name__}: {exc}", "ERROR", logger,
        )
    else:
        _add_check(checks, check_id, description, expected, actual, result, logger)
    return checks[check_id]


def _ordered_checks(checks: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """Return check rows in numeric order."""
    return [dict(checks[key]) for key in sorted(checks, key=lambda value: int(value[1:]))]


def _stage(name: str, timings: dict[str, float], logger: logging.Logger, function: Any) -> Any:
    """Run an audit stage and record its elapsed wall time."""
    started = time.perf_counter()
    try:
        return function()
    finally:
        timings[name] = time.perf_counter() - started
        logger.info("stage=%s wall_seconds=%.3f", name, timings[name])


def _logger(path: Path) -> logging.Logger:
    """Configure mirrored stdout and timestamped file logging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("x0c_tokenizers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _prompt_records(
    paths: Mapping[str, Path], limit: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """Load prompt files in their released order, optionally retaining only their first records."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    flattened: list[tuple[str, dict[str, Any]]] = []
    for stem, path in paths.items():
        records = load_prompt_file(path)
        if limit is not None:
            records = records[:limit]
        by_file[stem] = records
        flattened.extend((stem, record) for record in records)
    return by_file, flattened


def _prompt_duplicates(
    flattened: Sequence[tuple[str, dict[str, Any]]],
    concepts_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize exact duplicate prompt groups without saving or printing prompt text."""
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    file_prompts: dict[str, Counter[str]] = defaultdict(Counter)
    for stem, record in flattened:
        grouped[record["prompt"]].append((stem, record))
        file_prompts[stem][record["prompt"]] += 1
    per_file = {
        stem: {
            "groups": sum(count > 1 for count in counts.values()),
            "records": sum(count for count in counts.values() if count > 1),
        }
        for stem, counts in sorted(file_prompts.items())
    }
    duplicate_rows: list[dict[str, Any]] = []
    cross_groups: list[dict[str, Any]] = []
    directions_groups: list[dict[str, Any]] = []
    for prompt, members in grouped.items():
        if len(members) < 2:
            continue
        group_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        stems = sorted({stem for stem, _ in members})
        info = {
            "group_id": group_id,
            "files": stems,
            "records": len(members),
            "concept_ids": sorted(record["concept_id"] for _, record in members),
        }
        if len(stems) > 1:
            cross_groups.append(info)
        if stems == ["directions_matched"]:
            directions_groups.append(info)
        for stem, record in members:
            concept = concepts_by_id.get(record["concept_id"], {})
            duplicate_rows.append({
                "group_id": group_id,
                "file": stem,
                "concept_id": record["concept_id"],
                "format": record.get("format", ""),
                "lang": record.get("lang", ""),
                "split": record.get("split", ""),
                "m1_extension": record.get("m1_extension", ""),
                "vi": concept.get("vi", ""),
                "ru": concept.get("ru", ""),
            })
    groups = cross_groups + directions_groups
    return {
        "unique_prompts": len(grouped),
        "per_file": per_file,
        "cross_file": {"groups": len(cross_groups), "records": sum(row["records"] for row in cross_groups)},
        "directions_within": {"groups": len(directions_groups), "records": sum(row["records"] for row in directions_groups)},
    }, duplicate_rows, groups


def _jobs(
    by_file: Mapping[str, Sequence[dict[str, Any]]],
    concept_by_id: Mapping[str, Mapping[str, Any]],
    primary: Mapping[str, str],
) -> Iterator[tuple[tuple[str, str, str, str, str, str], str, str, str]]:
    """Yield stable prompt-target encodings and optional descriptive-table labels."""
    for stem in sorted(by_file):
        is_directions = stem == "directions_matched"
        for record in by_file[stem]:
            prompt = record["prompt"]
            fmt = record["format"]
            cid = record["concept_id"]
            if is_directions:
                lang = record["lang"]
                yield (stem, cid, fmt, lang, lang, "released"), prompt, record["target"], ""
                if lang == "zh":
                    target = record["target"]
                    no_space = target[1:] if target.startswith(" ") else target
                    yield (stem, cid, fmt, lang, lang, "zh_nospace"), prompt, no_space, ""
                continue
            for lang in LANGUAGES:
                descriptor = ""
                if stem == primary["repetition_diac"] and record["split"] == "test" and not record["m1_extension"]:
                    descriptor = lang
                elif stem == primary["repetition_nodiac"] and lang == "vi" and record["split"] == "test" and not record["m1_extension"]:
                    descriptor = "vi_nodiac_file"
                yield (stem, cid, fmt, "", lang, "released"), prompt, record[f"target_{lang}"], descriptor
                if lang == "zh":
                    target = record["target_zh"]
                    no_space = target[1:] if target.startswith(" ") else target
                    yield (
                        (stem, cid, fmt, "", lang, "zh_nospace"),
                        prompt,
                        no_space,
                        "zh_nospace" if descriptor else "",
                    )
                if lang == "vi" and record["condition"] == "diac":
                    target = " " + str(concept_by_id[cid]["vi_nodiac"])
                    yield (
                        (stem, cid, fmt, "", lang, "vi_nodiac_in_diac"),
                        prompt,
                        target,
                        "vi_nodiac_in_diac" if descriptor else "",
                    )


def _chunks(iterator: Iterator[Any], size: int) -> Iterator[list[Any]]:
    """Yield bounded chunks from an iterator."""
    while True:
        chunk: list[Any] = []
        for _ in range(size):
            try:
                chunk.append(next(iterator))
            except StopIteration:
                break
        if not chunk:
            return
        yield chunk


def _pair_shape(
    by_file: Mapping[str, Sequence[dict[str, Any]]],
    concept_by_id: Mapping[str, Mapping[str, Any]],
    primary: Mapping[str, str],
) -> tuple[dict[str, int], int, int]:
    """Count keyed pairs and duplicate keys without invoking a tokenizer."""
    counts = {variant: 0 for variant in VARIANTS}
    seen: set[tuple[str, str, str, str, str, str]] = set()
    duplicate_keys = 0
    for key, _, _, _ in _jobs(by_file, concept_by_id, primary):
        counts[key[-1]] += 1
        if key in seen:
            duplicate_keys += 1
        seen.add(key)
    return counts, len(seen), duplicate_keys


def _special_tokens(tokenizer: Any) -> dict[str, Any]:
    """Collect BOS, EOS, PAD and UNK special token strings and ids."""
    return {
        f"{name}_{field}": getattr(tokenizer, f"{name}_{field}", None)
        for name in ("bos", "eos", "pad", "unk")
        for field in ("token", "token_id")
    }


def _variant_counters() -> dict[str, dict[str, int]]:
    """Create aggregate counters for each encoding variant."""
    fields = (
        "pairs", "prefix_violations", "zero_target", "zero_content", "target_tokens",
        "content_tokens", "continuations", "first_whitespace", "byte_fragment",
        "any_whitespace",
    )
    return {variant: {field: 0 for field in fields} for variant in VARIANTS}


def _checkpoint_audit(
    spec: ModelSpec,
    tokenizer: Any,
    text_vocab_size: int,
    concepts: pd.DataFrame,
    concept_rows: Sequence[Mapping[str, Any]],
    concept_by_id: Mapping[str, Mapping[str, Any]],
    by_file: Mapping[str, Sequence[dict[str, Any]]],
    primary: Mapping[str, str],
    text_cfg: Mapping[str, Any],
    ref_prompt_digests: Mapping[str, str] | None,
    ref_joint_digests: Mapping[tuple[str, str, str, str, str, str], str] | None,
) -> dict[str, Any]:
    """Audit one tokenizer's contextual counts, all joint pairs, and primary records."""
    family = spec.family
    classifier = TokenClassifier(tokenizer)
    ntok_mismatches = {lang: 0 for lang in LANGUAGES}
    vi_one_ids: set[str] = set()
    stored_flag_mismatches = 0
    eligibility_mismatches = 0
    test_ones = 0
    main_ones = 0
    for row in concept_rows:
        cid = str(row["concept_id"])
        computed_vi = 0
        for lang in LANGUAGES:
            count = ntok_build_rule(tokenizer, str(row[lang]), str(text_cfg["ntok_context_prefix"]))
            if count != int(row[f"ntok_{family}_{lang}"]):
                ntok_mismatches[lang] += 1
            if lang == "vi":
                computed_vi = count
                if count == 1:
                    vi_one_ids.add(cid)
        actual_one = computed_vi == 1
        if actual_one != bool(row[f"single_token_vi_{family}"]):
            stored_flag_mismatches += 1
        if row["split"] == "test":
            if actual_one != bool(row[f"m1_eligible_{family}"]):
                eligibility_mismatches += 1
            test_ones += int(actual_one)
            if not bool(row["m1_extension"]):
                main_ones += int(actual_one)

    counters = _variant_counters()
    descriptors = {name: _variant_counters()["released"] for name in (
        "vi", "en", "zh", "zh_nospace", "fr", "id", "vi_nodiac_in_diac", "vi_nodiac_file",
    )}
    max_joint_len = 0
    max_joint_key = ""
    max_token_id = -1
    out_of_vocab = 0
    duplicate_keys = 0
    seen: set[tuple[str, str, str, str, str, str]] = set()
    joint_digests: dict[tuple[str, str, str, str, str, str], str] = {}
    prompt_cache: dict[str, tuple[int, ...]] = {}
    prompt_digest_mismatches = 0
    joint_digest_mismatches = 0
    primary_targets: dict[str, dict[str, dict[str, tuple[int, ...]]]] = defaultdict(lambda: defaultdict(dict))

    for chunk in _chunks(_jobs(by_file, concept_by_id, primary), int(text_cfg["audit_chunk_size"])):
        keys, prompts, targets, desc_names = zip(*chunk, strict=True)
        for key in keys:
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
        encodings = encode_joint_batch(
            tokenizer,
            prompts,
            targets,
            strict=False,
            prompt_cache=prompt_cache,
        )
        for key, prompt, descriptor, encoding in zip(keys, prompts, desc_names, encodings, strict=True):
            variant = key[-1]
            counts = counters[variant]
            counts["pairs"] += 1
            counts["prefix_violations"] += int(not encoding.prefix_ok)
            target_ids = encoding.target_ids
            content_mask = [classifier.is_content(token_id) for token_id in target_ids]
            content_n = sum(content_mask)
            counts["target_tokens"] += len(target_ids)
            counts["content_tokens"] += content_n
            counts["continuations"] += max(content_n - 1, 0)
            counts["zero_target"] += int(len(target_ids) == 0)
            counts["zero_content"] += int(content_n == 0)
            counts["first_whitespace"] += int(bool(target_ids) and not content_mask[0])
            has_byte = any(classifier.is_byte_fragment(token_id) for token_id in target_ids)
            has_whitespace = any(not value for value in content_mask)
            counts["byte_fragment"] += int(has_byte)
            counts["any_whitespace"] += int(has_whitespace)
            if descriptor:
                row = descriptors[descriptor]
                row["pairs"] += 1
                row["target_tokens"] += len(target_ids)
                row["content_tokens"] += content_n
                row["continuations"] += max(content_n - 1, 0)
                row["first_whitespace"] += int(bool(target_ids) and not content_mask[0])
                row["byte_fragment"] += int(has_byte)
                row["any_whitespace"] += int(has_whitespace)
            if len(encoding.joint_ids) > max_joint_len:
                max_joint_len = len(encoding.joint_ids)
                max_joint_key = repr(key)
            if encoding.joint_ids:
                max_token_id = max(max_token_id, max(encoding.joint_ids))
                out_of_vocab += sum(int(token_id) >= text_vocab_size for token_id in encoding.joint_ids)
            digest = _digest_ids(encoding.joint_ids)
            if ref_joint_digests is None:
                joint_digests[key] = digest
            elif ref_joint_digests.get(key) != digest:
                joint_digest_mismatches += 1
            if key[0] == primary["repetition_diac"] and encoding.prefix_ok:
                primary_targets[key[1]][variant][key[4]] = tuple(target_ids)

    prompt_digests = {prompt: _digest_ids(ids) for prompt, ids in prompt_cache.items()}
    if ref_prompt_digests is not None:
        prompt_digest_mismatches = sum(
            ref_prompt_digests.get(prompt) != digest for prompt, digest in prompt_digests.items()
        ) + len(set(ref_prompt_digests) - set(prompt_digests))

    primary_token_ids = {
        int(token_id)
        for variants in primary_targets.values()
        for language_map in variants.values()
        for target_ids in language_map.values()
        for token_id in target_ids
    }
    token_content = {token_id: classifier.is_content(token_id) for token_id in primary_token_ids}
    token_byte_fragment = {token_id: classifier.is_byte_fragment(token_id) for token_id in primary_token_ids}

    for name, row in descriptors.items():
        n_items = row["pairs"]
        if n_items:
            row["n_items"] = n_items
            row["mean_tokens"] = row["target_tokens"] / n_items
            row["mean_content_tokens"] = row["content_tokens"] / n_items
            row["continuation_share"] = row["continuations"] / row["content_tokens"] if row["content_tokens"] else 0.0
            row["first_whitespace_share"] = row["first_whitespace"] / n_items
            row["byte_fragment_share"] = row["byte_fragment"] / n_items
    return {
        "label": spec.label,
        "family": family,
        "set_name": spec.set_name,
        "ntok_mismatches": ntok_mismatches,
        "vi_one_ids": vi_one_ids,
        "stored_flag_mismatches": stored_flag_mismatches,
        "eligibility_mismatches": eligibility_mismatches,
        "test_ones": test_ones,
        "main_ones": main_ones,
        "counters": counters,
        "descriptors": descriptors,
        "max_joint_len": max_joint_len,
        "max_joint_key": max_joint_key,
        "max_token_id": max_token_id,
        "out_of_vocab": out_of_vocab,
        "duplicate_keys": duplicate_keys,
        "joint_digests": joint_digests if ref_joint_digests is None else None,
        "prompt_digests": prompt_digests if ref_prompt_digests is None else None,
        "prompt_digest_mismatches": prompt_digest_mismatches,
        "joint_digest_mismatches": joint_digest_mismatches,
        "prompt_count": len(prompt_cache),
        "pair_count": len(seen),
        "primary_targets": dict(primary_targets),
        "special_tokens": _special_tokens(tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": getattr(tokenizer, "is_fast", False) is True,
        "tokenizer_len": len(tokenizer),
        "token_content": token_content,
        "token_byte_fragment": token_byte_fragment,
    }


def _lstat_tree(roots: Sequence[Path]) -> list[tuple[Path, os.stat_result]]:
    """Return all existing cache entries and their lstat metadata without following symlinks."""
    found: list[tuple[Path, os.stat_result]] = []
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        path = stack.pop()
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        found.append((path, info))
        if os.path.isdir(path) and not os.path.islink(path):
            try:
                stack.extend(Path(entry.path) for entry in os.scandir(path))
            except OSError:
                pass
    return found


def _cache_roots(cache: Path, specs: Sequence[ModelSpec]) -> list[Path]:
    """Return repository and lock roots for all six candidates."""
    roots: list[Path] = []
    for spec in specs:
        name = "models--" + spec.hf_id.replace("/", "--")
        roots.extend((cache / name, cache / ".locks" / name))
    return roots


def _first_usable_counts(audit: Mapping[str, Any], concepts_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Count usable first target tokens for the five readouts defined in plan section 2.5."""
    token_content = audit["token_content"]
    token_byte_fragment = audit["token_byte_fragment"]
    family = str(audit["family"])
    result = {name: 0 for name in ("en", "fr", "id", "zh_released", "zh_nospace")}
    eligible_ids = [
        cid for cid, row in concepts_by_id.items()
        if row.get("m1_eligible_" + family) is not None
        and row.get("m1_eligible_" + family) is not pd.NA
        and bool(row.get("m1_eligible_" + family))
    ]
    for cid in eligible_ids:
        variants = audit["primary_targets"].get(cid, {})
        released = variants.get("released", {})
        no_space_zh = variants.get("zh_nospace", {}).get("zh", ())
        first = {lang: (released.get(lang, ()) or (None,))[0] for lang in LANGUAGES}
        options = {
            "en": (first["en"], [first[lang] for lang in ("vi", "fr", "id", "zh")]),
            "fr": (first["fr"], [first[lang] for lang in ("vi", "en", "id", "zh")]),
            "id": (first["id"], [first[lang] for lang in ("vi", "en", "fr", "zh")]),
            "zh_released": (first["zh"], [first[lang] for lang in ("vi", "en", "fr", "id")]),
            "zh_nospace": ((no_space_zh or (None,))[0], [first[lang] for lang in ("vi", "en", "fr", "id")]),
        }
        for name, (token_id, others) in options.items():
            if token_id is None or not token_content.get(token_id, False) or token_byte_fragment.get(token_id, False):
                continue
            if all(token_id != other for other in others):
                result[name] += 1
    return result


def _main_mask(concepts: pd.DataFrame) -> pd.Series:
    """Return the main-test row mask."""
    return (concepts["split"] == "test") & (~concepts["m1_extension"].fillna(False))


def _first_content_id(token_ids: Sequence[int], audit: Mapping[str, Any]) -> int | None:
    """Return the first primary-target id marked as content for this checkpoint."""
    token_content = audit["token_content"]
    for token_id in token_ids:
        if token_content.get(int(token_id), False):
            return int(token_id)
    return None


def _cognate_checks(
    concepts: pd.DataFrame,
    concepts_by_id: Mapping[str, Mapping[str, Any]],
    by_file: Mapping[str, Sequence[dict[str, Any]]],
    audits: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute all C35 distance, distinct-token, shared-subset and composition counts."""
    facts = cfg["reference_facts"]["x0c"]
    text_cfg = cfg["text"]
    main = concepts.loc[_main_mask(concepts)].copy()
    d_en_fr, d_en_id, distance_ok = distance_ok_ids(main, text_cfg)
    distinct_by_variant: dict[tuple[str, str], set[str]] = {}
    rows: list[dict[str, Any]] = []
    for label, audit in audits.items():
        token_content = audit["token_content"]
        family = str(audit["family"])
        for zh_variant in ("released", "zh_nospace"):
            first_ids: dict[str, dict[str, int | None]] = {}
            for cid in distance_ok:
                variants = audit["primary_targets"].get(cid, {})
                released = variants.get("released", {})
                lang_ids: dict[str, int | None] = {}
                for lang in LANGUAGES:
                    target_ids = released.get(lang, ())
                    if lang == "zh" and zh_variant == "zh_nospace":
                        target_ids = variants.get("zh_nospace", {}).get("zh", ())
                    lang_ids[lang] = _first_content_id(target_ids, audit) if token_content else None
                first_ids[cid] = lang_ids
            distinct = distinct_first_token_ids(first_ids)
            distinct_by_variant[(label, zh_variant)] = distinct
            expected = int(facts["cognate_distinct_first_tokens"][family])
            rows.append({
                "section": "distinct_first_tokens",
                "label": label,
                "metric": zh_variant,
                "actual": len(distinct),
                "expected": expected,
                "result": "MATCH" if len(distinct) == expected else "MISMATCH",
            })

    subset_by_variant: dict[tuple[str, str], list[str]] = {}
    for set_name in ("registered", "base"):
        for zh_variant in ("released", "zh_nospace"):
            family_sets = {
                family: distinct_by_variant[(f"{set_name}/{family}", zh_variant)]
                for family in ("gemma", "qwen", "llama")
            }
            subset = cognate_free_ids(distance_ok, family_sets)
            subset_by_variant[(set_name, zh_variant)] = subset
            digest = id_list_sha256(subset)
            expected_n = int(cfg["reference_facts"]["cognate_free_subset"])
            expected_sha = str(facts["cognate_free_sha256"])
            rows.append({
                "section": "cognate_free_subset",
                "label": set_name,
                "metric": zh_variant,
                "actual": f"{len(subset)}; sha256={digest}",
                "expected": f"{expected_n}; sha256={expected_sha}",
                "result": "MATCH" if len(subset) == expected_n and digest == expected_sha else "MISMATCH",
            })
    identical = (
        subset_by_variant[("registered", "released")]
        == subset_by_variant[("registered", "zh_nospace")]
        == subset_by_variant[("base", "released")]
        == subset_by_variant[("base", "zh_nospace")]
    )
    rows.append({
        "section": "cognate_free_subset",
        "label": "all",
        "metric": "identical_across_sets_and_zh_variants",
        "actual": identical,
        "expected": True,
        "result": "MATCH" if identical else "MISMATCH",
    })
    subset = set(subset_by_variant[("registered", "released")])
    main_ids = set(main["concept_id"].astype(str))
    coverage: dict[str, int] = {}
    for stem, expected_count in facts["cognate_free_coverage"].items():
        count = sum(record["concept_id"] in subset and record["concept_id"] in main_ids for record in by_file[stem])
        coverage[stem] = count
        rows.append({
            "section": "coverage",
            "label": stem,
            "metric": "main_records_in_subset",
            "actual": count,
            "expected": int(expected_count),
            "result": "MATCH" if count == int(expected_count) else "MISMATCH",
        })
    selected = [concepts_by_id[cid] for cid in subset if cid in main_ids]
    strata = Counter(str(row["stratum"]) for row in selected)
    pos = Counter(str(row["pos"]) for row in selected)
    for section, counts, expected_counts in (
        ("stratum", strata, facts["cognate_free_strata"]),
        ("pos", pos, facts["cognate_free_pos"]),
    ):
        for key in sorted(set(counts) | set(expected_counts)):
            actual = int(counts.get(key, 0))
            expected = int(expected_counts[key])
            rows.append({
                "section": section, "label": key, "metric": "count",
                "actual": actual, "expected": expected,
                "result": "MATCH" if actual == expected else "MISMATCH",
            })
    threshold = float(text_cfg["cognate_max_distance"])
    distance_counts = {
        "distance_ok_both": len(distance_ok),
        "en_fr_below_threshold": sum(value < threshold for value in d_en_fr.values()),
        "en_fr_surface_distance_zero": sum(value == 0.0 for value in d_en_fr.values()),
        "raw_en_equals_fr": sum(str(row.en) == str(row.fr) for row in main.itertuples(index=False)),
        "en_id_below_threshold": sum(value < threshold for value in d_en_id.values()),
        "en_id_surface_distance_zero": sum(value == 0.0 for value in d_en_id.values()),
    }
    rows.append({
        "section": "surface_distance", "label": "main", "metric": "distance_ok_both",
        "actual": distance_counts["distance_ok_both"], "expected": int(facts["cognate_distance_ok"]),
        "result": "MATCH" if distance_counts["distance_ok_both"] == int(facts["cognate_distance_ok"]) else "MISMATCH",
    })
    for metric in ("en_fr_below_threshold", "en_fr_surface_distance_zero", "raw_en_equals_fr", "en_id_below_threshold", "en_id_surface_distance_zero"):
        rows.append({
            "section": "surface_distance", "label": "main", "metric": metric,
            "actual": distance_counts[metric], "expected": "ungated", "result": "INFO",
        })
    return {
        "distance_counts": distance_counts,
        "distinct_counts": {
            label: {variant: len(distinct_by_variant[(label, variant)]) for variant in ("released", "zh_nospace")}
            for label in audits
        },
        "subset_count": len(subset),
        "subset_sha256": id_list_sha256(sorted(subset)),
        "subset_identical": identical,
        "coverage": coverage,
        "strata": {key: int(strata.get(key, 0)) for key in sorted(set(strata) | set(facts["cognate_free_strata"]))},
        "pos": {key: int(pos.get(key, 0)) for key in sorted(set(pos) | set(facts["cognate_free_pos"]))},
    }, rows


def _rss_mb() -> float:
    """Return process peak RSS in MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _output_directory(smoke: int | None, run_stamp: str) -> Path:
    """Choose reports/ for a full run or an isolated logs/ directory for smoke mode."""
    if smoke is not None:
        return REPO / "logs" / f"x0c_smoke_{run_stamp}"
    return REPO / "reports"


def _positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse the experiment config path and optional bounded smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--smoke", type=_positive_int, default=None, metavar="N")
    return parser.parse_args()


def _run_audit(args: argparse.Namespace) -> int:
    """Run the tokenizer audit, write its tables and return nonzero on a failed gate."""
    config_path = Path(args.config)
    cfg = load_config(config_path)
    cache = Path(cfg["paths"]["model_cache"])
    if os.path.realpath(cache) != os.path.realpath(os.environ["HF_HUB_CACHE"]):
        raise SystemExit("paths.model_cache does not resolve to HF_HUB_CACHE")
    overall_start = time.perf_counter()
    run_start = datetime.now(timezone.utc)
    run_start_ns = time.time_ns()
    run_stamp = run_start.strftime("%Y%m%dT%H%M%SZ")
    output_dir = _output_directory(args.smoke, run_stamp)
    if args.smoke is not None:
        logger_path = output_dir / "x0c_tokenizers.log"
    else:
        logger_path = REPO / cfg["paths"]["logs"] / f"x0c_tokenizers_{run_stamp}.log"
    logger = _logger(logger_path)
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger
    logger.info("T03a tokenizer-only audit start=%s", run_start.isoformat())
    timings: dict[str, float] = {}
    checks: dict[str, dict[str, str]] = {}
    specs = candidate_specs(cfg)
    prompt_paths = prompt_file_paths(cfg)
    by_file: dict[str, list[dict[str, Any]]] = {}
    flattened: list[tuple[str, dict[str, Any]]] = []
    concepts: pd.DataFrame | None = None
    concept_rows: list[dict[str, Any]] = []
    concept_by_id: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, Any] = {}
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    _ACTIVE_RUN.clear()
    _ACTIVE_RUN.update({
        "args": args, "cfg": cfg, "config_path": config_path, "cache": cache,
        "overall_start": overall_start, "run_start": run_start,
        "run_start_ns": run_start_ns, "run_stamp": run_stamp,
        "output_dir": output_dir, "logger": logger, "timings": timings,
        "checks": checks, "specs": specs, "prompt_paths": prompt_paths,
        "by_file": by_file, "flattened": flattened, "concepts": concepts,
        "concept_rows": concept_rows, "concept_by_id": concept_by_id,
        "duplicates": duplicates, "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups, "snapshots": {},
        "snapshot_sizes": {}, "tokenizer_rows": {}, "config_rows": {},
        "c33_rows": [], "c34_rows": [], "c35_rows": [],
        "descriptor_rows": [], "outputs_written": False,
    })

    def load_inputs() -> bool:
        """Verify the selected manifest entries and load JSONL concepts and prompts."""
        nonlocal concepts, concept_rows, concept_by_id, by_file, flattened, duplicates, duplicate_rows, duplicate_groups
        relative_prompts = {
            stem: (path if path.is_absolute() else (REPO / path)).resolve()
            for stem, path in prompt_paths.items()
        }
        selected = [cfg["paths"]["concepts_jsonl"]]
        selected.extend(path.relative_to(REPO).as_posix() for path in relative_prompts.values())
        try:
            def check_manifest() -> tuple[Any, str]:
                verified = verify_manifest(REPO / cfg["paths"]["manifest"], REPO, only=selected)
                expected_n = int(cfg["reference_facts"]["x0c"]["inputs_verified"])
                ok = len(verified) == expected_n and all(value for _, value in verified)
                actual = f"{sum(value for _, value in verified)}/{len(verified)} OK, {sum(not value for _, value in verified)} FAILED"
                return actual, "MATCH" if ok else "MISMATCH"

            c21 = _run_check(
                checks, logger, "C21", "Manifest verification of concepts JSONL and 27 prompt files",
                "28/28 OK", check_manifest,
            )
            ok = c21["result"] == "MATCH"
        except Exception as exc:
            # This catch protects setup around the check runner; the row is still isolated.
            logger.exception("C21 setup failed")
            ok = False
            _add_check(checks, "C21", "Manifest verification of concepts JSONL and 27 prompt files", "28/28 OK", f"{type(exc).__name__}: {exc}", "ERROR", logger)
        try:
            concepts = load_concepts_jsonl(REPO / cfg["paths"]["concepts_jsonl"])
            concept_rows = concepts.to_dict(orient="records")
            concept_by_id = {str(row["concept_id"]): row for row in concept_rows}
            by_file, flattened = _prompt_records(relative_prompts, args.smoke)
        except Exception:
            logger.exception("Loading verified tokenizer audit inputs failed")
            return False
        try:
            duplicates, duplicate_rows, duplicate_groups = _prompt_duplicates(flattened, concept_by_id)
            _ACTIVE_RUN.pop("duplicate_error", None)
        except Exception as exc:
            logger.exception("Prompt duplicate inventory failed")
            duplicates, duplicate_rows, duplicate_groups = {}, [], []
            _ACTIVE_RUN["duplicate_error"] = f"{type(exc).__name__}: {exc}"
        _ACTIVE_RUN.update({
            "by_file": by_file, "flattened": flattened, "concepts": concepts,
            "concept_rows": concept_rows, "concept_by_id": concept_by_id,
            "duplicates": duplicates, "duplicate_rows": duplicate_rows,
            "duplicate_groups": duplicate_groups,
        })
        return ok

    input_ok = _stage("manifest_and_input_load", timings, logger, load_inputs)
    if input_ok and concepts is not None and len(by_file) == 27:
        def check_c28() -> tuple[Any, str]:
            if _ACTIVE_RUN.get("duplicate_error"):
                raise RuntimeError(str(_ACTIVE_RUN["duplicate_error"]))
            shape_counts, shape_pairs, shape_duplicate_keys = _pair_shape(
                by_file, concept_by_id, cfg["data"]["primary_prompt_files"],
            )
            expected_shape = {key: int(value) for key, value in cfg["reference_facts"]["x0c"]["joint_encodings"].items()}
            expected_pair_total = sum(expected_shape.values())
            expected_duplicates = cfg["reference_facts"]["x0c"]["prompt_duplicates"]
            shape_ok = (
                shape_counts == expected_shape
                and shape_pairs == expected_pair_total
                and shape_duplicate_keys == 0
                and duplicates.get("unique_prompts") == int(cfg["reference_facts"]["x0c"]["unique_prompts"])
            )
            shape_ok &= all(
                counts == expected_duplicates["translation_diac_per_set"]
                for stem, counts in duplicates["per_file"].items()
                if stem.startswith("translation_diac_")
            )
            shape_ok &= all(
                counts == expected_duplicates["translation_nodiac_per_set"]
                for stem, counts in duplicates["per_file"].items()
                if stem.startswith("translation_nodiac_")
            )
            shape_ok &= all(
                counts == expected_duplicates["repetition_and_cloze_per_file"]
                for stem, counts in duplicates["per_file"].items()
                if stem.startswith(("repetition_", "cloze_"))
            )
            shape_ok &= duplicates.get("directions_within") == expected_duplicates["directions_within"]
            shape_ok &= duplicates.get("cross_file") == expected_duplicates["cross_file"]
            per_checkpoint_shape = {
                spec.label: {"pairs": shape_pairs, "variants": shape_counts}
                for spec in candidate_specs(cfg)
            }
            actual = {
                "unique_prompts": duplicates.get("unique_prompts"),
                "per_checkpoint": per_checkpoint_shape,
                "duplicate_keys": shape_duplicate_keys,
                "per_file": duplicates.get("per_file"),
                "directions_within": duplicates.get("directions_within"),
                "cross_file": duplicates.get("cross_file"),
            }
            return actual, "MATCH" if shape_ok else "MISMATCH"

        _stage("pair_prompt_structure", timings, logger, lambda: _run_check(
            checks, logger, "C28", CHECK_DESCRIPTIONS["C28"],
            "configured pair, prompt, and duplicate counts", check_c28,
        ))
    else:
        _add_check(
            checks, "C28", CHECK_DESCRIPTIONS["C28"],
            "configured pair, prompt, and duplicate counts",
            "NOT RUN: verified prompt inputs unavailable", "NOT RUN", logger,
        )
    snapshots: dict[str, Path] = {}
    snapshot_sizes: dict[str, dict[str, int | None]] = {}
    snapshot_issues: dict[str, str] = {}

    def audit_snapshots() -> tuple[Any, str]:
        """Resolve every pinned revision and check snapshot metadata/shards without reading weights."""
        complete = 0
        unexpected: list[str] = []
        for spec in specs:
            try:
                snapshot = resolve_snapshot(spec, cache)
                problems = snapshot_problems(snapshot)
                snapshots[spec.label] = snapshot
                snapshot_sizes[spec.label] = index_size_info(snapshot) or {
                    "index_total_size": None,
                    "shard_bytes_sum": None,
                }
                if problems:
                    snapshot_issues[spec.label] = "; ".join(problems)
                else:
                    complete += 1
            except SnapshotError as exc:
                snapshot_issues[spec.label] = str(exc)
            except Exception as exc:
                logger.exception("Unexpected snapshot audit failure: %s", spec.label)
                snapshot_issues[spec.label] = f"{type(exc).__name__}: {exc}"
                unexpected.append(f"{spec.label}: {type(exc).__name__}: {exc}")
        actual = f"{complete}/6 complete"
        if snapshot_issues:
            actual += "; " + " | ".join(f"{key}: {value}" for key, value in sorted(snapshot_issues.items()))
        if unexpected:
            raise RuntimeError("; ".join(unexpected))
        return actual, "MATCH" if complete == 6 else "MISMATCH"

    _stage("snapshot_audit", timings, logger, lambda: _run_check(
        checks, logger, "C22", "6/6 snapshots resolve at the pinned revision; snapshot_problems empty (OPS-6 criteria)",
        "6/6 complete", audit_snapshots,
    ))
    all_snapshots_ready = len(snapshots) == 6 and not snapshot_issues
    models_by_label = {spec.label: spec for spec in specs}
    audits: dict[str, dict[str, Any]] = {}
    config_rows: dict[str, dict[str, Any]] = {}
    tokenizer_rows: dict[str, dict[str, Any]] = {}
    identity: dict[str, dict[str, Any]] = {}
    component_errors: dict[str, str] = {}
    vocab_diff_details: list[str] = []
    primary = cfg["data"]["primary_prompt_files"]
    ref = cfg["reference_facts"]
    facts = ref["x0c"]
    audit_concept_rows = concept_rows if args.smoke is None else concept_rows[:args.smoke]
    _ACTIVE_RUN.update({
        "snapshots": snapshots, "snapshot_sizes": snapshot_sizes,
        "snapshot_issues": snapshot_issues, "audits": audits,
        "config_rows": config_rows, "tokenizer_rows": tokenizer_rows,
        "identity": identity, "component_errors": component_errors,
        "vocab_diff_details": vocab_diff_details,
    })

    def run_model_audits() -> None:
        """Load tokenizers/configs in family pairs and audit each tokenizer before releasing it."""
        if not all_snapshots_ready:
            for check_id in ("C23", "C24", "C25", "C26", "C27", "C29", "C30", "C31", "C32", "C33", "C34", "C35"):
                _add_check(checks, check_id, CHECK_DESCRIPTIONS.get(check_id, check_id), "configured result", "NOT RUN: C22 snapshot prerequisite failed", "NOT RUN")
            return
        if not input_ok or concepts is None or len(by_file) != 27:
            for check_id in ("C23", "C24", "C25", "C26", "C27", "C29", "C30", "C31", "C32", "C33", "C34", "C35"):
                _add_check(checks, check_id, CHECK_DESCRIPTIONS.get(check_id, check_id), "configured result", "NOT RUN: verified data prerequisite failed", "NOT RUN")
            return

        xcolon_problems: list[str] = []
        length_problems: list[str] = []
        dims_problems: list[str] = []
        fast_count = 0
        for family in ("gemma", "qwen", "llama"):
            reg_vocab: dict[str, int] | None = None
            reg_refs_prompt: Mapping[str, str] | None = None
            reg_refs_joint: Mapping[tuple[str, str, str, str, str, str], str] | None = None
            reg_pair_count = 0
            reg_prompt_count = 0
            family_identity = {"vocab_equal": False, "prompt_mismatches": -1, "joint_mismatches": -1, "prompt_count": 0, "joint_count": 0}
            for set_name in ("registered", "base"):
                spec = models_by_label[f"{set_name}/{family}"]
                try:
                    tokenizer = load_tokenizer(spec, cache)
                    vocab = tokenizer.get_vocab()
                    xcolon = [int(token_id) for token_id in tokenizer("x:")["input_ids"]]
                    expected_xcolon = [int(value) for value in facts["tokenizer_x_colon_ids"][family]]
                    if xcolon != expected_xcolon:
                        xcolon_problems.append(f"{spec.label}: {xcolon} expected {expected_xcolon}")
                    expected_len = int(facts["tokenizer_len"][family])
                    if len(tokenizer) != expected_len:
                        length_problems.append(f"{spec.label}: len={len(tokenizer)} expected={expected_len}")
                    fast_count += int(getattr(tokenizer, "is_fast", False) is True)
                    spec_row = {
                        "label": spec.label, "hf_id": spec.hf_id, "revision": spec.revision,
                        "snapshot_path": str(snapshots[spec.label]),
                        "tokenizer_class": type(tokenizer).__name__,
                        "is_fast": getattr(tokenizer, "is_fast", False) is True,
                        "tokenizer_len": len(tokenizer),
                        "x_colon_ids": _json(xcolon),
                        **_special_tokens(tokenizer),
                    }
                    tokenizer_rows[spec.label] = spec_row
                except Exception as exc:
                    component_errors[spec.label] = f"tokenizer: {type(exc).__name__}: {exc}"
                    logger.exception("Tokenizer load failed: %s", spec.label)
                    continue
                try:
                    hf_config = load_hf_config(spec, cache)
                    dims = text_dims(hf_config)
                    expected_dims = tuple(int(value) for value in facts["model_dims"][family])
                    if dims != expected_dims:
                        dims_problems.append(f"{spec.label}: dims={dims}, expected={expected_dims}")
                    text_config = hf_config.get_text_config()
                    config_row = {
                        "label": spec.label,
                        "hf_id": spec.hf_id,
                        "revision": spec.revision,
                        "model_type": getattr(hf_config, "model_type", ""),
                        "architectures": _json(getattr(hf_config, "architectures", None)),
                        "N": dims[0], "d": dims[1], "vocab_size": dims[2],
                        "tie_word_embeddings": getattr(hf_config, "tie_word_embeddings", None),
                        "final_logit_softcapping": getattr(text_config, "final_logit_softcapping", getattr(hf_config, "final_logit_softcapping", None)),
                        "sliding_window": getattr(text_config, "sliding_window", getattr(hf_config, "sliding_window", None)),
                    }
                    config_rows[spec.label] = config_row
                    tokenizer_rows[spec.label].update(config_row)
                except Exception as exc:
                    component_errors[spec.label] = f"config: {type(exc).__name__}: {exc}"
                    logger.exception("Config load failed: %s", spec.label)
                    continue

                try:
                    audit = _checkpoint_audit(
                        spec, tokenizer, dims[2], concepts, audit_concept_rows, concept_by_id, by_file,
                        primary, cfg["text"], reg_refs_prompt if set_name == "base" else None,
                        reg_refs_joint if set_name == "base" else None,
                    )
                except Exception as exc:
                    component_errors[spec.label] = f"audit: {type(exc).__name__}: {exc}"
                    logger.exception("Tokenizer audit failed: %s", spec.label)
                    del tokenizer
                    del hf_config
                    continue
                audits[spec.label] = audit
                if set_name == "registered":
                    reg_vocab = vocab
                    reg_refs_prompt = audit["prompt_digests"]
                    reg_refs_joint = audit["joint_digests"]
                    reg_pair_count = int(audit["pair_count"])
                    reg_prompt_count = int(audit["prompt_count"])
                else:
                    family_identity["vocab_equal"] = reg_vocab == vocab if reg_vocab is not None else False
                    family_identity["prompt_mismatches"] = int(audit["prompt_digest_mismatches"])
                    family_identity["joint_mismatches"] = int(audit["joint_digest_mismatches"])
                    family_identity["prompt_count"] = min(reg_prompt_count, int(audit["prompt_count"]))
                    family_identity["joint_count"] = min(reg_pair_count, int(audit["pair_count"]))
                del tokenizer
                del hf_config
            if f"registered/{family}" in audits:
                audits[f"registered/{family}"].pop("prompt_digests", None)
                audits[f"registered/{family}"].pop("joint_digests", None)
            identity[family] = family_identity

        def check_c23() -> tuple[Any, str]:
            tokenizer_errors = [f"{label}: {error}" for label, error in component_errors.items() if error.startswith("tokenizer:")]
            if tokenizer_errors:
                raise RuntimeError("; ".join(tokenizer_errors))
            ok = fast_count == 6 and not xcolon_problems and not length_problems and not any(
                error.startswith("tokenizer:") for error in component_errors.values()
            )
            actual = f"{fast_count}/6 fast"
            if xcolon_problems:
                actual += "; x: " + " | ".join(xcolon_problems)
            if length_problems:
                actual += "; tokenizer lengths " + " | ".join(length_problems)
            return actual, "MATCH" if ok else "MISMATCH"

        def check_c24() -> tuple[Any, str]:
            config_errors = [f"{label}: {error}" for label, error in component_errors.items() if error.startswith("config:")]
            if config_errors:
                raise RuntimeError("; ".join(config_errors))
            ok = len(config_rows) == 6 and not dims_problems and not any(
                error.startswith("config:") for error in component_errors.values()
            )
            actual = f"{len(config_rows)}/6 configs; dimensions match={not dims_problems}"
            if dims_problems:
                actual += "; " + " | ".join(dims_problems)
            return actual, "MATCH" if ok else "MISMATCH"

        def check_c25() -> tuple[Any, str]:
            for family in ("gemma", "qwen", "llama"):
                reg = tokenizer_rows.get(f"registered/{family}", {})
                base = tokenizer_rows.get(f"base/{family}", {})
                for field in ("bos_token", "bos_token_id", "eos_token", "eos_token_id", "pad_token", "pad_token_id", "unk_token", "unk_token_id"):
                    if reg.get(field) != base.get(field):
                        detail = f"{family} {field}: registered={reg.get(field)!r}; base={base.get(field)!r}"
                        if detail not in vocab_diff_details:
                            vocab_diff_details.append(detail)
            non_eos_differences = [item for item in vocab_diff_details if "eos_token" not in item]
            lengths_match = len(tokenizer_rows) == 6 and all(
                int(tokenizer_rows[label]["tokenizer_len"]) == int(facts["tokenizer_len"][label.split("/")[1]])
                for label in tokenizer_rows
            )
            ok = (
                all(bool(identity.get(family, {}).get("vocab_equal")) for family in ("gemma", "qwen", "llama"))
                and lengths_match
                and not non_eos_differences
            )
            actual = (
                f"vocabulary equality={sum(bool(identity.get(family, {}).get('vocab_equal')) for family in ('gemma','qwen','llama'))}/3; "
                f"lengths correct={sum(int(row['tokenizer_len']) == int(facts['tokenizer_len'][row['label'].split('/')[1]]) for row in tokenizer_rows.values())}/6; "
                f"special differences={' | '.join(vocab_diff_details) if vocab_diff_details else 'none'}"
            )
            return actual, "MATCH" if ok else "MISMATCH"

        _run_check(checks, logger, "C23", "Six fast tokenizers, expected x-colon ids and expected tokenizer lengths", "6/6 fast; x: IDs and lengths match", check_c23)
        _run_check(checks, logger, "C24", "Text config dimensions and checkpoint metadata", "6/6 dimensions match", check_c24)
        _run_check(checks, logger, "C25", "Registered/base tokenizer vocabulary and size identity; report special-token differences", "Vocabulary maps and lengths equal; eos-only difference allowed", check_c25)
        if len(audits) != 6:
            for check_id in ("C26", "C27", "C29", "C30", "C31", "C32", "C33", "C34", "C35"):
                _add_check(checks, check_id, CHECK_DESCRIPTIONS[check_id], "configured result", "NOT RUN: six tokenizer audits did not complete", "NOT RUN", logger)
            return

        def check_c26() -> tuple[Any, str]:
            details = [f"{spec.label}/{lang}={count}" for spec in specs for lang, count in audits[spec.label]["ntok_mismatches"].items()]
            return "; ".join(details), "MATCH" if all(audits[s.label]["ntok_mismatches"][lang] == 0 for s in specs for lang in LANGUAGES) else "MISMATCH"

        def check_c27() -> tuple[Any, str]:
            details = []
            ok = True
            for spec in specs:
                audit = audits[spec.label]
                family = spec.family
                matches = (audit["stored_flag_mismatches"] == 0 and audit["eligibility_mismatches"] == 0
                           and audit["test_ones"] == int(ref["m1_eligible"][family])
                           and audit["main_ones"] == int(ref["single_token_vi_main"][family]))
                ok &= matches
                details.append(f"{spec.label}: stored={audit['stored_flag_mismatches']}; eligible={audit['eligibility_mismatches']}; test/main={audit['test_ones']}/{audit['main_ones']}")
            return "; ".join(details), "MATCH" if ok else "MISMATCH"

        def check_c29() -> tuple[Any, str]:
            actual = {label: {"pairs": a["counters"]["released"]["pairs"], "prefix_violations": a["counters"]["released"]["prefix_violations"], "zero_target_tokens": a["counters"]["released"]["zero_target"], "zero_content_tokens": a["counters"]["released"]["zero_content"]} for label, a in audits.items()}
            ok = all(x["prefix_violations"] == int(ref["tokenizer_prefix_violations"]) and x["zero_target_tokens"] == 0 and x["zero_content_tokens"] == 0 for x in actual.values())
            return actual, "MATCH" if ok else "MISMATCH"

        def check_c30() -> tuple[Any, str]:
            actual = {label: {variant: {"pairs": a["counters"][variant]["pairs"], "prefix_violations": a["counters"][variant]["prefix_violations"], "zero_content_tokens": a["counters"][variant]["zero_content"]} for variant in ("zh_nospace", "vi_nodiac_in_diac")} for label, a in audits.items()}
            ok = all(row["prefix_violations"] == 0 and row["zero_content_tokens"] == 0 for variants in actual.values() for row in variants.values())
            return actual, "MATCH" if ok else "MISMATCH"

        def check_c31() -> tuple[Any, str]:
            expected_variants = facts["joint_encodings"]
            actual = {family: identity.get(family, {}) for family in ("gemma", "qwen", "llama")}
            expected_joint_count = sum(int(count) for count in expected_variants.values())
            ok = all(row.get("prompt_mismatches") == 0 and row.get("joint_mismatches") == 0
                     and row.get("prompt_count") == int(facts["unique_prompts"])
                     and row.get("joint_count") == expected_joint_count for row in actual.values())
            return actual, "MATCH" if ok else "MISMATCH"

        def check_c32() -> tuple[Any, str]:
            bound = int(facts["max_joint_len_bound"])
            detail = {label: {"out_of_vocabulary_ids": a["out_of_vocab"], "max_joint_len": a["max_joint_len"], "record": a["max_joint_key"], "max_token_id": a["max_token_id"]} for label, a in audits.items()}
            ok = all(a["out_of_vocab"] == 0 and a["max_joint_len"] <= bound for a in audits.values())
            return detail, "MATCH" if ok else "MISMATCH"

        c33_rows: list[dict[str, Any]] = []
        c34_rows: list[dict[str, Any]] = []
        c35_rows: list[dict[str, Any]] = []
        c35_result: dict[str, Any] | None = None

        def check_c33() -> tuple[Any, str]:
            ok = True
            names = ("en", "fr", "id", "zh_released", "zh_nospace")
            for spec in specs:
                counts = _first_usable_counts(audits[spec.label], concept_by_id)
                for index, name in enumerate(names):
                    expected = int(facts["m1_usable_first_token"][spec.family][index])
                    actual = counts[name]
                    result = "MATCH" if actual == expected else "MISMATCH"
                    c33_rows.append({"label": spec.label, "family": spec.family, "readout": name, "usable_count": actual, "expected": expected, "result": result})
                    ok &= result == "MATCH"
            grouped = {label: {row["readout"]: row["usable_count"] for row in c33_rows if row["label"] == label} for label in audits}
            return grouped, "MATCH" if ok else "MISMATCH"

        def check_c34() -> tuple[Any, str]:
            ok = True
            actual_by_label: dict[str, dict[str, int]] = {}
            for spec in specs:
                audit = audits[spec.label]
                counts = {"fr": 0, "id": 0}
                for cid, concept in concept_by_id.items():
                    if concept["split"] != "test" or bool(concept["m1_extension"]):
                        continue
                    targets = audit["primary_targets"].get(cid, {}).get("released", {})
                    en_token = _first_content_id(targets.get("en", ()), audit)
                    if en_token is None:
                        continue
                    for lang in ("fr", "id"):
                        token = _first_content_id(targets.get(lang, ()), audit)
                        if token is not None and token == en_token:
                            counts[lang] += 1
                actual_by_label[spec.label] = counts
                for lang in ("fr", "id"):
                    expected = int(facts["first_content_equals_en"][lang][spec.family])
                    result = "MATCH" if counts[lang] == expected else "MISMATCH"
                    c34_rows.append({"label": spec.label, "family": spec.family, "language": lang, "actual": counts[lang], "expected": expected, "result": result})
                    ok &= result == "MATCH"
            return actual_by_label, "MATCH" if ok else "MISMATCH"

        def check_c35() -> tuple[Any, str]:
            nonlocal c35_result, c35_rows
            c35_result, c35_rows = _cognate_checks(concepts, concept_by_id, by_file, audits, cfg)
            ok = all(row["result"] in {"MATCH", "INFO"} for row in c35_rows)
            return c35_result, "MATCH" if ok else "MISMATCH"

        _run_check(checks, logger, "C26", CHECK_DESCRIPTIONS["C26"], "0 mismatches per checkpoint x language", check_c26)
        _run_check(checks, logger, "C27", CHECK_DESCRIPTIONS["C27"], "0 flag/mask mismatches; test 186/166/162; main 108/96/96 per family", check_c27)
        _run_check(checks, logger, "C29", CHECK_DESCRIPTIONS["C29"], "0 prefix violations; every target has at least one target and content token", check_c29)
        _run_check(checks, logger, "C30", CHECK_DESCRIPTIONS["C30"], "0 prefix violations and at least one content token in every diagnostic target", check_c30)
        _run_check(checks, logger, "C31", CHECK_DESCRIPTIONS["C31"], "35,704 prompt digests and 229,487 joint digests identical per family", check_c31)
        _run_check(checks, logger, "C32", CHECK_DESCRIPTIONS["C32"], f"0 out-of-vocabulary IDs; max joint length <= {facts['max_joint_len_bound']}", check_c32)
        _run_check(checks, logger, "C33", CHECK_DESCRIPTIONS["C33"], "Configured five-count vector for every family and set", check_c33)
        _run_check(checks, logger, "C34", CHECK_DESCRIPTIONS["C34"], "Configured French/Indonesian counts for every family and set", check_c34)
        _run_check(checks, logger, "C35", CHECK_DESCRIPTIONS["C35"], "Configured distance, token-distinctness, subset hash, and composition", check_c35)
        descriptor_rows = []
        for spec in specs:
            audit = audits[spec.label]
            for name, stats in audit["descriptors"].items():
                if not stats.get("n_items"):
                    continue
                descriptor_rows.append({
                    "label": spec.label, "family": spec.family, "variant": name,
                    "n_items": stats["n_items"],
                    "mean_tokens": stats["target_tokens"] / stats["pairs"],
                    "mean_content_tokens": stats["content_tokens"] / stats["pairs"],
                    "continuation_share": stats["continuations"] / stats["content_tokens"] if stats["content_tokens"] else 0.0,
                    "first_token_whitespace_n": stats["first_whitespace"],
                    "first_token_whitespace_share": stats["first_whitespace"] / stats["pairs"],
                    "any_byte_fragment_n": stats["byte_fragment"],
                    "any_byte_fragment_share": stats["byte_fragment"] / stats["pairs"],
                    "any_whitespace_token_n": stats["any_whitespace"],
                })
        return c33_rows, c34_rows, c35_result, c35_rows, descriptor_rows

    try:
        model_result = _stage("tokenizer_config_ntok_joint_checks", timings, logger, run_model_audits)
    except Exception as exc:
        logger.exception("Tokenizer audit stage failed outside an individual check")
        model_result = None
        for check_id in ("C23", "C24", "C25", "C26", "C27", "C29", "C30", "C31", "C32", "C33", "C34", "C35"):
            if check_id not in checks:
                _add_check(checks, check_id, CHECK_DESCRIPTIONS.get(check_id, check_id), "configured result", f"{type(exc).__name__}: {exc}", "ERROR", logger)
    if isinstance(model_result, tuple):
        c33_rows, c34_rows, c35_result, c35_rows, descriptor_rows = model_result
    else:
        c33_rows, c34_rows, c35_result, c35_rows, descriptor_rows = [], [], None, [], []

    def audit_cache_mtimes() -> tuple[Any, str]:
        """Use lstat to find shared-cache paths modified at or after run start."""
        touched = [
            str(path)
            for path, info in _lstat_tree(_cache_roots(cache, specs))
            if info.st_mtime_ns >= run_start_ns
        ]
        return (
            f"{len(touched)} paths" + (": " + ", ".join(touched[:20]) if touched else ""),
            "MATCH" if not touched else "MISMATCH",
        )

    _stage("cache_mtime_check", timings, logger, lambda: _run_check(
        checks, logger, "C36", "Shared model cache paths and lock paths were untouched",
        "0 paths with mtime >= run start", audit_cache_mtimes,
    ))
    end_time = datetime.now(timezone.utc)
    total_seconds = time.perf_counter() - overall_start
    timings["total"] = total_seconds
    peak_rss = _rss_mb()
    check_rows = _ordered_checks(checks)
    token_table_rows = []
    for spec in specs:
        token_table_rows.append({
            **tokenizer_rows.get(spec.label, {}),
            **config_rows.get(spec.label, {}),
            "label": spec.label,
            "hf_id": spec.hf_id,
            "revision": spec.revision,
            "snapshot_path": str(snapshots.get(spec.label, "")),
            **snapshot_sizes.get(spec.label, {
                "index_total_size": None,
                "shard_bytes_sum": None,
            }),
        })
    _ACTIVE_RUN.update({
        "check_rows": check_rows, "token_table_rows": token_table_rows,
        "c33_rows": c33_rows, "c34_rows": c34_rows,
        "c35_result": c35_result, "c35_rows": c35_rows,
        "descriptor_rows": descriptor_rows,
        "end_time": end_time, "total_seconds": total_seconds,
        "peak_rss": peak_rss,
    })
    _write_csv(output_dir / "X0c_a_checks.csv", check_rows, CHECK_COLUMNS)
    _write_csv(
        output_dir / "X0c_a_tokenizers.csv",
        token_table_rows,
        (
            "label", "hf_id", "revision", "snapshot_path", "tokenizer_class", "is_fast", "tokenizer_len",
            "bos_token", "bos_token_id", "eos_token", "eos_token_id", "pad_token", "pad_token_id",
            "unk_token", "unk_token_id", "x_colon_ids", "model_type", "architectures", "N", "d",
            "vocab_size", "tie_word_embeddings", "final_logit_softcapping", "sliding_window",
            "index_total_size", "shard_bytes_sum",
        ),
    )
    descriptor_rows = locals().get("descriptor_rows", [])
    _write_csv(
        output_dir / "X0c_a_target_tokenization.csv", descriptor_rows,
        ("label", "family", "variant", "n_items", "mean_tokens", "mean_content_tokens",
         "continuation_share", "first_token_whitespace_n", "first_token_whitespace_share",
         "any_byte_fragment_n", "any_byte_fragment_share", "any_whitespace_token_n"),
    )
    _write_csv(
        output_dir / "X0c_a_m1_first_token.csv", c33_rows,
        ("label", "family", "readout", "usable_count", "expected", "result"),
    )
    _write_csv(
        output_dir / "X0c_a_cognates.csv", c35_rows,
        ("section", "label", "metric", "actual", "expected", "result"),
    )
    _write_csv(
        output_dir / "X0c_a_prompt_duplicates.csv", duplicate_rows,
        ("group_id", "file", "concept_id", "format", "lang", "split", "m1_extension", "vi", "ru"),
    )
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip())
    metadata = {
        "head": head,
        "dirty": dirty,
        "config_sha256": config_sha256(config_path),
        "manifest_sha256": sha256_file(REPO / cfg["paths"]["manifest"]),
        "python": sys.version.split()[0],
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "checkpoints": "; ".join(f"{spec.hf_id}@{spec.revision}" for spec in specs),
        "start": run_start.isoformat(),
        "end": end_time.isoformat(),
        "stage_times": _json(timings),
        "total_seconds": f"{total_seconds:.3f}",
        "peak_rss_mib": f"{peak_rss:.1f}",
        "special_differences": "; ".join(vocab_diff_details) if vocab_diff_details else "none",
        "c36": checks.get("C36", {}).get("result", "NOT RUN"),
    }
    report_lines = [
        "# X0c-a tokenizer audit (T03a)",
        "",
        "HEAD: " + metadata["head"] + "; dirty flag: " + str(metadata["dirty"]),
        "Config SHA-256: " + metadata["config_sha256"],
        "Manifest SHA-256: " + metadata["manifest_sha256"],
        f"Python {metadata['python']}; transformers {metadata['transformers']}; tokenizers {metadata['tokenizers']}; huggingface-hub {metadata['huggingface_hub']}",
        "Checkpoints: " + metadata["checkpoints"],
        "UTC start: " + metadata["start"] + "; end: " + metadata["end"],
        "Stage wall time seconds: " + metadata["stage_times"],
        "Total seconds: " + metadata["total_seconds"] + "; peak RSS MiB: " + metadata["peak_rss_mib"],
        "No model weights were loaded or opened.",
        "",
        "## C21–C36 checks",
        "",
        _markdown_table(CHECK_COLUMNS, [[row[column] for column in CHECK_COLUMNS] for row in check_rows]),
        "",
        "## Tokenizer inventory",
        "",
        _markdown_table(
            ("Label", "HF ID", "Revision", "Class", "Fast", "Len", "BOS", "EOS", "PAD", "UNK", "x:", "N/d/vocab", "Tie", "Softcap", "Sliding"),
            [[
                row.get("label", ""), row.get("hf_id", ""), row.get("revision", ""), row.get("tokenizer_class", ""),
                row.get("is_fast", ""), row.get("tokenizer_len", ""),
                f"{row.get('bos_token')} / {row.get('bos_token_id')}",
                f"{row.get('eos_token')} / {row.get('eos_token_id')}",
                f"{row.get('pad_token')} / {row.get('pad_token_id')}",
                f"{row.get('unk_token')} / {row.get('unk_token_id')}",
                row.get("x_colon_ids", ""),
                f"{row.get('N', '')}/{row.get('d', '')}/{row.get('vocab_size', '')}",
                row.get("tie_word_embeddings", ""), row.get("final_logit_softcapping", ""), row.get("sliding_window", ""),
            ] for row in token_table_rows],
        ),
        "",
        "Registered/base special-token differences: " + metadata["special_differences"],
        "",
        "## Section 2.4 target tokenization",
        "",
        _markdown_table(
            ("Checkpoint", "Variant", "Items", "Mean tokens", "Mean content", "Continuation", "First whitespace n/share", "Byte-fragment share", "Whitespace-token items"),
            [[
                row.get("label", ""), row.get("variant", ""), row.get("n_items", ""),
                f"{float(row['mean_tokens']):.2f}", f"{float(row['mean_content_tokens']):.2f}",
                f"{float(row['continuation_share']):.0%}",
                f"{row['first_token_whitespace_n']}/{float(row['first_token_whitespace_share']):.0%}",
                f"{float(row['any_byte_fragment_share']):.1%}", row.get("any_whitespace_token_n", ""),
            ] for row in descriptor_rows],
        ),
        "",
        "## Section 2.5 and 2.6 tables",
        "",
        "See X0c_a_m1_first_token.csv and X0c_a_cognates.csv for the complete per-checkpoint counts.",
        "C35 aggregate: " + _json(c35_result),
        "",
        "## Prompt duplicate inventory",
        "",
        "Unique exact prompt strings: " + str(duplicates.get("unique_prompts", "")),
        "Cross-file groups: " + _json(duplicates.get("cross_file", {})),
        "Directions-within groups: " + _json(duplicates.get("directions_within", {})),
        "Per-file groups/records: " + _json(duplicates.get("per_file", {})),
        "Prompt text is omitted. The CSV contains group id, source file, concept id, format, language, split, m1_extension and vi/ru forms.",
        "",
        "Commit hash: this file is included in the T03a commit; the hash is in the final run response.",
    ]
    (output_dir / "X0c_a.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _ACTIVE_RUN["outputs_written"] = True
    logger.info("C21-C36 results")
    for row in check_rows:
        logger.info("%s | %s | expected=%s | actual=%s | %s", row["check_id"], row["description"], row["expected"], row["actual"], row["result"])
    logger.info("run_end=%s elapsed=%.3fs peak_rss=%.1fMiB", end_time.isoformat(), total_seconds, peak_rss)
    return 0 if all(row["result"] == "MATCH" for row in check_rows) else 1


def _write_partial_outputs() -> None:
    """Write all available audit rows and table headers after an interrupted run."""
    if not _ACTIVE_RUN or _ACTIVE_RUN.get("outputs_written"):
        return
    output_dir: Path = _ACTIVE_RUN["output_dir"]
    checks: dict[str, dict[str, str]] = _ACTIVE_RUN["checks"]
    logger: logging.Logger = _ACTIVE_RUN["logger"]
    fatal = _ACTIVE_RUN.get("fatal")
    if fatal:
        for check_number in range(21, 37):
            check_id = f"C{check_number}"
            if check_id not in checks:
                _add_check(
                    checks, check_id, CHECK_DESCRIPTIONS.get(check_id, "Shared cache remained untouched" if check_id == "C36" else check_id),
                    "configured result", fatal, "ERROR", logger,
                )
    else:
        for check_number in range(21, 37):
            check_id = f"C{check_number}"
            if check_id not in checks:
                _add_check(checks, check_id, CHECK_DESCRIPTIONS.get(check_id, check_id), "configured result", "NOT RUN: stage did not produce a row", "NOT RUN", logger)

    cfg = _ACTIVE_RUN.get("cfg", {})
    specs: Sequence[ModelSpec] = _ACTIVE_RUN.get("specs", ())
    snapshots = _ACTIVE_RUN.get("snapshots", {})
    snapshot_sizes = _ACTIVE_RUN.get("snapshot_sizes", {})
    tokenizer_rows = _ACTIVE_RUN.get("tokenizer_rows", {})
    config_rows = _ACTIVE_RUN.get("config_rows", {})
    token_table_rows = _ACTIVE_RUN.get("token_table_rows")
    if token_table_rows is None:
        token_table_rows = [
            {
                **tokenizer_rows.get(spec.label, {}),
                **config_rows.get(spec.label, {}),
                "label": spec.label, "hf_id": spec.hf_id, "revision": spec.revision,
                "snapshot_path": str(snapshots.get(spec.label, "")),
                **snapshot_sizes.get(spec.label, {"index_total_size": None, "shard_bytes_sum": None}),
            }
            for spec in specs
        ]
    descriptor_rows = _ACTIVE_RUN.get("descriptor_rows", [])
    c33_rows = _ACTIVE_RUN.get("c33_rows", [])
    c35_rows = _ACTIVE_RUN.get("c35_rows", [])
    duplicate_rows = _ACTIVE_RUN.get("duplicate_rows", [])
    _write_csv(output_dir / "X0c_a_checks.csv", _ordered_checks(checks), CHECK_COLUMNS)
    _write_csv(output_dir / "X0c_a_tokenizers.csv", token_table_rows, (
        "label", "hf_id", "revision", "snapshot_path", "tokenizer_class", "is_fast", "tokenizer_len",
        "bos_token", "bos_token_id", "eos_token", "eos_token_id", "pad_token", "pad_token_id",
        "unk_token", "unk_token_id", "x_colon_ids", "model_type", "architectures", "N", "d",
        "vocab_size", "tie_word_embeddings", "final_logit_softcapping", "sliding_window",
        "index_total_size", "shard_bytes_sum",
    ))
    _write_csv(output_dir / "X0c_a_target_tokenization.csv", descriptor_rows, (
        "label", "family", "variant", "n_items", "mean_tokens", "mean_content_tokens",
        "continuation_share", "first_token_whitespace_n", "first_token_whitespace_share",
        "any_byte_fragment_n", "any_byte_fragment_share", "any_whitespace_token_n",
    ))
    _write_csv(output_dir / "X0c_a_m1_first_token.csv", c33_rows, ("label", "family", "readout", "usable_count", "expected", "result"))
    _write_csv(output_dir / "X0c_a_cognates.csv", c35_rows, ("section", "label", "metric", "actual", "expected", "result"))
    _write_csv(output_dir / "X0c_a_prompt_duplicates.csv", duplicate_rows, (
        "group_id", "file", "concept_id", "format", "lang", "split", "m1_extension", "vi", "ru",
    ))
    partial_lines = [
        "# X0c-a tokenizer audit (partial output)", "",
        f"Run stamp: {_ACTIVE_RUN.get('run_stamp', '')}",
        f"Failure: {fatal or 'normal finalization was not reached'}", "",
        "## C21–C36 checks", "",
        _markdown_table(CHECK_COLUMNS, [[row[column] for column in CHECK_COLUMNS] for row in _ordered_checks(checks)]),
        "",
        "No model weights were loaded or opened. Prompt text is omitted.",
    ]
    (output_dir / "X0c_a.md").write_text("\n".join(partial_lines) + "\n", encoding="utf-8")
    _ACTIVE_RUN["outputs_written"] = True


def main() -> int:
    """Run the audit, always persist available outputs, and print smoke result counts."""
    args = _parse_args()
    result = 1
    try:
        result = _run_audit(args)
    except Exception as exc:
        logger = _ACTIVE_RUN.get("logger")
        failure = f"{type(exc).__name__}: {exc}"
        if logger is not None:
            logger.exception("Audit terminated outside an individual check")
        else:
            print(f"Audit terminated: {failure}", file=sys.stderr)
        _ACTIVE_RUN["fatal"] = failure
        result = 1
    finally:
        _write_partial_outputs()
    if args.smoke is not None:
        counts = Counter(row["result"] for row in _ordered_checks(_ACTIVE_RUN.get("checks", {})))
        summary = ", ".join(f"{name}={counts.get(name, 0)}" for name in ("MATCH", "MISMATCH", "NOT RUN", "ERROR"))
        print(f"Smoke summary ({_ACTIVE_RUN.get('output_dir', '')}): {summary}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
