#!/datastore/luannt/.cache/sb/bin/python
"""Build, verify, and freeze the preregistered X0 derived manifests."""

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard; env_guard.guard(offline=True)

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from vilens.cognates import cognate_free_ids, distance_ok_ids, distinct_first_token_ids, id_list_sha256
from vilens.config import config_sha256, load_config, study_models
from vilens.data import load_concepts_jsonl, load_prompt_file, prompt_file_paths
from vilens.derived import (
    FormStat,
    StatKey,
    _match_distractors,
    e0_sample,
    e8_order_rows,
    form_stats_from_encoding,
    compare_directories,
    refuse_nonempty_derived,
    target_string,
)
from vilens.manifest import sha256_file, verify_manifest
from vilens.models import ModelSpec, load_tokenizer
from vilens.provenance import derived_sha256
from vilens.tokenize import TokenClassifier, encode_joint_batch

FAMILIES = ("gemma", "qwen", "llama")
LANGUAGES = ("vi", "en", "zh", "fr", "id")
INPUT_STEMS = tuple()
CHECK_COLUMNS = ("check_id", "description", "expected", "actual", "result")
OUTPUT_FILES = (
    "target_forms.jsonl",
    "cognate_flags.jsonl",
    "distractors.jsonl",
    "e8_order.jsonl",
    "e8_rule.md",
    "e0_pilot.jsonl",
    "e0_distractors.jsonl",
    "seeds.json",
    "README.md",
)


def _utc_now() -> str:
    """Return an ISO UTC timestamp for run reports and logs."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    """Return a compact UTC timestamp suitable for log filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _display(value: Any) -> str:
    """Render check values deterministically as compact JSON or strings."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _setup_logger(log_dir: Path) -> logging.Logger:
    """Configure timestamped stdout and file logging under the configured logs path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("x0d_derived")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"x0d_derived_{_utc_stamp()}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _concept_mapping(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Copy only released study fields to plain Python values for deterministic output."""
    columns = ("concept_id", "split", "m1_extension", "vi", "vi_nodiac", "en", "zh", "fr", "id", "stratum", "pos", "zipf_en", "collapsed")
    result: dict[str, dict[str, Any]] = {}
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        concept_id = str(row[0])
        result[concept_id] = {
            "concept_id": concept_id,
            "split": str(row[1]),
            "m1_extension": bool(row[2]),
            "vi": str(row[3]),
            "vi_nodiac": str(row[4]),
            "en": str(row[5]),
            "zh": str(row[6]),
            "fr": str(row[7]),
            "id": str(row[8]),
            "stratum": str(row[9]),
            "pos": str(row[10]),
            "zipf_en": float(row[11]),
            "collapsed": bool(row[12]),
        }
    return result


def _read_prompts(cfg: dict[str, Any], repo: Path) -> tuple[dict[str, Path], dict[str, list[dict[str, Any]]]]:
    """Load each of the 27 released prompt files keyed by its stem."""
    paths = {stem: repo / path for stem, path in prompt_file_paths(cfg).items()}
    prompts = {stem: load_prompt_file(path) for stem, path in paths.items()}
    return paths, prompts


def _input_paths(cfg: dict[str, Any], repo: Path) -> tuple[Path, dict[str, Path], list[str]]:
    """Resolve concepts, prompt paths, and the 28 selected manifest entries."""
    concepts_path = repo / cfg["paths"]["concepts_jsonl"]
    prompts = {stem: repo / path for stem, path in prompt_file_paths(cfg).items()}
    selected = [cfg["paths"]["concepts_jsonl"], *[path.relative_to(repo).as_posix() for path in prompts.values()]]
    return concepts_path, prompts, selected


def _main_concepts(concepts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return sorted main-test concept IDs, excluding the M1 extension."""
    return sorted(
        concept_id
        for concept_id, row in concepts.items()
        if row["split"] == "test" and not row["m1_extension"]
    )


def _write_check(
    checks: list[dict[str, str]],
    logger: logging.Logger,
    check_id: str,
    description: str,
    expected: Any,
    operation: Callable[[], Any],
    compare: Callable[[Any, Any], bool] | None = None,
) -> Any:
    """Run one independent check, log its row immediately, and retain failures."""
    try:
        actual = operation()
        passed = compare(actual, expected) if compare is not None else actual == expected
        result = "MATCH" if passed else "MISMATCH"
    except Exception as exc:
        logger.exception("%s raised during check", check_id)
        actual = f"{type(exc).__name__}: {exc}"
        result = "ERROR"
    row = {
        "check_id": check_id,
        "description": description,
        "expected": _display(expected),
        "actual": _display(actual),
        "result": result,
    }
    checks.append(row)
    logger.info("%s %s expected=%s actual=%s", check_id, result, row["expected"], row["actual"])
    return actual


def _precompute_stats(
    cfg: dict[str, Any],
    concepts: Mapping[str, Mapping[str, Any]],
    prompts: Mapping[str, list[dict[str, Any]]],
    tokenizers: Mapping[str, Any],
) -> tuple[dict[StatKey, FormStat], dict[StatKey, FormStat], dict[str, dict[str, tuple[int, ...]]], int, list[dict[str, Any]]]:
    """Batch-encode each family/combo/concept form once and cache all statistics before matching."""
    from vilens.derived import e0_sample

    seed = int(cfg["seed"])
    manifests = cfg["manifests"]
    chunk_size = int(cfg["text"]["audit_chunk_size"])
    main_ids = _main_concepts(concepts)
    rep_context = {
        row["concept_id"]: row["prompt"]
        for row in prompts[cfg["data"]["primary_prompt_files"]["repetition_diac"]]
    }
    directions_rows = prompts["directions_matched"]
    direction_context = {
        (row["concept_id"], row["lang"]): row["prompt"]
        for row in directions_rows
        if row["format"] == "repetition"
    }
    direction_ids = sorted(
        concept_id for concept_id, row in concepts.items() if row["split"] == "directions"
    )
    sample_rows = e0_sample(direction_ids, seed, int(manifests["e0_sample_size"]))
    e0_ids = [row["concept_id"] for row in sample_rows]
    main_stats: dict[StatKey, FormStat] = {}
    e0_stats: dict[StatKey, FormStat] = {}
    prompt_caches: dict[str, dict[str, tuple[int, ...]]] = {}
    prefix_failures = 0

    for family in FAMILIES:
        tokenizer = tokenizers[family]
        classifier = TokenClassifier(tokenizer)
        cache: dict[str, tuple[int, ...]] = {}
        prompt_caches[family] = cache
        for lang, variant in manifests["distractor_combos"]:
            pairs = [
                (rep_context[concept_id], target_string(concepts[concept_id], lang, variant), concept_id)
                for concept_id in main_ids
            ]
            for start in range(0, len(pairs), chunk_size):
                chunk = pairs[start : start + chunk_size]
                encodings = encode_joint_batch(
                    tokenizer,
                    [item[0] for item in chunk],
                    [item[1] for item in chunk],
                    strict=True,
                    prompt_cache=cache,
                )
                for (_, _, concept_id), encoding in zip(chunk, encodings, strict=True):
                    prefix_failures += int(not encoding.prefix_ok)
                    main_stats[(family, lang, variant, concept_id)] = form_stats_from_encoding(encoding, classifier)

        for lang, variant in manifests["e0_combos"]:
            pairs = [
                (direction_context[(concept_id, lang)], target_string(concepts[concept_id], lang, variant), concept_id)
                for concept_id in e0_ids
            ]
            for start in range(0, len(pairs), chunk_size):
                chunk = pairs[start : start + chunk_size]
                encodings = encode_joint_batch(
                    tokenizer,
                    [item[0] for item in chunk],
                    [item[1] for item in chunk],
                    strict=True,
                    prompt_cache=cache,
                )
                for (_, _, concept_id), encoding in zip(chunk, encodings, strict=True):
                    prefix_failures += int(not encoding.prefix_ok)
                    e0_stats[(family, lang, variant, concept_id)] = form_stats_from_encoding(encoding, classifier)
    return main_stats, e0_stats, prompt_caches, prefix_failures, sample_rows


def _make_target_forms(concepts: Mapping[str, Mapping[str, Any]], combos: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """Expand test and directions concepts into the configured seven target forms."""
    included = [row for row in concepts.values() if row["split"] in {"test", "directions"}]
    rows = [
        {
            "concept_id": row["concept_id"],
            "split": row["split"],
            "m1_extension": row["m1_extension"],
            "target_lang": lang,
            "target_variant": variant,
            "target": target_string(row, lang, variant),
        }
        for row in included
        for lang, variant in combos
    ]
    return sorted(rows, key=lambda row: (row["concept_id"], row["target_lang"], row["target_variant"]))


def _cognate_rows(
    concepts_frame: pd.DataFrame,
    concepts: Mapping[str, Mapping[str, Any]],
    text_cfg: Mapping[str, Any],
    stats: Mapping[StatKey, FormStat],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Build main-concept cognate flags from distances and all three tokenizer first IDs."""
    main_frame = concepts_frame[(concepts_frame["split"] == "test") & ~concepts_frame["m1_extension"]]
    d_en_fr, d_en_id, distance_ok = distance_ok_ids(main_frame, text_cfg)
    distinct_by_family: dict[str, set[str]] = {}
    for family in FAMILIES:
        firsts = {
            concept_id: {
                lang: stats[(family, lang, "released", concept_id)][1]
                for lang in LANGUAGES
            }
            for concept_id in distance_ok
        }
        distinct_by_family[family] = distinct_first_token_ids(firsts)
    cognate_free = cognate_free_ids(distance_ok, distinct_by_family)
    free = set(cognate_free)
    rows = [
        {
            "concept_id": str(row.concept_id),
            "d_en_fr": float(d_en_fr[str(row.concept_id)]),
            "d_en_id": float(d_en_id[str(row.concept_id)]),
            "distance_ok": str(row.concept_id) in distance_ok,
            "distinct_first_gemma": str(row.concept_id) in distinct_by_family["gemma"],
            "distinct_first_qwen": str(row.concept_id) in distinct_by_family["qwen"],
            "distinct_first_llama": str(row.concept_id) in distinct_by_family["llama"],
            "cognate_free": str(row.concept_id) in free,
        }
        for row in main_frame.itertuples(index=False)
    ]
    rows.sort(key=lambda row: row["concept_id"])
    return rows, cognate_free, {"distance_ok": distance_ok, "distinct_by_family": distinct_by_family}


def _distractor_checksum(rows: Sequence[Mapping[str, Any]], family: str) -> str:
    """Hash the sorted canonical distractor identity rows for one family."""
    selected = sorted(
        (row for row in rows if row["family"] == family),
        key=lambda row: (row["target_lang"], row["target_variant"], row["concept_id"]),
    )
    payload = "".join(
        f"{row['family']}\t{row['target_lang']}\t{row['target_variant']}\t{row['concept_id']}\t{row['distractor_id']}\n"
        for row in selected
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _e8_checksum(rows: Sequence[Mapping[str, Any]], family: str) -> str:
    """Hash sorted E8 order rows for one model family."""
    selected = sorted(
        (row for row in rows if row["family"] == family),
        key=lambda row: (row["stratum"], row["rank"]),
    )
    payload = "".join(
        f"{family}\t{row['stratum']}\t{row['rank']}\t{row['concept_id']}\n"
        for row in selected
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_e8_rule(seed: int, max_items: int) -> str:
    """Render the locked E8 selection rule, seed, and deterministic stream keys."""
    return (
        "# E8 selection rule\n\n"
        "A concept is competent when its unsteered greedy answer is scored VI under A10. Competent concepts "
        "are ranked within stratum by the frozen E8 order. Set n = min(e8_max_items, number of competent concepts). "
        "Allocate n proportionally to the "
        "competent stratum counts using largest remainder, breaking remainder ties by stratum name. "
        "Within each stratum, select the first quota competent concepts in rank order.\n\n"
        f"- `e8_max_items`: {max_items}\n"
        f"- `seed`: {seed}\n"
        "- E8 stream key: `e8|<family>|<stratum>`\n"
        "- E0 sample stream key: `e0_sample`\n"
        "- Main distractor key: `distractor|<family>|<lang>|<variant>|<concept_id>`\n"
        "- E0 distractor key: `e0_distractor|<family>|<lang>|<variant>|<concept_id>`\n"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]], sort_key: Sequence[str]) -> bytes:
    """Serialize sorted JSON objects as UTF-8 JSONL with a final newline."""
    ordered = sorted(rows, key=lambda row: tuple(row[key] for key in sort_key))
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in ordered
    ).encode("utf-8")


def _render_files(bundle: Mapping[str, Any], cfg: Mapping[str, Any], repo: Path, destination: Path) -> list[str]:
    """Write a deterministic manifest bundle into one temporary directory."""
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    manifests = cfg["manifests"]
    files: dict[str, bytes] = {
        "target_forms.jsonl": _jsonl_bytes(bundle.get("target_forms", []), ("concept_id", "target_lang", "target_variant")),
        "cognate_flags.jsonl": _jsonl_bytes(bundle.get("cognate_flags", []), ("concept_id",)),
        "distractors.jsonl": _jsonl_bytes(bundle.get("distractors", []), ("family", "target_lang", "target_variant", "concept_id")),
        "e8_order.jsonl": _jsonl_bytes(bundle.get("e8_order", []), ("family", "stratum", "rank")),
        "e8_rule.md": _make_e8_rule(seed, int(manifests["e8_max_items"])).encode("utf-8"),
        "e0_pilot.jsonl": _jsonl_bytes(bundle.get("e0_pilot", []), ("rank",)),
        "e0_distractors.jsonl": _jsonl_bytes(bundle.get("e0_distractors", []), ("family", "target_lang", "target_variant", "concept_id")),
    }
    stream_keys = [
        "distractor|<family>|<lang>|<variant>|<concept_id>",
        "e0_distractor|<family>|<lang>|<variant>|<concept_id>",
        "e8|<family>|<stratum>",
        "e0_sample",
    ]
    seeds = {
        "seed": seed,
        "rng_scheme": manifests["rng_scheme"],
        "stream_keys": stream_keys,
        "bootstrap": {"resamples": 1000, "key": "bootstrap|<analysis>|<model>"},
        "steer_random_direction": {"key": "steer_random|<family>|<layer>"},
    }
    files["seeds.json"] = (json.dumps(seeds, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    hash_lines = [f"- `{name}` — `{hashlib.sha256(content).hexdigest()}`" for name, content in sorted(files.items())]
    readme_path = repo / cfg["paths"]["derived"] / "README.md"
    if readme_path.is_file():
        existing = readme_path.read_text(encoding="utf-8")
        match = re.search(r"Generated by `scripts/x0d_derived\.py` from Git commit `([0-9a-f]{40})`\.", existing)
        if match is None:
            raise ValueError("existing derived/README.md does not identify the generator commit")
        head = match.group(1)
    else:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    readme = (
        "# Frozen X0 derived manifests\n\n"
        "These files implement D1/D3/A12 target variants and distractor matching, D4 cognate-free flags, "
        "D6 E8 order, the E0 directions pilot sample, and the registered seed streams.\n\n"
        f"Generated by `scripts/x0d_derived.py` from Git commit `{head}`.\n\n"
        "## Files and checksums\n\n"
        "- `target_forms.jsonl`: canonical released, Chinese no-space, and Vietnamese nodiac target forms.\n"
        "- `cognate_flags.jsonl`: main-test surface-distance and shared first-token flags.\n"
        "- `distractors.jsonl`: deterministic matched main-test distractors.\n"
        "- `e8_order.jsonl`: seeded within-stratum item order for E8.\n"
        "- `e8_rule.md`: frozen post-competence sampling rule.\n"
        "- `e0_pilot.jsonl`: seeded directions-split pilot concepts.\n"
        "- `e0_distractors.jsonl`: matched distractors for the E0 pilot.\n"
        "- `seeds.json`: registered seed and independent random-stream keys.\n\n"
        + "\n".join(hash_lines)
        + "\n"
    )
    files["README.md"] = readme.encode("utf-8")
    for name, content in files.items():
        (destination / name).write_bytes(content)
    return sorted(files)


def _markdown_table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Format plain mapping rows as a compact Markdown table."""
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n"
        for row in rows
    )
    return head + body


def _csv_markdown(path: Path, title: str) -> str:
    """Render a committed X0c summary CSV as a Markdown subsection and table."""
    if not path.is_file():
        return f"### {title}\n\nUnavailable: `{path}` is missing.\n\n"
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return f"### {title}\n\n" + _markdown_table(columns, rows) + "\n"


def _quality_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate main-distractor match quality by family and language variant."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("e0"):
            groups[(str(row["family"]), str(row["target_lang"]), str(row["target_variant"]))].append(row)
    quality: list[dict[str, Any]] = []
    for (family, lang, variant), group in sorted(groups.items()):
        values = np.asarray([float(row["abs_dzipf_en"]) for row in group], dtype=np.float64)
        quality.append(
            {
                "family": family,
                "target_lang": lang,
                "target_variant": variant,
                "n": len(group),
                "exact": sum(row["kind"] == "exact" for row in group),
                "count_fallback": sum(row["kind"] == "count_fallback" for row in group),
                "n_tied_gt1": sum(int(row["n_tied"]) > 1 for row in group),
                "mean_abs_dzipf_en": float(values.mean()) if values.size else 0.0,
                "median_abs_dzipf_en": float(np.median(values)) if values.size else 0.0,
                "max_abs_dzipf_en": float(values.max()) if values.size else 0.0,
                "share_abs_dzipf_en_zero": float((values == 0).mean()) if values.size else 0.0,
            }
        )
    return quality


def _write_quality_csv(path: Path, quality: Sequence[Mapping[str, Any]]) -> None:
    """Write the per-family/combo distractor quality summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "family", "target_lang", "target_variant", "n", "exact", "count_fallback", "n_tied_gt1",
        "mean_abs_dzipf_en", "median_abs_dzipf_en", "max_abs_dzipf_en", "share_abs_dzipf_en_zero",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(quality)


def _write_report(
    path: Path,
    *,
    cfg: Mapping[str, Any],
    repo: Path,
    checks: Sequence[Mapping[str, str]],
    bundle: Mapping[str, Any],
    concepts: Mapping[str, Mapping[str, Any]] | None,
    derived_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write the X0 checkpoint report with prior X0 findings and frozen checksums."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = (repo / cfg["paths"]["manifest"]).read_text(encoding="utf-8").rstrip()
    derived_files: list[tuple[str, str, int]] = []
    if derived_dir.is_dir():
        for file_path in sorted(item for item in derived_dir.iterdir() if item.is_file()):
            if file_path.suffix in {".jsonl"}:
                line_count = sum(1 for _ in file_path.open("r", encoding="utf-8"))
            else:
                line_count = sum(1 for _ in file_path.open("r", encoding="utf-8"))
            derived_files.append((file_path.name, sha256_file(file_path), line_count))
    digest = derived_sha256(repo)
    quality = _quality_rows(bundle.get("distractors", []))
    _write_quality_csv(repo / "reports/X0d_distractor_quality.csv", quality)
    checks_table = _markdown_table(CHECK_COLUMNS, checks)
    quality_table = _markdown_table(tuple(quality[0].keys()) if quality else (), quality)
    case_ids = bundle.get("nodiac_column_diff_ids", [])
    case_forms = []
    if concepts is not None:
        case_forms = [
            {"concept_id": concept_id, "vi": concepts[concept_id]["vi"], "vi_nodiac_column": concepts[concept_id]["vi_nodiac"]}
            for concept_id in case_ids
        ]
    d12 = cfg["data"]["pretag_excluded_directions"]
    e0_rows = bundle.get("e0_pilot", [])
    e0_excluded = sorted(set(row["concept_id"] for row in e0_rows) & {row["concept_id"] for row in d12})
    sections = [
        "# X0 checkpoint summary\n\n",
        "## Inputs (X0a)\n\nVerified using the committed `data/MANIFEST.sha256` (31/31 entries in T01/X0a). The manifest entries are reproduced below.\n\n",
        "```text\n" + manifest_text + "\n```\n\n",
        "## X0b and X0c-a\n\n",
        "- X0b: 20/20 data checks MATCH; commit `0a74438`.\n",
        "- X0c-a: C21–C36 all MATCH for both candidate sets; commit `4ca1825`.\n\n",
        "## X0c-a summary tables\n\n",
        _csv_markdown(repo / "reports/X0c_a_target_tokenization.csv", "§2.4 Target tokenization"),
        _csv_markdown(repo / "reports/X0c_a_m1_first_token.csv", "§2.5 M1 first-token usability"),
        _csv_markdown(repo / "reports/X0c_a_cognates.csv", "§2.6 Cognate-free subset"),
        "## T03b checks\n\n" + checks_table + "\n",
        "## Distractor match quality\n\n" + quality_table + "\n",
        "## Collapsed few-shot demonstrations (X0b)\n\n",
        "| Set | Vietnamese form |\n| ---: | --- |\n",
        "| 1 | thứ tư |\n| 3 | nhóm |\n| 4 | khí đốt |\n| 5 | vàng |\n| 6 | sắt |\n| 6 | tiệc |\n\n",
        "## D12 directions/test prompt exclusion\n\n",
        _markdown_table(("concept_id", "format", "lang"), d12),
        f"E0 sampled excluded concepts: `{', '.join(e0_excluded) if e0_excluded else 'none'}`.\n\n",
        "## Frozen derived files\n\n",
        _markdown_table(("file", "sha256", "wc -l"), [
            {"file": name, "sha256": sha, "wc -l": count} for name, sha, count in derived_files
        ]),
        f"\n`provenance.derived_sha256`: `{digest}`\n",
    ]
    path.write_text("".join(sections), encoding="utf-8")
    logger.info("Wrote X0 checkpoint report to %s", path)


def run_audit(cfg: dict[str, Any], repo: Path, logger: logging.Logger, *, check_mode: bool) -> tuple[list[dict[str, str]], dict[str, Any], Path | None, bool]:
    """Run C41–C53, build two deterministic temporary trees, and return audit state."""
    checks: list[dict[str, str]] = []
    facts = cfg["reference_facts"]["x0d"]
    concepts_path, prompt_paths, selected_inputs = _input_paths(cfg, repo)
    manifest_path = repo / cfg["paths"]["manifest"]
    paths: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    tokenizers: dict[str, Any] = {}
    classifiers: dict[str, TokenClassifier] = {}
    concepts_frame: pd.DataFrame | None = None
    concepts: dict[str, dict[str, Any]] | None = None
    prompts: dict[str, list[dict[str, Any]]] | None = None
    main_stats: dict[StatKey, FormStat] = {}
    e0_stats: dict[StatKey, FormStat] = {}
    prompt_caches: dict[str, dict[str, tuple[int, ...]]] = {}
    cognate_sets: dict[str, Any] = {}
    e8_rows: list[dict[str, Any]] = []
    main_distractors: list[dict[str, Any]] = []
    e0_distractors: list[dict[str, Any]] = []

    def load_concepts() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """Lazily load the strict JSONL table once for dependent checks."""
        nonlocal concepts_frame, concepts
        if concepts_frame is None or concepts is None:
            concepts_frame = load_concepts_jsonl(concepts_path)
            concepts = _concept_mapping(concepts_frame)
        return concepts_frame, concepts

    def load_all_prompts() -> dict[str, list[dict[str, Any]]]:
        """Lazily load and cache all released prompt records."""
        nonlocal prompts
        if prompts is None:
            _, prompts = _read_prompts(cfg, repo)
        return prompts

    def compare_expected(actual: Any, expected: Any) -> bool:
        """Compare only declared expected keys for informative actual mappings."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            return all(key in actual and compare_expected(actual[key], value) for key, value in expected.items())
        return actual == expected

    def c41() -> str:
        """Verify the concepts source and every released prompt file."""
        results = verify_manifest(manifest_path, repo, only=selected_inputs)
        good = sum(ok for _, ok in results)
        return f"{good}/{len(results)} OK"

    _write_check(checks, logger, "C41", "Input manifest verification (concepts plus 27 prompt files)", "28/28 OK", c41)

    def c42() -> dict[str, Any]:
        """Load the D10 base tokenizers from the pinned shared snapshots."""
        selected = study_models(cfg)
        pins_match = cfg["models"]["study_set"] == "base" and selected == cfg["models"]["base"]
        if not pins_match:
            raise ValueError("study_models(cfg) does not equal the D10 base pins")
        from vilens.models import SnapshotError

        for family in FAMILIES:
            item = selected[family]
            spec = ModelSpec("base", family, item["hf_id"], item["revision"])
            tokenizers[family] = load_tokenizer(spec, cfg["paths"]["model_cache"])
            classifiers[family] = TokenClassifier(tokenizers[family])
        logger.info("Loaded tokenizer-only study set: %s", ", ".join(
            f"{family}={selected[family]['hf_id']}@{selected[family]['revision']}" for family in FAMILIES
        ))
        return {"study_set": "base", "pins_match": True, "tokenizers_loaded": len(tokenizers)}

    expected_c42 = {"study_set": "base", "pins_match": True, "tokenizers_loaded": len(FAMILIES)}
    _write_check(checks, logger, "C42", "Study-set base tokenizers load at the D10 pins (no weights)", expected_c42, c42, compare_expected)

    def c43() -> dict[str, int]:
        """Build and count canonical target form rows."""
        _, concept_map = load_concepts()
        bundle["target_forms"] = _make_target_forms(concept_map, cfg["manifests"]["distractor_combos"])
        return {"rows": len(bundle["target_forms"]), "concepts": len({row["concept_id"] for row in bundle["target_forms"]})}

    _write_check(
        checks, logger, "C43", "Target-form rows for test and directions concepts",
        {"rows": facts["target_form_rows"], "concepts": facts["target_form_concepts"]}, c43, compare_expected,
    )

    def c44() -> int:
        """Compare released Chinese targets with the exact leading-space canonical form."""
        _, concept_map = load_concepts()
        prompt_map = load_all_prompts()
        mismatches = 0
        for stem, records in prompt_map.items():
            for record in records:
                concept = concept_map[record["concept_id"]]
                if stem == "directions_matched":
                    if record["lang"] == "zh" and record["target"] != " " + concept["zh"]:
                        mismatches += 1
                elif record["target_zh"] != " " + concept["zh"]:
                    mismatches += 1
        return mismatches

    _write_check(checks, logger, "C44", "Released Chinese targets equal one leading space plus zh", 0, c44)

    def c45() -> dict[str, Any]:
        """Verify the case-preserving nodiac rule across files and concept columns."""
        _, concept_map = load_concepts()
        prompt_map = load_all_prompts()
        prompt_mismatches = 0
        nodiac_records = 0
        for stem, records in prompt_map.items():
            if "nodiac" not in stem:
                continue
            for record in records:
                nodiac_records += 1
                concept = concept_map[record["concept_id"]]
                expected_target = target_string(concept, "vi", "nodiac")
                if record["target_vi"] != expected_target:
                    prompt_mismatches += 1
        column_diff_ids = sorted(
            concept_id
            for concept_id, concept in concept_map.items()
            if target_string(concept, "vi", "nodiac")[1:] != concept["vi_nodiac"]
        )
        bundle["nodiac_column_diff_ids"] = column_diff_ids
        return {
            "nodiac_records": nodiac_records,
            "prompt_mismatches": prompt_mismatches,
            "column_difference_count": len(column_diff_ids),
            "column_difference_ids": column_diff_ids,
        }

    _write_check(
        checks,
        logger,
        "C45",
        "Case-preserving nodiac target rule matches every nodiac prompt; column differences are case-only",
        {
            "nodiac_records": 14801,
            "prompt_mismatches": facts["nodiac_rule_mismatch_vs_prompts"],
            "column_difference_count": facts["nodiac_rule_differs_from_column"],
        },
        c45,
        compare_expected,
    )

    def c46() -> int:
        """Precompute batched token statistics once per family, combo, and concept."""
        _, concept_map = load_concepts()
        prompt_map = load_all_prompts()
        main_stats_local, e0_stats_local, caches, failures, sample = _precompute_stats(
            cfg, concept_map, prompt_map, tokenizers
        )
        main_stats.update(main_stats_local)
        e0_stats.update(e0_stats_local)
        prompt_caches.update(caches)
        bundle["e0_pilot"] = sample
        bundle["form_stats_count"] = len(main_stats) + len(e0_stats)
        return failures

    _write_check(
        checks,
        logger,
        "C46",
        "Strict joint encoding prefix checks for all main and E0 form statistics",
        0,
        c46,
    )

    def c47() -> dict[str, Any]:
        """Build the shared cognate-free set and per-main-concept flags."""
        frame, concept_map = load_concepts()
        rows, ids, sets = _cognate_rows(frame, concept_map, cfg["text"], main_stats)
        bundle["cognate_flags"] = rows
        bundle["cognate_free_ids"] = ids
        cognate_sets.update(sets)
        return {"rows": len(rows), "cognate_free": len(ids), "sha256": id_list_sha256(ids)}

    _write_check(
        checks,
        logger,
        "C47",
        "Cognate flags and shared D4 subset",
        {"rows": facts["cognate_rows"], "cognate_free": facts["cognate_free"], "sha256": cfg["reference_facts"]["x0c"]["cognate_free_sha256"]},
        c47,
        compare_expected,
    )

    def c48() -> dict[str, Any]:
        """Generate all main-test distractors using cached form statistics only."""
        _, concept_map = load_concepts()
        pool_ids = _main_concepts(concept_map)
        results: list[dict[str, Any]] = []
        for family in FAMILIES:
            for lang, variant in cfg["manifests"]["distractor_combos"]:
                results.extend(
                    _match_distractors(
                        concept_map,
                        main_stats,
                        family=family,
                        lang=lang,
                        variant=variant,
                        pool_ids=pool_ids,
                        seed=int(cfg["seed"]),
                        key_prefix="distractor",
                    )
                )
        bundle["distractors"] = sorted(
            results,
            key=lambda row: (row["family"], row["target_lang"], row["target_variant"], row["concept_id"]),
        )
        expected_kinds = facts["distractor_kinds"]
        kinds: dict[str, list[list[int]]] = {}
        for family in FAMILIES:
            family_kinds: list[list[int]] = []
            for lang, variant in cfg["manifests"]["distractor_combos"]:
                selected = [
                    row for row in bundle["distractors"]
                    if row["family"] == family and row["target_lang"] == lang and row["target_variant"] == variant
                ]
                family_kinds.append([
                    sum(row["kind"] == "exact" for row in selected),
                    sum(row["kind"] == "count_fallback" for row in selected),
                ])
            kinds[family] = family_kinds
        invalid = 0
        for row in bundle["distractors"]:
            concept_id, distractor_id = row["concept_id"], row["distractor_id"]
            invalid += int(concept_id == distractor_id or distractor_id not in pool_ids)
            if (row["target_lang"], row["target_variant"]) == ("vi", "nodiac"):
                invalid += int(bool(concept_map[distractor_id]["collapsed"]))
        expected = {
            "rows_per_family": facts["distractor_rows_per_family"],
            "kinds": expected_kinds,
            "invalid": 0,
        }
        actual = {
            "rows_per_family": {family: sum(row["family"] == family for row in bundle["distractors"]) for family in FAMILIES},
            "kinds": kinds,
            "invalid": invalid,
        }
        return actual

    _write_check(
        checks,
        logger,
        "C48",
        "Main distractor counts, exact/fallback kinds, pool membership, non-self and nodiac collapsed exclusion",
        {
            "rows_per_family": {family: facts["distractor_rows_per_family"] for family in FAMILIES},
            "kinds": facts["distractor_kinds"],
            "invalid": 0,
        },
        c48,
        compare_expected,
    )

    def c49() -> dict[str, str]:
        """Compare canonical per-family distractor checksum values."""
        rows = bundle["distractors"]
        return {family: _distractor_checksum(rows, family) for family in FAMILIES}

    _write_check(checks, logger, "C49", "Canonical main distractor checksums", facts["distractor_sha256"], c49)

    def c50() -> int:
        """Batch-check each c-context/d-target pair exactly once against its cached match statistics."""
        _, concept_map = load_concepts()
        prompt_map = load_all_prompts()
        context_records = {
            row["concept_id"]: row
            for row in prompt_map[cfg["data"]["primary_prompt_files"]["repetition_diac"]]
        }
        chunk_size = int(cfg["text"]["audit_chunk_size"])
        mismatches = 0
        for family in FAMILIES:
            tokenizer = tokenizers[family]
            classifier = classifiers[family]
            family_rows = [row for row in bundle["distractors"] if row["family"] == family]
            cache = prompt_caches[family]
            for start in range(0, len(family_rows), chunk_size):
                chunk = family_rows[start : start + chunk_size]
                prompts_chunk = [context_records[row["concept_id"]]["prompt"] for row in chunk]
                targets_chunk = [target_string(concept_map[row["distractor_id"]], row["target_lang"], row["target_variant"]) for row in chunk]
                encodings = encode_joint_batch(
                    tokenizer, prompts_chunk, targets_chunk, strict=False, prompt_cache=cache
                )
                for row, encoding in zip(chunk, encodings, strict=True):
                    content_ids = [token_id for token_id in encoding.target_ids if classifier.is_content(token_id)]
                    actual_count = len(content_ids)
                    actual_first = content_ids[0] if content_ids else None
                    mismatches += int(
                        not encoding.prefix_ok
                        or actual_count != row["d_content_count"]
                        or actual_first != row["d_first_id"]
                    )
        return mismatches

    _write_check(checks, logger, "C50", "Cross-context distractor tokenization (batch encoded once per pair)", 0, c50)

    def c51() -> dict[str, Any]:
        """Create and verify deterministic stratified E8 order rows."""
        _, concept_map = load_concepts()
        main_translation_ids = {
            row["concept_id"]
            for row in load_all_prompts()[cfg["data"]["primary_prompt_files"]["translation_diac"]]
            if row["m1_extension"] is False
        }
        rows = e8_order_rows(
            [concept_map[concept_id] for concept_id in sorted(main_translation_ids)], int(cfg["seed"])
        )
        bundle["e8_order"] = rows
        by_family = {family: [row for row in rows if row["family"] == family] for family in FAMILIES}
        return {
            "rows_per_family": {family: len(by_family[family]) for family in FAMILIES},
            "sha256": {family: _e8_checksum(rows, family) for family in FAMILIES},
            "concept_sets_match": all({row["concept_id"] for row in by_family[family]} == main_translation_ids for family in FAMILIES),
        }

    _write_check(
        checks,
        logger,
        "C51",
        "Seeded E8 order rows, checksums, and concept sets",
        {
            "rows_per_family": {family: facts["e8_rows_per_family"] for family in FAMILIES},
            "sha256": facts["e8_order_sha256"],
            "concept_sets_match": True,
        },
        c51,
        compare_expected,
    )

    def c52() -> dict[str, Any]:
        """Build E0 distractors for the already cached 50-item directions sample."""
        _, concept_map = load_concepts()
        sample_rows = bundle["e0_pilot"]
        sample_ids = [row["concept_id"] for row in sample_rows]
        output: list[dict[str, Any]] = []
        for family in FAMILIES:
            for lang, variant in cfg["manifests"]["e0_combos"]:
                output.extend(
                    _match_distractors(
                        concept_map,
                        e0_stats,
                        family=family,
                        lang=lang,
                        variant=variant,
                        pool_ids=sample_ids,
                        seed=int(cfg["seed"]),
                        key_prefix="e0_distractor",
                    )
                )
        bundle["e0_distractors"] = sorted(
            output,
            key=lambda row: (row["family"], row["target_lang"], row["target_variant"], row["concept_id"]),
        )
        sample_hash = hashlib.sha256("".join(
            f"{row['rank']}\t{row['concept_id']}\n" for row in sample_rows
        ).encode("utf-8")).hexdigest()
        bad_self = sum(row["concept_id"] == row["distractor_id"] for row in bundle["e0_distractors"])
        return {
            "sample_sha256": sample_hash,
            "sample_distinct": len(sample_ids) == len(set(sample_ids)),
            "sample_size": len(sample_ids),
            "distractor_rows_per_family": {family: sum(row["family"] == family for row in bundle["e0_distractors"]) for family in FAMILIES},
            "self_matches": bad_self,
        }

    _write_check(
        checks,
        logger,
        "C52",
        "E0 sample checksum and distinctness; E0 distractor sizes and non-self matching",
        {
            "sample_sha256": facts["e0_sample_sha256"],
            "sample_distinct": True,
            "sample_size": int(cfg["manifests"]["e0_sample_size"]),
            "distractor_rows_per_family": {family: facts["e0_distractor_rows_per_family"] for family in FAMILIES},
            "self_matches": 0,
        },
        c52,
        compare_expected,
    )

    repo_logs = repo / cfg["paths"]["logs"]
    repo_logs.mkdir(parents=True, exist_ok=True)
    temp1 = tempfile.TemporaryDirectory(prefix="x0d-build-a-", dir=repo_logs)
    temp2 = tempfile.TemporaryDirectory(prefix="x0d-build-b-", dir=repo_logs)
    generated_dir = Path(temp1.name) / "derived"
    second_dir = Path(temp2.name) / "derived"
    identical_to_committed = False
    try:
        _render_files(bundle, cfg, repo, generated_dir)
        _render_files(bundle, cfg, repo, second_dir)

        def c53() -> bool:
            """Compare two independent rendered builds byte-for-byte."""
            identical, different = compare_directories(generated_dir, second_dir)
            logger.info("C53 differing files=%s", different)
            return identical

        _write_check(checks, logger, "C53", "Two generated manifest trees are byte-identical", True, c53)
        if check_mode:
            identical_to_committed, differing = compare_directories(generated_dir, repo / cfg["paths"]["derived"])
            logger.info("--check committed comparison identical=%s differing_files=%s", identical_to_committed, differing)
        else:
            refuse_nonempty_derived(repo / cfg["paths"]["derived"])
            all_match = all(row["result"] == "MATCH" for row in checks)
            if all_match:
                final_dir = repo / cfg["paths"]["derived"]
                final_dir.mkdir(parents=True, exist_ok=True)
                for file_name in _render_files(bundle, cfg, repo, generated_dir):
                    os.replace(generated_dir / file_name, final_dir / file_name)
                logger.info("Published %d derived manifest files after all checks passed", len(OUTPUT_FILES))
            else:
                logger.error("At least one check failed; no files were written to derived/")
    finally:
        temp1.cleanup()
        temp2.cleanup()

    bundle["run_check_committed_identical"] = identical_to_committed
    bundle["output_files"] = list(OUTPUT_FILES)
    return checks, bundle, (repo / cfg["paths"]["derived"] if (repo / cfg["paths"]["derived"]).is_dir() else None), identical_to_committed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the X0d generator or its byte-for-byte reproducibility check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--check", action="store_true", help="compare generated files to the frozen derived directory")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    repo = Path.cwd()
    logger = _setup_logger(repo / cfg["paths"]["logs"])
    run_start = time.perf_counter()
    run_utc = _utc_now()
    logger.info("T03b start=%s mode=%s config_sha256=%s", run_utc, "check" if args.check else "build", config_sha256(args.config))
    if not args.check:
        try:
            refuse_nonempty_derived(repo / cfg["paths"]["derived"])
        except Exception:
            logger.exception("Default build refused before generation")
            return 2
    checks: list[dict[str, str]] = []
    bundle: dict[str, Any] = {}
    generated_dir: Path | None = None
    identical_to_committed = False
    try:
        checks, bundle, generated_dir, identical_to_committed = run_audit(
            cfg, repo, logger, check_mode=args.check
        )
    except Exception:
        logger.exception("Unexpected failure outside an individual check")
    finally:
        check_counts = Counter(row["result"] for row in checks)
        concepts: dict[str, dict[str, Any]] | None = None
        try:
            frame = load_concepts_jsonl(repo / cfg["paths"]["concepts_jsonl"])
            concepts = _concept_mapping(frame)
        except Exception:
            logger.exception("Could not reload concepts for final report");
        destination = repo / cfg["paths"]["derived"]
        quality = _quality_rows(bundle.get("distractors", []))
        _write_quality_csv(repo / cfg["paths"]["reports"] / "X0d_distractor_quality.csv", quality)
        try:
            _write_report(
                repo / cfg["paths"]["reports"] / "X0.md",
                cfg=cfg,
                repo=repo,
                checks=checks,
                bundle=bundle,
                concepts=concepts,
                derived_dir=destination,
                logger=logger,
            )
        except Exception:
            logger.exception("Could not write the X0 checkpoint report")
        elapsed = time.perf_counter() - run_start
        logger.info("check summary=%s", dict(sorted(check_counts.items())))
        logger.info("run end=%s wall_time_seconds=%.3f", _utc_now(), elapsed)
        if args.check:
            logger.info("--check result=%s", "IDENTICAL" if identical_to_committed else "DIFFERENT")
    passed = len(checks) == 13 and all(row["result"] == "MATCH" for row in checks)
    if args.check:
        passed = passed and identical_to_committed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
