#!/datastore/luannt/.cache/sb/bin/python
"""Verify the pinned X0-b inputs without loading tokenizers or models."""

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard; env_guard.guard(offline=True)

import argparse
import csv
import importlib.util
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from vilens.config import load_config
from vilens.checks import count_mask_disagreements
from vilens.data import JSON_EXPECTED_COLUMNS, load_concepts_jsonl, load_concepts_tsv, load_prompt_file, prompt_file_paths
from vilens.manifest import verify_manifest
from vilens.prompts import check_prompt_integrity


def _display(value: Any) -> str:
    """Format check values compactly and deterministically for CSV/Markdown."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_raw_tsv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Read raw TSV strings while skipping the pipeline comment line."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        next(stream)
        reader = csv.DictReader(stream, delimiter="\t")
        rows = {row["concept_id"]: dict(row) for row in reader}
        return list(reader.fieldnames or []), rows


def _read_raw_jsonl(path: Path) -> dict[str, dict[str, str]]:
    """Read raw JSONL dictionaries without casting any values."""
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["concept_id"]] = row
    return rows


def _setup_logger(log_path: Path) -> logging.Logger:
    """Create a stdout and file logger for one verification run."""
    logger = logging.getLogger("x0_verify")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _text_helper_parity(concepts: pd.DataFrame, prep_common_path: Path) -> dict[str, int]:
    """Compare local text helpers with the pinned preparation implementation."""
    module_spec = importlib.util.spec_from_file_location("prep_common_x0", prep_common_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("could not load preparation common.py")
    prep = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(prep)
    from vilens import text as local_text

    form_mismatches = 0
    for row in concepts.itertuples(index=False):
        for language in ("vi", "en", "zh", "fr", "id"):
            value = getattr(row, language)
            for preserve_case in (False, True):
                if local_text.strip_diacritics(value, preserve_case=preserve_case) != prep.strip_diacritics(value, preserve_case=preserve_case):
                    form_mismatches += 1
    distance_mismatches = 0
    for left_name, right_name in (("vi", "en"), ("en", "fr"), ("en", "id")):
        for row in concepts.itertuples(index=False):
            left, right = getattr(row, left_name), getattr(row, right_name)
            if local_text.normalized_levenshtein(left, right) != prep.normalized_levenshtein(left, right):
                distance_mismatches += 1
    return {"form_mismatches": form_mismatches, "distance_mismatches": distance_mismatches}


def _run_checks(cfg: dict[str, Any], repo: Path, logger: logging.Logger) -> list[dict[str, str]]:
    """Run C01–C20 and return the tabular results, retaining every failed gate."""
    checks: list[dict[str, str]] = []
    facts = cfg["reference_facts"]
    json_path = repo / cfg["paths"]["concepts_jsonl"]
    tsv_path = repo / cfg["paths"]["concepts_tsv"]
    manifest_path = repo / cfg["paths"]["manifest"]
    prep_root = repo / cfg["paths"]["prep_repo"]
    concepts: pd.DataFrame | None = None
    concept_table: pd.DataFrame | None = None
    prompts_by_stem: dict[str, list[dict[str, Any]]] = {}
    all_prompts: list[dict[str, Any]] = []

    def add(check_id: str, description: str, expected: Any, actual: Any) -> None:
        """Append a single check row and log its outcome."""
        result = "MATCH" if actual == expected else "MISMATCH"
        row = {
            "check_id": check_id,
            "description": description,
            "expected": _display(expected),
            "actual": _display(actual),
            "result": result,
        }
        checks.append(row)
        logger.info("%s %s expected=%s actual=%s", check_id, result, row["expected"], row["actual"])

    def run(check_id: str, description: str, expected: Any, operation: Callable[[], Any]) -> Any:
        """Run one check, converting exceptions to a visible mismatch row."""
        try:
            actual = operation()
        except Exception as exc:  # an independent later check should still be reported
            actual = f"ERROR {type(exc).__name__}: {exc}"
        add(check_id, description, expected, actual)
        return actual

    def check_manifest() -> str:
        """Verify every pinned input and summarize successful checksums."""
        result = verify_manifest(manifest_path, repo)
        okay = sum(status for _, status in result)
        return f"{okay}/{len(result)} OK, {len(result) - okay} FAILED"

    run("C01", "Manifest SHA-256 verification", "31/31 OK, 0 FAILED", check_manifest)

    def check_json() -> dict[str, Any]:
        """Load JSONL concepts and report counts without any deprecated field."""
        nonlocal concepts
        concepts = load_concepts_jsonl(json_path)
        with json_path.open("r", encoding="utf-8") as stream:
            first = json.loads(next(stream))
        return {
            "rows": len(concepts),
            "unique_ids": bool(concepts["concept_id"].is_unique),
            "raw_columns": len(first),
            "cloze_available_dropped": "cloze_available" not in concepts.columns,
        }

    run(
        "C02",
        "Strict JSONL concepts load; rows, unique IDs, raw columns before deprecated-column drop",
        {"rows": facts["concepts_total"], "unique_ids": True, "raw_columns": 69, "cloze_available_dropped": True},
        check_json,
    )

    def check_tsv() -> dict[str, Any]:
        """Load the 71-column frozen TSV and compare its concept ID set."""
        nonlocal concept_table
        concept_table = load_concepts_tsv(tsv_path)
        header, raw = _read_raw_tsv(tsv_path)
        json_ids = set(concepts["concept_id"].astype(str)) if concepts is not None else set()
        return {
            "rows": len(concept_table),
            "raw_columns": len(header),
            "loaded_columns": len(concept_table.columns),
            "same_concept_ids": set(raw) == json_ids,
        }

    run(
        "C03",
        "Strict TSV concepts load; row and column counts and concept ID set",
        {"rows": facts["concepts_total"], "raw_columns": 71, "loaded_columns": 71, "same_concept_ids": True},
        check_tsv,
    )

    def check_raw_shared() -> dict[str, Any]:
        """Compare all shared JSON and TSV fields as original strings."""
        raw_json = _read_raw_jsonl(json_path)
        _, raw_tsv = _read_raw_tsv(tsv_path)
        mismatch_count = 0
        shared_columns = set(raw_json[next(iter(raw_json))]) & set(JSON_EXPECTED_COLUMNS)
        for concept_id, json_row in raw_json.items():
            tsv_row = raw_tsv.get(concept_id)
            if tsv_row is None:
                mismatch_count += len(shared_columns)
                continue
            mismatch_count += sum(json_row[column] != tsv_row.get(column) for column in shared_columns)
        return {"shared_columns": len(shared_columns), "value_mismatches": mismatch_count}

    run("C04", "Raw-string equality for shared concept columns", {"shared_columns": 69, "value_mismatches": 0}, check_raw_shared)

    def check_concreteness_na() -> dict[str, Any]:
        """Compare only the concreteness missingness mask and aggregate count."""
        if concepts is None or concept_table is None:
            raise RuntimeError("concept tables unavailable")
        expected_na = concepts["concreteness_match"].eq("none")
        actual_na = concept_table["concreteness"].isna()
        return {"na_iff_none": bool((expected_na.to_numpy() == actual_na.to_numpy()).all()), "na_count": int(actual_na.sum())}

    run("C05", "Concreteness missingness mask against match status (aggregate only)", {"na_iff_none": True, "na_count": facts["concreteness_missing"]}, check_concreteness_na)

    def check_splits() -> dict[str, int]:
        """Count concept rows by preregistered split."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        return {key: int((concepts["split"] == key).sum()) for key in facts["split_counts"]}

    run("C06", "Concept split counts", facts["split_counts"], check_splits)

    def check_main_and_extension() -> dict[str, Any]:
        """Count main test and extension items and main-test strata."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        main = concepts[(concepts["split"] == "test") & ~concepts["m1_extension"]]
        extension = concepts[(concepts["split"] == "test") & concepts["m1_extension"]]
        strata = {key: int((main["stratum"] == key).sum()) for key in facts["strata_main"]}
        return {"main_test": len(main), "m1_extension": len(extension), "strata_main": strata}

    run("C07", "Main test and M1 extension counts; main strata", {"main_test": facts["main_test"], "m1_extension": facts["m1_extension"], "strata_main": facts["strata_main"]}, check_main_and_extension)

    def check_eligible() -> str:
        """Count M1 eligibility values and compare NA masks by their boolean values."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        names = {"gemma": "m1_eligible_gemma", "qwen": "m1_eligible_qwen", "llama": "m1_eligible_llama"}
        counts = {name: int(concepts[column].fillna(False).sum()) for name, column in names.items()}
        order = ("gemma", "qwen", "llama")
        disagreements = {
            name: count_mask_disagreements(concepts[names[name]].isna(), concepts["split"] != "test")
            for name in order
        }
        na_counts = {name: int(concepts[names[name]].isna().sum()) for name in order}
        disagreement_text = "/".join(str(disagreements[name]) for name in order)
        na_text = "/".join(str(na_counts[name]) for name in order)
        true_text = "/".join(str(counts[name]) for name in order)
        return (
            f"disagreeing rows gemma/qwen/llama = {disagreement_text}; "
            f"NA counts = {na_text}; true counts = {true_text}"
        )

    eligible_order = ("gemma", "qwen", "llama")
    eligible_true = "/".join(str(facts["m1_eligible"][name]) for name in eligible_order)
    eligible_na = "/".join(str(facts["m1_eligible_na_rows"]) for _ in eligible_order)
    eligible_expected = (
        f"disagreeing rows gemma/qwen/llama = 0/0/0; "
        f"NA counts = {eligible_na}; true counts = {eligible_true}"
    )
    run("C08", "M1 eligibility true counts and NA mask", eligible_expected, check_eligible)

    def check_single_token() -> dict[str, int]:
        """Count single-token Vietnamese flags over main-test concepts."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        main = concepts[(concepts["split"] == "test") & ~concepts["m1_extension"]]
        return {
            "gemma": int(main["single_token_vi_gemma"].sum()),
            "qwen": int(main["single_token_vi_qwen"].sum()),
            "llama": int(main["single_token_vi_llama"].sum()),
        }

    run("C09", "Single-token Vietnamese counts on main test", facts["single_token_vi_main"], check_single_token)

    def check_fewshot() -> dict[str, Any]:
        """Check selected few-shot concepts and the six collapsed demonstration forms."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        selected = concepts[concepts["fewshot_set"].notna()]
        per_set = {str(number): int((selected["fewshot_set"] == number).sum()) for number in range(1, 7)}
        all_reservoir = bool((selected["split"] == "fewshot_reservoir").all())
        collapsed = selected[selected["collapsed"]]
        collapsed_rows = [
            {"set": int(row.fewshot_set), "vi": str(row.vi)}
            for row in collapsed.itertuples(index=False)
        ]
        collapsed_rows.sort(key=lambda item: (item["set"], item["vi"]))
        expected_collapsed = sorted(facts["collapsed_fewshot_demos"], key=lambda item: (item["set"], item["vi"]))
        return {
            "selected": len(selected),
            "per_set": per_set,
            "all_in_fewshot_reservoir": all_reservoir,
            "collapsed_demos": collapsed_rows,
        }

    run(
        "C10",
        "Few-shot counts, reservoir membership, and collapsed demonstration set",
        {"selected": facts["fewshot_selected"], "per_set": {str(number): 5 for number in range(1, 7)}, "all_in_fewshot_reservoir": True, "collapsed_demos": sorted(facts["collapsed_fewshot_demos"], key=lambda item: (item["set"], item["vi"]))},
        check_fewshot,
    )

    def check_ru_missing() -> dict[str, int]:
        """Count only Russian-form missingness in the specified concept splits."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        main = concepts[(concepts["split"] == "test") & ~concepts["m1_extension"]]
        reservoir = concepts[concepts["split"] == "fewshot_reservoir"]
        return {"main_test": int(main["ru"].isna().sum()), "fewshot_reservoir": int(reservoir["ru"].isna().sum())}

    run("C11", "Russian-form missingness in main test and few-shot reservoir", facts["ru_missing"], check_ru_missing)

    def check_questionable() -> dict[str, int]:
        """Confirm each flagged concept is present and belongs to main test."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        ids = set(concepts["concept_id"].astype(str))
        main_ids = set(concepts.loc[(concepts["split"] == "test") & ~concepts["m1_extension"], "concept_id"].astype(str))
        target_ids = facts["questionable_targets"]
        return {"present": sum(item in ids for item in target_ids), "main_test": sum(item in main_ids for item in target_ids)}

    run("C12", "Questionable target identifiers exist and are main-test concepts", {"present": 6, "main_test": 6}, check_questionable)

    def load_all_prompts() -> dict[str, int]:
        """Load all 27 prompt files and validate the family schemas."""
        nonlocal all_prompts, prompts_by_stem
        paths = prompt_file_paths(cfg)
        prompts_by_stem = {stem: load_prompt_file(path) for stem, path in paths.items()}
        all_prompts = [record for records in prompts_by_stem.values() for record in records]
        return {stem: len(records) for stem, records in prompts_by_stem.items()}

    prompt_expected: dict[str, int] = {}
    for family in ("repetition", "translation"):
        for condition in ("diac", "nodiac"):
            count = facts["prompt_records"][f"{family}_{condition}"]["total"]
            for set_no in range(1, 7):
                prompt_expected[f"{family}_{condition}_set{set_no}"] = count
    prompt_expected["cloze_diac_set1"] = facts["prompt_records"]["cloze_diac"]["total"]
    prompt_expected["cloze_nodiac_set1"] = facts["prompt_records"]["cloze_nodiac"]["total"]
    prompt_expected["directions_matched"] = facts["prompt_records"]["directions"]["total"]
    run("C13", "All prompt families and per-file record counts", prompt_expected, load_all_prompts)

    def check_prompt_integrity_all() -> dict[str, int]:
        """Check prompt NFC/whitespace and concept identifier membership."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        known_ids = set(concepts["concept_id"].astype(str))
        return {
            "prompt_problems": len(check_prompt_integrity(all_prompts)),
            "unknown_concept_ids": sum(record["concept_id"] not in known_ids for record in all_prompts),
        }

    run("C14", "Prompt integrity and concept ID membership", {"prompt_problems": 0, "unknown_concept_ids": 0}, check_prompt_integrity_all)

    def check_prompt_splits_sets() -> int:
        """Check prompt split assignments and repetition/translation/cloze set numbers."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        concept_split = dict(zip(concepts["concept_id"].astype(str), concepts["split"].astype(str)))
        errors = 0
        for stem, records in prompts_by_stem.items():
            if stem == "directions_matched":
                errors += sum(concept_split.get(record["concept_id"]) != "directions" for record in records)
                continue
            errors += sum(record["split"] != "test" or concept_split.get(record["concept_id"]) != "test" for record in records)
            if stem.startswith("cloze_"):
                errors += sum(record["fewshot_set"] is not None for record in records)
            else:
                file_set = int(stem.rsplit("set", 1)[1])
                errors += sum(record["fewshot_set"] != file_set for record in records)
        return errors

    run("C15", "Prompt-to-concept splits and per-file few-shot set assignments", 0, check_prompt_splits_sets)

    def check_nodiac_collapsed() -> int:
        """Count nodiac prompts whose concept is marked collapsed."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        collapsed_by_id = dict(zip(concepts["concept_id"].astype(str), concepts["collapsed"]))
        return sum(
            bool(collapsed_by_id.get(record["concept_id"], False))
            for stem, records in prompts_by_stem.items() if "_nodiac_" in stem
            for record in records
        )

    run("C16", "No nodiac prompt uses a collapsed concept", 0, check_nodiac_collapsed)

    def check_cloze() -> dict[str, Any]:
        """Check cloze main strata, demos, and that demos are not test concepts."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        concept_by_id = concepts.set_index("concept_id", drop=False)
        actual_strata: dict[str, dict[str, int]] = {}
        demo_list = facts["cloze_demo_concept_ids"]
        demo_mismatches = 0
        test_demo_count = 0
        for condition, stem in (("diac", "cloze_diac_set1"), ("nodiac", "cloze_nodiac_set1")):
            records = prompts_by_stem[stem]
            counts = {stratum: 0 for stratum in facts["cloze_main_strata"][condition]}
            for record in records:
                if record["m1_extension"]:
                    continue
                stratum = str(concept_by_id.loc[record["concept_id"], "stratum"])
                counts[stratum] = counts.get(stratum, 0) + 1
            actual_strata[condition] = counts
            for record in records:
                demo_mismatches += record["demo_concept_ids"] != demo_list
                for demo_id in record["demo_concept_ids"]:
                    test_demo_count += demo_id not in concept_by_id.index or concept_by_id.loc[demo_id, "split"] == "test"
        return {
            "main_strata": actual_strata,
            "demo_list_mismatches": int(demo_mismatches),
            "test_or_missing_demo_ids": int(test_demo_count),
            "cloze_records": sum(len(prompts_by_stem[stem]) for stem in ("cloze_diac_set1", "cloze_nodiac_set1")),
        }

    run("C17", "Cloze main strata and configured non-test demonstrations", {"main_strata": facts["cloze_main_strata"], "demo_list_mismatches": 0, "test_or_missing_demo_ids": 0, "cloze_records": 70}, check_cloze)

    def check_directions() -> dict[str, Any]:
        """Validate directions grid size and concept-level split consistency."""
        records = prompts_by_stem["directions_matched"]
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_concept[record["concept_id"]].append(record)
        expected_combos = {(lang, fmt) for lang in ("vi", "en", "zh", "fr", "id") for fmt in ("repetition", "translation")}
        complete = all({(record["lang"], record["format"]) for record in rows} == expected_combos for rows in by_concept.values())
        constant = all(len({record["split"] for record in rows}) == 1 for rows in by_concept.values())
        splits = {split: sum(rows[0]["split"] == split for rows in by_concept.values()) for split in ("fit", "validate")}
        return {"records": len(records), "concepts": len(by_concept), "fit_concepts": splits["fit"], "validate_concepts": splits["validate"], "complete_cross_product": complete, "split_constant": constant}

    directions_facts = facts["prompt_records"]["directions"]
    run("C18", "Directions concept × language × format grid", {"records": directions_facts["total"], "concepts": directions_facts["concepts"], "fit_concepts": directions_facts["fit_concepts"], "validate_concepts": directions_facts["validate_concepts"], "complete_cross_product": True, "split_constant": True}, check_directions)

    def check_disjoint_splits() -> int:
        """Count few-shot or direction concepts used in test prompt files."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        prohibited = set(concepts.loc[concepts["split"].isin(["fewshot_reservoir", "directions"]), "concept_id"].astype(str))
        study_ids = {
            record["concept_id"]
            for stem, records in prompts_by_stem.items()
            if stem != "directions_matched"
            for record in records
        }
        return len(prohibited & study_ids)

    run("C19", "No few-shot or directions concepts occur in rep/translation/cloze files", 0, check_disjoint_splits)

    def check_text_parity() -> dict[str, int]:
        """Compare ported helpers with the pinned preparation source."""
        if concepts is None:
            raise RuntimeError("concept table unavailable")
        return _text_helper_parity(concepts, prep_root / "data/build/common.py")

    run("C20", "Preparation-repo text-helper parity over all concepts and form pairs", {"form_mismatches": 0, "distance_mismatches": 0}, check_text_parity)
    return checks


def _write_outputs(checks: list[dict[str, str]], report_dir: Path, logger: logging.Logger) -> bool:
    """Write the check CSV and Markdown summary and return whether all checks matched."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "X0b_checks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check_id", "description", "expected", "actual", "result"))
        writer.writeheader()
        writer.writerows(checks)
    matched = sum(row["result"] == "MATCH" for row in checks)
    failed = len(checks) - matched
    summary = f"X0b verification: {matched}/{len(checks)} MATCH; {failed} MISMATCH."
    lines = ["# X0b data and scaffold checks", "", summary, "", "| ID | Description | Expected | Actual | Result |", "|---|---|---|---|---|"]
    lines.extend(
        f"| {row['check_id']} | {row['description']} | {row['expected']} | {row['actual']} | {row['result']} |"
        for row in checks
    )
    (report_dir / "X0b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(summary)
    return failed == 0


def main() -> int:
    """Run X0-b checks from the repository root and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml", help="experiment config path")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = load_config(cfg_path)
    log_dir = REPO / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = _setup_logger(log_dir / f"x0_verify_{stamp}.log")
    checks = _run_checks(cfg, REPO, logger)
    okay = _write_outputs(checks, REPO / cfg["paths"]["reports"], logger)
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
