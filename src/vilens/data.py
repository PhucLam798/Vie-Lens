"""Strict readers for the frozen concept and prompt inputs."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

REQUIRED_STR = {
    "concept_id", "vi", "en", "zh", "fr", "id", "vi_nodiac", "en_sense_gloss",
}
ENUM_VALUES: dict[str, set[str]] = {
    "split": {"test", "directions", "fewshot_reservoir"},
    "pos": {"noun", "verb", "adj"},
    "stratum": {"sino", "nonsino", "ambiguous", "other_loan"},
    "sigA": {"sino", "nonsino", "conflict", "other_loan"},
    "sigB": {"sino", "nonsino"},
    "sigB_strict": {"sino", "nonsino"},
    "sigB_relaxed": {"sino", "nonsino"},
    "bt_route": {"strict", "rt_contain", "fwd_only", "none"},
    "concreteness_match": {"exact", "head", "none"},
    "reading_source": {"unihan", "wiktionary_char", "unihan+wiktionary_char"},
}
REQUIRED_ENUM = set(ENUM_VALUES) - {"reading_source"}
NULLABLE_STR = {"ru", "reading_source"}
JSON_LIST = {"vi_alts", "sino_via"}
REQUIRED_BOOL = {
    "collapsed", "native_strict", "in_vi_gloss", "in_wikidata", "external_attested",
    "bt_pass_strict_v1", "zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree",
    "single_token_vi_gemma", "single_token_vi_qwen", "single_token_vi_llama", "m1_extension",
    "cloze_available",
}
NULLABLE_BOOL = {"ru_nllb_agree", "m1_eligible_gemma", "m1_eligible_qwen", "m1_eligible_llama"}
REQUIRED_INT = {
    "n_sources", "n_senses_vi", "n_syllables", "n_chars_vi", "n_chars_en", "n_chars_zh",
    "n_chars_fr", "n_chars_id",
}
REQUIRED_INT.update(
    f"ntok_{model}_{lang}"
    for model in ("gemma", "qwen", "llama")
    for lang in ("vi", "en", "zh", "fr", "id")
)
NULLABLE_INT = {"fewshot_set"}
REQUIRED_FLOAT = {
    "min_surface_dist", "zipf_vi", "zipf_en", "zipf_zh", "zipf_fr", "zipf_id",
}
JSON_COLUMNS = (
    REQUIRED_STR | REQUIRED_ENUM | NULLABLE_STR | JSON_LIST | REQUIRED_BOOL | NULLABLE_BOOL
    | REQUIRED_INT | NULLABLE_INT | REQUIRED_FLOAT
)
TSV_EXTRA_COLUMNS = {"in_muse", "concreteness"}
JSON_EXPECTED_COLUMNS = JSON_COLUMNS
TSV_EXPECTED_COLUMNS = JSON_COLUMNS | TSV_EXTRA_COLUMNS


def _parse_bool(value: str, column: str, row_number: int, nullable: bool = False) -> bool | None:
    """Parse a strict lowercase boolean literal or nullable empty value."""
    if value == "true":
        return True
    if value == "false":
        return False
    if nullable and value == "":
        return None
    if column == "in_muse":
        raise ValueError(f"row {row_number}: {column} must be 'true' or 'false'")
    raise ValueError(f"row {row_number}: {column} must be 'true' or 'false', got {value!r}")


def _parse_int(value: str, column: str, row_number: int, nullable: bool = False) -> int | None:
    """Parse an integer literal or nullable empty value, rejecting other forms."""
    if nullable and value == "":
        return None
    if re.fullmatch(r"-?[0-9]+", value) is None:
        raise ValueError(f"row {row_number}: {column} must be an integer, got {value!r}")
    parsed = int(value)
    if column == "fewshot_set" and parsed not in range(1, 7):
        raise ValueError(f"row {row_number}: fewshot_set must be 1–6, got {parsed}")
    return parsed


def _parse_float(value: str, column: str, row_number: int, nullable: bool = False) -> float | None:
    """Parse a finite numeric literal or nullable empty value."""
    if nullable and value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        if column == "concreteness":
            raise ValueError(f"row {row_number}: {column} must be numeric") from exc
        raise ValueError(f"row {row_number}: {column} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        if column == "concreteness":
            raise ValueError(f"row {row_number}: {column} must be finite")
        raise ValueError(f"row {row_number}: {column} must be finite, got {value!r}")
    return parsed


def _cast_rows(
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    source: str,
    tsv: bool = False,
) -> pd.DataFrame:
    """Validate raw string records and cast every field to its declared pandas dtype."""
    parsed_rows: list[dict[str, Any]] = []
    bool_columns = REQUIRED_BOOL | ({"in_muse"} if tsv else set())
    nullable_bool_columns = NULLABLE_BOOL
    int_columns = REQUIRED_INT | NULLABLE_INT
    float_columns = REQUIRED_FLOAT | ({"concreteness"} if tsv else set())
    string_columns = REQUIRED_STR | REQUIRED_ENUM | NULLABLE_STR
    for index, raw in enumerate(rows, start=1):
        if set(raw) != set(columns):
            raise ValueError(f"{source} row {index}: column set changed during parsing")
        row: dict[str, Any] = {}
        for column in columns:
            value = raw[column]
            if not isinstance(value, str):
                raise ValueError(f"{source} row {index}: {column} must be a string")
            if column in REQUIRED_STR:
                if value == "":
                    raise ValueError(f"{source} row {index}: {column} must be non-empty")
                row[column] = value
            elif column in REQUIRED_ENUM:
                if value not in ENUM_VALUES[column]:
                    raise ValueError(f"{source} row {index}: invalid {column} value {value!r}")
                row[column] = value
            elif column in NULLABLE_STR:
                if value == "":
                    row[column] = None
                elif column in ENUM_VALUES and value not in ENUM_VALUES[column]:
                    raise ValueError(f"{source} row {index}: invalid {column} value {value!r}")
                else:
                    row[column] = value
            elif column in JSON_LIST:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source} row {index}: {column} is not valid JSON") from exc
                if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
                    raise ValueError(f"{source} row {index}: {column} must be a JSON list of strings")
                row[column] = parsed
            elif column in bool_columns:
                row[column] = _parse_bool(value, column, index)
            elif column in nullable_bool_columns:
                row[column] = _parse_bool(value, column, index, nullable=True)
            elif column in int_columns:
                row[column] = _parse_int(value, column, index, nullable=column in NULLABLE_INT)
            elif column in float_columns:
                row[column] = _parse_float(value, column, index, nullable=(column == "concreteness"))
            else:
                raise ValueError(f"{source} row {index}: no schema for column {column}")
        parsed_rows.append(row)
    frame = pd.DataFrame(parsed_rows, columns=columns)
    for column in REQUIRED_STR | REQUIRED_ENUM:
        frame[column] = pd.array(frame[column], dtype="string")
    for column in NULLABLE_STR:
        frame[column] = pd.array(frame[column], dtype="string")
    for column in bool_columns | nullable_bool_columns:
        frame[column] = pd.array(frame[column], dtype="boolean")
    for column in int_columns:
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in float_columns:
        frame[column] = pd.array(frame[column], dtype="Float64")
    if "vi_alts" in frame.columns:
        frame["vi_alts"] = pd.Series(parsed_rows and [row["vi_alts"] for row in parsed_rows] or [], dtype=object)
    if "sino_via" in frame.columns:
        frame["sino_via"] = pd.Series(parsed_rows and [row["sino_via"] for row in parsed_rows] or [], dtype=object)
    if frame["concept_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["concept_id"].duplicated(), "concept_id"].tolist()
        raise ValueError(f"{source}: concept_id must be unique; duplicates={duplicate_ids[:5]}")
    frame = frame.sort_values("concept_id", kind="stable").reset_index(drop=True)
    return frame


def load_concepts_jsonl(path: str | Path) -> pd.DataFrame:
    """Load string-valued concepts JSONL using the strict 69-column schema."""
    jsonl_path = Path(path)
    rows: list[dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{jsonl_path}:{line_no}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: malformed JSON") from exc
            if not isinstance(value, dict) or set(value) != JSON_EXPECTED_COLUMNS:
                actual = set(value) if isinstance(value, dict) else set()
                raise ValueError(
                    f"{jsonl_path}:{line_no}: expected {len(JSON_EXPECTED_COLUMNS)} columns; "
                    f"missing={sorted(JSON_EXPECTED_COLUMNS - actual)}, extra={sorted(actual - JSON_EXPECTED_COLUMNS)}"
                )
            if any(not isinstance(item, str) for item in value.values()):
                raise ValueError(f"{jsonl_path}:{line_no}: every raw JSONL value must be a string")
            rows.append(value)
    frame = _cast_rows(rows, sorted(JSON_EXPECTED_COLUMNS), source=str(jsonl_path))
    frame = frame.drop(columns=["cloze_available"])
    _LOG.info("Dropped deprecated cloze_available column from %s", jsonl_path)
    return frame


def load_concepts_tsv(path: str | Path) -> pd.DataFrame:
    """Load the frozen concept TSV with explicit string parsing and pandas dtypes."""
    tsv_path = Path(path)
    with tsv_path.open("r", encoding="utf-8", newline="") as stream:
        first_line = stream.readline()
        if not first_line.startswith("#"):
            raise ValueError(f"{tsv_path}: expected a leading pipeline comment line")
        header = next(csv.reader([stream.readline()], delimiter="\t"), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{tsv_path}: duplicate column names")
    if set(header) != TSV_EXPECTED_COLUMNS:
        raise ValueError(
            f"{tsv_path}: unexpected columns; missing={sorted(TSV_EXPECTED_COLUMNS - set(header))}, "
            f"extra={sorted(set(header) - TSV_EXPECTED_COLUMNS)}"
        )
    table = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, skiprows=1)
    if list(table.columns) != header:
        raise ValueError(f"{tsv_path}: TSV header changed while loading")
    raw_rows = table.to_dict(orient="records")
    # TSV keeps all 71 source fields for audit; deprecated cloze_available is validated but unused.
    return _cast_rows(raw_rows, header, source=str(tsv_path), tsv=True)


def _validate_prompt_record(record: dict[str, Any], family: str, source: str, line_no: int) -> None:
    """Validate a prompt record against its family-specific key and value schema."""
    common_keys = {
        "concept_id", "condition", "fewshot_set", "format", "m1_extension", "prompt", "split",
        "target_en", "target_fr", "target_id", "target_vi", "target_zh",
    }
    if family in {"repetition", "translation"}:
        expected = common_keys
    elif family == "cloze":
        expected = common_keys | {"demo_concept_ids"}
    elif family == "directions":
        expected = {"concept_id", "format", "lang", "prompt", "split", "target"}
    else:
        raise ValueError(f"{source}: unknown prompt family {family!r}")
    if set(record) != expected:
        raise ValueError(
            f"{source}:{line_no}: wrong keys; missing={sorted(expected - set(record))}, "
            f"extra={sorted(set(record) - expected)}"
        )
    if family == "directions":
        for key in ("concept_id", "prompt", "target"):
            if not isinstance(record[key], str):
                raise ValueError(f"{source}:{line_no}: {key} must be a string")
        if record["lang"] not in {"vi", "en", "zh", "fr", "id"}:
            raise ValueError(f"{source}:{line_no}: invalid direction language")
        if record["format"] not in {"repetition", "translation"}:
            raise ValueError(f"{source}:{line_no}: invalid direction format")
        if record["split"] not in {"fit", "validate"}:
            raise ValueError(f"{source}:{line_no}: invalid directions split")
        return
    string_fields = {"concept_id", "condition", "format", "prompt", "split", "target_en", "target_fr", "target_id", "target_vi", "target_zh"}
    if any(not isinstance(record[key], str) for key in string_fields):
        raise ValueError(f"{source}:{line_no}: prompt text and labels must be strings")
    if type(record["m1_extension"]) is not bool:
        raise ValueError(f"{source}:{line_no}: m1_extension must be a boolean")
    if family == "cloze":
        if record["fewshot_set"] is not None:
            raise ValueError(f"{source}:{line_no}: cloze fewshot_set must be null")
        demo_ids = record["demo_concept_ids"]
        if not isinstance(demo_ids, list) or len(demo_ids) != 3 or any(not isinstance(item, str) for item in demo_ids):
            raise ValueError(f"{source}:{line_no}: demo_concept_ids must be a list of three strings")
        if record["format"] != "cloze":
            raise ValueError(f"{source}:{line_no}: cloze format must be 'cloze'")
    else:
        if type(record["fewshot_set"]) is not int or record["fewshot_set"] not in range(1, 7):
            raise ValueError(f"{source}:{line_no}: fewshot_set must be an integer from 1 through 6")
        if record["format"] != family:
            raise ValueError(f"{source}:{line_no}: format does not match file family {family}")
    if record["condition"] not in {"diac", "nodiac"}:
        raise ValueError(f"{source}:{line_no}: invalid prompt condition")
    if record["split"] != "test":
        raise ValueError(f"{source}:{line_no}: repetition/translation/cloze split must be 'test'")


def load_prompt_file(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a released prompt JSONL file based on its filename family."""
    prompt_path = Path(path)
    stem = prompt_path.stem
    if stem == "directions_matched":
        family = "directions"
    elif stem.startswith("cloze_"):
        family = "cloze"
    elif stem.startswith("repetition_"):
        family = "repetition"
    elif stem.startswith("translation_"):
        family = "translation"
    else:
        raise ValueError(f"{prompt_path}: cannot infer prompt family from filename")
    records: list[dict[str, Any]] = []
    with prompt_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{prompt_path}:{line_no}: blank JSONL row")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{prompt_path}:{line_no}: malformed JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{prompt_path}:{line_no}: prompt row must be an object")
            _validate_prompt_record(record, family, str(prompt_path), line_no)
            records.append(record)
    return records


def prompt_file_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Return the 27 expected released prompt files keyed by filename stem."""
    directory = Path(cfg["paths"]["prompts_dir"])
    names = [
        f"{family}_{condition}_set{set_no}"
        for family in ("repetition", "translation")
        for condition in ("diac", "nodiac")
        for set_no in range(1, 7)
    ]
    names.extend(("cloze_diac_set1", "cloze_nodiac_set1", "directions_matched"))
    return {name: directory / f"{name}.jsonl" for name in names}
