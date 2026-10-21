#!/datastore/luannt/.cache/sb/bin/python
"""Summarize the descriptive R10 along-path readout from stored P-L rows."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vilens import env_guard

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vilens.analysis.bootstrap import concept_indices, percentile_interval
from vilens.config import load_config

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
LONG_COLUMNS = [
    "model", "model_revision", "pass", "format", "condition", "fewshot_set",
    "concept_id", "target_lang", "target_variant", "role", "ref_concept_id",
    "tok_idx", "token_id", "is_content", "layer", "metric", "value",
]
LANG_VARIANTS = {"en": "released", "zh": "nospace", "fr": "released", "id": "released"}


def _primary_cf_ids(repo: Path, cfg: Mapping[str, Any]) -> set[str]:
    """Resolve the registered cf729 set within the non-extension main scope."""
    from scripts import analyze_core as core

    main_ids, cf_ids = core._primary_scope_ids(repo, cfg)
    return main_ids & cf_ids


def _load_cell_rows(
    cfg: Mapping[str, Any], *, repo: Path, family: str, fmt: str, concept_ids: set[str]
) -> pd.DataFrame:
    """Read one family-format's selected primary P-L rows in bounded batches."""
    from scripts import analyze_core as core

    selected_paths = [
        path for path in core._primary_paths(repo, cfg, family=family)
        if f"format={fmt}__" in path.name and "fewshot_set=1" in path.name
    ]
    if not selected_paths:
        raise FileNotFoundError(f"No primary P-L units for {family}/{fmt}")
    frames: list[pd.DataFrame] = []
    for path in selected_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=LONG_COLUMNS, batch_size=65_536):
            frame = batch.to_pandas()
            keep = (
                frame["concept_id"].astype(str).isin(concept_ids)
                & frame["format"].astype(str).eq(fmt)
                & frame["condition"].astype(str).eq("diac")
                & frame["fewshot_set"].astype(str).eq("1")
            )
            if keep.any():
                frames.append(frame.loc[keep].copy())
    if not frames:
        raise FileNotFoundError(f"No cf729 primary rows for {family}/{fmt}")
    return pd.concat(frames, ignore_index=True)


def _content_position_map(raw: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Map raw Vietnamese target token indices to 1-based content positions."""
    vi = raw.loc[
        raw["role"].astype(str).eq("correct")
        & raw["target_lang"].astype(str).eq("vi")
        & raw["target_variant"].astype(str).eq("released")
        & raw["metric"].astype(str).eq("lens_logprob")
        & pd.to_numeric(raw["layer"], errors="coerce").eq(0)
        & raw["is_content"].astype(bool),
        ["concept_id", "tok_idx"],
    ].drop_duplicates()
    position_map: dict[tuple[str, int], int] = {}
    for concept_id, group in vi.groupby(vi["concept_id"].astype(str), sort=True):
        token_indices = sorted(pd.to_numeric(group["tok_idx"], errors="raise").astype(int).unique())
        if not token_indices:
            raise AssertionError(f"No Vietnamese content positions for {concept_id}")
        for position, token_index in enumerate(token_indices, start=1):
            position_map[(str(concept_id), int(token_index))] = position
    return position_map


def _path_delta_rows(
    raw: pd.DataFrame,
    *,
    language: str,
    variant: str,
    position_map: Mapping[tuple[str, int], int],
    n_layers: int,
) -> pd.DataFrame:
    """Pair each along-path correct readout with its stored A1 distractor."""
    path = raw.loc[
        raw["role"].astype(str).eq("path")
        & raw["target_lang"].astype(str).eq(language)
        & raw["target_variant"].astype(str).eq(variant)
        & raw["metric"].astype(str).eq("lens_logprob")
        & pd.to_numeric(raw["layer"], errors="coerce").between(1, n_layers),
        ["concept_id", "ref_concept_id", "tok_idx", "layer", "value"],
    ].copy()
    path["concept_id"] = path["concept_id"].astype(str)
    path["ref_concept_id"] = path["ref_concept_id"].astype(str)
    path["tok_idx"] = pd.to_numeric(path["tok_idx"], errors="raise").astype(int)
    path["layer"] = pd.to_numeric(path["layer"], errors="raise").astype(int)
    path["t"] = [position_map.get((cid, tok), np.nan) for cid, tok in zip(path["concept_id"], path["tok_idx"], strict=True)]
    path = path.loc[path["t"].notna()].copy()
    path["t"] = path["t"].astype(int)
    correct = path.loc[path["ref_concept_id"].eq(path["concept_id"]), ["concept_id", "t", "layer", "value"]]
    distractor = path.loc[~path["ref_concept_id"].eq(path["concept_id"]), ["concept_id", "t", "layer", "value"]]
    correct = correct.rename(columns={"value": "correct_value"})
    distractor = distractor.rename(columns={"value": "distractor_value"})
    paired = correct.merge(
        distractor, on=["concept_id", "t", "layer"], how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError(f"Incomplete path correct/distractor pairs for {language}/{variant}")
    paired["value"] = (
        pd.to_numeric(paired["correct_value"], errors="raise")
        - pd.to_numeric(paired["distractor_value"], errors="raise")
    )
    expected = paired.groupby(["concept_id", "t"], sort=True)["layer"].nunique()
    if not expected.eq(n_layers).all():
        raise AssertionError(f"Path layer coverage differs for {language}/{variant}")
    return paired[["concept_id", "t", "layer", "value"]].sort_values(
        ["t", "concept_id", "layer"], kind="stable"
    ).reset_index(drop=True)


def _answer_delta_rows(raw: pd.DataFrame, *, language: str, variant: str, n_layers: int) -> pd.DataFrame:
    """Pair first-content-token answer-position log-probs for the same P-L rows."""
    readout = raw.loc[
        raw["role"].astype(str).isin({"correct", "distractor"})
        & raw["target_lang"].astype(str).eq(language)
        & raw["target_variant"].astype(str).eq(variant)
        & raw["metric"].astype(str).eq("lens_logprob")
        & raw["is_content"].astype(bool)
        & pd.to_numeric(raw["layer"], errors="coerce").between(1, n_layers),
        ["concept_id", "role", "tok_idx", "layer", "value"],
    ].copy()
    readout["concept_id"] = readout["concept_id"].astype(str)
    readout["tok_idx"] = pd.to_numeric(readout["tok_idx"], errors="raise").astype(int)
    readout["layer"] = pd.to_numeric(readout["layer"], errors="raise").astype(int)
    readout = readout.sort_values(["concept_id", "role", "layer", "tok_idx"], kind="stable")
    first = readout.drop_duplicates(["concept_id", "role", "layer"], keep="first")
    correct = first.loc[first["role"].astype(str).eq("correct"), ["concept_id", "layer", "value"]]
    distractor = first.loc[first["role"].astype(str).eq("distractor"), ["concept_id", "layer", "value"]]
    correct = correct.rename(columns={"value": "correct_value"})
    distractor = distractor.rename(columns={"value": "distractor_value"})
    paired = correct.merge(
        distractor, on=["concept_id", "layer"], how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise AssertionError(f"Incomplete answer-position correct/distractor pairs for {language}/{variant}")
    paired["value"] = (
        pd.to_numeric(paired["correct_value"], errors="raise")
        - pd.to_numeric(paired["distractor_value"], errors="raise")
    )
    if not paired.groupby("concept_id", sort=True)["layer"].nunique().eq(n_layers).all():
        raise AssertionError(f"Answer-position layer coverage differs for {language}/{variant}")
    return paired[["concept_id", "layer", "value"]].sort_values(
        ["concept_id", "layer"], kind="stable"
    ).reset_index(drop=True)


def validate_cell(raw: pd.DataFrame, *, expected_ids: set[str] | None = None) -> tuple[int, dict[int, int]]:
    """Validate one cf729 cell and return its depth and concept count per t."""
    if raw.empty:
        raise AssertionError("R10 cell has no stored P-L rows")
    ids = set(raw["concept_id"].astype(str).unique())
    if expected_ids is not None and ids != expected_ids:
        raise AssertionError(
            f"R10 cf729 ids differ: observed={len(ids)}, expected={len(expected_ids)}"
        )
    vi_embedding = raw.loc[
        raw["role"].astype(str).eq("correct")
        & raw["target_lang"].astype(str).eq("vi")
        & raw["target_variant"].astype(str).eq("released")
        & raw["metric"].astype(str).eq("lens_logprob")
        & pd.to_numeric(raw["layer"], errors="coerce").eq(0)
    ]
    layers = sorted(pd.to_numeric(raw["layer"], errors="raise").astype(int).unique())
    if len(layers) < 2 or layers[0] != 0 or layers != list(range(layers[-1] + 1)):
        raise AssertionError("R10 layer axis must contain consecutive residual layers 0..N")
    n_layers = layers[-1]
    if vi_embedding.empty:
        raise AssertionError("R10 lacks Vietnamese correct-token rows at residual layer 0")
    position_map = _content_position_map(raw)
    expected_positions = {
        (cid, tok)
        for cid, group in vi_embedding.loc[vi_embedding["is_content"].astype(bool)].groupby(
            vi_embedding.loc[vi_embedding["is_content"].astype(bool), "concept_id"].astype(str), sort=True
        )
        for tok in pd.to_numeric(group["tok_idx"], errors="raise").astype(int).unique()
    }
    if set(position_map) != expected_positions:
        raise AssertionError("R10 Vietnamese content-position map is incomplete")
    for language, variant in LANG_VARIANTS.items():
        path_rows = _path_delta_rows(
            raw, language=language, variant=variant, position_map=position_map, n_layers=n_layers
        )
        ans_rows = _answer_delta_rows(raw, language=language, variant=variant, n_layers=n_layers)
        path_ids = set(path_rows["concept_id"].astype(str).unique())
        ans_ids = set(ans_rows["concept_id"].astype(str).unique())
        if path_ids != ids or ans_ids != ids:
            raise AssertionError(
                f"R10 {language}/{variant} concept coverage differs: path={len(path_ids)}, answer={len(ans_ids)}, expected={len(ids)}"
            )
        expected_pairs = {
            (cid, position) for (cid, _token_index), position in position_map.items()
        }
        observed_pairs = set(zip(path_rows["concept_id"].astype(str), path_rows["t"].astype(int), strict=True))
        if observed_pairs != expected_pairs:
            raise AssertionError(f"R10 {language}/{variant} does not cover every Vietnamese content position")
    counts: dict[int, int] = {}
    for t in position_map.values():
        counts[t] = counts.get(t, 0) + 1
    return n_layers, dict(sorted(counts.items()))


def preflight(cfg: Mapping[str, Any], *, repo: Path = REPO) -> tuple[str, bool]:
    """Check P-L path-pair coverage and report concept counts by t."""
    cf_ids = _primary_cf_ids(repo, cfg)
    output: list[str] = ["FAMILY | FORMAT | CONCEPTS | t | CONCEPTS_AT_t | STATUS"]
    output.append("---|---|---:|---:|---:|---")
    okay = True
    for family in FAMILIES:
        for fmt in FORMATS:
            expected_count = 729 if fmt == "repetition" else 721
            expected_ids = cf_ids if fmt == "repetition" else set()
            try:
                raw = _load_cell_rows(cfg, repo=repo, family=family, fmt=fmt, concept_ids=cf_ids)
                observed_ids = set(raw["concept_id"].astype(str).unique())
                if len(observed_ids) != expected_count:
                    raise AssertionError(f"cf729 row IDs={len(observed_ids)}, expected={expected_count}")
                if fmt == "repetition" and observed_ids != expected_ids:
                    raise AssertionError("repetition IDs do not equal cf729")
                n_layers, counts = validate_cell(raw, expected_ids=observed_ids)
                for t, count in counts.items():
                    output.append(f"{family} | {fmt} | {len(observed_ids)} | {t} | {count} | PASS (N={n_layers})")
            except (AssertionError, FileNotFoundError, KeyError, ValueError, TypeError) as exc:
                okay = False
                output.append(f"{family} | {fmt} | ? | ? | ? | FAIL: {type(exc).__name__}: {exc}")
    return "\n".join(output), okay


def _cell_summary(
    raw: pd.DataFrame,
    *,
    family: str,
    fmt: str,
    seed: int,
    resamples: int,
) -> pd.DataFrame:
    """Calculate the along-path and answer-position maxima for one cell."""
    n_layers, counts = validate_cell(raw)
    position_map = _content_position_map(raw)
    model_values = sorted(raw["model"].astype(str).unique())
    if len(model_values) != 1:
        raise AssertionError(f"R10 {family}/{fmt} has non-unique model IDs")
    model_id = model_values[0]
    rows: list[dict[str, Any]] = []
    layers = list(range(1, n_layers + 1))
    for language, variant in LANG_VARIANTS.items():
        path = _path_delta_rows(
            raw, language=language, variant=variant, position_map=position_map, n_layers=n_layers
        )
        answer = _answer_delta_rows(raw, language=language, variant=variant, n_layers=n_layers)
        for t, n_concepts in counts.items():
            path_t = path.loc[path["t"].astype(int).eq(t)]
            ids = sorted(path_t["concept_id"].astype(str).unique())
            if len(ids) != n_concepts:
                raise AssertionError(f"R10 {family}/{fmt}/{language}/t{t} concept count mismatch")
            path_matrix = path_t.pivot(index="concept_id", columns="layer", values="value").reindex(
                index=ids, columns=layers
            ).to_numpy(dtype=np.float64)
            answer_matrix = answer.pivot(index="concept_id", columns="layer", values="value").reindex(
                index=ids, columns=layers
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(path_matrix).all() or not np.isfinite(answer_matrix).all():
                raise AssertionError(f"R10 matrix has missing values for {family}/{fmt}/{language}/t{t}")
            key = f"bootstrap|r10|{model_id}|{fmt}|{language}|t{t}"
            indices = concept_indices(ids, resamples=resamples, seed=seed, key=key)
            path_boot = path_matrix[indices].mean(axis=1)
            answer_boot = answer_matrix[indices].mean(axis=1)
            path_estimates = path_matrix.mean(axis=0)
            answer_estimates = answer_matrix.mean(axis=0)
            path_layer_index = int(np.argmax(path_estimates))
            answer_layer_index = int(np.argmax(answer_estimates))
            path_max_samples = path_boot.max(axis=1)
            answer_max_samples = answer_boot.max(axis=1)
            path_low, path_high = percentile_interval(path_max_samples)
            answer_low, answer_high = percentile_interval(answer_max_samples)
            rows.append({
                "model": model_id,
                "family": family,
                "format": fmt,
                "target_lang": language,
                "target_variant": variant,
                "t": int(t),
                "concepts_at_t": int(n_concepts),
                "quantity": "max_concept_mean_delta",
                "along_path_value": float(path_estimates[path_layer_index]),
                "along_path_layer": int(layers[path_layer_index]),
                "along_path_ci_low": path_low,
                "along_path_ci_high": path_high,
                "answer_position_value": float(answer_estimates[answer_layer_index]),
                "answer_position_layer": int(layers[answer_layer_index]),
                "answer_position_ci_low": answer_low,
                "answer_position_ci_high": answer_high,
                "bootstrap_resamples": int(resamples),
                "bootstrap_key": key,
            })
    return pd.DataFrame.from_records(rows)


def analyze(
    cfg: Mapping[str, Any],
    *,
    repo: Path = REPO,
    input_loader: Callable[[str, str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Compute the descriptive R10 output, optionally using synthetic cell rows."""
    cf_ids = _primary_cf_ids(repo, cfg) if input_loader is None else set()
    loader = input_loader or (
        lambda family, fmt: _load_cell_rows(cfg, repo=repo, family=family, fmt=fmt, concept_ids=cf_ids)
    )
    summary = pd.concat(
        [
            _cell_summary(
                loader(family, fmt), family=family, fmt=fmt,
                seed=int(cfg["seed"]), resamples=int(cfg["analysis"]["bootstrap_resamples"]),
            )
            for family in FAMILIES for fmt in FORMATS
        ],
        ignore_index=True,
    )
    summary = summary.sort_values(["family", "format", "target_lang", "t"], kind="stable").reset_index(drop=True)
    output_cfg = cfg["r10_analysis"]
    csv_path = Path(str(output_cfg["csv"]))
    report_path = Path(str(output_cfg["report"]))
    if not csv_path.is_absolute():
        csv_path = repo / csv_path
    if not report_path.is_absolute():
        report_path = repo / report_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    report_lines = [
        "# R10 — along-path readout",
        "",
        "Descriptive analysis under OPS-46; readout is position-specific and is not a decision-rule input.",
        "For each cell, language and Vietnamese content position, the table reports the maximum over layers of the concept-mean along-path delta and the matching answer-position maximum on the same concepts.",
        "",
        f"Bootstrap: B={int(cfg['analysis']['bootstrap_resamples'])}, 95% percentile CI over concepts.",
        "",
        "Full numeric output: [R10.csv](R10.csv).",
        "",
        f"Rows: {len(summary)}; cells: {summary[['family', 'format']].drop_duplicates().shape[0]}.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary, csv_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Load the declared config and run the R10 analysis."""
    env_guard.guard(offline=True)
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.preflight_only:
        table, okay = preflight(cfg, repo=REPO)
        print(table)
        return 0 if okay else 2
    result, csv_path, report_path = analyze(cfg, repo=REPO)
    print(f"R10 rows={len(result)} csv={csv_path} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
