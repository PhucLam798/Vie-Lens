"""End-to-end synthetic coverage for the E7 analyzer entry function."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vilens import attribute, env_guard
from vilens.io import unit_paths


def _synthetic_projection_rows(
    *, family: str, model: str, format_name: str, item_set: str,
    n_layers: int, n_heads: int, concepts: list[str],
) -> list[dict[str, Any]]:
    """Build a compact P-A unit in the analyzer's production column schema."""
    rows: list[dict[str, Any]] = []
    for layer in range(1, n_layers + 1):
        layer_type = (
            "local" if family == "gemma" and layer % 2 else
            "global" if family == "gemma" else "standard"
        )
        for concept_index, concept_id in enumerate(concepts):
            for direction_index, direction in enumerate(attribute.DIRECTIONS):
                for kind_index, kind in enumerate(("attn", "mlp")):
                    rows.append({
                        "model": model,
                        "model_revision": "synthetic-revision",
                        "pass": "pa",
                        "format": format_name,
                        "item_set": item_set,
                        "concept_id": concept_id,
                        "layer": layer,
                        "layer_type": layer_type,
                        "kind": kind,
                        "head": -1,
                        "direction": direction,
                        "projection": float((1 if kind_index == 0 else -1) * (concept_index + 1) * (layer + direction_index)),
                        "update_norm": float(layer + kind_index + 1),
                    })
            for head in range(n_heads):
                sign = 1.0 if head < n_heads // 2 else -1.0
                rows.append({
                    "model": model,
                    "model_revision": "synthetic-revision",
                    "pass": "pa",
                    "format": format_name,
                    "item_set": item_set,
                    "concept_id": concept_id,
                    "layer": layer,
                    "layer_type": layer_type,
                    "kind": "head",
                    "head": head,
                    "direction": "d_vi_en",
                    "projection": sign * (head + 1) * (concept_index + 1) * (layer + 1) / n_heads,
                    "update_norm": float(head + 1),
                })
    return rows


def test_analyze_writes_full_synthetic_e7_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    """The real analyze() path handles tiny multi-family P-A units and writes all outputs."""
    # Pytest has already imported the suite's tiny torch models during collection.
    # This analyzer is an offline report function; keep its production guard's
    # environment checks while bypassing only the guard's import-order check.
    monkeypatch.setattr(env_guard, "check_no_heavy_imports", lambda: None)
    from scripts import analyze_e7

    families = ("gemma", "llama")
    formats = ("repetition", "translation")
    dims = {"gemma": (8, 8), "llama": (3, 10)}
    head_counts = {"gemma": 24, "llama": 22}
    model_ids = {family: f"synthetic/{family}" for family in families}
    concepts = ["concept-a", "concept-b", "concept-c"]
    concepts_by_set = {
        "cf729": concepts,
        "main": ["main-a", "main-b", "main-c"],
    }

    monkeypatch.setattr(analyze_e7, "REPO", tmp_path)
    monkeypatch.setattr(analyze_e7, "FAMILIES", families)
    monkeypatch.setattr(analyze_e7, "FORMATS", formats)

    pa_dir = tmp_path / "results" / "pa"
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "derived").mkdir()
    (tmp_path / "results" / "analysis" / "primary" / "family=gemma").mkdir(parents=True)
    (tmp_path / "results" / "analysis" / "primary" / "family=llama").mkdir(parents=True)
    with (tmp_path / "derived" / "cognate_flags.jsonl").open("w", encoding="utf-8") as stream:
        for concept_id in concepts:
            stream.write(json.dumps({"concept_id": concept_id, "cognate_free": True}) + "\n")

    cfg: dict[str, Any] = {
        "seed": 20260925,
        "models": {
            "study_set": "registered",
            "registered": {family: {"hf_id": model_ids[family]} for family in families},
        },
        "reference_facts": {
            "x0c": {"model_dims": {family: [dims[family][0], dims[family][1]] for family in families}}
        },
        "e1": {"expected_early_blocks": {family: [1, 1] for family in families}},
        "e7": {
            "results_dir": "results/pa",
            "analysis_results_dir": "results/analysis/primary",
            "replay": {"p_a": {"source": "pl"}},
            "bootstrap_resamples": 20,
            "bootstrap_key_template": "bootstrap|e7|{model}|{format}",
            "head_rank_n": 10,
            "share_epsilon": 1e-8,
            "report": "reports/E7.md",
            "outputs": {
                "layers": "reports/E7_layers.csv",
                "band": "reports/E7_band.csv",
                "heads": "reports/E7_heads.csv",
                "heads_all": "reports/E7_heads_all.csv",
            },
        },
    }

    family_summary: dict[str, Any] = {}
    for family in families:
        family_summary[family] = {fmt: {"replay_batch_count": 1} for fmt in formats}
    pa_dir.mkdir(parents=True)
    (pa_dir / "summary.json").write_text(
        json.dumps({"families": family_summary}), encoding="utf-8"
    )

    unit_keys = [
        {
            "family": family,
            "pass": "pa",
            "format": fmt,
            "item_set": item_set,
            "chunk": 0,
            "replay_source": "pl",
        }
        for family in families
        for fmt in formats
        for item_set in ("main", "cf729")
    ]
    for unit_key in unit_keys:
        family = str(unit_key["family"])
        fmt = str(unit_key["format"])
        item_set = str(unit_key["item_set"])
        rows = _synthetic_projection_rows(
            family=family,
            model=model_ids[family],
            format_name=fmt,
            item_set=item_set,
            n_layers=dims[family][0],
            n_heads=head_counts[family],
            concepts=concepts_by_set[item_set],
        )
        path, _sidecar = unit_paths(pa_dir, unit_key)
        pd.DataFrame(rows, columns=analyze_e7.PA_COLUMNS).to_parquet(path, index=False)

    landmark_rows: list[dict[str, Any]] = []
    for family in families:
        for fmt in formats:
            common = {
                "model": model_ids[family],
                "format": fmt,
                "pass": "analysis",
                "condition": "diac",
                "fewshot_set": 1,
                "role": "path",
                "item_set": "cf729",
                "metric_variant": "s",
            }
            landmark_rows.extend([
                {**common, "metric": "L_peak", "value": float(dims[family][0] - 1)},
                {
                    **common,
                    "metric": "L_cross",
                    "value": float(dims[family][0]) if family == "gemma" and fmt == "repetition" else np.nan,
                },
            ])
    pd.DataFrame(landmark_rows).to_csv(tmp_path / "reports" / "E2-E5_landmarks.csv", index=False)

    gate_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for family in families:
        n_layers = dims[family][0]
        for fmt in formats:
            for layer in range(1, n_layers + 1):
                gate_rows.append({
                    "family": family, "fit_format": fmt, "layer": layer, "gate_pass": True,
                })
            for concept_id in concepts:
                for layer in range(n_layers + 1):
                    delta_rows.append({
                        "pass": "pl", "format": fmt, "condition": "diac", "fewshot_set": 1,
                        "concept_id": concept_id, "target_lang": "vi", "target_variant": "released",
                        "role": "path", "layer": layer, "metric": "delta", "value": float(layer),
                    })
        delta_path = tmp_path / "results" / "analysis" / "primary" / f"family={family}" / "deltas.parquet"
        pd.DataFrame([row for row in delta_rows if row["format"] in formats]).to_parquet(delta_path, index=False)
        delta_rows.clear()
    pd.DataFrame(gate_rows).to_csv(tmp_path / "reports" / "E5_gate.csv", index=False)

    analyze_e7.analyze(cfg)

    assert (tmp_path / "reports" / "E7.md").is_file()
    report_text = (tmp_path / "reports" / "E7.md").read_text(encoding="utf-8")
    assert "exploratory" in report_text
    assert "v2: head ranking corrected to (layer, head) per OPS-37-7" in report_text
    layers = pd.read_csv(tmp_path / "reports" / "E7_layers.csv")
    bands = pd.read_csv(tmp_path / "reports" / "E7_band.csv")
    heads = pd.read_csv(tmp_path / "reports" / "E7_heads.csv")
    heads_all = pd.read_csv(tmp_path / "reports" / "E7_heads_all.csv")

    target = bands.loc[bands["item_set"].eq("cf729")]
    expected = {
        (family, fmt, label, component, direction)
        for family in families
        for fmt in formats
        for label in ("primary", "slope_sensitivity", "ops37_as_written")
        for component in ("attn", "mlp", "share")
        for direction in attribute.DIRECTIONS
    }
    actual = set(zip(
        target["model"].map(lambda model: next(f for f in families if model_ids[f] == model)),
        target["format"], target["band_label"], target["component"], target["direction"],
    ))
    assert actual == expected
    shares = target.loc[target["component"].eq("share"), "estimate"].to_numpy(dtype=float)
    assert all(np.isnan(value) or 0.0 <= value <= 1.0 for value in shares)

    for table in (layers, bands, heads, heads_all):
        low = pd.to_numeric(table["ci_low"], errors="coerce")
        high = pd.to_numeric(table["ci_high"], errors="coerce")
        assert ((low <= high) | (low.isna() & high.isna())).all()

    primary_heads = heads.loc[heads["item_set"].eq("cf729")]
    for family in families:
        for fmt in formats:
            family_heads = primary_heads.loc[
                primary_heads["model"].eq(model_ids[family])
                & primary_heads["format"].eq(fmt)
            ]
            expected_types = {"local", "global"} if family == "gemma" else {"standard"}
            assert set(family_heads["layer_type"]) == expected_types
            assert {"layer", "head"}.issubset(family_heads.columns)
            if family == "llama":
                assert set(family_heads["layer_type"]) == {"standard"}
            for _layer_type, typed in family_heads.groupby("layer_type"):
                assert typed["top_positive_rank"].notna().sum() == 10
                assert typed["top_negative_rank"].notna().sum() == 10

            all_cell_heads = heads_all.loc[
                heads_all["item_set"].eq("cf729")
                & heads_all["model"].eq(model_ids[family])
                & heads_all["format"].eq(fmt)
            ]
            assert len(all_cell_heads) == dims[family][0] * head_counts[family]
            repeated_index = all_cell_heads.loc[all_cell_heads["head"].eq(0)]
            assert repeated_index["layer"].nunique() == dims[family][0]
            assert repeated_index["mean_projection"].nunique() == dims[family][0]
    assert "layer.head" in report_text
