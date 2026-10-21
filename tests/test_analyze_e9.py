from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import analyze_e9
from vilens.config import load_config


def _synthetic_rows(condition: str, concept_ids: list[str]) -> pd.DataFrame:
    rows = []
    languages = ("vi", "en", "zh", "fr", "id")
    for concept_index, concept_id in enumerate(concept_ids):
        for language_index, language in enumerate(languages):
            variant = "nodiac" if language == "vi" and condition == "nodiac" else (
                "nospace" if language == "zh" else "released"
            )
            for role, ref_id, base in (
                ("correct", concept_id, -5.0 + language_index * 0.1),
                ("distractor", f"d-{concept_id}", -5.5 + language_index * 0.1),
            ):
                for token_index in (0, 1):
                    for layer in range(3):
                        rows.append({
                            "model": "gemma",
                            "model_revision": "synthetic",
                            "pass": "pl",
                            "format": "repetition",
                            "condition": condition,
                            "fewshot_set": 1,
                            "concept_id": concept_id,
                            "target_lang": language,
                            "target_variant": variant,
                            "role": role,
                            "ref_concept_id": ref_id,
                            "tok_idx": token_index,
                            "token_id": token_index + 1,
                            "is_content": True,
                            "layer": layer,
                            "metric": "lens_logprob",
                            "value": base + 0.04 * layer + 0.001 * concept_index,
                        })
    return pd.DataFrame.from_records(rows)


def _la_tables(condition: str, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for language_index, language in enumerate(("en", "zh", "fr", "id")):
        summary_rows.append({
            "source": "primary", "model": "gemma", "format": "repetition",
            "metric": "max_LA", "target_lang": language, "layer": 2,
            "value": 0.55 - language_index * 0.04 + (0.01 if condition == "nodiac" else 0.0),
            "ci_low": 0.5, "ci_high": 0.6, "n_concepts": len(ids),
        })
        for concept_index, concept_id in enumerate(ids):
            for layer in (1, 2):
                long_rows.append({
                    "model": "gemma", "format": "repetition", "concept_id": concept_id,
                    "target_lang": language, "metric": "LA", "layer": layer,
                    "value": 0.2 + 0.03 * layer + 0.001 * concept_index,
                })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def _generations(condition: str, ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "model": "gemma", "format": "repetition", "condition": condition,
            "fewshot_set": 1, "concept_id": concept_id, "answer": f"viet-{concept_id}",
        }
        for concept_id in ids
    ]


def test_e9_analyze_end_to_end_paired_scopes_and_deterministic_bootstrap(tmp_path: Path) -> None:
    cfg = load_config(analyze_e9.REPO / "configs/experiments.yaml")
    cfg["analysis"]["bootstrap_resamples"] = 20
    cfg["analysis"]["window"] = [0.25, 0.75]
    cfg["e1"]["early_depth_max"] = 0.5
    cfg["e9_analysis"] = {"report": "reports/E9.md", "csv": "reports/E9.csv"}
    (tmp_path / "reports").mkdir()
    all_ids = ["c1", "c2", "c3"]
    cell = {
        "paired_ids": set(all_ids),
        "cf_ids": {"c1", "c2"},
        "diac_raw": _synthetic_rows("diac", all_ids),
        "nodiac_raw": _synthetic_rows("nodiac", all_ids),
        "diac_la_long": _la_tables("diac", all_ids)[0],
        "diac_la_summary": _la_tables("diac", all_ids)[1],
        "nodiac_la_long": _la_tables("nodiac", all_ids)[0],
        "nodiac_la_summary": _la_tables("nodiac", all_ids)[1],
        "diac_generations": _generations("diac", all_ids),
        "nodiac_generations": _generations("nodiac", all_ids),
        "concepts": {
            concept_id: {
                "concept_id": concept_id, "vi": f"viet-{concept_id}",
                "en": f"english-{concept_id}", "zh": "漢", "fr": "français",
                "id": "indonesia", "ru": None,
            }
            for concept_id in all_ids
        },
    }
    result = analyze_e9.analyze(
        cfg, repo=tmp_path, synthetic_inputs={("gemma", "repetition"): cell}
    )
    assert (tmp_path / "reports/E9.md").is_file()
    assert (tmp_path / "reports/E9.csv").is_file()
    assert set(result["scope"]) == {"paired_main", "paired_cf729"}
    assert set(result["condition"]) >= {"diac", "nodiac", "nodiac_minus_diac"}
    differences = result.loc[result["condition"].eq("nodiac_minus_diac")]
    assert {"Q_H1", "H1_ratio", "H2_band_ratio", "max_LA_en"}.issubset(set(differences["quantity"]))
    assert np.isfinite(differences.loc[differences["quantity"].isin({"Q_H1", "max_LA_en"}), "value"].to_numpy(dtype=float)).all()
    for row in differences.itertuples(index=False):
        if pd.notna(row.ci_low) and pd.notna(row.ci_high):
            assert row.ci_low <= row.ci_high
    assert sorted(result.loc[result["quantity"].eq("A10_vi_rate"), "a10_denominator"].tolist()) == [2, 2, 3, 3]


def test_e9_bootstrap_key_is_stable_and_scope_specific() -> None:
    assert analyze_e9._paired_boot_key("M", "repetition", "paired_main") == "bootstrap|e9|M|repetition"
    assert analyze_e9._paired_boot_key("M", "repetition", "paired_cf729") == "bootstrap|e9|M|repetition|cf729"
    from vilens.analysis.bootstrap import concept_indices

    first = concept_indices(["c1", "c2", "c3"], resamples=20, seed=20260925, key="bootstrap|e9|M|repetition")
    second = concept_indices(["c1", "c2", "c3"], resamples=20, seed=20260925, key="bootstrap|e9|M|repetition")
    assert np.array_equal(first, second)


def test_e9_unit_key_uses_none_for_cloze() -> None:
    assert analyze_e9._unit_key("gemma", "cloze", 0)["fewshot_set"] == "none"
    assert analyze_e9._unit_key("gemma", "repetition", 0)["fewshot_set"] == "1"


def test_e9_analysis_wrapper_uses_two_thread_caps() -> None:
    from job_analyze_e9 import THREAD_CAPS

    assert THREAD_CAPS == {
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2", "ARROW_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
