from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import analyze_r10


def _synthetic_rows(family: str, fmt: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    concepts = {"concept-a": [0, 1], "concept-b": [0]}

    def append(
        concept: str, lang: str, variant: str, role: str, ref: str,
        tok: int, layer: int, value: float,
    ) -> None:
        rows.append({
            "model": f"model-{family}", "model_revision": "tiny", "pass": "pl",
            "format": fmt, "condition": "diac", "fewshot_set": 1,
            "concept_id": concept, "target_lang": lang, "target_variant": variant,
            "role": role, "ref_concept_id": ref, "tok_idx": tok,
            "token_id": tok + 1, "is_content": True, "layer": layer,
            "metric": "lens_logprob", "value": value,
        })

    for c_index, (concept, tokens) in enumerate(concepts.items()):
        for tok in tokens:
            for layer in range(3):
                append(concept, "vi", "released", "correct", concept, tok, layer, -4.0 + layer * 0.1 + c_index * 0.02)
        for l_index, (language, variant) in enumerate(analyze_r10.LANG_VARIANTS.items()):
            distractor = f"d-{concept}-{language}"
            for layer in range(1, 3):
                append(concept, language, variant, "correct", concept, 0, layer, -5.0 + layer * 0.1 + l_index * 0.03)
                append(concept, language, variant, "distractor", distractor, 0, layer, -5.3 + layer * 0.04)
                for tok in tokens:
                    append(concept, language, variant, "path", concept, tok, layer, -4.8 + layer * 0.1 + tok * 0.02)
                    append(concept, language, variant, "path", distractor, tok, layer, -5.2 + layer * 0.03)
    return pd.DataFrame(rows)


def test_r10_end_to_end_position_availability_counts_and_key_determinism(tmp_path: Path) -> None:
    frames = {
        (family, fmt): _synthetic_rows(family, fmt)
        for family in analyze_r10.FAMILIES for fmt in analyze_r10.FORMATS
    }
    cfg = {
        "seed": 20260925,
        "analysis": {"bootstrap_resamples": 20},
        "r10_analysis": {"report": "reports/R10.md", "csv": "reports/R10.csv"},
    }
    loader = lambda family, fmt: frames[(family, fmt)].copy()
    first, csv_path, report_path = analyze_r10.analyze(cfg, repo=tmp_path, input_loader=loader)
    second, _csv_path, _report_path = analyze_r10.analyze(cfg, repo=tmp_path, input_loader=loader)

    assert csv_path.is_file() and report_path.is_file()
    assert len(first) == 6 * len(analyze_r10.LANG_VARIANTS) * 2
    row_t1 = first.loc[
        first["family"].eq("gemma") & first["format"].eq("repetition")
        & first["target_lang"].eq("en") & first["t"].eq(1)
    ].iloc[0]
    row_t2 = first.loc[
        first["family"].eq("gemma") & first["format"].eq("repetition")
        & first["target_lang"].eq("en") & first["t"].eq(2)
    ].iloc[0]
    assert row_t1["concepts_at_t"] == 2
    assert row_t2["concepts_at_t"] == 1
    assert row_t1["bootstrap_key"] == "bootstrap|r10|model-gemma|repetition|en|t1"
    assert first["bootstrap_key"].tolist() == second["bootstrap_key"].tolist()
    pd.testing.assert_frame_equal(first, second)

