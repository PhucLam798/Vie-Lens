"""Tiny end-to-end coverage for OPS-44 R-battery E4 sensitivities."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_rows(model: str, fmt: str, ids: list[str]) -> pd.DataFrame:
    rows = []
    slopes = {"vi": 0.45, "en": 0.35, "zh": 0.29, "fr": 0.22, "id": 0.20}
    variants = {"vi": "released", "en": "released", "zh": "nospace", "fr": "released", "id": "released"}
    for concept_index, concept_id in enumerate(ids):
        for language, slope in slopes.items():
            for role, ref_id, offset in (("correct", concept_id, 0.0), ("distractor", f"d-{concept_id}", -0.03)):
                for layer in range(11):
                    for token in range(2):
                        rows.append({
                            "model": model, "model_revision": "tiny", "pass": "pl", "format": fmt,
                            "condition": "diac", "fewshot_set": 1, "concept_id": concept_id,
                            "target_lang": language, "target_variant": variants[language], "role": role,
                            "ref_concept_id": ref_id, "tok_idx": token, "token_id": token,
                            "is_content": True, "layer": layer, "metric": "lens_logprob",
                            "value": -5.0 + offset + slope * layer + concept_index * 0.001 + token * 0.0001,
                        })
    return pd.DataFrame(rows)


def _fake_la(model: str, fmt: str) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {
            "model": model, "format": fmt, "metric": "max_LA", "target_lang": language,
            "layer": 3, "value": 0.1 if language != "en" else 0.6,
            "ci_low": 0.08, "ci_high": 0.62, "la_evaluable": True, "n_concepts": 4,
        }
        for language in ("en", "zh", "fr", "id")
    ])


def _metadata(ids: list[str]) -> dict[str, dict[str, str]]:
    values = {
        "c0": (True, True, True, 0.8),
        "c1": (True, False, False, 0.6),
        "c2": (False, True, True, 0.59),
        "c3": (True, True, False, 0.61),
        "c4": (False, False, True, 0.7),
        "c5": (False, True, False, 0.2),
        "c6": (False, False, False, 0.1),
        "c7": (False, False, False, 0.1),
    }
    output = {}
    for concept_id in ids:
        bt, external, nllb, distance = values[concept_id]
        output[concept_id] = {
            "bt_pass_strict_v1": str(bt).lower(), "external_attested": str(external).lower(),
            "zh_nllb_agree": str(nllb).lower(), "fr_nllb_agree": str(nllb).lower(),
            "id_nllb_agree": str(nllb).lower(), "min_surface_dist": str(distance),
        }
    return output


def test_r_battery_variant_filters_have_expected_item_counts() -> None:
    from scripts.analyze_rbattery import variant_ids

    ids = [f"c{i}" for i in range(8)]
    main_ids = set(ids)
    cf_ids = set(ids[:6])
    meta = _metadata(ids)
    assert len(variant_ids({"scope": "cf729", "filter": "none"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 6
    assert len(variant_ids({"scope": "cf729", "filter": "bt_pass_strict_v1"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 3
    assert len(variant_ids({"scope": "cf729", "filter": "external_attested"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 4
    assert len(variant_ids({"scope": "cf729", "filter": "all_three_nllb"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 3
    assert len(variant_ids({"scope": "cf729", "filter": "min_surface_dist_ge_060"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 4
    assert len(variant_ids({"scope": "cf729", "filter": "competent_vi"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta, competent_ids={"c0", "c1", "c7"})) == 2
    assert len(variant_ids({"scope": "main", "filter": "none"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta)) == 8
    assert len(variant_ids({"scope": "cf729", "filter": "exclude_r11_ids"}, main_ids=main_ids, cf_ids=cf_ids, metadata=meta, r11_ids={"c0", "c1"})) == 4


def test_r_battery_bootstrap_key_is_variant_deterministic() -> None:
    from scripts.analyze_rbattery import _variant_bootstrap_config

    variant = {"name": "R6_bt_pass_strict_v1", "code": "R6"}
    cfg = {"analysis": {"bootstrap_resamples": 20, "bootstrap_key_template": "frozen"}}
    first = _variant_bootstrap_config(cfg, variant)["bootstrap_key_template"]
    second = _variant_bootstrap_config(cfg, variant)["bootstrap_key_template"]
    assert first == second == "bootstrap|r6|{model}|{format}|bt_pass_strict_v1"


def test_r_battery_analyze_end_to_end_tiny_inputs(tmp_path: Path, monkeypatch) -> None:
    from scripts import analyze_core as core
    from scripts import analyze_rbattery as rb
    from vilens.config import load_config
    import vilens.analysis.competence as competence

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/experiments.yaml")
    cfg["analysis"] = {**cfg["analysis"], "bootstrap_resamples": 20}
    ids = [f"c{i}" for i in range(8)]
    main_ids, cf_ids = set(ids), set(ids[:6])
    metadata = _metadata(ids)
    repo = tmp_path
    (repo / "reports").mkdir()
    (repo / "data/external/viLens-concepts").mkdir(parents=True)
    (repo / "results/pl").mkdir(parents=True)
    fake_path = repo / "results/pl/fake.parquet"
    fake_path.touch()
    for suffix in (".provenance.json", ".resid.npy", ".generations.jsonl"):
        fake_path.with_suffix(suffix).touch()
    monkeypatch.setattr(core, "_primary_scope_ids", lambda *_args: (main_ids, cf_ids))
    monkeypatch.setattr(core, "_primary_paths", lambda *_args, **_kwargs: [fake_path])
    monkeypatch.setattr(core, "_primary_prompt_records", lambda *_args: [
        {"concept_id": concept_id, "format": fmt, "split": "test", "m1_extension": False}
        for fmt in rb.FORMATS for concept_id in ids
    ])
    monkeypatch.setattr(core, "_load_primary_generations", lambda _repo, _cfg, _ids, family=None: [
        {"model": family, "format": fmt, "condition": "diac", "concept_id": concept_id, "answer": ""}
        for fmt in rb.FORMATS for concept_id in ids
    ])
    monkeypatch.setattr(rb, "_load_metadata", lambda _repo, _ids: metadata)
    monkeypatch.setattr(rb, "_load_primary_long", lambda _cfg, _repo, family: pd.concat([
        _synthetic_rows(f"model-{family}", fmt, ids) for fmt in rb.FORMATS
    ], ignore_index=True))
    monkeypatch.setattr(competence, "competence_rows", lambda generations, _concepts: pd.DataFrame.from_records([
        {"model": row["model"], "format": row["format"], "concept_id": row["concept_id"], "competent": row["concept_id"] in {"c0", "c1"}}
        for row in generations
    ]))

    def fake_la(source, *, config, repo, main_ids, family_filter=None, formats_override=None,
                query_scope_ids=None, expected_query_counts=None, bootstrap_key_template=None, **_kwargs):
        assert source == "primary"
        assert bootstrap_key_template and "bootstrap|" in bootstrap_key_template
        fmt = tuple(formats_override)[0]
        model = f"model-{family_filter}"
        return pd.DataFrame(), _fake_la(model, fmt)

    monkeypatch.setattr(core, "_compute_la_source", fake_la)

    base_rule_parts, base_landmark_parts = [], []
    for family in rb.FAMILIES:
        for fmt in rb.FORMATS:
            raw = _synthetic_rows(f"model-{family}", fmt, ids[:6])
            scores = core.content_token_scores(raw)
            deltas = core.matched_distractor_deltas(scores)
            readouts = core._readout_rows(deltas)
            score_readouts = core._readout_rows(scores.loc[scores["role"].astype(str).eq("correct")])
            rises_s = core.concept_rise_curves(readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="delta_s")
            rises_s1 = core.concept_rise_curves(readouts, early_depth_max=float(cfg["e1"]["early_depth_max"]), value_column="delta_s1")
            for metric_variant, rises, score_column in (("s", rises_s, "s"), ("s1", rises_s1, "s1")):
                rules, landmarks, _ = core._bootstrap_core(
                    per_concept_rise=rises, per_concept_scores=score_readouts,
                    source="primary", item_set="cf729", seed=cfg["seed"],
                    analysis_config=cfg["analysis"], score_column=score_column,
                    metric_variant=metric_variant,
                )
                base_rule_parts.append(core._summary_metric_long(rules, source_rows=score_readouts, kind="rule"))
                base_landmark_parts.append(core._summary_metric_long(landmarks, source_rows=score_readouts, kind="landmark"))
    pd.concat(base_rule_parts, ignore_index=True).to_csv(repo / "reports/E2-E5_rules.csv", index=False)
    pd.concat(base_landmark_parts, ignore_index=True).to_csv(repo / "reports/E2-E5_landmarks.csv", index=False)

    summary, _matches = rb.analyze(cfg, repo=repo)
    assert (repo / "reports/R_battery_e4.csv").is_file()
    assert (repo / "reports/R_battery_e4.md").is_file()
    assert len(summary) == len(cfg["r_battery_e4"]["variants"]) * len(rb.FAMILIES) * len(rb.FORMATS)
    counts = summary.loc[summary["model"].eq("model-gemma") & summary["format"].eq("repetition")].set_index("variant")["item_count"].to_dict()
    assert counts == {
        "R4_raw_s": 6, "R6_bt_pass_strict_v1": 3, "R6_external_attested": 4,
        "R6_all_three_nllb": 3, "R6_min_surface_dist_ge_060": 4,
        "R7_competent": 2, "R9_main": 8, "R11_excluded": 6,
    }


def test_r_battery_wrapper_sets_primary_analysis_thread_caps(monkeypatch) -> None:
    wrapper = importlib.import_module("job_analyze_rbattery")
    analyzer = importlib.import_module("scripts.analyze_rbattery")
    config = importlib.import_module("vilens.config")
    locks = importlib.import_module("vilens.locks")
    provenance = importlib.import_module("vilens.provenance")
    primary = importlib.import_module("job_analyze_core")
    monkeypatch.setattr(wrapper.env_guard, "guard", lambda **_kwargs: None)
    monkeypatch.setattr(config, "load_config", lambda _path: {})
    monkeypatch.setattr(locks, "ensure_adjudication_committed", lambda _repo: None)
    monkeypatch.setattr(provenance, "git_state", lambda _repo: ("test", False))
    monkeypatch.setattr(analyzer, "preflight", lambda _cfg, repo: ("ready", True))
    for name in primary.THREAD_CAPS:
        monkeypatch.setenv(name, "4" if name != "TOKENIZERS_PARALLELISM" else "true")
    assert wrapper.main(["--preflight-only"]) == 0
    assert wrapper.THREAD_CAPS == primary.THREAD_CAPS
    assert {name: os.environ[name] for name in primary.THREAD_CAPS} == primary.THREAD_CAPS
