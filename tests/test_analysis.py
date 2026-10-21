"""Unit tests for the pure E2–E5 analysis helpers."""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vilens.analysis.bootstrap import bootstrap_key, bootstrap_means, concept_indices
from vilens.analysis.c1 import paired_selection_draws
from vilens.analysis.competence import competence_rows
from vilens.analysis.holm import holm_adjust, holm_table, window_positive_test
from vilens.analysis.la import (
    classification_accuracy,
    fit_layer_directions,
    d_en_curve,
    alignment_layer_status,
    la_layer_selections,
    la_rule_sensitivity_changed,
    native_validation_cosines,
    normalized_alignment,
    random_cosine_baseline,
)
from vilens.analysis.landmarks import concept_layer, landmarks, probability_concept_layer
from vilens.analysis.m1 import (
    absolute_delta_gate_passes,
    cyclic_shift_lens_control_deltas,
    first_token_lens_rows,
    lens_control_passes,
    lens_consistency_deltas,
    prefix_token_ids,
    require_nonempty_prefix_token_ids,
    summarize_lens_control_deltas,
    summarize_report_only_lens_deltas,
    unicode_prefix_token_id,
    unicode_prefix_token_id_for_family,
)
from vilens.analysis.rise import concept_rise_curves, decision_rule_quantities
from vilens.analysis.scores import content_token_scores, matched_distractor_deltas
from scripts import analyze_core


class TinyTokenizer:
    """Minimal vocabulary fixture with model-specific space markers."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [8 if text else 9]


def _long_rows() -> pd.DataFrame:
    """Make correct/distractor target rows with one two-token sequence."""
    rows = []
    for role, ref_id, values in (
        ("correct", "c1", (-1.0, -3.0, -2.0)),
        ("distractor", "d1", (-2.0, -4.0, -4.0)),
    ):
        for tok_idx, value in enumerate(values):
            rows.append(
                {
                    "model": "m",
                    "model_revision": "rev",
                    "pass": "pl",
                    "format": "repetition",
                    "condition": "diac",
                    "fewshot_set": 1,
                    "concept_id": "c1",
                    "target_lang": "en",
                    "target_variant": "released",
                    "role": role,
                    "ref_concept_id": ref_id,
                    "tok_idx": tok_idx,
                    "token_id": 10 + tok_idx,
                    "is_content": tok_idx != 1,
                    "layer": 1,
                    "metric": "lens_logprob",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def test_content_scores_and_matched_delta_exclude_noncontent_tokens() -> None:
    """s, s1, s+, and Δ use only content-token log probabilities."""
    scores = content_token_scores(_long_rows())
    correct = scores.loc[scores["role"].eq("correct")].iloc[0]
    assert correct["s"] == pytest.approx(-1.5)
    assert correct["s1"] == pytest.approx(-1.0)
    assert correct["s_plus"] == pytest.approx(-2.0)
    assert correct["sequence_sum"] == pytest.approx(-3.0)
    delta = matched_distractor_deltas(scores).iloc[0]
    assert delta["delta_s"] == pytest.approx(1.5)
    assert delta["delta_s1"] == pytest.approx(1.0)
    assert delta["delta_s_plus"] == pytest.approx(2.0)


def test_c1_stage_runs_end_to_end_on_tiny_trial_fixture(tmp_path: Path) -> None:
    """C1 reads its E3/E2 tables, computes effects, and writes the trial output."""
    family = "llama"
    core_root = tmp_path / "results/analysis/trial" / f"family={family}"
    e2_root = tmp_path / "results/analysis/m1/trial"
    core_root.mkdir(parents=True)
    e2_root.mkdir(parents=True)
    core_rows = []
    e2_rows = []
    variants = {"vi": "released", "en": "released", "zh": "nospace", "fr": "released", "id": "released"}
    for fmt in ("repetition", "translation"):
        for concept_index, concept_id in enumerate(("c1", "c2")):
            for language_index, language in enumerate(("vi", "en", "zh", "fr", "id")):
                for layer in (1, 2):
                    common = {
                        "model": "fixture-model",
                        "model_revision": "fixture-revision",
                        "format": fmt,
                        "concept_id": concept_id,
                        "target_lang": language,
                        "target_variant": variants[language],
                        "role": "correct",
                        "layer": layer,
                    }
                    core_rows.append(
                        {**common, "metric": "s", "value": -4.0 + concept_index + language_index * 0.1 + layer * 0.2}
                    )
                    e2_rows.append(
                        {
                            **common,
                            "metric": "prefix_sum_probability",
                            "value": 0.1 + concept_index * 0.05 + language_index * 0.01 + layer * 0.02,
                        }
                    )
    pd.DataFrame(core_rows).to_parquet(core_root / "scores.parquet", index=False)
    # Each format has its own E2 unit file.
    e2_by_format = pd.DataFrame(e2_rows)
    for fmt in ("repetition", "translation"):
        e2_by_format.loc[e2_by_format["format"].eq(fmt)].to_parquet(
            e2_root / f"family={family}__format={fmt}__pass=m1__source=trial.parquet",
            index=False,
        )

    cfg = {
        "seed": 20260925,
        "analysis": {
            "bootstrap_resamples": 4,
            "bootstrap_key_template": "bootstrap|{item_set}|{model}|{format}",
            "l_concept_fractions": [0.05, 0.10, 0.20],
        },
        "runtime": {"parquet_compression": "zstd"},
    }
    summary = analyze_core._run_c1_stage(cfg, tmp_path, [family], source="trial")

    output = core_root / "c1.parquet"
    assert output.is_file()
    assert summary["rows"] == 36
    assert len(pd.read_parquet(output)) == 36


def _checkpoint_h_trial_fixture(tmp_path: Path) -> dict[str, Any]:
    """Write small Checkpoint-H inputs using the production table schemas."""
    families = ("gemma", "qwen", "llama")
    languages = ("vi", "en", "zh", "fr", "id")
    summary_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("la_summary", "rule_quantities", "holm_w_raw", "landmarks", "c1", "competence")
    }

    def long_row(
        family: str, fmt: str, concept: str, language: str | None, layer: int, metric: str, value: float
    ) -> dict[str, Any]:
        return {
            "model": f"fixture/{family}",
            "model_revision": "fixture-revision",
            "pass": "analysis",
            "format": fmt,
            "condition": "diac",
            "fewshot_set": 1,
            "concept_id": concept,
            "target_lang": language,
            "target_variant": "released" if language != "zh" else "nospace",
            "role": "correct",
            "ref_concept_id": None,
            "tok_idx": None,
            "token_id": None,
            "is_content": None,
            "layer": layer,
            "metric": metric,
            "value": value,
        }

    for family in families:
        family_root = tmp_path / "results/analysis/trial" / f"family={family}"
        family_root.mkdir(parents=True)
        la_rows: list[dict[str, Any]] = []
        rise_rows: list[dict[str, Any]] = []
        for fmt in ("repetition", "translation"):
            for layer, layer_pass in ((1, 0.0), (2, 1.0)):
                la_rows.append(
                    long_row(family, fmt, "", None, layer, "layer_pass", layer_pass)
                )
                la_rows.append(
                    long_row(family, fmt, "", None, layer, "undefined_denominator", 0.0)
                )
            for concept_index, concept in enumerate(("c1", "c2")):
                for layer in (1, 2):
                    la_rows.append(
                        long_row(family, fmt, concept, None, layer, "D_en", 0.05 * layer + concept_index * 0.01)
                    )
                    for language_index, language in enumerate(languages):
                        la_rows.append(
                            long_row(
                                family, fmt, concept, language, layer, "LA",
                                0.1 + 0.01 * layer + 0.005 * language_index + 0.001 * concept_index,
                            )
                        )
                        rise_rows.append(
                            long_row(
                                family, fmt, concept, language, layer, "rise_cf729",
                                0.02 * layer + 0.003 * language_index + 0.001 * concept_index,
                            )
                        )
            for language in ("en", "zh", "fr", "id"):
                for metric, layer, value in (
                    ("max_LA", 2, 0.2),
                    ("max_LA_all_layer_sensitivity", 1, 0.3),
                ):
                    summary_rows["la_summary"].append(
                        {
                            "source": "trial", "model": f"fixture/{family}", "format": fmt,
                            "metric": metric, "target_lang": language, "layer": layer,
                            "value": value, "n_concepts": 2, "gate_correct": 2, "gate_pass": True,
                            "undefined_denominator": False, "c_random": 0.0,
                            "would_change_rule_outcome": metric == "max_LA_all_layer_sensitivity",
                            "model_revision": "fixture-revision", "pass": "analysis", "condition": "diac",
                            "fewshot_set": 1, "concept_id": None, "target_variant": None,
                            "ref_concept_id": None, "tok_idx": None, "token_id": None,
                            "is_content": None, "role": "path",
                        }
                    )
            for variant in ("s", "s1"):
                for language in languages:
                    summary_rows["rule_quantities"].append(
                        {
                            **long_row(family, fmt, "", language, 2, "M_L", 0.2),
                            "ci_low": 0.1, "ci_high": 0.3, "n_concepts": 2, "valid_draws": 4,
                            "n_layers": 2, "item_set": "cf729", "metric_variant": variant,
                            "category_value": None,
                        }
                    )
                for metric in ("h1_q", "h1_ratio", "h2_band_ratio"):
                    summary_rows["rule_quantities"].append(
                        {
                            **long_row(family, fmt, "", "__H1_H2__", 2, metric, 0.4),
                            "ci_low": 0.2, "ci_high": 0.6, "n_concepts": 2, "valid_draws": 4,
                            "n_layers": 2, "item_set": "cf729", "metric_variant": variant,
                            "category_value": None,
                        }
                    )
            for language in languages:
                for metric, value in (("W_L", 0.2), ("p_value", 0.5)):
                    summary_rows["holm_w_raw"].append(
                        {
                            **long_row(family, fmt, "", language, 2, metric, value),
                            "ci_low": 0.1, "ci_high": 0.3, "n_concepts": 2,
                            "valid_draws": 4, "p_value": value, "p_holm": value,
                            "item_set": "main", "metric_variant": "s",
                        }
                    )
            summary_rows["landmarks"].append(
                {
                    **long_row(family, fmt, "c1", "en", 2, "L_concept_10", 2),
                    "ci_low": 1, "ci_high": 2, "n_concepts": 2, "valid_draws": 4,
                    "n_layers": 2, "item_set": "cf729", "metric_variant": "s",
                    "category_value": None,
                }
            )
            summary_rows["c1"].append(
                {
                    "model": f"fixture/{family}", "format": fmt, "item_set": "m1_main",
                    "contrast": "instrument_effect", "threshold": 0.1,
                    "metric": "instrument_effect_10_blocks", "value": 0.0,
                    "ci_low": 0.0, "ci_high": 0.0, "valid_draws": 4,
                    "bootstrap_resamples": 4, "n_concepts": 2, "units": "blocks",
                    "model_revision": "fixture-revision", "pass": "analysis", "condition": "diac",
                    "fewshot_set": 1, "concept_id": None, "target_lang": None,
                    "target_variant": None, "role": "path", "ref_concept_id": None,
                    "tok_idx": None, "token_id": None, "is_content": None, "layer": None,
                }
            )
            summary_rows["competence"].append(
                {
                    "model": f"fixture/{family}", "format": fmt, "concept_id": "c1",
                    "condition": "diac", "answer": "fixture answer",
                    "competence_class": "VI", "competent": True,
                    "shared_ru_source": False, "shared_ru_group_n": 0,
                }
            )
        pd.DataFrame(la_rows, columns=analyze_core.LONG_COLUMNS).to_parquet(
            family_root / "la.parquet", index=False
        )
        pd.DataFrame(rise_rows, columns=analyze_core.LONG_COLUMNS).to_parquet(
            family_root / "rise.parquet", index=False
        )

    for family in families:
        family_root = tmp_path / "results/analysis/trial" / f"family={family}"
        for name in ("la_summary", "rule_quantities", "holm_w_raw", "landmarks", "c1", "competence"):
            rows = [row for row in summary_rows[name] if row.get("model") == f"fixture/{family}"]
            pd.DataFrame.from_records(rows).to_parquet(family_root / f"{name}.parquet", index=False)

    m1_root = tmp_path / "results/analysis/m1/trial"
    m1_root.mkdir(parents=True)
    code_tree = "fixture-code-tree"
    gate_by_family = {family: {"max_abs_delta": 0.001} for family in families}
    control_by_family = {family: {"median_abs_delta": 0.5} for family in families}
    gate_b_by_family = {family: {"max_abs_delta": 0.5} for family in families}
    unit_rows = []
    for family in families:
        for fmt in ("repetition", "translation"):
            filename = f"family={family}__format={fmt}__pass=m1__source=trial.parquet"
            pd.DataFrame(
                [{
                    "model": f"fixture/{family}", "model_revision": "fixture-revision", "pass": "m1",
                    "format": fmt, "condition": "diac", "fewshot_set": 1,
                    "concept_id": "c1", "target_lang": "vi", "target_variant": "released",
                    "role": "correct", "ref_concept_id": None, "tok_idx": None,
                    "token_id": None, "is_content": None, "layer": 1,
                    "metric": "prefix_sum_probability", "value": 0.5,
                }]
            ).to_parquet(m1_root / filename, index=False)
            (m1_root / f"{Path(filename).stem}.provenance.json").write_text(
                json.dumps(
                    {"git_commit": "fixture-commit", "code_tree_sha256": code_tree,
                     "git_dirty": False, "row_count": 1, "expected_row_count": 1}
                ),
                encoding="utf-8",
            )
            unit_rows.append(
                {"family": family, "format": fmt, "rows": 1, "expected_rows": 1, "units": 1}
            )
    (m1_root / "e2_job_summary.json").write_text(
        json.dumps(
            {
                "git_commit": "fixture-commit", "git_dirty": False,
                "code_tree_sha256": code_tree,
                "result": {
                    "source": "trial", "units": unit_rows, "gate_a_max_abs_delta": 0.001,
                    "gate_a_by_family": gate_by_family,
                    "gate_a_control_by_family": control_by_family,
                    "gate_b_by_family": gate_b_by_family,
                },
            }
        ),
        encoding="utf-8",
    )
    return {"families": list(families)}


def test_checkpoint_h_stage_runs_end_to_end_on_real_trial_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint-H tables work without matplotlib and include the required diagnostics."""
    original_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("mocked matplotlib absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    fixture_cfg = analyze_core.load_config(str(analyze_core.REPO / "configs/experiments.yaml"))
    fixture_cfg["pd"]["families"] = _checkpoint_h_trial_fixture(tmp_path)["families"]
    fixture_cfg["analysis"]["bootstrap_resamples"] = 4
    monkeypatch.setattr(analyze_core, "load_config", lambda _path: fixture_cfg)

    result = analyze_core.run_analysis(
        "trial", config_path="fixture-config", repo=tmp_path, stages="checkpoint_h"
    )

    outputs = result["files_and_rows"]
    assert len(outputs) == 14
    assert all((tmp_path / name).is_file() for name in outputs)
    assert outputs["results/analysis/trial/checkpoint_h_smoke/checkpoint_h_rules.csv"] == 6
    assert outputs["results/analysis/trial/checkpoint_h_smoke/checkpoint_h_verdicts.csv"] == 3
    assert outputs["results/analysis/trial/checkpoint_h_smoke/E2-E5_holm.csv"] == 30
    assert outputs["results/analysis/trial/checkpoint_h_smoke/E2-E5_a10.csv"] == 6
    sensitivity_path = tmp_path / "results/analysis/trial/checkpoint_h_smoke/checkpoint_h_la_sensitivity.csv"
    argmax_path = tmp_path / "results/analysis/trial/checkpoint_h_smoke/checkpoint_h_argmax_layers.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    argmax_layers = pd.read_csv(argmax_path)
    assert len(sensitivity) == 24
    assert sensitivity["would_change_rule_outcome"].all()
    assert {"argmax_gate_status", "previous_gate_status", "next_gate_status"}.issubset(
        argmax_layers.columns
    )
    m_argmax = argmax_layers.loc[
        argmax_layers["max_quantity"].eq("M_L")
        & argmax_layers["target_lang"].eq("en")
        & argmax_layers["metric_variant"].eq("s")
        & argmax_layers["family"].eq("gemma")
    ].iloc[0]
    assert m_argmax["argmax_gate_status"] == "pass"
    assert m_argmax["previous_gate_status"] == "failed"
    sensitivity_argmax = argmax_layers.loc[
        argmax_layers["max_quantity"].eq("max_LA_all_layer_sensitivity")
        & argmax_layers["target_lang"].eq("en")
        & argmax_layers["family"].eq("gemma")
        & argmax_layers["format"].eq("repetition")
    ].iloc[0]
    assert sensitivity_argmax["argmax_gate_status"] == "failed"


def test_matched_distractor_pairs_null_fewshot_sets() -> None:
    """No-few-shot parquet rows still form correct/distractor pairs."""
    rows = _long_rows().assign(fewshot_set=np.nan)
    deltas = matched_distractor_deltas(content_token_scores(rows))
    assert len(deltas) == 1
    assert deltas.iloc[0]["delta_s"] == pytest.approx(1.5)


def test_rise_uses_configured_early_blocks_and_excludes_layer_zero() -> None:
    """A synthetic known rise ignores an extreme embedding row at layer zero."""
    rows = []
    values = {0: 1_000.0, 1: 2.0, 2: 4.0, 3: 5.0, 4: 8.0, 5: 10.0, 6: 12.0, 7: 14.0, 8: 16.0, 9: 18.0, 10: 20.0}
    for layer, delta in values.items():
        rows.append(
            {
                "model": "m", "format": "repetition", "concept_id": "c1",
                "target_lang": "vi", "target_variant": "released", "layer": layer, "delta": delta,
            }
        )
    result = concept_rise_curves(pd.DataFrame(rows), early_depth_max=0.15)
    layer_one = result.loc[result["layer"].eq(1), "rise"].iloc[0]
    layer_two = result.loc[result["layer"].eq(2), "rise"].iloc[0]
    assert layer_one == pytest.approx(0.0)
    assert layer_two == pytest.approx(2.0)


def test_rise_keeps_single_token_continuation_undefined() -> None:
    """An unavailable continuation curve stays NA instead of failing rise."""
    rows = [
        {
            "model": "m",
            "format": "repetition",
            "concept_id": "c1",
            "target_lang": "zh",
            "target_variant": "nospace",
            "layer": layer,
            "delta_s_plus": float("nan"),
        }
        for layer in range(1, 5)
    ]
    result = concept_rise_curves(
        pd.DataFrame(rows), early_depth_max=0.5, value_column="delta_s_plus"
    )
    assert result["rise"].isna().all()


def test_h1_and_h2_rule_edges() -> None:
    """The H1 contrast includes equality and H2 requires positive maxima."""
    h1 = decision_rule_quantities({"vi": 0.5, "en": 1.0, "zh": 0.7, "fr": 0.5, "id": 0.5})
    assert h1["h1_q"] == pytest.approx(0.0)
    assert h1["h1_rise_holds"] is True
    assert h1["h1_ratio"] == pytest.approx(2.0)

    band = decision_rule_quantities({"vi": 0.5, "en": 0.5, "zh": 0.4, "fr": 0.6, "id": 0.5})
    assert band["h2_band_ratio"] == pytest.approx(1.5)
    assert band["h2_band_holds"] is True

    nonpositive = decision_rule_quantities({"vi": 0.0, "en": 0.2, "zh": 0.1, "fr": 0.1, "id": 0.1})
    assert nonpositive["h2_band_holds"] is False
    assert nonpositive["h2_band_ratio"] is None


def test_concept_landmarks_on_handmade_curves() -> None:
    """L_concept and L_cross return the manually specified block positions."""
    layers = [1, 2, 3, 4]
    mean_scores = {
        "vi": [-5.0, -4.0, -3.0, -2.0],
        "en": [-6.0, -5.0, -4.5, -3.0],
        "zh": [-6.0, -5.5, -5.0, -4.0],
        "fr": [-6.0, -5.5, -5.0, -4.0],
        "id": [-6.0, -5.5, -5.0, -4.0],
    }
    assert concept_layer(mean_scores, layers=layers, threshold=0.10) == 2
    result = landmarks(
        mean_scores,
        {
            "vi": [0.0, 1.0, 2.5, 3.0],
            "en": [0.0, 2.0, 1.0, 0.0],
            "zh": [0.0, 0.5, 0.4, 0.3],
            "fr": [0.0, 0.4, 0.3, 0.2],
            "id": [0.0, 0.3, 0.2, 0.1],
        },
        layers=layers,
        n_layers=4,
    )
    assert result["L_concept_10"] == 2
    assert result["L_peak"] == 2
    assert result["L_cross"] == 3
    assert result["pivot_width"] == pytest.approx(0.25)
    assert result["pivot_magnitude"] == pytest.approx(2.0)
    no_cross = landmarks(
        mean_scores,
        {
            "vi": [0.0, 0.1, 0.2, 0.3],
            "en": [0.0, 2.0, 2.0, 2.0],
            "zh": [0.0, 0.5, 0.4, 0.3],
            "fr": [0.0, 0.4, 0.3, 0.2],
            "id": [0.0, 0.3, 0.2, 0.1],
        },
        layers=layers,
        n_layers=4,
    )
    assert no_cross["L_cross"] is None
    assert no_cross["pivot_width"] is None


def test_probability_l_concept_uses_mean_probability_anchor() -> None:
    """C1 M1 landmarks compare mean prefix probability to final mean Vietnamese probability."""
    curves = {
        "vi": np.asarray([0.02, 0.10, 0.20]),
        "en": np.asarray([0.01, 0.04, 0.05]),
        "zh": np.asarray([0.01, 0.02, 0.04]),
        "fr": np.asarray([0.01, 0.03, 0.04]),
        "id": np.asarray([0.01, 0.01, 0.03]),
    }
    assert probability_concept_layer(curves, layers=[1, 2, 3], threshold=0.20) == 2


def test_m_l_is_nonnegative_for_every_rise_curve() -> None:
    """An early-block-centered rise has an early mean of zero, so its maximum cannot be negative."""
    rng = np.random.default_rng(91)
    for curve in rng.standard_normal((200, 12)):
        rise = curve - curve[:2].mean()
        assert float(np.max(rise)) >= 0.0


def test_bootstrap_is_deterministic_and_reuses_shared_indices() -> None:
    """A shared index matrix preserves cross-language pairing."""
    ids = ["c3", "c1", "c2"]
    first = concept_indices(ids, resamples=25, seed=20260925, key="bootstrap|test|m|f")
    second = concept_indices(ids, resamples=25, seed=20260925, key="bootstrap|test|m|f")
    np.testing.assert_array_equal(first, second)
    values = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    means = bootstrap_means(values, first)
    np.testing.assert_allclose(means[:, 1], means[:, 0] * 10.0)


def test_bootstrap_keys_are_stable_and_item_set_specific() -> None:
    """OPS-14 bootstrap streams name their item set as well as model and format."""
    main = bootstrap_key("main", "m", "repetition", template="bootstrap|{item_set}|{model}|{format}")
    cf = bootstrap_key("cf729", "m", "repetition", template="bootstrap|{item_set}|{model}|{format}")
    assert main == "bootstrap|main|m|repetition"
    assert cf == "bootstrap|cf729|m|repetition"
    assert main != cf
    assert bootstrap_key("main", "m", "repetition", template="bootstrap|{item_set}|{model}|{format}") == main


def test_w_l_bootstrap_test_and_holm_use_the_window_statistic() -> None:
    """Negative and positive OPS-14 window means produce the expected tail probabilities."""
    indices = concept_indices(["a", "b", "c", "d"], resamples=200, seed=10, key="window")
    layers = [1, 2, 3, 4]
    negative = window_positive_test(
        np.asarray([[-2.0] * 4, [-1.0] * 4, [-0.5] * 4, [-1.5] * 4]),
        layers=layers,
        window=[0.25, 0.75],
        indices=indices,
    )
    positive = window_positive_test(
        np.asarray([[1.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4]),
        layers=layers,
        window=[0.25, 0.75],
        indices=indices,
    )
    assert negative["p_value"] > 0.9
    assert positive["p_value"] < 0.02
    table = holm_table(
        pd.DataFrame({"model": ["a", "b", "c"], "p_value": [0.01, 0.04, 0.03]})
    )
    np.testing.assert_allclose(table["p_holm"], [0.03, 0.06, 0.06])


def test_paired_selection_draw_filters_full_draw_and_preserves_duplicates() -> None:
    """Selection-effect samples are the eligible concepts from the same main draw."""
    draws = np.asarray([[0, 1, 1, 2], [2, 0, 2, 1]])
    selected = paired_selection_draws(["a", "b", "c"], {"b"}, draws)
    assert len(selected) == 2
    np.testing.assert_array_equal(selected[0], [1, 1])
    np.testing.assert_array_equal(selected[1], [1])


def test_la_excludes_failed_and_undefined_layers_and_flags_sensitivity() -> None:
    """Only passing, defined blocks enter primary summaries; all defined blocks enter sensitivity."""
    usable, sensitivity = la_layer_selections(
        [True, False, True, True], [False, False, True, False]
    )
    np.testing.assert_array_equal(np.flatnonzero(usable), [0, 3])
    np.testing.assert_array_equal(np.flatnonzero(sensitivity), [0, 1, 3])
    assert alignment_layer_status(False, {"en": 0.5}, 0.1) == (False, False)
    assert alignment_layer_status(True, {"en": 0.05}, 0.1) == (False, True)
    assert la_rule_sensitivity_changed(0.55, 0.45, "en") is True


def test_landmark_bootstrap_reports_zero_valid_draws_for_no_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-crossing L_cross remains NA and carries an explicit zero valid-draw count."""
    from vilens import env_guard

    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_core.py"
    spec = importlib.util.spec_from_file_location("analyze_core_landmark_bootstrap_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rise_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    rise_by_language = {
        "vi": [0.0, 0.0, 0.0, 0.0],
        "en": [0.0, 2.0, 2.0, 2.0],
        "zh": [0.0, 0.5, 0.4, 0.3],
        "fr": [0.0, 0.4, 0.3, 0.2],
        "id": [0.0, 0.3, 0.2, 0.1],
    }
    for concept_id in ("c1", "c2"):
        for language, curve in rise_by_language.items():
            for layer, value in enumerate(curve, start=1):
                rise_rows.append(
                    {"model": "m", "format": "repetition", "concept_id": concept_id,
                     "target_lang": language, "layer": layer, "rise": value}
                )
                score_rows.append(
                    {"model": "m", "format": "repetition", "concept_id": concept_id,
                     "target_lang": language, "layer": layer, "s": -5.0 + layer}
                )
    _rules, landmarks_table, _holm = module._bootstrap_core(
        per_concept_rise=pd.DataFrame(rise_rows),
        per_concept_scores=pd.DataFrame(score_rows),
        source="synthetic",
        item_set="cf729",
        seed=20260925,
        analysis_config={
            "bootstrap_resamples": 1000,
            "bootstrap_key_template": "bootstrap|{item_set}|{model}|{format}",
            "window": [0.25, 0.75],
            "l_concept_fractions": [0.05, 0.1, 0.2],
        },
    )
    row = landmarks_table.loc[landmarks_table["metric"].eq("L_cross")].iloc[0]
    assert pd.isna(row["value"])
    assert row["valid_draws"] == 0


def test_la_gate_counts_errors_and_native_normalizes_to_one() -> None:
    """Centered synthetic language clusters classify and normalize as specified."""
    languages = ("vi", "en", "zh", "fr", "id")
    basis = np.eye(5, dtype=np.float32)
    fit_acts = np.repeat(basis, repeats=4, axis=0)
    fit_labels = [language for language in languages for _ in range(4)]
    center, directions, _means = fit_layer_directions(fit_acts, fit_labels, class_order=languages)
    accuracy, correct, count = classification_accuracy(basis, languages, center, directions)
    assert (accuracy, correct, count) == (1.0, 5, 5)
    wrong_labels = ["en", "en", "zh", "fr", "id"]
    wrong_accuracy, wrong_correct, wrong_count = classification_accuracy(
        basis, wrong_labels, center, directions
    )
    assert (wrong_accuracy, wrong_correct, wrong_count) == (0.8, 4, 5)

    centered = basis - center
    c_random = random_cosine_baseline(centered, seed=20260925, n_directions=5_000)
    assert abs(c_random) < 0.02
    native = native_validation_cosines(basis, languages, center, directions)
    alignments = normalized_alignment(
        basis,
        center,
        directions,
        native_cosines=native,
        c_random=c_random,
    )
    for language_index, language in enumerate(languages):
        assert alignments[language][language_index] == pytest.approx(1.0)

    random_queries = np.random.default_rng(17).standard_normal((1_000, 5)).astype(np.float32)
    random_alignment = normalized_alignment(
        random_queries,
        center,
        directions,
        native_cosines=native,
        c_random=c_random,
    )
    assert max(abs(float(values.mean())) for values in random_alignment.values()) < 0.05
    np.testing.assert_allclose(
        d_en_curve({"en": np.asarray([1.0, 2.0]), "fr": np.asarray([0.2, 0.4]), "id": np.asarray([0.4, 0.6])}),
        [0.7, 1.5],
    )


def test_holm_adjust_matches_hand_calculation() -> None:
    """Holm adjusted values match the hand-computed example."""
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_m1_prefix_ids_include_both_markers_and_deduplicate() -> None:
    """Vocabulary prefix IDs are deduplicated across plain and marked pieces."""
    gemma = TinyTokenizer({"a": 1, "ab": 2, "▁a": 3, "▁ab": 4, "ac": 5, "<bos>": 6})
    ids = prefix_token_ids(
        gemma,
        " ab",
        family="gemma",
        language="zh",
        unicode_prefix_tokid=2,
    )
    assert ids == (1, 2, 3, 4)
    qwen = TinyTokenizer({"a": 1, "ab": 2, "Ġa": 3, "Ġab": 4, "è": 8})
    assert prefix_token_ids(qwen, "ab", family="qwen", language="en") == (1, 2, 3, 4)
    assert unicode_prefix_token_id(qwen, " 路", family="qwen") == 8
    assert prefix_token_ids(
        qwen,
        " 路",
        family="qwen",
        language="zh",
        unicode_prefix_tokid=unicode_prefix_token_id(qwen, " 路", family="qwen"),
    ) == (8,)
    assert prefix_token_ids(
        qwen, "ab", family="qwen", language="zh", unicode_prefix_tokid=1
    ) == (1, 2, 3, 4)
    assert unicode_prefix_token_id(TinyTokenizer({"a": 1}), "路", family="gemma") is None


def test_unicode_byte_token_resolution_for_sentencepiece_and_byte_bpe() -> None:
    """Resolve first-byte fallback via SentencePiece spelling and in-memory byte BPE vocab."""
    sentencepiece = TinyTokenizer({"<0xE8>": 17})
    assert unicode_prefix_token_id_for_family(sentencepiece, "路", family="gemma") == 17

    from tokenizers import Tokenizer, models, pre_tokenizers

    byte_bpe = Tokenizer(models.BPE(vocab={"è": 23}, merges=[]))
    byte_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    assert unicode_prefix_token_id_for_family(byte_bpe, "路", family="llama") == 23


def test_m1_prefix_ids_use_tokenizer_alphabet_for_unicode_and_ascii() -> None:
    """Byte BPE and SentencePiece match initial-character tokens in both marker forms."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    from vilens.analysis.m1 import _byte_to_unicode

    words = ("đường", "学校", "apple")
    byte_vocab: dict[str, int] = {}
    expected_byte_ids: dict[str, set[int]] = {}
    for word in words:
        encoded_first = "".join(_byte_to_unicode()[value] for value in word[0].encode("utf-8"))
        ids = set()
        for token in (encoded_first, "Ġ" + encoded_first):
            token_id = len(byte_vocab)
            byte_vocab[token] = token_id
            ids.add(token_id)
        expected_byte_ids[word] = ids
    byte_bpe = Tokenizer(models.BPE(vocab=byte_vocab, merges=[]))
    byte_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    for word in words:
        actual = set(prefix_token_ids(byte_bpe, word, family="llama", language="vi"))
        assert expected_byte_ids[word].issubset(actual)

    sentencepiece_vocab: dict[str, int] = {}
    expected_sentencepiece_ids: dict[str, set[int]] = {}
    for word in words:
        first = word[0]
        ids = set()
        for token in (first, "▁" + first):
            token_id = len(sentencepiece_vocab)
            sentencepiece_vocab[token] = token_id
            ids.add(token_id)
        expected_sentencepiece_ids[word] = ids
    sentencepiece = TinyTokenizer(sentencepiece_vocab)

    for word in words:
        actual = set(prefix_token_ids(sentencepiece, word, family="gemma", language="vi"))
        assert expected_sentencepiece_ids[word].issubset(actual)


def test_m1_empty_prefix_error_names_the_item() -> None:
    """A missing prefix is a named failure, never an empty result row."""
    with pytest.raises(
        ValueError,
        match=r"concept_id=broken readout=vi/released.*target='missing'",
    ):
        require_nonempty_prefix_token_ids(
            TinyTokenizer({}),
            "missing",
            family="gemma",
            language="vi",
            item_description="concept_id=broken readout=vi/released",
        )


def test_e2_consistency_indexes_all_layers_and_uses_primary_rows_over_paths() -> None:
    """P-L consistency spans layer 0..N, with correct rows preferred over paths."""
    rows: list[dict[str, Any]] = []
    for layer in range(3):
        rows.extend(
            [
                {"concept_id": "c1", "target_lang": "vi", "target_variant": "released",
                 "role": "correct", "layer": layer, "token_id": 10, "value": -1.0},
                {"concept_id": "c1", "target_lang": "en", "target_variant": "released",
                 "role": "path", "layer": layer, "token_id": 20, "value": -2.0},
                {"concept_id": "c1", "target_lang": "en", "target_variant": "released",
                 "role": "correct", "layer": layer, "token_id": 30, "value": -3.0},
            ]
        )
    indexed = first_token_lens_rows(
        rows,
        concept_ids={"c1"},
        readouts=(("vi", "released"), ("en", "released")),
        n_layers=2,
    )
    assert len(indexed) == 6
    assert indexed[("c1", "vi", "released", 0)] == (10, -1.0)
    assert indexed[("c1", "en", "released", 0)] == (30, -3.0)


def test_e2_gate_a_absolute_tolerance_passes_observed_gemma_and_fails_point_05() -> None:
    """The absolute Gate A bound accepts observed fp32 noise and rejects 0.05."""
    observed_delta = abs(-6.17694139 - -6.17661476)
    assert observed_delta == pytest.approx(0.00032663, abs=1.0e-8)
    assert absolute_delta_gate_passes(observed_delta, tolerance=1.0e-2)
    assert not absolute_delta_gate_passes(0.05, tolerance=1.0e-2)


def test_e2_gate_a_control_uses_next_prompt_same_layer_and_skips_layer_zero() -> None:
    """The negative control wraps prompt rows cyclically and excludes layer 0."""
    prompt_ids = ["p0", "p1", "p2"]
    next_prompt = {"p0": "p1", "p1": "p2", "p2": "p0"}
    current_log_probs = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=np.float64,
    )
    answer_topk = {
        ("p1", 0): [{"token_id": 1, "value": -9.0}],
        ("p2", 0): [{"token_id": 2, "value": -9.0}],
        ("p0", 0): [{"token_id": 0, "value": -9.0}],
        ("p1", 1): [{"token_id": 1, "value": 0.45}],
        ("p2", 1): [{"token_id": 2, "value": 0.6}],
        ("p0", 1): [{"token_id": 0, "value": 0.55}],
    }

    layer_zero = cyclic_shift_lens_control_deltas(
        current_log_probs, prompt_ids, next_prompt, answer_topk, layer=0
    )
    assert layer_zero.size == 0

    shifted = cyclic_shift_lens_control_deltas(
        current_log_probs, prompt_ids, next_prompt, answer_topk, layer=1
    )
    assert shifted.tolist() == pytest.approx([0.25, 0.0, 0.15])
    summary = summarize_lens_control_deltas(shifted)
    assert summary["matched_count"] == 3
    assert summary["median_abs_delta"] == pytest.approx(0.15)
    assert lens_control_passes(summary, minimum_median=0.1)


def test_e2_gate_a_control_fails_when_cyclically_shifted_rows_are_identical() -> None:
    """Identical prompt rows produce a zero median and fail discrimination."""
    prompt_ids = ["p0", "p1"]
    next_prompt = {"p0": "p1", "p1": "p0"}
    current_log_probs = np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float64)
    answer_topk = {
        ("p1", 1): [{"token_id": 0, "value": 0.2}],
        ("p0", 1): [{"token_id": 1, "value": 0.8}],
    }
    deltas = cyclic_shift_lens_control_deltas(
        current_log_probs, prompt_ids, next_prompt, answer_topk, layer=1
    )
    summary = summarize_lens_control_deltas(deltas)
    assert summary["median_abs_delta"] == 0.0
    assert not lens_control_passes(summary, minimum_median=0.1)


def test_e2_gate_b_reports_small_noise_and_layer_offset_without_failing() -> None:
    """Gate B reports cross-forward deltas without applying a failure threshold."""
    e2_log_probs = np.asarray([-0.1, -1.5, -0.2, -1.7], dtype=np.float64)
    small_noise = np.asarray([0.02, -0.03, 0.04, -0.01], dtype=np.float64)
    noisy_reference = e2_log_probs + small_noise
    deltas = lens_consistency_deltas(e2_log_probs, noisy_reference)
    summary = summarize_report_only_lens_deltas(deltas)
    assert summary["max_abs_delta"] <= 1.0
    assert summary["matched_count"] == 4

    layer_shifted = e2_log_probs.reshape(2, 2)[:, ::-1].reshape(-1)
    shifted_summary = summarize_report_only_lens_deltas(
        lens_consistency_deltas(e2_log_probs, layer_shifted)
    )
    assert shifted_summary["max_abs_delta"] > 1.0
    assert shifted_summary["matched_count"] == 4


def test_e2_trial_check_failure_stops_before_primary_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed directions-trial check prevents every primary E2 unit from running."""
    from vilens import env_guard

    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    script = Path(__file__).resolve().parents[1] / "job_e2_m1.py"
    spec = importlib.util.spec_from_file_location("job_e2_m1_trial_check_failure", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[str] = []

    def failed_trial(_cfg: Any, source: str, _run_git: Any) -> Any:
        calls.append(source)
        if source == "trial":
            raise AssertionError("trial row/probability/lens check failed")
        pytest.fail("primary E2 must not run after a failed trial check")

    monkeypatch.setattr(module, "_run_source_once", failed_trial)
    with pytest.raises(AssertionError, match="trial row/probability/lens check failed"):
        module._run_source({}, "primary", None, with_trial_check=True)
    assert calls == ["trial"]


def test_competence_flags_shared_russian_source() -> None:
    """A Russian source group can score a sibling Vietnamese concept as VI."""
    concepts = {
        "c1": {"concept_id": "c1", "ru": "палец", "vi": "ngón tay", "vi_alts": [], "en": "finger", "zh": "手指", "fr": "doigt", "id": "jari"},
        "c2": {"concept_id": "c2", "ru": "палец", "vi": "ngón chân", "vi_alts": [], "en": "toe", "zh": "趾", "fr": "orteil", "id": "jari kaki"},
    }
    rows = competence_rows(
        [{"model": "m", "format": "translation", "condition": "diac", "concept_id": "c1", "answer": "ngón chân"}],
        concepts,
    )
    assert rows.loc[0, "competence_class"] == "VI"
    assert bool(rows.loc[0, "shared_ru_source"]) is True
    assert rows.loc[0, "shared_ru_group_n"] == 2


def test_primary_cli_refuses_before_opening_results_pl(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primary guard fires before any P-L file operation can occur."""
    from vilens import env_guard

    # The entrypoint checks for heavyweight imports at process startup. Pytest
    # has already imported torch, so bypass only that startup guard while
    # loading the module for this isolated CLI guard test.
    monkeypatch.setattr(env_guard, "guard", lambda offline: None)
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_core.py"
    spec = importlib.util.spec_from_file_location("analyze_core_for_guard_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_open = Path.open
    original_glob = Path.glob

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if "results" in path.parts and "pl" in path.parts:
            raise AssertionError("the guard attempted to open results/pl")
        return original_open(path, *args, **kwargs)

    def guarded_glob(path: Path, pattern: str) -> Any:
        if "results" in path.parts and "pl" in path.parts:
            raise AssertionError("the guard attempted to enumerate results/pl")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "glob", guarded_glob)
    original_run_analysis = module.run_analysis
    monkeypatch.setattr(module, "run_analysis", lambda *args, **kwargs: pytest.fail("analysis ran"))
    with pytest.raises(SystemExit) as error:
        module.main(["--source", "primary"])
    assert error.value.code == 2
    with pytest.raises(PermissionError, match="confirm-checkpoint-h"):
        original_run_analysis("primary")
