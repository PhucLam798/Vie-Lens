"""Synthetic tests for R1/R2 descriptive robustness helpers."""

from __future__ import annotations

from pathlib import Path
import importlib
import os

import numpy as np
import pytest


def test_r1_query_index_accepts_prompt_row_without_language_field() -> None:
    from scripts.analyze_core import r1_query_index_row_selected

    assert r1_query_index_row_selected({"fewshot_set": 4, "split": "test"}, 4)
    assert not r1_query_index_row_selected({"fewshot_set": 3, "split": "test"}, 4)
    assert not r1_query_index_row_selected({"fewshot_set": 4, "split": "fit"}, 4)


def test_r1_expected_units_follow_set_order_and_shard_counts() -> None:
    from scripts.analyze_r1 import expected_unit_keys

    cfg = {
        "pl": {
            "shard_prompts": 250,
            "r1": {
                "fewshot_sets": [2, 3, 4, 5, 6],
                "expected": {"repetition": 729, "translation": 721},
            },
        }
    }
    units = expected_unit_keys(cfg)
    assert len(units) == 90
    assert [int(unit["fewshot_set"]) for unit in units[:6]] == [2, 2, 2, 2, 2, 2]
    assert [unit["format"] for unit in units[:6]] == [
        "repetition", "repetition", "repetition", "translation", "translation", "translation"
    ]


def test_r1_a10_summary_uses_shared_scorer_by_set() -> None:
    from scripts.analyze_r1 import r1_a10_summary

    concepts = {
        "c1": {
            "concept_id": "c1", "vi": "synthetic_vi_one", "en": "synthetic_en_one",
            "zh": "synthetic_zh_one", "fr": "synthetic_fr_one", "id": "synthetic_id_one",
            "ru": "synthetic_ru_one", "vi_alts": [],
        }
    }
    generations = [
        {
            "model": "llama", "format": "repetition", "condition": "diac",
            "fewshot_set": 1, "concept_id": "c1", "answer": "synthetic_en_one",
        },
        {
            "model": "llama", "format": "repetition", "condition": "diac",
            "fewshot_set": 2, "concept_id": "c1", "answer": "synthetic_vi_one",
        },
    ]
    summary = r1_a10_summary(generations, concepts).set_index("fewshot_set")

    assert summary.loc[1, "a10_en_n"] == 1
    assert summary.loc[1, "a10_vi_n"] == 0
    assert summary.loc[1, "a10_denominator"] == 1
    assert summary.loc[1, "value"] == 0.0
    assert summary.loc[2, "a10_vi_n"] == 1
    assert summary.loc[2, "a10_denominator"] == 1
    assert summary.loc[2, "value"] == 1.0


def test_r2_identical_direction_inputs_have_unit_cosines() -> None:
    from scripts.analyze_r2 import flores_direction_cosines

    labels = np.repeat(np.asarray(["vi", "en", "zh", "fr", "id"], dtype=object), 2)
    basis = np.eye(5, dtype=np.float32)
    activations = np.repeat(basis, 2, axis=0)
    values = flores_direction_cosines(activations, labels, activations.copy(), labels.copy())
    assert set(values) == {"v_vi", "v_en", "v_zh", "v_fr", "v_id", "d_vi_minus_en"}
    assert all(np.isclose(value, 1.0) for value in values.values())


def test_r2_classification_reports_count_and_registered_criterion() -> None:
    from scripts.analyze_r2 import flores_classification

    labels = np.repeat(np.asarray(["vi", "en", "zh", "fr", "id"], dtype=object), 2)
    fit = np.repeat(np.eye(5, dtype=np.float32), 2, axis=0)
    valid_labels = np.repeat(np.asarray(["vi", "en", "zh", "fr", "id"], dtype=object), 26)
    valid = np.repeat(np.eye(5, dtype=np.float32), 26, axis=0)
    correct, total, passes = flores_classification(fit, labels, valid, valid_labels)
    assert (correct, total, passes) == (130, 130, True)


def test_r2_bootstrap_key_is_deterministic_and_pooling_specific() -> None:
    from scripts.analyze_r2 import r2_bootstrap_key

    key = r2_bootstrap_key("model-id", "repetition", "last")
    assert key == "bootstrap|r2|model-id|repetition|last"
    assert key == r2_bootstrap_key("model-id", "repetition", "last")
    assert key != r2_bootstrap_key("model-id", "repetition", "mean")


def test_r1_wrapper_resolves_config_from_repo_root() -> None:
    from job_analyze_r1 import resolve_repo_path

    assert resolve_repo_path("configs/experiments.yaml", repo_root=Path("/tmp/project")) == Path(
        "/tmp/project/configs/experiments.yaml"
    )


def test_r2_wrapper_resolves_config_from_repo_root() -> None:
    from job_r2_directions import resolve_repo_path

    assert resolve_repo_path("configs/experiments.yaml", repo_root=Path("/tmp/project")) == Path(
        "/tmp/project/configs/experiments.yaml"
    )


@pytest.mark.parametrize(
    ("wrapper_name", "analysis_name"),
    (("job_analyze_r1", "scripts.analyze_r1"), ("job_r2_directions", "scripts.analyze_r2")),
)
def test_r1_r2_wrappers_apply_primary_analysis_thread_caps(
    monkeypatch: pytest.MonkeyPatch, wrapper_name: str, analysis_name: str
) -> None:
    wrapper = importlib.import_module(wrapper_name)
    analyzer = importlib.import_module(analysis_name)
    primary = importlib.import_module("job_analyze_core")
    config = importlib.import_module("vilens.config")
    locks = importlib.import_module("vilens.locks")
    provenance = importlib.import_module("vilens.provenance")

    monkeypatch.setattr(wrapper.env_guard, "guard", lambda **_kwargs: None)
    monkeypatch.setattr(config, "load_config", lambda _path: {})
    monkeypatch.setattr(locks, "ensure_adjudication_committed", lambda _repo: None)
    monkeypatch.setattr(provenance, "git_state", lambda _repo: ("test-commit", False))
    monkeypatch.setattr(analyzer, "preflight", lambda _cfg, repo: ("ready", True))
    for name in primary.THREAD_CAPS:
        monkeypatch.setenv(name, "4" if name != "TOKENIZERS_PARALLELISM" else "true")

    assert wrapper.main(["--preflight-only"]) == 0
    assert wrapper.THREAD_CAPS == primary.THREAD_CAPS
    assert {name: os.environ[name] for name in primary.THREAD_CAPS} == primary.THREAD_CAPS
