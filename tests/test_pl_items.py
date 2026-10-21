"""Synthetic coverage for the expanded P-L readout item set."""

from __future__ import annotations

import copy
import importlib.util
import logging
from typing import Any
from pathlib import Path

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vilens import models as model_api
from vilens.config import load_config
from vilens.derived import target_string
from vilens.lens import LensHead
from vilens.passes import build_pl_items, run_pl


def _small_tokenizer() -> PreTrainedTokenizerFast:
    """Build a deterministic in-memory byte-level tokenizer."""
    backend = Tokenizer(
        models.BPE(
            vocab={character: index for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
            merges=[],
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _fixtures(family: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create main, extension, and distractor forms for item expansion tests."""
    concepts = {
        concept_id: {
            "concept_id": concept_id,
            "vi": vi,
            "en": en,
            "zh": zh,
            "fr": fr,
            "id": ind,
            "ru": ru,
            "m1_extension": extension,
            "collapsed": "false",
        }
        for concept_id, vi, en, zh, fr, ind, ru, extension in (
            ("main", "một", "one", "一", "un", "satu", "один", "false"),
            ("extension", "hai", "two", "二", "deux", "dua", "два", "true"),
            ("distractor", "ba", "three", "三", "trois", "tiga", "три", "false"),
        )
    }
    combos = [tuple(pair) for pair in load_config("configs/experiments.yaml")["pl"]["correct_readouts"]]
    target_rows = [
        {
            "concept_id": concept_id,
            "target_lang": lang,
            "target_variant": variant,
            "target": target_string(concept, lang, variant),
        }
        for concept_id, concept in concepts.items()
        for lang, variant in combos
    ]
    distractors = [
        {
            "family": family,
            "target_lang": lang,
            "target_variant": variant,
            "concept_id": "main",
            "distractor_id": "distractor",
        }
        for lang, variant in combos
        if lang != "ru"
    ]
    records = [
        {"concept_id": "main", "format": "repetition", "condition": "diac", "fewshot_set": 1,
         "split": "test", "prompt": "x:"},
        {"concept_id": "extension", "format": "repetition", "condition": "diac", "fewshot_set": 1,
         "split": "test", "prompt": "y:"},
        {"concept_id": "main", "format": "translation", "condition": "diac", "fewshot_set": 1,
         "split": "test", "prompt": "z:"},
    ]
    return concepts, records, target_rows, distractors


def test_primary_item_expansion_counts_and_extension_ru_behavior(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expand main, extension, and translation prompts into the declared sequence totals."""
    loaded = tiny_bundle["loaded"]
    tokenizer = _small_tokenizer()
    loaded.tokenizer = tokenizer
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    concepts, records, targets, distractors = _fixtures(loaded.family)
    items = build_pl_items(
        tiny_bundle["cfg"], records, concepts, targets, distractors, loaded.family, "primary"
    )
    assert len(items) == 22
    assert sum(1 + (item.distractor_encoding is not None) for item in items if item.record["format"] == "repetition") == 21
    assert sum(1 + (item.distractor_encoding is not None) for item in items if item.record["format"] == "translation") == 15
    extension_items = [item for item in items if item.concept_id == "extension"]
    assert len(extension_items) == 7
    assert all(item.distractor_encoding is None for item in extension_items)
    assert sum(item.target_lang == "ru" for item in items) == 1
    assert all(item.distractor_encoding is None for item in items if item.target_lang == "ru")
    assert all(item.correct_encoding is not None and item.correct_encoding.prefix_ok for item in items)


def test_r1_uses_only_the_five_primary_readouts_and_matched_distractors(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1's fixed P-L capture expands five primary readouts and five distractors."""
    tokenizer = _small_tokenizer()
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    concepts, records, targets, distractors = _fixtures(tiny_bundle["family"])
    cfg = copy.deepcopy(tiny_bundle["cfg"])
    cfg["pl"]["r1"]["fewshot_sets"] = [2]
    r1_record = {**records[0], "fewshot_set": 2}
    items = build_pl_items(
        cfg, [r1_record], concepts, targets, distractors, tiny_bundle["family"], "r1"
    )
    assert len(items) == 5
    assert [item.target_lang for item in items] == ["vi", "en", "zh", "fr", "id"]
    assert all(item.distractor_encoding is not None for item in items)
    assert sum(1 + (item.distractor_encoding is not None) for item in items) == 10
    assert all(not item.path_forms for item in items)


def test_r1_item_expansion_rejects_m1_extension(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 cannot accidentally capture an extension concept."""
    tokenizer = _small_tokenizer()
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    concepts, records, targets, distractors = _fixtures(tiny_bundle["family"])
    cfg = copy.deepcopy(tiny_bundle["cfg"])
    cfg["pl"]["r1"]["fewshot_sets"] = [2]
    extension_record = {**records[1], "fewshot_set": 2}
    with pytest.raises(ValueError, match="excludes M1 extension"):
        build_pl_items(
            cfg, [extension_record], concepts, targets, distractors, tiny_bundle["family"], "r1"
        )


def test_e9_item_selection_excludes_extensions_and_collapsed_collisions(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Select only E9 main, non-collapsed nodiac records and wire vi-nodiac targets."""
    from vilens import env_guard

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(env_guard, "guard", lambda **_kwargs: None)
    spec = importlib.util.spec_from_file_location("t25_job_pl_lens_e9_test", repo_root / "job_pl_lens.py")
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    monkeypatch.setattr(job, "REPO", tmp_path)

    cfg = copy.deepcopy(tiny_bundle["cfg"])
    cfg["pl"]["e9"]["expected"] = {
        "repetition": 1,
        "translation": 1,
        "cloze": 1,
        "sequences": 30,
        "units_per_family": 3,
    }
    cfg["pl"]["e9"]["results_dir"] = str(tmp_path / "e9")
    concepts, fixture_records, targets, distractors = _fixtures(tiny_bundle["family"])
    concepts["collapsed"] = {
        **concepts["main"], "concept_id": "collapsed", "collapsed": "true",
    }
    prompt_rows = {
        "repetition_nodiac_set1.jsonl": [
            {**fixture_records[0], "condition": "nodiac"},
            {**fixture_records[1], "concept_id": "extension", "condition": "nodiac"},
            {**fixture_records[0], "concept_id": "collapsed", "condition": "nodiac"},
        ],
        "translation_nodiac_set1.jsonl": [
            {**fixture_records[2], "condition": "nodiac"},
        ],
        "cloze_nodiac_set1.jsonl": [
            {**fixture_records[0], "format": "cloze", "condition": "nodiac", "fewshot_set": None},
        ],
    }
    monkeypatch.setattr(job, "_read_jsonl", lambda path: prompt_rows[path.name])
    monkeypatch.setattr(job, "_guard_model_records", lambda _cfg, records: list(records))
    selected = job._e9_records(cfg, concepts)
    assert [(row["format"], row["concept_id"]) for row in selected] == [
        ("repetition", "main"),
        ("translation", "main"),
        ("cloze", "main"),
    ]
    e9_counts = job._e9_counts(cfg, selected)
    assert e9_counts["sequences"] == 30
    manifest_path = tmp_path / cfg["paths"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_inputs = job._selected_inputs("e9", cfg)[1:]
    manifest_path.write_text(
        "".join(f"{'a' * 64}  {relative}\n" for relative in prompt_inputs),
        encoding="utf-8",
    )
    assert set(job._e9_prompt_manifest_hashes(cfg)) == {
        relative.rsplit("/", maxsplit=1)[-1] for relative in prompt_inputs
    }

    r1_root = tmp_path / cfg["pl"]["r1"]["results_dir"]
    r1_root.mkdir(parents=True)
    for family in cfg["pl"]["families"]:
        (r1_root / f"unit__family={family}.parquet").write_bytes(b"x" * 1000)
    projection = job._e9_disk_projection(cfg, e9_counts)
    assert set(projection["projected_bytes_by_family"]) == set(cfg["pl"]["families"])
    assert projection["projected_total_bytes"] == sum(projection["projected_bytes_by_family"].values())

    tokenizer = _small_tokenizer()
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    items = build_pl_items(
        cfg, selected, concepts, targets, distractors, tiny_bundle["family"], "e9"
    )
    assert len(items) == 15
    assert all(item.target_lang != "vi" or item.target_variant == "nodiac" for item in items)
    assert all(item.distractor_encoding is not None for item in items)
    assert all(not item.path_forms for item in items)


def test_missing_main_distractor_raises(tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail item construction when a required main-concept distractor is absent."""
    tokenizer = _small_tokenizer()
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    concepts, records, targets, _distractors = _fixtures(tiny_bundle["family"])
    with pytest.raises(KeyError, match="Missing main-concept distractor"):
        build_pl_items(tiny_bundle["cfg"], records[:1], concepts, targets, [], tiny_bundle["family"], "primary")


def test_path_rows_are_scored_only_from_vi_correct_sequences(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit along-path rows only for the Vietnamese released correct readout."""
    loaded = tiny_bundle["loaded"]
    tokenizer = _small_tokenizer()
    loaded.tokenizer = tokenizer
    monkeypatch.setattr(model_api, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    concepts, records, targets, distractors = _fixtures(loaded.family)
    items = build_pl_items(
        tiny_bundle["cfg"], records, concepts, targets, distractors, loaded.family, "primary"
    )
    expected = {
        (item.record["format"], item.concept_id)
        for item in items
        if item.target_lang == "vi" and item.target_variant == "released"
    }

    def fake_generate(input_ids: torch.Tensor, *, do_sample: bool, max_new_tokens: int) -> torch.Tensor:
        return torch.cat((input_ids, torch.tensor([[1]], device=input_ids.device)), dim=1)

    monkeypatch.setattr(loaded.model, "generate", fake_generate)
    output = run_pl(
        loaded,
        LensHead.from_loaded(loaded, "fp32"),
        items,
        batch_size=32,
        top_k=2,
        gen_max_new_tokens=1,
    )
    path_rows = output.long.loc[output.long["role"] == "path"]
    observed = set(zip(path_rows["format"], path_rows["concept_id"], strict=True))
    assert observed == expected
    assert set(path_rows["target_lang"]) == {"en", "zh", "fr", "id"}


def test_primary_entrypoint_refuses_closed_test_item_lock(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse primary P-L before model work when the fixture's test lock is closed."""
    repo_root = Path(__file__).resolve().parents[1]
    from vilens import env_guard

    # The entrypoint's module guard runs before its torch imports. The test has
    # already built a tiny model, so stub only that import-time guard here.
    monkeypatch.setattr(env_guard, "guard", lambda **_kwargs: None)
    spec = importlib.util.spec_from_file_location("t11_job_pl_lens_test", repo_root / "job_pl_lens.py")
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    cfg = copy.deepcopy(tiny_bundle["cfg"])
    cfg["prereg"]["analysis_tag"] = None
    cfg["prereg"]["test_items_unlocked"] = False
    monkeypatch.setattr(job, "load_config", lambda _path: cfg)
    monkeypatch.setattr(job, "setup_logging", lambda *_args, **_kwargs: logging.getLogger("t11.primary.lock"))
    monkeypatch.setattr(job, "capture_run_git", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        job,
        "_read_jsonl",
        lambda _path: [
            {
                "concept_id": "synthetic-test",
                "format": "repetition",
                "condition": "diac",
                "fewshot_set": 1,
                "split": "test",
                "prompt": "x:",
            }
        ],
    )

    with caplog.at_level(logging.ERROR):
        result = job.main(["--items", "primary", "--preflight-only"])

    assert result == 2
    assert "IntegrityLockError" in caplog.text
    assert "Test-item model access" in caplog.text


def _primary_entrypoint_fixture(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load the entrypoint with a fixture config and isolated output directory."""
    repo_root = Path(__file__).resolve().parents[1]
    from vilens import env_guard

    monkeypatch.setattr(env_guard, "guard", lambda **_kwargs: None)
    spec = importlib.util.spec_from_file_location("t12_job_pl_lens_test", repo_root / "job_pl_lens.py")
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    cfg = copy.deepcopy(tiny_bundle["cfg"])
    cfg["pl"]["results_dir"] = str(tmp_path)
    record = {
        "concept_id": "synthetic-test",
        "format": "repetition",
        "condition": "diac",
        "fewshot_set": 1,
        "split": "test",
        "prompt": "x:",
    }
    monkeypatch.setattr(job, "load_config", lambda _path: cfg)
    monkeypatch.setattr(job, "setup_logging", lambda *_args, **_kwargs: logging.getLogger("t12.primary"))
    monkeypatch.setattr(job, "capture_run_git", lambda *_args, **_kwargs: None)
    return job, cfg, record


def test_primary_nonpreflight_dispatches_with_open_test_item_lock(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dispatch primary model work when both human-controlled test locks are open."""
    from vilens.preflight import Check

    job, cfg, record = _primary_entrypoint_fixture(tiny_bundle, monkeypatch, tmp_path)
    cfg["prereg"]["analysis_tag"] = "prereg-v1.4"
    cfg["prereg"]["test_items_unlocked"] = True

    def passing_preflight(
        candidate_cfg: dict[str, Any], mode: str, *, gpu: bool
    ) -> tuple[list[dict[str, Any]], dict[str, int], list[Any]]:
        assert mode == "primary" and gpu
        checked = job._guard_model_records(candidate_cfg, [record])
        return checked, {"prompts": 1}, [Check("integrity_locks", "PASS", "fixture lock pass")]

    monkeypatch.setattr(job, "_preflight", passing_preflight)
    model_calls: list[tuple[str, int]] = []

    def mocked_model_run(
        candidate_cfg: dict[str, Any], logger: logging.Logger, run_git: Any,
        records: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        model_calls.append((candidate_cfg["models"]["study_set"], len(records)))
        return 0, {"mocked_model_run": True}

    monkeypatch.setattr(job, "_run_primary", mocked_model_run)
    assert job.main(["--items", "primary"]) == 0
    assert model_calls == [(cfg["models"]["study_set"], 1)]


@pytest.mark.parametrize(
    ("analysis_tag", "test_items_unlocked"),
    [("prereg-v1.4", False), (None, True)],
)
def test_primary_nonpreflight_refuses_closed_lock_before_model_call(
    tiny_bundle: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    analysis_tag: str | None,
    test_items_unlocked: bool,
) -> None:
    """Keep primary model work behind both integrity-lock values."""
    job, cfg, record = _primary_entrypoint_fixture(tiny_bundle, monkeypatch, tmp_path)
    cfg["prereg"]["analysis_tag"] = analysis_tag
    cfg["prereg"]["test_items_unlocked"] = test_items_unlocked

    def locked_preflight(
        candidate_cfg: dict[str, Any], mode: str, *, gpu: bool
    ) -> tuple[list[dict[str, Any]], dict[str, int], list[Any]]:
        assert mode == "primary" and gpu
        checked = job._guard_model_records(candidate_cfg, [record])
        return checked, {"prompts": 1}, []

    monkeypatch.setattr(job, "_preflight", locked_preflight)
    model_calls: list[bool] = []
    monkeypatch.setattr(
        job,
        "_run_primary",
        lambda *_args: (model_calls.append(True) or 0, {"unexpected": True}),
    )
    with caplog.at_level(logging.ERROR):
        assert job.main(["--items", "primary"]) == 2
    assert model_calls == []
    assert "IntegrityLockError" in caplog.text
    assert "Test-item model access" in caplog.text
