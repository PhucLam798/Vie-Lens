"""CPU-only coverage for reusable prompt and teacher-forced passes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from vilens import hooks
from vilens import passes
from vilens.io import LENS_COLUMNS
from vilens.provenance import capture_run_git
from vilens.passes import (
    Item,
    PathForm,
    build_items,
    persist_pl_outputs,
    run_pd,
    run_pd_resume_safe,
    run_pl,
    run_pl_resume_safe,
)


def _byte_tokenizer(add_bos: bool = False) -> PreTrainedTokenizerFast:
    """Build an in-memory byte-level tokenizer matching the X0 fixtures."""
    backend = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
            merges=[],
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    if add_bos:
        token = "<bos>"
        backend.add_special_tokens([token])
        token_id = backend.token_to_id(token)
        backend.post_processor = processors.TemplateProcessing(
            single=f"{token} $A", special_tokens=[(token, token_id)]
        )
        return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token=token)
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def test_pl_shards_restart_within_each_fewshot_set() -> None:
    """R1 P-L shards each format/few-shot group independently in stable order."""
    records = [
        {"concept_id": f"s2-{index}", "format": "repetition", "condition": "diac", "fewshot_set": 2}
        for index in range(4)
    ] + [
        {"concept_id": f"s3-{index}", "format": "repetition", "condition": "diac", "fewshot_set": 3}
        for index in range(4)
    ]
    observed = passes.with_pl_shards(records, shard_prompts=3)
    assert [row["pl_shard"] for row in observed] == [0, 0, 0, 1, 0, 0, 0, 1]


def _synthetic_items(tokenizer: PreTrainedTokenizerFast) -> list[Item]:
    """Create three directions-like records and matched targets for P-D/P-L tests."""
    samples = [
        ("c_vi", "d_vi", "vi", "repetition", " dog", " cat"),
        ("c_en", "d_en", "en", "translation", " house", " mouse"),
        ("c_zh", "d_zh", "zh", "repetition", "猫", "狗"),
    ]
    items: list[Item] = []
    for concept_id, distractor_id, lang, fmt, target, distractor_target in samples:
        variant = "nospace" if lang == "zh" else "released"
        record: dict[str, Any] = {
            "concept_id": concept_id,
            "format": fmt,
            "lang": lang,
            "prompt": "x:",
            "target": " " + target if lang == "zh" else target,
            "split": "fit",
        }
        forms = (PathForm("en", "released", concept_id, " path"),) if lang == "vi" else ()
        items.append(
            Item(
                record=record,
                concept_id=concept_id,
                target_lang=lang,
                target_variant=variant,
                prompt="x:",
                target=target,
                distractor_id=distractor_id,
                distractor_target=distractor_target,
                path_forms=forms,
            )
        )
    return items


def _records(items: list[Item]) -> list[dict[str, Any]]:
    """Return the underlying synthetic prompt records."""
    return [item.record for item in items]


def test_run_pd_shapes_and_last_prompt_position(tiny_bundle: dict[str, Any]) -> None:
    """Capture one float32 residual stack per prompt at its final prompt token."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = _byte_tokenizer()
    items = _synthetic_items(loaded.tokenizer)
    index, residual = run_pd(loaded, _records(items), batch_size=2)
    assert residual.shape == (3, loaded.n_layers + 1, loaded.d_model)
    assert residual.dtype == np.float32
    first = index.iloc[0]
    record = next(
        item.record
        for item in items
        if item.record["format"] == first["format"]
        and item.record["lang"] == first["lang"]
        and item.concept_id == first["concept_id"]
    )
    ids = torch.tensor([loaded.tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]])
    mask = torch.ones_like(ids)
    selected = torch.tensor([[ids.shape[1] - 1]], dtype=torch.long)
    expected = hooks.record(loaded, ids, mask, positions=selected).resid[:, 0, 0].float().numpy()
    row = int(first["array_row"])
    # CPU kernels can differ by a few float32 ulps between batched and
    # one-at-a-time forwards; this remains well below the residual gate.
    np.testing.assert_allclose(residual[row], expected, rtol=1.0e-5, atol=5.0e-6)


def test_run_pd_uses_last_real_token_with_right_padding(tiny_bundle: dict[str, Any]) -> None:
    """Capture each prompt's final actual token, not the right-padding position."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = _byte_tokenizer()
    records = [
        {"concept_id": "short", "format": "repetition", "lang": "vi", "prompt": "x:", "split": "fit"},
        {"concept_id": "long", "format": "repetition", "lang": "vi", "prompt": "a longer x:", "split": "fit"},
    ]
    index, residuals = run_pd(loaded, records, batch_size=2)
    for _, row in index.iterrows():
        record = next(item for item in records if item["concept_id"] == row["concept_id"])
        ids = torch.tensor([loaded.tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]])
        mask = torch.ones_like(ids)
        selected = torch.tensor([[ids.shape[1] - 1]], dtype=torch.long)
        expected = hooks.record(loaded, ids, mask, positions=selected).resid[:, 0, 0].float().numpy()
        np.testing.assert_allclose(residuals[int(row["array_row"])], expected, rtol=1.0e-5, atol=5.0e-6)


def test_run_pd_flores_padding_mean_and_specials(tiny_bundle: dict[str, Any]) -> None:
    """Exclude BOS and padding while matching batched and one-at-a-time FLORES residuals."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = _byte_tokenizer(add_bos=loaded.family != "qwen")
    records = [
        {"lang": "eng_Latn", "sentence_id": "s3", "split": "fit", "sentence": "a longer sentence"},
        {"lang": "fra_Latn", "sentence_id": "s1", "split": "validate", "sentence": "short"},
        {"lang": "eng_Latn", "sentence_id": "s2", "split": "fit", "sentence": "middle text"},
    ]
    batched_index, batched_last, batched_mean = passes.run_pd_flores(
        loaded, records, batch_size=3
    )
    single_rows: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        single_index, single_last, single_mean = passes.run_pd_flores(
            loaded, [record], batch_size=1
        )
        key = (str(single_index.iloc[0]["lang"]), str(single_index.iloc[0]["sentence_id"]))
        single_rows[key] = (single_last[0], single_mean[0])
    expected_specials = 0 if loaded.family == "qwen" else 1
    assert set(batched_index["n_special"].astype(int)) == {expected_specials}
    assert (batched_index["n_tokens"].astype(int) > 0).all()
    for row_number, row in batched_index.iterrows():
        key = (str(row["lang"]), str(row["sentence_id"]))
        expected_last, expected_mean = single_rows[key]
        np.testing.assert_allclose(batched_last[row_number], expected_last, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(batched_mean[row_number], expected_mean, rtol=1.0e-5, atol=1.0e-5)

    record = next(item for item in records if item["sentence_id"] == "s3")
    encoded = loaded.tokenizer(record["sentence"], add_special_tokens=True)["input_ids"]
    specials = loaded.tokenizer.get_special_tokens_mask(encoded, already_has_special_tokens=True)
    content_positions = [i for i, special in enumerate(specials) if not special]
    ids = torch.tensor([encoded], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.tensor([content_positions], dtype=torch.long)
    direct = hooks.record(loaded, ids, mask, positions=positions).resid[:, 0].float().numpy()
    expected_mean = direct.mean(axis=1, dtype=np.float32)
    row_number = int(batched_index.index[batched_index["sentence_id"] == "s3"][0])
    np.testing.assert_allclose(batched_mean[row_number], expected_mean, rtol=1.0e-5, atol=1.0e-5)


def test_pd_storage_unit_skips_complete_before_forward(
    tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse a completed matched unit without entering the model forward method."""
    loaded = tiny_bundle["loaded"]
    loaded.tokenizer = _byte_tokenizer()
    cfg = tiny_bundle["cfg"]
    record = {
        "concept_id": "c1", "format": "repetition", "lang": "vi",
        "prompt": "x:", "split": "fit",
    }
    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)
    first = passes.run_pd_storage_unit(
        loaded, [record], tmp_path, cfg, source="matched", format_name="repetition",
        repo_root=repo_root, entrypoint=None, batch_size=1, run_git=run_git,
    )
    assert first["status"] == "ran"

    def fail_forward(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("forward must not run for a complete P-D unit")

    monkeypatch.setattr(loaded.model, "forward", fail_forward)
    second = passes.run_pd_storage_unit(
        loaded, [record], tmp_path, cfg, source="matched", format_name="repetition",
        repo_root=repo_root, entrypoint=None, batch_size=1, run_git=run_git,
    )
    assert second["status"] == "skipped"


def test_pd_storage_unit_nonfinite_values_are_not_written(
    tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject non-finite residuals before writing any P-D completion files."""
    loaded = tiny_bundle["loaded"]
    cfg = tiny_bundle["cfg"]
    record = {
        "concept_id": "c1", "format": "repetition", "lang": "vi",
        "prompt": "x:", "split": "fit",
    }
    index = pd.DataFrame(
        [{"array_row": 0, "concept_id": "c1", "lang": "vi", "format": "repetition", "split": "fit", "n_tokens": 2, "n_special": 0}]
    )
    monkeypatch.setattr(
        passes,
        "run_pd",
        lambda *_args, **_kwargs: (
            index,
            np.full((1, loaded.n_layers + 1, loaded.d_model), np.nan, dtype=np.float32),
        ),
    )
    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)
    with pytest.raises(FloatingPointError, match="non-finite"):
        passes.run_pd_storage_unit(
            loaded, [record], tmp_path, cfg, source="matched", format_name="repetition",
            repo_root=repo_root, entrypoint=None, batch_size=1, run_git=run_git,
        )
    assert list(tmp_path.iterdir()) == []


def test_run_pl_persistence_and_generation(tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify P-L schemas, counts, path rows, top-k rows, deterministic reruns and resume skips."""
    loaded = tiny_bundle["loaded"]
    model = tiny_bundle["model"]
    cfg = tiny_bundle["cfg"]
    tokenizer = _byte_tokenizer()
    loaded.tokenizer = tokenizer
    items = _synthetic_items(tokenizer)
    generated_calls = 0

    def fake_generate(input_ids: torch.Tensor, *, do_sample: bool, max_new_tokens: int) -> torch.Tensor:
        nonlocal generated_calls
        assert do_sample is False
        assert max_new_tokens == 4
        text = "hello\nrest" if generated_calls % 2 == 0 else "hello -rest"
        generated_calls += 1
        extra = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    lens = __import__("vilens.lens", fromlist=["LensHead"]).LensHead.from_loaded(loaded, "fp32")
    output = run_pl(loaded, lens, items, batch_size=2, top_k=5, gen_max_new_tokens=4)
    assert list(output.long.columns) == LENS_COLUMNS
    assert set(output.long["role"]) == {"correct", "distractor", "path"}
    token_rows = output.long.loc[output.long["metric"] == "model_logprob"]
    assert len(token_rows) == sum(
        len(tokenizer(item.prompt + target, add_special_tokens=True)["input_ids"])
        - len(tokenizer(item.prompt, add_special_tokens=True)["input_ids"])
        for item in items
        for target in (item.target, item.distractor_target)
    )
    assert set(output.long.loc[output.long["role"] == "path", "concept_id"]) == {"c_vi"}
    assert len(output.answer_topk) == 5 * (loaded.n_layers + 1) * len(items)
    assert output.generations.iloc[0]["text"] == "hello"
    assert output.generations.iloc[1]["text"] == "hello"
    assert len(output.long) == output.expected_row_count

    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)
    first_skipped, total = persist_pl_outputs(
        loaded,
        output,
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=2,
        run_git=run_git,
    )
    assert (first_skipped, total) == (0, 2)
    second_skipped, total = persist_pl_outputs(
        loaded,
        output,
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=2,
        run_git=run_git,
    )
    assert (second_skipped, total) == (2, 2)

    generated_calls = 0
    rerun = run_pl(loaded, lens, items, batch_size=2, top_k=5, gen_max_new_tokens=4)
    pd.testing.assert_frame_equal(
        output.long.sort_values([column for column in LENS_COLUMNS if column != "value"]).reset_index(drop=True),
        rerun.long.sort_values([column for column in LENS_COLUMNS if column != "value"]).reset_index(drop=True),
        check_exact=False,
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(output.resid, rerun.resid)
    assert output.generations["text"].tolist() == rerun.generations["text"].tolist()


def test_run_pl_columnar_path_is_bitwise_equal_to_reference(
    tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the optimized sequence-batched output bitwise equal for identical batches."""
    loaded = tiny_bundle["loaded"]
    tokenizer = _byte_tokenizer()
    loaded.tokenizer = tokenizer
    items = _synthetic_items(tokenizer)
    model = tiny_bundle["model"]

    def fake_generate(input_ids: torch.Tensor, *, do_sample: bool, max_new_tokens: int) -> torch.Tensor:
        assert do_sample is False
        assert max_new_tokens == 3
        return torch.cat([input_ids, torch.tensor([[11, 12]], device=input_ids.device)], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    lens = __import__("vilens.lens", fromlist=["LensHead"]).LensHead.from_loaded(loaded, "fp32")
    reference = passes._run_pl_reference(
        loaded, lens, items, batch_size=2, top_k=5, gen_max_new_tokens=3
    )
    optimized = run_pl(
        loaded, lens, items, batch_size=4, top_k=5, gen_max_new_tokens=3
    )
    key_columns = [column for column in LENS_COLUMNS if column != "value"]
    expected_long = reference.long.sort_values(key_columns, kind="stable").reset_index(drop=True)
    actual_long = optimized.long.sort_values(key_columns, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_long, actual_long, check_exact=True, check_dtype=False)
    topk_keys = [column for column in reference.answer_topk.columns if column != "value"]
    pd.testing.assert_frame_equal(
        reference.answer_topk.sort_values(topk_keys, kind="stable").reset_index(drop=True),
        optimized.answer_topk.sort_values(topk_keys, kind="stable").reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    np.testing.assert_array_equal(reference.resid, optimized.resid)
    np.testing.assert_array_equal(reference.attn, optimized.attn)
    np.testing.assert_array_equal(reference.mlp, optimized.mlp)
    assert reference.generations["token_ids"].tolist() == optimized.generations["token_ids"].tolist()


def test_pl_resume_units_have_independent_prompt_group_batches_and_bitwise_outputs(
    tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a unit's batches and outputs identical when other units are present."""
    loaded = tiny_bundle["loaded"]
    model = tiny_bundle["model"]
    cfg = tiny_bundle["cfg"]
    tokenizer = _byte_tokenizer()
    loaded.tokenizer = tokenizer
    base = _synthetic_items(tokenizer)

    def with_unit(item: Item, *, format_name: str, shard: int, prompt: str) -> Item:
        record = {**item.record, "format": format_name, "pl_shard": shard, "prompt": prompt}
        return replace(item, record=record, prompt=prompt)

    repetition = [
        with_unit(base[0], format_name="repetition", shard=0, prompt="x:"),
        with_unit(base[2], format_name="repetition", shard=0, prompt="a longer repetition context:"),
    ]
    translation = [
        with_unit(base[1], format_name="translation", shard=0, prompt="a much longer translation context with padding:"),
    ]
    all_items = [*repetition, *translation]
    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)

    isolated_plans = passes.plan_pl_units(
        loaded, repetition, tmp_path / "isolated", cfg, repo_root=repo_root,
        entrypoint=None, batch_size=3, run_git=run_git,
    )
    combined_plans = passes.plan_pl_units(
        loaded, all_items, tmp_path / "combined", cfg, repo_root=repo_root,
        entrypoint=None, batch_size=3, run_git=run_git,
    )
    isolated_plan = isolated_plans[0]
    combined_plan = next(plan for plan in combined_plans if plan.format_name == "repetition")

    def batch_membership(plan: Any) -> list[list[tuple[str, str, str, int]]]:
        specs = passes._sequence_specs(plan.items, loaded.tokenizer, plan.encodings)
        batches = passes._sequence_batches(specs, 3, keep_prompt_groups=True)
        return [
            [
                (entry[1].concept_id, entry[1].target_lang, entry[2], len(entry[4].joint_ids))
                for entry in batch
            ]
            for batch in batches
        ]

    assert batch_membership(isolated_plan) == batch_membership(combined_plan)
    for batch in passes._sequence_batches(
        passes._sequence_specs(combined_plan.items, loaded.tokenizer, combined_plan.encodings),
        3,
        keep_prompt_groups=True,
    ):
        assert len({passes._prompt_record_key(entry[1]) for entry in batch}) == 1

    def fake_generate(input_ids: torch.Tensor, *, do_sample: bool, max_new_tokens: int) -> torch.Tensor:
        assert do_sample is False
        assert max_new_tokens == 2
        token = torch.full((input_ids.shape[0], 1), 11, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, token], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    lens = __import__("vilens.lens", fromlist=["LensHead"]).LensHead.from_loaded(loaded, "fp32")
    isolated, skipped, total, _rows = run_pl_resume_safe(
        loaded, lens, repetition, tmp_path / "alone", cfg, repo_root=repo_root,
        entrypoint=None, batch_size=3, top_k=5, gen_max_new_tokens=2, run_git=run_git,
    )
    combined, skipped_full, total_full, _rows_full = run_pl_resume_safe(
        loaded, lens, all_items, tmp_path / "together", cfg, repo_root=repo_root,
        entrypoint=None, batch_size=3, top_k=5, gen_max_new_tokens=2, run_git=run_git,
    )
    assert isolated is not None and combined is not None
    assert (skipped, total) == (0, 1)
    assert (skipped_full, total_full) == (0, 2)

    unit_key = {
        "family": loaded.family,
        "pass": "pl",
        "condition": "diac",
        "fewshot_set": "none",
        "format": "repetition",
        "shard": 0,
    }
    isolated_path, _ = __import__("vilens.io", fromlist=["unit_paths"]).unit_paths(
        tmp_path / "alone", unit_key
    )
    combined_path, _ = __import__("vilens.io", fromlist=["unit_paths"]).unit_paths(
        tmp_path / "together", unit_key
    )
    left = pd.read_parquet(isolated_path)
    right = pd.read_parquet(combined_path)
    keys = [column for column in LENS_COLUMNS if column != "value"]
    pd.testing.assert_frame_equal(
        left.sort_values(keys, kind="stable").reset_index(drop=True),
        right.sort_values(keys, kind="stable").reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    for suffix in (".resid.npy", ".attn.npy", ".mlp.npy"):
        np.testing.assert_array_equal(
            np.load(isolated_path.with_suffix(suffix), allow_pickle=False),
            np.load(combined_path.with_suffix(suffix), allow_pickle=False),
        )
    for suffix in (".resid_index.jsonl", ".answer_topk.jsonl", ".generations.jsonl"):
        assert isolated_path.with_suffix(suffix).read_bytes() == combined_path.with_suffix(suffix).read_bytes()


def test_build_items_rejects_manifest_target_disagreement() -> None:
    """Reject directions records whose target differs from the frozen target form."""
    cfg: dict[str, Any] = {}
    record = {
        "concept_id": "c1",
        "format": "translation",
        "lang": "en",
        "prompt": "x:",
        "target": " wrong",
        "split": "fit",
    }
    targets = [
        {"concept_id": "c1", "target_lang": "en", "target_variant": "released", "target": " right"},
        {"concept_id": "d1", "target_lang": "en", "target_variant": "released", "target": " other"},
    ]
    distractors = [
        {
            "family": "llama",
            "target_lang": "en",
            "target_variant": "released",
            "concept_id": "c1",
            "distractor_id": "d1",
        }
    ]
    with pytest.raises(ValueError, match="Directions target mismatch"):
        build_items(cfg, [record], targets, distractors, "llama")


def test_pl_shard_resume_skip_precedes_model_forward(
    tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip a completed prompt shard before forward, and reject changed provenance."""
    loaded = tiny_bundle["loaded"]
    model = tiny_bundle["model"]
    cfg = tiny_bundle["cfg"]
    tokenizer = _byte_tokenizer()
    loaded.tokenizer = tokenizer
    item = _synthetic_items(tokenizer)[0]
    item = Item(
        record={**item.record, "pl_shard": 2},
        concept_id=item.concept_id,
        target_lang=item.target_lang,
        target_variant=item.target_variant,
        prompt=item.prompt,
        target=item.target,
        distractor_id=item.distractor_id,
        distractor_target=item.distractor_target,
        path_forms=item.path_forms,
    )

    def fake_generate(input_ids: torch.Tensor, *, do_sample: bool, max_new_tokens: int) -> torch.Tensor:
        assert do_sample is False
        return torch.cat([input_ids, torch.tensor([[1]], device=input_ids.device)], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    lens = __import__("vilens.lens", fromlist=["LensHead"]).LensHead.from_loaded(loaded, "fp32")
    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)
    first, skipped, total, expected_rows = run_pl_resume_safe(
        loaded,
        lens,
        [item],
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=2,
        top_k=5,
        gen_max_new_tokens=4,
        run_git=run_git,
    )
    assert first is not None and len(first.long) == expected_rows
    assert (skipped, total) == (0, 1)
    plan = passes.plan_pl_units(
        loaded, [item], tmp_path, cfg, repo_root=repo_root, entrypoint=None,
        batch_size=2, run_git=run_git,
    )
    assert plan[0].unit_key["shard"] == 2

    def unexpected_forward(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("model forward ran before the resume skip")

    monkeypatch.setattr(model, "forward", unexpected_forward)
    second, skipped, total, expected_rows = run_pl_resume_safe(
        loaded,
        lens,
        [item],
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=2,
        top_k=5,
        gen_max_new_tokens=4,
        run_git=run_git,
    )
    assert second is None
    assert expected_rows > 0
    assert (skipped, total) == (1, 1)

    original_provenance = passes._unit_provenance

    def changed_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_provenance(*args, **kwargs)
        result["model_revision"] = "changed-provenance"
        return result

    monkeypatch.setattr(passes, "_unit_provenance", changed_provenance)
    with pytest.raises(AssertionError, match="model forward ran"):
        run_pl_resume_safe(
            loaded,
            lens,
            [item],
            tmp_path,
            cfg,
            repo_root=repo_root,
            entrypoint=None,
            batch_size=2,
            top_k=5,
            gen_max_new_tokens=4,
            run_git=run_git,
        )


def test_pd_resume_skip_precedes_model_forward(
    tiny_bundle: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip a complete prompt-only unit before the model's forward method is called."""
    loaded = tiny_bundle["loaded"]
    model = tiny_bundle["model"]
    cfg = tiny_bundle["cfg"]
    loaded.tokenizer = _byte_tokenizer()
    records = [
        {
            "format": "repetition",
            "lang": "vi",
            "concept_id": "tiny-001",
            "split": "fit",
            "condition": "diac",
            "prompt": "a:",
        }
    ]
    repo_root = Path(__file__).resolve().parents[1]
    run_git = capture_run_git(repo_root, cfg)
    _index, _resid, skipped, total, expected_rows = run_pd_resume_safe(
        loaded,
        records,
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=1,
        run_git=run_git,
    )
    assert (skipped, total) == (0, 1)
    assert expected_rows == (loaded.n_layers + 1)

    def unexpected_forward(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("prompt-only model forward ran before the resume skip")

    monkeypatch.setattr(model, "forward", unexpected_forward)
    _index, _resid, skipped, total, _expected_rows = run_pd_resume_safe(
        loaded,
        records,
        tmp_path,
        cfg,
        repo_root=repo_root,
        entrypoint=None,
        batch_size=1,
        run_git=run_git,
    )
    assert (skipped, total) == (1, 1)
