"""Tests for duplicate-key rejection and human-only study selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vilens.config import ConfigError, load_config, study_models

ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_key_fails(tmp_path: Path) -> None:
    """Reject repeated YAML keys before schema validation."""
    path = tmp_path / "duplicate.yaml"
    path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate YAML key"):
        load_config(path)


def test_unknown_top_level_key_fails(tmp_path: Path) -> None:
    """Reject config keys outside the declared top-level schema."""
    path = tmp_path / "unknown.yaml"
    path.write_text((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8") + "extra: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Top-level keys differ"):
        load_config(path)


def test_bad_revision_length_fails(tmp_path: Path) -> None:
    """Reject a model revision that is not 40 hexadecimal characters."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    text = text.replace("093f9f388b31de276ce2de164bdc2081324b9767", "093f9f388b31de276ce2de164bdc2081324b976")
    path = tmp_path / "bad_revision.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="40 hexadecimal characters"):
        load_config(path)


def test_study_models_requires_d10_decision() -> None:
    """Do not select study checkpoints while D10 remains null."""
    cfg = load_config(ROOT / "configs/experiments.yaml")
    cfg["models"]["study_set"] = None
    with pytest.raises(ConfigError, match="D10 undecided: models.study_set is null"):
        study_models(cfg)


def test_e6_closeout_workers_is_validated_as_one(tmp_path: Path) -> None:
    """Closeout defaults to and requires sequential processing."""
    cfg = load_config(ROOT / "configs/experiments.yaml")
    assert cfg["e6"]["closeout_workers"] == 1
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    path = tmp_path / "parallel_closeout.yaml"
    path.write_text(text.replace("  closeout_workers: 1", "  closeout_workers: '1'", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="closeout_workers must be 1"):
        load_config(path)


def test_wrong_model_cache_fails(tmp_path: Path) -> None:
    """Require the explicitly shared read-only model cache."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    text = text.replace("/datastore/shared/models   # read-only shared HF cache (AGENTS.md rule 20)", "/tmp/models")
    path = tmp_path / "wrong_model_cache.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="paths.model_cache"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("window: [0.25, 0.75]", "window: [0.20, 0.80]", "analysis.window"),
        ("gate_min_correct: 124", "gate_min_correct: 123", "analysis.gate_min_correct"),
        ("bootstrap_key_template: \"bootstrap|{item_set}|{model}|{format}\"",
         "bootstrap_key_template: \"bootstrap|{model}|{format}\"", "bootstrap_key_template"),
    ],
)
def test_ops14_analysis_constants_are_validated(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject changes to the human-frozen analysis threshold and stream key."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_analysis.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize("threshold", ["0.0", "1.0"])
def test_cognate_threshold_bounds_fail(tmp_path: Path, threshold: str) -> None:
    """Reject endpoint values for the cognate distance threshold."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    text = text.replace("cognate_max_distance: 0.45", f"cognate_max_distance: {threshold}")
    path = tmp_path / "bad_cognate_threshold.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="cognate_max_distance"):
        load_config(path)


def test_unknown_primary_prompt_stem_fails(tmp_path: Path) -> None:
    """Require each primary prompt value to name one released prompt file."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "repetition_diac: repetition_diac_set1",
        "repetition_diac: repetition_diac_set99",
    )
    path = tmp_path / "bad_primary_prompt.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="prompt_file_paths stems"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  synthetic_min_valid: 99\n", "", "exactly the OPS-28 through OPS-34 settings"),
        ("  maxiter: 500\n", "  maxiter: '500'\n", "e6.maxiter"),
    ],
)
def test_ops29_required_fit_settings_are_validated(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject missing or wrongly typed OPS-29 fit-validity settings."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_ops29.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("    target_abs_bias_max: 0.1\n", "", "e6.recovery"),
        ("    target_abs_bias_max: 0.1\n", "    target_abs_bias_max: '0.1'\n", "target_abs_bias_max"),
    ],
)
def test_ops30_recovery_settings_are_required_and_typed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject missing or wrongly typed OPS-30 recovery fields."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_ops30.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fit_methods", ["lbfgs", "lbfgs_cold", "bfgs", "powell"], "e6.fit_methods"),
        ("boundary_rule", "true", "e6.boundary_rule"),
        ("llf_tie_tol", -1.0, "e6.llf_tie_tol"),
    ],
)
def test_ops31_fit_settings_reject_bad_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Reject malformed OPS-31 method ordering and boundary controls."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["e6"][field] = value
    path = tmp_path / "bad_ops31_fit.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("refits", None, "e6.recovery"),
        ("refits", "20", "e6.recovery.refits"),
        ("min_valid_refits", 18, "e6.recovery.min_valid_refits"),
    ],
)
def test_ops31_recovery_settings_reject_missing_or_bad_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Require typed recovery refit count and minimum valid count."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    if value is None:
        del cfg["e6"]["recovery"][field]
    else:
        cfg["e6"]["recovery"][field] = value
    path = tmp_path / "bad_ops31_recovery.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("weights_dtype: bfloat16", "weights_dtype: float16", "weights_dtype"),
        ("lens_dtype: float32", "lens_dtype: float16", "lens_dtype"),
        ("attn_implementation: eager", "attn_implementation: flash", "attn_implementation"),
        ("padding_side: right", "padding_side: left", "padding_side"),
        ("lens_position_chunk: 256", "lens_position_chunk: 0", "lens_position_chunk"),
        ("gemma: 64   # from E0 job 1484 sweep (peak 14.49 GB)",
         "gemma: 0   # from E0 job 1484 sweep (peak 14.49 GB)", "batch_size"),
        ("llama: LlamaForCausalLM", "llama: null", "architectures"),
        ("llama: {decoder: model, lm_head: lm_head, attn_out: self_attn, mlp_out: mlp}",
         "llama: {decoder: model, lm_head: lm_head, attn_out: self_attn}", "module_paths"),
    ],
)
def test_invalid_inference_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject one invalid value or schema shape for each inference setting kind."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_inference.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("    - tokenizers_version\n", "    - not_a_sidecar_field\n", "provenance_compare"),
        ("    - config_sha256\n", "    - unit_key\n", "provenance_compare"),
        ('git_clean_paths: ["src", "configs", "derived"]', 'git_clean_paths: [src, 4]', "git_clean_paths"),
        ("parquet_compression: zstd", "parquet_compression: gzip", "parquet_compression"),
    ],
)
def test_invalid_runtime_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject invalid provenance fields, Git path lists, and Parquet compression."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_runtime.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_malformed_pretag_exclusion_fails(tmp_path: Path) -> None:
    """Require each D12 exclusion to contain exactly three string fields."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    old = '- {concept_id: "9b523e0c2ba7", format: translation, lang: vi}'
    new = '- {concept_id: "9b523e0c2ba7", format: translation}'
    assert text.count(old) == 1
    path = tmp_path / "bad_exclusion.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"pretag_excluded_directions\[0\]"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_k", 0, r"pl\.top_k"),
        ("gen_max_new_tokens", None, r"pl\.gen_max_new_tokens"),
        ("shard_prompts", -1, r"pl\.shard_prompts"),
        ("batch_size", {"llama": 64, "qwen": 0, "gemma": 64}, r"pl\.batch_size"),
        ("correct_readouts", [["vi"]], r"pl\.correct_readouts"),
        ("expected", {"prompts": 1}, r"pl\.expected"),
    ],
)
def test_invalid_pl_settings_fail(tmp_path: Path, field: str, value: object, message: str) -> None:
    """Reject malformed P-L readout, batching, count, and execution settings."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["pl"][field] = value
    path = tmp_path / "bad_pl.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_invalid_pl_trial_tolerance_fails(tmp_path: Path) -> None:
    """Require a positive rerun tolerance for the trial."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["pl"]["trial"]["rerun_tolerance"] = 0
    path = tmp_path / "bad_pl_trial.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"pl\.trial\.rerun_tolerance"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fewshot_sets", [2, 3]),
        ("correct_readouts", [["vi", "released"]]),
        ("generation", False),
        ("results_dir", "results/pl"),
    ],
)
def test_r1_config_requires_the_fixed_capture_spec(
    tmp_path: Path, field: str, value: object
) -> None:
    """Reject R1 config edits that alter its registered rows, readouts, or output isolation."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["pl"]["r1"][field] = value
    path = tmp_path / "bad_r1.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"pl\.r1"):
        load_config(path)


def test_invalid_steering_settings_fail(tmp_path: Path) -> None:
    """Validate the OPS-27 steering protocol and tie its item cap to the frozen manifest."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["steering"]["max_items"] = 299
    path = tmp_path / "bad_steering.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"steering\.max_items"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("families", ["gemma", "qwen", "llama"]),
        ("sources", ["matched"]),
        ("matched_formats", ["translation"]),
        ("flores_pooling", ["last"]),
        ("expected_matched_records", 0),
        ("expected_flores_records", -1),
        ("expected_flores_validate_ids", 0),
        ("results_dir", None),
    ],
)
def test_invalid_pd_setting_fails(
    tmp_path: Path, field: str, value: object
) -> None:
    """Reject malformed P-D family, source, format, pooling, count, and path settings."""
    cfg = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    cfg["pd"][field] = value
    path = tmp_path / "bad_pd.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=f"pd\\.{field}"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("revision: 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06", "revision: 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06x", "flores.revision"),
        ("languages: [vie_Latn, eng_Latn, cmn_Hans, fra_Latn, ind_Latn]", "languages: [vie_Latn, eng_Latn, cmn_Hans, fra_Latn]", "flores.languages"),
        ("expected_records: 4985", "expected_records: 0", "flores.expected_records"),
    ],
)
def test_invalid_flores_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject malformed revision, language list, and record count for FLORES config."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "bad_flores.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("rng_scheme: \"numpy.random.default_rng([seed, zlib.crc32(key.encode('utf-8'))])\"",
         "rng_scheme: []", "rng_scheme"),
        ("    - [vi, released]\n    - [en, released]",
         "    - [vi]\n    - [en, released]", "distractor_combos"),
        ("e0_sample_size: 50", "e0_sample_size: 0", "e0_sample_size"),
        ("e8_max_items: 300", "e8_max_items: null", "e8_max_items"),
    ],
)
def test_invalid_manifest_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject malformed manifest RNG, combination lists, and size limits."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_manifests.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("main_family: llama", "main_family: unknown", "main_family"),
        ("smoke_families: [gemma, qwen]", "smoke_families: [gemma, unknown]", "smoke_families"),
        ("smoke_concepts: 10", "smoke_concepts: 0", "smoke_concepts"),
        ("batch_size: 16", "batch_size: -1", "batch_size"),
        ("sweep_batch_sizes: [8, 16, 32, 64]", "sweep_batch_sizes: [8, 0]", "sweep_batch_sizes"),
        ("gen_max_new_tokens: 8", "gen_max_new_tokens: null", "gen_max_new_tokens"),
        ("top_k: 20", "top_k: 0", "top_k"),
        ("rerun_tolerance: 1.0e-4", "rerun_tolerance: -1.0", "rerun_tolerance"),
        ("results_dir: results/e0", "results_dir: null", "results_dir"),
        ("rerun_dir: results/e0_rerun", "rerun_dir: ''", "rerun_dir"),
    ],
)
def test_invalid_e0_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject invalid types, values, and paths in the E0 pilot config."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_e0.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("families: [llama, qwen, gemma]", "families: [llama, llama]", "e1.families"),
        ("lens_gate_max_abs_logprob: 1.0e-3", "lens_gate_max_abs_logprob: 0", "lens_gate_max_abs_logprob"),
        ("lens_gate_top1_agreement: 1.0", "lens_gate_top1_agreement: 2.0", "lens_gate_top1_agreement"),
        ("expected_early_blocks: {gemma: [1, 5], qwen: [1, 5], llama: [1, 4]}",
         "expected_early_blocks: {gemma: [1, 5], qwen: [1, 5]}", "expected_early_blocks"),
        ("precision_check_concepts: 50", "precision_check_concepts: 0", "precision_check_concepts"),
        ("competence_formats: [repetition, translation]", "competence_formats: [repetition]", "competence_formats"),
        ("competence_flag_min_translation_vi: 0.20", "competence_flag_min_translation_vi: 1.5", "competence_flag_min_translation_vi"),
        (
            "results_dir: results/e1_r2  # OPS-11 rerun; results/e1 = job 1512 (superseded)",
            "results_dir: ''",
            "results_dir",
        ),
    ],
)
def test_invalid_e1_setting_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Reject malformed E1 family, gate, block, size, format, and output settings."""
    text = (ROOT / "configs/experiments.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "bad_e1.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)
