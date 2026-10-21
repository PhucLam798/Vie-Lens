"""Validated loading for the central experiment configuration."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

_EXPECTED_TOP_LEVEL = {
    "seed", "paths", "data", "prereg", "models", "text", "manifests", "runtime",
    "e0", "e1", "flores", "pd", "pl", "e9_analysis", "r10_analysis", "analysis", "r_battery_e4", "steering", "e6", "e7", "inference", "reference_facts"
}
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_MODEL_FAMILIES = {"gemma", "qwen", "llama"}
_MODULE_PATH_KEYS = {"decoder", "lm_head", "attn_out", "mlp_out"}


class ConfigError(ValueError):
    """Raised when the experiment configuration is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects repeated mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ConfigError(f"Duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _validate_model_group(group: Any, name: str) -> None:
    """Validate the three pinned model entries in a checkpoint group."""
    if not isinstance(group, dict) or set(group) != {"gemma", "qwen", "llama"}:
        raise ConfigError(f"models.{name} must contain gemma, qwen, and llama")
    for model_name, model in group.items():
        if not isinstance(model, dict) or not isinstance(model.get("hf_id"), str):
            raise ConfigError(f"models.{name}.{model_name}.hf_id must be a string")
        revision = model.get("revision")
        if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
            raise ConfigError(f"models.{name}.{model_name}.revision must be 40 hexadecimal characters")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML safely, reject duplicate keys, and validate required config fields."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.load(stream, Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise ConfigError(f"Could not read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("The configuration must be a top-level mapping")
    if set(cfg) != _EXPECTED_TOP_LEVEL:
        missing = sorted(_EXPECTED_TOP_LEVEL - set(cfg))
        extra = sorted(set(cfg) - _EXPECTED_TOP_LEVEL)
        raise ConfigError(f"Top-level keys differ; missing={missing}, extra={extra}")
    manifests = cfg.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {
        "rng_scheme", "distractor_combos", "e0_combos", "e0_sample_size", "e8_max_items"
    }:
        raise ConfigError("manifests must contain exactly the declared manifest settings")
    if not isinstance(manifests.get("rng_scheme"), str) or not manifests["rng_scheme"]:
        raise ConfigError("manifests.rng_scheme must be a non-empty string")
    for field in ("distractor_combos", "e0_combos"):
        combos = manifests.get(field)
        if (
            not isinstance(combos, list)
            or not combos
            or any(
                not isinstance(combo, list)
                or len(combo) != 2
                or any(not isinstance(value, str) for value in combo)
                for combo in combos
            )
        ):
            raise ConfigError(f"manifests.{field} must be a non-empty list of two-string lists")
    for field in ("e0_sample_size", "e8_max_items"):
        if type(manifests.get(field)) is not int or manifests[field] <= 0:
            raise ConfigError(f"manifests.{field} must be a positive integer")
    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "provenance_compare", "git_clean_paths", "parquet_compression"
    }:
        raise ConfigError("runtime must contain exactly provenance_compare, git_clean_paths, and parquet_compression")
    from .provenance import SIDECAR_FIELDS

    provenance_compare = runtime.get("provenance_compare")
    if (
        not isinstance(provenance_compare, list)
        or not provenance_compare
        or any(not isinstance(field, str) for field in provenance_compare)
        or len(set(provenance_compare)) != len(provenance_compare)
        or not set(provenance_compare).issubset(SIDECAR_FIELDS)
    ):
        raise ConfigError("runtime.provenance_compare must be a non-empty list of unique sidecar field names")
    clean_paths = runtime.get("git_clean_paths")
    if not isinstance(clean_paths, list) or any(not isinstance(path, str) for path in clean_paths):
        raise ConfigError("runtime.git_clean_paths must be a list of strings")
    if runtime.get("parquet_compression") not in {"zstd", "snappy"}:
        raise ConfigError("runtime.parquet_compression must be zstd or snappy")
    analysis = cfg.get("analysis")
    analysis_keys = {
        "window", "item_sets", "gate_min_correct", "l_concept_fractions",
        "bootstrap_resamples", "bootstrap_key_template", "la_random_key_template",
    }
    if not isinstance(analysis, dict) or set(analysis) != analysis_keys:
        raise ConfigError("analysis must contain exactly the declared OPS-14 constants")
    window = analysis.get("window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(value) not in {int, float} for value in window)
        or window != [0.25, 0.75]
    ):
        raise ConfigError("analysis.window must be the frozen inclusive window [0.25, 0.75]")
    if analysis.get("item_sets") != ["cf729", "main", "m1_main", "m1_all"]:
        raise ConfigError("analysis.item_sets must be [cf729, main, m1_main, m1_all]")
    if type(analysis.get("gate_min_correct")) is not int or analysis["gate_min_correct"] != 124:
        raise ConfigError("analysis.gate_min_correct must be the frozen value 124")
    fractions = analysis.get("l_concept_fractions")
    if (
        not isinstance(fractions, list)
        or any(type(value) not in {int, float} for value in fractions)
        or fractions != [0.05, 0.1, 0.2]
    ):
        raise ConfigError("analysis.l_concept_fractions must be [0.05, 0.1, 0.2]")
    if type(analysis.get("bootstrap_resamples")) is not int or analysis["bootstrap_resamples"] != 1000:
        raise ConfigError("analysis.bootstrap_resamples must be 1000")
    if analysis.get("bootstrap_key_template") != "bootstrap|{item_set}|{model}|{format}":
        raise ConfigError("analysis.bootstrap_key_template is not the OPS-14 key template")
    if analysis.get("la_random_key_template") != "la_random|{family}|{format}|{layer}":
        raise ConfigError("analysis.la_random_key_template is not the OPS-14 key template")
    flores = cfg.get("flores")
    flores_keys = {
        "repo", "revision", "split", "languages", "hub_cache", "datasets_cache",
        "parquet_dir", "prep_download_script", "prep_directions_script", "prep_sources",
        "output", "expected_ids", "expected_validate_ids", "expected_records",
    }
    if not isinstance(flores, dict) or set(flores) != flores_keys:
        raise ConfigError("flores must contain exactly the declared download and output settings")
    for field in (
        "repo", "split", "hub_cache", "datasets_cache", "parquet_dir",
        "prep_download_script", "prep_directions_script", "prep_sources", "output",
    ):
        if not isinstance(flores.get(field), str) or not flores[field]:
            raise ConfigError(f"flores.{field} must be a non-empty string")
    if not isinstance(flores.get("revision"), str) or _REVISION.fullmatch(flores["revision"]) is None:
        raise ConfigError("flores.revision must be 40 hexadecimal characters")
    languages = flores.get("languages")
    if (
        not isinstance(languages, list)
        or len(languages) != 5
        or any(not isinstance(language, str) or not language for language in languages)
        or len(set(languages)) != 5
    ):
        raise ConfigError("flores.languages must contain five unique language codes")
    for field in ("expected_ids", "expected_validate_ids", "expected_records"):
        if type(flores.get(field)) is not int or flores[field] <= 0:
            raise ConfigError(f"flores.{field} must be a positive integer")
    pd = cfg.get("pd")
    pd_keys = {
        "families", "sources", "matched_formats", "flores_pooling",
        "expected_matched_records", "expected_flores_records",
        "expected_flores_validate_ids", "results_dir",
    }
    if not isinstance(pd, dict) or set(pd) != pd_keys:
        raise ConfigError("pd must contain exactly the declared directions pass settings")
    for field, expected in (
        ("families", ["llama", "qwen", "gemma"]),
        ("sources", ["matched", "flores"]),
        ("matched_formats", ["repetition", "translation"]),
        ("flores_pooling", ["last", "mean"]),
    ):
        if pd.get(field) != expected:
            raise ConfigError(f"pd.{field} must equal {expected}")
    for field in ("expected_matched_records", "expected_flores_records", "expected_flores_validate_ids"):
        if type(pd.get(field)) is not int or pd[field] <= 0:
            raise ConfigError(f"pd.{field} must be a positive integer")
    if not isinstance(pd.get("results_dir"), str) or not pd["results_dir"]:
        raise ConfigError("pd.results_dir must be a non-empty string")
    pl = cfg.get("pl")
    pl_keys = {
        "families", "primary", "correct_readouts", "translation_extra_readouts",
        "path_langs", "top_k", "gen_max_new_tokens", "shard_prompts", "batch_size",
        "expected", "results_dir", "trial", "r1", "e9",
    }
    if not isinstance(pl, dict) or set(pl) != pl_keys:
        raise ConfigError("pl must contain exactly the declared P-L settings")
    if pl.get("families") != ["llama", "qwen", "gemma"]:
        raise ConfigError("pl.families must be [llama, qwen, gemma]")
    primary = pl.get("primary")
    if (
        not isinstance(primary, dict)
        or set(primary) != {"condition", "fewshot_set", "files"}
        or primary.get("condition") != "diac"
        or type(primary.get("fewshot_set")) is not int
        or primary["fewshot_set"] != 1
        or primary.get("files") != ["repetition_diac_set1", "translation_diac_set1", "cloze_diac_set1"]
    ):
        raise ConfigError("pl.primary must select diac set 1 and the three declared prompt files")

    def _readout_pairs(value: Any, name: str) -> None:
        """Validate configured language/variant pairs."""
        if (
            not isinstance(value, list)
            or not value
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(part, str) or not part for part in pair)
                for pair in value
            )
        ):
            raise ConfigError(f"pl.{name} must be a non-empty list of two-string lists")

    _readout_pairs(pl.get("correct_readouts"), "correct_readouts")
    if pl["correct_readouts"] != [
        ["vi", "released"], ["en", "released"], ["fr", "released"], ["id", "released"],
        ["zh", "nospace"], ["zh", "released"], ["vi", "nodiac"],
    ]:
        raise ConfigError("pl.correct_readouts differs from the registered P-L readouts")
    _readout_pairs(pl.get("translation_extra_readouts"), "translation_extra_readouts")
    if pl["translation_extra_readouts"] != [["ru", "released"]]:
        raise ConfigError("pl.translation_extra_readouts must contain ru/released")
    _readout_pairs(pl.get("path_langs"), "path_langs")
    if pl["path_langs"] != [["en", "released"], ["zh", "nospace"], ["fr", "released"], ["id", "released"]]:
        raise ConfigError("pl.path_langs differs from the registered path readouts")
    for field in ("top_k", "gen_max_new_tokens", "shard_prompts"):
        if type(pl.get(field)) is not int or pl[field] <= 0:
            raise ConfigError(f"pl.{field} must be a positive integer")
    pl_batch_sizes = pl.get("batch_size")
    if (
        not isinstance(pl_batch_sizes, dict)
        or set(pl_batch_sizes) != _MODEL_FAMILIES
        or any(type(value) is not int or value <= 0 for value in pl_batch_sizes.values())
    ):
        raise ConfigError("pl.batch_size must contain positive sizes for gemma, qwen, and llama")
    expected_pl = pl.get("expected")
    if not isinstance(expected_pl, dict) or set(expected_pl) != {
        "prompts", "sequences", "repetition", "translation", "cloze"
    } or any(type(value) is not int or value <= 0 for value in expected_pl.values()):
        raise ConfigError("pl.expected must contain positive prompt, sequence, and per-format counts")
    if not isinstance(pl.get("results_dir"), str) or not pl["results_dir"]:
        raise ConfigError("pl.results_dir must be a non-empty string")
    r1 = pl.get("r1")
    r1_keys = {
        "condition", "fewshot_sets", "files", "item_set", "correct_readouts",
        "translation_extra_readouts", "path_langs", "generation", "expected", "results_dir",
    }
    expected_r1_files = [
        *(f"repetition_diac_set{set_id}" for set_id in range(2, 7)),
        *(f"translation_diac_set{set_id}" for set_id in range(2, 7)),
    ]
    if (
        not isinstance(r1, dict)
        or set(r1) != r1_keys
        or r1.get("condition") != "diac"
        or r1.get("fewshot_sets") != [2, 3, 4, 5, 6]
        or r1.get("files") != expected_r1_files
        or r1.get("item_set") != "cf729"
        or r1.get("correct_readouts") != [
            ["vi", "released"], ["en", "released"], ["zh", "nospace"],
            ["fr", "released"], ["id", "released"],
        ]
        or r1.get("translation_extra_readouts") != []
        or r1.get("path_langs") != []
        or type(r1.get("generation")) is not bool
        or r1.get("generation") is not True
        or r1.get("expected") != {
            "repetition": 729,
            "translation": 721,
            "prompts_per_set": 1450,
            "sequences_per_prompt": 10,
        }
        or not isinstance(r1.get("results_dir"), str)
        or not r1["results_dir"]
        or r1["results_dir"] == pl["results_dir"]
    ):
        raise ConfigError("pl.r1 must contain the fixed cf729 few-shot sets 2–6 capture settings")
    e9 = pl.get("e9")
    if (
        not isinstance(e9, dict)
        or set(e9) != {
            "condition", "fewshot_set", "files", "scope", "exclude_collapsed",
            "correct_readouts", "translation_extra_readouts", "path_langs", "generation",
            "expected", "results_dir",
        }
        or e9.get("condition") != "nodiac"
        or type(e9.get("fewshot_set")) is not int
        or e9.get("fewshot_set") != 1
        or e9.get("files") != [
            "repetition_nodiac_set1", "translation_nodiac_set1", "cloze_nodiac_set1"
        ]
        or e9.get("scope") != "noncollapsed_main"
        or e9.get("exclude_collapsed") is not True
        or e9.get("correct_readouts") != [
            ["vi", "nodiac"], ["en", "released"], ["fr", "released"],
            ["id", "released"], ["zh", "nospace"],
        ]
        or e9.get("translation_extra_readouts") != []
        or e9.get("path_langs") != []
        or e9.get("generation") is not True
        or e9.get("expected") != {
            "repetition": 1231, "translation": 1221, "cloze": 29,
            "sequences": 24810, "units_per_family": 11,
        }
        or not isinstance(e9.get("results_dir"), str)
        or not e9["results_dir"]
        or e9["results_dir"] in {pl["results_dir"], r1["results_dir"]}
    ):
        raise ConfigError("pl.e9 must match the fixed OPS-45 nodiac capture settings")
    e9_analysis = cfg.get("e9_analysis")
    if not isinstance(e9_analysis, dict) or e9_analysis != {
        "report": "reports/E9.md", "csv": "reports/E9.csv"
    }:
        raise ConfigError("e9_analysis must name the registered E9 report and CSV")
    r10_analysis = cfg.get("r10_analysis")
    if not isinstance(r10_analysis, dict) or r10_analysis != {
        "report": "reports/R10.md", "csv": "reports/R10.csv"
    }:
        raise ConfigError("r10_analysis must name the registered R10 report and CSV")
    r_battery_e4 = cfg.get("r_battery_e4")
    expected_r_battery_variants = [
        {"name": "R4_raw_s", "code": "R4", "scope": "cf729", "filter": "none", "score_source": "raw"},
        {"name": "R6_bt_pass_strict_v1", "code": "R6", "scope": "cf729", "filter": "bt_pass_strict_v1", "score_source": "delta"},
        {"name": "R6_external_attested", "code": "R6", "scope": "cf729", "filter": "external_attested", "score_source": "delta"},
        {"name": "R6_all_three_nllb", "code": "R6", "scope": "cf729", "filter": "all_three_nllb", "score_source": "delta"},
        {"name": "R6_min_surface_dist_ge_060", "code": "R6", "scope": "cf729", "filter": "min_surface_dist_ge_060", "score_source": "delta"},
        {"name": "R7_competent", "code": "R7", "scope": "cf729", "filter": "competent_vi", "score_source": "delta"},
        {"name": "R9_main", "code": "R9", "scope": "main", "filter": "none", "score_source": "delta"},
        {"name": "R11_excluded", "code": "R11", "scope": "cf729", "filter": "exclude_r11_ids", "score_source": "delta"},
    ]
    if (
        not isinstance(r_battery_e4, dict)
        or set(r_battery_e4) != {"variants", "outputs"}
        or r_battery_e4.get("variants") != expected_r_battery_variants
        or r_battery_e4.get("outputs") != {
            "report": "reports/R_battery_e4.md", "csv": "reports/R_battery_e4.csv"
        }
    ):
        raise ConfigError("r_battery_e4 must match the fixed OPS-44 variant list and output paths")
    trial = pl.get("trial")
    if not isinstance(trial, dict) or set(trial) != {
        "results_dir", "rerun_dir", "rerun_tolerance"
    }:
        raise ConfigError("pl.trial must contain the declared trial settings")
    for field in ("results_dir", "rerun_dir"):
        if not isinstance(trial.get(field), str) or not trial[field]:
            raise ConfigError(f"pl.trial.{field} must be a non-empty string")
    tolerance = trial.get("rerun_tolerance")
    if type(tolerance) not in (float, int) or tolerance <= 0:
        raise ConfigError("pl.trial.rerun_tolerance must be positive")
    steering = cfg.get("steering")
    steering_keys = {
        "k_grid", "random_control_k", "layers", "arms", "gen_max_new_tokens",
        "stop_strings", "max_items", "batch_size", "profile", "bootstrap_resamples",
        "bootstrap_key_template", "random_key_template", "results_dir", "trial_results_dir",
    }
    if not isinstance(steering, dict) or set(steering) != steering_keys:
        raise ConfigError("steering must contain exactly the declared E8 settings")
    if steering.get("k_grid") != [0.5, 1.0, 2.0]:
        raise ConfigError("steering.k_grid must be [0.5, 1.0, 2.0]")
    if type(steering.get("random_control_k")) not in {int, float} or steering["random_control_k"] != 1.0:
        raise ConfigError("steering.random_control_k must be 1.0")
    if steering.get("layers") != "all":
        raise ConfigError('steering.layers must be "all"')
    if steering.get("arms") != {"translation": ["generation", "delta"], "repetition": ["delta"]}:
        raise ConfigError("steering.arms differs from OPS-27")
    if steering.get("gen_max_new_tokens") != 8 or type(steering.get("gen_max_new_tokens")) is not int:
        raise ConfigError("steering.gen_max_new_tokens must be 8")
    if steering.get("stop_strings") != ["\n", " -"]:
        raise ConfigError('steering.stop_strings must be ["\\n", " -"]')
    if steering.get("max_items") != manifests["e8_max_items"] or type(steering.get("max_items")) is not int:
        raise ConfigError("steering.max_items must equal manifests.e8_max_items")
    steer_batches = steering.get("batch_size")
    if not isinstance(steer_batches, dict) or set(steer_batches) != _MODEL_FAMILIES:
        raise ConfigError("steering.batch_size must contain gemma, qwen, and llama")
    if any(type(value) is not int or value <= 0 for value in steer_batches.values()):
        raise ConfigError("steering.batch_size values must be positive integers")
    if steering.get("profile") != "normal":
        raise ConfigError("steering.profile must be normal")
    if steering.get("bootstrap_resamples") != 1000 or type(steering.get("bootstrap_resamples")) is not int:
        raise ConfigError("steering.bootstrap_resamples must be 1000")
    if steering.get("bootstrap_key_template") != "bootstrap|e8|{family}|{format}":
        raise ConfigError("steering.bootstrap_key_template differs from OPS-27")
    if steering.get("random_key_template") != "steer_random|{family}|{layer}":
        raise ConfigError("steering.random_key_template differs from OPS-27")
    for field in ("results_dir", "trial_results_dir"):
        if not isinstance(steering.get(field), str) or not steering[field]:
            raise ConfigError(f"steering.{field} must be a non-empty string")
    e6 = cfg.get("e6")
    e6_keys = {
        "strata", "window", "bootstrap_resamples", "dzh_key_template", "model_key_template",
        "matched_key_template", "match_stream", "complete_case_counts", "caliper", "reference_levels", "spline_df",
        "workers", "bootstrap_chunk", "max_projected_hours", "r11_concept_ids", "sensitivity_variants", "output_dir",
        "synthetic_dir", "closeout_dir", "moment_refit_json", "closeout_workers", "report", "outputs", "fit_methods", "maxiter", "min_valid_draws",
        "synthetic_validation_draws", "synthetic_min_valid", "synthetic_sd", "recovery",
        "boundary_rule", "llf_tie_tol",
    }
    if not isinstance(e6, dict) or set(e6) != e6_keys:
        raise ConfigError("e6 must contain exactly the OPS-28 through OPS-34 settings")
    if e6.get("strata") != ["sino", "nonsino"]:
        raise ConfigError("e6.strata must be [sino, nonsino]")
    if e6.get("window") != [0.25, 0.75]:
        raise ConfigError("e6.window must be the inclusive [0.25, 0.75]")
    if type(e6.get("bootstrap_resamples")) is not int or e6["bootstrap_resamples"] != 1000:
        raise ConfigError("e6.bootstrap_resamples must be 1000")
    expected_e6_keys = {
        "dzh_key_template": "bootstrap|h3|{model}|pooled",
        "model_key_template": "bootstrap|h3cc|{model}|pooled",
        "matched_key_template": "bootstrap|h3m|{model}|pooled",
        "match_stream": "e6_match",
    }
    for field, expected in expected_e6_keys.items():
        if e6.get(field) != expected:
            raise ConfigError(f"e6.{field} must equal {expected!r}")
    if e6.get("complete_case_counts") != {"sino": 469, "nonsino": 688}:
        raise ConfigError("e6.complete_case_counts must be sino=469 and nonsino=688")
    if type(e6.get("caliper")) not in {int, float} or e6["caliper"] != 0.2:
        raise ConfigError("e6.caliper must be 0.2")
    if e6.get("reference_levels") != {"language": "en", "stratum": "nonsino"}:
        raise ConfigError("e6.reference_levels must use en and nonsino")
    if type(e6.get("spline_df")) is not int or e6["spline_df"] != 5:
        raise ConfigError("e6.spline_df must be 5")
    if type(e6.get("workers")) is not int or e6["workers"] != 8:
        raise ConfigError("e6.workers must be 8")
    if type(e6.get("closeout_workers")) is not int or e6["closeout_workers"] != 1:
        raise ConfigError("e6.closeout_workers must be 1 for sequential closeout")
    if type(e6.get("bootstrap_chunk")) is not int or e6["bootstrap_chunk"] != 100:
        raise ConfigError("e6.bootstrap_chunk must be 100")
    if type(e6.get("max_projected_hours")) is not int or e6["max_projected_hours"] != 48:
        raise ConfigError("e6.max_projected_hours must be 48")
    if e6.get("fit_methods") != ["lbfgs", "lbfgs_mom", "lbfgs_cold", "bfgs", "powell"]:
        raise ConfigError("e6.fit_methods must be [lbfgs, lbfgs_mom, lbfgs_cold, bfgs, powell]")
    if type(e6.get("boundary_rule")) is not bool or e6["boundary_rule"] is not True:
        raise ConfigError("e6.boundary_rule must be true")
    if type(e6.get("llf_tie_tol")) not in {int, float} or e6["llf_tie_tol"] != 1.0e-6:
        raise ConfigError("e6.llf_tie_tol must be 1.0e-6")
    if type(e6.get("maxiter")) is not int or e6["maxiter"] != 500:
        raise ConfigError("e6.maxiter must be the positive integer 500")
    if type(e6.get("min_valid_draws")) is not int or e6["min_valid_draws"] != 950:
        raise ConfigError("e6.min_valid_draws must be 950")
    if type(e6.get("synthetic_validation_draws")) is not int or e6["synthetic_validation_draws"] != 100:
        raise ConfigError("e6.synthetic_validation_draws must be 100")
    if type(e6.get("synthetic_min_valid")) is not int or e6["synthetic_min_valid"] != 99:
        raise ConfigError("e6.synthetic_min_valid must be 99")
    synthetic_sd = e6.get("synthetic_sd")
    if (
        not isinstance(synthetic_sd, dict)
        or set(synthetic_sd) != {"concept", "concept_language", "residual"}
        or any(type(value) not in {int, float} or value <= 0 for value in synthetic_sd.values())
        or synthetic_sd != {"concept": 1.0, "concept_language": 1.0, "residual": 2.0}
    ):
        raise ConfigError("e6.synthetic_sd must be positive numeric concept=1.0, concept_language=1.0, residual=2.0")
    recovery = e6.get("recovery")
    if not isinstance(recovery, dict) or set(recovery) != {
        "planted_coefficients", "concept_sd_ratio_bounds", "target_wald_se_multiple", "target_abs_bias_max",
        "refits", "min_valid_refits",
    }:
        raise ConfigError("e6.recovery must contain exactly the OPS-30 and OPS-31 recovery settings")
    planted_coefficients = recovery.get("planted_coefficients")
    expected_planted_coefficients = {
        "language_en": 0.35,
        "depth": 0.16,
        "task_format_translation": 0.03,
        "stratum_sino_language_zh": 0.5,
    }
    if (
        not isinstance(planted_coefficients, dict)
        or set(planted_coefficients) != set(expected_planted_coefficients)
        or any(type(value) not in {int, float} for value in planted_coefficients.values())
        or planted_coefficients != expected_planted_coefficients
    ):
        raise ConfigError("e6.recovery.planted_coefficients must state the OPS-30 fixed effects")
    bounds = recovery.get("concept_sd_ratio_bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(value) not in {int, float} for value in bounds)
        or bounds != [0.5, 1.5]
    ):
        raise ConfigError("e6.recovery.concept_sd_ratio_bounds must be [0.5, 1.5]")
    if type(recovery.get("target_wald_se_multiple")) not in {int, float} or recovery["target_wald_se_multiple"] != 3.0:
        raise ConfigError("e6.recovery.target_wald_se_multiple must be 3.0")
    if type(recovery.get("target_abs_bias_max")) not in {int, float} or recovery["target_abs_bias_max"] != 0.1:
        raise ConfigError("e6.recovery.target_abs_bias_max must be 0.1")
    if type(recovery.get("refits")) is not int or recovery["refits"] != 20:
        raise ConfigError("e6.recovery.refits must be 20")
    if type(recovery.get("min_valid_refits")) is not int or recovery["min_valid_refits"] != 19:
        raise ConfigError("e6.recovery.min_valid_refits must be 19")
    if e6.get("r11_concept_ids") != [
        "151a3fe224ea", "17781bd98b3c", "922cd1167021", "c9584b841817",
        "c505c130104c", "f5411853b246",
    ]:
        raise ConfigError("e6.r11_concept_ids differs from OPS-28")
    if e6.get("sensitivity_variants") != [
        "zipf_vi", "zipf_zh", "native_strict", "bt_pass_strict_v1", "external_attested",
        "all_three_nllb", "min_surface_dist_ge_060", "R11_excluded",
    ]:
        raise ConfigError("e6.sensitivity_variants must list the eight human-named OPS-28 variants")
    for field in ("output_dir", "closeout_dir", "moment_refit_json", "synthetic_dir", "report"):
        if not isinstance(e6.get(field), str) or not e6[field]:
            raise ConfigError(f"e6.{field} must be a non-empty path")
    e6_outputs = e6.get("outputs")
    if not isinstance(e6_outputs, dict) or set(e6_outputs) != {
        "dzh", "model_terms", "holm", "matched", "sensitivity", "matched_pairs", "balance", "closeout", "moment_refit"
    } or any(not isinstance(value, str) or not value.startswith("reports/E6_") for value in e6_outputs.values()):
        raise ConfigError("e6.outputs must contain the declared reports/E6_*.csv paths")
    e7 = cfg.get("e7")
    e7_keys = {
        "formats", "languages", "primary_item_set", "sensitivity_item_set", "prompt_condition",
        "fewshot_set", "trial_items_per_cell", "unit_chunk_prompts", "batch_size", "expected_prompt_counts",
        "primary_metric", "vietnamese_target_language", "profile", "smoke_profile", "bootstrap_resamples",
        "bootstrap_key_template", "head_rank_n", "share_epsilon", "check_head_median_relative",
        "check_head_p99_relative", "check_residual_median_relative", "check_residual_p99_relative",
        "check_residual_model_order_max", "check_residual_replay_max_relative", "replay",
        "check_pl_median_relative", "check_direction_relative", "max_pd_e7_overlap", "p_l_results_dir",
        "p_d_results_dir", "e8_results_dir", "analysis_results_dir", "results_dir", "trial_results_dir",
        "smoke_results_dir",
        "report", "outputs",
    }
    if not isinstance(e7, dict) or set(e7) != e7_keys:
        raise ConfigError("e7 must contain exactly the declared OPS-37/OPS-38 settings")
    if e7.get("formats") != ["repetition", "translation"]:
        raise ConfigError("e7.formats must be [repetition, translation]")
    if e7.get("languages") != ["vi", "en", "zh", "fr", "id"]:
        raise ConfigError("e7.languages must follow the frozen five-language order")
    if e7.get("primary_item_set") != "cf729" or e7.get("sensitivity_item_set") != "main":
        raise ConfigError("e7 item sets must be cf729 primary and main sensitivity")
    if e7.get("prompt_condition") != "diac" or type(e7.get("fewshot_set")) is not int or e7["fewshot_set"] != 1:
        raise ConfigError("e7 prompts must use condition diac and fewshot_set 1")
    if type(e7.get("trial_items_per_cell")) is not int or e7["trial_items_per_cell"] != 32:
        raise ConfigError("e7.trial_items_per_cell must be 32")
    if type(e7.get("unit_chunk_prompts")) is not int or e7["unit_chunk_prompts"] != 32:
        raise ConfigError("e7.unit_chunk_prompts must be 32")
    expected_e7_counts = {
        "main": {"repetition": 1535, "translation": 1524},
        "cf729": {"repetition": 729, "translation": 721},
    }
    if e7.get("expected_prompt_counts") != expected_e7_counts:
        raise ConfigError("e7.expected_prompt_counts differs from OPS-37 reference sets")
    if e7.get("primary_metric") != "s" or e7.get("vietnamese_target_language") != "vi":
        raise ConfigError("e7 band curve must use primary metric s and Vietnamese target rows")
    e7_batches = e7.get("batch_size")
    if not isinstance(e7_batches, dict) or set(e7_batches) != _MODEL_FAMILIES or any(
        type(value) is not int or value <= 0 for value in e7_batches.values()
    ):
        raise ConfigError("e7.batch_size must contain positive values for all three families")
    if e7.get("profile") != "normal":
        raise ConfigError("e7.profile must be normal for P-A")
    if e7.get("smoke_profile") != "small":
        raise ConfigError("e7.smoke_profile must be small for the directions smoke")
    replay = e7.get("replay")
    replay_expected = {
        "p_a": {"source": "pl", "batch_size_config": "pl.batch_size", "divisor": 2},
        "smoke": {"source": "pd", "batch_size_config": "inference.batch_size", "divisor": 1},
    }
    if replay != replay_expected:
        raise ConfigError("e7.replay must reference P-L batches for P-A and P-D batches for smoke")
    if type(e7.get("bootstrap_resamples")) is not int or e7["bootstrap_resamples"] != 1000:
        raise ConfigError("e7.bootstrap_resamples must be 1000")
    if e7.get("bootstrap_key_template") != "bootstrap|e7|{model}|{format}":
        raise ConfigError("e7.bootstrap_key_template differs from OPS-37")
    if type(e7.get("head_rank_n")) is not int or e7["head_rank_n"] != 10:
        raise ConfigError("e7.head_rank_n must be 10")
    numeric_e7 = (
        "share_epsilon", "check_head_median_relative", "check_head_p99_relative",
        "check_residual_median_relative", "check_residual_p99_relative", "check_pl_median_relative",
        "check_direction_relative", "check_residual_replay_max_relative",
    )
    for field in numeric_e7:
        value = e7.get(field)
        if type(value) not in {int, float} or value <= 0:
            raise ConfigError(f"e7.{field} must be a positive number")
    expected_check_values = {
        "share_epsilon": 1.0e-8,
        "check_head_median_relative": 1.0e-2,
        "check_head_p99_relative": 5.0e-2,
        "check_residual_median_relative": 1.0e-2,
        "check_residual_p99_relative": 5.0e-2,
        "check_pl_median_relative": 1.0e-2,
        "check_direction_relative": 1.0e-5,
        "check_residual_replay_max_relative": 1.0e-6,
    }
    if any(float(e7[field]) != expected for field, expected in expected_check_values.items()):
        raise ConfigError("e7 validity-check thresholds must match OPS-37")
    if type(e7.get("check_residual_model_order_max")) not in {int, float} or float(
        e7["check_residual_model_order_max"]
    ) != 1.0e-4:
        raise ConfigError("e7.check_residual_model_order_max must match OPS-38")
    if float(e7["check_residual_replay_max_relative"]) != 1.0e-6:
        raise ConfigError("e7.check_residual_replay_max_relative must match OPS-39")
    if type(e7.get("max_pd_e7_overlap")) is not int or e7["max_pd_e7_overlap"] != 0:
        raise ConfigError("e7.max_pd_e7_overlap must be 0")
    for field in (
        "p_l_results_dir", "p_d_results_dir", "e8_results_dir", "analysis_results_dir",
        "results_dir", "trial_results_dir", "smoke_results_dir", "report",
    ):
        if not isinstance(e7.get(field), str) or not e7[field]:
            raise ConfigError(f"e7.{field} must be a non-empty path")
    e7_outputs = e7.get("outputs")
    if not isinstance(e7_outputs, dict) or set(e7_outputs) != {"layers", "band", "heads", "heads_all"} or any(
        not isinstance(path, str) or not path.startswith("reports/E7_") for path in e7_outputs.values()
    ):
        raise ConfigError("e7.outputs must map layers, band, heads, and heads_all to reports/E7_*.csv")
    inference = cfg.get("inference")
    if not isinstance(inference, dict):
        raise ConfigError("inference must be a mapping")
    inference_keys = {
        "weights_dtype", "lens_dtype", "attn_implementation", "padding_side",
        "lens_position_chunk", "batch_size", "architectures", "module_paths",
    }
    if set(inference) != inference_keys:
        raise ConfigError("inference must contain exactly the declared inference settings")
    if inference.get("weights_dtype") != "bfloat16":
        raise ConfigError("inference.weights_dtype must be bfloat16")
    if inference.get("lens_dtype") != "float32":
        raise ConfigError("inference.lens_dtype must be float32")
    if inference.get("attn_implementation") not in {"eager", "sdpa"}:
        raise ConfigError("inference.attn_implementation must be eager or sdpa")
    if inference.get("padding_side") != "right":
        raise ConfigError("inference.padding_side must be right")
    if type(inference.get("lens_position_chunk")) is not int or inference["lens_position_chunk"] <= 0:
        raise ConfigError("inference.lens_position_chunk must be a positive integer")
    batch_sizes = inference.get("batch_size")
    if not isinstance(batch_sizes, dict) or set(batch_sizes) != _MODEL_FAMILIES:
        raise ConfigError("inference.batch_size must contain gemma, qwen, and llama")
    if any(value is not None and (type(value) is not int or value <= 0) for value in batch_sizes.values()):
        raise ConfigError("inference.batch_size values must be null or positive integers")
    architectures = inference.get("architectures")
    if (
        not isinstance(architectures, dict)
        or set(architectures) != _MODEL_FAMILIES
        or any(not isinstance(value, str) or not value for value in architectures.values())
    ):
        raise ConfigError("inference.architectures must map gemma, qwen, and llama to class names")
    module_paths = inference.get("module_paths")
    if not isinstance(module_paths, dict) or set(module_paths) != _MODEL_FAMILIES:
        raise ConfigError("inference.module_paths must contain gemma, qwen, and llama")
    for family, paths in module_paths.items():
        if (
            not isinstance(paths, dict)
            or set(paths) != _MODULE_PATH_KEYS
            or any(not isinstance(value, str) or not value for value in paths.values())
        ):
            raise ConfigError(
                f"inference.module_paths.{family} must contain decoder, lm_head, attn_out, and mlp_out paths"
            )
    if type(cfg.get("seed")) is not int:
        raise ConfigError("seed must be an integer")
    prereg = cfg.get("prereg")
    if not isinstance(prereg, dict):
        raise ConfigError("prereg must be a mapping")
    if prereg.get("analysis_tag") is not None and not isinstance(prereg["analysis_tag"], str):
        raise ConfigError("prereg.analysis_tag must be a string or null")
    if type(prereg.get("test_items_unlocked")) is not bool:
        raise ConfigError("prereg.test_items_unlocked must be a boolean")
    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ConfigError("models must be a mapping")
    if models.get("study_set") not in (None, "registered", "base"):
        raise ConfigError('models.study_set must be null, "registered", or "base"')
    _validate_model_group(models.get("registered"), "registered")
    _validate_model_group(models.get("base"), "base")
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or paths.get("model_cache") != "/datastore/shared/models":
        raise ConfigError("paths.model_cache must be /datastore/shared/models")
    data = cfg.get("data")
    primary = data.get("primary_prompt_files") if isinstance(data, dict) else None
    primary_keys = {
        "repetition_diac", "translation_diac", "repetition_nodiac",
        "translation_nodiac", "cloze_diac", "cloze_nodiac",
    }
    if not isinstance(primary, dict) or set(primary) != primary_keys:
        raise ConfigError("data.primary_prompt_files must contain the six primary prompt keys")
    from .data import prompt_file_paths

    valid_stems = set(prompt_file_paths(cfg))
    invalid_stems = {value for value in primary.values() if not isinstance(value, str) or value not in valid_stems}
    if invalid_stems:
        raise ConfigError(f"data.primary_prompt_files values must be prompt_file_paths stems: {sorted(map(str, invalid_stems))}")
    e0 = cfg.get("e0")
    e0_keys = {
        "main_family", "smoke_families", "smoke_concepts", "batch_size", "sweep_batch_sizes",
        "gen_max_new_tokens", "top_k", "rerun_tolerance", "results_dir", "rerun_dir",
    }
    if not isinstance(e0, dict) or set(e0) != e0_keys:
        raise ConfigError("e0 must contain exactly the declared pilot settings")
    if e0.get("main_family") not in _MODEL_FAMILIES:
        raise ConfigError("e0.main_family must be gemma, qwen, or llama")
    if (
        not isinstance(e0.get("smoke_families"), list)
        or not e0["smoke_families"]
        or any(family not in _MODEL_FAMILIES for family in e0["smoke_families"])
    ):
        raise ConfigError("e0.smoke_families must be a non-empty list of model families")
    for field in ("smoke_concepts", "batch_size", "gen_max_new_tokens", "top_k"):
        if type(e0.get(field)) is not int or e0[field] <= 0:
            raise ConfigError(f"e0.{field} must be a positive integer")
    sweep_sizes = e0.get("sweep_batch_sizes")
    if (
        not isinstance(sweep_sizes, list)
        or not sweep_sizes
        or any(type(size) is not int or size <= 0 for size in sweep_sizes)
    ):
        raise ConfigError("e0.sweep_batch_sizes must be a non-empty list of positive integers")
    tolerance = e0.get("rerun_tolerance")
    if type(tolerance) not in {int, float} or tolerance < 0:
        raise ConfigError("e0.rerun_tolerance must be a non-negative number")
    for field in ("results_dir", "rerun_dir"):
        if not isinstance(e0.get(field), str) or not e0[field]:
            raise ConfigError(f"e0.{field} must be a non-empty string")
    e1 = cfg.get("e1")
    e1_keys = {
        "families", "lens_gate_max_abs_logprob", "lens_gate_top1_agreement",
        "residual_gate_max_rel_error", "early_depth_max", "expected_early_blocks",
        "precision_check_concepts", "competence_formats",
        "competence_flag_min_translation_vi", "results_dir",
    }
    if not isinstance(e1, dict) or set(e1) != e1_keys:
        raise ConfigError("e1 must contain exactly the declared audit settings")
    families = e1.get("families")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(family, str) for family in families)
        or len(set(families)) != len(families)
        or any(family not in _MODEL_FAMILIES for family in families)
    ):
        raise ConfigError("e1.families must be a non-empty unique list of model families")
    for field in (
        "lens_gate_max_abs_logprob", "lens_gate_top1_agreement",
        "residual_gate_max_rel_error", "early_depth_max",
        "competence_flag_min_translation_vi",
    ):
        value = e1.get(field)
        if type(value) not in {int, float} or not 0 < float(value) <= 1:
            raise ConfigError(f"e1.{field} must be a positive number no greater than 1")
    early_blocks = e1.get("expected_early_blocks")
    if (
        not isinstance(early_blocks, dict)
        or set(early_blocks) != _MODEL_FAMILIES
        or any(
            not isinstance(blocks, list)
            or not blocks
            or any(type(block) is not int or block <= 0 for block in blocks)
            or blocks != sorted(set(blocks))
            for blocks in early_blocks.values()
        )
    ):
        raise ConfigError("e1.expected_early_blocks must map each family to sorted positive block indices")
    if type(e1.get("precision_check_concepts")) is not int or e1["precision_check_concepts"] <= 0:
        raise ConfigError("e1.precision_check_concepts must be a positive integer")
    competence_formats = e1.get("competence_formats")
    if (
        not isinstance(competence_formats, list)
        or any(not isinstance(value, str) for value in competence_formats)
        or set(competence_formats) != {"repetition", "translation"}
        or len(competence_formats) != 2
    ):
        raise ConfigError("e1.competence_formats must contain repetition and translation exactly once")
    if not isinstance(e1.get("results_dir"), str) or not e1["results_dir"]:
        raise ConfigError("e1.results_dir must be a non-empty string")
    excluded_directions = data.get("pretag_excluded_directions") if isinstance(data, dict) else None
    excluded_keys = {"concept_id", "format", "lang"}
    if not isinstance(excluded_directions, list):
        raise ConfigError("data.pretag_excluded_directions must be a list")
    seen_exclusions: set[tuple[str, str, str]] = set()
    for index, exclusion in enumerate(excluded_directions):
        if (
            not isinstance(exclusion, dict)
            or set(exclusion) != excluded_keys
            or any(not isinstance(exclusion.get(key), str) for key in excluded_keys)
        ):
            raise ConfigError(
                f"data.pretag_excluded_directions[{index}] must contain string concept_id, format, and lang"
            )
        identity = (exclusion["concept_id"], exclusion["format"], exclusion["lang"])
        if identity in seen_exclusions:
            raise ConfigError(f"data.pretag_excluded_directions contains duplicate {identity}")
        seen_exclusions.add(identity)
    text = cfg.get("text")
    if not isinstance(text, dict):
        raise ConfigError("text must be a mapping")
    if not isinstance(text.get("ntok_context_prefix"), str) or not text["ntok_context_prefix"]:
        raise ConfigError("text.ntok_context_prefix must be a non-empty string")
    distance = text.get("cognate_max_distance")
    if type(distance) is not float or not 0 < distance < 1:
        raise ConfigError("text.cognate_max_distance must be a float between 0 and 1")
    chunk_size = text.get("audit_chunk_size")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ConfigError("text.audit_chunk_size must be a positive integer")
    reference_facts = cfg.get("reference_facts")
    if not isinstance(reference_facts, dict) or not isinstance(reference_facts.get("x0c"), dict):
        raise ConfigError("reference_facts.x0c must be a mapping")
    if not isinstance(reference_facts.get("x0d"), dict):
        raise ConfigError("reference_facts.x0d must be a mapping")
    return cfg


def config_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a configuration file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_models(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the human-selected study checkpoint mapping or reject undecided D10."""
    selected = cfg["models"]["study_set"]
    if selected is None:
        raise ConfigError("D10 undecided: models.study_set is null")
    return cfg["models"][selected]
