#!/datastore/luannt/.cache/sb/bin/python
"""Job wrapper for the OPS-28 E6 synthetic timing and primary analysis."""

from __future__ import annotations

from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from vilens import env_guard

if __name__ == "__main__":
    env_guard.guard(offline=True)

import argparse
import gc
import importlib.metadata
import json
import logging
import resource
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

THREAD_CAPS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
    "ARROW_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}
os.environ.update(THREAD_CAPS)

import numpy as np

from vilens.analysis import e6
from vilens.config import load_config
from vilens.locks import ensure_adjudication_committed, ensure_model_access_allowed
from vilens.manifest import verify_manifest
from vilens.provenance import git_state
from vilens.preflight import Check, format_checks, preflight_ok


def _utc_now() -> str:
    """Return a UTC timestamp with a Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _logger() -> logging.Logger:
    """Configure stdout and file logging for this job."""
    from vilens.cli import setup_logging

    return setup_logging("job_analyze_e6", REPO / "logs")


def _check(name: str, passed: bool, detail: str) -> Check:
    """Build a standard preflight record."""
    return Check(name, "PASS" if passed else "FAIL", detail)


def _preflight(cfg: Mapping[str, Any], source: str, *, gpu: bool) -> list[Check]:
    """Check data/config/locks and optionally GPU, without loading study weights."""
    checks: list[Check] = []
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                checks.append(_check("gpu_visible", False, "torch.cuda.is_available() is false"))
                checks.append(_check("bf16_matmul", False, "GPU is unavailable"))
            else:
                checks.append(_check("gpu_visible", True, torch.cuda.get_device_name(0)))
                left = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
                checks.append(_check("bf16_matmul", bool(torch.isfinite(left @ left).all().item()), "64x64 bf16 CUDA matmul"))
        except Exception as exc:
            checks.append(_check("gpu", False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.extend((
            Check("gpu_visible", "SKIPPED", "login-node preflight"),
            Check("bf16_matmul", "SKIPPED", "login-node preflight"),
        ))

    try:
        statsmodels_version = importlib.metadata.version("statsmodels")
        checks.append(_check(
            "statsmodels_0_15", statsmodels_version == "0.15.0",
            f"installed={statsmodels_version}; required=0.15.0 (D11)",
        ))
    except importlib.metadata.PackageNotFoundError as exc:
        checks.append(_check("statsmodels_0_15", False, f"package missing: {exc}"))
    try:
        patsy_version = importlib.metadata.version("patsy")
        checks.append(_check("patsy_available", True, f"installed={patsy_version}"))
    except importlib.metadata.PackageNotFoundError as exc:
        checks.append(_check("patsy_available", False, f"package missing: {exc}"))

    if source == "primary":
        try:
            ensure_adjudication_committed(REPO)
            ensure_model_access_allowed(dict(cfg), [{"split": "test"}])
            checks.append(_check("integrity_locks", True, "Checkpoint H and test-item locks pass"))
        except Exception as exc:
            checks.append(_check("integrity_locks", False, f"{type(exc).__name__}: {exc}"))
        try:
            data = verify_manifest(REPO / str(cfg["paths"]["manifest"]), REPO, only=[str(cfg["paths"]["concepts_tsv"])])
            checks.append(_check("concepts_tsv_sha256", True, f"{len(data)} manifest input verified"))
        except Exception as exc:
            checks.append(_check("concepts_tsv_sha256", False, f"{type(exc).__name__}: {exc}"))
        try:
            from scripts.analyze_e6 import preflight_primary_inputs

            inputs = preflight_primary_inputs(cfg, REPO)
            checks.extend(_check(name, status == "MATCH", detail) for name, status, detail in inputs)
        except Exception as exc:
            checks.append(_check("primary_inputs", False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(_check("synthetic_shapes", True, "in-memory data only; no primary outcomes opened"))

    output_dirs = [
        REPO / str(cfg["e6"]["output_dir"]),
        REPO / str(cfg["e6"]["synthetic_dir"]),
        REPO / "reports",
    ]
    for directory in output_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".e6-preflight-{os.getpid()}"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
            checks.append(_check(f"writable_{directory.name}", True, str(directory.relative_to(REPO))))
        except Exception as exc:
            checks.append(_check(f"writable_{directory.name}", False, f"{type(exc).__name__}: {exc}"))

    try:
        commit, dirty = git_state(REPO)
        checks.append(_check("git_clean", not dirty, f"HEAD={commit}; dirty={str(dirty).lower()}"))
    except Exception as exc:
        checks.append(_check("git_clean", False, f"{type(exc).__name__}: {exc}"))
    try:
        from scripts.analyze_e6 import construct_provenance_preflight_records

        records = construct_provenance_preflight_records(cfg, REPO)
        labels = sorted({label.split(":", 1)[1].split(":", 1)[0] for label, _ in records})
        checks.append(_check(
            "e6_provenance_records",
            bool(records),
            f"constructed={len(records)}; families=3; unit_types={','.join(labels)}; writes=0",
        ))
    except Exception as exc:
        checks.append(_check("e6_provenance_records", False, f"{type(exc).__name__}: {exc}"))
    return checks


def _summary_path() -> Path:
    """Return a unique timing summary path under ignored logs/."""
    job = os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO / "logs" / f"e6_timing_{job}_{stamp}.json"


def _recovery_summary_path() -> Path:
    """Return a unique synthetic recovery summary path under ignored logs/."""
    job = os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO / "logs" / f"e6_recovery_{job}_{stamp}.json"


def _synthetic_effect_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return configured fixed effects for synthetic-only designs."""
    planted = dict(cfg["e6"]["recovery"]["planted_coefficients"])
    target = float(planted.pop("stratum_sino_language_zh"))
    return {"synthetic_fixed_effects": planted, "target_interaction": target}


def _peak_rss_bytes() -> int:
    """Return the larger peak RSS observed for the parent and its workers."""
    own = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    children = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    multiplier = 1 if sys.platform == "darwin" else 1024
    return max(own, children) * multiplier


def _run_synthetic_timing(cfg: Mapping[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Fit full production-shaped synthetic models and project the primary plan."""
    import torch
    from scripts.analyze_e6 import (
        _assert_matching_committed,
        primary_complete_case_format_counts,
        primary_dzh_format_counts,
        primary_matched_pair_format_ids,
    )

    started_utc = _utc_now()
    wall_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    timings: dict[str, dict[str, float | int | bool]] = {}
    family_n = {
        family: int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
        for family in ("gemma", "qwen", "llama")
    }
    complete_counts = {str(key): int(value) for key, value in cfg["e6"]["complete_case_counts"].items()}
    model_draws = int(cfg["e6"]["bootstrap_resamples"])
    effect_settings = _synthetic_effect_settings(cfg)
    matched_path = REPO / str(cfg["e6"]["outputs"]["matched_pairs"])
    if not matched_path.is_file():
        raise FileNotFoundError("commit the covariate-only matched pair file before timing E6")
    import pandas as pd

    committed_pairs = _assert_matching_committed(REPO, cfg)
    matched_pairs = committed_pairs.loc[:, ["pair_id"]]
    matched_n = int(len(matched_pairs))
    format_concept_counts = primary_complete_case_format_counts(cfg, REPO)
    matched_concept_ids = set(committed_pairs["sino_concept_id"].astype(str)) | set(
        committed_pairs["nonsino_concept_id"].astype(str)
    )
    matched_format_concept_counts = primary_complete_case_format_counts(
        cfg, REPO, selected_concept_ids=matched_concept_ids
    )
    full_row_concepts = sum(sum(counts.values()) for counts in format_concept_counts.values())
    matched_row_concepts = sum(sum(counts.values()) for counts in matched_format_concept_counts.values())
    matched_fraction = matched_row_concepts / full_row_concepts
    dzh_format_counts = primary_dzh_format_counts(cfg, REPO)
    matched_pair_ids_by_format = primary_matched_pair_format_ids(cfg, REPO, committed_pairs)
    total_projected = 0.0
    family_projection: dict[str, float] = {}
    for family, layers in family_n.items():
        frame = e6.make_synthetic_model_frame(
            n_sino=complete_counts["sino"], n_nonsino=complete_counts["nonsino"],
            n_layers=layers, seed=int(cfg["seed"]), synthetic_sd=cfg["e6"]["synthetic_sd"],
            format_concept_counts=format_concept_counts, **effect_settings,
        )
        started = time.perf_counter()
        fitted = e6.fit_mixedlm(
            frame, spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]), fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
        )
        full_fit_seconds = time.perf_counter() - started
        draw_ids = e6.stratified_draws(
            {"sino": sorted(frame.loc[frame["stratum"].eq("sino"), "concept_id"].unique()),
             "nonsino": sorted(frame.loc[frame["stratum"].eq("nonsino"), "concept_id"].unique())},
            resamples=20, seed=int(cfg["seed"]),
            key=str(cfg["e6"]["model_key_template"]).format(model=family),
        )
        bootstrap_start = time.perf_counter()
        if fitted.diagnostics.valid:
            values, flags, fit_records = e6.model_bootstrap(
                frame, draw_ids, workers=int(cfg["e6"]["workers"]),
                start_params=fitted.params.to_numpy(dtype="float64"),
                spline_df=int(cfg["e6"]["spline_df"]), maxiter=int(cfg["e6"]["maxiter"]),
                fit_methods=cfg["e6"]["fit_methods"], chunk_size=20,
                boundary_rule=cfg["e6"]["boundary_rule"],
                llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            )
        else:
            values = np.full(len(draw_ids), np.nan)
            flags = np.zeros(len(draw_ids), dtype=bool)
            fit_records = [e6.empty_fit_diagnostics() for _ in draw_ids]
        refit_batch_seconds = time.perf_counter() - bootstrap_start
        fit_summary = e6.fit_diagnostics_summary(fit_records)
        mean_refit_seconds = refit_batch_seconds / 20.0
        fit_projection = full_fit_seconds + model_draws * mean_refit_seconds
        matched_projection = fit_projection * matched_fraction
        sensitivity_fit_count = len(cfg["e6"]["sensitivity_variants"])
        sensitivity_projection = sensitivity_fit_count * full_fit_seconds
        family_total = fit_projection + matched_projection + sensitivity_projection
        family_projection[family] = family_total
        total_projected += family_total
        timings[family] = {
            "rows": int(len(frame)),
            "format_concept_counts": format_concept_counts,
            "full_fit_seconds": full_fit_seconds,
            "20_refit_batch_seconds": refit_batch_seconds,
            "seconds_per_refit_wall": mean_refit_seconds,
            "converged_refits": int(flags.sum()),
            "full_fit_valid": bool(fitted.diagnostics.valid),
            "full_fit_method": fitted.diagnostics.method_used,
            "full_fit_tries": fitted.diagnostics.tries,
            "full_fit_boundary": fitted.diagnostics.variance_at_boundary,
            "refit_method_mix": fit_summary["method_mix"],
            "refit_boundary_count": fit_summary["boundary_count"],
            "refit_warning_categories": fit_summary["warning_categories"],
            "mean_refit_fit_seconds": fit_summary["mean_fit_seconds"],
            "mean_refit_tries": fit_summary["mean_tries"],
            "refit_count": int(len(flags)),
            "peak_rss_bytes": _peak_rss_bytes(),
            "full_plus_1000_refits_projection_seconds": fit_projection,
            "matched_projection_seconds": matched_projection,
            "eight_sensitivity_fits_projection_seconds": sensitivity_projection,
            "primary_projection_seconds": family_total,
        }
        logger.info(
            "timing family=%s rows=%d full_fit_seconds=%.3f full_fit_valid=%s full_method=%s full_tries=%d "
            "refit_batch_20_seconds=%.3f mean_refit_wall_seconds=%.3f valid=%d/20 method_mix=%s "
            "boundary_count=%d peak_rss_bytes=%d",
            family, len(frame), full_fit_seconds, fitted.diagnostics.valid, fitted.diagnostics.method_used,
            fitted.diagnostics.tries, refit_batch_seconds, mean_refit_seconds, int(flags.sum()),
            fit_summary["method_mix"], fit_summary["boundary_count"], timings[family]["peak_rss_bytes"],
        )
        del frame, fitted, draw_ids, values, flags
        gc.collect()

    # The D_zh bootstrap arithmetic is cheap, but time its production-size arrays
    # using a vectorized synthetic table and the frozen concept keys.
    dzh_seconds_by_family: dict[str, float] = {}
    matched_dzh_seconds_by_family: dict[str, float] = {}
    for family, layers in family_n.items():
        numpy = __import__("numpy")
        rng = numpy.random.default_rng(int(cfg["seed"]))
        rows: list[dict[str, Any]] = []
        for fmt in ("repetition", "translation"):
            for stratum in ("sino", "nonsino"):
                count = int(dzh_format_counts[fmt][stratum])
                for concept in range(count):
                    concept_id = f"{family}-{stratum}-{concept:04d}"
                    for layer in range(1, layers + 1):
                        rows.append({
                            "format": fmt, "concept_id": concept_id, "stratum": stratum,
                            "layer": layer, "value": float(rng.normal()),
                        })
        la = pd.DataFrame.from_records(rows)
        ids = {
            name: sorted(la.loc[la["stratum"].eq(name), "concept_id"].astype(str).unique())
            for name in ("sino", "nonsino")
        }
        draws = e6.stratified_draws(
            ids, resamples=model_draws, seed=int(cfg["seed"]),
            key=str(cfg["e6"]["dzh_key_template"]).format(model=family),
        )
        started = time.perf_counter()
        e6.dzh_from_frame(la, n_layers=layers, window=cfg["e6"]["window"], draws=draws)
        dzh_seconds_by_family[family] = time.perf_counter() - started
        logger.info(
            "timing D_zh_bootstrap family=%s draws=%d wall_seconds=%.3f",
            family, model_draws, dzh_seconds_by_family[family],
        )
        del la, rows, ids, draws, rng
        gc.collect()
        low, high = map(float, cfg["e6"]["window"])
        eligible_layers = [
            layer for layer in range(1, layers + 1)
            if low <= layer / layers <= high
        ]
        pair_rng = numpy.random.default_rng(int(cfg["seed"]) + 1)
        pair_values_array = pair_rng.normal(
            size=sum(len(ids) for ids in matched_pair_ids_by_format.values()) * len(eligible_layers)
        )
        pair_rows: list[dict[str, Any]] = []
        value_index = 0
        pair_index_by_id = {
            str(pair_id): index for index, pair_id in enumerate(committed_pairs["pair_id"].astype(str))
        }
        for pair in committed_pairs.itertuples(index=False):
            for fmt in ("repetition", "translation"):
                if str(pair.pair_id) not in matched_pair_ids_by_format[fmt]:
                    continue
                for layer in eligible_layers:
                    pair_rows.append({
                        "pair_id": f"pair_{pair_index_by_id[str(pair.pair_id)]:04d}",
                        "format": fmt,
                        "layer": layer,
                        "difference": float(pair_values_array[value_index]),
                    })
                    value_index += 1
        pair_values = pd.DataFrame.from_records(pair_rows)
        started = time.perf_counter()
        e6.dzh_pair_bootstrap(
            pair_values, resamples=model_draws, seed=int(cfg["seed"]),
            key=str(cfg["e6"]["matched_key_template"]).format(model=family),
            chunk_size=int(cfg["e6"]["bootstrap_chunk"]),
        )
        matched_dzh_seconds_by_family[family] = time.perf_counter() - started
        logger.info(
            "timing matched_D_zh_bootstrap family=%s pairs=%d draws=%d wall_seconds=%.3f",
            family, matched_n, model_draws, matched_dzh_seconds_by_family[family],
        )
        del pair_rows, pair_values, pair_values_array, pair_rng
        gc.collect()
    dzh_projection = sum(dzh_seconds_by_family.values())
    # The base D_zh and each non-frequency sensitivity receive a bootstrap CI.
    dzh_sensitivity_count = sum(
        str(name) not in {"zipf_vi", "zipf_zh"}
        for name in cfg["e6"]["sensitivity_variants"]
    )
    dzh_projection *= 1 + dzh_sensitivity_count
    matched_dzh_projection = sum(matched_dzh_seconds_by_family.values())
    total_projected += dzh_projection
    total_projected += matched_dzh_projection
    summary: dict[str, Any] = {
        "utc_start": started_utc,
        "families": timings,
        "matched_pairs": matched_n,
        "format_concept_counts": format_concept_counts,
        "matched_format_concept_counts": matched_format_concept_counts,
        "dzh_format_concept_counts": dzh_format_counts,
        "matched_pair_counts_by_format": {
            fmt: len(pair_ids) for fmt, pair_ids in matched_pair_ids_by_format.items()
        },
        "sensitivity_variants": list(cfg["e6"]["sensitivity_variants"]),
        "sensitivity_variant_count": len(cfg["e6"]["sensitivity_variants"]),
        "dzh_bootstrap_seconds_per_family": dzh_seconds_by_family,
        "dzh_projection_seconds": dzh_projection,
        "matched_dzh_bootstrap_seconds_per_family": matched_dzh_seconds_by_family,
        "matched_dzh_projection_seconds": matched_dzh_projection,
        "projected_primary_wall_seconds": total_projected,
        "projected_primary_wall_hours": total_projected / 3600.0,
        "projected_primary_cpu_hours": total_projected * int(cfg["e6"]["workers"]) / 3600.0,
        "max_projected_hours": int(cfg["e6"]["max_projected_hours"]),
        "projection_within_48h": total_projected <= float(cfg["e6"]["max_projected_hours"]) * 3600,
        "sensitivity_fit_count": len(cfg["e6"]["sensitivity_variants"]),
        "dzh_bootstrap_multiplier": 1 + dzh_sensitivity_count,
        "projected_primary_gpu_hours": total_projected / 3600.0,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "wall_seconds": time.perf_counter() - wall_started,
        "utc_end": _utc_now(),
    }
    path = _summary_path()
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("synthetic timing summary=%s", path.relative_to(REPO))
    logger.info("E6 timing projection %s", json.dumps(summary, sort_keys=True))
    return summary


def _run_synthetic_validation(cfg: Mapping[str, Any], logger: logging.Logger) -> tuple[dict[str, Any], bool]:
    """Validate OPS-29 on production-shaped current and old stress fixtures."""
    import pandas as pd
    import torch

    from scripts.analyze_e6 import (
        _assert_matching_committed,
        primary_complete_case_format_counts,
        primary_matched_pair_format_ids,
    )

    started_utc = _utc_now()
    wall_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    complete_counts = {str(key): int(value) for key, value in cfg["e6"]["complete_case_counts"].items()}
    family_n = {
        family: int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
        for family in ("gemma", "qwen", "llama")
    }
    draws_count = int(cfg["e6"]["synthetic_validation_draws"])
    old_sd = {"concept": 0.25, "concept_language": 0.0, "residual": 0.04}
    current_sd = {key: float(value) for key, value in cfg["e6"]["synthetic_sd"].items()}
    method_list = cfg["e6"]["fit_methods"]
    maxiter = int(cfg["e6"]["maxiter"])
    workers = int(cfg["e6"]["workers"])
    effect_settings = _synthetic_effect_settings(cfg)
    matched_pairs = _assert_matching_committed(REPO, cfg)
    matched_ids = set(matched_pairs["sino_concept_id"].astype(str)) | set(
        matched_pairs["nonsino_concept_id"].astype(str)
    )
    full_format_counts = primary_complete_case_format_counts(cfg, REPO)
    matched_format_counts = primary_complete_case_format_counts(
        cfg, REPO, selected_concept_ids=matched_ids
    )
    full_row_concepts = sum(sum(counts.values()) for counts in full_format_counts.values())
    matched_row_concepts = sum(sum(counts.values()) for counts in matched_format_counts.values())
    matched_fraction = matched_row_concepts / full_row_concepts
    pair_format_ids = primary_matched_pair_format_ids(cfg, REPO, matched_pairs)
    matched_pair_format_fraction = sum(map(len, pair_format_ids.values())) / (len(matched_pairs) * 2.0)
    primary_outputs: dict[str, Any] = {}
    passed_all = True

    def run_setting(family: str, layers: int, label: str, sd: Mapping[str, float]) -> dict[str, Any]:
        frame = e6.make_synthetic_model_frame(
            n_sino=complete_counts["sino"],
            n_nonsino=complete_counts["nonsino"],
            n_layers=layers,
            seed=int(cfg["seed"]),
            synthetic_sd=sd,
            **effect_settings,
        )
        fit_started = time.perf_counter()
        full_fit = e6.fit_mixedlm(
            frame,
            spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=maxiter,
            fit_methods=method_list,
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
        )
        full_wall = time.perf_counter() - fit_started
        full_info = e6.model_instrument(full_fit)
        ids = {
            stratum: sorted(frame.loc[frame["stratum"].astype(str).eq(stratum), "concept_id"].astype(str).unique())
            for stratum in ("sino", "nonsino")
        }
        draw_key = f"bootstrap|e6_validation_{label}|{family}|pooled"
        draws = e6.stratified_draws(ids, resamples=draws_count, seed=int(cfg["seed"]), key=draw_key)
        bootstrap_started = time.perf_counter()
        # OPS-29 suppresses bootstrap work for non-estimable analysis cells.
        # The old-scale stress case is validation-only, so still run its 100
        # refits cold when its full-sample fit cannot provide valid warm starts.
        if full_info["valid"] or label == "old_stress":
            values, flags, records = e6.model_bootstrap(
                frame,
                draws,
                workers=workers,
                start_params=(
                    full_fit.params.to_numpy(dtype=np.float64) if full_info["valid"] else None
                ),
                spline_df=int(cfg["e6"]["spline_df"]),
                maxiter=maxiter,
                fit_methods=method_list,
                chunk_size=int(cfg["e6"]["bootstrap_chunk"]),
                boundary_rule=cfg["e6"]["boundary_rule"],
                llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            )
        else:
            values = np.full(draws_count, np.nan, dtype=np.float64)
            flags = np.zeros(draws_count, dtype=bool)
            records = [e6.empty_fit_diagnostics() for _ in draws]
        bootstrap_wall = time.perf_counter() - bootstrap_started
        summary = e6.fit_diagnostics_summary(records)
        result = {
            "full_fit_valid": bool(full_info["valid"]),
            "full_fit_method": str(full_info["method_used"]),
            "full_fit_tries": int(full_info["tries"]),
            "full_fit_boundary": bool(full_info["variance_at_boundary"]),
            "full_fit_warning_categories": full_info["warning_categories"],
            "full_fit_seconds": float(full_info["fit_seconds"]),
            "valid_draws": int(flags.sum()),
            "draw_count": draws_count,
            "method_mix": summary["method_mix"],
            "boundary_count": summary["boundary_count"],
            "warning_categories": summary["warning_categories"],
            "mean_seconds_per_fit": summary["mean_fit_seconds"],
            "mean_tries_per_fit": summary["mean_tries"],
            "bootstrap_batch_seconds": float(bootstrap_wall),
            "mean_seconds_per_draw_wall": float(bootstrap_wall / draws_count),
            "peak_rss_bytes": _peak_rss_bytes(),
            "rows": int(len(frame)),
        }
        logger.info(
            "validation family=%s setting=%s full_valid=%s full_method=%s full_tries=%d "
            "valid_draws=%d/%d method_mix=%s boundary_count=%d mean_fit_seconds=%.4f "
            "mean_tries=%.3f peak_rss_bytes=%d",
            family, label, result["full_fit_valid"], result["full_fit_method"], result["full_fit_tries"],
            result["valid_draws"], draws_count, result["method_mix"], result["boundary_count"],
            result["mean_seconds_per_fit"], result["mean_tries_per_fit"], result["peak_rss_bytes"],
        )
        del frame, full_fit, values, flags, records, draws
        gc.collect()
        return result

    for family, layers in family_n.items():
        current = run_setting(family, layers, "current", current_sd)
        current["gate_passed"] = current["valid_draws"] >= int(cfg["e6"]["synthetic_min_valid"])
        passed_all = passed_all and bool(current["gate_passed"])
        stress = run_setting(family, layers, "old_stress", old_sd)
        primary_outputs[family] = {"current": current, "old_stress": {
            "valid_draws": stress["valid_draws"],
            "draw_count": stress["draw_count"],
            "method_mix": stress["method_mix"],
        }}

    # Reuse the already measured synthetic D_zh timings from job 3280; only the
    # MixedLM portion is recomputed from the observed OPS-29 fit timings here.
    old_timing_path = REPO / "logs" / "e6_timing_3280_20261013T195620Z.json"
    if not old_timing_path.is_file():
        raise FileNotFoundError(f"required prior synthetic D_zh timing summary is missing: {old_timing_path}")
    old_timing = json.loads(old_timing_path.read_text(encoding="utf-8"))
    non_frequency_sensitivities = sum(
        str(item) not in {"zipf_vi", "zipf_zh"} for item in cfg["e6"]["sensitivity_variants"]
    )
    dzh_multiplier = 1 + non_frequency_sensitivities
    projection_by_family: dict[str, dict[str, float]] = {}
    projected_wall = 0.0
    for family, values in primary_outputs.items():
        current = values["current"]
        full_fit_seconds = float(current["full_fit_seconds"])
        # The measured 100-draw batch captures the configured worker pool and
        # all optimizer retries; scale that observed wall duration to B=1,000.
        full_set_seconds = full_fit_seconds + float(current["bootstrap_batch_seconds"]) * (
            int(cfg["e6"]["bootstrap_resamples"]) / draws_count
        )
        matched_seconds = full_set_seconds * matched_fraction
        sensitivity_seconds = full_fit_seconds * len(cfg["e6"]["sensitivity_variants"])
        dzh_seconds = float(old_timing["dzh_bootstrap_seconds_per_family"][family]) * dzh_multiplier
        matched_dzh_seconds = float(old_timing["matched_dzh_bootstrap_seconds_per_family"][family]) * matched_pair_format_fraction
        family_total = full_set_seconds + matched_seconds + sensitivity_seconds + dzh_seconds + matched_dzh_seconds
        projection_by_family[family] = {
            "full_set_seconds": full_set_seconds,
            "matched_set_seconds": matched_seconds,
            "sensitivity_fits_seconds": sensitivity_seconds,
            "dzh_bootstraps_seconds": dzh_seconds,
            "matched_dzh_bootstrap_seconds": matched_dzh_seconds,
            "projected_wall_seconds": family_total,
            "observed_mean_fit_seconds": float(current["mean_seconds_per_fit"]),
            "observed_mean_tries": float(current["mean_tries_per_fit"]),
        }
        projected_wall += family_total
    summary = {
        "utc_start": started_utc,
        "families": primary_outputs,
        "synthetic_sd_current": current_sd,
        "synthetic_sd_old_stress": old_sd,
        "validation_draws": draws_count,
        "minimum_valid_draws": int(cfg["e6"]["synthetic_min_valid"]),
        "validation_passed": passed_all,
        "projection_by_family": projection_by_family,
        "projected_primary_wall_seconds": projected_wall,
        "projected_primary_wall_hours": projected_wall / 3600.0,
        "projected_primary_cpu_hours": projected_wall * workers / 3600.0,
        "max_projected_hours": int(cfg["e6"]["max_projected_hours"]),
        "projection_within_48h": projected_wall <= float(cfg["e6"]["max_projected_hours"]) * 3600,
        "matched_pairs": int(len(matched_pairs)),
        "matched_fraction": matched_fraction,
        "matched_pair_format_fraction": matched_pair_format_fraction,
        "peak_rss_bytes": _peak_rss_bytes(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "wall_seconds": time.perf_counter() - wall_started,
        "utc_end": _utc_now(),
    }
    path = _summary_path()
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("synthetic validation summary=%s", path.relative_to(REPO))
    logger.info("E6 synthetic validation %s", json.dumps(summary, sort_keys=True))
    return summary, passed_all


def _run_synthetic_recovery(cfg: Mapping[str, Any], logger: logging.Logger) -> tuple[dict[str, Any], bool]:
    """Run OPS-30 full-fit and OPS-31 refit recovery checks on synthetic rows."""
    import torch

    from scripts.analyze_e6 import (
        _assert_matching_committed,
        primary_complete_case_format_counts,
        primary_matched_pair_format_ids,
    )

    started_utc = _utc_now()
    wall_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    complete_counts = {str(key): int(value) for key, value in cfg["e6"]["complete_case_counts"].items()}
    family_layers = {
        family: int(cfg["reference_facts"]["x0c"]["model_dims"][family][0])
        for family in ("gemma", "qwen", "llama")
    }
    recovery = cfg["e6"]["recovery"]
    planted = recovery["planted_coefficients"]
    effects = _synthetic_effect_settings(cfg)
    family_results: dict[str, dict[str, Any]] = {}
    all_passed = True
    refit_count = int(recovery["refits"])
    minimum_valid_refits = int(recovery["min_valid_refits"])
    for family, layers in family_layers.items():
        frame = e6.make_synthetic_model_frame(
            n_sino=complete_counts["sino"],
            n_nonsino=complete_counts["nonsino"],
            n_layers=layers,
            seed=int(cfg["seed"]),
            synthetic_sd=cfg["e6"]["synthetic_sd"],
            **effects,
        )
        fit_started = time.perf_counter()
        fitted = e6.fit_mixedlm(
            frame,
            spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]),
            fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
        )
        fit_wall = time.perf_counter() - fit_started
        result = e6.synthetic_recovery_checks(
            frame,
            fitted,
            n_layers=layers,
            planted_concept_sd=float(cfg["e6"]["synthetic_sd"]["concept"]),
            planted_interaction=float(planted["stratum_sino_language_zh"]),
            concept_sd_ratio_bounds=recovery["concept_sd_ratio_bounds"],
            target_wald_se_multiple=float(recovery["target_wald_se_multiple"]),
            target_abs_bias_max=float(recovery["target_abs_bias_max"]),
        )
        ids = {
            stratum: sorted(
                frame.loc[frame["stratum"].astype(str).eq(stratum), "concept_id"].astype(str).unique()
            )
            for stratum in ("sino", "nonsino")
        }
        refit_draws = e6.stratified_draws(
            ids,
            resamples=refit_count,
            seed=int(cfg["seed"]),
            key=str(cfg["e6"]["model_key_template"]).format(model=family),
        )
        refit_started = time.perf_counter()
        start_params = (
            fitted.params.to_numpy(dtype=np.float64)
            if fitted.result is not None and len(fitted.params) > 0
            else None
        )
        _refit_estimates, refit_valid, refit_records = e6.model_bootstrap(
            frame,
            refit_draws,
            workers=int(cfg["e6"]["workers"]),
            start_params=start_params,
            spline_df=int(cfg["e6"]["spline_df"]),
            maxiter=int(cfg["e6"]["maxiter"]),
            fit_methods=cfg["e6"]["fit_methods"],
            boundary_rule=cfg["e6"]["boundary_rule"],
            llf_tie_tol=float(cfg["e6"]["llf_tie_tol"]),
            chunk_size=refit_count,
        )
        refit_wall = time.perf_counter() - refit_started
        refit_summary = e6.synthetic_refit_recovery_summary(
            refit_records,
            planted_concept_sd=float(cfg["e6"]["synthetic_sd"]["concept"]),
            concept_sd_ratio_bounds=recovery["concept_sd_ratio_bounds"],
            expected_refits=refit_count,
            min_valid_refits=minimum_valid_refits,
        )
        result["full_fit_checks_passed"] = bool(result["passed"])
        result["refit_summary"] = refit_summary
        result["refit_batch_seconds"] = refit_wall
        result["refit_valid_flags_match"] = int(refit_valid.sum()) == int(refit_summary["valid_refit_count"])
        result["passed"] = bool(result["full_fit_checks_passed"] and refit_summary["gate_passed"])
        result.update({
            "family": family,
            "layers": layers,
            "rows": int(len(frame)),
            "fit_wall_seconds": fit_wall,
            "warning_categories": fitted.diagnostics.warning_categories,
            "failure_reasons": list(fitted.diagnostics.failure_reasons),
            "peak_rss_bytes": _peak_rss_bytes(),
        })
        all_passed = all_passed and bool(result["passed"])
        logger.info(
            "recovery family=%s pass=%s planted_concept_sd=%.6f estimated_concept_sd=%s "
            "residual_sd=%s target=%.6f estimate=%s wald_se=%s abs_bias=%s "
            "groups=%d/%d rows_per_group=%d..%d expected=%d method=%s tries=%d checks=%s "
            "boundary=%s boundary_rule_ran=%s boundary_rule_changed=%s llf_first_valid=%s llf_kept=%s "
            "refits_valid=%d/%d refits_recovered=%d sd_min=%s sd_median=%s sd_max=%s "
            "refit_method_mix=%s refit_boundary_triggers=%d refit_boundary_changes=%d "
            "refit_mean_seconds=%.4f refit_mean_tries=%.3f refit_wall_seconds=%.3f "
            "fit_wall_seconds=%.3f peak_rss_bytes=%d",
            family, result["passed"], result["planted_concept_sd"], result["estimated_concept_sd"],
            result["residual_sd"], result["planted_interaction"], result["interaction_estimate"],
            result["interaction_wald_se"], result["interaction_abs_bias"], result["group_count"],
            result["concept_count"], result["observed_rows_per_group_min"],
            result["observed_rows_per_group_max"], result["expected_rows_per_group"],
            result["method_used"], result["tries"], result["checks"],
            result["variance_at_boundary"], result["boundary_rule_ran"],
            result["boundary_rule_changed_fit"], result["llf_first_valid"],
            result["llf_kept"], refit_summary["valid_refit_count"], refit_count,
            refit_summary["recovered_refit_count"], refit_summary["concept_sd_min"],
            refit_summary["concept_sd_median"], refit_summary["concept_sd_max"], refit_summary["method_mix"],
            refit_summary["boundary_rule_trigger_count"], refit_summary["boundary_rule_changed_fit_count"],
            refit_summary["mean_seconds_per_fit"], refit_summary["mean_tries"],
            refit_wall, fit_wall, result["peak_rss_bytes"],
        )
        family_results[family] = result
        del frame, fitted
        gc.collect()
    matched_pairs = _assert_matching_committed(REPO, cfg)
    matched_ids = set(matched_pairs["sino_concept_id"].astype(str)) | set(
        matched_pairs["nonsino_concept_id"].astype(str)
    )
    full_format_counts = primary_complete_case_format_counts(cfg, REPO)
    matched_format_counts = primary_complete_case_format_counts(
        cfg, REPO, selected_concept_ids=matched_ids
    )
    full_row_concepts = sum(sum(counts.values()) for counts in full_format_counts.values())
    matched_row_concepts = sum(sum(counts.values()) for counts in matched_format_counts.values())
    matched_fraction = matched_row_concepts / full_row_concepts
    pair_format_ids = primary_matched_pair_format_ids(cfg, REPO, matched_pairs)
    matched_pair_format_fraction = sum(map(len, pair_format_ids.values())) / (len(matched_pairs) * 2.0)
    previous_timing_path = REPO / "logs" / "e6_timing_3280_20261013T195620Z.json"
    if not previous_timing_path.is_file():
        raise FileNotFoundError(f"required prior synthetic D_zh timing summary is missing: {previous_timing_path}")
    previous_timing = json.loads(previous_timing_path.read_text(encoding="utf-8"))
    non_frequency_sensitivities = sum(
        str(item) not in {"zipf_vi", "zipf_zh"} for item in cfg["e6"]["sensitivity_variants"]
    )
    dzh_multiplier = 1 + non_frequency_sensitivities
    projection_by_family: dict[str, dict[str, float]] = {}
    projected_wall_seconds = 0.0
    for family, family_result in family_results.items():
        full_fit_seconds = float(family_result["fit_wall_seconds"])
        refit_batch_seconds = float(family_result["refit_batch_seconds"])
        full_set_seconds = full_fit_seconds + refit_batch_seconds * (
            int(cfg["e6"]["bootstrap_resamples"]) / refit_count
        )
        matched_set_seconds = full_set_seconds * matched_fraction
        sensitivity_seconds = full_fit_seconds * len(cfg["e6"]["sensitivity_variants"])
        dzh_seconds = float(previous_timing["dzh_bootstrap_seconds_per_family"][family]) * dzh_multiplier
        matched_dzh_seconds = float(previous_timing["matched_dzh_bootstrap_seconds_per_family"][family]) * matched_pair_format_fraction
        family_seconds = full_set_seconds + matched_set_seconds + sensitivity_seconds + dzh_seconds + matched_dzh_seconds
        projection_by_family[family] = {
            "full_set_seconds": full_set_seconds,
            "matched_set_seconds": matched_set_seconds,
            "sensitivity_fits_seconds": sensitivity_seconds,
            "dzh_bootstraps_seconds": dzh_seconds,
            "matched_dzh_bootstrap_seconds": matched_dzh_seconds,
            "projected_wall_seconds": family_seconds,
            "observed_mean_refit_fit_seconds": float(family_result["refit_summary"]["mean_seconds_per_fit"]),
            "observed_mean_refit_tries": float(family_result["refit_summary"]["mean_tries"]),
        }
        projected_wall_seconds += family_seconds
    summary = {
        "utc_start": started_utc,
        "families": family_results,
        "synthetic_sd": {key: float(value) for key, value in cfg["e6"]["synthetic_sd"].items()},
        "planted_coefficients": {key: float(value) for key, value in planted.items()},
        "recovery_settings": recovery,
        "refit_count": refit_count,
        "min_valid_refits": minimum_valid_refits,
        "matched_pairs": int(len(matched_pairs)),
        "matched_fraction": matched_fraction,
        "matched_pair_format_fraction": matched_pair_format_fraction,
        "primary_projection_by_family": projection_by_family,
        "projected_primary_wall_seconds": projected_wall_seconds,
        "projected_primary_wall_hours": projected_wall_seconds / 3600.0,
        "projected_primary_cpu_hours": projected_wall_seconds * int(cfg["e6"]["workers"]) / 3600.0,
        "projected_primary_gpu_hours": projected_wall_seconds / 3600.0,
        "max_projected_hours": int(cfg["e6"]["max_projected_hours"]),
        "projection_within_48h": projected_wall_seconds <= float(cfg["e6"]["max_projected_hours"]) * 3600,
        "passed": all_passed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "wall_seconds": time.perf_counter() - wall_started,
        "utc_end": _utc_now(),
    }
    path = _recovery_summary_path()
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("synthetic recovery summary=%s", path.relative_to(REPO))
    logger.info("E6 synthetic recovery %s", json.dumps(summary, sort_keys=True))
    return summary, all_passed


def _validation_exit_code(valid_by_family: Mapping[str, int], minimum: int) -> int:
    """Return non-zero when any synthetic family misses the OPS-29 valid-draw gate."""
    return 0 if valid_by_family and all(int(count) >= int(minimum) for count in valid_by_family.values()) else 3


def _parser() -> argparse.ArgumentParser:
    """Build the root job parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--source", choices=("synthetic", "primary"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--confirm-e6", action="store_true")
    parser.add_argument("--closeout", action="store_true")
    parser.add_argument("--moment-refit", metavar="FAMILY:FIT_SET[,FAMILY:FIT_SET]")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run input/GPU preflight and then a synthetic timing or primary analysis."""
    parser = _parser()
    args = parser.parse_args(argv)
    cfg = load_config(REPO / args.config)
    logger = _logger()
    synthetic_modes = sum(bool(value) for value in (args.timing, args.validate, args.recovery))
    if args.moment_refit and (
        args.source != "primary" or not args.confirm_e6 or args.closeout or synthetic_modes
    ):
        parser.error("--moment-refit requires --source primary --confirm-e6 and excludes other modes")
    if args.closeout and (args.source != "primary" or not args.confirm_e6 or synthetic_modes):
        parser.error("--closeout requires --source primary --confirm-e6")
    if args.source == "synthetic" and not (synthetic_modes or args.preflight_only):
        parser.error("--source synthetic requires --timing, --validate, --recovery or --preflight-only")
    if args.source == "primary" and synthetic_modes:
        parser.error("--timing, --validate and --recovery are only valid with --source synthetic")
    if synthetic_modes > 1:
        parser.error("--timing, --validate and --recovery are mutually exclusive")
    checks = _preflight(cfg, args.source, gpu=not args.preflight_only)
    print(format_checks(checks), flush=True)
    if not preflight_ok(checks):
        logger.error("E6 preflight failed; no outcome computation started")
        return 2
    if args.preflight_only:
        logger.info("preflight-only complete; model weights were not loaded")
        return 0
    if args.source == "primary":
        if not args.confirm_e6:
            parser.error("--source primary requires --confirm-e6")
        from scripts import analyze_e6
        forwarded = ["--config", args.config, "--source", "primary", "--confirm-e6"]
        if args.closeout:
            forwarded.append("--closeout")
        elif args.moment_refit:
            forwarded.extend(("--moment-refit", args.moment_refit))
        else:
            forwarded.extend(("--stages", "all"))
        return analyze_e6.main(forwarded)
    if args.validate:
        summary, _passed = _run_synthetic_validation(cfg, logger)
        valid_by_family = {
            family: int(data["current"]["valid_draws"])
            for family, data in summary["families"].items()
        }
        return _validation_exit_code(valid_by_family, int(cfg["e6"]["synthetic_min_valid"]))
    if args.recovery:
        _summary, passed = _run_synthetic_recovery(cfg, logger)
        return 0 if passed else 4
    if not args.timing:
        parser.error("synthetic execution requires --timing")
    summary = _run_synthetic_timing(cfg, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
