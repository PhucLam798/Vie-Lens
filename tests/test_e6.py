"""Synthetic-only coverage for OPS-28 analysis and matching helpers."""

from __future__ import annotations

import builtins
import copy
import logging
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_e6
from vilens.analysis import e6
from vilens.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _committed_e6_smoke_repo(root: Path) -> Path:
    """Create a tiny committed repo with every file provenance requires."""
    for relative in (
        "src", "configs", "derived", "scripts", "data", "reports", "logs",
        "results/analysis/primary/e6",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "src/fixture.py").write_text("FIXTURE = True\n", encoding="utf-8")
    (root / "derived/seeds.json").write_text("{}\n", encoding="utf-8")
    (root / "scripts/analyze_e6.py").write_text("# fixture entrypoint\n", encoding="utf-8")
    (root / "job_analyze_e6.py").write_text("# fixture job\n", encoding="utf-8")
    (root / "configs/experiments.yaml").write_bytes(
        (REPO_ROOT / "configs/experiments.yaml").read_bytes()
    )
    (root / "data/MANIFEST.sha256").write_bytes(
        (REPO_ROOT / "data/MANIFEST.sha256").read_bytes()
    )
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "E6 test", "GIT_AUTHOR_EMAIL": "e6-test@example.invalid",
        "GIT_COMMITTER_NAME": "E6 test", "GIT_COMMITTER_EMAIL": "e6-test@example.invalid",
    })
    subprocess.run(["git", "init", "-q"], cwd=root, env=env, check=True)
    subprocess.run(["git", "add", "src", "configs", "derived", "scripts", "data", "job_analyze_e6.py"], cwd=root, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, env=env, check=True)
    return root


def _toy_match_frame() -> pd.DataFrame:
    """Build a covariate-only matching fixture with two nearby pairs."""
    rows = []
    values = {
        "s1": ("sino", 0.0), "s2": ("sino", 10.0),
        "n1": ("nonsino", 0.1), "n2": ("nonsino", 10.1),
    }
    for concept_id, (stratum, value) in values.items():
        rows.append({
            "concept_id": concept_id, "split": "test", "m1_extension": False,
            "stratum": stratum, "zipf_en": value, "n_syllables": value + 0.2,
            "concreteness": value + 0.4, "ntok_vi_mean": value + 0.6,
        })
    return pd.DataFrame(rows)


def test_matching_is_deterministic_caliper_limited_and_without_replacement() -> None:
    """Matching reuses no nonsino ID and enforces every standardized caliper."""
    frame = _toy_match_frame()
    first, balance, unmatched = e6.match_without_replacement(
        frame, seed=20260925, stream_key="e6_match", enforce_reference_counts=False
    )
    second, _balance_again, _ = e6.match_without_replacement(
        frame, seed=20260925, stream_key="e6_match", enforce_reference_counts=False
    )
    pd.testing.assert_frame_equal(first, second)
    assert unmatched == 0
    assert first["sino_concept_id"].is_unique
    assert first["nonsino_concept_id"].is_unique
    complete = e6.complete_case_concepts(frame, expected_counts={})
    coordinates, _ = e6.standardized_covariates(complete)
    index = {cid: i for i, cid in enumerate(complete["concept_id"].astype(str))}
    for row in first.itertuples(index=False):
        difference = coordinates[index[row.sino_concept_id]] - coordinates[index[row.nonsino_concept_id]]
        assert np.max(np.abs(difference)) <= 0.2
    assert set(balance["covariate"]) == set(e6.MATCH_COVARIATES)


def test_matching_reports_unmatched_sino_when_caliper_fails() -> None:
    """A point outside any per-covariate caliper remains unmatched."""
    frame = _toy_match_frame().iloc[[0, 2]].copy()
    frame.loc[frame["concept_id"].eq("n1"), list(e6.MATCH_COVARIATES)] = [50.0, 50.0, 50.0, 50.0]
    pairs, _balance, unmatched = e6.match_without_replacement(
        frame, enforce_reference_counts=False
    )
    assert pairs.empty
    assert unmatched == 1


def test_matching_reader_refuses_any_results_path(tmp_path: Path) -> None:
    """The outcome-free match stage cannot open an input under results/."""
    result_path = tmp_path / "results" / "match.tsv"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("# metadata\nconcept_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="may not read anything under results"):
        e6.read_matching_covariates(str(result_path), repo_root=str(tmp_path))


def test_per_concept_response_matches_stored_rise_and_aggregate() -> None:
    """OPS-28-2 subtraction reproduces concept-level and aggregate rise rows."""
    rows = []
    for concept_index, concept_id in enumerate(("c1", "c2")):
        for language in e6.LANGUAGES:
            for layer in range(1, 8):
                rows.append({
                    "model": "m", "format": "translation", "concept_id": concept_id,
                    "target_lang": language,
                    "target_variant": "nospace" if language == "zh" else "released",
                    "layer": layer, "value": float(concept_index + layer * (1 + e6.LANGUAGES.index(language))),
                })
    delta = pd.DataFrame(rows)
    response = e6.per_concept_response(delta, early_depth_max=0.15)
    stored = response[[
        "model", "format", "concept_id", "target_lang", "target_variant", "layer",
    ]].copy()
    stored["value"] = response["rise"]
    assert e6.compare_with_stored_rise(response, stored) == pytest.approx(0.0)
    assert e6.assert_concept_mean_matches_aggregate(response, stored) == pytest.approx(0.0)
    stored.loc[0, "value"] += 0.01
    with pytest.raises(AssertionError, match="aggregate rise"):
        e6.assert_concept_mean_matches_aggregate(response, stored)


def test_dzh_uses_passing_layers_inside_the_inclusive_depth_window() -> None:
    """D_zh ignores a gate-failed layer and layers outside the window."""
    rows = []
    for fmt, contrast in (("repetition", 1.0), ("translation", 3.0)):
        for stratum, ids, value in (("sino", ("s1", "s2"), contrast), ("nonsino", ("n1", "n2"), 0.0)):
            for concept_id in ids:
                for layer in (1, 2, 4):  # layer 3 is gate-failed; 4 is outside this window
                    rows.append({
                        "format": fmt, "stratum": stratum, "concept_id": concept_id,
                        "layer": layer, "value": value,
                    })
    frame = pd.DataFrame(rows)
    analyze_e6.validate_la_gate_layers(
        frame, {"repetition": {3}, "translation": {3}}, n_layers=4
    )
    table, _ = e6.dzh_from_frame(frame, n_layers=4, window=(0.25, 0.75))
    per_format = table.loc[table["format"].isin(e6.FORMATS)].set_index("format")
    pooled = table.loc[table["format"].eq("pooled")].iloc[0]
    assert per_format.loc["repetition", "layers"] == "1;2"
    assert per_format.loc["translation", "layers"] == "1;2"
    assert per_format.loc["repetition", "estimate"] == pytest.approx(1.0)
    assert per_format.loc["translation", "estimate"] == pytest.approx(3.0)
    assert pooled["estimate"] == pytest.approx(2.0)


def test_stratified_and_pair_bootstrap_streams_are_deterministic() -> None:
    """Bootstrap keys reproduce the same fixed-stratum and pair resamples."""
    ids = {"sino": ["s1", "s2"], "nonsino": ["n1", "n2", "n3"]}
    draws_a = e6.stratified_draws(ids, resamples=8, seed=20260925, key="bootstrap|h3|m|pooled")
    draws_b = e6.stratified_draws(ids, resamples=8, seed=20260925, key="bootstrap|h3|m|pooled")
    draws_other_key = e6.stratified_draws(ids, resamples=8, seed=20260925, key="bootstrap|h3cc|m|pooled")
    assert draws_a == draws_b
    assert draws_a != draws_other_key
    for draw in draws_a:
        assert sum(stratum == "sino" for stratum, _ in draw) == 2
        assert sum(stratum == "nonsino" for stratum, _ in draw) == 3
    frame = pd.DataFrame({
        "concept_id": ["s1", "n1"], "stratum": ["sino", "nonsino"], "x": [1, 2],
    })
    duplicated = e6.expand_draw_groups(frame, [("sino", "s1"), ("sino", "s1")])
    assert duplicated["concept_group"].nunique() == 2
    pairs = pd.DataFrame({"pair_id": ["p1", "p2"]})
    matched_draws = e6.match_bootstrap_draws(pairs, resamples=8, seed=1, key="bootstrap|h3m|m|pooled")
    assert matched_draws == e6.match_bootstrap_draws(
        pairs, resamples=8, seed=1, key="bootstrap|h3m|m|pooled"
    )
    assert matched_draws != e6.match_bootstrap_draws(
        pairs, resamples=8, seed=1, key="bootstrap|h3m|other|pooled"
    )
    pair_values = pd.DataFrame([
        {"pair_id": pair_id, "format": fmt, "layer": layer, "difference": value}
        for pair_id, value in (("p1", 1.0), ("p2", 3.0), ("p3", 5.0))
        for fmt in e6.FORMATS
        for layer in (1, 2)
    ])
    uninterrupted = e6.dzh_pair_bootstrap(
        pair_values, resamples=12, seed=20260925, key="bootstrap|h3m|m|pooled", chunk_size=3,
    )
    resumed = e6.dzh_pair_bootstrap(
        pair_values, resamples=12, seed=20260925, key="bootstrap|h3m|m|pooled", chunk_size=3,
        completed={0: uninterrupted[:3]},
    )
    np.testing.assert_array_equal(resumed, uninterrupted)


def test_expected_s71_row_grid_counts_only_formats_present_per_concept() -> None:
    """OPS-33 grid arithmetic uses concept counts available in each format."""
    expected_by_format, total = e6.expected_s71_row_grid(
        {
            "repetition": {"sino": 2, "nonsino": 3},
            "translation": {"sino": 2, "nonsino": 2},
        },
        n_layers=6,
    )
    assert expected_by_format == {"repetition": 150, "translation": 120}
    assert total == 270


def test_bootstrap_keeps_all_available_rows_of_each_concept_together() -> None:
    """A format-missing concept is resampled with its complete observed row set."""
    frame = pd.DataFrame([
        {"concept_id": "s1", "stratum": "sino", "format": "repetition", "row": 1},
        {"concept_id": "s1", "stratum": "sino", "format": "repetition", "row": 2},
        {"concept_id": "s1", "stratum": "sino", "format": "translation", "row": 3},
        {"concept_id": "n1", "stratum": "nonsino", "format": "repetition", "row": 4},
    ])
    expanded = e6.expand_draw_groups(
        frame, [("sino", "s1"), ("nonsino", "n1"), ("sino", "s1")]
    )
    for _group, rows in expanded.groupby("concept_group", sort=False):
        assert rows["concept_id"].nunique() == 1
        source_id = str(rows["concept_id"].iloc[0])
        if source_id == "s1":
            assert rows["format"].value_counts().to_dict() == {"repetition": 2, "translation": 1}
        else:
            assert rows["format"].tolist() == ["repetition"]


def test_pair_bootstrap_skips_unavailable_format_pair_rows() -> None:
    """Matched D_zh bootstrap remains finite when one pair lacks a format prompt."""
    records = [
        {"pair_id": f"p{index:03d}", "format": fmt, "layer": layer, "difference": float(index + 1)}
        for index in range(40)
        for fmt in e6.FORMATS
        for layer in (1, 2)
    ]
    records.extend(
        {"pair_id": "p040", "format": "repetition", "layer": layer, "difference": 2.0}
        for layer in (1, 2)
    )
    records.extend(
        {"pair_id": "p040", "format": "translation", "layer": layer, "difference": float("nan")}
        for layer in (1, 2)
    )
    samples = e6.dzh_pair_bootstrap(
        pd.DataFrame.from_records(records), resamples=20, seed=20260925,
        key="bootstrap|h3m|format-missing|pooled",
    )
    assert np.isfinite(samples).all()


def test_string_false_nllb_flags_are_not_counted_as_true() -> None:
    """TSV boolean strings are parsed explicitly before forming the subset."""
    frame = pd.DataFrame({
        "zh_nllb_agree": ["true", "true", "false"],
        "fr_nllb_agree": ["true", "false", "true"],
        "id_nllb_agree": ["true", "true", "true"],
    })
    result = analyze_e6._all_true_bool_flags(
        frame, ("zh_nllb_agree", "fr_nllb_agree", "id_nllb_agree")
    )
    assert result.tolist() == [True, False, False]


def test_bootstrap_p_holm_classification_and_unresolved_examples() -> None:
    """Apply the exact plus-one p, Holm adjustment and class comparison rules."""
    assert e6.two_sided_bootstrap_p([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.4)
    assert e6.two_sided_bootstrap_p([-1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.8)
    adjusted = e6.holm_family_b(pd.DataFrame({
        "model": ["g", "g", "q", "q", "l", "l"],
        "instrument": ["D_zh", "rise", "D_zh", "rise", "D_zh", "rise"],
        "p_value": [0.01, 0.04, 0.03, 0.8, 0.6, 0.9],
    }))
    holm_by_key = adjusted.set_index(["model", "instrument"])["p_holm"].to_dict()
    assert holm_by_key == pytest.approx({
        ("g", "D_zh"): 0.06, ("g", "rise"): 0.16,
        ("l", "D_zh"): 1.0, ("l", "rise"): 1.0,
        ("q", "D_zh"): 0.15, ("q", "rise"): 1.0,
    })
    assert e6.interval_class(0.1, 0.2) == "+"
    assert e6.interval_class(-0.2, -0.1) == "−"
    assert e6.interval_class(-0.1, 0.1) == "0"
    assert e6.unresolved_from_classes({"m:D_zh": "+"}, {"m:D_zh": "0"})
    assert not e6.unresolved_from_classes({"m:D_zh": "0"}, {"m:D_zh": "0"})


def test_mixedlm_recovers_planted_interaction_and_keeps_task_format_fixed() -> None:
    """A small synthetic fit recovers the planted sino:zh interaction."""
    formula = e6.mixedlm_formula()
    assert "C(task_format)" in formula
    assert "vc_formula" not in formula
    frame = e6.make_synthetic_model_frame(
        n_sino=24, n_nonsino=24, n_layers=8, seed=41,
        synthetic_sd={"concept": 0.25, "concept_language": 0.0, "residual": 0.04},
    )
    fitted = e6.fit_mixedlm(frame, maxiter=300)
    term = e6.model_instrument(fitted)
    assert term["converged"]
    assert term["valid"]
    assert float(term["estimate"]) == pytest.approx(0.5, abs=0.15)


def _fake_fit_result(
    *, converged: bool = True, standard_error: float = 0.1, variance: float = 0.2, llf: float = -10.0
) -> SimpleNamespace:
    """Build a statsmodels-shaped result for optimizer validity-path tests."""
    interaction = "C(stratum, Treatment(reference='nonsino'))[T.sino]:C(language, Treatment(reference='en'))[T.zh]"
    sino = "C(stratum, Treatment(reference='nonsino'))[T.sino]"
    params = pd.Series({"Intercept": 0.1, sino: 0.2, interaction: 0.5}, dtype=np.float64)
    bse = pd.Series({"Intercept": 0.1, sino: 0.1, interaction: standard_error}, dtype=np.float64)
    conf = pd.DataFrame({0: params - 1.96 * bse, 1: params + 1.96 * bse})
    return SimpleNamespace(
        params=params,
        fe_params=params,
        bse_fe=bse,
        bse=bse,
        cov_re=pd.DataFrame([[variance]]),
        llf=float(llf),
        converged=converged,
        conf_int=lambda: conf,
    )


def _synthetic_recovery_fixture_fit(*, interaction: float = 0.5) -> e6.MixedLMFit:
    """Return a statsmodels-shaped fit result for recovery-gate unit tests."""
    result = _fake_fit_result(standard_error=0.01, variance=1.0)
    target, _sino = e6._interaction_terms(list(map(str, result.params.index)))
    assert target is not None
    result.params.loc[target] = float(interaction)
    result.scale = 4.0
    diagnostics = e6.FitDiagnostics(
        valid=True,
        converged=True,
        method_used="lbfgs_cold",
        tries=1,
        variance_at_boundary=False,
        warning_categories={},
        failure_reasons=(),
        elapsed_seconds=0.01,
        random_intercept_variance=1.0,
    )
    return e6.MixedLMFit(result, diagnostics)


def _run_synthetic_recovery_check(
    frame: pd.DataFrame, fitted: e6.MixedLMFit
) -> dict[str, object]:
    """Use the configured OPS-30 thresholds on a tiny synthetic fixture."""
    recovery = {
        "concept_sd_ratio_bounds": [0.5, 1.5],
        "target_wald_se_multiple": 3.0,
        "target_abs_bias_max": 0.1,
    }
    return e6.synthetic_recovery_checks(
        frame,
        fitted,
        n_layers=2,
        planted_concept_sd=1.0,
        planted_interaction=0.5,
        **recovery,
    )


def test_synthetic_recovery_passes_on_correctly_grouped_fixture() -> None:
    """The OPS-30 checks pass when synthetic rows and the fitted terms agree."""
    frame = e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=4, n_layers=2, seed=20260925,
        synthetic_sd={"concept": 1.0, "concept_language": 1.0, "residual": 2.0},
    )
    recovery = _run_synthetic_recovery_check(frame, _synthetic_recovery_fixture_fit())
    assert recovery["passed"]
    assert recovery["group_count"] == recovery["concept_count"] == 7
    assert recovery["observed_rows_per_group_min"] == recovery["observed_rows_per_group_max"] == 20


def test_synthetic_recovery_fails_for_wrong_grouping() -> None:
    """Merged groups fail the concept-count and per-group-row checks."""
    frame = e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=4, n_layers=2, seed=20260925,
    )
    frame["concept_group"] = "merged"
    recovery = _run_synthetic_recovery_check(frame, _synthetic_recovery_fixture_fit())
    assert not recovery["checks"]["group_count_matches_concepts"]
    assert not recovery["checks"]["rows_per_group_match_5xNx2"]
    assert not recovery["passed"]


def test_synthetic_recovery_fails_when_target_coefficient_is_not_recovered() -> None:
    """A target estimate outside the declared bias and Wald-SE bounds fails."""
    frame = e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=4, n_layers=2, seed=20260925,
    )
    recovery = _run_synthetic_recovery_check(frame, _synthetic_recovery_fixture_fit(interaction=0.75))
    assert not recovery["checks"]["interaction_within_3_wald_se"]
    assert not recovery["checks"]["interaction_abs_bias_below_limit"]
    assert not recovery["passed"]


def _patch_fake_mixedlm(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> list[dict[str, object]]:
    """Replace only statsmodels' model constructor with ordered fake fit results."""
    import statsmodels.formula.api as smf

    calls: list[dict[str, object]] = []

    class FakeModel:
        def fit(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            current = outcomes.pop(0)
            if isinstance(current, Exception):
                raise current
            return current

    monkeypatch.setattr(smf, "mixedlm", lambda *_args, **_kwargs: FakeModel())
    return calls


def _tiny_mixedlm_frame() -> pd.DataFrame:
    """Supply required fields without asking the statsmodels constructor to fit."""
    return e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=3, n_layers=8, seed=18,
    )


def test_mixedlm_falls_through_and_records_the_kept_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed warm-started lbfgs attempt falls through to the moment start."""
    calls = _patch_fake_mixedlm(monkeypatch, [
        _fake_fit_result(converged=False),
        _fake_fit_result(converged=True),
    ])
    fitted = e6.fit_mixedlm(_tiny_mixedlm_frame(), start_params=[1.0, 0.0])
    assert fitted.diagnostics.valid
    assert fitted.diagnostics.method_used == "lbfgs_mom"
    assert fitted.diagnostics.tries == 2
    assert calls[0]["method"] == "lbfgs"
    assert calls[0]["start_params"] == [1.0, 0.0]
    assert calls[1]["method"] == "lbfgs"
    assert calls[1]["start_params"] is not None


def test_mixedlm_nonfinite_target_standard_error_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target interaction's non-finite standard error invalidates each fit."""
    calls = _patch_fake_mixedlm(monkeypatch, [_fake_fit_result(standard_error=float("nan")) for _ in range(4)])
    fitted = e6.fit_mixedlm(_tiny_mixedlm_frame())
    assert not fitted.diagnostics.valid
    assert fitted.diagnostics.tries == 4
    assert len(calls) == 4
    assert any("standard error is not finite" in reason for reason in fitted.diagnostics.failure_reasons)


def test_mixedlm_boundary_random_intercept_is_valid_and_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A true zero-variance fixture keeps its boundary fit under OPS-31."""
    _patch_fake_mixedlm(monkeypatch, [_fake_fit_result(variance=0.0, llf=-10.0 - i) for i in range(5)])
    zero_variance_frame = e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=3, n_layers=8, seed=19,
        synthetic_sd={"concept": 0.0, "concept_language": 0.0, "residual": 2.0},
    )
    fitted = e6.fit_mixedlm(zero_variance_frame, start_params=[1.0, 0.0])
    assert fitted.diagnostics.valid
    assert fitted.diagnostics.variance_at_boundary
    assert fitted.diagnostics.random_intercept_variance == 0.0
    assert fitted.diagnostics.boundary_rule_ran
    assert not fitted.diagnostics.boundary_rule_changed_fit


def test_bootstrap_valid_draw_gate_flags_without_raising() -> None:
    """A count below 950 is represented as a failed gate while retaining the result."""
    summary = e6.bootstrap_summary(np.ones(949), min_valid_draws=950)
    assert summary["valid_draws"] == 949
    assert not summary["valid_draw_gate_passed"]


def test_mixedlm_not_estimable_after_all_optimizer_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """All optimizer attempts can fail without raising out of the analysis loop."""
    calls = _patch_fake_mixedlm(monkeypatch, [_fake_fit_result(converged=False) for _ in range(4)])
    fitted = e6.fit_mixedlm(_tiny_mixedlm_frame())
    instrument = e6.model_instrument(fitted)
    assert not instrument["valid"]
    assert instrument["method_used"] == "not_estimable"
    assert instrument["tries"] == 4
    assert len(calls) == 4


def test_moment_start_matches_hand_computed_one_way_anova() -> None:
    """The variance start equals between-group mean residual variance minus within noise."""
    import statsmodels.formula.api as smf

    frame = pd.DataFrame({
        "concept_group": np.repeat(["a", "b", "c"], 2),
        "x": [0.0, 1.0] * 3,
        "rise": [0.1, 3.9, 2.1, 5.9, -0.9, 2.9],
    })
    design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
    beta = np.linalg.lstsq(design, frame["rise"].to_numpy(), rcond=None)[0]
    residuals = frame["rise"].to_numpy() - design @ beta
    group_means = np.array([residuals[0:2].mean(), residuals[2:4].mean(), residuals[4:6].mean()])
    between = float(np.var(group_means, ddof=1))
    residual_variance = float(np.sum(residuals**2) / (len(residuals) - np.linalg.matrix_rank(design)))
    expected = max(1e-4, between - residual_variance / 2)
    start = e6._mixedlm_moment_start(None, frame, "rise ~ x")
    assert start.cov_re[0, 0] == pytest.approx(expected)
    assert start.fe_params == pytest.approx(beta)
    assert smf.ols("rise ~ x", data=frame).fit().scale == pytest.approx(residual_variance)


def test_generalized_moment_start_matches_unequal_group_anova() -> None:
    """OPS-34 uses the effective-size ANOVA estimate for unbalanced groups."""
    frame = pd.DataFrame({
        "concept_group": ["a"] * 2 + ["b"] * 3 + ["c"] * 4,
        "x": [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0],
        "rise": [0.2, 1.4, 2.3, 3.1, 4.8, -1.1, 0.4, 2.8, 6.2],
    })
    design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
    beta = np.linalg.lstsq(design, frame["rise"].to_numpy(), rcond=None)[0]
    residuals = frame["rise"].to_numpy() - design @ beta
    sizes = np.array([2.0, 3.0, 4.0])
    means = np.array([
        residuals[:2].mean(), residuals[2:5].mean(), residuals[5:].mean(),
    ])
    grand = residuals.mean()
    msb = float(np.sum(sizes * (means - grand) ** 2) / 2)
    within = np.concatenate([
        residuals[:2] - means[0], residuals[2:5] - means[1], residuals[5:] - means[2],
    ])
    msw = float(np.sum(within**2) / (len(frame) - 3))
    n0 = (len(frame) - float(np.sum(sizes**2)) / len(frame)) / 2
    expected = max(1e-4, (msb - msw) / n0)
    start = e6.generalized_moment_start(None, frame, "rise ~ x")
    assert start.cov_re[0, 0] == pytest.approx(expected)
    assert start.fe_params == pytest.approx(beta)


def test_generalized_moment_start_reduces_to_ops31_for_equal_groups() -> None:
    """Equal group sizes yield the OPS-31 one-way ANOVA start."""
    frame = pd.DataFrame({
        "concept_group": np.repeat(["a", "b", "c"], 2),
        "x": [0.0, 1.0] * 3,
        "rise": [0.1, 3.9, 2.1, 5.9, -0.9, 2.9],
    })
    equal = e6._mixedlm_moment_start(None, frame, "rise ~ x")
    generalized = e6.generalized_moment_start(None, frame, "rise ~ x")
    assert generalized.cov_re[0, 0] == pytest.approx(equal.cov_re[0, 0], abs=1e-12)
    assert generalized.fe_params == pytest.approx(equal.fe_params)


def _rank_fixture(*, omit_sino_zh: bool) -> pd.DataFrame:
    """Build covariate-only rows with either a complete or absent sino/zh cell."""
    rng = np.random.default_rng(20260925)
    rows = []
    for stratum in ("nonsino", "sino"):
        for language in e6.LANGUAGES:
            if omit_sino_zh and stratum == "sino" and language == "zh":
                continue
            for task_format in e6.FORMATS:
                for layer in range(1, 9):
                    rows.append({
                        "language": language, "depth": layer / 8,
                        "stratum": stratum, "task_format": task_format,
                        "zipf_en": float(rng.normal()), "n_tokens": int(rng.integers(1, 12)),
                        "n_syllables": float(rng.uniform(1, 5)),
                        "concreteness": float(rng.normal()),
                    })
    return pd.DataFrame.from_records(rows)


def test_fixed_design_estimability_flags_missing_sino_zh_cell() -> None:
    """The §7.1 interaction contrast is outside the row space when its cell is absent."""
    result = e6.fixed_design_estimability(_rank_fixture(omit_sino_zh=True))
    assert not result["interaction_estimable"]
    assert not result["sum_effect_estimable"]
    assert result["nullity"] > 0
    assert result["interaction_estimability_residual"] > 1e-8


def test_fixed_design_estimability_accepts_estimable_contrasts() -> None:
    """Both §7.1 contrasts are estimable when all stratum/language cells exist."""
    result = e6.fixed_design_estimability(_rank_fixture(omit_sino_zh=False))
    assert result["interaction_estimable"]
    assert result["sum_effect_estimable"]
    assert result["interaction_estimability_residual"] <= 1e-8
    assert result["sum_effect_estimability_residual"] <= 1e-8


@pytest.mark.parametrize("omit_sino_zh", [False, True])
def test_economy_svd_matches_direct_full_svd(
    omit_sino_zh: bool,
) -> None:
    """Economy SVD preserves rank and contrast estimability from the full factorization."""
    import patsy

    frame = _rank_fixture(omit_sino_zh=omit_sino_zh)
    result = e6.fixed_design_estimability(frame)
    design_columns = [
        "language", "depth", "stratum", "task_format", "zipf_en",
        "n_tokens", "n_syllables", "concreteness",
    ]
    design_data = frame.loc[:, design_columns].drop_duplicates().reset_index(drop=True)
    rhs = e6.mixedlm_formula().split("~", 1)[1]
    values = patsy.dmatrix(rhs, design_data, return_type="dataframe").to_numpy(dtype=np.float64)
    _u, singular_values, vh = np.linalg.svd(values, full_matrices=True)
    cutoff = max(values.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    null_basis = vh[rank:, :].T
    interaction, sino = e6._interaction_terms(list(map(str, patsy.dmatrix(rhs, design_data).design_info.column_names)))

    def residual(weights: dict[str, float]) -> float:
        columns = list(map(str, patsy.dmatrix(rhs, design_data).design_info.column_names))
        contrast = np.zeros(len(columns), dtype=np.float64)
        for name, weight in weights.items():
            contrast[columns.index(name)] = weight
        norm = float(np.linalg.norm(contrast))
        return float(np.linalg.norm(null_basis.T @ contrast) / norm) if null_basis.size else 0.0

    assert result["rank"] == rank
    assert result["interaction_estimability_residual"] == pytest.approx(residual({interaction: 1.0}), abs=1e-12)
    assert result["sum_effect_estimability_residual"] == pytest.approx(residual({sino: 1.0, interaction: 1.0}), abs=1e-12)


def test_closeout_fit_sets_are_processed_sequentially_when_worker_count_is_one() -> None:
    """Closeout callback execution is ordered and never overlaps with one worker."""
    active = 0
    peak_active = 0
    events: list[tuple[str, int]] = []

    def process(value: int) -> int:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        events.append(("start", value))
        active -= 1
        events.append(("end", value))
        return value * 2

    results = analyze_e6._closeout_serial_map(iter((1, 2, 3)), process, workers=1)
    assert results == [2, 4, 6]
    assert peak_active == 1
    assert events == [("start", 1), ("end", 1), ("start", 2), ("end", 2), ("start", 3), ("end", 3)]


def test_closeout_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    """A previous closeout artifact blocks any new closeout write."""
    closeout_dir = tmp_path / "closeout"
    closeout_dir.mkdir()
    (closeout_dir / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        analyze_e6.ensure_closeout_outputs_absent(closeout_dir, tmp_path / "E6_closeout.csv")
    report = tmp_path / "E6_closeout.csv"
    report.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        analyze_e6.ensure_closeout_outputs_absent(tmp_path / "empty", report)


def test_moment_refit_reproduces_stored_llf_and_refuses_overwrite(tmp_path: Path) -> None:
    """OPS-36 refits the stored moment candidate and protects existing outputs."""
    frame = e6.make_synthetic_model_frame(
        n_sino=12, n_nonsino=12, n_layers=6, seed=3601,
        synthetic_sd={"concept": 1.0, "concept_language": 1.0, "residual": 2.0},
    )
    first = analyze_e6._fit_generalized_moment_candidate(
        frame, frequency_covariate="zipf_en", maxiter=500,
    )
    assert first["llf_mom"] is not None
    stored_llf = float(first["llf_mom"])
    second = analyze_e6._fit_generalized_moment_candidate(
        frame, frequency_covariate="zipf_en", maxiter=500,
    )
    assert second["llf_mom"] is not None
    assert analyze_e6._moment_refit_llf_matches(second["llf_mom"], stored_llf)

    result_path = tmp_path / "moment_refit.json"
    report_path = tmp_path / "E6_moment_refit.csv"
    sidecar_path = tmp_path / "moment_refit.provenance.json"
    analyze_e6.ensure_moment_refit_outputs_absent(result_path, report_path, sidecar_path)
    report_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exist"):
        analyze_e6.ensure_moment_refit_outputs_absent(result_path, report_path, sidecar_path)


def test_moment_refit_mode_writes_reproduced_synthetic_rows_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OPS-36 mode records matching tiny synthetic llfs and refuses a second write."""
    repo = tmp_path
    for relative in ("results/analysis/primary/e6/closeout", "reports"):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": "qwen", "fit_set": "full", "moment_valid": True, "llf_mom": -12.0},
        {"model": "qwen", "fit_set": "sensitivity:min_surface_dist_ge_060", "moment_valid": True, "llf_mom": -22.0},
    ]).to_csv(repo / "reports/E6_closeout.csv", index=False)
    pd.DataFrame([{
        "model": "qwen", "instrument": "rise_zh_interaction", "estimate": 0.5,
        "wald_ci_low": 0.3, "wald_ci_high": 0.7, "llf_kept": -13.0,
    }]).to_csv(repo / "reports/E6_model_terms.csv", index=False)
    pd.DataFrame([{
        "model": "qwen", "sensitivity": "min_surface_dist_ge_060",
        "instrument": "rise_zh_interaction", "estimate": 0.4,
        "ci_low": 0.2, "ci_high": 0.6, "llf_kept": -23.0,
    }]).to_csv(repo / "reports/E6_sensitivity.csv", index=False)
    frame = pd.DataFrame({"concept_id": ["s1", "n1"], "concept_group": ["s1", "n1"]})
    monkeypatch.setattr(analyze_e6, "_load_response_inputs", lambda *_args: (frame.copy(), None, None, None, 0.0))
    monkeypatch.setattr(
        analyze_e6, "_sensitivity_model_frames",
        lambda *_args: iter((
            ("min_surface_dist_ge_060", "zipf_en", frame.copy()),
        )),
    )
    llfs = iter((-12.0, -22.0))

    def fake_candidate(*_args: object, **_kwargs: object) -> dict[str, object]:
        llf = next(llfs)
        result = _fake_fit_result(llf=llf)
        names = list(map(str, result.fe_params.index))
        result.cov_params = lambda: pd.DataFrame(
            np.eye(len(names), dtype=np.float64) * 0.01, index=names, columns=names,
        )
        return {
            "valid": True, "converged": True, "method": "lbfgs_mom_unequal",
            "variance_at_boundary": False, "random_intercept_variance": 1.0,
            "llf_mom": llf, "failure_reasons": [], "warning_categories": {},
            "seconds": 0.01, "result": result,
        }

    monkeypatch.setattr(analyze_e6, "_fit_generalized_moment_candidate", fake_candidate)
    monkeypatch.setattr(analyze_e6, "study_models", lambda _cfg: {"qwen": {"hf_id": "qwen/test", "revision": "rev"}})
    monkeypatch.setattr(analyze_e6, "build_provenance", lambda *_args, **_kwargs: {"stub": True})
    cfg = {"e6": {
        "sensitivity_variants": ["min_surface_dist_ge_060"],
        "moment_refit_json": "results/analysis/primary/e6/closeout/moment_refit.json",
        "maxiter": 500,
        "outputs": {
            "moment_refit": "reports/E6_moment_refit.csv",
            "closeout": "reports/E6_closeout.csv",
            "model_terms": "reports/E6_model_terms.csv",
            "sensitivity": "reports/E6_sensitivity.csv",
        },
    }}
    table, passed = analyze_e6.run_moment_refits(
        cfg, repo, logging.getLogger("e6-moment-refit-fixture"), SimpleNamespace(),
        "qwen:full,qwen:sensitivity:min_surface_dist_ge_060",
    )
    assert passed
    assert len(table) == 2
    assert table["llf_reproduction_match"].tolist() == [True, True]
    assert (repo / "reports/E6_moment_refit.csv").is_file()
    assert (repo / "results/analysis/primary/e6/closeout/moment_refit.json").is_file()
    with pytest.raises(FileExistsError, match="already exist"):
        analyze_e6.run_moment_refits(
            cfg, repo, logging.getLogger("e6-moment-refit-fixture"), SimpleNamespace(),
            "qwen:full,qwen:sensitivity:min_surface_dist_ge_060",
        )


def test_boundary_rule_runs_remaining_methods_and_keeps_highest_llf(monkeypatch: pytest.MonkeyPatch) -> None:
    """A boundary first-valid fit triggers the chain and is replaced by higher REML llf."""
    _patch_fake_mixedlm(monkeypatch, [
        _fake_fit_result(variance=0.0, llf=-120.0),
        _fake_fit_result(variance=0.8, llf=-100.0),
        _fake_fit_result(variance=0.7, llf=-110.0),
        _fake_fit_result(variance=0.6, llf=-115.0),
        _fake_fit_result(variance=0.5, llf=-116.0),
    ])
    fitted = e6.fit_mixedlm(_tiny_mixedlm_frame(), start_params=[1.0, 0.0])
    assert fitted.diagnostics.valid
    assert fitted.diagnostics.method_used == "lbfgs_mom"
    assert fitted.diagnostics.tries == 5
    assert fitted.diagnostics.boundary_rule_ran
    assert fitted.diagnostics.boundary_rule_changed_fit
    assert fitted.diagnostics.llf_first_valid == pytest.approx(-120.0)
    assert fitted.diagnostics.llf_kept == pytest.approx(-100.0)


def test_boundary_rule_tie_break_keeps_earlier_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """REML llf candidates within the configured tolerance keep the earlier method."""
    _patch_fake_mixedlm(monkeypatch, [
        _fake_fit_result(variance=0.0, llf=-100.0),
        _fake_fit_result(variance=0.0, llf=-100.0 + 0.5e-6),
        _fake_fit_result(variance=0.0, llf=-101.0),
        _fake_fit_result(variance=0.0, llf=-102.0),
        _fake_fit_result(variance=0.0, llf=-103.0),
    ])
    fitted = e6.fit_mixedlm(_tiny_mixedlm_frame(), start_params=[1.0, 0.0])
    assert fitted.diagnostics.method_used == "lbfgs"
    assert fitted.diagnostics.llf_kept == pytest.approx(-100.0)
    assert not fitted.diagnostics.boundary_rule_changed_fit


def test_ops31_recovery_refit_gate_requires_19_valid_recovered_fits() -> None:
    """OPS-31 requires at least 19 of 20 valid refits with recovered concept SD."""
    records = [
        {"valid": True, "random_intercept_variance": 1.0, "fit_seconds": 0.1}
        for _ in range(19)
    ] + [{"valid": False, "random_intercept_variance": 0.0, "fit_seconds": 0.2}]
    passed = e6.synthetic_refit_recovery_summary(
        records, planted_concept_sd=1.0, concept_sd_ratio_bounds=[0.5, 1.5],
        expected_refits=20, min_valid_refits=19,
    )
    assert passed["valid_refit_count"] == 19
    assert passed["recovered_refit_count"] == 19
    assert passed["gate_passed"]
    records[0] = {"valid": True, "random_intercept_variance": 0.0, "fit_seconds": 0.2}
    records[-1] = {"valid": True, "random_intercept_variance": 0.0, "fit_seconds": 0.2}
    failed = e6.synthetic_refit_recovery_summary(
        records, planted_concept_sd=1.0, concept_sd_ratio_bounds=[0.5, 1.5],
        expected_refits=20, min_valid_refits=19,
    )
    assert failed["valid_refit_count"] == 20
    assert failed["recovered_refit_count"] == 18
    assert not failed["gate_passed"]


def test_synthetic_generator_plants_configured_random_effect_scales() -> None:
    """Synthetic random intercept, concept-language term and residual use configured SDs."""
    sd = {"concept": 1.0, "concept_language": 1.0, "residual": 2.0}
    frame = e6.make_synthetic_model_frame(
        n_sino=100, n_nonsino=100, n_layers=2, seed=20260925, synthetic_sd=sd,
    )
    concept_sd = frame.drop_duplicates("concept_id")["synthetic_random_intercept"].std(ddof=1)
    language_sd = frame.drop_duplicates(["concept_id", "language"])["synthetic_concept_language"].std(ddof=1)
    residual_sd = frame["synthetic_residual"].std(ddof=1)
    assert concept_sd == pytest.approx(1.0, abs=0.15)
    assert language_sd == pytest.approx(1.0, abs=0.08)
    assert residual_sd == pytest.approx(2.0, abs=0.07)


def test_synthetic_generator_supports_concepts_with_one_format() -> None:
    """Production-shape fixtures can omit a format for selected concepts."""
    frame = e6.make_synthetic_model_frame(
        n_sino=3, n_nonsino=4, n_layers=2, seed=20260925,
        format_concept_counts={
            "repetition": {"sino": 3, "nonsino": 4},
            "translation": {"sino": 3, "nonsino": 3},
        },
    )
    assert len(frame.loc[frame["format"].eq("repetition")]) == 7 * 5 * 2
    assert len(frame.loc[frame["format"].eq("translation")]) == 6 * 5 * 2
    assert set(frame.loc[frame["concept_id"].eq("concept_00006"), "format"]) == {"repetition"}


def test_summary_writer_blocks_raw_concreteness_column(tmp_path: Path) -> None:
    """Summary writers reject a raw concrete covariate column."""
    with pytest.raises(ValueError, match="not concreteness"):
        analyze_e6._write_csv(pd.DataFrame({"concreteness": [0.1]}), tmp_path / "bad.csv")
    output = tmp_path / "balance.csv"
    analyze_e6._write_csv(pd.DataFrame({"covariate": ["concreteness"], "SMD_after": [0.0]}), output)
    assert output.is_file()
    assert "0.1" not in output.read_text(encoding="utf-8")


def test_primary_requires_confirmation_and_both_integrity_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary mode refuses an omitted confirmation and propagates a failed lock."""
    with pytest.raises(SystemExit):
        analyze_e6.main(["--source", "primary", "--stages", "match"])
    monkeypatch.setattr(analyze_e6, "ensure_adjudication_committed", lambda _repo: (_ for _ in ()).throw(RuntimeError("lock closed")))
    with pytest.raises(RuntimeError, match="lock closed"):
        analyze_e6._ensure_primary_locks(Path.cwd(), {"prereg": {}})


def test_preflight_projection_never_requests_outcome_values(tmp_path: Path) -> None:
    """Metadata-only LA selection accepts its null variant and omits values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({
        "format": ["repetition"], "concept_id": ["c1"], "target_lang": ["zh"],
        "target_variant": ["nospace"], "layer": [1], "metric": ["LA"], "value": [0.5],
    }), path)
    frame = analyze_e6.read_parquet_columns(path, analyze_e6.PREFLIGHT_LA_COLUMNS)
    assert "value" not in frame.columns
    assert "value" not in analyze_e6.PREFLIGHT_LA_COLUMNS
    rows = pd.DataFrame({
        "format": ["repetition", "translation", "cloze"],
        "concept_id": ["c1", "c2", "c3"],
        "target_lang": ["zh", "en", "zh"],
        "target_variant": [None, None, None],
        "layer": [1, 1, 1],
        "metric": ["LA", "LA", "LA"],
    })
    selected = analyze_e6._select_zh_la_rows(rows)
    assert selected["concept_id"].tolist() == ["c1"]


def test_job_preflight_only_does_not_enter_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preflight-only invocation returns before timing or analysis execution."""
    import job_analyze_e6
    from vilens.preflight import Check

    monkeypatch.setattr(job_analyze_e6, "_run_synthetic_timing", lambda *_args: (_ for _ in ()).throw(AssertionError("timing ran")))
    monkeypatch.setattr(job_analyze_e6, "_preflight", lambda *_args, **_kwargs: [Check("synthetic", "PASS", "fixture")])
    monkeypatch.setattr(job_analyze_e6, "_logger", lambda: __import__("logging").getLogger("e6-test"))
    assert job_analyze_e6.main(["--source", "synthetic", "--preflight-only"]) == 0


def test_primary_closeout_mode_forwards_closeout_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job wrapper routes approved primary closeout into the analysis entrypoint."""
    import job_analyze_e6
    from vilens.preflight import Check

    monkeypatch.setattr(job_analyze_e6, "_preflight", lambda *_args, **_kwargs: [Check("fixture", "PASS", "ok")])
    monkeypatch.setattr(job_analyze_e6, "_logger", lambda: __import__("logging").getLogger("e6-closeout-test"))
    forwarded: list[list[str]] = []
    monkeypatch.setattr(analyze_e6, "main", lambda args: forwarded.append(list(args)) or 0)
    assert job_analyze_e6.main([
        "--source", "primary", "--confirm-e6", "--closeout",
    ]) == 0
    assert forwarded and "--closeout" in forwarded[0]
    assert "--stages" not in forwarded[0]


def test_primary_moment_refit_mode_forwards_requested_fit_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job wrapper forwards the OPS-36 selection without starting all stages."""
    import job_analyze_e6
    from vilens.preflight import Check

    monkeypatch.setattr(job_analyze_e6, "_preflight", lambda *_args, **_kwargs: [Check("fixture", "PASS", "ok")])
    monkeypatch.setattr(job_analyze_e6, "_logger", lambda: __import__("logging").getLogger("e6-moment-refit-test"))
    forwarded: list[list[str]] = []
    monkeypatch.setattr(analyze_e6, "main", lambda args: forwarded.append(list(args)) or 0)
    spec = "qwen:full,qwen:sensitivity:min_surface_dist_ge_060"
    assert job_analyze_e6.main([
        "--source", "primary", "--confirm-e6", "--moment-refit", spec,
    ]) == 0
    assert forwarded and "--moment-refit" in forwarded[0]
    assert forwarded[0][forwarded[0].index("--moment-refit") + 1] == spec
    assert "--stages" not in forwarded[0]


def test_synthetic_validation_exit_code_requires_99_of_100() -> None:
    """The validation job returns non-zero when any family misses 99 valid draws."""
    import job_analyze_e6

    assert job_analyze_e6._validation_exit_code({"gemma": 99, "qwen": 100, "llama": 99}, 99) == 0
    assert job_analyze_e6._validation_exit_code({"gemma": 98, "qwen": 100, "llama": 100}, 99) != 0


def test_e6_provenance_preflight_constructs_all_unit_records_without_writes(tmp_path: Path) -> None:
    """Construct the full primary E6 provenance inventory without writing outputs."""
    repo = _committed_e6_smoke_repo(tmp_path / "provenance-repo")
    cfg = load_config(repo / "configs/experiments.yaml")
    before = {path.relative_to(repo) for path in repo.rglob("*")}

    records = analyze_e6.construct_provenance_preflight_records(cfg, repo)

    labels = {label for label, _payload in records}
    assert len(records) >= 50
    assert any(label.endswith(":dzh_chunk") for label in labels)
    assert any(label.endswith(":model_full_fit") for label in labels)
    assert any(label.endswith(":bootstrap_chunk") for label in labels)
    assert any(label.endswith(":matched_dzh_chunk") for label in labels)
    assert any(label.endswith(":matched_model_chunk") for label in labels)
    assert any(label.endswith(":sensitivity_fit:zipf_vi") for label in labels)
    assert any(label.endswith(":sensitivity_chunk:R11_excluded") for label in labels)
    assert any(label.endswith(":report") for label in labels)
    for _label, payload in records:
        assert "scripts/analyze_e6.py" in payload["code_tree"]
        assert payload["code_sha256"]
    assert {path.relative_to(repo) for path in repo.rglob("*")} == before


def test_primary_e6_stages_smoke_end_to_end_with_resume_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run every primary E6 stage twice on tiny synthetic data and verify chunks resume."""
    repo = _committed_e6_smoke_repo(tmp_path / "primary-smoke")
    cfg = copy.deepcopy(load_config(repo / "configs/experiments.yaml"))
    cfg["paths"]["manifest"] = "data/MANIFEST.sha256"
    cfg["e6"]["bootstrap_resamples"] = 5
    cfg["e6"]["bootstrap_chunk"] = 2
    cfg["e6"]["workers"] = 2
    cfg["e6"]["min_valid_draws"] = 3
    cfg["e6"]["output_dir"] = "results/analysis/primary/e6"
    cfg["e6"]["report"] = "reports/E6.md"
    cfg["e6"]["outputs"].update({
        "dzh": "reports/E6_dzh.csv",
        "model_terms": "reports/E6_model_terms.csv",
        "holm": "reports/E6_holm.csv",
        "matched": "reports/E6_matched.csv",
        "sensitivity": "reports/E6_sensitivity.csv",
        "matched_pairs": "reports/E6_matched_pairs.csv",
        "balance": "reports/E6_balance.csv",
    })

    synthetic = e6.make_synthetic_model_frame(
        n_sino=40,
        n_nonsino=40,
        n_layers=6,
        seed=int(cfg["seed"]),
        synthetic_sd={"concept": 1.0, "concept_language": 1.0, "residual": 2.0},
        format_concept_counts={
            "repetition": {"sino": 40, "nonsino": 40},
            "translation": {"sino": 40, "nonsino": 39},
        },
    )
    missing_translation = synthetic.loc[
        synthetic["concept_id"].eq("concept_00079") & synthetic["format"].eq("translation")
    ]
    assert missing_translation.empty
    ids = synthetic[["concept_id", "stratum"]].drop_duplicates().reset_index(drop=True)
    ids["zipf_vi"] = np.linspace(-1.0, 1.0, len(ids))
    ids["zipf_zh"] = np.cos(np.linspace(0.0, 4.0, len(ids)))
    ids["native_strict"] = ids["stratum"].eq("nonsino")
    for column in ("bt_pass_strict_v1", "external_attested", "all_three_nllb", "min_surface_dist_ge_060"):
        ids[column] = True
    metadata = ids.copy()
    for column in ("zipf_vi", "zipf_zh"):
        synthetic[column] = synthetic["concept_id"].map(ids.set_index("concept_id")[column])

    rng = np.random.default_rng(20260925)
    la = synthetic[["concept_id", "stratum", "format", "layer"]].drop_duplicates().copy()
    la["value"] = rng.normal(size=len(la)) + la["stratum"].eq("sino").astype(float) * 0.2
    la["model"] = "fixture/model"
    la["target_lang"] = "zh"
    la["target_variant"] = "nospace"
    la["metric"] = "LA"
    sino_ids = ids.loc[ids["stratum"].eq("sino"), "concept_id"].tolist()
    nonsino_ids = ids.loc[ids["stratum"].eq("nonsino"), "concept_id"].tolist()
    nonsino_ids = [nonsino_ids[-1], *nonsino_ids[:-1]]
    pairs = pd.DataFrame({
        "pair_id": [f"pair_{index:03d}" for index in range(20)],
        "sino_concept_id": sino_ids[:20],
        "nonsino_concept_id": nonsino_ids[:20],
    })
    assert pairs.loc[0, "nonsino_concept_id"] == "concept_00079"

    monkeypatch.setattr(analyze_e6, "REPO", repo)
    smoke_families = ("gemma", "qwen", "llama")
    monkeypatch.setattr(analyze_e6, "FAMILIES", smoke_families)
    monkeypatch.setattr(analyze_e6, "_family_n_layers", lambda _cfg, _family: 6)
    monkeypatch.setattr(analyze_e6, "load_config", lambda _path: cfg)
    monkeypatch.setattr(analyze_e6, "verify_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(analyze_e6, "_ensure_primary_locks", lambda *_args: None)
    monkeypatch.setattr(analyze_e6, "_ensure_match_files_committed", lambda *_args: None)
    monkeypatch.setattr(analyze_e6, "_load_passing_la_zh", lambda *_args: la.copy())
    monkeypatch.setattr(
        analyze_e6,
        "_load_response_inputs",
        lambda *_args: (synthetic.copy(), pd.DataFrame(), metadata.copy(), pd.DataFrame(), 0.0),
    )
    monkeypatch.setattr(analyze_e6, "_main_strata_metadata", lambda *_args, **_kwargs: metadata.copy())
    monkeypatch.setattr(analyze_e6, "_assert_matching_committed", lambda *_args: pairs.copy())
    monkeypatch.setattr(
        analyze_e6,
        "study_models",
        lambda _cfg: {
            family: {"hf_id": f"fixture/{family}", "revision": "a" * 40}
            for family in smoke_families
        },
    )
    monkeypatch.setattr(analyze_e6, "setup_logger", lambda: logging.getLogger("e6-primary-smoke"))
    argv = [
        "--config", "configs/experiments.yaml", "--source", "primary", "--confirm-e6",
        "--stages", "dzh", "model", "matched", "sensitivity", "report",
    ]
    assert analyze_e6.main(argv) == 0

    report_files = [
        repo / cfg["e6"]["report"],
        *(repo / cfg["e6"]["outputs"][name] for name in ("dzh", "model_terms", "holm", "matched", "sensitivity")),
        repo / "results/analysis/primary/e6/dzh_p_values.csv",
        repo / "results/analysis/primary/e6/model_p_values.csv",
    ]
    assert all(path.is_file() for path in report_files)
    archives = sorted((repo / cfg["e6"]["output_dir"] / Path("checkpoints")).rglob("*.npz"))
    assert archives
    assert all(path.with_suffix(".provenance.json").is_file() for path in archives)

    expected_columns = {
        "dzh": [
            "format", "instrument", "estimate", "n_sino", "n_nonsino", "layers", "ci_low", "ci_high",
            "valid_draws", "p_value", "model", "bootstrap_key", "result_class",
        ],
        "model_terms": [
            "model", "instrument", "term", "estimate", "wald_ci_low", "wald_ci_high", "standard_error",
            "converged", "fit_valid", "fit_method", "fit_tries", "variance_at_boundary", "boundary_rule_ran",
            "boundary_rule_changed_fit", "llf_kept", "llf_first_valid", "random_intercept_variance",
            "fit_warning_categories", "fit_failure_reasons", "full_fit_seconds", "zh_specific_stratum_effect",
            "n_sino", "n_nonsino", "bootstrap_key", "rise_main_max_abs_diff", "bootstrap_ci_low",
            "bootstrap_ci_high", "bootstrap_p", "result_class", "valid_draws", "nonconverged_draws",
            "valid_draw_gate_passed", "bootstrap_method_mix", "bootstrap_boundary_count",
            "bootstrap_boundary_rule_trigger_count", "bootstrap_boundary_rule_changed_fit_count",
            "bootstrap_warning_categories", "bootstrap_mean_fit_seconds", "bootstrap_mean_tries",
        ],
        "holm": ["model", "instrument", "p_value", "p_holm", "reject_0_05"],
        "matched": [
            "model", "instrument", "estimate", "ci_low", "ci_high", "valid_draws", "bootstrap_key",
            "result_class", "standard_error", "wald_ci_low", "wald_ci_high", "nonconverged_draws",
            "valid_draw_gate_passed", "bootstrap_method_mix", "bootstrap_boundary_count",
            "bootstrap_boundary_rule_trigger_count", "bootstrap_boundary_rule_changed_fit_count",
            "bootstrap_warning_categories", "fit_valid", "fit_method", "fit_tries", "variance_at_boundary",
            "random_intercept_variance", "boundary_rule_ran", "boundary_rule_changed_fit", "llf_kept",
            "llf_first_valid", "fit_warning_categories", "fit_failure_reasons", "full_result_class",
            "class_changed", "H3_unresolved",
        ],
        "sensitivity": [
            "model", "sensitivity", "instrument", "estimate", "ci_low", "ci_high", "standard_error",
            "converged", "fit_valid", "fit_method", "fit_tries", "variance_at_boundary",
            "random_intercept_variance", "boundary_rule_ran", "boundary_rule_changed_fit", "llf_kept",
            "llf_first_valid", "fit_warning_categories", "fit_failure_reasons", "n_rows", "valid_draws",
            "p_value", "bootstrap_key",
        ],
    }
    for name, expected in expected_columns.items():
        frame = pd.read_csv(repo / cfg["e6"]["outputs"][name])
        assert list(frame.columns) == expected
        assert not any("concreteness" in str(column).lower() for column in frame.columns)

    saved_calls: list[str] = []
    original_save = analyze_e6.BootstrapChunkStore.save
    original_save_values = analyze_e6.BootstrapChunkStore.save_values

    def track_save(self: Any, *args: Any, **kwargs: Any) -> None:
        saved_calls.append("fit")
        original_save(self, *args, **kwargs)

    def track_save_values(self: Any, *args: Any, **kwargs: Any) -> None:
        saved_calls.append("values")
        original_save_values(self, *args, **kwargs)

    monkeypatch.setattr(analyze_e6.BootstrapChunkStore, "save", track_save)
    monkeypatch.setattr(analyze_e6.BootstrapChunkStore, "save_values", track_save_values)
    assert analyze_e6.main(argv) == 0
    assert saved_calls == []


def test_e6_job_preflight_marks_provenance_construction_failure_as_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface any provenance construction error as a preflight failure."""
    import job_analyze_e6

    repo = tmp_path / "preflight-repo"
    for relative in ("configs", "results", "reports"):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(load_config(REPO_ROOT / "configs/experiments.yaml"))
    cfg["e6"]["output_dir"] = "results/primary-e6"
    cfg["e6"]["synthetic_dir"] = "results/synthetic-e6"
    monkeypatch.setattr(job_analyze_e6, "REPO", repo)
    monkeypatch.setattr(job_analyze_e6, "git_state", lambda _repo: ("fixture", False))
    monkeypatch.setattr(
        analyze_e6,
        "construct_provenance_preflight_records",
        lambda *_args: (_ for _ in ()).throw(ValueError("fixture provenance error")),
    )

    checks = job_analyze_e6._preflight(cfg, "synthetic", gpu=False)

    provenance_check = next(check for check in checks if check.name == "e6_provenance_records")
    assert provenance_check.status == "FAIL"
    assert "fixture provenance error" in provenance_check.detail

