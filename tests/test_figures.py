"""Synthetic sidecar and plotting smoke tests for every paper figure."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from scripts.figures import (
    FIGURE_FUNCTIONS,
    FIGURE_SPECS,
    FigureContext,
    MODEL_ORDER,
    verify_sidecars,
)


def _unique(*parts: object) -> float:
    """Return a deterministic, label-specific float for correspondence fixtures."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return 0.05 + 0.9 * integer / float(2**64 - 1)


def _signed(*parts: object) -> float:
    """Return a deterministic signed float for projection fixtures."""
    return 2.0 * _unique(*parts) - 1.0


def _write_csv(root: Path, name: str, rows: list[dict[str, Any]]) -> None:
    """Write a tiny aggregate source table."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_reports(root: Path) -> None:
    """Create distinct source values for each model, cell, set, variant, layer, and language."""
    reports = root / "reports"
    formats = ("repetition", "translation")

    e2_rows: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for readout in ("vi", "en", "zh"):
                e2_rows.append(
                    {
                        "model": family,
                        "format": fmt,
                        "readout": readout,
                        "max_mean_P_M1": _unique("e2", family, fmt, readout),
                    }
                )
            for threshold in (.05, .10, .20):
                for units in ("blocks", "depth"):
                    for contrast in ("instrument_effect", "selection_effect", "extension_inclusive_instrument_effect"):
                        value = _signed("c1", family, fmt, threshold, units, contrast)
                        c1_rows.append(dict(model=family, format=fmt, item_set="m1_main", contrast=contrast,
                                            threshold=threshold, units=units, metric=f"{contrast}_{threshold}_{units}",
                                            value=value, ci_low=value-.03, ci_high=value+.03))
    # Cloze and extension-inclusive rows have independent synthetic values.
    for row in list(c1_rows):
        if row["format"] == "repetition":
            clone = dict(row, format="cloze")
            clone["value"] = _signed("cloze", row["model"], row["metric"])
            clone["ci_low"], clone["ci_high"] = clone["value"]-.03, clone["value"]+.03
            c1_rows.append(clone)
    _write_csv(reports, "E2-E5_e2.csv", e2_rows)
    _write_csv(reports, "E2-E5_c1.csv", c1_rows)

    curves: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for language in ("vi", "en", "zh", "fr", "id"):
                for layer in (1, 2, 3):
                    for metric in ("rise_cf729", "LA"):
                        value = _signed("curve", family, fmt, language, layer, metric)
                        curves.append(
                            {
                                "model": family,
                                "format": fmt,
                                "target_lang": language,
                                "metric": metric,
                                "layer": layer,
                                "value": value,
                                "ci_low": value - 0.02,
                                "ci_high": value + 0.02,
                            }
                        )
            for metric in ("L_peak", "L_concept_10", "L_cross"):
                value = 10 + 80 * _unique("landmark", family, fmt, metric)
                landmarks.append(
                    {
                        "model": family,
                        "format": fmt,
                        "condition": "diac",
                        "metric_variant": "s",
                        "item_set": "cf729",
                        "metric": metric,
                        "value": value,
                        "ci_low": value - 1,
                        "ci_high": value + 1,
                        "n_layers": 100,
                    }
                )
            for layer in (1, 2, 3):
                gates.append(
                    {
                        "family": family,
                        "fit_format": fmt,
                        "layer": layer,
                        "gate_pass": str(layer != 1),
                        "label_dominated_last": str(layer == 2),
                        "label_dominated_mean": str(layer == 3),
                    }
                )
    _write_csv(reports, "E2-E5_la_curves.csv", curves)
    _write_csv(reports, "E2-E5_landmarks.csv", landmarks)
    _write_csv(reports, "E5_gate.csv", gates)

    dzh_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivities = (
        "bt_pass_strict_v1",
        "external_attested",
        "all_three_nllb",
        "min_surface_dist_ge_060",
        "R11_excluded", "native_strict", "zipf_vi", "zipf_zh",
    )
    for family in MODEL_ORDER:
        for fmt in (*formats, "pooled"):
            value = _signed("dzh", family, fmt)
            dzh_rows.append(
                {
                    "model": family,
                    "format": fmt,
                    "instrument": "D_zh",
                    "estimate": value,
                    "ci_low": value - 0.05,
                    "ci_high": value + 0.05,
                    "bootstrap_key": f"bootstrap|{family}|{fmt}",
                }
            )
        value = _signed("term", family)
        term_rows.append(
            {
                "model": family,
                "instrument": "rise_zh_interaction",
                "term": "interaction",
                "estimate": value,
                "wald_ci_low": value - 0.1,
                "wald_ci_high": value + 0.1,
                "bootstrap_ci_low": value - 0.08,
                "bootstrap_ci_high": value + 0.08,
            }
        )
        for sensitivity in sensitivities:
            for instrument in ("D_zh", "rise_zh_interaction"):
                value = _signed("sensitivity", family, sensitivity, instrument)
                bootstrap = instrument == "D_zh"
                sensitivity_rows.append(
                    {
                        "model": family,
                        "sensitivity": sensitivity,
                        "instrument": instrument,
                        "estimate": value,
                        "ci_low": value - 0.04,
                        "ci_high": value + 0.04,
                        "standard_error": "" if bootstrap else 0.03,
                        "valid_draws": 1000 if bootstrap else "",
                        "bootstrap_key": f"bootstrap|{family}|{sensitivity}" if bootstrap else "",
                    }
                )
    _write_csv(reports, "E6_dzh.csv", dzh_rows)
    _write_csv(reports, "E6_model_terms.csv", term_rows)
    _write_csv(reports, "E6_sensitivity.csv", sensitivity_rows)

    rate_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            baseline = _signed("delta", family, fmt, "baseline", 0)
            delta_rows.append(
                {
                    "family": family,
                    "format": fmt,
                    "setting": "baseline",
                    "layer": 0,
                    "n": 100,
                    "mean_delta": baseline,
                    "ci_low": baseline - 0.03,
                    "ci_high": baseline + 0.03,
                }
            )
            for setting in ("k0.5", "k1", "k2", "random"):
                for layer in (1, {"gemma":18,"qwen":24,"llama":14}[family], 28):
                    rate_rows.append(
                        {
                            "family": family,
                            "format": fmt,
                            "setting": setting,
                            "layer": layer,
                            "n": 100 + len(rate_rows),
                            "en_rate": (0.99 if setting == "k1" and layer != 1 else _unique("en-rate", family, fmt, setting, layer)),
                            "damage_count": len(rate_rows) + 1,
                        }
                    )
                    value = _signed("delta", family, fmt, setting, layer)
                    delta_rows.append(
                        {
                            "family": family,
                            "format": fmt,
                            "setting": setting,
                            "layer": layer,
                            "n": 100 + layer,
                            "mean_delta": value,
                            "ci_low": value - 0.03,
                            "ci_high": value + 0.03,
                        }
                    )
    _write_csv(reports, "E8_rates.csv", rate_rows)
    _write_csv(reports, "E8_delta.csv", delta_rows)

    band_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for band in ("primary", "slope_sensitivity", "ops37_as_written"):
                value = _unique("band", family, fmt, band)
                band_rows.append(
                    {
                        "model": family,
                        "format": fmt,
                        "item_set": "cf729",
                        "direction": "d_vi_en",
                        "band": band,
                        "component": "share",
                        "estimate": value,
                        "ci_low": value - 0.02,
                        "ci_high": value + 0.02,
                        "blocks": "1;2;3;4;5;6;7;8;9",
                        "included_blocks": "1;2;3", "requested_size": 9,
                    }
                )
            for component in ("attn", "mlp"):
                value = _signed("signed-band", family, fmt, component)
                band_rows.append(dict(model=family, format=fmt, item_set="cf729", direction="d_vi_en", band="primary",
                                      component=component, estimate=value, sign=1.0 if value > 0 else -1.0, ci_low=value-.01, ci_high=value+.01))
            for layer in (1, 2, 3):
                for component in ("attn", "mlp"):
                    value = _signed("layer-projection", family, fmt, layer, component)
                    layer_rows.append(
                        {
                            "model": family,
                            "format": fmt,
                            "item_set": "cf729",
                            "direction": "d_vi_en",
                            "layer": layer,
                            "component": component,
                            "mean_projection": value,
                            "ci_low": value - 0.02,
                            "ci_high": value + 0.02,
                            "gate_pass": str(layer != 1),
                        }
                    )
    _write_csv(reports, "E7_band.csv", band_rows)
    _write_csv(reports, "E7_layers.csv", layer_rows)

    r1_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for variant in ("s", "s1"):
                for quantity in ("Q_H1", "H1_ratio", "max_LA_en"):
                    set1_value = _signed("r1", family, fmt, variant, quantity, 1)
                    for source_set in range(2, 7):
                        value = _signed("r1", family, fmt, variant, quantity, source_set)
                        r1_rows.append(
                            {
                                "model": family,
                                "format": fmt,
                                "fewshot_set": source_set,
                                "metric_variant": variant,
                                "quantity": quantity,
                                "value": value,
                                "ci_low": value - 0.02,
                                "ci_high": value + 0.02,
                                "set1_value": set1_value,
                                "set1_ci_low": set1_value - 0.02,
                                "set1_ci_high": set1_value + 0.02,
                            }
                        )
    _write_csv(reports, "R1_metrics.csv", r1_rows)

    classification_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    directions_rows: list[dict[str, Any]] = []
    class_index = 0
    for family in MODEL_ORDER:
        for fmt in formats:
            for pooling in ("last", "mean"):
                for layer in (1, 2, 3):
                    classification_rows.append(
                        {
                            "model": family,
                            "format": fmt,
                            "pooling": pooling,
                            "layer": layer,
                            "flores_correct": 90 + class_index,
                            "matched_correct": 95 + class_index,
                            "total": 130,
                        }
                    )
                    class_index += 1
                alignment = _signed("alignment", family, fmt, pooling)
                primary = _signed("alignment-primary", family, fmt, pooling)
                alignment_rows.append(
                    {
                        "model": family,
                        "format": fmt,
                        "pooling": pooling,
                        "max_LA_en": alignment,
                        "layer": 2,
                        "ci_low": alignment - 0.03,
                        "ci_high": alignment + 0.03,
                        "primary_max_LA_en": primary,
                        "primary_layer": 2,
                        "primary_ci_low": primary - 0.03,
                        "primary_ci_high": primary + 0.03,
                    }
                )
                directions_rows.append(
                    {
                        "model": family,
                        "format": fmt,
                        "pooling": pooling,
                        "layer": 2,
                        "v_vi": _signed("direction", family, fmt, pooling, "vi"),
                        "v_en": _signed("direction", family, fmt, pooling, "en"),
                        **{f"v_{l}": _signed("direction", family, fmt, pooling, l) for l in ("zh", "fr", "id")},
                    }
                )
    _write_csv(reports, "R2_classification.csv", classification_rows)
    _write_csv(reports, "R2_alignment.csv", alignment_rows)
    _write_csv(reports, "R2_directions.csv", directions_rows)

    e9_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for scope in ("paired_main", "paired_cf729"):
            for fmt in formats:
                for variant in ("s", "s1"):
                    for quantity in ("Q_H1", "H1_ratio", "H2_band_ratio", "max_LA_en"):
                        value = _signed("e9-diff", family, scope, fmt, variant, quantity)
                        e9_rows.append(
                            {
                                "model": family,
                                "format": fmt,
                                "scope": scope,
                                "condition": "nodiac_minus_diac",
                                "metric_variant": variant,
                                "quantity": quantity,
                                "value": value,
                                "ci_low": value - 0.025,
                                "ci_high": value + 0.025,
                            }
                        )
            for fmt in ("repetition", "translation", "cloze"):
                for cond_idx, condition in enumerate(("diac", "nodiac")):
                    denominator = 100 + len(e9_rows)
                    numerator = 1 + len(e9_rows)
                    e9_rows.append(
                        {
                            "model": family,
                            "format": fmt,
                            "scope": scope,
                            "condition": condition,
                            "metric_variant": "s",
                            "quantity": "A10_vi_rate",
                            "value": numerator / denominator,
                            "a10_vi_n": numerator,
                            "a10_denominator": denominator,
                        }
                    )
    _write_csv(reports, "E9.csv", e9_rows)

    head_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for layer in range(1, 5):
                for head in range(4):
                    value = _signed("head", family, fmt, layer, head)
                    head_rows.append(
                        {
                            "model": family,
                            "format": fmt,
                            "item_set": "cf729",
                            "direction": "d_vi_en",
                            "layer": layer,
                            "head": head,
                            "mean_projection": value,
                            "ci_low": value - 0.06,
                            "ci_high": value + 0.06,
                            "primary_band": "2;3",
                        }
                    )
    _write_csv(reports, "E7_heads_all.csv", head_rows)
    rankings = []
    for family in MODEL_ORDER:
        for fmt in formats:
            subset = [r for r in head_rows if r["model"] == family and r["format"] == fmt]
            # Deliberately different from heatmap extrema: plotting must respect source ranks.
            for i, row in enumerate(subset[:12]):
                rankings.append(dict(row, layer_type="local" if family == "gemma" else "global",
                                     top_positive_rank=float(i+1) if i < 6 else "", top_negative_rank=float(i-5) if i >= 6 else ""))
    _write_csv(reports, "E7_heads.csv", rankings)

    r10_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for language in ("vi", "en", "zh", "fr", "id"):
                for position in (1, 2, 3):
                    along = _signed("r10-along", family, fmt, language, position)
                    answer = _signed("r10-answer", family, fmt, language, position)
                    r10_rows.append(
                        {
                            "model": family,
                            "format": fmt,
                            "target_lang": language,
                            "target_variant": "primary",
                            "t": position,
                            "concepts_at_t": 30 + position,
                            "quantity": "max_concept_mean_delta",
                            "along_path_value": along,
                            "along_path_layer": 2,
                            "along_path_ci_low": along - 0.03,
                            "along_path_ci_high": along + 0.03,
                            "answer_position_value": answer,
                            "answer_position_layer": 3,
                            "answer_position_ci_low": answer - 0.03,
                            "answer_position_ci_high": answer + 0.03,
                        }
                    )
    _write_csv(reports, "R10.csv", r10_rows)

    battery_rows: list[dict[str, Any]] = []
    for family in MODEL_ORDER:
        for fmt in formats:
            for variant in ("primary", "variant_a", "variant_b"):
                suffix = hashlib.sha256(f"{family}|{fmt}|{variant}".encode()).hexdigest()[:8]
                outcome_s = f"outcome_s_{suffix}"
                outcome_s1 = f"outcome_s1_{suffix}"
                battery_rows.append(
                    {
                        "variant": variant,
                        "model": family,
                        "format": fmt,
                        "Outcome_s": outcome_s,
                        "Outcome_s1": outcome_s1,
                        "primary_Outcome": outcome_s if variant == "primary" else f"primary_s_{suffix}",
                        "primary_s1_outcome": outcome_s1 if variant == "primary" else f"primary_s1_{suffix}",
                    }
                )
    _write_csv(reports, "R_battery_e4.csv", battery_rows)


def test_every_figure_writes_a_sidecar_verified_against_synthetic_csvs(
    tmp_path: Path,
) -> None:
    """Build all panels from synthetic reports and verify every source mapping."""
    _synthetic_reports(tmp_path)
    reports = tmp_path / "reports"
    context = FigureContext(
        repo=tmp_path,
        reports_dir=reports,
        output_dir=reports / "figures",
        committed_only=False,
        dpi=45,
    )
    for spec, function in zip(FIGURE_SPECS, FIGURE_FUNCTIONS):
        function(context)
        for suffix in ("pdf", "png"):
            output = reports / "figures" / f"{spec.stem}.{suffix}"
            assert output.is_file(), f"{spec.key} did not write {output.name}"
        sidecar = reports / "figures" / "data" / f"{spec.stem}.csv"
        assert sidecar.is_file(), f"{spec.key} did not write its plotted-data sidecar"
        assert (reports / "figures" / f"{spec.stem}.png").stat().st_size < 1_000_000
        with sidecar.open(newline="", encoding="utf-8") as stream:
            copied = list(csv.DictReader(stream))
        if spec.key == "Fig 2":
            c1 = [r for r in copied if r["source_file"] == "E2-E5_c1.csv"]
            assert len(c1) == 12
            assert all(float(r["key_threshold"]) == .10 and r["x_label"] == "blocks" for r in c1)
            assert {r["key_contrast"] for r in c1} == {"instrument_effect", "selection_effect"}
        if spec.key == "S7":
            assert len(copied) == 3 * 3 * 3 * 3 * 2
            assert {float(r["key_threshold"]) for r in copied} == {.05, .10, .20}
            assert {r["x_label"] for r in copied} == {"blocks", "depth"}
        if spec.key == "Fig 5":
            for panel in ("(a)", "(b)"):
                native = [r for r in copied if r["panel"] == panel and r.get("key_sensitivity") == "native_strict"]
                assert len(native) == 3
            for sensitivity in ("zipf_vi", "zipf_zh"):
                assert len([r for r in copied if r["panel"] == "(b)" and r.get("key_sensitivity") == sensitivity]) == 3
        if spec.key == "Fig 7":
            assert len([r for r in copied if r["source_value_column"] == "sign"]) == 12
        if spec.key == "S2":
            assert {r["source_value_column"] for r in copied if r["source_file"] == "R2_directions.csv"} == {f"v_{l}" for l in ("vi", "en", "zh", "fr", "id")}
    assert context.read_paths
    assert all(path.parent == reports for path in context.read_paths)
    assert all("results" not in path.parts for path in context.read_paths)
    verification = verify_sidecars(tmp_path, committed_only=False)
    assert len(verification) == len(FIGURE_SPECS)
    assert all(row["fails"] == 0 for row in verification)
    assert all(float(row["max_diff"]) == 0 for row in verification)


def test_s1_labels_keep_quantity_variant_and_set_with_the_source_value(
    tmp_path: Path,
) -> None:
    """Regression guard for the former reversed-point / forward-label S1 row rotation."""
    _synthetic_reports(tmp_path)
    reports = tmp_path / "reports"
    context = FigureContext(
        repo=tmp_path,
        reports_dir=reports,
        output_dir=reports / "figures",
        committed_only=False,
        dpi=40,
    )
    from scripts.figures import supp1_r1_forest

    supp1_r1_forest(context)
    sidecar_path = reports / "figures" / "data" / "supp01_r1_forest.csv"
    with sidecar_path.open(encoding="utf-8", newline="") as stream:
        plotted_rows = list(csv.DictReader(stream))
    r1_rows = [row for row in plotted_rows if row["source_file"] == "R1_metrics.csv"]
    assert r1_rows
    assert len(r1_rows) == 3 * 2 * 2 * 3 * 6
    for row in r1_rows:
        match = re.fullmatch(r"(.+) · (s1?) · set ([1-6])", row["series"])
        assert match is not None
        quantity, variant, set_text = match.groups()
        plotted_set = int(set_text)
        assert row["key_quantity"] == quantity
        assert row["key_metric_variant"] == variant
        assert float(row["y"]) == float(plotted_set) + (-.15 if variant == "s" else .15)
        if plotted_set == 1:
            assert row["source_value_column"] == "set1_value"
            assert f"plotted set 1;" in row["annotation_text"]
        else:
            assert row["source_value_column"] == "value"
            assert int(float(row["key_fewshot_set"])) == plotted_set
    labels = [
        (row["key_model"], row["key_format"], row["key_quantity"], row["key_metric_variant"], row["series"])
        for row in r1_rows
    ]
    assert len(labels) == len(set(labels))


def test_la_lines_have_explicit_gaps() -> None:
    """Excluded, missing, and undefined layers cannot be bridged by an LA line."""
    import pandas as pd
    from scripts.figures import la_with_gaps
    part = pd.DataFrame(dict(layer=[1,2,4,5], value=[.11,.22,float("nan"),.55], ci_low=[.1,.2,.3,.5], ci_high=[.2,.3,.4,.6]))
    gate = pd.DataFrame(dict(layer=[1,2,3,4,5], gate_pass=[True,False,True,True,True]))
    result = la_with_gaps(part, gate, 5)
    assert result["layer"].tolist() == [1,2,3,4,5]
    assert result["value"].isna().tolist() == [False,True,True,True,False]
    assert result.loc[1:3, "ci_low"].isna().all()


def test_flip_peak_uses_earliest_maximum() -> None:
    """OPS-27 lookup uses translation k1 and breaks ties by earliest block."""
    import pandas as pd
    from scripts.figures import earliest_flip_peak
    rows = pd.DataFrame(dict(format=["translation"]*4+["repetition"], setting=["k1","k1","k1","k2","k1"],
                             layer=[24,18,3,1,1], en_rate=[.8,.8,.2,1.,1.]))
    assert earliest_flip_peak(rows)["layer"] == 18


def test_fig6_same_setting_color_in_every_row(tmp_path: Path, monkeypatch: Any) -> None:
    """The artist colors follow settings independently of measure and row."""
    from scripts.figures import fig6_e8_rates, SETTING_COLORS
    import matplotlib.pyplot as plt
    _synthetic_reports(tmp_path)
    captured = []
    def capture(self: Any, stem: str, figure: Any) -> None:
        captured.append(figure)
    monkeypatch.setattr(FigureContext, "save", capture)
    ctx = FigureContext(tmp_path, tmp_path/"reports", tmp_path/"reports/figures", committed_only=False)
    fig6_e8_rates(ctx)
    figure = captured[0]
    for axis in figure.axes:
        for line in axis.lines:
            label = line.get_label()
            for setting, color in SETTING_COLORS.items():
                if label == setting or label.startswith(setting+" "):
                    assert line.get_color() == color
    assert all(SETTING_COLORS[s] in [l.get_color() for l in axis.lines] for axis in figure.axes for s in ("k0.5","k1","k2","random"))
    plt.close(figure)


def test_s4_outlines_copy_source_ranking(tmp_path: Path, monkeypatch: Any) -> None:
    """Ranked-head outlines must follow E7_heads rather than heatmap extrema."""
    from scripts.figures import supp4_e7_heads
    import pandas as pd
    import matplotlib.pyplot as plt
    _synthetic_reports(tmp_path)
    captured = []
    def capture(self: Any, stem: str, figure: Any) -> None:
        captured.append(figure)
    monkeypatch.setattr(FigureContext, "save", capture)
    ctx = FigureContext(tmp_path, tmp_path/"reports", tmp_path/"reports/figures", committed_only=False)
    supp4_e7_heads(ctx)
    source = pd.read_csv(tmp_path/"reports/E7_heads.csv")
    outlines = [r for r in ctx.plotted_rows if r["source_file"] == "E7_heads.csv"]
    assert len(outlines) == len(source)
    for row in outlines:
        found = source.loc[(source.model == row["key_model"]) & (source["format"] == row["key_format"]) &
                           (source.layer == int(row["key_layer"])) & (source["head"] == int(row["key_head"]))].iloc[0]
        assert float(row["source_value"]) == found[row["source_value_column"]]
    heatmaps = [ax for ax in captured[0].axes if ax.images]
    assert sum(len(ax.patches) for ax in heatmaps) == len(source)
    for axis in heatmaps:
        for leader in axis.lines:
            assert all(x >= 1 for x in leader.get_xdata())
            assert leader.get_ydata()[0] == leader.get_ydata()[1]
    plt.close(captured[0])


def test_s4_label_anchors_are_own_layer_centres(tmp_path: Path, monkeypatch: Any) -> None:
    """Text may dodge, but every leader and annotation stays anchored to its head."""
    from scripts.figures import supp4_e7_heads
    import matplotlib.pyplot as plt
    _synthetic_reports(tmp_path)
    captured = []
    monkeypatch.setattr(FigureContext, "save", lambda self, stem, figure: captured.append(figure))
    ctx = FigureContext(tmp_path, tmp_path/"reports", tmp_path/"reports/figures", committed_only=False)
    supp4_e7_heads(ctx)
    count = 0
    for axis in captured[0].axes:
        if not axis.images:
            continue
        labels = [text for text in axis.texts if re.fullmatch(r"\d+\.\d+", text.get_text())]
        assert labels
        for label, leader in zip(labels, axis.lines, strict=True):
            layer = int(label.get_text().split(".")[0])
            # Synthetic heatmaps contain distinct layers 1..4 in ascending order.
            assert label.xy == (1, layer - 1)
            assert leader.get_ydata()[0] == layer - 1
            assert leader.get_ydata()[1] == layer - 1
            assert min(leader.get_xdata()) >= 1
            count += 1
    assert count == 36
    plt.close(captured[0])


def test_table_a_copies_strings_and_preserves_absent_fields(tmp_path: Path, monkeypatch: Any) -> None:
    """Joining tokenization variants must not derive an eligibility count."""
    import scripts.figures as figures
    reports = tmp_path / "reports"
    reports.mkdir()
    sources = {
        "X0c_a_target_tokenization.csv": b"label,family,variant,n_items,mean_content_tokens,continuation_share\nbase/gemma,gemma,zh,17,2.1234567890123456,0.3456789012345678\nbase/gemma,gemma,vi,19,3.2345678901234567,0.4567890123456789\n",
        "X0c_a_m1_first_token.csv": b"label,family,readout,usable_count,expected\nbase/gemma,gemma,zh_released,11,13\n",
    }
    monkeypatch.setattr(figures, "_read_csv_bytes", lambda repo, filename, **kwargs: sources[filename])
    monkeypatch.setattr(figures, "_latest_source_commit", lambda repo, filename: "synthetic-source-commit")
    figures.write_table_a(tmp_path)
    with (reports / "tables/tableA_tokenization.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["mean_content_tokens"] == "2.1234567890123456"
    assert rows[0]["continuation_share"] == "0.3456789012345678"
    assert rows[0]["usable_count"] == "11" and rows[0]["expected"] == "13"
    assert rows[1]["usable_count"] == ""
    assert "m1_eligible" not in rows[0]
    assert "synthetic-source-commit" in (reports / "tables/tableA_tokenization.md").read_text()


def test_t28_changed_figure_presentations(tmp_path: Path, monkeypatch: Any) -> None:
    """Distinct fixtures check the six T28 builders' actual artist encodings."""
    import matplotlib.pyplot as plt
    import scripts.figures as figures
    _synthetic_reports(tmp_path)
    captured = []
    def capture(self: Any, stem: str, figure: Any) -> None:
        captured.append((stem, figure))
    monkeypatch.setattr(FigureContext, "save", capture)
    for builder in (figures.fig4_la_gate, figures.fig5_e6_forest, figures.fig6_e8_rates,
                    figures.fig7_e7_projection, figures.supp3_e9_paired, figures.supp4_e7_heads):
        ctx = FigureContext(tmp_path, tmp_path/"reports", tmp_path/"reports/figures", committed_only=False)
        builder(ctx)
        stem, figure = captured[-1]
        if stem == "fig04_la_gate":
            labels = [text.get_text() for text in figure.legends[0].get_texts()]
            assert labels.count("label-dominated (last)") == 1
            assert labels.count("label-dominated (mean)") == 1
            assert not any(label.startswith(("×", "+")) for label in labels)
        elif stem == "fig05_e6_forest":
            labels = [t.get_text() for axis in figure.axes for t in axis.get_yticklabels()]
            assert any("OPS-28: native_strict" in label for label in labels)
            assert any("A8: zipf_vi" in label for label in labels)
            assert any("A8: zipf_zh" in label for label in labels)
            assert not any("R6: native_strict" in label or "R6: zipf_" in label for label in labels)
        elif stem == "fig06_e8_rates":
            assert len(figure.axes) == 12
            for axis in figure.axes[:3]:
                assert not any("DAMAGE" in line.get_label() for line in axis.lines)
            for axis in figure.axes[3:6]:
                assert any("DAMAGE" in line.get_label() for line in axis.lines)
                assert not any("en_rate" in line.get_label() for line in axis.lines)
            assert {r["panel"] for r in ctx.plotted_rows} == {"(a)", "(b)", "(c)", "(d)"}
        elif stem == "fig07_e7_projection":
            axis = figure.axes[0]
            signs = [t for t in axis.texts if t.get_text().startswith("(") and "," in t.get_text()]
            assert len(signs) == 6
            assert {round(float(t.xy[1]), 2) for t in signs} == {round(i-.18, 2) for i in range(6)}
            written = next(t for t in axis.texts if t.get_text() == "3 of 9 blocks")
            assert written.xy[1] == 1.18
            assert written.get_position()[1] < 0
        elif stem == "supp03_e9_paired":
            assert len(figure.legends) == 2
            assert [t.get_text() for t in figure.legends[0].get_texts()] == ["s", "s¹", "paired main", "paired cf729"]
            assert [t.get_text() for t in figure.legends[1].get_texts()] == ["diac", "nodiac", "main", "cf729"]
        elif stem == "supp04_e7_heads":
            for axis in figure.axes:
                if not axis.images:
                    continue
                for line in axis.lines:
                    assert min(line.get_xdata()) >= 1
                    assert line.get_ydata()[0] == line.get_ydata()[1]
        plt.close(figure)
