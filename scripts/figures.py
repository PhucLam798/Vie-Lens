"""Build figures and plotted-data sidecars from committed aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ["MPLCONFIGDIR"] = "/datastore/luannt/.cache/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/datastore/luannt/.cache/xdg"
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANG_ORDER = ("vi", "en", "zh", "fr", "id")
LANG_COLORS = {
    "vi": "#0072B2",
    "en": "#E69F00",
    "zh": "#009E73",
    "fr": "#CC79A7",
    "id": "#D55E00",
}
MODEL_ORDER = ("gemma", "qwen", "llama")
MODEL_LABELS = {"gemma": "Gemma", "qwen": "Qwen", "llama": "Llama"}
MODEL_COLORS = {"gemma": "#0072B2", "qwen": "#E69F00", "llama": "#009E73"}
SAFE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
SETTING_ORDER = ("k0.5", "k1", "k2", "random")
SETTING_COLORS = {"baseline": "#000000", "k0.5": "#9ecae1", "k1": "#4292c6", "k2": "#08519c", "random": "#888888"}
ALLOWED_CSVS = frozenset(
    {
        "X0c_a_target_tokenization.csv", "X0c_a_m1_first_token.csv",
        "E2-E5_e2.csv", "E2-E5_c1.csv", "E2-E5_landmarks.csv",
        "E2-E5_la_curves.csv", "E5_gate.csv", "E6_dzh.csv",
        "E6_model_terms.csv", "E6_sensitivity.csv", "E8_rates.csv",
        "E8_delta.csv", "E7_band.csv", "E7_layers.csv",
        "E7_heads_all.csv", "E7_heads.csv", "R1_metrics.csv", "R2_classification.csv",
        "R2_alignment.csv", "R2_directions.csv", "E9.csv", "R10.csv",
        "R_battery_e4.csv",
    }
)
SIDECAR_FIELDS = (
    "panel", "series", "x_label", "x", "y", "plotted_value", "ci_low", "ci_high",
    "source_file", "source_key_columns", "source_value_column", "source_value",
    "source_value_axis", "value_transform", "source_value_aux_column",
    "source_value_aux", "source_x_column", "x_transform", "x_denominator",
    "source_ci_low_column", "source_ci_high_column", "ci_transform",
    "ci_denominator", "annotation_text", "annotation_transform",
)


class FigureSkip(RuntimeError):
    """Raised when a required source value column is absent."""


@dataclass
class FigureContext:
    """Source reader, plotted-data recorder, and deterministic output writer."""

    repo: Path
    reports_dir: Path
    output_dir: Path
    committed_only: bool = True
    dpi: int = 300
    read_paths: list[Path] = field(default_factory=list)
    plotted_rows: list[dict[str, str]] = field(default_factory=list)

    def csv(self, filename: str) -> pd.DataFrame:
        """Read an allow-listed aggregate CSV from HEAD or a synthetic fixture."""
        if filename not in ALLOWED_CSVS:
            raise ValueError(f"CSV is not allow-listed: {filename}")
        if self.committed_only:
            relpath = f"reports/{filename}"
            result = subprocess.run(
                ["git", "show", f"HEAD:{relpath}"],
                cwd=self.repo,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            data = result.stdout
            path = self.repo / relpath
        else:
            path = (self.reports_dir / filename).resolve()
            if "results" in path.parts:
                raise ValueError("Figure sources may not be read from results/")
            data = path.read_bytes()
        self.read_paths.append(path)
        return pd.read_csv(io.BytesIO(data), keep_default_na=True)

    def record_point(
        self,
        *,
        panel: str,
        series: str,
        x_label: str,
        x: object,
        y: object,
        source_file: str,
        source_row: Mapping[str, Any],
        key_columns: Sequence[str],
        source_value_column: str,
        plotted_value: object | None = None,
        source_value_axis: str = "y",
        value_transform: str = "identity",
        source_value_aux_column: str = "",
        source_x_column: str = "",
        x_transform: str = "",
        x_denominator: object | None = None,
        source_ci_low_column: str = "",
        source_ci_high_column: str = "",
        ci_transform: str = "identity",
        ci_denominator: object | None = None,
        ci_low: object | None = None,
        ci_high: object | None = None,
        annotation_text: str = "",
        annotation_transform: str = "identity",
    ) -> None:
        """Record one displayed estimate or categorical annotation and its source row."""
        if source_value_axis not in {"x", "y", "color", "annotation"}:
            raise ValueError(f"Unsupported source value axis: {source_value_axis}")
        keys = list(key_columns)
        raw_value = source_row[source_value_column]
        aux_value = source_row[source_value_aux_column] if source_value_aux_column else ""
        row = {
            "panel": str(panel),
            "series": str(series),
            "x_label": str(x_label),
            "x": _cell_text(x),
            "y": _cell_text(y),
            "plotted_value": _cell_text(raw_value if plotted_value is None else plotted_value),
            "ci_low": _cell_text(ci_low),
            "ci_high": _cell_text(ci_high),
            "source_file": source_file,
            "source_key_columns": json.dumps(keys, ensure_ascii=False, separators=(",", ":")),
            "source_value_column": source_value_column,
            "source_value": _cell_text(raw_value),
            "source_value_axis": source_value_axis,
            "value_transform": value_transform,
            "source_value_aux_column": source_value_aux_column,
            "source_value_aux": _cell_text(aux_value),
            "source_x_column": source_x_column,
            "x_transform": x_transform,
            "x_denominator": _cell_text(x_denominator),
            "source_ci_low_column": source_ci_low_column,
            "source_ci_high_column": source_ci_high_column,
            "ci_transform": ci_transform,
            "ci_denominator": _cell_text(ci_denominator),
            "annotation_text": annotation_text,
            "annotation_transform": annotation_transform,
        }
        for key in keys:
            row[f"key_{key}"] = _cell_text(source_row.get(key, ""))
        self.plotted_rows.append(row)

    def write_sidecar(self, stem: str) -> Path:
        """Write the plotted rows, including a header for an intentionally skipped figure."""
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{stem}.csv"
        key_fields = sorted(
            {key for row in self.plotted_rows for key in row if key.startswith("key_")}
        )
        fields = (*SIDECAR_FIELDS, *key_fields)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, extrasaction="ignore",
                lineterminator="\r\n" if stem in {"fig03_rise_landmarks", "supp06_r_battery_outcomes"} else "\n",
            )
            writer.writeheader()
            writer.writerows(self.plotted_rows)
        self.plotted_rows.clear()
        return path

    def save(self, stem: str, figure: plt.Figure) -> tuple[str, str, str]:
        """Save deterministic PDF, sub-1-MB PNG, and plotted-data CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.output_dir / f"{stem}.pdf"
        png_path = self.output_dir / f"{stem}.png"
        figure.savefig(
            pdf_path,
            format="pdf",
            metadata={"CreationDate": None, "ModDate": None},
            bbox_inches="tight",
        )
        dpi = self.dpi
        figure.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", metadata={})
        if png_path.stat().st_size >= 1_000_000 and dpi > 200:
            figure.savefig(png_path, format="png", dpi=200, bbox_inches="tight", metadata={})
        if png_path.stat().st_size >= 1_000_000:
            raise RuntimeError(f"PNG exceeds 1 MB after the 200 dpi fallback: {png_path.name}")
        plt.close(figure)
        self.write_sidecar(stem)
        return pdf_path.name, png_path.name, f"data/{stem}.csv"


@dataclass(frozen=True)
class FigureSpec:
    """Source and caption metadata for one requested figure."""

    key: str
    title: str
    stem: str
    sources: tuple[str, ...]
    filters: str
    columns: str
    transforms: str
    caption: str
    notes: str = ""


FIGURE_SPECS = (
    FigureSpec(
        "Fig 2", "M1 and M2 readouts", "fig02_m1_m2",
        ("E2-E5_e2.csv", "E2-E5_c1.csv"),
        "All E2 model × format × readout rows; all aggregate C1 rows with a reported value.",
        "E2: model, format, readout, max_mean_P_M1, layer; C1: model, format, item_set, contrast, threshold, metric, value, ci_low, ci_high.",
        "No transformation. E2 has no interval in source; C1 intervals copied where present.",
        "Panel (a) plots maximum mean P_M1 by readout; panel (b) plots the reported C1 contrasts with source intervals.",
    ),
    FigureSpec(
        "Fig 3", "Five-way rise by normalized depth", "fig03_rise_landmarks",
        ("E2-E5_la_curves.csv", "E2-E5_landmarks.csv"),
        "rise_cf729 curves; diac, metric variant s, cf729 landmarks; repetition and translation.",
        "Curves: model, format, target_lang, metric, layer, value, optional ci_low/ci_high. Landmarks: model, format, metric, value, ci_low, ci_high, n_layers.",
        "Curve x=layer/N. Landmark x and CI bounds=value or CI bound divided by n_layers.",
        "Five language curves show rise over normalized layer depth; separate landmark markers and intervals are shown where defined.",
    ),
    FigureSpec(
        "Fig 4", "LA curves and gate diagnostics", "fig04_la_gate",
        ("E2-E5_la_curves.csv", "E5_gate.csv"),
        "LA curves grouped by model, format, and target language; A6 gate rows matched by family, format, and layer.",
        "Curves: model, format, target_lang, metric, layer, value, optional ci_low/ci_high. Gate: family, fit_format, layer, gate_pass, label_dominated_last, label_dominated_mean.",
        "x=layer/N. Reference lines at −0.25, 0.20, and 0.50; gate failures shade the layer interval.",
        "Language curves show LA estimates by normalized layer depth with source intervals where present; gate-failing layers and label-dominated flags are marked.",
    ),
    FigureSpec(
        "Fig 5", "E6 forest estimates and sensitivities", "fig05_e6_forest",
        ("E6_dzh.csv", "E6_model_terms.csv", "E6_sensitivity.csv"),
        "D_zh by model and format; primary rise_zh_interaction; four R6 subsets and R11_excluded.",
        "D_zh: model, format, estimate, ci_low, ci_high, bootstrap_key. Interaction: model, instrument, term, estimate, bootstrap_ci_low/high, wald_ci_low/high. Sensitivity: model, sensitivity, instrument, estimate, ci_low/high, standard_error, valid_draws, bootstrap_key.",
        "No estimate transformation. Filled/solid = bootstrap; open/dashed = Wald.",
        "Forest panels show D_zh and the §7.1 interaction for each model, alongside the four R6 subsets and R11 sensitivity rows.",
        "Rows with no estimate are dropped within their panel.",
    ),
    FigureSpec(
        "Fig 6", "E8 flip and damage rates", "fig06_e8_rates",
        ("E8_rates.csv", "E8_delta.csv", "E2-E5_landmarks.csv"),
        "Top: translation settings k0.5, k1, k2, random; damage for k1 and random. Bottom: all reported Δ settings in both formats. L_cross where defined.",
        "Rates: family, format, setting, layer, n, en_rate, damage_count. Delta: family, format, setting, layer, mean_delta, ci_low, ci_high. Landmarks: model, format, metric, value, n_layers.",
        "x=layer/N; DAMAGE=damage_count/n; landmark x=value/n_layers.",
        "The top row plots translation en_rate and DAMAGE by normalized depth and setting; each point's n is copied from E8_rates.csv. The lower rows plot mean Δ and its intervals by format.",
        "No interval in source for the rate curves. Repetition rates are not part of this E8 plot; repetition Δ is shown below.",
    ),
    FigureSpec(
        "Fig 7", "E7 attention share and layer projections", "fig07_e7_projection",
        ("E7_band.csv", "E7_layers.csv"),
        "cf729, d_vi_en share rows for primary and both sensitivities; cf729 d_vi_en attention and MLP layer rows.",
        "Band: model, format, item_set, direction, band, component, estimate, ci_low, ci_high, sign. Layers: model, format, item_set, direction, layer, component, mean_projection, ci_low, ci_high, gate_pass.",
        "Layer x=layer/N. Source interval ribbons are drawn where both CI columns are present.",
        "Panel (a) shows attention-share estimates for the primary and sensitivity bands. Panel (b) shows attention and MLP projections over normalized layer depth.",
        "The source has no S_attn or S_mlp signed-value columns, so sign-pair annotations are not plotted.",
    ),
    FigureSpec(
        "S1", "R1 forest summaries", "supp01_r1_forest",
        ("R1_metrics.csv",),
        "Few-shot sets 1–6; metric variants s and s1; Q_H1, H1_ratio, max_LA_en.",
        "model, format, fewshot_set, metric_variant, quantity, value, ci_low, ci_high; set 1 uses set1_value and set1_ci_low/high.",
        "No value transformation. Set 1 is copied from the dedicated set1_* columns.",
        "Three columns show Q_H1, H1 ratio, and maximum LA_en across sets 1–6 for each model and format.",
    ),
    FigureSpec(
        "S2", "R2 validation and alignment summaries", "supp02_r2_validation",
        ("R2_classification.csv", "R2_alignment.csv", "R2_directions.csv"),
        "All reported pooling modes and validation layers; both alignment rows per model × format × pooling.",
        "Classification: model, format, pooling, layer, flores_correct, matched_correct, total. Alignment: model, format, pooling, max_LA_en, CIs, primary_max_LA_en, primary CIs. Directions: component columns only.",
        "x=layer/N for classifications; reference line at 124/130. Alignment values copied without transformation.",
        "Upper panels show per-layer validation counts and the 124/130 reference. Lower panels show max LA_en and the matched primary reference marker.",
        "No interval in source for classification counts. No cosine min/median/max summary value exists in the committed R2 CSVs, so a cosine-summary panel is not plotted.",
    ),
    FigureSpec(
        "S3", "E9 paired differences and A10 rates", "supp03_e9_paired",
        ("E9.csv",),
        "Paired main and paired cf729; nodiac_minus_diac differences for s/s1; A10 VI rates by condition and format.",
        "model, format, scope, condition, metric_variant, quantity, value, ci_low/high, a10_vi_n, a10_denominator.",
        "A10 VI rate=a10_vi_n/a10_denominator. Difference values and intervals are copied.",
        "Panel (a) shows paired nodiac-minus-diac estimates with source intervals; panel (b) shows A10 VI rates by condition and format.",
        "No interval in source for the A10 rates.",
    ),
    FigureSpec(
        "S4", "E7 per-head projection heatmaps", "supp04_e7_heads",
        ("E7_heads_all.csv",),
        "cf729, d_vi_en; one mean-projection heatmap per model × format; primary band outlined.",
        "model, format, item_set, direction, layer, head, mean_projection, ci_low, ci_high, primary_band.",
        "Heatmap uses mean_projection with a symmetric per-cell diverging scale; a dot marks a source CI excluding zero.",
        "Each cell displays layer-by-head mean projections; dots mark intervals excluding zero and the primary band is outlined.",
    ),
    FigureSpec(
        "S5", "R10 along-path and answer-position summaries", "supp05_r10_positions",
        ("R10.csv",),
        "t=1, 2, 3; all reported model, format, target-language and target-variant rows.",
        "model, format, target_lang, target_variant, t, concepts_at_t, along_path_value/layer/CI, answer_position_value/layer/CI.",
        "No value transformation. concepts_at_t is copied once per t under each panel x-axis.",
        "Lines show along-path and answer-position maximum Δ values with source intervals at t=1, 2, and 3; each t label gives concepts_at_t.",
    ),
    FigureSpec(
        "S6", "R-battery outcome grid", "supp06_r_battery_outcomes",
        ("R_battery_e4.csv",),
        "All variants and model × format cells; score s and s1 outcomes beside primary outcomes.",
        "variant, model, format, Outcome_s, Outcome_s1, primary_Outcome, primary_s1_outcome.",
        "No value transformation; categorical outcomes copied as displayed.",
        "Panels show each variant outcome beside the corresponding primary outcome for every model × format cell.",
    ),
    FigureSpec("S7", "Complete C1 contrasts", "supp07_c1_full", ("E2-E5_c1.csv",),
               "All thresholds 0.05/0.10/0.20, formats and extension-inclusive contrasts.",
               "model, format, item_set, contrast, threshold, metric, units, value, ci_low, ci_high.",
               "Values and intervals copied; blocks and depth use separate axes.",
               "Complete C1 set: one row per model × format; blocks and depth in separate columns."),
)


# Caption metadata follows the final presentation; source tables remain immutable.
_SPEC_UPDATES = {
    "Fig 2": dict(filters="E2 unchanged; C1 instrument_effect and selection_effect, threshold 0.10, blocks, repetition and translation.",
                  caption="E2 readouts (a)–(c); C1 effects in blocks (d)–(f) with copied source intervals; format is marker shape. Full C1 appears in S7."),
    "Fig 4": dict(notes="label-dominated: matched-fit directions classify FLORES+ dev sentences with < 95% accuracy at that layer and pooling (scripts/a6_directions.py:171)", caption="LA curves for en/zh/fr/id; excluded and undefined layers are gaps. × label-dominated (last) and + label-dominated (mean) mean matched-fit directions classify FLORES+ dev sentences with <95% accuracy at that layer and pooling (scripts/a6_directions.py:171). Grey shading: A6 gate fail. Neutral null −0.25 (black dotted), H2 0.20 (grey dashed), H1 0.50 (black dash-dot). No LA_vi rows exist."),
    "Fig 5": dict(filters="Primary and sensitivity rows including native_strict in both panels and zipf_vi/zipf_zh in the interaction panel.",
                  caption="D_zh and §7.1 stratum[sino] × language[zh] term; H3-rule row labels are bold. Filled points: bootstrap; open points: Wald. Intervals are copied."),
    "Fig 6": dict(title="E8 flip, DAMAGE and Δ", filters="Translation rates and DAMAGE for all four settings; both-format Δ and unsteered baseline.",
                  caption="(a) Translation EN flip rate; (b) translation DAMAGE rate; (c) repetition Δ; (d) translation Δ. Setting colours are fixed across rows: black baseline, sequential light-to-dark blue k0.5/k1/k2, grey random. en_rate: solid circles; DAMAGE: dashed squares; Δ: solid. OPS-27 flip peak is the earliest block attaining maximum translation k1 en_rate: Gemma 18, Qwen 24, Llama 14; copied-rate lookup only."),
    "Fig 7": dict(notes="Projections are in residual-stream units and are not comparable across models.",
                  caption="(a) Six-cell attention-share dot plot: primary filled circle, slope sensitivity open square, OPS-37 as written open triangle. Primary sign pairs (attn, mlp), copied from the sign fields: Gemma rep (+,+), tr (−,−); Qwen rep (+,+), tr (+,+); Llama rep (+,+), tr (+,+). Gemma translation as written includes 3 of 9 requested blocks. (b) Per-layer residual projections, separate Repetition and Translation rows."),
    "S1": dict(caption="s is a solid whisker with filled circle; s¹ a dashed whisker with open square, dodged ±0.15 within each set row. Three quantity columns retain separate axes."),
    "S2": dict(columns="Classification counts; alignment estimates and source CIs; R2_directions keys model/format/pooling/layer and v_vi/v_en/v_zh/v_fr/v_id cosine columns.",
               notes="No intervals in source for counts or direction cosines.",
               caption="Classification, alignment and per-layer direction cosine rows. Alignment FLORES estimates are blue; matched primary last-pooling reference is a black diamond. Cosine colour is language; last pooling solid, mean dashed."),
    "S3": dict(caption="(a) Paired differences: separate quantity columns and model × format rows. s filled circles, s¹ open squares; main dark, cf729 light; vertical dodge. Each panel has its own encoding legend. (b) Per-model A10 dumbbells link diac filled circle to nodiac open square; main and cf729 are dodged."),
    "S4": dict(sources=("E7_heads_all.csv", "E7_heads.csv"),
               transforms="Mean projections copied to heatmaps; outlines and labels copied from primary-band rank columns in E7_heads.csv (cf729, d_vi_en). No CI-exclusion dots.",
               caption="Top-10 positive/negative primary-band heads outlined by source ranking, with top 3 in each sign/type labelled layer.head in a separate side column; short horizontal leaders remain outside the heatmap. Gemma local/global rankings retained. Gemma global layers 6,12,18,24,30 carry bold G ticks."),
}
FIGURE_SPECS = tuple(replace(spec, **_SPEC_UPDATES.get(spec.key, {})) for spec in FIGURE_SPECS)


def _style() -> None:
    """Set stable typography and layout defaults for publication figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _fam(value: object) -> str:
    """Map a model ID or short name to the stable family key."""
    text = str(value).lower()
    if "gemma" in text:
        return "gemma"
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    return text


def _cell_text(value: object | None) -> str:
    """Serialize a scalar as a stable CSV cell, leaving missing values empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _num(value: object) -> float:
    """Convert a finite source scalar to float, raising on missing plot values."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value cannot be plotted: {value!r}")
    return result


def _finite(value: object) -> bool:
    """Return whether a value is a finite numeric scalar."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require(frame: pd.DataFrame, filename: str, columns: Iterable[str]) -> None:
    """Require keys and values needed to select the requested plotted rows."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FigureSkip(f"{filename} is missing required value/key column(s): {', '.join(missing)}")


def _interval(
    row: Mapping[str, Any], low_col: str, high_col: str
) -> tuple[float, float] | None:
    """Return a source interval only when both source bounds are present."""
    if low_col not in row or high_col not in row:
        return None
    if not _finite(row[low_col]) or not _finite(row[high_col]):
        return None
    return _num(row[low_col]), _num(row[high_col])


def _record(
    ctx: FigureContext,
    row: Mapping[str, Any],
    *,
    panel: str,
    series: str,
    x_label: str,
    x: object,
    y: object,
    source_file: str,
    keys: Sequence[str],
    value_col: str,
    value_axis: str = "y",
    value_transform: str = "identity",
    value_aux_col: str = "",
    x_source_col: str = "",
    x_transform: str = "",
    x_denominator: object | None = None,
    ci_low_col: str = "",
    ci_high_col: str = "",
    ci_transform: str = "identity",
    ci_denominator: object | None = None,
    ci: tuple[float, float] | None = None,
    plotted_value: object | None = None,
    annotation: str = "",
    annotation_transform: str = "identity",
) -> None:
    """Record a source-backed plotted point and optional source interval."""
    ctx.record_point(
        panel=panel,
        series=series,
        x_label=x_label,
        x=x,
        y=y,
        source_file=source_file,
        source_row=row,
        key_columns=keys,
        source_value_column=value_col,
        plotted_value=plotted_value,
        source_value_axis=value_axis,
        value_transform=value_transform,
        source_value_aux_column=value_aux_col,
        source_x_column=x_source_col,
        x_transform=x_transform,
        x_denominator=x_denominator,
        source_ci_low_column=ci_low_col,
        source_ci_high_column=ci_high_col,
        ci_transform=ci_transform,
        ci_denominator=ci_denominator,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        annotation_text=annotation,
        annotation_transform=annotation_transform,
    )


def _line_with_ci(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci_low: np.ndarray | None,
    ci_high: np.ndarray | None,
    *,
    color: str,
    linestyle: str = "-",
    label: str | None = None,
    marker: str | None = None,
    alpha: float = 0.15,
) -> None:
    """Draw source points as a line and draw an interval ribbon if available."""
    axis.plot(x, y, color=color, linestyle=linestyle, lw=1.1, label=label, marker=marker, ms=2.5)
    if ci_low is not None and ci_high is not None:
        axis.fill_between(x, ci_low, ci_high, color=color, alpha=alpha, linewidth=0)


def _facet_matrix(
    figure: plt.Figure,
    axes: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> None:
    """Place model and format labels outside panels to avoid title collisions."""
    for col, label in enumerate(col_labels):
        axes[0, col].text(
            0.5, 1.08, label, transform=axes[0, col].transAxes,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.28, 0.5, label, transform=axes[row, 0].transAxes,
            ha="right", va="center", rotation=90, fontsize=8, fontweight="bold",
        )


def _c1_point(ctx: FigureContext, axis: plt.Axes, row: Mapping[str, Any], y: float, panel: str) -> None:
    """Copy a C1 estimate and its source interval without altering units."""
    value = _num(row["value"])
    interval = _interval(row, "ci_low", "ci_high")
    family = _fam(row["model"])
    marker = {"repetition": "o", "translation": "s", "cloze": "D"}[str(row["format"])]
    if interval:
        axis.errorbar(value, y, xerr=[[max(0, value-interval[0])], [max(0, interval[1]-value)]],
                      fmt=marker, color=MODEL_COLORS[family], capsize=2, ms=4)
    else:
        axis.plot(value, y, marker=marker, color=MODEL_COLORS[family], ls="none")
    _record(ctx, row, panel=panel, series=f"{family} · {row['format']} · {row['contrast']} · {row['threshold']} · {row['metric']}",
            x_label=str(row.get("units", "C1")), x=value, y=y, source_file="E2-E5_c1.csv",
            keys=("model", "format", "item_set", "contrast", "threshold", "metric"), value_col="value", value_axis="x",
            ci_low_col="ci_low" if interval else "", ci_high_col="ci_high" if interval else "", ci=interval)


def fig2_m1_m2(ctx: FigureContext) -> None:
    """Keep E2 panels and show primary C1 effects in blocks at threshold 0.10."""
    e2 = ctx.csv("E2-E5_e2.csv")
    c1 = ctx.csv("E2-E5_c1.csv")
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    e2 = e2.loc[e2["max_mean_P_M1"].notna()].copy()
    e2["family"] = e2["model"].map(_fam)
    format_markers = {"repetition": "o", "translation": "s", "cloze": "D"}
    _facet_matrix(figure, axes, row_labels=("E2 readouts", "C1 threshold 0.10"),
                  col_labels=tuple(MODEL_LABELS[f] for f in MODEL_ORDER))
    panel_index = 0
    for col_idx, family in enumerate(MODEL_ORDER):
        axis = axes[0, col_idx]
        cell = e2.loc[e2["family"].eq(family)].sort_values(["format", "readout"], kind="stable")
        labels: list[str] = []
        for index, (_, row) in enumerate(cell.iterrows()):
            value = _num(row["max_mean_P_M1"])
            y = float(index)
            fmt = str(row["format"])
            axis.plot(
                value, y, marker=format_markers.get(fmt, "o"),
                color=MODEL_COLORS[family], ms=3.8, linestyle="none",
            )
            label = f"{fmt} · {row['readout']}"
            labels.append(label)
            _record(
                ctx, row, panel=f"({chr(97 + panel_index)})", series=label,
                x_label="max mean P_M1", x=value, y=y,
                source_file="E2-E5_e2.csv", keys=("model", "format", "readout"),
                value_col="max_mean_P_M1", value_axis="x",
            )
        axis.set_yticks(range(len(labels)), labels, fontsize=5.8)
        axis.set_xlabel("max mean P_M1 (no interval in source)")
        axis.set_title(f"({chr(97 + panel_index)})", loc="left", fontsize=7)
        panel_index += 1

    for col, family in enumerate(MODEL_ORDER):
        axis = axes[1, col]
        cell = c1.loc[c1["model"].map(_fam).eq(family) & c1["threshold"].eq(0.10)
                      & c1["contrast"].isin(("instrument_effect", "selection_effect"))
                      & c1["units"].eq("blocks") & c1["format"].isin(("repetition", "translation"))]
        for _, row in cell.iterrows():
            y = (0 if row["contrast"] == "instrument_effect" else 1) + (-0.12 if row["format"] == "repetition" else 0.12)
            _c1_point(ctx, axis, row, y, f"({chr(100+col)})")
        axis.set_yticks((0, 1), ("instrument effect", "selection effect"))
        axis.axvline(0, color="#888888", ls=":", lw=0.8)
        axis.set_xlabel("C1 effect (blocks; source 95% CI)")
        axis.set_title(f"({chr(100+col)})", loc="left")
    figure.legend(handles=[plt.Line2D([], [], marker=m, color="#555555", ls="none", label=f)
                           for f, m in format_markers.items()], loc="lower center", ncol=3)
    figure.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.12, hspace=0.45, wspace=0.4)
    ctx.save("fig02_m1_m2", figure)


def supp7_c1_full(ctx: FigureContext) -> None:
    """Show complete C1 contrasts, separating block and depth axes."""
    frame = ctx.csv("E2-E5_c1.csv")
    frame = frame.loc[frame["value"].notna() & frame["threshold"].isin((0.05, 0.10, 0.20))]
    _style()
    figure, axes = plt.subplots(9, 2, figsize=(13, 23), squeeze=False)
    for i, (family, fmt) in enumerate((f, t) for f in MODEL_ORDER for t in ("repetition", "translation", "cloze")):
        for j, unit in enumerate(("blocks", "depth")):
            axis = axes[i, j]
            cell = frame.loc[frame["model"].map(_fam).eq(family) & frame["format"].eq(fmt) & frame["units"].eq(unit)]
            cell = cell.sort_values(["contrast", "threshold", "item_set"], kind="stable")
            labels = []
            for y, (_, row) in enumerate(cell.iterrows()):
                _c1_point(ctx, axis, row, float(y), f"{family} · {fmt} · {unit}")
                contrast = str(row["contrast"]).replace("extension_inclusive_instrument_effect", "extension-inclusive instrument").replace("_effect", "").replace("_", " ")
                labels.append(f"{contrast} · {row['threshold']:.2f} · {row['item_set']}")
            axis.set_yticks(range(len(labels)), labels, fontsize=6)
            axis.axvline(0, color="#888888", ls=":", lw=0.7)
            axis.set_xlabel(f"C1 effect ({unit}; source 95% CI)")
            axis.set_title(f"{MODEL_LABELS[family]} · {fmt} · {unit}")
    figure.subplots_adjust(left=0.29, right=0.99, top=0.98, bottom=0.03, hspace=0.65, wspace=1.1)
    ctx.save("supp07_c1_full", figure)


def fig3_rise_landmarks(ctx: FigureContext) -> None:
    """Plot five-way rise curves and normalized landmark markers."""
    curves = ctx.csv("E2-E5_la_curves.csv")
    landmarks = ctx.csv("E2-E5_landmarks.csv")
    _require(curves, "E2-E5_la_curves.csv", ("model", "format", "target_lang", "metric", "layer", "value"))
    _require(landmarks, "E2-E5_landmarks.csv", ("model", "format", "condition", "metric_variant", "item_set", "metric", "value", "n_layers"))
    curves = curves.loc[curves["metric"].eq("rise_cf729")].copy()
    landmarks = landmarks.loc[
        landmarks["condition"].eq("diac")
        & landmarks["metric_variant"].eq("s")
        & landmarks["item_set"].eq("cf729")
        & landmarks["metric"].isin(("L_peak", "L_concept_10", "L_cross"))
        & landmarks["format"].isin(("repetition", "translation"))
    ].copy()
    if curves.empty:
        raise FigureSkip("E2-E5_la_curves.csv has no rise_cf729 value rows")
    curves["family"] = curves["model"].map(_fam)
    landmarks["family"] = landmarks["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    _facet_matrix(
        figure, axes, row_labels=("Repetition", "Translation"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    lm_style = {"L_peak": "--", "L_concept_10": ":", "L_cross": "-."}
    lane = {"L_peak": 0.045, "L_concept_10": 0.10, "L_cross": 0.155}
    for row_idx, fmt in enumerate(("repetition", "translation")):
        for col_idx, family in enumerate(MODEL_ORDER):
            axis = axes[row_idx, col_idx]
            cell = curves.loc[curves["family"].eq(family) & curves["format"].eq(fmt)]
            for lang in LANG_ORDER:
                part = cell.loc[cell["target_lang"].eq(lang)].sort_values("layer", kind="stable")
                if part.empty:
                    continue
                n_layers = float(part["layer"].max())
                xx = part["layer"].to_numpy(dtype=float) / n_layers
                yy = part["value"].to_numpy(dtype=float)
                have_ci = {"ci_low", "ci_high"}.issubset(part.columns) and part["ci_low"].notna().all() and part["ci_high"].notna().all()
                lo = part["ci_low"].to_numpy(dtype=float) if have_ci else None
                hi = part["ci_high"].to_numpy(dtype=float) if have_ci else None
                _line_with_ci(axis, xx, yy, lo, hi, color=LANG_COLORS[lang], label=lang)
                for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                    interval = _interval(data_row, "ci_low", "ci_high")
                    _record(
                        ctx, data_row, panel="(a)", series=f"{lang} · {MODEL_LABELS[family]} · {fmt}",
                        x_label="Normalized depth ℓ/N", x=xval, y=yval,
                        source_file="E2-E5_la_curves.csv",
                        keys=("model", "format", "target_lang", "metric", "layer"),
                        value_col="value", x_source_col="layer", x_transform="layer/N",
                        x_denominator=n_layers,
                        ci_low_col="ci_low" if interval else "",
                        ci_high_col="ci_high" if interval else "", ci=interval,
                    )
            lm = landmarks.loc[landmarks["family"].eq(family) & landmarks["format"].eq(fmt)]
            for _, mark in lm.iterrows():
                metric = str(mark["metric"])
                if not _finite(mark["value"]) or not _finite(mark["n_layers"]) or _num(mark["n_layers"]) <= 0:
                    continue
                n_layers = _num(mark["n_layers"])
                xpos = _num(mark["value"]) / n_layers
                style = lm_style[metric]
                axis.axvline(xpos, color="#333333", linestyle=style, lw=0.85, alpha=0.75)
                interval = _interval(mark, "ci_low", "ci_high")
                if interval:
                    lo, hi = interval[0] / n_layers, interval[1] / n_layers
                    bars = axis.errorbar(
                        xpos, lane[metric], xerr=[[max(0.0, xpos - lo)], [max(0.0, hi - xpos)]],
                        transform=axis.get_xaxis_transform(), fmt="none", ecolor="#333333",
                        elinewidth=0.9, capsize=2.3,
                    )
                    for bar in bars.lines[2]:
                        bar.set_linestyle(style)
                keys = tuple(
                    key for key in
                    ("model", "format", "condition", "metric_variant", "item_set", "metric")
                    if key in mark.index
                )
                _record(
                    ctx, mark, panel="(a)", series=f"{metric} · {MODEL_LABELS[family]} · {fmt}",
                    x_label="Normalized landmark depth ℓ/N", x=xpos, y=lane[metric],
                    source_file="E2-E5_landmarks.csv", keys=keys, value_col="value",
                    value_axis="x", value_transform="layer/N",
                    x_source_col="value", x_transform="layer/N", x_denominator=n_layers,
                    plotted_value=xpos,
                    ci_low_col="ci_low" if interval else "",
                    ci_high_col="ci_high" if interval else "",
                    ci_transform="layer/N", ci_denominator=n_layers, ci=(
                        (interval[0] / n_layers, interval[1] / n_layers) if interval else None
                    ),
                )
            axis.set_xlim(0, 1)
            axis.set_xlabel("Normalized depth ℓ/N")
            if col_idx == 0:
                axis.set_ylabel("rise (nats)")
            axis.text(0.02, 0.94, f"({chr(97 + row_idx * 3 + col_idx)})", transform=axis.transAxes)
    lang_handles = [
        plt.Line2D([0], [0], color=LANG_COLORS[lang], label=lang, lw=1.2) for lang in LANG_ORDER
    ]
    lm_handles = [
        plt.Line2D([0], [0], color="#333333", linestyle=style, label=metric.replace("_", " "), lw=0.9)
        for metric, style in lm_style.items()
    ]
    figure.legend(
        handles=lang_handles, title="Target language", loc="lower center",
        ncol=5, bbox_to_anchor=(0.5, 0.035), handletextpad=0.35,
        columnspacing=0.8, labelspacing=0.25, borderpad=0.25,
    )
    figure.legend(
        handles=lm_handles, title="Landmark", loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.005), handletextpad=0.35,
        columnspacing=0.8, labelspacing=0.25, borderpad=0.25,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.17, hspace=0.32, wspace=0.22)
    ctx.save("fig03_rise_landmarks", figure)


def la_with_gaps(part: pd.DataFrame, gate: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """Insert missing layers and mask gate-excluded/undefined estimates as gaps."""
    result = part.set_index("layer").reindex(range(1, n_layers + 1)).reset_index()
    excluded = gate.loc[gate["gate_pass"].astype(str).str.lower().eq("false"), "layer"]
    columns = [c for c in ("value", "ci_low", "ci_high") if c in result]
    result.loc[result["layer"].isin(excluded) | result["value"].isna(), columns] = np.nan
    return result


def fig4_la_gate(ctx: FigureContext) -> None:
    """Plot language-specific LA curves, rule references, and A6 gate flags."""
    curves = ctx.csv("E2-E5_la_curves.csv")
    gate = ctx.csv("E5_gate.csv")
    _require(curves, "E2-E5_la_curves.csv", ("model", "format", "target_lang", "metric", "layer", "value"))
    _require(gate, "E5_gate.csv", ("family", "fit_format", "layer", "gate_pass"))
    curves = curves.loc[curves["metric"].eq("LA")].copy()
    if curves.empty:
        raise FigureSkip("E2-E5_la_curves.csv has no LA value rows")
    curves["family"] = curves["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    _facet_matrix(
        figure, axes, row_labels=("Repetition", "Translation"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    for row_idx, fmt in enumerate(("repetition", "translation")):
        for col_idx, family in enumerate(MODEL_ORDER):
            axis = axes[row_idx, col_idx]
            cell = curves.loc[curves["family"].eq(family) & curves["format"].eq(fmt)]
            gcell = gate.loc[gate["family"].map(_fam).eq(family) & gate["fit_format"].eq(fmt)]
            max_layer = float(gcell["layer"].max())
            for lang in LANG_ORDER:
                part = cell.loc[cell["target_lang"].eq(lang)].sort_values("layer", kind="stable")
                if part.empty:
                    continue
                part = la_with_gaps(part, gcell, int(max_layer))
                xx = part["layer"].to_numpy(dtype=float) / max_layer
                yy = part["value"].to_numpy(dtype=float)
                have_ci = {"ci_low", "ci_high"}.issubset(part.columns) and True
                lo = part["ci_low"].to_numpy(dtype=float) if have_ci else None
                hi = part["ci_high"].to_numpy(dtype=float) if have_ci else None
                _line_with_ci(axis, xx, yy, lo, hi, color=LANG_COLORS[lang], label=lang)
                for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                    if not _finite(yval):
                        continue
                    interval = _interval(data_row, "ci_low", "ci_high")
                    _record(
                        ctx, data_row, panel="(a)", series=f"{lang} · {MODEL_LABELS[family]} · {fmt}",
                        x_label="Normalized depth ℓ/N", x=xval, y=yval,
                        source_file="E2-E5_la_curves.csv",
                        keys=("model", "format", "target_lang", "metric", "layer"),
                        value_col="value", x_source_col="layer", x_transform="layer/N",
                        x_denominator=max_layer,
                        ci_low_col="ci_low" if interval else "",
                        ci_high_col="ci_high" if interval else "", ci=interval,
                    )
            gcell = gate.loc[
                gate["family"].map(_fam).eq(family) & gate["fit_format"].eq(fmt)
            ]
            for _, gate_row in gcell.iterrows():
                layer = _num(gate_row["layer"])
                xpos = layer / max_layer
                failed = str(gate_row["gate_pass"]).lower() == "false"
                if failed:
                    axis.axvspan(max(0.0, (layer - 0.5) / max_layer), (layer + 0.5) / max_layer, color="#888888", alpha=0.12, lw=0)
                    _record(
                        ctx, gate_row, panel="(a)", series=f"A6 gate-failing layer · {family} · {fmt}",
                        x_label="Normalized depth ℓ/N", x=xpos, y=0.02,
                        source_file="E5_gate.csv", keys=("family", "fit_format", "layer"),
                        value_col="gate_pass", value_axis="annotation",
                        annotation="shaded A6 gate-failing layer",
                        annotation_transform="presence",
                    )
                for flag_col, marker, label in (
                    ("label_dominated_last", "x", "label-dominated last"),
                    ("label_dominated_mean", "+", "label-dominated mean"),
                ):
                    if flag_col not in gate.columns or str(gate_row[flag_col]).lower() != "true":
                        continue
                    lane = 0.0 if flag_col == "label_dominated_last" else 0.025
                    axis.scatter(
                        [xpos], [lane], transform=axis.get_xaxis_transform(),
                        marker=marker, color="#111111", s=18, clip_on=False, zorder=5,
                    )
                    _record(
                        ctx, gate_row, panel="(a)", series=f"{label} · {family} · {fmt}",
                        x_label="Normalized depth ℓ/N", x=xpos, y=lane,
                        source_file="E5_gate.csv", keys=("family", "fit_format", "layer"),
                        value_col=flag_col, value_axis="annotation",
                        annotation=label, annotation_transform="presence",
                    )
            for reference, color, style in (
                (-0.25, "#000000", ":"), (0.20, "#999999", "--"), (0.50, "#000000", "-."),
            ):
                axis.axhline(reference, color=color, linestyle=style, lw=0.8)
            axis.set_xlim(0, 1)
            axis.set_xlabel("Normalized depth ℓ/N")
            if col_idx == 0:
                axis.set_ylabel("LA_L(ℓ)")
    legend = [
        plt.Line2D([0], [0], color=LANG_COLORS[lang], label=lang, lw=1.1) for lang in LANG_ORDER if lang in set(curves["target_lang"])
    ]
    legend.extend(
        [
            plt.Line2D([0], [0], color="#000000", linestyle=":", label="Null −0.25"),
            plt.Line2D([0], [0], color="#999999", linestyle="--", label="H2 0.20"),
            plt.Line2D([0], [0], color="#000000", linestyle="-.", label="H1 0.50"),
        ]
    )
    legend.extend([
        plt.Line2D([], [], marker="x", color="black", ls="none", label="label-dominated (last)"),
        plt.Line2D([], [], marker="+", color="black", ls="none", label="label-dominated (mean)"),
        plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.12, label="A6 gate fail"),
    ])
    for axis in axes.flat:
        axis.tick_params(labelbottom=True, labelleft=True)
    figure.legend(
        handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015),
        handletextpad=0.35, columnspacing=0.8, labelspacing=0.25, borderpad=0.25,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.18, hspace=0.32, wspace=0.22)
    ctx.save("fig04_la_gate", figure)


def _ci_type_sensitivity(row: Mapping[str, Any]) -> str:
    """Classify a sensitivity interval from its stored bootstrap key or SE."""
    if "bootstrap_key" in row and not pd.isna(row["bootstrap_key"]) and str(row["bootstrap_key"]).strip():
        return "bootstrap"
    if "standard_error" in row and _finite(row["standard_error"]):
        return "Wald"
    return "CI type unstated"


def _forest_point(
    axis: plt.Axes,
    value: float,
    low: float | None,
    high: float | None,
    y: float,
    *,
    color: str,
    ci_type: str,
) -> None:
    """Draw one horizontal forest point with the marker/whisker style for its CI."""
    if low is not None and high is not None:
        container = axis.errorbar(
            value, y, xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
            fmt="o", color=color, ecolor=color, capsize=2.2, ms=3.6,
            markerfacecolor="none" if ci_type == "Wald" else color,
            markeredgecolor=color,
        )
        if ci_type == "Wald":
            for bar in container.lines[2]:
                bar.set_linestyle("--")
    else:
        axis.plot(
            value, y, marker="o", linestyle="none", color=color, ms=3.6,
            markerfacecolor="none" if ci_type == "Wald" else color,
        )


def sensitivity_label(sensitivity: str) -> str:
    """Name each sensitivity by its registered source, without reclassifying it."""
    if sensitivity == "R11_excluded":
        return "R11: excluded"
    if sensitivity == "native_strict":
        return "OPS-28: native_strict"
    if sensitivity in {"zipf_vi", "zipf_zh"}:
        return f"A8: {sensitivity}"
    return f"R6: {sensitivity.removeprefix('R6_')}"


def fig5_e6_forest(ctx: FigureContext) -> None:
    """Plot E6 D_zh, §7.1 interaction, and R6/R11 sensitivities."""
    dzh = ctx.csv("E6_dzh.csv")
    terms = ctx.csv("E6_model_terms.csv")
    sensitivity = ctx.csv("E6_sensitivity.csv")
    _require(dzh, "E6_dzh.csv", ("model", "format", "instrument", "estimate"))
    _require(terms, "E6_model_terms.csv", ("model", "instrument", "term", "estimate"))
    _require(sensitivity, "E6_sensitivity.csv", ("model", "sensitivity", "instrument", "estimate"))
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(18, 12))
    selected_sensitivities = {
        "bt_pass_strict_v1", "external_attested", "all_three_nllb",
        "min_surface_dist_ge_060", "R11_excluded", "native_strict",
    }
    specs = (
        ("D_zh", dzh, "primary", "estimate", "ci_low", "ci_high", ("model", "format", "instrument")),
        ("rise_zh_interaction", terms, "primary", "estimate", "bootstrap_ci_low", "bootstrap_ci_high", ("model", "instrument", "term")),
    )
    for panel_idx, (instrument, source, _, value_col, low_col, high_col, keys) in enumerate(specs):
        axis = axes[panel_idx]
        rows: list[tuple[str, Mapping[str, Any], str, str, str, str, tuple[str, ...]]] = []
        if instrument == "D_zh":
            for _, row in source.loc[source["instrument"].eq("D_zh")].iterrows():
                if not _finite(row[value_col]):
                    continue
                rows.append((f"{MODEL_LABELS[_fam(row['model'])]} · {row['format']}", row, "E6_dzh.csv", value_col, low_col, high_col, keys))
        else:
            for _, row in source.loc[source["instrument"].eq(instrument)].iterrows():
                if not _finite(row[value_col]):
                    continue
                rows.append((f"{MODEL_LABELS[_fam(row['model'])]} · primary", row, "E6_model_terms.csv", value_col, low_col, high_col, keys))
        for _, row in sensitivity.loc[
            sensitivity["instrument"].eq(instrument)
            & sensitivity["sensitivity"].isin(selected_sensitivities | ({"zipf_vi", "zipf_zh"} if instrument != "D_zh" else set()))
        ].iterrows():
            if not _finite(row[value_col]):
                continue
            sens = str(row["sensitivity"])
            label = sensitivity_label(sens)
            row_keys = ("model", "sensitivity", "instrument")
            rows.append((f"{MODEL_LABELS[_fam(row['model'])]} · {label}", row, "E6_sensitivity.csv", value_col, "ci_low", "ci_high", row_keys))
        rows.sort(key=lambda item: (
            MODEL_ORDER.index(_fam(item[1]["model"])),
            0 if item[2] != "E6_sensitivity.csv" else 1,
            str(item[0]),
        ))
        labels: list[str] = []
        for index, (label, row, source_file, value_col, low_col, high_col, keys) in enumerate(rows):
            value = _num(row[value_col])
            interval = _interval(row, low_col, high_col)
            if source_file == "E6_model_terms.csv":
                ci_type = "bootstrap"
            elif source_file == "E6_sensitivity.csv":
                ci_type = _ci_type_sensitivity(row)
            else:
                ci_type = "bootstrap"
            if ci_type == "CI type unstated":
                label = f"{label} · CI type unstated"
            color = MODEL_COLORS[_fam(row["model"])]
            y = float(index)
            low = interval[0] if interval else None
            high = interval[1] if interval else None
            _forest_point(axis, value, low, high, y, color=color, ci_type=ci_type)
            labels.append(label)
            _record(
                ctx, row, panel="(a)" if panel_idx == 0 else "(b)",
                series=f"{label} · {ci_type}", x_label="Estimate",
                x=value, y=y, source_file=source_file, keys=keys,
                value_col=value_col, value_axis="x",
                ci_low_col=low_col if interval else "", ci_high_col=high_col if interval else "",
                ci=interval,
            )
        axis.axvline(0.0, color="#555555", linestyle=":", lw=0.8)
        axis.set_yticks(range(len(labels)), labels, fontsize=6.5)
        axis.set_xlabel("Estimate (95% CI: filled = bootstrap, open = Wald)")
        axis.set_title("(a) D_zh: LA_zh(sino) − LA_zh(nonsino)" if panel_idx == 0 else "(b) §7.1 term stratum[sino] × language[zh]")
        for tick, label in zip(axis.get_yticklabels(), labels):
            if "pooled" in label or "· primary" in label:
                tick.set_fontweight("bold")
        axis.invert_yaxis()
    figure.subplots_adjust(left=0.24, right=0.99, top=0.95, bottom=0.09, wspace=0.95)
    ctx.save("fig05_e6_forest", figure)


def earliest_flip_peak(rates: pd.DataFrame) -> pd.Series:
    """Look up the earliest maximum translation k1 en_rate (OPS-27)."""
    part = rates.loc[rates["setting"].eq("k1") & rates["format"].eq("translation")]
    return part.loc[part["en_rate"].eq(part["en_rate"].max())].sort_values("layer", kind="stable").iloc[0]


def fig6_e8_rates(ctx: FigureContext) -> None:
    """Plot translation flip/damage rates and both-format mean deltas."""
    rates = ctx.csv("E8_rates.csv")
    delta = ctx.csv("E8_delta.csv")
    landmarks = ctx.csv("E2-E5_landmarks.csv")
    _require(rates, "E8_rates.csv", ("family", "format", "setting", "layer", "n", "en_rate", "damage_count"))
    _require(delta, "E8_delta.csv", ("family", "format", "setting", "layer", "mean_delta"))
    _require(landmarks, "E2-E5_landmarks.csv", ("model", "format", "condition", "metric_variant", "item_set", "metric", "value", "n_layers"))
    rates["family_key"] = rates["family"].map(_fam)
    delta["family_key"] = delta["family"].map(_fam)
    landmarks = landmarks.loc[
        landmarks["condition"].eq("diac") & landmarks["metric_variant"].eq("s")
        & landmarks["item_set"].eq("cf729") & landmarks["metric"].eq("L_cross")
    ].copy()
    landmarks["family_key"] = landmarks["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(4, 3, figsize=(14, 12), sharex="col")
    _facet_matrix(
        figure, axes, row_labels=("(a) Translation EN flip rate", "(b) Translation DAMAGE rate", "(c) Repetition Δ", "(d) Translation Δ"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    marker_styles = {setting: "-" for setting in SETTING_ORDER}
    for col_idx, family in enumerate(MODEL_ORDER):
        axis = axes[0, col_idx]
        damage_axis = axes[1, col_idx]
        cell = rates.loc[
            rates["family_key"].eq(family) & rates["format"].eq("translation")
            & rates["setting"].isin(("baseline", *SETTING_ORDER))
        ]
        max_layer = float(cell["layer"].max())
        for setting in ("baseline", *SETTING_ORDER):
            part = cell.loc[cell["setting"].eq(setting)].sort_values("layer", kind="stable")
            if part.empty:
                continue
            xx = part["layer"].to_numpy(dtype=float) / max_layer
            yy = part["en_rate"].to_numpy(dtype=float)
            axis.plot(xx, yy, color=SETTING_COLORS[setting], linestyle="-", marker="o", ms=2.2, lw=0.9, label=f"{setting} en_rate")
            if setting == "baseline":
                axis.annotate("unsteered", (xx[0], yy[0]), xytext=(5, 7), textcoords="offset points", fontsize=7)
            for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                _record(
                    ctx, data_row, panel="(a)", series=f"{setting} en_rate · {family}",
                    x_label="Normalized depth ℓ/N", x=xval, y=yval,
                    source_file="E8_rates.csv",
                    keys=("family", "format", "setting", "layer"),
                    value_col="en_rate", x_source_col="layer", x_transform="layer/N",
                    x_denominator=max_layer,
                )
            if setting in ("baseline", *SETTING_ORDER):
                damage = part["damage_count"].to_numpy(dtype=float) / part["n"].to_numpy(dtype=float)
                damage_axis.plot(xx, damage, color=SETTING_COLORS[setting], linestyle="--", marker="s", ms=2.0, lw=0.85, label=f"{setting} DAMAGE")
                if setting == "baseline":
                    damage_axis.annotate("unsteered", (xx[0], damage[0]), xytext=(5,7), textcoords="offset points", fontsize=7)
                for (_, data_row), xval, yval in zip(part.iterrows(), xx, damage):
                    _record(
                        ctx, data_row, panel="(b)", series=f"{setting} DAMAGE · {family}",
                        x_label="Normalized depth ℓ/N", x=xval, y=yval,
                        source_file="E8_rates.csv",
                        keys=("family", "format", "setting", "layer"),
                        value_col="damage_count", value_transform="damage_count/n",
                        value_aux_col="n", plotted_value=yval,
                        x_source_col="layer", x_transform="layer/N",
                        x_denominator=max_layer,
                    )
        peak = earliest_flip_peak(cell)
        expected = {"gemma": 18, "qwen": 24, "llama": 14}[family]
        assert int(peak["layer"]) == expected, f"OPS-27 flip peak mismatch: {family}: {peak['layer']} != {expected}"
        peak_x = _num(peak["layer"]) / max_layer
        axis.scatter([peak_x], [peak["en_rate"]], marker="*", s=75, color=SETTING_COLORS["k1"], edgecolors="black", zorder=6)
        axis.annotate(f"flip peak: {expected}", (peak_x, peak["en_rate"]), xytext=(4, 12), textcoords="offset points", fontsize=7)
        _record(ctx, peak, panel="(a)", series=f"OPS-27 flip peak · {family}", x_label="Normalized depth ℓ/N",
                x=peak_x, y=peak["en_rate"], source_file="E8_rates.csv", keys=("family", "format", "setting", "layer"),
                value_col="en_rate", x_source_col="layer", x_transform="layer/N", x_denominator=max_layer,
                annotation=f"flip peak: {expected}")
        mark = landmarks.loc[landmarks["family_key"].eq(family) & landmarks["format"].eq("translation")]
        for _, lm in mark.iterrows():
            if not _finite(lm["value"]) or not _finite(lm["n_layers"]) or _num(lm["n_layers"]) <= 0:
                continue
            n_layers = _num(lm["n_layers"])
            xpos = _num(lm["value"]) / n_layers
            axis.axvline(xpos, color="#111111", linestyle="-.", lw=0.8)
            keys = ("model", "format", "condition", "metric_variant", "item_set", "metric")
            _record(
                ctx, lm, panel="(a)", series=f"L_cross · {family} · translation",
                x_label="Normalized L_cross depth", x=xpos, y=0.0,
                source_file="E2-E5_landmarks.csv", keys=keys,
                value_col="value", value_axis="x", value_transform="layer/N",
                x_source_col="value", x_transform="layer/N", x_denominator=n_layers,
                plotted_value=xpos,
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("Normalized depth ℓ/N")
        if col_idx == 0:
            axis.set_ylabel("Rate (no interval in source)")

        damage_axis.set_xlim(0, 1)
        damage_axis.set_ylim(bottom=0)
        damage_axis.set_xlabel("Normalized depth ℓ/N")
        if col_idx == 0:
            damage_axis.set_ylabel("DAMAGE rate (no interval in source)")
        for row_idx, fmt in ((2, "repetition"), (3, "translation")):
            delta_axis = axes[row_idx, col_idx]
            delta_cell = delta.loc[delta["family_key"].eq(family) & delta["format"].eq(fmt)]
            settings = sorted(delta_cell["setting"].dropna().astype(str).unique())
            for setting_idx, setting in enumerate(settings):
                part = delta_cell.loc[delta_cell["setting"].eq(setting)].sort_values("layer", kind="stable")
                if part.empty:
                    continue
                max_layer = max(1.0, float(part["layer"].max()))
                xx = part["layer"].to_numpy(dtype=float) / max_layer
                yy = part["mean_delta"].to_numpy(dtype=float)
                lo = part["ci_low"].to_numpy(dtype=float) if "ci_low" in part.columns and part["ci_low"].notna().all() else None
                hi = part["ci_high"].to_numpy(dtype=float) if "ci_high" in part.columns and part["ci_high"].notna().all() else None
                color = SETTING_COLORS[setting]
                _line_with_ci(delta_axis, xx, yy, lo, hi, color=color, label=setting,
                              marker="o" if setting == "baseline" else None, alpha=0.12)
                if setting == "baseline":
                    delta_axis.annotate("unsteered", (xx[0], yy[0]), xytext=(5, 5), textcoords="offset points", fontsize=7)
                for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                    interval = _interval(data_row, "ci_low", "ci_high")
                    _record(
                        ctx, data_row, panel="(c)" if fmt == "repetition" else "(d)", series=f"{fmt} · {setting} · {family}",
                        x_label="Normalized depth ℓ/N", x=xval, y=yval,
                        source_file="E8_delta.csv", keys=("family", "format", "setting", "layer"),
                        value_col="mean_delta", x_source_col="layer",
                        x_transform="layer/N", x_denominator=max_layer,
                        ci_low_col="ci_low" if interval else "",
                        ci_high_col="ci_high" if interval else "", ci=interval,
                    )
            delta_axis.axhline(0.0, color="#555555", linestyle=":", lw=0.7)
            delta_axis.set_xlim(0, 1)
            delta_axis.set_xlabel("Normalized depth ℓ/N")
            if col_idx == 0:
                delta_axis.set_ylabel("Mean Δ (95% CI where present)")
    for axis in axes.flat:
        axis.tick_params(labelbottom=True, labelleft=True)
    setting_handles = [plt.Line2D([], [], color=color, label="unsteered baseline" if setting == "baseline" else setting)
                       for setting, color in SETTING_COLORS.items()]
    measure_handles = [plt.Line2D([], [], color="black", ls="-", marker="o", label="en_rate"),
                       plt.Line2D([], [], color="black", ls="--", marker="s", label="DAMAGE"),
                       plt.Line2D([], [], color="black", ls="-", label="Δ"),
                       plt.Line2D([], [], color="black", marker="*", ls="none", label="OPS-27 flip peak (k1)")]
    figure.legend(handles=setting_handles, title="Setting", loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.035))
    figure.legend(handles=measure_handles, title="Measure", loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015))
    figure.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.17, hspace=0.45, wspace=0.3)
    ctx.save("fig06_e8_rates", figure)


def fig7_e7_projection(ctx: FigureContext) -> None:
    """Plot six attention-share cells with signed band sums and residual projections."""
    all_bands = ctx.csv("E7_band.csv")
    all_bands = all_bands.loc[all_bands["item_set"].eq("cf729") & all_bands["direction"].eq("d_vi_en")].copy()
    all_bands["family"] = all_bands["model"].map(_fam)
    bands = all_bands.loc[all_bands["component"].eq("share")]
    layers = ctx.csv("E7_layers.csv")
    layers = layers.loc[layers["item_set"].eq("cf729") & layers["direction"].eq("d_vi_en")].copy()
    layers["family"] = layers["model"].map(_fam)
    _style()
    figure = plt.figure(figsize=(15, 12))
    grid = figure.add_gridspec(3, 3, height_ratios=(1.5, 1, 1), hspace=0.6, wspace=0.3)
    share_axis = figure.add_subplot(grid[0, :])
    axes = np.array([[figure.add_subplot(grid[r+1, c]) for c in range(3)] for r in range(2)])
    cells = [(f, t) for f in MODEL_ORDER for t in ("repetition", "translation")]
    band_styles = (("primary", "o", -0.18), ("slope_sensitivity", "s", 0), ("ops37_as_written", "^", 0.18))
    for y, (family, fmt) in enumerate(cells):
        cell = bands.loc[bands["family"].eq(family) & bands["format"].eq(fmt)]
        for band, marker, dodge in band_styles:
            datum = cell.loc[cell["band"].eq(band)].iloc[0]
            value = _num(datum["estimate"])
            interval = _interval(datum, "ci_low", "ci_high")
            color = MODEL_COLORS[family]
            share_axis.errorbar(value, y+dodge, xerr=[[value-interval[0]], [interval[1]-value]] if interval else None,
                                fmt=marker, color=color, mfc=color if band == "primary" else "white", capsize=2, ms=5)
            _record(ctx, datum, panel="(a)", series=f"{family} · {fmt} · {band}", x_label="Attention share", x=value, y=y+dodge,
                    source_file="E7_band.csv", keys=("model", "format", "item_set", "direction", "band", "component"),
                    value_col="estimate", value_axis="x", ci_low_col="ci_low" if interval else "", ci_high_col="ci_high" if interval else "", ci=interval)
            if band == "primary":
                signs = []
                signed = all_bands.loc[all_bands["family"].eq(family) & all_bands["format"].eq(fmt) & all_bands["band"].eq(band)]
                for component in ("attn", "mlp"):
                    signrow = signed.loc[signed["component"].eq(component)].iloc[0]
                    sign = _num(signrow["sign"])
                    signs.append("+" if sign > 0 else "−" if sign < 0 else "0")
                    _record(ctx, signrow, panel="(a)", series=f"{family} · {fmt} · {component} sign", x_label="Attention share", x=value, y=y+dodge,
                            source_file="E7_band.csv", keys=("model", "format", "item_set", "direction", "band", "component"),
                            value_col="sign", value_axis="annotation", annotation_transform="numeric")
                share_axis.annotate(f"({signs[0]}, {signs[1]})", (value, y+dodge), xytext=(8, 7), textcoords="offset points", fontsize=8, va="bottom")
            if family == "gemma" and fmt == "translation" and band == "ops37_as_written":
                included = str(datum["included_blocks"]).split(";")
                text = f"{len(included)} of {int(datum['requested_size'])} blocks"
                share_axis.annotate(text, (value, y+dodge), xytext=(-8, -10), textcoords="offset points", ha="right", va="top", fontsize=8)
                for column in ("included_blocks", "requested_size"):
                    _record(ctx, datum, panel="(a)", series=f"Gemma translation as written {column}", x_label="Attention share", x=value, y=y+dodge,
                            source_file="E7_band.csv", keys=("model", "format", "item_set", "direction", "band", "component"),
                            value_col=column, value_axis="annotation")
    share_axis.set_yticks(range(6), [f"{MODEL_LABELS[f]} {'rep' if t == 'repetition' else 'tr'}" for f,t in cells])
    share_axis.set_xlim(0, 1); share_axis.set_xticks((0, .25, .5, .75, 1))
    share_axis.set_xlabel("Attention share (source 95% CI)")
    share_axis.set_title("(a)", loc="left")
    share_axis.invert_yaxis()
    share_axis.legend(handles=[plt.Line2D([], [], marker=m, color="#555555", mfc="#555555" if band == "primary" else "white", ls="none", label=label)
                              for (band,m,_),label in zip(band_styles,("Primary", "Slope sensitivity", "OPS-37 as written"))], loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3)
    _facet_matrix(figure, axes, row_labels=("Repetition", "Translation"), col_labels=tuple(MODEL_LABELS[f] for f in MODEL_ORDER))
    for row_idx, fmt in enumerate(("repetition", "translation")):
        for col_idx, family in enumerate(MODEL_ORDER):
            cell = bands.loc[bands["family"].eq(family) & bands["format"].eq(fmt)]
            layer_axis = axes[row_idx, col_idx]
            layer_cell = layers.loc[layers["family"].eq(family) & layers["format"].eq(fmt)]
            for component, color in (("attn", "#0072B2"), ("mlp", "#E69F00")):
                subset = layer_cell.loc[layer_cell["component"].eq(component)].sort_values("layer", kind="stable")
                if subset.empty:
                    continue
                max_layer = float(subset["layer"].max())
                xx = subset["layer"].to_numpy(dtype=float) / max_layer
                yy = subset["mean_projection"].to_numpy(dtype=float)
                have_ci = {"ci_low", "ci_high"}.issubset(subset.columns) and subset["ci_low"].notna().all() and subset["ci_high"].notna().all()
                lo = subset["ci_low"].to_numpy(dtype=float) if have_ci else None
                hi = subset["ci_high"].to_numpy(dtype=float) if have_ci else None
                _line_with_ci(layer_axis, xx, yy, lo, hi, color=color, label=component, marker="o")
                for (_, data_row), xval, yval in zip(subset.iterrows(), xx, yy):
                    interval = _interval(data_row, "ci_low", "ci_high")
                    _record(
                        ctx, data_row, panel="(b)", series=f"{component} · {family} · {fmt}",
                        x_label="Normalized depth ℓ/N", x=xval, y=yval,
                        source_file="E7_layers.csv",
                        keys=("model", "format", "item_set", "direction", "layer", "component"),
                        value_col="mean_projection", x_source_col="layer",
                        x_transform="layer/N", x_denominator=max_layer,
                        ci_low_col="ci_low" if interval else "",
                        ci_high_col="ci_high" if interval else "", ci=interval,
                    )
            primary = cell.loc[cell["band"].eq("primary")]
            if not primary.empty:
                blocks_value = primary.iloc[0].get("blocks", "")
                if pd.notna(blocks_value):
                    for block in str(blocks_value).split(";"):
                        if not block:
                            continue
                        layer_value = float(block)
                        max_layer = float(layer_cell["layer"].max()) if not layer_cell.empty else 1.0
                        layer_axis.axvspan(
                            max(0.0, (layer_value - 0.5) / max_layer),
                            min(1.0, (layer_value + 0.5) / max_layer),
                            color="#999999", alpha=0.12, lw=0,
                        )
            if not layer_cell.empty:
                max_layer = float(layer_cell["layer"].max())
                for _, gate_row in layer_cell.loc[layer_cell["gate_pass"].astype(str).str.lower().eq("false")].iterrows():
                    xpos = _num(gate_row["layer"]) / max_layer
                    layer_axis.scatter([xpos], [0.0], transform=layer_axis.get_xaxis_transform(), marker="x", color="#D55E00", s=18, clip_on=False)
            layer_axis.axhline(0, color="#888888", ls=":", lw=0.7)
            layer_axis.tick_params(labelbottom=True, labelleft=True)
            layer_axis.set_xlim(0, 1)
            layer_axis.set_xlabel("Normalized depth ℓ/N")
            if col_idx == 0:
                layer_axis.set_ylabel("Projection onto d̂ (95% CI where present)")
            layer_axis.legend(loc="best", fontsize=6)
    axes[0, 0].set_title("(b)", loc="left", pad=20)
    figure.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.07)
    ctx.save("fig07_e7_projection", figure)


def supp1_r1_forest(ctx: FigureContext) -> None:
    """Plot R1 quantities in metric columns and model-format row blocks."""
    frame = ctx.csv("R1_metrics.csv")
    _require(frame, "R1_metrics.csv", ("model", "format", "fewshot_set", "metric_variant", "quantity", "value", "set1_value"))
    frame = frame.loc[frame["quantity"].isin(("Q_H1", "H1_ratio", "max_LA_en"))].copy()
    _style()
    quantity_order = ("Q_H1", "H1_ratio", "max_LA_en")
    cells = [(family, fmt) for family in MODEL_ORDER for fmt in ("repetition", "translation")]
    figure, axes = plt.subplots(6, 3, figsize=(13.5, 11), sharey=True)
    for row_idx, (family, fmt) in enumerate(cells):
        cell = frame.loc[frame["format"].eq(fmt) & frame["model"].map(_fam).eq(family)]
        for col_idx, quantity in enumerate(quantity_order):
            axis = axes[row_idx, col_idx]
            axis.axhspan(0.5, 1.5, color="#F0E442", alpha=0.22, zorder=0)
            if quantity == "Q_H1":
                axis.axvline(0.0, color="#555555", linestyle=":", lw=0.8)
            if quantity == "max_LA_en":
                axis.axvline(0.50, color="#000000", linestyle="-.", lw=0.8)
            subset = cell.loc[cell["quantity"].eq(quantity)]
            for variant, marker in (("s", "o"), ("s1", "s")):
                vrows = subset.loc[subset["metric_variant"].eq(variant)].sort_values("fewshot_set", kind="stable")
                for plotted_set in range(1, 7):
                    if plotted_set == 1:
                        if vrows.empty or not _finite(vrows.iloc[0].get("set1_value")):
                            continue
                        datum = vrows.iloc[0]
                        value_col = "set1_value"
                        low_col, high_col = "set1_ci_low", "set1_ci_high"
                        value = _num(datum[value_col])
                        interval = _interval(datum, low_col, high_col)
                        source_set = datum["fewshot_set"]
                    else:
                        hit = vrows.loc[vrows["fewshot_set"].eq(plotted_set)]
                        if hit.empty or not _finite(hit.iloc[0].get("value")):
                            continue
                        datum = hit.iloc[0]
                        value_col = "value"
                        low_col, high_col = "ci_low", "ci_high"
                        value = _num(datum[value_col])
                        interval = _interval(datum, low_col, high_col)
                        source_set = datum["fewshot_set"]
                    y = float(plotted_set) + (-0.15 if variant == "s" else 0.15)
                    color = MODEL_COLORS[family]
                    if interval:
                        bars = axis.errorbar(
                            value, y,
                            xerr=[[max(0.0, value - interval[0])], [max(0.0, interval[1] - value)]],
                            fmt=marker, color=color, markerfacecolor=color if variant == "s" else "white",
                            markeredgecolor=color, ms=4.1 if plotted_set == 1 else 3.4,
                            capsize=2, elinewidth=0.8, zorder=3,
                        )
                        if variant == "s1":
                            for bar in bars.lines[2]:
                                bar.set_linestyle("--")
                    else:
                        axis.plot(
                            value, y, marker=marker, color=color, linestyle="none",
                            markerfacecolor=color if variant == "s" else "white",
                            markeredgecolor=color, ms=4.1 if plotted_set == 1 else 3.4,
                        )
                    _record(
                        ctx, datum, panel=f"{MODEL_LABELS[family]} · {fmt}",
                        series=f"{quantity} · {variant} · set {plotted_set}",
                        x_label=quantity, x=value, y=y,
                        source_file="R1_metrics.csv",
                        keys=("model", "format", "fewshot_set", "metric_variant", "quantity"),
                        value_col=value_col, value_axis="x",
                        ci_low_col=low_col if interval else "",
                        ci_high_col=high_col if interval else "", ci=interval,
                        annotation=f"plotted set {plotted_set}; source fewshot_set {source_set}",
                    )
            axis.set_yticks(range(1, 7), ("1", "2", "3", "4", "5", "6"))
            axis.tick_params(labelleft=True)
            axis.set_ylim(6.5, 0.5)
            axis.grid(axis="y", color="#dddddd", lw=0.4)
            if col_idx == 0:
                axis.set_ylabel(f"{MODEL_LABELS[family]} · {fmt}\nSet")
            if row_idx == 0:
                axis.text(0.5, 1.10, quantity.replace("_", " "), transform=axis.transAxes, ha="center", fontsize=8, fontweight="bold")
            axis.set_xlabel("Estimate (95% CI where present)")
    legend = [
        plt.Line2D([0], [0], marker="o", color="#555555", linestyle="none", label="s"),
        plt.Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="#555555", color="#555555", linestyle="none", label="s¹"),
        plt.Line2D([0], [0], color="#F0E442", alpha=0.7, lw=5, label="Set 1"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    figure.subplots_adjust(left=0.19, right=0.99, top=0.95, bottom=0.07, hspace=0.5, wspace=0.34)
    ctx.save("supp01_r1_forest", figure)


def supp2_r2_validation(ctx: FigureContext) -> None:
    """Plot validation counts and matched primary max-LA reference markers."""
    classification = ctx.csv("R2_classification.csv")
    alignment = ctx.csv("R2_alignment.csv")
    directions = ctx.csv("R2_directions.csv")
    _require(classification, "R2_classification.csv", ("model", "format", "pooling", "layer", "flores_correct", "matched_correct", "total"))
    _require(alignment, "R2_alignment.csv", ("model", "format", "pooling", "max_LA_en", "primary_max_LA_en"))
    classification["family"] = classification["model"].map(_fam)
    alignment["family"] = alignment["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(6, 3, figsize=(14, 14))
    _facet_matrix(
        figure, axes, row_labels=("Repetition", "Translation", "Repetition alignment", "Translation alignment", "Repetition cosine", "Translation cosine"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    for row_idx, fmt in enumerate(("repetition", "translation")):
        for col_idx, family in enumerate(MODEL_ORDER):
            axis = axes[row_idx, col_idx]
            cell = classification.loc[classification["family"].eq(family) & classification["format"].eq(fmt)]
            for pool_idx, pooling in enumerate(sorted(cell["pooling"].dropna().astype(str).unique())):
                part = cell.loc[cell["pooling"].eq(pooling)].sort_values("layer", kind="stable")
                if part.empty:
                    continue
                n_layers = float(part["layer"].max())
                xx = part["layer"].to_numpy(dtype=float) / n_layers
                for value_col, color, label in (
                    ("flores_correct", "#0072B2", "FLORES+"),
                    ("matched_correct", "#D55E00", "Matched"),
                ):
                    if value_col not in part.columns:
                        continue
                    yy = part[value_col].to_numpy(dtype=float)
                    axis.plot(xx, yy, color=color, linestyle=("-" if pool_idx == 0 else "--"), marker="o", ms=2.3, lw=0.9, label=f"{label} · {pooling}")
                    for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                        _record(
                            ctx, data_row, panel="(a)", series=f"{label} · {pooling} · {family} · {fmt}",
                            x_label="Normalized depth ℓ/N", x=xval, y=yval,
                            source_file="R2_classification.csv",
                            keys=("model", "format", "pooling", "layer"),
                            value_col=value_col, x_source_col="layer", x_transform="layer/N",
                            x_denominator=n_layers,
                        )
            axis.axhline(124, color="#333333", linestyle=":", lw=0.9)
            axis.set_xlim(0, 1)
            axis.set_xlabel("Normalized depth ℓ/N")
            if col_idx == 0:
                axis.set_ylabel("Correct of 130")
            align_axis = axes[row_idx + 2, col_idx]
            align = alignment.loc[alignment["family"].eq(family) & alignment["format"].eq(fmt)].sort_values("pooling", kind="stable")
            labels: list[str] = []
            for yidx, (_, datum) in enumerate(align.iterrows()):
                labels.append(str(datum["pooling"]))
                for colname, low_col, high_col, marker, label in (
                    ("max_LA_en", "ci_low", "ci_high", "o", "FLORES"),
                    ("primary_max_LA_en", "primary_ci_low", "primary_ci_high", "D", "Matched (primary, last pooling)"),
                ):
                    if colname == "primary_max_LA_en" and datum["pooling"] != "last":
                        continue
                    if colname not in datum.index or not _finite(datum[colname]):
                        continue
                    value = _num(datum[colname])
                    interval = _interval(datum, low_col, high_col)
                    color = "#0072B2" if label == "FLORES" else "#000000"
                    ycoord = float(yidx) + (0.12 if label == "Matched (primary, last pooling)" else -0.12)
                    if interval:
                        align_axis.errorbar(
                            value, ycoord,
                            xerr=[[max(0.0, value - interval[0])], [max(0.0, interval[1] - value)]],
                            fmt=marker, color=color, capsize=2, ms=3.2,
                        )
                    else:
                        align_axis.plot(value, ycoord, marker=marker, color=color, linestyle="none", ms=3.2)
                    _record(
                        ctx, datum, panel="(b)", series=f"{label} · {family} · {fmt} · {datum['pooling']}",
                        x_label="Max LA_en", x=value, y=ycoord,
                        source_file="R2_alignment.csv",
                        keys=("model", "format", "pooling"), value_col=colname,
                        value_axis="x", ci_low_col=low_col if interval else "",
                        ci_high_col=high_col if interval else "", ci=interval,
                    )
            align_axis.set_yticks(range(len(labels)), labels, fontsize=6)
            align_axis.set_xlabel("Max LA_en (95% CI where present)")
            cos_axis = axes[row_idx + 4, col_idx]
            cosine = directions.loc[directions["model"].map(_fam).eq(family) & directions["format"].eq(fmt)]
            n_layers = float(classification.loc[classification["family"].eq(family), "layer"].max())
            for lang in LANG_ORDER:
                for pool in ("last", "mean"):
                    part = cosine.loc[cosine["pooling"].eq(pool)].sort_values("layer", kind="stable")
                    column = f"v_{lang}"
                    xx = part["layer"].to_numpy(dtype=float) / n_layers
                    cos_axis.plot(xx, part[column], color=LANG_COLORS[lang], ls="-" if pool == "last" else "--", lw=1)
                    for (_, datum), x in zip(part.iterrows(), xx):
                        _record(ctx, datum, panel="(c)", series=f"{family} · {fmt} · {lang} · {pool}", x_label="Normalized depth ℓ/N", x=x, y=datum[column],
                                source_file="R2_directions.csv", keys=("model", "format", "pooling", "layer"), value_col=column,
                                x_source_col="layer", x_transform="layer/N", x_denominator=n_layers)
            cos_axis.set_xlabel("Normalized depth ℓ/N")
            cos_axis.set_ylabel("Direction cosine")
            cos_axis.axhline(0, color="#888888", ls=":", lw=.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015), fontsize=6)
    figure.legend(handles=[plt.Line2D([], [], marker="o", color="#0072B2", ls="none", label="FLORES"),
                           plt.Line2D([], [], marker="D", color="black", ls="none", label="Matched (primary, last pooling)")],
                  loc="lower center", ncol=2, bbox_to_anchor=(.5, -.01))
    figure.legend(handles=[plt.Line2D([], [], color=LANG_COLORS[l], label=l) for l in LANG_ORDER] +
                  [plt.Line2D([], [], color="black", ls=style, label=pool) for pool,style in (("last", "-"),("mean", "--"))],
                  loc="lower center", ncol=7, bbox_to_anchor=(.5, -.04))
    figure.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.09, hspace=0.65, wspace=0.24)
    ctx.save("supp02_r2_validation", figure)


def supp3_e9_paired(ctx: FigureContext) -> None:
    """Separate paired quantities and show per-model A10 dumbbells."""
    frame = ctx.csv("E9.csv")
    _style()
    figure = plt.figure(figsize=(18, 18))
    quantities = ("Q_H1", "H1_ratio", "H2_band_ratio", "max_LA_en")
    cells = [(f,t) for f in MODEL_ORDER for t in ("repetition", "translation", "cloze")]
    # Three formats remain separate; nine row blocks each carry independent quantity axes.
    grid = figure.add_gridspec(11, 12, height_ratios=(*([1]*9), .3, 2), hspace=.9, wspace=2.4)
    for i, (family, fmt) in enumerate(cells):
        for j, quantity in enumerate(quantities):
            axis = figure.add_subplot(grid[i, j*3:(j+1)*3])
            part = frame.loc[frame["model"].map(_fam).eq(family) & frame["format"].eq(fmt)
                             & frame["quantity"].eq(quantity) & frame["condition"].eq("nodiac_minus_diac")]
            for _, datum in part.iterrows():
                variant = str(datum["metric_variant"])
                scope = str(datum["scope"])
                if scope not in ("paired_main", "paired_cf729") or variant not in ("s", "s1"):
                    continue
                y = (0 if scope == "paired_main" else 1) + (-.15 if variant == "s" else .15)
                value = _num(datum["value"]); interval = _interval(datum, "ci_low", "ci_high")
                color = "#333333" if scope == "paired_main" else "#aaaaaa"
                bars = axis.errorbar(value, y, xerr=[[value-interval[0]], [interval[1]-value]] if interval else None,
                                     fmt="o" if variant == "s" else "s", color=color, mfc=color if variant == "s" else "white", ms=4, capsize=2)
                if variant == "s1":
                    for bar in bars.lines[2]: bar.set_linestyle("--")
                _record(ctx, datum, panel="(a)", series=f"{family} · {fmt} · {quantity} · {scope} · {variant}", x_label=quantity, x=value, y=y,
                        source_file="E9.csv", keys=("model", "format", "scope", "condition", "metric_variant", "quantity"),
                        value_col="value", value_axis="x", ci_low_col="ci_low" if interval else "", ci_high_col="ci_high" if interval else "", ci=interval)
            axis.axvline(0, color="#888888", ls=":", lw=.7)
            axis.set_yticks((0,1), ("main", "cf729")); axis.set_ylim(1.5,-.5)
            if j == 0: axis.set_ylabel(f"{MODEL_LABELS[family]} · {fmt}")
            if i == 0: axis.set_title(f"{'(a) ' if j == 0 else ''}{quantity}")
    for j, family in enumerate(MODEL_ORDER):
        axis = figure.add_subplot(grid[10, j*4:(j+1)*4])
        for i, fmt in enumerate(("repetition", "translation", "cloze")):
            for scope, dodge, color in (("paired_main", -.15, "#333333"),("paired_cf729", .15, "#aaaaaa")):
                part = frame.loc[frame["model"].map(_fam).eq(family) & frame["format"].eq(fmt) & frame["scope"].eq(scope)
                                 & frame["quantity"].eq("A10_vi_rate") & frame["metric_variant"].eq("s")]
                points = []
                for condition in ("diac", "nodiac"):
                    datum = part.loc[part["condition"].eq(condition)].iloc[0]
                    rate = _num(datum["a10_vi_n"]) / _num(datum["a10_denominator"]); points.append(rate)
                    y = i+dodge
                    axis.plot(rate, y, marker="o" if condition == "diac" else "s", color=color, mfc=color if condition == "diac" else "white", ms=5, clip_on=False)
                    _record(ctx, datum, panel="(b)", series=f"{family} · {fmt} · {scope} · {condition}", x_label="A10 VI rate", x=rate, y=y,
                            source_file="E9.csv", keys=("model", "format", "scope", "condition", "metric_variant", "quantity"),
                            value_col="a10_vi_n", value_axis="x", value_transform="a10_vi_n/a10_denominator", value_aux_col="a10_denominator", plotted_value=rate)
                axis.plot(points, [i+dodge]*2, color=color, lw=1)
        axis.set_yticks(range(3), ("repetition", "translation", "cloze")); axis.set_xlim(0,1)
        axis.set_title(f"{'(b) ' if j == 0 else ''}{MODEL_LABELS[family]}"); axis.set_xlabel("A10 VI rate")
    figure.legend(handles=[plt.Line2D([], [], marker="o", color="#333333", ls="none", label="s"),
                           plt.Line2D([], [], marker="s", color="#333333", mfc="white", ls="none", label="s¹"),
                           plt.Line2D([], [], color="#333333", label="paired main"), plt.Line2D([], [], color="#aaaaaa", label="paired cf729")],
                  title="(a)", loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=4)
    figure.legend(handles=[plt.Line2D([], [], marker="o", color="#333333", ls="none", label="diac"),
                           plt.Line2D([], [], marker="s", color="#333333", mfc="white", ls="none", label="nodiac"),
                           plt.Line2D([], [], color="#333333", label="main"), plt.Line2D([], [], color="#aaaaaa", label="cf729")],
                  title="(b)", loc="lower center", ncol=4)
    figure.subplots_adjust(left=.12, right=.98, top=.96, bottom=.06)
    ctx.save("supp03_e9_paired", figure)


def _head_labels(axis: Any, rows: list[tuple[Any, int, int]]) -> None:
    """Anchor head labels to their rows, moving text only to prevent overlap."""
    from matplotlib.transforms import blended_transform_factory

    ordered = sorted(rows, key=lambda row: (row[2], row[1]))
    if not ordered:
        return
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    # Pack text in display coordinates after the final axes layout. Merging
    # colliding groups shares only the displacement needed for readable labels.
    gap = 9 * axis.figure.dpi / 72
    groups: list[list[float]] = []
    for index, (_, _, anchor_y) in enumerate(ordered):
        desired = axis.transData.transform((0, anchor_y))[1] - index * gap
        groups.append([desired, 1])
        while len(groups) > 1 and groups[-2][0] / groups[-2][1] > groups[-1][0] / groups[-1][1]:
            right = groups.pop()
            groups[-1][0] += right[0]
            groups[-1][1] += right[1]
    packed = [total / count for total, count in groups for _ in range(int(count))]
    for index, (datum, _, anchor_y) in enumerate(ordered):
        label_y = axis.transData.inverted().transform((0, packed[index] + index * gap))[1]
        axis.plot([1, 1.025, 1.045], [anchor_y, anchor_y, label_y],
                  transform=transform, color="#555555", lw=.6, clip_on=False,
                  scalex=False, scaley=False)
        axis.annotate(f"{int(datum['layer'])}.{int(datum['head'])}",
                      xy=(1, anchor_y), xytext=(1.055, label_y),
                      xycoords=transform, textcoords=transform,
                      ha="left", va="center", fontsize=7, annotation_clip=False)


def supp4_e7_heads(ctx: FigureContext) -> None:
    """Plot one E7 mean-projection heatmap per cell with source-CI annotations."""
    frame = ctx.csv("E7_heads_all.csv")
    ranked = ctx.csv("E7_heads.csv")
    ranked = ranked.loc[ranked["item_set"].eq("cf729") & ranked["direction"].eq("d_vi_en")]
    _require(frame, "E7_heads_all.csv", ("model", "format", "item_set", "direction", "layer", "head", "mean_projection", "primary_band"))
    frame = frame.loc[frame["item_set"].eq("cf729") & frame["direction"].eq("d_vi_en")].copy()
    if frame.empty:
        raise FigureSkip("E7_heads_all.csv has no cf729 d_vi_en mean_projection values")
    frame["family"] = frame["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(18, 8.5), squeeze=False)
    _facet_matrix(
        figure, axes, row_labels=("Repetition", "Translation"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    cell_pairs = [(family, fmt) for fmt in ("repetition", "translation") for family in MODEL_ORDER]
    head_labels = []
    for cell_idx, (family, fmt) in enumerate(cell_pairs):
        row_idx = 0 if fmt == "repetition" else 1
        col_idx = MODEL_ORDER.index(family)
        axis = axes[row_idx, col_idx]
        cell = frame.loc[frame["family"].eq(family) & frame["format"].eq(fmt)].copy()
        if cell.empty:
            axis.set_axis_off()
            continue
        cell = cell.sort_values(["layer", "head"], kind="stable")
        layer_values = sorted(cell["layer"].dropna().astype(float).unique())
        head_values = sorted(cell["head"].dropna().astype(float).unique())
        matrix = cell.pivot(index="layer", columns="head", values="mean_projection").reindex(index=layer_values, columns=head_values)
        raw = cell["mean_projection"].to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(raw))) if len(raw) else 1.0
        if not math.isfinite(limit) or limit == 0:
            limit = 1.0
        image = axis.imshow(
            matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit, interpolation="nearest", origin="lower",
        )
        layer_index = {float(value): idx for idx, value in enumerate(layer_values)}
        head_index = {float(value): idx for idx, value in enumerate(head_values)}
        selected = ranked.loc[ranked["model"].map(_fam).eq(family) & ranked["format"].eq(fmt)]
        label_rows = []
        for _, datum in selected.iterrows():
            positive = _finite(datum.get("top_positive_rank")) and 1 <= float(datum["top_positive_rank"]) <= 10
            negative = _finite(datum.get("top_negative_rank")) and 1 <= float(datum["top_negative_rank"]) <= 10
            if not (positive or negative):
                continue
            xcoord = head_index[float(datum["head"])]
            ycoord = layer_index[float(datum["layer"])]
            axis.add_patch(plt.Rectangle((xcoord-.5, ycoord-.5), 1, 1, fill=False, edgecolor="black", lw=1))
            for column, present in (("top_positive_rank", positive), ("top_negative_rank", negative)):
                if not present: continue
                _record(ctx, datum, panel=f"({chr(97+cell_idx)})", series=f"{family} · {fmt} · {column} outline", x_label="Head / layer index", x=xcoord, y=ycoord,
                        source_file="E7_heads.csv", keys=("model", "format", "item_set", "direction", "layer", "head", "layer_type"),
                        value_col=column, value_axis="annotation", annotation_transform="numeric")
                if float(datum[column]) <= 3:
                    label_rows.append((datum, xcoord, ycoord))
        head_labels.append((axis, label_rows))
        for _, datum in cell.iterrows():
            xcoord = head_index[float(datum["head"])]
            ycoord = layer_index[float(datum["layer"])]
            interval = _interval(datum, "ci_low", "ci_high")
            _record(
                ctx, datum, panel=f"({chr(97 + cell_idx)})",
                series=f"{family} · {fmt} · layer {int(datum['layer'])} head {int(datum['head'])}",
                x_label="Head index / layer index", x=xcoord, y=ycoord,
                source_file="E7_heads_all.csv",
                keys=("model", "format", "item_set", "direction", "layer", "head"),
                value_col="mean_projection", value_axis="color",
                ci_low_col="ci_low" if interval else "",
                ci_high_col="ci_high" if interval else "", ci=interval,
                annotation=f"{int(datum['layer'])}.{int(datum['head'])}",
                annotation_transform="layer.head",
            )
        tick_indices = [idx for idx, layer in enumerate(layer_values) if int(layer) % 2 == 0]
        tick_labels = [str(int(layer_values[idx])) for idx in tick_indices]
        if family == "gemma":
            for global_layer in (6, 12, 18, 24, 30):
                if global_layer in layer_index and layer_index[global_layer] not in tick_indices:
                    tick_indices.append(layer_index[global_layer])
                    tick_labels.append(str(global_layer))
            ordered = sorted(zip(tick_indices, tick_labels))
            tick_indices, tick_labels = [item[0] for item in ordered], [item[1] for item in ordered]
        axis.set_yticks(tick_indices, tick_labels, fontsize=5)
        axis.set_xticks(range(len(head_values)), [str(int(head)) for head in head_values], rotation=90, fontsize=4.5)
        axis.set_xlabel("Head")
        axis.set_ylabel("Layer")
        if family == "gemma":
            for tick, label in zip(axis.get_yticklabels(), tick_labels):
                if int(label) in (6,12,18,24,30):
                    tick.set_text(label + " G")
                    tick.set_fontweight("bold")
            axis.set_yticklabels([label + " G" if int(label) in (6,12,18,24,30) else label for label in tick_labels])
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.23)
    figure.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.12, hspace=0.4, wspace=0.6)
    for axis, rows in head_labels:
        _head_labels(axis, rows)
    ctx.save("supp04_e7_heads", figure)


def supp5_r10_positions(ctx: FigureContext) -> None:
    """Plot R10 along-path and answer-position values and one count label per t."""
    frame = ctx.csv("R10.csv")
    _require(frame, "R10.csv", ("model", "format", "target_lang", "t", "concepts_at_t", "along_path_value", "along_path_layer", "answer_position_value", "answer_position_layer"))
    frame = frame.loc[frame["t"].isin((1, 2, 3))].copy()
    if frame.empty:
        raise FigureSkip("R10.csv has no value rows for t=1, 2, 3")
    frame["family"] = frame["model"].map(_fam)
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    _facet_matrix(
        figure, axes, row_labels=("Repetition", "Translation"),
        col_labels=tuple(MODEL_LABELS[name] for name in MODEL_ORDER),
    )
    for row_idx, fmt in enumerate(("repetition", "translation")):
        for col_idx, family in enumerate(MODEL_ORDER):
            axis = axes[row_idx, col_idx]
            cell = frame.loc[frame["family"].eq(family) & frame["format"].eq(fmt)]
            for lang in LANG_ORDER:
                part = cell.loc[cell["target_lang"].eq(lang)].sort_values("t", kind="stable")
                if part.empty:
                    continue
                xx = part["t"].to_numpy(dtype=float)
                color = LANG_COLORS[lang]
                for value_col, low_col, high_col, style, label in (
                    ("along_path_value", "along_path_ci_low", "along_path_ci_high", "-", f"{lang} along path"),
                    ("answer_position_value", "answer_position_ci_low", "answer_position_ci_high", "--", f"{lang} answer position"),
                ):
                    yy = part[value_col].to_numpy(dtype=float)
                    have_ci = {low_col, high_col}.issubset(part.columns) and part[low_col].notna().all() and part[high_col].notna().all()
                    lo = part[low_col].to_numpy(dtype=float) if have_ci else None
                    hi = part[high_col].to_numpy(dtype=float) if have_ci else None
                    _line_with_ci(axis, xx, yy, lo, hi, color=color, linestyle=style, label=label, marker="o", alpha=0.10)
                    for (_, data_row), xval, yval in zip(part.iterrows(), xx, yy):
                        interval = _interval(data_row, low_col, high_col)
                        _record(
                            ctx, data_row, panel="(a)", series=f"{label} · {family} · {fmt}",
                            x_label="Vietnamese content position t", x=xval, y=yval,
                            source_file="R10.csv",
                            keys=("model", "format", "target_lang", "target_variant", "t", "quantity"),
                            value_col=value_col, x_source_col="t",
                            ci_low_col=low_col if interval else "",
                            ci_high_col=high_col if interval else "", ci=interval,
                        )
            for t in (1, 2, 3):
                at_t = cell.loc[cell["t"].eq(t)]
                if at_t.empty:
                    continue
                datum = at_t.iloc[0]
                count = _num(datum["concepts_at_t"])
                axis.annotate(
                    str(int(count)), (float(t), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -18), textcoords="offset points", ha="center", va="top", fontsize=5.2,
                )
                _record(
                    ctx, datum, panel="(a)", series=f"concepts_at_t · {family} · {fmt} · t={t}",
                    x_label="Vietnamese content position t", x=float(t), y=0.0,
                    source_file="R10.csv",
                    keys=("model", "format", "target_lang", "target_variant", "t", "quantity"),
                    value_col="concepts_at_t", value_axis="annotation",
                    annotation=str(int(count)),
                )
            axis.set_xticks((1, 2, 3), ("1", "2", "3"))
            axis.tick_params(labelbottom=True, labelleft=True)
            if col_idx == 0:
                axis.set_ylabel("Maximum Δ (95% CI where present)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01), fontsize=5.8)
    figure.supxlabel("Vietnamese content position t; numbers below ticks are concepts_at_t", y=0.06, fontsize=6.5)
    figure.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.2, hspace=0.48, wspace=0.22)
    ctx.save("supp05_r10_positions", figure)


def supp6_r_battery(ctx: FigureContext) -> None:
    """Plot categorical score outcomes against their primary counterparts."""
    frame = ctx.csv("R_battery_e4.csv")
    _require(frame, "R_battery_e4.csv", ("variant", "model", "format", "Outcome_s", "Outcome_s1", "primary_Outcome", "primary_s1_outcome"))
    frame["family"] = frame["model"].map(_fam)
    variants = sorted(frame["variant"].dropna().astype(str).unique())
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.34 * len(variants))))
    columns = (
        ("Outcome_s", "primary_Outcome", "(a) primary score s"),
        ("Outcome_s1", "primary_s1_outcome", "(b) first-token score s¹"),
    )
    cell_order = [(family, fmt) for family in MODEL_ORDER for fmt in ("repetition", "translation")]

    for axis, (observed, primary, heading) in zip(axes, columns):
        for col_idx, (family, fmt) in enumerate(cell_order):
            cell = frame.loc[frame["family"].eq(family) & frame["format"].eq(fmt)]
            for row_idx, variant in enumerate(variants):
                match = cell.loc[cell["variant"].eq(variant)]
                if match.empty:
                    continue
                datum = match.iloc[0]
                y = float(len(variants) - row_idx - 1)
                x = float(col_idx)
                observed_text = str(datum[observed])
                primary_text = str(datum[primary])
                differs = observed_text != primary_text
                text_y = y + 0.08
                axis.text(x, text_y, observed_text, ha="center", va="center", fontsize=5.2)
                axis.text(x, y - 0.12, f"primary {primary_text}", ha="center", va="center", fontsize=4.8, color="#555555")
                edge = "#E69F00" if differs else "#009E73"
                axis.add_patch(plt.Rectangle((x - 0.47, y - 0.43), 0.94, 0.86, fill=False, edgecolor=edge, linewidth=0.8))
                key_cols = ("variant", "model", "format")
                _record(
                    ctx, datum, panel=heading, series=f"variant outcome · {family} · {fmt} · {variant}",
                    x_label="Model × format", x=x, y=text_y,
                    source_file="R_battery_e4.csv", keys=key_cols, value_col=observed,
                    value_axis="annotation", annotation=observed_text,
                )
                _record(
                    ctx, datum, panel=heading, series=f"primary outcome · {family} · {fmt} · {variant}",
                    x_label="Model × format", x=x, y=y - 0.12,
                    source_file="R_battery_e4.csv", keys=key_cols, value_col=primary,
                    value_axis="annotation", annotation=primary_text,
                )
        axis.set_xlim(-0.5, len(cell_order) - 0.5)
        axis.set_ylim(-0.5, len(variants) - 0.5)
        axis.set_xticks(range(len(cell_order)), [f"{MODEL_LABELS[family]}\n{fmt}" for family, fmt in cell_order], fontsize=6)
        axis.set_yticks(range(len(variants)), list(reversed(variants)), fontsize=5.5)
        axis.set_title(heading)
        axis.set_xlabel("Model × format")
        axis.set_ylabel("Variant")
    handles = [
        plt.Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor="#009E73", color="#009E73", linestyle="none", label="Matches primary"),
        plt.Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor="#E69F00", color="#E69F00", linestyle="none", label="orange outline = differs from primary"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), fontsize=6.5)
    figure.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.13, wspace=0.30)
    ctx.save("supp06_r_battery_outcomes", figure)


FIGURE_FUNCTIONS: tuple[Callable[[FigureContext], None], ...] = (
    fig2_m1_m2,
    fig3_rise_landmarks,
    fig4_la_gate,
    fig5_e6_forest,
    fig6_e8_rates,
    fig7_e7_projection,
    supp1_r1_forest,
    supp2_r2_validation,
    supp3_e9_paired,
    supp4_e7_heads,
    supp5_r10_positions,
    supp6_r_battery,
    supp7_c1_full,
)


def _latest_source_commit(repo: Path, filename: str) -> str:
    """Return the commit that last changed a committed report CSV."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "HEAD", "--", f"reports/{filename}"],
        cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip() or "unknown"


def _read_csv_bytes(repo: Path, filename: str, *, committed_only: bool) -> bytes:
    """Read an allowed source CSV from HEAD or from a synthetic report directory."""
    if filename not in ALLOWED_CSVS:
        raise ValueError(f"CSV is not allow-listed: {filename}")
    if committed_only:
        result = subprocess.run(
            ["git", "show", f"HEAD:reports/{filename}"], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout
    path = (repo / "reports" / filename).resolve()
    if "results" in path.parts:
        raise ValueError("Figure verification may not read results/")
    return path.read_bytes()


def _numeric_transform(
    value: str,
    transform: str,
    *,
    aux_value: str = "",
    denominator: str = "",
) -> float:
    """Apply only a named, allowed plotting transformation to a source scalar."""
    number = float(value)
    if transform == "" or transform == "identity":
        return number
    if transform in {"damage_count/n", "a10_vi_n/a10_denominator"}:
        return number / float(aux_value)
    if transform == "layer/N":
        return number / float(denominator)
    if transform == "sign":
        return 1.0 if number > 0 else (-1.0 if number < 0 else 0.0)
    raise ValueError(f"Unrecognized plotted-data transform: {transform}")


def verify_sidecars(repo: Path, *, committed_only: bool = True) -> list[dict[str, object]]:
    """Verify each sidecar point and interval against its committed source row."""
    report_rows: list[dict[str, object]] = []
    source_cache: dict[str, pd.DataFrame] = {}
    selection_cache: dict[tuple[object, ...], pd.DataFrame] = {}
    for spec in FIGURE_SPECS:
        path = repo / "reports" / "figures" / "data" / f"{spec.stem}.csv"
        if not path.is_file():
            report_rows.append({"figure": spec.key, "rows": 0, "max_diff": math.inf, "fails": 1})
            continue
        sidecar = pd.read_csv(path, keep_default_na=False, dtype=str)
        fail_rows = 0
        max_diff = 0.0
        failed_samples: list[dict[str, object]] = []
        for _, plotted in sidecar.iterrows():
            row_failed = False
            row_reasons: list[str] = []

            def fail(reason: str) -> None:
                nonlocal row_failed
                row_failed = True
                row_reasons.append(reason)

            source_file = plotted["source_file"]
            if source_file not in source_cache:
                source_bytes = _read_csv_bytes(repo, source_file, committed_only=committed_only)
                source_cache[source_file] = pd.read_csv(io.BytesIO(source_bytes), keep_default_na=False, dtype=str)
            source = source_cache[source_file]
            key_cols = json.loads(plotted["source_key_columns"])
            selected = source
            prefix: tuple[object, ...] = (source_file,)
            for key in key_cols:
                source_key = plotted.get(f"key_{key}", "")
                prefix += (key, source_key)
                if prefix not in selection_cache:
                    selection_cache[prefix] = selected.loc[
                        selected[key].map(lambda source_key_value: _key_equal(source_key_value, source_key))
                    ]
                selected = selection_cache[prefix]
            if len(selected) != 1:
                fail(f"source row matches={len(selected)}")
                fail_rows += int(row_failed)
                failed_samples.append({"series": plotted["series"], "reasons": row_reasons})
                continue
            source_row = selected.iloc[0]
            value_col = plotted["source_value_column"]
            if value_col not in source_row.index:
                fail(f"missing source value column {value_col}")
                fail_rows += int(row_failed)
                failed_samples.append({"series": plotted["series"], "reasons": row_reasons})
                continue
            raw_value = source_row[value_col]
            if raw_value != plotted["source_value"]:
                if _numeric_equal(raw_value, plotted["source_value"]):
                    pass
                else:
                    fail("stored source value differs from CSV")
                    fail_rows += 1
                    failed_samples.append({"series": plotted["series"], "reasons": row_reasons})
                    continue
            transform = plotted["value_transform"] or "identity"
            axis_type = plotted["source_value_axis"]
            if axis_type == "annotation":
                expected_value: object = raw_value
                if plotted["annotation_transform"] == "layer.head":
                    expected_value = f"{int(float(source_row['layer']))}.{int(float(source_row['head']))}"
                annotation_matches = (
                    _numeric_equal(plotted["plotted_value"], str(expected_value))
                    if plotted["annotation_transform"] == "numeric"
                    else plotted["plotted_value"] == str(expected_value)
                )
                if not annotation_matches:
                    fail("annotation plotted value differs from source")
                if plotted["annotation_transform"] == "layer.head":
                    if plotted["annotation_text"] != str(expected_value):
                        fail("annotation text differs from source")
            else:
                aux = source_row[plotted["source_value_aux_column"]] if plotted["source_value_aux_column"] else ""
                expected_value = _numeric_transform(
                    raw_value, transform, aux_value=aux, denominator=plotted["x_denominator"]
                )
                observed_value = float(plotted["plotted_value"])
                max_diff = max(max_diff, abs(observed_value - expected_value))
                if not _float_close(observed_value, expected_value):
                    fail("plotted value differs from source or stated transform")
                if axis_type in {"x", "y"}:
                    coordinate = float(plotted[axis_type])
                    max_diff = max(max_diff, abs(coordinate - expected_value))
                    if not _float_close(coordinate, expected_value):
                        fail(f"{axis_type} coordinate differs from plotted value")
            if plotted["annotation_transform"] == "layer.head":
                expected_annotation = f"{int(float(source_row['layer']))}.{int(float(source_row['head']))}"
                if plotted["annotation_text"] != expected_annotation:
                    fail("layer.head annotation differs from source")
            for bound, colname in (
                ("ci_low", plotted["source_ci_low_column"]),
                ("ci_high", plotted["source_ci_high_column"]),
            ):
                observed = plotted[bound]
                if not colname:
                    if observed:
                        fail(f"unexpected {bound} without source interval")
                    continue
                source_bound = source_row[colname]
                if not source_bound:
                    if observed:
                        fail(f"unexpected {bound} for empty source interval")
                    continue
                if observed:
                    if plotted["ci_transform"] == "layer/N":
                        expected_bound = float(source_bound) / float(plotted["ci_denominator"])
                    else:
                        expected_bound = float(source_bound)
                    diff = abs(float(observed) - expected_bound)
                    max_diff = max(max_diff, diff)
                    if not _float_close(float(observed), expected_bound):
                        fail(f"{bound} differs from source interval")
                else:
                    fail(f"missing plotted {bound} for source interval")
            if plotted["x_transform"] == "layer/N":
                if not plotted["source_x_column"]:
                    fail("normalized x has no source layer column")
                else:
                    expected_x = float(source_row[plotted["source_x_column"]]) / float(plotted["x_denominator"])
                    actual_x = float(plotted["x"])
                    diff = abs(actual_x - expected_x)
                    max_diff = max(max_diff, diff)
                    if not _float_close(actual_x, expected_x):
                        fail("normalized x differs from source layer/N")
            fail_rows += int(row_failed)
            if row_failed:
                failed_samples.append({"series": plotted["series"], "reasons": row_reasons})
        report_rows.append(
            {
                "figure": spec.key,
                "rows": int(len(sidecar)),
                "max_diff": 0.0 if max_diff <= 1e-12 else max_diff,
                "fails": fail_rows,
                "failed_samples": failed_samples[:5],
            }
        )
    return report_rows


def _numeric_equal(left: str, right: str) -> bool:
    """Compare numeric-looking source strings without altering categorical values."""
    try:
        return _float_close(float(left), float(right))
    except (TypeError, ValueError):
        return left == right


def _key_equal(left: object, right: str) -> bool:
    """Match source keys exactly, with numeric equivalence for serialized numbers."""
    left_text = _cell_text(left)
    return left_text == right or _numeric_equal(left_text, right)


def _float_close(left: float, right: float) -> bool:
    """Use a tight serialization tolerance for copied and transformed floats."""
    return abs(left - right) <= 1e-12


def _empty_sidecar(context: FigureContext, stem: str) -> None:
    """Write an empty plotted-data sidecar for a value-column skip."""
    context.plotted_rows.clear()
    context.write_sidecar(stem)


def write_readme(repo: Path, output_dir: Path, results: Mapping[str, str]) -> None:
    """Write per-figure files, source commits, filters, transforms, and captions."""
    lines = [
        "# Paper figures", "",
        "Fig 1 is removed. Table A replaces it: [tokenization table](../tables/tableA_tokenization.md), with copied source values and an explicit absent-field note.", "",
        "Figures use committed aggregate report CSVs only. No figure builder reads results/ or includes item text.",
        "PNG files use 300 dpi; a PNG above 1 MB is regenerated at 200 dpi.",
        "Each figure has a plotted-data sidecar in data/; --verify checks every row against its committed source CSV.", "",
    ]
    for spec in FIGURE_SPECS:
        lines.extend((f"## {spec.key} — {spec.title}", ""))
        result = results.get(spec.key, "not generated")
        if result.startswith("SKIPPED:"):
            files = f"SKIPPED; data/{spec.stem}.csv (header only)"
        else:
            files = result
        lines.append(f"- Files: {files}")
        for name in (f"{spec.stem}.pdf", f"{spec.stem}.png", f"data/{spec.stem}.csv"):
            path = output_dir / name
            if path.is_file():
                lines.append(f"  - {name}: {path.stat().st_size} bytes")
        lines.append(f"- Filters: {spec.filters}")
        lines.append(f"- Source columns: {spec.columns}")
        lines.append(f"- Plotting transformations: {spec.transforms}")
        lines.append("- Sources:")
        for filename in spec.sources:
            lines.append(f"  - reports/{filename} — commit {_latest_source_commit(repo, filename)}")
        if spec.notes:
            lines.append(f"- Note: {spec.notes}")
        lines.append(f"- Caption draft: {spec.caption}")
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_table_a(repo: Path) -> None:
    """Copy tokenization fields verbatim, joining only exact family/variant keys."""
    target_name = "X0c_a_target_tokenization.csv"
    first_name = "X0c_a_m1_first_token.csv"
    targets = list(csv.DictReader(io.StringIO(_read_csv_bytes(repo, target_name, committed_only=True).decode())))
    first = list(csv.DictReader(io.StringIO(_read_csv_bytes(repo, first_name, committed_only=True).decode())))
    lookup = {(r["label"], r["family"], r["readout"]): r for r in first}
    columns = ("label", "family", "target_variant", "n_items", "mean_content_tokens", "continuation_share", "usable_count", "expected")
    rows = []
    for row in targets:
        readout = "zh_released" if row["variant"] == "zh" else row["variant"]
        matched = lookup.get((row["label"], row["family"], readout), {})
        rows.append({"label":row["label"], "family":row["family"], "target_variant":row["variant"],
                     **{c: row.get(c, "") for c in ("n_items", "mean_content_tokens", "continuation_share")},
                     **{c: matched.get(c, "") for c in ("usable_count", "expected")}})
    output = repo / "reports" / "tables"
    output.mkdir(exist_ok=True)
    with (output / "tableA_tokenization.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Table A — tokenization", "", "All values are copied as source strings; no statistic or interval is computed.", "",
             f"Source reports/{target_name}, commit {_latest_source_commit(repo,target_name)}: label, family, variant → target_variant, n_items, mean_content_tokens, continuation_share.",
             f"Source reports/{first_name}, commit {_latest_source_commit(repo,first_name)}: label, family, readout (join key), usable_count, expected.", "",
             "Join: label × family × target variant; target variant zh matches first-token readout zh_released, and other variants match by exact name. Registered and base labels remain separate.", "",
             "Absent fields: neither source has an M1-eligible count column; it is left absent rather than inferred. expected is the source's expected usable first-token count, not an M1-eligible count. First-token rows for vi and Vietnamese nodiac variants are absent; blank cells retain that absence. No single-token coverage column is present.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row[c] for c in columns) + " |" for row in rows)
    (output / "tableA_tokenization.md").write_text("\n".join(lines)+"\n")


def build_all(repo: Path, *, dpi: int = 300, committed_only: bool = True) -> dict[str, str]:
    """Build all figure functions from committed report CSVs and save skip records."""
    _style()
    output_dir = repo / "reports" / "figures"
    context = FigureContext(
        repo=repo, reports_dir=repo / "reports", output_dir=output_dir,
        committed_only=committed_only, dpi=dpi,
    )
    results: dict[str, str] = {}
    for spec, function in zip(FIGURE_SPECS, FIGURE_FUNCTIONS):
        context.plotted_rows.clear()
        try:
            function(context)
        except FigureSkip as exc:
            results[spec.key] = f"SKIPPED: {exc}"
            _empty_sidecar(context, spec.stem)
        else:
            results[spec.key] = f"{spec.stem}.pdf, {spec.stem}.png, data/{spec.stem}.csv"
    write_table_a(repo)
    write_readme(repo, output_dir, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Build figures or verify their plotted-data sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--verify", action="store_true", help="verify existing sidecars against committed CSVs")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.verify:
        rows = verify_sidecars(repo, committed_only=True)
        failures = 0
        for row in rows:
            max_diff = row["max_diff"]
            shown = "inf" if not math.isfinite(float(max_diff)) else f"{float(max_diff):.17g}"
            print(
                f"{row['figure']}: rows checked={row['rows']}; "
                f"max |difference|={shown}; FAIL rows={row['fails']}"
            )
            failures += int(row["fails"])
        return 1 if failures else 0
    results = build_all(repo, dpi=args.dpi, committed_only=True)
    for key, result in results.items():
        print(f"{key}: {result}")
    total = sum(path.stat().st_size for path in (repo / "reports" / "figures").rglob("*") if path.is_file())
    print(f"Total reports/figures size: {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
