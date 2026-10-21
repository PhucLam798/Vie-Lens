"""A6 language-direction fitting, validation, and alignment calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Normalize row vectors, returning NaNs for zero-norm vectors."""
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    output = np.full_like(array, np.nan, dtype=np.float32)
    np.divide(array, norms, out=output, where=norms > 0)
    return output


def fit_layer_directions(
    activations: np.ndarray,
    labels: Sequence[str],
    *,
    class_order: Sequence[str] = ("vi", "en", "zh", "fr", "id"),
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit centered per-language mean directions from one fit layer."""
    x = np.asarray(activations, dtype=np.float32)
    y = np.asarray(labels, dtype=object)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("activations must be rows×hidden and labels must align")
    means: dict[str, np.ndarray] = {}
    for language in class_order:
        selected = x[y == language]
        if selected.shape[0] == 0:
            raise ValueError(f"No fit activations for language {language}")
        means[str(language)] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    center = np.stack(list(means.values()), axis=0).mean(axis=0, dtype=np.float64).astype(np.float32)
    directions = {language: mean - center for language, mean in means.items()}
    return center, directions, means


def cosine_scores(activations: np.ndarray, directions: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute rowwise cosine scores in the mapping's insertion order."""
    x = np.asarray(activations, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("activations must be a two-dimensional matrix")
    direction_values = np.stack([np.asarray(value, dtype=np.float32) for value in directions.values()])
    if direction_values.ndim != 2 or direction_values.shape[1] != x.shape[1]:
        raise ValueError("direction vectors must align with activation width")
    x_unit = _unit_rows(x)
    v_unit = _unit_rows(direction_values)
    return x_unit @ v_unit.T


def predict_languages(
    activations: np.ndarray,
    center: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Classify activations by maximum centered cosine and return scores."""
    centered = np.asarray(activations, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    scores = cosine_scores(centered, directions)
    languages = np.asarray(list(directions), dtype=object)
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    if np.any(np.all(~np.isfinite(scores), axis=1)):
        raise ValueError("At least one activation has no finite direction cosine")
    return languages[np.argmax(safe_scores, axis=1)], scores


def classification_accuracy(
    activations: np.ndarray,
    labels: Sequence[str],
    center: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> tuple[float, int, int]:
    """Return language accuracy, number correct, and sample count."""
    expected = np.asarray(labels, dtype=object)
    predicted, _ = predict_languages(activations, center, directions)
    if expected.shape != predicted.shape:
        raise ValueError("validation labels do not align with activations")
    correct = int(np.count_nonzero(predicted == expected))
    return correct / len(expected), correct, int(len(expected))


def native_validation_cosines(
    activations: np.ndarray,
    labels: Sequence[str],
    center: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Return each language's mean held-out cosine to its own direction."""
    values = np.asarray(activations, dtype=np.float32)
    targets = np.asarray(labels, dtype=object)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("validation activations and labels must align")
    centered = values - np.asarray(center, dtype=np.float32)
    output: dict[str, float] = {}
    for language, direction in directions.items():
        selected = centered[targets == language]
        if selected.shape[0] == 0:
            raise ValueError(f"No validation prompts for language {language}")
        cosine = cosine_scores(selected, {language: direction})[:, 0]
        if not np.isfinite(cosine).all():
            raise ValueError(f"Non-finite native validation cosine for language {language}")
        output[language] = float(np.mean(cosine, dtype=np.float64))
    return output


def random_cosine_baseline(
    activations: np.ndarray,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    dimensions: int | None = None,
    n_directions: int = 1_000,
) -> float:
    """Average cosine to seeded standard-Gaussian random unit directions."""
    x = np.asarray(activations, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("activations must be a non-empty rows×hidden matrix")
    if dimensions is not None and int(dimensions) != x.shape[1]:
        raise ValueError("requested random-vector width does not match activations")
    if n_directions <= 0:
        raise ValueError("n_directions must be positive")
    if (seed is None) == (rng is None):
        raise ValueError("supply exactly one of seed or rng")
    generator = np.random.default_rng(int(seed)) if rng is None else rng
    random = generator.standard_normal((int(n_directions), x.shape[1])).astype(np.float32)
    random_unit = _unit_rows(random)
    cosine = _unit_rows(x) @ random_unit.T
    return float(np.nanmean(cosine, dtype=np.float64))


def random_unit_directions(
    dimensions: int,
    *,
    rng: np.random.Generator,
    n_directions: int = 1_000,
) -> np.ndarray:
    """Draw seeded unit Gaussian directions for chunked query evaluation."""
    if int(dimensions) <= 0 or int(n_directions) <= 0:
        raise ValueError("dimensions and n_directions must be positive")
    values = rng.standard_normal((int(n_directions), int(dimensions))).astype(np.float32)
    return _unit_rows(values)


def random_cosine_values(activations: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return rowwise mean cosine to an already-drawn random-direction set."""
    values = _unit_rows(np.asarray(activations, dtype=np.float32))
    units = _unit_rows(np.asarray(directions, dtype=np.float32))
    if values.ndim != 2 or units.ndim != 2 or values.shape[1] != units.shape[1]:
        raise ValueError("activations and random directions must have matching widths")
    return np.mean(values @ units.T, axis=1, dtype=np.float64)


def normalized_alignment(
    query_activations: np.ndarray,
    center: np.ndarray,
    directions: Mapping[str, np.ndarray],
    *,
    native_cosines: Mapping[str, float],
    c_random: float,
) -> dict[str, np.ndarray]:
    """Normalize query cosine curves to random and native held-out baselines."""
    centered = np.asarray(query_activations, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    output: dict[str, np.ndarray] = {}
    for language, direction in directions.items():
        raw = cosine_scores(centered, {language: direction})[:, 0].astype(np.float64)
        denominator = float(native_cosines[language]) - float(c_random)
        if denominator <= 0.0 or not np.isfinite(denominator):
            output[language] = np.full(raw.shape, np.nan, dtype=np.float64)
        else:
            output[language] = (raw - float(c_random)) / denominator
    return output


def alignment_layer_status(
    gate_pass: bool,
    native_cosines: Mapping[str, float],
    c_random: float,
) -> tuple[bool, bool]:
    """Return (usable, undefined) under the gate and positive-denominator rule."""
    undefined = any(
        not np.isfinite(float(value) - float(c_random))
        or float(value) - float(c_random) <= 0.0
        for value in native_cosines.values()
    )
    return bool(gate_pass) and not undefined, undefined


def la_layer_selections(
    gate_pass: Sequence[bool], undefined: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Return passing-layer and defined-layer masks for primary/sensitivity summaries."""
    passed = np.asarray(gate_pass, dtype=bool)
    is_undefined = np.asarray(undefined, dtype=bool)
    if passed.ndim != 1 or is_undefined.shape != passed.shape:
        raise ValueError("gate and undefined flags must be aligned one-dimensional vectors")
    sensitivity = ~is_undefined
    return passed & sensitivity, sensitivity


def la_rule_sensitivity_changed(
    passing_maximum: float,
    all_layer_maximum: float,
    language: str,
) -> bool | None:
    """Compare the threshold outcome with and without failed gate layers."""
    passed = float(passing_maximum)
    all_layers = float(all_layer_maximum)
    if not np.isfinite(passed) or not np.isfinite(all_layers):
        return None
    passing_rule = passed >= 0.50 if language == "en" else passed < 0.20
    all_rule = all_layers >= 0.50 if language == "en" else all_layers < 0.20
    return bool(passing_rule != all_rule)


def d_en_curve(alignment: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the pointwise A6 contrast LA_en − (LA_fr + LA_id)/2."""
    required = {"en", "fr", "id"}
    missing = required - set(alignment)
    if missing:
        raise ValueError(f"D_en requires alignment curves for: {sorted(missing)}")
    english = np.asarray(alignment["en"], dtype=np.float64)
    french = np.asarray(alignment["fr"], dtype=np.float64)
    indonesian = np.asarray(alignment["id"], dtype=np.float64)
    if english.shape != french.shape or english.shape != indonesian.shape:
        raise ValueError("D_en alignment curves must have matching shapes")
    return english - (french + indonesian) / 2.0
