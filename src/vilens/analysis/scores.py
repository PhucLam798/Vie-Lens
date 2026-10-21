"""Content-token M2 scores and matched-distractor contrasts."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

SCORE_KEYS = (
    "model",
    "model_revision",
    "pass",
    "format",
    "condition",
    "fewshot_set",
    "concept_id",
    "target_lang",
    "target_variant",
    "role",
    "ref_concept_id",
    "layer",
    "metric",
)
DELTA_KEYS = tuple(key for key in SCORE_KEYS if key not in {"role", "ref_concept_id", "metric"})


def content_token_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Return per-sequence means, first-token scores, continuation means, and sums.

    Whitespace-only tokens remain in the input table but are excluded using
    ``is_content``. ``s_plus`` is the mean over content tokens after the first;
    it is NaN when a sequence has only one content token.
    """
    missing = (set(SCORE_KEYS) | {"tok_idx", "is_content", "value"}) - set(rows.columns)
    if missing:
        raise ValueError(f"Lens table is missing columns: {sorted(missing)}")
    selected = rows.loc[
        rows["metric"].eq("lens_logprob") & rows["is_content"].fillna(False)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*SCORE_KEYS, "s", "s1", "s_plus", "sequence_sum", "n_content"])
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    key_columns = list(SCORE_KEYS)
    selected = selected.sort_values([*key_columns, "tok_idx"], kind="stable")
    grouped = selected.groupby(key_columns, sort=True, dropna=False, observed=True)
    selected["_content_position"] = grouped.cumcount()

    # Keep the sequence statistics in compiled group reductions. The previous
    # per-sequence Python loop repeatedly materialized Arrow-backed string keys
    # from the large P-L table and dominated the analysis profile.
    grouped = selected.groupby(key_columns, sort=True, dropna=False, observed=True)
    summary = grouped["value"].agg(["mean", "first", "sum", "size"])
    continuation = (
        selected.loc[selected["_content_position"].gt(0)]
        .groupby(key_columns, sort=True, dropna=False, observed=True)["value"]
        .mean()
    )
    summary = summary.rename(
        columns={"mean": "s", "first": "s1", "sum": "sequence_sum", "size": "n_content"}
    )
    summary["s_plus"] = continuation.reindex(summary.index)
    summary["n_content"] = summary["n_content"].astype("int64")
    return summary.reset_index()[[*SCORE_KEYS, "s", "s1", "s_plus", "sequence_sum", "n_content"]]


def matched_distractor_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """Join correct and A1 distractor sequences and compute score contrasts.

    The distractor sequence is already teacher-forced after the concept's own
    prompt. Pairing is by source concept, model, format, condition, readout and
    layer; the distractor's ``ref_concept_id`` is retained for audit.
    """
    measures = ("s", "s1", "s_plus", "sequence_sum")
    required = set(DELTA_KEYS) | {"role", "ref_concept_id", *measures}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")
    key_columns = list(DELTA_KEYS)
    selected = scores.loc[scores["role"].astype(str).isin({"correct", "distractor"})]
    role_frames: dict[str, pd.DataFrame] = {}
    for role in ("correct", "distractor"):
        role_frame = selected.loc[selected["role"].astype(str).eq(role)]
        duplicate = role_frame.duplicated(key_columns, keep=False)
        if duplicate.any():
            row = role_frame.loc[duplicate].iloc[0]
            key = tuple(None if pd.isna(row[name]) else row[name] for name in key_columns)
            raise ValueError(f"Duplicate {role} score for key {key}")
        role_frames[role] = role_frame

    correct = role_frames["correct"].loc[:, [*key_columns, *measures]].rename(
        columns={measure: f"{measure}_correct" for measure in measures}
    )
    distractor = role_frames["distractor"].loc[
        :, [*key_columns, "ref_concept_id", *measures]
    ].rename(
        columns={
            "ref_concept_id": "distractor_id",
            **{measure: f"{measure}_distractor" for measure in measures},
        }
    )
    paired = correct.merge(
        distractor,
        on=key_columns,
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame(
            columns=[
                *key_columns, "ref_concept_id", "distractor_id",
                *[name for measure in measures for name in (
                    f"{measure}_correct", f"{measure}_distractor", f"delta_{measure}"
                )],
                "delta",
            ]
        )

    # Preserve the frozen output order: the original implementation sorted the
    # normalized pairing keys by their string representations (not numerically).
    order_columns = []
    for index, key_column in enumerate(key_columns):
        order_column = f"_order_{index}"
        paired[order_column] = paired[key_column].map(
            lambda value: "None" if pd.isna(value) else str(value)
        ).astype(object)
        order_columns.append(order_column)
    paired = paired.sort_values(order_columns, kind="stable").reset_index(drop=True)

    output = paired.loc[:, key_columns].copy()
    output["ref_concept_id"] = paired["distractor_id"]
    output["distractor_id"] = paired["distractor_id"]
    for measure in measures:
        correct_values = pd.to_numeric(paired[f"{measure}_correct"], errors="raise")
        distractor_values = pd.to_numeric(paired[f"{measure}_distractor"], errors="raise")
        output[f"{measure}_correct"] = correct_values
        output[f"{measure}_distractor"] = distractor_values
        valid = correct_values.notna() & distractor_values.notna()
        delta = pd.Series(float("nan"), index=paired.index, dtype="float64")
        delta.loc[valid] = (
            correct_values.loc[valid].astype("float64")
            - distractor_values.loc[valid].astype("float64")
        )
        output[f"delta_{measure}"] = delta
    output["delta"] = output["delta_s"]
    return output


def restrict_concepts(rows: pd.DataFrame, concept_ids: Iterable[str]) -> pd.DataFrame:
    """Filter an analysis table to an explicit, frozen concept-id collection."""
    selected = {str(value) for value in concept_ids}
    return rows.loc[rows["concept_id"].astype(str).isin(selected)].copy()
