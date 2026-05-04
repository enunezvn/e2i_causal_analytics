"""Positivity-violation hazard for the ml_patients cohort.

Forces ``treatment_initiated=1`` with high probability for patients in a
specific segment (default ``age_group == ">65"``), producing a near-empty
control region in that segment. This violates the propensity-overlap
assumption that downstream causal estimators rely on.

Detection signal (any one of):
  - SHAP shows the segment column as a near-perfect predictor of treatment
    (or, in the discontinuation context, as an inflated predictor of the
    target through correlated side-effects).
  - Stratified CV ROC-AUC std > 0.07 (high fold variance from the
    near-empty stratum).
  - Sampling-frame audit flags a subgroup distribution mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_positivity_violation(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    segment_col: str = "age_group",
    segment_value: str = ">65",
    p_treat_in_segment: float = 0.95,
    p_treat_outside: float = 0.4,
) -> pd.DataFrame:
    """Plant a near-zero control region in a specific segment.

    Args:
        df: ml_patients DataFrame (input is not mutated).
        seed: RNG seed for reproducibility.
        segment_col: Column whose value defines the targeted segment. The
            default is ``age_group`` (``ml_patients()`` emits literal values
            ``"<50" | "50-65" | ">65"``).
        segment_value: Value within ``segment_col`` to target. Defaults to
            ``">65"`` (the elderly stratum). The string ``"elderly"`` is
            also accepted as an alias and remapped automatically.
        p_treat_in_segment: P(treatment_initiated=1 | segment). Default 0.95
            (near-saturation in the targeted segment).
        p_treat_outside:    P(treatment_initiated=1 | not segment). Default
            0.40 — chosen so the contingency table on
            ``(segment_col, treatment_initiated)`` is asymmetric enough that
            the leakage detector's categorical_class_separation check
            (Cramer's V > 0.5 HIGH threshold) fires on n=1500 cohorts.

    Returns:
        New DataFrame with ``treatment_initiated`` column appended, planted
        with the near-zero control region.
    """
    if segment_col not in df.columns:
        raise ValueError(
            f"segment_col {segment_col!r} not found in DataFrame columns; "
            f"available: {list(df.columns)}"
        )

    # Convenience: callers using the spec wording "elderly" map to the
    # actual ml_patients literal ">65".
    resolved_value = (
        ">65" if (segment_col == "age_group" and segment_value == "elderly") else segment_value
    )

    rng = np.random.default_rng(seed)
    out = df.copy()

    in_segment = out[segment_col].astype(str) == str(resolved_value)
    n = len(out)
    treatment_p = np.where(in_segment, p_treat_in_segment, p_treat_outside)
    out["treatment_initiated"] = rng.binomial(1, treatment_p, size=n).astype(int)
    return out
