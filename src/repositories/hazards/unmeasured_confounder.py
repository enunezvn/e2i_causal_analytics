"""Unmeasured-confounder hazard for the ml_patients cohort.

Plants a hidden binary confounder ``U`` that simultaneously raises both
treatment-initiation probability and discontinuation probability. ``U`` is
NOT exposed as a column on the returned DataFrame — that is the point of
the hazard: the modelling agent sees an unexplained variance pattern but
cannot regress on the actual driver.

Detection signal:
  Stratified k-fold CV ROC-AUC std elevated above the clean baseline
  (``cv_roc_auc_std > 0.04`` on n=1500 ml_patients with default kwargs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_unmeasured_confounder(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    p_treat_when_u_high: float = 0.8,
    p_treat_when_u_low: float = 0.2,
    discontinuation_shift: float = 0.15,
) -> pd.DataFrame:
    """Inject a hidden binary confounder U into the cohort.

    Args:
        df: ml_patients DataFrame (input is not mutated).
        seed: RNG seed for reproducibility.
        p_treat_when_u_high: P(treatment_initiated=1 | U=1).
        p_treat_when_u_low:  P(treatment_initiated=1 | U=0).
        discontinuation_shift: Additive bump applied to
            ``discontinuation_flag`` for the U=1 stratum (probability of
            re-flipping a previously-zero outcome). Keep small (~0.15) so
            the propensity model still trains but fold variance jumps.

    Returns:
        New DataFrame with ``treatment_initiated`` column appended and
        ``discontinuation_flag`` partially shifted by U. ``U`` itself is
        intentionally NOT included.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    n = len(out)
    u = rng.binomial(1, 0.5, size=n)

    treatment_p = np.where(u == 1, p_treat_when_u_high, p_treat_when_u_low)
    out["treatment_initiated"] = rng.binomial(1, treatment_p, size=n).astype(int)

    # Shift the outcome partially based on U so the confounder has a real
    # effect on both treatment AND outcome — without obliterating the
    # baseline signal entirely.
    flip_prob = np.where(u == 1, discontinuation_shift, 0.0)
    flips = rng.binomial(1, flip_prob, size=n).astype(bool)
    out.loc[flips, "discontinuation_flag"] = 1

    return out
