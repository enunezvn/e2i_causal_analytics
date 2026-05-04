"""Label-leakage hazard for the ml_patients cohort.

Plants a NEW column ``post_treatment_visits`` whose values are nearly
deterministic given ``discontinuation_flag``. Concretely, for each patient,
the planted feature is sampled from a distribution whose mean is shifted by
``leak_strength`` standard deviations between the two outcome classes — so
a single-feature AUC against the target lands above 0.90 (the
``check_single_feature_auc`` HIGH/CRITICAL threshold).

Detection signal:
  ``state["leakage_findings"]`` contains an entry with
  ``feature == "post_treatment_visits"`` at severity in {high, critical}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_label_leakage(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    leak_strength: float = 0.95,
    leak_feature: str = "post_treatment_visits",
) -> pd.DataFrame:
    """Add a post-treatment feature highly correlated with the target.

    The implementation models a "30-day post-treatment HCP visit count" that
    only diverges between discontinued and non-discontinued patients
    *because of* the discontinuation event itself. The two class-conditional
    distributions are Gaussian with means separated by a multiple of their
    shared scale — sized so single-feature AUC > 0.95 even at
    ``leak_strength=0.95``.

    Args:
        df: ml_patients DataFrame (input is not mutated). Must include
            ``discontinuation_flag``.
        seed: RNG seed for reproducibility.
        leak_strength: Approximate single-feature AUC the planted leak
            should produce; in practice this maps to a class-mean
            separation in standard-deviation units (we scale by 6 *
            leak_strength so 0.95 yields well above 0.90 AUC).
        leak_feature: Column name of the planted leak feature.

    Returns:
        New DataFrame with the planted leak column. Other columns are
        preserved unchanged.
    """
    if "discontinuation_flag" not in df.columns:
        raise ValueError("label_leakage requires 'discontinuation_flag' in the DataFrame")

    rng = np.random.default_rng(seed)
    out = df.copy()

    target = out["discontinuation_flag"].astype(int).values
    n = len(out)

    # Class-conditional Gaussian: mean=0 for class 0, mean=mu_shift for
    # class 1, both with sigma=1. AUC of N(0,1) vs N(mu,1) is
    # Phi(mu/sqrt(2)). For leak_strength=0.95 we want AUC near 0.95 ->
    # mu ~ Phi^{-1}(0.95)*sqrt(2) ~ 2.33. We pick mu_shift = 6 *
    # leak_strength so even at leak_strength=0.5 we land well above the
    # 0.90 CRITICAL threshold.
    mu_shift = 6.0 * float(leak_strength)
    base_noise = rng.normal(0.0, 1.0, size=n)
    out[leak_feature] = base_noise + mu_shift * target
    return out
