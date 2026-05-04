"""Measurement-error hazard for the ml_patients cohort.

Adds Gaussian noise to a single feature at sigma_noise = ``noise_sigma_frac
* df[target_feature].std()``. Default targets ``hcp_visits`` (the highest-
SHAP feature in the canonical ml_patients DGP per
``src/repositories/sample_data.py:654``).

Detection signal:
  When called repeatedly with noise levels {0.1, 0.2, 0.3}, the model_trainer's
  CV ROC-AUC degrades monotonically (auc_clean > auc_10 > auc_20 > auc_30).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_measurement_error(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    target_feature: str = "hcp_visits",
    noise_sigma_frac: float = 0.1,
) -> pd.DataFrame:
    """Add Gaussian noise to a single feature.

    Args:
        df: ml_patients DataFrame (input is not mutated).
        seed: RNG seed for reproducibility.
        target_feature: Column to noise. Must be numeric.
        noise_sigma_frac: Noise std expressed as a fraction of the feature's
            empirical std. Use 0.1 / 0.2 / 0.3 to drive the
            monotonic-degradation test.

    Returns:
        New DataFrame with ``target_feature`` perturbed by N(0, sigma^2)
        where sigma = noise_sigma_frac * std(target_feature).
    """
    if target_feature not in df.columns:
        raise ValueError(
            f"target_feature {target_feature!r} not found in DataFrame columns; "
            f"available: {list(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    out = df.copy()

    series = pd.to_numeric(out[target_feature], errors="coerce")
    if series.isna().all():
        raise ValueError(
            f"target_feature {target_feature!r} has no numeric values; "
            "cannot inject measurement error"
        )

    sigma = float(series.std()) * float(noise_sigma_frac)
    if sigma <= 0:
        # Constant column — fall back to a unit-scale jitter so the noise
        # is detectable while not silently producing identity output.
        sigma = float(noise_sigma_frac)

    noise = rng.normal(0.0, sigma, size=len(out))
    perturbed = series.fillna(series.median()) + noise

    # Preserve the original integer dtype where possible (most ml_patients
    # numeric features are int counts).
    if pd.api.types.is_integer_dtype(out[target_feature]):
        perturbed = perturbed.round().astype(int)

    out[target_feature] = perturbed.values
    return out
