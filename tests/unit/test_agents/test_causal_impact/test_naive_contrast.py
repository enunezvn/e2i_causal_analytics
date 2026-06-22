"""Unit coverage for the naive (unadjusted) difference-in-means foil.

``_compute_naive_contrast`` is the diagnostic that powers the "confounding bias
removed" surfacing (Option D): it reports the UNADJUSTED diff-in-means alongside
the adjusted ATE so the analyst sees how much bias adjustment removed. It is
DEFINED ONLY for a binary 0/1 treatment (the ``treatment_arm`` case) — for a
continuous treatment a median-split "difference in means" would be a *different
estimand* than the adjusted slope, so the honest answer is not-applicable
(``None``), never a fabricated number. Being a non-essential foil, it fails
OPEN to ``None`` on any problem rather than breaking the real estimate.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.estimation import _compute_naive_contrast


@pytest.mark.unit
def test_binary_treatment_returns_diff_in_means_with_ci():
    # mean(Y|T=1)=0.70, mean(Y|T=0)=0.40 -> naive diff = 0.30.
    df = pd.DataFrame(
        {
            "treatment_arm": [1] * 100 + [0] * 100,
            "persistent_180d": [1] * 70 + [0] * 30 + [1] * 40 + [0] * 60,
        }
    )
    point, lo, hi = _compute_naive_contrast(df, "treatment_arm", "persistent_180d")
    assert point == pytest.approx(0.30, abs=1e-9)
    assert lo is not None and hi is not None
    assert lo < point < hi  # the CI brackets the point estimate


@pytest.mark.unit
def test_continuous_treatment_is_not_applicable():
    df = pd.DataFrame(
        {
            "engagement_score": np.linspace(0.0, 10.0, 200),
            "persistent_180d": np.random.default_rng(0).random(200),
        }
    )
    assert _compute_naive_contrast(df, "engagement_score", "persistent_180d") == (
        None,
        None,
        None,
    )


@pytest.mark.unit
def test_single_arm_present_is_not_applicable():
    # Only treated rows -> no control contrast -> not applicable.
    df = pd.DataFrame({"treatment_arm": [1] * 50, "persistent_180d": [1] * 25 + [0] * 25})
    assert _compute_naive_contrast(df, "treatment_arm", "persistent_180d") == (None, None, None)


@pytest.mark.unit
def test_three_level_treatment_is_not_applicable():
    df = pd.DataFrame({"arm": [0] * 30 + [1] * 30 + [2] * 30, "y": ([1, 0] * 45)})
    assert _compute_naive_contrast(df, "arm", "y") == (None, None, None)


@pytest.mark.unit
def test_naive_overstates_relative_to_a_stratified_estimate_under_confounding():
    """Faithful miniature of the DGP confounding: arm assigned by a covariate
    propensity, and that covariate also drives the outcome. The naive (pooled)
    diff-in-means must exceed a severity-STRATIFIED diff-in-means — i.e. the
    naive estimate is inflated by exactly the confounding that stratification
    (a crude adjustment) removes. This is the phenomenon Option D surfaces."""
    rng = np.random.default_rng(7)
    n = 8000
    severity = rng.normal(5.0, 1.0, n)
    # Propensity rises with severity -> treated patients are sicker (confounding).
    propensity = 1.0 / (1.0 + np.exp(-(0.9 * (severity - 5.0))))
    arm = (rng.random(n) < propensity).astype(int)
    # Outcome driven by severity (prognostic) + a small TRUE arm effect.
    latent = (severity - 5.0) * 0.45 + 0.05 * arm + rng.normal(0.0, 0.3, n)
    y = (latent > 0.0).astype(int)
    df = pd.DataFrame({"treatment_arm": arm, "persistent_180d": y})

    naive, _, _ = _compute_naive_contrast(df, "treatment_arm", "persistent_180d")
    assert naive is not None

    # Crude deconfounding: weighted within-severity-band diff-in-means.
    bands = pd.qcut(severity, 6, labels=False)
    num, den = 0.0, 0
    for b in np.unique(bands):
        grp = df[bands == b]
        treated = grp.loc[grp["treatment_arm"] == 1, "persistent_180d"]
        control = grp.loc[grp["treatment_arm"] == 0, "persistent_180d"]
        if len(treated) and len(control):
            num += len(grp) * (treated.mean() - control.mean())
            den += len(grp)
    stratified = num / den

    # The naive estimate is materially inflated by the confounding.
    assert naive > stratified
