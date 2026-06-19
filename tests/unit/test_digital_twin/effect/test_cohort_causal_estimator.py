"""Recover-known-effect + de-confounding tests for the cohort causal estimator.

The synthetic-gold cohort DGP (scripts/backfill_segment_engagement.py) plants a
region-heterogeneous TRUE causal effect of engagement on conversion, CONFOUNDED by
market_share. A valid estimator must (a) recover the planted per-region CATE and the
population ATE, and (b) be LESS biased than a naive estimator that omits the confounder.
These are the acceptance gates for Direction 2 (design doc 2026-06-19).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import (
    CohortCausalEffect,
    estimate_cohort_effect,
)
from src.digital_twin.effect.errors import EffectDataUnavailable

# Planted truth (mirrors TRUE_CATE_BY_REGION in the backfill DGP).
TRUE_CATE = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}


def _make_confounded_cohort(n_per_region: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Region-heterogeneous causal effect of engagement on conversion, CONFOUNDED
    by market_share (drives BOTH the treatment propensity AND the outcome, beta=0.8),
    mirroring the gold-standard DGP. Treatment is binarized at the engagement median,
    and the planted effect is on that same binarization (so a correct estimator that
    binarizes at the median recovers tau)."""
    rng = np.random.default_rng(seed)
    frames = []
    for region, tau in TRUE_CATE.items():
        n = n_per_region
        market = rng.uniform(0.0, 1.0, n)  # confounder
        logvol = rng.normal(0.0, 1.0, n)
        # engagement (treatment) is confounded by market_share
        eng_logit = 1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n)
        engagement = 10.0 / (1.0 + np.exp(-eng_logit))  # domain 0..10
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "engagement_score": engagement,
                    "market_share": market,
                    "total_rx_count": np.expm1(np.abs(logvol) * 2.0),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    t_bin = (df["engagement_score"] > df["engagement_score"].median()).astype(float)
    df["conversion_rate"] = (
        0.5
        + 0.8 * df["market_share"]  # the strong confounder on the outcome
        + df["_tau"] * t_bin  # the planted causal effect
        + rng.normal(0.0, 0.25, len(df))
    ).clip(lower=0.0)
    return df.drop(columns="_tau")


@pytest.mark.slow
def test_recovers_true_cate_by_region_and_population_ate():
    cohort = _make_confounded_cohort()
    eff = estimate_cohort_effect(cohort, "engagement_score")

    assert isinstance(eff, CohortCausalEffect)
    # Population ATE ~ n-weighted mean of region taus (~0.2525).
    true_ate = float(np.mean(list(TRUE_CATE.values())))
    assert abs(eff.ate - true_ate) < 0.10, f"ate {eff.ate} vs true {true_ate}"
    # Per-region CATE recovered within tolerance + correct ordering.
    for region, tau in TRUE_CATE.items():
        assert abs(eff.cate_by_region[region] - tau) < 0.12, (
            f"{region}: {eff.cate_by_region[region]} vs {tau}"
        )
    assert (
        eff.cate_by_region["northeast"]
        > eff.cate_by_region["south"]
        > eff.cate_by_region["midwest"]
    )
    # Honest CI: contains the ATE and is NOT the fake-tight 0.003-0.009 width.
    assert eff.ate_ci_lower < eff.ate < eff.ate_ci_upper
    assert (eff.ate_ci_upper - eff.ate_ci_lower) > 0.01


@pytest.mark.slow
def test_deconfounding_reduces_bias():
    """Omitting the market_share confounder INFLATES the estimate; adjusting for it
    moves the estimate closer to the planted truth."""
    cohort = _make_confounded_cohort()
    true_ate = float(np.mean(list(TRUE_CATE.values())))

    deconfounded = estimate_cohort_effect(
        cohort, "engagement_score", confounders=("market_share", "total_rx_count")
    )
    naive = estimate_cohort_effect(cohort, "engagement_score", confounders=())

    assert abs(deconfounded.ate - true_ate) < abs(naive.ate - true_ate)
    assert naive.ate > deconfounded.ate  # confounding inflates upward


def test_fail_honest_on_degenerate_treatment():
    """All-constant treatment cannot identify an effect -> raise, never fabricate."""
    cohort = _make_confounded_cohort(n_per_region=200)
    cohort["engagement_score"] = 5.0  # no variation -> no contrast
    with pytest.raises(EffectDataUnavailable):
        estimate_cohort_effect(cohort, "engagement_score")


def test_fail_honest_on_insufficient_rows():
    cohort = _make_confounded_cohort(n_per_region=5)  # 20 rows total
    with pytest.raises(EffectDataUnavailable):
        estimate_cohort_effect(cohort, "engagement_score")
