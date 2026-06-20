"""Cohort effect provider (Direction 2).

``CohortEffectDataProvider`` now exposes the brand's synthetic-gold cohort as a RAW
labeled frame for DIRECT causal estimation (paired with ``CohortCausalEstimator``) —
NOT a region-only ATE laundered through a synthetic injected-effect frame. These tests
pin: (1) the retained ``region_standardized_ate`` baseline still behaves, (2) the
provider returns the raw cohort frame (no synthetic handoff, ground_truth_ate=None),
(3) only the IDENTIFIED intervention (digital_engagement) is estimable — call_frequency
is now honestly unavailable, (4) the end-to-end estimate reports cohort provenance with
an honest CI.
"""

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import PROVENANCE_COHORT
from src.digital_twin.effect.provider import (
    CohortEffectDataProvider,
    region_standardized_ate,
)

_REGIONS = ["northeast", "south", "midwest", "west"]
# Region base-rate confounder: regions with higher base ALSO get higher
# engagement below, so the RAW gap overstates the true engagement effect.
_REGION_BASE = {"northeast": 0.60, "south": 0.40, "midwest": 0.40, "west": 0.50}


def _make_cohort(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = rng.choice(_REGIONS, size=n)
    base = np.array([_REGION_BASE[r] for r in regions])
    # Engagement is region-correlated (confounding) and continuous 0..10.
    engagement = np.clip(base * 6 + rng.uniform(0, 5, size=n), 0, 10)
    call_frequency = rng.uniform(0, 14, size=n)
    # Conversion: STRONG engagement effect, WEAK call_frequency effect, + region base.
    conversion = base + 0.06 * engagement + 0.01 * call_frequency + rng.normal(0, 0.08, size=n)
    # Pre-treatment confounder columns the direct estimator adjusts for (present subset).
    market_share = np.clip(base * 0.5 + rng.uniform(0, 0.5, size=n), 0, 1)
    total_rx_count = rng.poisson(lam=np.clip(50 + 80 * base, 1, None)).astype(float)
    return pd.DataFrame(
        {
            "region": regions,
            "engagement_score": engagement,
            "call_frequency": call_frequency,
            "conversion_rate": conversion,
            "market_share": market_share,
            "total_rx_count": total_rx_count,
        }
    )


# --- retained region-only baseline (no longer the production path, kept as a reference) ---


def test_region_standardized_ate_differentiates_treatments():
    cohort = _make_cohort()
    eng = region_standardized_ate(cohort, "engagement_score")
    cf = region_standardized_ate(cohort, "call_frequency")
    assert eng > cf > 0.0
    assert eng > 0.15
    assert -0.6 <= eng <= 0.6 and -0.6 <= cf <= 0.6


def test_region_standardized_ate_removes_region_confounding():
    cohort = _make_cohort()
    work = cohort.dropna()
    t_thr = work["engagement_score"].median()
    y_thr = work["conversion_rate"].median()
    t = (work["engagement_score"] > t_thr).astype(int)
    y = (work["conversion_rate"] > y_thr).astype(int)
    raw = y[t == 1].mean() - y[t == 0].mean()
    standardized = region_standardized_ate(cohort, "engagement_score")
    assert raw > standardized
    assert standardized > 0.0


# --- new direct-estimation provider contract ---


def test_cohort_provider_returns_raw_cohort_frame():
    """The provider returns the RAW cohort (no synthetic injected-effect handoff):
    treatment=engagement_score, outcome=conversion_rate, region as effect modifier,
    the present pre-treatment confounders, and ground_truth_ate=None (estimated, not
    injected)."""
    cohort = _make_cohort()
    provider = CohortEffectDataProvider(cohort, seed=42)
    frame = provider.get_training_frame("digital_engagement", brand="Remibrutinib", twin_type="hcp")
    assert frame.treatment_var == "engagement_score"
    assert frame.outcome_var == "conversion_rate"
    assert frame.ground_truth_ate is None  # estimated from data, NOT injected
    assert frame.confounders == ["market_share", "total_rx_count"]
    assert frame.effect_modifiers == ["region"]
    assert frame.df is cohort  # raw cohort, not a synthetic resample


def test_cohort_provider_non_mappable_intervention_fails_closed():
    provider = CohortEffectDataProvider(_make_cohort())
    with pytest.raises(EffectDataUnavailable):
        provider.get_training_frame("email_campaign", brand="X", twin_type="hcp")


def test_cohort_provider_call_frequency_now_unavailable():
    """call_frequency is an exposure CORRELATE, explicitly NOT in the causal path, so
    call_frequency_increase is no longer identified -> honest unavailable (was estimable
    in the prior region-only version)."""
    provider = CohortEffectDataProvider(_make_cohort())
    with pytest.raises(EffectDataUnavailable):
        provider.get_training_frame("call_frequency_increase", brand="Kisqali", twin_type="hcp")


@pytest.mark.slow
def test_cohort_provider_end_to_end_reports_cohort_provenance():
    cohort = _make_cohort()
    rng = np.random.default_rng(2)
    twins = pd.DataFrame({"region": rng.choice(_REGIONS, size=600)})
    provider = CohortEffectDataProvider(cohort, seed=42)
    frame = provider.get_training_frame(
        "digital_engagement", brand="Remibrutinib", twin_type="hcp", reference_covariates=twins
    )
    estimate = CohortCausalEstimator().estimate(frame, twins)

    assert estimate.data_provenance == PROVENANCE_COHORT
    assert estimate.estimator_type == "cohort_causal_forest_dml"
    assert estimate.ate > 0.0  # engagement has a real positive effect in this cohort
    # Honest inference CI (NOT the prior fake-tight synthetic-frame width).
    assert estimate.ate_ci_lower < estimate.ate < estimate.ate_ci_upper
    assert estimate.ci_width() > 0.01
    assert len(estimate.per_twin_uplift) == len(twins)
