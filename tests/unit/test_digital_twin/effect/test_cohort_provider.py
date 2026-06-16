"""Phase 2: cohort-estimated effect provider.

The effect MAGNITUDE is a region-standardized treatment effect estimated from the
brand's synthetic-gold cohort, so it is brand- and intervention-differentiated
(NOT the flat synthetic uplift). These tests pin: (1) the estimator differentiates
treatments and removes region confounding, (2) the provider carries the
data-derived ATE through a twin-aligned frame, (3) the end-to-end estimate reports
the cohort provenance.
"""

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import PROVENANCE_COHORT
from src.digital_twin.effect.estimator import TwinEffectEstimator
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
    conversion = (
        base
        + 0.06 * engagement
        + 0.01 * call_frequency
        + rng.normal(0, 0.08, size=n)
    )
    return pd.DataFrame(
        {
            "region": regions,
            "engagement_score": engagement,
            "call_frequency": call_frequency,
            "conversion_rate": conversion,
        }
    )


def test_region_standardized_ate_differentiates_treatments():
    cohort = _make_cohort()
    eng = region_standardized_ate(cohort, "engagement_score")
    cf = region_standardized_ate(cohort, "call_frequency")
    # Engagement has the far stronger planted effect than call frequency.
    assert eng > cf > 0.0
    assert eng > 0.15  # a clearly non-trivial, deployable effect
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
    # Region positively confounds engagement→conversion, so the unadjusted gap
    # is LARGER than the region-standardized estimate (adjustment does real work).
    assert raw > standardized
    assert standardized > 0.0


def test_region_standardized_ate_insufficient_rows_fails_closed():
    with pytest.raises(EffectDataUnavailable):
        region_standardized_ate(_make_cohort(n=100), "engagement_score")


def test_region_standardized_ate_missing_treatment_column_fails_closed():
    cohort = _make_cohort().drop(columns=["call_frequency"])
    with pytest.raises(EffectDataUnavailable):
        region_standardized_ate(cohort, "call_frequency")


def test_cohort_provider_frame_carries_data_derived_ate():
    cohort = _make_cohort()
    expected = region_standardized_ate(cohort, "engagement_score")
    provider = CohortEffectDataProvider(cohort, seed=42)
    frame = provider.get_training_frame(
        "digital_engagement", brand="Remibrutinib", twin_type="hcp"
    )
    # The frame's injected ground-truth equals the data-derived cohort ATE.
    assert frame.ground_truth_ate == pytest.approx(expected)
    assert frame.treatment_var == "treatment"
    assert frame.outcome_var == "outcome"
    assert len(frame.df) >= 2000
    assert frame.confounders


def test_cohort_provider_non_mappable_intervention_fails_closed():
    provider = CohortEffectDataProvider(_make_cohort())
    with pytest.raises(EffectDataUnavailable):
        provider.get_training_frame("email_campaign", brand="X", twin_type="hcp")


def test_cohort_provider_confounders_align_with_twin_covariates():
    rng = np.random.default_rng(1)
    twins = pd.DataFrame(
        {
            "decile": rng.integers(1, 11, size=400).astype(float),
            "years_experience": rng.normal(15, 5, size=400),
            "peer_influence_score": rng.uniform(0, 1, size=400),
        }
    )
    provider = CohortEffectDataProvider(_make_cohort())
    frame = provider.get_training_frame(
        "call_frequency_increase",
        brand="Kisqali",
        twin_type="hcp",
        reference_covariates=twins,
    )
    # Confounders are exactly the twin numeric columns → estimator can score twins.
    assert frame.confounders == ["decile", "years_experience", "peer_influence_score"]


def test_cohort_provider_end_to_end_reports_cohort_provenance():
    cohort = _make_cohort()
    expected_ate = region_standardized_ate(cohort, "engagement_score")
    rng = np.random.default_rng(2)
    twins = pd.DataFrame(
        {
            "decile": rng.integers(1, 11, size=600).astype(float),
            "years_experience": rng.normal(15, 5, size=600),
            "peer_influence_score": rng.uniform(0, 1, size=600),
        }
    )
    provider = CohortEffectDataProvider(cohort, seed=42)
    frame = provider.get_training_frame(
        "digital_engagement", brand="Remibrutinib", twin_type="hcp", reference_covariates=twins
    )
    estimator = TwinEffectEstimator(provenance=PROVENANCE_COHORT)
    estimate = estimator.estimate(frame, twins)
    assert estimate.data_provenance == PROVENANCE_COHORT
    # The estimator recovers (approximately) the injected, data-derived ATE.
    assert estimate.ate == pytest.approx(expected_ate, abs=0.08)
