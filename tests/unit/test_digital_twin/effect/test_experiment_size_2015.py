"""One experiment-size rule for the Digital Twin page and the chat simulator (#2015).

The page's ``recommended_sample_size`` came from ``RecommendationPolicy``'s two-proportion
formula fed the twins' mean ``baseline_propensity`` — a heuristic TREATMENT propensity
(``TwinGenerator._calculate_propensity``) — as the control conversion proportion, while the
outcome is a continuous per-HCP rate (live: mean 1.17-1.19, SD 0.64-0.65, max ~4). The chat
simulator sized from the cohort outcome instead, so the two surfaces stated different
numbers for the same run (164-166 vs 274 per arm on Kisqali email_campaign).

``experiment_size`` is the single rule both use: a two-sided, equal-allocation test of a
continuous outcome at the policy's power and alpha, Cohen's d = |effect| / the outcome SD in
the estimate's comparison arm. The expectations below are closed-form, from a frame whose
comparison-arm SD is planted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.digital_twin.effect.provider import CohortEffectDataProvider
from src.digital_twin.effect.recommendation import experiment_size

_REGIONS = ["northeast", "south", "midwest", "west"]


def _frame(control_outcomes, treated_outcome=1.3, n_per_arm=600):
    """Low-intensity rows (0-4 emails) carry ``control_outcomes`` cyclically; high-intensity
    rows (6-10) a constant. The cohort median intensity is 5, so the comparison arm is
    exactly the low-intensity rows."""
    rows = []
    for i in range(n_per_arm):
        rows.append(
            {
                "region": _REGIONS[i % 4],
                "email_campaign_count": float(i % 5),
                "conversion_rate": control_outcomes[i % len(control_outcomes)],
                "market_share": 0.3,
                "total_rx_count": 50.0,
            }
        )
    for i in range(n_per_arm):
        rows.append(
            {
                "region": _REGIONS[i % 4],
                "email_campaign_count": 6.0 + i % 5,
                "conversion_rate": treated_outcome,
                "market_share": 0.3,
                "total_rx_count": 50.0,
            }
        )
    provider = CohortEffectDataProvider(pd.DataFrame(rows))
    return provider.get_training_frame("email_campaign", brand="Kisqali", twin_type="hcp")


def _closed_form(effect, sd):
    z = norm.ppf(1 - 0.05 / 2) + norm.ppf(0.80)
    return math.ceil(2 * (z / (abs(effect) / sd)) ** 2)


# Alternating 1 -/+ 0.5 over 600 rows: mean 1, sample SD 0.5 * sqrt(600 / 599).
PLANTED_SD = 0.5 * math.sqrt(600 / 599)


def test_the_size_is_the_closed_form_for_the_planted_spread():
    frame = _frame([0.5, 1.5])
    assert float(np.std([0.5, 1.5] * 300, ddof=1)) == pytest.approx(PLANTED_SD)
    size, note = experiment_size(frame, 0.1)
    assert size == _closed_form(0.1, PLANTED_SD)
    assert f"{size} per arm" in note


def test_a_negative_effect_is_sized_by_its_magnitude():
    frame = _frame([0.5, 1.5])
    assert experiment_size(frame, -0.1)[0] == _closed_form(-0.1, PLANTED_SD)
    assert experiment_size(frame, -0.1)[0] == experiment_size(frame, 0.1)[0]


def test_target_regions_use_their_own_comparison_arm():
    # A 3-value outcome cycle against the 4-region cycle, so each region's arm varies.
    frame = _frame([0.5, 1.0, 1.5])
    size, _ = experiment_size(frame, 0.1, regions=["northeast"])
    control = frame.df[frame.df["email_campaign_count"] <= 5]
    sd = control.loc[control["region"] == "northeast", "conversion_rate"].std(ddof=1)
    assert size == _closed_form(0.1, sd)


@pytest.mark.parametrize(
    ("control_outcomes", "effect", "regions", "reason"),
    [
        ([1.0], 0.1, (), "does not vary"),
        ([0.5, 1.5], 0.1, ("atlantis",), "comparison-arm rows"),
        ([0.5, 1.5], 0.0, (), "zero effect"),
        ([0.5, 1.5], 1e6, (), "per arm"),
    ],
)
def test_no_size_is_given_when_it_cannot_be_computed(control_outcomes, effect, regions, reason):
    size, note = experiment_size(_frame(control_outcomes), effect, regions=list(regions))
    assert size is None
    assert "not given" in note and reason in note


def test_a_frame_without_the_cohort_columns_gives_no_size_not_a_propensity_size():
    """The synthetic uplift provider's frame has no region column: no size, with the reason,
    and never the old two-proportion figure. That provider backs the dormant engine defaults
    tracked in #2025 (not fixed here)."""
    from src.digital_twin.effect.provider import SyntheticEffectDataProvider

    frame = SyntheticEffectDataProvider(n=300, true_ate=0.2, seed=42).get_training_frame(
        "email_campaign", brand="Kisqali", twin_type="hcp"
    )
    size, note = experiment_size(frame, 0.2)
    assert size is None and "not given" in note


def test_the_engine_sizes_with_the_shared_rule_and_explains_a_missing_size():
    from src.digital_twin.effect.estimator import TwinEffectEstimator
    from src.digital_twin.effect.provider import SyntheticEffectDataProvider
    from src.digital_twin.models.simulation_models import InterventionConfig
    from src.digital_twin.simulation_engine import SimulationEngine
    from tests.unit.test_digital_twin.test_engine_real_effect import _population

    result = SimulationEngine(
        population=_population(),
        effect_provider=SyntheticEffectDataProvider(n=300, true_ate=0.2, seed=42),
        effect_estimator=TwinEffectEstimator(
            n_estimators=25, max_depth=3, min_training_samples=100
        ),
    ).simulate(InterventionConfig(intervention_type="email_campaign"), use_cache=False)
    assert result.status.value == "completed"
    assert result.recommended_sample_size is None
    assert "not given" in result.recommendation_rationale
