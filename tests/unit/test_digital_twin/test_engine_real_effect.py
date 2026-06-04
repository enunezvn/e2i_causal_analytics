"""Focused integration test: SimulationEngine wired to the REAL uplift effect engine.

Proves the H5 rewire end-to-end with a SMALL injected estimator (one fast uplift
fit) instead of the deleted INTERVENTION_EFFECTS heuristic:

* a labeled (treatment, outcome, confounders) frame from the synthetic provider
  is fitted and scored over the twin population,
* the result carries a CI-based ATE, a DEPLOY/REFINE/SKIP recommendation, and a
  ``data_provenance`` label, and
* the engine FAILS CLOSED (status=failed, no fabricated ATE) when the twin
  features have no numeric covariates the estimator can use.
"""

from __future__ import annotations

import numpy as np

from src.digital_twin.effect.estimator import TwinEffectEstimator
from src.digital_twin.effect.provider import SyntheticEffectDataProvider
from src.digital_twin.models.simulation_models import InterventionConfig
from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.simulation_engine import SimulationEngine


def _population(n: int = 150) -> TwinPopulation:
    """A population of real DigitalTwin objects with NUMERIC features.

    The numeric columns (decile / digital_engagement_score / peer_influence_score /
    years_experience) become the uplift confounders; varying them gives the model
    in-distribution covariates to score.
    """
    rng = np.random.default_rng(7)
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "decile": int(rng.integers(1, 11)),
                "digital_engagement_score": float(rng.uniform(0.2, 0.9)),
                "peer_influence_score": float(rng.uniform(0.3, 0.9)),
                "years_experience": int(rng.integers(1, 40)),
            },
            baseline_outcome=float(rng.uniform(0.05, 0.25)),
            baseline_propensity=float(rng.uniform(0.3, 0.8)),
        )
        for _ in range(n)
    ]
    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=twins,
        size=len(twins),
    )


def _non_numeric_population(n: int = 150) -> TwinPopulation:
    """Population whose twin features are ALL strings -> no numeric covariates."""
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "specialty": "rheumatology",
                "region": "northeast",
                "adoption_stage": "early_majority",
            },
            baseline_outcome=0.1,
            baseline_propensity=0.5,
        )
        for _ in range(n)
    ]
    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=twins,
        size=len(twins),
    )


def test_simulate_produces_labeled_ci_based_result() -> None:
    pop = _population()
    engine = SimulationEngine(
        population=pop,
        effect_provider=SyntheticEffectDataProvider(n=300, true_ate=0.2, seed=42),
        effect_estimator=TwinEffectEstimator(
            n_estimators=25, max_depth=3, min_training_samples=100
        ),
    )
    result = engine.simulate(
        InterventionConfig(intervention_type="email_campaign"), use_cache=False
    )
    assert result.status.value == "completed"
    assert result.data_provenance == "synthetic_uplift_v1"
    assert result.simulated_ci_lower <= result.simulated_ate <= result.simulated_ci_upper
    assert result.recommendation.value in {"deploy", "refine", "skip"}


def test_simulate_fails_closed_on_non_numeric_twins() -> None:
    pop = _non_numeric_population()
    engine = SimulationEngine(
        population=pop,
        effect_provider=SyntheticEffectDataProvider(n=300, true_ate=0.2, seed=42),
        effect_estimator=TwinEffectEstimator(
            n_estimators=25, max_depth=3, min_training_samples=100
        ),
    )
    result = engine.simulate(
        InterventionConfig(intervention_type="email_campaign"), use_cache=False
    )
    assert result.status.value == "failed"
    # Fail-closed contract: no fabricated effect leaks out.
    assert result.simulated_ate == 0.0
    assert result.data_provenance is None
