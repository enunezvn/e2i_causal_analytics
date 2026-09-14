"""#2054: the engine reports only the subgroup axes its estimator resolves.

On the live cohort path the estimator's per-twin score is a step function of region, so
averaging it over the twins' specialty / decile / adoption_stage draws produces numbers
whose only content is the twin draw. Measured on the base commit with the real
``_calculate_heterogeneity``:

    n_twins   specialty spread   region spread
        100            0.04901         0.23339
      1,000            0.02086         0.23339
     10,000            0.00714         0.23339
    100,000            0.00207         0.23339

Region is invariant because it is what the estimate resolves; the other axes decay
towards zero like sampling noise. (The design doc's table reads 0.01376 / 0.00287 at the
same twin counts: it drove an extracted copy of ``calc_group_stats`` over a different twin
draw. The decay and region's invariance are the finding; the digits are harness-specific.) Those three axes are therefore not reported at all, and
``by_region`` is reported from the COHORT rows the effect was estimated on.

The synthetic path (``TwinEffectEstimator``) declares every axis and keeps reporting all
four — ``test_simulation_engine.py::TestHeterogeneousEffects`` pins that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
from src.digital_twin.effect.provider import CohortEffectDataProvider
from src.digital_twin.models.simulation_models import InterventionConfig, PopulationFilter
from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.simulation_engine import SimulationEngine

TRUE_CATE = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}
SPECIALTIES = ("oncology", "cardiology", "neurology", "primary_care")
ADOPTION = ("innovator", "early_majority", "late_majority")


def _cohort(n_per_region: int = 200, seed: int = 42) -> pd.DataFrame:
    """Region-heterogeneous effect of engagement on conversion, confounded by
    market_share (the synthetic-gold DGP), with unequal region sizes."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, (region, tau) in enumerate(TRUE_CATE.items()):
        n = n_per_region + 40 * i
        market = rng.uniform(0.0, 1.0, n)
        logvol = rng.normal(0.0, 1.0, n)
        eng_logit = 1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "engagement_score": 10.0 / (1.0 + np.exp(-eng_logit)),
                    "market_share": market,
                    "total_rx_count": np.expm1(np.abs(logvol) * 2.0),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    t_bin = (df["engagement_score"] > df["engagement_score"].median()).astype(float)
    df["conversion_rate"] = (
        0.5 + 0.8 * df["market_share"] + df["_tau"] * t_bin + rng.normal(0.0, 0.25, len(df))
    ).clip(lower=0.0)
    return df.drop(columns="_tau")


def _population(n: int, regions: tuple[str, ...] = tuple(TRUE_CATE), seed: int = 7):
    """Twins whose specialty / decile / adoption_stage are drawn INDEPENDENTLY of region,
    exactly as ``twin_generator`` draws them."""
    rng = np.random.default_rng(seed)
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.KISQALI,
            features={
                "region": str(rng.choice(regions)),
                "specialty": str(rng.choice(SPECIALTIES)),
                "decile": int(rng.integers(1, 11)),
                "adoption_stage": str(rng.choice(ADOPTION)),
                "market_share": float(rng.uniform(0.0, 1.0)),
                "total_rx_count": float(rng.uniform(0.0, 500.0)),
            },
            baseline_outcome=float(rng.uniform(0.05, 0.25)),
            baseline_propensity=float(rng.uniform(0.3, 0.8)),
        )
        for _ in range(n)
    ]
    return TwinPopulation(twin_type=TwinType.HCP, brand=Brand.KISQALI, twins=twins, size=len(twins))


def _engine(cohort: pd.DataFrame, population, target_regions=()):
    return SimulationEngine(
        population=population,
        effect_provider=CohortEffectDataProvider(cohort),
        effect_estimator=CohortCausalEstimator(target_regions=target_regions),
    )


def _config():
    return InterventionConfig(intervention_type="digital_engagement", duration_weeks=8)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return _cohort()


def test_cohort_path_reports_only_region_and_is_invariant_to_twin_count(cohort):
    """The reported ``by_region`` must be identical at 100 / 1,000 / 10,000 twins, and the
    three axes the estimate does not resolve must be absent rather than noise."""
    reported = []
    for n_twins in (100, 1000, 10000):
        result = _engine(cohort, _population(n_twins)).simulate(_config(), use_cache=False)
        assert result.error_message is None, result.error_message

        het = result.effect_heterogeneity
        assert het.by_specialty == {}
        assert het.by_decile == {}
        assert het.by_adoption_stage == {}
        assert set(het.by_region) == set(TRUE_CATE)
        reported.append(het.by_region)

    for other in reported[1:]:
        assert set(other) == set(reported[0])
        for region, stats in other.items():
            # abs=1e-12 is repeated-fit float jitter, twelve orders below the
            # twin-count-driven spread this pins out (0.049 -> 0.002 on by_specialty).
            assert stats["ate"] == pytest.approx(reported[0][region]["ate"], abs=1e-12)
            assert stats["n"] == reported[0][region]["n"]


def test_reported_region_n_is_cohort_rows_not_twins(cohort):
    """The evidence behind a reported subgroup effect is the rows it was estimated on."""
    result = _engine(cohort, _population(1000)).simulate(_config(), use_cache=False)

    expected = cohort["region"].value_counts().to_dict()
    assert {r: int(s["n"]) for r, s in result.effect_heterogeneity.by_region.items()} == {
        r: int(expected[r]) for r in expected
    }
    assert sum(int(s["n"]) for s in result.effect_heterogeneity.by_region.values()) == len(cohort)


def test_a_single_targeted_regions_subgroup_equals_the_headline(cohort):
    """The #2054 headline harm: a subgroup contradicting the headline for what is
    nominally the same population. With one region targeted there is one population, so
    the two numbers must be the same number."""
    for region in TRUE_CATE:
        population = _population(300, regions=(region,))
        result = _engine(cohort, population, target_regions=(region,)).simulate(
            _config(),
            population_filter=PopulationFilter(regions=[region]),
            use_cache=False,
        )
        assert result.error_message is None, result.error_message

        by_region = result.effect_heterogeneity.by_region
        assert set(by_region) == {region}
        assert by_region[region]["ate"] == pytest.approx(result.simulated_ate, abs=1e-9)


def test_the_engine_reports_every_axis_in_the_declaration_vocabulary(cohort):
    """Guards the wiring between `SUBGROUP_AXES` and the `by_*` fields: an axis added to the
    vocabulary and declared by an estimator must actually reach the response. Before the
    engine looped over `SUBGROUP_AXES` it called four hardcoded `axis_stats`, so a fifth
    axis would have been declared and silently never reported."""
    from src.digital_twin.effect.estimate import SUBGROUP_AXES

    result = _engine(cohort, _population(300)).simulate(_config(), use_cache=False)
    reported = {
        name.removeprefix("by_")
        for name in type(result.effect_heterogeneity).model_fields
        if name.startswith("by_")
    }
    assert reported == set(SUBGROUP_AXES)
    for axis in SUBGROUP_AXES:
        assert isinstance(getattr(result.effect_heterogeneity, f"by_{axis}"), dict)
