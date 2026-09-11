"""``counterfactual_simulator`` runs the digital-twin simulation instead of scaling a constant (#2015).

The tool returned ``predicted_lift = expected_effect * 0.85`` with ``confidence="medium"``
and ``uncertainty_range = [x0.6, x1.1]``, ignoring ``intervention`` and ``target_entities``
— output shaped like a simulation that was not one (the #621 class).

Measured on the live platform (2026-09-11, Kisqali, ``email_campaign``, 1,000 twins): the
twin engine that serves ``/digital-twin/simulate`` returns a DML estimate on the brand's
per-HCP cohort, ATE 0.135 with a 95% interval [0.092, 0.178]. It serves an intervention
from the twin catalog on a brand, NOT an upstream effect (it estimates its own), and a
region filter leaves the ATE and its interval unchanged — only the per-region effects
respond. The tool now takes what the engine can use (intervention, brand, target regions),
reports what the engine computed with its provenance, and refuses the rest.

The engine runs for real here: real twins, the real cohort provider over a planted-effect
cohort, the real ``CohortCausalEstimator`` (CausalForestDML). Only the database and MLflow
lookups in front of it are outside a unit test; the live cert covers them.
"""

from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
from src.digital_twin.effect.provider import CohortEffectDataProvider
from src.digital_twin.models.simulation_models import InterventionConfig, PopulationFilter
from src.digital_twin.models.twin_models import Brand, DigitalTwin, TwinPopulation, TwinType
from src.digital_twin.simulation_engine import SimulationEngine

pytestmark = pytest.mark.timeout(180)

# Planted per-region effect of a high (above-median) email_campaign_count on conversion.
PLANTED = {"northeast": 0.30, "west": 0.20, "south": 0.10, "midwest": 0.02}


def _run(**kwargs):
    result = tr.counterfactual_simulator(**kwargs)
    return asyncio.run(result) if inspect.isawaitable(result) else result


def _cohort(n_per_region: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for region, tau in PLANTED.items():
        market = rng.uniform(0.0, 1.0, n_per_region)
        emails = rng.poisson(3 + 4 * market).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "email_campaign_count": emails,
                    "market_share": market,
                    "total_rx_count": rng.poisson(60, n_per_region).astype(float),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    treated = (df["email_campaign_count"] > df["email_campaign_count"].median()).astype(float)
    df["conversion_rate"] = (
        0.2 + 0.3 * df["market_share"] + df["_tau"] * treated + rng.normal(0, 0.05, len(df))
    )
    return df.drop(columns="_tau")


def _population(n_per_region: int = 130) -> TwinPopulation:
    rng = np.random.default_rng(5)
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.KISQALI,
            features={"region": region, "decile": int(rng.integers(1, 11))},
            baseline_outcome=float(rng.uniform(0.1, 0.3)),
            baseline_propensity=float(rng.uniform(0.2, 0.5)),
        )
        for region in PLANTED
        for _ in range(n_per_region)
    ]
    return TwinPopulation(twin_type=TwinType.HCP, brand=Brand.KISQALI, twins=twins, size=len(twins))


@pytest.fixture(scope="module")
def provider() -> CohortEffectDataProvider:
    return CohortEffectDataProvider(_cohort())


@pytest.fixture(scope="module")
def engine(provider) -> SimulationEngine:
    return SimulationEngine(
        population=_population(), effect_provider=provider, effect_estimator=CohortCausalEstimator()
    )


def _simulate(engine, regions):
    return engine.simulate(
        InterventionConfig(intervention_type="email_campaign", target_regions=regions),
        population_filter=PopulationFilter(regions=regions) if regions else None,
        use_cache=False,
    )


@pytest.fixture(scope="module")
def whole_population(engine):
    return _simulate(engine, [])


@pytest.fixture(scope="module")
def northeast_only(engine):
    return _simulate(engine, ["northeast"])


def _frame(provider):
    return provider.get_training_frame("email_campaign", brand="Kisqali", twin_type="hcp")


# ---------------------------------------------------------------------------
# What the engine computed is what the tool reports
# ---------------------------------------------------------------------------


def _baseline(engine, regions):
    twins = [t for t in engine.population.twins if not regions or t.features["region"] in regions]
    return float(np.mean([t.baseline_propensity for t in twins]))


def test_the_result_is_the_engine_estimate_not_a_scaled_constant(whole_population, provider):
    out = tr._simulation_results(
        whole_population,
        brand="Kisqali",
        intervention_type="email_campaign",
        frame=_frame(provider),
        targeted=None,
    )
    assert out.effect == pytest.approx(whole_population.simulated_ate)
    assert (out.ci_lower, out.ci_upper) == pytest.approx(
        (whole_population.simulated_ci_lower, whole_population.simulated_ci_upper)
    )
    assert (out.cohort_effect, out.cohort_ci_lower, out.cohort_ci_upper) == (
        out.effect,
        out.ci_lower,
        out.ci_upper,
    )
    assert out.ci_lower < out.effect < out.ci_upper
    assert out.effect_scope == "cohort"
    # The planted cohort-wide effect is the mean of the per-region effects (equal regions).
    assert out.effect == pytest.approx(np.mean(list(PLANTED.values())), abs=0.05)
    assert set(out.region_effects) == set(PLANTED)
    for region, tau in PLANTED.items():
        assert out.region_effects[region] == pytest.approx(tau, abs=0.07)
    assert out.twin_count == len(_population().twins)
    assert out.data_provenance == "cohort_estimated_synthetic_gold_v1"
    assert out.recommendation == whole_population.recommendation.value
    assert out.recommended_sample_size == whole_population.recommended_sample_size


def test_a_targeted_request_is_answered_with_inference_on_the_targeted_regions(
    engine, whole_population, provider
):
    """codex iter-1 F1: the engine re-uses the cohort-wide interval and recommendation for a
    region filter (measured), so a targeted question needs its own inference. Midwest's
    planted effect (0.02) is below the policy's 0.05 minimum while the cohort's is not."""
    frame = _frame(provider)
    targeted = tr._targeted_effect(frame, ["midwest"], baseline_rate=_baseline(engine, ["midwest"]))
    midwest_run = _simulate(engine, ["midwest"])
    out = tr._simulation_results(
        midwest_run,
        brand="Kisqali",
        intervention_type="email_campaign",
        frame=frame,
        targeted=targeted,
    )
    assert out.effect_scope == "targeted regions ['midwest']"
    assert out.target_regions == ["midwest"]
    assert out.effect == pytest.approx(PLANTED["midwest"], abs=0.07)
    assert out.ci_lower < out.effect < out.ci_upper
    assert out.ci_lower < PLANTED["midwest"] < out.ci_upper
    # Same forest (same seed and frame) as the engine's fit: the targeted point estimate
    # is the engine's region effect, now with an interval.
    assert out.effect == pytest.approx(out.region_effects["midwest"], abs=1e-9)
    assert whole_population.recommendation.value == "deploy"
    assert out.recommendation != "deploy"
    assert out.cohort_effect == pytest.approx(whole_population.simulated_ate)
    assert out.twin_count == 130


def test_a_multi_region_target_is_the_average_effect_over_those_regions(engine, provider):
    frame = _frame(provider)
    targeted = tr._targeted_effect(
        frame, ["northeast", "west"], baseline_rate=_baseline(engine, ["northeast", "west"])
    )
    assert targeted.effect == pytest.approx((PLANTED["northeast"] + PLANTED["west"]) / 2, abs=0.07)
    assert targeted.ci_lower < targeted.effect < targeted.ci_upper
    assert targeted.recommendation == "deploy"
    assert targeted.cohort_rows == 600


def test_a_target_region_the_cohort_does_not_cover_is_refused(engine):
    """codex iter-1 F3: ``CohortCausalEstimator`` gives a twin whose region is absent from
    the cohort the COHORT ATE; that must never be reported as the region's effect."""
    no_west = CohortEffectDataProvider(_cohort().query("region != 'west'"))
    frame = _frame(no_west)
    with pytest.raises(ToolRefusalError, match="west"):
        tr._targeted_effect(frame, ["west"], baseline_rate=0.3)
    run = SimulationEngine(
        population=_population(), effect_provider=no_west, effect_estimator=CohortCausalEstimator()
    ).simulate(InterventionConfig(intervention_type="email_campaign"), use_cache=False)
    assert run.effect_heterogeneity.by_region["west"]["ate"] == pytest.approx(run.simulated_ate)
    out = tr._simulation_results(
        run, brand="Kisqali", intervention_type="email_campaign", frame=frame, targeted=None
    )
    assert "west" not in out.region_effects
    assert set(out.region_effects) == {"northeast", "south", "midwest"}


def test_the_assumptions_name_the_contrast_the_estimate_answers(whole_population, provider):
    out = tr._simulation_results(
        whole_population,
        brand="Kisqali",
        intervention_type="email_campaign",
        frame=_frame(provider),
        targeted=None,
    )
    text = " ".join(out.assumptions)
    for fragment in ("email_campaign_count", "conversion_rate", "median", "market_share"):
        assert fragment in text, fragment
    assert "synthetic" in text


def test_a_failed_engine_run_is_refused_not_reported(engine, provider):
    small = SimulationEngine(
        population=TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.KISQALI,
            twins=_population(n_per_region=20).twins,
            size=80,
        ),
        effect_provider=provider,
        effect_estimator=CohortCausalEstimator(),
    )
    failed = _simulate(small, [])
    assert failed.status.value == "failed"
    with pytest.raises(ToolRefusalError, match="Insufficient twins"):
        tr._simulation_results(
            failed,
            brand="Kisqali",
            intervention_type="email_campaign",
            frame=_frame(provider),
            targeted=None,
        )


# ---------------------------------------------------------------------------
# Unreachable services are retried, not refused (codex iter-1 F2)
# ---------------------------------------------------------------------------


async def test_an_unreachable_database_is_not_a_refusal():
    """The unit tree pins Supabase to a dead port (#1420). A real client against it must
    surface as a retryable failure: neither the model lookup nor the cohort load may turn a
    connection error into 'not identified'."""
    from src.memory.services.factories import get_async_supabase_client

    with pytest.raises(RuntimeError) as model_lookup:
        await tr.counterfactual_simulator(intervention="email_campaign", brand="Kisqali")
    assert not isinstance(model_lookup.value, (ToolRefusalError, ToolInputError))
    assert "could be read" in str(model_lookup.value)

    load = tr._load_cohort_provider
    client = await get_async_supabase_client()
    with pytest.raises(Exception) as cohort_load:
        await load(client, "email_campaign", "Kisqali")
    assert not isinstance(cohort_load.value, (ToolRefusalError, ToolInputError))
    assert "connect" in f"{type(cohort_load.value).__name__} {cohort_load.value}".lower()


def test_the_route_loader_still_degrades_to_none(provider):
    """The route keeps its contract: ``build_cohort_provider_or_none`` never raises. The
    shared usability check is the same function the tool uses."""
    from src.digital_twin.effect.cohort_loader import cohort_provider_from_frame

    cohort = _cohort()
    built = cohort_provider_from_frame(cohort, "email_campaign")
    assert built is not None and len(built._cohort) == len(cohort)
    assert cohort_provider_from_frame(cohort.head(100), "email_campaign") is None
    assert cohort_provider_from_frame(cohort.drop(columns="market_share"), "email_campaign") is None
    assert cohort_provider_from_frame(pd.DataFrame(), "email_campaign") is None


# ---------------------------------------------------------------------------
# Inputs the engine cannot use are refused before any lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"intervention": None, "brand": "Kisqali"}, "intervention"),
        ({"intervention": "increase rep visits", "brand": "Kisqali"}, "email_campaign"),
        ({"intervention": "email_campaign", "brand": None}, "brand"),
        ({"intervention": "email_campaign", "brand": "Cosentyx"}, "Kisqali"),
        (
            {"intervention": "email_campaign", "brand": "Kisqali", "target_entities": ["oncology"]},
            "northeast",
        ),
        (
            {"intervention": "email_campaign", "brand": "Kisqali", "target_entities": "northeast"},
            "target_entities",
        ),
        (
            {"intervention": "email_campaign", "brand": "Kisqali", "target_entities": [None]},
            "target_entities",
        ),
    ],
)
def test_inputs_the_twin_engine_cannot_use_are_refused(kwargs, reason):
    with pytest.raises(ToolInputError, match=reason):
        _run(**kwargs)


def test_a_legacy_expected_effect_does_not_produce_a_lift():
    """The #1573 q08 plan shape (free-text intervention + entities + expected_effect) no
    longer yields a number: the text is not a catalog intervention, and an upstream effect
    is not an input the engine reads."""
    with pytest.raises(ToolInputError, match="not a digital-twin intervention"):
        _run(
            intervention="increase rep visits",
            brand="Kisqali",
            target_entities=["west"],
            expected_effect=0.2,
        )


def test_catalog_labels_brands_and_regions_are_matched_case_insensitively():
    assert tr._counterfactual_inputs("Email Campaign", "kisqali", ["NorthEast", "west"]) == (
        "email_campaign",
        "Kisqali",
        ["northeast", "west"],
    )
    assert tr._counterfactual_inputs("call-frequency-increase", "FABHALTA", None) == (
        "call_frequency_increase",
        "Fabhalta",
        [],
    )


def test_the_planner_is_offered_exactly_the_catalog_brands_and_regions():
    """The planner-facing descriptions spell out the vocabulary (the planner sees no enum);
    they must list exactly what ``_counterfactual_inputs`` accepts."""
    from src.digital_twin.effect.provider import INTERVENTION_CATALOG
    from src.digital_twin.models.twin_models import Region
    from src.tool_registry.registry import get_registry

    params = {
        p.name: p.description
        for p in get_registry().get("counterfactual_simulator").schema.input_parameters
    }
    listed = params["intervention"].split("One of: ", 1)[1].split(", ")
    assert listed == [value for value, _label in INTERVENTION_CATALOG]
    for brand in Brand:
        assert brand.value in params["brand"]
    for region in Region:
        assert region.value in params["target_entities"]


# ---------------------------------------------------------------------------
# The orchestration after twin generation (codex iter-2 F2)
# ---------------------------------------------------------------------------


def _population_with_regional_baselines() -> TwinPopulation:
    """Twins whose baseline propensity depends on region, so a policy fed the whole
    population's baseline gives a different sample size than one fed the targeted twins'."""
    baselines = {"northeast": 0.10, "west": 0.60, "south": 0.35, "midwest": 0.35}
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.KISQALI,
            features={"region": region},
            baseline_outcome=0.2,
            baseline_propensity=baselines[region],
        )
        for region in PLANTED
        for _ in range(130)
    ]
    return TwinPopulation(twin_type=TwinType.HCP, brand=Brand.KISQALI, twins=twins, size=len(twins))


def test_simulating_a_targeted_population_applies_the_policy_to_the_targeted_twins(provider):
    from uuid import uuid4

    from src.digital_twin.effect.estimate import PROVENANCE_COHORT, EffectEstimate
    from src.digital_twin.effect.recommendation import PolicyThresholds, RecommendationPolicy

    frame = _frame(provider)
    model_id = uuid4()
    result, targeted = tr._simulate_population(
        _population_with_regional_baselines(),
        provider=provider,
        frame=frame,
        intervention_type="email_campaign",
        regions=["northeast"],
        model_id=model_id,
    )
    assert result.status.value == "completed" and str(result.model_id) == str(model_id)
    assert targeted is not None and targeted.regions == ["northeast"]

    def policy_n(baseline):
        estimate = EffectEstimate(
            ate=targeted.effect,
            ate_ci_lower=targeted.ci_lower,
            ate_ci_upper=targeted.ci_upper,
            att=None,
            atc=None,
            per_twin_uplift=np.array([targeted.effect]),
            auuc=None,
            qini=None,
            feature_importances=None,
            n_train=targeted.cohort_rows,
            estimator_type="cohort_causal_forest_dml",
            data_provenance=PROVENANCE_COHORT,
        )
        return RecommendationPolicy(PolicyThresholds(min_effect=0.05)).decide(
            estimate, baseline_rate=baseline
        )

    decision, rationale, n_targeted = policy_n(0.10)
    assert targeted.recommended_sample_size == n_targeted
    assert (targeted.recommendation, targeted.recommendation_rationale) == (
        decision.value,
        rationale,
    )
    assert policy_n(float(np.mean([0.10, 0.60, 0.35, 0.35])))[2] != n_targeted

    out = tr._simulation_results(
        result, brand="Kisqali", intervention_type="email_campaign", frame=frame, targeted=targeted
    )
    assert out.effect == targeted.effect and out.recommended_sample_size == n_targeted
    assert out.effect_scope == "targeted regions ['northeast']"
    assert out.model_id == str(model_id)


def test_simulating_the_whole_population_has_no_targeted_inference(provider):
    from uuid import uuid4

    result, targeted = tr._simulate_population(
        _population_with_regional_baselines(),
        provider=provider,
        frame=_frame(provider),
        intervention_type="email_campaign",
        regions=[],
        model_id=uuid4(),
    )
    assert targeted is None and result.twin_count == 520
