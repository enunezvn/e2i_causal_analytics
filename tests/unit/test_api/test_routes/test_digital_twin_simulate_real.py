"""R1 regression: /simulate must use the real SimulationEngine signature and must
NOT 200-persist a FAILED result. These tests deliberately do NOT mock
SimulationEngine — that mock is exactly what hid H4 (the model_id= TypeError)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.models.twin_models import Brand, TwinType
from src.digital_twin.simulation_engine import SimulationEngine


@pytest.mark.unit
def test_engine_does_not_accept_model_id_kwarg():
    """Pin the real engine signature: model_id is NOT a constructor kwarg.
    (RED today only if someone re-adds it; locks the contract the route relies on.)"""
    from src.digital_twin.models.twin_models import TwinPopulation

    pop = TwinPopulation(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB, twins=[], size=0)
    with pytest.raises(TypeError):
        SimulationEngine(population=pop, model_id=uuid4())  # type: ignore[call-arg]


@pytest.mark.unit
def test_route_construction_smoke_uses_real_engine(monkeypatch):
    """The route's construction pattern must work against the REAL engine:
    construct with population only, then assign model_id. RED before fix because
    the route passes model_id= as a kwarg → TypeError → 500."""
    from src.digital_twin.models.twin_models import TwinPopulation

    pop = TwinPopulation(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB, twins=[], size=0)
    model_id = uuid4()
    engine = SimulationEngine(population=pop)  # no model_id kwarg
    engine.model_id = model_id
    assert engine.model_id == model_id


@pytest.mark.unit
def test_failed_simulation_is_not_persisted_as_200(monkeypatch):
    """N1: a FAILED SimulationResult must NOT be saved + returned 200.
    RED today: save_simulation runs and a 200 is returned."""
    import asyncio

    from fastapi import HTTPException

    from src.api.routes import digital_twin as dt
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        SimulationRecommendation,
        SimulationResult,
        SimulationStatus,
    )

    failed = SimulationResult(
        model_id=uuid4(),
        intervention_config=InterventionConfig(intervention_type="email_campaign"),
        twin_count=0,
        simulated_ate=0.0,
        simulated_ci_lower=0.0,
        simulated_ci_upper=0.0,
        simulated_std_error=0.0,
        recommendation=SimulationRecommendation.REFINE,
        recommendation_rationale="Insufficient twins after filtering (need >= 100)",
        simulation_confidence=0.0,
        status=SimulationStatus.FAILED,
        execution_time_ms=1,
    )
    saved = {"called": False}

    async def fake_save(result, brand):
        saved["called"] = True
        return result.simulation_id

    repo = SimpleNamespace(
        save_simulation=fake_save,
        list_active_models=AsyncMock(
            return_value=[
                {
                    "model_id": str(uuid4()),
                    "mlflow_model_uri": "models:/x/1",
                    "mlflow_run_id": "r",
                }
            ]
        ),
        # Phase-2 cohort lookup reads repo.client; None → no cohort provider
        # (synthetic-uplift fallback), which is what this real-engine test wants.
        client=None,
    )
    monkeypatch.setattr(dt, "_get_twin_repo", AsyncMock(return_value=repo))

    # _load_trained_generator must yield a real generator whose .generate() returns
    # a real TwinPopulation (NOT a bare AsyncMock, whose .generate() is itself async
    # and returns an un-awaited coroutine the real engine then chokes on). Only
    # SimulationEngine.simulate is patched — the real __init__ must accept the pop.
    from src.digital_twin.models.twin_models import TwinPopulation

    real_pop = TwinPopulation(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB, twins=[], size=0)
    fake_gen = SimpleNamespace(model_id=uuid4(), generate=lambda n: real_pop)
    monkeypatch.setattr(dt, "_load_trained_generator", AsyncMock(return_value=fake_gen))

    with patch.object(SimulationEngine, "simulate", return_value=failed):

        async def _call():
            from src.api.routes.digital_twin import (
                InterventionConfigRequest,
                SimulateRequest,
                run_simulation,
            )

            req = SimulateRequest(
                intervention=InterventionConfigRequest(intervention_type="email_campaign"),
                brand=dt.BrandEnum.REMIBRUTINIB,
                twin_count=100,
            )
            with pytest.raises(HTTPException) as ei:
                await run_simulation(req, user={"sub": "op", "roles": ["operator"]})
            assert ei.value.status_code == 422

        asyncio.run(_call())
    assert saved["called"] is False, "a FAILED result must not be persisted"


@pytest.mark.unit
def test_simulation_response_exposes_data_provenance():
    from src.api.routes.digital_twin import SimulationResponse

    assert "data_provenance" in SimulationResponse.model_fields


# =============================================================================
# #2023: a region filter must change the headline effect, not only the twins
# =============================================================================


def _region_heterogeneous_cohort(n_per_region: int = 250, seed: int = 7):
    """A per-HCP cohort whose treatment effect differs sharply BY REGION.

    Not the planted-truth DGP (that lives with the estimator's recovery tests) — this only
    has to give the four regions clearly different effects, so a run filtered to one region
    cannot accidentally agree with the cohort-wide number.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    tau_by_region = {"northeast": 0.45, "west": 0.30, "south": 0.18, "midwest": 0.08}
    frames = []
    for region, tau in tau_by_region.items():
        market = rng.uniform(0.0, 1.0, n_per_region)
        engagement = 10.0 / (
            1.0 + np.exp(-(1.6 * (market - 0.5) + rng.normal(0.0, 0.5, n_per_region)))
        )
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "email_campaign_count": engagement,
                    "market_share": market,
                    "total_rx_count": rng.gamma(2.0, 50.0, n_per_region),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    treated = (df["email_campaign_count"] > df["email_campaign_count"].median()).astype(float)
    df["conversion_rate"] = (
        0.5 + 0.8 * df["market_share"] + df["_tau"] * treated + rng.normal(0.0, 0.25, len(df))
    ).clip(lower=0.0)
    return df.drop(columns="_tau")


def _twin_population(brand, n_per_region: int = 150):
    """A real TwinPopulation spread over the cohort's regions (>= 100 per region, so a
    single-region filter still clears the engine's minimum)."""
    from src.digital_twin.models.twin_models import DigitalTwin, TwinPopulation

    twins = []
    for region in ("northeast", "west", "south", "midwest"):
        for i in range(n_per_region):
            twins.append(
                DigitalTwin(
                    twin_type=TwinType.HCP,
                    brand=brand,
                    features={
                        "region": region,
                        "specialty": "oncology",
                        "decile": 5,
                        "adoption_stage": "early_majority",
                    },
                    baseline_outcome=0.5,
                    baseline_propensity=0.4,
                )
            )
    return TwinPopulation(twin_type=TwinType.HCP, brand=brand, twins=twins, size=len(twins))


class _CapturingClient:
    """Minimal async Supabase stand-in that records the row handed to insert().

    Lets a test drive the REAL SimulationRepository.save_simulation serializer instead of
    asserting on the in-memory SimulationResult, so what actually reaches twin_simulations
    is what gets pinned.
    """

    def __init__(self, sink):
        self._sink = sink

    def table(self, name):
        self._sink["table"] = name
        return self

    def insert(self, row):
        self._sink["row"] = row
        return self

    async def execute(self):
        return SimpleNamespace(data=[self._sink["row"]])


def _run_simulate(monkeypatch, cohort, *, regions, capture=None):
    """Drive the REAL route against the REAL engine + cohort estimator. Only the two
    external seams are faked: the DB frame load and the MLflow generator hydration.

    ``capture``: a dict to receive the row the REAL repository serializer would insert.
    """
    import asyncio

    from src.api.routes import digital_twin as dt
    from src.digital_twin.effect import cohort_loader

    brand = Brand.KISQALI
    saved = {}

    async def fake_load_cohort_frame(client, brand_value):
        return cohort

    monkeypatch.setattr(cohort_loader, "load_cohort_frame", fake_load_cohort_frame)

    async def fake_save(result, brand_value):
        saved["result"] = result
        if capture is not None:
            from src.digital_twin.twin_repository import SimulationRepository

            real = SimulationRepository(supabase_client=_CapturingClient(capture))
            await real.save_simulation(result, brand_value)
        return result.simulation_id

    repo = SimpleNamespace(
        client=object(),  # non-None: the real cohort provider path runs
        save_simulation=fake_save,
        list_active_models=AsyncMock(
            return_value=[
                {
                    "model_id": str(uuid4()),
                    "mlflow_model_uri": "models:/x/1",
                    "mlflow_run_id": "r",
                }
            ]
        ),
    )
    monkeypatch.setattr(dt, "_get_twin_repo", AsyncMock(return_value=repo))

    population = _twin_population(brand)
    fake_gen = SimpleNamespace(model_id=uuid4(), generate=lambda n: population)
    monkeypatch.setattr(dt, "_load_trained_generator", AsyncMock(return_value=fake_gen))

    req = dt.SimulateRequest(
        intervention=dt.InterventionConfigRequest(intervention_type="email_campaign"),
        brand=dt.BrandEnum.KISQALI,
        twin_count=600,
        population_filters=dt.PopulationFilterRequest(regions=list(regions)) if regions else None,
        calculate_heterogeneity=False,
    )
    response = asyncio.run(dt.run_simulation(req, user={"sub": "op", "roles": ["operator"]}))
    return response, saved.get("result")


@pytest.mark.unit
def test_region_filtered_simulate_reports_that_regions_effect(monkeypatch):
    """#2023: filtering to a region must move the headline effect, CI and recommendation to
    that region's estimate — the number the chat ``counterfactual_simulator`` gives for the
    same question — with the cohort-wide effect reported alongside.

    RED before the fix: the filtered run returns the cohort-wide 'northeast == cohort'
    effect byte-for-byte, changing only which twins were simulated.
    """
    from src.digital_twin.effect.cohort_causal_estimator import (
        CohortCausalEstimator,
        estimate_cohort_effect,
    )
    from src.digital_twin.effect.provider import CohortEffectDataProvider
    from src.digital_twin.effect.recommendation import PolicyThresholds, experiment_size
    from src.digital_twin.simulation_engine import SimulationEngine

    cohort = _region_heterogeneous_cohort()
    unfiltered, _ = _run_simulate(monkeypatch, cohort, regions=[])
    filtered, persisted = _run_simulate(monkeypatch, cohort, regions=["northeast"])

    # The reference the chat tool computes for the same region, from the same frame.
    defaults = CohortCausalEstimator()
    reference = estimate_cohort_effect(
        cohort,
        "email_campaign_count",
        alpha=defaults.alpha,
        seed=defaults.seed,
        target_regions=["northeast"],
    )

    assert filtered.simulated_ate != unfiltered.simulated_ate, (
        "#2023: the region filter changed only the twins, not the effect"
    )
    assert filtered.simulated_ate == pytest.approx(round(reference.target_ate, 4), abs=1e-9)
    assert filtered.simulated_ci_lower == pytest.approx(
        round(reference.target_ci_lower, 4), abs=1e-9
    )
    assert filtered.simulated_ci_upper == pytest.approx(
        round(reference.target_ci_upper, 4), abs=1e-9
    )
    assert filtered.target_regions == ["northeast"]

    # The cohort-wide effect is reported alongside, and is the unfiltered headline.
    assert filtered.cohort_effect == pytest.approx(unfiltered.simulated_ate, abs=1e-9)
    assert filtered.cohort_ci_lower == pytest.approx(unfiltered.simulated_ci_lower, abs=1e-9)
    assert filtered.cohort_ci_upper == pytest.approx(unfiltered.simulated_ci_upper, abs=1e-9)

    # An unfiltered run is unchanged: cohort-wide headline, nothing narrowed away.
    assert unfiltered.target_regions == []
    assert unfiltered.cohort_effect is None

    # Recommendation and sizing follow the SAME estimate the headline came from.
    policy = PolicyThresholds(min_effect=SimulationEngine.DEFAULT_MIN_EFFECT_THRESHOLD)
    frame = CohortEffectDataProvider(cohort).get_training_frame(
        "email_campaign", brand="Kisqali", twin_type="hcp"
    )
    expected_n, _ = experiment_size(
        frame, reference.target_ate, regions=["northeast"], thresholds=policy
    )
    assert filtered.recommended_sample_size == expected_n
    assert filtered.recommended_sample_size != unfiltered.recommended_sample_size

    # The persisted history row carries the same numbers the response states.
    assert persisted is not None
    assert persisted.simulated_ate == pytest.approx(reference.target_ate, abs=1e-9)
    assert persisted.target_regions == ["northeast"]


@pytest.mark.unit
def test_region_filtered_simulate_refuses_a_region_the_cohort_cannot_estimate(monkeypatch):
    """#2023: a region the cohort cannot contrast is refused with an honest 422, the way the
    chat tool refuses it — never answered with the cohort-wide effect."""
    from fastapi import HTTPException

    cohort = _region_heterogeneous_cohort()
    # Twins exist in every region; drop one region from the COHORT so its effect is
    # unestimable while the twin filter still yields a population.
    cohort = cohort[cohort["region"] != "midwest"].reset_index(drop=True)

    with pytest.raises(HTTPException) as ei:
        _run_simulate(monkeypatch, cohort, regions=["midwest"])
    assert ei.value.status_code == 422
    assert "midwest" in str(ei.value.detail)


@pytest.mark.unit
def test_region_filtered_simulate_changes_the_verdict_not_only_the_number(monkeypatch):
    """#2023's user-visible symptom: the page recommended DEPLOY for a region whose own
    interval does not clear the minimum effect, because only the twins were filtered.

    The verdict, its rationale and the standard error must all come from the targeted
    estimate — an implementation that moved the ATE and CI but kept a cohort-wide DEPLOY
    would still show the wrong call to the user.
    """
    cohort = _region_heterogeneous_cohort()
    unfiltered, _ = _run_simulate(monkeypatch, cohort, regions=[])
    filtered, _ = _run_simulate(monkeypatch, cohort, regions=["midwest"])

    # midwest's own interval straddles the 0.05 minimum effect; the cohort's clears it.
    assert unfiltered.recommendation.value == "deploy"
    assert filtered.recommendation.value == "refine"

    # The rationale quotes the TARGETED bounds, so the page cannot state cohort numbers.
    assert (
        f"[{filtered.simulated_ci_lower:.3f}, {filtered.simulated_ci_upper:.3f}]"
        in filtered.recommendation_rationale
    )

    # SE is the targeted interval's half-width, not the cohort's.
    assert filtered.simulated_std_error != unfiltered.simulated_std_error
    assert filtered.simulated_std_error == pytest.approx(
        (filtered.simulated_ci_upper - filtered.simulated_ci_lower) / (2 * 1.96), abs=1e-3
    )


@pytest.mark.unit
def test_region_filtered_simulate_persists_the_targeted_numbers(monkeypatch):
    """The history row must carry the numbers the response stated, asserted through the
    REAL repository serializer rather than the in-memory result object.

    It also pins what is NOT persisted: twin_simulations has no column for the estimate's
    scope, so target_regions / cohort_* do not survive the write and a history read cannot
    tell a region-scoped row from a cohort-wide one. That gap is deliberate here (no
    migration in this lane) and is pinned so it cannot be forgotten (#2023 follow-up).
    """
    cohort = _region_heterogeneous_cohort()
    sink = {}
    filtered, _ = _run_simulate(monkeypatch, cohort, regions=["midwest"], capture=sink)

    row = sink["row"]
    assert sink["table"] == "twin_simulations"
    assert row["simulated_ate"] == pytest.approx(filtered.simulated_ate, abs=5e-5)
    assert row["simulated_ci_lower"] == pytest.approx(filtered.simulated_ci_lower, abs=5e-5)
    assert row["simulated_ci_upper"] == pytest.approx(filtered.simulated_ci_upper, abs=5e-5)
    assert row["recommendation"] == "refine"
    assert row["recommended_sample_size"] == filtered.recommended_sample_size
    # The filter that produced the scope IS recorded, even though the scope is not.
    assert row["population_filters"]["regions"] == ["midwest"]
    assert "target_regions" not in row
    assert "cohort_ate" not in row
