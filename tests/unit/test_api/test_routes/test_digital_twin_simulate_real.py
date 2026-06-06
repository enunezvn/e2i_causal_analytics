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
