"""Route-level tests for the P2 heavy-compute offload (DARK by default).

Covers BOTH endpoints' two code paths:

* ``HEAVY_OFFLOAD_ENABLED`` unset/false  -> the P1 inline path runs and the
  Celery task is NEVER enqueued (dark default is inert).
* ``HEAVY_OFFLOAD_ENABLED`` true         -> the route enqueues via
  ``apply_async`` and builds the SAME response model from a stubbed
  ``AsyncResult``; a timeout maps to HTTP 408.

The heavy compute is mocked; the assertions target the wiring + response-shape
contract (no hollow asserts -- the offload path rebuilds the real response model).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException


@pytest.fixture
def _no_eager_celery():
    """Disable the autouse eager-celery mode for offload tests.

    ``tests/conftest.py`` autouse-sets ``task_always_eager=True`` so
    ``.apply_async()`` runs the task body inline. The offload-route tests must
    instead assert that the route ENQUEUES (and reads back a stubbed
    ``AsyncResult``), so we turn eager off here and patch ``apply_async``
    ourselves. Restored after the test.
    """
    from src.workers.celery_app import celery_app

    prev = celery_app.conf.task_always_eager
    celery_app.conf.task_always_eager = False
    yield
    celery_app.conf.task_always_eager = prev


# =============================================================================
# Feature-flag helper (the dark switch itself)
# =============================================================================


def test_heavy_offload_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HEAVY_OFFLOAD_ENABLED", raising=False)
    from src.api.dependencies.compute import heavy_offload_enabled

    assert heavy_offload_enabled() is False


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "", "garbage"])
def test_heavy_offload_stays_off_for_non_truthy(monkeypatch, val):
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", val)
    from src.api.dependencies.compute import heavy_offload_enabled

    assert heavy_offload_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "On"])
def test_heavy_offload_on_for_truthy(monkeypatch, val):
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", val)
    from src.api.dependencies.compute import heavy_offload_enabled

    assert heavy_offload_enabled() is True


# =============================================================================
# Digital-twin /simulate
# =============================================================================


def _real_simulation_result():
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        SimulationRecommendation,
        SimulationResult,
    )

    return SimulationResult(
        model_id=UUID(int=7),
        intervention_config=InterventionConfig(intervention_type="email_campaign"),
        twin_count=1000,
        simulated_ate=0.075,
        simulated_ci_lower=0.050,
        simulated_ci_upper=0.100,
        simulated_std_error=0.012,
        recommendation=SimulationRecommendation("deploy"),
        recommendation_rationale="Strong positive effect",
        simulation_confidence=0.92,
        execution_time_ms=250,
    )


def _twin_request():
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
    )

    return SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign", duration_weeks=8
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )


@pytest.mark.asyncio
async def test_simulate_inline_path_does_not_enqueue_when_flag_off(monkeypatch):
    """Dark default: /simulate runs inline and NEVER calls apply_async."""
    monkeypatch.delenv("HEAVY_OFFLOAD_ENABLED", raising=False)
    from src.api.routes.digital_twin import run_simulation

    real_result = _real_simulation_result()

    with (
        patch("src.digital_twin.twin_generator.TwinGenerator") as mock_gen,
        patch("src.digital_twin.simulation_engine.SimulationEngine") as mock_engine,
        patch("src.digital_twin.twin_repository.TwinRepository") as mock_repo,
        patch("src.tasks.heavy_offload_tasks.simulate_population.apply_async") as mock_apply,
    ):
        gen_instance = MagicMock()
        gen_instance.model_id = uuid4()
        gen_instance.generate.return_value = MagicMock()
        mock_gen.return_value = gen_instance

        engine_instance = MagicMock()
        engine_instance.simulate.return_value = real_result
        mock_engine.return_value = engine_instance

        repo_instance = AsyncMock()
        mock_repo.return_value = repo_instance

        result = await run_simulation(_twin_request(), {"role": "operator"})

    mock_apply.assert_not_called()  # dark default is inert
    engine_instance.simulate.assert_called_once()  # inline path ran
    assert result.simulated_ate == 0.075
    assert result.recommendation.value == "deploy"


@pytest.mark.asyncio
async def test_simulate_offload_path_enqueues_and_builds_same_response(
    monkeypatch, _no_eager_celery
):
    """Flag on: /simulate enqueues the task and rebuilds the SAME response model
    from the stubbed AsyncResult result dict -- no inline generate/simulate."""
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")
    from src.api.routes.digital_twin import run_simulation
    from src.digital_twin.simulation_runner import simulation_result_to_dict

    result_dict = simulation_result_to_dict(_real_simulation_result())

    fake_async = MagicMock()
    fake_async.ready.return_value = True
    fake_async.successful.return_value = True
    fake_async.result = result_dict

    with (
        patch("src.digital_twin.twin_generator.TwinGenerator") as mock_gen,
        patch("src.digital_twin.simulation_engine.SimulationEngine") as mock_engine,
        patch("src.digital_twin.twin_repository.TwinRepository") as mock_repo,
        patch(
            "src.tasks.heavy_offload_tasks.simulate_population.apply_async",
            return_value=fake_async,
        ) as mock_apply,
    ):
        gen_instance = MagicMock()
        gen_instance.model_id = uuid4()
        mock_gen.return_value = gen_instance
        engine_instance = MagicMock()
        mock_engine.return_value = engine_instance
        repo_instance = AsyncMock()
        mock_repo.return_value = repo_instance

        result = await run_simulation(_twin_request(), {"role": "operator"})

    mock_apply.assert_called_once()
    # routed to the twins queue
    assert mock_apply.call_args.kwargs.get("queue") == "twins"
    # inline heavy compute must NOT have run on the offload path
    engine_instance.simulate.assert_not_called()
    gen_instance.generate.assert_not_called()
    # SAME response shape as the inline path
    assert result.simulated_ate == 0.075
    assert result.recommendation.value == "deploy"
    assert result.is_significant is True
    assert result.effect_direction == "positive"
    # result still persisted (save happens after compute on both paths)
    repo_instance.save_simulation.assert_awaited_once()


@pytest.mark.asyncio
async def test_simulate_offload_timeout_returns_408(monkeypatch, _no_eager_celery):
    """Flag on: a task that never readies maps to HTTP 408 (synchronous contract)."""
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")
    import src.api.routes.digital_twin as dt_mod
    from src.api.routes.digital_twin import run_simulation

    fake_async = MagicMock()
    fake_async.ready.return_value = False  # never completes

    with (
        patch("src.digital_twin.twin_generator.TwinGenerator") as mock_gen,
        patch("src.digital_twin.simulation_engine.SimulationEngine"),
        patch("src.digital_twin.twin_repository.TwinRepository"),
        patch(
            "src.tasks.heavy_offload_tasks.simulate_population.apply_async",
            return_value=fake_async,
        ),
        patch.object(dt_mod, "_OFFLOAD_TIMEOUT_SECONDS", 0.05),
    ):
        gen_instance = MagicMock()
        gen_instance.model_id = uuid4()
        mock_gen.return_value = gen_instance

        with pytest.raises(HTTPException) as exc:
            await run_simulation(_twin_request(), {"role": "operator"})

    assert exc.value.status_code == 408


# =============================================================================
# SHAP compute_shap (covers /explain/predict and /predict/batch)
# =============================================================================


def _shap_service():
    from src.api.routes.explain import RealTimeSHAPService

    explainer = AsyncMock()
    explainer.compute_shap_values = AsyncMock(
        return_value=MagicMock(
            shap_values={"f1": 0.2, "f2": -0.1},
            base_value=0.4,
            explainer_type=MagicMock(value="TreeExplainer"),
            computation_time_ms=10.0,
        )
    )
    svc = RealTimeSHAPService(
        bentoml_client=AsyncMock(),
        shap_explainer=explainer,
        shap_repo=MagicMock(),
        feast_client=AsyncMock(),
    )
    svc._initialized = True
    return svc, explainer


@pytest.mark.asyncio
async def test_compute_shap_inline_when_flag_off(monkeypatch):
    """Dark default: compute_shap uses the in-process explainer, no apply_async."""
    monkeypatch.delenv("HEAVY_OFFLOAD_ENABLED", raising=False)
    from src.api.routes.explain import ModelType

    svc, explainer = _shap_service()

    with patch("src.tasks.heavy_offload_tasks.compute_shap_values.apply_async") as mock_apply:
        out = await svc.compute_shap(
            features={"f1": 1.0, "f2": 2.0},
            model_type=ModelType.PROPENSITY,
            model_version_id="v1",
            top_k=5,
        )

    mock_apply.assert_not_called()
    explainer.compute_shap_values.assert_awaited_once()
    assert out["base_value"] == 0.4
    assert out["explainer_type"] == "TreeExplainer"
    assert {c.feature_name for c in out["contributions"]} == {"f1", "f2"}
    assert out["shap_sum"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_compute_shap_offload_enqueues_and_same_shape(monkeypatch, _no_eager_celery):
    """Flag on: compute_shap enqueues on the shap queue and builds the SAME dict
    from the stubbed AsyncResult; the in-process explainer is NOT used."""
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")
    from src.api.routes.explain import ModelType

    svc, explainer = _shap_service()

    fake_async = MagicMock()
    fake_async.ready.return_value = True
    fake_async.successful.return_value = True
    fake_async.result = {
        "base_value": 0.4,
        "shap_values": {"f1": 0.2, "f2": -0.1},
        "explainer_type": "TreeExplainer",
        "computation_time_ms": 10.0,
    }

    with patch(
        "src.tasks.heavy_offload_tasks.compute_shap_values.apply_async",
        return_value=fake_async,
    ) as mock_apply:
        out = await svc.compute_shap(
            features={"f1": 1.0, "f2": 2.0},
            model_type=ModelType.PROPENSITY,
            model_version_id="v1",
            top_k=5,
        )

    mock_apply.assert_called_once()
    assert mock_apply.call_args.kwargs.get("queue") == "shap"
    explainer.compute_shap_values.assert_not_awaited()  # offloaded, not inline
    # SAME response shape as inline
    assert out["base_value"] == 0.4
    assert out["explainer_type"] == "TreeExplainer"
    assert {c.feature_name for c in out["contributions"]} == {"f1", "f2"}
    assert out["shap_sum"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_compute_shap_offload_timeout_returns_408(monkeypatch, _no_eager_celery):
    """Flag on: a task that never readies maps to HTTP 408."""
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")
    import src.api.routes.explain as explain_mod
    from src.api.routes.explain import ModelType

    svc, _ = _shap_service()

    fake_async = MagicMock()
    fake_async.ready.return_value = False

    with (
        patch(
            "src.tasks.heavy_offload_tasks.compute_shap_values.apply_async",
            return_value=fake_async,
        ),
        patch.object(explain_mod, "_SHAP_OFFLOAD_TIMEOUT_SECONDS", 0.05),
    ):
        with pytest.raises(HTTPException) as exc:
            await svc.compute_shap(
                features={"f1": 1.0},
                model_type=ModelType.PROPENSITY,
                model_version_id="v1",
                top_k=5,
            )

    assert exc.value.status_code == 408
