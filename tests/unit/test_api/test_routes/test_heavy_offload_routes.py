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


def _active_model_row():
    """A persisted active-model row /simulate resolves before generating (#705 H4)."""
    return {
        "model_id": str(uuid4()),
        "twin_type": "hcp",
        "brand": "Remibrutinib",
        "is_active": True,
        "mlflow_model_uri": "models:/m-test",
        "mlflow_run_id": "run-test",
    }


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
        patch("src.workers.celery_app.celery_app.send_task") as mock_apply,
        patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=True),
        # /simulate resolves the repo via _get_twin_repo -> get_async_supabase_client
        # FIRST (#705 H6); without this the real client raises ServiceConnectionError
        # in keyless CI -> caught -> 500. Mirrors test_digital_twin.mock_twin_repository.
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        # Direction-2 identification gate must PASS so the inline path runs (the engine
        # is mocked; the real estimate is exercised in the effect/ unit tests).
        patch(
            "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        gen_instance = MagicMock()
        gen_instance.model_id = uuid4()
        gen_instance.generate.return_value = MagicMock()
        mock_gen.return_value = gen_instance

        engine_instance = MagicMock()
        engine_instance.simulate.return_value = real_result
        mock_engine.return_value = engine_instance

        repo_instance = AsyncMock()
        repo_instance.list_active_models.return_value = [_active_model_row()]
        mock_repo.return_value = repo_instance

        result = await run_simulation(_twin_request(), {"role": "operator"})

    mock_apply.assert_not_called()  # dark default is inert
    engine_instance.simulate.assert_called_once()  # inline path ran
    assert result.simulated_ate == 0.075
    assert result.recommendation.value == "deploy"


@pytest.mark.asyncio
async def test_simulate_offload_path_returns_503_not_supported(monkeypatch):
    """Flag on: /simulate's offload path is NOT wired for cohort-causal estimation, so it
    fails closed (503) rather than enqueuing a worker that would fabricate a synthetic
    effect. The Direction-2 identification gate passes here (provider mocked non-None);
    the offload guard then rejects without enqueuing."""
    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")
    from src.api.routes.digital_twin import run_simulation

    with (
        patch("src.digital_twin.twin_repository.TwinRepository") as mock_repo,
        patch("src.workers.celery_app.celery_app.send_task") as mock_apply,
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        repo_instance = AsyncMock()
        repo_instance.list_active_models.return_value = [_active_model_row()]
        mock_repo.return_value = repo_instance

        with pytest.raises(HTTPException) as exc:
            await run_simulation(_twin_request(), {"role": "operator"})

    assert exc.value.status_code == 503
    mock_apply.assert_not_called()  # fails closed — never enqueues a fabricating worker


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

    with patch("src.workers.celery_app.celery_app.send_task") as mock_apply:
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
        "src.workers.celery_app.celery_app.send_task",
        return_value=fake_async,
    ) as mock_apply:
        out = await svc.compute_shap(
            features={"f1": 1.0, "f2": 2.0},
            model_type=ModelType.PROPENSITY,
            model_version_id="v1",
            top_k=5,
        )

    mock_apply.assert_called_once()
    # FIX 2: enqueued by registered task NAME via send_task (NOT a task-object
    # import, which would pull sklearn into the API process via src.tasks.__init__)
    assert mock_apply.call_args.args[0] == "src.tasks.compute_shap_values"
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
            "src.workers.celery_app.celery_app.send_task",
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


# =============================================================================
# FIX 2 — routes must NOT import the heavy task package on the offload path.
# src/tasks/__init__ imports ab_testing_tasks -> TwinGenerator (sklearn); a
# `from src.tasks.heavy_offload_tasks import ...` would pull heavy ML libs into
# the API process, defeating P2's memory isolation. The routes must enqueue by
# registered task NAME via celery_app.send_task instead.
# =============================================================================


def test_explain_route_does_not_import_heavy_task_package():
    from src.api.routes import explain

    text = open(explain.__file__).read()
    assert "from src.tasks.heavy_offload_tasks import" not in text, (
        "explain route must enqueue via celery_app.send_task, not import the heavy task package"
    )
    assert "send_task(" in text and "src.tasks.compute_shap_values" in text, (
        "explain route must enqueue the SHAP task by registered name via send_task"
    )


def test_digital_twin_route_does_not_import_heavy_task_package():
    from src.api.routes import digital_twin

    text = open(digital_twin.__file__).read()
    assert "from src.tasks.heavy_offload_tasks import" not in text, (
        "twin route must not import the heavy task package into the API process"
    )
    # Direction 2: the offload path is NOT wired for cohort-causal estimation — the route
    # fails closed (503) instead of enqueuing a worker. So it must make NO send_task call
    # (robust to comments that may mention the task name).
    assert "send_task(" not in text, (
        "twin /simulate must not enqueue a worker on the cohort-causal path (fails closed)"
    )


# =============================================================================
# FIX 1 + FIX 3 — explain_prediction endpoint behavior.
#
# FIX 1: a 408 bubbling up from service.compute_shap (the offload timeout) must
#        propagate as 408, not be re-wrapped into a generic 500.
# FIX 3: the reject-fast heavy_compute_slot is held ONLY on the inline
#        (flag-off / DARK default) path; on the flag-on offload path it must NOT
#        be held (the heavy SHAP runs on worker_heavy, so holding the API slot
#        would needlessly 503 concurrent requests).
# =============================================================================


def _explain_service_for_route():
    """A RealTimeSHAPService whose feature/predict deps are stubbed so the route
    reaches the compute_shap step."""
    from src.api.routes.explain import RealTimeSHAPService

    svc = RealTimeSHAPService(
        bentoml_client=AsyncMock(),
        shap_explainer=AsyncMock(),
        shap_repo=MagicMock(),
        feast_client=AsyncMock(),
    )
    svc._initialized = True
    svc.get_prediction = AsyncMock(
        return_value={
            "model_version_id": "v1",
            "prediction_class": "adopter",
            "prediction_probability": 0.9,
            # The route now feeds compute_shap/audit the canonical, strictly-
            # validated numeric feature dict that get_prediction resolves from
            # /model_info (not the raw request) and fail-closes (500) if it is
            # absent. Provide it so these tests still exercise the compute_shap /
            # slot behavior they actually guard, rather than tripping the
            # internal-contract guard before reaching the heavy body.
            "model_features": {"f1": 1.0},
        }
    )
    return svc


def _explain_request():
    from src.api.routes.explain import ExplainRequest, ModelType

    return ExplainRequest(
        patient_id="P-1",
        model_type=ModelType.PROPENSITY,
        features={"f1": 1.0},
    )


@pytest.mark.asyncio
async def test_explain_prediction_propagates_408_not_500(monkeypatch):
    """FIX 1: a 408 raised by service.compute_shap (the offload timeout) must surface from
    the explain_prediction endpoint as 408, NOT be re-wrapped into a generic 500 by the
    endpoint's outer ``except Exception`` handler."""
    from fastapi import BackgroundTasks

    from src.api.routes import explain

    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "true")  # offload path; nullcontext slot

    svc = _explain_service_for_route()

    async def _raise_408(**kwargs):
        raise HTTPException(status_code=408, detail="SHAP computation timed out; retry shortly.")

    svc.compute_shap = _raise_408

    async def _get_svc():
        return svc

    monkeypatch.setattr(explain, "get_shap_service", _get_svc)

    with pytest.raises(HTTPException) as exc:
        await explain.explain_prediction(_explain_request(), BackgroundTasks(), user={"sub": "t"})

    assert exc.value.status_code == 408, "408 offload timeout must propagate, not become 500"


def _saturate_slot():
    import src.api.dependencies.compute as compute

    compute._reset_limiter_cache_for_tests()
    compute.get_heavy_compute_limiter().acquire()  # consume the single slot


@pytest.mark.asyncio
async def test_explain_prediction_holds_slot_when_flag_off(monkeypatch):
    """FIX 3 (DARK default): a saturated slot must 503 (HeavyComputeSaturated) here, and
    the heavy compute_shap must NOT run (the slot rejects before the body)."""
    from fastapi import BackgroundTasks

    from src.api.dependencies.compute import HeavyComputeSaturated
    from src.api.routes import explain

    monkeypatch.delenv("HEAVY_OFFLOAD_ENABLED", raising=False)

    svc = _explain_service_for_route()
    compute_shap_ran = {"v": False}

    async def _spy_compute_shap(**kwargs):
        compute_shap_ran["v"] = True
        return {}

    svc.compute_shap = _spy_compute_shap

    async def _get_svc():
        return svc

    monkeypatch.setattr(explain, "get_shap_service", _get_svc)

    _saturate_slot()

    with pytest.raises(HeavyComputeSaturated):
        await explain.explain_prediction(_explain_request(), BackgroundTasks(), user={"sub": "t"})

    assert compute_shap_ran["v"] is False, "the heavy body must not run when the slot rejects"


@pytest.mark.asyncio
async def test_explain_prediction_does_not_require_slot_when_flag_on(monkeypatch):
    """FIX 3 (flag on): a saturated slot must NOT block; the request reaches the heavy
    body (compute_shap) because the offload path uses nullcontext, not the slot."""
    from fastapi import BackgroundTasks

    from src.api.dependencies.compute import HeavyComputeSaturated
    from src.api.routes import explain

    monkeypatch.setenv("HEAVY_OFFLOAD_ENABLED", "1")

    svc = _explain_service_for_route()
    compute_shap_ran = {"v": False}

    async def _spy_compute_shap(**kwargs):
        compute_shap_ran["v"] = True
        return {
            "base_value": 0.4,
            "contributions": [],
            "shap_sum": 0.0,
            "explainer_type": "TreeExplainer",
            "computation_time_ms": 10.0,
        }

    svc.compute_shap = _spy_compute_shap

    async def _get_svc():
        return svc

    monkeypatch.setattr(explain, "get_shap_service", _get_svc)

    _saturate_slot()  # slot is full; with the flag on this must be irrelevant

    try:
        await explain.explain_prediction(_explain_request(), BackgroundTasks(), user={"sub": "t"})
    except HeavyComputeSaturated:  # pragma: no cover - the regression we guard against
        raise AssertionError("flag-on offload path must NOT acquire the reject-fast slot")
    except HTTPException:
        pass  # response-model shaping may fail on the empty stub; not what we assert here

    assert compute_shap_ran["v"] is True, "flag-on path must reach the heavy body (slot not held)"
