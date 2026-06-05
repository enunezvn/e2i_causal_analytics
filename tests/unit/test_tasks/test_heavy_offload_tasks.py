"""Unit tests for the P2 heavy-compute offload Celery tasks.

Covers:
* The task names match the pre-existing ``task_routes`` entries (so routing to
  ``worker_heavy`` works without touching the routing table).
* Each task returns a JSON-SERIALIZABLE scalar dict with the REAL extraction
  shape (asserted against real ``SHAPResult`` / ``SimulationResult`` models, not
  hollow stubs).
* The shared runners serialize/round-trip correctly so the inline and offloaded
  API paths produce an identical response.

The heavy libraries (the SHAP explainer, the twin generator/engine) are mocked
where a real run would be too slow, but the serialization boundary under test is
exercised with the REAL pydantic/dataclass models.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from src.workers.celery_app import celery_app

# =============================================================================
# Routing: task names match the pre-existing task_routes entries
# =============================================================================


def _route_queue(task_name: str) -> str:
    route = celery_app.conf.task_routes.get(task_name)
    assert route is not None, f"no task_routes entry for {task_name}"
    return route["queue"]


def test_task_names_match_existing_routes():
    """The authored tasks must carry the exact names the routing table expects."""
    from src.tasks.heavy_offload_tasks import compute_shap_values, simulate_population

    assert compute_shap_values.name == "src.tasks.compute_shap_values"
    assert simulate_population.name == "src.tasks.simulate_population"
    assert _route_queue(compute_shap_values.name) == "shap"
    assert _route_queue(simulate_population.name) == "twins"


def test_tasks_are_registered_in_the_app():
    """Importing src.tasks must register both tasks for worker_heavy discovery."""
    import src.tasks  # noqa: F401  (fires the registration imports)

    assert "src.tasks.compute_shap_values" in celery_app.tasks
    assert "src.tasks.simulate_population" in celery_app.tasks
    assert "src.tasks.train_twin_model" in celery_app.tasks


def test_train_twin_model_task_name_and_route():
    """The offline twin-training task must carry the name its ml-queue route
    (celery_app.py) already expects (#705 H4)."""
    from src.tasks.heavy_offload_tasks import train_twin_model

    assert train_twin_model.name == "src.tasks.train_twin_model"
    assert _route_queue(train_twin_model.name) == "ml"


def test_train_twin_model_forwards_to_training_job():
    """The task must build a real repo and forward the payload to the training job."""
    from src.tasks.heavy_offload_tasks import train_twin_model

    payload = {
        "twin_type": "hcp",
        "brand": "Remibrutinib",
        "synthetic": True,
        "n_rows": 1100,
        "seed": 1,
    }
    with (
        patch(
            "src.digital_twin.training_job.train_and_persist_twin",
            new=AsyncMock(return_value={"model_id": "m1", "data_provenance": "synthetic"}),
        ) as mock_train,
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("src.digital_twin.twin_repository.TwinRepository", MagicMock()),
    ):
        out = train_twin_model.apply(args=[payload]).get()

    assert out["model_id"] == "m1"
    mock_train.assert_awaited_once()
    kwargs = mock_train.await_args.kwargs
    assert kwargs["synthetic"] is True
    assert kwargs["n_rows"] == 1100


# =============================================================================
# SHAP task: returns the real JSON-safe SHAP dict
# =============================================================================


def test_compute_shap_values_returns_json_safe_scalar_dict():
    """The SHAP task returns exactly the four scalar keys the route consumes,
    and the result is JSON-serializable (Celery result serializer = json)."""
    from src.mlops.shap_explainer_realtime import ExplainerType, SHAPResult
    from src.tasks.heavy_offload_tasks import compute_shap_values

    real_result = SHAPResult(
        shap_values={"days_since_last_hcp_visit": 0.15, "adherence": -0.05},
        base_value=0.42,
        expected_value=0.42,
        computation_time_ms=120.5,
        explainer_type=ExplainerType.TREE,
        feature_count=2,
        model_version_id="v2.3.1-prod",
    )

    # Mock only the (slow) explainer call; the runner's REAL extraction +
    # serialization shape is what we assert. The patched method replaces the
    # class attribute, so it is called bound (``self`` first) — accept + ignore it.
    async def _fake_compute(_self, **_kwargs):
        return real_result

    with patch(
        "src.mlops.shap_explainer_realtime.RealTimeSHAPExplainer.compute_shap_values",
        new=_fake_compute,
    ):
        payload = {
            "features": {"days_since_last_hcp_visit": 45.0, "adherence": 0.7},
            "model_type": "propensity",
            "model_version_id": "v2.3.1-prod",
            "top_k": 5,
        }
        eager = compute_shap_values.apply(args=[payload])

    assert eager.successful()
    out = eager.result
    # JSON round-trip must not raise (json serializer requirement).
    json.loads(json.dumps(out))
    assert set(out.keys()) == {
        "base_value",
        "shap_values",
        "explainer_type",
        "computation_time_ms",
    }
    assert out["base_value"] == pytest.approx(0.42)
    assert out["shap_values"] == {
        "days_since_last_hcp_visit": pytest.approx(0.15),
        "adherence": pytest.approx(-0.05),
    }
    assert out["explainer_type"] == "TreeExplainer"  # enum -> .value string
    assert out["computation_time_ms"] == pytest.approx(120.5)
    # No non-JSON types leaked through.
    assert all(isinstance(v, float) for v in out["shap_values"].values())


# =============================================================================
# Twin task: returns the real round-trippable SimulationResult dict
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


def test_simulate_population_returns_json_safe_roundtrippable_dict():
    """The twin task returns a JSON-safe dict that rebuilds into the SAME
    SimulationResult (so the route's response extraction is byte-identical)."""
    from src.digital_twin.simulation_runner import simulation_result_from_dict
    from src.tasks.heavy_offload_tasks import simulate_population

    real_result = _real_simulation_result()

    # Mock only the heavy generate+simulate compute; assert the REAL pydantic
    # model_dump(mode="json") serialization shape.
    with patch(
        "src.digital_twin.simulation_runner.run_simulation_compute",
        return_value=real_result,
    ):
        payload = {
            "twin_type_value": "hcp",
            "brand_value": "Remibrutinib",
            "twin_count": 1000,
            "intervention_dict": {"intervention_type": "email_campaign"},
            "population_filter_dict": None,
            "calculate_heterogeneity": True,
            "model_id_value": str(UUID(int=7)),
        }
        eager = simulate_population.apply(args=[payload])

    assert eager.successful()
    out = eager.result
    # JSON round-trip must not raise.
    json.loads(json.dumps(out))
    # The dict rebuilds into an equivalent SimulationResult with working methods.
    rebuilt = simulation_result_from_dict(out)
    assert rebuilt.simulated_ate == pytest.approx(0.075)
    assert rebuilt.twin_count == 1000
    assert rebuilt.recommendation.value == "deploy"
    assert rebuilt.is_significant() is True
    assert rebuilt.effect_direction() == "positive"
    assert str(rebuilt.model_id) == str(UUID(int=7))


def test_simulate_population_passes_payload_through_to_compute():
    """The task must forward the JSON payload fields to the shared compute helper
    unchanged (no silent dropping of filter/heterogeneity/model_id)."""
    from src.tasks.heavy_offload_tasks import simulate_population

    real_result = _real_simulation_result()
    with patch(
        "src.digital_twin.simulation_runner.run_simulation_compute",
        return_value=real_result,
    ) as mock_compute:
        payload = {
            "twin_type_value": "hcp",
            "brand_value": "Remibrutinib",
            "twin_count": 2500,
            "intervention_dict": {"intervention_type": "call_frequency_increase"},
            "population_filter_dict": {"deciles": [1, 2, 3]},
            "calculate_heterogeneity": False,
            "model_id_value": str(UUID(int=9)),
        }
        simulate_population.apply(args=[payload])

    mock_compute.assert_called_once()
    kwargs = mock_compute.call_args.kwargs
    assert kwargs["twin_type_value"] == "hcp"
    assert kwargs["twin_count"] == 2500
    assert kwargs["intervention_dict"] == {"intervention_type": "call_frequency_increase"}
    assert kwargs["population_filter_dict"] == {"deciles": [1, 2, 3]}
    assert kwargs["calculate_heterogeneity"] is False
    assert kwargs["model_id_value"] == str(UUID(int=9))


def test_run_simulation_compute_fails_closed_without_loadable_model():
    """Worker-side fail-closed guard (#705 H4): with no loadable model the shared
    compute raises RuntimeError and never generates from an untrained generator.
    This exercises the REAL guard (unmocked) — the offload route tests stub the
    function it should exercise."""
    from src.digital_twin.simulation_runner import run_simulation_compute

    with pytest.raises(RuntimeError):
        run_simulation_compute(
            twin_type_value="hcp",
            brand_value="Remibrutinib",
            twin_count=10,
            intervention_dict={"intervention_type": "email_campaign"},
            population_filter_dict=None,
            calculate_heterogeneity=False,
            model_id_value=None,
            model_uri=None,
            model_run_id=None,
        )
