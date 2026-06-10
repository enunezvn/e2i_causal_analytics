"""Orchestrator registry-gating regression (#840, codex HIGH-2).

When a model_registry IS wired but resolves NO model for the requested target,
the orchestrator must FAIL CLOSED — it must NOT fall back to running every
loaded manifest client, which would serve another target's models (a
target-agnostic fabrication). The legacy no-registry path still falls back to
the loaded clients.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.prediction_synthesizer.nodes.model_orchestrator import ModelOrchestratorNode


class _Client:
    """Minimal real async prediction client."""

    def __init__(self, prediction: float = 0.7):
        self._p = prediction

    async def predict(
        self, entity_id: str, features: Dict[str, Any], time_horizon: str
    ) -> Dict[str, Any]:
        return {"model_type": "logistic_regression", "prediction": self._p, "confidence": 0.8}


class _ErroringClient:
    """A client that swallows an inference failure into a synthetic neutral
    result (the InProcessModelClient on-error contract)."""

    async def predict(
        self, entity_id: str, features: Dict[str, Any], time_horizon: str
    ) -> Dict[str, Any]:
        return {
            "model_type": "logistic_regression",
            "prediction": 0.5,
            "confidence": 0.3,
            "error": "boom",
        }


class _EmptyRegistry:
    async def get_models_for_target(self, target: str, entity_type: str) -> List[str]:
        return []


class _ResolvingRegistry:
    def __init__(self, names: List[str]):
        self._names = names

    async def get_models_for_target(self, target: str, entity_type: str) -> List[str]:
        return list(self._names)


@pytest.mark.asyncio
async def test_registry_present_but_empty_fails_closed(base_state):
    """csu-only clients loaded, but registry resolves nothing for the requested
    target -> status=failed, NOT a prediction from the wrong target's clients."""
    node = ModelOrchestratorNode(
        model_registry=_EmptyRegistry(),
        model_clients={"csu_model_a": _Client(0.7), "csu_model_b": _Client(0.6)},
    )
    state = {**base_state, "prediction_target": "pnh_persistence", "models_to_use": None}
    result = await node.execute(state)
    assert result["status"] == "failed"
    assert "No models available" in result["errors"][0]["error"]


@pytest.mark.asyncio
async def test_registry_resolves_models_and_runs_them(base_state):
    """Registry resolves models that exist in clients -> they run -> success."""
    node = ModelOrchestratorNode(
        model_registry=_ResolvingRegistry(["m1", "m2"]),
        model_clients={"m1": _Client(0.6), "m2": _Client(0.7)},
    )
    result = await node.execute({**base_state, "models_to_use": None})
    assert result["status"] == "combining"
    assert result["models_succeeded"] == 2


@pytest.mark.asyncio
async def test_no_registry_legacy_fallback_to_clients(base_state):
    """No registry wired -> legacy fallback to loaded clients is preserved."""
    node = ModelOrchestratorNode(
        model_registry=None,
        model_clients={"m1": _Client(0.6)},
    )
    result = await node.execute({**base_state, "models_to_use": None})
    assert result["status"] == "combining"
    assert result["models_succeeded"] == 1


@pytest.mark.asyncio
async def test_explicit_override_narrows_within_registry_approval(base_state):
    """An explicit models_to_use may only NARROW within registry-approved models;
    names not approved for the target are dropped."""
    node = ModelOrchestratorNode(
        model_registry=_ResolvingRegistry(["m1", "m2"]),
        model_clients={"m1": _Client(0.6), "m2": _Client(0.7)},
    )
    # override asks for m1 + a non-approved name -> only m1 runs
    result = await node.execute({**base_state, "models_to_use": ["m1", "not_approved"]})
    assert result["status"] == "combining"
    assert result["models_succeeded"] == 1


@pytest.mark.asyncio
async def test_explicit_override_for_unapproved_target_fails_closed(base_state):
    """The dispatcher forwards models_to_use from external params (#840 HIGH):
    naming a loaded client for a target the registry does not approve must FAIL
    CLOSED, not run that client as a target-agnostic prediction."""
    node = ModelOrchestratorNode(
        model_registry=_EmptyRegistry(),  # no deployable model for this target
        model_clients={"csu_model_a": _Client(0.7)},
    )
    state = {
        **base_state,
        "prediction_target": "nonexistent_target",
        "models_to_use": ["csu_model_a"],
    }
    result = await node.execute(state)
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_duplicate_override_is_deduplicated(base_state):
    """A repeated model_id in the override must NOT be scheduled twice — that
    would let the combiner count one model as ensemble diversity and bypass the
    single-model confidence cap (fabricated diversity)."""
    node = ModelOrchestratorNode(
        model_registry=_ResolvingRegistry(["m1", "m2"]),
        model_clients={"m1": _Client(0.6), "m2": _Client(0.7)},
    )
    result = await node.execute({**base_state, "models_to_use": ["m1", "m1"]})
    assert result["status"] == "combining"
    assert result["models_succeeded"] == 1, "duplicate model_id must run once"


@pytest.mark.asyncio
async def test_error_flagged_prediction_is_not_counted_success(base_state):
    """A client that returns an error-flagged synthetic 0.5 (swallowed inference
    failure) must be counted as FAILED, not a real ensemble member — otherwise a
    broken model fabricates a prediction."""
    node = ModelOrchestratorNode(
        model_registry=_ResolvingRegistry(["m1", "m2"]),
        model_clients={"m1": _ErroringClient(), "m2": _ErroringClient()},
    )
    result = await node.execute({**base_state, "models_to_use": None})
    assert result["status"] == "failed"
    assert result["models_succeeded"] == 0
