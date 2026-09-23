"""#2238: prediction synthesizer nodes must not echo the ``operator.add`` channels.

``PredictionSynthesizerState`` declares ``errors`` and ``warnings`` as
``Annotated[List, operator.add]``. LangGraph APPENDS whatever a node returns
for such a channel, so a node returning ``{**state, ...}`` (or ``state``)
re-submits everything already accumulated and every entry doubles per node.

Measured before the fix (compiled graph, seeded ``warnings=["seed"]``,
``errors=[seed]``): failure path seed warning x4, the orchestrator's own
"No models available" error x2; success path seed error x8.

Each test drives the COMPILED graph with the audit service absent (no rows)
and model clients stubbed at the boundary only.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from src.agents.prediction_synthesizer.graph import (
    build_prediction_synthesizer_graph,
    build_simple_prediction_graph,
)

SEED_W = "seed-warning"
SEED_E = {"node": "seed", "error": "seed-error"}


def _base_state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "query": "What is the churn risk?",
        "entity_id": "hcp_1",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "features": {"f": 1.0},
        "time_horizon": "30d",
        "models_to_use": None,
        "ensemble_method": "weighted",
        "confidence_level": 0.95,
        "include_context": False,
        "individual_predictions": None,
        "models_succeeded": 0,
        "models_failed": 0,
        "ensemble_prediction": None,
        "prediction_summary": None,
        "prediction_context": None,
        "orchestration_latency_ms": 0,
        "ensemble_latency_ms": 0,
        "total_latency_ms": 0,
        "timestamp": "",
        "errors": [SEED_E],
        "warnings": [SEED_W],
        "status": "pending",
    }
    state.update(overrides)
    return state


class _StubClient:
    """Boundary stub for a model client: a fixed prediction, no I/O."""

    def __init__(self, value: float):
        self.value = value

    async def predict(self, entity_id: str, features: Dict[str, Any], time_horizon: str):
        return {"prediction": self.value, "confidence": 0.8}


@pytest.fixture
def no_audit_service():
    with patch("src.agents.base.audit_chain_mixin.get_audit_chain_service", return_value=None) as p:
        yield p


@pytest.mark.asyncio
async def test_failure_path_seeds_and_own_error_come_out_once(no_audit_service):
    """orchestrate (no models) -> error_handler: the seeded warning/error and the
    orchestrator's own error must each appear exactly once."""
    graph = build_simple_prediction_graph(model_clients={})
    final = await graph.ainvoke(_base_state())

    assert final["status"] == "failed"
    assert final["warnings"] == [SEED_W]
    assert final["errors"].count(SEED_E) == 1
    own = [e for e in final["errors"] if e.get("node") == "orchestrator"]
    assert len(own) == 1, final["errors"]


@pytest.mark.asyncio
async def test_success_path_without_context_seeds_come_out_once(no_audit_service):
    """orchestrate -> combine -> enrich(skip): seeds survive each node once."""
    graph = build_prediction_synthesizer_graph(
        model_clients={"m1": _StubClient(0.4), "m2": _StubClient(0.6)}
    )
    final = await graph.ainvoke(_base_state())

    assert final["status"] == "completed"
    assert final["warnings"] == [SEED_W]
    assert final["errors"] == [SEED_E]


@pytest.mark.asyncio
async def test_success_path_with_context_seeds_and_per_dep_warnings_once(no_audit_service):
    """orchestrate -> combine -> enrich(no stores): the seeds and every per-dep
    enrichment warning appear once; the enrichment total-failure error once."""
    graph = build_prediction_synthesizer_graph(
        model_clients={"m1": _StubClient(0.4), "m2": _StubClient(0.6)}
    )
    final = await graph.ainvoke(_base_state(include_context=True))

    assert final["warnings"].count(SEED_W) == 1
    assert len(final["warnings"]) == len(set(final["warnings"])), final["warnings"]
    assert final["errors"].count(SEED_E) == 1
    total_failures = [
        e for e in final["errors"] if e.get("code") == "context_enrichment_total_failure"
    ]
    assert len(total_failures) == 1, final["errors"]
