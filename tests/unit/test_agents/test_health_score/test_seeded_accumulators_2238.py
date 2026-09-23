"""#2238: health score nodes must not echo the ``operator.add`` ``errors`` channel.

``HealthScoreState`` declares ``errors`` as ``Annotated[List, operator.add]``
(``warnings`` there is a plain field). The five business nodes returned
``{**state, ...}``, and the composer additionally re-emitted
``"errors": state.get("errors")`` as a "contract-required field" — LangGraph
already materialises a reducer channel as ``[]`` when nothing writes it, so
that re-emit only doubled.

Measured before the fix (compiled graph, no stores wired): full graph seeded
error x32 (2^5 nodes), quick graph x4; with broken stores every node's own
error x2.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from src.agents.health_score.graph import build_health_score_graph, build_quick_check_graph

SEED_E = {"node": "seed", "error": "seed-error"}


def _base_state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "query": "",
        "check_scope": "full",
        "component_statuses": None,
        "component_health_score": None,
        "model_metrics": None,
        "model_health_score": None,
        "pipeline_statuses": None,
        "pipeline_health_score": None,
        "agent_statuses": None,
        "agent_health_score": None,
        "overall_health_score": None,
        "health_grade": None,
        "critical_issues": None,
        "warnings": None,
        "health_summary": None,
        "total_latency_ms": 0,
        "timestamp": "",
        "errors": [SEED_E],
        "status": "pending",
    }
    state.update(overrides)
    return state


class _BrokenStore:
    """Boundary stub: every backend read raises, so each node takes its
    error branch and appends its own error row."""

    async def get_active_models(self):
        raise RuntimeError("store down")

    async def get_all_pipelines(self):
        raise RuntimeError("store down")

    async def get_all_agents(self):
        raise RuntimeError("registry down")


@pytest.fixture
def no_audit_service():
    with patch("src.agents.base.audit_chain_mixin.get_audit_chain_service", return_value=None) as p:
        yield p


@pytest.mark.asyncio
async def test_full_graph_no_stores_seed_comes_out_once(no_audit_service):
    final = await build_health_score_graph().ainvoke(_base_state())

    assert final["status"] == "completed"
    assert final["errors"] == [SEED_E]


@pytest.mark.asyncio
async def test_quick_graph_seed_comes_out_once(no_audit_service):
    final = await build_quick_check_graph().ainvoke(_base_state(check_scope="quick"))

    assert final["status"] == "completed"
    assert final["errors"] == [SEED_E]


@pytest.mark.asyncio
async def test_broken_stores_each_node_error_once(no_audit_service):
    graph = build_health_score_graph(
        metrics_store=_BrokenStore(), pipeline_store=_BrokenStore(), agent_registry=_BrokenStore()
    )
    final = await graph.ainvoke(_base_state())

    assert final["status"] == "completed"
    nodes = [e.get("node") for e in final["errors"]]
    assert nodes.count("seed") == 1, nodes
    for node in ("model_health", "pipeline_health", "agent_health"):
        assert nodes.count(node) == 1, nodes
