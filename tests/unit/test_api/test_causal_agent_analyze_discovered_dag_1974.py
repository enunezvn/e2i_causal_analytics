"""agent-analyze response carries the persisted DAG id and any persist warning (#1974).

``_agent_state_to_response`` maps the causal_impact final state onto the API
response. The graph_builder node now returns ``discovered_dag_id`` (success)
or ``discovered_dag_persist_error`` + a ``warnings`` entry (failure); the
response must surface both so an analyst can find the durable row — or see
why there is none — without reading server logs.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.api.routes.causal import _agent_state_to_response
from src.api.schemas.causal import AgentCausalAnalysisRequest

PERSIST_WARNING = (
    "Discovered DAG persistence FAILED: record_discovered_dag failed: boom "
    "[dag_version_hash=abc session_id=None query_id=a1]"
)


def _request() -> AgentCausalAnalysisRequest:
    return AgentCausalAnalysisRequest(
        dataset="patient_journeys", treatment_var="t", outcome_var="y"
    )


def _final_state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "causal_graph": {
            "nodes": ["t", "y"],
            "edges": [("t", "y")],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
            "adjustment_sets": [[]],
            "discovery_gate_decision": "accept",
        },
        "discovery_result": {"success": True},
        "estimation_result": {"ate": 0.1},
        "refutation_results": {"gate_decision": "proceed", "tests_passed": 1, "total_tests": 1},
        "warnings": [],
    }
    state.update(overrides)
    return state


def _response(final_state: Dict[str, Any]):
    return _agent_state_to_response(
        analysis_id="a1",
        request=_request(),
        data_source="database",
        n_rows=10,
        final_state=final_state,
        latency_ms=5,
    )


@pytest.mark.unit
def test_discovered_dag_id_surfaces_on_the_response():
    response = _response(_final_state(discovered_dag_id="dag-777"))
    assert response.discovered_dag_id == "dag-777"


@pytest.mark.unit
def test_persist_failure_surfaces_as_warning_and_null_id():
    response = _response(
        _final_state(discovered_dag_persist_error=PERSIST_WARNING, warnings=[PERSIST_WARNING])
    )
    assert response.discovered_dag_id is None
    assert PERSIST_WARNING in response.warnings


@pytest.mark.unit
def test_no_discovery_leaves_id_null_without_warning():
    """Positive control for the warning assertion above: a run that never
    persisted (manual DAG) has neither the id nor a persist warning."""
    state = _final_state(discovery_result=None)
    response = _response(state)
    assert response.discovered_dag_id is None
    assert not any("persist" in w.lower() for w in response.warnings)
