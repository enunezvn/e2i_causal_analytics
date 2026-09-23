"""#2238: orchestrator nodes must not echo the ``operator.add`` channels.

``OrchestratorState`` declares ``agent_results``, ``errors`` and ``warnings``
as ``Annotated[List, operator.add]``. Every node (classify, rag_context, route,
dispatch, synthesize) returned ``{**state, ...}``, so each re-submitted the
accumulated lists and LangGraph appended them again.

Measured before the fix (compiled graph, RAG off): seeded warning/error x16,
seeded agent result x8, and the ONE dispatched agent card came out TWICE
(``agent_results`` = 10 rows for 1 seed + 1 dispatch). With RAG on: x32/x16
and the card still twice. The duplicated card is user-visible: the agent's
``_format_output`` builds ``agents_dispatched`` / ``agent_results`` from that
channel.

External calls stubbed at the boundary only: audit service absent, RAG
dependencies empty (retriever-less mock mode), classifier pipeline off, the
dispatcher's test-only mock scaffold (``allow_mock=True``), and the seed
agent result marked ``success=False`` so the synthesizer takes the single
result path (no LLM call).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from src.agents.orchestrator.agent import OrchestratorAgent
from src.agents.orchestrator.graph import create_orchestrator_graph

SEED_W = "seed-warning"
SEED_E = {"node": "seed", "error": "seed-error"}
SEED_R = {
    "agent_name": "seed_agent",
    "success": False,
    "result": None,
    "error": "seeded prior-turn failure",
    "latency_ms": 1,
}


def _base_state() -> Dict[str, Any]:
    return {
        "query": "what is the impact of hcp engagement on patient conversions?",
        "query_id": "q-2238",
        "user_id": None,
        "session_id": None,
        "user_context": {},
        "conversation_history": [],
        "start_time": "",
        "current_phase": "classifying",
        "status": "pending",
        "agent_results": [SEED_R],
        "errors": [SEED_E],
        "warnings": [SEED_W],
        "fallback_used": False,
        "total_latency_ms": 0,
        "classification_latency_ms": 0,
        "rag_latency_ms": 0,
        "routing_latency_ms": 0,
        "dispatch_latency_ms": 0,
        "synthesis_latency_ms": 0,
        "response_confidence": 0.0,
        "agents_dispatched": [],
    }


async def _no_rag_deps() -> Dict[str, Any]:
    return {}


@pytest.fixture
def boundary_stubs(monkeypatch):
    monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
    monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "off")
    with (
        patch("src.agents.base.audit_chain_mixin.get_audit_chain_service", return_value=None),
        patch("src.api.dependencies.get_rag_dependencies", _no_rag_deps),
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_rag", [False, True])
async def test_seeds_and_dispatched_card_come_out_once(boundary_stubs, enable_rag):
    graph = create_orchestrator_graph(agent_registry=None, enable_rag=enable_rag, allow_mock=True)
    final = await graph.ainvoke(_base_state())

    assert final["status"] == "completed"
    assert final["warnings"] == [SEED_W]
    assert final["errors"] == [SEED_E]
    names = [r.get("agent_name") for r in final["agent_results"]]
    assert names.count("seed_agent") == 1, names
    dispatched = [n for n in names if n != "seed_agent"]
    assert len(dispatched) >= 1, names
    assert len(dispatched) == len(set(dispatched)), names


@pytest.mark.asyncio
async def test_agent_run_emits_each_agent_card_once(boundary_stubs):
    """The user-facing path: ``OrchestratorAgent.run`` formats its output from
    the ``agent_results`` channel, so an echoing synthesizer showed every
    agent card twice."""
    result = await OrchestratorAgent(allow_mock=True).run(
        {"query": "what is the impact of hcp engagement on patient conversions?"}
    )

    names = [r["agent_name"] for r in result["agent_results"]]
    assert names, result
    assert len(names) == len(set(names)), names
    assert len(result["agents_dispatched"]) == len(set(result["agents_dispatched"]))
