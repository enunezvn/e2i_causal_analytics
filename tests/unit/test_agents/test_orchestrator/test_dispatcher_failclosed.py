"""Regression suite for #814: the dispatcher must FAIL CLOSED for a missing
registry entry instead of fabricating plausible analytics values.

Background
----------
``DispatcherNode._mock_agent_execution`` returns canned narratives with
fabricated numbers (``ATE=0.12``, ``$2.5M ROI``, ``2x higher response rate``)
when the routed ``agent_name`` is absent from the registry. That mock is a
legitimate **test-only** scaffold for exercising dispatch mechanics without
instantiating real agents — but it must be UNREACHABLE in production, where a
missing agent (e.g. a partial registry from a swallowed instantiation failure)
must surface a loud, honest degradation rather than a fake ``success=True``
result.

The fix gates the mock behind an explicit, default-off ``allow_mock`` flag:
production constructs the dispatcher with ``allow_mock=False`` (the default) and
a missing agent fails closed; unit tests that want the canned scaffold opt in
with ``allow_mock=True``.

Discipline: no mocking of the dispatcher logic itself — these drive the real
``DispatcherNode.execute`` / ``_dispatch_agent`` code paths. Registered agents
are lightweight ``MagicMock`` stand-ins only at the agent boundary.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, dispatch_to_agents

# Fabricated fragments that must NEVER reach a user via a fail-closed path.
_FABRICATED_FRAGMENTS = ("ATE=0.12", "2x higher response rate", "$2.5M", "15% increase")


def _state(agent_name: str) -> Dict[str, Any]:
    return {
        "query": "what drove conversion and which segments respond best?",
        "dispatch_plan": [
            {
                "agent_name": agent_name,
                "priority": 1,
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [[agent_name]],
    }


def _assert_no_fabrication(result: Dict[str, Any]) -> None:
    blob = str(result.get("result")) + str(result.get("error") or "")
    for frag in _FABRICATED_FRAGMENTS:
        assert frag not in blob, f"fabricated fragment leaked: {frag!r} in {blob!r}"


@pytest.mark.asyncio
async def test_missing_agent_fails_closed_by_default():
    """Default (prod-shaped) dispatcher with no registry entry → success=False,
    structured error, no fabricated result."""
    dispatcher = DispatcherNode()  # allow_mock defaults to False
    result = await dispatcher.execute(_state("causal_impact"))

    res = result["agent_results"][0]
    assert res["agent_name"] == "causal_impact"
    assert res["success"] is False
    assert res["result"] is None
    assert "causal_impact" in res["error"]
    _assert_no_fabrication(res)


@pytest.mark.asyncio
async def test_partial_registry_does_not_fabricate():
    """Faithful degraded repro (#814): a PARTIAL registry that is missing the
    routed agent must fail closed — never surface the canned 'heterogeneous'
    fabrication — even though other agents are present."""
    present = MagicMock()
    present.analyze = AsyncMock(return_value={"narrative": "real", "confidence": 0.9})
    # heterogeneous_optimizer is intentionally absent (simulates a dropped agent).
    dispatcher = DispatcherNode(agent_registry={"causal_impact": present})

    result = await dispatcher.execute(_state("heterogeneous_optimizer"))
    res = result["agent_results"][0]
    assert res["success"] is False
    assert res["result"] is None
    _assert_no_fabrication(res)


@pytest.mark.asyncio
async def test_allow_mock_true_preserves_test_scaffold():
    """The test-only scaffold is preserved behind the explicit flag: with
    allow_mock=True a missing agent still yields the canned mock (so dispatch-
    mechanics unit tests keep working)."""
    dispatcher = DispatcherNode(allow_mock=True)
    result = await dispatcher.execute(_state("causal_impact"))

    res = result["agent_results"][0]
    assert res["success"] is True
    assert res["result"] is not None
    assert "narrative" in res["result"]


@pytest.mark.asyncio
async def test_dispatch_to_agents_module_fn_fails_closed():
    """The registry-less graph node function (graph else-branch) fails closed —
    it builds a DispatcherNode with no registry and allow_mock off."""
    result = await dispatch_to_agents(_state("causal_impact"))
    res = result["agent_results"][0]
    assert res["success"] is False
    _assert_no_fabrication(res)


@pytest.mark.asyncio
async def test_registered_agent_still_dispatched():
    """A registered agent is unaffected by the fail-closed change — it is
    dispatched to the real agent and succeeds."""
    agent = MagicMock()
    agent.analyze = AsyncMock(
        return_value={"narrative": "real response", "recommendations": [], "confidence": 0.91}
    )
    dispatcher = DispatcherNode(agent_registry={"my_agent": agent})

    result = await dispatcher.execute(_state("my_agent"))
    res = result["agent_results"][0]
    assert res["success"] is True
    assert res["result"]["narrative"] == "real response"
    assert agent.analyze.called


def test_failclosed_output_failure_details_matches_state_contract():
    """#814 contract regression: the fail-closed path populates ``failure_details``
    as a ``List[Dict]`` (one entry per failed agent) — see
    ``OrchestratorAgent._build_output``. Before #814 the dispatcher fabricated a
    ``success=True`` mock, so the failure branch (and thus this list) was never
    populated in the registry-less harness/prod-degraded path; the
    ``OrchestratorState.failure_details`` annotation was therefore left as
    ``Optional[Dict]`` and never exercised. The fail-closed default makes that path
    production-reachable under a partial registry, so the state contract must accept
    the real ``List[Dict]`` shape with NO ContractValidator type error (this is the
    exact Tier 1-5 harness ``type_error`` the fail-closed path would otherwise emit).

    Drives the REAL ``_build_output`` and the REAL ``ContractValidator`` — no mocks.
    """
    from src.agents.orchestrator.agent import OrchestratorAgent
    from src.agents.orchestrator.state import OrchestratorState
    from src.testing.contract_validator import ContractValidator

    failclosed_state: Dict[str, Any] = {
        "query_id": "q-failclosed",
        "status": "failed",
        "synthesized_response": "I was unable to complete the analysis.",
        "response_confidence": 0.0,
        "agent_results": [
            {
                "agent_name": "causal_impact",
                "success": False,
                "result": None,
                "error": (
                    "Agent 'causal_impact' is not available in the dispatcher "
                    "registry; dispatch fails closed (no fabricated result)."
                ),
                "latency_ms": 0,
            },
            {
                "agent_name": "explainer",
                "success": False,
                "result": None,
                "error": (
                    "Agent 'explainer' is not available in the dispatcher "
                    "registry; dispatch fails closed (no fabricated result)."
                ),
                "latency_ms": 0,
            },
        ],
    }

    output = OrchestratorAgent(enable_opik=False)._build_output(failclosed_state)

    # Producer reality: a LIST of per-agent failure dicts (already true today).
    assert isinstance(output["failure_details"], list)
    assert {d["agent_name"] for d in output["failure_details"]} == {"causal_impact", "explainer"}

    # The STATE CONTRACT must accept that list shape with zero type errors.
    result = ContractValidator().validate_state(output, OrchestratorState)
    fd_errors = [te for te in result.type_errors if te.get("field") == "failure_details"]
    assert fd_errors == [], f"failure_details violates OrchestratorState contract: {fd_errors}"
