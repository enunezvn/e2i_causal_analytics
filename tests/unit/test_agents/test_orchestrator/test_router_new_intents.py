"""Router coverage for intents added when wiring up the Tier 1-5 workflow.

Before this change the orchestrator graph's router (``nodes/router.py``)
had no entry for ``multi_faceted`` or ``experiment_monitor`` — the only
multi-faceted detector lived in the unwired standalone ``router_v42.py``,
and ``experiment_monitor`` was registered in the factory but unreachable.
"""

from __future__ import annotations

import pytest

from src.agents.orchestrator.nodes.router import RouterNode


@pytest.mark.asyncio
async def test_router_dispatches_multi_faceted_to_tool_composer() -> None:
    router = RouterNode()
    state = {
        "intent": {
            "primary_intent": "multi_faceted",
            "confidence": 0.9,
            "secondary_intents": [],
            "requires_multi_agent": False,
        }
    }
    result = await router.execute(state)
    plan = result["dispatch_plan"]
    assert len(plan) == 1
    assert plan[0]["agent_name"] == "tool_composer"
    assert plan[0]["timeout_ms"] == 180_000, "tool_composer SLA is 180s"
    assert plan[0]["fallback_agent"] == "explainer"
    assert result["current_phase"] == "dispatching"


@pytest.mark.asyncio
async def test_router_dispatches_experiment_monitor() -> None:
    router = RouterNode()
    state = {
        "intent": {
            "primary_intent": "experiment_monitor",
            "confidence": 0.95,
            "secondary_intents": [],
            "requires_multi_agent": False,
        }
    }
    result = await router.execute(state)
    plan = result["dispatch_plan"]
    assert len(plan) == 1
    assert plan[0]["agent_name"] == "experiment_monitor"
    assert plan[0]["timeout_ms"] == 15_000
    assert plan[0]["fallback_agent"] is None


@pytest.mark.asyncio
async def test_router_still_handles_existing_cohort_intent() -> None:
    """cohort_definition predates this change and must still route correctly."""
    router = RouterNode()
    state = {
        "intent": {
            "primary_intent": "cohort_definition",
            "confidence": 0.93,
            "secondary_intents": [],
            "requires_multi_agent": False,
        }
    }
    result = await router.execute(state)
    plan = result["dispatch_plan"]
    # Re-pointed to cohort_profiler (chat companion with real per-segment counts);
    # cohort_constructor is the ML-pipeline agent and can't run from a chat payload.
    assert plan[0]["agent_name"] == "cohort_profiler"
