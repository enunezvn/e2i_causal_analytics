"""Regression: cohort_constructor must FAIL CLOSED with an actionable message
when routed from a chat dispatch — never leak the raw registry error.

Background
----------
``cohort_constructor`` is a Tier-0 ML-pipeline agent that the chatbot's router
lists in ``VALID_AGENTS``, so a cohort-definition query CAN be routed to it. But
its ``run(patient_df, brand, ...)`` entry point needs a real patient DataFrame +
brand config that the conversational orchestrator payload never carries, and it
was deliberately never added to ``AGENT_METHOD_MAP`` (Tier 1–5 only). Before the
fix, dispatch fell through to the default ``analyze`` method the agent doesn't
implement and surfaced the raw internal error::

    Agent 'cohort_constructor' is registered but has no method 'analyze'.
    Check AGENT_METHOD_MAP.

The fix registers a fail-closed ``INPUT_RESOLVERS`` entry (the established
#F12/F13/F14/#814 pattern) that short-circuits BEFORE the method lookup and
returns a clear, actionable ``NeedsStructuredInput`` message, fabricating nothing.

Discipline: these drive the REAL ``DispatcherNode._dispatch_agent`` path; the
cohort_constructor agent is a lightweight ``MagicMock`` only at the agent
boundary (and the resolver short-circuits before any of its methods are called).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.agents.orchestrator.nodes.dispatcher import (
    INPUT_RESOLVERS,
    NeedsStructuredInput,
    _resolve_cohort_constructor_input,
)


def _state() -> Dict[str, Any]:
    return {
        "query": "build a cohort of eligible CSU patients for Remibrutinib",
        "dispatch_plan": [
            {
                "agent_name": "cohort_constructor",
                "priority": 1,
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["cohort_constructor"]],
    }


def test_cohort_constructor_registered_in_input_resolvers():
    """The resolver is wired into the single-source-of-truth registry."""
    assert INPUT_RESOLVERS.get("cohort_constructor") is _resolve_cohort_constructor_input


def test_resolver_always_fails_closed_without_fabricating():
    """The resolver never returns fabricated inputs — a chat payload cannot
    ground a patient dataset, so it always signals NeedsStructuredInput."""
    signal = _resolve_cohort_constructor_input({"query": "anything"}, {"parameters": {}})
    assert isinstance(signal, NeedsStructuredInput)
    assert signal.agent_name == "cohort_constructor"
    assert set(signal.missing) == {"patient_df", "brand"}
    err = signal.to_error()
    assert "no values were fabricated" in err


@pytest.mark.asyncio
async def test_dispatch_fails_closed_not_raw_registry_error():
    """End-to-end through the REAL dispatcher: a routed cohort query yields an
    actionable fail-closed message, NOT the raw 'no method analyze' registry
    error (the pre-fix behavior)."""
    from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

    # Registered (so we pass the registry check and reach the resolver), but the
    # resolver short-circuits before any method on this mock is touched.
    dispatcher = DispatcherNode(agent_registry={"cohort_constructor": MagicMock()})
    result = await dispatcher.execute(_state())

    res = result["agent_results"][0]
    assert res["agent_name"] == "cohort_constructor"
    assert res["success"] is False
    assert res["result"] is None

    err = res["error"]
    # Actionable fail-closed message ...
    assert "cohort_constructor" in err
    assert "patient_df" in err
    assert "no values were fabricated" in err
    # ... and NOT the raw internal registry/method error that leaked pre-fix.
    assert "Check AGENT_METHOD_MAP" not in err
    assert "has no method" not in err
