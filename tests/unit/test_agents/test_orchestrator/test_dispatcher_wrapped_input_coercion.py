"""Tests that the dispatcher coerces the generic orchestrator payload into the
shape required by each wrapped-input agent (issue #260).

Issue #260: ``DispatcherNode._dispatch_agent`` was instantiating the per-agent
``input_model`` (``ExperimentMonitorInput``, ``DriftMonitorInput``,
``ExperimentDesignerInput``) directly from the generic payload returned by
``_prepare_agent_input``. The generic payload contains
``user_context``/``parsed_query``/``span_id``/etc. — fields none of these
wrapped models declare. The strict ``@dataclass`` (ExperimentMonitorInput)
raised ``TypeError`` on extra kwargs; the pydantic models with required fields
(``features_to_monitor`` / ``business_question``) raised ``ValidationError``.

The dispatcher caught those and reported success=False, so the agent was
silently NEVER RUN while routing reported success. This file pins the
remediation: the dispatcher must project the generic payload onto the
declared fields of the target ``input_model`` and supply ACs-#3 defaults for
required missing fields.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from src.agents.orchestrator import _agent_method_map as mm
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_for(agent_name: str, query: str = "monitor experiments") -> dict:
    """Return a minimally-populated OrchestratorState for the dispatcher."""
    return {
        "query": query,
        "user_context": {"user_id": "u1", "brand": "Brand-X"},
        "session_id": "sess-1",
        "parsed_query": {"intent": "experiment_monitor"},
        "dispatch_plan": [
            {
                "agent_name": agent_name,
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 15000,
                "fallback_agent": None,
                "execution_mode": "parallel",
            }
        ],
        "parallel_groups": [[agent_name]],
    }


# ---------------------------------------------------------------------------
# AC-1: experiment_monitor (strict @dataclass input model) must succeed.
# AC-2: dataclass branch of the coercion logic.
# AC-3: ``experiment_ids`` default not supplied by orchestrator payload — must
#        be left to the dataclass default (None / empty list).
# AC-5: no full agent registry needed; stub agent + real AGENT_METHOD_MAP entry.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_monitor_dataclass_input_dispatches_successfully() -> None:
    """ExperimentMonitorInput is a strict ``@dataclass``.

    Before the fix the dispatcher would call ``ExperimentMonitorInput(**payload)``
    with the generic payload that contains ``user_context``/``parsed_query``/
    ``dispatch_id``/``span_id``/``execution_mode`` — all of which raise
    ``TypeError: __init__() got an unexpected keyword argument``.

    After the fix the dispatcher projects to the declared fields only and
    succeeds. ``query`` from the orchestrator payload makes it through.
    """
    # Bind a fake agent whose ``run_async`` we can inspect.
    captured: dict = {}

    async def fake_run_async(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        # Return a minimal output dict so the synthesizer path is not exercised.
        return {"monitor_summary": "1 critical alert", "experiments_checked": 3}

    agent = MagicMock()
    agent.run_async = fake_run_async
    # Strip ``analyze`` so the dispatcher cannot fall through silently.
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"experiment_monitor": agent})
    result = await dispatcher.execute(_state_for("experiment_monitor"))

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, (
        f"experiment_monitor dispatch failed: {agent_result.get('error')!r}"
    )
    assert agent_result["result"]["monitor_summary"] == "1 critical alert"
    # The agent received an ExperimentMonitorInput instance with the right shape.
    from src.agents.experiment_monitor.agent import ExperimentMonitorInput

    received = captured["input_obj"]
    assert isinstance(received, ExperimentMonitorInput), type(received)
    assert received.query == "monitor experiments"


# ---------------------------------------------------------------------------
# AC-1: drift_monitor (pydantic BaseModel with REQUIRED field).
# AC-2: Pydantic branch of the coercion logic.
# AC-3: ``features_to_monitor`` required → dispatcher must supply default ``[]``
#        when orchestrator payload doesn't carry it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_monitor_pydantic_input_with_required_field_defaults() -> None:
    """DriftMonitorInput.features_to_monitor is required (min_length=1).

    Before the fix the dispatcher would call ``DriftMonitorInput(**payload)``
    without ``features_to_monitor`` → pydantic ``ValidationError``.

    After the fix the dispatcher supplies a sensible default. The
    pydantic ``min_length=1`` validator means an empty list is also a
    ValidationError, so the dispatcher must pass a *non-empty* placeholder
    or let parameters override. Per AC-3, we accept ``features_to_monitor``
    sourced from ``dispatch.parameters`` first.
    """
    captured: dict = {}

    async def fake_run(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"drift_summary": "no drift", "overall_drift_score": 0.0}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    # Use ``parameters`` to thread ``features_to_monitor`` through — this is
    # the production path (the router stores per-agent parameters in the
    # dispatch entry).
    state = _state_for("drift_monitor", query="check drift on conversion features")
    state["dispatch_plan"][0]["parameters"] = {
        "features_to_monitor": ["calls_per_hcp", "samples_distributed"],
    }

    dispatcher = DispatcherNode(agent_registry={"drift_monitor": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, (
        f"drift_monitor dispatch failed: {agent_result.get('error')!r}"
    )
    from src.agents.drift_monitor.agent import DriftMonitorInput

    received = captured["input_obj"]
    assert isinstance(received, DriftMonitorInput), type(received)
    assert received.query == "check drift on conversion features"
    assert received.features_to_monitor == ["calls_per_hcp", "samples_distributed"]


# ---------------------------------------------------------------------------
# AC-1: experiment_designer (pydantic BaseModel with REQUIRED field).
# AC-3: ``business_question`` required → dispatcher must default it from the
#        orchestrator-level ``query`` if dispatch params don't override it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_designer_business_question_defaults_from_query() -> None:
    """ExperimentDesignerInput.business_question is required (min_length=10).

    Per AC-3, the dispatcher should default ``business_question`` from the
    orchestrator-level ``query``. The query in this test is well above the
    10-char min_length so the pydantic validator must accept it.
    """
    captured: dict = {}

    def fake_run(input_obj):  # noqa: ANN001 — experiment_designer is sync
        captured["input_obj"] = input_obj
        return {"design_summary": "RCT with 500 HCPs per arm"}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    state = _state_for(
        "experiment_designer",
        query="design an experiment to measure the effect of email cadence on adherence",
    )

    dispatcher = DispatcherNode(agent_registry={"experiment_designer": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, (
        f"experiment_designer dispatch failed: {agent_result.get('error')!r}"
    )
    from src.agents.experiment_designer.agent import ExperimentDesignerInput

    received = captured["input_obj"]
    assert isinstance(received, ExperimentDesignerInput), type(received)
    assert "email cadence" in received.business_question
    assert len(received.business_question) >= 10


# ---------------------------------------------------------------------------
# AC-3: explicit ``parameters`` should win over the orchestrator-level
#        ``query`` fallback when both are present.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_parameters_override_payload_defaults() -> None:
    """When the dispatch entry has ``parameters`` that match the input_model
    fields, those values must take precedence over generic-payload values."""
    captured: dict = {}

    def fake_run(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"design_summary": "ok"}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    state = _state_for(
        "experiment_designer",
        query="this is the orchestrator query, more than 10 characters",
    )
    state["dispatch_plan"][0]["parameters"] = {
        "business_question": ("explicit business question from dispatch parameters - quite long"),
        "max_redesign_iterations": 1,
    }

    dispatcher = DispatcherNode(agent_registry={"experiment_designer": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, agent_result.get("error")
    from src.agents.experiment_designer.agent import ExperimentDesignerInput

    received = captured["input_obj"]
    assert isinstance(received, ExperimentDesignerInput)
    assert received.business_question == (
        "explicit business question from dispatch parameters - quite long"
    )
    assert received.max_redesign_iterations == 1


# ---------------------------------------------------------------------------
# AC-4: existing non-wrapped agents must continue to work. The dispatcher
#        does NOT touch agents without an ``input_model`` (e.g. causal_impact,
#        health_score). Asserts no regression.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_wrapped_agent_dispatch_unaffected_by_fix() -> None:
    """``health_score`` has no input_model and uses ``uses_kwargs=True``.

    The dispatcher's wrapped-input coercion path must NOT engage. The agent
    receives the full generic payload as kwargs.
    """
    captured: dict = {}

    async def fake_check_health(**kwargs):  # noqa: ANN003
        captured["kwargs"] = kwargs
        return {
            "overall_health_score": 92.0,
            "health_grade": "A",
            "health_summary": "all green",
        }

    agent = MagicMock()
    agent.check_health = fake_check_health
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"health_score": agent})
    state = _state_for("health_score", query="check system health")

    result = await dispatcher.execute(state)
    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, agent_result.get("error")
    assert agent_result["result"]["health_grade"] == "A"
    # Full generic payload passed through (uses_kwargs=True, no input_model).
    assert captured["kwargs"]["query"] == "check system health"
    assert "user_context" in captured["kwargs"]
    assert "parsed_query" in captured["kwargs"]


# ---------------------------------------------------------------------------
# AC-2: the coercion must work for a pydantic BaseModel that DOES tolerate
#        extra fields (so the projection step doesn't accidentally drop fields
#        for permissive models). Demonstrates the projection is conservative.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_filters_to_declared_fields_only() -> None:
    """A pydantic model declaring only ``query`` must receive only ``query``,
    not the rest of the generic payload, even though pydantic by default
    forbids extras and would error otherwise.

    Uses a synthetic model registered into AGENT_METHOD_MAP at test scope.
    """

    class MinimalInput(BaseModel):
        query: str = Field(..., min_length=1)

    fake_module = types.ModuleType("__test_minimal_input__")
    fake_module.MinimalInput = MinimalInput  # type: ignore[attr-defined]
    sys.modules["__test_minimal_input__"] = fake_module

    original = mm.AGENT_METHOD_MAP.get("explainer")
    mm.AGENT_METHOD_MAP["explainer"] = mm.AgentMethodSpec(
        method="explain",
        is_async=True,
        uses_kwargs=False,  # input_model wrapping path: single arg
        input_model="MinimalInput",
        input_module="__test_minimal_input__",
    )
    try:
        captured: dict = {}

        async def fake_explain(input_obj):  # noqa: ANN001
            captured["input_obj"] = input_obj
            return {"narrative": "ok"}

        agent = MagicMock()
        agent.explain = fake_explain
        del agent.analyze

        dispatcher = DispatcherNode(agent_registry={"explainer": agent})
        result = await dispatcher.execute(_state_for("explainer", query="explain X"))

        agent_result = result["agent_results"][0]
        assert agent_result["success"] is True, agent_result.get("error")
        assert isinstance(captured["input_obj"], MinimalInput)
        assert captured["input_obj"].query == "explain X"
    finally:
        if original is not None:
            mm.AGENT_METHOD_MAP["explainer"] = original


# ---------------------------------------------------------------------------
# AC-3: an experiment_monitor query that supplies ``experiment_ids`` via
#        ``parameters`` must reach the agent intact.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_monitor_experiment_ids_threaded_through_parameters() -> None:
    """When the router puts ``experiment_ids`` in dispatch parameters, the
    dispatcher must thread them into the ExperimentMonitorInput dataclass."""
    captured: dict = {}

    async def fake_run_async(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"monitor_summary": "specific check"}

    agent = MagicMock()
    agent.run_async = fake_run_async
    del agent.analyze

    state = _state_for("experiment_monitor", query="check exp_2026_05")
    state["dispatch_plan"][0]["parameters"] = {
        "experiment_ids": ["exp_2026_05", "exp_2026_06"],
        "check_all_active": False,
    }

    dispatcher = DispatcherNode(agent_registry={"experiment_monitor": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, agent_result.get("error")
    received = captured["input_obj"]
    assert received.experiment_ids == ["exp_2026_05", "exp_2026_06"]
    assert received.check_all_active is False
