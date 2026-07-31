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
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_experiment_monitor_dataclass_input_dispatches_successfully() -> None:
    """ExperimentMonitorInput is a strict ``@dataclass``."""
    captured: dict = {}

    async def fake_run_async(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"monitor_summary": "1 critical alert", "experiments_checked": 3}

    agent = MagicMock()
    agent.run_async = fake_run_async
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"experiment_monitor": agent})
    result = await dispatcher.execute(_state_for("experiment_monitor"))

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, (
        f"experiment_monitor dispatch failed: {agent_result.get('error')!r}"
    )
    assert agent_result["result"]["monitor_summary"] == "1 critical alert"
    from src.agents.experiment_monitor.agent import ExperimentMonitorInput

    received = captured["input_obj"]
    assert isinstance(received, ExperimentMonitorInput), type(received)
    assert received.query == "monitor experiments"


@pytest.mark.asyncio
async def test_drift_monitor_pydantic_input_with_required_field_defaults() -> None:
    """DriftMonitorInput.features_to_monitor is required (min_length=1).

    Use ``parameters`` to thread ``features_to_monitor`` through — this is
    the production path (the router stores per-agent parameters in the
    dispatch entry).
    """
    captured: dict = {}

    async def fake_run(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"drift_summary": "no drift", "overall_drift_score": 0.0}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

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


@pytest.mark.asyncio
async def test_experiment_designer_business_question_defaults_from_query() -> None:
    """ExperimentDesignerInput.business_question is required (min_length=10)."""
    captured: dict = {}

    def fake_run(input_obj):  # noqa: ANN001
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


@pytest.mark.asyncio
async def test_non_wrapped_kwargs_agent_gets_clean_resolver_kwargs() -> None:
    """``health_score`` has no input_model and uses ``uses_kwargs=True``.

    Pre-#883 this test pinned the generic-payload LEAK (``user_context`` /
    ``parsed_query`` splatted into ``check_health``) as the status quo — the
    exact TypeError that made the 'system_health' intent dead against the REAL
    agent (#883 §3). The resolver registry now hands kwargs agents a CLEAN
    kwarg set, so the contract this test pins is the clean one.
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
    assert captured["kwargs"]["query"] == "check system health"
    # #883 §3: the generic payload must NOT leak into the kwargs splat — the
    # real ``check_health`` signature rejects these with TypeError.
    assert "user_context" not in captured["kwargs"]
    assert "parsed_query" not in captured["kwargs"]
    assert set(captured["kwargs"]) <= {"scope", "query", "experiment_name", "session_id"}


@pytest.mark.asyncio
async def test_projection_filters_to_declared_fields_only() -> None:
    """A pydantic model declaring only ``query`` must receive only ``query``."""

    class MinimalInput(BaseModel):
        query: str = Field(..., min_length=1)

    fake_module = types.ModuleType("__test_minimal_input__")
    fake_module.MinimalInput = MinimalInput  # type: ignore[attr-defined]
    sys.modules["__test_minimal_input__"] = fake_module

    # Host the synthetic spec on a resolver-LESS agent (#883 added an explainer
    # resolver that fails closed pre-projection, #1351 added a causal_impact
    # resolver; this test is about projection — drift_monitor has no resolver).
    original = mm.AGENT_METHOD_MAP.get("drift_monitor")
    mm.AGENT_METHOD_MAP["drift_monitor"] = mm.AgentMethodSpec(
        method="explain",
        is_async=True,
        uses_kwargs=False,
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

        dispatcher = DispatcherNode(agent_registry={"drift_monitor": agent})
        result = await dispatcher.execute(_state_for("drift_monitor", query="explain X"))

        agent_result = result["agent_results"][0]
        assert agent_result["success"] is True, agent_result.get("error")
        assert isinstance(captured["input_obj"], MinimalInput)
        assert captured["input_obj"].query == "explain X"
    finally:
        if original is not None:
            mm.AGENT_METHOD_MAP["drift_monitor"] = original


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


# ---------------------------------------------------------------------------
# Iter-1 (codex MED-required): drift_monitor with NO dispatch.parameters must
# still dispatch successfully by sourcing features from parsed_query.entities.
# Without this fallback the production path is broken even after the
# extra-kwargs leak is fixed — pydantic min_length=1 trips on empty list.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_monitor_features_sourced_from_parsed_query_entities() -> None:
    """When ``dispatch.parameters`` does NOT carry ``features_to_monitor``,
    the dispatcher must derive features from ``parsed_query.entities`` (KPI
    mentions in the user's query). Otherwise pydantic ``min_length=1`` trips
    on an empty list and the agent never runs.
    """
    captured: dict = {}

    async def fake_run(input_obj):  # noqa: ANN001
        captured["input_obj"] = input_obj
        return {"drift_summary": "ok", "overall_drift_score": 0.0}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze

    state = _state_for("drift_monitor", query="is there drift on conversion_rate?")
    state["parsed_query"] = {
        "entities": [
            {"type": "kpi", "value": "conversion_rate", "confidence": 0.92},
            {"type": "brand", "value": "Brand-X", "confidence": 0.99},
        ],
    }
    state["dispatch_plan"][0]["parameters"] = {}

    dispatcher = DispatcherNode(agent_registry={"drift_monitor": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True, agent_result.get("error")
    from src.agents.drift_monitor.agent import DriftMonitorInput

    received = captured["input_obj"]
    assert isinstance(received, DriftMonitorInput)
    # Only the kpi-typed entity becomes a feature; the brand entity filtered.
    assert received.features_to_monitor == ["conversion_rate"]


@pytest.mark.asyncio
async def test_drift_monitor_no_features_anywhere_produces_clear_error() -> None:
    """When neither dispatch.parameters nor parsed_query.entities supply
    features, the dispatcher surfaces a CLEAR structured error referencing
    ``features_to_monitor`` — not the old cryptic 'unexpected keyword
    user_context' leak. Don't fabricate phantom feature names.
    """
    agent = MagicMock()
    agent.run = AsyncMock(return_value={"drift_summary": "ok"})
    del agent.analyze

    state = _state_for("drift_monitor", query="something about drift")
    state["dispatch_plan"][0]["parameters"] = {}
    state["parsed_query"] = {"entities": []}

    dispatcher = DispatcherNode(agent_registry={"drift_monitor": agent})
    result = await dispatcher.execute(state)

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is False
    error_msg = (agent_result["error"] or "").lower()
    assert "features_to_monitor" in error_msg or "validation" in error_msg, (
        f"expected pydantic-validation-style error, got: {error_msg}"
    )
    assert "user_context" not in error_msg, (
        f"old leak shouldn't appear in fixed error message: {error_msg}"
    )


# ---------------------------------------------------------------------------
# Iter-1 (codex MED-tracker): a future AGENT_METHOD_MAP entry combining
# input_model + uses_kwargs=True must be supported via re-flattening.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kwargs_plus_input_model_combo_robust() -> None:
    """If a future AGENT_METHOD_MAP entry combines ``uses_kwargs=True`` with
    ``input_model``, the dispatcher must re-flatten the validated model into
    a kwargs dict before splatting. Without this guard,
    ``method(**model_instance)`` raises TypeError.

    No production entry combines both today; this pins defense-in-depth.
    """

    class KwargsAndModelInput(BaseModel):
        query: str = Field(..., min_length=1)
        flag: bool = Field(False)

    fake_module = types.ModuleType("__test_kwargs_and_model__")
    fake_module.KwargsAndModelInput = KwargsAndModelInput  # type: ignore[attr-defined]
    sys.modules["__test_kwargs_and_model__"] = fake_module

    # Host the synthetic spec on a resolver-LESS agent (#883/#1351 resolver
    # output would REPLACE the payload before the combo under test;
    # drift_monitor has no resolver).
    original = mm.AGENT_METHOD_MAP.get("drift_monitor")
    mm.AGENT_METHOD_MAP["drift_monitor"] = mm.AgentMethodSpec(
        method="explain",
        is_async=True,
        uses_kwargs=True,
        input_model="KwargsAndModelInput",
        input_module="__test_kwargs_and_model__",
    )
    try:
        captured: dict = {}

        async def fake_explain(**kwargs):  # noqa: ANN003
            captured["kwargs"] = kwargs
            return {"narrative": "explained"}

        agent = MagicMock()
        agent.explain = fake_explain
        del agent.analyze

        dispatcher = DispatcherNode(agent_registry={"drift_monitor": agent})
        state = _state_for("drift_monitor", query="explain this thing")
        state["dispatch_plan"][0]["parameters"] = {"flag": True}

        result = await dispatcher.execute(state)
        agent_result = result["agent_results"][0]
        assert agent_result["success"] is True, agent_result.get("error")
        assert captured["kwargs"]["query"] == "explain this thing"
        assert captured["kwargs"]["flag"] is True
    finally:
        if original is not None:
            mm.AGENT_METHOD_MAP["drift_monitor"] = original
