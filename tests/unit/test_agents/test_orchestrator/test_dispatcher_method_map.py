"""Dispatcher tests for real-agent method mapping.

The orchestrator's dispatcher must call each registered agent via the method
name declared in ``AGENT_METHOD_MAP``, not the legacy ``.analyze()`` contract.
Without these tests it's easy to register, say, ``health_score`` (which uses
``check_health``) into the orchestrator and have every call silently raise
``AttributeError: 'HealthScoreAgent' object has no attribute 'analyze'`` —
caught only at runtime when the dispatcher swallows the exception into an
AgentResult.error string.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.nodes.dispatcher import DispatcherNode


@pytest.mark.asyncio
async def test_dispatcher_calls_check_health_for_health_score_agent() -> None:
    """health_score uses ``check_health`` and ``uses_kwargs=True``."""
    agent = MagicMock()
    agent.check_health = AsyncMock(
        return_value={
            "overall_health_score": 88.5,
            "health_grade": "B",
            "health_summary": "all green",
            "status": "healthy",
        }
    )
    # The agent must NOT have ``analyze`` — that's the bug we're guarding against.
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"health_score": agent})

    state = {
        "query": "system health",
        "dispatch_plan": [
            {
                "agent_name": "health_score",
                "priority": "critical",
                "parameters": {"scope": "all"},
                "timeout_ms": 5000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["health_score"]],
    }

    result = await dispatcher.execute(state)

    assert agent.check_health.called, "dispatcher must call check_health, not analyze"
    call_kwargs = agent.check_health.call_args.kwargs
    assert call_kwargs["query"] == "system health"
    assert result["agent_results"][0]["success"] is True
    assert result["agent_results"][0]["result"]["health_grade"] == "B"


@pytest.mark.asyncio
async def test_dispatcher_calls_optimize_for_resource_optimizer() -> None:
    """resource_optimizer uses ``optimize`` and ``uses_kwargs=True``.

    With a REAL allocation problem supplied via ``dispatch.parameters`` (an
    API/router-driven call), the F13 input resolver passes it through as a CLEAN
    kwarg set and ``optimize(**kwargs)`` is called — with NONE of the generic
    payload leak (``user_context``/``parsed_query``/``span_id``/...) that used to
    crash the splat. (A bare chat dispatch with empty parameters now fails closed;
    see test_dispatcher_input_resolver.py.)
    """
    captured: dict = {}

    async def fake_optimize(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"optimization_summary": "shift 20% to channel A", "status": "completed"}

    agent = MagicMock()
    agent.optimize = fake_optimize
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"resource_optimizer": agent})
    state = {
        "query": "optimize budget",
        "dispatch_plan": [
            {
                "agent_name": "resource_optimizer",
                "priority": "critical",
                "parameters": {
                    "allocation_targets": [
                        {
                            "entity_id": "ch_a",
                            "entity_type": "territory",
                            "current_allocation": 40000.0,
                            "expected_response": 1.2,
                        }
                    ],
                    "constraints": [{"constraint_type": "budget", "value": 40000.0}],
                },
                "timeout_ms": 20000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["resource_optimizer"]],
    }

    result = await dispatcher.execute(state)

    assert result["agent_results"][0]["success"] is True, result["agent_results"][0].get("error")
    assert "optimization_summary" in result["agent_results"][0]["result"]
    # Real structured inputs reached optimize(); the generic-payload leak did not.
    assert captured["allocation_targets"][0]["entity_id"] == "ch_a"
    for leaked in ("user_context", "parsed_query", "span_id", "dispatch_id", "execution_mode"):
        assert leaked not in captured, f"{leaked} leaked into optimize(): {sorted(captured)}"


@pytest.mark.asyncio
async def test_dispatcher_reports_missing_method_clearly() -> None:
    """Registered agent without the expected method must produce a useful error.

    Silent fallback to ``_mock_agent_execution`` would mask the
    misconfiguration and ship fabricated narratives — the dispatcher must
    surface ``AttributeError`` as a structured agent failure instead.
    """
    agent = MagicMock(spec=["__class__"])  # no methods exposed

    dispatcher = DispatcherNode(agent_registry={"explainer": agent})
    state = {
        "query": "explain this",
        "dispatch_plan": [
            {
                "agent_name": "explainer",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 5000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["explainer"]],
    }

    result = await dispatcher.execute(state)
    agent_result = result["agent_results"][0]
    assert agent_result["success"] is False
    assert "explain" in (agent_result["error"] or ""), agent_result["error"]


@pytest.mark.asyncio
async def test_dispatcher_normalizes_dataclass_output() -> None:
    """Agents like experiment_monitor return a dataclass; dispatcher flattens to dict.

    ``AgentResult.result`` must be a plain dict so downstream synthesizer and
    audit serialization work. Don't break for output objects that aren't
    TypedDicts.
    """

    class FakeOutput:
        def __init__(self) -> None:
            self.monitor_summary = "0 critical, 2 warn"
            self.experiments_checked = 5
            self._private = "should be dropped"

    agent = MagicMock()
    agent.run_async = AsyncMock(return_value=FakeOutput())
    del agent.analyze

    dispatcher = DispatcherNode(agent_registry={"experiment_monitor": agent})
    state = {
        "query": "monitor experiments",
        "dispatch_plan": [
            {
                "agent_name": "experiment_monitor",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 15000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["experiment_monitor"]],
    }

    # NOTE: this test deliberately avoids passing input_model — the registered
    # agent in tests is a plain MagicMock without the real ExperimentMonitorInput
    # signature, so the dispatcher falls back to building the model and will
    # fail with "Failed to build". To exercise the normalization path itself we
    # patch ``get_method_spec`` to return a spec without the input wrapper.
    from src.agents.orchestrator import _agent_method_map as mm

    original = mm.AGENT_METHOD_MAP["experiment_monitor"]
    mm.AGENT_METHOD_MAP["experiment_monitor"] = mm.AgentMethodSpec(
        method="run_async",
        is_async=True,
        uses_kwargs=False,
        input_model=None,
        input_module=None,
    )
    try:
        result = await dispatcher.execute(state)
    finally:
        mm.AGENT_METHOD_MAP["experiment_monitor"] = original

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is True
    assert isinstance(agent_result["result"], dict)
    assert agent_result["result"]["monitor_summary"] == "0 critical, 2 warn"
    assert agent_result["result"]["experiments_checked"] == 5
    assert "_private" not in agent_result["result"], "private fields must be stripped"


@pytest.mark.asyncio
async def test_dispatcher_fails_closed_when_agent_not_registered() -> None:
    """#814: an empty/partial registry FAILS CLOSED by default — a missing agent
    yields a structured ``success=False`` error, NOT a fabricated mock narrative.
    The canned scaffold is reachable only via ``DispatcherNode(allow_mock=True)``."""
    dispatcher = DispatcherNode()
    state = {
        "query": "what causes adoption?",
        "dispatch_plan": [
            {
                "agent_name": "causal_impact",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["causal_impact"]],
    }

    result = await dispatcher.execute(state)
    agent_result = result["agent_results"][0]
    assert agent_result["success"] is False
    assert agent_result["result"] is None
    assert "causal_impact" in agent_result["error"]


@pytest.mark.asyncio
async def test_dispatcher_catches_pydantic_validation_error_on_input_wrapper() -> None:
    """Pydantic ValidationError (subclass of ValueError) on input wrapping must
    produce a structured AgentResult.error, not propagate uncaught.

    Plan review-polish: catch (ImportError, AttributeError, TypeError, ValueError).
    """
    import sys
    import types

    from pydantic import BaseModel, Field

    class StrictInput(BaseModel):
        required_field: str = Field(..., min_length=1)

    # Inject a fake module so dispatcher's import_module resolves StrictInput.
    fake_module = types.ModuleType("__test_strict_input__")
    fake_module.StrictInput = StrictInput  # type: ignore[attr-defined]
    sys.modules["__test_strict_input__"] = fake_module

    from src.agents.orchestrator import _agent_method_map as mm

    original = mm.AGENT_METHOD_MAP.get("explainer")
    mm.AGENT_METHOD_MAP["explainer"] = mm.AgentMethodSpec(
        method="explain",
        is_async=True,
        uses_kwargs=True,
        input_model="StrictInput",
        input_module="__test_strict_input__",
    )

    agent = MagicMock()
    agent.explain = AsyncMock(return_value={"narrative": "ok"})

    dispatcher = DispatcherNode(agent_registry={"explainer": agent})
    state = {
        "query": "explain something",
        "dispatch_plan": [
            {
                "agent_name": "explainer",
                "priority": "critical",
                "parameters": {"required_field": ""},  # min_length=1 → ValidationError
                "timeout_ms": 5000,
                "fallback_agent": None,
            }
        ],
        "parallel_groups": [["explainer"]],
    }

    try:
        result = await dispatcher.execute(state)
    finally:
        if original is not None:
            mm.AGENT_METHOD_MAP["explainer"] = original

    agent_result = result["agent_results"][0]
    assert agent_result["success"] is False, agent_result
    # The error message should mention either the input model or validation.
    error_msg = (agent_result["error"] or "").lower()
    assert "validation" in error_msg or "strictinput" in error_msg or "min_length" in error_msg, (
        f"expected validation-related error, got: {error_msg}"
    )
