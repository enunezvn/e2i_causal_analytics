"""F2-core — executor threads a real context DataFrame into tool kwargs.

The composable tools read the frame from ``**kwargs`` via
``_extract_dataframe_from_kwargs`` (tool_registrations.py:606-622), which
scans ``_DATAFRAME_KWARGS_KEYS = ("data", "dataframe", "estimation_data")``.
Before this fix NO production path delivered a DataFrame: the executor only
resolved the planner's ``input_mapping`` (which omits ``data``) and never
read the frame from ``context``. These tests pin the auto-injection seam at
``_execute_step`` — right after the experiment_id auto-pop block.

The injection key is ``estimation_data`` (NOT ``data``): ``discover_dag``'s
``DiscoverDagInput.data`` is a Dict and must not receive a DataFrame.

Falsifiability: each test enumerates an exact regression path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStep,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolParameter, ToolRegistry, ToolSchema


def _make_data_consuming_registry(captured: Dict[str, Any]) -> ToolRegistry:
    """A tool that records the DataFrame it receives via **kwargs.

    Mirrors the real composable tools: the frame arrives in ``**kwargs``
    under one of the canonical keys, NOT as a named positional param.
    """
    registry = ToolRegistry()
    registry.clear()

    def cate_like_tool(treatment: str = "rx", **kwargs: Any) -> Dict[str, Any]:
        # Mirror _extract_dataframe_from_kwargs scan order.
        frame = None
        for key in ("data", "dataframe", "estimation_data"):
            candidate = kwargs.get(key)
            if candidate is not None and hasattr(candidate, "columns"):
                frame = candidate
                break
        captured["frame"] = frame
        captured["frame_id"] = id(frame) if frame is not None else None
        captured["kwargs_keys"] = sorted(kwargs.keys())
        return {"n_rows": int(len(frame)) if frame is not None else 0}

    schema = ToolSchema(
        name="cate_like_tool",
        description="Estimates CATE; consumes a DataFrame from kwargs.",
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter("treatment", "str", "Treatment column", True),
        ],
        output_schema="CateResult",
        avg_execution_ms=50,
    )
    registry.register(schema=schema, callable=cate_like_tool)
    return registry


def _make_step(input_mapping: Dict[str, Any]) -> ExecutionStep:
    return ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="cate_like_tool",
        source_agent="causal_impact",
        input_mapping=input_mapping,
        dependency_type=DependencyType.PARALLEL,
        depends_on_steps=[],
    )


def _make_plan(step: ExecutionStep) -> ExecutionPlan:
    decomposition = DecompositionResult(
        original_query="What is the CATE of rx?",
        sub_questions=[
            SubQuestion(id="sq_1", question="cate?", intent="CAUSAL", entities=[], depends_on=[]),
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    return ExecutionPlan(
        decomposition=decomposition,
        steps=[step],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="cate_like_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        parallel_groups=[["step_1"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_context_dataframe_is_injected_when_input_mapping_omits_it() -> None:
    """Falsifiability: without auto-injection the tool sees NO frame
    (input_mapping omits data) and ``captured['frame']`` is ``None``.
    """
    df = pd.DataFrame({"rx": [0, 1, 1, 0], "outcome": [1, 0, 1, 1]})
    captured: Dict[str, Any] = {}
    registry = _make_data_consuming_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step({"treatment": "rx"}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"estimation_data": df})

    assert captured["frame"] is not None, "context DataFrame must reach tool kwargs"
    assert captured["frame_id"] == id(df), "the SAME frame object must be injected (no copy)"
    assert "estimation_data" in captured["kwargs_keys"], (
        f"frame must be injected under 'estimation_data'; got keys {captured['kwargs_keys']!r}"
    )


@pytest.mark.asyncio
async def test_explicit_estimation_data_in_input_mapping_is_not_overridden() -> None:
    """Falsifiability: a buggy hook that unconditionally overwrites
    ``resolved_inputs['estimation_data']`` from context would replace the
    caller's explicit frame. Caller-explicit wins (C1 trust-gate parity).
    """
    explicit_df = pd.DataFrame({"rx": [1, 1], "outcome": [0, 1]})
    context_df = pd.DataFrame({"rx": [0, 0, 0], "outcome": [1, 1, 1]})
    captured: Dict[str, Any] = {}
    registry = _make_data_consuming_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    # Caller supplies an explicit estimation_data via input_mapping.
    plan = _make_plan(_make_step({"treatment": "rx", "estimation_data": explicit_df}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"estimation_data": context_df})

    assert captured["frame_id"] == id(explicit_df), (
        "explicit caller estimation_data must win over the context frame"
    )


@pytest.mark.asyncio
async def test_no_frame_in_context_leaves_tool_without_one() -> None:
    """Falsifiability: a hook that injects a non-DataFrame context value
    (e.g. a dict) under a DataFrame key would trip the duck-typed gate.
    """
    captured: Dict[str, Any] = {}
    registry = _make_data_consuming_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step({"treatment": "rx"}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        # context carries a NON-DataFrame under a canonical key.
        await executor.execute(plan, context={"data": {"not": "a frame"}})

    assert captured["frame"] is None, "non-DataFrame context value must NOT be injected"
    assert "estimation_data" not in captured["kwargs_keys"], (
        "no estimation_data kwarg should be added when context has no real frame"
    )


# ---------------------------------------------------------------------------
# T2 — ToolComposerAgent.run threads input_data["data"] / ["data_source"]
# into the merged_context the composer receives.
#
# The composer is replaced with a lightweight spy returning a SimpleNamespace
# stand-in (the real CompositionResult requires fully-built decomposition/plan/
# execution/response sub-models; run() reads every field behind an
# ``if result.X else`` guard, so a namespace with the attributes run() touches
# is a faithful test double for the injected composer dependency).
# ---------------------------------------------------------------------------
def _spy_result() -> SimpleNamespace:
    return SimpleNamespace(
        success=False,
        response=None,
        composition_id="comp_spy",
        decomposition=None,
        plan=None,
        execution=None,
        status=None,
        total_duration_ms=0,
        phase_durations={},
    )


@pytest.mark.asyncio
async def test_agent_run_threads_data_into_composer_context() -> None:
    """Falsifiability: if run() does NOT copy input_data['data'] into
    merged_context['estimation_data'], the spy composer would see no frame
    and no data_source — the executor downstream could never auto-inject.
    """
    from src.agents.tool_composer.agent import ToolComposerAgent

    df = pd.DataFrame({"rx": [0, 1], "outcome": [1, 0]})
    seen: Dict[str, Any] = {}

    class _SpyComposer:
        async def compose(self, query: str, context: Dict[str, Any]) -> SimpleNamespace:
            seen["context"] = context
            return _spy_result()

    agent = ToolComposerAgent()
    agent._composer = _SpyComposer()  # bypass real LLM init

    await agent.run(
        {
            "query": "What is the CATE of rx?",
            "data": df,
            "data_source": "optum_mart",
        }
    )

    ctx = seen["context"]
    assert "estimation_data" in ctx, "run() must normalize input_data['data'] -> estimation_data"
    assert id(ctx["estimation_data"]) == id(df), "the SAME frame object must be threaded through"
    assert ctx.get("data_source") == "optum_mart", "data_source must pass through to the context"


@pytest.mark.asyncio
async def test_agent_run_without_data_leaves_context_frameless() -> None:
    """Falsifiability: a buggy run() that writes estimation_data=None
    unconditionally would put a None frame into context and trip the
    executor's duck-typed gate downstream.
    """
    from src.agents.tool_composer.agent import ToolComposerAgent

    seen: Dict[str, Any] = {}

    class _SpyComposer:
        async def compose(self, query: str, context: Dict[str, Any]) -> SimpleNamespace:
            seen["context"] = context
            return _spy_result()

    agent = ToolComposerAgent()
    agent._composer = _SpyComposer()

    await agent.run({"query": "no data here"})

    assert "estimation_data" not in seen["context"], (
        "run() must NOT inject an estimation_data key when no data was provided"
    )
