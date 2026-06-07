"""F5 — a step that depends on a FAILED upstream step is SKIPPED, not crashed.

Before this fix, when step_1 failed it never entered ``outputs`` (only
successful steps are stored — executor.py:455,465-466), so step_2's
``$step_1.field`` reference resolved to None and the tool then crashed on
None (AttributeError) or ran with garbage inputs. The fix records failed
step_ids and short-circuits dependents to ExecutionStatus.SKIPPED with a
clear 'dependency unmet' error, WITHOUT invoking the tool.

Falsifiability: each test pins an exact regression path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolParameter, ToolRegistry, ToolSchema


def _make_two_step_registry(invoked: Dict[str, int]) -> ToolRegistry:
    """step_1's tool always RAISES; step_2's tool records that it ran."""
    registry = ToolRegistry()
    registry.clear()

    def failing_tool(**kwargs: Any) -> Dict[str, Any]:
        invoked["failing_tool"] = invoked.get("failing_tool", 0) + 1
        raise RuntimeError("upstream boom")

    def dependent_tool(upstream: Any = None, **kwargs: Any) -> Dict[str, Any]:
        invoked["dependent_tool"] = invoked.get("dependent_tool", 0) + 1
        # Would crash on None if it ran: upstream.some_attr — but it must NOT run.
        return {"used": upstream.nonexistent_attr}  # noqa: F821 — never reached

    registry.register(
        schema=ToolSchema(
            name="failing_tool",
            description="Always raises.",
            source_agent="causal_impact",
            tier=2,
            input_parameters=[],
            output_schema="X",
            avg_execution_ms=10,
        ),
        callable=failing_tool,
    )
    registry.register(
        schema=ToolSchema(
            name="dependent_tool",
            description="Depends on step_1 output.",
            source_agent="causal_impact",
            tier=2,
            input_parameters=[ToolParameter("upstream", "str", "upstream ref", False)],
            output_schema="Y",
            avg_execution_ms=10,
        ),
        callable=dependent_tool,
    )
    return registry


def _make_dependent_plan() -> ExecutionPlan:
    decomposition = DecompositionResult(
        original_query="chained?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[]),
            SubQuestion(
                id="sq_2", question="q2", intent="CAUSAL", entities=[], depends_on=["sq_1"]
            ),
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    step_1 = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="failing_tool",
        source_agent="causal_impact",
        input_mapping={},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=[],
    )
    step_2 = ExecutionStep(
        step_id="step_2",
        sub_question_id="sq_2",
        tool_name="dependent_tool",
        source_agent="causal_impact",
        input_mapping={"upstream": "$step_1.value"},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=["step_1"],
    )
    return ExecutionPlan(
        decomposition=decomposition,
        steps=[step_1, step_2],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="failing_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
            ToolMapping(
                sub_question_id="sq_2",
                tool_name="dependent_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        # Two separate waves so step_1's failure is recorded before step_2 runs.
        parallel_groups=[["step_1"], ["step_2"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_dependent_of_failed_step_is_skipped_not_crashed() -> None:
    """Falsifiability: without the skip guard the executor either (a) raises
    AttributeError when dependent_tool runs on a None upstream, or (b)
    records step_2 as FAILED via the generic retry path. Either way the
    tool would be INVOKED. The fix must short-circuit BEFORE invocation."""
    invoked: Dict[str, int] = {}
    registry = _make_two_step_registry(invoked)
    # No retries so the failing step fails fast.
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)
    plan = _make_dependent_plan()

    trace = await executor.execute(plan, context={})

    step_1_result = trace.get_result("step_1")
    step_2_result = trace.get_result("step_2")

    assert step_1_result is not None and step_2_result is not None
    assert step_1_result.status == ExecutionStatus.FAILED
    assert step_2_result.status == ExecutionStatus.SKIPPED, (
        f"dependent step must be SKIPPED, got {step_2_result.status!r}"
    )
    # The dependent tool must NOT have been invoked at all.
    assert invoked.get("dependent_tool", 0) == 0, "dependent tool must not run when upstream failed"
    # The skip reason must name the unmet upstream.
    assert "step_1" in (step_2_result.output.error or ""), (
        f"skip error must name the unmet dependency; got {step_2_result.output.error!r}"
    )
    assert "dependency" in (step_2_result.output.error or "").lower()


@pytest.mark.asyncio
async def test_independent_step_still_runs_when_a_sibling_fails() -> None:
    """Falsifiability: a too-broad skip (e.g. skipping every later step once
    ANY step failed) would wrongly SKIP a step that does NOT depend on the
    failed one. Only true dependents are skipped."""
    invoked: Dict[str, int] = {}
    registry = _make_two_step_registry(invoked)

    def ok_tool(**kwargs: Any) -> Dict[str, Any]:
        invoked["ok_tool"] = invoked.get("ok_tool", 0) + 1
        return {"value": 42}

    registry.register(
        schema=ToolSchema(
            name="ok_tool",
            description="Independent, succeeds.",
            source_agent="causal_impact",
            tier=2,
            input_parameters=[],
            output_schema="Z",
            avg_execution_ms=10,
        ),
        callable=ok_tool,
    )

    decomposition = DecompositionResult(
        original_query="mixed?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[]),
            SubQuestion(id="sq_3", question="q3", intent="CAUSAL", entities=[], depends_on=[]),
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    step_1 = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="failing_tool",
        source_agent="causal_impact",
        input_mapping={},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=[],
    )
    step_3 = ExecutionStep(
        step_id="step_3",
        sub_question_id="sq_3",
        tool_name="ok_tool",
        source_agent="causal_impact",
        input_mapping={},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=[],  # independent of step_1
    )
    plan = ExecutionPlan(
        decomposition=decomposition,
        steps=[step_1, step_3],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="failing_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
            ToolMapping(
                sub_question_id="sq_3",
                tool_name="ok_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        parallel_groups=[["step_1"], ["step_3"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )

    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)
    trace = await executor.execute(plan, context={})

    assert trace.get_result("step_1").status == ExecutionStatus.FAILED
    assert trace.get_result("step_3").status == ExecutionStatus.COMPLETED, (
        "an independent step must still run after a sibling fails"
    )
    assert invoked.get("ok_tool", 0) == 1


@pytest.mark.asyncio
async def test_three_level_transitive_skip() -> None:
    """A SKIPPED step's OWN dependents must also be SKIPPED (transitive cascade):
    step_1 FAILS -> step_2 SKIPPED -> step_3 SKIPPED. This proves SKIPPED results
    (output.success == False) are added to failed_step_ids, so the skip
    propagates across more than one level. Without that, step_3 would run on a
    None upstream and crash."""
    invoked: Dict[str, int] = {}
    registry = _make_two_step_registry(invoked)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)

    decomposition = DecompositionResult(
        original_query="3-level chain?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[]),
            SubQuestion(
                id="sq_2", question="q2", intent="CAUSAL", entities=[], depends_on=["sq_1"]
            ),
            SubQuestion(
                id="sq_3", question="q3", intent="CAUSAL", entities=[], depends_on=["sq_2"]
            ),
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    step_1 = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="failing_tool",
        source_agent="causal_impact",
        input_mapping={},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=[],
    )
    step_2 = ExecutionStep(
        step_id="step_2",
        sub_question_id="sq_2",
        tool_name="dependent_tool",
        source_agent="causal_impact",
        input_mapping={"upstream": "$step_1.value"},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=["step_1"],
    )
    step_3 = ExecutionStep(
        step_id="step_3",
        sub_question_id="sq_3",
        tool_name="dependent_tool",
        source_agent="causal_impact",
        input_mapping={"upstream": "$step_2.value"},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=["step_2"],
    )
    plan = ExecutionPlan(
        decomposition=decomposition,
        steps=[step_1, step_2, step_3],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="failing_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
            ToolMapping(
                sub_question_id="sq_2",
                tool_name="dependent_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
            ToolMapping(
                sub_question_id="sq_3",
                tool_name="dependent_tool",
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        parallel_groups=[["step_1"], ["step_2"], ["step_3"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )

    trace = await executor.execute(plan, context={})

    assert trace.get_result("step_1").status == ExecutionStatus.FAILED
    assert trace.get_result("step_2").status == ExecutionStatus.SKIPPED
    assert trace.get_result("step_3").status == ExecutionStatus.SKIPPED, (
        "a dependent of a SKIPPED step must also be SKIPPED (transitive cascade)"
    )
    # The dependent tool must not run anywhere in a fully-failed chain.
    assert invoked.get("dependent_tool", 0) == 0, "no dependent tool may run in a failed chain"
    # step_3's skip reason names its direct unmet upstream (step_2).
    assert "step_2" in (trace.get_result("step_3").output.error or "")
