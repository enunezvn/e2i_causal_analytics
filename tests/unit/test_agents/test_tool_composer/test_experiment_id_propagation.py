"""S14 — experiment_id propagation tests for tool_composer.

Issue: #360 (tool_composer: extract experiment_id from existing context
dict carrier).

Acceptance per issue body:

    Test: assert ``composer.compose(query, context={"experiment_id":
    "exp-1", ...})`` propagates ``experiment_id`` to the executed tool's
    invocation kwargs.

The fix mirrors the existing ``_maybe_autopopulate_confounders`` pattern
from PR #367 / Phase 7.2: when a tool's schema declares ``experiment_id``
as a parameter AND the caller did NOT supply an explicit value AND the
context carries an ``experiment_id``, pre-fill the kwarg from the
context. Explicit caller value always wins (C1 trust-gate parity).

These tests pin the propagation contract at the executor seam — the
exact place issue #360 references (``executor.py:431-433`` per the
issue body; the actual line is inside ``_execute_step`` where
``resolved_inputs`` is built before the tool callable is invoked).

Falsifiability: each test enumerates an exact way the implementation
could regress (mutation testing one-step ahead).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch

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

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_experiment_aware_registry(captured: dict[str, Any]) -> ToolRegistry:
    """Build a registry with a tool that declares ``experiment_id`` as a
    parameter and records the kwargs it receives.
    """
    registry = ToolRegistry()
    registry.clear()

    def role_query_tool(
        feature_name: str = "X",
        experiment_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        captured["experiment_id"] = experiment_id
        captured["feature_name"] = feature_name
        captured["all_kwargs"] = {
            "feature_name": feature_name,
            "experiment_id": experiment_id,
            **kwargs,
        }
        return {"role": "confounder"}

    schema = ToolSchema(
        name="role_query_tool",
        description="Query causal-role attributions for a feature.",
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter("feature_name", "str", "Feature name", True),
            ToolParameter("experiment_id", "str", "Experiment ID carrier", False),
        ],
        output_schema="RoleQueryResult",
        avg_execution_ms=50,
    )
    registry.register(schema=schema, callable=role_query_tool)
    return registry


def _make_experiment_unaware_registry(captured: dict[str, Any]) -> ToolRegistry:
    """Build a registry with a tool that does NOT declare ``experiment_id``
    as a parameter — used to verify the kwarg is not injected blindly.
    """
    registry = ToolRegistry()
    registry.clear()

    def plain_tool(metric: str = "sales", **kwargs: Any) -> Dict[str, Any]:
        captured["metric"] = metric
        captured["kwargs"] = kwargs
        return {"value": 1}

    schema = ToolSchema(
        name="plain_tool",
        description="Tool without an experiment_id parameter.",
        source_agent="gap_analyzer",
        tier=2,
        input_parameters=[ToolParameter("metric", "str", "Metric", True)],
        output_schema="GapAnalysis",
        avg_execution_ms=50,
    )
    registry.register(schema=schema, callable=plain_tool)
    return registry


def _make_step(
    tool_name: str,
    input_mapping: Dict[str, Any],
) -> ExecutionStep:
    return ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name=tool_name,
        source_agent="causal_impact",
        input_mapping=input_mapping,
        dependency_type=DependencyType.PARALLEL,
        depends_on_steps=[],
    )


def _make_plan(step: ExecutionStep) -> ExecutionPlan:
    decomposition = DecompositionResult(
        original_query="What is the experiment id propagation?",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question="prop?",
                intent="CAUSAL",
                entities=[],
                depends_on=[],
            ),
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
                tool_name=step.tool_name,
                source_agent=step.source_agent,
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        parallel_groups=[["step_1"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


# ----------------------------------------------------------------------------
# Case 1 (S14 happy path): tool declares experiment_id, caller omits it,
# context carries it → tool receives the auto-populated kwarg.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_id_propagates_to_tool_kwargs_when_declared() -> None:
    """Falsifiability: if the executor does NOT auto-inject experiment_id
    into ``resolved_inputs`` when the tool schema declares it,
    ``captured['experiment_id']`` would be ``None`` (the registered
    default), not ``"exp-001"``.
    """
    captured: dict[str, Any] = {}
    registry = _make_experiment_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step("role_query_tool", {"feature_name": "rep_visits"}))

    # Patch the role-attributions repo so the Phase 7.2 hook (also reading
    # context["experiment_id"]) doesn't error out on the unrelated path.
    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    assert captured["experiment_id"] == "exp-001", (
        f"experiment_id must propagate to tool kwargs from context; "
        f"got: {captured.get('experiment_id')!r}"
    )


# ----------------------------------------------------------------------------
# Case 2 (caller-explicit wins): when input_mapping already supplies
# experiment_id, the context value MUST NOT overwrite it.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_experiment_id_in_input_mapping_wins() -> None:
    """Falsifiability: a buggy implementation that unconditionally
    overwrites ``resolved_inputs["experiment_id"]`` from the context
    would trip this assertion. Explicit-caller-wins is the C1 contract
    parity with confounders auto-pop (the most-authoritative source).
    """
    captured: dict[str, Any] = {}
    registry = _make_experiment_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(
        _make_step(
            "role_query_tool",
            {"feature_name": "rep_visits", "experiment_id": "exp-explicit"},
        )
    )

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-context"})

    assert captured["experiment_id"] == "exp-explicit", (
        f"explicit caller experiment_id must win over context; "
        f"got: {captured.get('experiment_id')!r}"
    )


# ----------------------------------------------------------------------------
# Case 3 (S14 isolation): no experiment_id in context → tool keeps
# default None for the declared parameter.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absent_experiment_id_leaves_tool_default() -> None:
    """Falsifiability: a buggy implementation that injects a default
    string (e.g. ``""``) or otherwise touches the kwarg when the
    context does NOT carry an experiment_id would trip this.
    """
    captured: dict[str, Any] = {}
    registry = _make_experiment_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step("role_query_tool", {"feature_name": "rep_visits"}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"brand": "acme"})

    assert captured["experiment_id"] is None, (
        f"absent context experiment_id must leave tool default; "
        f"got: {captured.get('experiment_id')!r}"
    )


# ----------------------------------------------------------------------------
# Case 4 (schema gate): tool does NOT declare experiment_id → no
# injection (no spurious kwargs that the tool's signature wouldn't
# accept).
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_without_experiment_id_param_is_untouched() -> None:
    """Falsifiability: an implementation that injects ``experiment_id``
    into every tool's resolved_inputs would (a) blow up if the tool's
    signature didn't accept ``**kwargs``, or (b) silently bypass the
    schema's parameter-list contract. The hook MUST inspect the
    schema's input_parameters before injecting.
    """
    captured: dict[str, Any] = {}
    registry = _make_experiment_unaware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step("plain_tool", {"metric": "sales"}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    assert "experiment_id" not in captured["kwargs"], (
        f"experiment_id must not be injected into tools that don't "
        f"declare the parameter; got kwargs: {captured.get('kwargs')}"
    )


# ----------------------------------------------------------------------------
# Case 5 (empty-string guard): an empty experiment_id is treated as
# absent — mirrors the S14 gate in ``_maybe_autopopulate_confounders``
# (``not isinstance(...) or not experiment_id``).
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_experiment_id_treated_as_absent() -> None:
    """Falsifiability: an implementation using ``"experiment_id" in
    context`` (key-presence) instead of ``context.get(...)`` + truthy
    check would inject ``""`` here, breaking parity with the S14 gate
    already in ``_maybe_autopopulate_confounders``.
    """
    captured: dict[str, Any] = {}
    registry = _make_experiment_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step("role_query_tool", {"feature_name": "rep_visits"}))

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[],
    ):
        await executor.execute(plan, context={"experiment_id": ""})

    assert captured["experiment_id"] is None, (
        f"empty-string experiment_id must be treated as absent; "
        f"got: {captured.get('experiment_id')!r}"
    )


# ----------------------------------------------------------------------------
# Case 6 (regression guard): the confounders auto-pop still works
# alongside the new experiment_id propagation — both must coexist.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_id_and_confounders_coexist() -> None:
    """Falsifiability: a refactor that consolidates both hooks but
    accidentally drops one (e.g. iterating over a single param list and
    ``return``-ing after the first hit) would trip this.
    """
    captured: dict[str, Any] = {}
    registry = ToolRegistry()
    registry.clear()

    def combined_tool(
        treatment: str = "T",
        outcome: str = "Y",
        confounders: List[str] | None = None,
        experiment_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        captured["confounders"] = confounders
        captured["experiment_id"] = experiment_id
        return {"ate": 0.1}

    schema = ToolSchema(
        name="combined_tool",
        description="Tool with both confounders and experiment_id.",
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter("treatment", "str", "Treatment", True),
            ToolParameter("outcome", "str", "Outcome", True),
            ToolParameter("confounders", "List[str]", "Confounders", False),
            ToolParameter("experiment_id", "str", "Experiment ID", False),
        ],
        output_schema="EffectEstimate",
        avg_execution_ms=100,
    )
    registry.register(schema=schema, callable=combined_tool)

    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(
        _make_step("combined_tool", {"treatment": "rep_visits", "outcome": "rx_volume"})
    )

    from src.data.role_attribution import RoleAttribution

    def _attr(feature: str, role: str) -> RoleAttribution:
        return RoleAttribution(
            feature=feature,
            causal_role=role,
            source="manifest",  # type: ignore[typeddict-item]
            evaluator_satisfied=True,
            evaluator_model="n/a",
        )

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[_attr("age", "confounder")],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    assert captured["experiment_id"] == "exp-001", (
        f"experiment_id must propagate alongside confounders auto-pop; "
        f"got: {captured.get('experiment_id')!r}"
    )
    assert captured["confounders"] == ["age"], (
        f"confounders auto-pop must still fire when experiment_id also "
        f"auto-pops; got: {captured.get('confounders')}"
    )
