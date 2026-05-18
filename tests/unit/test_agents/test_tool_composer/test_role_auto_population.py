"""S14 + Phase 7.2 forcing tests for tool-composer role auto-population.

Plan reference: ``.claude/plans/causal_role_propagation_FINAL.md`` §7.2
+ v3 §7.3 (3 case Vector). S14 line 420.

**S14**: extract ``experiment_id`` from the existing
``context: Dict[str, Any]`` carrier at
``src/agents/tool_composer/executor.py`` (the executor.execute seam) so
Phase 7.2's pre-fill hook has access to it.

**Phase 7.2**: when the planned tool accepts ``confounders`` and the
caller did NOT supply an explicit value AND ``experiment_id`` is
present in the context, pre-fill the ``confounders`` parameter from
``query_active_role_attributions(experiment_id)`` filtered to
``causal_role == "confounder"``. Explicit caller value always wins.
Repository returning ``evaluator_satisfied=False`` rows is impossible
under the default gate (Phase 7.1 filters), but is rechecked here for
defense-in-depth (C1 trust-gate consistency).

Tests stub the repository at the import-into-executor boundary so the
SQL layer is not exercised — Phase 7.1 owns SQL falsifiability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DependencyType,
    ExecutionPlan,
    ExecutionStep,
)
from src.data.role_attribution import RoleAttribution
from src.tool_registry.registry import ToolParameter, ToolRegistry, ToolSchema

# ----------------------------------------------------------------------------
# Helpers — minimal registry with a confounders-accepting tool and a
# capture-call tool that records the kwargs it receives.
# ----------------------------------------------------------------------------


def _make_confounder_aware_registry(captured: dict[str, Any]) -> ToolRegistry:
    """Build a registry with ``causal_effect_estimator`` whose callable
    records its received kwargs into ``captured``.
    """
    registry = ToolRegistry()
    registry.clear()

    def estimator(
        treatment: str = "T",
        outcome: str = "Y",
        confounders: List[str] | None = None,
        method: str = "backdoor.linear_regression",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        captured["confounders"] = confounders
        captured["treatment"] = treatment
        captured["outcome"] = outcome
        captured["all_kwargs"] = {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "method": method,
            **kwargs,
        }
        return {"ate": 0.12}

    schema = ToolSchema(
        name="causal_effect_estimator",
        description="Estimate ATE.",
        source_agent="causal_impact",
        tier=2,
        input_parameters=[
            ToolParameter("treatment", "str", "Treatment", True),
            ToolParameter("outcome", "str", "Outcome", True),
            ToolParameter("confounders", "List[str]", "Confounders", False),
        ],
        output_schema="EffectEstimate",
        avg_execution_ms=100,
    )
    registry.register(schema=schema, callable=estimator)
    return registry


def _make_step_without_confounders() -> ExecutionStep:
    """An execution step that does NOT supply ``confounders`` in the
    input_mapping — the seam where Phase 7.2 must fire.
    """
    return ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="causal_effect_estimator",
        source_agent="causal_impact",
        input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
        dependency_type=DependencyType.PARALLEL,
        depends_on_steps=[],
    )


def _make_step_with_explicit_confounders() -> ExecutionStep:
    return ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name="causal_effect_estimator",
        source_agent="causal_impact",
        input_mapping={
            "treatment": "rep_visits",
            "outcome": "rx_volume",
            "confounders": ["age_explicit", "gender_explicit"],
        },
        dependency_type=DependencyType.PARALLEL,
        depends_on_steps=[],
    )


def _make_plan(step: ExecutionStep) -> ExecutionPlan:
    from src.agents.tool_composer.models.composition_models import (
        DecompositionResult,
        SubQuestion,
        ToolMapping,
    )

    decomposition = DecompositionResult(
        original_query="What is the causal effect?",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question="effect?",
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
                tool_name="causal_effect_estimator",
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


def _attr(
    feature: str,
    causal_role: str,
    *,
    source: str = "llm",
    satisfied: bool = True,
) -> RoleAttribution:
    return RoleAttribution(
        feature=feature,
        causal_role=causal_role,
        source=source,  # type: ignore[typeddict-item]
        evaluator_satisfied=satisfied,
        evaluator_model="haiku" if source == "llm" else "n/a",
    )


# ----------------------------------------------------------------------------
# Case 1 (S14 + 7.2 happy path): tool has no explicit confounders, repo
# returns a satisfied confounder → tool receives the auto-populated list.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_populates_confounders_when_caller_omits_them() -> None:
    """Falsifiability: revert the executor's pre-fill hook (or skip it
    when ``experiment_id`` is None) — captured["confounders"] is None,
    not ["X"]. This is the load-bearing v3 §7.3 case 1.
    """
    captured: dict[str, Any] = {}
    registry = _make_confounder_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step_without_confounders())

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[
            _attr("X", "confounder", source="manifest", satisfied=True),
            _attr("Z", "instrument", source="llm", satisfied=True),
        ],
    ) as mock_query:
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    # S14 verification: the experiment_id reached the repository.
    assert mock_query.called, "repository must be queried when experiment_id is in context"
    assert mock_query.call_args[0][0] == "exp-001", (
        f"experiment_id must be passed positional; got: {mock_query.call_args}"
    )
    # Phase 7.2 verification: only the confounder-role attribution
    # surfaces, NOT the instrument.
    assert captured["confounders"] == ["X"], (
        f"confounders must be auto-populated with the manifest "
        f"confounder; got: {captured.get('confounders')}"
    )


# ----------------------------------------------------------------------------
# Case 2 (v3 §7.3 case 2): caller-supplied explicit confounders take
# precedence over the auto-population.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_confounders_win_over_auto_population() -> None:
    """Falsifiability: a buggy implementation that unconditionally
    overwrites the resolved ``confounders`` kwarg from the repository
    would trip this assertion. Explicit-caller-wins is the C1 contract
    (the caller is the most-authoritative source).
    """
    captured: dict[str, Any] = {}
    registry = _make_confounder_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step_with_explicit_confounders())

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[_attr("auto_X", "confounder")],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    assert captured["confounders"] == ["age_explicit", "gender_explicit"], (
        f"explicit caller value must win; got: {captured.get('confounders')}"
    )


# ----------------------------------------------------------------------------
# Case 3 (v3 §7.3 case 3): repo returns an UNSATISFIED LLM attribution
# (evaluator_satisfied=False). Phase 7.2 must drop it — the auto-pop
# list is empty.
#
# Note: Phase 7.1's default ``only_evaluator_satisfied=True`` would have
# filtered this row at the SQL layer, but Phase 7.2 must NOT rely on
# that — a future caller passing ``only_evaluator_satisfied=False``
# would otherwise leak unverified roles into tools. Defense-in-depth at
# the consumer boundary.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsatisfied_llm_attribution_is_dropped_by_auto_population() -> None:
    """Falsifiability: remove the ``should_act`` / evaluator_satisfied
    check in the executor pre-fill hook — captured["confounders"]
    becomes ["leaked"], violating C1 at the tool-call boundary.
    """
    captured: dict[str, Any] = {}
    registry = _make_confounder_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step_without_confounders())

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
        return_value=[
            _attr("leaked", "confounder", source="llm", satisfied=False),
        ],
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    # Either explicitly empty list or None — both honor C1. We pin
    # "empty list" here to make the contract crisp: the hook DID fire
    # (None would suggest the hook never ran), it just filtered the
    # unsatisfied row out.
    assert captured["confounders"] == [], (
        f"unsatisfied LLM attribution must be filtered; got: {captured.get('confounders')}"
    )


# ----------------------------------------------------------------------------
# Case 4 (S14 isolation): no experiment_id in context → repo is not
# called, confounders remain at the caller-supplied (or None) state.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_experiment_id_skips_repository_query() -> None:
    """Falsifiability: a buggy impl that calls the repo with None
    experiment_id (e.g. forgetting the ``if experiment_id is None``
    guard) would trip this — repo gets called, possibly with a SQL
    error in production. The guard is the S14 propagation discipline.
    """
    captured: dict[str, Any] = {}
    registry = _make_confounder_aware_registry(captured)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False)
    plan = _make_plan(_make_step_without_confounders())

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
    ) as mock_query:
        # No experiment_id in context.
        await executor.execute(plan, context={"brand": "acme"})

    assert not mock_query.called, "repository must not be queried when experiment_id is absent"
    # confounders not touched — receives None (the registered default).
    assert captured["confounders"] is None


# ----------------------------------------------------------------------------
# Case 5 (Phase 7.2 scope): tool does NOT accept confounders → no
# pre-fill attempted (no key error, no spurious kwargs).
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_without_confounders_parameter_is_untouched() -> None:
    """Falsifiability: an impl that injects ``confounders`` into every
    tool's resolved_inputs would either (a) blow up when the tool's
    signature doesn't accept the kwarg, or (b) silently bypass the
    parameter-list contract. The hook MUST inspect the schema's
    input_parameters before injecting.
    """
    captured: dict[str, Any] = {}
    registry = ToolRegistry()
    registry.clear()

    def no_confounders_tool(metric: str, **kwargs: Any) -> Dict[str, Any]:
        captured["metric"] = metric
        captured["kwargs"] = kwargs
        return {"value": 1}

    registry.register(
        schema=ToolSchema(
            name="gap_calculator",
            description="t",
            source_agent="gap_analyzer",
            tier=2,
            input_parameters=[ToolParameter("metric", "str", "m", True)],
            output_schema="GapAnalysis",
            avg_execution_ms=100,
        ),
        callable=no_confounders_tool,
    )

    executor = PlanExecutor(tool_registry=registry, enable_caching=False)

    from src.agents.tool_composer.models.composition_models import (
        DecompositionResult,
        SubQuestion,
        ToolMapping,
    )

    plan = ExecutionPlan(
        decomposition=DecompositionResult(
            original_query="q",
            sub_questions=[
                SubQuestion(
                    id="sq_1",
                    question="?",
                    intent="DESCRIPTIVE",
                    entities=[],
                    depends_on=[],
                ),
            ],
            decomposition_reasoning="t",
            timestamp=datetime.now(timezone.utc),
        ),
        steps=[
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="gap_calculator",
                source_agent="gap_analyzer",
                input_mapping={"metric": "sales"},
                dependency_type=DependencyType.PARALLEL,
                depends_on_steps=[],
            ),
        ],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="gap_calculator",
                source_agent="gap_analyzer",
                confidence=0.9,
                reasoning="t",
            ),
        ],
        estimated_duration_ms=100,
        parallel_groups=[["step_1"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )

    with patch(
        "src.agents.tool_composer.executor.query_active_role_attributions",
    ):
        await executor.execute(plan, context={"experiment_id": "exp-001"})

    # The repo MAY be called (the executor doesn't have to know in
    # advance that this tool has no confounders param) OR MAY NOT —
    # both are acceptable. What's NOT acceptable is injecting the
    # ``confounders`` kwarg into a tool that doesn't accept it.
    assert "confounders" not in captured["kwargs"], (
        f"confounders must not be injected into tools that don't accept "
        f"the parameter; got kwargs: {captured.get('kwargs')}"
    )
