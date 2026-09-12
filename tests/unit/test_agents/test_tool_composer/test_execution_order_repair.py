"""Every planned step runs, in dependency order (spec §7.4).

``ExecutionPlan.get_execution_order()`` used to return the LLM's ``parallel_groups`` unchecked
(measured 2026-09-11): a step left out of the groups never ran, a consumer grouped with its
producer ran before its input existed, and with no groups the steps ran in list order even when
a consumer was listed first. Duplicate step ids were not rejected, so the first of two different
steps named ``a`` executed twice.

Now the plan model rejects duplicate ids, unknown dependencies and cycles wherever a plan is
built, and ``get_execution_order()`` returns the given groups only when every step appears
exactly once, no group names an unknown step and every dependency sits in a strictly earlier
group; otherwise the topological levels of ``depends_on_steps``, in plan order within a level,
with ``execution_order_repaired`` naming the violated condition. That is the intent of the dropped
``get_tool_execution_order`` SQL function ("dependency-aware execution DAG"), met at step level.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Sequence

import pandas as pd
import pytest

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.executor import ExecutionError, PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    ORDER_REPAIR_REASONS,
    DecompositionResult,
    ExecutionPlan,
    ExecutionStep,
    SubQuestion,
)
from src.agents.tool_composer.planner import PlanningError, ToolPlanner
from src.agents.tool_composer.serialization import dump_json_safe
from src.tool_registry.registry import ToolSchema, get_registry

REPO_ROOT = Path(__file__).resolve().parents[4]


def _decomposition() -> DecompositionResult:
    return DecompositionResult(
        original_query="what drove it?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[])
        ],
        decomposition_reasoning="t",
    )


def _step(step_id: str, deps: Sequence[str] = (), tool: str = "probe") -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        sub_question_id="sq_1",
        tool_name=tool,
        source_agent="gap_analyzer",
        depends_on_steps=list(deps),
    )


def _plan(steps: List[ExecutionStep], groups: List[List[str]], **extra: Any) -> ExecutionPlan:
    return ExecutionPlan(
        decomposition=_decomposition(),
        steps=steps,
        tool_mappings=[],
        parallel_groups=groups,
        planning_reasoning="t",
        **extra,
    )


# ---------------------------------------------------------------------------
# The three measured probes, and the other violated conditions
# ---------------------------------------------------------------------------


def test_omitted_step_is_scheduled():
    plan = _plan([_step("a"), _step("b", ["a"])], [["a"]])
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "step_missing_from_groups"


def test_consumer_grouped_with_producer_is_split():
    plan = _plan([_step("a"), _step("b", ["a"])], [["a", "b"]])
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "dependency_not_in_earlier_group"


def test_empty_groups_follow_dependencies_not_list_order():
    plan = _plan([_step("b", ["a"]), _step("a")], [])
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "no_groups"


def test_levels_keep_plan_order_and_parallelise_independent_steps():
    plan = _plan([_step("c"), _step("b", ["a"]), _step("a")], [])
    assert plan.get_execution_order() == [["c", "a"], ["b"]]


def test_valid_groups_unchanged():
    groups = [["a", "c"], ["b"]]
    plan = _plan([_step("a"), _step("b", ["a"]), _step("c")], groups)
    assert plan.get_execution_order() == groups
    assert plan.execution_order_repaired is None


def test_unknown_step_in_groups_is_repaired():
    plan = _plan([_step("a"), _step("b", ["a"])], [["a", "ghost"], ["b"]])
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "unknown_step_in_groups"


def test_step_repeated_in_groups_is_repaired():
    plan = _plan([_step("a"), _step("b", ["a"])], [["a"], ["a", "b"]])
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "step_repeated_in_groups"


def test_repair_reasons_are_the_recording_vocabulary():
    # ml/041 keeps an order-repair reason in the recorded plan only if it is one of these.
    sql = (REPO_ROOT / "database" / "ml" / "041_composer_learning_loop_recording.sql").read_text()
    block = re.search(r"execution_order_repaired' IN \((.*?)\)", sql, re.S)
    assert block, "vocabulary not found in ml/041"
    assert set(re.findall(r"'([a-z_]+)'", block.group(1))) == set(ORDER_REPAIR_REASONS)


def test_order_is_recomputed_from_current_state():
    plan = _plan([_step("a"), _step("b", ["a"])], [["a"], ["b"]])
    assert plan.execution_order_repaired is None
    plan.parallel_groups = [["a"]]
    assert plan.get_execution_order() == [["a"], ["b"]]
    assert plan.execution_order_repaired == "step_missing_from_groups"


# ---------------------------------------------------------------------------
# Plan errors, at construction
# ---------------------------------------------------------------------------


def test_duplicate_step_ids_rejected():
    with pytest.raises(ValueError, match="duplicate step id"):
        _plan([_step("a"), _step("a")], [])


def test_unknown_dependency_rejected():
    with pytest.raises(ValueError, match="unknown step"):
        _plan([_step("a", ["ghost"])], [])


@pytest.mark.parametrize(
    "steps",
    [
        [_step("a", ["b"]), _step("b", ["a"])],
        [_step("a", ["a"])],
        [_step("a"), _step("b", ["a", "c"]), _step("c", ["b"])],
    ],
)
def test_cycle_rejected(steps):
    with pytest.raises(ValueError, match="cycle"):
        _plan(steps, [])


@pytest.mark.parametrize("groups", [[], [["a"], ["b"]]])
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda plan: plan.steps[0].depends_on_steps.append("b"), "cycle"),
        (lambda plan: plan.steps[1].depends_on_steps.append("ghost"), "unknown step"),
        (lambda plan: setattr(plan.steps[1], "step_id", "a"), "duplicate step id"),
    ],
)
def test_graph_broken_after_construction_fails_instead_of_scheduling(mutate, message, groups):
    plan = _plan([_step("a"), _step("b", ["a"])], groups)
    mutate(plan)
    with pytest.raises(ValueError, match=message):
        plan.get_execution_order()
    with pytest.raises(ValueError, match=message):
        _ = plan.execution_order_repaired


def test_error_result_plan_still_constructs():
    plan = _plan([], [])
    assert plan.get_execution_order() == []
    assert plan.execution_order_repaired is None


def test_kpi_plan_groups_unchanged():
    frame = pd.DataFrame(
        {
            "converted": [0, 1] * 20,
            "accepted": [1, 0] * 20,
            "confidence_score": [0.5 + 0.01 * i for i in range(40)],
            "trigger_id": [f"T{i}" for i in range(40)],
            "delivery_channel": ["email", "crm", "phone", "portal"] * 10,
        }
    )
    decomposition = DecompositionResult(
        original_query="what drove conversion and which segments respond best",
        sub_questions=[
            SubQuestion(
                id="sq_1", question="drivers?", intent="CAUSAL", entities=[], depends_on=[]
            ),
            SubQuestion(
                id="sq_2", question="segments?", intent="COMPARATIVE", entities=[], depends_on=[]
            ),
        ],
        decomposition_reasoning="t",
    )
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    plan = composer._build_kpi_causal_plan(
        decomposition, {"estimation_data": frame, "kpi_outcome": "converted"}, "converted"
    )
    assert plan is not None
    assert plan.get_execution_order() == plan.parallel_groups
    assert plan.execution_order_repaired is None


async def test_planner_wraps_validator_error_in_planning_error(mock_llm_client, mock_tool_registry):
    # The LLM transport is the conftest double; the planner's parsing, validation and plan
    # construction are real.
    duplicate = {
        "step_id": "step_1",
        "sub_question_id": "sq_1",
        "tool_name": "causal_effect_estimator",
        "input_mapping": {"treatment": "accepted", "outcome": "converted"},
        "depends_on_steps": [],
    }
    mock_llm_client.set_planning_response(
        json.dumps(
            {
                "reasoning": "t",
                "tool_mappings": [
                    {
                        "sub_question_id": "sq_1",
                        "tool_name": "causal_effect_estimator",
                        "confidence": 0.9,
                    }
                ],
                "execution_steps": [duplicate, dict(duplicate)],
                "parallel_groups": [["step_1"]],
            }
        )
    )
    planner = ToolPlanner(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        use_episodic_memory=False,
        enable_caching=False,
    )
    with pytest.raises(PlanningError, match="duplicate step id"):
        await planner.plan(_decomposition())


# ---------------------------------------------------------------------------
# The executor runs the repaired order; plan provenance fields
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    live = get_registry()
    snapshot = live.snapshot()
    live.clear()
    try:
        yield live
    finally:
        live.restore_snapshot(snapshot)


async def test_executor_runs_every_step_of_omitted_step_plan(registry):
    ran: List[str] = []

    async def probe(**kwargs: Any) -> Any:
        return {"ok": True}

    registry.register(
        schema=ToolSchema(
            name="probe",
            description="execution-order probe tool.",
            source_agent="gap_analyzer",
            tier=2,
            avg_execution_ms=10,
        ),
        callable=probe,
    )
    plan = _plan([_step("a"), _step("b", ["a"])], [["a"]])
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)
    trace = await executor.execute(plan, on_step_result=lambda n, r: ran.append(r.step_id))
    assert trace.tools_executed == 2 and trace.tools_succeeded == 2
    assert ran == ["a", "b"]


def test_plan_source_and_cache_key_fields():
    plan = _plan([_step("a")], [["a"]])
    assert plan.plan_source is None and plan.plan_cache_key is None

    tagged = _plan([_step("a")], [["a"]], plan_source="plan_cache", plan_cache_key="sig_123")
    dumped = tagged.model_dump(mode="json")
    assert dumped["plan_source"] == "plan_cache" and dumped["plan_cache_key"] == "sig_123"
    assert dump_json_safe(tagged)["plan_source"] == "plan_cache"
    restored = ExecutionPlan.model_validate(dumped)
    assert (restored.plan_source, restored.plan_cache_key) == ("plan_cache", "sig_123")

    with pytest.raises(ValueError):
        _plan([_step("a")], [["a"]], plan_source="guess")


async def test_executor_runs_no_tool_for_a_graph_broken_after_construction(registry):
    calls: List[int] = []

    async def probe(**kwargs: Any) -> Any:
        calls.append(1)
        return {"ok": True}

    registry.register(
        schema=ToolSchema(
            name="probe",
            description="execution-order probe tool.",
            source_agent="gap_analyzer",
            tier=2,
            avg_execution_ms=10,
        ),
        callable=probe,
    )
    plan = _plan([_step("a"), _step("b", ["a"])], [["a"], ["b"]])
    plan.steps[0].depends_on_steps.append("b")
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)
    with pytest.raises(ExecutionError, match="cycle"):
        await executor.execute(plan)
    assert calls == []
