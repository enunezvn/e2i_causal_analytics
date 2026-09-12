"""The recorder's serializer sends structure only (spec §5.5).

Everything a composition record carries, except the redacted query in the seed, is a registry
identifier, a number or an enum. Anything the LLM or the data authored is dropped or reduced:
sub-questions and steps are positional, intents are normalized, input maps keep only the tool's
declared parameter names, string values become a catalog column name or a length, ``$step``
references become step numbers with a field only when the producer's output model declares it,
output keys are the output model's fields, and no error text is sent. ``to_record`` is the single
path to the RPCs, so a sentinel planted in every LLM- and data-authored position must be absent
from what it returns.

Pure functions over real model objects and the live tool registry; no database.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.agents.tool_composer import composer as _composer  # noqa: F401 - registers every tool
from src.agents.tool_composer.learning_recorder import (
    groups_record,
    plan_record,
    step_record,
    structure_value,
    sub_questions_record,
    to_record,
)
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    ToolInput,
    ToolOutput,
)

SENTINEL = "PT-SENTINEL-9c1"
CATALOG = frozenset({"treatment"})


def _decomposition(intents: List[str], ids: Optional[List[str]] = None) -> DecompositionResult:
    ids = ids or [f"sq_{i}" for i in range(len(intents))]
    return DecompositionResult(
        original_query=f"query about {SENTINEL}",
        sub_questions=[
            SubQuestion(id=sq_id, question=f"{SENTINEL}?", intent=intent, entities=[SENTINEL])
            for sq_id, intent in zip(ids, intents, strict=True)
        ],
        decomposition_reasoning=SENTINEL,
    )


def _plan(
    decomposition: DecompositionResult, steps: List[ExecutionStep], groups: List[List[str]]
) -> ExecutionPlan:
    return ExecutionPlan(
        decomposition=decomposition,
        steps=steps,
        tool_mappings=[],
        parallel_groups=groups,
        planning_reasoning=SENTINEL,
    )


def _step(
    step_id: str,
    tool: str,
    sq_id: str,
    mapping: Dict[str, Any],
    deps: Optional[List[str]] = None,
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        sub_question_id=sq_id,
        tool_name=tool,
        source_agent="causal_impact",
        input_mapping=mapping,
        depends_on_steps=deps or [],
    )


def _result(
    step: ExecutionStep,
    *,
    outcome_class: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    attempts: int = 1,
) -> StepResult:
    started = datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)
    return StepResult(
        step_id=step.step_id,
        sub_question_id=step.sub_question_id,
        tool_name=step.tool_name,
        input=ToolInput(
            tool_name=step.tool_name, parameters={"leak": SENTINEL}, context={"x": SENTINEL}
        ),
        output=ToolOutput(
            tool_name=step.tool_name,
            success=result is not None,
            result=result,
            error=error,
        ),
        status=ExecutionStatus.COMPLETED if result is not None else ExecutionStatus.FAILED,
        started_at=started,
        completed_at=started + timedelta(milliseconds=250),
        outcome_class=outcome_class,
        attempts=attempts,
        error_type=error_type,
    )


def _sentinel_models():
    d = _decomposition(["causal", SENTINEL], ids=[f"{SENTINEL}-sq1", f"{SENTINEL}-sq2"])
    sq1, sq2 = (sq.id for sq in d.sub_questions)
    ate = _step(
        f"{SENTINEL}_ate",
        "causal_effect_estimator",
        sq1,
        {
            "treatment": "treatment",  # a catalog column: kept by name
            "outcome": SENTINEL,  # a frame column that is not in the catalog
            "confounders": {SENTINEL: "x"},  # a dict key
            SENTINEL: 1,  # an undeclared parameter key
        },
    )
    dag = _step(f"{SENTINEL}_dag", "discover_dag", sq1, {"alpha": 0.05})
    rank = _step(
        f"{SENTINEL}_rank",
        "rank_drivers",
        sq2,
        {"dag_edge_list": f"${SENTINEL}_dag.{SENTINEL}", "target": f"$context.{SENTINEL}"},
        deps=[f"{SENTINEL}_dag"],
    )
    gap = _step(
        f"{SENTINEL}_gap", "gap_calculator", sq2, {"metric": SENTINEL, "entities": [SENTINEL]}
    )
    plan = _plan(
        d, [ate, dag, rank, gap], [[ate.step_id, dag.step_id], [rank.step_id], [gap.step_id]]
    )
    trace = ExecutionTrace(plan_id=plan.plan_id)
    trace.add_result(_result(ate, outcome_class="succeeded", result={"ate": 0.1, SENTINEL: 2}))
    trace.add_result(
        _result(
            dag,
            outcome_class="error",
            error=f"KeyError: '{SENTINEL}'",
            error_type="KeyError",
            attempts=3,
        )
    )
    trace.add_result(
        _result(
            rank,
            outcome_class="input_rejected",
            error=f"bad {SENTINEL}",
            error_type="ToolInputError",
        )
    )
    trace.add_result(
        _result(
            gap,
            outcome_class="refused",
            error=f"estimation_data_scope={SENTINEL}",
            error_type="ToolRefusalError",
        )
    )
    return d, plan, trace


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------


def test_sentinel_absent_everywhere():
    d, plan, trace = _sentinel_models()
    record = to_record(decomposition=d, plan=plan, trace=trace, allowlist=CATALOG)
    text = json.dumps(record, allow_nan=False)
    assert SENTINEL not in text
    # Not vacuous: the record carries every part, and the catalog name survived.
    assert set(record) == {"sub_questions", "tool_plan", "parallelizable_groups", "steps"}
    assert len(record["steps"]) == 4 and len(record["tool_plan"]["steps"]) == 4
    assert '"name": "treatment"' in text


def test_record_is_strict_json():
    d, plan, trace = _sentinel_models()
    json.dumps(
        to_record(decomposition=d, plan=plan, trace=trace, allowlist=CATALOG), allow_nan=False
    )


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_catalog_column_kept_and_non_catalog_reduced():
    assert structure_value("treatment", allowlist=CATALOG) == {
        "type": "column",
        "name": "treatment",
    }
    assert structure_value("geo_x", allowlist=CATALOG) == {"type": "str", "len": 5}


def test_allowlist_unavailable_keeps_no_names():
    assert structure_value("treatment", allowlist=None) == {"type": "str", "len": 9}


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.05, 0.05),
        (7, 7),
        (True, True),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (["NE", "SE"], {"type": "list", "len": 2}),
        ({"a": 1, "b": 2}, {"type": "dict", "len": 2}),
        (
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"type": "frame", "rows": 3, "columns": 2},
        ),
    ],
)
def test_values_reduced_to_structure(value, expected):
    assert structure_value(value, allowlist=CATALOG) == expected


def test_numpy_numbers_become_plain_json_numbers():
    import numpy as np

    assert structure_value(np.float64(0.5), allowlist=None) == 0.5
    assert structure_value(np.int64(3), allowlist=None) == 3
    assert structure_value(np.float64(math.nan), allowlist=None) is None
    assert structure_value(np.bool_(True), allowlist=None) is True


# ---------------------------------------------------------------------------
# Positional identity, references, intents, outputs
# ---------------------------------------------------------------------------


def _kpi_like_plan():
    d = _decomposition(["CAUSAL", "COMPARATIVE"])
    dag = _step("kpi_dag", "discover_dag", "sq_0", {})
    rank = _step(
        "kpi_rank",
        "rank_drivers",
        "sq_1",
        {
            "dag_edge_list": "$kpi_dag.edge_list",
            "target": "$kpi_dag.not_an_output_field",
            "importance_percentile": "$context.kpi_outcome",
        },
        deps=["kpi_dag"],
    )
    return d, _plan(d, [dag, rank], [["kpi_dag"], ["kpi_rank"]])


def test_positional_ids_and_remapped_refs():
    d, plan = _kpi_like_plan()
    record = plan_record(plan, allowlist=CATALOG)
    first, second = record["steps"]
    assert (first["step_number"], second["step_number"]) == (0, 1)
    assert second["depends_on_steps"] == [0]
    assert second["input_params"] == {
        "dag_edge_list": {"type": "ref", "step": 0, "field": "edge_list"},
        "target": {"type": "ref", "step": 0, "field": None},
        "importance_percentile": {"type": "ref", "step": None, "field": None},
    }
    assert record["execution_order_repaired"] is None
    assert groups_record(plan) == [[0], [1]]


def test_repaired_order_is_recorded_as_step_numbers_with_its_reason():
    d = _decomposition(["CAUSAL"])
    a = _step("a", "discover_dag", "sq_0", {})
    b = _step("b", "rank_drivers", "sq_0", {}, deps=["a"])
    plan = _plan(d, [b, a], [])
    assert plan_record(plan, allowlist=None)["execution_order_repaired"] == "no_groups"
    assert groups_record(plan) == [[1], [0]]


def test_intent_normalized_and_index_positional():
    d = _decomposition(["causal", "sell more", " Predictive "])
    assert sub_questions_record(d) == [
        {"index": 0, "intent": "CAUSAL"},
        {"index": 1, "intent": "OTHER"},
        {"index": 2, "intent": "PREDICTIVE"},
    ]


def test_input_map_keeps_declared_names_and_counts_the_rest():
    d = _decomposition(["CAUSAL"])
    gap = _step(
        "g",
        "gap_calculator",
        "sq_0",
        {
            "metric": "treatment",
            "entities": ["a", "b"],
            "group_by": "geo_x",
            "invented": 1,
            "other": "x",
        },
    )
    record = plan_record(_plan(d, [gap], [["g"]]), allowlist=CATALOG)
    assert record["steps"][0]["input_params"] == {
        "metric": {"type": "column", "name": "treatment"},
        "entities": {"type": "list", "len": 2},
        "group_by": {"type": "str", "len": 5},
        "undeclared_params": 2,
    }


def test_output_keys_only_model_fields():
    d = _decomposition(["CAUSAL"])
    gap = _step("g", "gap_calculator", "sq_0", {})
    plan = _plan(d, [gap], [["g"]])
    result = _result(
        gap,
        outcome_class="succeeded",
        result={"entity_values": {"NE": 1}, "gap": 0.2, "per_segment_NE": 3, SENTINEL: 4},
    )
    record = step_record(0, result, plan, allowlist=CATALOG)
    assert record["output_keys"] == {"keys": ["gap", "entity_values"], "other_keys": 2}


def test_step_record_fields():
    d = _decomposition(["CAUSAL", "COMPARATIVE"])
    dag = _step("kpi_dag", "discover_dag", "sq_0", {})
    rank = _step("kpi_rank", "rank_drivers", "sq_1", {}, deps=["kpi_dag"])
    plan = _plan(d, [dag, rank], [["kpi_dag"], ["kpi_rank"]])
    result = _result(
        rank, outcome_class="timeout", error="took too long", error_type="SyncToolTimeout"
    )
    record = step_record(1, result, plan, allowlist=None)
    assert record == {
        "step_number": 1,
        "tool_name": "rank_drivers",
        "input_params": {},
        "output_keys": {"keys": [], "other_keys": 0},
        "depends_on_steps": [0],
        "serves_sub_question": "1",
        "started_at": "2026-09-11T10:00:00+00:00",
        "completed_at": "2026-09-11T10:00:00.250000+00:00",
        "latency_ms": 250,
        "outcome_class": "timeout",
        "attempts": 1,
        "cache_hit": False,
        "error_type": "SyncToolTimeout",
    }


def test_steps_of_unregistered_tools_are_not_sent_and_their_names_not_kept():
    d = _decomposition(["CAUSAL"])
    ghost = _step("ghost_step", f"{SENTINEL}_tool", "sq_0", {"x": 1})
    plan = _plan(d, [ghost], [["ghost_step"]])
    trace = ExecutionTrace(plan_id=plan.plan_id)
    trace.add_result(_result(ghost, outcome_class="not_registered"))
    record = to_record(decomposition=d, plan=plan, trace=trace, allowlist=None)
    assert record["steps"] == []
    assert record["tool_plan"]["steps"] == [
        {
            "step_number": 0,
            "tool_name": None,
            "depends_on_steps": [],
            "input_params": {"undeclared_params": 1},
        }
    ]
    assert SENTINEL not in json.dumps(record)
