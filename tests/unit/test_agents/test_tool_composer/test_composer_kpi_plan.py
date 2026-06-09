"""Deterministic KPI causal-plan builder (issue #810).

For a defined-KPI causal query the composer builds a deterministic causal plan
(causal_effect_estimator + discover_dag -> rank_drivers + cate_analyzer) over the
KPI substrate, instead of relying on unreliable free-form LLM planning. Treatment
is a binary/numeric driver; segments are LOW-cardinality categoricals (high-card
ID columns excluded); the outcome is the KPI column.
"""

from __future__ import annotations

import pandas as pd

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    SubQuestion,
)


def _decomp() -> DecompositionResult:
    return DecompositionResult(
        original_query="what drove conversion and which segments respond best",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question="what drove conversion?",
                intent="CAUSAL",
                entities=[],
                depends_on=[],
            ),
            SubQuestion(
                id="sq_2",
                question="which segments respond best?",
                intent="COMPARATIVE",
                entities=[],
                depends_on=[],
            ),
        ],
        decomposition_reasoning="t",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "converted": [0, 1] * 20,  # binary outcome (KPI)
            "accepted": [1, 0] * 20,  # binary driver -> treatment
            "confidence_score": [0.5 + 0.01 * i for i in range(40)],  # numeric -> confounder
            "trigger_id": [f"T{i}" for i in range(40)],  # high-card ID -> excluded from segments
            "delivery_channel": (["email", "crm", "phone", "portal"] * 10),  # low-card -> segment
        }
    )


def _composer() -> ToolComposer:
    return ToolComposer(llm_client=object(), enable_memory_contribution=False)


def test_build_kpi_plan_binds_treatment_outcome_segments():
    ctx = {"estimation_data": _frame(), "kpi_outcome": "converted"}
    plan = _composer()._build_kpi_causal_plan(_decomp(), ctx, "converted")
    assert plan is not None
    tools = {s.tool_name for s in plan.steps}
    assert {"causal_effect_estimator", "discover_dag", "rank_drivers", "cate_analyzer"} <= tools

    ate = next(s for s in plan.steps if s.tool_name == "causal_effect_estimator")
    assert ate.input_mapping["treatment"] == "accepted"  # binary driver, not outcome/ID
    assert ate.input_mapping["outcome"] == "converted"
    assert "confidence_score" in ate.input_mapping["confounders"]

    cate = next(s for s in plan.steps if s.tool_name == "cate_analyzer")
    assert cate.input_mapping["outcome"] == "converted"
    assert "delivery_channel" in cate.input_mapping["segments"]
    # high-cardinality ID column must NOT be used as a segment.
    assert "trigger_id" not in cate.input_mapping["segments"]

    rank = next(s for s in plan.steps if s.tool_name == "rank_drivers")
    assert rank.input_mapping["target"] == "converted"
    assert rank.depends_on_steps == ["kpi_dag"]


def test_build_kpi_plan_none_without_usable_treatment():
    # Only the outcome + a high-card ID: no binary/numeric driver -> fall back to LLM.
    df = pd.DataFrame({"converted": [0, 1] * 20, "trigger_id": [f"T{i}" for i in range(40)]})
    ctx = {"estimation_data": df, "kpi_outcome": "converted"}
    assert _composer()._build_kpi_causal_plan(_decomp(), ctx, "converted") is None


def test_build_kpi_plan_none_without_frame():
    assert _composer()._build_kpi_causal_plan(_decomp(), {}, "converted") is None
