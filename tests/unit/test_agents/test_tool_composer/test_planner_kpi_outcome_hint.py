"""Planner KPI outcome-hint binding (issue #810).

When a query targets a defined KPI, the causal outcome is definitionally the
KPI's outcome column. The planner must bind every ``outcome``/``target`` arg to
it deterministically (not rely on the LLM guessing), while never clobbering a
``$step`` reference or injecting a non-existent column.
"""

from __future__ import annotations

import pytest

from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionStep,
    SubQuestion,
)
from src.agents.tool_composer.planner import ToolPlanner


def _planner(mock_llm_client, mock_tool_registry) -> ToolPlanner:
    return ToolPlanner(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_caching=False,
        use_episodic_memory=False,
    )


class TestApplyOutcomeHint:
    def test_overrides_outcome_to_kpi_column(self, mock_llm_client, mock_tool_registry):
        steps = [
            ExecutionStep(
                step_id="s1",
                sub_question_id="q1",
                source_agent="composable",
                tool_name="causal_effect_estimator",
                input_mapping={"treatment": "accepted", "outcome": "wrong_col"},
            )
        ]
        _planner(mock_llm_client, mock_tool_registry)._apply_outcome_hint(
            steps, "converted", ["converted", "accepted", "wrong_col"]
        )
        assert steps[0].input_mapping["outcome"] == "converted"
        # treatment (a driver) is untouched.
        assert steps[0].input_mapping["treatment"] == "accepted"

    def test_overrides_target_arg_too(self, mock_llm_client, mock_tool_registry):
        steps = [
            ExecutionStep(
                step_id="s1",
                sub_question_id="q1",
                source_agent="composable",
                tool_name="rank_drivers",
                input_mapping={"target": "foo"},
            )
        ]
        _planner(mock_llm_client, mock_tool_registry)._apply_outcome_hint(
            steps, "converted", ["converted", "foo"]
        )
        assert steps[0].input_mapping["target"] == "converted"

    def test_does_not_clobber_step_reference(self, mock_llm_client, mock_tool_registry):
        steps = [
            ExecutionStep(
                step_id="s2",
                sub_question_id="q1",
                source_agent="composable",
                tool_name="cate_analyzer",
                input_mapping={"outcome": "$step_1.converted"},
            )
        ]
        _planner(mock_llm_client, mock_tool_registry)._apply_outcome_hint(
            steps, "converted", ["converted"]
        )
        assert steps[0].input_mapping["outcome"] == "$step_1.converted"

    def test_noop_when_hint_not_a_real_column(self, mock_llm_client, mock_tool_registry):
        steps = [
            ExecutionStep(
                step_id="s1",
                sub_question_id="q1",
                source_agent="composable",
                tool_name="causal_effect_estimator",
                input_mapping={"outcome": "engagement"},
            )
        ]
        # 'converted' is NOT in the available columns -> never inject it.
        _planner(mock_llm_client, mock_tool_registry)._apply_outcome_hint(
            steps, "converted", ["engagement", "accepted"]
        )
        assert steps[0].input_mapping["outcome"] == "engagement"

    def test_noop_when_no_hint(self, mock_llm_client, mock_tool_registry):
        steps = [
            ExecutionStep(
                step_id="s1",
                sub_question_id="q1",
                source_agent="composable",
                tool_name="causal_effect_estimator",
                input_mapping={"outcome": "engagement"},
            )
        ]
        _planner(mock_llm_client, mock_tool_registry)._apply_outcome_hint(
            steps, None, ["engagement"]
        )
        assert steps[0].input_mapping["outcome"] == "engagement"


_GUARD_PROFILES = [
    {
        "name": "converted",
        "dtype_family": "binary",
        "n_unique": 2,
        "n_nonnull": 100,
        "values": [0, 1],
    },
    {
        "name": "accepted",
        "dtype_family": "binary",
        "n_unique": 2,
        "n_nonnull": 100,
        "values": [0, 1],
    },
    {
        "name": "confidence_score",
        "dtype_family": "numeric-continuous",
        "n_unique": 80,
        "n_nonnull": 100,
        "values": None,
    },
    {
        "name": "trigger_type",
        "dtype_family": "categorical",
        "n_unique": 6,
        "n_nonnull": 100,
        "values": ["adherence_risk", "churn_prevention", "cross_sell"],
    },
]


class TestApplyTreatmentGuard:
    def _step(self, treatment):
        return ExecutionStep(
            step_id="s1",
            sub_question_id="q1",
            source_agent="composable",
            tool_name="causal_effect_estimator",
            input_mapping={"treatment": treatment, "outcome": "converted"},
        )

    def test_overrides_categorical_treatment_to_binary(self, mock_llm_client, mock_tool_registry):
        steps = [self._step("trigger_type")]  # categorical -> unusable as DoWhy treatment
        _planner(mock_llm_client, mock_tool_registry)._apply_treatment_guard(
            steps, _GUARD_PROFILES, "converted"
        )
        # bound to the binary driver (not the outcome, not the categorical).
        assert steps[0].input_mapping["treatment"] == "accepted"

    def test_keeps_valid_binary_treatment(self, mock_llm_client, mock_tool_registry):
        steps = [self._step("accepted")]
        _planner(mock_llm_client, mock_tool_registry)._apply_treatment_guard(
            steps, _GUARD_PROFILES, "converted"
        )
        assert steps[0].input_mapping["treatment"] == "accepted"

    def test_keeps_valid_numeric_treatment(self, mock_llm_client, mock_tool_registry):
        steps = [self._step("confidence_score")]
        _planner(mock_llm_client, mock_tool_registry)._apply_treatment_guard(
            steps, _GUARD_PROFILES, "converted"
        )
        assert steps[0].input_mapping["treatment"] == "confidence_score"

    def test_noop_without_outcome_hint_non_kpi(self, mock_llm_client, mock_tool_registry):
        # Non-KPI plan (no outcome_hint): the LLM's treatment choice is preserved.
        steps = [self._step("trigger_type")]
        _planner(mock_llm_client, mock_tool_registry)._apply_treatment_guard(
            steps, _GUARD_PROFILES, None
        )
        assert steps[0].input_mapping["treatment"] == "trigger_type"

    def test_does_not_clobber_step_reference(self, mock_llm_client, mock_tool_registry):
        steps = [self._step("$step_1.treatment")]
        _planner(mock_llm_client, mock_tool_registry)._apply_treatment_guard(
            steps, _GUARD_PROFILES, "converted"
        )
        assert steps[0].input_mapping["treatment"] == "$step_1.treatment"


@pytest.mark.asyncio
async def test_plan_binds_outcome_hint_end_to_end(mock_llm_client, mock_tool_registry):
    """Full plan(): the LLM emits a wrong outcome; the hint overrides it to the
    KPI outcome column (and F6(b) enforcement then passes on the real column)."""
    decomposition = DecompositionResult(
        original_query="what drove conversion",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question="what drove conversion?",
                intent="CAUSAL",
                entities=[],
                depends_on=[],
            )
        ],
        decomposition_reasoning="t",
    )
    mock_llm_client.set_planning_response(
        '{"reasoning":"r",'
        '"tool_mappings":[{"sub_question_id":"sq_1","tool_name":"causal_effect_estimator",'
        '"confidence":0.9,"reasoning":"x"}],'
        '"execution_steps":[{"step_id":"step_1","sub_question_id":"sq_1",'
        '"tool_name":"causal_effect_estimator",'
        '"input_mapping":{"treatment":"accepted","outcome":"prescriptions"},'
        '"depends_on_steps":[]}],'
        '"parallel_groups":[["step_1"]]}'
    )
    profiles = [
        {
            "name": "converted",
            "dtype_family": "binary",
            "n_unique": 2,
            "n_nonnull": 100,
            "values": [0, 1],
        },
        {
            "name": "accepted",
            "dtype_family": "binary",
            "n_unique": 2,
            "n_nonnull": 100,
            "values": [0, 1],
        },
    ]
    planner = _planner(mock_llm_client, mock_tool_registry)
    plan = await planner.plan(decomposition, column_profiles=profiles, outcome_hint="converted")
    assert plan.steps[0].input_mapping["outcome"] == "converted"
    assert plan.steps[0].input_mapping["treatment"] == "accepted"
