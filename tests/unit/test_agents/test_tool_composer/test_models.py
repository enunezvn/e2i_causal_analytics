"""
Tests for Tool Composer Pydantic models.

Validates data models for all 4 phases of the composition pipeline.
"""

from datetime import datetime, timezone

import pytest

from src.agents.tool_composer.models.composition_models import (
    ComposedResponse,
    CompositionPhase,
    CompositionResult,
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    SynthesisInput,
    ToolInput,
    ToolMapping,
    ToolOutput,
)


class TestEnums:
    """Tests for enumeration types"""

    def test_composition_phase_values(self):
        """Verify CompositionPhase enum values"""
        assert CompositionPhase.DECOMPOSE.value == "decompose"
        assert CompositionPhase.PLAN.value == "plan"
        assert CompositionPhase.EXECUTE.value == "execute"
        assert CompositionPhase.SYNTHESIZE.value == "synthesize"

    def test_execution_status_values(self):
        """Verify ExecutionStatus enum values"""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.SKIPPED.value == "skipped"

    def test_dependency_type_values(self):
        """Verify DependencyType enum values"""
        assert DependencyType.SEQUENTIAL.value == "sequential"
        assert DependencyType.PARALLEL.value == "parallel"
        assert DependencyType.CONDITIONAL.value == "conditional"


class TestSubQuestion:
    """Tests for SubQuestion model"""

    def test_create_sub_question(self):
        """Test creating a sub-question"""
        sq = SubQuestion(
            id="sq_1",
            question="What is the causal effect?",
            intent="CAUSAL",
            entities=["treatment", "outcome"],
            depends_on=[],
        )
        assert sq.id == "sq_1"
        assert sq.question == "What is the causal effect?"
        assert sq.intent == "CAUSAL"
        assert sq.entities == ["treatment", "outcome"]
        assert sq.depends_on == []

    def test_sub_question_auto_id(self):
        """Test auto-generated ID"""
        sq = SubQuestion(question="Test question", intent="DESCRIPTIVE")
        assert sq.id.startswith("sq_")
        assert len(sq.id) == 11  # sq_ + 8 hex chars

    def test_sub_question_with_dependencies(self):
        """Test sub-question with dependencies"""
        sq = SubQuestion(
            id="sq_2", question="How does it vary?", intent="COMPARATIVE", depends_on=["sq_1"]
        )
        assert sq.depends_on == ["sq_1"]

    def test_sub_question_immutable(self):
        """Test that SubQuestion is frozen (immutable)"""
        sq = SubQuestion(id="sq_1", question="Test", intent="DESCRIPTIVE")
        with pytest.raises(Exception):
            sq.question = "Changed"


class TestDecompositionResult:
    """Tests for DecompositionResult model"""

    def test_create_decomposition_result(self, sample_sub_questions):
        """Test creating a decomposition result"""
        result = DecompositionResult(
            original_query="Test query",
            sub_questions=sample_sub_questions,
            decomposition_reasoning="Test reasoning",
        )
        assert result.original_query == "Test query"
        assert len(result.sub_questions) == 2
        assert result.decomposition_reasoning == "Test reasoning"

    def test_question_count_property(self, sample_sub_questions):
        """Test question_count property"""
        result = DecompositionResult(
            original_query="Test",
            sub_questions=sample_sub_questions,
            decomposition_reasoning="Test",
        )
        assert result.question_count == 2

    def test_get_root_questions(self, sample_sub_questions):
        """Test get_root_questions method"""
        result = DecompositionResult(
            original_query="Test",
            sub_questions=sample_sub_questions,
            decomposition_reasoning="Test",
        )
        roots = result.get_root_questions()
        assert len(roots) == 1
        assert roots[0].id == "sq_1"

    def test_timestamp_auto_set(self, sample_sub_questions):
        """Test timestamp is automatically set"""
        result = DecompositionResult(
            original_query="Test",
            sub_questions=sample_sub_questions,
            decomposition_reasoning="Test",
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestToolMapping:
    """Tests for ToolMapping model"""

    def test_create_tool_mapping(self):
        """Test creating a tool mapping"""
        mapping = ToolMapping(
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            confidence=0.95,
            reasoning="Good match",
        )
        assert mapping.sub_question_id == "sq_1"
        assert mapping.tool_name == "causal_effect_estimator"
        assert mapping.confidence == 0.95

    def test_confidence_bounds(self):
        """Test confidence value bounds"""
        # Valid confidence
        mapping = ToolMapping(
            sub_question_id="sq_1",
            tool_name="test",
            source_agent="test",
            confidence=0.5,
            reasoning="Test",
        )
        assert mapping.confidence == 0.5

        # Test boundary values
        with pytest.raises(ValueError):
            ToolMapping(
                sub_question_id="sq_1",
                tool_name="test",
                source_agent="test",
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test",
            )


class TestExecutionStep:
    """Tests for ExecutionStep model"""

    def test_create_execution_step(self):
        """Test creating an execution step"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "rep_visits"},
            dependency_type=DependencyType.PARALLEL,
            depends_on_steps=[],
        )
        assert step.step_id == "step_1"
        assert step.status == ExecutionStatus.PENDING

    def test_execution_step_auto_id(self):
        """Test auto-generated step ID"""
        step = ExecutionStep(sub_question_id="sq_1", tool_name="test", source_agent="test")
        assert step.step_id.startswith("step_")

    def test_execution_step_with_dependencies(self):
        """Test step with dependencies"""
        step = ExecutionStep(
            step_id="step_2",
            sub_question_id="sq_2",
            tool_name="cate_analyzer",
            source_agent="heterogeneous_optimizer",
            input_mapping={"effect": "$step_1.effect"},
            dependency_type=DependencyType.SEQUENTIAL,
            depends_on_steps=["step_1"],
        )
        assert step.depends_on_steps == ["step_1"]
        assert step.dependency_type == DependencyType.SEQUENTIAL


class TestExecutionPlan:
    """Tests for ExecutionPlan model"""

    def test_create_execution_plan(
        self, sample_decomposition, sample_execution_steps, sample_tool_mappings
    ):
        """Test creating an execution plan"""
        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=sample_execution_steps,
            tool_mappings=sample_tool_mappings,
            planning_reasoning="Test plan",
        )
        assert plan.plan_id.startswith("plan_")
        assert plan.step_count == 2

    def test_get_step(self, sample_execution_plan):
        """Test get_step method"""
        step = sample_execution_plan.get_step("step_1")
        assert step is not None
        assert step.tool_name == "causal_effect_estimator"

        # Non-existent step
        assert sample_execution_plan.get_step("nonexistent") is None

    def test_get_ready_steps(self, sample_execution_plan):
        """Test get_ready_steps method"""
        ready = sample_execution_plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "step_1"

    def test_get_execution_order(self, sample_execution_plan):
        """Test get_execution_order method"""
        order = sample_execution_plan.get_execution_order()
        assert order == [["step_1"], ["step_2"]]


class TestToolOutput:
    """Tests for ToolOutput model"""

    def test_successful_output(self):
        """Test successful tool output"""
        output = ToolOutput(
            tool_name="test", success=True, result={"effect": 0.15}, execution_time_ms=500
        )
        assert output.is_success is True

    def test_failed_output(self):
        """Test failed tool output"""
        output = ToolOutput(tool_name="test", success=False, error="Tool failed")
        assert output.is_success is False

    def test_output_with_none_result(self):
        """Test output with None result"""
        output = ToolOutput(tool_name="test", success=True, result=None)
        assert output.is_success is False  # success but no result


class TestExecutionTrace:
    """Tests for ExecutionTrace model"""

    def test_create_trace(self):
        """Test creating an execution trace"""
        trace = ExecutionTrace(plan_id="plan_123")
        assert trace.plan_id == "plan_123"
        assert trace.tools_executed == 0
        assert trace.tools_succeeded == 0

    def test_add_result(self, sample_step_result):
        """Test adding a result to trace"""
        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(sample_step_result)
        assert trace.tools_executed == 1
        assert trace.tools_succeeded == 1
        assert len(trace.step_results) == 1

    def test_add_failed_result(self):
        """Test adding a failed result"""
        trace = ExecutionTrace(plan_id="plan_123")
        failed_result = StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="test",
            input=ToolInput(tool_name="test", parameters={}),
            output=ToolOutput(tool_name="test", success=False, error="Failed"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        trace.add_result(failed_result)
        assert trace.tools_executed == 1
        assert trace.tools_succeeded == 0
        assert trace.tools_failed == 1

    def test_get_all_outputs(self, sample_step_result):
        """Test get_all_outputs method"""
        trace = ExecutionTrace(plan_id="plan_123")
        trace.add_result(sample_step_result)
        outputs = trace.get_all_outputs()
        assert "step_1" in outputs
        assert outputs["step_1"]["effect"] == 0.15


class TestComposedResponse:
    """Tests for ComposedResponse model"""

    def test_create_response(self):
        """Test creating a composed response"""
        response = ComposedResponse(
            answer="The effect is significant.",
            confidence=0.85,
            supporting_data={"effect": 0.15},
            citations=["step_1"],
            caveats=["Observational data"],
        )
        assert response.response_id.startswith("resp_")
        assert response.confidence == 0.85

    def test_response_confidence_bounds(self):
        """Test confidence value bounds"""
        with pytest.raises(ValueError):
            ComposedResponse(answer="Test", confidence=1.5)  # Invalid


class TestSynthesisInput:
    """Tests for SynthesisInput model"""

    def test_create_synthesis_input(self, sample_decomposition, sample_execution_trace):
        """Test creating synthesis input"""
        synthesis = SynthesisInput(
            original_query="Test query",
            decomposition=sample_decomposition,
            execution_trace=sample_execution_trace,
        )
        assert synthesis.original_query == "Test query"

    def test_get_context_for_synthesis(self, sample_synthesis_input):
        """Test get_context_for_synthesis method"""
        context = sample_synthesis_input.get_context_for_synthesis()
        assert "query" in context
        assert "sub_questions" in context
        assert "results" in context
        assert len(context["sub_questions"]) == 2


class TestCompositionResult:
    """Tests for CompositionResult model"""

    def test_create_composition_result(
        self, sample_decomposition, sample_execution_plan, sample_execution_trace
    ):
        """Test creating a composition result"""
        response = ComposedResponse(answer="Test answer", confidence=0.85)
        result = CompositionResult(
            query="Test query",
            decomposition=sample_decomposition,
            plan=sample_execution_plan,
            execution=sample_execution_trace,
            response=response,
            total_duration_ms=1500,
            phase_durations={"decompose": 100, "plan": 200, "execute": 1000, "synthesize": 200},
            success=True,
        )
        assert result.composition_id.startswith("comp_")
        assert result.success is True

    def test_to_summary(self, sample_decomposition, sample_execution_plan, sample_execution_trace):
        """Test to_summary method"""
        response = ComposedResponse(answer="Test", confidence=0.85)
        result = CompositionResult(
            query="Test query",
            decomposition=sample_decomposition,
            plan=sample_execution_plan,
            execution=sample_execution_trace,
            response=response,
            total_duration_ms=1500,
            success=True,
        )
        summary = result.to_summary()
        assert "composition_id" in summary
        assert "sub_questions" in summary
        assert "tools_executed" in summary
        assert "success" in summary
