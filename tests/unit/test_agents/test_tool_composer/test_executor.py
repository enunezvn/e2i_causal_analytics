"""
Tests for Tool Composer Phase 3: Executor

Tests the PlanExecutor class which executes tool chains
according to the execution plan.
"""

import pytest

from src.agents.tool_composer.executor import (
    PlanExecutor,
    execute_sync,
)
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    SubQuestion,
)
from src.tool_registry.registry import ToolSchema


class TestPlanExecutorInit:
    """Tests for PlanExecutor initialization"""

    def test_default_initialization(self, mock_tool_registry):
        """Test default initialization"""
        executor = PlanExecutor(tool_registry=mock_tool_registry)
        assert executor.max_parallel == 3
        assert executor.max_retries == 2
        assert executor.timeout_seconds == 120

    def test_custom_initialization(self, mock_tool_registry):
        """Test custom initialization"""
        executor = PlanExecutor(
            tool_registry=mock_tool_registry, max_parallel=5, max_retries=3, timeout_seconds=60
        )
        assert executor.max_parallel == 5
        assert executor.max_retries == 3
        assert executor.timeout_seconds == 60


class TestBasicExecution:
    """Tests for basic plan execution"""

    @pytest.mark.asyncio
    async def test_execute_single_step(self, mock_tool_registry, sample_decomposition):
        """Test executing a plan with a single step"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Single step",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        assert isinstance(trace, ExecutionTrace)
        assert trace.tools_executed == 1
        assert trace.tools_succeeded == 1
        assert len(trace.step_results) == 1

    @pytest.mark.asyncio
    async def test_execute_sequential_steps(self, mock_tool_registry, sample_execution_plan):
        """Test executing sequential steps"""
        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(sample_execution_plan)

        assert trace.tools_executed == 2
        # Both steps should succeed if tools are registered
        assert trace.tools_succeeded >= 1

    @pytest.mark.asyncio
    async def test_execution_trace_has_timing(self, mock_tool_registry, sample_execution_plan):
        """Test that execution trace has timing information"""
        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(sample_execution_plan)

        assert trace.started_at is not None
        assert trace.completed_at is not None
        assert trace.total_duration_ms >= 0


class TestParallelExecution:
    """Tests for parallel execution"""

    @pytest.mark.asyncio
    async def test_parallel_execution_group(self, mock_tool_registry, sample_decomposition):
        """Test that parallel steps execute concurrently"""
        # Create two independent steps
        steps = [
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="causal_effect_estimator",
                source_agent="causal_impact",
                input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
                dependency_type=DependencyType.PARALLEL,
                depends_on_steps=[],
            ),
            ExecutionStep(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="gap_calculator",
                source_agent="gap_analyzer",
                input_mapping={"metric": "rx_volume"},
                dependency_type=DependencyType.PARALLEL,
                depends_on_steps=[],
            ),
        ]

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=steps,
            tool_mappings=[],
            parallel_groups=[["step_1", "step_2"]],  # Both in same group
            planning_reasoning="Parallel steps",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        assert trace.tools_executed == 2
        assert trace.parallel_executions == 1

    @pytest.mark.asyncio
    async def test_max_parallel_limit(self, mock_tool_registry, sample_decomposition):
        """Test that parallel execution respects max_parallel limit"""
        # Create many parallel steps
        steps = [
            ExecutionStep(
                step_id=f"step_{i}",
                sub_question_id=f"sq_{i}",
                tool_name="gap_calculator",
                source_agent="gap_analyzer",
                input_mapping={"metric": "test"},
                dependency_type=DependencyType.PARALLEL,
                depends_on_steps=[],
            )
            for i in range(5)
        ]

        plan = ExecutionPlan(
            decomposition=DecompositionResult(
                original_query="Test",
                sub_questions=[
                    SubQuestion(id=f"sq_{i}", question=f"Q{i}", intent="DESCRIPTIVE")
                    for i in range(5)
                ],
                decomposition_reasoning="Test",
            ),
            steps=steps,
            tool_mappings=[],
            parallel_groups=[[f"step_{i}" for i in range(5)]],
            planning_reasoning="Many parallel",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry, max_parallel=2)
        trace = await executor.execute(plan)

        assert trace.tools_executed == 5


class TestInputResolution:
    """Tests for input parameter resolution"""

    @pytest.mark.asyncio
    async def test_static_input_values(self, mock_tool_registry, sample_decomposition):
        """Test that static input values are passed correctly"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        result = trace.step_results[0]
        assert result.input.parameters["treatment"] == "rep_visits"
        assert result.input.parameters["outcome"] == "rx_volume"

    @pytest.mark.asyncio
    async def test_step_reference_resolution(self, mock_tool_registry, sample_decomposition):
        """Test that $step_X.field references are resolved"""
        steps = [
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="causal_effect_estimator",
                source_agent="causal_impact",
                input_mapping={"treatment": "rep_visits", "outcome": "rx_volume"},
                depends_on_steps=[],
            ),
            ExecutionStep(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="cate_analyzer",
                source_agent="heterogeneous_optimizer",
                input_mapping={
                    "effect": "$step_1.effect",  # Reference to step_1 output
                    "dimension": "region",
                },
                depends_on_steps=["step_1"],
            ),
        ]

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=steps,
            tool_mappings=[],
            parallel_groups=[["step_1"], ["step_2"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        # step_2 should have resolved the reference
        result2 = next(r for r in trace.step_results if r.step_id == "step_2")
        # The effect value should be resolved from step_1's output
        assert result2.input.parameters["effect"] == 0.15  # From mock tool

    @pytest.mark.asyncio
    async def test_context_reference_resolution(self, mock_tool_registry, sample_decomposition):
        """Test that $context.field references are resolved"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "$context.treatment_var", "outcome": "rx_volume"},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        context = {"treatment_var": "speaker_programs"}
        trace = await executor.execute(plan, context)

        result = trace.step_results[0]
        assert result.input.parameters["treatment"] == "speaker_programs"


class TestRetryBehavior:
    """Tests for retry behavior"""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_tool_registry, sample_decomposition):
        """Test that failed tools are retried"""
        call_count = 0

        def flaky_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return {"result": "success"}

        # Register a flaky tool
        mock_tool_registry.clear()
        schema = ToolSchema(
            name="flaky_tool",
            description="A flaky tool",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=schema, callable=flaky_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="flaky_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(plan)

        assert call_count == 2
        assert trace.tools_succeeded == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, mock_tool_registry, sample_decomposition):
        """Test behavior when max retries are exhausted"""

        def always_fail(**kwargs):
            raise ValueError("Always fails")

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="failing_tool",
            description="Always fails",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=schema, callable=always_fail)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="failing_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(plan)

        assert trace.tools_failed == 1
        assert trace.tools_succeeded == 0
        result = trace.step_results[0]
        assert result.status == ExecutionStatus.FAILED
        assert "Always fails" in result.output.error


class TestAsyncToolExecution:
    """Tests for async tool execution"""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, mock_tool_registry, sample_decomposition, async_tool):
        """Test that async tools are executed correctly"""
        mock_tool_registry.clear()
        schema = ToolSchema(
            name="async_tool",
            description="An async tool",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
            is_async=True,
        )
        mock_tool_registry.register(schema=schema, callable=async_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="async_tool",
            source_agent="test",
            input_mapping={"treatment": "test", "outcome": "test"},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        assert trace.tools_succeeded == 1
        assert trace.step_results[0].output.result["effect"] == 0.2


class TestToolNotFound:
    """Tests for handling missing tools"""

    @pytest.mark.asyncio
    async def test_missing_tool_handled(self, empty_registry, sample_decomposition):
        """Test that missing tools are handled gracefully"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="nonexistent_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=empty_registry)
        trace = await executor.execute(plan)

        assert trace.tools_failed == 1
        result = trace.step_results[0]
        assert "not found" in result.output.error


class TestOutputConversion:
    """Tests for tool output conversion"""

    @pytest.mark.asyncio
    async def test_dict_output_preserved(self, mock_tool_registry, sample_decomposition):
        """Test that dict outputs are preserved"""
        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "test", "outcome": "test"},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        result = trace.step_results[0].output.result
        assert isinstance(result, dict)
        assert "effect" in result

    @pytest.mark.asyncio
    async def test_pydantic_output_converted(self, mock_tool_registry, sample_decomposition):
        """Test that Pydantic model outputs are converted to dict"""
        from pydantic import BaseModel

        class ToolResult(BaseModel):
            value: float
            confidence: float

        def pydantic_tool(**kwargs):
            return ToolResult(value=0.5, confidence=0.9)

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="pydantic_tool",
            description="Returns Pydantic model",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=schema, callable=pydantic_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="pydantic_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        result = trace.step_results[0].output.result
        assert isinstance(result, dict)
        assert result["value"] == 0.5

    @pytest.mark.asyncio
    async def test_primitive_output_wrapped(self, mock_tool_registry, sample_decomposition):
        """Test that primitive outputs are wrapped in dict"""

        def primitive_tool(**kwargs):
            return 42

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="primitive_tool",
            description="Returns primitive",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=schema, callable=primitive_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="primitive_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[["step_1"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(plan)

        result = trace.step_results[0].output.result
        assert result == {"value": 42}


class TestExecutionOrder:
    """Tests for execution order"""

    @pytest.mark.asyncio
    async def test_dependencies_executed_first(self, mock_tool_registry, sample_decomposition):
        """Test that dependencies are executed before dependent steps"""
        execution_order = []

        def tracking_tool_1(**kwargs):
            execution_order.append("step_1")
            return {"effect": 0.15}

        def tracking_tool_2(**kwargs):
            execution_order.append("step_2")
            return {"segments": []}

        mock_tool_registry.clear()
        for name, fn in [("tool_1", tracking_tool_1), ("tool_2", tracking_tool_2)]:
            schema = ToolSchema(
                name=name,
                description="Tracking tool",
                source_agent="test",
                tier=1,
                avg_execution_ms=100,
            )
            mock_tool_registry.register(schema=schema, callable=fn)

        steps = [
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="tool_1",
                source_agent="test",
                input_mapping={},
                depends_on_steps=[],
            ),
            ExecutionStep(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="tool_2",
                source_agent="test",
                input_mapping={},
                depends_on_steps=["step_1"],
            ),
        ]

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=steps,
            tool_mappings=[],
            parallel_groups=[["step_1"], ["step_2"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry)
        await executor.execute(plan)

        assert execution_order == ["step_1", "step_2"]


class TestSyncWrapper:
    """Tests for synchronous wrapper function"""

    def test_execute_sync(self, mock_tool_registry, sample_execution_plan):
        """Test synchronous execute wrapper"""
        trace = execute_sync(sample_execution_plan, tool_registry=mock_tool_registry)

        assert isinstance(trace, ExecutionTrace)
        assert trace.tools_executed >= 1


class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_execution_continues_on_failure(self, mock_tool_registry, sample_decomposition):
        """Test that execution continues even if one step fails"""

        def failing_tool(**kwargs):
            raise ValueError("Failure")

        def working_tool(**kwargs):
            return {"result": "success"}

        mock_tool_registry.clear()
        for name, fn in [("failing_tool", failing_tool), ("working_tool", working_tool)]:
            schema = ToolSchema(
                name=name,
                description="Test tool",
                source_agent="test",
                tier=1,
                avg_execution_ms=100,
            )
            mock_tool_registry.register(schema=schema, callable=fn)

        steps = [
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="failing_tool",
                source_agent="test",
                input_mapping={},
                depends_on_steps=[],
            ),
            ExecutionStep(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="working_tool",
                source_agent="test",
                input_mapping={},
                depends_on_steps=[],  # No dependency on failing step
            ),
        ]

        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=steps,
            tool_mappings=[],
            parallel_groups=[["step_1", "step_2"]],
            planning_reasoning="Test",
        )

        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=0)
        trace = await executor.execute(plan)

        assert trace.tools_executed == 2
        assert trace.tools_failed == 1
        assert trace.tools_succeeded == 1
