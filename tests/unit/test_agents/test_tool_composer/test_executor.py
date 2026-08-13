"""
Tests for Tool Composer Phase 3: Executor

Tests the PlanExecutor class which executes tool chains
according to the execution plan.
"""

import pytest

from src.agents.tool_composer.executor import (
    PlanExecutor,
    ToolFailureStats,
    ToolFailureTracker,
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


@pytest.mark.xdist_group(name="sync_wrappers")
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


class TestExponentialBackoff:
    """Tests for ExponentialBackoff class (G7)"""

    def test_backoff_default_values(self):
        """Test default initialization"""
        from src.agents.tool_composer.executor import ExponentialBackoff

        backoff = ExponentialBackoff()
        assert backoff.base_delay == 1.0
        assert backoff.max_delay == 30.0
        assert backoff.factor == 2.0

    def test_backoff_delay_calculation(self):
        """Test delay calculation follows exponential pattern"""
        from src.agents.tool_composer.executor import ExponentialBackoff

        backoff = ExponentialBackoff(base_delay=1.0, factor=2.0, jitter=0.0)

        # Without jitter, delays should be exact
        assert backoff.get_delay(0) == 1.0  # 1 * 2^0 = 1
        assert backoff.get_delay(1) == 2.0  # 1 * 2^1 = 2
        assert backoff.get_delay(2) == 4.0  # 1 * 2^2 = 4
        assert backoff.get_delay(3) == 8.0  # 1 * 2^3 = 8

    def test_backoff_max_delay_cap(self):
        """Test that delay is capped at max_delay"""
        from src.agents.tool_composer.executor import ExponentialBackoff

        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.0)

        # Delay should be capped at 10
        assert backoff.get_delay(5) == 10.0  # 1 * 2^5 = 32, but capped at 10

    def test_backoff_jitter_adds_variation(self):
        """Test that jitter adds random variation"""
        from src.agents.tool_composer.executor import ExponentialBackoff

        backoff = ExponentialBackoff(base_delay=10.0, jitter=0.1)

        delays = [backoff.get_delay(0) for _ in range(10)]
        # With jitter, delays should vary
        unique_delays = set(delays)
        assert len(unique_delays) > 1  # Multiple unique values


class TestCircuitBreaker:
    """Tests for CircuitBreaker class (G7)"""

    def test_circuit_starts_closed(self):
        """Test that circuit starts in closed state"""
        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after failure_threshold failures"""
        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        """Test that success resets the failure count"""
        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_transitions_to_half_open(self):
        """Test that open circuit transitions to half-open after timeout"""
        import time

        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should transition to half-open when checked
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test that success in half-open state closes the circuit"""
        import time

        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait and transition to half-open
        time.sleep(0.15)
        cb.can_execute()  # Triggers transition

        # Record success
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit"""
        import time

        from src.agents.tool_composer.executor import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait and transition to half-open
        time.sleep(0.15)
        cb.can_execute()  # Triggers transition

        # Record failure
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_get_state_info(self):
        """Test circuit breaker state info for observability"""
        from src.agents.tool_composer.executor import CircuitBreaker

        cb = CircuitBreaker()
        cb.record_success()
        cb.record_failure()

        info = cb.get_state_info()
        assert info["state"] == "closed"
        assert info["success_count"] == 1
        assert info["failure_count"] == 1


class TestToolFailureTracker:
    """Tests for ToolFailureTracker class (G7)"""

    def test_tracker_creates_stats_on_demand(self):
        """Test that tracker creates stats for new tools"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        assert tracker.can_execute("new_tool")
        assert tracker.get_stats("new_tool") is not None

    def test_tracker_records_success(self):
        """Test that tracker records successful executions"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        tracker.record_success("test_tool", latency_ms=100)

        stats = tracker.get_stats("test_tool")
        assert stats.total_calls == 1
        assert stats.total_successes == 1
        assert stats.total_latency_ms == 100
        assert stats.success_rate == 1.0

    def test_tracker_records_failure(self):
        """Test that tracker records failed executions"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        tracker.record_failure("test_tool", reason="Test error")

        stats = tracker.get_stats("test_tool")
        assert stats.total_calls == 1
        assert stats.total_failures == 1
        assert stats.last_failure_reason == "Test error"
        assert stats.success_rate == 0.0

    def test_tracker_independent_per_tool(self):
        """Test that each tool has independent stats"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        tracker.record_success("tool_a", latency_ms=50)
        tracker.record_failure("tool_b", reason="Error")

        stats_a = tracker.get_stats("tool_a")
        stats_b = tracker.get_stats("tool_b")

        assert stats_a.total_successes == 1
        assert stats_a.total_failures == 0
        assert stats_b.total_successes == 0
        assert stats_b.total_failures == 1

    def test_tracker_blocks_when_circuit_opens(self):
        """Test that tracker blocks execution when circuit opens"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker(failure_threshold=2)

        assert tracker.can_execute("failing_tool")
        tracker.record_failure("failing_tool", reason="Error 1")
        assert tracker.can_execute("failing_tool")
        tracker.record_failure("failing_tool", reason="Error 2")
        assert not tracker.can_execute("failing_tool")

    def test_tracker_reset_single_tool(self):
        """Test resetting stats for a single tool"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        tracker.record_success("tool_a", 100)
        tracker.record_success("tool_b", 100)

        tracker.reset("tool_a")

        assert tracker.get_stats("tool_a") is None
        assert tracker.get_stats("tool_b") is not None

    def test_tracker_reset_all(self):
        """Test resetting all stats"""
        from src.agents.tool_composer.executor import ToolFailureTracker

        tracker = ToolFailureTracker()
        tracker.record_success("tool_a", 100)
        tracker.record_success("tool_b", 100)

        tracker.reset()

        assert tracker.get_stats("tool_a") is None
        assert tracker.get_stats("tool_b") is None


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with PlanExecutor (G7)"""

    @pytest.mark.asyncio
    async def test_executor_skips_tool_with_open_circuit(
        self, mock_tool_registry, sample_decomposition
    ):
        """Test that executor skips tools when circuit breaker is open"""
        call_count = 0

        def failing_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="circuit_test_tool",
            description="Test tool",
            source_agent="test",
            tier=1,
            avg_execution_ms=100,
        )
        mock_tool_registry.register(schema=schema, callable=failing_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="circuit_test_tool",
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

        # Use low threshold and no retries
        executor = PlanExecutor(
            tool_registry=mock_tool_registry,
            max_retries=0,
            circuit_failure_threshold=2,
        )

        # First execution - fails and counts
        await executor.execute(plan)
        assert call_count == 1

        # Second execution - fails, opens circuit
        await executor.execute(plan)
        assert call_count == 2

        # Third execution - circuit is OPEN, should skip
        trace = await executor.execute(plan)
        assert call_count == 2  # No additional call
        assert trace.step_results[0].status == ExecutionStatus.SKIPPED
        assert "Circuit breaker open" in trace.step_results[0].output.error

    @pytest.mark.asyncio
    async def test_executor_tracks_stats(self, mock_tool_registry, sample_decomposition):
        """Test that executor tracks tool statistics"""
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
        await executor.execute(plan)

        stats = executor.get_tool_stats()
        assert "causal_effect_estimator" in stats
        assert stats["causal_effect_estimator"]["total_calls"] == 1
        assert stats["causal_effect_estimator"]["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_executor_reset_stats(self, mock_tool_registry, sample_decomposition):
        """Test that executor can reset tool statistics"""
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
        await executor.execute(plan)

        assert len(executor.get_tool_stats()) > 0

        executor.reset_tool_stats()
        assert len(executor.get_tool_stats()) == 0


# ============================================================================
# G8: TOOL PERFORMANCE LEARNING TESTS
# ============================================================================


class TestReferenceContract1573:
    """#1573 — unresolvable plan references must FAIL FAST, never degrade to None.

    Live evidence (q08, req 7dad3a48, 2026-08-12): the planner emitted
    ``$dataset.*`` (an invented source) and ``$step_5.top_segments`` /
    ``$step_5.cate_estimate`` (fields the CATE output does not carry). The
    executor resolved all of them to silent None; the counterfactual step then
    crashed with ``NoneType * float`` and was retried 3 times identically.

    Contract pinned here:
    (a) unknown reference source ('$dataset.*') -> explicit step failure, tool
        NOT invoked, no silent None;
    (b) known source / missing field ('$step_1.cate_estimate') -> explicit
        step failure naming the field and the available fields;
    (c) deterministic failures are NOT retried (resolution failures: zero tool
        invocations; ToolInputError: exactly one invocation);
    (d) the failure reason text reaches the synthesis input (honest disclosure
        with the TRUE cause);
    (e) the counterfactual simulator declines None input with a stated reason
        instead of TypeError;
    (f) F7 GUARD: references NESTED inside dict values keep degrading to None
        (the discover_dag ``data={'col': '$ref'}`` artifact) — blanket
        strictness would break q08's five working steps.
    """

    @staticmethod
    def _single_step_plan(decomposition, step):
        return ExecutionPlan(
            decomposition=decomposition,
            steps=[step],
            tool_mappings=[],
            parallel_groups=[[step.step_id]],
            planning_reasoning="1573",
        )

    @pytest.mark.asyncio
    async def test_unknown_reference_source_fails_step_without_invoking_tool(
        self, mock_tool_registry, sample_decomposition
    ):
        """(a)+(c): '$dataset.sales' must fail the step BEFORE the tool runs."""
        calls = {"n": 0}

        def counting_tool(**kwargs):
            calls["n"] += 1
            return {"predicted_lift": 0.1}

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="counting_tool",
            description="Counts invocations",
            source_agent="test",
            tier=1,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=counting_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="counting_tool",
            source_agent="test",
            input_mapping={"sales": "$dataset.sales"},  # planner-invented source (q08)
            depends_on_steps=[],
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(self._single_step_plan(sample_decomposition, step))

        result = trace.step_results[0]
        assert result.status == ExecutionStatus.FAILED
        assert trace.tools_failed == 1
        assert calls["n"] == 0, "tool must NOT be invoked with silently-degraded None inputs"

        err = result.output.error or ""
        assert "$dataset.sales" in err, f"failure reason must name the reference; got {err!r}"
        assert "dataset" in err
        # A plan defect must NOT poison the healthy tool's circuit breaker:
        # the tool never ran, and opening its circuit would block other valid
        # steps that use the same tool.
        stats = executor.failure_tracker.get_stats("counting_tool")
        assert stats is None or stats.total_failures == 0

    @pytest.mark.asyncio
    async def test_missing_field_on_known_source_fails_dependent_step(
        self, mock_tool_registry, sample_decomposition
    ):
        """(b): '$step_1.cate_estimate' where step_1's output lacks the field."""
        calls = {"upstream": 0, "downstream": 0}

        def upstream_tool(**kwargs):
            calls["upstream"] += 1
            # CATEResults shape: the fields the REAL cate_analyzer carries.
            return {
                "segments": [{"segment": "west", "cate": 0.1}],
                "high_responders": ["west"],
                "effect_by_segment": {"west": 0.1},
            }

        def downstream_tool(**kwargs):
            calls["downstream"] += 1
            return {"predicted_lift": 0.1}

        mock_tool_registry.clear()
        for name, fn in [("upstream_tool", upstream_tool), ("downstream_tool", downstream_tool)]:
            schema = ToolSchema(
                name=name,
                description="1573 test tool",
                source_agent="test",
                tier=1,
                avg_execution_ms=10,
            )
            mock_tool_registry.register(schema=schema, callable=fn)

        steps = [
            ExecutionStep(
                step_id="step_1",
                sub_question_id="sq_1",
                tool_name="upstream_tool",
                source_agent="test",
                input_mapping={},
                depends_on_steps=[],
            ),
            ExecutionStep(
                step_id="step_2",
                sub_question_id="sq_2",
                tool_name="downstream_tool",
                source_agent="test",
                # The q08 defect: a field the upstream output does not carry.
                input_mapping={"expected_effect": "$step_1.cate_estimate"},
                depends_on_steps=["step_1"],
            ),
        ]
        plan = ExecutionPlan(
            decomposition=sample_decomposition,
            steps=steps,
            tool_mappings=[],
            parallel_groups=[["step_1"], ["step_2"]],
            planning_reasoning="1573",
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(plan)

        r1 = trace.get_result("step_1")
        r2 = trace.get_result("step_2")
        # The working upstream step is untouched (do-not-break-5/6 constraint).
        assert r1.status == ExecutionStatus.COMPLETED
        assert calls["upstream"] == 1
        # The doomed downstream step fails fast, tool never invoked, no retries.
        assert r2.status == ExecutionStatus.FAILED
        assert calls["downstream"] == 0

        err = r2.output.error or ""
        assert "cate_estimate" in err
        assert "step_1" in err
        # Honest disclosure includes what IS available so synthesis can say why.
        assert "effect_by_segment" in err, f"available fields must be listed; got {err!r}"

    @pytest.mark.asyncio
    async def test_deterministic_input_rejection_not_retried(
        self, mock_tool_registry, sample_decomposition
    ):
        """(c)+(e): a ToolInputError is raised ONCE — never retried identically."""
        from src.agents.tool_composer.errors import ToolInputError

        calls = {"n": 0}

        def declining_tool(**kwargs):
            calls["n"] += 1
            raise ToolInputError("declined: expected_effect is None — cannot simulate a lift")

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="declining_tool",
            description="Declines deterministically",
            source_agent="test",
            tier=1,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=declining_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="declining_tool",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(self._single_step_plan(sample_decomposition, step))

        assert calls["n"] == 1, "deterministic input rejection must not be retried"
        result = trace.step_results[0]
        assert result.status == ExecutionStatus.FAILED
        err = result.output.error or ""
        assert "declined" in err
        assert "expected_effect" in err

    @pytest.mark.asyncio
    async def test_transient_tool_errors_are_still_retried(
        self, mock_tool_registry, sample_decomposition
    ):
        """Regression guard: plain exceptions keep the existing retry behavior."""
        calls = {"n": 0}

        def flaky_tool(**kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("transient")
            return {"ok": True}

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="flaky_tool_1573",
            description="Fails once then succeeds",
            source_agent="test",
            tier=1,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=flaky_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="flaky_tool_1573",
            source_agent="test",
            input_mapping={},
            depends_on_steps=[],
        )
        executor = PlanExecutor(
            tool_registry=mock_tool_registry, max_retries=2, backoff_base_delay=0.01
        )
        trace = await executor.execute(self._single_step_plan(sample_decomposition, step))

        assert calls["n"] == 2
        assert trace.tools_succeeded == 1

    @pytest.mark.asyncio
    async def test_unresolvable_reference_reason_reaches_synthesis_input(
        self, mock_tool_registry, sample_decomposition
    ):
        """(d): the TRUE cause must reach the synthesis input and prompt."""
        from src.agents.tool_composer.models.composition_models import SynthesisInput
        from src.agents.tool_composer.synthesizer import ResponseSynthesizer

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="any_tool",
            description="Never reached",
            source_agent="test",
            tier=1,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=lambda **k: {"x": 1})

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="any_tool",
            source_agent="test",
            input_mapping={"expected_effect": "$dataset.cate_estimate"},
            depends_on_steps=[],
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(self._single_step_plan(sample_decomposition, step))

        synthesis_input = SynthesisInput(
            original_query=sample_decomposition.original_query,
            decomposition=sample_decomposition,
            execution_trace=trace,
        )
        ctx = synthesis_input.get_context_for_synthesis()
        failed = [r for r in ctx["results"] if not r["success"]]
        assert failed, "the failed step must appear in the synthesis context"
        assert "$dataset.cate_estimate" in (failed[0]["error"] or "")

        # And the synthesizer's prompt formatter carries the same reason.
        formatted = ResponseSynthesizer(llm_client=object())._format_results(synthesis_input)
        assert "$dataset.cate_estimate" in formatted

    @pytest.mark.asyncio
    async def test_nested_dict_reference_still_degrades_to_none_f7(
        self, mock_tool_registry, sample_decomposition
    ):
        """(f) F7 GUARD: nested refs (the discover_dag data artifact) stay lenient.

        q08's five WORKING steps resolved planner-invented ``$dataset.*`` refs
        nested inside dict params to None and still succeeded (the F7 DataFrame
        auto-injection repairs the broken ``data`` dict downstream). Top-level
        strictness must NOT extend into nested constructions.
        """
        captured = {}

        def dict_consuming_tool(**kwargs):
            captured["data"] = kwargs.get("data")
            return {"edge_list": []}

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="dict_consuming_tool",
            description="Consumes a dict param",
            source_agent="test",
            tier=1,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=dict_consuming_tool)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="dict_consuming_tool",
            source_agent="test",
            input_mapping={
                "data": {
                    "patient_journey_id": "patient_journey_id",
                    "conversion_rate": "$dataset.conversion_rate",
                }
            },
            depends_on_steps=[],
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry)
        trace = await executor.execute(self._single_step_plan(sample_decomposition, step))

        assert trace.step_results[0].status == ExecutionStatus.COMPLETED
        assert captured["data"] == {
            "patient_journey_id": "patient_journey_id",
            "conversion_rate": None,
        }

    def test_counterfactual_simulator_declines_none_expected_effect(self):
        """(e): stated-reason decline instead of ``NoneType * float`` TypeError."""
        from src.agents.tool_composer import tool_registrations as tr
        from src.agents.tool_composer.errors import ToolInputError

        with pytest.raises(ToolInputError, match="expected_effect"):
            tr.counterfactual_simulator(
                intervention="increase rep visits",
                target_entities=["west"],
                expected_effect=None,
            )

    @pytest.mark.asyncio
    async def test_real_simulator_none_from_context_fails_once_with_reason(
        self, mock_tool_registry, sample_decomposition
    ):
        """(c)+(e) end-to-end: a LEGITIMATELY-resolved None (context field whose
        value is None) reaches the real simulator, which declines once — the
        executor must not retry the deterministic rejection."""
        from src.agents.tool_composer import tool_registrations as tr

        calls = {"n": 0}

        def counting_simulator(**kwargs):
            calls["n"] += 1
            return tr.counterfactual_simulator(**kwargs)

        mock_tool_registry.clear()
        schema = ToolSchema(
            name="counterfactual_simulator",
            description="Real simulator behind a call counter",
            source_agent="experiment_designer",
            tier=3,
            avg_execution_ms=10,
        )
        mock_tool_registry.register(schema=schema, callable=counting_simulator)

        step = ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="counterfactual_simulator",
            source_agent="experiment_designer",
            input_mapping={
                "intervention": "increase rep visits",
                "target_entities": ["west"],
                "expected_effect": "$context.expected_effect",
            },
            depends_on_steps=[],
        )
        executor = PlanExecutor(tool_registry=mock_tool_registry, max_retries=2)
        trace = await executor.execute(
            self._single_step_plan(sample_decomposition, step),
            context={"expected_effect": None},
        )

        assert calls["n"] == 1, "deterministic None rejection must not be retried"
        result = trace.step_results[0]
        assert result.status == ExecutionStatus.FAILED
        assert "expected_effect" in (result.output.error or "")


class TestToolFailureStatsG8:
    """Tests for G8 performance learning metrics in ToolFailureStats."""

    def test_ema_latency_first_observation(self):
        """First observation sets EMA directly."""
        stats = ToolFailureStats()
        assert stats.ema_latency_ms == 0.0

        stats.update_ema_latency(100)
        assert stats.ema_latency_ms == 100.0

    def test_ema_latency_smoothing(self):
        """EMA smooths latency over multiple observations."""
        stats = ToolFailureStats(ema_alpha=0.2)

        # First observation
        stats.update_ema_latency(100)
        assert stats.ema_latency_ms == 100.0

        # Second observation: EMA = 0.2*200 + 0.8*100 = 40 + 80 = 120
        stats.update_ema_latency(200)
        assert abs(stats.ema_latency_ms - 120.0) < 0.01

        # Third observation: EMA = 0.2*100 + 0.8*120 = 20 + 96 = 116
        stats.update_ema_latency(100)
        assert abs(stats.ema_latency_ms - 116.0) < 0.01

    def test_ema_responds_to_spikes(self):
        """EMA responds to latency spikes but smooths them."""
        stats = ToolFailureStats(ema_alpha=0.2)

        # Establish baseline
        for _ in range(5):
            stats.update_ema_latency(100)

        baseline = stats.ema_latency_ms

        # Introduce spike
        stats.update_ema_latency(500)

        # EMA should increase but not jump to 500
        assert stats.ema_latency_ms > baseline
        assert stats.ema_latency_ms < 500

    def test_sliding_window_records_results(self):
        """Sliding window records success/failure results."""
        stats = ToolFailureStats(sliding_window_size=5)
        assert len(stats.recent_results) == 0

        stats.record_result(True)
        stats.record_result(True)
        stats.record_result(False)

        assert len(stats.recent_results) == 3
        assert stats.recent_results == [True, True, False]

    def test_sliding_window_trims_to_size(self):
        """Sliding window trims old results."""
        stats = ToolFailureStats(sliding_window_size=3)

        stats.record_result(True)
        stats.record_result(False)
        stats.record_result(True)
        stats.record_result(False)  # Should trim first True
        stats.record_result(True)  # Should trim first False

        assert len(stats.recent_results) == 3
        assert stats.recent_results == [True, False, True]

    def test_recent_success_rate_empty(self):
        """Empty window returns 100% success rate."""
        stats = ToolFailureStats()
        assert stats.recent_success_rate == 1.0

    def test_recent_success_rate_calculation(self):
        """Recent success rate calculates correctly."""
        stats = ToolFailureStats()

        stats.record_result(True)
        stats.record_result(True)
        stats.record_result(False)
        stats.record_result(True)

        # 3 successes out of 4 = 75%
        assert stats.recent_success_rate == 0.75

    def test_recent_success_rate_all_failures(self):
        """Recent success rate is 0 when all failures."""
        stats = ToolFailureStats()

        stats.record_result(False)
        stats.record_result(False)
        stats.record_result(False)

        assert stats.recent_success_rate == 0.0


class TestToolFailureTrackerG8:
    """Tests for G8 integration in ToolFailureTracker."""

    def test_record_success_updates_ema(self):
        """record_success updates EMA latency."""
        tracker = ToolFailureTracker()

        tracker.record_success("test_tool", 100)
        stats = tracker.get_stats("test_tool")
        assert stats.ema_latency_ms == 100.0

        tracker.record_success("test_tool", 200)
        # EMA with alpha=0.2: 0.2*200 + 0.8*100 = 120
        assert abs(stats.ema_latency_ms - 120.0) < 0.01

    def test_record_success_records_sliding_window(self):
        """record_success records to sliding window."""
        tracker = ToolFailureTracker()

        tracker.record_success("test_tool", 100)
        stats = tracker.get_stats("test_tool")

        assert len(stats.recent_results) == 1
        assert stats.recent_results[-1] is True

    def test_record_failure_records_sliding_window(self):
        """record_failure records to sliding window."""
        tracker = ToolFailureTracker()

        tracker.record_failure("test_tool", "error")
        stats = tracker.get_stats("test_tool")

        assert len(stats.recent_results) == 1
        assert stats.recent_results[-1] is False

    def test_get_all_stats_includes_g8_metrics(self):
        """get_all_stats includes G8 metrics."""
        tracker = ToolFailureTracker()

        tracker.record_success("test_tool", 150)
        tracker.record_success("test_tool", 250)
        tracker.record_failure("test_tool", "error")

        all_stats = tracker.get_all_stats()

        assert "test_tool" in all_stats
        assert "ema_latency_ms" in all_stats["test_tool"]
        assert "recent_success_rate" in all_stats["test_tool"]

        # Check values
        assert all_stats["test_tool"]["ema_latency_ms"] > 0
        # 2 successes, 1 failure = ~66.7%
        assert abs(all_stats["test_tool"]["recent_success_rate"] - 2 / 3) < 0.01


class TestUpdateToolPerformance:
    """Tests for the update_tool_performance method (G8)."""

    def test_update_requires_minimum_calls(self, mock_tool_registry):
        """update_tool_performance skips tools with too few calls."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Record fewer than min_calls
        for _i in range(5):
            executor.failure_tracker.record_success("causal_effect_estimator", 100)

        results = executor.update_tool_performance(min_calls=10)

        assert results.get("causal_effect_estimator") is False

    def test_update_with_enough_calls(self, mock_tool_registry):
        """update_tool_performance updates registry with enough calls."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Get original latency
        original = mock_tool_registry.get_schema("causal_effect_estimator")
        original_latency = original.avg_execution_ms

        # Record enough calls with consistent latency
        for _ in range(15):
            executor.failure_tracker.record_success("causal_effect_estimator", 200)

        results = executor.update_tool_performance(min_calls=10)

        assert results.get("causal_effect_estimator") is True

        # Check registry was updated
        updated = mock_tool_registry.get_schema("causal_effect_estimator")
        assert updated.avg_execution_ms != original_latency
        assert updated.avg_execution_ms == 200  # EMA converges to constant value

    def test_update_specific_tool(self, mock_tool_registry):
        """update_tool_performance can target a specific tool."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Record for two tools
        for _ in range(15):
            executor.failure_tracker.record_success("causal_effect_estimator", 200)
            executor.failure_tracker.record_success("cate_analyzer", 300)

        # Update only one
        results = executor.update_tool_performance(
            tool_name="causal_effect_estimator", min_calls=10
        )

        assert "causal_effect_estimator" in results
        assert "cate_analyzer" not in results

    def test_update_nonexistent_tool(self, mock_tool_registry):
        """update_tool_performance handles tools not in registry."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Record for a fake tool
        for _ in range(15):
            executor.failure_tracker.record_success("nonexistent_tool", 100)

        results = executor.update_tool_performance(min_calls=10)

        assert results.get("nonexistent_tool") is False

    def test_update_all_tools(self, mock_tool_registry):
        """update_tool_performance can update all tools at once."""
        executor = PlanExecutor(tool_registry=mock_tool_registry)

        # Record for multiple tools
        for _ in range(15):
            executor.failure_tracker.record_success("causal_effect_estimator", 150)
            executor.failure_tracker.record_success("cate_analyzer", 250)
            executor.failure_tracker.record_success("gap_calculator", 100)

        results = executor.update_tool_performance(min_calls=10)

        # All should be updated
        assert results.get("causal_effect_estimator") is True
        assert results.get("cate_analyzer") is True
        assert results.get("gap_calculator") is True

        # Check registry values
        assert mock_tool_registry.get_schema("causal_effect_estimator").avg_execution_ms == 150
        assert mock_tool_registry.get_schema("cate_analyzer").avg_execution_ms == 250
        assert mock_tool_registry.get_schema("gap_calculator").avg_execution_ms == 100
