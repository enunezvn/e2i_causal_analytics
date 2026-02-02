"""Tests for Resource Optimizer Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Resource Optimizer agent's optimization pipeline.
"""

from unittest.mock import MagicMock

import pytest

from src.agents.resource_optimizer.opik_tracer import (
    OPTIMIZATION_OBJECTIVES,
    PIPELINE_NODES,
    SOLVER_TYPES,
    NodeSpanContext,
    OptimizationTraceContext,
    ResourceOptimizerOpikTracer,
    get_resource_optimizer_tracer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    ResourceOptimizerOpikTracer._instance = None
    ResourceOptimizerOpikTracer._initialized = False
    # Also reset module-level singleton
    import src.agents.resource_optimizer.opik_tracer as tracer_module

    tracer_module._tracer_instance = None
    yield
    ResourceOptimizerOpikTracer._instance = None
    ResourceOptimizerOpikTracer._initialized = False
    tracer_module._tracer_instance = None


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_optimization_objectives(self):
        """Test OPTIMIZATION_OBJECTIVES constant."""
        assert "maximize_outcome" in OPTIMIZATION_OBJECTIVES
        assert "maximize_roi" in OPTIMIZATION_OBJECTIVES
        assert "minimize_cost" in OPTIMIZATION_OBJECTIVES
        assert "balance" in OPTIMIZATION_OBJECTIVES

    def test_solver_types(self):
        """Test SOLVER_TYPES constant."""
        assert "linear" in SOLVER_TYPES
        assert "milp" in SOLVER_TYPES
        assert "nonlinear" in SOLVER_TYPES

    def test_pipeline_nodes(self):
        """Test PIPELINE_NODES constant."""
        assert "formulate" in PIPELINE_NODES
        assert "optimize" in PIPELINE_NODES
        assert "scenario" in PIPELINE_NODES
        assert "project" in PIPELINE_NODES


# ============================================================================
# NODE SPAN CONTEXT TESTS
# ============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_create_node_span_context(self):
        """Test NodeSpanContext creation with required fields."""
        ctx = NodeSpanContext(
            span=None,
            node_name="formulate",
        )
        assert ctx.span is None
        assert ctx.node_name == "formulate"
        assert ctx.start_time > 0
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to span context."""
        ctx = NodeSpanContext(
            span=None,
            node_name="optimize",
        )
        ctx.add_metadata("key", "value")
        assert ctx.metadata["key"] == "value"

    def test_end_without_span(self):
        """Test ending span context without Opik span."""
        ctx = NodeSpanContext(
            span=None,
            node_name="formulate",
        )
        ctx.add_metadata("test", "value")
        # Should not raise
        ctx.end(status="completed")

    def test_end_with_mock_span(self):
        """Test ending span context with mock Opik span."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="optimize",
        )
        ctx.add_metadata("solver", "linear")
        ctx.end(status="completed")
        mock_span.end.assert_called_once()

    def test_end_handles_span_error(self):
        """Test that end handles span errors gracefully."""
        mock_span = MagicMock()
        mock_span.end.side_effect = Exception("Span error")
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="formulate",
        )
        # Should not raise
        ctx.end(status="completed")


# ============================================================================
# OPTIMIZATION TRACE CONTEXT TESTS
# ============================================================================


class TestOptimizationTraceContext:
    """Tests for OptimizationTraceContext class."""

    def test_create_trace_context(self):
        """Test OptimizationTraceContext creation."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        assert ctx.trace is None
        assert ctx.tracer is tracer
        assert ctx.resource_type == "budget"
        assert ctx.objective == "maximize_outcome"
        assert ctx.start_time > 0
        assert ctx.trace_metadata == {}
        assert ctx.active_spans == {}

    def test_log_optimization_started(self):
        """Test logging optimization started."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        # Should not raise without trace
        ctx.log_optimization_started(
            resource_type="budget",
            objective="maximize_outcome",
            solver_type="linear",
            target_count=50,
            constraint_count=10,
            run_scenarios=True,
        )

    def test_log_optimization_started_with_trace(self):
        """Test logging optimization started with mock trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.log_optimization_started(
            resource_type="budget",
            objective="maximize_outcome",
            solver_type="linear",
            target_count=50,
            constraint_count=10,
            run_scenarios=True,
        )
        assert ctx.trace_metadata.get("resource_type") == "budget"
        assert ctx.trace_metadata.get("solver_type") == "linear"
        mock_trace.update.assert_called()

    def test_log_problem_formulation(self):
        """Test logging problem formulation."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_roi",
        )
        # Should not raise without trace
        ctx.log_problem_formulation(
            target_count=50,
            constraint_count=10,
            variables_count=100,
            formulation_latency_ms=150,
        )

    def test_log_problem_formulation_with_trace(self):
        """Test logging problem formulation with trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_roi",
        )
        ctx.log_problem_formulation(
            target_count=50,
            constraint_count=10,
            variables_count=100,
            formulation_latency_ms=150,
        )
        assert ctx.trace_metadata.get("target_count") == 50
        assert ctx.trace_metadata.get("constraint_count") == 10
        assert ctx.trace_metadata.get("variables_count") == 100
        mock_trace.update.assert_called()

    def test_log_solver_execution(self):
        """Test logging solver execution."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        # Should not raise without trace
        ctx.log_solver_execution(
            solver_type="linear",
            solver_status="optimal",
            solve_time_ms=150,
            objective_value=450000.0,
            allocations_count=50,
        )

    def test_log_solver_execution_with_trace(self):
        """Test logging solver execution with trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.log_solver_execution(
            solver_type="linear",
            solver_status="optimal",
            solve_time_ms=150,
            objective_value=450000.0,
            allocations_count=50,
        )
        assert ctx.trace_metadata.get("solver_type") == "linear"
        assert ctx.trace_metadata.get("solver_status") == "optimal"
        assert ctx.trace_metadata.get("solve_time_ms") == 150
        mock_trace.update.assert_called()

    def test_log_scenario_analysis(self):
        """Test logging scenario analysis."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        # Should not raise without trace
        ctx.log_scenario_analysis(
            scenario_count=3,
            best_scenario="aggressive",
            scenario_latency_ms=200,
        )

    def test_log_scenario_analysis_with_trace(self):
        """Test logging scenario analysis with trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.log_scenario_analysis(
            scenario_count=3,
            best_scenario="aggressive",
            scenario_latency_ms=200,
        )
        assert ctx.trace_metadata.get("scenario_count") == 3
        assert ctx.trace_metadata.get("best_scenario") == "aggressive"
        mock_trace.update.assert_called()

    def test_log_impact_projection(self):
        """Test logging impact projection."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        # Should not raise without trace
        ctx.log_impact_projection(
            projected_outcome=450000.0,
            projected_roi=2.25,
            segments_count=5,
            projection_latency_ms=100,
        )

    def test_log_impact_projection_with_trace(self):
        """Test logging impact projection with trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.log_impact_projection(
            projected_outcome=450000.0,
            projected_roi=2.25,
            segments_count=5,
            projection_latency_ms=100,
        )
        assert ctx.trace_metadata.get("projected_outcome") == 450000.0
        assert ctx.trace_metadata.get("projected_roi") == 2.25
        mock_trace.update.assert_called()

    def test_log_optimization_complete(self):
        """Test logging optimization complete."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        # Should not raise without trace
        ctx.log_optimization_complete(
            status="completed",
            success=True,
            total_duration_ms=2000,
            objective_value=450000.0,
            solver_status="optimal",
            projected_outcome=450000.0,
            projected_roi=2.25,
            allocations_count=50,
            increases_count=20,
            decreases_count=10,
            recommendations=["Increase budget for Northeast by 15%"],
            errors=[],
            warnings=[],
        )

    def test_log_optimization_complete_with_trace(self):
        """Test logging optimization complete with trace."""
        mock_trace = MagicMock()
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=mock_trace,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.log_optimization_complete(
            status="completed",
            success=True,
            total_duration_ms=2000,
            objective_value=450000.0,
            solver_status="optimal",
            projected_outcome=450000.0,
            projected_roi=2.25,
            allocations_count=50,
            increases_count=20,
            decreases_count=10,
            recommendations=["Increase budget for Northeast by 15%"],
            errors=[],
            warnings=[],
        )
        assert ctx.trace_metadata.get("status") == "completed"
        assert ctx.trace_metadata.get("success") is True
        assert ctx.trace_metadata.get("objective_value") == 450000.0
        assert ctx.trace_metadata.get("projected_roi") == 2.25
        mock_trace.update.assert_called()

    def test_start_node_span_without_trace(self):
        """Test starting a node span without trace."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        node_ctx = ctx.start_node_span("formulate", {"constraints": 10})
        assert isinstance(node_ctx, NodeSpanContext)
        assert node_ctx.node_name == "formulate"
        assert node_ctx.span is None
        assert "formulate" in ctx.active_spans

    def test_end_node_span(self):
        """Test ending a node span."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        ctx = OptimizationTraceContext(
            trace=None,
            tracer=tracer,
            resource_type="budget",
            objective="maximize_outcome",
        )
        ctx.start_node_span("optimize")
        assert "optimize" in ctx.active_spans
        ctx.end_node_span("optimize", {"status": "optimal"}, "completed")
        assert "optimize" not in ctx.active_spans


# ============================================================================
# RESOURCE OPTIMIZER OPIK TRACER TESTS
# ============================================================================


class TestResourceOptimizerOpikTracer:
    """Tests for ResourceOptimizerOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = ResourceOptimizerOpikTracer()
        assert tracer.project_name == "e2i-resource-optimizer"
        assert tracer.enabled is True
        assert tracer.sampling_rate == 1.0
        assert tracer._client is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = ResourceOptimizerOpikTracer(
            project_name="custom-project",
            enabled=False,
            sampling_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sampling_rate == 0.5

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        tracer1 = ResourceOptimizerOpikTracer()
        tracer2 = ResourceOptimizerOpikTracer()
        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test that singleton skips reinitialization."""
        tracer1 = ResourceOptimizerOpikTracer(project_name="first")
        tracer2 = ResourceOptimizerOpikTracer(project_name="second")
        assert tracer1 is tracer2
        assert tracer1.project_name == "first"

    def test_get_client_disabled(self):
        """Test _get_client returns None when disabled."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        assert tracer._get_client() is None

    def test_should_sample_full_rate(self):
        """Test _should_sample at 100% rate."""
        tracer = ResourceOptimizerOpikTracer(sampling_rate=1.0)
        assert tracer._should_sample() is True

    def test_should_sample_zero_rate(self):
        """Test _should_sample at 0% rate."""
        tracer = ResourceOptimizerOpikTracer(sampling_rate=0.0)
        assert tracer._should_sample() is False

    def test_generate_trace_id_format(self):
        """Test trace ID generation format."""
        tracer = ResourceOptimizerOpikTracer()
        trace_id = tracer._generate_trace_id()
        # Should be UUID format (with hyphens)
        assert len(trace_id) == 36
        assert trace_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_trace_optimization_disabled(self):
        """Test trace_optimization when disabled."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        async with tracer.trace_optimization(
            resource_type="budget",
            objective="maximize_outcome",
        ) as ctx:
            assert ctx is not None
            assert ctx.resource_type == "budget"
            assert ctx.objective == "maximize_outcome"

    @pytest.mark.asyncio
    async def test_trace_optimization_not_sampled(self):
        """Test trace_optimization when not sampled."""
        tracer = ResourceOptimizerOpikTracer(sampling_rate=0.0)
        async with tracer.trace_optimization(
            resource_type="rep_time",
            objective="maximize_roi",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace is None

    @pytest.mark.asyncio
    async def test_trace_optimization_full_pipeline(self):
        """Test full optimization pipeline trace."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        async with tracer.trace_optimization(
            resource_type="budget",
            objective="maximize_outcome",
            solver_type="linear",
            query="Optimize budget allocation",
        ) as ctx:
            # Simulate full pipeline
            ctx.log_optimization_started(
                resource_type="budget",
                objective="maximize_outcome",
                solver_type="linear",
                target_count=50,
                constraint_count=10,
                run_scenarios=True,
            )

            ctx.start_node_span("formulate", {"constraints": 10})
            ctx.end_node_span("formulate", {"variables": 100})
            ctx.log_problem_formulation(
                target_count=50,
                constraint_count=10,
                variables_count=100,
                formulation_latency_ms=50,
            )

            ctx.start_node_span("optimize")
            ctx.end_node_span("optimize", {"status": "optimal"})
            ctx.log_solver_execution(
                solver_type="linear",
                solver_status="optimal",
                solve_time_ms=150,
                objective_value=450000.0,
                allocations_count=50,
            )

            ctx.start_node_span("scenario")
            ctx.end_node_span("scenario", {"best": "aggressive"})
            ctx.log_scenario_analysis(
                scenario_count=3,
                best_scenario="aggressive",
                scenario_latency_ms=200,
            )

            ctx.start_node_span("project")
            ctx.end_node_span("project", {"roi": 2.25})
            ctx.log_impact_projection(
                projected_outcome=450000.0,
                projected_roi=2.25,
                segments_count=5,
                projection_latency_ms=100,
            )

            ctx.log_optimization_complete(
                status="completed",
                success=True,
                total_duration_ms=500,
                objective_value=450000.0,
                solver_status="optimal",
                projected_outcome=450000.0,
                projected_roi=2.25,
                allocations_count=50,
                increases_count=20,
                decreases_count=10,
                recommendations=["Increase Northeast budget"],
                errors=[],
                warnings=[],
            )

        # Verify all spans were ended
        assert len(ctx.active_spans) == 0

    def test_flush_without_client(self):
        """Test flush when client is not initialized."""
        tracer = ResourceOptimizerOpikTracer(enabled=False)
        # Should not raise
        tracer.flush()


# ============================================================================
# SINGLETON FUNCTIONS TESTS
# ============================================================================


class TestGetResourceOptimizerTracer:
    """Tests for get_resource_optimizer_tracer function."""

    def test_returns_tracer_instance(self):
        """Test that get_resource_optimizer_tracer returns a tracer."""
        tracer = get_resource_optimizer_tracer()
        assert isinstance(tracer, ResourceOptimizerOpikTracer)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        tracer1 = get_resource_optimizer_tracer()
        tracer2 = get_resource_optimizer_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_resource_optimizer_tracer(
            project_name="custom-project",
            sampling_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sampling_rate == 0.5


# ============================================================================
# INTEGRATION TESTS (WITH MOCKED OPIK)
# ============================================================================


class TestOpikIntegration:
    """Integration tests with mocked Opik client."""

    @pytest.mark.asyncio
    async def test_full_trace_with_opik(self):
        """Test full tracing with mocked Opik client."""
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_trace.span.return_value = mock_span

        tracer = ResourceOptimizerOpikTracer()
        tracer._client = mock_client

        async with tracer.trace_optimization(
            resource_type="budget",
            objective="maximize_outcome",
        ) as ctx:
            ctx.log_optimization_started(
                resource_type="budget",
                objective="maximize_outcome",
                solver_type="linear",
                target_count=50,
                constraint_count=10,
                run_scenarios=True,
            )
            ctx.log_problem_formulation(
                target_count=50,
                constraint_count=10,
                variables_count=100,
                formulation_latency_ms=50,
            )
            ctx.log_solver_execution(
                solver_type="linear",
                solver_status="optimal",
                solve_time_ms=150,
                objective_value=450000.0,
                allocations_count=50,
            )
            ctx.log_optimization_complete(
                status="completed",
                success=True,
                total_duration_ms=2000,
                objective_value=450000.0,
                solver_status="optimal",
                projected_outcome=450000.0,
                projected_roi=2.25,
                allocations_count=50,
                increases_count=20,
                decreases_count=10,
                recommendations=["Increase Northeast budget"],
                errors=[],
                warnings=[],
            )

        mock_client.trace.assert_called_once()
        mock_trace.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_client_error(self):
        """Test that errors during tracing don't break execution."""
        tracer = ResourceOptimizerOpikTracer()

        # Mock client that raises
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Opik error")
        tracer._client = mock_client

        # Should not raise
        async with tracer.trace_optimization(
            resource_type="budget",
            objective="maximize_outcome",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace is None
            ctx.log_optimization_complete(
                status="completed",
                success=True,
                total_duration_ms=1000,
                objective_value=450000.0,
                solver_status="optimal",
                projected_outcome=450000.0,
                projected_roi=2.25,
                allocations_count=50,
                increases_count=20,
                decreases_count=10,
                recommendations=[],
                errors=[],
                warnings=[],
            )
