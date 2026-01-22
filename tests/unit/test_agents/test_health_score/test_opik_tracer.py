"""Tests for Health Score Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Health Score agent's fast path pipeline.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.health_score import opik_tracer as health_score_opik_module
from src.agents.health_score.opik_tracer import (
    GRADE_THRESHOLDS,
    HEALTH_CHECK_TYPES,
    HEALTH_NODES,
    HealthCheckTraceContext,
    HealthScoreOpikTracer,
    NodeSpanContext,
    get_health_score_tracer,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    HealthScoreOpikTracer._instance = None
    HealthScoreOpikTracer._initialized = False
    health_score_opik_module._tracer_instance = None
    yield
    HealthScoreOpikTracer._instance = None
    HealthScoreOpikTracer._initialized = False
    health_score_opik_module._tracer_instance = None


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_health_check_types(self):
        """Test HEALTH_CHECK_TYPES constant."""
        assert "full" in HEALTH_CHECK_TYPES
        assert "quick" in HEALTH_CHECK_TYPES
        assert "models" in HEALTH_CHECK_TYPES
        assert "pipelines" in HEALTH_CHECK_TYPES
        assert "agents" in HEALTH_CHECK_TYPES

    def test_health_nodes(self):
        """Test HEALTH_NODES constant."""
        assert "component" in HEALTH_NODES
        assert "model" in HEALTH_NODES
        assert "pipeline" in HEALTH_NODES
        assert "agent" in HEALTH_NODES
        assert "compose" in HEALTH_NODES

    def test_grade_thresholds(self):
        """Test GRADE_THRESHOLDS constant."""
        assert GRADE_THRESHOLDS["A"] == 90
        assert GRADE_THRESHOLDS["B"] == 80
        assert GRADE_THRESHOLDS["C"] == 70
        assert GRADE_THRESHOLDS["D"] == 60
        assert GRADE_THRESHOLDS["F"] == 0


# ============================================================================
# NODE SPAN CONTEXT TESTS
# ============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_create_node_span_context(self):
        """Test NodeSpanContext creation with required fields."""
        ctx = NodeSpanContext(
            span=None,
            node_name="component",
        )
        assert ctx.span is None
        assert ctx.node_name == "component"
        assert ctx.start_time > 0
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to span context."""
        ctx = NodeSpanContext(
            span=None,
            node_name="component",
        )
        ctx.add_metadata("key", "value")
        assert ctx.metadata["key"] == "value"

    def test_end_without_span(self):
        """Test ending span context without Opik span."""
        ctx = NodeSpanContext(
            span=None,
            node_name="component",
        )
        ctx.add_metadata("test", "value")
        # Should not raise
        ctx.end(status="completed")

    def test_end_with_mock_span(self):
        """Test ending span context with mock Opik span."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="model",
        )
        ctx.add_metadata("accuracy", 0.95)
        ctx.end(status="completed")
        mock_span.end.assert_called_once()

    def test_end_handles_span_error(self):
        """Test that end handles span errors gracefully."""
        mock_span = MagicMock()
        mock_span.end.side_effect = Exception("Span error")
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="component",
        )
        # Should not raise
        ctx.end(status="completed")


# ============================================================================
# HEALTH CHECK TRACE CONTEXT TESTS
# ============================================================================


class TestHealthCheckTraceContext:
    """Tests for HealthCheckTraceContext class."""

    def test_create_trace_context(self):
        """Test HealthCheckTraceContext creation."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        assert ctx.trace is None
        assert ctx.tracer is tracer
        assert ctx.check_scope == "full"

    def test_log_check_started(self):
        """Test logging health check started."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_check_started(check_scope="full")
        # No trace, so metadata won't be stored

    def test_log_component_health(self):
        """Test logging component health."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_component_health(
            score=0.95,
            statuses={"database": "healthy", "cache": "healthy"},
            issues=[],
            duration_ms=150,
        )
        # No trace, so won't store but should not raise

    def test_log_model_health(self):
        """Test logging model health."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_model_health(
            score=0.85,
            model_count=5,
            degraded_models=["churn_predictor"],
            duration_ms=250,
        )
        # No trace, so won't store but should not raise

    def test_log_pipeline_health(self):
        """Test logging pipeline health."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_pipeline_health(
            score=0.90,
            pipeline_count=4,
            stale_pipelines=["legacy_etl"],
            duration_ms=200,
        )
        # No trace, so won't store but should not raise

    def test_log_agent_health(self):
        """Test logging agent health."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_agent_health(
            score=0.92,
            agent_count=18,
            unavailable_agents=["experiment_designer"],
            duration_ms=180,
        )
        # No trace, so won't store but should not raise

    def test_log_check_complete(self):
        """Test logging health check complete."""
        tracer = HealthScoreOpikTracer(enabled=False)
        ctx = HealthCheckTraceContext(
            trace=None,
            tracer=tracer,
            check_scope="full",
        )
        ctx.log_check_complete(
            status="completed",
            success=True,
            total_duration_ms=1500,
            overall_score=85.5,
            health_grade="B",
            component_score=0.90,
            model_score=0.85,
            pipeline_score=0.85,
            agent_score=0.90,
            critical_issues=[],
            warnings=["Model degraded accuracy"],
        )
        # No trace, so won't store but should not raise


# ============================================================================
# HEALTH SCORE OPIK TRACER TESTS
# ============================================================================


class TestHealthScoreOpikTracer:
    """Tests for HealthScoreOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = HealthScoreOpikTracer()
        assert tracer.project_name == "e2i-health-score"
        assert tracer.enabled is True
        assert tracer.sampling_rate == 1.0
        assert tracer._client is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = HealthScoreOpikTracer(
            project_name="custom-project",
            enabled=False,
            sampling_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sampling_rate == 0.5

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        tracer1 = HealthScoreOpikTracer()
        tracer2 = HealthScoreOpikTracer()
        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test that singleton skips reinitialization."""
        tracer1 = HealthScoreOpikTracer(project_name="first")
        tracer2 = HealthScoreOpikTracer(project_name="second")
        assert tracer1 is tracer2
        assert tracer1.project_name == "first"

    def test_get_client_disabled(self):
        """Test _get_client returns None when disabled."""
        tracer = HealthScoreOpikTracer(enabled=False)
        assert tracer._get_client() is None

    def test_should_sample_full_rate(self):
        """Test _should_sample at 100% rate."""
        tracer = HealthScoreOpikTracer(sampling_rate=1.0)
        assert tracer._should_sample() is True

    def test_should_sample_zero_rate(self):
        """Test _should_sample at 0% rate."""
        tracer = HealthScoreOpikTracer(sampling_rate=0.0)
        assert tracer._should_sample() is False

    def test_generate_trace_id_format(self):
        """Test trace ID generation format."""
        tracer = HealthScoreOpikTracer()
        trace_id = tracer._generate_trace_id()
        # Should be UUID format (with hyphens)
        assert len(trace_id) == 36
        assert trace_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_trace_health_check_disabled(self):
        """Test trace_health_check when disabled."""
        tracer = HealthScoreOpikTracer(enabled=False)
        async with tracer.trace_health_check(check_scope="full") as ctx:
            assert ctx is not None
            assert ctx.check_scope == "full"

    @pytest.mark.asyncio
    async def test_trace_health_check_not_sampled(self):
        """Test trace_health_check when not sampled."""
        tracer = HealthScoreOpikTracer(sampling_rate=0.0)
        async with tracer.trace_health_check(check_scope="quick") as ctx:
            assert ctx is not None


# ============================================================================
# SINGLETON FUNCTIONS TESTS
# ============================================================================


class TestGetHealthScoreTracer:
    """Tests for get_health_score_tracer function."""

    def test_returns_tracer_instance(self):
        """Test that get_health_score_tracer returns a tracer."""
        tracer = get_health_score_tracer()
        assert isinstance(tracer, HealthScoreOpikTracer)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        tracer1 = get_health_score_tracer()
        tracer2 = get_health_score_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_health_score_tracer(
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
        mock_client.trace.return_value = mock_trace

        tracer = HealthScoreOpikTracer()
        tracer._client = mock_client

        async with tracer.trace_health_check(check_scope="full") as ctx:
            assert ctx.check_scope == "full"
            ctx.log_check_started(check_scope="full")
            ctx.log_component_health(
                score=0.95,
                statuses={"database": "healthy"},
                issues=[],
                duration_ms=100,
            )
            ctx.log_check_complete(
                status="completed",
                success=True,
                total_duration_ms=1000,
                overall_score=95.0,
                health_grade="A",
                component_score=0.95,
                model_score=0.95,
                pipeline_score=0.95,
                agent_score=0.95,
                critical_issues=[],
                warnings=[],
            )

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_client_error(self):
        """Test that errors during tracing don't break execution."""
        tracer = HealthScoreOpikTracer()

        # Mock client that raises
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Opik error")
        tracer._client = mock_client

        # Should not raise
        async with tracer.trace_health_check(check_scope="quick") as ctx:
            assert ctx is not None
            ctx.log_check_complete(
                status="completed",
                success=True,
                total_duration_ms=500,
                overall_score=90.0,
                health_grade="A",
                component_score=0.90,
                model_score=0.90,
                pipeline_score=0.90,
                agent_score=0.90,
                critical_issues=[],
                warnings=[],
            )
