"""Tests for Heterogeneous Optimizer Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Heterogeneous Optimizer CATE pipeline.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.heterogeneous_optimizer.opik_tracer import (
    AGENT_METADATA,
    CATEAnalysisTraceContext,
    HeterogeneousOptimizerOpikTracer,
    NodeSpanContext,
    get_heterogeneous_optimizer_tracer,
    reset_heterogeneous_optimizer_tracer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    reset_heterogeneous_optimizer_tracer()
    yield
    reset_heterogeneous_optimizer_tracer()


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_agent_metadata(self):
        """Test AGENT_METADATA constant."""
        assert AGENT_METADATA["name"] == "heterogeneous_optimizer"
        assert AGENT_METADATA["tier"] == 2
        assert AGENT_METADATA["type"] == "standard"
        assert "pipeline" in AGENT_METADATA


# ============================================================================
# NODE SPAN CONTEXT TESTS
# ============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_create_node_span_context(self):
        """Test NodeSpanContext creation with required fields."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="estimate_cate",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.node_name == "estimate_cate"
        assert isinstance(ctx.start_time, datetime)
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to span context."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="estimate_cate",
        )
        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"

    def test_log_cate_estimation(self):
        """Test logging CATE estimation results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="estimate_cate",
        )
        ctx.log_cate_estimation(
            segments_count=10,
            overall_ate=0.15,
            heterogeneity_score=0.72,
            n_estimators=100,
            estimation_method="CausalForestDML",
        )
        assert ctx.metadata["segments_count"] == 10
        assert ctx.metadata["overall_ate"] == 0.15
        assert ctx.metadata["heterogeneity_score"] == 0.72
        assert ctx.metadata["n_estimators"] == 100
        assert ctx.metadata["estimation_method"] == "CausalForestDML"

    def test_log_cate_estimation_defaults(self):
        """Test logging CATE estimation with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="estimate_cate",
        )
        ctx.log_cate_estimation(
            segments_count=5,
            overall_ate=0.10,
            heterogeneity_score=0.50,
        )
        assert ctx.metadata["segments_count"] == 5
        assert ctx.metadata["overall_ate"] == 0.10
        assert ctx.metadata["heterogeneity_score"] == 0.50
        assert ctx.metadata["n_estimators"] == 100
        assert ctx.metadata["estimation_method"] == "CausalForestDML"

    def test_log_segment_analysis(self):
        """Test logging segment analysis results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="analyze_segments",
        )
        ctx.log_segment_analysis(
            high_responders_count=5,
            low_responders_count=3,
            total_segments_analyzed=10,
            significant_effects_count=4,
            max_cate=0.35,
            min_cate=-0.10,
        )
        assert ctx.metadata["high_responders_count"] == 5
        assert ctx.metadata["low_responders_count"] == 3
        assert ctx.metadata["total_segments_analyzed"] == 10
        assert ctx.metadata["significant_effects_count"] == 4
        assert ctx.metadata["max_cate"] == 0.35
        assert ctx.metadata["min_cate"] == -0.10

    def test_log_segment_analysis_defaults(self):
        """Test logging segment analysis with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="analyze_segments",
        )
        ctx.log_segment_analysis(
            high_responders_count=3,
            low_responders_count=2,
            total_segments_analyzed=8,
        )
        assert ctx.metadata["high_responders_count"] == 3
        assert ctx.metadata["low_responders_count"] == 2
        assert ctx.metadata["total_segments_analyzed"] == 8
        assert ctx.metadata["significant_effects_count"] == 0
        assert ctx.metadata["max_cate"] == 0.0
        assert ctx.metadata["min_cate"] == 0.0

    def test_log_policy_learning(self):
        """Test logging policy learning results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="learn_policy",
        )
        ctx.log_policy_learning(
            recommendations_count=5,
            expected_total_lift=0.25,
            reallocations_suggested=3,
            budget_impact=15000.0,
        )
        assert ctx.metadata["recommendations_count"] == 5
        assert ctx.metadata["expected_total_lift"] == 0.25
        assert ctx.metadata["reallocations_suggested"] == 3
        assert ctx.metadata["budget_impact"] == 15000.0

    def test_log_policy_learning_defaults(self):
        """Test logging policy learning with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="learn_policy",
        )
        ctx.log_policy_learning(
            recommendations_count=3,
            expected_total_lift=0.15,
        )
        assert ctx.metadata["recommendations_count"] == 3
        assert ctx.metadata["expected_total_lift"] == 0.15
        assert ctx.metadata["reallocations_suggested"] == 0
        assert ctx.metadata["budget_impact"] == 0.0

    def test_log_profile_generation(self):
        """Test logging profile generation results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate_profiles",
        )
        ctx.log_profile_generation(
            profiles_generated=5,
            insights_count=8,
            summary_length=1200,
        )
        assert ctx.metadata["profiles_generated"] == 5
        assert ctx.metadata["insights_count"] == 8
        assert ctx.metadata["summary_length"] == 1200

    def test_log_profile_generation_defaults(self):
        """Test logging profile generation with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate_profiles",
        )
        ctx.log_profile_generation(
            profiles_generated=3,
            insights_count=4,
        )
        assert ctx.metadata["profiles_generated"] == 3
        assert ctx.metadata["insights_count"] == 4
        assert ctx.metadata["summary_length"] == 0

    def test_set_output(self):
        """Test setting output on span context."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate_profiles",
            _opik_span=mock_span,
        )
        output = {"result": "test"}
        ctx.set_output(output)
        mock_span.set_output.assert_called_once_with(output)

    def test_set_output_without_span(self):
        """Test setting output without Opik span doesn't raise."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate_profiles",
        )
        # Should not raise
        ctx.set_output({"result": "test"})


# ============================================================================
# CATE ANALYSIS TRACE CONTEXT TESTS
# ============================================================================


class TestCATEAnalysisTraceContext:
    """Tests for CATEAnalysisTraceContext class."""

    def test_create_trace_context(self):
        """Test CATEAnalysisTraceContext creation."""
        ctx = CATEAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Which segments respond best?",
            treatment_var="rep_visits",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.query == "Which segments respond best?"
        assert ctx.treatment_var == "rep_visits"
        assert isinstance(ctx.start_time, datetime)
        assert ctx.node_spans == {}
        assert ctx.node_durations == {}
        assert ctx.metadata == {}

    def test_log_analysis_complete(self):
        """Test logging analysis complete."""
        ctx = CATEAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Which segments respond best?",
            treatment_var="rep_visits",
        )
        ctx.log_analysis_complete(
            status="completed",
            success=True,
            total_duration_ms=2500,
            overall_ate=0.15,
            heterogeneity_score=0.72,
            high_responders_count=5,
            low_responders_count=3,
            recommendations_count=4,
            expected_total_lift=0.25,
            confidence=0.88,
            errors=[],
            suggested_next_agent="resource_optimizer",
        )
        # Log method doesn't store in metadata directly, it logs to Opik

    def test_log_analysis_complete_with_errors(self):
        """Test logging analysis complete with errors."""
        ctx = CATEAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="Which segments respond best?",
            treatment_var="rep_visits",
        )
        ctx.log_analysis_complete(
            status="partial",
            success=False,
            total_duration_ms=1800,
            overall_ate=0.10,
            heterogeneity_score=0.50,
            high_responders_count=2,
            low_responders_count=1,
            recommendations_count=1,
            expected_total_lift=0.10,
            confidence=0.65,
            errors=["Insufficient data for some segments"],
        )
        # Should not raise


# ============================================================================
# HETEROGENEOUS OPTIMIZER OPIK TRACER TESTS
# ============================================================================


class TestHeterogeneousOptimizerOpikTracer:
    """Tests for HeterogeneousOptimizerOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = HeterogeneousOptimizerOpikTracer()
        assert tracer.project_name == "e2i-heterogeneous-optimizer"
        assert tracer.enabled is True
        assert tracer.sample_rate == 1.0
        assert tracer._opik_connector is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = HeterogeneousOptimizerOpikTracer(
            project_name="custom-project",
            enabled=False,
            sample_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sample_rate == 0.5

    def test_singleton_pattern(self):
        """Test singleton pattern via get_heterogeneous_optimizer_tracer."""
        tracer1 = get_heterogeneous_optimizer_tracer()
        tracer2 = get_heterogeneous_optimizer_tracer()
        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test that singleton skips reinitialization."""
        tracer1 = get_heterogeneous_optimizer_tracer(project_name="first")
        tracer2 = get_heterogeneous_optimizer_tracer(project_name="second")
        assert tracer1 is tracer2
        assert tracer1.project_name == "first"

    def test_reset_singleton(self):
        """Test resetting the singleton."""
        tracer1 = get_heterogeneous_optimizer_tracer()
        reset_heterogeneous_optimizer_tracer()
        tracer2 = get_heterogeneous_optimizer_tracer()
        assert tracer1 is not tracer2

    def test_should_trace_full_rate(self):
        """Test _should_trace at 100% rate."""
        tracer = HeterogeneousOptimizerOpikTracer(sample_rate=1.0)
        assert tracer._should_trace() is True

    def test_should_trace_zero_rate(self):
        """Test _should_trace at 0% rate."""
        tracer = HeterogeneousOptimizerOpikTracer(sample_rate=0.0)
        assert tracer._should_trace() is False

    @pytest.mark.asyncio
    async def test_trace_analysis_disabled(self):
        """Test trace_analysis when disabled."""
        tracer = HeterogeneousOptimizerOpikTracer(enabled=False)
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="rep_visits",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace_id is not None
            assert ctx.span_id is not None

    @pytest.mark.asyncio
    async def test_trace_analysis_not_sampled(self):
        """Test trace_analysis when not sampled."""
        tracer = HeterogeneousOptimizerOpikTracer(sample_rate=0.0)
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="rep_visits",
        ) as ctx:
            assert ctx is not None

    @pytest.mark.asyncio
    async def test_trace_analysis_with_metadata(self):
        """Test trace_analysis with additional metadata."""
        tracer = HeterogeneousOptimizerOpikTracer(enabled=False)
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="rep_visits",
            outcome_var="trx",
            segment_vars=["region", "specialty"],
            brand="Kisqali",
            session_id="session-123",
            metadata={"custom_key": "custom_value"},
        ) as ctx:
            assert ctx is not None
            assert ctx.treatment_var == "rep_visits"


# ============================================================================
# SINGLETON FUNCTIONS TESTS
# ============================================================================


class TestGetHeterogeneousOptimizerTracer:
    """Tests for get_heterogeneous_optimizer_tracer function."""

    def test_returns_tracer_instance(self):
        """Test that get_heterogeneous_optimizer_tracer returns a tracer."""
        tracer = get_heterogeneous_optimizer_tracer()
        assert isinstance(tracer, HeterogeneousOptimizerOpikTracer)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        tracer1 = get_heterogeneous_optimizer_tracer()
        tracer2 = get_heterogeneous_optimizer_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_heterogeneous_optimizer_tracer(
            project_name="custom-project",
            sample_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sample_rate == 0.5


class TestResetHeterogeneousOptimizerTracer:
    """Tests for reset_heterogeneous_optimizer_tracer function."""

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        tracer1 = get_heterogeneous_optimizer_tracer()
        reset_heterogeneous_optimizer_tracer()
        tracer2 = get_heterogeneous_optimizer_tracer()
        assert tracer1 is not tracer2


# ============================================================================
# INTEGRATION TESTS (WITH MOCKED OPIK)
# ============================================================================


class TestOpikIntegration:
    """Integration tests with mocked Opik connector."""

    @pytest.mark.asyncio
    async def test_full_trace_with_opik(self):
        """Test full tracing with mocked Opik connector."""
        mock_connector = MagicMock()
        mock_span_context = MagicMock()
        mock_connector.trace_agent = AsyncMock()
        mock_connector.trace_agent.return_value.__aenter__ = AsyncMock(
            return_value=mock_span_context
        )
        mock_connector.trace_agent.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_connector.is_enabled = True

        tracer = HeterogeneousOptimizerOpikTracer()
        tracer._opik_connector = mock_connector
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="Which segments respond best?",
            treatment_var="rep_visits",
        ) as trace:
            assert trace.trace_id is not None
            trace.log_analysis_complete(
                status="completed",
                success=True,
                total_duration_ms=2000,
                overall_ate=0.15,
                heterogeneity_score=0.72,
                high_responders_count=5,
                low_responders_count=3,
                recommendations_count=4,
                expected_total_lift=0.25,
                confidence=0.88,
            )

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_trace_error(self):
        """Test that errors during tracing don't break execution."""
        tracer = HeterogeneousOptimizerOpikTracer()

        # Mock connector that raises on trace_agent
        mock_connector = MagicMock()
        mock_connector.is_enabled = True
        mock_connector.trace_agent = MagicMock(side_effect=Exception("Opik error"))
        tracer._opik_connector = mock_connector
        tracer._initialized = True

        # Should not raise
        async with tracer.trace_analysis(
            query="Test query",
            treatment_var="rep_visits",
        ) as trace:
            assert trace is not None
            trace.log_analysis_complete(
                status="completed",
                success=True,
                total_duration_ms=1000,
                overall_ate=0.10,
                heterogeneity_score=0.50,
                high_responders_count=2,
                low_responders_count=1,
                recommendations_count=2,
                expected_total_lift=0.15,
            )
