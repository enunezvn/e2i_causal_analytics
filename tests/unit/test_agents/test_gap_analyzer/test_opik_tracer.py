"""Tests for Gap Analyzer Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Gap Analyzer agent's 4-node pipeline.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.agents.gap_analyzer.opik_tracer import (
    AGENT_METADATA,
    GapAnalysisTraceContext,
    GapAnalyzerOpikTracer,
    NodeSpanContext,
    get_gap_analyzer_tracer,
    reset_gap_analyzer_tracer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    reset_gap_analyzer_tracer()
    yield
    reset_gap_analyzer_tracer()


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_agent_metadata(self):
        """Test AGENT_METADATA constant."""
        assert AGENT_METADATA["name"] == "gap_analyzer"
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
            node_name="gap_detector",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.node_name == "gap_detector"
        assert isinstance(ctx.start_time, datetime)
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to span context."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="gap_detector",
        )
        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"

    def test_log_gap_detection(self):
        """Test logging gap detection results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="gap_detector",
        )
        ctx.log_gap_detection(
            gaps_detected=5,
            segments_analyzed=10,
            gap_types=["market_share", "trx"],
            avg_gap_percentage=15.5,
        )
        assert ctx.metadata["gaps_detected"] == 5
        assert ctx.metadata["segments_analyzed"] == 10
        assert ctx.metadata["gap_types"] == ["market_share", "trx"]
        assert ctx.metadata["avg_gap_percentage"] == 15.5

    def test_log_gap_detection_defaults(self):
        """Test logging gap detection with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="gap_detector",
        )
        ctx.log_gap_detection(gaps_detected=3, segments_analyzed=8)
        assert ctx.metadata["gaps_detected"] == 3
        assert ctx.metadata["segments_analyzed"] == 8
        assert ctx.metadata["gap_types"] == []
        assert ctx.metadata["avg_gap_percentage"] == 0.0

    def test_log_roi_calculation(self):
        """Test logging ROI calculation results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="roi_calculator",
        )
        ctx.log_roi_calculation(
            opportunities_analyzed=5,
            total_addressable_value=150000.0,
            avg_roi=2.5,
            roi_confidence=0.85,
        )
        assert ctx.metadata["opportunities_analyzed"] == 5
        assert ctx.metadata["total_addressable_value"] == 150000.0
        assert ctx.metadata["avg_roi"] == 2.5
        assert ctx.metadata["roi_confidence"] == 0.85

    def test_log_roi_calculation_defaults(self):
        """Test logging ROI calculation with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="roi_calculator",
        )
        ctx.log_roi_calculation(
            opportunities_analyzed=3,
            total_addressable_value=100000.0,
        )
        assert ctx.metadata["opportunities_analyzed"] == 3
        assert ctx.metadata["total_addressable_value"] == 100000.0
        assert ctx.metadata["avg_roi"] == 0.0
        assert ctx.metadata["roi_confidence"] == 0.0

    def test_log_prioritization(self):
        """Test logging prioritization results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="prioritizer",
        )
        ctx.log_prioritization(
            total_opportunities=10,
            quick_wins=3,
            strategic_bets=2,
            top_priority_value=50000.0,
        )
        assert ctx.metadata["total_opportunities"] == 10
        assert ctx.metadata["quick_wins"] == 3
        assert ctx.metadata["strategic_bets"] == 2
        assert ctx.metadata["top_priority_value"] == 50000.0

    def test_log_prioritization_defaults(self):
        """Test logging prioritization with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="prioritizer",
        )
        ctx.log_prioritization(
            total_opportunities=5,
            quick_wins=2,
            strategic_bets=1,
        )
        assert ctx.metadata["total_opportunities"] == 5
        assert ctx.metadata["quick_wins"] == 2
        assert ctx.metadata["strategic_bets"] == 1
        assert ctx.metadata["top_priority_value"] == 0.0

    def test_log_formatting(self):
        """Test logging formatting results."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="formatter",
        )
        ctx.log_formatting(
            summary_length=500,
            insights_count=4,
            recommendations_count=3,
        )
        assert ctx.metadata["summary_length"] == 500
        assert ctx.metadata["insights_count"] == 4
        assert ctx.metadata["recommendations_count"] == 3

    def test_log_formatting_defaults(self):
        """Test logging formatting with default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="formatter",
        )
        ctx.log_formatting(summary_length=300, insights_count=2)
        assert ctx.metadata["summary_length"] == 300
        assert ctx.metadata["insights_count"] == 2
        assert ctx.metadata["recommendations_count"] == 0

    def test_set_output(self):
        """Test setting output on span context."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="formatter",
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
            node_name="formatter",
        )
        # Should not raise
        ctx.set_output({"result": "test"})


# ============================================================================
# GAP ANALYSIS TRACE CONTEXT TESTS
# ============================================================================


class TestGapAnalysisTraceContext:
    """Tests for GapAnalysisTraceContext class."""

    def test_create_trace_context(self):
        """Test GapAnalysisTraceContext creation."""
        ctx = GapAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="What are the gaps for Kisqali?",
            brand="Kisqali",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.query == "What are the gaps for Kisqali?"
        assert ctx.brand == "Kisqali"
        assert isinstance(ctx.start_time, datetime)
        assert ctx.node_spans == {}
        assert ctx.node_durations == {}
        assert ctx.metadata == {}

    def test_log_analysis_complete(self):
        """Test logging analysis complete."""
        ctx = GapAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="What are the gaps?",
            brand="Kisqali",
        )
        ctx.log_analysis_complete(
            status="completed",
            success=True,
            total_duration_ms=2000,
            gaps_detected=5,
            opportunities_count=8,
            quick_wins_count=3,
            strategic_bets_count=2,
            total_addressable_value=150000.0,
            confidence=0.85,
            errors=[],
            suggested_next_agent="resource_optimizer",
        )
        # Log method doesn't store in metadata directly, it updates trace

    def test_log_analysis_complete_with_errors(self):
        """Test logging analysis complete with errors."""
        ctx = GapAnalysisTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="What are the gaps?",
            brand="Kisqali",
        )
        ctx.log_analysis_complete(
            status="partial",
            success=False,
            total_duration_ms=1500,
            gaps_detected=2,
            opportunities_count=3,
            errors=["Data unavailable for some segments"],
        )
        # Should not raise


# ============================================================================
# GAP ANALYZER OPIK TRACER TESTS
# ============================================================================


class TestGapAnalyzerOpikTracer:
    """Tests for GapAnalyzerOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = GapAnalyzerOpikTracer()
        assert tracer.project_name == "e2i-gap-analyzer"
        assert tracer.enabled is True
        assert tracer.sample_rate == 1.0
        assert tracer._opik_connector is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = GapAnalyzerOpikTracer(
            project_name="custom-project",
            enabled=False,
            sample_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sample_rate == 0.5

    def test_singleton_pattern(self):
        """Test singleton pattern via get_gap_analyzer_tracer."""
        tracer1 = get_gap_analyzer_tracer()
        tracer2 = get_gap_analyzer_tracer()
        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test that singleton skips reinitialization."""
        tracer1 = get_gap_analyzer_tracer(project_name="first")
        tracer2 = get_gap_analyzer_tracer(project_name="second")
        assert tracer1 is tracer2
        assert tracer1.project_name == "first"

    def test_reset_singleton(self):
        """Test resetting the singleton."""
        tracer1 = get_gap_analyzer_tracer()
        reset_gap_analyzer_tracer()
        tracer2 = get_gap_analyzer_tracer()
        assert tracer1 is not tracer2

    def test_should_trace_full_rate(self):
        """Test _should_trace at 100% rate."""
        tracer = GapAnalyzerOpikTracer(sample_rate=1.0)
        assert tracer._should_trace() is True

    def test_should_trace_zero_rate(self):
        """Test _should_trace at 0% rate."""
        tracer = GapAnalyzerOpikTracer(sample_rate=0.0)
        assert tracer._should_trace() is False

    @pytest.mark.asyncio
    async def test_trace_analysis_disabled(self):
        """Test trace_analysis when disabled."""
        tracer = GapAnalyzerOpikTracer(enabled=False)
        async with tracer.trace_analysis(
            query="Test query",
            brand="Kisqali",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace_id is not None
            assert ctx.span_id is not None

    @pytest.mark.asyncio
    async def test_trace_analysis_not_sampled(self):
        """Test trace_analysis when not sampled."""
        tracer = GapAnalyzerOpikTracer(sample_rate=0.0)
        async with tracer.trace_analysis(
            query="Test query",
            brand="Kisqali",
        ) as ctx:
            assert ctx is not None

    @pytest.mark.asyncio
    async def test_trace_analysis_with_metadata(self):
        """Test trace_analysis with additional metadata."""
        tracer = GapAnalyzerOpikTracer(enabled=False)
        async with tracer.trace_analysis(
            query="Test query",
            brand="Kisqali",
            metrics=["trx", "nrx"],
            segments=["high_value", "low_value"],
            gap_type="vs_potential",
            metadata={"custom_key": "custom_value"},
        ) as ctx:
            assert ctx is not None
            assert ctx.brand == "Kisqali"


# ============================================================================
# SINGLETON FUNCTIONS TESTS
# ============================================================================


class TestGetGapAnalyzerTracer:
    """Tests for get_gap_analyzer_tracer function."""

    def test_returns_tracer_instance(self):
        """Test that get_gap_analyzer_tracer returns a tracer."""
        tracer = get_gap_analyzer_tracer()
        assert isinstance(tracer, GapAnalyzerOpikTracer)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        tracer1 = get_gap_analyzer_tracer()
        tracer2 = get_gap_analyzer_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_gap_analyzer_tracer(
            project_name="custom-project",
            sample_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sample_rate == 0.5


class TestResetGapAnalyzerTracer:
    """Tests for reset_gap_analyzer_tracer function."""

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        tracer1 = get_gap_analyzer_tracer()
        reset_gap_analyzer_tracer()
        tracer2 = get_gap_analyzer_tracer()
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
        mock_connector.start_span.return_value.__enter__ = MagicMock(return_value=mock_span_context)
        mock_connector.start_span.return_value.__exit__ = MagicMock(return_value=None)

        tracer = GapAnalyzerOpikTracer()
        tracer._opik_connector = mock_connector
        tracer._initialized = True

        async with tracer.trace_analysis(
            query="What gaps exist for Kisqali?",
            brand="Kisqali",
        ) as trace:
            assert trace.trace_id is not None
            trace.log_analysis_complete(
                status="completed",
                success=True,
                total_duration_ms=2000,
                gaps_detected=3,
                opportunities_count=5,
            )

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_trace_error(self):
        """Test that errors during tracing don't break execution."""
        tracer = GapAnalyzerOpikTracer()

        # Mock connector that raises on start_span
        mock_connector = MagicMock()
        mock_connector.start_span.side_effect = Exception("Opik error")
        tracer._opik_connector = mock_connector
        tracer._initialized = True

        # Should not raise
        async with tracer.trace_analysis(
            query="Test query",
            brand="Kisqali",
        ) as trace:
            assert trace is not None
            trace.log_analysis_complete(
                status="completed",
                success=True,
                total_duration_ms=1000,
            )
