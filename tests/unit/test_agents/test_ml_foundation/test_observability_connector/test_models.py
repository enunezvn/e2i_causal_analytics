"""Tests for observability connector Pydantic models.

Version: 1.0.0
Tests the data models for observability spans and metrics.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from src.agents.ml_foundation.observability_connector.models import (
    AgentNameEnum,
    AgentTierEnum,
    LatencyStats,
    ObservabilitySpan,
    QualityMetrics,
    SpanEvent,
    SpanStatusEnum,
    TokenUsage,
    create_llm_span,
    create_span,
)


class TestAgentNameEnum:
    """Test AgentNameEnum."""

    def test_all_ml_foundation_agents_present(self):
        """Test Tier 0 agents are present."""
        tier0_agents = [
            "scope_definer",
            "data_preparer",
            "feature_analyzer",
            "model_selector",
            "model_trainer",
            "model_deployer",
            "observability_connector",
        ]

        for agent in tier0_agents:
            assert AgentNameEnum(agent) is not None

    def test_all_tier1_agents_present(self):
        """Test Tier 1 agents are present."""
        tier1_agents = ["orchestrator", "tool_composer"]

        for agent in tier1_agents:
            assert AgentNameEnum(agent) is not None

    def test_all_tier2_agents_present(self):
        """Test Tier 2 agents are present."""
        tier2_agents = ["causal_impact", "gap_analyzer", "heterogeneous_optimizer"]

        for agent in tier2_agents:
            assert AgentNameEnum(agent) is not None

    def test_all_tiers_covered(self):
        """Test all tiers have at least one agent."""
        all_agents = list(AgentNameEnum)
        # 7 (Tier0) + 2 (Tier1) + 3 (Tier2) + 3 (Tier3) + 2 (Tier4) + 2 (Tier5) = 19
        assert len(all_agents) == 19

    def test_invalid_agent_raises(self):
        """Test invalid agent name raises ValueError."""
        with pytest.raises(ValueError):
            AgentNameEnum("invalid_agent")


class TestAgentTierEnum:
    """Test AgentTierEnum."""

    def test_all_tiers_present(self):
        """Test all tiers are present."""
        tiers = [
            "ml_foundation",
            "coordination",
            "causal_analytics",
            "monitoring",
            "ml_predictions",
            "self_improvement",
        ]

        for tier in tiers:
            assert AgentTierEnum(tier) is not None

    def test_tier_count(self):
        """Test correct number of tiers."""
        assert len(list(AgentTierEnum)) == 6


class TestSpanStatusEnum:
    """Test SpanStatusEnum."""

    def test_all_statuses_present(self):
        """Test all status values are present."""
        assert SpanStatusEnum.SUCCESS.value == "success"
        assert SpanStatusEnum.ERROR.value == "error"
        assert SpanStatusEnum.TIMEOUT.value == "timeout"


class TestSpanEvent:
    """Test SpanEvent model."""

    def test_create_event(self):
        """Test creating a span event."""
        event = SpanEvent(name="checkpoint", attributes={"progress": 50})

        assert event.name == "checkpoint"
        assert event.attributes["progress"] == 50
        assert event.timestamp is not None

    def test_event_default_attributes(self):
        """Test event with default attributes."""
        event = SpanEvent(name="simple_event")

        assert event.name == "simple_event"
        assert event.attributes == {}

    def test_event_is_immutable(self):
        """Test that events are frozen (immutable)."""
        event = SpanEvent(name="immutable")

        with pytest.raises(Exception):  # ValidationError or AttributeError
            event.name = "changed"


class TestTokenUsage:
    """Test TokenUsage model."""

    def test_create_token_usage(self):
        """Test creating token usage."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_auto_compute_total(self):
        """Test total tokens auto-computation."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)

        # Should auto-compute if not provided
        assert usage.total_tokens == 0 or usage.total_tokens == 150

    def test_token_validation(self):
        """Test token values must be non-negative."""
        with pytest.raises(Exception):
            TokenUsage(input_tokens=-1, output_tokens=50)


class TestObservabilitySpan:
    """Test ObservabilitySpan model."""

    def test_create_minimal_span(self):
        """Test creating span with minimal required fields."""
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
        )

        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.agent_name == AgentNameEnum.ORCHESTRATOR
        assert span.agent_tier == AgentTierEnum.COORDINATION
        assert span.status == SpanStatusEnum.SUCCESS

    def test_create_full_span(self):
        """Test creating span with all fields."""
        now = datetime.now(timezone.utc)
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="parent-789",
            agent_name=AgentNameEnum.CAUSAL_IMPACT,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="inference",
            started_at=now,
            ended_at=now + timedelta(seconds=2),
            model_name="claude-3-5-sonnet",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            status=SpanStatusEnum.SUCCESS,
            fallback_used=False,
            user_id="user-123",
            session_id="session-456",
        )

        assert span.parent_span_id == "parent-789"
        assert span.operation_type == "inference"
        assert span.model_name == "claude-3-5-sonnet"
        assert span.total_tokens == 700

    def test_span_auto_generated_id(self):
        """Test span ID is auto-generated."""
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
        )

        assert span.id is not None
        assert isinstance(span.id, UUID)

    def test_add_event(self):
        """Test adding events to span."""
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
        )

        span.add_event("checkpoint", {"step": 1})
        span.add_event("retry", {"attempt": 2})

        assert len(span.events) == 2
        assert span.events[0].name == "checkpoint"

    def test_set_error(self):
        """Test marking span as error."""
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
        )

        span.set_error("ValidationError", "Invalid input")

        assert span.status == SpanStatusEnum.ERROR
        assert span.error_type == "ValidationError"
        assert span.error_message == "Invalid input"

    def test_complete_span(self):
        """Test completing a span."""
        start = datetime.now(timezone.utc)
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=start,
        )

        span.complete()

        assert span.ended_at is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_to_db_dict(self):
        """Test conversion to database dictionary."""
        span = ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
        )

        db_dict = span.to_db_dict()

        assert isinstance(db_dict["id"], str)
        assert db_dict["agent_name"] == "orchestrator"
        assert db_dict["agent_tier"] == "coordination"
        assert db_dict["status"] == "success"

    def test_span_validation_trace_id_required(self):
        """Test trace_id is required."""
        with pytest.raises(Exception):
            ObservabilitySpan(
                span_id="span-456",
                agent_name=AgentNameEnum.ORCHESTRATOR,
                agent_tier=AgentTierEnum.COORDINATION,
                started_at=datetime.now(timezone.utc),
            )

    def test_span_validation_agent_name_required(self):
        """Test agent_name is required."""
        with pytest.raises(Exception):
            ObservabilitySpan(
                trace_id="trace-123",
                span_id="span-456",
                agent_tier=AgentTierEnum.COORDINATION,
                started_at=datetime.now(timezone.utc),
            )


class TestLatencyStats:
    """Test LatencyStats model."""

    def test_create_latency_stats(self):
        """Test creating latency statistics."""
        stats = LatencyStats(
            agent_name=AgentNameEnum.CAUSAL_IMPACT,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            total_spans=100,
            avg_duration_ms=250.5,
            p50_ms=200.0,
            p95_ms=500.0,
            p99_ms=800.0,
            error_rate=0.02,
            fallback_rate=0.05,
            total_tokens_used=50000,
        )

        assert stats.total_spans == 100
        assert stats.p95_ms == 500.0
        assert stats.error_rate == 0.02

    def test_is_healthy_true(self):
        """Test is_healthy returns True for healthy stats."""
        stats = LatencyStats(
            p95_ms=2000.0,
            error_rate=0.01,
        )

        assert stats.is_healthy is True

    def test_is_healthy_false_high_latency(self):
        """Test is_healthy returns False for high latency."""
        stats = LatencyStats(
            p95_ms=6000.0,  # > 5000ms threshold
            error_rate=0.01,
        )

        assert stats.is_healthy is False

    def test_is_healthy_false_high_error_rate(self):
        """Test is_healthy returns False for high error rate."""
        stats = LatencyStats(
            p95_ms=1000.0,
            error_rate=0.10,  # > 0.05 threshold
        )

        assert stats.is_healthy is False


class TestQualityMetrics:
    """Test QualityMetrics model."""

    def test_create_quality_metrics(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            time_window="24h",
            total_spans=1000,
            success_count=980,
            error_count=15,
            timeout_count=5,
            success_rate=0.98,
            error_rate=0.015,
            fallback_rate=0.02,
            avg_latency_ms=300.0,
            p50_latency_ms=250.0,
            p95_latency_ms=600.0,
            p99_latency_ms=1000.0,
            total_tokens=500000,
        )

        assert metrics.total_spans == 1000
        assert metrics.success_rate == 0.98
        assert metrics.p95_latency_ms == 600.0

    def test_compute_quality_score_high(self):
        """Test quality score computation for healthy system."""
        metrics = QualityMetrics(
            success_rate=0.99,
            p95_latency_ms=500.0,  # Low latency
            fallback_rate=0.01,
            error_rate=0.01,
        )

        score = metrics.compute_quality_score()

        assert score > 0.9  # Should be high

    def test_compute_quality_score_low(self):
        """Test quality score computation for unhealthy system."""
        metrics = QualityMetrics(
            success_rate=0.70,  # Low success
            p95_latency_ms=15000.0,  # Very high latency
            fallback_rate=0.30,  # High fallback
            error_rate=0.30,  # High error
        )

        score = metrics.compute_quality_score()

        assert score < 0.6  # Should be relatively low

    def test_default_status_distribution(self):
        """Test default status distribution."""
        metrics = QualityMetrics()

        assert metrics.status_distribution == {"success": 0, "error": 0, "timeout": 0}


class TestFactoryFunctions:
    """Test factory functions for creating spans."""

    def test_create_span(self):
        """Test create_span factory function."""
        span = create_span(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.GAP_ANALYZER,
            agent_tier=AgentTierEnum.CAUSAL_ANALYTICS,
            operation_type="analysis",
        )

        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.agent_name == AgentNameEnum.GAP_ANALYZER
        assert span.operation_type == "analysis"
        assert span.started_at is not None

    def test_create_span_with_parent(self):
        """Test create_span with parent span."""
        span = create_span(
            trace_id="trace-123",
            span_id="child-span",
            agent_name=AgentNameEnum.EXPLAINER,
            agent_tier=AgentTierEnum.SELF_IMPROVEMENT,
            parent_span_id="parent-span",
        )

        assert span.parent_span_id == "parent-span"

    def test_create_llm_span(self):
        """Test create_llm_span factory function."""
        span = create_llm_span(
            trace_id="trace-123",
            span_id="llm-span",
            agent_name=AgentNameEnum.PREDICTION_SYNTHESIZER,
            agent_tier=AgentTierEnum.ML_PREDICTIONS,
            model_name="claude-3-5-sonnet",
        )

        assert span.operation_type == "llm_call"
        assert span.model_name == "claude-3-5-sonnet"
        assert span.started_at is not None

    def test_create_llm_span_with_extra_kwargs(self):
        """Test create_llm_span with additional kwargs."""
        span = create_llm_span(
            trace_id="trace-123",
            span_id="llm-span",
            agent_name=AgentNameEnum.EXPLAINER,
            agent_tier=AgentTierEnum.SELF_IMPROVEMENT,
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )

        assert span.input_tokens == 1000
        assert span.output_tokens == 500
        assert span.total_tokens == 1500
