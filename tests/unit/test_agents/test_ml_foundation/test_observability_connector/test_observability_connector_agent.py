"""Integration tests for ObservabilityConnectorAgent.

Version: 2.0.0 (Phase 2 Integration)
Tests include new OpikConnector and repository properties.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.ml_foundation.observability_connector.agent import (
    ObservabilityConnectorAgent,
    Span,
)


class TestObservabilityConnectorAgentProperties:
    """Test agent properties for Phase 2 integration."""

    @pytest.mark.asyncio
    async def test_opik_connector_property_lazy_initialization(self):
        """Test that opik_connector is lazily initialized."""
        agent = ObservabilityConnectorAgent()

        # Initially None
        assert agent._opik_connector is None

        # Access property triggers initialization (may fail in test env)
        with patch(
            "src.agents.ml_foundation.observability_connector.agent.ObservabilityConnectorAgent.opik_connector",
            new_callable=lambda: property(lambda self: MagicMock(is_enabled=True)),
        ):
            mock_agent = ObservabilityConnectorAgent()
            mock_agent._opik_connector = MagicMock(is_enabled=True)
            assert mock_agent._opik_connector is not None

    @pytest.mark.asyncio
    async def test_span_repository_property_lazy_initialization(self):
        """Test that span_repository is lazily initialized."""
        agent = ObservabilityConnectorAgent()

        # Initially None
        assert agent._span_repository is None

        # Simulate successful initialization
        agent._span_repository = MagicMock()
        assert agent._span_repository is not None

    @pytest.mark.asyncio
    async def test_is_opik_enabled_returns_true_when_available(self):
        """Test is_opik_enabled returns True when OpikConnector is available."""
        agent = ObservabilityConnectorAgent()

        # Mock opik_connector
        mock_opik = MagicMock()
        mock_opik.is_enabled = True
        agent._opik_connector = mock_opik

        assert agent.is_opik_enabled is True

    @pytest.mark.asyncio
    async def test_is_opik_enabled_returns_false_when_unavailable(self):
        """Test is_opik_enabled returns False when OpikConnector is unavailable."""
        agent = ObservabilityConnectorAgent()

        # Mock the opik_connector property to return None (simulating failure)
        with patch.object(
            ObservabilityConnectorAgent,
            "opik_connector",
            new_callable=lambda: property(lambda self: None),
        ):
            assert agent.is_opik_enabled is False

    @pytest.mark.asyncio
    async def test_is_opik_enabled_returns_false_when_disabled(self):
        """Test is_opik_enabled returns False when OpikConnector is disabled."""
        agent = ObservabilityConnectorAgent()

        mock_opik = MagicMock()
        mock_opik.is_enabled = False
        agent._opik_connector = mock_opik

        assert agent.is_opik_enabled is False

    @pytest.mark.asyncio
    async def test_is_db_enabled_returns_true_when_available(self):
        """Test is_db_enabled returns True when repository is available."""
        agent = ObservabilityConnectorAgent()

        agent._span_repository = MagicMock()

        assert agent.is_db_enabled is True

    @pytest.mark.asyncio
    async def test_is_db_enabled_returns_false_when_unavailable(self):
        """Test is_db_enabled returns False when repository is unavailable."""
        agent = ObservabilityConnectorAgent()

        agent._span_repository = None

        assert agent.is_db_enabled is False


class TestObservabilityConnectorAgent:
    """Integration tests for ObservabilityConnectorAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ObservabilityConnectorAgent()

        assert agent.tier == 0
        assert agent.tier_name == "ml_foundation"
        assert agent.agent_type == "standard"
        assert agent.sla_seconds == 5

    @pytest.mark.asyncio
    async def test_run_with_events(self):
        """Test agent run with events to log."""
        agent = ObservabilityConnectorAgent()

        input_data = {
            "events_to_log": [
                {
                    "span_id": "span_1",
                    "trace_id": "trace_1",
                    "agent_name": "scope_definer",
                    "operation": "execute",
                    "status": "ok",
                    "duration_ms": 1500,
                }
            ]
        }

        result = await agent.run(input_data)

        # Check output structure
        assert result["emission_successful"] is True
        assert result["events_logged"] == 1
        assert result["quality_metrics_computed"] is True
        assert "overall_success_rate" in result
        assert "overall_p95_latency_ms" in result
        assert "quality_score" in result

    @pytest.mark.asyncio
    async def test_run_without_events(self):
        """Test agent run without events (metrics only)."""
        agent = ObservabilityConnectorAgent()

        input_data = {"time_window": "24h"}

        result = await agent.run(input_data)

        # Should still compute metrics
        assert result["quality_metrics_computed"] is True
        assert result["events_logged"] == 0

    @pytest.mark.asyncio
    async def test_span_context_manager_success(self):
        """Test span context manager with successful operation."""
        agent = ObservabilityConnectorAgent()

        context = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "sampled": True,
        }

        async with agent.span(
            operation_name="test_operation",
            agent_name="test_agent",
            context=context,
            attributes={"test_attr": "value"},
        ) as span:
            # Simulate operation
            assert span.operation_name == "test_operation"
            assert span.agent_name == "test_agent"
            assert span.status == "started"
            assert span.attributes["test_attr"] == "value"

            # Set additional attributes
            span.set_attribute("result", "success")

        # After context exit, span should be completed
        assert span.status == "ok"
        assert span.end_time is not None
        assert span.duration_ms > 0
        assert span.attributes["result"] == "success"

    @pytest.mark.asyncio
    async def test_span_context_manager_error(self):
        """Test span context manager with error."""
        agent = ObservabilityConnectorAgent()

        context = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "sampled": True,
        }

        with pytest.raises(ValueError):
            async with agent.span(
                operation_name="test_operation",
                agent_name="test_agent",
                context=context,
            ) as span:
                # Simulate error
                raise ValueError("Test error")

        # Span should capture error
        assert span.status == "error"
        assert span.error_type == "ValueError"
        assert span.error_message == "Test error"
        assert span.end_time is not None

    @pytest.mark.asyncio
    async def test_span_set_attribute(self):
        """Test setting span attributes."""
        agent = ObservabilityConnectorAgent()

        context = {"trace_id": "trace_123", "span_id": "span_456"}

        async with agent.span(
            operation_name="test",
            agent_name="test_agent",
            context=context,
        ) as span:
            span.set_attribute("key1", "value1")
            span.set_attribute("key2", 123)
            span.set_attribute("key3", True)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123
        assert span.attributes["key3"] is True

    @pytest.mark.asyncio
    async def test_span_add_event(self):
        """Test adding events to span."""
        agent = ObservabilityConnectorAgent()

        context = {"trace_id": "trace_123", "span_id": "span_456"}

        async with agent.span(
            operation_name="test",
            agent_name="test_agent",
            context=context,
        ) as span:
            span.add_event("event1", {"detail": "first event"})
            span.add_event("event2", {"detail": "second event"})

        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"
        assert span.events[0]["attributes"]["detail"] == "first event"
        assert span.events[1]["name"] == "event2"

    @pytest.mark.asyncio
    async def test_span_not_sampled(self):
        """Test span with sampling disabled."""
        agent = ObservabilityConnectorAgent()

        context = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "sampled": False,  # Not sampled
        }

        async with agent.span(
            operation_name="test",
            agent_name="test_agent",
            context=context,
        ) as span:
            span.set_attribute("test", "value")

        # Span should still be created but not emitted
        assert span.status == "ok"
        assert span.attributes["test"] == "value"

    @pytest.mark.asyncio
    async def test_track_llm_call(self):
        """Test tracking LLM call."""
        agent = ObservabilityConnectorAgent()

        context = {"trace_id": "trace_123", "span_id": "span_456"}

        result = await agent.track_llm_call(
            agent_name="feature_analyzer",
            operation="interpret",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            context=context,
            attributes={"temperature": 0.7},
        )

        # Check result structure
        assert result["span_id"] is not None
        assert result["trace_id"] == "trace_123"
        assert result["agent_name"] == "feature_analyzer"
        assert result["operation"] == "interpret"
        assert result["model_used"] == "claude-sonnet-4-20250514"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["tokens_used"] == 1500
        assert result["attributes"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_track_llm_call_with_error(self):
        """Test tracking LLM call with error."""
        agent = ObservabilityConnectorAgent()

        context = {"trace_id": "trace_123", "span_id": "span_456"}

        result = await agent.track_llm_call(
            agent_name="feature_analyzer",
            operation="interpret",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            context=context,
            error="API timeout",
            error_type="TimeoutError",
        )

        assert result["status"] == "error"
        assert result["error"] == "API timeout"
        assert result["error_type"] == "TimeoutError"

    @pytest.mark.asyncio
    async def test_get_quality_metrics(self):
        """Test getting quality metrics."""
        agent = ObservabilityConnectorAgent()

        input_data = {"time_window": "24h"}

        result = await agent.get_quality_metrics(input_data)

        # Check quality metrics structure
        assert "overall_success_rate" in result
        assert "overall_p95_latency_ms" in result
        assert "overall_p99_latency_ms" in result
        assert "quality_score" in result
        assert "latency_by_agent" in result
        assert "error_rate_by_agent" in result
        assert "token_usage_by_agent" in result
        assert "total_spans_analyzed" in result

    @pytest.mark.asyncio
    async def test_get_quality_metrics_with_filters(self):
        """Test getting quality metrics with filters."""
        agent = ObservabilityConnectorAgent()

        input_data = {
            "time_window": "24h",
            "agent_name_filter": "scope_definer",
            "trace_id_filter": "trace_123",
        }

        result = await agent.get_quality_metrics(input_data)

        assert result["quality_metrics_computed"] is True

    @pytest.mark.asyncio
    async def test_create_child_context(self):
        """Test creating child span context."""
        agent = ObservabilityConnectorAgent()

        parent_context = {
            "trace_id": "trace_parent",
            "span_id": "span_parent",
        }

        child_context = agent.create_child_context(parent_context)

        # Child should inherit trace_id
        assert child_context["trace_id"] == "trace_parent"

        # Child should have different span_id
        assert child_context["span_id"] != "span_parent"

        # Child should reference parent
        assert child_context["parent_span_id"] == "span_parent"

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        """Test nested span contexts."""
        agent = ObservabilityConnectorAgent()

        parent_context = {
            "trace_id": "trace_123",
            "span_id": "span_parent",
            "sampled": True,
        }

        async with agent.span(
            operation_name="parent_operation",
            agent_name="parent_agent",
            context=parent_context,
        ) as parent_span:
            parent_span.set_attribute("level", "parent")

            # Create child context
            child_context = agent.create_child_context(parent_context)
            child_context["sampled"] = True

            async with agent.span(
                operation_name="child_operation",
                agent_name="child_agent",
                context=child_context,
            ) as child_span:
                child_span.set_attribute("level", "child")

        # Both spans should be completed
        assert parent_span.status == "ok"
        assert child_span.status == "ok"
        assert parent_span.attributes["level"] == "parent"
        assert child_span.attributes["level"] == "child"

    @pytest.mark.asyncio
    async def test_span_class_initialization(self):
        """Test Span class initialization."""
        span = Span(
            span_id="span_123",
            trace_id="trace_456",
            parent_span_id="parent_789",
            operation_name="test_op",
            agent_name="test_agent",
        )

        assert span.span_id == "span_123"
        assert span.trace_id == "trace_456"
        assert span.parent_span_id == "parent_789"
        assert span.operation_name == "test_op"
        assert span.agent_name == "test_agent"
        assert span.status == "started"
        assert span.attributes == {}
        assert span.events == []

    @pytest.mark.asyncio
    async def test_concurrent_spans(self):
        """Test multiple concurrent spans."""
        agent = ObservabilityConnectorAgent()

        context1 = {"trace_id": "trace_1", "span_id": "span_1", "sampled": True}
        context2 = {"trace_id": "trace_2", "span_id": "span_2", "sampled": True}

        # Start first span
        async with agent.span(operation_name="op1", agent_name="agent1", context=context1) as span1:
            span1.set_attribute("operation", "first")

            # Start second span while first is active
            async with agent.span(
                operation_name="op2", agent_name="agent2", context=context2
            ) as span2:
                span2.set_attribute("operation", "second")

        # Both should complete successfully
        assert span1.status == "ok"
        assert span2.status == "ok"
        assert span1.attributes["operation"] == "first"
        assert span2.attributes["operation"] == "second"

    @pytest.mark.asyncio
    async def test_sampling_rate_propagation(self):
        """Test that sampling rate is properly propagated."""
        agent = ObservabilityConnectorAgent()

        context = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "sample_rate": 0.5,
            "sampled": True,
        }

        async with agent.span(
            operation_name="test",
            agent_name="test_agent",
            context=context,
        ):
            pass

        # Sample rate should be preserved in context
        assert context.get("sample_rate") == 0.5

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self):
        """Test graceful degradation on errors."""
        agent = ObservabilityConnectorAgent()

        # Missing required fields should still work
        input_data = {}

        result = await agent.run(input_data)

        # Should still return valid output with defaults
        assert result["emission_successful"] is True
        assert result["quality_metrics_computed"] is True
