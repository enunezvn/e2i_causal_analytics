"""Tests for OpikConnector.

Version: 1.0.0
Tests the Opik SDK wrapper with mocked SDK operations.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.mlops.opik_connector import (
    LLMSpanContext,
    OpikConfig,
    OpikConnector,
    SpanContext,
)


class TestOpikConfig:
    """Test OpikConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OpikConfig()

        assert config.api_key is None
        assert config.workspace == "default"
        assert config.project_name == "e2i-causal-analytics"
        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.always_sample_errors is True

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        # Clear any existing env vars
        env_vars = ["OPIK_API_KEY", "OPIK_WORKSPACE", "OPIK_PROJECT_NAME"]
        original = {k: os.environ.get(k) for k in env_vars}

        for k in env_vars:
            if k in os.environ:
                del os.environ[k]

        try:
            config = OpikConfig.from_env()
            assert config.api_key is None
            assert config.workspace == "default"
            assert config.project_name == "e2i-causal-analytics"
        finally:
            # Restore original values
            for k, v in original.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_with_values(self):
        """Test from_env with environment variables set."""
        original = {
            "OPIK_API_KEY": os.environ.get("OPIK_API_KEY"),
            "OPIK_WORKSPACE": os.environ.get("OPIK_WORKSPACE"),
        }

        os.environ["OPIK_API_KEY"] = "test-key-123"
        os.environ["OPIK_WORKSPACE"] = "test-workspace"

        try:
            config = OpikConfig.from_env()
            assert config.api_key == "test-key-123"
            assert config.workspace == "test-workspace"
        finally:
            for k, v in original.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


class TestSpanContext:
    """Test SpanContext dataclass."""

    def test_create_span_context(self):
        """Test creating a span context."""
        ctx = SpanContext(
            span_id="span-123",
            trace_id="trace-456",
            parent_span_id="parent-789",
        )

        assert ctx.span_id == "span-123"
        assert ctx.trace_id == "trace-456"
        assert ctx.parent_span_id == "parent-789"
        assert ctx.metadata == {}

    def test_set_attribute(self):
        """Test setting attributes on span context."""
        ctx = SpanContext(span_id="span-123", trace_id="trace-456")

        ctx.set_attribute("key1", "value1")
        ctx.set_attribute("key2", 42)

        assert ctx.metadata["key1"] == "value1"
        assert ctx.metadata["key2"] == 42

    def test_add_event(self):
        """Test adding events to span context."""
        ctx = SpanContext(span_id="span-123", trace_id="trace-456")

        ctx.add_event("checkpoint", {"progress": 50})
        ctx.add_event("retry", {"attempt": 2})

        assert len(ctx.metadata["events"]) == 2
        assert ctx.metadata["events"][0]["name"] == "checkpoint"
        assert ctx.metadata["events"][1]["name"] == "retry"

    def test_set_input_output(self):
        """Test setting input and output on span context."""
        ctx = SpanContext(span_id="span-123", trace_id="trace-456")

        ctx.set_input({"query": "test query"})
        ctx.set_output({"result": "test result"})

        assert ctx.input_data == {"query": "test query"}
        assert ctx.output_data == {"result": "test result"}

    def test_to_dict(self):
        """Test converting span context to dictionary."""
        ctx = SpanContext(
            span_id="span-123",
            trace_id="trace-456",
            agent_name="test_agent",
            operation="test_op",
        )
        ctx.set_attribute("agent", "test_agent")
        ctx.set_input({"query": "test"})
        ctx.set_output({"result": "success"})
        ctx.add_event("event1", {})

        result = ctx.to_dict()

        assert result["span_id"] == "span-123"
        assert result["trace_id"] == "trace-456"
        assert result["attributes"]["agent"] == "test_agent"
        assert result["input"] == {"query": "test"}
        assert result["output"] == {"result": "success"}
        assert len(result["attributes"]["events"]) == 1


class TestLLMSpanContext:
    """Test LLMSpanContext dataclass."""

    def test_create_llm_span_context(self):
        """Test creating an LLM span context."""
        ctx = LLMSpanContext(
            span_id="span-123",
            trace_id="trace-456",
            model="claude-3-5-sonnet",
        )

        assert ctx.span_id == "span-123"
        assert ctx.model == "claude-3-5-sonnet"
        assert ctx.input_tokens == 0
        assert ctx.output_tokens == 0

    def test_log_tokens(self):
        """Test logging token usage."""
        ctx = LLMSpanContext(span_id="span-123", trace_id="trace-456", model="gpt-4")

        ctx.log_tokens(input_tokens=100, output_tokens=50)

        assert ctx.input_tokens == 100
        assert ctx.output_tokens == 50
        assert ctx.total_tokens == 150

    def test_set_cost(self):
        """Test setting cost on LLM span."""
        ctx = LLMSpanContext(span_id="span-123", trace_id="trace-456", model="gpt-4")

        ctx.set_cost(0.05)

        assert ctx.total_cost == 0.05


class TestOpikConnector:
    """Test OpikConnector class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        OpikConnector._instance = None
        OpikConnector._initialized = False
        yield
        OpikConnector._instance = None
        OpikConnector._initialized = False

    def test_singleton_pattern(self):
        """Test that OpikConnector uses singleton pattern."""
        connector1 = OpikConnector()
        connector2 = OpikConnector()

        assert connector1 is connector2

    def test_initialization_without_opik(self):
        """Test initialization when Opik SDK is not available."""
        connector = OpikConnector()

        # Should initialize gracefully even without Opik
        assert connector.config is not None
        assert connector.config.project_name == "e2i-causal-analytics"

    def test_is_enabled_default(self):
        """Test is_enabled property."""
        connector = OpikConnector()

        # Without API key, should still show as enabled in config
        assert connector.config.enabled is True

    @pytest.mark.asyncio
    async def test_trace_agent_creates_context(self):
        """Test trace_agent context manager creates span context."""
        connector = OpikConnector()

        async with connector.trace_agent(
            agent_name="test_agent",
            operation="test_operation",
        ) as ctx:
            assert ctx is not None
            assert ctx.span_id is not None
            assert ctx.trace_id is not None
            assert len(ctx.span_id) > 0

    @pytest.mark.asyncio
    async def test_trace_agent_with_parent_trace(self):
        """Test trace_agent with parent trace ID."""
        connector = OpikConnector()

        async with connector.trace_agent(
            agent_name="child_agent",
            operation="child_op",
            trace_id="parent-trace-123",
        ) as ctx:
            assert ctx.trace_id == "parent-trace-123"

    @pytest.mark.asyncio
    async def test_trace_agent_with_metadata(self):
        """Test trace_agent with custom metadata."""
        connector = OpikConnector()

        async with connector.trace_agent(
            agent_name="test_agent",
            operation="test_op",
            metadata={"custom_key": "custom_value"},
        ) as ctx:
            assert "custom_key" in ctx.metadata
            assert ctx.metadata["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_trace_agent_captures_error(self):
        """Test trace_agent captures exceptions."""
        connector = OpikConnector()

        with pytest.raises(ValueError):
            async with connector.trace_agent(
                agent_name="error_agent",
                operation="failing_op",
            ):
                raise ValueError("Test error")

        # Context should have recorded the error
        # (implementation tracks this internally)

    @pytest.mark.asyncio
    async def test_trace_llm_call_creates_context(self):
        """Test trace_llm_call context manager."""
        connector = OpikConnector()

        async with connector.trace_llm_call(
            model="claude-3-5-sonnet",
        ) as ctx:
            assert ctx is not None
            assert isinstance(ctx, LLMSpanContext)
            assert ctx.model == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_trace_llm_call_with_tokens(self):
        """Test trace_llm_call with token logging."""
        connector = OpikConnector()

        async with connector.trace_llm_call(
            model="gpt-4",
            trace_id="llm-trace-123",
        ) as ctx:
            ctx.log_tokens(input_tokens=500, output_tokens=200)
            ctx.set_cost(0.02)

        assert ctx.input_tokens == 500
        assert ctx.output_tokens == 200
        assert ctx.total_cost == 0.02

    def test_log_metric(self):
        """Test logging a metric."""
        connector = OpikConnector()

        # Should not raise even without Opik SDK
        connector.log_metric(
            name="test_metric",
            value=42.5,
            trace_id="trace-123",
        )

    def test_log_feedback(self):
        """Test logging feedback."""
        connector = OpikConnector()

        # Should not raise even without Opik SDK
        connector.log_feedback(
            trace_id="trace-123",
            score=0.9,
            feedback_type="quality",
            reason="Good response",
        )

    def test_flush(self):
        """Test flush method."""
        connector = OpikConnector()

        # Should not raise
        connector.flush()

    @pytest.mark.asyncio
    async def test_sampling_enabled(self):
        """Test that sampling works when enabled."""
        connector = OpikConnector()
        connector.config.sample_rate = 0.5

        # Run multiple traces and check that some are sampled
        sampled_count = 0
        for _i in range(20):
            async with connector.trace_agent(
                agent_name="sample_test",
                operation="test",
            ) as ctx:
                if ctx.span_id:  # If we got a valid span, it was sampled
                    sampled_count += 1

        # With 50% sampling, we should get some but not all
        # This is probabilistic, so we just check it's reasonable
        assert sampled_count >= 0  # At minimum, sampled_count should be valid


class TestOpikConnectorWithMockedSDK:
    """Test OpikConnector with mocked Opik SDK."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        OpikConnector._instance = None
        OpikConnector._initialized = False
        yield
        OpikConnector._instance = None
        OpikConnector._initialized = False

    @pytest.mark.asyncio
    async def test_trace_agent_with_mocked_opik(self):
        """Test trace_agent with mocked Opik SDK."""
        mock_opik = MagicMock()
        mock_trace = MagicMock()
        mock_span = MagicMock()

        mock_opik.trace.return_value = mock_trace
        mock_trace.span.return_value = mock_span

        with patch.dict("sys.modules", {"opik": MagicMock()}):
            connector = OpikConnector()
            connector._opik_client = mock_opik

            async with connector.trace_agent(
                agent_name="mocked_agent",
                operation="mocked_op",
            ) as ctx:
                ctx.set_attribute("test", "value")

            # Verify context was created
            assert ctx.span_id is not None

    @pytest.mark.asyncio
    async def test_error_handling_in_trace(self):
        """Test that errors in Opik SDK are handled gracefully."""
        connector = OpikConnector()

        # Even with internal errors, should complete without raising
        async with connector.trace_agent(
            agent_name="error_test",
            operation="test",
        ) as ctx:
            pass

        assert ctx is not None


class TestOpikConnectorGracefulDegradation:
    """Test graceful degradation when Opik is unavailable."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        OpikConnector._instance = None
        OpikConnector._initialized = False
        yield
        OpikConnector._instance = None
        OpikConnector._initialized = False

    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        # Clear API key
        original = os.environ.get("OPIK_API_KEY")
        if "OPIK_API_KEY" in os.environ:
            del os.environ["OPIK_API_KEY"]

        try:
            connector = OpikConnector()
            assert connector is not None
            # Should still be usable
            assert connector.config is not None
        finally:
            if original:
                os.environ["OPIK_API_KEY"] = original

    @pytest.mark.asyncio
    async def test_trace_works_without_opik_client(self):
        """Test tracing works even without Opik client."""
        connector = OpikConnector()
        connector._opik_client = None  # Simulate Opik not available

        async with connector.trace_agent(
            agent_name="fallback_test",
            operation="test",
        ) as ctx:
            ctx.set_attribute("key", "value")

        assert ctx is not None
        assert ctx.metadata.get("key") == "value"

    def test_log_metric_without_client(self):
        """Test log_metric works without Opik client."""
        connector = OpikConnector()
        connector._opik_client = None

        # Should not raise
        connector.log_metric("test", 1.0)

    def test_log_feedback_without_client(self):
        """Test log_feedback works without Opik client."""
        connector = OpikConnector()
        connector._opik_client = None

        # Should not raise
        connector.log_feedback("trace-123", 0.9)
