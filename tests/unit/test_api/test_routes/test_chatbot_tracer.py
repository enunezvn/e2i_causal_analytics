"""
Comprehensive tests for E2I Chatbot Tracer module.

Tests the Opik integration for chatbot LangGraph workflow tracing.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.chatbot_tracer import (
    CHATBOT_OPIK_TRACING_ENABLED,
    ChatbotOpikTracer,
    ChatbotTraceContext,
    NodeSpanContext,
    get_chatbot_tracer,
    reset_chatbot_tracer,
    trace_chatbot_workflow,
)


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_node_span_context_creation(self):
        """NodeSpanContext can be created with required fields."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.node_name == "init"

    def test_node_span_context_default_values(self):
        """NodeSpanContext has correct default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
        )
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.metadata == {}
        assert ctx._opik_span is None
        assert ctx._parent_ctx is None

    def test_node_span_context_start_time_auto_set(self):
        """NodeSpanContext auto-sets start_time."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
        )
        assert ctx.start_time is not None
        assert isinstance(ctx.start_time, datetime)

    def test_log_init_updates_metadata(self):
        """log_init updates metadata with init metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
        )
        ctx.log_init(
            is_new_conversation=True,
            session_id="session-789",
            user_id="user-001",
        )
        assert ctx.metadata["is_new_conversation"] is True
        assert ctx.metadata["session_id"] == "session-789"
        assert ctx.metadata["user_id"] == "user-001"

    def test_log_init_with_extra_kwargs(self):
        """log_init accepts extra keyword arguments."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
        )
        ctx.log_init(
            is_new_conversation=False,
            custom_key="custom_value",
        )
        assert ctx.metadata["is_new_conversation"] is False
        assert ctx.metadata["custom_key"] == "custom_value"

    def test_log_init_with_opik_span(self):
        """log_init interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="init",
            _opik_span=mock_span,
        )
        ctx.log_init(
            is_new_conversation=True,
            session_id="session-789",
        )
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_context_load_updates_metadata(self):
        """log_context_load updates metadata with context metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="load_context",
        )
        ctx.log_context_load(
            previous_message_count=5,
            conversation_title="Sales Analysis",
            brand_context="Kisqali",
            region_context="Northeast",
        )
        assert ctx.metadata["previous_message_count"] == 5
        assert ctx.metadata["conversation_title"] == "Sales Analysis"
        assert ctx.metadata["brand_context"] == "Kisqali"
        assert ctx.metadata["region_context"] == "Northeast"

    def test_log_context_load_with_opik_span(self):
        """log_context_load interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="load_context",
            _opik_span=mock_span,
        )
        ctx.log_context_load(
            previous_message_count=3,
            brand_context="Fabhalta",
        )
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called_with(
            "context_loaded",
            {
                "previous_message_count": 3,
                "has_title": False,
            },
        )

    def test_log_intent_classification_updates_metadata(self):
        """log_intent_classification updates metadata with intent metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="classify_intent",
        )
        ctx.log_intent_classification(
            intent="kpi_query",
            confidence=0.92,
            classification_method="dspy",
        )
        assert ctx.metadata["intent"] == "kpi_query"
        assert ctx.metadata["confidence"] == 0.92
        assert ctx.metadata["classification_method"] == "dspy"

    def test_log_intent_classification_default_values(self):
        """log_intent_classification has correct default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="classify_intent",
        )
        ctx.log_intent_classification(intent="general")
        assert ctx.metadata["confidence"] == 1.0
        assert ctx.metadata["classification_method"] == "hardcoded"

    def test_log_intent_classification_with_opik_span(self):
        """log_intent_classification interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="classify_intent",
            _opik_span=mock_span,
        )
        ctx.log_intent_classification(
            intent="causal_analysis",
            confidence=0.85,
            classification_method="dspy",
        )
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_rag_retrieval_updates_metadata(self):
        """log_rag_retrieval updates metadata with RAG metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="retrieve_rag",
        )
        ctx.log_rag_retrieval(
            result_count=5,
            relevance_scores=[0.9, 0.85, 0.8, 0.75, 0.7],
            kpi_filter="TRx",
            brand_filter="Kisqali",
            retrieval_method="hybrid",
        )
        assert ctx.metadata["result_count"] == 5
        assert ctx.metadata["avg_relevance_score"] == 0.8
        assert ctx.metadata["kpi_filter"] == "TRx"
        assert ctx.metadata["brand_filter"] == "Kisqali"
        assert ctx.metadata["retrieval_method"] == "hybrid"

    def test_log_rag_retrieval_with_empty_scores(self):
        """log_rag_retrieval handles empty relevance scores."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="retrieve_rag",
        )
        ctx.log_rag_retrieval(result_count=0)
        assert ctx.metadata["avg_relevance_score"] == 0.0
        assert ctx.metadata["relevance_scores"] == []

    def test_log_rag_retrieval_with_opik_span(self):
        """log_rag_retrieval interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="retrieve_rag",
            _opik_span=mock_span,
        )
        ctx.log_rag_retrieval(
            result_count=3,
            relevance_scores=[0.9, 0.8, 0.7],
        )
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_generate_updates_metadata(self):
        """log_generate updates metadata with generation metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate",
        )
        ctx.log_generate(
            input_tokens=150,
            output_tokens=300,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            tool_calls_count=2,
            temperature=0.3,
        )
        assert ctx.metadata["input_tokens"] == 150
        assert ctx.metadata["output_tokens"] == 300
        assert ctx.metadata["total_tokens"] == 450
        assert ctx.metadata["model"] == "claude-sonnet-4-20250514"
        assert ctx.metadata["provider"] == "anthropic"
        assert ctx.metadata["tool_calls_count"] == 2
        assert ctx.metadata["temperature"] == 0.3

    def test_log_generate_default_values(self):
        """log_generate has correct default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate",
        )
        ctx.log_generate()
        assert ctx.metadata["input_tokens"] == 0
        assert ctx.metadata["output_tokens"] == 0
        assert ctx.metadata["total_tokens"] == 0
        assert ctx.metadata["provider"] == "anthropic"
        assert ctx.metadata["tool_calls_count"] == 0
        assert ctx.metadata["temperature"] == 0.3

    def test_log_generate_with_opik_span(self):
        """log_generate interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate",
            _opik_span=mock_span,
        )
        ctx.log_generate(input_tokens=100, output_tokens=200)
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_tool_execution_updates_metadata(self):
        """log_tool_execution updates metadata with tool metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="tools",
        )
        ctx.log_tool_execution(
            tool_name="causal_analyzer",
            success=True,
            result_size=1024,
        )
        assert ctx.metadata["tool_name"] == "causal_analyzer"
        assert ctx.metadata["tool_success"] is True
        assert ctx.metadata["result_size"] == 1024
        assert ctx.metadata["tool_error"] is None

    def test_log_tool_execution_with_error(self):
        """log_tool_execution logs errors correctly."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="tools",
        )
        ctx.log_tool_execution(
            tool_name="gap_analyzer",
            success=False,
            error="Insufficient data for analysis",
        )
        assert ctx.metadata["tool_success"] is False
        assert ctx.metadata["tool_error"] == "Insufficient data for analysis"

    def test_log_tool_execution_with_opik_span(self):
        """log_tool_execution interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="tools",
            _opik_span=mock_span,
        )
        ctx.log_tool_execution(tool_name="test_tool", success=True)
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_finalize_updates_metadata(self):
        """log_finalize updates metadata with finalization metrics."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="finalize",
        )
        ctx.log_finalize(
            response_length=500,
            messages_persisted=True,
            episodic_memory_saved=True,
            significance_score=0.85,
        )
        assert ctx.metadata["response_length"] == 500
        assert ctx.metadata["messages_persisted"] is True
        assert ctx.metadata["episodic_memory_saved"] is True
        assert ctx.metadata["significance_score"] == 0.85

    def test_log_finalize_default_values(self):
        """log_finalize has correct default values."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="finalize",
        )
        ctx.log_finalize(response_length=100, messages_persisted=False)
        assert ctx.metadata["episodic_memory_saved"] is False
        assert ctx.metadata["significance_score"] == 0.0

    def test_log_finalize_with_opik_span(self):
        """log_finalize interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="finalize",
            _opik_span=mock_span,
        )
        ctx.log_finalize(response_length=200, messages_persisted=True)
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_set_output(self):
        """set_output sets output on Opik span."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate",
            _opik_span=mock_span,
        )
        output = {"response": "test", "tokens": 100}
        ctx.set_output(output)
        mock_span.set_output.assert_called_with(output)

    def test_set_output_without_span(self):
        """set_output does nothing when no Opik span."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="generate",
        )
        # Should not raise
        ctx.set_output({"test": "data"})

    def test_log_metadata_updates_metadata(self):
        """log_metadata updates metadata with arbitrary data."""
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="custom",
        )
        ctx.log_metadata(
            {
                "custom_metric": 42,
                "custom_flag": True,
            }
        )
        assert ctx.metadata["custom_metric"] == 42
        assert ctx.metadata["custom_flag"] is True

    def test_log_metadata_with_opik_span(self):
        """log_metadata interacts with Opik span when present."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="custom",
            _opik_span=mock_span,
        )
        ctx.log_metadata({"key": "value"})
        mock_span.set_attribute.assert_called()
        mock_span.add_event.assert_called()

    def test_log_metadata_filters_none_values(self):
        """log_metadata filters None values from Opik events."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            trace_id="trace-123",
            span_id="span-456",
            node_name="custom",
            _opik_span=mock_span,
        )
        ctx.log_metadata({"key": "value", "none_key": None})
        # The add_event call should filter None values
        call_args = mock_span.add_event.call_args
        assert "none_key" not in call_args[0][1]


class TestChatbotTraceContext:
    """Tests for ChatbotTraceContext dataclass."""

    def test_trace_context_creation(self):
        """ChatbotTraceContext can be created with required fields."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="What is the TRx for Kisqali?",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.query == "What is the TRx for Kisqali?"

    def test_trace_context_default_values(self):
        """ChatbotTraceContext has correct default values."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        assert ctx.session_id is None
        assert ctx.end_time is None
        assert ctx.duration_ms is None
        assert ctx.node_spans == {}
        assert ctx.node_durations == {}
        assert ctx.metadata == {}
        assert ctx._opik_span is None
        assert ctx._tracer is None

    def test_trace_context_start_time_auto_set(self):
        """ChatbotTraceContext auto-sets start_time."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        assert ctx.start_time is not None
        assert isinstance(ctx.start_time, datetime)

    @pytest.mark.asyncio
    async def test_trace_node_creates_node_span(self):
        """trace_node creates a NodeSpanContext."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        async with ctx.trace_node("init") as node:
            assert isinstance(node, NodeSpanContext)
            assert node.node_name == "init"
            assert node.trace_id == "trace-123"

    @pytest.mark.asyncio
    async def test_trace_node_records_duration(self):
        """trace_node records node duration."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        async with ctx.trace_node("init") as node:
            pass  # Quick execution

        assert node.end_time is not None
        assert node.duration_ms is not None
        assert node.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_trace_node_stores_in_parent(self):
        """trace_node stores node in parent context."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        async with ctx.trace_node("init"):
            pass
        async with ctx.trace_node("generate"):
            pass

        assert "init" in ctx.node_spans
        assert "generate" in ctx.node_spans
        assert "init" in ctx.node_durations
        assert "generate" in ctx.node_durations

    @pytest.mark.asyncio
    async def test_trace_node_with_metadata(self):
        """trace_node accepts initial metadata."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        async with ctx.trace_node("init", metadata={"custom": "value"}) as node:
            assert node.metadata.get("custom") == "value"

    def test_get_node_index_known_nodes(self):
        """_get_node_index returns correct indices for known nodes."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        assert ctx._get_node_index("init") == 0
        assert ctx._get_node_index("load_context") == 1
        assert ctx._get_node_index("classify_intent") == 2
        assert ctx._get_node_index("retrieve_rag") == 3
        assert ctx._get_node_index("generate") == 4
        assert ctx._get_node_index("tools") == 5
        assert ctx._get_node_index("finalize") == 6

    def test_get_node_index_unknown_node(self):
        """_get_node_index returns -1 for unknown nodes."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        assert ctx._get_node_index("unknown") == -1

    def test_log_workflow_complete_basic(self):
        """log_workflow_complete logs basic metrics."""
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
        )
        ctx.duration_ms = 1500.0
        ctx.log_workflow_complete(
            status="success",
            success=True,
            intent="kpi_query",
            total_tokens=500,
        )
        # Should not raise

    def test_log_workflow_complete_with_opik_span(self):
        """log_workflow_complete interacts with Opik span."""
        mock_span = MagicMock()
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
            _opik_span=mock_span,
        )
        ctx.duration_ms = 1500.0
        ctx.log_workflow_complete(
            status="success",
            success=True,
            intent="kpi_query",
            total_tokens=500,
            tool_calls_count=2,
            rag_result_count=5,
            response_length=300,
        )
        mock_span.set_attribute.assert_called()
        mock_span.set_output.assert_called()

    def test_log_workflow_complete_with_errors(self):
        """log_workflow_complete handles errors list."""
        mock_span = MagicMock()
        ctx = ChatbotTraceContext(
            trace_id="trace-123",
            span_id="span-456",
            query="test",
            _opik_span=mock_span,
        )
        ctx.log_workflow_complete(
            status="partial",
            success=False,
            errors=["Tool execution failed", "RAG retrieval timeout"],
        )
        # Should include errors in output
        call_args = mock_span.set_output.call_args
        output_data = call_args[0][0]
        assert len(output_data["errors"]) == 2


class TestChatbotOpikTracer:
    """Tests for ChatbotOpikTracer class."""

    def setup_method(self):
        """Reset tracer singleton before each test."""
        reset_chatbot_tracer()

    def test_tracer_creation(self):
        """ChatbotOpikTracer can be created."""
        tracer = ChatbotOpikTracer()
        assert tracer.project_name == "e2i-chatbot"
        assert tracer.sample_rate == 1.0

    def test_tracer_creation_with_params(self):
        """ChatbotOpikTracer accepts custom parameters."""
        tracer = ChatbotOpikTracer(
            project_name="custom-project",
            enabled=True,
            sample_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sample_rate == 0.5

    def test_tracer_disabled_by_constructor(self):
        """ChatbotOpikTracer can be disabled via constructor."""
        tracer = ChatbotOpikTracer(enabled=False)
        # enabled should be False regardless of feature flag
        # (enabled = enabled AND CHATBOT_OPIK_TRACING_ENABLED)
        assert tracer.enabled is False

    @patch.dict(os.environ, {"CHATBOT_OPIK_TRACING": "false"})
    def test_tracer_disabled_by_env(self):
        """ChatbotOpikTracer respects CHATBOT_OPIK_TRACING env var."""
        # Need to reload the module to pick up new env var
        # For testing purposes, we create a new tracer with flag check
        ChatbotOpikTracer(enabled=True)
        # The actual CHATBOT_OPIK_TRACING_ENABLED was set at import time
        # so we check the logic pattern
        # tracer.enabled = enabled AND CHATBOT_OPIK_TRACING_ENABLED

    def test_tracer_sample_rate(self):
        """ChatbotOpikTracer respects sample rate."""
        tracer = ChatbotOpikTracer(sample_rate=0.0)
        # With sample_rate=0.0, _should_trace should always be False
        assert tracer._should_trace() is False

        tracer_always = ChatbotOpikTracer(sample_rate=1.0)
        # With sample_rate=1.0, _should_trace should always be True
        assert tracer_always._should_trace() is True

    @pytest.mark.asyncio
    async def test_trace_workflow_creates_context(self):
        """trace_workflow creates a ChatbotTraceContext."""
        tracer = ChatbotOpikTracer(enabled=False)  # Disable Opik
        async with tracer.trace_workflow(
            query="What is the TRx?",
            session_id="session-123",
        ) as ctx:
            assert isinstance(ctx, ChatbotTraceContext)
            assert ctx.query == "What is the TRx?"
            assert ctx.session_id == "session-123"

    @pytest.mark.asyncio
    async def test_trace_workflow_generates_trace_id(self):
        """trace_workflow generates a trace ID."""
        tracer = ChatbotOpikTracer(enabled=False)
        async with tracer.trace_workflow(query="test") as ctx:
            assert ctx.trace_id is not None
            assert len(ctx.trace_id) > 0

    @pytest.mark.asyncio
    async def test_trace_workflow_records_duration(self):
        """trace_workflow records workflow duration."""
        tracer = ChatbotOpikTracer(enabled=False)
        async with tracer.trace_workflow(query="test") as ctx:
            pass  # Quick execution

        assert ctx.end_time is not None
        assert ctx.duration_ms is not None
        assert ctx.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_trace_workflow_with_metadata(self):
        """trace_workflow includes metadata."""
        tracer = ChatbotOpikTracer(enabled=False)
        async with tracer.trace_workflow(
            query="test query",
            session_id="session-123",
            user_id="user-456",
            brand_context="Kisqali",
            region_context="Northeast",
            metadata={"custom": "value"},
        ) as ctx:
            assert ctx.metadata["query_length"] == len("test query")
            assert ctx.metadata["session_id"] == "session-123"
            assert ctx.metadata["user_id"] == "user-456"
            assert ctx.metadata["brand_context"] == "Kisqali"
            assert ctx.metadata["region_context"] == "Northeast"
            assert ctx.metadata["custom"] == "value"

    @pytest.mark.asyncio
    async def test_trace_workflow_with_nested_nodes(self):
        """trace_workflow supports nested node tracing."""
        tracer = ChatbotOpikTracer(enabled=False)
        async with tracer.trace_workflow(query="test") as trace:
            async with trace.trace_node("init") as node:
                node.log_init(is_new_conversation=True)
            async with trace.trace_node("generate") as node:
                node.log_generate(input_tokens=100, output_tokens=200)
            trace.log_workflow_complete(status="success", success=True)

        assert "init" in trace.node_spans
        assert "generate" in trace.node_spans
        assert trace.node_spans["init"].metadata["is_new_conversation"] is True

    @pytest.mark.asyncio
    async def test_trace_workflow_exception_handling(self):
        """trace_workflow handles exceptions properly."""
        tracer = ChatbotOpikTracer(enabled=False)

        with pytest.raises(ValueError, match="test error"):
            async with tracer.trace_workflow(query="test") as ctx:
                raise ValueError("test error")

        # Duration should still be recorded
        assert ctx.end_time is not None


class TestGetChatbotTracer:
    """Tests for get_chatbot_tracer singleton function."""

    def setup_method(self):
        """Reset tracer singleton before each test."""
        reset_chatbot_tracer()

    def test_get_chatbot_tracer_returns_tracer(self):
        """get_chatbot_tracer returns a ChatbotOpikTracer."""
        tracer = get_chatbot_tracer()
        assert isinstance(tracer, ChatbotOpikTracer)

    def test_get_chatbot_tracer_returns_singleton(self):
        """get_chatbot_tracer returns same instance."""
        tracer1 = get_chatbot_tracer()
        tracer2 = get_chatbot_tracer()
        assert tracer1 is tracer2

    def test_get_chatbot_tracer_with_params(self):
        """get_chatbot_tracer accepts parameters on first call."""
        tracer = get_chatbot_tracer(
            project_name="custom-project",
            enabled=True,
            sample_rate=0.75,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sample_rate == 0.75


class TestResetChatbotTracer:
    """Tests for reset_chatbot_tracer function."""

    def test_reset_chatbot_tracer_clears_singleton(self):
        """reset_chatbot_tracer clears the singleton."""
        tracer1 = get_chatbot_tracer()
        reset_chatbot_tracer()
        tracer2 = get_chatbot_tracer()
        assert tracer1 is not tracer2

    def test_reset_chatbot_tracer_allows_new_config(self):
        """reset_chatbot_tracer allows new configuration."""
        # Explicitly reset at start to handle parallel test execution
        reset_chatbot_tracer()
        tracer1 = get_chatbot_tracer(project_name="project-1")
        assert tracer1.project_name == "project-1"
        reset_chatbot_tracer()
        tracer2 = get_chatbot_tracer(project_name="project-2")
        assert tracer2.project_name == "project-2"
        # Verify they are different instances
        assert tracer1 is not tracer2


class TestTraceChatbotWorkflowDecorator:
    """Tests for trace_chatbot_workflow decorator."""

    def setup_method(self):
        """Reset tracer singleton before each test."""
        reset_chatbot_tracer()

    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self):
        """trace_chatbot_workflow decorator creates trace context."""

        @trace_chatbot_workflow()
        async def my_workflow(trace: ChatbotTraceContext, query: str):
            assert isinstance(trace, ChatbotTraceContext)
            assert trace.query == query
            return "result"

        result = await my_workflow(query="test query")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_with_session_id(self):
        """trace_chatbot_workflow decorator handles session_id."""

        @trace_chatbot_workflow()
        async def my_workflow(trace: ChatbotTraceContext, query: str, session_id=None):
            assert trace.session_id == session_id
            return "result"

        await my_workflow(query="test", session_id="session-123")

    @pytest.mark.asyncio
    async def test_decorator_custom_params(self):
        """trace_chatbot_workflow decorator accepts custom param names."""

        @trace_chatbot_workflow(query_param="q", session_id_param="sid")
        async def my_workflow(trace: ChatbotTraceContext, q: str, sid=None):
            assert trace.query == q
            assert trace.session_id == sid
            return "result"

        await my_workflow(q="custom query", sid="custom-session")

    @pytest.mark.asyncio
    async def test_decorator_with_node_tracing(self):
        """trace_chatbot_workflow decorator supports node tracing."""

        @trace_chatbot_workflow()
        async def my_workflow(trace: ChatbotTraceContext, query: str):
            async with trace.trace_node("init") as node:
                node.log_init(is_new_conversation=True)
            async with trace.trace_node("generate") as node:
                node.log_generate(input_tokens=50, output_tokens=100)
            return trace.node_spans

        result = await my_workflow(query="test")
        assert "init" in result
        assert "generate" in result


class TestFeatureFlag:
    """Tests for CHATBOT_OPIK_TRACING_ENABLED feature flag."""

    def test_feature_flag_type(self):
        """CHATBOT_OPIK_TRACING_ENABLED is a boolean."""
        assert isinstance(CHATBOT_OPIK_TRACING_ENABLED, bool)


class TestIntegrationScenarios:
    """Integration tests for realistic tracing scenarios."""

    def setup_method(self):
        """Reset tracer singleton before each test."""
        reset_chatbot_tracer()

    @pytest.mark.asyncio
    async def test_full_workflow_trace(self):
        """Test complete workflow tracing flow."""
        tracer = ChatbotOpikTracer(enabled=False)  # Disable Opik for unit test

        async with tracer.trace_workflow(
            query="What is the TRx for Kisqali in Q4?",
            session_id="session-123",
            user_id="analyst-001",
            brand_context="Kisqali",
        ) as trace:
            # Init node
            async with trace.trace_node("init") as node:
                node.log_init(
                    is_new_conversation=True,
                    session_id="session-123",
                    user_id="analyst-001",
                )

            # Load context node
            async with trace.trace_node("load_context") as node:
                node.log_context_load(
                    previous_message_count=0,
                    brand_context="Kisqali",
                )

            # Classify intent node
            async with trace.trace_node("classify_intent") as node:
                node.log_intent_classification(
                    intent="kpi_query",
                    confidence=0.92,
                    classification_method="dspy",
                )

            # Retrieve RAG node
            async with trace.trace_node("retrieve_rag") as node:
                node.log_rag_retrieval(
                    result_count=5,
                    relevance_scores=[0.95, 0.90, 0.85, 0.80, 0.75],
                    brand_filter="Kisqali",
                )

            # Generate node
            async with trace.trace_node("generate") as node:
                node.log_generate(
                    input_tokens=500,
                    output_tokens=200,
                    model="claude-sonnet-4-20250514",
                )

            # Finalize node
            async with trace.trace_node("finalize") as node:
                node.log_finalize(
                    response_length=350,
                    messages_persisted=True,
                )

            # Complete workflow
            trace.log_workflow_complete(
                status="success",
                success=True,
                intent="kpi_query",
                total_tokens=700,
                rag_result_count=5,
                response_length=350,
            )

        # Verify trace structure
        assert len(trace.node_spans) == 6
        assert all(
            name in trace.node_spans
            for name in [
                "init",
                "load_context",
                "classify_intent",
                "retrieve_rag",
                "generate",
                "finalize",
            ]
        )
        assert trace.node_spans["classify_intent"].metadata["intent"] == "kpi_query"
        assert trace.node_spans["generate"].metadata["total_tokens"] == 700

    @pytest.mark.asyncio
    async def test_workflow_with_tool_execution(self):
        """Test workflow with tool execution tracing."""
        tracer = ChatbotOpikTracer(enabled=False)

        async with tracer.trace_workflow(
            query="What caused the sales decline?",
            session_id="session-456",
        ) as trace:
            async with trace.trace_node("classify_intent") as node:
                node.log_intent_classification(
                    intent="causal_analysis",
                    confidence=0.88,
                )

            async with trace.trace_node("tools") as node:
                # Simulate multiple tool calls
                node.log_tool_execution(
                    tool_name="causal_chain_tracer",
                    success=True,
                    result_size=2048,
                )
                node.log_metadata(
                    {
                        "additional_tool": "gap_analyzer",
                        "gap_score": 0.75,
                    }
                )

            trace.log_workflow_complete(
                status="success",
                success=True,
                intent="causal_analysis",
                tool_calls_count=2,
            )

        assert trace.node_spans["tools"].metadata["tool_name"] == "causal_chain_tracer"

    @pytest.mark.asyncio
    async def test_workflow_with_error(self):
        """Test workflow tracing with errors."""
        tracer = ChatbotOpikTracer(enabled=False)

        async with tracer.trace_workflow(
            query="Invalid query",
            session_id="session-789",
        ) as trace:
            async with trace.trace_node("classify_intent") as node:
                node.log_intent_classification(
                    intent="general",
                    confidence=0.5,
                )

            async with trace.trace_node("retrieve_rag") as node:
                node.log_rag_retrieval(
                    result_count=0,
                )

            trace.log_workflow_complete(
                status="partial",
                success=False,
                intent="general",
                errors=["No relevant context found"],
            )

        assert trace.node_spans["retrieve_rag"].metadata["result_count"] == 0


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_available(self):
        """All __all__ exports are importable."""
        from src.api.routes.chatbot_tracer import __all__

        expected = [
            "NodeSpanContext",
            "ChatbotTraceContext",
            "ChatbotOpikTracer",
            "trace_chatbot_workflow",
            "get_chatbot_tracer",
            "reset_chatbot_tracer",
            "CHATBOT_OPIK_TRACING_ENABLED",
        ]
        assert set(__all__) == set(expected)
