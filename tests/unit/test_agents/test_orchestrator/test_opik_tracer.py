"""Tests for Orchestrator Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Orchestrator agent.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator.opik_tracer import (
    NodeSpanContext,
    OrchestrationTraceContext,
    OrchestratorOpikTracer,
    ORCHESTRATION_PHASES,
    PIPELINE_NODES,
    get_orchestrator_tracer,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    # Reset the module-level singleton
    import src.agents.orchestrator.opik_tracer as tracer_module
    tracer_module._tracer_instance = None
    # Reset class-level singleton
    OrchestratorOpikTracer._instance = None
    OrchestratorOpikTracer._initialized = False
    yield
    tracer_module._tracer_instance = None
    OrchestratorOpikTracer._instance = None
    OrchestratorOpikTracer._initialized = False


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_orchestration_phases(self):
        """Test ORCHESTRATION_PHASES constant."""
        assert len(ORCHESTRATION_PHASES) == 5
        assert "classifying" in ORCHESTRATION_PHASES
        assert "routing" in ORCHESTRATION_PHASES
        assert "dispatching" in ORCHESTRATION_PHASES
        assert "synthesizing" in ORCHESTRATION_PHASES

    def test_pipeline_nodes(self):
        """Test PIPELINE_NODES constant."""
        assert len(PIPELINE_NODES) == 5
        assert "classify" in PIPELINE_NODES
        assert "rag_context" in PIPELINE_NODES
        assert "route" in PIPELINE_NODES
        assert "dispatch" in PIPELINE_NODES
        assert "synthesize" in PIPELINE_NODES


# ============================================================================
# NODE SPAN CONTEXT TESTS
# ============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_create_node_span_context(self):
        """Test creating a node span context."""
        ctx = NodeSpanContext(
            span=None,
            node_name="classify",
        )

        assert ctx.span is None
        assert ctx.node_name == "classify"
        assert ctx.start_time > 0
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to node span."""
        ctx = NodeSpanContext(span=None, node_name="route")

        ctx.add_metadata("agents_selected", ["causal_impact", "gap_analyzer"])
        ctx.add_metadata("confidence", 0.85)

        assert ctx.metadata["agents_selected"] == ["causal_impact", "gap_analyzer"]
        assert ctx.metadata["confidence"] == 0.85

    def test_end_without_span(self):
        """Test ending node span without Opik span attached."""
        ctx = NodeSpanContext(span=None, node_name="dispatch")

        # Should not raise even without span
        ctx.end(status="completed")

        assert ctx.metadata == {}  # No metadata added since no span

    def test_end_with_mock_span(self):
        """Test ending node span with mock Opik span."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(span=mock_span, node_name="synthesize")
        ctx.add_metadata("response_length", 500)

        ctx.end(status="completed")

        mock_span.end.assert_called_once()
        call_kwargs = mock_span.end.call_args[1]
        assert call_kwargs["metadata"]["status"] == "completed"
        assert call_kwargs["metadata"]["response_length"] == 500
        assert "duration_ms" in call_kwargs["metadata"]

    def test_end_handles_span_error(self):
        """Test end handles span error gracefully."""
        mock_span = MagicMock()
        mock_span.end.side_effect = Exception("Opik error")
        ctx = NodeSpanContext(span=mock_span, node_name="classify")

        # Should not raise
        ctx.end(status="error")


# ============================================================================
# ORCHESTRATION TRACE CONTEXT TESTS
# ============================================================================


class TestOrchestrationTraceContext:
    """Tests for OrchestrationTraceContext dataclass."""

    def test_create_trace_context(self):
        """Test creating an orchestration trace context."""
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(
            trace=None,
            tracer=tracer,
            query_id="q-123",
        )

        assert ctx.trace is None
        assert ctx.tracer is tracer
        assert ctx.query_id == "q-123"
        assert ctx.start_time > 0
        assert ctx.trace_metadata == {}
        assert ctx.active_spans == {}

    def test_log_orchestration_started(self):
        """Test logging orchestration start without trace."""
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=None, tracer=tracer, query_id="q-123")

        # Should not raise without trace
        ctx.log_orchestration_started(
            query="What caused the revenue drop?",
            user_id="user-456",
            session_id="sess-789",
        )

    def test_log_orchestration_started_with_mock_trace(self):
        """Test logging orchestration start with mock trace."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_orchestration_started(
            query="Analyze rep visit effectiveness",
            user_id="user-456",
        )

        assert ctx.trace_metadata["user_id"] == "user-456"
        assert "Analyze rep visit" in ctx.trace_metadata["query"]
        mock_trace.update.assert_called()

    def test_start_node_span_without_trace(self):
        """Test starting a node span without trace."""
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=None, tracer=tracer, query_id="q-123")

        node_ctx = ctx.start_node_span("classify", {"query": "test"})

        assert isinstance(node_ctx, NodeSpanContext)
        assert node_ctx.node_name == "classify"
        assert node_ctx.span is None
        assert "classify" in ctx.active_spans

    def test_start_node_span_with_mock_trace(self):
        """Test starting a node span with mock trace."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        tracer = OrchestratorOpikTracer(enabled=False)
        tracer._client = MagicMock()  # Simulate initialized client
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        node_ctx = ctx.start_node_span("route", {"intent": "CAUSAL"})

        assert node_ctx.span is mock_span
        mock_trace.span.assert_called_once()

    def test_end_node_span(self):
        """Test ending a node span."""
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=None, tracer=tracer, query_id="q-123")

        ctx.start_node_span("dispatch")
        assert "dispatch" in ctx.active_spans

        ctx.end_node_span("dispatch", {"agents": ["causal_impact"]}, "completed")
        assert "dispatch" not in ctx.active_spans

    def test_end_node_span_with_mock_span(self):
        """Test ending a node span with mock span."""
        mock_span = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=None, tracer=tracer, query_id="q-123")
        ctx.active_spans["synthesize"] = NodeSpanContext(span=mock_span, node_name="synthesize")

        ctx.end_node_span("synthesize", {"response": "Analysis complete"}, "completed")

        mock_span.end.assert_called_once()

    def test_log_intent_classification(self):
        """Test logging intent classification."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_intent_classification(
            primary_intent="CAUSAL",
            confidence=0.92,
            secondary_intents=["COMPARATIVE", "DESCRIPTIVE"],
            classification_latency_ms=150,
        )

        assert ctx.trace_metadata["primary_intent"] == "CAUSAL"
        assert ctx.trace_metadata["intent_confidence"] == 0.92
        assert ctx.trace_metadata["classification_latency_ms"] == 150

    def test_log_rag_retrieval(self):
        """Test logging RAG retrieval."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_rag_retrieval(
            context_retrieved=True,
            chunks_count=5,
            rag_latency_ms=200,
        )

        assert ctx.trace_metadata["context_retrieved"] is True
        assert ctx.trace_metadata["rag_chunks_count"] == 5
        assert ctx.trace_metadata["rag_latency_ms"] == 200

    def test_log_agent_routing(self):
        """Test logging agent routing."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_agent_routing(
            agents_selected=["causal_impact", "gap_analyzer"],
            routing_rationale="Query requires causal analysis and ROI estimation",
            routing_latency_ms=30,
        )

        assert ctx.trace_metadata["agents_selected"] == ["causal_impact", "gap_analyzer"]
        assert ctx.trace_metadata["agents_count"] == 2
        assert ctx.trace_metadata["routing_latency_ms"] == 30

    def test_log_agent_dispatch(self):
        """Test logging agent dispatch."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_agent_dispatch(
            agents_dispatched=["causal_impact", "gap_analyzer"],
            successful_agents=["causal_impact", "gap_analyzer"],
            failed_agents=[],
            dispatch_latency_ms=500,
        )

        assert ctx.trace_metadata["agents_dispatched"] == ["causal_impact", "gap_analyzer"]
        assert ctx.trace_metadata["dispatch_success_rate"] == 1.0
        assert ctx.trace_metadata["dispatch_latency_ms"] == 500

    def test_log_agent_dispatch_with_failures(self):
        """Test logging agent dispatch with failures."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_agent_dispatch(
            agents_dispatched=["causal_impact", "gap_analyzer", "prediction_synthesizer"],
            successful_agents=["causal_impact", "gap_analyzer"],
            failed_agents=["prediction_synthesizer"],
            dispatch_latency_ms=800,
        )

        assert len(ctx.trace_metadata["failed_agents"]) == 1
        assert ctx.trace_metadata["dispatch_success_rate"] == pytest.approx(0.666, rel=0.01)

    def test_log_response_synthesis(self):
        """Test logging response synthesis."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_response_synthesis(
            response_length=1500,
            citations_count=3,
            synthesis_latency_ms=250,
        )

        assert ctx.trace_metadata["response_length"] == 1500
        assert ctx.trace_metadata["citations_count"] == 3
        assert ctx.trace_metadata["synthesis_latency_ms"] == 250

    def test_log_orchestration_complete(self):
        """Test logging orchestration completion."""
        mock_trace = MagicMock()
        tracer = OrchestratorOpikTracer(enabled=False)
        ctx = OrchestrationTraceContext(trace=mock_trace, tracer=tracer, query_id="q-123")

        ctx.log_orchestration_complete(
            status="success",
            success=True,
            total_duration_ms=1500,
            response_confidence=0.9,
            agents_dispatched=["causal_impact"],
            successful_agents=["causal_impact"],
            failed_agents=[],
            has_partial_failure=False,
            primary_intent="CAUSAL",
            classification_latency_ms=100,
            rag_latency_ms=200,
            routing_latency_ms=50,
            dispatch_latency_ms=800,
            synthesis_latency_ms=350,
            errors=[],
            warnings=[],
        )

        assert ctx.trace_metadata["status"] == "success"
        assert ctx.trace_metadata["success"] is True
        assert ctx.trace_metadata["total_duration_ms"] == 1500
        mock_trace.update.assert_called()


# ============================================================================
# ORCHESTRATOR OPIK TRACER TESTS
# ============================================================================


class TestOrchestratorOpikTracer:
    """Tests for OrchestratorOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = OrchestratorOpikTracer()

        assert tracer.project_name == "e2i-orchestrator"
        assert tracer.sampling_rate == 1.0
        assert tracer.enabled is True
        assert tracer._client is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = OrchestratorOpikTracer(
            project_name="custom-orchestrator",
            sampling_rate=0.5,
            enabled=False,
        )

        assert tracer.project_name == "custom-orchestrator"
        assert tracer.sampling_rate == 0.5
        assert tracer.enabled is False

    def test_singleton_pattern(self):
        """Test tracer uses singleton pattern."""
        tracer1 = OrchestratorOpikTracer()
        tracer2 = OrchestratorOpikTracer()

        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test singleton skips re-initialization."""
        tracer1 = OrchestratorOpikTracer(project_name="first-project")
        tracer2 = OrchestratorOpikTracer(project_name="second-project")

        # Both should reference the same instance with first config
        assert tracer1 is tracer2
        assert tracer1.project_name == "first-project"

    def test_get_client_lazy_init(self):
        """Test client is lazily initialized."""
        opik = pytest.importorskip("opik", reason="opik not installed")

        with patch("opik.Opik") as mock_opik_class:
            mock_client = MagicMock()
            mock_opik_class.return_value = mock_client

            tracer = OrchestratorOpikTracer()
            assert tracer._client is None

            client = tracer._get_client()
            assert client is mock_client
            mock_opik_class.assert_called_once_with(project_name="e2i-orchestrator")

    def test_get_client_disabled(self):
        """Test _get_client returns None when disabled."""
        tracer = OrchestratorOpikTracer(enabled=False)

        client = tracer._get_client()
        assert client is None

    def test_get_client_handles_import_error(self):
        """Test _get_client handles ImportError gracefully."""
        opik = pytest.importorskip("opik", reason="opik not installed")

        with patch("opik.Opik") as mock_opik_class:
            mock_opik_class.side_effect = ImportError("opik not installed")

            tracer = OrchestratorOpikTracer()
            client = tracer._get_client()

            assert client is None
            assert tracer.enabled is False  # Should be disabled after error

    def test_should_sample_full_rate(self):
        """Test _should_sample with 1.0 sample rate."""
        tracer = OrchestratorOpikTracer(sampling_rate=1.0)

        # Should always return True with 1.0 rate
        results = [tracer._should_sample() for _ in range(10)]
        assert all(results)

    def test_should_sample_zero_rate(self):
        """Test _should_sample with 0.0 sample rate."""
        tracer = OrchestratorOpikTracer(sampling_rate=0.0)

        # Should always return False with 0.0 rate
        results = [tracer._should_sample() for _ in range(10)]
        assert not any(results)

    def test_generate_trace_id_format(self):
        """Test _generate_trace_id generates valid UUID format."""
        tracer = OrchestratorOpikTracer()

        trace_id = tracer._generate_trace_id()

        # Should be valid UUID format
        assert len(trace_id) == 36  # UUID string length with hyphens
        assert trace_id.count("-") == 4  # UUID has 4 hyphens

    @pytest.mark.asyncio
    async def test_trace_orchestration_disabled(self):
        """Test trace_orchestration when disabled."""
        tracer = OrchestratorOpikTracer(enabled=False)

        async with tracer.trace_orchestration(query_id="q-123") as ctx:
            assert isinstance(ctx, OrchestrationTraceContext)
            assert ctx.trace is None
            assert ctx.query_id == "q-123"

    @pytest.mark.asyncio
    async def test_trace_orchestration_not_sampled(self):
        """Test trace_orchestration when not sampled."""
        tracer = OrchestratorOpikTracer(sampling_rate=0.0)

        async with tracer.trace_orchestration(query_id="q-123") as ctx:
            assert ctx.trace is None

    @pytest.mark.asyncio
    async def test_trace_orchestration_full_pipeline(self):
        """Test full orchestration trace with all nodes."""
        tracer = OrchestratorOpikTracer(enabled=False)

        async with tracer.trace_orchestration(
            query_id="q-123",
            query="What is the causal impact of rep visits?",
            user_id="user-456",
            session_id="sess-789",
        ) as ctx:
            # Simulate full orchestration pipeline
            ctx.log_orchestration_started("What is the causal impact of rep visits?", "user-456", "sess-789")

            node = ctx.start_node_span("classify", {"query": "test"})
            node.add_metadata("intent", "CAUSAL")
            ctx.end_node_span("classify", {"intent": "CAUSAL"})
            ctx.log_intent_classification("CAUSAL", 0.95, [], 100)

            ctx.start_node_span("rag_context")
            ctx.end_node_span("rag_context", {"chunks": 3})
            ctx.log_rag_retrieval(True, 3, 200)

            ctx.start_node_span("route")
            ctx.end_node_span("route", {"agent": "causal_impact"})
            ctx.log_agent_routing(["causal_impact"], "Query requires causal analysis", 30)

            ctx.start_node_span("dispatch")
            ctx.end_node_span("dispatch", {"status": "success"})
            ctx.log_agent_dispatch(["causal_impact"], ["causal_impact"], [], 500)

            ctx.start_node_span("synthesize")
            ctx.end_node_span("synthesize", {"response": "Analysis..."})
            ctx.log_response_synthesis(1000, 2, 200)

            ctx.log_orchestration_complete(
                status="success",
                success=True,
                total_duration_ms=1030,
                response_confidence=0.9,
                agents_dispatched=["causal_impact"],
                successful_agents=["causal_impact"],
                failed_agents=[],
                has_partial_failure=False,
                primary_intent="CAUSAL",
                classification_latency_ms=100,
                rag_latency_ms=200,
                routing_latency_ms=30,
                dispatch_latency_ms=500,
                synthesis_latency_ms=200,
                errors=[],
                warnings=[],
            )

        # Verify all nodes were processed
        assert len(ctx.active_spans) == 0  # All spans ended

    def test_flush_without_client(self):
        """Test flush when client is not initialized."""
        tracer = OrchestratorOpikTracer(enabled=False)

        # Should not raise
        tracer.flush()

    def test_flush_with_client(self):
        """Test flush calls client flush."""
        opik = pytest.importorskip("opik", reason="opik not installed")

        with patch("opik.Opik") as mock_opik_class:
            mock_client = MagicMock()
            mock_opik_class.return_value = mock_client

            tracer = OrchestratorOpikTracer()
            tracer._get_client()  # Initialize client

            tracer.flush()

            mock_client.flush.assert_called_once()


# ============================================================================
# SINGLETON FACTORY TESTS
# ============================================================================


class TestGetOrchestratorTracer:
    """Tests for get_orchestrator_tracer singleton factory."""

    def test_returns_tracer_instance(self):
        """Test returns OrchestratorOpikTracer instance."""
        tracer = get_orchestrator_tracer()

        assert isinstance(tracer, OrchestratorOpikTracer)

    def test_returns_same_instance(self):
        """Test returns same singleton instance."""
        tracer1 = get_orchestrator_tracer()
        tracer2 = get_orchestrator_tracer()

        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_orchestrator_tracer(
            project_name="custom-project",
            sampling_rate=0.5,
            enabled=False,
        )

        assert tracer.project_name == "custom-project"
        assert tracer.sampling_rate == 0.5
        assert tracer.enabled is False

    def test_subsequent_calls_use_cached_instance(self):
        """Test subsequent calls use cached instance."""
        tracer1 = get_orchestrator_tracer(project_name="first")
        tracer2 = get_orchestrator_tracer(project_name="second")

        assert tracer1 is tracer2
        assert tracer2.project_name == "first"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestOpikIntegration:
    """Integration tests with mocked Opik."""

    @pytest.mark.asyncio
    async def test_full_trace_with_opik(self):
        """Test full trace with Opik client."""
        opik = pytest.importorskip("opik", reason="opik not installed")

        with patch("opik.Opik") as mock_opik_class:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_span = MagicMock()
            mock_client.trace.return_value = mock_trace
            mock_trace.span.return_value = mock_span
            mock_opik_class.return_value = mock_client

            tracer = OrchestratorOpikTracer(sampling_rate=1.0)

            async with tracer.trace_orchestration(
                query_id="q-123",
                query="Test query",
            ) as ctx:
                node_ctx = ctx.start_node_span("classify")
                ctx.end_node_span("classify")

            mock_client.trace.assert_called_once()
            mock_trace.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_trace_error(self):
        """Test graceful degradation when trace creation fails."""
        opik = pytest.importorskip("opik", reason="opik not installed")

        with patch("opik.Opik") as mock_opik_class:
            mock_client = MagicMock()
            mock_client.trace.side_effect = Exception("Trace creation failed")
            mock_opik_class.return_value = mock_client

            tracer = OrchestratorOpikTracer(sampling_rate=1.0)

            # Should not raise, should fall back gracefully
            async with tracer.trace_orchestration(query_id="q-123") as ctx:
                assert ctx.trace is None

                # Operations should still work
                ctx.start_node_span("classify")
                ctx.end_node_span("classify")
