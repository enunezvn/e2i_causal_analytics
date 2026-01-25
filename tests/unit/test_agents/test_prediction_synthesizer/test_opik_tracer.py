"""Tests for Prediction Synthesizer Opik Tracer.

Version: 1.0.0
Tests the Opik observability integration for Prediction Synthesizer agent's synthesis pipeline.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.prediction_synthesizer.opik_tracer import (
    ENSEMBLE_METHODS,
    ENTITY_TYPES,
    PIPELINE_NODES,
    NodeSpanContext,
    SynthesisTraceContext,
    PredictionSynthesizerOpikTracer,
    get_prediction_synthesizer_tracer,
    reset_tracer,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the tracer singleton before each test."""
    reset_tracer()
    yield
    reset_tracer()


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_ensemble_methods(self):
        """Test ENSEMBLE_METHODS constant."""
        assert "average" in ENSEMBLE_METHODS
        assert "weighted" in ENSEMBLE_METHODS
        assert "stacking" in ENSEMBLE_METHODS
        assert "voting" in ENSEMBLE_METHODS

    def test_entity_types(self):
        """Test ENTITY_TYPES constant."""
        assert "hcp" in ENTITY_TYPES
        assert "territory" in ENTITY_TYPES
        assert "patient" in ENTITY_TYPES

    def test_pipeline_nodes(self):
        """Test PIPELINE_NODES constant."""
        assert "orchestrate" in PIPELINE_NODES
        assert "combine" in PIPELINE_NODES
        assert "enrich" in PIPELINE_NODES


# ============================================================================
# NODE SPAN CONTEXT TESTS
# ============================================================================


class TestNodeSpanContext:
    """Tests for NodeSpanContext dataclass."""

    def test_create_node_span_context(self):
        """Test NodeSpanContext creation with required fields."""
        ctx = NodeSpanContext(
            span=None,
            node_name="orchestrate",
        )
        assert ctx.span is None
        assert ctx.node_name == "orchestrate"
        assert ctx.start_time > 0
        assert ctx.metadata == {}

    def test_add_metadata(self):
        """Test adding metadata to span context."""
        ctx = NodeSpanContext(
            span=None,
            node_name="combine",
        )
        ctx.add_metadata("key", "value")
        assert ctx.metadata["key"] == "value"

    def test_end_without_span(self):
        """Test ending span context without Opik span."""
        ctx = NodeSpanContext(
            span=None,
            node_name="orchestrate",
        )
        ctx.add_metadata("test", "value")
        # Should not raise
        ctx.end(status="completed")

    def test_end_with_mock_span(self):
        """Test ending span context with mock Opik span."""
        mock_span = MagicMock()
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="combine",
        )
        ctx.add_metadata("ensemble_method", "weighted")
        ctx.end(status="completed")
        mock_span.end.assert_called_once()

    def test_end_handles_span_error(self):
        """Test that end handles span errors gracefully."""
        mock_span = MagicMock()
        mock_span.end.side_effect = Exception("Span error")
        ctx = NodeSpanContext(
            span=mock_span,
            node_name="orchestrate",
        )
        # Should not raise
        ctx.end(status="completed")


# ============================================================================
# SYNTHESIS TRACE CONTEXT TESTS
# ============================================================================


class TestSynthesisTraceContext:
    """Tests for SynthesisTraceContext class."""

    def test_create_trace_context(self):
        """Test SynthesisTraceContext creation."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        assert ctx.trace is None
        assert ctx.tracer is tracer
        assert ctx.entity_type == "hcp"
        assert ctx.prediction_target == "churn"
        assert ctx.start_time > 0
        assert ctx.trace_metadata == {}
        assert ctx.active_spans == {}

    def test_log_synthesis_started(self):
        """Test logging synthesis started."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        # Should not raise without trace
        ctx.log_synthesis_started(
            entity_id="HCP-12345",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=3,
            ensemble_method="weighted",
            include_context=True,
        )

    def test_log_synthesis_started_with_trace(self):
        """Test logging synthesis started with mock trace."""
        mock_trace = MagicMock()
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=mock_trace,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        ctx.log_synthesis_started(
            entity_id="HCP-12345",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=3,
            ensemble_method="weighted",
            include_context=True,
        )
        assert ctx.trace_metadata.get("entity_id") == "HCP-12345"
        assert ctx.trace_metadata.get("ensemble_method") == "weighted"
        mock_trace.update.assert_called()

    def test_log_model_orchestration(self):
        """Test logging model orchestration."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="conversion",
        )
        # Should not raise without trace
        ctx.log_model_orchestration(
            models_requested=3,
            models_succeeded=3,
            models_failed=0,
            orchestration_latency_ms=500,
        )

    def test_log_model_orchestration_with_trace(self):
        """Test logging model orchestration with trace."""
        mock_trace = MagicMock()
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=mock_trace,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="conversion",
        )
        ctx.log_model_orchestration(
            models_requested=3,
            models_succeeded=2,
            models_failed=1,
            orchestration_latency_ms=500,
        )
        assert ctx.trace_metadata.get("models_requested") == 3
        assert ctx.trace_metadata.get("models_succeeded") == 2
        assert ctx.trace_metadata.get("models_failed") == 1
        assert ctx.trace_metadata.get("model_success_rate") == 2 / 3
        mock_trace.update.assert_called()

    def test_log_ensemble_combination(self):
        """Test logging ensemble combination."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        # Should not raise without trace
        ctx.log_ensemble_combination(
            ensemble_method="weighted",
            point_estimate=0.75,
            prediction_interval_lower=0.65,
            prediction_interval_upper=0.85,
            confidence=0.85,
            model_agreement=0.92,
            ensemble_latency_ms=200,
        )

    def test_log_ensemble_combination_with_trace(self):
        """Test logging ensemble combination with trace."""
        mock_trace = MagicMock()
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=mock_trace,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        ctx.log_ensemble_combination(
            ensemble_method="weighted",
            point_estimate=0.75,
            prediction_interval_lower=0.65,
            prediction_interval_upper=0.85,
            confidence=0.85,
            model_agreement=0.92,
            ensemble_latency_ms=200,
        )
        assert ctx.trace_metadata.get("ensemble_method") == "weighted"
        assert ctx.trace_metadata.get("point_estimate") == 0.75
        assert ctx.trace_metadata.get("confidence") == 0.85
        assert ctx.trace_metadata.get("prediction_interval_width") == pytest.approx(0.20)
        mock_trace.update.assert_called()

    def test_log_context_enrichment(self):
        """Test logging context enrichment."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        # Should not raise without trace
        ctx.log_context_enrichment(
            similar_cases_found=5,
            feature_importance_calculated=True,
            historical_accuracy=0.82,
            trend_direction="increasing",
            enrichment_latency_ms=150,
        )

    def test_log_context_enrichment_with_trace(self):
        """Test logging context enrichment with trace."""
        mock_trace = MagicMock()
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=mock_trace,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        ctx.log_context_enrichment(
            similar_cases_found=5,
            feature_importance_calculated=True,
            historical_accuracy=0.82,
            trend_direction="increasing",
            enrichment_latency_ms=150,
        )
        assert ctx.trace_metadata.get("similar_cases_found") == 5
        assert ctx.trace_metadata.get("feature_importance_calculated") is True
        assert ctx.trace_metadata.get("historical_accuracy") == 0.82
        assert ctx.trace_metadata.get("trend_direction") == "increasing"
        mock_trace.update.assert_called()

    def test_log_synthesis_complete(self):
        """Test logging synthesis complete."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        # Should not raise without trace
        ctx.log_synthesis_complete(
            status="completed",
            success=True,
            total_duration_ms=1000,
            point_estimate=0.75,
            confidence=0.85,
            model_agreement=0.92,
            models_succeeded=3,
            models_failed=0,
            prediction_summary="High churn probability predicted.",
            errors=[],
            warnings=[],
        )

    def test_log_synthesis_complete_with_trace(self):
        """Test logging synthesis complete with trace."""
        mock_trace = MagicMock()
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=mock_trace,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        ctx.log_synthesis_complete(
            status="completed",
            success=True,
            total_duration_ms=1000,
            point_estimate=0.75,
            confidence=0.85,
            model_agreement=0.92,
            models_succeeded=3,
            models_failed=0,
            prediction_summary="High churn probability predicted.",
            errors=[],
            warnings=[],
        )
        assert ctx.trace_metadata.get("status") == "completed"
        assert ctx.trace_metadata.get("success") is True
        assert ctx.trace_metadata.get("point_estimate") == 0.75
        assert ctx.trace_metadata.get("confidence") == 0.85
        mock_trace.update.assert_called()

    def test_start_node_span_without_trace(self):
        """Test starting a node span without trace."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        node_ctx = ctx.start_node_span("orchestrate", {"models_requested": 3})
        assert isinstance(node_ctx, NodeSpanContext)
        assert node_ctx.node_name == "orchestrate"
        assert node_ctx.span is None
        assert "orchestrate" in ctx.active_spans

    def test_end_node_span(self):
        """Test ending a node span."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        ctx = SynthesisTraceContext(
            trace=None,
            tracer=tracer,
            entity_type="hcp",
            prediction_target="churn",
        )
        ctx.start_node_span("combine")
        assert "combine" in ctx.active_spans
        ctx.end_node_span("combine", {"point_estimate": 0.75}, "completed")
        assert "combine" not in ctx.active_spans


# ============================================================================
# PREDICTION SYNTHESIZER OPIK TRACER TESTS
# ============================================================================


class TestPredictionSynthesizerOpikTracer:
    """Tests for PredictionSynthesizerOpikTracer class."""

    def test_init_defaults(self):
        """Test tracer initialization with defaults."""
        tracer = PredictionSynthesizerOpikTracer()
        assert tracer.project_name == "e2i-prediction-synthesizer"
        assert tracer.enabled is True
        assert tracer.sampling_rate == 1.0
        assert tracer._client is None

    def test_init_custom_params(self):
        """Test tracer initialization with custom parameters."""
        tracer = PredictionSynthesizerOpikTracer(
            project_name="custom-project",
            enabled=False,
            sampling_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False
        assert tracer.sampling_rate == 0.5

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        tracer1 = PredictionSynthesizerOpikTracer()
        tracer2 = PredictionSynthesizerOpikTracer()
        assert tracer1 is tracer2

    def test_singleton_skips_reinit(self):
        """Test that singleton skips reinitialization."""
        tracer1 = PredictionSynthesizerOpikTracer(project_name="first")
        tracer2 = PredictionSynthesizerOpikTracer(project_name="second")
        assert tracer1 is tracer2
        assert tracer1.project_name == "first"

    def test_get_client_disabled(self):
        """Test _get_client returns None when disabled."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        assert tracer._get_client() is None

    def test_should_sample_full_rate(self):
        """Test _should_sample at 100% rate."""
        tracer = PredictionSynthesizerOpikTracer(sampling_rate=1.0)
        assert tracer._should_sample() is True

    def test_should_sample_zero_rate(self):
        """Test _should_sample at 0% rate."""
        tracer = PredictionSynthesizerOpikTracer(sampling_rate=0.0)
        assert tracer._should_sample() is False

    def test_generate_trace_id_format(self):
        """Test trace ID generation format."""
        tracer = PredictionSynthesizerOpikTracer()
        trace_id = tracer._generate_trace_id()
        # Should be UUID format (with hyphens)
        assert len(trace_id) == 36
        assert trace_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_trace_synthesis_disabled(self):
        """Test trace_synthesis when disabled."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        async with tracer.trace_synthesis(
            entity_type="hcp",
            prediction_target="churn",
        ) as ctx:
            assert ctx is not None
            assert ctx.entity_type == "hcp"
            assert ctx.prediction_target == "churn"

    @pytest.mark.asyncio
    async def test_trace_synthesis_not_sampled(self):
        """Test trace_synthesis when not sampled."""
        tracer = PredictionSynthesizerOpikTracer(sampling_rate=0.0)
        async with tracer.trace_synthesis(
            entity_type="territory",
            prediction_target="conversion",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace is None

    @pytest.mark.asyncio
    async def test_trace_synthesis_full_pipeline(self):
        """Test full synthesis pipeline trace."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        async with tracer.trace_synthesis(
            entity_type="hcp",
            prediction_target="churn",
            ensemble_method="weighted",
            query="Predict churn for HCP-12345",
        ) as ctx:
            # Simulate full pipeline
            ctx.log_synthesis_started(
                entity_id="HCP-12345",
                entity_type="hcp",
                prediction_target="churn",
                time_horizon="30d",
                models_requested=3,
                ensemble_method="weighted",
                include_context=True,
            )

            node = ctx.start_node_span("orchestrate", {"models_requested": 3})
            ctx.end_node_span("orchestrate", {"models_succeeded": 3})
            ctx.log_model_orchestration(
                models_requested=3,
                models_succeeded=3,
                models_failed=0,
                orchestration_latency_ms=500,
            )

            ctx.start_node_span("combine")
            ctx.end_node_span("combine", {"point_estimate": 0.75})
            ctx.log_ensemble_combination(
                ensemble_method="weighted",
                point_estimate=0.75,
                prediction_interval_lower=0.65,
                prediction_interval_upper=0.85,
                confidence=0.85,
                model_agreement=0.92,
                ensemble_latency_ms=200,
            )

            ctx.start_node_span("enrich")
            ctx.end_node_span("enrich", {"similar_cases": 5})
            ctx.log_context_enrichment(
                similar_cases_found=5,
                feature_importance_calculated=True,
                historical_accuracy=0.82,
                trend_direction="increasing",
                enrichment_latency_ms=150,
            )

            ctx.log_synthesis_complete(
                status="completed",
                success=True,
                total_duration_ms=850,
                point_estimate=0.75,
                confidence=0.85,
                model_agreement=0.92,
                models_succeeded=3,
                models_failed=0,
                prediction_summary="High churn probability.",
                errors=[],
                warnings=[],
            )

        # Verify all spans were ended
        assert len(ctx.active_spans) == 0

    def test_flush_without_client(self):
        """Test flush when client is not initialized."""
        tracer = PredictionSynthesizerOpikTracer(enabled=False)
        # Should not raise
        tracer.flush()


# ============================================================================
# SINGLETON FUNCTIONS TESTS
# ============================================================================


class TestGetPredictionSynthesizerTracer:
    """Tests for get_prediction_synthesizer_tracer function."""

    def test_returns_tracer_instance(self):
        """Test that get_prediction_synthesizer_tracer returns a tracer."""
        tracer = get_prediction_synthesizer_tracer()
        assert isinstance(tracer, PredictionSynthesizerOpikTracer)

    def test_returns_same_instance(self):
        """Test singleton behavior."""
        tracer1 = get_prediction_synthesizer_tracer()
        tracer2 = get_prediction_synthesizer_tracer()
        assert tracer1 is tracer2

    def test_first_call_sets_config(self):
        """Test first call configures the tracer."""
        tracer = get_prediction_synthesizer_tracer(
            project_name="custom-project",
            sampling_rate=0.5,
        )
        assert tracer.project_name == "custom-project"
        assert tracer.sampling_rate == 0.5


class TestResetTracer:
    """Tests for reset_tracer function."""

    def test_reset_clears_singleton(self):
        """Test that reset_tracer clears the singleton."""
        tracer1 = get_prediction_synthesizer_tracer()
        reset_tracer()
        tracer2 = get_prediction_synthesizer_tracer()
        assert tracer1 is not tracer2


# ============================================================================
# INTEGRATION TESTS (WITH MOCKED OPIK)
# ============================================================================


class TestAgentOpikIntegration:
    """Tests for Opik integration with PredictionSynthesizerAgent."""

    def test_agent_enable_opik_default(self):
        """Test agent has enable_opik flag defaulting to True."""
        from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

        agent = PredictionSynthesizerAgent()
        assert agent.enable_opik is True

    def test_agent_enable_opik_disabled(self):
        """Test agent can be created with Opik disabled."""
        from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

        agent = PredictionSynthesizerAgent(enable_opik=False)
        assert agent.enable_opik is False

    def test_agent_all_integrations_enabled(self):
        """Test agent can have all integrations enabled."""
        from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

        agent = PredictionSynthesizerAgent(
            enable_memory=True,
            enable_dspy=True,
            enable_opik=True,
        )
        assert agent.enable_memory is True
        assert agent.enable_dspy is True
        assert agent.enable_opik is True

    def test_agent_all_integrations_disabled(self):
        """Test agent can have all integrations disabled."""
        from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

        agent = PredictionSynthesizerAgent(
            enable_memory=False,
            enable_dspy=False,
            enable_opik=False,
        )
        assert agent.enable_memory is False
        assert agent.enable_dspy is False
        assert agent.enable_opik is False

    def test_agent_tracer_property_disabled(self):
        """Test tracer property returns None when disabled."""
        from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

        agent = PredictionSynthesizerAgent(enable_opik=False)
        assert agent.tracer is None


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

        tracer = PredictionSynthesizerOpikTracer()
        tracer._client = mock_client

        async with tracer.trace_synthesis(
            entity_type="hcp",
            prediction_target="churn",
        ) as ctx:
            ctx.log_synthesis_started(
                entity_id="HCP-12345",
                entity_type="hcp",
                prediction_target="churn",
                time_horizon="30d",
                models_requested=3,
                ensemble_method="weighted",
                include_context=True,
            )
            ctx.log_model_orchestration(
                models_requested=3,
                models_succeeded=3,
                models_failed=0,
                orchestration_latency_ms=500,
            )
            ctx.log_ensemble_combination(
                ensemble_method="weighted",
                point_estimate=0.75,
                prediction_interval_lower=0.65,
                prediction_interval_upper=0.85,
                confidence=0.85,
                model_agreement=0.92,
                ensemble_latency_ms=200,
            )
            ctx.log_synthesis_complete(
                status="completed",
                success=True,
                total_duration_ms=700,
                point_estimate=0.75,
                confidence=0.85,
                model_agreement=0.92,
                models_succeeded=3,
                models_failed=0,
                prediction_summary="High churn probability.",
                errors=[],
                warnings=[],
            )

        mock_client.trace.assert_called_once()
        mock_trace.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_client_error(self):
        """Test that errors during tracing don't break execution."""
        tracer = PredictionSynthesizerOpikTracer()

        # Mock client that raises
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Opik error")
        tracer._client = mock_client

        # Should not raise
        async with tracer.trace_synthesis(
            entity_type="hcp",
            prediction_target="churn",
        ) as ctx:
            assert ctx is not None
            assert ctx.trace is None
            ctx.log_synthesis_complete(
                status="completed",
                success=True,
                total_duration_ms=500,
                point_estimate=0.75,
                confidence=0.85,
                model_agreement=0.92,
                models_succeeded=3,
                models_failed=0,
                prediction_summary="Prediction completed.",
                errors=[],
                warnings=[],
            )
