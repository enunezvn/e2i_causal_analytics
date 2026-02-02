"""
Tests for Discovery Observability Module
========================================

Tests for the DiscoveryTracer and related observability components.
"""

from unittest.mock import MagicMock, patch

import pytest
from uuid_utils import uuid7 as uuid7_func

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    GateDecision,
)
from src.causal_engine.discovery.observability import (
    DiscoverySpan,
    DiscoverySpanMetadata,
    DiscoveryTracer,
    get_discovery_tracer,
    reset_discovery_tracer,
)


class TestDiscoverySpanMetadata:
    """Tests for DiscoverySpanMetadata dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        metadata = DiscoverySpanMetadata()
        assert metadata.session_id is None
        assert metadata.algorithms == []
        assert metadata.n_variables == 0
        assert metadata.n_samples == 0
        assert metadata.config is None

    def test_initialization_with_values(self):
        """Test initialization with values."""
        session_id = uuid7_func()
        metadata = DiscoverySpanMetadata(
            session_id=session_id,
            algorithms=["ges", "pc"],
            n_variables=10,
            n_samples=1000,
            config={"threshold": 0.5},
        )
        assert metadata.session_id == session_id
        assert metadata.algorithms == ["ges", "pc"]
        assert metadata.n_variables == 10
        assert metadata.n_samples == 1000
        assert metadata.config == {"threshold": 0.5}

    def test_to_dict(self):
        """Test to_dict conversion."""
        session_id = uuid7_func()
        metadata = DiscoverySpanMetadata(
            session_id=session_id,
            algorithms=["ges", "pc"],
            n_variables=5,
            n_samples=500,
            config={"key": "value"},
        )
        result = metadata.to_dict()
        assert result["session_id"] == str(session_id)
        assert result["algorithms"] == ["ges", "pc"]
        assert result["n_variables"] == 5
        assert result["n_samples"] == 500
        assert result["config"] == {"key": "value"}

    def test_to_dict_with_none_session(self):
        """Test to_dict with None session_id."""
        metadata = DiscoverySpanMetadata()
        result = metadata.to_dict()
        assert result["session_id"] is None


class TestDiscoverySpan:
    """Tests for DiscoverySpan dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
        )
        assert span.span_id == "test-span-id"
        assert span.trace_id == "test-trace-id"
        assert span.name == "test_operation"
        assert span.end_time is None
        assert span.duration_ms is None
        assert span.algorithm_results == []
        assert span.gate_decision is None
        assert span.n_edges_discovered == 0

    def test_set_attribute_without_opik(self):
        """Test set_attribute without Opik span."""
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
        )
        # Should not raise even without _opik_span
        span.set_attribute("key", "value")

    def test_set_attribute_with_opik(self):
        """Test set_attribute with mocked Opik span."""
        mock_opik_span = MagicMock()
        mock_opik_span._metadata = {}
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
            _opik_span=mock_opik_span,
        )
        span.set_attribute("custom_key", "custom_value")
        mock_opik_span.update.assert_called_once()

    def test_add_event_without_opik(self):
        """Test add_event without Opik span."""
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
        )
        # Should not raise even without _opik_span
        span.add_event("test_event", {"key": "value"})

    def test_add_event_with_opik(self):
        """Test add_event with mocked Opik span."""
        mock_opik_span = MagicMock()
        mock_opik_span._metadata = {}
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
            _opik_span=mock_opik_span,
        )
        span.add_event("test_event", {"attr": "value"})
        mock_opik_span.update.assert_called_once()

    def test_to_dict(self):
        """Test to_dict conversion."""
        span = DiscoverySpan(
            span_id="test-span-id",
            trace_id="test-trace-id",
            name="test_operation",
            n_edges_discovered=5,
            algorithm_agreement=0.85,
            gate_decision="ACCEPT",
            gate_confidence=0.9,
        )
        result = span.to_dict()
        assert result["span_id"] == "test-span-id"
        assert result["trace_id"] == "test-trace-id"
        assert result["name"] == "test_operation"
        assert result["n_edges_discovered"] == 5
        assert result["algorithm_agreement"] == 0.85
        assert result["gate_decision"] == "ACCEPT"
        assert result["gate_confidence"] == 0.9


class TestDiscoveryTracer:
    """Tests for DiscoveryTracer class."""

    def setup_method(self):
        """Reset tracer singleton before each test."""
        reset_discovery_tracer()

    def test_initialization_disabled(self):
        """Test initialization with tracing disabled."""
        tracer = DiscoveryTracer(enabled=False)
        assert tracer._enabled is False
        assert tracer.is_enabled is False

    def test_initialization_without_opik(self):
        """Test initialization when Opik is not available."""
        with patch(
            "src.causal_engine.discovery.observability.DiscoveryTracer._init_opik"
        ) as mock_init:
            DiscoveryTracer(enabled=True)
            mock_init.assert_called_once()

    def test_is_enabled_property(self):
        """Test is_enabled property logic."""
        tracer = DiscoveryTracer(enabled=False)
        assert tracer.is_enabled is False

        tracer._enabled = True
        tracer._opik = None
        assert tracer.is_enabled is False

        mock_opik = MagicMock()
        mock_opik.is_enabled = True
        tracer._opik = mock_opik
        assert tracer.is_enabled is True

    @pytest.mark.asyncio
    async def test_trace_discovery_without_opik(self):
        """Test trace_discovery context manager without Opik."""
        tracer = DiscoveryTracer(enabled=False)
        session_id = uuid7_func()

        async with tracer.trace_discovery(
            session_id=session_id,
            algorithms=["ges", "pc"],
            n_variables=10,
            n_samples=1000,
        ) as span:
            assert span is not None
            assert span.metadata.session_id == session_id
            assert span.metadata.algorithms == ["ges", "pc"]
            span.n_edges_discovered = 5
            span.algorithm_agreement = 0.8

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_trace_discovery_captures_error(self):
        """Test that trace_discovery captures errors."""
        tracer = DiscoveryTracer(enabled=False)

        with pytest.raises(ValueError, match="Test error"):
            async with tracer.trace_discovery(
                algorithms=["ges"],
            ) as span:
                raise ValueError("Test error")

        assert span.error == "Test error"

    @pytest.mark.asyncio
    async def test_log_algorithm_result(self):
        """Test log_algorithm_result method."""
        tracer = DiscoveryTracer(enabled=False)

        span = DiscoverySpan(
            span_id="test-span",
            trace_id="test-trace",
            name="test",
        )

        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.GES,
            adjacency_matrix=MagicMock(),
            edge_list=[("A", "B"), ("B", "C")],
            runtime_seconds=1.5,
            converged=True,
            score=100.0,
        )

        await tracer.log_algorithm_result(span, result)

        assert len(span.algorithm_results) == 1
        assert span.algorithm_results[0]["algorithm"] == "ges"
        assert span.algorithm_results[0]["runtime_seconds"] == 1.5
        assert span.algorithm_results[0]["n_edges"] == 2
        assert span.algorithm_results[0]["converged"] is True

    @pytest.mark.asyncio
    async def test_log_gate_decision(self):
        """Test log_gate_decision method."""
        tracer = DiscoveryTracer(enabled=False)

        span = DiscoverySpan(
            span_id="test-span",
            trace_id="test-trace",
            name="test",
        )

        await tracer.log_gate_decision(
            parent_span=span,
            decision=GateDecision.ACCEPT,
            confidence=0.85,
            reasons=["High confidence", "Good agreement"],
        )

        # GateDecision enum uses lowercase values
        assert span.gate_decision == GateDecision.ACCEPT.value
        assert span.gate_confidence == 0.85
        assert span.gate_reasons == ["High confidence", "Good agreement"]

    @pytest.mark.asyncio
    async def test_log_cache_event(self):
        """Test log_cache_event method."""
        tracer = DiscoveryTracer(enabled=False)

        span = DiscoverySpan(
            span_id="test-span",
            trace_id="test-trace",
            name="test",
        )

        await tracer.log_cache_event(
            parent_span=span,
            hit=True,
            latency_ms=5.2,
        )

        assert span.cache_hit is True
        assert span.cache_latency_ms == 5.2

    @pytest.mark.asyncio
    async def test_log_ensemble_result(self):
        """Test log_ensemble_result method."""
        tracer = DiscoveryTracer(enabled=False)

        span = DiscoverySpan(
            span_id="test-span",
            trace_id="test-trace",
            name="test",
        )

        await tracer.log_ensemble_result(
            parent_span=span,
            n_edges=10,
            agreement=0.9,
            runtime_seconds=5.5,
        )

        assert span.n_edges_discovered == 10
        assert span.algorithm_agreement == 0.9

    def test_get_status(self):
        """Test get_status method."""
        tracer = DiscoveryTracer(enabled=True, project_name="test-project")
        tracer._initialized = True

        status = tracer.get_status()

        assert status["enabled"] is True
        assert status["initialized"] is True
        assert status["project_name"] == "test-project"


class TestDiscoveryTracerSingleton:
    """Tests for DiscoveryTracer singleton functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_discovery_tracer()

    def test_get_discovery_tracer_creates_singleton(self):
        """Test that get_discovery_tracer creates a singleton."""
        with patch("src.causal_engine.discovery.observability.DiscoveryTracer._init_opik"):
            tracer1 = get_discovery_tracer()
            tracer2 = get_discovery_tracer()
            assert tracer1 is tracer2

    def test_get_discovery_tracer_force_new(self):
        """Test that force_new creates a new instance."""
        with patch("src.causal_engine.discovery.observability.DiscoveryTracer._init_opik"):
            tracer1 = get_discovery_tracer()
            tracer2 = get_discovery_tracer(force_new=True)
            assert tracer1 is not tracer2

    def test_reset_discovery_tracer(self):
        """Test reset_discovery_tracer clears singleton."""
        with patch("src.causal_engine.discovery.observability.DiscoveryTracer._init_opik"):
            tracer1 = get_discovery_tracer()
            reset_discovery_tracer()
            tracer2 = get_discovery_tracer()
            assert tracer1 is not tracer2
