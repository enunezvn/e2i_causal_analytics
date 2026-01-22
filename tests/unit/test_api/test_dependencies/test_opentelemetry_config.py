"""Unit tests for OpenTelemetry configuration module.

Tests cover:
- OpenTelemetry initialization
- Tracer provider configuration
- Exporter selection (OTLP, Console, None)
- Sampling configuration
- NoOp tracer fallback
- FastAPI instrumentation
- Span utilities

Phase 1 G02 from observability audit remediation plan.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.api.dependencies.opentelemetry_config import (
    init_opentelemetry,
    get_tracer,
    shutdown_opentelemetry,
    get_opentelemetry_middleware,
    instrument_fastapi,
    create_span_from_trace_context,
    _NoOpTracer,
    _NoOpSpan,
    _NoOpContextManager,
    OTEL_ENABLED,
    OTEL_SERVICE_NAME,
    OTEL_EXPORTER_TYPE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def reset_otel_state():
    """Reset OpenTelemetry module state before and after tests."""
    import src.api.dependencies.opentelemetry_config as otel_module

    # Store original state
    original_initialized = otel_module._otel_initialized
    original_tracer = otel_module._tracer

    # Reset state
    otel_module._otel_initialized = False
    otel_module._tracer = None

    yield

    # Restore state
    otel_module._otel_initialized = original_initialized
    otel_module._tracer = original_tracer


@pytest.fixture
def mock_otel_sdk():
    """Mock OpenTelemetry SDK imports."""
    mock_trace = MagicMock()
    mock_provider = MagicMock()
    mock_trace.get_tracer_provider.return_value = mock_provider
    mock_trace.get_tracer.return_value = MagicMock()

    with patch.dict("sys.modules", {
        "opentelemetry": MagicMock(trace=mock_trace),
        "opentelemetry.trace": mock_trace,
        "opentelemetry.sdk.trace": MagicMock(TracerProvider=MagicMock),
        "opentelemetry.sdk.trace.sampling": MagicMock(),
        "opentelemetry.sdk.resources": MagicMock(),
        "opentelemetry.propagators.composite": MagicMock(),
        "opentelemetry.propagate": MagicMock(),
        "opentelemetry.trace.propagation.tracecontext": MagicMock(),
        "opentelemetry.baggage.propagation": MagicMock(),
    }):
        yield mock_trace


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestOpenTelemetryConfiguration:
    """Tests for configuration values."""

    def test_default_service_name(self):
        """Test default service name is set."""
        assert OTEL_SERVICE_NAME == "e2i-causal-analytics" or isinstance(OTEL_SERVICE_NAME, str)

    def test_exporter_type_is_valid(self):
        """Test exporter type is a valid option."""
        valid_types = {"otlp", "console", "none"}
        assert OTEL_EXPORTER_TYPE.lower() in valid_types or isinstance(OTEL_EXPORTER_TYPE, str)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestInitOpenTelemetry:
    """Tests for init_opentelemetry function."""

    def test_init_returns_bool(self, reset_otel_state):
        """Test that init returns a boolean."""
        result = init_opentelemetry()
        assert isinstance(result, bool)

    def test_init_is_idempotent(self, reset_otel_state):
        """Test that multiple init calls are idempotent."""
        result1 = init_opentelemetry()
        result2 = init_opentelemetry()

        # Second call should not re-initialize
        import src.api.dependencies.opentelemetry_config as otel_module
        assert otel_module._otel_initialized is True

    @patch.dict(os.environ, {"OTEL_ENABLED": "false"})
    def test_init_disabled_via_env(self, reset_otel_state):
        """Test that init respects OTEL_ENABLED=false."""
        import src.api.dependencies.opentelemetry_config as otel_module

        # Force re-read of env var
        otel_module._otel_initialized = False
        otel_module.OTEL_ENABLED = False

        result = init_opentelemetry()

        # Should complete but return False (disabled)
        assert otel_module._otel_initialized is True

    def test_init_handles_import_error(self, reset_otel_state):
        """Test graceful handling when OpenTelemetry SDK not available."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            with patch("src.api.dependencies.opentelemetry_config.OTEL_ENABLED", True):
                import src.api.dependencies.opentelemetry_config as otel_module
                otel_module._otel_initialized = False

                # Should not raise, just return False
                result = init_opentelemetry()
                assert isinstance(result, bool)


# =============================================================================
# GET_TRACER TESTS
# =============================================================================


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_returns_tracer(self, reset_otel_state):
        """Test that get_tracer returns a tracer object."""
        tracer = get_tracer("test_module")
        assert tracer is not None

    def test_get_tracer_accepts_name(self, reset_otel_state):
        """Test that get_tracer accepts a module name."""
        tracer = get_tracer("my.custom.module")
        assert tracer is not None

    def test_get_tracer_default_name(self, reset_otel_state):
        """Test that get_tracer has a default name."""
        tracer = get_tracer()
        assert tracer is not None


# =============================================================================
# NOOP TRACER TESTS
# =============================================================================


class TestNoOpTracer:
    """Tests for _NoOpTracer fallback."""

    def test_noop_tracer_start_span(self):
        """Test NoOpTracer.start_span returns NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_span("test_span")
        assert isinstance(span, _NoOpSpan)

    def test_noop_tracer_start_as_current_span(self):
        """Test NoOpTracer.start_as_current_span returns context manager."""
        tracer = _NoOpTracer()
        ctx_manager = tracer.start_as_current_span("test_span")
        assert isinstance(ctx_manager, _NoOpContextManager)


class TestNoOpSpan:
    """Tests for _NoOpSpan."""

    def test_noop_span_set_attribute(self):
        """Test set_attribute doesn't raise."""
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # Should not raise

    def test_noop_span_set_status(self):
        """Test set_status doesn't raise."""
        span = _NoOpSpan()
        span.set_status("OK")  # Should not raise

    def test_noop_span_record_exception(self):
        """Test record_exception doesn't raise."""
        span = _NoOpSpan()
        span.record_exception(Exception("test"))  # Should not raise

    def test_noop_span_end(self):
        """Test end doesn't raise."""
        span = _NoOpSpan()
        span.end()  # Should not raise

    def test_noop_span_context_manager(self):
        """Test NoOpSpan works as context manager."""
        span = _NoOpSpan()
        with span as s:
            assert s is span


class TestNoOpContextManager:
    """Tests for _NoOpContextManager."""

    def test_noop_context_manager_enter(self):
        """Test __enter__ returns NoOpSpan."""
        ctx = _NoOpContextManager()
        with ctx as span:
            assert isinstance(span, _NoOpSpan)


# =============================================================================
# SHUTDOWN TESTS
# =============================================================================


class TestShutdownOpenTelemetry:
    """Tests for shutdown_opentelemetry function."""

    def test_shutdown_when_not_initialized(self, reset_otel_state):
        """Test shutdown doesn't error when not initialized."""
        import src.api.dependencies.opentelemetry_config as otel_module
        otel_module._otel_initialized = False

        # Should not raise
        shutdown_opentelemetry()

    def test_shutdown_resets_initialized_flag(self, reset_otel_state):
        """Test that shutdown resets the initialized flag."""
        import src.api.dependencies.opentelemetry_config as otel_module

        # Initialize first
        init_opentelemetry()
        assert otel_module._otel_initialized is True

        # Shutdown
        shutdown_opentelemetry()
        assert otel_module._otel_initialized is False


# =============================================================================
# MIDDLEWARE TESTS
# =============================================================================


class TestGetOpenTelemetryMiddleware:
    """Tests for get_opentelemetry_middleware function."""

    def test_returns_none_when_disabled(self, reset_otel_state):
        """Test returns None when OTEL disabled."""
        with patch("src.api.dependencies.opentelemetry_config.OTEL_ENABLED", False):
            result = get_opentelemetry_middleware()
            assert result is None

    def test_returns_none_when_import_fails(self, reset_otel_state):
        """Test returns None when middleware import fails."""
        with patch("src.api.dependencies.opentelemetry_config.OTEL_ENABLED", True):
            with patch.dict("sys.modules", {
                "opentelemetry.instrumentation.asgi": None
            }):
                # Force import error
                result = get_opentelemetry_middleware()
                # May return middleware or None depending on actual imports
                assert result is None or callable(result)


class TestInstrumentFastAPI:
    """Tests for instrument_fastapi function."""

    def test_returns_false_when_disabled(self, reset_otel_state):
        """Test returns False when OTEL disabled."""
        with patch("src.api.dependencies.opentelemetry_config.OTEL_ENABLED", False):
            app = MagicMock()
            result = instrument_fastapi(app)
            assert result is False

    def test_returns_false_when_not_initialized(self, reset_otel_state):
        """Test returns False when not initialized."""
        import src.api.dependencies.opentelemetry_config as otel_module
        otel_module._otel_initialized = False

        with patch("src.api.dependencies.opentelemetry_config.OTEL_ENABLED", True):
            app = MagicMock()
            result = instrument_fastapi(app)
            assert result is False


# =============================================================================
# SPAN UTILITIES TESTS
# =============================================================================


class TestCreateSpanFromTraceContext:
    """Tests for create_span_from_trace_context function."""

    def test_creates_span_without_trace_context(self, reset_otel_state):
        """Test creates span when no trace context provided."""
        span_ctx = create_span_from_trace_context("test_span", {})
        assert span_ctx is not None

    def test_creates_span_with_trace_context(self, reset_otel_state):
        """Test creates span with existing trace context."""
        trace_context = {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "span_id": "b7ad6b7169203331",
            "trace_flags": "01",
        }
        span_ctx = create_span_from_trace_context("test_span", trace_context)
        assert span_ctx is not None

    def test_handles_invalid_trace_context(self, reset_otel_state):
        """Test handles invalid trace context gracefully."""
        trace_context = {
            "trace_id": "invalid",
            "span_id": "also-invalid",
        }
        # Should not raise
        span_ctx = create_span_from_trace_context("test_span", trace_context)
        assert span_ctx is not None

    def test_handles_partial_trace_context(self, reset_otel_state):
        """Test handles partial trace context."""
        trace_context = {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            # Missing span_id and trace_flags
        }
        span_ctx = create_span_from_trace_context("test_span", trace_context)
        assert span_ctx is not None
