"""Unit tests for trace header extraction middleware.

Tests cover:
- Request ID generation and extraction
- W3C Trace Context (traceparent) parsing
- Zipkin B3 header support
- Context variable management
- Response header propagation
- TraceContext dataclass

QW5 from observability audit remediation plan.
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.tracing import (
    TracingMiddleware,
    TraceContext,
    get_request_id,
    get_correlation_id,
    get_trace_id,
    get_span_id,
    get_trace_context,
    with_trace_context,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_logging_context():
    """Mock the logging context functions."""
    with patch("src.api.middleware.tracing.set_logging_context") as mock_set, \
         patch("src.api.middleware.tracing.clear_logging_context") as mock_clear:
        yield mock_set, mock_clear


def create_test_app(middleware_kwargs=None):
    """Create a test Starlette app with TracingMiddleware."""
    middleware_kwargs = middleware_kwargs or {}

    async def homepage(request):
        # Return the request ID from state
        request_id = getattr(request.state, "request_id", "unknown")
        return Response(request_id, media_type="text/plain")

    async def trace_info(request):
        # Return trace context info
        trace_ctx = getattr(request.state, "trace_context", None)
        if trace_ctx:
            return Response(
                f"request_id={trace_ctx.request_id},"
                f"trace_id={trace_ctx.trace_id},"
                f"span_id={trace_ctx.span_id}",
                media_type="text/plain"
            )
        return Response("no trace context", media_type="text/plain")

    app = Starlette(
        routes=[
            Route("/", homepage),
            Route("/trace-info", trace_info),
        ]
    )
    app.add_middleware(TracingMiddleware, **middleware_kwargs)
    return app


@pytest.fixture
def client(mock_logging_context):
    """Create a test client with mocked logging context."""
    app = create_test_app()
    return TestClient(app)


# =============================================================================
# TRACE CONTEXT DATACLASS TESTS
# =============================================================================


class TestTraceContextDataclass:
    """Tests for TraceContext dataclass."""

    def test_from_headers_generates_request_id(self):
        """Test that request ID is generated when not provided."""
        ctx = TraceContext.from_headers({})
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0

    def test_from_headers_extracts_request_id(self):
        """Test extraction of X-Request-ID header."""
        ctx = TraceContext.from_headers({"x-request-id": "my-request-123"})
        assert ctx.request_id == "my-request-123"

    def test_from_headers_extracts_correlation_id(self):
        """Test extraction of X-Correlation-ID header."""
        ctx = TraceContext.from_headers({
            "x-request-id": "req-123",
            "x-correlation-id": "corr-456"
        })
        assert ctx.request_id == "req-123"
        assert ctx.correlation_id == "corr-456"

    def test_from_headers_correlation_defaults_to_request_id(self):
        """Test that correlation ID defaults to request ID."""
        ctx = TraceContext.from_headers({"x-request-id": "req-123"})
        assert ctx.correlation_id == "req-123"

    def test_from_headers_parses_w3c_traceparent(self):
        """Test parsing of W3C Trace Context traceparent header."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = TraceContext.from_headers({"traceparent": traceparent})

        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.parent_id == "b7ad6b7169203331"
        assert ctx.trace_flags == "01"
        assert ctx.span_id is not None  # Generated for this request

    def test_from_headers_parses_zipkin_b3(self):
        """Test parsing of Zipkin B3 headers."""
        ctx = TraceContext.from_headers({
            "x-b3-traceid": "80f198ee56343ba864fe8b2a57d3eff7",
            "x-b3-spanid": "e457b5a2e4d86bd1",
            "x-b3-parentspanid": "05e3ac9a4f6e3b90",
            "x-b3-sampled": "1",
        })

        assert ctx.trace_id == "80f198ee56343ba864fe8b2a57d3eff7"
        assert ctx.span_id == "e457b5a2e4d86bd1"
        assert ctx.parent_id == "05e3ac9a4f6e3b90"
        assert ctx.trace_flags == "01"  # Sampled

    def test_from_headers_parses_generic_trace_id(self):
        """Test parsing of generic X-Trace-ID header."""
        ctx = TraceContext.from_headers({"x-trace-id": "my-trace-12345"})
        assert ctx.trace_id == "my-trace-12345"

    def test_from_headers_extracts_amzn_trace_id(self):
        """Test extraction of X-Amzn-Trace-Id header (AWS)."""
        ctx = TraceContext.from_headers({
            "x-amzn-trace-id": "Root=1-5759e988-bd862e3fe1be46a994272793"
        })
        assert ctx.request_id == "Root=1-5759e988-bd862e3fe1be46a994272793"


class TestTraceContextInvalidFormats:
    """Tests for handling invalid trace header formats."""

    def test_invalid_traceparent_format_ignored(self):
        """Test that invalid traceparent format is ignored."""
        ctx = TraceContext.from_headers({"traceparent": "invalid-format"})
        assert ctx.trace_id is None

    def test_malformed_traceparent_ignored(self):
        """Test that malformed traceparent is ignored."""
        # Missing parts
        ctx = TraceContext.from_headers({"traceparent": "00-abcdef"})
        assert ctx.trace_id is None


# =============================================================================
# TRACING MIDDLEWARE TESTS
# =============================================================================


class TestTracingMiddlewareBasic:
    """Basic functionality tests for TracingMiddleware."""

    def test_middleware_generates_request_id(self, client):
        """Test that middleware generates request ID when not provided."""
        response = client.get("/")
        # Body contains the request ID from request.state
        request_id = response.text
        assert len(request_id) > 0
        # Should be a UUID-like format
        assert "-" in request_id or len(request_id) == 36

    def test_middleware_uses_provided_request_id(self, client):
        """Test that middleware uses provided X-Request-ID."""
        response = client.get("/", headers={"X-Request-ID": "custom-req-123"})
        assert response.text == "custom-req-123"

    def test_middleware_adds_request_id_to_response(self, client):
        """Test that response includes X-Request-ID header."""
        response = client.get("/")
        assert "X-Request-ID" in response.headers

    def test_middleware_adds_correlation_id_to_response(self, client):
        """Test that response includes X-Correlation-ID header."""
        response = client.get("/", headers={"X-Correlation-ID": "corr-456"})
        assert response.headers.get("X-Correlation-ID") == "corr-456"


class TestTracingMiddlewareResponseHeaders:
    """Tests for response header propagation."""

    def test_response_includes_traceparent(self, mock_logging_context):
        """Test that response includes traceparent when trace context exists."""
        app = create_test_app()
        client = TestClient(app)

        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        response = client.get("/", headers={"traceparent": traceparent})

        assert "traceparent" in response.headers
        # Should contain the same trace_id
        assert "0af7651916cd43dd8448eb211c80319c" in response.headers["traceparent"]

    def test_response_headers_disabled(self, mock_logging_context):
        """Test that response headers can be disabled."""
        app = create_test_app(middleware_kwargs={"add_response_headers": False})
        client = TestClient(app)

        response = client.get("/", headers={"X-Request-ID": "req-123"})

        # Should NOT include trace headers
        assert "X-Request-ID" not in response.headers


class TestTracingMiddlewareLogging:
    """Tests for trace context logging."""

    def test_sets_logging_context(self, mock_logging_context):
        """Test that middleware sets logging context."""
        mock_set, mock_clear = mock_logging_context
        app = create_test_app()
        client = TestClient(app)

        client.get("/", headers={"X-Request-ID": "req-123"})

        mock_set.assert_called()
        call_kwargs = mock_set.call_args.kwargs
        assert call_kwargs.get("request_id") == "req-123"

    def test_clears_logging_context(self, mock_logging_context):
        """Test that middleware clears logging context after request."""
        mock_set, mock_clear = mock_logging_context
        app = create_test_app()
        client = TestClient(app)

        client.get("/")

        mock_clear.assert_called()

    def test_log_trace_context_option(self, mock_logging_context):
        """Test log_trace_context option."""
        with patch("src.api.middleware.tracing.logger") as mock_logger:
            app = create_test_app(middleware_kwargs={"log_trace_context": True})
            client = TestClient(app)

            client.get("/", headers={"X-Request-ID": "req-123"})

            mock_logger.debug.assert_called()


# =============================================================================
# CONTEXT VARIABLE ACCESSOR TESTS
# =============================================================================


class TestContextVariableAccessors:
    """Tests for context variable accessor functions."""

    def test_get_request_id_returns_empty_when_not_set(self):
        """Test get_request_id returns empty string when not set."""
        # Outside of request context
        result = get_request_id()
        # Default is empty string
        assert result == ""

    def test_get_correlation_id_returns_empty_when_not_set(self):
        """Test get_correlation_id returns empty string when not set."""
        result = get_correlation_id()
        assert result == ""

    def test_get_trace_id_returns_empty_when_not_set(self):
        """Test get_trace_id returns empty string when not set."""
        result = get_trace_id()
        assert result == ""

    def test_get_span_id_returns_empty_when_not_set(self):
        """Test get_span_id returns empty string when not set."""
        result = get_span_id()
        assert result == ""

    def test_get_trace_context_returns_context(self):
        """Test get_trace_context returns TraceContext."""
        ctx = get_trace_context()
        assert isinstance(ctx, TraceContext)


# =============================================================================
# WITH_TRACE_CONTEXT UTILITY TESTS
# =============================================================================


class TestWithTraceContext:
    """Tests for with_trace_context utility function."""

    def test_with_trace_context_adds_request_id(self):
        """Test that request_id is added to extra dict."""
        result = with_trace_context()
        assert "request_id" in result

    def test_with_trace_context_adds_correlation_id(self):
        """Test that correlation_id is added to extra dict."""
        result = with_trace_context()
        assert "correlation_id" in result

    def test_with_trace_context_preserves_existing_extra(self):
        """Test that existing extra dict values are preserved."""
        existing = {"custom_field": "value", "another": 123}
        result = with_trace_context(existing)

        assert result["custom_field"] == "value"
        assert result["another"] == 123
        assert "request_id" in result

    def test_with_trace_context_handles_none_extra(self):
        """Test that None extra is handled."""
        result = with_trace_context(None)
        assert isinstance(result, dict)
        assert "request_id" in result
