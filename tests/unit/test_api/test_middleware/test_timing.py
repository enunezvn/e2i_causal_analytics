"""Unit tests for request timing middleware.

Tests cover:
- Request timing measurement
- Server-Timing header addition
- Slow request logging
- Path exclusion logic
- Path normalization for metrics
- Error handling

QW3 from observability audit remediation plan.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.timing import (
    RequestTimingContext,
    TimingMiddleware,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_record_request():
    """Mock the record_request function."""
    with patch("src.api.middleware.timing.record_request") as mock:
        yield mock


@pytest.fixture
def mock_record_error():
    """Mock the record_error function."""
    with patch("src.api.middleware.timing.record_error") as mock:
        yield mock


def create_test_app(middleware_kwargs=None):
    """Create a test Starlette app with TimingMiddleware."""
    middleware_kwargs = middleware_kwargs or {}

    async def homepage(request):
        return Response("OK", media_type="text/plain")

    async def slow_endpoint(request):
        time.sleep(0.1)  # 100ms delay
        return Response("Slow", media_type="text/plain")

    async def health_endpoint(request):
        return Response("Healthy", media_type="text/plain")

    async def error_endpoint(request):
        return Response("Not Found", status_code=404, media_type="text/plain")

    async def server_error_endpoint(request):
        return Response("Server Error", status_code=500, media_type="text/plain")

    app = Starlette(
        routes=[
            Route("/", homepage),
            Route("/slow", slow_endpoint),
            Route("/health", health_endpoint),
            Route("/error", error_endpoint),
            Route("/server-error", server_error_endpoint),
            Route("/users/123/profile", homepage),
            Route("/orders/550e8400-e29b-41d4-a716-446655440000", homepage),
        ]
    )
    app.add_middleware(TimingMiddleware, **middleware_kwargs)
    return app


@pytest.fixture
def client(mock_record_request, mock_record_error):
    """Create a test client with mocked metrics."""
    app = create_test_app()
    return TestClient(app)


# =============================================================================
# TIMING MIDDLEWARE TESTS
# =============================================================================


class TestTimingMiddlewareBasic:
    """Basic functionality tests for TimingMiddleware."""

    def test_middleware_adds_server_timing_header(self, client):
        """Test that Server-Timing header is added to responses."""
        response = client.get("/")
        assert "Server-Timing" in response.headers
        # Should contain "total;dur=" pattern
        assert "total;dur=" in response.headers["Server-Timing"]

    def test_middleware_records_request_metrics(self, client, mock_record_request):
        """Test that request metrics are recorded."""
        client.get("/")
        mock_record_request.assert_called()

    def test_middleware_records_correct_method(self, client, mock_record_request):
        """Test that HTTP method is recorded correctly."""
        client.post("/")
        call_args = mock_record_request.call_args
        assert call_args.kwargs["method"] == "POST"

    def test_middleware_records_latency(self, client, mock_record_request):
        """Test that latency is recorded."""
        client.get("/")
        call_args = mock_record_request.call_args
        assert "latency" in call_args.kwargs
        assert call_args.kwargs["latency"] > 0


class TestPathExclusion:
    """Tests for path exclusion logic."""

    def test_health_endpoint_excluded_by_default(self, mock_record_request, mock_record_error):
        """Test that /health is excluded from metrics by default."""
        app = create_test_app()
        client = TestClient(app)
        client.get("/health")
        mock_record_request.assert_not_called()

    def test_custom_exclude_paths(self, mock_record_request, mock_record_error):
        """Test custom path exclusion."""
        # Note: Don't use "/" as exclude path - it would exclude ALL paths
        # since _should_track uses startswith() matching
        app = create_test_app(middleware_kwargs={"exclude_paths": ["/slow", "/error"]})
        client = TestClient(app)

        # Excluded path
        client.get("/slow")
        mock_record_request.assert_not_called()

        # Non-excluded path (health is not in custom list)
        client.get("/health")
        mock_record_request.assert_called()

    def test_should_track_method(self):
        """Test _should_track method directly."""
        middleware = TimingMiddleware(app=MagicMock(), exclude_paths=["/health", "/metrics"])

        assert middleware._should_track("/api/users") is True
        assert middleware._should_track("/health") is False
        assert middleware._should_track("/healthz") is False  # starts with /health
        assert middleware._should_track("/metrics") is False
        assert middleware._should_track("/metrics/health") is False


class TestPathNormalization:
    """Tests for path normalization to prevent metric cardinality explosion."""

    def test_uuid_normalization(self):
        """Test that UUIDs are normalized."""
        middleware = TimingMiddleware(app=MagicMock())

        path = "/orders/550e8400-e29b-41d4-a716-446655440000"
        normalized = middleware._normalize_path(path)
        assert "{uuid}" in normalized
        assert "550e8400" not in normalized

    def test_numeric_id_normalization(self):
        """Test that numeric IDs are normalized."""
        middleware = TimingMiddleware(app=MagicMock())

        path = "/users/12345/profile"
        normalized = middleware._normalize_path(path)
        assert "{id}" in normalized
        assert "12345" not in normalized

    def test_date_normalization(self):
        """Test that dates are normalized."""
        middleware = TimingMiddleware(app=MagicMock())

        path = "/reports/2024-01-15/summary"
        normalized = middleware._normalize_path(path)
        assert "{date}" in normalized
        assert "2024-01-15" not in normalized

    def test_multiple_normalizations(self):
        """Test that multiple patterns are normalized."""
        middleware = TimingMiddleware(app=MagicMock())

        path = "/users/123/orders/550e8400-e29b-41d4-a716-446655440000/items/456"
        normalized = middleware._normalize_path(path)
        assert normalized == "/users/{id}/orders/{uuid}/items/{id}"


class TestSlowRequestLogging:
    """Tests for slow request logging."""

    def test_slow_request_logs_warning(self, mock_record_request, mock_record_error):
        """Test that slow requests are logged."""
        with patch("src.api.middleware.timing.logger") as mock_logger:
            # Set a very low threshold
            app = create_test_app(middleware_kwargs={"slow_threshold_ms": 1.0})
            client = TestClient(app)

            # Make a request that takes more than 1ms
            client.get("/slow")

            # Should have logged a warning
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "Slow request" in str(call_args)


class TestErrorTracking:
    """Tests for error tracking in timing middleware."""

    def test_records_client_error(self, mock_record_request, mock_record_error):
        """Test that client errors (4xx) are recorded."""
        app = create_test_app()
        client = TestClient(app)

        client.get("/error")

        # Should record the error
        mock_record_error.assert_called()
        call_args = mock_record_error.call_args
        assert call_args.kwargs["error_type"] == "not_found"

    def test_records_server_error(self, mock_record_request, mock_record_error):
        """Test that server errors (5xx) are recorded."""
        app = create_test_app()
        client = TestClient(app)

        client.get("/server-error")

        mock_record_error.assert_called()
        call_args = mock_record_error.call_args
        assert call_args.kwargs["error_type"] == "server_error"


class TestServerTimingDisabled:
    """Tests for disabling Server-Timing header."""

    def test_server_timing_can_be_disabled(self, mock_record_request, mock_record_error):
        """Test that Server-Timing header can be disabled."""
        app = create_test_app(middleware_kwargs={"add_server_timing": False})
        client = TestClient(app)

        response = client.get("/")

        assert "Server-Timing" not in response.headers


# =============================================================================
# REQUEST TIMING CONTEXT TESTS
# =============================================================================


class TestRequestTimingContext:
    """Tests for RequestTimingContext utility class."""

    def test_context_tracks_elapsed_time(self):
        """Test that context tracks elapsed time."""
        ctx = RequestTimingContext()
        time.sleep(0.01)  # 10ms
        elapsed = ctx.elapsed_ms
        assert elapsed >= 10

    def test_context_checkpoint(self):
        """Test checkpoint recording."""
        ctx = RequestTimingContext()
        time.sleep(0.005)  # 5ms

        checkpoint1 = ctx.checkpoint("db_query")
        assert checkpoint1 >= 5

        time.sleep(0.005)
        checkpoint2 = ctx.checkpoint("api_call")
        assert checkpoint2 > checkpoint1

        assert "db_query" in ctx.checkpoints
        assert "api_call" in ctx.checkpoints

    def test_context_to_server_timing(self):
        """Test Server-Timing header generation."""
        ctx = RequestTimingContext()
        ctx.checkpoint("db")
        ctx.checkpoint("render")

        header = ctx.to_server_timing()

        assert "db;dur=" in header
        assert "render;dur=" in header
        assert "total;dur=" in header

    def test_empty_context_to_server_timing(self):
        """Test Server-Timing with no checkpoints."""
        ctx = RequestTimingContext()
        header = ctx.to_server_timing()

        # Should still have total
        assert "total;dur=" in header
