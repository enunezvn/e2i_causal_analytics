"""Unit tests for Prometheus metrics endpoint.

Tests cover:
- Metrics endpoint returns valid Prometheus format
- Metric recording functions work correctly
- Health endpoint returns proper status
- Graceful degradation when prometheus_client unavailable

QW1 from observability audit remediation plan.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create a FastAPI app with metrics router."""
    from src.api.routes.metrics import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def reset_metrics_state():
    """Reset metrics module state before and after tests."""
    import src.api.routes.metrics as metrics_module

    # Store original state
    original_initialized = metrics_module._metrics_initialized
    original_registry = metrics_module._metrics_registry

    # Reset state
    metrics_module._metrics_initialized = False
    metrics_module._metrics_registry = None

    yield

    # Restore state
    metrics_module._metrics_initialized = original_initialized
    metrics_module._metrics_registry = original_registry


# =============================================================================
# METRICS ENDPOINT TESTS
# =============================================================================


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_endpoint_returns_200(self, client, reset_metrics_state):
        """Test that /metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self, client, reset_metrics_state):
        """Test that /metrics returns Prometheus text format."""
        response = client.get("/metrics")
        # Prometheus format includes HELP and TYPE lines
        content = response.text
        # Should at least have some metric definitions or empty comment
        assert "#" in content or "e2i_" in content or "not installed" in content.lower()

    def test_metrics_endpoint_content_type(self, client, reset_metrics_state):
        """Test that /metrics has correct content type."""
        response = client.get("/metrics")
        # Should be text/plain or Prometheus MIME type
        assert "text/" in response.headers.get("content-type", "")

    @patch("src.api.routes.metrics.PROMETHEUS_AVAILABLE", False)
    def test_metrics_endpoint_graceful_degradation(self, reset_metrics_state):
        """Test graceful degradation when prometheus_client unavailable."""
        from src.api.routes.metrics import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "not installed" in response.text.lower()


class TestMetricsHealthEndpoint:
    """Tests for /metrics/health endpoint."""

    def test_metrics_health_returns_json(self, client, reset_metrics_state):
        """Test that /metrics/health returns JSON."""
        response = client.get("/metrics/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "prometheus_available" in data
        assert "metrics_initialized" in data

    def test_metrics_health_shows_prometheus_status(self, client, reset_metrics_state):
        """Test that health shows prometheus availability."""
        response = client.get("/metrics/health")
        data = response.json()
        # prometheus_available should be a boolean
        assert isinstance(data["prometheus_available"], bool)


# =============================================================================
# METRIC RECORDING FUNCTION TESTS
# =============================================================================


class TestRecordRequest:
    """Tests for record_request function."""

    def test_record_request_no_error_when_not_initialized(self, reset_metrics_state):
        """Test record_request doesn't error when metrics not initialized."""
        from src.api.routes.metrics import record_request

        # Should not raise any exception
        record_request(method="GET", endpoint="/test", status_code=200, latency=0.1)

    def test_record_request_after_initialization(self, reset_metrics_state):
        """Test record_request works after initialization."""
        from src.api.routes.metrics import get_metrics_registry, record_request

        # Initialize metrics
        get_metrics_registry()

        # Should not raise
        record_request(method="GET", endpoint="/test", status_code=200, latency=0.1)

    def test_record_request_handles_various_status_codes(self, reset_metrics_state):
        """Test record_request handles various HTTP status codes."""
        from src.api.routes.metrics import get_metrics_registry, record_request

        get_metrics_registry()

        # Should handle all status code types
        record_request(method="GET", endpoint="/test", status_code=200, latency=0.1)
        record_request(method="POST", endpoint="/test", status_code=201, latency=0.2)
        record_request(method="GET", endpoint="/test", status_code=404, latency=0.05)
        record_request(method="POST", endpoint="/test", status_code=500, latency=0.3)


class TestRecordError:
    """Tests for record_error function."""

    def test_record_error_no_error_when_not_initialized(self, reset_metrics_state):
        """Test record_error doesn't error when metrics not initialized."""
        from src.api.routes.metrics import record_error

        # Should not raise any exception
        record_error(method="GET", endpoint="/test", error_type="server_error")

    def test_record_error_after_initialization(self, reset_metrics_state):
        """Test record_error works after initialization."""
        from src.api.routes.metrics import get_metrics_registry, record_error

        get_metrics_registry()

        # Should not raise
        record_error(method="GET", endpoint="/test", error_type="server_error")

    def test_record_error_various_error_types(self, reset_metrics_state):
        """Test record_error handles various error types."""
        from src.api.routes.metrics import get_metrics_registry, record_error

        get_metrics_registry()

        # Should handle all error types
        record_error(method="GET", endpoint="/test", error_type="client_error")
        record_error(method="GET", endpoint="/test", error_type="server_error")
        record_error(method="GET", endpoint="/test", error_type="auth_error")
        record_error(method="GET", endpoint="/test", error_type="not_found")


class TestRecordAgentInvocation:
    """Tests for record_agent_invocation function."""

    def test_record_agent_invocation_no_error_when_not_initialized(self, reset_metrics_state):
        """Test record_agent_invocation doesn't error when not initialized."""
        from src.api.routes.metrics import record_agent_invocation

        # Should not raise
        record_agent_invocation(agent_name="causal_impact", tier="2", status="success")

    def test_record_agent_invocation_after_initialization(self, reset_metrics_state):
        """Test record_agent_invocation works after initialization."""
        from src.api.routes.metrics import get_metrics_registry, record_agent_invocation

        get_metrics_registry()

        # Should not raise
        record_agent_invocation(agent_name="causal_impact", tier="2", status="success")
        record_agent_invocation(agent_name="orchestrator", tier="1", status="failure")


class TestSetComponentHealth:
    """Tests for set_component_health function."""

    def test_set_component_health_no_error_when_not_initialized(self, reset_metrics_state):
        """Test set_component_health doesn't error when not initialized."""
        from src.api.routes.metrics import set_component_health

        # Should not raise
        set_component_health(component="api", healthy=True)

    def test_set_component_health_after_initialization(self, reset_metrics_state):
        """Test set_component_health works after initialization."""
        from src.api.routes.metrics import get_metrics_registry, set_component_health

        get_metrics_registry()

        # Should not raise
        set_component_health(component="api", healthy=True)
        set_component_health(component="database", healthy=False)


# =============================================================================
# METRICS REGISTRY TESTS
# =============================================================================


class TestMetricsRegistry:
    """Tests for metrics registry initialization."""

    def test_get_metrics_registry_initializes_on_first_call(self, reset_metrics_state):
        """Test that registry is initialized on first call."""
        from src.api.routes.metrics import get_metrics_registry

        registry = get_metrics_registry()

        # Should return a registry
        assert registry is not None

    def test_get_metrics_registry_returns_same_instance(self, reset_metrics_state):
        """Test that subsequent calls return the same registry."""
        from src.api.routes.metrics import get_metrics_registry

        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()

        assert registry1 is registry2

    def test_metrics_have_e2i_prefix(self, reset_metrics_state):
        """Test that all metrics use e2i_ prefix."""
        from src.api.routes.metrics import get_metrics_registry

        registry = get_metrics_registry()

        if registry:
            # Get metric names from registry
            collectors = list(registry._names_to_collectors.keys())
            for name in collectors:
                assert name.startswith("e2i_"), f"Metric {name} should start with e2i_"
