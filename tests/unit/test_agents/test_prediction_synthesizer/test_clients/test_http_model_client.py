"""Unit tests for HTTPModelClient.

Tests cover:
- Client initialization and configuration
- Prediction requests with mocked endpoints
- Circuit breaker behavior
- Retry logic
- Health checks
"""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.agents.prediction_synthesizer.clients.http_model_client import (
    CircuitBreaker,
    CircuitState,
    HTTPModelClient,
    HTTPModelClientConfig,
)


def make_response(status_code: int, json_data: dict) -> httpx.Response:
    """Create a properly mocked httpx Response with request set."""
    request = httpx.Request("POST", "http://test/predict")
    response = httpx.Response(status_code, json=json_data, request=request)
    return response


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestHTTPModelClientConfig:
    """Tests for HTTPModelClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HTTPModelClientConfig()
        assert config.timeout == 5.0
        assert config.max_retries == 3
        assert config.max_connections == 10
        assert config.circuit_failure_threshold == 5

    def test_custom_values(self):
        """Test custom configuration."""
        config = HTTPModelClientConfig(
            model_id="test_model",
            endpoint_url="http://custom:3000",
            timeout=10.0,
            max_retries=5,
        )
        assert config.model_id == "test_model"
        assert config.endpoint_url == "http://custom:3000"
        assert config.timeout == 10.0
        assert config.max_retries == 5


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts closed."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0

    def test_half_open_after_timeout(self):
        """Test circuit goes half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)

        cb.record_failure()  # Opens circuit
        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        assert cb.can_execute()  # Should transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_after_successes(self):
        """Test circuit closes after consecutive successes in half-open."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)

        cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()  # Transition to HALF_OPEN

        # 3 consecutive successes should close
        for _ in range(3):
            cb.record_success()

        assert cb.state == CircuitState.CLOSED


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestHTTPModelClient:
    """Tests for HTTPModelClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return HTTPModelClient(
            model_id="test_model",
            endpoint_url="http://test-bentoml:3000/test_model",
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_http_client(self, client):
        """Test initialization creates HTTP client."""
        assert not client._initialized

        await client.initialize()

        assert client._initialized
        assert client._client is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, client):
        """Test close cleans up resources."""
        await client.initialize()
        await client.close()

        assert client._client is None
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_predict_success(self, client):
        """Test successful prediction request."""
        mock_response = make_response(
            200,
            {
                "prediction": 0.85,
                "confidence": 0.92,
                "model_type": "xgboost",
                "model_version": "1.0.0",
            },
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10, "frequency": 5},
                time_horizon="30d",
            )

            assert result["prediction"] == 0.85
            assert result["confidence"] == 0.92
            assert result["model_type"] == "xgboost"
            assert "latency_ms" in result
            assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_transforms_response(self, client):
        """Test prediction transforms BentoML response correctly."""
        mock_response = make_response(
            200,
            {
                "prediction": 0.75,
                "probabilities": {"positive": 0.75, "negative": 0.25},
                "confidence": 0.88,
                "features_used": ["recency", "frequency"],
            },
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )

            assert result["prediction"] == 0.75
            assert result["proba"]["positive"] == 0.75
            assert result["confidence"] == 0.88
            assert "recency" in result["features_used"]

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_retries_on_server_error(self, client):
        """Test prediction retries on server errors."""
        error_response = make_response(500, {"error": "Internal error"})
        success_response = make_response(200, {"prediction": 0.5, "confidence": 0.8})

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.HTTPStatusError(
                    "Server error",
                    request=error_response.request,
                    response=error_response,
                )
            return success_response

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            await client.initialize()
            result = await client.predict(
                entity_id="HCP001",
                features={"x": 1},
                time_horizon="30d",
            )

            assert result["prediction"] == 0.5
            assert call_count == 2  # Initial + 1 retry

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_circuit_breaker_opens(self, client):
        """Test circuit breaker opens after repeated failures."""
        # Set low threshold for testing
        client._circuit_breaker.failure_threshold = 2
        client.config.max_retries = 1

        error_response = make_response(500, {"error": "Server down"})

        async def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "Server error",
                request=error_response.request,
                response=error_response,
            )

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            await client.initialize()

            # First call should fail after retries
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict(
                    entity_id="HCP001",
                    features={},
                    time_horizon="30d",
                )

            # Second call should fail after retries
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict(
                    entity_id="HCP001",
                    features={},
                    time_horizon="30d",
                )

            # Circuit should now be open
            with pytest.raises(RuntimeError, match="Circuit breaker open"):
                await client.predict(
                    entity_id="HCP001",
                    features={},
                    time_horizon="30d",
                )

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check returns healthy status."""
        request = httpx.Request("GET", "http://test/healthz")
        mock_response = httpx.Response(200, json={"status": "ok"}, request=request)

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            await client.initialize()
            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["model_id"] == "test_model"
            assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Test health check returns unhealthy on error."""
        with patch.object(
            httpx.AsyncClient, "get", side_effect=httpx.RequestError("Connection refused")
        ):
            await client.initialize()
            result = await client.health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result

        await client.close()

    def test_properties(self, client):
        """Test client properties."""
        assert client.model_id == "test_model"
        assert not client.is_initialized
        assert client.circuit_state == "closed"
