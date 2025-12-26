"""Unit tests for BentoML client wrapper.

Tests cover:
- Client initialization and configuration
- Prediction requests with mocked endpoints
- Circuit breaker behavior
- Retry logic
- Health checks
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    CircuitBreaker,
    CircuitState,
    close_bentoml_client,
    configure_bentoml_endpoints,
    get_bentoml_client,
)


def make_response(status_code: int, json_data: dict) -> httpx.Response:
    """Create a properly mocked httpx Response with request set."""
    request = httpx.Request("POST", "http://test/predict")
    response = httpx.Response(status_code, json=json_data, request=request)
    return response


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestBentoMLClientConfig:
    """Tests for BentoMLClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BentoMLClientConfig()
        assert config.timeout == 10.0
        assert config.max_retries == 3
        assert config.max_connections == 20
        assert config.circuit_failure_threshold == 5

    def test_get_endpoint_url_default(self):
        """Test endpoint URL generation with default pattern."""
        config = BentoMLClientConfig(base_url="http://localhost:3000")
        assert config.get_endpoint_url("churn_model") == "http://localhost:3000/churn_model"

    def test_get_endpoint_url_custom(self):
        """Test endpoint URL with custom mapping."""
        config = BentoMLClientConfig(
            model_endpoints={"churn_model": "http://churn-service:3000"}
        )
        assert config.get_endpoint_url("churn_model") == "http://churn-service:3000"


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
        import time
        time.sleep(0.15)

        assert cb.can_execute()  # Should transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_after_successes(self):
        """Test circuit closes after consecutive successes in half-open."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)

        cb.record_failure()
        import time
        time.sleep(0.02)
        cb.can_execute()  # Transition to HALF_OPEN

        # 3 consecutive successes should close
        for _ in range(3):
            cb.record_success()

        assert cb.state == CircuitState.CLOSED


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestBentoMLClient:
    """Tests for BentoMLClient."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return BentoMLClientConfig(
            base_url="http://test-bentoml:3000",
            timeout=5.0,
            max_retries=2,
            enable_tracing=False,
        )

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return BentoMLClient(config)

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
                "model_version": "1.0.0",
            },
        )

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict(
                "churn_model",
                {"features": [[0.1, 0.2, 0.3]]},
            )

            assert result["prediction"] == 0.85
            assert result["confidence"] == 0.92
            assert "_metadata" in result
            assert result["_metadata"]["model_name"] == "churn_model"

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_adds_metadata(self, client):
        """Test prediction adds metadata to response."""
        mock_response = make_response(200, {"prediction": 0.5})

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict("test_model", {"features": []})

            assert "_metadata" in result
            assert "latency_ms" in result["_metadata"]
            assert "timestamp" in result["_metadata"]
            assert result["_metadata"]["model_name"] == "test_model"

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_retries_on_server_error(self, client):
        """Test prediction retries on server errors."""
        error_response = make_response(500, {"error": "Internal error"})
        success_response = make_response(200, {"prediction": 0.5})

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

        with patch.object(
            httpx.AsyncClient, "post", side_effect=mock_post
        ):
            await client.initialize()
            result = await client.predict("test_model", {"features": []})

            assert result["prediction"] == 0.5
            assert call_count == 2  # Initial + 1 retry

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_circuit_breaker_opens(self, client):
        """Test circuit breaker opens after repeated failures."""
        # Set low threshold for testing
        client.config.circuit_failure_threshold = 2
        # Reduce retries to speed up test
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
                await client.predict("test_model", {"features": []})

            # Second call should fail after retries
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("test_model", {"features": []})

            # Circuit should now be open
            with pytest.raises(RuntimeError, match="Circuit breaker open"):
                await client.predict("test_model", {"features": []})

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check returns healthy status."""
        request = httpx.Request("GET", "http://test/healthz")
        mock_response = httpx.Response(200, json={"status": "ok"}, request=request)

        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            await client.initialize()
            result = await client.health_check("test_model")

            assert result["status"] == "healthy"
            assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Test health check returns unhealthy on error."""
        with patch.object(
            httpx.AsyncClient, "get", side_effect=httpx.RequestError("Connection refused")
        ):
            await client.initialize()
            result = await client.health_check("test_model")

            assert result["status"] == "unhealthy"
            assert "error" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, client):
        """Test batch prediction request."""
        mock_response = make_response(
            200,
            {
                "predictions": [
                    {"prediction": 0.8},
                    {"prediction": 0.6},
                    {"prediction": 0.9},
                ]
            },
        )

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict_batch(
                "test_model",
                [{"features": [0.1]}, {"features": [0.2]}, {"features": [0.3]}],
            )

            assert len(result["predictions"]) == 3
            assert result["predictions"][0]["prediction"] == 0.8

        await client.close()


# =============================================================================
# DEPENDENCY INJECTION TESTS
# =============================================================================


class TestDependencyInjection:
    """Tests for FastAPI dependency injection functions."""

    @pytest.mark.asyncio
    async def test_get_bentoml_client_returns_singleton(self):
        """Test get_bentoml_client returns singleton."""
        # Reset global state
        await close_bentoml_client()

        with patch.object(BentoMLClient, "initialize", new_callable=AsyncMock):
            client1 = await get_bentoml_client()
            client2 = await get_bentoml_client()

            assert client1 is client2

        await close_bentoml_client()

    @pytest.mark.asyncio
    async def test_configure_bentoml_endpoints(self):
        """Test configure_bentoml_endpoints updates config."""
        await close_bentoml_client()

        with patch.object(BentoMLClient, "initialize", new_callable=AsyncMock):
            client = await get_bentoml_client()

            configure_bentoml_endpoints({
                "custom_model": "http://custom-service:3000"
            })

            assert client.config.model_endpoints["custom_model"] == "http://custom-service:3000"

        await close_bentoml_client()

    @pytest.mark.asyncio
    async def test_close_bentoml_client_cleans_up(self):
        """Test close_bentoml_client cleans up resources."""
        await close_bentoml_client()

        with patch.object(BentoMLClient, "initialize", new_callable=AsyncMock):
            with patch.object(BentoMLClient, "close", new_callable=AsyncMock) as mock_close:
                client = await get_bentoml_client()
                await close_bentoml_client()

                mock_close.assert_called_once()
