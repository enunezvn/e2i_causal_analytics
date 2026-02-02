"""Unit tests for BentoML client dependency.

Tests cover:
- Client initialization and configuration
- Prediction requests with retries
- Circuit breaker pattern
- Batch prediction
- Health checks
- Error handling and resilience
- Connection pooling
- Opik tracing integration

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    CircuitBreaker,
    CircuitState,
)


@pytest.mark.unit
class TestCircuitBreaker:
    """Test suite for CircuitBreaker pattern."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

        # Threshold reached
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_circuit_breaker_success_resets_failures(self):
        """Test successful requests reset failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker enters half-open state after reset timeout."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        import time

        time.sleep(0.15)

        # Should enter half-open state
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_recovery(self):
        """Test circuit breaker closes after successful recovery."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        import time

        time.sleep(0.15)

        # Enter half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # 3 consecutive successes to close
        cb.record_success()
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


@pytest.mark.unit
class TestBentoMLClientConfig:
    """Test suite for BentoMLClientConfig."""

    def test_config_defaults(self):
        """Test config uses default values."""
        config = BentoMLClientConfig()

        assert config.base_url == "http://localhost:3000"
        assert config.timeout == 10.0
        assert config.max_retries == 3
        assert config.max_connections == 20
        assert config.circuit_failure_threshold == 5
        assert config.enable_tracing is True

    def test_config_from_environment(self):
        """Test config reads from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "BENTOML_SERVICE_URL": "http://custom:8080",
                "BENTOML_TIMEOUT": "20.0",
                "BENTOML_MAX_RETRIES": "5",
                "BENTOML_MAX_CONNECTIONS": "50",
                "BENTOML_ENABLE_TRACING": "false",
            },
        ):
            config = BentoMLClientConfig()

            assert config.base_url == "http://custom:8080"
            assert config.timeout == 20.0
            assert config.max_retries == 5
            assert config.max_connections == 50
            assert config.enable_tracing is False

    def test_config_get_endpoint_url_default(self):
        """Test get_endpoint_url returns default pattern."""
        config = BentoMLClientConfig(base_url="http://localhost:3000")

        url = config.get_endpoint_url("churn_model")

        assert url == "http://localhost:3000/churn_model"

    def test_config_get_endpoint_url_custom_mapping(self):
        """Test get_endpoint_url uses custom mapping if provided."""
        config = BentoMLClientConfig(
            base_url="http://localhost:3000",
            model_endpoints={"churn_model": "http://custom:9000/churn"},
        )

        url = config.get_endpoint_url("churn_model")

        assert url == "http://custom:9000/churn"


@pytest.mark.unit
class TestBentoMLClient:
    """Test suite for BentoMLClient."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        client = BentoMLClient()

        assert client._initialized is False
        assert client._client is None

        await client.initialize()

        assert client._initialized is True
        assert client._client is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_client_initialize_idempotent(self):
        """Test multiple initialize calls are safe."""
        client = BentoMLClient()

        await client.initialize()
        first_client = client._client

        await client.initialize()
        second_client = client._client

        assert first_client is second_client

        await client.close()

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Test client cleanup."""
        client = BentoMLClient()

        await client.initialize()
        assert client._initialized is True

        await client.close()

        assert client._initialized is False
        assert client._client is None

    @pytest.mark.asyncio
    async def test_predict_success(self):
        """Test successful prediction request."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": [0.8, 0.2]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.predict("churn_model", {"features": [[0.1, 0.2]]})

            assert "prediction" in result
            assert "_metadata" in result
            assert result["_metadata"]["model_name"] == "churn_model"
            assert "latency_ms" in result["_metadata"]

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_wraps_input_data(self):
        """Test predict wraps input in expected BentoML format."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": [0.8]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.predict("churn_model", {"features": [[0.1]]})

            # Verify wrapped input
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs
            assert call_kwargs["json"] == {"input_data": {"features": [[0.1]]}}

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_with_trace_id(self):
        """Test predict includes trace ID in headers."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": [0.8]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.predict("churn_model", {"features": [[0.1]]}, trace_id="test-trace-123")

            call_kwargs = mock_post.call_args.kwargs
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["X-Trace-ID"] == "test-trace-123"

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_retries_on_server_error(self):
        """Test predict retries on 5xx errors."""
        config = BentoMLClientConfig(max_retries=3, retry_backoff_base=0.01)
        client = BentoMLClient(config)

        # First two calls fail, third succeeds
        error_response = MagicMock()
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=error_response
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"prediction": [0.8]}
        success_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                error_response,
                error_response,
                success_response,
            ]

            result = await client.predict("churn_model", {"features": [[0.1]]})

            assert "prediction" in result
            assert mock_post.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_no_retry_on_client_error(self):
        """Test predict doesn't retry on 4xx errors."""
        config = BentoMLClientConfig(max_retries=3)
        client = BentoMLClient(config)

        error_response = MagicMock()
        error_response.status_code = 400
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=error_response
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_response

            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("churn_model", {"features": [[0.1]]})

            # Should not retry on 4xx
            assert mock_post.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_circuit_breaker_opens(self):
        """Test circuit breaker opens after repeated failures."""
        config = BentoMLClientConfig(
            max_retries=1,
            circuit_failure_threshold=2,
            retry_backoff_base=0.01,
        )
        client = BentoMLClient(config)

        error_response = MagicMock()
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=error_response
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_response

            # First request fails and exhausts retries
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("churn_model", {"features": [[0.1]]})

            # Second request fails and opens circuit
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("churn_model", {"features": [[0.1]]})

            # Circuit is now open - should reject immediately
            with pytest.raises(RuntimeError, match="Circuit breaker open"):
                await client.predict("churn_model", {"features": [[0.1]]})

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_batch(self):
        """Test batch prediction request."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": [[0.8], [0.2]]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            batch_data = [{"features": [[0.1]]}, {"features": [[0.2]]}]
            result = await client.predict_batch("churn_model", batch_data)

            assert "predictions" in result
            # Verify endpoint
            assert "predict_batch" in mock_post.call_args.args[0]

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert "endpoint" in result
            assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_specific_model(self):
        """Test health check for specific model."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await client.health_check(model_name="churn_model")

            assert result["status"] == "healthy"
            # Verify correct endpoint
            call_args = mock_get.call_args.args[0]
            assert "churn_model" in call_args
            assert "healthz" in call_args

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check handles failures."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")

            result = await client.health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test get model info request."""
        config = BentoMLClientConfig()
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "churn_model",
            "version": "1.0.0",
            "framework": "sklearn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await client.get_model_info("churn_model")

            assert result["name"] == "churn_model"
            assert result["version"] == "1.0.0"
            # Verify metadata endpoint
            assert "metadata" in mock_get.call_args.args[0]

        await client.close()

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_per_endpoint(self):
        """Test circuit breakers are per-endpoint."""
        client = BentoMLClient()

        cb1 = client._get_circuit_breaker("model_a")
        cb2 = client._get_circuit_breaker("model_b")
        cb3 = client._get_circuit_breaker("model_a")

        assert cb1 is not cb2  # Different endpoints, different breakers
        assert cb1 is cb3  # Same endpoint, same breaker

        await client.close()

    @pytest.mark.asyncio
    async def test_opik_tracing_enabled(self):
        """Test Opik tracing when enabled."""
        config = BentoMLClientConfig(enable_tracing=True)
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": [0.8]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            # Patch opik.track directly since opik is imported inside _log_trace
            with patch("opik.track") as mock_track:
                mock_post.return_value = mock_response

                await client.predict("churn_model", {"features": [[0.1]]})

                # Verify opik.track was called
                mock_track.assert_called_once()
                call_kwargs = mock_track.call_args.kwargs
                assert "bentoml.predict.churn_model" in call_kwargs["name"]

        await client.close()

    @pytest.mark.asyncio
    async def test_opik_tracing_disabled(self):
        """Test Opik tracing when disabled."""
        config = BentoMLClientConfig(enable_tracing=False)
        client = BentoMLClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": [0.8]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            # Patch opik.track directly
            with patch("opik.track") as mock_track:
                mock_post.return_value = mock_response

                await client.predict("churn_model", {"features": [[0.1]]})

                # Opik should not be called when tracing is disabled
                mock_track.assert_not_called()

        await client.close()


@pytest.mark.unit
class TestBentoMLDependencyInjection:
    """Test suite for BentoML dependency injection functions."""

    @pytest.fixture(autouse=True)
    async def reset_global_client(self):
        """Reset global client before each test."""
        import src.api.dependencies.bentoml_client as bentoml_module

        bentoml_module._bentoml_client = None
        yield
        if bentoml_module._bentoml_client:
            await bentoml_module._bentoml_client.close()
        bentoml_module._bentoml_client = None

    @pytest.mark.asyncio
    async def test_get_bentoml_client_singleton(self):
        """Test get_bentoml_client returns singleton instance."""
        from src.api.dependencies.bentoml_client import get_bentoml_client

        client1 = await get_bentoml_client()
        client2 = await get_bentoml_client()

        assert client1 is client2
        assert client1._initialized is True

        await client1.close()

    @pytest.mark.asyncio
    async def test_close_bentoml_client(self):
        """Test close_bentoml_client cleanup."""
        from src.api.dependencies.bentoml_client import close_bentoml_client, get_bentoml_client

        client = await get_bentoml_client()
        assert client._initialized is True

        await close_bentoml_client()

        # Verify cleanup
        import src.api.dependencies.bentoml_client as bentoml_module

        assert bentoml_module._bentoml_client is None

    def test_configure_bentoml_endpoints(self):
        """Test configure_bentoml_endpoints updates config."""
        import src.api.dependencies.bentoml_client as bentoml_module
        from src.api.dependencies.bentoml_client import BentoMLClient, configure_bentoml_endpoints

        # Set up a client first
        client = BentoMLClient()
        bentoml_module._bentoml_client = client

        configure_bentoml_endpoints(
            {
                "churn_model": "http://custom:9000/churn",
                "conversion_model": "http://custom:9001/conversion",
            }
        )

        assert client.config.model_endpoints["churn_model"] == "http://custom:9000/churn"
        assert client.config.model_endpoints["conversion_model"] == "http://custom:9001/conversion"
