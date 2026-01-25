"""
End-to-End Integration Tests for Prediction Flow.

Tests the complete prediction flow:
- BentoML client → HTTP endpoints
- ModelClientFactory → HTTPModelClient
- prediction_synthesizer agent integration
- Latency validation (target: <100ms p95)
- Prometheus metrics collection

These tests use mocked HTTP endpoints to simulate BentoML services.
Use pytest markers to skip when running with live services.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from src.agents.prediction_synthesizer.clients import (
    HTTPModelClient,
    HTTPModelClientConfig,
    ModelClientFactory,
    get_model_client,
)
from src.agents.prediction_synthesizer.clients.factory import (
    MockModelClient,
    ModelEndpointConfig,
    ModelEndpointsConfig,
    close_model_clients,
    configure_model_endpoints,
)
from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    close_bentoml_client,
    get_bentoml_client,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Environment checks for live service testing
BENTOML_SERVICE_URL = os.getenv("BENTOML_SERVICE_URL", "")


def _check_bentoml_service_available() -> bool:
    """Check if BentoML service is actually reachable."""
    if not BENTOML_SERVICE_URL:
        return False
    try:
        # Quick sync check during test collection
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(BENTOML_SERVICE_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 3000
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


HAS_BENTOML_SERVICE = _check_bentoml_service_available()

requires_bentoml = pytest.mark.skipif(
    not HAS_BENTOML_SERVICE,
    reason="BentoML service not available (URL not set or service unreachable)",
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_mock_response(status_code: int, json_data: Dict[str, Any]) -> httpx.Response:
    """Create a properly mocked httpx Response."""
    request = httpx.Request("POST", "http://test/predict")
    return httpx.Response(status_code, json=json_data, request=request)


def create_mock_prediction_response(
    prediction: float = 0.75,
    confidence: float = 0.85,
    model_type: str = "xgboost",
    model_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Create a mock prediction response."""
    return {
        "prediction": prediction,
        "confidence": confidence,
        "proba": {"positive": prediction, "negative": 1 - prediction},
        "model_type": model_type,
        "model_version": model_version,
        "features_used": ["recency", "frequency", "monetary"],
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trace_id() -> str:
    """Generate unique trace ID for test isolation."""
    return f"test-prediction-{uuid.uuid4().hex[:16]}"


@pytest.fixture
async def mock_factory():
    """Create factory with mock clients for testing."""
    config = ModelEndpointsConfig(
        default_base_url="http://mock:3000",
        endpoints={
            "churn_model": ModelEndpointConfig(
                model_id="churn_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.7,
                default_confidence=0.85,
                enabled=True,
            ),
            "conversion_model": ModelEndpointConfig(
                model_id="conversion_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.6,
                default_confidence=0.8,
                enabled=True,
            ),
            "adoption_model": ModelEndpointConfig(
                model_id="adoption_model",
                endpoint_url="",
                client_type="mock",
                default_prediction=0.5,
                default_confidence=0.75,
                enabled=True,
            ),
        },
    )
    factory = ModelClientFactory(config)
    yield factory
    await factory.close_all()


@pytest.fixture
def bentoml_config() -> BentoMLClientConfig:
    """Create BentoML client configuration for testing."""
    return BentoMLClientConfig(
        base_url="http://test-bentoml:3000",
        timeout=5.0,
        max_retries=2,
        enable_tracing=False,
    )


@pytest.fixture(autouse=True)
async def cleanup_clients():
    """Cleanup global client instances after tests."""
    yield
    await close_model_clients()
    await close_bentoml_client()


# =============================================================================
# MockModelClient Tests
# =============================================================================


class TestMockClientPrediction:
    """Tests for mock client prediction flow."""

    @pytest.mark.asyncio
    async def test_single_prediction(self, mock_factory):
        """Test single prediction through mock client."""
        client = await mock_factory.get_client("churn_model")

        result = await client.predict(
            entity_id="HCP001",
            features={"recency": 10, "frequency": 5, "monetary": 500},
            time_horizon="30d",
        )

        assert "prediction" in result
        assert 0 <= result["prediction"] <= 1
        assert "confidence" in result
        assert result["model_type"] == "mock"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_multiple_predictions(self, mock_factory):
        """Test multiple predictions in sequence."""
        client = await mock_factory.get_client("churn_model")

        predictions = []
        for i in range(10):
            result = await client.predict(
                entity_id=f"HCP{i:03d}",
                features={"recency": i, "frequency": i * 2},
                time_horizon="30d",
            )
            predictions.append(result)

        assert len(predictions) == 10
        assert all("prediction" in p for p in predictions)

    @pytest.mark.asyncio
    async def test_parallel_predictions(self, mock_factory):
        """Test parallel predictions to multiple models."""
        clients = await mock_factory.get_clients(
            ["churn_model", "conversion_model", "adoption_model"]
        )

        async def get_prediction(model_id: str, client):
            return await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )

        tasks = [
            get_prediction(model_id, client)
            for model_id, client in clients.items()
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all("prediction" in r for r in results)


# =============================================================================
# HTTPModelClient Tests with Mocked Endpoints
# =============================================================================


class TestHTTPClientPrediction:
    """Tests for HTTP client prediction flow with mocked endpoints."""

    @pytest.mark.asyncio
    async def test_http_client_prediction(self):
        """Test HTTP client prediction with mocked response."""
        mock_response = make_mock_response(200, create_mock_prediction_response())

        client = HTTPModelClient(
            model_id="churn_model",
            endpoint_url="http://test-bentoml:3000/churn_model",
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10, "frequency": 5},
                time_horizon="30d",
            )

        assert result["prediction"] == 0.75
        assert result["confidence"] == 0.85
        assert "latency_ms" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_http_client_retry_success(self):
        """Test HTTP client retries and succeeds."""
        error_response = make_mock_response(500, {"error": "Internal error"})
        success_response = make_mock_response(200, create_mock_prediction_response())

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

        client = HTTPModelClient(
            model_id="churn_model",
            endpoint_url="http://test:3000/churn_model",
            config=HTTPModelClientConfig(max_retries=3),
        )

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            await client.initialize()
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )

        assert result["prediction"] == 0.75
        assert call_count == 2

        await client.close()


# =============================================================================
# BentoML Client Integration Tests
# =============================================================================


class TestBentoMLClientIntegration:
    """Tests for BentoML FastAPI client integration."""

    @pytest.mark.asyncio
    async def test_bentoml_client_prediction(self, bentoml_config):
        """Test BentoML client prediction with mocked response."""
        mock_response = make_mock_response(200, create_mock_prediction_response())

        client = BentoMLClient(bentoml_config)

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.initialize()
            result = await client.predict(
                model_name="churn_model",
                input_data={"features": [[0.1, 0.2, 0.3]]},
            )

        assert result["prediction"] == 0.75
        assert "_metadata" in result
        assert result["_metadata"]["model_name"] == "churn_model"
        assert "latency_ms" in result["_metadata"]

        await client.close()

    @pytest.mark.asyncio
    async def test_bentoml_health_check(self, bentoml_config):
        """Test BentoML health check."""
        health_response = httpx.Response(
            200,
            json={"status": "healthy"},
            request=httpx.Request("GET", "http://test/healthz"),
        )

        client = BentoMLClient(bentoml_config)

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = health_response

            await client.initialize()
            result = await client.health_check()

        assert result["status"] == "healthy"
        assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_bentoml_circuit_breaker(self, bentoml_config):
        """Test BentoML circuit breaker opens after failures."""
        error_response = make_mock_response(500, {"error": "Server down"})

        client = BentoMLClient(bentoml_config)
        client.config.circuit_failure_threshold = 2
        client.config.max_retries = 1

        async def mock_post(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "Server error",
                request=error_response.request,
                response=error_response,
            )

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            await client.initialize()

            # First call should fail
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("churn_model", {"features": []})

            # Second call should fail and open circuit
            with pytest.raises(httpx.HTTPStatusError):
                await client.predict("churn_model", {"features": []})

            # Third call should be rejected by circuit breaker
            with pytest.raises(RuntimeError, match="Circuit breaker open"):
                await client.predict("churn_model", {"features": []})

        await client.close()


# =============================================================================
# Latency Performance Tests
# =============================================================================


class TestLatencyPerformance:
    """Tests for prediction latency requirements."""

    @pytest.mark.asyncio
    async def test_mock_prediction_latency(self, mock_factory):
        """Test mock prediction latency is within acceptable bounds."""
        client = await mock_factory.get_client("churn_model")

        latencies: List[float] = []

        for _ in range(100):
            start = time.perf_counter()
            await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        latencies.sort()
        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]

        # Mock client should be very fast
        assert p50 < 10, f"P50 latency {p50:.2f}ms > 10ms"
        assert p95 < 50, f"P95 latency {p95:.2f}ms > 50ms"
        assert p99 < 100, f"P99 latency {p99:.2f}ms > 100ms"

    @pytest.mark.asyncio
    async def test_concurrent_prediction_latency(self, mock_factory):
        """Test latency under concurrent load."""
        clients = await mock_factory.get_clients(
            ["churn_model", "conversion_model", "adoption_model"]
        )

        async def single_prediction():
            start = time.perf_counter()
            await clients["churn_model"].predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
            return (time.perf_counter() - start) * 1000

        # Run 50 concurrent predictions
        start = time.perf_counter()
        tasks = [single_prediction() for _ in range(50)]
        latencies = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start) * 1000

        latencies = sorted(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]

        # P95 should be under 100ms even with concurrency
        assert p95 < 100, f"P95 latency {p95:.2f}ms > 100ms under concurrent load"
        # Total time for 50 concurrent requests should be reasonable
        assert total_time < 500, f"Total time {total_time:.2f}ms > 500ms for 50 concurrent requests"


# =============================================================================
# End-to-End Prediction Flow Tests
# =============================================================================


class TestEndToEndPredictionFlow:
    """End-to-end tests for complete prediction flow."""

    @pytest.mark.asyncio
    async def test_full_prediction_flow_with_mock(self, mock_factory, trace_id):
        """Test complete prediction flow from client to result."""
        # Get clients for multiple models
        clients = await mock_factory.get_clients(
            ["churn_model", "conversion_model"]
        )

        # Simulate orchestrated prediction flow
        entity_id = "HCP001"
        features = {"recency": 10, "frequency": 5, "monetary": 500}

        predictions = {}
        for model_id, client in clients.items():
            result = await client.predict(
                entity_id=entity_id,
                features=features,
                time_horizon="30d",
            )
            predictions[model_id] = result

        # Verify all predictions received
        assert len(predictions) == 2
        assert "churn_model" in predictions
        assert "conversion_model" in predictions

        # Ensemble calculation (simple average)
        avg_prediction = sum(
            p["prediction"] for p in predictions.values()
        ) / len(predictions)

        assert 0 <= avg_prediction <= 1

    @pytest.mark.asyncio
    async def test_prediction_flow_with_failure_recovery(self, mock_factory):
        """Test prediction flow handles partial failures."""
        # Configure one model to fail
        config = ModelEndpointsConfig(
            endpoints={
                "working_model": ModelEndpointConfig(
                    model_id="working_model",
                    endpoint_url="",
                    client_type="mock",
                    default_prediction=0.7,
                    enabled=True,
                ),
                "disabled_model": ModelEndpointConfig(
                    model_id="disabled_model",
                    endpoint_url="http://fail:3000",
                    client_type="http",  # Will fail without real endpoint
                    enabled=False,  # Disabled
                ),
            },
        )
        factory = ModelClientFactory(config)

        # Get available models (should exclude disabled)
        available = factory.list_available_models()
        assert "working_model" in available
        assert "disabled_model" not in available

        # Prediction should work with available model
        client = await factory.get_client("working_model")
        result = await client.predict(
            entity_id="HCP001",
            features={"x": 1},
            time_horizon="30d",
        )

        assert result["prediction"] is not None

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_global_convenience_functions(self):
        """Test global convenience functions work correctly."""
        # Configure with mock
        configure_model_endpoints({
            "endpoints": {
                "test_model": {
                    "client_type": "mock",
                    "enabled": True,
                    "default_prediction": 0.65,
                },
            },
        })

        # Get client through convenience function
        client = await get_model_client("test_model")
        assert isinstance(client, MockModelClient)

        # Make prediction
        result = await client.predict(
            entity_id="HCP001",
            features={"x": 1},
            time_horizon="30d",
        )

        assert "prediction" in result

        # Cleanup
        await close_model_clients()


# =============================================================================
# Live Service Tests (Requires Running BentoML)
# =============================================================================


class TestLiveBentoMLService:
    """Tests that require running BentoML services."""

    @requires_bentoml
    @pytest.mark.asyncio
    async def test_live_prediction(self):
        """Test live prediction against running BentoML service."""
        client = await get_bentoml_client()

        # Check health first
        health = await client.health_check()
        assert health["status"] == "healthy", f"BentoML service unhealthy: {health}"

        # Make prediction
        result = await client.predict(
            model_name="churn_model",
            input_data={"features": [[0.5, 0.3, 0.7, 0.2, 0.9]]},
        )

        assert "prediction" in result
        assert "_metadata" in result

    @requires_bentoml
    @pytest.mark.asyncio
    async def test_live_latency_requirement(self):
        """Test live prediction meets latency requirements."""
        client = await get_bentoml_client()

        latencies = []

        for _ in range(10):
            start = time.perf_counter()
            await client.predict(
                model_name="churn_model",
                input_data={"features": [[0.5, 0.3, 0.7]]},
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        # Target: <100ms P95
        assert p95 < 100, f"P95 latency {p95:.2f}ms exceeds 100ms target"


# =============================================================================
# Metrics Collection Tests
# =============================================================================


class TestMetricsCollection:
    """Tests for metrics collection and observability."""

    @pytest.mark.asyncio
    async def test_prediction_captures_latency(self, mock_factory):
        """Test that predictions capture latency metrics."""
        client = await mock_factory.get_client("churn_model")

        result = await client.predict(
            entity_id="HCP001",
            features={"x": 1},
            time_horizon="30d",
        )

        # Mock client should include latency
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))
        assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_prediction_includes_timestamp(self, mock_factory):
        """Test that predictions include timestamp."""
        client = await mock_factory.get_client("churn_model")

        result = await client.predict(
            entity_id="HCP001",
            features={"x": 1},
            time_horizon="30d",
        )

        assert "timestamp" in result
        # Verify it's a valid timestamp format
        timestamp = result["timestamp"]
        assert "T" in timestamp or "-" in timestamp
