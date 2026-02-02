"""Unit tests for ModelClientFactory.

Tests cover:
- Factory initialization and configuration
- Client creation (HTTP and mock)
- YAML configuration loading
- Client lifecycle management
"""

import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.prediction_synthesizer.clients.factory import (
    MockModelClient,
    ModelClientFactory,
    ModelEndpointConfig,
    ModelEndpointsConfig,
    close_model_clients,
    configure_model_endpoints,
    get_model_client,
)
from src.agents.prediction_synthesizer.clients.http_model_client import HTTPModelClient

# =============================================================================
# MOCK CLIENT TESTS
# =============================================================================


class TestMockModelClient:
    """Tests for MockModelClient."""

    @pytest.mark.asyncio
    async def test_predict_returns_mock_values(self):
        """Test mock client returns predictions."""
        client = MockModelClient(
            model_id="test_model",
            default_prediction=0.7,
            default_confidence=0.9,
        )

        await client.initialize()
        result = await client.predict(
            entity_id="HCP001",
            features={"x": 1},
            time_horizon="30d",
        )

        assert "prediction" in result
        assert "confidence" in result
        assert result["model_type"] == "mock"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_predict_varies_values(self):
        """Test mock client adds variation to predictions."""
        client = MockModelClient(
            model_id="test_model",
            default_prediction=0.5,
            default_confidence=0.8,
        )

        results = []
        for _ in range(5):
            result = await client.predict(
                entity_id="HCP001",
                features={"x": 1},
                time_horizon="30d",
            )
            results.append(result["prediction"])

        # Results should vary slightly
        assert len(set(results)) > 1  # Not all the same


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestModelEndpointsConfig:
    """Tests for ModelEndpointsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelEndpointsConfig()
        assert "localhost:3000" in config.default_base_url
        assert config.default_timeout == 5.0
        assert config.default_max_retries == 3
        assert config.endpoints == {}

    def test_from_yaml_file_not_found(self):
        """Test loading non-existent YAML file."""
        config = ModelEndpointsConfig.from_yaml("/nonexistent/path.yaml")
        assert config.endpoints == {}

    def test_from_yaml_valid_file(self):
        """Test loading valid YAML configuration."""
        yaml_content = """
default_base_url: http://test:3000
default_timeout: 10.0
default_max_retries: 5

endpoints:
  churn_model:
    url: http://churn:3000
    timeout: 5.0
    client_type: http
    enabled: true

  mock_model:
    client_type: mock
    default_prediction: 0.8
    enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ModelEndpointsConfig.from_yaml(f.name)

            assert config.default_base_url == "http://test:3000"
            assert config.default_timeout == 10.0
            assert "churn_model" in config.endpoints
            assert config.endpoints["churn_model"].endpoint_url == "http://churn:3000"
            assert config.endpoints["mock_model"].client_type == "mock"


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestModelClientFactory:
    """Tests for ModelClientFactory."""

    @pytest.fixture
    def factory(self):
        """Create factory with test configuration."""
        config = ModelEndpointsConfig(
            default_base_url="http://test:3000",
            endpoints={
                "http_model": ModelEndpointConfig(
                    model_id="http_model",
                    endpoint_url="http://test:3000/http_model",
                    client_type="http",
                    enabled=True,
                ),
                "mock_model": ModelEndpointConfig(
                    model_id="mock_model",
                    endpoint_url="",
                    client_type="mock",
                    default_prediction=0.75,
                    enabled=True,
                ),
                "disabled_model": ModelEndpointConfig(
                    model_id="disabled_model",
                    endpoint_url="http://test:3000/disabled_model",
                    client_type="http",
                    enabled=False,
                ),
            },
        )
        return ModelClientFactory(config)

    @pytest.mark.asyncio
    async def test_get_mock_client(self, factory):
        """Test creating a mock client."""
        client = await factory.get_client("mock_model")

        assert isinstance(client, MockModelClient)
        assert client.model_id == "mock_model"
        assert client.default_prediction == 0.75

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_http_client(self, factory):
        """Test creating an HTTP client."""
        with patch.object(HTTPModelClient, "initialize", new_callable=AsyncMock):
            client = await factory.get_client("http_model")

            assert isinstance(client, HTTPModelClient)
            assert client.model_id == "http_model"

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_disabled_model_raises(self, factory):
        """Test getting disabled model raises ValueError."""
        with pytest.raises(ValueError, match="disabled"):
            await factory.get_client("disabled_model")

    @pytest.mark.asyncio
    async def test_client_caching(self, factory):
        """Test clients are cached."""
        client1 = await factory.get_client("mock_model")
        client2 = await factory.get_client("mock_model")

        assert client1 is client2

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_unknown_model(self, factory):
        """Test getting unknown model creates HTTP client with defaults."""
        with patch.object(HTTPModelClient, "initialize", new_callable=AsyncMock):
            client = await factory.get_client("unknown_model")

            assert isinstance(client, HTTPModelClient)
            assert client.model_id == "unknown_model"
            assert "unknown_model" in client.endpoint_url

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_get_clients_multiple(self, factory):
        """Test getting multiple clients."""
        with patch.object(HTTPModelClient, "initialize", new_callable=AsyncMock):
            clients = await factory.get_clients(["mock_model", "http_model"])

            assert len(clients) == 2
            assert "mock_model" in clients
            assert "http_model" in clients

        await factory.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self, factory):
        """Test closing all clients."""
        with patch.object(HTTPModelClient, "initialize", new_callable=AsyncMock):
            await factory.get_client("mock_model")
            await factory.get_client("http_model")

            assert len(factory._clients) == 2

            await factory.close_all()

            assert len(factory._clients) == 0

    def test_list_available_models(self, factory):
        """Test listing available models."""
        models = factory.list_available_models()

        assert "http_model" in models
        assert "mock_model" in models
        assert "disabled_model" not in models


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_get_model_client(self):
        """Test get_model_client function."""
        # Reset global state
        await close_model_clients()

        # Configure with mock
        configure_model_endpoints(
            {
                "endpoints": {
                    "test_model": {
                        "client_type": "mock",
                        "enabled": True,
                    },
                },
            }
        )

        client = await get_model_client("test_model")
        assert isinstance(client, MockModelClient)

        await close_model_clients()

    @pytest.mark.asyncio
    async def test_configure_model_endpoints(self):
        """Test configure_model_endpoints function."""
        await close_model_clients()

        configure_model_endpoints(
            {
                "default_base_url": "http://custom:3000",
                "endpoints": {
                    "custom_model": {
                        "url": "http://custom:3000/custom",
                        "client_type": "mock",
                    },
                },
            }
        )

        client = await get_model_client("custom_model")
        assert client is not None

        await close_model_clients()

    @pytest.mark.asyncio
    async def test_close_model_clients(self):
        """Test close_model_clients cleanup."""
        await close_model_clients()

        configure_model_endpoints(
            {
                "endpoints": {
                    "test_model": {"client_type": "mock"},
                },
            }
        )

        await get_model_client("test_model")
        await close_model_clients()

        # After cleanup, getting client again should work
        configure_model_endpoints(
            {
                "endpoints": {
                    "test_model": {"client_type": "mock"},
                },
            }
        )
        client = await get_model_client("test_model")
        assert client is not None

        await close_model_clients()
