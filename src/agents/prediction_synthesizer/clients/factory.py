"""Model Client Factory for Prediction Synthesizer.

This module provides a factory for creating model clients, supporting both
HTTP-based (BentoML endpoints) and in-process mock clients.

Features:
---------
- YAML configuration for model endpoints
- Support for multiple client types (HTTP, mock)
- Connection pooling management
- Lazy initialization

Usage:
------
    from src.agents.prediction_synthesizer.clients import (
        ModelClientFactory,
        get_model_client,
    )

    # Using factory directly
    factory = ModelClientFactory.from_config("config/model_endpoints.yaml")
    client = await factory.get_client("churn_model")
    result = await client.predict(...)

    # Using convenience function
    client = await get_model_client("churn_model")

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import yaml  # type: ignore[import-untyped]

from src.agents.prediction_synthesizer.clients.http_model_client import (
    HTTPModelClient,
    HTTPModelClientConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================


class ModelClient(Protocol):
    """Protocol for model prediction client.

    Matches the protocol defined in model_orchestrator.py.
    """

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Get prediction from model."""
        ...


# =============================================================================
# MOCK CLIENT (for testing)
# =============================================================================


class MockModelClient:
    """Mock model client for testing and development."""

    def __init__(
        self,
        model_id: str,
        default_prediction: float = 0.5,
        default_confidence: float = 0.8,
    ):
        """Initialize mock client.

        Args:
            model_id: Model identifier
            default_prediction: Default prediction value
            default_confidence: Default confidence value
        """
        self.model_id = model_id
        self.default_prediction = default_prediction
        self.default_confidence = default_confidence

    async def initialize(self) -> None:
        """No-op initialization."""
        pass

    async def close(self) -> None:
        """No-op cleanup."""
        pass

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Return mock prediction."""
        import random
        import time

        # Add some variation
        prediction = self.default_prediction + random.uniform(-0.1, 0.1)
        prediction = max(0.0, min(1.0, prediction))

        return {
            "prediction": prediction,
            "proba": {
                "positive": prediction,
                "negative": 1 - prediction,
            },
            "confidence": self.default_confidence + random.uniform(-0.05, 0.05),
            "model_type": "mock",
            "model_version": "mock-1.0",
            "features_used": list(features.keys()),
            "latency_ms": random.uniform(10, 50),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default base URL constant
DEFAULT_BASE_URL = os.environ.get("BENTOML_SERVICE_URL", "http://localhost:3000")


@dataclass
class ModelEndpointConfig:
    """Configuration for a single model endpoint."""

    model_id: str
    endpoint_url: str
    client_type: str = "http"  # "http" or "mock"
    timeout: float = 5.0
    max_retries: int = 3
    enabled: bool = True
    default_prediction: float = 0.5  # For mock clients
    default_confidence: float = 0.8  # For mock clients


@dataclass
class ModelEndpointsConfig:
    """Configuration for all model endpoints."""

    default_base_url: str = field(default_factory=lambda: DEFAULT_BASE_URL)
    default_timeout: float = 5.0
    default_max_retries: int = 3
    endpoints: Dict[str, ModelEndpointConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelEndpointsConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ModelEndpointsConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        base_url = data.get("default_base_url", DEFAULT_BASE_URL)

        endpoints = {}
        for model_id, model_config in data.get("endpoints", {}).items():
            endpoints[model_id] = ModelEndpointConfig(
                model_id=model_id,
                endpoint_url=model_config.get("url", f"{base_url}/{model_id}"),
                client_type=model_config.get("client_type", "http"),
                timeout=model_config.get("timeout", data.get("default_timeout", 5.0)),
                max_retries=model_config.get("max_retries", data.get("default_max_retries", 3)),
                enabled=model_config.get("enabled", True),
                default_prediction=model_config.get("default_prediction", 0.5),
                default_confidence=model_config.get("default_confidence", 0.8),
            )

        return cls(
            default_base_url=base_url,
            default_timeout=data.get("default_timeout", 5.0),
            default_max_retries=data.get("default_max_retries", 3),
            endpoints=endpoints,
        )


# =============================================================================
# FACTORY
# =============================================================================


class ModelClientFactory:
    """Factory for creating and managing model clients.

    This factory creates appropriate client instances based on configuration
    and manages their lifecycle.

    Attributes:
        config: Endpoint configuration
        _clients: Cache of initialized clients

    Example:
        factory = ModelClientFactory.from_config("config/model_endpoints.yaml")
        client = await factory.get_client("churn_model")
        result = await client.predict(...)
        await factory.close_all()
    """

    def __init__(self, config: Optional[ModelEndpointsConfig] = None):
        """Initialize factory.

        Args:
            config: Endpoint configuration. Uses defaults if not provided.
        """
        self.config = config or ModelEndpointsConfig()
        self._clients: Dict[str, ModelClient] = {}

    @classmethod
    def from_config(cls, config_path: str) -> "ModelClientFactory":
        """Create factory from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ModelClientFactory instance
        """
        config = ModelEndpointsConfig.from_yaml(config_path)
        return cls(config)

    async def get_client(self, model_id: str) -> ModelClient:
        """Get or create a model client.

        Args:
            model_id: Model identifier

        Returns:
            Model client instance

        Raises:
            ValueError: If model is not found or disabled
        """
        # Return cached client if exists
        if model_id in self._clients:
            return self._clients[model_id]

        # Get endpoint config
        endpoint_config = self.config.endpoints.get(model_id)

        if endpoint_config and not endpoint_config.enabled:
            raise ValueError(f"Model '{model_id}' is disabled")

        # Create client based on type
        client: Union[MockModelClient, HTTPModelClient]
        if endpoint_config and endpoint_config.client_type == "mock":
            client = MockModelClient(
                model_id=model_id,
                default_prediction=endpoint_config.default_prediction,
                default_confidence=endpoint_config.default_confidence,
            )
        else:
            # Default to HTTP client
            endpoint_url = (
                endpoint_config.endpoint_url
                if endpoint_config
                else f"{self.config.default_base_url}/{model_id}"
            )

            http_config = HTTPModelClientConfig(
                model_id=model_id,
                endpoint_url=endpoint_url,
                timeout=endpoint_config.timeout if endpoint_config else self.config.default_timeout,
                max_retries=endpoint_config.max_retries
                if endpoint_config
                else self.config.default_max_retries,
            )

            client = HTTPModelClient(
                model_id=model_id,
                endpoint_url=endpoint_url,
                config=http_config,
            )

        # Initialize and cache
        await client.initialize()
        self._clients[model_id] = client

        logger.info(f"Created {type(client).__name__} for model={model_id}")

        return client

    async def get_clients(self, model_ids: List[str]) -> Dict[str, ModelClient]:
        """Get or create multiple model clients.

        Args:
            model_ids: List of model identifiers

        Returns:
            Dictionary mapping model_id to client
        """
        clients = {}
        for model_id in model_ids:
            try:
                clients[model_id] = await self.get_client(model_id)
            except Exception as e:
                logger.warning(f"Failed to create client for {model_id}: {e}")
        return clients

    async def close_all(self) -> None:
        """Close all cached clients."""
        for model_id, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                logger.debug(f"Closed client for model={model_id}")
            except Exception as e:
                logger.warning(f"Error closing client for {model_id}: {e}")

        self._clients.clear()

    def list_available_models(self) -> List[str]:
        """List all configured model IDs.

        Returns:
            List of model identifiers
        """
        return [model_id for model_id, config in self.config.endpoints.items() if config.enabled]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global factory instance (singleton pattern)
_factory: Optional[ModelClientFactory] = None


async def get_model_client(
    model_id: str,
    config_path: Optional[str] = None,
) -> ModelClient:
    """Get a model client using the global factory.

    This is a convenience function for getting model clients without
    manually managing the factory lifecycle.

    Args:
        model_id: Model identifier
        config_path: Optional path to configuration file

    Returns:
        Model client instance
    """
    global _factory

    if _factory is None:
        # Look for config in standard locations
        if config_path is None:
            for path in [
                "config/model_endpoints.yaml",
                "config/model_endpoints.yml",
            ]:
                if Path(path).exists():
                    config_path = path
                    break

        if config_path:
            _factory = ModelClientFactory.from_config(config_path)
        else:
            _factory = ModelClientFactory()

    return await _factory.get_client(model_id)


async def close_model_clients() -> None:
    """Close all global model clients."""
    global _factory
    if _factory:
        await _factory.close_all()
        _factory = None


def configure_model_endpoints(config: Dict[str, Any]) -> None:
    """Configure model endpoints programmatically.

    Args:
        config: Configuration dictionary with endpoint definitions

    Example:
        configure_model_endpoints({
            "endpoints": {
                "churn_model": {
                    "url": "http://localhost:3000/churn_model",
                    "timeout": 5.0,
                },
            }
        })
    """
    global _factory

    endpoints = {}
    for model_id, model_config in config.get("endpoints", {}).items():
        endpoints[model_id] = ModelEndpointConfig(
            model_id=model_id,
            endpoint_url=model_config.get("url", f"http://localhost:3000/{model_id}"),
            client_type=model_config.get("client_type", "http"),
            timeout=model_config.get("timeout", 5.0),
            max_retries=model_config.get("max_retries", 3),
            enabled=model_config.get("enabled", True),
        )

    _factory = ModelClientFactory(
        ModelEndpointsConfig(
            default_base_url=config.get("default_base_url", "http://localhost:3000"),
            endpoints=endpoints,
        )
    )
