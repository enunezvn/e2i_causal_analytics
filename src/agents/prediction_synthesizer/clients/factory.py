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

# F-012 (#430, codex iter-1 H1): explicit allowlist of dev/test ENVIRONMENT
# values. Anything outside this set — unset, misspelled, "production",
# "staging" — skips dev YAML merge AND forbids the implicit-HTTP fallback
# at the factory boundary.
_KNOWN_DEV_ENVIRONMENTS = {"development", "dev", "test", "testing", "local"}


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

        F-012 (#430): When ``ENVIRONMENT`` is explicitly set to one of
        ``{development, dev, test, testing, local}`` this method also
        merges ``config/dev_model_endpoints.yaml`` (if present and
        adjacent to the primary config file) so dev-only ``mock_model``
        entries become available. Any other ENVIRONMENT value
        — including UNSET, misspelled, ``production`` — skips the merge.
        Selecting a non-existent model id then raises ValueError
        downstream.

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
                endpoint_url=model_config.get(
                    "url", base_url
                ),  # flat single-model service: no /{model_id} suffix
                client_type=model_config.get("client_type", "http"),
                timeout=model_config.get("timeout", data.get("default_timeout", 5.0)),
                max_retries=model_config.get("max_retries", data.get("default_max_retries", 3)),
                enabled=model_config.get("enabled", True),
                default_prediction=model_config.get("default_prediction", 0.5),
                default_confidence=model_config.get("default_confidence", 0.8),
            )

        # F-012 (#430, codex iter-1 H1): merge dev-only endpoints ONLY when
        # ENVIRONMENT is an EXPLICIT dev value. Unset/misspelled/production
        # all skip the merge — missing metadata must not enable mock clients.
        environment = os.environ.get("ENVIRONMENT", "").strip().lower()
        if environment in _KNOWN_DEV_ENVIRONMENTS:
            dev_path = config_path.with_name("dev_model_endpoints.yaml")
            if dev_path.exists():
                with open(dev_path) as f:
                    dev_data = yaml.safe_load(f) or {}
                for model_id, model_config in dev_data.get("endpoints", {}).items():
                    endpoints[model_id] = ModelEndpointConfig(
                        model_id=model_id,
                        endpoint_url=model_config.get(
                            "url", base_url
                        ),  # flat single-model service: no /{model_id} suffix
                        client_type=model_config.get("client_type", "http"),
                        timeout=model_config.get("timeout", data.get("default_timeout", 5.0)),
                        max_retries=model_config.get(
                            "max_retries", data.get("default_max_retries", 3)
                        ),
                        enabled=model_config.get("enabled", True),
                        default_prediction=model_config.get("default_prediction", 0.5),
                        default_confidence=model_config.get("default_confidence", 0.8),
                    )
                logger.info(
                    "Loaded dev endpoints from %s (ENVIRONMENT=%s)",
                    dev_path,
                    environment,
                )
        else:
            logger.debug(
                "Skipping dev_model_endpoints.yaml load (ENVIRONMENT=%r is not dev)",
                environment,
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

        # F-012 (#430, codex iter-1 H2): require an explicit endpoint
        # declaration. Previously, an unknown ``model_id`` would silently
        # construct an HTTPModelClient pointing at
        # ``{default_base_url}/{model_id}`` — so removing ``mock_model``
        # from the prod yaml didn't actually fail-closed at this boundary.
        # Now any undeclared model raises ValueError immediately.
        if endpoint_config is None:
            raise ValueError(
                f"Model '{model_id}' has no endpoint declaration in "
                "config/model_endpoints.yaml. Add an explicit entry "
                "(client_type + url) to make this model available."
            )

        if not endpoint_config.enabled:
            raise ValueError(f"Model '{model_id}' is disabled")

        # Create client based on type
        client: Union[MockModelClient, HTTPModelClient]
        if endpoint_config.client_type == "mock":
            client = MockModelClient(
                model_id=model_id,
                default_prediction=endpoint_config.default_prediction,
                default_confidence=endpoint_config.default_confidence,
            )
        else:
            # HTTP client uses the explicitly declared endpoint URL.
            http_config = HTTPModelClientConfig(
                model_id=model_id,
                endpoint_url=endpoint_config.endpoint_url,
                timeout=endpoint_config.timeout,
                max_retries=endpoint_config.max_retries,
            )

            client = HTTPModelClient(
                model_id=model_id,
                endpoint_url=endpoint_config.endpoint_url,
                config=http_config,
            )

        # Initialize and cache
        await client.initialize()
        self._clients[model_id] = client

        logger.info(f"Created {type(client).__name__} for model={model_id}")

        return client

    async def get_clients(self, model_ids: List[str]) -> Dict[str, ModelClient]:
        """Get or create multiple model clients.

        F-012 (#430, codex iter-2 M1 + iter-3 H4): undeclared models AND
        declared-model init failures now both propagate to the caller.
        The previous broad ``except Exception`` was a backward-compat shim
        — it would silently drop a failed declared model and return a
        partial dict, masking real misconfiguration. Per the no-silent-
        fabrication directive, the caller (planner or scheduler) sees the
        error and decides whether to retry, fail the whole job, or
        proceed without that model.

        Args:
            model_ids: List of model identifiers

        Returns:
            Dictionary mapping model_id to client (always complete on
            success; partial dicts are never returned).

        Raises:
            ValueError: If any requested model has no endpoint declaration
                or is disabled.
            Exception: Any client-init error (network, auth, etc.) from
                ``get_client`` is propagated unchanged.
        """
        clients = {}
        for model_id in model_ids:
            clients[model_id] = await self.get_client(model_id)
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
            endpoint_url=model_config.get(
                "url", config.get("default_base_url", "http://localhost:3000")
            ),  # flat: no /{model_id}
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
