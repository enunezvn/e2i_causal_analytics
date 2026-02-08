"""HTTP Model Client for Prediction Synthesizer.

This module provides an HTTP-based implementation of the ModelClient protocol
for calling BentoML model serving endpoints.

Features:
---------
- Async HTTP calls with httpx
- Connection pooling for high-throughput
- Automatic retries with exponential backoff
- Circuit breaker pattern for failing services
- Opik tracing integration

Usage:
------
    from src.agents.prediction_synthesizer.clients import HTTPModelClient

    client = HTTPModelClient(
        model_id="churn_model",
        endpoint_url="http://localhost:3000/churn_model",
    )
    await client.initialize()
    result = await client.predict(
        entity_id="HCP001",
        features={"recency": 10, "frequency": 5},
        time_horizon="30d",
    )

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for resilient service calls."""

    failure_threshold: int = 5
    reset_timeout: float = 30.0
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # 3 consecutive successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker closed after recovery")
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (
                time.time() - self.last_failure_time >= self.reset_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker half-open, testing recovery")
                return True
            return False

        # HALF_OPEN
        return True


@dataclass
class HTTPModelClientConfig:
    """Configuration for HTTP model client."""

    # Model identifier
    model_id: str = ""

    # Endpoint URL for this model
    endpoint_url: str = ""

    # Request timeout in seconds
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("MODEL_CLIENT_TIMEOUT", "5.0"))
    )

    # Maximum retries for failed requests
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("MODEL_CLIENT_MAX_RETRIES", "3"))
    )

    # Retry backoff base (exponential)
    retry_backoff_base: float = 0.5

    # Connection pool size
    max_connections: int = field(
        default_factory=lambda: int(os.environ.get("MODEL_CLIENT_MAX_CONNECTIONS", "10"))
    )

    # Circuit breaker threshold (failures before opening)
    circuit_failure_threshold: int = 5

    # Circuit breaker reset timeout (seconds)
    circuit_reset_timeout: float = 30.0

    # Enable Opik tracing
    enable_tracing: bool = field(
        default_factory=lambda: os.environ.get("MODEL_CLIENT_ENABLE_TRACING", "true").lower()
        == "true"
    )


# =============================================================================
# HTTP MODEL CLIENT
# =============================================================================


class HTTPModelClient:
    """HTTP-based implementation of ModelClient protocol.

    This client provides an async interface for calling BentoML model endpoints
    that conforms to the ModelClient protocol expected by ModelOrchestratorNode.

    Attributes:
        model_id: Identifier for this model
        config: Client configuration
        _client: httpx async client
        _circuit_breaker: Circuit breaker for resilience

    Example:
        client = HTTPModelClient("churn_model", "http://localhost:3000/churn_model")
        await client.initialize()
        try:
            result = await client.predict(
                entity_id="HCP001",
                features={"recency": 10},
                time_horizon="30d",
            )
        finally:
            await client.close()
    """

    def __init__(
        self,
        model_id: str,
        endpoint_url: str,
        config: Optional[HTTPModelClientConfig] = None,
    ):
        """Initialize the HTTP model client.

        Args:
            model_id: Identifier for this model
            endpoint_url: Base URL for the model endpoint
            config: Optional client configuration
        """
        self.model_id = model_id
        self.endpoint_url = endpoint_url
        self.config = config or HTTPModelClientConfig(
            model_id=model_id,
            endpoint_url=endpoint_url,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            reset_timeout=self.config.circuit_reset_timeout,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the HTTP client with connection pool."""
        if self._initialized:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_connections // 2,
            ),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "E2I-ModelClient/1.0",
            },
        )
        self._initialized = True
        logger.debug(f"HTTPModelClient initialized for model={self.model_id}")

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.debug(f"HTTPModelClient closed for model={self.model_id}")

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Get prediction from model.

        This method conforms to the ModelClient protocol defined in model_orchestrator.py.

        Args:
            entity_id: Entity to make prediction for
            features: Feature dictionary for prediction
            time_horizon: Prediction time horizon (e.g., "30d", "90d")

        Returns:
            Dictionary with prediction, confidence, model_type, features_used

        Raises:
            RuntimeError: If circuit breaker is open
            httpx.HTTPError: If request fails after retries
        """
        if not self._initialized:
            await self.initialize()

        # After initialization, _client is guaranteed to be set
        assert self._client is not None

        if not self._circuit_breaker.can_execute():
            raise RuntimeError(
                f"Circuit breaker open for model '{self.model_id}'. Service may be unavailable."
            )

        start_time = time.time()
        last_exception: Optional[Exception] = None

        # Build request payload
        request_payload = {
            "entity_id": entity_id,
            "features": features,
            "time_horizon": time_horizon,
            "return_proba": True,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    f"{self.endpoint_url}/predict",
                    json=request_payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

                result = response.json()
                latency_ms = (time.time() - start_time) * 1000

                self._circuit_breaker.record_success()

                # Transform BentoML response to ModelClient format
                prediction_result = {
                    "prediction": result.get("prediction"),
                    "proba": result.get("probabilities"),
                    "confidence": result.get("confidence", 0.5),
                    "model_type": result.get("model_type", "unknown"),
                    "model_version": result.get("model_version"),
                    "features_used": result.get("features_used", list(features.keys())),
                    "latency_ms": latency_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if self.config.enable_tracing:
                    self._log_trace(entity_id, features, prediction_result, latency_ms)

                return prediction_result

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                    await self._backoff(attempt)
                else:
                    # Client error, don't retry
                    raise

            except httpx.RequestError as e:
                last_exception = e
                self._circuit_breaker.record_failure()
                await self._backoff(attempt)

        # All retries exhausted
        raise last_exception or RuntimeError(f"Failed to call model '{self.model_id}'")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the model endpoint.

        Returns:
            Health status dictionary
        """
        if not self._initialized:
            await self.initialize()

        # After initialization, _client is guaranteed to be set
        assert self._client is not None

        try:
            response = await self._client.get(
                f"{self.endpoint_url}/healthz",
                timeout=5.0,
            )
            response.raise_for_status()
            return {
                "status": "healthy",
                "model_id": self.model_id,
                "endpoint": self.endpoint_url,
                "circuit_breaker": self._circuit_breaker.state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except httpx.RequestError as e:
            return {
                "status": "unhealthy",
                "model_id": self.model_id,
                "endpoint": self.endpoint_url,
                "error": str(e),
                "circuit_breaker": self._circuit_breaker.state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        delay = self.config.retry_backoff_base * (2**attempt)
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        await asyncio.sleep(delay + jitter)

    def _log_trace(
        self,
        entity_id: str,
        features: Dict[str, Any],
        result: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Log prediction to Opik for observability."""
        try:
            import opik

            opik.track(  # type: ignore[call-arg]
                name=f"model_client.predict.{self.model_id}",
                input={
                    "entity_id": entity_id,
                    "feature_count": len(features),
                },
                output={
                    "prediction": str(result.get("prediction"))[:100],
                    "confidence": result.get("confidence"),
                },
                metadata={
                    "model_id": self.model_id,
                    "latency_ms": latency_ms,
                },
            )
        except Exception as e:
            logger.debug(f"Opik tracing failed (non-critical): {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state."""
        return self._circuit_breaker.state.value
