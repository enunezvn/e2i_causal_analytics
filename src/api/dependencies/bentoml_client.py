"""BentoML Client Wrapper for FastAPI.

This module provides an async HTTP client for BentoML model serving endpoints.

Features:
---------
- Async HTTP calls with httpx
- Connection pooling for high-throughput
- Automatic retries with exponential backoff
- Circuit breaker pattern for failing services
- Health check support
- Opik tracing integration

Usage:
------
    from src.api.dependencies import get_bentoml_client, BentoMLClient

    @app.get("/predict")
    async def predict(client: BentoMLClient = Depends(get_bentoml_client)):
        result = await client.predict("churn_model", {"features": [...]})
        return result

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
from typing import Any, Dict, List, Optional

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
class BentoMLClientConfig:
    """Configuration for BentoML client."""

    # Base URL for BentoML service
    base_url: str = field(
        default_factory=lambda: os.environ.get("BENTOML_SERVICE_URL", "http://localhost:3000")
    )

    # Request timeout in seconds
    timeout: float = field(default_factory=lambda: float(os.environ.get("BENTOML_TIMEOUT", "10.0")))

    # Maximum retries for failed requests
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("BENTOML_MAX_RETRIES", "3"))
    )

    # Retry backoff base (exponential)
    retry_backoff_base: float = 0.5

    # Connection pool size
    max_connections: int = field(
        default_factory=lambda: int(os.environ.get("BENTOML_MAX_CONNECTIONS", "20"))
    )

    # Circuit breaker threshold (failures before opening)
    circuit_failure_threshold: int = 5

    # Circuit breaker reset timeout (seconds)
    circuit_reset_timeout: float = 30.0

    # Enable Opik tracing
    enable_tracing: bool = field(
        default_factory=lambda: os.environ.get("BENTOML_ENABLE_TRACING", "true").lower() == "true"
    )

    # Model endpoint mapping (model_name -> endpoint_url)
    model_endpoints: Dict[str, str] = field(default_factory=dict)

    def get_endpoint_url(self, model_name: str) -> str:
        """Get endpoint URL for a model."""
        if model_name in self.model_endpoints:
            return self.model_endpoints[model_name]
        # Default pattern: base_url/model_name
        return f"{self.base_url}/{model_name}"


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


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


# =============================================================================
# BENTOML CLIENT
# =============================================================================


class BentoMLClient:
    """Async HTTP client for BentoML model serving endpoints.

    This client provides a high-level interface for calling BentoML prediction
    endpoints with built-in resilience patterns.

    Attributes:
        config: Client configuration
        _client: httpx async client
        _circuit_breakers: Per-endpoint circuit breakers

    Example:
        client = BentoMLClient(config)
        await client.initialize()
        try:
            result = await client.predict("churn_model", {"features": [[0.1, 0.2]]})
        finally:
            await client.close()
    """

    def __init__(self, config: Optional[BentoMLClientConfig] = None):
        """Initialize the BentoML client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self.config = config or BentoMLClientConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
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
                "User-Agent": "E2I-BentoML-Client/1.0",
            },
        )
        self._initialized = True
        logger.info(f"BentoML client initialized with base_url={self.config.base_url}")

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("BentoML client closed")

    def _get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an endpoint."""
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                reset_timeout=self.config.circuit_reset_timeout,
            )
        return self._circuit_breakers[endpoint]

    async def predict(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a prediction request to a BentoML model endpoint.

        Args:
            model_name: Name of the model to call
            input_data: Input data for prediction (JSON-serializable)
            timeout: Optional request timeout override
            trace_id: Optional trace ID for observability

        Returns:
            Prediction result from the model

        Raises:
            httpx.HTTPError: If request fails after retries
            RuntimeError: If circuit breaker is open
        """
        if not self._initialized:
            await self.initialize()

        endpoint_url = self.config.get_endpoint_url(model_name)
        circuit = self._get_circuit_breaker(model_name)

        if not circuit.can_execute():
            raise RuntimeError(
                f"Circuit breaker open for model '{model_name}'. Service may be unavailable."
            )

        start_time = time.time()
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                headers = {}
                if trace_id:
                    headers["X-Trace-ID"] = trace_id

                # BentoML expects input wrapped in parameter name from method signature
                # e.g., async def predict(self, input_data: Model) expects {"input_data": {...}}
                wrapped_input = {"input_data": input_data}

                assert self._client is not None
                response = await self._client.post(
                    f"{endpoint_url}/predict",
                    json=wrapped_input,
                    timeout=timeout or self.config.timeout,
                    headers=headers,
                )
                response.raise_for_status()

                result = response.json()
                latency_ms = (time.time() - start_time) * 1000

                circuit.record_success()

                # Add metadata
                result["_metadata"] = {
                    "model_name": model_name,
                    "latency_ms": latency_ms,
                    "endpoint": endpoint_url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if self.config.enable_tracing:
                    self._log_trace(model_name, input_data, result, latency_ms, trace_id)

                return result  # type: ignore[no-any-return]

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500:
                    circuit.record_failure()
                    await self._backoff(attempt)
                else:
                    # Client error, don't retry
                    raise

            except httpx.RequestError as e:
                last_exception = e
                circuit.record_failure()
                await self._backoff(attempt)

        # All retries exhausted
        raise last_exception or RuntimeError(f"Failed to call model '{model_name}'")

    async def predict_batch(
        self,
        model_name: str,
        batch_data: List[Dict[str, Any]],
        *,
        timeout: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Make batch prediction requests.

        Args:
            model_name: Name of the model to call
            batch_data: List of input data dictionaries
            timeout: Optional request timeout override
            trace_id: Optional trace ID for observability

        Returns:
            List of prediction results
        """
        if not self._initialized:
            await self.initialize()

        endpoint_url = self.config.get_endpoint_url(model_name)

        start_time = time.time()

        assert self._client is not None
        response = await self._client.post(
            f"{endpoint_url}/predict_batch",
            json={"instances": batch_data},
            timeout=timeout or self.config.timeout * 2,  # Longer timeout for batch
        )
        response.raise_for_status()

        results = response.json()
        latency_ms = (time.time() - start_time) * 1000

        if self.config.enable_tracing:
            self._log_trace(
                model_name,
                {"batch_size": len(batch_data)},
                {"batch_results": len(results.get("predictions", []))},
                latency_ms,
                trace_id,
            )

        return results  # type: ignore[no-any-return]

    async def health_check(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Check health of BentoML service or specific model.

        Args:
            model_name: Optional specific model to check. If None, checks base service.

        Returns:
            Health status dictionary
        """
        if not self._initialized:
            await self.initialize()

        if model_name:
            endpoint_url = self.config.get_endpoint_url(model_name)
        else:
            endpoint_url = self.config.base_url

        try:
            assert self._client is not None
            response = await self._client.get(
                f"{endpoint_url}/healthz",
                timeout=5.0,
            )
            response.raise_for_status()
            return {
                "status": "healthy",
                "endpoint": endpoint_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except httpx.RequestError as e:
            return {
                "status": "unhealthy",
                "endpoint": endpoint_url,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a deployed model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata and configuration
        """
        if not self._initialized:
            await self.initialize()

        endpoint_url = self.config.get_endpoint_url(model_name)

        assert self._client is not None
        response = await self._client.get(
            f"{endpoint_url}/metadata",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        delay = self.config.retry_backoff_base * (2**attempt)
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        await asyncio.sleep(delay + jitter)

    def _log_trace(
        self,
        model_name: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        trace_id: Optional[str] = None,
    ) -> None:
        """Log prediction to Opik for observability."""
        try:
            import opik

            opik.track(  # type: ignore[call-arg]
                name=f"bentoml.predict.{model_name}",
                input={"model": model_name, "input_summary": str(input_data)[:200]},
                output={"prediction_summary": str(output_data)[:200]},
                metadata={
                    "latency_ms": latency_ms,
                    "trace_id": trace_id or "unknown",
                },
            )
        except Exception as e:
            logger.debug(f"Opik tracing failed (non-critical): {e}")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

# Global client instance (singleton pattern)
_bentoml_client: Optional[BentoMLClient] = None


async def get_bentoml_client() -> BentoMLClient:
    """FastAPI dependency for BentoML client.

    Returns:
        Singleton BentoML client instance

    Example:
        @app.get("/predict/{model_name}")
        async def predict(
            model_name: str,
            client: BentoMLClient = Depends(get_bentoml_client)
        ):
            return await client.predict(model_name, {...})
    """
    global _bentoml_client
    if _bentoml_client is None:
        _bentoml_client = BentoMLClient()
        await _bentoml_client.initialize()
    return _bentoml_client


async def close_bentoml_client() -> None:
    """Cleanup function for application shutdown."""
    global _bentoml_client
    if _bentoml_client:
        await _bentoml_client.close()
        _bentoml_client = None


def configure_bentoml_endpoints(endpoints: Dict[str, str]) -> None:
    """Configure model endpoint mappings.

    Args:
        endpoints: Mapping of model_name -> endpoint_url

    Example:
        configure_bentoml_endpoints({
            "churn_model": "http://churn-service:3000",
            "conversion_model": "http://conversion-service:3001",
        })
    """
    global _bentoml_client
    if _bentoml_client:
        _bentoml_client.config.model_endpoints.update(endpoints)
