"""Redis client dependency for FastAPI.

Provides async Redis connection pool management for:
- Caching
- Session storage
- Rate limiting
- Pub/Sub messaging

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6382")
REDIS_SOCKET_TIMEOUT = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "3.0"))
REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", "10"))

# Global client reference
_redis_client: Optional[Redis] = None

# Circuit breaker for health checks
_health_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=3, reset_timeout_seconds=30.0)
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
async def init_redis() -> Redis:
    """
    Initialize Redis connection pool.

    Returns:
        Redis client instance

    Raises:
        ConnectionError: If Redis connection fails after retries
    """
    global _redis_client

    # Reset stale reference so retries don't short-circuit
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None

    logger.info(f"Initializing Redis connection to {REDIS_URL}")

    try:
        _redis_client = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            max_connections=REDIS_MAX_CONNECTIONS,
        )

        # Verify connection
        await _redis_client.ping()
        logger.info("Redis connection established successfully")

        return _redis_client

    except Exception as e:
        _redis_client = None
        logger.error(f"Failed to connect to Redis: {e}")
        raise ConnectionError(f"Redis connection failed: {e}") from e


async def get_redis() -> Redis:
    """
    Get Redis client instance.

    Returns:
        Redis client (initializes if needed)

    Raises:
        RuntimeError: If Redis is not initialized
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = await init_redis()

    return _redis_client


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis_client

    if _redis_client is not None:
        logger.info("Closing Redis connection")
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


async def redis_health_check() -> dict:
    """
    Check Redis health status.

    Returns:
        Dict with status and latency
    """
    import time

    if not _health_circuit_breaker.allow_request():
        return {"status": "circuit_open"}

    try:
        client = await get_redis()
        start = time.time()
        await client.ping()
        latency_ms = (time.time() - start) * 1000

        # Get info for additional metrics
        info = await client.info("clients")

        _health_circuit_breaker.record_success()

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "connected_clients": info.get("connected_clients", 0),
        }

    except Exception as e:
        _health_circuit_breaker.record_failure()
        return {
            "status": "unhealthy",
            "error": str(e),
        }
