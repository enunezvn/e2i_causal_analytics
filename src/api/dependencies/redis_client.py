"""Redis client dependency for FastAPI.

Provides async Redis connection pool management for:
- Caching
- Session storage
- Rate limiting
- Pub/Sub messaging

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import logging
import os
import time
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


async def _discard(client: Redis) -> None:
    """Close a client that was never published (best effort)."""
    try:
        await client.aclose()
    except Exception as e:  # noqa: BLE001 - the attempt is already over
        logger.debug(f"Closing an unpublished Redis client failed: {e}")


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
    existing = _redis_client
    if existing is not None:
        try:
            await existing.ping()  # type: ignore[misc]
            return existing
        except Exception:
            # #1999: clear only the client this call found; another initialiser
            # may have published a working one during the ping.
            if _redis_client is existing:
                _redis_client = None

    logger.info(f"Initializing Redis connection to {REDIS_URL}")

    # #1999: the candidate is published only after its ping succeeds. Request
    # paths read the module client concurrently (current_client(), and the
    # background reconnect runs while requests are served), so an unverified
    # client must never be visible, and a failed or cancelled attempt must not
    # clear a client someone else published meanwhile.
    candidate: Optional[Redis] = None
    try:
        candidate = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            max_connections=REDIS_MAX_CONNECTIONS,
        )

        # Verify connection
        await candidate.ping()  # type: ignore[misc]
    except BaseException as e:
        if candidate is not None:
            await _discard(candidate)
        if not isinstance(e, Exception):
            raise  # cancellation: nothing was published
        logger.error(f"Failed to connect to Redis: {e}")
        raise ConnectionError(f"Redis connection failed: {e}") from e

    if _redis_client is not None:
        # Another initialiser published first; keep one pool.
        await _discard(candidate)
        return _redis_client
    _redis_client = candidate
    logger.info("Redis connection established successfully")
    return _redis_client


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


def current_client() -> Optional[Redis]:
    """The client the lifespan initialised, or ``None``. Never connects or waits.

    For request paths: ``get_redis()`` runs ``init_redis()``'s multi-attempt
    backoff when the client is unset (startup in Redis-degraded mode), which
    measured 16 s per call against a refused Redis (#1999).
    """
    return _redis_client


# Per-process background reconnect (#1999): request paths that find no client
# degrade at once and schedule ONE connection attempt here, so a Redis that
# comes back makes them durable again without a request waiting on it. One
# attempt per round and a cooldown between failed rounds bound the probing to
# one connect per RECONNECT_COOLDOWN_SECONDS per worker while Redis stays down.
RECONNECT_COOLDOWN_SECONDS = 30.0
_reconnect_task: "Optional[asyncio.Task[None]]" = None
_reconnect_failed_at: Optional[float] = None


async def _reconnect_once() -> None:
    global _reconnect_failed_at

    try:
        await init_redis.retry_with(stop=stop_after_attempt(1))()
    except Exception as e:
        _reconnect_failed_at = time.monotonic()
        logger.warning(
            f"Background Redis reconnect failed; next attempt in "
            f"{RECONNECT_COOLDOWN_SECONDS:g}s at the earliest: {e}"
        )
        return
    _reconnect_failed_at = None
    logger.info("Background Redis reconnect succeeded; request paths are durable again")


def _schedule_reconnect() -> None:
    global _reconnect_task

    if _redis_client is not None:
        return
    loop = asyncio.get_running_loop()
    task = _reconnect_task
    if task is not None and not task.done() and task.get_loop() is loop:
        return  # single-flight
    if (
        _reconnect_failed_at is not None
        and time.monotonic() - _reconnect_failed_at < RECONNECT_COOLDOWN_SECONDS
    ):
        return
    _reconnect_task = loop.create_task(_reconnect_once(), name="redis-background-reconnect")


async def request_path_client() -> Redis:
    """``current_client()`` for request paths that degrade without Redis.

    Returns the initialised client, or schedules the background reconnect and
    raises ``RuntimeError`` (which every durable store and ``InflightLock``
    already treat as "Redis unavailable, degrade") without waiting.
    """
    client = current_client()
    if client is None:
        _schedule_reconnect()
        raise RuntimeError("Redis client not initialised (startup degraded mode)")
    return client


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis_client, _reconnect_task, _reconnect_failed_at

    # Cancel a pending background reconnect FIRST: one whose ping succeeds after
    # this close would otherwise publish a new client on a shutting-down worker.
    task = _reconnect_task
    _reconnect_task = None
    _reconnect_failed_at = None
    if task is not None and not task.done() and task.get_loop() is asyncio.get_running_loop():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

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
        await client.ping()  # type: ignore[misc]
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
