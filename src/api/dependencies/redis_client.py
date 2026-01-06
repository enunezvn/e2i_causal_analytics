"""Redis client dependency for FastAPI.

Provides async Redis connection pool management for:
- Caching
- Session storage
- Rate limiting
- Pub/Sub messaging

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6382")
REDIS_SOCKET_TIMEOUT = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "3.0"))
REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", "10"))

# Global client reference
_redis_client: Optional[Redis] = None


async def init_redis() -> Redis:
    """
    Initialize Redis connection pool.

    Returns:
        Redis client instance

    Raises:
        ConnectionError: If Redis connection fails
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

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

    try:
        client = await get_redis()
        start = time.time()
        await client.ping()
        latency_ms = (time.time() - start) * 1000

        # Get info for additional metrics
        info = await client.info("clients")

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "connected_clients": info.get("connected_clients", 0),
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
