"""
E2I Agentic Memory - LangGraph Checkpointer
Redis-backed checkpointer for LangGraph workflow state persistence.

This module provides a factory function for creating LangGraph checkpointers
that persist workflow state to Redis. Falls back to in-memory storage if
Redis is not available.

Usage:
    from src.memory.langgraph_saver import create_checkpointer

    # Create checkpointer for workflow
    checkpointer = create_checkpointer()

    # Use in LangGraph workflow compilation
    workflow = StateGraph(MyState)
    # ... add nodes and edges ...
    compiled = workflow.compile(checkpointer=checkpointer)
"""

import os
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def create_checkpointer(
    redis_url: Optional[str] = None,
    fallback_to_memory: bool = True
):
    """
    Create a LangGraph checkpointer.

    Attempts to create a Redis-backed checkpointer. If Redis is not available
    and fallback_to_memory is True, falls back to an in-memory checkpointer.

    Args:
        redis_url: Redis connection URL. If not provided, uses REDIS_URL env var.
        fallback_to_memory: If True, fall back to MemorySaver when Redis unavailable.

    Returns:
        Checkpointer: RedisSaver or MemorySaver instance

    Raises:
        ImportError: If required packages are not installed and fallback is disabled
        ConnectionError: If Redis connection fails and fallback is disabled
    """
    url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")

    try:
        from langgraph.checkpoint.redis import RedisSaver

        checkpointer = RedisSaver.from_conn_string(url)
        logger.info(f"Created RedisSaver checkpointer for {url.split('@')[-1]}")
        return checkpointer

    except ImportError as e:
        if not fallback_to_memory:
            raise ImportError(
                "langgraph-checkpoint-redis is required. "
                "Install with: pip install langgraph-checkpoint-redis"
            ) from e

        logger.warning(
            "langgraph-checkpoint-redis not installed, "
            "falling back to in-memory checkpointer"
        )

    except Exception as e:
        if not fallback_to_memory:
            raise ConnectionError(
                f"Failed to connect to Redis at {url}: {e}"
            ) from e

        logger.warning(f"Redis connection failed, falling back to in-memory: {e}")

    # Fall back to memory checkpointer
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using in-memory MemorySaver checkpointer")
    return MemorySaver()


def create_async_checkpointer(
    redis_url: Optional[str] = None,
    fallback_to_memory: bool = True
):
    """
    Create an async LangGraph checkpointer.

    Similar to create_checkpointer but uses async Redis client.

    Args:
        redis_url: Redis connection URL. If not provided, uses REDIS_URL env var.
        fallback_to_memory: If True, fall back to MemorySaver when Redis unavailable.

    Returns:
        Checkpointer: AsyncRedisSaver or MemorySaver instance
    """
    url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")

    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        checkpointer = AsyncRedisSaver.from_conn_string(url)
        logger.info(f"Created AsyncRedisSaver checkpointer for {url.split('@')[-1]}")
        return checkpointer

    except ImportError as e:
        if not fallback_to_memory:
            raise ImportError(
                "langgraph-checkpoint-redis is required. "
                "Install with: pip install langgraph-checkpoint-redis"
            ) from e

        logger.warning(
            "langgraph-checkpoint-redis not installed, "
            "falling back to in-memory checkpointer"
        )

    except Exception as e:
        if not fallback_to_memory:
            raise ConnectionError(
                f"Failed to connect to Redis at {url}: {e}"
            ) from e

        logger.warning(f"Redis connection failed, falling back to in-memory: {e}")

    # Fall back to memory checkpointer
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using in-memory MemorySaver checkpointer (async fallback)")
    return MemorySaver()


class CheckpointerConfig:
    """
    Configuration for checkpointer creation.

    Attributes:
        redis_url: Redis connection URL
        checkpoint_prefix: Prefix for checkpoint keys in Redis
        ttl_seconds: TTL for checkpoint entries (None = no expiry)
        fallback_to_memory: Whether to fall back to in-memory when Redis unavailable
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        checkpoint_prefix: str = "e2i:checkpoint:",
        ttl_seconds: Optional[int] = 86400,
        fallback_to_memory: bool = True
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.checkpoint_prefix = checkpoint_prefix
        self.ttl_seconds = ttl_seconds
        self.fallback_to_memory = fallback_to_memory

    def create_checkpointer(self):
        """Create a checkpointer using this configuration."""
        return create_checkpointer(
            redis_url=self.redis_url,
            fallback_to_memory=self.fallback_to_memory
        )

    def create_async_checkpointer(self):
        """Create an async checkpointer using this configuration."""
        return create_async_checkpointer(
            redis_url=self.redis_url,
            fallback_to_memory=self.fallback_to_memory
        )
