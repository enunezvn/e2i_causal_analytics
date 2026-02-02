"""
Metrics Cache for Observability Connector.

Provides caching for frequently accessed observability metrics with:
- Redis backend (primary) for distributed caching
- In-memory fallback for development/testing
- TTL-based expiration based on time window
- Cache invalidation on new span insertion

Version: 1.0.0 (Phase 3.3)

Usage:
    from src.agents.ml_foundation.observability_connector.cache import (
        MetricsCache,
        get_metrics_cache,
    )

    # Get singleton instance
    cache = get_metrics_cache()

    # Cache metrics
    await cache.set_metrics("1h", "orchestrator", {"p50": 100, "p99": 500})

    # Get cached metrics
    metrics = await cache.get_metrics("1h", "orchestrator")

    # Invalidate on new span
    await cache.invalidate("orchestrator")

Author: E2I Causal Analytics Team
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Supported cache backends."""

    REDIS = "redis"
    MEMORY = "memory"


@dataclass
class CacheConfig:
    """Configuration for metrics cache."""

    # Backend selection
    backend: CacheBackend = CacheBackend.MEMORY  # Default to memory for safety
    fallback_to_memory: bool = True  # Fall back to memory if Redis fails

    # Key prefix
    key_prefix: str = "obs_metrics"

    # TTL settings (in seconds)
    ttl_1h: int = 60  # 1 minute TTL for "1h" window
    ttl_24h: int = 300  # 5 minutes TTL for "24h" window
    ttl_7d: int = 600  # 10 minutes TTL for "7d" window
    ttl_default: int = 120  # 2 minutes default TTL

    # Memory cache settings
    max_memory_entries: int = 1000  # Max entries in memory cache
    cleanup_interval: int = 60  # Cleanup interval in seconds

    def get_ttl(self, window: str) -> int:
        """Get TTL for a specific time window."""
        ttl_map = {
            "1h": self.ttl_1h,
            "24h": self.ttl_24h,
            "7d": self.ttl_7d,
        }
        return ttl_map.get(window, self.ttl_default)


@dataclass
class CacheMetrics:
    """Metrics for cache monitoring."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    invalidations: int = 0
    errors: int = 0
    redis_failures: int = 0
    memory_fallbacks: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "invalidations": self.invalidations,
            "errors": self.errors,
            "redis_failures": self.redis_failures,
            "memory_fallbacks": self.memory_fallbacks,
            "hit_rate": round(self.hit_rate, 3),
        }


@dataclass
class CacheEntry:
    """Entry in memory cache with TTL."""

    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class MetricsCache:
    """
    Cache for observability metrics with Redis and in-memory backends.

    Provides:
    - Fast access to frequently queried metrics
    - TTL-based expiration based on time window
    - Automatic invalidation on span insertion
    - Redis primary with memory fallback

    Cache Key Pattern:
        {prefix}:{window}:{agent}
        e.g., obs_metrics:1h:orchestrator

    Example:
        cache = MetricsCache()

        # Cache quality metrics
        await cache.set_metrics("1h", "orchestrator", {
            "latency_p50": 100,
            "latency_p99": 500,
            "error_rate": 0.01
        })

        # Get cached metrics
        metrics = await cache.get_metrics("1h", "orchestrator")
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        redis_client: Optional[Any] = None,
    ):
        """Initialize metrics cache.

        Args:
            config: Cache configuration
            redis_client: Optional pre-configured Redis client (for testing)
        """
        self._config = config or CacheConfig()
        self._redis_client = redis_client
        self._redis_available = False

        # In-memory cache (for fallback or when Redis unavailable)
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_lock = asyncio.Lock()

        # Track invalidation subscriptions
        self._invalidation_callbacks: List[Any] = []

        # Metrics
        self._metrics = CacheMetrics()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._metrics

    @property
    def backend(self) -> CacheBackend:
        """Get active cache backend."""
        if self._redis_available and self._config.backend == CacheBackend.REDIS:
            return CacheBackend.REDIS
        return CacheBackend.MEMORY

    async def initialize(self) -> bool:
        """
        Initialize the cache backend.

        Returns:
            bool: True if initialization successful
        """
        if self._config.backend == CacheBackend.REDIS:
            try:
                if self._redis_client is None:
                    from src.memory.services.factories import get_redis_client

                    self._redis_client = get_redis_client()

                # Test connection
                await self._redis_client.ping()
                self._redis_available = True
                logger.info("MetricsCache: Redis backend initialized")
                return True
            except Exception as e:
                logger.warning(f"MetricsCache: Redis initialization failed: {e}")
                self._metrics.redis_failures += 1
                if self._config.fallback_to_memory:
                    logger.info("MetricsCache: Falling back to memory backend")
                    self._metrics.memory_fallbacks += 1
                    return True
                return False
        else:
            logger.info("MetricsCache: Memory backend initialized")
            return True

    def _make_key(self, window: str, agent: Optional[str] = None) -> str:
        """
        Generate cache key.

        Args:
            window: Time window (1h, 24h, 7d)
            agent: Optional agent name filter

        Returns:
            Cache key string
        """
        parts = [self._config.key_prefix, window]
        if agent:
            parts.append(agent)
        else:
            parts.append("_all")
        return ":".join(parts)

    async def get_metrics(
        self,
        window: str,
        agent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached metrics for a time window and agent.

        Args:
            window: Time window (1h, 24h, 7d)
            agent: Optional agent name filter

        Returns:
            Cached metrics dict or None if not cached
        """
        key = self._make_key(window, agent)

        try:
            # Try Redis first if available
            if self._redis_available:
                try:
                    data = await self._redis_client.get(key)
                    if data:
                        self._metrics.hits += 1
                        return json.loads(data)
                except Exception as e:
                    logger.debug(f"Redis get failed, trying memory: {e}")
                    self._metrics.redis_failures += 1

            # Fall back to memory cache
            async with self._memory_lock:
                entry = self._memory_cache.get(key)
                if entry and not entry.is_expired:
                    self._metrics.hits += 1
                    return entry.value
                elif entry and entry.is_expired:
                    del self._memory_cache[key]

            self._metrics.misses += 1
            return None

        except Exception as e:
            logger.warning(f"MetricsCache get error: {e}")
            self._metrics.errors += 1
            self._metrics.misses += 1
            return None

    async def set_metrics(
        self,
        window: str,
        agent: Optional[str],
        metrics: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache metrics for a time window and agent.

        Args:
            window: Time window (1h, 24h, 7d)
            agent: Optional agent name filter
            metrics: Metrics dictionary to cache
            ttl: Optional TTL override (uses window default if not provided)

        Returns:
            bool: True if caching successful
        """
        key = self._make_key(window, agent)
        ttl = ttl or self._config.get_ttl(window)

        try:
            # Try Redis first if available
            if self._redis_available:
                try:
                    await self._redis_client.setex(
                        key,
                        ttl,
                        json.dumps(metrics),
                    )
                    self._metrics.sets += 1
                    return True
                except Exception as e:
                    logger.debug(f"Redis set failed, using memory: {e}")
                    self._metrics.redis_failures += 1

            # Fall back to memory cache
            async with self._memory_lock:
                # Cleanup if at capacity
                if len(self._memory_cache) >= self._config.max_memory_entries:
                    self._cleanup_memory_cache_unlocked()

                self._memory_cache[key] = CacheEntry(
                    value=metrics,
                    expires_at=time.time() + ttl,
                )

            self._metrics.sets += 1
            return True

        except Exception as e:
            logger.warning(f"MetricsCache set error: {e}")
            self._metrics.errors += 1
            return False

    async def invalidate(
        self,
        agent: Optional[str] = None,
        window: Optional[str] = None,
    ) -> int:
        """
        Invalidate cached metrics.

        Args:
            agent: Optional agent to invalidate (None = all agents)
            window: Optional window to invalidate (None = all windows)

        Returns:
            int: Number of keys invalidated
        """
        count = 0
        windows = [window] if window else ["1h", "24h", "7d"]

        try:
            for w in windows:
                key = self._make_key(w, agent)

                # Invalidate in Redis
                if self._redis_available:
                    try:
                        deleted = await self._redis_client.delete(key)
                        count += deleted
                    except Exception as e:
                        logger.debug(f"Redis invalidate failed: {e}")

                # Invalidate in memory
                async with self._memory_lock:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                        count += 1

                    # Also invalidate the "all agents" key
                    if agent:
                        all_key = self._make_key(w, None)
                        if all_key in self._memory_cache:
                            del self._memory_cache[all_key]
                            count += 1

            self._metrics.invalidations += count
            return count

        except Exception as e:
            logger.warning(f"MetricsCache invalidate error: {e}")
            self._metrics.errors += 1
            return count

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching a pattern (Redis only).

        Args:
            pattern: Redis key pattern (e.g., obs_metrics:*:orchestrator)

        Returns:
            int: Number of keys invalidated
        """
        count = 0

        try:
            if self._redis_available:
                # Use SCAN to find matching keys
                cursor = 0
                while True:
                    cursor, keys = await self._redis_client.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100,
                    )
                    if keys:
                        deleted = await self._redis_client.delete(*keys)
                        count += deleted
                    if cursor == 0:
                        break

            # Invalidate matching keys in memory
            async with self._memory_lock:
                import fnmatch

                keys_to_delete = [
                    k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)
                ]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    count += 1

            self._metrics.invalidations += count
            return count

        except Exception as e:
            logger.warning(f"MetricsCache invalidate_pattern error: {e}")
            self._metrics.errors += 1
            return count

    async def get_or_compute(
        self,
        window: str,
        agent: Optional[str],
        compute_fn: Any,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get cached metrics or compute and cache them.

        Args:
            window: Time window (1h, 24h, 7d)
            agent: Optional agent name filter
            compute_fn: Async function to compute metrics if not cached
            ttl: Optional TTL override

        Returns:
            Dict[str, Any]: Cached or computed metrics
        """
        # Try cache first
        cached = await self.get_metrics(window, agent)
        if cached is not None:
            return cached

        # Compute and cache
        metrics = await compute_fn()
        await self.set_metrics(window, agent, metrics, ttl)
        return metrics

    def _cleanup_memory_cache_unlocked(self) -> int:
        """
        Clean up expired entries from memory cache.

        Note: Caller must hold self._memory_lock.

        Returns:
            int: Number of entries removed
        """
        count = 0
        now = time.time()

        expired_keys = [k for k, v in self._memory_cache.items() if v.expires_at <= now]
        for key in expired_keys:
            del self._memory_cache[key]
            count += 1

        # If still over capacity, remove oldest entries
        if len(self._memory_cache) >= self._config.max_memory_entries:
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].created_at,
            )
            to_remove = len(self._memory_cache) - int(self._config.max_memory_entries * 0.8)
            for key, _ in sorted_entries[:to_remove]:
                del self._memory_cache[key]
                count += 1

        return count

    async def _cleanup_memory_cache(self) -> int:
        """
        Clean up expired entries from memory cache.

        Returns:
            int: Number of entries removed
        """
        async with self._memory_lock:
            return self._cleanup_memory_cache_unlocked()

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("MetricsCache: Cleanup task started")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.debug("MetricsCache: Cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval)
                removed = await self._cleanup_memory_cache()
                if removed > 0:
                    logger.debug(f"MetricsCache: Cleaned up {removed} expired entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"MetricsCache cleanup error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            "backend": self.backend.value,
            "redis_available": self._redis_available,
            "memory_entries": len(self._memory_cache),
            "config": {
                "key_prefix": self._config.key_prefix,
                "ttl_1h": self._config.ttl_1h,
                "ttl_24h": self._config.ttl_24h,
                "ttl_7d": self._config.ttl_7d,
                "max_memory_entries": self._config.max_memory_entries,
            },
            "metrics": self._metrics.to_dict(),
        }

    async def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            int: Number of entries cleared
        """
        count = 0

        try:
            # Clear Redis
            if self._redis_available:
                pattern = f"{self._config.key_prefix}:*"
                count += await self.invalidate_pattern(pattern)

            # Clear memory
            async with self._memory_lock:
                count += len(self._memory_cache)
                self._memory_cache.clear()

            return count

        except Exception as e:
            logger.warning(f"MetricsCache clear error: {e}")
            return count


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_metrics_cache: Optional[MetricsCache] = None


def get_metrics_cache(config: Optional[CacheConfig] = None) -> MetricsCache:
    """
    Get the singleton MetricsCache instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        MetricsCache: Singleton instance
    """
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = MetricsCache(config=config)
    return _metrics_cache


async def reset_metrics_cache() -> None:
    """Reset the singleton instance (for testing)."""
    global _metrics_cache
    if _metrics_cache is not None:
        await _metrics_cache.stop_cleanup_task()
        _metrics_cache = None
