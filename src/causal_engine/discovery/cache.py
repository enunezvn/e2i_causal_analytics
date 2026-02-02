"""
E2I Causal Analytics - Discovery Result Cache
==============================================

Caching layer for causal discovery results with Redis + in-memory fallback.

Features:
- Data-aware hashing (DataFrame content + column order)
- Config-aware hashing (algorithm selection, thresholds)
- TTL-based expiration (default 1 hour)
- LRU eviction for in-memory cache
- Cache statistics tracking

Usage:
    cache = DiscoveryCache(redis_url="redis://localhost:6379")

    # Check cache
    cached = await cache.get(data, config)
    if cached:
        return cached

    # Run discovery and cache result
    result = await runner.discover_dag(data, config)
    await cache.set(data, config, result)

Author: E2I Causal Analytics Team
"""

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from .hasher import hash_config, hash_dataframe, make_cache_key

if TYPE_CHECKING:
    from .base import DiscoveryConfig, DiscoveryResult


@dataclass
class CacheConfig:
    """Configuration for DiscoveryCache.

    Attributes:
        redis_url: Redis connection URL (optional)
        ttl_seconds: Time-to-live for cached entries (default 1 hour)
        max_memory_items: Maximum items in in-memory cache (default 100)
        enable_redis: Whether to try Redis (default True)
        enable_memory: Whether to use in-memory fallback (default True)
    """

    redis_url: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour
    max_memory_items: int = 100
    enable_redis: bool = True
    enable_memory: bool = True


@dataclass
class CacheStats:
    """Statistics for cache performance tracking.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of cache sets
        evictions: Number of LRU evictions
        redis_errors: Number of Redis errors
        memory_items: Current in-memory item count
        last_hit_at: Timestamp of last hit
        last_miss_at: Timestamp of last miss
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    redis_errors: int = 0
    memory_items: int = 0
    last_hit_at: Optional[datetime] = None
    last_miss_at: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "redis_errors": self.redis_errors,
            "memory_items": self.memory_items,
            "hit_rate": round(self.hit_rate, 4),
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
            "last_miss_at": self.last_miss_at.isoformat() if self.last_miss_at else None,
        }


@dataclass
class CacheEntry:
    """A cached discovery result entry.

    Attributes:
        result_json: Serialized DiscoveryResult
        created_at: When the entry was created
        expires_at: When the entry expires
        hit_count: Number of times this entry was hit
    """

    result_json: str
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now(timezone.utc) > self.expires_at


class DiscoveryCache:
    """Cache for causal discovery results.

    Provides a two-tier caching system:
    1. Redis (if available) - persistent, shared across processes
    2. In-memory LRU cache - fast fallback when Redis unavailable

    Example:
        >>> cache = DiscoveryCache()
        >>> cached = await cache.get(data, config)
        >>> if cached is None:
        ...     result = await runner.discover_dag(data, config)
        ...     await cache.set(data, config, result)
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
    ):
        """Initialize DiscoveryCache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._stats = CacheStats()
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._redis_client = None
        self._redis_available = False

        if self.config.enable_redis:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection if configured."""
        try:
            import redis.asyncio as redis

            redis_url = self.config.redis_url or "redis://localhost:6379"
            self._redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
            )
            self._redis_available = True
            logger.debug(f"Redis cache initialized: {redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using memory cache only")
            self._redis_available = False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache only")
            self._redis_available = False

    async def get(
        self,
        data: pd.DataFrame,
        config: "DiscoveryConfig",
    ) -> Optional["DiscoveryResult"]:
        """Get cached discovery result.

        Args:
            data: Input DataFrame
            config: Discovery configuration

        Returns:
            Cached DiscoveryResult if found and not expired, None otherwise
        """
        cache_key = self._make_key(data, config)

        # Try Redis first
        if self._redis_available and self.config.enable_redis:
            try:
                result = await self._get_from_redis(cache_key)
                if result is not None:
                    self._stats.hits += 1
                    self._stats.last_hit_at = datetime.now(timezone.utc)
                    logger.debug(f"Cache hit (Redis): {cache_key[:32]}...")
                    return result
            except Exception as e:
                self._stats.redis_errors += 1
                logger.warning(f"Redis get error: {e}")

        # Try memory cache
        if self.config.enable_memory:
            result = self._get_from_memory(cache_key)
            if result is not None:
                self._stats.hits += 1
                self._stats.last_hit_at = datetime.now(timezone.utc)
                logger.debug(f"Cache hit (memory): {cache_key[:32]}...")
                return result

        self._stats.misses += 1
        self._stats.last_miss_at = datetime.now(timezone.utc)
        return None

    async def set(
        self,
        data: pd.DataFrame,
        config: "DiscoveryConfig",
        result: "DiscoveryResult",
    ) -> None:
        """Cache a discovery result.

        Args:
            data: Input DataFrame
            config: Discovery configuration
            result: DiscoveryResult to cache
        """
        cache_key = self._make_key(data, config)
        result_json = self._serialize_result(result)
        now = datetime.now(timezone.utc)

        # Store in Redis
        if self._redis_available and self.config.enable_redis:
            try:
                await self._set_in_redis(cache_key, result_json)
            except Exception as e:
                self._stats.redis_errors += 1
                logger.warning(f"Redis set error: {e}")

        # Store in memory
        if self.config.enable_memory:
            self._set_in_memory(cache_key, result_json, now)

        self._stats.sets += 1
        logger.debug(f"Cached result: {cache_key[:32]}...")

    def invalidate(self, data_hash: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            data_hash: If provided, only invalidate entries for this data hash.
                       If None, invalidate all entries.

        Returns:
            Number of entries invalidated
        """
        count = 0

        if data_hash is None:
            # Clear everything
            count = len(self._memory_cache)
            self._memory_cache.clear()
            self._stats.memory_items = 0
        else:
            # Clear entries matching data hash
            keys_to_remove = [key for key in self._memory_cache.keys() if data_hash in key]
            for key in keys_to_remove:
                del self._memory_cache[key]
                count += 1
            self._stats.memory_items = len(self._memory_cache)

        logger.debug(f"Invalidated {count} cache entries")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current metrics
        """
        self._stats.memory_items = len(self._memory_cache)
        return self._stats

    def _make_key(
        self,
        data: pd.DataFrame,
        config: "DiscoveryConfig",
    ) -> str:
        """Generate cache key for data and config."""
        data_hash = hash_dataframe(data)
        config_hash = hash_config(config)
        return make_cache_key(data_hash, config_hash)

    async def _get_from_redis(self, key: str) -> Optional["DiscoveryResult"]:
        """Get entry from Redis."""
        if self._redis_client is None:
            return None

        value = await self._redis_client.get(key)
        if value is None:
            return None

        return self._deserialize_result(value)

    async def _set_in_redis(self, key: str, result_json: str) -> None:
        """Set entry in Redis with TTL."""
        if self._redis_client is None:
            return

        await self._redis_client.setex(
            key,
            self.config.ttl_seconds,
            result_json,
        )

    def _get_from_memory(self, key: str) -> Optional["DiscoveryResult"]:
        """Get entry from in-memory cache."""
        if key not in self._memory_cache:
            return None

        entry = self._memory_cache[key]

        # Check expiration
        if entry.is_expired():
            del self._memory_cache[key]
            return None

        # Update LRU order
        self._memory_cache.move_to_end(key)
        entry.hit_count += 1

        return self._deserialize_result(entry.result_json)

    def _set_in_memory(
        self,
        key: str,
        result_json: str,
        now: datetime,
    ) -> None:
        """Set entry in in-memory cache with LRU eviction."""
        from datetime import timedelta

        # Evict if at capacity
        while len(self._memory_cache) >= self.config.max_memory_items:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            self._stats.evictions += 1

        # Create entry
        entry = CacheEntry(
            result_json=result_json,
            created_at=now,
            expires_at=now + timedelta(seconds=self.config.ttl_seconds),
        )

        self._memory_cache[key] = entry
        self._stats.memory_items = len(self._memory_cache)

    def _serialize_result(self, result: "DiscoveryResult") -> str:
        """Serialize DiscoveryResult to JSON."""
        return json.dumps(result.to_dict())

    def _deserialize_result(self, result_json: str) -> "DiscoveryResult":
        """Deserialize JSON to DiscoveryResult."""
        from .base import (
            DiscoveredEdge,
            DiscoveryAlgorithmType,
            DiscoveryConfig,
            DiscoveryResult,
            EdgeType,
            GateDecision,
        )

        data = json.loads(result_json)

        # Reconstruct config
        config_data = data.get("config", {})
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType(alg) for alg in config_data.get("algorithms", [])],
            alpha=config_data.get("alpha", 0.05),
            max_cond_vars=config_data.get("max_cond_vars"),
            ensemble_threshold=config_data.get("ensemble_threshold", 0.5),
            max_iter=config_data.get("max_iter", 10000),
            random_state=config_data.get("random_state", 42),
            score_func=config_data.get("score_func", "local_score_BIC"),
            assume_linear=config_data.get("assume_linear", True),
            assume_gaussian=config_data.get("assume_gaussian", False),
        )

        # Reconstruct edges
        edges = []
        for edge_data in data.get("edges", []):
            edges.append(
                DiscoveredEdge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    edge_type=EdgeType(edge_data.get("edge_type", "directed")),
                    confidence=edge_data.get("confidence", 1.0),
                    algorithm_votes=edge_data.get("algorithm_votes", 1),
                    algorithms=edge_data.get("algorithms", []),
                )
            )

        # Reconstruct gate decision
        gate_decision = None
        if data.get("gate_decision"):
            gate_decision = GateDecision(data["gate_decision"])

        return DiscoveryResult(
            success=data.get("success", True),
            config=config,
            edges=edges,
            gate_decision=gate_decision,
            gate_confidence=data.get("gate_confidence", 0.0),
            metadata=data.get("metadata", {}),
        )


# Singleton cache instance
_default_cache: Optional[DiscoveryCache] = None


def get_discovery_cache(config: Optional[CacheConfig] = None) -> DiscoveryCache:
    """Get or create the default DiscoveryCache instance.

    Args:
        config: Optional configuration for new cache

    Returns:
        DiscoveryCache singleton instance
    """
    global _default_cache

    if _default_cache is None:
        _default_cache = DiscoveryCache(config)

    return _default_cache
