"""
Data Cache - Phase 1: Data Loading Foundation

Redis-based caching layer for ML data loading:
- Cache DataFrames for repeated experiments
- Configurable TTL for cache expiration
- Automatic serialization/deserialization
- Cache invalidation on data updates

Version: 1.0.0
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for data caching."""

    ttl_seconds: int = 3600  # 1 hour default
    prefix: str = "ml_data"
    serialize_format: str = "pickle"  # pickle or json
    compression: bool = True


@dataclass
class CacheEntry:
    """Cached data entry with metadata."""

    data: Any
    created_at: str
    expires_at: str
    key: str
    hit_count: int = 0


class DataCache:
    """
    Redis-based cache for ML data.

    Provides:
    - DataFrame caching with automatic serialization
    - TTL-based expiration
    - Cache key generation from query parameters
    - Cache invalidation

    Example:
        cache = DataCache()

        # Cache a DataFrame
        await cache.set("my_key", df, ttl_seconds=3600)

        # Get from cache
        df = await cache.get("my_key")

        # Use decorator for automatic caching
        @cache.cached(ttl_seconds=3600)
        async def load_data(table, filters):
            return await loader.load_for_training(table, filters)
    """

    def __init__(
        self,
        redis_client=None,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize data cache.

        Args:
            redis_client: Redis client (optional, uses factory if not provided)
            config: Cache configuration
        """
        self._client = redis_client
        self.config = config or CacheConfig()
        self._initialized = False

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                from src.memory.services.factories import get_redis_client

                self._client = get_redis_client()
            except Exception as e:
                logger.warning(f"Could not get Redis client: {e}")
                return None
        return self._client

    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.config.prefix}:{key}"

    def _generate_cache_key(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a cache key from query parameters.

        Args:
            table: Table name
            filters: Query filters
            **kwargs: Additional parameters

        Returns:
            Hash-based cache key
        """
        # Create deterministic string from parameters
        params = {
            "table": table,
            "filters": filters or {},
            **kwargs,
        }
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        return f"{table}:{hash_val}"

    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get cached DataFrame.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None if not found
        """
        client = await self._get_client()
        if client is None:
            return None

        full_key = self._make_key(key)

        try:
            data = await client.get(full_key)
            if data is None:
                logger.debug(f"Cache miss for key: {key}")
                return None

            # Deserialize
            if self.config.serialize_format == "pickle":
                df = pickle.loads(data.encode("latin-1") if isinstance(data, str) else data)
            else:
                df = pd.read_json(data)

            logger.debug(f"Cache hit for key: {key}")

            # Increment hit count
            await client.hincrby(f"{full_key}:meta", "hit_count", 1)

            return df

        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            return None

    async def set(
        self,
        key: str,
        df: pd.DataFrame,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Cache a DataFrame.

        Args:
            key: Cache key
            df: DataFrame to cache
            ttl_seconds: Time to live (uses default if not specified)

        Returns:
            True if cached successfully
        """
        client = await self._get_client()
        if client is None:
            return False

        full_key = self._make_key(key)
        ttl = ttl_seconds or self.config.ttl_seconds

        try:
            # Serialize
            if self.config.serialize_format == "pickle":
                data = pickle.dumps(df).decode("latin-1")
            else:
                data = df.to_json()

            # Store with TTL
            await client.setex(full_key, ttl, data)

            # Store metadata
            now = datetime.now()
            meta = {
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "hit_count": 0,
            }
            await client.hset(f"{full_key}:meta", mapping=meta)
            await client.expire(f"{full_key}:meta", ttl)

            logger.debug(f"Cached {len(df)} rows with key: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete cached data.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        client = await self._get_client()
        if client is None:
            return False

        full_key = self._make_key(key)

        try:
            await client.delete(full_key, f"{full_key}:meta")
            logger.debug(f"Deleted cache key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from cache: {e}")
            return False

    async def invalidate_table(self, table: str) -> int:
        """
        Invalidate all cached data for a table.

        Args:
            table: Table name

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        if client is None:
            return 0

        pattern = self._make_key(f"{table}:*")

        try:
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for table: {table}")

            return len(keys)
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0

    async def clear_all(self) -> int:
        """
        Clear all cached ML data.

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        if client is None:
            return 0

        pattern = self._make_key("*")

        try:
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")

            return len(keys)
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        client = await self._get_client()
        if client is None:
            return {"status": "unavailable"}

        pattern = self._make_key("*:meta")

        try:
            stats = {
                "total_entries": 0,
                "total_hits": 0,
                "tables": {},
            }

            async for key in client.scan_iter(match=pattern):
                meta = await client.hgetall(key)
                if meta:
                    stats["total_entries"] += 1
                    stats["total_hits"] += int(meta.get("hit_count", 0))

                    # Extract table name from key
                    # Key format: ml_data:table:hash:meta
                    parts = key.split(":")
                    if len(parts) >= 3:
                        table = parts[1]
                        if table not in stats["tables"]:
                            stats["tables"][table] = {"entries": 0, "hits": 0}
                        stats["tables"][table]["entries"] += 1
                        stats["tables"][table]["hits"] += int(meta.get("hit_count", 0))

            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    def cached(
        self,
        ttl_seconds: Optional[int] = None,
        key_func: Optional[Callable[..., str]] = None,
    ):
        """
        Decorator for caching async function results.

        Args:
            ttl_seconds: Cache TTL
            key_func: Custom key generation function

        Returns:
            Decorator function

        Example:
            @cache.cached(ttl_seconds=3600)
            async def load_data(table: str, filters: Dict):
                return await loader.load_for_training(table, filters)
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            async def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Default: use function name and args hash
                    arg_str = json.dumps(
                        {"args": [str(a) for a in args], "kwargs": kwargs}, sort_keys=True
                    )
                    key = f"{func.__name__}:{hashlib.sha256(arg_str.encode()).hexdigest()[:16]}"

                # Try to get from cache
                cached = await self.get(key)
                if cached is not None:
                    return cached

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    await self.set(key, result, ttl_seconds)

                return result

            return wrapper

        return decorator


# Convenience function
def get_data_cache(redis_client=None, config: Optional[CacheConfig] = None) -> DataCache:
    """Get a DataCache instance."""
    return DataCache(redis_client, config)
