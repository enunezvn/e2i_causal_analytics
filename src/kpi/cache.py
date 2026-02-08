"""
KPI Cache

Redis-based caching layer for expensive KPI calculations.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, cast

logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:
    redis = None  # type: ignore

from src.kpi.models import KPIResult, KPIStatus


class KPICache:
    """Redis-based cache for KPI calculation results."""

    DEFAULT_TTL_SECONDS = 300  # 5 minutes default
    KEY_PREFIX = "kpi:result:"

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl: int = DEFAULT_TTL_SECONDS,
    ):
        """Initialize KPI cache.

        Args:
            redis_url: Redis connection URL. If None, uses REDIS_URL env var.
            default_ttl: Default time-to-live in seconds for cached results.
        """
        self._redis: "redis.Redis | None" = None
        self._default_ttl = default_ttl
        self._enabled = False

        if redis is None:
            logger.warning("Redis not installed. KPI caching disabled.")
            return

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.from_url(url, decode_responses=True)
            # Test connection
            self._redis.ping()
            self._enabled = True
            logger.info(f"KPI cache connected to Redis: {url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self._redis = None

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled and self._redis is not None

    def _make_key(self, kpi_id: str, **context: Any) -> str:
        """Create a cache key for a KPI result.

        Args:
            kpi_id: The KPI identifier
            **context: Additional context (e.g., brand, date_range)
        """
        parts = [self.KEY_PREFIX, kpi_id]
        if context:
            # Sort for consistent keys
            for key in sorted(context.keys()):
                parts.append(f"{key}:{context[key]}")
        return ":".join(parts)

    def get(self, kpi_id: str, **context: Any) -> KPIResult | None:
        """Get a cached KPI result.

        Args:
            kpi_id: The KPI identifier
            **context: Additional context for the cache key

        Returns:
            Cached KPIResult if found and not expired, None otherwise.
        """
        if not self.enabled:
            return None

        key = self._make_key(kpi_id, **context)
        try:
            data: Optional[str] = cast(Optional[str], self._redis.get(key))  # type: ignore[union-attr]
            if data is None:
                return None

            result_dict = json.loads(data)
            result = KPIResult(
                kpi_id=result_dict["kpi_id"],
                value=result_dict.get("value"),
                status=KPIStatus(result_dict.get("status", "unknown")),
                calculated_at=datetime.fromisoformat(result_dict["calculated_at"]),
                cached=True,
                cache_expires_at=datetime.fromisoformat(result_dict["cache_expires_at"])
                if result_dict.get("cache_expires_at")
                else None,
                error=result_dict.get("error"),
                metadata=result_dict.get("metadata", {}),
                causal_library_used=result_dict.get("causal_library_used"),
                confidence_interval=tuple(result_dict["confidence_interval"])
                if result_dict.get("confidence_interval")
                else None,
                p_value=result_dict.get("p_value"),
                effect_size=result_dict.get("effect_size"),
            )
            logger.debug(f"Cache hit for KPI {kpi_id}")
            return result

        except Exception as e:
            logger.warning(f"Cache get failed for {kpi_id}: {e}")
            return None

    def set(
        self,
        result: KPIResult,
        ttl: int | None = None,
        **context: Any,
    ) -> bool:
        """Cache a KPI result.

        Args:
            result: The KPIResult to cache
            ttl: Time-to-live in seconds (uses default if None)
            **context: Additional context for the cache key

        Returns:
            True if cached successfully, False otherwise.
        """
        if not self.enabled:
            return False

        key = self._make_key(result.kpi_id, **context)
        ttl_seconds = ttl or self._default_ttl

        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

            result_dict = {
                "kpi_id": result.kpi_id,
                "value": result.value,
                "status": result.status.value
                if isinstance(result.status, KPIStatus)
                else result.status,
                "calculated_at": result.calculated_at.isoformat(),
                "cache_expires_at": expires_at.isoformat(),
                "error": result.error,
                "metadata": result.metadata,
                "causal_library_used": result.causal_library_used,
                "confidence_interval": list(result.confidence_interval)
                if result.confidence_interval
                else None,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
            }

            self._redis.setex(  # type: ignore
                key,
                ttl_seconds,
                json.dumps(result_dict, default=str),
            )
            logger.debug(f"Cached KPI {result.kpi_id} with TTL {ttl_seconds}s")
            return True

        except Exception as e:
            logger.warning(f"Cache set failed for {result.kpi_id}: {e}")
            return False

    def invalidate(self, kpi_id: str, **context: Any) -> bool:
        """Invalidate a cached KPI result.

        Args:
            kpi_id: The KPI identifier
            **context: Additional context for the cache key

        Returns:
            True if invalidated successfully, False otherwise.
        """
        if not self.enabled:
            return False

        key = self._make_key(kpi_id, **context)
        try:
            self._redis.delete(key)  # type: ignore
            logger.debug(f"Invalidated cache for KPI {kpi_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache invalidation failed for {kpi_id}: {e}")
            return False

    def invalidate_all(self) -> int:
        """Invalidate all cached KPI results.

        Returns:
            Number of keys invalidated.
        """
        if not self.enabled:
            return 0

        try:
            pattern = f"{self.KEY_PREFIX}*"
            keys: List[str] = cast(List[str], self._redis.keys(pattern))  # type: ignore[union-attr]
            if keys:
                count: int = cast(int, self._redis.delete(*keys))  # type: ignore[union-attr]
                logger.info(f"Invalidated {count} cached KPI results")
                return count
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidate_all failed: {e}")
            return 0

    def get_ttl(self, kpi_id: str, **context: Any) -> int | None:
        """Get remaining TTL for a cached result.

        Returns:
            Remaining TTL in seconds, or None if not cached.
        """
        if not self.enabled:
            return None

        key = self._make_key(kpi_id, **context)
        try:
            ttl: int = cast(int, self._redis.ttl(key))  # type: ignore[union-attr]
            return ttl if ttl > 0 else None
        except Exception:
            return None

    def size(self) -> int:
        """Get the number of cached KPI results.

        Returns:
            Number of cached KPI entries.
        """
        if not self.enabled:
            return 0

        try:
            pattern = f"{self.KEY_PREFIX}*"
            keys: List[str] = cast(List[str], self._redis.keys(pattern))  # type: ignore[union-attr]
            return len(keys) if keys else 0
        except Exception as e:
            logger.warning(f"Cache size check failed: {e}")
            return 0
