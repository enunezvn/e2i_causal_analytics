"""
Simulation Cache
================

Redis-based caching layer for digital twin simulation results.
Provides caching of frequently simulated interventions to reduce
computation time for repeated queries.

Features:
    - Cache simulation results by intervention config + population filters
    - TTL-based expiration
    - Cache invalidation on model updates
    - Cache statistics tracking

Version: 1.0.0
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from .models.simulation_models import (
    InterventionConfig,
    PopulationFilter,
    SimulationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationCacheConfig:
    """Configuration for simulation result caching."""

    ttl_seconds: int = 1800  # 30 minutes default
    prefix: str = "twin_sim"
    enabled: bool = True
    max_cached_results: int = 1000  # LRU eviction threshold


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SimulationCache:
    """
    Redis-based cache for simulation results.

    Caches frequently simulated intervention configurations to reduce
    computation time for repeated queries. Cache keys are generated
    from the intervention config, population filters, and model ID.

    Example:
        cache = SimulationCache()

        # Check cache before simulation
        cached = await cache.get_cached_result(config, filters, model_id)
        if cached:
            return cached

        # Run simulation and cache result
        result = engine.simulate(config, filters)
        await cache.cache_result(result)
    """

    def __init__(
        self,
        redis_client=None,
        config: Optional[SimulationCacheConfig] = None,
    ):
        """
        Initialize simulation cache.

        Args:
            redis_client: Redis client (optional, uses factory if not provided)
            config: Cache configuration
        """
        self._client = redis_client
        self.config = config or SimulationCacheConfig()
        self._stats = CacheStats()
        self._initialized = False

        logger.info(
            f"Initialized SimulationCache (enabled={self.config.enabled}, "
            f"ttl={self.config.ttl_seconds}s)"
        )

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
        intervention_config: InterventionConfig,
        population_filter: Optional[PopulationFilter],
        model_id: UUID,
    ) -> str:
        """
        Generate a unique cache key from simulation parameters.

        Args:
            intervention_config: Intervention configuration
            population_filter: Population filters
            model_id: Twin model ID

        Returns:
            Hash-based cache key
        """
        # Build deterministic representation
        params = {
            "model_id": str(model_id),
            "intervention_type": intervention_config.intervention_type,
            "channel": intervention_config.channel,
            "frequency": intervention_config.frequency,
            "duration_weeks": intervention_config.duration_weeks,
            "intensity_multiplier": intervention_config.intensity_multiplier,
            "target_deciles": sorted(intervention_config.target_deciles),
            "target_specialties": sorted(intervention_config.target_specialties),
            "target_regions": sorted(intervention_config.target_regions),
        }

        # Add population filter parameters
        if population_filter:
            params["filter_specialties"] = sorted(population_filter.specialties)
            params["filter_deciles"] = sorted(population_filter.deciles)
            params["filter_regions"] = sorted(population_filter.regions)
            params["filter_adoption_stages"] = sorted(population_filter.adoption_stages)
            params["min_baseline"] = population_filter.min_baseline_outcome
            params["max_baseline"] = population_filter.max_baseline_outcome

        # Create hash
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:24]

        return f"{intervention_config.intervention_type}:{hash_val}"

    async def get_cached_result(
        self,
        intervention_config: InterventionConfig,
        population_filter: Optional[PopulationFilter],
        model_id: UUID,
    ) -> Optional[SimulationResult]:
        """
        Get cached simulation result if available.

        Args:
            intervention_config: Intervention configuration
            population_filter: Population filters
            model_id: Twin model ID

        Returns:
            Cached SimulationResult or None if not found/expired
        """
        if not self.config.enabled:
            return None

        client = await self._get_client()
        if client is None:
            return None

        cache_key = self._generate_cache_key(intervention_config, population_filter, model_id)
        full_key = self._make_key(cache_key)

        try:
            data = await client.get(full_key)
            if data is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                self._stats.misses += 1
                return None

            # Deserialize
            result_dict = pickle.loads(data.encode("latin-1") if isinstance(data, str) else data)
            result = SimulationResult(**result_dict)

            logger.debug(f"Cache hit for key: {cache_key}")
            self._stats.hits += 1

            # Update hit count metadata
            await client.hincrby(f"{full_key}:meta", "hit_count", 1)

            return result

        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            self._stats.errors += 1
            return None

    async def cache_result(
        self,
        result: SimulationResult,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Cache a simulation result.

        Args:
            result: SimulationResult to cache
            ttl_seconds: Time to live (uses default if not specified)

        Returns:
            True if cached successfully
        """
        if not self.config.enabled:
            return False

        client = await self._get_client()
        if client is None:
            return False

        cache_key = self._generate_cache_key(
            result.intervention_config,
            result.population_filters,
            result.model_id,
        )
        full_key = self._make_key(cache_key)
        ttl = ttl_seconds or self.config.ttl_seconds

        try:
            # Serialize result to dict for pickle
            result_dict = result.model_dump(mode="json")
            data = pickle.dumps(result_dict).decode("latin-1")

            # Store with TTL
            await client.setex(full_key, ttl, data)

            # Store metadata
            now = datetime.now(timezone.utc)
            meta = {
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
                "intervention_type": result.intervention_config.intervention_type,
                "model_id": str(result.model_id),
                "twin_count": result.twin_count,
                "ate": str(result.simulated_ate),
                "hit_count": 0,
            }
            await client.hset(f"{full_key}:meta", mapping=meta)
            await client.expire(f"{full_key}:meta", ttl)

            logger.debug(
                f"Cached simulation result with key: {cache_key} "
                f"(ATE={result.simulated_ate:.4f}, ttl={ttl}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache simulation result: {e}")
            self._stats.errors += 1
            return False

    async def invalidate_model_cache(self, model_id: UUID) -> int:
        """
        Invalidate all cached results for a specific model.

        Called when a model is updated or retrained to ensure
        stale predictions are not served.

        Args:
            model_id: Model ID to invalidate cache for

        Returns:
            Number of keys deleted
        """
        client = await self._get_client()
        if client is None:
            return 0

        pattern = self._make_key("*")

        try:
            keys_to_delete = []

            async for key in client.scan_iter(match=pattern):
                # Check if this key belongs to the model
                if key.endswith(":meta"):
                    meta = await client.hgetall(key)
                    if meta and meta.get("model_id") == str(model_id):
                        # Add both data key and meta key
                        data_key = key[:-5]  # Remove ":meta" suffix
                        keys_to_delete.extend([data_key, key])

            if keys_to_delete:
                await client.delete(*keys_to_delete)
                count = len(keys_to_delete) // 2  # Count actual results (not meta)
                self._stats.invalidations += count
                logger.info(f"Invalidated {count} cached simulations for model {model_id}")
                return count

            return 0

        except Exception as e:
            logger.error(f"Failed to invalidate model cache: {e}")
            self._stats.errors += 1
            return 0

    async def invalidate_all(self) -> int:
        """
        Clear all cached simulation results.

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
                count = len([k for k in keys if not k.endswith(":meta")])
                self._stats.invalidations += count
                logger.info(f"Invalidated all {count} cached simulations")
                return count

            return 0

        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            self._stats.errors += 1
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats including hit rate
        """
        client = await self._get_client()

        stats = {
            "enabled": self.config.enabled,
            "ttl_seconds": self.config.ttl_seconds,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "invalidations": self._stats.invalidations,
            "errors": self._stats.errors,
        }

        if client is None:
            stats["redis_status"] = "unavailable"
            return stats

        # Count cached entries
        pattern = self._make_key("*:meta")

        try:
            entry_count = 0
            total_hits = 0
            by_intervention = {}

            async for key in client.scan_iter(match=pattern):
                meta = await client.hgetall(key)
                if meta:
                    entry_count += 1
                    total_hits += int(meta.get("hit_count", 0))

                    intervention = meta.get("intervention_type", "unknown")
                    if intervention not in by_intervention:
                        by_intervention[intervention] = {"count": 0, "hits": 0}
                    by_intervention[intervention]["count"] += 1
                    by_intervention[intervention]["hits"] += int(meta.get("hit_count", 0))

            stats["redis_status"] = "connected"
            stats["cached_entries"] = entry_count
            stats["total_redis_hits"] = total_hits
            stats["by_intervention_type"] = by_intervention

        except Exception as e:
            stats["redis_status"] = "error"
            stats["error"] = str(e)

        return stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()
        logger.debug("Reset cache statistics")


def get_simulation_cache(
    redis_client=None,
    config: Optional[SimulationCacheConfig] = None,
) -> SimulationCache:
    """
    Get a SimulationCache instance.

    Args:
        redis_client: Optional Redis client
        config: Optional cache configuration

    Returns:
        SimulationCache instance
    """
    return SimulationCache(redis_client, config)
