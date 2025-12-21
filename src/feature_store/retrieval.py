"""
Feature Retrieval Service

Handles online and offline feature retrieval with Redis caching.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from supabase import Client

from .models import EntityFeatures

logger = logging.getLogger(__name__)


class FeatureRetriever:
    """Handles feature retrieval with caching."""

    def __init__(
        self,
        supabase: Client,
        redis_client: Optional[redis.Redis],
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize Feature Retriever.

        Args:
            supabase: Supabase client
            redis_client: Redis client (optional)
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.supabase = supabase
        self.redis_client = redis_client
        self.cache_ttl_seconds = cache_ttl_seconds

    def get_entity_features(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        include_stale: bool = True,
        use_cache: bool = True,
    ) -> EntityFeatures:
        """
        Get all features for a specific entity.

        Flow:
        1. Check Redis cache (if enabled)
        2. If miss, query Supabase
        3. Cache results in Redis
        4. Return EntityFeatures

        Args:
            entity_values: Entity key-value pairs
            feature_group: Optional feature group filter
            feature_names: Optional list of specific features
            include_stale: Include stale features
            use_cache: Use Redis cache

        Returns:
            EntityFeatures object
        """
        # Generate cache key
        cache_key = self._generate_cache_key(entity_values, feature_group, feature_names)

        # Try cache first
        if use_cache and self.redis_client:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache HIT for entity: {entity_values}")
                return cached

        logger.debug(f"Cache MISS for entity: {entity_values}")

        # Query Supabase
        features_data = self._query_supabase_features(
            entity_values, feature_group, feature_names, include_stale
        )

        # Build EntityFeatures object
        entity_features = self._build_entity_features(entity_values, features_data)

        # Cache results
        if use_cache and self.redis_client:
            self._set_in_cache(cache_key, entity_features)

        return entity_features

    def get_historical_features(
        self,
        entity_values: Dict[str, Any],
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical feature values (time-series).

        Args:
            entity_values: Entity key-value pairs
            feature_names: List of feature names
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of historical feature records
        """
        # Build query
        query = (
            self.supabase.table("feature_values")
            .select(
                """
                *,
                features!inner(name, feature_groups!inner(name))
            """
            )
            .eq("entity_values", json.dumps(entity_values))
        )

        # Add time filters
        if start_time:
            query = query.gte("event_timestamp", start_time.isoformat())
        if end_time:
            query = query.lte("event_timestamp", end_time.isoformat())

        # Execute query
        response = query.order("event_timestamp", desc=True).execute()

        # Filter by feature names if specified
        results = []
        for record in response.data:
            feature_name = record["features"]["name"]
            if feature_name in feature_names:
                results.append(
                    {
                        "feature_name": feature_name,
                        "feature_group": record["features"]["feature_groups"]["name"],
                        "value": record["value"],
                        "event_timestamp": record["event_timestamp"],
                        "freshness_status": record["freshness_status"],
                    }
                )

        return results

    def _query_supabase_features(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str],
        feature_names: Optional[List[str]],
        include_stale: bool,
    ) -> List[Dict[str, Any]]:
        """Query Supabase for latest feature values."""
        try:
            # Use stored function for efficient retrieval
            rpc_params = {
                "p_entity_values": json.dumps(entity_values),
                "p_feature_group_name": feature_group,
                "p_include_stale": include_stale,
            }

            response = self.supabase.rpc("get_entity_features", rpc_params).execute()

            # Filter by feature names if specified
            if feature_names:
                return [
                    record for record in response.data if record["feature_name"] in feature_names
                ]

            return response.data

        except Exception as e:
            logger.error(f"Error querying Supabase features: {e}")
            return []

    def _build_entity_features(
        self, entity_values: Dict[str, Any], features_data: List[Dict[str, Any]]
    ) -> EntityFeatures:
        """Build EntityFeatures object from query results."""
        entity_features = EntityFeatures(entity_values=entity_values)

        for record in features_data:
            feature_name = record["feature_name"]

            # Add feature value
            entity_features.features[feature_name] = record["value"]

            # Add metadata
            entity_features.metadata[feature_name] = {
                "feature_group": record["feature_group"],
                "event_timestamp": record["event_timestamp"],
                "freshness_status": record["freshness_status"],
            }

        return entity_features

    def _generate_cache_key(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str],
        feature_names: Optional[List[str]],
    ) -> str:
        """Generate Redis cache key."""
        # Sort entity values for consistent keys
        sorted_entity = json.dumps(entity_values, sort_keys=True)

        # Build key components
        key_parts = [
            "fs",  # feature store prefix
            hashlib.md5(sorted_entity.encode()).hexdigest(),
        ]

        if feature_group:
            key_parts.append(f"fg:{feature_group}")

        if feature_names:
            sorted_features = "|".join(sorted(feature_names))
            key_parts.append(f"fn:{hashlib.md5(sorted_features.encode()).hexdigest()}")

        return ":".join(key_parts)

    def _get_from_cache(self, cache_key: str) -> Optional[EntityFeatures]:
        """Get EntityFeatures from Redis cache."""
        try:
            if not self.redis_client:
                return None

            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return EntityFeatures(**data)

        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")

        return None

    def _set_in_cache(self, cache_key: str, entity_features: EntityFeatures) -> None:
        """Set EntityFeatures in Redis cache."""
        try:
            if not self.redis_client:
                return

            # Serialize to JSON
            cache_data = entity_features.model_dump(mode="json")
            self.redis_client.setex(cache_key, self.cache_ttl_seconds, json.dumps(cache_data))

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def invalidate_cache(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str] = None,
    ) -> None:
        """
        Invalidate cache for an entity.

        Args:
            entity_values: Entity key-value pairs
            feature_group: Optional feature group (invalidates all if None)
        """
        try:
            if not self.redis_client:
                return

            # Generate pattern to match all cache keys for this entity
            sorted_entity = json.dumps(entity_values, sort_keys=True)
            entity_hash = hashlib.md5(sorted_entity.encode()).hexdigest()

            if feature_group:
                pattern = f"fs:{entity_hash}:fg:{feature_group}*"
            else:
                pattern = f"fs:{entity_hash}*"

            # Delete all matching keys
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cache keys for entity")

        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
