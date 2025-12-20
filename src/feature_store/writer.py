"""
Feature Writer Service

Handles writing feature values to Supabase and cache invalidation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import redis
from supabase import Client

from .models import Feature, FeatureValue

logger = logging.getLogger(__name__)


class FeatureWriter:
    """Handles feature value writing and cache invalidation."""

    def __init__(
        self,
        supabase: Client,
        redis_client: Optional[redis.Redis],
    ):
        """
        Initialize Feature Writer.

        Args:
            supabase: Supabase client
            redis_client: Redis client (optional)
        """
        self.supabase = supabase
        self.redis_client = redis_client

    def write_feature_value(
        self,
        feature_name: str,
        entity_values: Dict[str, Any],
        value: Any,
        event_timestamp: datetime,
        feature_group: Optional[str] = None,
        source_job_id: Optional[str] = None,
        invalidate_cache: bool = True,
    ) -> FeatureValue:
        """
        Write a single feature value.

        Args:
            feature_name: Name of the feature
            entity_values: Entity key-value pairs
            value: Feature value (any JSON-serializable type)
            event_timestamp: When the event occurred
            feature_group: Feature group name (required if feature name is ambiguous)
            source_job_id: ID of the job that created this value
            invalidate_cache: Invalidate Redis cache for this entity

        Returns:
            Created FeatureValue instance
        """
        # Get feature ID
        feature_id = self._get_feature_id(feature_name, feature_group)
        if not feature_id:
            raise ValueError(
                f"Feature not found: {feature_name}"
                + (f" in group {feature_group}" if feature_group else "")
            )

        # Prepare data
        data = {
            "feature_id": str(feature_id),
            "entity_values": entity_values,
            "value": value,
            "event_timestamp": event_timestamp.isoformat(),
            "source_job_id": source_job_id,
        }

        # Insert into Supabase
        try:
            response = self.supabase.table("feature_values").insert(data).execute()

            if not response.data:
                raise RuntimeError("No data returned from insert")

            feature_value = FeatureValue(**response.data[0])
            logger.debug(f"Wrote feature value: {feature_name} for {entity_values}")

            # Invalidate cache
            if invalidate_cache:
                self._invalidate_entity_cache(entity_values, feature_group)

            return feature_value

        except Exception as e:
            logger.error(f"Error writing feature value: {e}")
            raise

    def write_batch_features(
        self,
        feature_values: List[Dict[str, Any]],
        invalidate_cache: bool = True,
    ) -> int:
        """
        Write multiple feature values in batch.

        Args:
            feature_values: List of dicts with keys:
                - feature_name: str
                - entity_values: dict
                - value: any
                - event_timestamp: datetime
                - feature_group: str (optional)
                - source_job_id: str (optional)
            invalidate_cache: Invalidate Redis cache

        Returns:
            Number of features successfully written
        """
        if not feature_values:
            return 0

        # Resolve feature IDs
        resolved_data = []
        entities_to_invalidate = set()

        for fv in feature_values:
            feature_id = self._get_feature_id(
                fv["feature_name"], fv.get("feature_group")
            )

            if not feature_id:
                logger.warning(
                    f"Feature not found: {fv['feature_name']}, skipping..."
                )
                continue

            resolved_data.append(
                {
                    "feature_id": str(feature_id),
                    "entity_values": fv["entity_values"],
                    "value": fv["value"],
                    "event_timestamp": fv["event_timestamp"].isoformat(),
                    "source_job_id": fv.get("source_job_id"),
                }
            )

            # Track entities for cache invalidation
            entity_key = json.dumps(fv["entity_values"], sort_keys=True)
            entities_to_invalidate.add(entity_key)

        if not resolved_data:
            logger.warning("No valid features to write")
            return 0

        # Batch insert
        try:
            response = self.supabase.table("feature_values").insert(resolved_data).execute()
            count = len(response.data) if response.data else 0

            logger.info(f"Wrote {count} feature values in batch")

            # Invalidate cache for all affected entities
            if invalidate_cache:
                for entity_key in entities_to_invalidate:
                    entity_values = json.loads(entity_key)
                    self._invalidate_entity_cache(entity_values)

            return count

        except Exception as e:
            logger.error(f"Error in batch write: {e}")
            raise

    def _get_feature_id(
        self, feature_name: str, feature_group: Optional[str] = None
    ) -> Optional[UUID]:
        """
        Get feature ID by name and optional feature group.

        Args:
            feature_name: Feature name
            feature_group: Feature group name (optional)

        Returns:
            Feature UUID or None if not found
        """
        try:
            # Query for feature
            query = (
                self.supabase.table("features")
                .select("id, feature_groups!inner(name)")
                .eq("name", feature_name)
            )

            if feature_group:
                # Join with feature_groups to filter by group name
                query = query.eq("feature_groups.name", feature_group)

            response = query.execute()

            if response.data:
                if len(response.data) > 1 and not feature_group:
                    logger.warning(
                        f"Multiple features found with name '{feature_name}'. "
                        "Consider specifying feature_group."
                    )
                return UUID(response.data[0]["id"])

            return None

        except Exception as e:
            logger.error(f"Error getting feature ID: {e}")
            return None

    def _invalidate_entity_cache(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str] = None,
    ) -> None:
        """
        Invalidate Redis cache for an entity.

        Args:
            entity_values: Entity key-value pairs
            feature_group: Optional feature group filter
        """
        if not self.redis_client:
            return

        try:
            # Generate cache key pattern
            sorted_entity = json.dumps(entity_values, sort_keys=True)
            import hashlib

            entity_hash = hashlib.md5(sorted_entity.encode()).hexdigest()

            if feature_group:
                pattern = f"fs:{entity_hash}:fg:{feature_group}*"
            else:
                pattern = f"fs:{entity_hash}*"

            # Delete matching keys
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} cache entries")

        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
