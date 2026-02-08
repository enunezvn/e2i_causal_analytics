"""
Feature Store Client

Main client for interacting with the E2I lightweight feature store.
Provides unified interface for feature retrieval, writing, and management.
"""

import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import mlflow
import redis
from supabase import Client, create_client

from .models import (
    EntityFeatures,
    Feature,
    FeatureGroup,
    FeatureStatistics,
    FeatureValue,
)
from .retrieval import FeatureRetriever
from .writer import FeatureWriter

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """
    Lightweight Feature Store Client.

    Integrates Supabase (offline storage), Redis (online caching),
    and MLflow (feature tracking).

    Example:
        ```python
        from src.feature_store import FeatureStoreClient

        # Initialize client (self-hosted Supabase)
        fs = FeatureStoreClient(
            supabase_url="http://localhost:54321",  # or http://138.197.4.36:54321 on droplet
            supabase_key="your-anon-key-from-self-hosted",
            redis_url="redis://localhost:6382"
        )

        # Get features for an entity
        features = fs.get_entity_features(
            entity_values={"hcp_id": "HCP123"},
            feature_group="hcp_demographics"
        )

        # Write new feature values
        fs.write_feature_value(
            feature_name="specialty",
            entity_values={"hcp_id": "HCP123"},
            value="Oncology",
            event_timestamp=datetime.now()
        )
        ```
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        redis_url: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize Feature Store Client.

        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase anon/service key (defaults to SUPABASE_KEY env var)
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            mlflow_tracking_uri: MLflow tracking server URI
            cache_ttl_seconds: Default TTL for cached features (default: 1 hour)
            enable_cache: Enable Redis caching (default: True)
        """
        # Read from environment variables if not provided
        supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        supabase_key = (
            supabase_key or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        )
        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6382")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL and key are required. "
                "Provide them as arguments or set SUPABASE_URL and SUPABASE_KEY environment variables."
            )

        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Initialize Redis client
        self.redis_client: Optional[redis.Redis] = None
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds

        if enable_cache:
            try:
                self.redis_client = redis.from_url(
                    redis_url, decode_responses=True, socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis connection successful")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
                self.enable_cache = False

        # Initialize MLflow
        if mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

        # Initialize retrieval and writing components
        self.retriever = FeatureRetriever(
            supabase=self.supabase,
            redis_client=self.redis_client,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.writer = FeatureWriter(
            supabase=self.supabase,
            redis_client=self.redis_client,
        )

        logger.info("FeatureStoreClient initialized successfully")

    # =========================================================================
    # Feature Group Management
    # =========================================================================

    def create_feature_group(
        self,
        name: str,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        source_table: Optional[str] = None,
        expected_update_frequency_hours: int = 24,
        max_age_hours: int = 168,
        tags: Optional[List[str]] = None,
        mlflow_experiment_name: Optional[str] = None,
    ) -> FeatureGroup:
        """
        Create a new feature group.

        Args:
            name: Unique feature group name
            description: Feature group description
            owner: Owner/team name
            source_table: Source table in Supabase
            expected_update_frequency_hours: Expected update frequency
            max_age_hours: Maximum age before features expire
            tags: List of tags for organization
            mlflow_experiment_name: MLflow experiment name for tracking

        Returns:
            Created FeatureGroup instance
        """
        # Create MLflow experiment if specified
        mlflow_experiment_id = None
        if mlflow_experiment_name:
            try:
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment is None:
                    mlflow_experiment_id = mlflow.create_experiment(
                        mlflow_experiment_name,
                        artifact_location="mlflow-artifacts:/",
                    )
                    logger.info(
                        f"Created MLflow experiment: {mlflow_experiment_name} ({mlflow_experiment_id})"
                    )
                else:
                    mlflow_experiment_id = experiment.experiment_id
                    logger.info(
                        f"Using existing MLflow experiment: {mlflow_experiment_name} ({mlflow_experiment_id})"
                    )
            except Exception as e:
                logger.warning(f"Failed to create/get MLflow experiment: {e}")

        # Insert into Supabase
        data = {
            "name": name,
            "description": description,
            "owner": owner,
            "source_table": source_table,
            "expected_update_frequency_hours": expected_update_frequency_hours,
            "max_age_hours": max_age_hours,
            "tags": tags or [],
            "mlflow_experiment_id": mlflow_experiment_id,
        }

        response = self.supabase.table("feature_groups").insert(data).execute()

        if response.data:
            feature_group = FeatureGroup(**response.data[0])  # type: ignore[arg-type]
            logger.info(f"Created feature group: {name}")
            return feature_group
        else:
            raise RuntimeError(f"Failed to create feature group: {name}")

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Get feature group by name."""
        response = self.supabase.table("feature_groups").select("*").eq("name", name).execute()

        if response.data:
            return FeatureGroup(**response.data[0])  # type: ignore[arg-type]
        return None

    def list_feature_groups(self, owner: Optional[str] = None) -> List[FeatureGroup]:
        """List all feature groups, optionally filtered by owner."""
        query = self.supabase.table("feature_groups").select("*")

        if owner:
            query = query.eq("owner", owner)

        response = query.execute()
        return [FeatureGroup(**item) for item in response.data]  # type: ignore[arg-type]

    # =========================================================================
    # Feature Management
    # =========================================================================

    def create_feature(
        self,
        feature_group_name: str,
        name: str,
        value_type: str,
        entity_keys: List[str],
        description: Optional[str] = None,
        computation_query: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Feature:
        """
        Create a new feature within a feature group.

        Args:
            feature_group_name: Name of parent feature group
            name: Feature name
            value_type: Feature value type (e.g., "int64", "float64")
            entity_keys: List of entity keys this feature describes
            description: Feature description
            computation_query: SQL query to compute this feature
            owner: Owner/team name
            tags: List of tags

        Returns:
            Created Feature instance
        """
        # Get feature group ID
        feature_group = self.get_feature_group(feature_group_name)
        if not feature_group:
            raise ValueError(f"Feature group not found: {feature_group_name}")

        # Insert into Supabase
        data = {
            "feature_group_id": str(feature_group.id),
            "name": name,
            "value_type": value_type,
            "entity_keys": entity_keys,
            "description": description,
            "computation_query": computation_query,
            "owner": owner,
            "tags": tags or [],
        }

        response = (
            self.supabase.table("features")
            .upsert(data, on_conflict="feature_group_id,name")
            .execute()
        )

        if response.data:
            feature = Feature(**response.data[0])  # type: ignore[arg-type]
            logger.info(f"Upserted feature: {feature_group_name}.{name}")

            # Log to MLflow if experiment is linked
            if feature_group.mlflow_experiment_id:
                self._log_feature_to_mlflow(feature, feature_group)

            return feature
        else:
            raise RuntimeError(f"Failed to create feature: {name}")

    def get_feature(self, feature_group_name: str, feature_name: str) -> Optional[Feature]:
        """Get feature by feature group and name."""
        # Get feature group
        feature_group = self.get_feature_group(feature_group_name)
        if not feature_group:
            return None

        # Get feature
        response = (
            self.supabase.table("features")
            .select("*")
            .eq("feature_group_id", str(feature_group.id))
            .eq("name", feature_name)
            .execute()
        )

        if response.data:
            return Feature(**response.data[0])  # type: ignore[arg-type]
        return None

    def list_features(self, feature_group_name: str) -> List[Feature]:
        """List all features in a feature group."""
        feature_group = self.get_feature_group(feature_group_name)
        if not feature_group:
            return []

        response = (
            self.supabase.table("features")
            .select("*")
            .eq("feature_group_id", str(feature_group.id))
            .execute()
        )

        return [Feature(**item) for item in response.data]  # type: ignore[arg-type]

    # =========================================================================
    # Feature Retrieval (Online Serving)
    # =========================================================================

    def get_entity_features(
        self,
        entity_values: Dict[str, Any],
        feature_group: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        include_stale: bool = True,
        use_cache: bool = True,
    ) -> EntityFeatures:
        """
        Get all features for a specific entity (online serving).

        Args:
            entity_values: Entity key-value pairs (e.g., {"hcp_id": "HCP123"})
            feature_group: Optional feature group filter
            feature_names: Optional list of specific features to retrieve
            include_stale: Include stale features (default: True)
            use_cache: Use Redis cache if available (default: True)

        Returns:
            EntityFeatures with all features and metadata
        """
        return self.retriever.get_entity_features(
            entity_values=entity_values,
            feature_group=feature_group,
            feature_names=feature_names,
            include_stale=include_stale,
            use_cache=use_cache and self.enable_cache,
        )

    def get_historical_features(
        self,
        entity_values: Dict[str, Any],
        feature_names: List[str],
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical feature values for an entity (time-series).

        Args:
            entity_values: Entity key-value pairs
            feature_names: List of feature names to retrieve
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)

        Returns:
            List of feature value records with timestamps
        """
        return self.retriever.get_historical_features(
            entity_values=entity_values,
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time,
        )

    # =========================================================================
    # Feature Writing
    # =========================================================================

    def write_feature_value(
        self,
        feature_name: str,
        entity_values: Dict[str, Any],
        value: Any,
        event_timestamp: Any,
        feature_group: Optional[str] = None,
        invalidate_cache: bool = True,
    ) -> FeatureValue:
        """
        Write a single feature value.

        Args:
            feature_name: Name of the feature
            entity_values: Entity key-value pairs
            value: Feature value
            event_timestamp: When the event occurred
            feature_group: Feature group name (required if feature name is ambiguous)
            invalidate_cache: Invalidate Redis cache for this entity

        Returns:
            Created FeatureValue instance
        """
        return self.writer.write_feature_value(
            feature_name=feature_name,
            entity_values=entity_values,
            value=value,
            event_timestamp=event_timestamp,
            feature_group=feature_group,
            invalidate_cache=invalidate_cache and self.enable_cache,
        )

    def write_batch_features(
        self,
        feature_values: List[Dict[str, Any]],
        invalidate_cache: bool = True,
    ) -> int:
        """
        Write multiple feature values in batch.

        Args:
            feature_values: List of feature value dicts with keys:
                - feature_name
                - entity_values
                - value
                - event_timestamp
                - feature_group (optional)
            invalidate_cache: Invalidate Redis cache

        Returns:
            Number of features written
        """
        return self.writer.write_batch_features(
            feature_values=feature_values,
            invalidate_cache=invalidate_cache and self.enable_cache,
        )

    # =========================================================================
    # Monitoring & Statistics
    # =========================================================================

    def get_feature_statistics(
        self, feature_group_name: str, feature_name: str
    ) -> Optional[FeatureStatistics]:
        """
        Get statistics for a specific feature.

        Computes aggregation statistics from feature_values table including
        count, min, max, mean, std, percentiles (for numeric), and unique counts.

        Args:
            feature_group_name: Name of the feature group
            feature_name: Name of the feature

        Returns:
            FeatureStatistics with computed metrics or None if feature not found
        """
        # Get feature definition
        feature = self.get_feature(feature_group_name, feature_name)
        if not feature:
            logger.warning(f"Feature not found: {feature_group_name}.{feature_name}")
            return None

        # Query all values for this feature
        try:
            response = (
                self.supabase.table("feature_values")
                .select("value, event_timestamp")
                .eq("feature_id", str(feature.id))
                .order("event_timestamp", desc=True)
                .limit(10000)  # Limit for performance
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to query feature values: {e}")
            return None

        if not response.data:
            # Return empty statistics if no values
            assert feature.id is not None
            return FeatureStatistics(
                feature_id=feature.id,
                feature_name=feature_name,
                feature_group=feature_group_name,
                count=0,
                null_count=0,
                computed_at=datetime.now(timezone.utc),
            )

        # Extract values
        values = [row.get("value") for row in response.data]  # type: ignore[union-attr]
        total_count = len(values)

        # Count nulls
        null_count = sum(1 for v in values if v is None)
        non_null_values = [v for v in values if v is not None]

        # Initialize statistics
        assert feature.id is not None
        stats = FeatureStatistics(
            feature_id=feature.id,
            feature_name=feature_name,
            feature_group=feature_group_name,
            count=total_count,
            null_count=null_count,
            computed_at=datetime.now(timezone.utc),
        )

        # Compute unique count
        try:
            stats.unique_count = len({str(v) for v in non_null_values})
        except Exception:
            stats.unique_count = None

        # Compute numeric statistics if applicable
        numeric_values = []
        for v in non_null_values:
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_values.append(float(v))
                elif isinstance(v, str):
                    # Try to parse as number
                    numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue

        if numeric_values:
            stats.min_value = min(numeric_values)
            stats.max_value = max(numeric_values)
            stats.mean = statistics.mean(numeric_values)

            if len(numeric_values) > 1:
                stats.std = statistics.stdev(numeric_values)

                # Compute percentiles
                sorted_values = sorted(numeric_values)
                n = len(sorted_values)
                percentiles_raw: Dict[str, Optional[float]] = {
                    "p25": sorted_values[int(n * 0.25)],
                    "p50": sorted_values[int(n * 0.50)],
                    "p75": sorted_values[int(n * 0.75)],
                    "p90": sorted_values[int(n * 0.90)],
                    "p95": sorted_values[int(n * 0.95)] if n >= 20 else None,
                    "p99": sorted_values[int(n * 0.99)] if n >= 100 else None,
                }
                # Remove None percentiles
                stats.percentiles = {k: v for k, v in percentiles_raw.items() if v is not None}

        logger.debug(
            f"Computed statistics for {feature_group_name}.{feature_name}: "
            f"count={stats.count}, mean={stats.mean}"
        )

        return stats

    def get_feature_group_statistics(self, feature_group_name: str) -> Dict[str, FeatureStatistics]:
        """
        Get statistics for all features in a feature group.

        Args:
            feature_group_name: Name of the feature group

        Returns:
            Dict mapping feature names to their FeatureStatistics
        """
        features = self.list_features(feature_group_name)
        if not features:
            logger.warning(f"No features found in group: {feature_group_name}")
            return {}

        stats_dict = {}
        for feature in features:
            stats = self.get_feature_statistics(feature_group_name, feature.name)
            if stats:
                stats_dict[feature.name] = stats

        logger.info(
            f"Computed statistics for {len(stats_dict)} features in group {feature_group_name}"
        )
        return stats_dict

    def update_freshness_status(self) -> None:
        """Update freshness status for all features."""
        try:
            self.supabase.rpc("update_feature_freshness").execute()
            logger.info("Updated feature freshness status")
        except Exception as e:
            logger.error(f"Failed to update freshness status: {e}")

    # =========================================================================
    # MLflow Integration
    # =========================================================================

    def _log_feature_to_mlflow(self, feature: Feature, feature_group: FeatureGroup) -> None:
        """Log feature definition to MLflow."""
        try:
            if not feature_group.mlflow_experiment_id:
                return

            mlflow.set_experiment(experiment_id=feature_group.mlflow_experiment_id)

            with mlflow.start_run(run_name=f"feature_{feature.name}"):
                # Log parameters
                mlflow.log_param("feature_name", feature.name)
                mlflow.log_param("feature_group", feature_group.name)
                mlflow.log_param("value_type", feature.value_type)
                mlflow.log_param("entity_keys", ",".join(feature.entity_keys))
                mlflow.log_param("version", feature.version)

                # Log description as text artifact
                if feature.description:
                    mlflow.log_text(feature.description, "feature_description.txt")

                # Log computation query if exists
                if feature.computation_query:
                    mlflow.log_text(feature.computation_query, "computation_query.sql")

                logger.info(f"Logged feature {feature.name} to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log feature to MLflow: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {
            "supabase": False,
            "redis": False,
            "mlflow": False,
        }

        # Check Supabase
        try:
            self.supabase.table("feature_groups").select("id").limit(1).execute()
            health["supabase"] = True
        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")

        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                health["redis"] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")

        # Check MLflow
        try:
            mlflow.get_tracking_uri()
            health["mlflow"] = True
        except Exception as e:
            logger.error(f"MLflow health check failed: {e}")

        return health

    def close(self) -> None:
        """Close all connections."""
        if self.redis_client:
            self.redis_client.close()
        logger.info("FeatureStoreClient connections closed")
