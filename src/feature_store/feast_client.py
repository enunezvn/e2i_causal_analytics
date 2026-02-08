"""Feast Feature Store Client for E2I Causal Analytics.

This module provides a unified interface to Feast for:
- Online feature retrieval (low-latency inference)
- Offline feature retrieval (point-in-time correct training data)
- Feature materialization (syncing offline to online store)
- Feature statistics and metadata
- Fallback to custom feature store during migration

Usage:
    from src.feature_store.feast_client import FeastClient

    client = FeastClient()
    features = await client.get_online_features(
        entity_rows=[{"hcp_id": "123", "brand_id": "remibrutinib"}],
        feature_refs=["hcp_conversion_features:engagement_score"],
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from feast import FeatureStore

    from src.feature_store.client import FeatureStoreClient
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _expand_env_vars(text: str) -> str:
    """Expand shell-style environment variables in text.

    Supports both ${VAR} and ${VAR:-default} syntax.
    Handles YAML-safe empty string output for values in key: value positions.

    Args:
        text: Text containing environment variable references.

    Returns:
        Text with environment variables expanded.
    """
    # Pattern matches ${VAR} or ${VAR:-default}
    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        value = os.environ.get(var_name, default_value)
        # For empty values that will be in YAML key: value positions,
        # return quoted empty string to prevent YAML null
        if value == "":
            return '""'
        return value

    return re.sub(pattern, replace_var, text)


# Feature repo path - relative to project root
FEATURE_REPO_PATH = Path(__file__).parent.parent.parent / "feature_repo"

# Config path
FEAST_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "feast_materialization.yaml"


class FreshnessStatus(str, Enum):
    """Feature freshness status levels."""

    FRESH = "fresh"
    WARNING = "warning"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class FeastConfig(BaseModel):
    """Configuration for Feast client."""

    repo_path: Path = Field(default=FEATURE_REPO_PATH)
    enable_fallback: bool = Field(default=True, description="Enable fallback to custom store")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL for feature statistics")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")
    max_retries: int = Field(default=3, description="Max retries for failed requests")


class FeatureStatistics(BaseModel):
    """Feature statistics from Feast."""

    feature_view: str
    feature_name: str
    count: int
    null_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    stddev_value: Optional[float] = None
    last_updated: datetime


class FeatureFreshness(BaseModel):
    """Feature freshness information for a feature view."""

    feature_view: str
    last_materialized: Optional[datetime] = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    age_hours: Optional[float] = None
    ttl_hours: Optional[float] = None
    max_staleness_hours: float = 24.0
    warning_threshold_hours: Optional[float] = None
    is_fresh: bool = False
    message: Optional[str] = None


def load_feast_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Feast materialization configuration from YAML file.

    Args:
        config_path: Path to config file. Uses default if not provided.

    Returns:
        Dictionary with Feast configuration.
    """
    path = config_path or FEAST_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Feast config not found at {path}. Using defaults.")
        return {
            "materialization": {
                "max_staleness_hours": 24.0,
            },
            "feature_views": {},
            "alerting": {"enabled": True},
        }

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
            logger.debug(f"Loaded Feast config from {path}")
            return config or {}
    except Exception as e:
        logger.error(f"Error loading Feast config: {e}")
        return {}


class FeastClient:
    """Unified Feast client for E2I feature store.

    Provides async interface to Feast feature store with:
    - Point-in-time correct historical feature retrieval
    - Low-latency online feature serving
    - Feature materialization management
    - Graceful fallback to custom store

    Example:
        client = FeastClient()

        # Online features for inference
        features = await client.get_online_features(
            entity_rows=[{"hcp_id": "123"}],
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Historical features for training
        training_data = await client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["hcp_conversion_features:*"],
        )
    """

    def __init__(
        self,
        config: Optional[FeastConfig] = None,
        materialization_config_path: Optional[Path] = None,
    ):
        """Initialize Feast client.

        Args:
            config: Optional configuration. Uses defaults if not provided.
            materialization_config_path: Optional path to materialization config.
        """
        self.config = config or FeastConfig()
        self._store: Optional[FeatureStore] = None
        self._initialized = False
        self._custom_store: Optional[FeatureStoreClient] = None  # Fallback custom store
        self._stats_cache: Dict[str, FeatureStatistics] = {}
        self._stats_cache_time: Dict[str, datetime] = {}
        self._temp_dir: Optional[str] = None  # Temp directory for expanded config

        # Load materialization config for freshness thresholds
        self._materialization_config = load_feast_config(materialization_config_path)

        # Track last materialization timestamps per feature view
        self._materialization_timestamps: Dict[str, datetime] = {}

    async def initialize(self) -> None:
        """Initialize Feast store connection.

        Lazily initializes the connection to avoid blocking at import time.
        Pre-processes feature_store.yaml to expand environment variables.
        """
        if self._initialized:
            return

        try:
            from feast import FeatureStore

            # Initialize Feast store
            repo_path = str(self.config.repo_path)
            if not os.path.exists(repo_path):
                logger.warning(f"Feast repo not found at {repo_path}. Creating minimal setup.")
                os.makedirs(repo_path, exist_ok=True)

            # Pre-process feature_store.yaml to expand environment variables
            feature_store_yaml = Path(repo_path) / "feature_store.yaml"
            expanded_repo_path = repo_path

            if feature_store_yaml.exists():
                logger.info(f"Found feature_store.yaml at {feature_store_yaml}")
                with open(feature_store_yaml, "r") as f:
                    original_yaml = f.read()

                # Check if YAML contains env var syntax that needs expansion
                if "${" in original_yaml:
                    logger.info("Expanding environment variables in Feast config")
                    expanded_yaml = _expand_env_vars(original_yaml)

                    # Create temp directory with expanded config
                    self._temp_dir = tempfile.mkdtemp(prefix="feast_expanded_")
                    expanded_repo_path = self._temp_dir
                    expanded_yaml_path = Path(self._temp_dir) / "feature_store.yaml"

                    with open(expanded_yaml_path, "w") as f:
                        f.write(expanded_yaml)

                    logger.info(f"Feast config expanded to {expanded_repo_path}")
                    logger.info(f"Using expanded repo path: {expanded_repo_path}")

            self._store = FeatureStore(repo_path=expanded_repo_path)
            self._initialized = True
            logger.info(f"Feast client initialized with repo: {repo_path}")

        except ImportError as e:
            logger.error(f"Feast not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Feast store: {e}")
            if self.config.enable_fallback:
                logger.info("Falling back to custom feature store")
                await self._init_fallback_store()
            else:
                raise

    async def _init_fallback_store(self) -> None:
        """Initialize fallback to custom feature store."""
        try:
            from src.feature_store.client import FeatureStoreClient

            self._custom_store = FeatureStoreClient()
            logger.info("Custom feature store fallback initialized")
        except ImportError:
            logger.warning("Custom feature store not available for fallback")

    def _ensure_initialized(self) -> None:
        """Ensure the client is initialized."""
        if not self._initialized and self._store is None and self._custom_store is None:
            raise RuntimeError("FeastClient not initialized. Call initialize() first.")

    async def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str],
        full_feature_names: bool = True,
    ) -> Dict[str, List[Any]]:
        """Get features for online inference.

        Retrieves feature values from the online store (Redis) for
        low-latency inference.

        Args:
            entity_rows: List of entity key dictionaries.
                Example: [{"hcp_id": "123", "brand_id": "remibrutinib"}]
            feature_refs: List of feature references.
                Format: "feature_view_name:feature_name" or "feature_view_name:*"
                Example: ["hcp_conversion_features:engagement_score"]
            full_feature_names: If True, use fully qualified names in output.

        Returns:
            Dictionary mapping feature names to lists of values.
            Example: {"hcp_conversion_features__engagement_score": [0.85]}

        Raises:
            RuntimeError: If client not initialized.
            ValueError: If entity_rows or feature_refs are empty.
        """
        await self.initialize()
        self._ensure_initialized()

        if not entity_rows:
            raise ValueError("entity_rows cannot be empty")
        if not feature_refs:
            raise ValueError("feature_refs cannot be empty")

        try:
            if self._store:
                # Use Feast online store
                response = self._store.get_online_features(
                    entity_rows=entity_rows,
                    features=feature_refs,
                    full_feature_names=full_feature_names,
                )
                return response.to_dict()

            elif self._custom_store:
                # Fallback to custom store
                logger.debug("Using custom store fallback for online features")
                return await self._get_online_features_fallback(
                    entity_rows, feature_refs, full_feature_names
                )

            else:
                raise RuntimeError("No feature store available")

        except Exception as e:
            logger.error(f"Error getting online features: {e}")
            if self.config.enable_fallback and self._custom_store:
                logger.info("Attempting fallback to custom store")
                return await self._get_online_features_fallback(
                    entity_rows, feature_refs, full_feature_names
                )
            raise

    async def _get_online_features_fallback(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str],
        full_feature_names: bool,
    ) -> Dict[str, List[Any]]:
        """Fallback implementation using custom feature store."""
        if not self._custom_store:
            raise RuntimeError("Custom store not available for fallback")

        # Parse feature refs to get feature names
        features: Dict[str, List[Any]] = {}
        for ref in feature_refs:
            if ":" in ref:
                view_name, feat_name = ref.split(":", 1)
                if feat_name == "*":
                    continue  # Skip wildcard refs for fallback
                key = f"{view_name}__{feat_name}" if full_feature_names else feat_name
                features[key] = []

        # Get features for each entity row
        for entity_row in entity_rows:
            entity_id = entity_row.get("hcp_id") or entity_row.get("patient_id")
            if entity_id:
                result = self._custom_store.get_entity_features(
                    entity_values=entity_row,
                    feature_names=list(features.keys()),
                )
                for key in features:
                    features[key].append(result.features.get(key))

        return features

    async def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        full_feature_names: bool = True,
    ) -> pd.DataFrame:
        """Get point-in-time correct historical features.

        Performs point-in-time joins to get feature values as they were
        at each entity's event timestamp. Critical for preventing data leakage.

        Args:
            entity_df: DataFrame with entity keys and event_timestamp column.
                Required columns: entity key columns, event_timestamp
            feature_refs: List of feature references.
                Format: "feature_view_name:feature_name"
            full_feature_names: If True, use fully qualified names.

        Returns:
            DataFrame with entity columns plus feature columns.
            Features are joined at the correct point in time.

        Example:
            entity_df = pd.DataFrame({
                "hcp_id": ["123", "456"],
                "brand_id": ["remibrutinib", "remibrutinib"],
                "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
            })

            features_df = await client.get_historical_features(
                entity_df=entity_df,
                feature_refs=["hcp_conversion_features:engagement_score"],
            )
        """
        await self.initialize()
        self._ensure_initialized()

        if entity_df.empty:
            raise ValueError("entity_df cannot be empty")
        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must have 'event_timestamp' column")
        if not feature_refs:
            raise ValueError("feature_refs cannot be empty")

        try:
            if self._store:
                # Use Feast offline store with point-in-time joins
                retrieval_job = self._store.get_historical_features(
                    entity_df=entity_df,
                    features=feature_refs,
                    full_feature_names=full_feature_names,
                )
                return retrieval_job.to_df()

            elif self._custom_store:
                logger.debug("Using custom store fallback for historical features")
                return await self._get_historical_features_fallback(
                    entity_df, feature_refs, full_feature_names
                )

            else:
                raise RuntimeError("No feature store available")

        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            if self.config.enable_fallback and self._custom_store:
                return await self._get_historical_features_fallback(
                    entity_df, feature_refs, full_feature_names
                )
            raise

    async def _get_historical_features_fallback(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        full_feature_names: bool,
    ) -> pd.DataFrame:
        """Fallback implementation for historical features.

        Note: This fallback does NOT support true point-in-time joins.
        It returns the latest feature values instead.
        """
        logger.warning(
            "Using fallback for historical features. Point-in-time correctness NOT guaranteed."
        )

        if not self._custom_store:
            raise RuntimeError("Custom store not available for fallback")

        # Get latest features (not point-in-time correct)
        result_df = entity_df.copy()

        for ref in feature_refs:
            if ":" not in ref:
                continue
            view_name, feat_name = ref.split(":", 1)
            if feat_name == "*":
                continue

            col_name = f"{view_name}__{feat_name}" if full_feature_names else feat_name
            result_df[col_name] = None

        return result_df

    async def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Materialize features to online store.

        Syncs feature values from offline store (Supabase) to online store (Redis)
        for the specified time range.

        Args:
            start_date: Start of materialization window.
            end_date: End of materialization window.
            feature_views: Optional list of feature views to materialize.
                If None, materializes all feature views.

        Returns:
            Dictionary with materialization results:
            - feature_views: List of materialized views
            - rows_materialized: Estimated row count
            - duration_seconds: Time taken
        """
        await self.initialize()
        self._ensure_initialized()

        if not self._store:
            logger.warning("Feast store not available. Skipping materialization.")
            return {"feature_views": [], "rows_materialized": 0, "status": "skipped"}

        try:
            start_time = datetime.now()

            if feature_views:
                self._store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=feature_views,
                )
            else:
                self._store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                )

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Materialization completed in {duration:.2f}s. Views: {feature_views or 'all'}"
            )

            # Track materialization timestamps
            materialized_views = feature_views or self._get_all_feature_view_names()
            for view_name in materialized_views:
                self._materialization_timestamps[view_name] = datetime.now()

            return {
                "feature_views": materialized_views,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_seconds": duration,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            return {
                "feature_views": feature_views or ["all"],
                "status": "failed",
                "error": str(e),
            }

    async def materialize_incremental(
        self,
        end_date: datetime,
        feature_views: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Incrementally materialize features from last materialization.

        More efficient than full materialization for regular updates.

        Args:
            end_date: End date for incremental materialization.
            feature_views: Optional list of feature views to materialize.

        Returns:
            Materialization results dictionary.
        """
        await self.initialize()
        self._ensure_initialized()

        if not self._store:
            logger.warning("Feast store not available. Skipping materialization.")
            return {"status": "skipped"}

        try:
            start_time = datetime.now()

            if feature_views:
                self._store.materialize_incremental(
                    end_date=end_date,
                    feature_views=feature_views,
                )
            else:
                self._store.materialize_incremental(end_date=end_date)

            duration = (datetime.now() - start_time).total_seconds()

            # Track materialization timestamps
            materialized_views = feature_views or self._get_all_feature_view_names()
            for view_name in materialized_views:
                self._materialization_timestamps[view_name] = datetime.now()

            return {
                "feature_views": materialized_views,
                "end_date": end_date.isoformat(),
                "duration_seconds": duration,
                "status": "completed",
                "incremental": True,
            }

        except Exception as e:
            logger.error(f"Incremental materialization failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _get_all_feature_view_names(self) -> List[str]:
        """Get all registered feature view names from config or store."""
        # First check config
        feature_views_config = self._materialization_config.get("feature_views", {})
        if feature_views_config:
            return list(feature_views_config.keys())

        # Fall back to store
        if self._store:
            try:
                return [fv.name for fv in self._store.list_feature_views()]
            except Exception:
                pass

        return []

    def _get_freshness_thresholds(self, feature_view: str) -> tuple[float, float]:
        """Get freshness thresholds for a feature view.

        Args:
            feature_view: Name of the feature view.

        Returns:
            Tuple of (max_staleness_hours, warning_threshold_hours).
        """
        # Check for view-specific config
        view_config = self._materialization_config.get("feature_views", {}).get(feature_view, {})
        global_config = self._materialization_config.get("materialization", {})

        # Get max staleness (view-specific or global default)
        max_staleness = view_config.get(
            "max_staleness_hours",
            global_config.get("max_staleness_hours", 24.0),
        )

        # Warning threshold is typically 50% of max staleness
        warning_threshold = view_config.get(
            "warning_threshold_hours",
            max_staleness * 0.5,
        )

        return float(max_staleness), float(warning_threshold)

    async def get_feature_statistics(
        self,
        feature_view: str,
        feature_name: str,
        refresh: bool = False,
        supabase_client=None,
    ) -> Optional[FeatureStatistics]:
        """Get statistics for a specific feature.

        Computes actual statistics by querying the offline store (Supabase).
        Results are cached for efficiency.

        Args:
            feature_view: Name of the feature view.
            feature_name: Name of the feature.
            refresh: If True, bypass cache and fetch fresh stats.
            supabase_client: Optional Supabase client for querying offline store.

        Returns:
            FeatureStatistics or None if not available.
        """
        await self.initialize()

        cache_key = f"{feature_view}:{feature_name}"

        # Check cache
        if not refresh and cache_key in self._stats_cache:
            cache_time = self._stats_cache_time.get(cache_key)
            if (
                cache_time
                and (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_seconds
            ):
                return self._stats_cache[cache_key]

        # Compute statistics from offline store
        try:
            stats = await self._compute_feature_statistics(
                feature_view=feature_view,
                feature_name=feature_name,
                supabase_client=supabase_client,
            )

            if stats:
                self._stats_cache[cache_key] = stats
                self._stats_cache_time[cache_key] = datetime.now()

            return stats

        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return None

    async def _compute_feature_statistics(
        self,
        feature_view: str,
        feature_name: str,
        supabase_client=None,
    ) -> Optional[FeatureStatistics]:
        """Compute actual statistics for a feature from offline store.

        Args:
            feature_view: Name of the feature view.
            feature_name: Name of the feature.
            supabase_client: Supabase client for querying.

        Returns:
            FeatureStatistics with computed values.
        """
        # Map feature views to source tables
        feature_view_tables = self._materialization_config.get("feature_views", {})
        view_config = feature_view_tables.get(feature_view, {})
        source_table = view_config.get("source_table")

        if not source_table and supabase_client:
            # Try to infer table from feature view name
            # Convention: hcp_conversion_features -> hcp_profiles or similar
            source_table = self._infer_source_table(feature_view)

        if not source_table:
            logger.debug(f"No source table found for feature view: {feature_view}")
            # Return placeholder if no source table configured
            return FeatureStatistics(
                feature_view=feature_view,
                feature_name=feature_name,
                count=0,
                null_count=0,
                last_updated=datetime.now(),
            )

        # Query statistics from Supabase
        if supabase_client:
            try:
                stats = await self._query_statistics_from_supabase(
                    client=supabase_client,
                    table_name=source_table,
                    column_name=feature_name,
                    feature_view=feature_view,
                )
                return stats
            except Exception as e:
                logger.warning(f"Failed to query stats from Supabase: {e}")

        # Fallback: try to compute from Feast offline store
        if self._store:
            try:
                return await self._compute_stats_from_feast(
                    feature_view=feature_view,
                    feature_name=feature_name,
                )
            except Exception as e:
                logger.warning(f"Failed to compute stats from Feast: {e}")

        # Final fallback: placeholder stats
        return FeatureStatistics(
            feature_view=feature_view,
            feature_name=feature_name,
            count=0,
            null_count=0,
            last_updated=datetime.now(),
        )

    def _infer_source_table(self, feature_view: str) -> Optional[str]:
        """Infer source table name from feature view name.

        Convention mappings:
        - hcp_conversion_features -> hcp_profiles
        - patient_journey_features -> patient_journeys
        - trigger_features -> triggers
        - market_features -> business_metrics
        """
        mappings = {
            "hcp_conversion_features": "hcp_profiles",
            "hcp_features": "hcp_profiles",
            "patient_journey_features": "patient_journeys",
            "patient_features": "patient_journeys",
            "trigger_features": "triggers",
            "market_features": "business_metrics",
            "business_features": "business_metrics",
        }
        return mappings.get(feature_view)

    async def _query_statistics_from_supabase(
        self,
        client,
        table_name: str,
        column_name: str,
        feature_view: str,
    ) -> FeatureStatistics:
        """Query feature statistics directly from Supabase.

        Uses SQL aggregate functions for efficient computation.

        Args:
            client: Supabase client.
            table_name: Source table name.
            column_name: Column to compute stats for.
            feature_view: Feature view name for result.

        Returns:
            FeatureStatistics with computed values.
        """
        # Build SQL for statistics computation
        stats_query = f"""
            SELECT
                COUNT(*) as total_count,
                COUNT({column_name}) as non_null_count,
                COUNT(*) - COUNT({column_name}) as null_count,
                MIN({column_name}::numeric) as min_val,
                MAX({column_name}::numeric) as max_val,
                AVG({column_name}::numeric) as mean_val,
                STDDEV({column_name}::numeric) as stddev_val
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
        """

        try:
            # Execute via RPC or direct query
            result = await asyncio.to_thread(
                lambda: client.rpc("execute_sql", {"query": stats_query}).execute()
            )

            if result.data and len(result.data) > 0:
                row = result.data[0]
                return FeatureStatistics(
                    feature_view=feature_view,
                    feature_name=column_name,
                    count=int(row.get("total_count", 0)),
                    null_count=int(row.get("null_count", 0)),
                    min_value=float(row["min_val"]) if row.get("min_val") else None,
                    max_value=float(row["max_val"]) if row.get("max_val") else None,
                    mean_value=float(row["mean_val"]) if row.get("mean_val") else None,
                    stddev_value=float(row["stddev_val"]) if row.get("stddev_val") else None,
                    last_updated=datetime.now(),
                )
        except Exception as e:
            # Try simpler count-only query if stats query fails
            logger.debug(f"Full stats query failed: {e}. Trying count-only.")

            try:
                count_result = await asyncio.to_thread(
                    lambda: client.table(table_name).select("*", count="exact").limit(0).execute()
                )
                total_count = count_result.count if hasattr(count_result, "count") else 0

                return FeatureStatistics(
                    feature_view=feature_view,
                    feature_name=column_name,
                    count=total_count,
                    null_count=0,
                    last_updated=datetime.now(),
                )
            except Exception as count_error:
                logger.warning(f"Count query also failed: {count_error}")

        # Return placeholder if all queries fail
        return FeatureStatistics(
            feature_view=feature_view,
            feature_name=column_name,
            count=0,
            null_count=0,
            last_updated=datetime.now(),
        )

    async def _compute_stats_from_feast(
        self,
        feature_view: str,
        feature_name: str,
    ) -> FeatureStatistics:
        """Compute statistics from Feast offline store.

        Uses Feast's data source to compute statistics.

        Args:
            feature_view: Feature view name.
            feature_name: Feature name.

        Returns:
            FeatureStatistics with computed values.
        """
        if not self._store:
            raise RuntimeError("Feast store not initialized")

        try:
            # Get feature view definition
            fv = self._store.get_feature_view(feature_view)

            # Access the data source
            data_source = fv.batch_source if hasattr(fv, "batch_source") else None

            if data_source and hasattr(data_source, "get_table_query_string"):
                # Could query the source directly
                # This is implementation-specific to the data source type
                pass

            # For now, get a sample of recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # Create minimal entity df for sampling
            entity_df = pd.DataFrame(
                {
                    "event_timestamp": [start_date, end_date],
                }
            )

            # Add required entity columns based on feature view
            for entity in fv.entity_columns if hasattr(fv, "entity_columns") else []:
                entity_df[entity] = ["sample_id", "sample_id"]

            # Get historical features
            retrieval_job = self._store.get_historical_features(
                entity_df=entity_df,
                features=[f"{feature_view}:{feature_name}"],
            )
            df = retrieval_job.to_df()

            # Compute statistics
            col_name = f"{feature_view}__{feature_name}"
            if col_name not in df.columns:
                col_name = feature_name

            if col_name in df.columns:
                series = df[col_name]
                return FeatureStatistics(
                    feature_view=feature_view,
                    feature_name=feature_name,
                    count=len(series),
                    null_count=int(series.isna().sum()),
                    min_value=float(series.min()) if pd.notna(series.min()) else None,
                    max_value=float(series.max()) if pd.notna(series.max()) else None,
                    mean_value=float(series.mean()) if pd.notna(series.mean()) else None,
                    stddev_value=float(series.std()) if pd.notna(series.std()) else None,
                    last_updated=datetime.now(),
                )

        except Exception as e:
            logger.warning(f"Failed to compute stats from Feast: {e}")

        # Fallback
        return FeatureStatistics(
            feature_view=feature_view,
            feature_name=feature_name,
            count=0,
            null_count=0,
            last_updated=datetime.now(),
        )

    async def list_feature_views(self) -> List[Dict[str, Any]]:
        """List all registered feature views.

        Returns:
            List of feature view metadata dictionaries.
        """
        await self.initialize()
        self._ensure_initialized()

        if not self._store:
            return []

        try:
            feature_views = self._store.list_feature_views()
            return [
                {
                    "name": fv.name,
                    "entities": list(fv.entity_columns) if hasattr(fv, "entity_columns") else [],
                    "features": [f.name for f in fv.schema] if hasattr(fv, "schema") else [],
                    "ttl": str(fv.ttl) if hasattr(fv, "ttl") else None,
                    "online": fv.online if hasattr(fv, "online") else False,
                    "tags": dict(fv.tags) if hasattr(fv, "tags") else {},
                }
                for fv in feature_views
            ]
        except Exception as e:
            logger.error(f"Error listing feature views: {e}")
            return []

    async def list_entities(self) -> List[Dict[str, Any]]:
        """List all registered entities.

        Returns:
            List of entity metadata dictionaries.
        """
        await self.initialize()
        self._ensure_initialized()

        if not self._store:
            return []

        try:
            entities = self._store.list_entities()
            return [
                {
                    "name": e.name,
                    "join_keys": [e.join_key] if hasattr(e, "join_key") else [],
                    "description": e.description,
                    "tags": dict(e.tags) if hasattr(e, "tags") else {},
                }
                for e in entities
            ]
        except Exception as e:
            logger.error(f"Error listing entities: {e}")
            return []

    async def get_feature_freshness(
        self,
        feature_view: str,
    ) -> FeatureFreshness:
        """Check feature freshness for a feature view.

        Determines freshness based on:
        1. Time since last materialization
        2. Configured staleness thresholds

        Status levels:
        - FRESH: Within warning threshold
        - WARNING: Between warning and max staleness
        - STALE: Beyond max staleness but within 2x
        - EXPIRED: Beyond 2x max staleness
        - UNKNOWN: No materialization record

        Args:
            feature_view: Name of the feature view.

        Returns:
            FeatureFreshness object with status and timing info.
        """
        await self.initialize()
        self._ensure_initialized()

        # Get thresholds from config
        max_staleness_hours, warning_threshold_hours = self._get_freshness_thresholds(feature_view)

        # Get last materialization time
        last_materialized = self._materialization_timestamps.get(feature_view)

        # Calculate age and determine status
        if last_materialized is None:
            return FeatureFreshness(
                feature_view=feature_view,
                last_materialized=None,
                freshness_status=FreshnessStatus.UNKNOWN,
                age_hours=None,
                max_staleness_hours=max_staleness_hours,
                warning_threshold_hours=warning_threshold_hours,
                is_fresh=False,
                message="No materialization record found",
            )

        # Calculate age in hours
        age_seconds = (datetime.now() - last_materialized).total_seconds()
        age_hours = age_seconds / 3600.0

        # Determine status
        if age_hours <= warning_threshold_hours:
            status = FreshnessStatus.FRESH
            is_fresh = True
            message = f"Features are fresh (age: {age_hours:.1f}h)"
        elif age_hours <= max_staleness_hours:
            status = FreshnessStatus.WARNING
            is_fresh = True
            message = f"Features approaching staleness (age: {age_hours:.1f}h, max: {max_staleness_hours:.1f}h)"
        elif age_hours <= max_staleness_hours * 2:
            status = FreshnessStatus.STALE
            is_fresh = False
            message = f"Features are stale (age: {age_hours:.1f}h, max: {max_staleness_hours:.1f}h)"
        else:
            status = FreshnessStatus.EXPIRED
            is_fresh = False
            message = (
                f"Features are expired (age: {age_hours:.1f}h, max: {max_staleness_hours:.1f}h)"
            )

        return FeatureFreshness(
            feature_view=feature_view,
            last_materialized=last_materialized,
            freshness_status=status,
            age_hours=age_hours,
            max_staleness_hours=max_staleness_hours,
            warning_threshold_hours=warning_threshold_hours,
            is_fresh=is_fresh,
            message=message,
        )

    async def get_all_freshness(self) -> Dict[str, FeatureFreshness]:
        """Get freshness status for all tracked feature views.

        Returns:
            Dictionary mapping feature view names to freshness status.
        """
        await self.initialize()

        result = {}

        # Get all known feature views
        feature_views = self._get_all_feature_view_names()

        # Also include any views we've tracked materialization for
        for view_name in set(list(feature_views) + list(self._materialization_timestamps.keys())):
            result[view_name] = await self.get_feature_freshness(view_name)

        return result

    def record_materialization(
        self,
        feature_view: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Manually record a materialization timestamp.

        Useful for external materialization tracking or testing.

        Args:
            feature_view: Name of the feature view.
            timestamp: Materialization timestamp (defaults to now).
        """
        self._materialization_timestamps[feature_view] = timestamp or datetime.now()
        logger.debug(f"Recorded materialization for {feature_view}")

    async def apply(self) -> bool:
        """Apply feature definitions to the registry.

        Typically called during deployment to register/update features.

        Returns:
            True if successful, False otherwise.
        """
        await self.initialize()

        if not self._store:
            logger.warning("Feast store not available. Cannot apply.")
            return False

        try:
            self._store.apply([])  # Apply all definitions from repo
            logger.info("Feature definitions applied to registry")
            return True
        except Exception as e:
            logger.error(f"Error applying feature definitions: {e}")
            return False

    async def close(self) -> None:
        """Close client connections and cleanup temp files."""
        import shutil

        self._store = None
        self._custom_store = None
        self._initialized = False
        self._stats_cache.clear()
        self._stats_cache_time.clear()
        self._materialization_timestamps.clear()

        # Clean up temp directory if created
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
            self._temp_dir = None

        logger.info("Feast client closed")


# Singleton instance for convenience
_client: Optional[FeastClient] = None


async def get_feast_client(config: Optional[FeastConfig] = None) -> FeastClient:
    """Get or create the singleton Feast client.

    Args:
        config: Optional FeastConfig. Only used when creating new client.

    Returns:
        Initialized FeastClient instance.
    """
    global _client
    if _client is None:
        _client = FeastClient(config=config)
        await _client.initialize()
    return _client
