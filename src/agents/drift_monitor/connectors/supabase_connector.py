"""Supabase data connector for production drift detection.

This module provides the production implementation of BaseDataConnector
that queries real data from Supabase for drift detection.

The connector queries:
- Feature values from the feature store tables
- Predictions from the ml_predictions table (with optional actual_outcome for concept drift)
- Model registry for available models

Example:
    connector = SupabaseDataConnector(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )

    data = await connector.query_features(
        feature_names=["age", "income"],
        time_window=TimeWindow(start=..., end=..., label="baseline"),
        filters={"brand": "remibrutinib"}
    )
"""

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np

from src.agents.drift_monitor.connectors.base import (
    BaseDataConnector,
    FeatureData,
    PredictionData,
    TimeWindow,
)

logger = logging.getLogger(__name__)


class SupabaseDataConnector(BaseDataConnector):
    """Production data connector using Supabase.

    This connector queries real data from Supabase for drift detection:
    - Feature data from feature_values table
    - Prediction data from predictions table
    - Model metadata from ml_model_registry

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        _client: Supabase client instance
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        """Initialize Supabase connector.

        Args:
            supabase_url: Supabase project URL (defaults to env var)
            supabase_key: Supabase API key (defaults to env var)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = (
            supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        )

        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazily initialize Supabase client."""
        if not self._initialized:
            try:
                from supabase import create_client

                if not self.supabase_url or not self.supabase_key:
                    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

                self._client = create_client(self.supabase_url, self.supabase_key)
                self._initialized = True
                logger.info("SupabaseDataConnector initialized successfully")
            except ImportError:
                raise ImportError("supabase package not installed. Run: pip install supabase")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                raise

    async def query_features(
        self,
        feature_names: list[str],
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, FeatureData]:
        """Query feature values from Supabase feature store.

        Queries the feature_values table for the specified features within
        the given time window. Supports filtering by brand, geography, etc.

        Args:
            feature_names: List of feature names to retrieve
            time_window: Time window for the query
            filters: Optional filters (brand, geography_id, etc.)

        Returns:
            Dictionary mapping feature name to FeatureData
        """
        await self._ensure_initialized()

        result = {}

        for feature_name in feature_names:
            try:
                # Query feature values for this feature
                query = (
                    self._client.table("feature_values")
                    .select("value, event_timestamp, entity_values")
                    .eq("feature_id", self._get_feature_id_subquery(feature_name))
                    .gte("event_timestamp", time_window.start.isoformat())
                    .lte("event_timestamp", time_window.end.isoformat())
                )

                # Apply filters
                if filters:
                    for key, value in filters.items():
                        # Filter on entity_values JSONB field
                        query = query.contains("entity_values", {key: value})

                response = query.order("event_timestamp", desc=False).execute()

                if response.data:
                    values = np.array([self._extract_value(row["value"]) for row in response.data])
                    timestamps = np.array(
                        [
                            datetime.fromisoformat(row["event_timestamp"].replace("Z", "+00:00"))
                            for row in response.data
                        ]
                    )
                    entity_ids = np.array(
                        [str(row.get("entity_values", {})) for row in response.data]
                    )

                    result[feature_name] = FeatureData(
                        feature_name=feature_name,
                        values=values,
                        timestamps=timestamps,
                        entity_ids=entity_ids,
                        time_window=time_window,
                    )
                else:
                    # No data found - return empty FeatureData
                    result[feature_name] = FeatureData(
                        feature_name=feature_name,
                        values=np.array([]),
                        timestamps=np.array([]),
                        time_window=time_window,
                    )
                    logger.warning(f"No data found for feature '{feature_name}' in time window")

            except Exception as e:
                logger.error(f"Error querying feature '{feature_name}': {e}")
                # Return empty data for failed features
                result[feature_name] = FeatureData(
                    feature_name=feature_name,
                    values=np.array([]),
                    time_window=time_window,
                )

        return result

    async def query_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Query prediction data from Supabase.

        Queries the predictions table for model predictions within
        the given time window.

        Args:
            model_id: Model identifier (UUID or name)
            time_window: Time window for the query
            filters: Optional filters (segment, brand, etc.)

        Returns:
            PredictionData containing predictions
        """
        await self._ensure_initialized()

        try:
            # Query ml_predictions table
            query = (
                self._client.table("ml_predictions")
                .select("confidence_score, prediction_value, created_at, entity_id")
                .eq("model_version", model_id)
                .gte("created_at", time_window.start.isoformat())
                .lte("created_at", time_window.end.isoformat())
            )

            # Apply filters
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)

            response = query.order("created_at", desc=False).execute()

            if response.data:
                scores = np.array([row.get("confidence_score", 0.5) for row in response.data])
                labels = np.array(
                    [
                        self._prediction_to_label(row.get("prediction_value"))
                        for row in response.data
                    ]
                )
                timestamps = np.array(
                    [
                        datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                        for row in response.data
                    ]
                )
                entity_ids = np.array([row.get("entity_id", "") for row in response.data])

                return PredictionData(
                    model_id=model_id,
                    scores=scores,
                    labels=labels,
                    timestamps=timestamps,
                    entity_ids=entity_ids,
                    time_window=time_window,
                )
            else:
                logger.warning(f"No predictions found for model '{model_id}' in time window")
                return PredictionData(
                    model_id=model_id,
                    scores=np.array([]),
                    labels=np.array([]),
                    time_window=time_window,
                )

        except Exception as e:
            logger.error(f"Error querying predictions for model '{model_id}': {e}")
            return PredictionData(
                model_id=model_id,
                scores=np.array([]),
                labels=np.array([]),
                time_window=time_window,
            )

    async def query_labeled_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: dict[str, Any] | None = None,
    ) -> PredictionData:
        """Query predictions with actual labels for concept drift.

        Joins predictions with actual outcomes for concept drift detection.
        This requires ground truth to be available.

        Args:
            model_id: Model identifier
            time_window: Time window for the query
            filters: Optional filters

        Returns:
            PredictionData with both predicted and actual labels
        """
        await self._ensure_initialized()

        try:
            # Query ml_predictions with ground truth outcomes
            query = (
                self._client.table("ml_predictions")
                .select("confidence_score, prediction_value, created_at, entity_id, actual_outcome")
                .eq("model_version", model_id)
                .gte("created_at", time_window.start.isoformat())
                .lte("created_at", time_window.end.isoformat())
                .not_.is_("actual_outcome", "null")  # Only include labeled data
            )

            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)

            response = query.order("created_at", desc=False).execute()

            if response.data:
                scores = np.array([row.get("confidence_score", 0.5) for row in response.data])
                labels = np.array(
                    [
                        self._prediction_to_label(row.get("prediction_value"))
                        for row in response.data
                    ]
                )
                actual_labels = np.array(
                    [self._prediction_to_label(row.get("actual_outcome")) for row in response.data]
                )
                timestamps = np.array(
                    [
                        datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                        for row in response.data
                    ]
                )
                entity_ids = np.array([row.get("entity_id", "") for row in response.data])

                return PredictionData(
                    model_id=model_id,
                    scores=scores,
                    labels=labels,
                    actual_labels=actual_labels,
                    timestamps=timestamps,
                    entity_ids=entity_ids,
                    time_window=time_window,
                )
            else:
                logger.warning(f"No labeled predictions found for model '{model_id}'")
                return PredictionData(
                    model_id=model_id,
                    scores=np.array([]),
                    labels=np.array([]),
                    actual_labels=np.array([]),
                    time_window=time_window,
                )

        except Exception as e:
            logger.error(f"Error querying labeled predictions: {e}")
            return PredictionData(
                model_id=model_id,
                scores=np.array([]),
                labels=np.array([]),
                actual_labels=np.array([]),
                time_window=time_window,
            )

    async def get_available_features(
        self,
        source_table: str | None = None,
    ) -> list[str]:
        """Get list of available features from feature store.

        Args:
            source_table: Optional table name to filter features

        Returns:
            List of available feature names
        """
        await self._ensure_initialized()

        try:
            query = self._client.table("features").select("name, feature_group_id")

            if source_table:
                # Filter by feature group's source table
                query = query.eq("feature_groups.source_table", source_table)

            response = query.execute()

            if response.data:
                return [row["name"] for row in response.data]
            return []

        except Exception as e:
            logger.error(f"Error getting available features: {e}")
            return []

    async def get_available_models(
        self,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of available models from model registry.

        Args:
            stage: Optional stage filter ("production", "staging", etc.)

        Returns:
            List of model metadata dictionaries
        """
        await self._ensure_initialized()

        try:
            query = self._client.table("ml_model_registry").select(
                "id, name, version, stage, metrics, created_at"
            )

            if stage:
                query = query.eq("stage", stage)

            response = query.order("created_at", desc=True).execute()

            if response.data:
                return response.data
            return []

        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    async def health_check(self) -> dict[str, bool]:
        """Check connector health and connectivity.

        Returns:
            Dictionary with health status for each component
        """
        health = {
            "connected": False,
            "database": False,
            "predictions_table": False,
            "features_table": False,
            "models_table": False,
        }

        try:
            await self._ensure_initialized()
            health["connected"] = True

            # Check database connectivity
            response = self._client.table("ml_model_registry").select("id").limit(1).execute()
            health["database"] = True
            health["models_table"] = len(response.data) >= 0

            # Check ml_predictions table
            response = self._client.table("ml_predictions").select("id").limit(1).execute()
            health["predictions_table"] = len(response.data) >= 0

            # Check features table
            response = self._client.table("features").select("id").limit(1).execute()
            health["features_table"] = len(response.data) >= 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return health

    async def close(self) -> None:
        """Close Supabase client connection."""
        self._client = None
        self._initialized = False
        logger.info("SupabaseDataConnector closed")

    def _get_feature_id_subquery(self, feature_name: str) -> str:
        """Get feature ID for a feature name.

        This is a simplified version - in production, you'd want to
        cache feature IDs or use a join.

        Args:
            feature_name: Name of the feature

        Returns:
            Feature ID (or the name as fallback)
        """
        # For simplicity, using feature name directly
        # In production, implement proper feature ID lookup
        return feature_name

    def _extract_value(self, value: Any) -> float:
        """Extract numeric value from feature value.

        Handles JSONB values that may be wrapped.

        Args:
            value: Raw value from database

        Returns:
            Numeric value as float
        """
        if isinstance(value, dict):
            # Handle JSONB wrapper
            return float(value.get("value", value.get("v", 0)))
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    def _prediction_to_label(self, value: Any) -> int:
        """Convert prediction value to integer label.

        Args:
            value: Prediction value

        Returns:
            Integer label (0 or 1 for binary)
        """
        if isinstance(value, bool):
            return int(value)
        elif isinstance(value, (int, float)):
            return int(value > 0.5) if isinstance(value, float) else int(value)
        elif isinstance(value, str):
            return 1 if value.lower() in ("true", "yes", "1", "positive") else 0
        return 0
