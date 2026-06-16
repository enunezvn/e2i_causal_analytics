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
        supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),  # or SUPABASE_SERVICE_ROLE_KEY
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
from src.utils.supabase_env import resolve_supabase_service_key

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
        # Prefer the service-role key (any of its deployment env-var names) so
        # reads of the service_role-only ml_* tables are not denied (42501);
        # anon is a dev/test fallback only. See src/utils/supabase_env.py.
        self.supabase_key = resolve_supabase_service_key(supabase_key)

        self._client: Any = None
        self._initialized = False
        # feature_name -> feature_id (uuid) cache. feature_values.feature_id is a
        # uuid FK to features.id; query_features resolves names through this.
        self._feature_id_cache: dict[str, str] = {}

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
        include_synthetic: bool = False,
    ) -> dict[str, FeatureData]:
        """Query feature values from Supabase feature store.

        Queries the feature_values table for the specified features within
        the given time window. Supports filtering by brand, geography, etc.

        Args:
            feature_names: List of feature names to retrieve
            time_window: Time window for the query
            filters: Optional filters (brand, geography_id, etc.)
            include_synthetic: When False (default) synthetic feature_values
                rows are excluded (#894 codex R2 — feature_values is tagged by
                migration 069; real drift checks must not ingest planted
                values). Validation runs opt in, mirroring get_predictions.

        Returns:
            Dictionary mapping feature name to FeatureData
        """
        await self._ensure_initialized()

        from src.repositories.provenance import apply_provenance_filter

        result = {}

        feature_id_map = self._resolve_feature_ids(feature_names)
        for feature_name in feature_names:
            feature_id = feature_id_map.get(feature_name)
            if feature_id is None:
                # Not in the `features` registry -> honest empty FeatureData.
                # NEVER query feature_values by a bare name (the 22P02 uuid bug
                # that made every monitoring run hollow).
                result[feature_name] = FeatureData(
                    feature_name=feature_name,
                    values=np.array([]),
                    timestamps=np.array([]),
                    time_window=time_window,
                )
                logger.warning(f"Feature '{feature_name}' not in `features` registry; skipping")
                continue
            try:
                # Query feature values for this feature by its resolved uuid id
                query = (
                    self._client.table("feature_values")
                    .select("value, event_timestamp, entity_values")
                    .eq("feature_id", feature_id)
                    .gte("event_timestamp", time_window.start.isoformat())
                    .lte("event_timestamp", time_window.end.isoformat())
                )

                # Apply filters
                if filters:
                    for key, value in filters.items():
                        # Filter on entity_values JSONB field
                        query = query.contains("entity_values", {key: value})
                query = apply_provenance_filter(query, include_synthetic=include_synthetic)

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
        include_synthetic: bool = False,
    ) -> PredictionData:
        """Query prediction data from Supabase.

        Queries the predictions table for model predictions within
        the given time window.

        Args:
            model_id: Model identifier (UUID or name)
            time_window: Time window for the query
            filters: Optional filters (segment, brand, etc.)
            include_synthetic: When False (default) synthetic ml_predictions
                rows are excluded from drift detection (Shard 07) — a
                synthetic prediction must not register as real input drift.

        Returns:
            PredictionData containing predictions
        """
        await self._ensure_initialized()

        try:
            # Query ml_predictions table.
            #
            # Issue #188: exclude gated audit rows from drift monitoring.
            # Gated rows (prediction_class==GATED_HONEST_FAILURE_SENTINEL,
            # written by src/tasks/risk_score_prediction_tasks.py when a
            # model failed its honest-failure gate) carry raw un-gated
            # scores that MUST NOT feed drift detection; including them
            # would silently double-count the failure as either input
            # drift or unstable predictions. Codex pass-5 MEDIUM: use a
            # NULL-preserving filter (prediction_class.is.null OR
            # neq.sentinel) since prediction_class is nullable and
            # historical rows may have it set to NULL.
            from src.repositories.prediction import GATED_HONEST_FAILURE_SENTINEL
            from src.repositories.provenance import apply_provenance_filter

            query = (
                self._client.table("ml_predictions")
                .select("confidence_score, prediction_value, created_at, patient_id, hcp_id")
                .eq("model_version", model_id)
                .or_(
                    f"prediction_class.is.null,prediction_class.neq.{GATED_HONEST_FAILURE_SENTINEL}"
                )
                .gte("created_at", time_window.start.isoformat())
                .lte("created_at", time_window.end.isoformat())
            )
            # Shard 07: default-exclude synthetic rows from drift monitoring.
            query = apply_provenance_filter(query, include_synthetic=include_synthetic)

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
                entity_ids = np.array(
                    [row.get("patient_id") or row.get("hcp_id") or "" for row in response.data]
                )

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
        include_synthetic: bool = False,
    ) -> PredictionData:
        """Query predictions with actual labels for concept drift.

        Joins predictions with actual outcomes for concept drift detection.
        This requires ground truth to be available.

        Args:
            model_id: Model identifier
            time_window: Time window for the query
            filters: Optional filters
            include_synthetic: When False (default) synthetic ml_predictions
                rows are excluded from concept-drift detection (Shard 07).

        Returns:
            PredictionData with both predicted and actual labels
        """
        await self._ensure_initialized()

        try:
            # Query ml_predictions with ground truth outcomes.
            # Issue #188: exclude gated audit rows with a NULL-preserving
            # filter (see codex pass-5 MEDIUM rationale on
            # query_predictions above).
            from src.repositories.prediction import GATED_HONEST_FAILURE_SENTINEL
            from src.repositories.provenance import apply_provenance_filter

            query = (
                self._client.table("ml_predictions")
                .select(
                    "confidence_score, prediction_value, created_at, "
                    "patient_id, hcp_id, actual_outcome"
                )
                .eq("model_version", model_id)
                .or_(
                    f"prediction_class.is.null,prediction_class.neq.{GATED_HONEST_FAILURE_SENTINEL}"
                )
                .gte("created_at", time_window.start.isoformat())
                .lte("created_at", time_window.end.isoformat())
                .not_.is_("actual_outcome", "null")  # Only include labeled data
            )
            # Shard 07: default-exclude synthetic rows from concept drift.
            query = apply_provenance_filter(query, include_synthetic=include_synthetic)

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
                entity_ids = np.array(
                    [row.get("patient_id") or row.get("hcp_id") or "" for row in response.data]
                )

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
            from src.repositories.provenance import apply_provenance_filter

            # #894 codex R2: features is is_synthetic-tagged (migration 069) —
            # the sweep must not auto-select planted feature names.
            query = self._client.table("features").select("name, feature_group_id")

            if source_table:
                # Filter by feature group's source table
                query = query.eq("feature_groups.source_table", source_table)
            query = apply_provenance_filter(query)

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
            from src.repositories.provenance import apply_provenance_filter

            # #894: ml_model_registry is is_synthetic-tagged and the synthetic
            # generator stamps stage='production' — the drift sweep must never
            # enumerate planted models. Codex R1 also surfaced that the old
            # projection named columns that do not exist on the live schema
            # (name/version/metrics/created_at vs the canonical model_name/
            # model_version/registered_at — see explain.py's note and
            # database/ml/mlops_tables.sql), so every call 42703'd into the
            # except-branch [] and the 6-hourly production sweep reported
            # "No production models found" forever. Realigned to live columns.
            query = self._client.table("ml_model_registry").select(
                "id, model_name, model_version, stage, registered_at"
            )

            if stage:
                query = query.eq("stage", stage)
            query = apply_provenance_filter(query)

            response = query.order("registered_at", desc=True).execute()

            if response.data:
                return response.data  # type: ignore[no-any-return]
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

            # Check ml_predictions table (#894 codex R2: the live PK is
            # prediction_id — probing "id" was a reachable 42703 that aborted
            # the whole try block, leaving predictions/features flags False)
            response = (
                self._client.table("ml_predictions").select("prediction_id").limit(1).execute()
            )
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

    def _resolve_feature_ids(self, feature_names: list[str]) -> dict[str, str]:
        """Resolve feature names to their ``feature_id`` (uuid) via the
        ``features`` registry, cached on the instance.

        ``feature_values.feature_id`` is a UUID FK to ``features.id``. Filtering
        it with a bare feature NAME raises 22P02 (invalid input syntax for type
        uuid) and silently drops every feature — the bug that left all
        monitoring runs hollow ("15 checks / 0 drift") and ml_drift_history
        empty. Resolve each unseen name once and reuse. Names absent from the
        registry are omitted; the caller emits honest empty FeatureData.
        """
        # Lazy-init: some call sites construct the connector via __new__ (tests),
        # bypassing __init__, so don't assume the attribute exists.
        cache = getattr(self, "_feature_id_cache", None)
        if cache is None:
            cache = self._feature_id_cache = {}
        missing = [n for n in feature_names if n not in cache]
        if missing:
            resp = (
                self._client.table("features").select("id, name").in_("name", missing).execute()
            )
            for row in resp.data or []:
                cache[row["name"]] = row["id"]
        return {n: cache[n] for n in feature_names if n in cache}

    def _extract_value(self, value: Any) -> Any:
        """Extract a feature value, PRESERVING categorical labels.

        Numeric values (incl. numeric strings and JSONB-wrapped numbers) become
        float. Non-numeric categorical labels (e.g. 'rheumatology', 'low') are
        returned UNCHANGED so the drift node's non-numeric path label-encodes
        them. Forcing float here silently zeroed categoricals (str branch) AND
        raised on a JSONB-wrapped category (dict branch) — leaving every
        categorical feature unusable for drift.
        """
        if isinstance(value, dict):
            # Unwrap JSONB {"value": ...} / {"v": ...}
            value = value.get("value", value.get("v"))
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value  # categorical label preserved for the drift node
        return value

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
