"""Feast Feature Store adapter for prediction_synthesizer.

This module provides Feast integration for online feature retrieval
during prediction time. Implements the FeatureStore protocol with
extended capabilities for real-time feature lookup.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_feature_analyzer_adapter():
    """Get FeatureAnalyzerAdapter (lazy import to avoid circular deps)."""
    try:
        from src.feature_store.client import FeatureStoreClient
        from src.feature_store.feature_analyzer_adapter import (
            get_feature_analyzer_adapter,
        )

        # Create client - uses default Supabase config
        fs_client = FeatureStoreClient()
        return get_feature_analyzer_adapter(
            feature_store_client=fs_client,
            enable_feast=True,
        )
    except Exception as e:
        logger.warning(f"Could not initialize FeatureAnalyzerAdapter: {e}")
        return None


class FeastFeatureStore:
    """Feast-backed feature store for prediction_synthesizer.

    This adapter provides:
    1. Feature importance retrieval (via SHAP values in Feast)
    2. Online feature retrieval for real-time predictions
    3. Feature freshness validation

    The adapter implements graceful degradation - if Feast is unavailable,
    methods return empty results rather than raising exceptions.
    """

    def __init__(
        self,
        adapter=None,
        default_feature_view: str = "hcp_features",
        entity_key: str = "hcp_id",
    ):
        """Initialize the Feast feature store adapter.

        Args:
            adapter: Optional FeatureAnalyzerAdapter instance. If None,
                will be lazily initialized on first use.
            default_feature_view: Default feature view for lookups
            entity_key: Default entity key column name
        """
        self._adapter = adapter
        self._default_feature_view = default_feature_view
        self._entity_key = entity_key
        self._initialized = False

    def _ensure_adapter(self):
        """Ensure adapter is initialized."""
        if self._adapter is None and not self._initialized:
            self._adapter = _get_feature_analyzer_adapter()
            self._initialized = True
        return self._adapter

    async def get_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for a model.

        Retrieves SHAP-based feature importance values stored in Feast
        metadata or from the model registry.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary mapping feature names to importance scores (0-1)
        """
        adapter = self._ensure_adapter()
        if adapter is None:
            logger.debug(f"Feast unavailable, no importance for {model_id}")
            return {}

        try:
            # Try to get importance from Feast feature metadata
            importance = await adapter.get_feature_importance(
                model_id=model_id,
                feature_view=self._default_feature_view,
            )
            return importance if importance else {}
        except AttributeError:
            # Adapter doesn't have get_feature_importance - fallback
            logger.debug(f"get_feature_importance not available for {model_id}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get feature importance for {model_id}: {e}")
            return {}

    async def get_online_features(
        self,
        entity_id: str,
        feature_refs: Optional[List[str]] = None,
        feature_view: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get online (real-time) features for an entity.

        Fetches the latest feature values from Feast online store.
        This is used during prediction time to get current feature values.

        Args:
            entity_id: Entity identifier (e.g., HCP ID)
            feature_refs: Optional list of specific features to retrieve.
                If None, retrieves all features from the feature view.
            feature_view: Optional feature view name. Uses default if not specified.

        Returns:
            Dictionary mapping feature names to current values
        """
        adapter = self._ensure_adapter()
        if adapter is None:
            logger.debug(f"Feast unavailable, no online features for {entity_id}")
            return {}

        view = feature_view or self._default_feature_view

        try:
            # Build feature references
            if not feature_refs:
                feature_refs = [f"{view}:*"]

            # Fetch from Feast online store
            features = await adapter.get_online_features(
                entity_dict={self._entity_key: entity_id},
                feature_refs=feature_refs,
            )

            if features is None:
                return {}

            # Remove entity key and metadata from result
            result = {
                k: v
                for k, v in features.items()
                if k not in [self._entity_key, "event_timestamp", "__feast_event_timestamp"]
            }

            logger.debug(f"Retrieved {len(result)} online features for entity {entity_id}")
            return result

        except Exception as e:
            logger.warning(f"Failed to get online features for {entity_id}: {e}")
            return {}

    async def get_online_features_batch(
        self,
        entity_ids: List[str],
        feature_refs: Optional[List[str]] = None,
        feature_view: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get online features for multiple entities.

        Batch retrieval for efficiency when predicting multiple entities.

        Args:
            entity_ids: List of entity identifiers
            feature_refs: Optional list of specific features
            feature_view: Optional feature view name

        Returns:
            Dictionary mapping entity_id to feature dictionaries
        """
        adapter = self._ensure_adapter()
        if adapter is None:
            return {eid: {} for eid in entity_ids}

        view = feature_view or self._default_feature_view

        try:
            if not feature_refs:
                feature_refs = [f"{view}:*"]

            # Build entity list for batch lookup
            entity_dicts = [{self._entity_key: eid} for eid in entity_ids]

            # Fetch from Feast
            batch_features = await adapter.get_online_features_batch(
                entity_dicts=entity_dicts,
                feature_refs=feature_refs,
            )

            if batch_features is None:
                return {eid: {} for eid in entity_ids}

            # Process results
            result = {}
            for entity_id, features in zip(entity_ids, batch_features, strict=False):
                result[entity_id] = {
                    k: v
                    for k, v in features.items()
                    if k not in [self._entity_key, "event_timestamp"]
                }

            return result

        except Exception as e:
            logger.warning(f"Failed to get batch online features: {e}")
            return {eid: {} for eid in entity_ids}

    async def check_feature_freshness(
        self,
        entity_id: str,
        max_staleness_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """Check freshness of features for an entity.

        Validates that features are not stale before using for prediction.

        Args:
            entity_id: Entity identifier
            max_staleness_hours: Maximum acceptable age in hours

        Returns:
            Dictionary with freshness status:
                - fresh: bool indicating if features are fresh
                - stale_features: List of stale feature names
                - last_updated: ISO timestamp of last update
        """
        adapter = self._ensure_adapter()
        if adapter is None:
            return {"fresh": True, "stale_features": [], "last_updated": None}

        try:
            freshness = await adapter.check_entity_freshness(
                entity_id=entity_id,
                entity_key=self._entity_key,
                max_staleness_hours=max_staleness_hours,
            )
            return freshness if freshness else {"fresh": True, "stale_features": []}
        except AttributeError:
            # Method not available
            return {"fresh": True, "stale_features": [], "last_updated": None}
        except Exception as e:
            logger.warning(f"Failed to check freshness for {entity_id}: {e}")
            return {"fresh": True, "stale_features": [], "last_updated": None}

    @property
    def is_available(self) -> bool:
        """Check if Feast is available."""
        return self._ensure_adapter() is not None


def get_feast_feature_store(
    default_feature_view: str = "hcp_features",
    entity_key: str = "hcp_id",
) -> FeastFeatureStore:
    """Factory function to get FeastFeatureStore instance.

    Args:
        default_feature_view: Default feature view for lookups
        entity_key: Default entity key column

    Returns:
        Configured FeastFeatureStore instance
    """
    return FeastFeatureStore(
        default_feature_view=default_feature_view,
        entity_key=entity_key,
    )
