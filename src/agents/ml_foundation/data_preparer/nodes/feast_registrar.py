"""Feast registrar node for data_preparer agent.

This node registers validated features in Feast feature store.
It executes AFTER validation and BEFORE baseline computation to ensure:
1. Features are registered for point-in-time retrieval
2. Feature freshness is checked as part of QC
3. Features are available for model_trainer
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..state import DataPreparerState

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


async def register_features_in_feast(state: DataPreparerState) -> Dict[str, Any]:
    """Register validated features in Feast feature store.

    This node:
    1. Registers features from validated training data in Feast
    2. Checks feature freshness for QC gate
    3. Syncs feature metadata to Feast registry

    The node runs AFTER transform_data and BEFORE compute_baseline_metrics.
    Feature registration is non-blocking - failures generate warnings, not errors.

    Args:
        state: Current agent state

    Returns:
        Updated state with Feast registration results
    """
    logger.info(f"Registering features in Feast for experiment {state['experiment_id']}")

    updates: Dict[str, Any] = {
        "feast_registration_status": "skipped",
        "feast_features_registered": 0,
        "feast_freshness_check": None,
        "feast_warnings": [],
        "feast_registered_at": None,
    }

    try:
        # Get adapter
        adapter = _get_feature_analyzer_adapter()
        if adapter is None:
            updates["feast_warnings"].append(
                "Feast adapter not available - skipping feature registration"
            )
            logger.warning("Feast adapter not available")
            return updates

        # Get training data
        train_df = state.get("train_df")
        if train_df is None:
            updates["feast_warnings"].append("No training data available for registration")
            return updates

        # Get scope spec for entity and feature info
        scope_spec = state.get("scope_spec", {})
        experiment_id = state["experiment_id"]
        entity_key = scope_spec.get("entity_key", "hcp_id")
        required_features = scope_spec.get("required_features", [])

        # Build feature metadata from training data
        generated_features = []
        for feature_name in required_features:
            if feature_name in train_df.columns:
                generated_features.append(
                    {
                        "name": feature_name,
                        "type": "prepared",
                        "transformation": "data_preparer",
                        "source": state.get("data_source", "unknown"),
                    }
                )

        # Build state dict for adapter
        adapter_state = {
            "generated_features": generated_features,
            "selected_features": required_features,
            "feature_importance": {},  # Not computed yet
            "X_train_selected": train_df[[f for f in required_features if f in train_df.columns]]
            if required_features
            else train_df,
        }

        # Register features
        registration_result = await adapter.register_features_from_state(
            state=adapter_state,
            experiment_id=experiment_id,
            entity_key=entity_key,
            owner="data_preparer",
            tags=["data_preparer", "validated", f"exp_{experiment_id}"],
        )

        updates["feast_features_registered"] = registration_result.get("features_registered", 0)
        updates["feast_registration_status"] = (
            "completed" if registration_result.get("features_registered", 0) > 0 else "empty"
        )

        if registration_result.get("errors"):
            for error in registration_result["errors"]:
                updates["feast_warnings"].append(
                    f"Registration error: {error.get('error', str(error))}"
                )

        # Check feature freshness (if features are already in Feast)
        freshness_result = await _check_feature_freshness(adapter, experiment_id, required_features)
        updates["feast_freshness_check"] = freshness_result

        # Add freshness warnings to state
        if freshness_result and not freshness_result.get("fresh", True):
            for recommendation in freshness_result.get("recommendations", []):
                updates["feast_warnings"].append(f"Freshness: {recommendation}")

        updates["feast_registered_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Feast registration completed: {updates['feast_features_registered']} features"
        )

        return updates

    except Exception as e:
        logger.error(f"Feast registration failed: {e}", exc_info=True)
        updates["feast_registration_status"] = "error"
        updates["feast_warnings"].append(f"Registration error: {str(e)}")
        return updates


async def _check_feature_freshness(
    adapter: Any,
    experiment_id: str,
    feature_names: list,
    max_staleness_hours: float = 24.0,
) -> Optional[Dict[str, Any]]:
    """Check feature freshness in Feast.

    Args:
        adapter: FeatureAnalyzerAdapter instance
        experiment_id: Experiment identifier
        feature_names: List of feature names to check
        max_staleness_hours: Maximum allowed staleness

    Returns:
        Freshness check result or None if check fails
    """
    try:
        # Build feature refs for the experiment's feature view
        feature_view = f"feature_analyzer_{experiment_id}"
        feature_refs = [f"{feature_view}:{name}" for name in feature_names[:5]]  # Check first 5

        if not feature_refs:
            return None

        result = await adapter.check_feature_freshness(
            feature_refs=feature_refs,
            max_staleness_hours=max_staleness_hours,
        )

        return result

    except Exception as e:
        logger.debug(f"Freshness check failed (non-critical): {e}")
        return None
