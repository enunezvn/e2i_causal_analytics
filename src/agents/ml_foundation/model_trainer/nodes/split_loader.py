"""Split loading for model_trainer.

This module loads and validates ML data splits (train/val/test/holdout).
Supports point-in-time correct feature retrieval from Feast feature store.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

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


def _get_ml_data_loader():
    """Get MLDataLoader (lazy import to avoid circular deps)."""
    try:
        from src.ml.data_loader import MLDataLoader

        return MLDataLoader()
    except Exception as e:
        logger.warning(f"Could not initialize MLDataLoader: {e}")
        return None


async def _fetch_splits_from_feast(
    experiment_id: str,
    feature_refs: Optional[List[str]] = None,
    entity_key: str = "hcp_id",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch point-in-time correct splits from Feast feature store.

    This retrieves training features with proper temporal joins to prevent
    data leakage. Features must have been registered by data_preparer.

    Args:
        experiment_id: Experiment identifier for feature lookup
        feature_refs: Optional list of feature references. If None, uses
            default features for the experiment's feature view.
        entity_key: Entity column name (default: hcp_id)

    Returns:
        Dictionary with train/validation/test/holdout split data, or None
        if Feast retrieval fails.
    """
    adapter = _get_feature_analyzer_adapter()
    if adapter is None:
        logger.warning("FeatureAnalyzerAdapter not available for split loading")
        return None

    ml_loader = _get_ml_data_loader()
    if ml_loader is None:
        logger.warning("MLDataLoader not available for entity retrieval")
        return None

    try:
        # Get entity data with timestamps for each split
        # This loads the split assignments and event timestamps from the database
        split_metadata = await ml_loader.get_split_metadata(experiment_id)
        if not split_metadata:
            logger.warning(f"No split metadata found for experiment {experiment_id}")
            return None

        # Build feature references from experiment's registered features
        if not feature_refs:
            # Use default feature view based on experiment
            feature_view = f"feature_analyzer_{experiment_id}"
            feature_refs = [f"{feature_view}:*"]  # All features from the view

        splits = {}
        for split_name in ["train", "validation", "test", "holdout"]:
            split_entities = split_metadata.get(split_name)
            if split_entities is None or len(split_entities) == 0:
                logger.warning(f"No entities for {split_name} split")
                splits[split_name] = {
                    "X": pd.DataFrame(),
                    "y": pd.Series(dtype=float),
                    "row_count": 0,
                }
                continue

            # Build entity DataFrame with timestamps for point-in-time joins
            entity_df = pd.DataFrame(
                {
                    entity_key: split_entities.get("entity_ids", []),
                    "event_timestamp": pd.to_datetime(split_entities.get("event_timestamps", [])),
                }
            )

            # Add brand_id if available (multi-entity join)
            if "brand_ids" in split_entities:
                entity_df["brand_id"] = split_entities["brand_ids"]

            # Retrieve point-in-time correct features from Feast
            features_df = await adapter.get_training_features(
                entity_df=entity_df,
                feature_refs=feature_refs,
                full_feature_names=True,
            )

            # Separate target from features
            target_col = split_entities.get("target_column", "target")
            if target_col in features_df.columns:
                y = features_df[target_col]
                X = features_df.drop(
                    columns=[target_col, entity_key, "event_timestamp"], errors="ignore"
                )
            else:
                # Target not in Feast - get from split metadata
                y = pd.Series(split_entities.get("targets", []))
                X = features_df.drop(columns=[entity_key, "event_timestamp"], errors="ignore")

            splits[split_name] = {
                "X": X,
                "y": y,
                "row_count": len(X),
                "entity_ids": split_entities.get("entity_ids", []),
                "feast_retrieved": True,
                "feature_refs": feature_refs,
            }

            logger.info(
                f"Loaded {split_name} split from Feast: {len(X)} samples, {len(X.columns)} features"
            )

        return {
            "train_data": splits["train"],
            "validation_data": splits["validation"],
            "test_data": splits["test"],
            "holdout_data": splits["holdout"],
            "feast_source": True,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Feast split loading failed: {e}", exc_info=True)
        return None


async def load_splits(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load train/validation/test/holdout data splits.

    CRITICAL: This loads the enforced ML splits from data_preparer.
    These splits MUST be respected throughout training to prevent data leakage.

    Expected split ratios:
    - Train: 60%
    - Validation: 20%
    - Test: 15%
    - Holdout: 5% (LOCKED until post-deployment)

    Args:
        state: ModelTrainerState with experiment_id or direct split data

    Returns:
        Dictionary with train_data, validation_data, test_data, holdout_data,
        and sample counts

    Raises:
        No exceptions - returns error in state if splits unavailable
    """
    # Extract experiment_id for fetching splits
    experiment_id = state.get("experiment_id")

    # Check if splits are already in state (direct pass from data_preparer)
    if all(key in state for key in ["train_data", "validation_data", "test_data", "holdout_data"]):
        # Splits already loaded, validate and calculate sizes
        train_data = state["train_data"]
        validation_data = state["validation_data"]
        test_data = state["test_data"]
        holdout_data = state["holdout_data"]

    elif experiment_id:
        # Fetch splits from Feast feature store with point-in-time correct joins
        logger.info(f"Fetching splits from Feast for experiment {experiment_id}")

        # Get feature refs from state if provided
        feature_refs = state.get("feature_refs")
        entity_key = state.get("entity_key", "hcp_id")

        feast_splits = await _fetch_splits_from_feast(
            experiment_id=experiment_id,
            feature_refs=feature_refs,
            entity_key=entity_key,
        )

        if feast_splits is None:
            # Feast unavailable - try fallback to database
            logger.warning(
                f"Feast split loading failed for {experiment_id}, attempting database fallback"
            )
            ml_loader = _get_ml_data_loader()
            if ml_loader is not None:
                try:
                    db_splits = await ml_loader.load_experiment_splits(experiment_id)
                    if db_splits:
                        train_data = db_splits.get("train")
                        validation_data = db_splits.get("validation")
                        test_data = db_splits.get("test")
                        holdout_data = db_splits.get("holdout")
                        logger.info(f"Loaded splits from database for {experiment_id}")
                    else:
                        return {
                            "error": f"No splits found in database for experiment {experiment_id}",
                            "error_type": "split_not_found_error",
                        }
                except Exception as e:
                    return {
                        "error": f"Failed to load splits from database: {e}",
                        "error_type": "split_load_error",
                    }
            else:
                return {
                    "error": f"Cannot load splits: Feast and database both unavailable. "
                    f"Experiment ID: {experiment_id}",
                    "error_type": "split_loader_unavailable",
                }
        else:
            # Use Feast splits
            train_data = feast_splits["train_data"]
            validation_data = feast_splits["validation_data"]
            test_data = feast_splits["test_data"]
            holdout_data = feast_splits["holdout_data"]
            logger.info(
                f"Loaded splits from Feast for {experiment_id} "
                f"(feast_source={feast_splits.get('feast_source', False)})"
            )

    else:
        # No splits in state and no experiment_id
        return {
            "error": "Cannot load splits: no splits in state and no experiment_id provided",
            "error_type": "missing_splits_error",
        }

    # Validate splits have required fields
    required_keys = ["X", "y", "row_count"]
    for split_name, split_data in [
        ("train_data", train_data),
        ("validation_data", validation_data),
        ("test_data", test_data),
        ("holdout_data", holdout_data),
    ]:
        if not isinstance(split_data, dict):
            return {
                "error": f"{split_name} is not a dictionary",
                "error_type": "invalid_split_format",
            }
        for key in required_keys:
            if key not in split_data:
                return {
                    "error": f"{split_name} missing required key: {key}",
                    "error_type": "invalid_split_format",
                }

    # Extract sample counts
    train_samples = train_data["row_count"]
    validation_samples = validation_data["row_count"]
    test_samples = test_data["row_count"]
    holdout_samples = holdout_data["row_count"]
    total_samples = train_samples + validation_samples + test_samples + holdout_samples

    # Validate all splits have samples
    if total_samples == 0:
        return {
            "error": "All splits are empty (0 samples)",
            "error_type": "empty_splits_error",
        }

    # Calculate actual ratios
    train_ratio = train_samples / total_samples
    validation_ratio = validation_samples / total_samples
    test_ratio = test_samples / total_samples
    holdout_ratio = holdout_samples / total_samples

    return {
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "holdout_data": holdout_data,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "holdout_samples": holdout_samples,
        "total_samples": total_samples,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "holdout_ratio": holdout_ratio,
    }
