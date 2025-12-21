"""Split loading for model_trainer.

This module loads and validates ML data splits (train/val/test/holdout).
"""

from typing import Any, Dict


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
        # TODO: Fetch splits from Feast feature store or database
        # For now, return error - this requires Feast integration
        return {
            "error": f"Split fetching from Feast not yet implemented. "
            f"Experiment ID: {experiment_id}",
            "error_type": "split_fetch_not_implemented",
        }

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
