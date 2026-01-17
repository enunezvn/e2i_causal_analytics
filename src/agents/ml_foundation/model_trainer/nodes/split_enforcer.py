"""Split enforcement for model_trainer.

This module validates ML split ratios and detects potential data leakage.
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


async def enforce_splits(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ML split ratios and detect data leakage.

    CRITICAL: This enforces the E2I split policy:
    - Train: 60% ± 2%
    - Validation: 20% ± 2%
    - Test: 15% ± 2%
    - Holdout: 5% ± 2%

    This also checks for potential data leakage by ensuring splits are properly
    isolated.

    Args:
        state: ModelTrainerState with split data and ratios

    Returns:
        Dictionary with split_ratios_valid, split_validation_message,
        leakage_warnings

    Raises:
        No exceptions - returns error in state if validation fails
    """
    # Extract split ratios
    train_ratio = state.get("train_ratio", 0.0)
    validation_ratio = state.get("validation_ratio", 0.0)
    test_ratio = state.get("test_ratio", 0.0)
    holdout_ratio = state.get("holdout_ratio", 0.0)

    # Extract sample counts
    train_samples = state.get("train_samples", 0)
    validation_samples = state.get("validation_samples", 0)
    test_samples = state.get("test_samples", 0)
    holdout_samples = state.get("holdout_samples", 0)
    total_samples = state.get("total_samples", 0)

    # Expected ratios (E2I ML Foundation split policy)
    expected_ratios = {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.15,
        "holdout": 0.05,
    }
    tolerance = 0.02  # ±2%
    # Add small epsilon for floating point comparison (0.62 - 0.60 may be 0.0200000001)
    epsilon = 1e-9

    # Validate each split ratio
    ratio_checks = []
    ratios_valid = True

    # Train split
    if abs(train_ratio - expected_ratios["train"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Train split ratio {train_ratio:.2%} outside expected "
            f"{expected_ratios['train']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Train split: {train_ratio:.2%} ✓")

    # Validation split
    if abs(validation_ratio - expected_ratios["validation"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Validation split ratio {validation_ratio:.2%} outside expected "
            f"{expected_ratios['validation']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Validation split: {validation_ratio:.2%} ✓")

    # Test split
    if abs(test_ratio - expected_ratios["test"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Test split ratio {test_ratio:.2%} outside expected "
            f"{expected_ratios['test']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Test split: {test_ratio:.2%} ✓")

    # Holdout split
    if abs(holdout_ratio - expected_ratios["holdout"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Holdout split ratio {holdout_ratio:.2%} outside expected "
            f"{expected_ratios['holdout']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Holdout split: {holdout_ratio:.2%} ✓")

    # Check for minimum sample sizes
    min_samples_per_split = 10  # Minimum viable samples
    leakage_warnings = []

    if train_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Train split has only {train_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    if validation_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Validation split has only {validation_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    if test_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Test split has only {test_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    # Check holdout - warn if zero (holdout is optional but important for final validation)
    if holdout_samples == 0 and holdout_ratio > 0:
        leakage_warnings.append(f"Holdout split has 0 samples but ratio is {holdout_ratio:.2%}")
        ratios_valid = False
    elif holdout_samples == 0:
        leakage_warnings.append(
            "Holdout split has 0 samples - final model validation will be limited"
        )
        ratios_valid = False

    # Check if total adds up (should be close to 1.0)
    total_ratio = train_ratio + validation_ratio + test_ratio + holdout_ratio
    if abs(total_ratio - 1.0) > 0.01:
        leakage_warnings.append(
            f"Split ratios sum to {total_ratio:.4f}, expected 1.0 (possible overlap or missing data)"
        )
        ratios_valid = False

    # Advanced leakage detection (when data is available)
    train_data = state.get("train_data")
    validation_data = state.get("validation_data")
    test_data = state.get("test_data")
    holdout_data = state.get("holdout_data")

    if train_data and validation_data and test_data:
        advanced_warnings = _check_advanced_leakage(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            holdout_data=holdout_data,
            target_column=state.get("target_column"),
            time_column=state.get("time_column"),
        )
        leakage_warnings.extend(advanced_warnings)

        if advanced_warnings:
            logger.warning(f"Advanced leakage checks found {len(advanced_warnings)} issues")

    # Construct validation message
    if ratios_valid:
        split_validation_message = (
            f"Split ratios validated: "
            f"{train_ratio:.1%}/{validation_ratio:.1%}/{test_ratio:.1%}/{holdout_ratio:.1%} "
            f"({total_samples:,} total samples)"
        )
    else:
        split_validation_message = (
            f"Split ratio validation FAILED: "
            f"{', '.join(ratio_checks[:2])}. "
            f"Warnings: {len(leakage_warnings)}"
        )

    result = {
        "split_ratios_valid": ratios_valid,
        "split_validation_message": split_validation_message,
        "split_ratio_checks": ratio_checks,
        "leakage_warnings": leakage_warnings if leakage_warnings else [],
    }

    # Set error when validation fails so agent can detect it
    if not ratios_valid:
        failed_checks = [c for c in ratio_checks if "✓" not in c]
        result["error"] = (
            f"Split validation failed: {split_validation_message}. "
            f"Failed checks: {', '.join(failed_checks[:3])}"
        )
        result["error_type"] = "split_validation_error"

    return result


def _check_advanced_leakage(
    train_data: Dict[str, Any],
    validation_data: Dict[str, Any],
    test_data: Dict[str, Any],
    holdout_data: Optional[Dict[str, Any]],
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> List[str]:
    """Perform advanced leakage detection on split data.

    Checks for:
    1. Duplicate row indices across splits (data leakage)
    2. Feature leakage (target variable in features)
    3. Temporal ordering violations (future data in training)

    Args:
        train_data: Training split dict with 'X', 'y', optional 'indices'
        validation_data: Validation split dict
        test_data: Test split dict
        holdout_data: Holdout split dict (optional)
        target_column: Target column name for feature leakage check
        time_column: Time column name for temporal ordering check

    Returns:
        List of warning messages for detected issues
    """
    warnings = []

    try:
        # Extract training data for feature leakage check
        X_train = train_data.get("X")

        # 1. Check for duplicate row indices across splits
        duplicate_warnings = _check_duplicate_indices(
            train_data, validation_data, test_data, holdout_data
        )
        warnings.extend(duplicate_warnings)

        # 2. Check for feature leakage (target in features)
        if target_column and X_train is not None:
            feature_leakage = _check_feature_leakage(X_train, target_column)
            if feature_leakage:
                warnings.append(feature_leakage)

        # 3. Check temporal ordering (train should be before validation/test)
        if time_column:
            temporal_warnings = _check_temporal_ordering(
                train_data, validation_data, test_data, time_column
            )
            warnings.extend(temporal_warnings)

    except Exception as e:
        logger.warning(f"Advanced leakage detection error: {e}")
        # Don't fail the pipeline for leakage detection errors

    return warnings


def _check_duplicate_indices(
    train_data: Dict[str, Any],
    validation_data: Dict[str, Any],
    test_data: Dict[str, Any],
    holdout_data: Optional[Dict[str, Any]],
) -> List[str]:
    """Check for duplicate row indices across splits."""
    warnings = []

    # Get indices from each split
    train_indices = _get_indices(train_data)
    val_indices = _get_indices(validation_data)
    test_indices = _get_indices(test_data)
    holdout_indices = _get_indices(holdout_data) if holdout_data else set()

    if not train_indices:
        return warnings  # No indices available to check

    # Check for overlaps
    train_val_overlap = train_indices & val_indices
    if train_val_overlap:
        warnings.append(
            f"CRITICAL: {len(train_val_overlap)} duplicate indices between train and validation "
            f"(data leakage detected)"
        )

    train_test_overlap = train_indices & test_indices
    if train_test_overlap:
        warnings.append(
            f"CRITICAL: {len(train_test_overlap)} duplicate indices between train and test "
            f"(data leakage detected)"
        )

    val_test_overlap = val_indices & test_indices
    if val_test_overlap:
        warnings.append(
            f"CRITICAL: {len(val_test_overlap)} duplicate indices between validation and test "
            f"(data leakage detected)"
        )

    if holdout_indices:
        train_holdout_overlap = train_indices & holdout_indices
        if train_holdout_overlap:
            warnings.append(
                f"CRITICAL: {len(train_holdout_overlap)} duplicate indices between train and holdout"
            )

    return warnings


def _get_indices(split_data: Dict[str, Any]) -> Set:
    """Extract row indices from split data."""
    if not split_data:
        return set()

    # Check for explicit indices
    indices = split_data.get("indices")
    if indices is not None:
        return set(indices)

    # Try to extract from DataFrame index
    X = split_data.get("X")
    if X is not None:
        if isinstance(X, pd.DataFrame):
            return set(X.index.tolist())
        elif isinstance(X, np.ndarray):
            # Use row numbers as indices
            return set(range(len(X)))

    return set()


def _check_feature_leakage(X: Any, target_column: str) -> Optional[str]:
    """Check if target column appears in features."""
    if X is None:
        return None

    # Get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    elif hasattr(X, "feature_names_in_"):
        feature_names = list(X.feature_names_in_)
    else:
        return None

    # Check for target in features
    target_lower = target_column.lower()
    for feature in feature_names:
        feature_lower = str(feature).lower()
        if target_lower == feature_lower:
            return f"CRITICAL: Target column '{target_column}' found in features (direct leakage)"
        if target_lower in feature_lower and feature_lower != target_lower:
            return (
                f"WARNING: Feature '{feature}' may leak target '{target_column}' "
                f"(similar name detected)"
            )

    return None


def _check_temporal_ordering(
    train_data: Dict[str, Any],
    validation_data: Dict[str, Any],
    test_data: Dict[str, Any],
    time_column: str,
) -> List[str]:
    """Check that training data is temporally before validation/test."""
    warnings = []

    try:
        train_max_time = _get_max_time(train_data, time_column)
        val_min_time = _get_min_time(validation_data, time_column)
        test_min_time = _get_min_time(test_data, time_column)

        if train_max_time is None or val_min_time is None:
            return warnings

        # Check temporal ordering
        if train_max_time > val_min_time:
            warnings.append(
                f"WARNING: Training data max time ({train_max_time}) > "
                f"validation min time ({val_min_time}). "
                f"Possible temporal leakage for time-series data."
            )

        if train_max_time is not None and test_min_time is not None:
            if train_max_time > test_min_time:
                warnings.append(
                    f"WARNING: Training data max time ({train_max_time}) > "
                    f"test min time ({test_min_time}). "
                    f"Possible temporal leakage for time-series data."
                )

    except Exception as e:
        logger.debug(f"Temporal ordering check skipped: {e}")

    return warnings


def _get_max_time(split_data: Dict[str, Any], time_column: str) -> Any:
    """Get maximum time value from split data."""
    X = split_data.get("X")
    if isinstance(X, pd.DataFrame) and time_column in X.columns:
        return X[time_column].max()
    return None


def _get_min_time(split_data: Dict[str, Any], time_column: str) -> Any:
    """Get minimum time value from split data."""
    X = split_data.get("X")
    if isinstance(X, pd.DataFrame) and time_column in X.columns:
        return X[time_column].min()
    return None
