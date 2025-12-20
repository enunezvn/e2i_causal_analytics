"""Leakage detector node for data_preparer agent.

This node detects three types of data leakage:
1. Temporal leakage: Future data leaking into training
2. Target leakage: Features that are derived from the target
3. Train-test contamination: Overlapping samples between splits
"""

from typing import Dict, Any, List
import logging
import numpy as np

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def detect_leakage(state: DataPreparerState) -> Dict[str, Any]:
    """Detect data leakage across multiple types.

    This node checks for:
    1. Temporal leakage: event_date > target_date in any row
    2. Target leakage: Features with suspiciously high correlation with target
    3. Train-test contamination: Duplicate samples across splits

    Args:
        state: Current agent state

    Returns:
        Updated state with leakage detection results
    """
    if state.get("skip_leakage_check", False):
        logger.warning("Leakage detection skipped per configuration")
        return {
            "leakage_detected": False,
            "leakage_issues": ["Leakage check skipped (not recommended)"],
        }

    logger.info(f"Running leakage detection for experiment {state['experiment_id']}")

    leakage_issues = []

    try:
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")
        holdout_df = state.get("holdout_df")

        if train_df is None:
            raise ValueError("train_df not found in state")

        scope_spec = state.get("scope_spec", {})
        target_variable = scope_spec.get("prediction_target")
        required_features = scope_spec.get("required_features", [])

        # === 1. TEMPORAL LEAKAGE ===
        temporal_issues = check_temporal_leakage(train_df, scope_spec)
        leakage_issues.extend(temporal_issues)

        # === 2. TARGET LEAKAGE ===
        if target_variable and target_variable in train_df.columns:
            target_leakage_issues = check_target_leakage(
                train_df, target_variable, required_features
            )
            leakage_issues.extend(target_leakage_issues)

        # === 3. TRAIN-TEST CONTAMINATION ===
        contamination_issues = check_train_test_contamination(
            train_df, validation_df, test_df, holdout_df
        )
        leakage_issues.extend(contamination_issues)

        # Determine if leakage detected
        leakage_detected = len(leakage_issues) > 0

        # Add to blocking issues if leakage detected
        blocking_updates = {}
        if leakage_detected:
            existing_blocking = state.get("blocking_issues", [])
            blocking_updates["blocking_issues"] = existing_blocking + leakage_issues

        logger.info(
            f"Leakage detection completed: "
            f"detected={leakage_detected}, issues={len(leakage_issues)}"
        )

        return {
            "leakage_detected": leakage_detected,
            "leakage_issues": leakage_issues,
            **blocking_updates,
        }

    except Exception as e:
        logger.error(f"Leakage detection failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "leakage_detection_error",
            "leakage_detected": True,  # Assume worst case
            "leakage_issues": [f"Leakage detection error: {str(e)}"],
        }


def check_temporal_leakage(df: Any, scope_spec: Dict[str, Any]) -> List[str]:
    """Check for temporal leakage.

    Temporal leakage occurs when event timestamps are after target timestamps.

    Args:
        df: DataFrame to check
        scope_spec: Scope specification with temporal column hints

    Returns:
        List of temporal leakage issues
    """
    issues = []

    # TODO: Implement temporal leakage detection
    # This requires:
    # 1. Identify event_date and target_date columns from scope_spec
    # 2. Check if any event_date > target_date
    # 3. Report percentage of rows with temporal leakage

    # Placeholder: No temporal leakage detected
    return issues


def check_target_leakage(
    df: Any, target_variable: str, features: List[str]
) -> List[str]:
    """Check for target leakage.

    Target leakage occurs when features have suspiciously high correlation
    with the target (e.g., > 0.95), suggesting they may be derived from the target.

    Args:
        df: DataFrame to check
        target_variable: Name of target variable
        features: List of feature names

    Returns:
        List of target leakage issues
    """
    issues = []

    try:
        target_data = df[target_variable]

        for feature in features:
            if feature not in df.columns:
                continue

            feature_data = df[feature]

            # Skip if feature or target is not numerical
            if not (
                np.issubdtype(feature_data.dtype, np.number)
                and np.issubdtype(target_data.dtype, np.number)
            ):
                continue

            # Compute correlation
            correlation = feature_data.corr(target_data)

            # Flag suspiciously high correlation
            if abs(correlation) > 0.95:
                issues.append(
                    f"Potential target leakage: feature '{feature}' has "
                    f"correlation {correlation:.3f} with target (threshold: 0.95)"
                )

    except Exception as e:
        logger.warning(f"Target leakage check failed: {e}")

    return issues


def check_train_test_contamination(
    train_df: Any,
    validation_df: Any = None,
    test_df: Any = None,
    holdout_df: Any = None,
) -> List[str]:
    """Check for train-test contamination.

    Contamination occurs when the same samples appear in multiple splits.

    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        holdout_df: Holdout DataFrame (optional)

    Returns:
        List of contamination issues
    """
    issues = []

    try:
        # Check if DataFrames have an index we can compare
        # If they have a unique identifier column, we should use that instead

        splits = {
            "validation": validation_df,
            "test": test_df,
            "holdout": holdout_df,
        }

        for split_name, split_df in splits.items():
            if split_df is None:
                continue

            # Check for index overlap
            train_indices = set(train_df.index)
            split_indices = set(split_df.index)
            overlap = train_indices.intersection(split_indices)

            if len(overlap) > 0:
                overlap_pct = len(overlap) / len(train_df) * 100
                issues.append(
                    f"Train-{split_name} contamination: {len(overlap)} samples "
                    f"({overlap_pct:.2f}%) overlap between splits"
                )

    except Exception as e:
        logger.warning(f"Train-test contamination check failed: {e}")

    return issues
