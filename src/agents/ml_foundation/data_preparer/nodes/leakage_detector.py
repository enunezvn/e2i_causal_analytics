"""Leakage detector node for data_preparer agent.

This node detects three types of data leakage:
1. Temporal leakage: Future data leaking into training
2. Target leakage: Features that are derived from the target
3. Train-test contamination: Overlapping samples between splits
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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

    Temporal leakage occurs when event timestamps are after target timestamps,
    or when features contain information from the future relative to prediction time.

    Detection strategies:
    1. Explicit: event_date_column vs target_date_column comparison
    2. Split-based: feature dates vs split_date (prediction boundary)
    3. Generic: auto-detect date columns and check for future data

    Args:
        df: DataFrame to check
        scope_spec: Scope specification with temporal column hints

    Returns:
        List of temporal leakage issues
    """
    issues = []

    if df is None or len(df) == 0:
        return issues

    try:
        # Strategy 1: Explicit event_date vs target_date comparison
        event_date_col = scope_spec.get("event_date_column")
        target_date_col = scope_spec.get("target_date_column")

        if event_date_col and target_date_col:
            if event_date_col in df.columns and target_date_col in df.columns:
                leakage_count, leakage_pct = _check_date_ordering(
                    df, event_date_col, target_date_col
                )
                if leakage_count > 0:
                    issues.append(
                        f"Temporal leakage: {leakage_count} rows ({leakage_pct:.2f}%) "
                        f"have {event_date_col} > {target_date_col}"
                    )

        # Strategy 2: Check feature date columns against split_date
        split_date_str = scope_spec.get("split_date")
        feature_date_columns = scope_spec.get("feature_date_columns", [])

        if split_date_str and feature_date_columns:
            split_date = _parse_date(split_date_str)
            if split_date:
                for col in feature_date_columns:
                    if col in df.columns:
                        future_count, future_pct = _check_future_dates(
                            df, col, split_date
                        )
                        if future_count > 0:
                            issues.append(
                                f"Temporal leakage: {future_count} rows ({future_pct:.2f}%) "
                                f"in '{col}' have dates after split_date ({split_date_str})"
                            )

        # Strategy 3: Generic auto-detection of date columns
        date_column = scope_spec.get("date_column")
        if split_date_str and date_column:
            split_date = _parse_date(split_date_str)
            if split_date:
                # Find all date-like columns (excluding the main date column)
                date_cols = _detect_date_columns(df, exclude=[date_column])
                for col in date_cols:
                    future_count, future_pct = _check_future_dates(df, col, split_date)
                    if future_count > 0:
                        issues.append(
                            f"Potential temporal leakage: {future_count} rows ({future_pct:.2f}%) "
                            f"in auto-detected date column '{col}' have dates after split_date"
                        )

    except Exception as e:
        logger.warning(f"Temporal leakage check failed: {e}")
        issues.append(f"Temporal leakage check incomplete: {str(e)}")

    return issues


def _check_date_ordering(df: Any, event_col: str, target_col: str) -> tuple:
    """Check if event dates occur after target dates.

    Args:
        df: DataFrame
        event_col: Column with event dates
        target_col: Column with target dates

    Returns:
        Tuple of (leakage_count, leakage_percentage)
    """
    try:
        event_dates = pd.to_datetime(df[event_col], errors="coerce")
        target_dates = pd.to_datetime(df[target_col], errors="coerce")

        # Count rows where event > target (future leakage)
        valid_mask = event_dates.notna() & target_dates.notna()
        leakage_mask = valid_mask & (event_dates > target_dates)

        leakage_count = leakage_mask.sum()
        leakage_pct = (leakage_count / len(df)) * 100 if len(df) > 0 else 0

        return leakage_count, leakage_pct
    except Exception:
        return 0, 0.0


def _check_future_dates(df: Any, col: str, reference_date: datetime) -> tuple:
    """Check for dates after a reference date.

    Args:
        df: DataFrame
        col: Date column to check
        reference_date: Reference date (typically split_date)

    Returns:
        Tuple of (future_count, future_percentage)
    """
    try:
        dates = pd.to_datetime(df[col], errors="coerce")
        valid_mask = dates.notna()

        # Make reference_date timezone-naive for comparison
        ref_date = pd.Timestamp(reference_date).tz_localize(None)
        dates_naive = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates

        future_mask = valid_mask & (dates_naive > ref_date)
        future_count = future_mask.sum()
        future_pct = (future_count / len(df)) * 100 if len(df) > 0 else 0

        return future_count, future_pct
    except Exception:
        return 0, 0.0


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime.

    Args:
        date_str: Date string in various formats

    Returns:
        datetime object or None if parsing fails
    """
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except Exception:
        return None


def _detect_date_columns(df: Any, exclude: List[str] = None) -> List[str]:
    """Auto-detect date columns in DataFrame.

    Args:
        df: DataFrame
        exclude: Columns to exclude from detection

    Returns:
        List of detected date column names
    """
    exclude = exclude or []
    date_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        # Check if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        # Check column name patterns
        date_patterns = ["_date", "_time", "_at", "_timestamp", "date_", "time_"]
        if any(pattern in col.lower() for pattern in date_patterns):
            # Try to parse as date
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() > len(sample) * 0.5:
                        date_cols.append(col)
            except Exception:
                pass

    return date_cols


def check_target_leakage(df: Any, target_variable: str, features: List[str]) -> List[str]:
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
