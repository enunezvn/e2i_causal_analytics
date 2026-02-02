"""Quality checker node for data_preparer agent.

This node runs data quality validation and generates a QC report.
In Phase 3, this will integrate with Great Expectations.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def run_quality_checks(state: DataPreparerState) -> Dict[str, Any]:
    """Run data quality validation and compute QC scores.

    This node:
    1. Validates completeness (missing values)
    2. Validates data types and value ranges
    3. Checks consistency across columns
    4. Checks for duplicates
    5. Validates data freshness
    6. Computes overall QC score
    7. Identifies blocking issues

    Args:
        state: Current agent state

    Returns:
        Updated state with QC results
    """
    start_time = datetime.now()
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Starting quality checks for experiment {experiment_id}")

    try:
        # Generate report ID
        report_id = f"qc_{experiment_id}_{uuid.uuid4().hex[:8]}"

        train_df = state.get("train_df")
        if train_df is None:
            raise ValueError("train_df not found in state")

        validation_df = state.get("validation_df")
        test_df = state.get("test_df")

        # Get scope spec for validation configuration
        scope_spec = state.get("scope_spec", {})
        date_column = scope_spec.get("date_column", "created_at")
        required_columns = scope_spec.get("required_columns", [])
        expected_dtypes = scope_spec.get("expected_dtypes", {})
        unique_columns = scope_spec.get("unique_columns", [])
        max_staleness_days = scope_spec.get("max_staleness_days", 30)

        # Initialize results
        expectation_results = []
        failed_expectations = []
        warnings = []
        remediation_steps = []
        blocking_issues = []

        # === COMPLETENESS CHECKS ===
        completeness_score, completeness_results = _check_completeness(train_df, required_columns)
        expectation_results.extend(completeness_results)
        for result in completeness_results:
            if not result["success"]:
                if result.get("severity") == "blocking":
                    failed_expectations.append(result)
                    blocking_issues.append(
                        f"Critical missing values in column: {result.get('column')}"
                    )
                else:
                    warnings.append(result)

        # === VALIDITY CHECKS ===
        validity_score, validity_results = _check_validity(train_df, expected_dtypes)
        expectation_results.extend(validity_results)
        for result in validity_results:
            if not result["success"]:
                warnings.append(result)
                remediation_steps.append(
                    f"Fix data type for column {result.get('column')}: "
                    f"expected {result.get('expected_dtype')}, "
                    f"got {result.get('actual_dtype')}"
                )

        # === CONSISTENCY CHECKS ===
        consistency_score, consistency_results = _check_consistency(
            train_df, validation_df, test_df
        )
        expectation_results.extend(consistency_results)
        for result in consistency_results:
            if not result["success"]:
                warnings.append(result)

        # === UNIQUENESS CHECKS ===
        uniqueness_score, uniqueness_results = _check_uniqueness(train_df, unique_columns)
        expectation_results.extend(uniqueness_results)
        for result in uniqueness_results:
            if not result["success"]:
                warnings.append(result)
                remediation_steps.append(
                    f"Remove {result.get('duplicate_count')} duplicates in column: {result.get('column')}"
                )

        # === TIMELINESS CHECKS ===
        timeliness_score, timeliness_results = _check_timeliness(
            train_df, date_column, max_staleness_days
        )
        expectation_results.extend(timeliness_results)
        for result in timeliness_results:
            if not result["success"]:
                warnings.append(result)
                remediation_steps.append(
                    f"Data is {result.get('staleness_days')} days stale, consider refreshing"
                )

        # === OVERALL SCORE ===
        overall_score = (
            completeness_score * 0.25
            + validity_score * 0.25
            + consistency_score * 0.20
            + uniqueness_score * 0.15
            + timeliness_score * 0.15
        )

        # Check for blocking threshold
        if overall_score < 0.80:
            blocking_issues.append(
                f"Overall QC score ({overall_score:.2f}) below minimum threshold (0.80)"
            )
            remediation_steps.append("Review data quality and address failing expectations")

        # Determine QC status
        if blocking_issues:
            qc_status = "failed"
        elif warnings:
            qc_status = "warning"
        else:
            qc_status = "passed"

        # Calculate stats
        row_count = len(train_df)
        column_count = len(train_df.columns)
        validation_duration_seconds = (datetime.now() - start_time).total_seconds()

        # Update state
        updates = {
            "report_id": report_id,
            "qc_status": qc_status,
            "overall_score": overall_score,
            "completeness_score": completeness_score,
            "validity_score": validity_score,
            "consistency_score": consistency_score,
            "uniqueness_score": uniqueness_score,
            "timeliness_score": timeliness_score,
            "expectation_results": expectation_results,
            "failed_expectations": failed_expectations,
            "warnings": warnings,
            "remediation_steps": remediation_steps,
            "blocking_issues": blocking_issues,
            "row_count": row_count,
            "column_count": column_count,
            "validated_at": datetime.now().isoformat(),
            "validation_duration_seconds": validation_duration_seconds,
        }

        logger.info(
            f"Quality checks completed: status={qc_status}, "
            f"score={overall_score:.2f}, duration={validation_duration_seconds:.2f}s"
        )

        return updates

    except Exception as e:
        logger.error(f"Quality check failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "quality_check_error",
            "qc_status": "failed",
            "blocking_issues": [f"Quality check error: {str(e)}"],
        }


def _check_completeness(
    df: pd.DataFrame, required_columns: List[str]
) -> tuple[float, List[Dict[str, Any]]]:
    """Check for missing/null values.

    Args:
        df: DataFrame to check
        required_columns: Columns that must have no nulls

    Returns:
        Tuple of (score, expectation_results)
    """
    results = []

    # Overall null percentage
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    overall_completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0

    results.append(
        {
            "expectation_type": "expect_table_completeness",
            "success": overall_completeness >= 0.90,
            "result": {
                "total_cells": total_cells,
                "null_cells": int(null_cells),
                "completeness_ratio": overall_completeness,
            },
        }
    )

    # Check required columns have no nulls
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            col_completeness = 1 - (null_count / len(df)) if len(df) > 0 else 0

            success = null_count == 0
            results.append(
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "column": col,
                    "success": success,
                    "severity": "blocking" if not success else "info",
                    "result": {
                        "element_count": len(df),
                        "null_count": int(null_count),
                        "completeness_ratio": col_completeness,
                    },
                }
            )

    # Per-column null percentages for non-required columns
    for col in df.columns:
        if col not in required_columns:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.1:  # More than 10% nulls
                results.append(
                    {
                        "expectation_type": "expect_column_null_percentage",
                        "column": col,
                        "success": False,
                        "severity": "warning",
                        "result": {
                            "null_percentage": null_pct,
                            "threshold": 0.10,
                        },
                    }
                )

    return overall_completeness, results


def _check_validity(
    df: pd.DataFrame, expected_dtypes: Dict[str, str]
) -> tuple[float, List[Dict[str, Any]]]:
    """Check data types and value ranges.

    Args:
        df: DataFrame to check
        expected_dtypes: Expected data types for columns

    Returns:
        Tuple of (score, expectation_results)
    """
    results = []
    valid_columns = 0
    total_checks = 0

    # Check expected data types
    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            total_checks += 1
            actual_dtype = str(df[col].dtype)

            # Flexible type matching
            type_matches = _dtype_matches(actual_dtype, expected_dtype)

            if type_matches:
                valid_columns += 1

            results.append(
                {
                    "expectation_type": "expect_column_values_to_be_of_type",
                    "column": col,
                    "success": type_matches,
                    "expected_dtype": expected_dtype,
                    "actual_dtype": actual_dtype,
                    "result": {"type_matches": type_matches},
                }
            )

    # Check for numeric columns with obvious issues
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        total_checks += 1
        # Check for infinity values
        inf_count = np.isinf(df[col]).sum() if np.issubdtype(df[col].dtype, np.floating) else 0

        if inf_count == 0:
            valid_columns += 1
        else:
            results.append(
                {
                    "expectation_type": "expect_column_values_to_be_finite",
                    "column": col,
                    "success": False,
                    "result": {"infinity_count": int(inf_count)},
                }
            )

    validity_score = valid_columns / total_checks if total_checks > 0 else 1.0
    return validity_score, results


def _dtype_matches(actual: str, expected: str) -> bool:
    """Check if actual dtype matches expected dtype.

    Args:
        actual: Actual dtype string
        expected: Expected dtype string

    Returns:
        True if types are compatible
    """
    # Normalize type strings
    actual = actual.lower()
    expected = expected.lower()

    # Direct match
    if actual == expected:
        return True

    # Numeric type matches
    numeric_types = {"int64", "int32", "float64", "float32", "int", "float", "number"}
    if expected in numeric_types and any(t in actual for t in ["int", "float"]):
        return True

    # String type matches
    string_types = {"object", "string", "str"}
    if expected in string_types and actual in string_types:
        return True

    # Datetime matches
    datetime_types = {"datetime64", "datetime", "date", "timestamp"}
    if expected in datetime_types and "datetime" in actual:
        return True

    return False


def _check_consistency(
    train_df: pd.DataFrame,
    validation_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
) -> tuple[float, List[Dict[str, Any]]]:
    """Check consistency across train/val/test splits.

    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of (score, expectation_results)
    """
    results = []
    consistent_checks = 0
    total_checks = 0

    # Check column consistency across splits
    train_cols = set(train_df.columns)

    if validation_df is not None:
        total_checks += 1
        val_cols = set(validation_df.columns)
        cols_match = train_cols == val_cols

        if cols_match:
            consistent_checks += 1

        results.append(
            {
                "expectation_type": "expect_column_set_to_match",
                "comparison": "train_vs_validation",
                "success": cols_match,
                "result": {
                    "train_only": list(train_cols - val_cols),
                    "validation_only": list(val_cols - train_cols),
                },
            }
        )

    if test_df is not None:
        total_checks += 1
        test_cols = set(test_df.columns)
        cols_match = train_cols == test_cols

        if cols_match:
            consistent_checks += 1

        results.append(
            {
                "expectation_type": "expect_column_set_to_match",
                "comparison": "train_vs_test",
                "success": cols_match,
                "result": {
                    "train_only": list(train_cols - test_cols),
                    "test_only": list(test_cols - train_cols),
                },
            }
        )

    # Check dtype consistency
    for col in train_df.columns:
        train_dtype = str(train_df[col].dtype)

        if validation_df is not None and col in validation_df.columns:
            total_checks += 1
            val_dtype = str(validation_df[col].dtype)
            if train_dtype == val_dtype:
                consistent_checks += 1
            else:
                results.append(
                    {
                        "expectation_type": "expect_dtype_to_match",
                        "column": col,
                        "comparison": "train_vs_validation",
                        "success": False,
                        "result": {"train_dtype": train_dtype, "val_dtype": val_dtype},
                    }
                )

        if test_df is not None and col in test_df.columns:
            total_checks += 1
            test_dtype = str(test_df[col].dtype)
            if train_dtype == test_dtype:
                consistent_checks += 1
            else:
                results.append(
                    {
                        "expectation_type": "expect_dtype_to_match",
                        "column": col,
                        "comparison": "train_vs_test",
                        "success": False,
                        "result": {"train_dtype": train_dtype, "test_dtype": test_dtype},
                    }
                )

    consistency_score = consistent_checks / total_checks if total_checks > 0 else 1.0
    return consistency_score, results


def _check_uniqueness(
    df: pd.DataFrame, unique_columns: List[str]
) -> tuple[float, List[Dict[str, Any]]]:
    """Check for duplicate values.

    Args:
        df: DataFrame to check
        unique_columns: Columns that should have unique values

    Returns:
        Tuple of (score, expectation_results)
    """
    results = []
    unique_checks_passed = 0
    total_checks = 0

    # Check for duplicate rows
    total_checks += 1
    duplicate_rows = df.duplicated().sum()
    duplicate_ratio = duplicate_rows / len(df) if len(df) > 0 else 0

    rows_unique = duplicate_ratio <= 0.05  # Allow up to 5% duplicates
    if rows_unique:
        unique_checks_passed += 1

    results.append(
        {
            "expectation_type": "expect_table_row_uniqueness",
            "success": rows_unique,
            "result": {
                "duplicate_count": int(duplicate_rows),
                "duplicate_ratio": duplicate_ratio,
                "threshold": 0.05,
            },
        }
    )

    # Check specific columns for uniqueness
    for col in unique_columns:
        if col in df.columns:
            total_checks += 1
            duplicate_count = df[col].duplicated().sum()
            is_unique = duplicate_count == 0

            if is_unique:
                unique_checks_passed += 1

            results.append(
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "column": col,
                    "success": is_unique,
                    "duplicate_count": int(duplicate_count),
                    "result": {"duplicate_count": int(duplicate_count)},
                }
            )

    uniqueness_score = unique_checks_passed / total_checks if total_checks > 0 else 1.0
    return uniqueness_score, results


def _check_timeliness(
    df: pd.DataFrame, date_column: str, max_staleness_days: int
) -> tuple[float, List[Dict[str, Any]]]:
    """Check data freshness.

    Args:
        df: DataFrame to check
        date_column: Column containing dates
        max_staleness_days: Maximum acceptable staleness in days

    Returns:
        Tuple of (score, expectation_results)
    """
    results = []

    if date_column not in df.columns:
        # No date column, assume fresh
        results.append(
            {
                "expectation_type": "expect_data_timeliness",
                "success": True,
                "result": {"message": "No date column to check"},
            }
        )
        return 1.0, results

    try:
        # Convert to datetime
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid_dates = dates.dropna()

        if len(valid_dates) == 0:
            results.append(
                {
                    "expectation_type": "expect_data_timeliness",
                    "success": False,
                    "result": {"message": "No valid dates found"},
                }
            )
            return 0.5, results

        # Calculate staleness
        max_date = valid_dates.max()
        now = datetime.now()
        staleness_days = (
            (now - max_date).days
            if hasattr(max_date, "days")
            else (now - max_date.to_pydatetime()).days
        )

        is_fresh = staleness_days <= max_staleness_days

        # Calculate score based on freshness
        if staleness_days <= max_staleness_days:
            timeliness_score = 1.0
        elif staleness_days <= max_staleness_days * 2:
            timeliness_score = 0.7
        else:
            timeliness_score = 0.4

        results.append(
            {
                "expectation_type": "expect_data_timeliness",
                "success": is_fresh,
                "staleness_days": int(staleness_days),
                "result": {
                    "max_date": str(max_date),
                    "staleness_days": int(staleness_days),
                    "max_allowed_days": max_staleness_days,
                },
            }
        )

        return timeliness_score, results

    except Exception as e:
        logger.warning(f"Error checking timeliness: {e}")
        results.append(
            {
                "expectation_type": "expect_data_timeliness",
                "success": True,
                "result": {"message": f"Could not check timeliness: {e}"},
            }
        )
        return 0.8, results
