"""Schema validator node for data_preparer agent.

This node runs Pandera schema validation as the FIRST validation step,
before quality_checker and Great Expectations. It provides fast-fail
on schema issues (types, nullability, enums).

Integration Order:
1. load_data
2. run_schema_validation (THIS NODE) <-- Fast schema checks
3. run_quality_checks <-- 5 dimension scoring
4. run_ge_validation <-- Business rules
"""

import logging
import time
from typing import Any, Dict, List

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def run_schema_validation(state: DataPreparerState) -> Dict[str, Any]:
    """Run Pandera schema validation on loaded data.

    This node validates DataFrames against Pandera schemas defined in
    src/mlops/pandera_schemas.py. Schema validation is fast (~10ms)
    and catches structural issues early.

    Validation includes:
    - Column existence and naming
    - Data types (int, float, str, datetime)
    - Nullability constraints
    - Value ranges and categories (E2I enums)

    If schema validation fails, blocking_issues are added to state,
    which will block downstream training.

    Args:
        state: Current agent state with train_df, validation_df, test_df

    Returns:
        Updated state with schema validation results:
        - schema_validation_status: "passed", "failed", "skipped", "error"
        - schema_validation_errors: List of error dicts
        - schema_splits_validated: Number of splits validated
        - schema_validation_time_ms: Execution time in milliseconds
        - blocking_issues: Extended with schema errors if failed
    """
    start_time = time.perf_counter()
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Starting schema validation for experiment {experiment_id}")

    try:
        # Import here to avoid circular imports
        from src.mlops.pandera_schemas import (
            get_schema,
            list_registered_schemas,
            validate_dataframe,
        )

        # Get data source name
        data_source = state.get("data_source", "")
        scope_spec = state.get("scope_spec", {})

        # Try to infer data source from scope_spec if not set
        if not data_source:
            data_source = scope_spec.get("data_source", "")
            if not data_source:
                data_source = scope_spec.get("table_name", "")

        logger.debug(f"Data source for schema validation: {data_source}")

        # Check if we have a schema for this data source
        schema = get_schema(data_source)

        if schema is None:
            # No schema defined - skip validation
            logger.info(
                f"No Pandera schema defined for data source '{data_source}'. "
                f"Registered schemas: {list(list_registered_schemas().keys())}"
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "schema_validation_status": "skipped",
                "schema_validation_errors": [],
                "schema_splits_validated": 0,
                "schema_validation_time_ms": elapsed_ms,
            }

        # Collect DataFrames to validate
        splits_to_validate = []
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")

        if train_df is not None:
            splits_to_validate.append(("train", train_df))
        if validation_df is not None:
            splits_to_validate.append(("validation", validation_df))
        if test_df is not None:
            splits_to_validate.append(("test", test_df))

        if not splits_to_validate:
            logger.warning("No DataFrames found in state to validate")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "schema_validation_status": "skipped",
                "schema_validation_errors": [],
                "schema_splits_validated": 0,
                "schema_validation_time_ms": elapsed_ms,
                "warnings": state.get("warnings", [])
                + ["No DataFrames available for schema validation"],
            }

        # Validate each split
        all_errors: List[Dict[str, Any]] = []
        splits_validated = 0
        splits_passed = 0

        for split_name, df in splits_to_validate:
            logger.debug(
                f"Validating {split_name} split: {len(df)} rows, {len(df.columns)} columns"
            )

            result = validate_dataframe(df, data_source, lazy=True)
            splits_validated += 1

            if result["status"] == "passed":
                splits_passed += 1
                logger.debug(f"{split_name} split passed schema validation")
            elif result["status"] == "failed":
                # Add split context to errors
                for error in result.get("errors", []):
                    error_with_context = {
                        "split": split_name,
                        "data_source": data_source,
                        **error,
                    }
                    all_errors.append(error_with_context)
                logger.warning(
                    f"{split_name} split failed schema validation: "
                    f"{len(result.get('errors', []))} errors"
                )
            else:
                # Error or other status
                for error in result.get("errors", []):
                    error_with_context = {
                        "split": split_name,
                        "data_source": data_source,
                        **error,
                    }
                    all_errors.append(error_with_context)

        # Determine overall status
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        if all_errors:
            # Schema validation failed
            status = "failed"

            # Build blocking issues
            blocking_issues = state.get("blocking_issues", []).copy()
            error_summary = _summarize_schema_errors(all_errors)
            blocking_issues.append(f"Schema validation failed: {error_summary}")

            logger.warning(
                f"Schema validation FAILED for {data_source}: "
                f"{len(all_errors)} errors across {splits_validated} splits "
                f"({elapsed_ms}ms)"
            )

            return {
                "schema_validation_status": status,
                "schema_validation_errors": all_errors,
                "schema_splits_validated": splits_validated,
                "schema_validation_time_ms": elapsed_ms,
                "blocking_issues": blocking_issues,
            }
        else:
            # All splits passed
            logger.info(
                f"Schema validation PASSED for {data_source}: "
                f"{splits_passed}/{splits_validated} splits validated ({elapsed_ms}ms)"
            )

            return {
                "schema_validation_status": "passed",
                "schema_validation_errors": [],
                "schema_splits_validated": splits_validated,
                "schema_validation_time_ms": elapsed_ms,
            }

    except ImportError as e:
        logger.error(f"Failed to import Pandera schemas: {e}")
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "schema_validation_status": "error",
            "schema_validation_errors": [
                {"message": f"Import error: {str(e)}", "type": "ImportError"}
            ],
            "schema_splits_validated": 0,
            "schema_validation_time_ms": elapsed_ms,
            "error": str(e),
            "error_type": "schema_import_error",
        }

    except Exception as e:
        logger.error(f"Schema validation error: {e}", exc_info=True)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "schema_validation_status": "error",
            "schema_validation_errors": [{"message": str(e), "type": type(e).__name__}],
            "schema_splits_validated": 0,
            "schema_validation_time_ms": elapsed_ms,
            "error": str(e),
            "error_type": "schema_validation_error",
        }


def _summarize_schema_errors(errors: List[Dict[str, Any]]) -> str:
    """Create a human-readable summary of schema errors.

    Args:
        errors: List of error dictionaries

    Returns:
        Summary string for blocking_issues
    """
    if not errors:
        return "No errors"

    # Group by column
    columns_with_errors = set()
    check_types = set()

    for error in errors:
        if error.get("column"):
            columns_with_errors.add(error["column"])
        if error.get("check"):
            check_types.add(error["check"])

    parts = []
    if columns_with_errors:
        parts.append(f"columns [{', '.join(sorted(columns_with_errors)[:5])}]")
        if len(columns_with_errors) > 5:
            parts.append(f"(+{len(columns_with_errors) - 5} more)")

    if check_types:
        parts.append(f"checks [{', '.join(sorted(check_types)[:3])}]")

    return f"{len(errors)} error(s) in {' '.join(parts)}"
