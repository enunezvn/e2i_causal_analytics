"""Great Expectations validator node for data_preparer agent.

This node runs Great Expectations validation after data loading.
It uses the DataQualityValidator from src/mlops/data_quality.py.
"""

import logging
from typing import Any, Dict

from src.mlops.data_quality import get_data_quality_validator

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def run_ge_validation(state: DataPreparerState) -> Dict[str, Any]:
    """Run Great Expectations validation on loaded data.

    This node:
    1. Gets the appropriate expectation suite based on data source
    2. Validates train, validation, and test splits
    3. Aggregates results and updates state
    4. Adds blocking issues if validation fails

    Args:
        state: Current agent state

    Returns:
        Updated state with GE validation results
    """
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Running GE validation for experiment {experiment_id}")

    try:
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")

        if train_df is None:
            logger.warning("No train_df found, skipping GE validation")
            return {
                "ge_validation_status": "skipped",
                "ge_validation_reason": "No training data available",
            }

        # Determine suite name from data source.
        # data_source may now be a dict (file ingestion) — in that case we
        # default to ``patient_journeys`` so the auto-detect block below
        # picks the appropriate suite from the loaded DataFrame's columns.
        # Falling back to ``business_metrics`` (the previous behavior)
        # surfaced 7 spurious GE failures on real CSU runs because that
        # suite expects ``id`` / ``metric_value`` cols absent in
        # patient_journeys (backlog item #12).
        #
        # Codex review MEDIUM-E on PR #105: dict shapes the agent does NOT
        # know about (``type`` not in {file_dir, files}) used to silently
        # fall through to ``business_metrics``, masking a real
        # data_source-routing regression. Now they fail closed with an
        # explicit error so a future shape change is loud, not silent.
        scope_spec = state.get("scope_spec", {})
        _ds_raw = state.get("data_source") or scope_spec.get("data_source", "business_metrics")
        if isinstance(_ds_raw, str):
            data_source: str = _ds_raw
        elif isinstance(_ds_raw, dict) and _ds_raw.get("type") in ("file_dir", "files"):
            data_source = "patient_journeys"
        elif isinstance(_ds_raw, dict):
            return {
                "ge_validation_status": "error",
                "ge_validation_error": (
                    f"Unknown data_source dict shape: type="
                    f"{_ds_raw.get('type')!r}; expected one of "
                    f"('file_dir', 'files') or a string suite name"
                ),
                "blocking_issues": state.get("blocking_issues", [])
                + [
                    f"GE validation: unrecognised data_source dict shape "
                    f"(type={_ds_raw.get('type')!r})"
                ],
            }
        else:
            data_source = "business_metrics"

        # Get the validator
        validator = get_data_quality_validator()

        # Check if suite exists for this data source
        available_suites = list(validator.SUITES.keys())

        # Auto-detect ML patient data format vs event-level patient_journeys
        # ML patient data has patient_journey_id and discontinuation_flag but no event_type
        if data_source == "patient_journeys" and train_df is not None:
            has_ml_patient_cols = (
                "patient_journey_id" in train_df.columns
                and "discontinuation_flag" in train_df.columns
            )
            has_event_cols = "event_type" in train_df.columns
            if has_ml_patient_cols and not has_event_cols:
                logger.info("Detected ML patient data format, using 'ml_patients' suite")
                data_source = "ml_patients"

        if data_source not in available_suites:
            logger.info(
                f"No GE suite for '{data_source}', using generic validation. "
                f"Available suites: {available_suites}"
            )
            # Use business_metrics as fallback (most generic)
            suite_name = "business_metrics"
        else:
            suite_name = data_source

        # Validate all splits
        training_run_id_raw = state.get("training_run_id")
        training_run_id = str(training_run_id_raw) if training_run_id_raw is not None else None
        results = await validator.validate_splits(
            train_df=train_df,
            val_df=validation_df,
            test_df=test_df,
            suite_name=suite_name,
            table_name=data_source,
            training_run_id=training_run_id,
        )

        # Aggregate results
        ge_results = []
        all_passed = True
        total_expectations = 0
        total_passed = 0
        blocking_issues = []

        for split_name, result in results.items():
            ge_results.append(result.to_dict())
            total_expectations += result.expectations_evaluated
            total_passed += result.expectations_passed

            if result.blocking:
                all_passed = False
                blocking_issues.append(
                    f"GE validation failed for {split_name}: "
                    f"{result.expectations_failed} expectations failed"
                )

                # Add details of failed expectations
                for failed in result.failed_expectations[:3]:  # Limit to top 3
                    col = failed.get("column", "table")
                    exp_type = failed.get("expectation_type", "unknown")
                    blocking_issues.append(f"  - {exp_type} on {col}")

        # Calculate overall success rate
        overall_success_rate = total_passed / total_expectations if total_expectations > 0 else 1.0

        # Determine overall status
        if all_passed:
            if overall_success_rate >= 0.95:
                ge_status = "passed"
            else:
                ge_status = "warning"
        else:
            ge_status = "failed"

        # Update blocking issues in state
        existing_blocking = state.get("blocking_issues", [])
        updated_blocking = existing_blocking + blocking_issues

        logger.info(
            f"GE validation completed: status={ge_status}, "
            f"passed={total_passed}/{total_expectations}, "
            f"splits_validated={len(results)}"
        )

        return {
            "ge_validation_status": ge_status,
            "ge_validation_results": ge_results,
            "ge_expectations_evaluated": total_expectations,
            "ge_expectations_passed": total_passed,
            "ge_success_rate": overall_success_rate,
            "blocking_issues": updated_blocking if blocking_issues else None,
        }

    except Exception as e:
        logger.error(f"GE validation failed: {e}", exc_info=True)
        return {
            "ge_validation_status": "error",
            "ge_validation_error": str(e),
            "blocking_issues": state.get("blocking_issues", [])
            + [f"GE validation error: {str(e)}"],
        }
