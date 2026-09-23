"""Great Expectations validator node for data_preparer agent.

This node runs Great Expectations validation after data loading.
It uses the DataQualityValidator from src/mlops/data_quality.py.
"""

import logging
from typing import Any, Dict, List, Optional

from src.mlops.data_quality import ExpectationSuiteBuilder, get_data_quality_validator

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


def _register_contract_suite(
    validator: Any,
    table: str,
    base_suite: Optional[str],
    columns: List[str],
    target: Optional[str],
) -> str:
    """Register (idempotently) the suite a COLUMN-SCOPED table contract is held to.

    #2207 split contract (codex r1 HIGH on PR #2241, owner decision 2026-09-23): the
    per-table suites describe the WHOLE table — ``patient_journeys`` expects
    ``event_type`` / ``event_date`` / ``patient_id`` — while a contract's ``columns`` is a
    deliberate PROJECTION (measured on the live Kisqali initiation contract: 3/18,
    blocking, against the whole-table suite). So the table's suite is kept but
    FILTERED: table-level expectations and those on projected columns stay and still
    block; expectations on columns outside the projection are NOT APPLICABLE — skipped
    and logged at INFO, never failed and never dropped as a whole. The contract adds its
    own checks: every declared column exists and the prediction target is non-null.
    Re-registering the same name just overwrites it.
    """
    suite_name = f"{table}__contract"
    projection = set(columns)
    kept: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(expectation: Dict[str, Any]) -> None:
        kwargs = expectation.get("kwargs", {}) or {}
        key = (expectation.get("expectation_type"), tuple(sorted(kwargs.items())))
        if key not in seen:
            seen.add(key)
            kept.append(expectation)

    skipped: List[str] = []
    base = list(validator.SUITES.get(base_suite, [])) if base_suite else []
    for expectation in base:
        column = (expectation.get("kwargs", {}) or {}).get("column")
        if column is None or column in projection:
            _add(expectation)
        else:
            skipped.append(f"{expectation.get('expectation_type')} on {column}")
    for name in skipped:
        logger.info(
            "GE suite %r: skipping %s — not in contract projection (table %s)",
            base_suite,
            name,
            table,
        )

    builder = ExpectationSuiteBuilder(suite_name).expect_table_row_count_to_be_between(min_value=1)
    for column in columns:
        builder = builder.expect_column_to_exist(column)
    if target and target in columns:
        builder = builder.expect_column_values_to_not_be_null(target)
    for expectation in builder.build():
        _add(expectation)

    validator.register_suite(suite_name, kept)
    n_kept_base = len(base) - len(skipped)
    logger.info(
        "GE contract suite %s: %d expectation(s) (%d kept from %r, %d skipped, %d contract checks)",
        suite_name,
        len(kept),
        n_kept_base,
        base_suite,
        len(skipped),
        len(kept) - n_kept_base,
    )
    return suite_name


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
        contract_columns: Optional[List[str]] = None
        if isinstance(_ds_raw, str):
            data_source: str = _ds_raw
        elif isinstance(_ds_raw, dict) and _ds_raw.get("type") in ("file_dir", "files"):
            data_source = "patient_journeys"
        elif isinstance(_ds_raw, dict) and _ds_raw.get("type") == "table":
            # #2207 split contract: a table cohort dict validates against its table's
            # own suite; a column-scoped one against a suite derived from the contract
            # (see _register_contract_suite).
            data_source = str(_ds_raw.get("table") or "")
            raw_columns = _ds_raw.get("columns")
            if raw_columns:
                contract_columns = [str(c) for c in raw_columns]
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

        contract_table = data_source
        contract_suite: Optional[str] = None

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

        if contract_columns:
            contract_suite = _register_contract_suite(
                validator,
                contract_table,
                data_source if data_source in available_suites else None,
                contract_columns,
                scope_spec.get("prediction_target"),
            )
            suite_name = contract_suite
        elif data_source not in available_suites:
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

            # A contract suite is a SPECIFICATION, not a quality score: one missing
            # declared column or a null target is a contract violation even though it
            # scores above the validator's 0.8 success-rate threshold (5/6 = 83 %).
            contract_violation = contract_suite is not None and result.expectations_failed > 0
            if result.blocking or contract_violation:
                all_passed = False
                blocking_issues.append(
                    f"GE validation failed for {split_name}: "
                    f"{result.expectations_failed} expectations failed"
                    + (" (table cohort contract violated)" if contract_violation else "")
                )

                # Add details of failed expectations
                for failed in result.failed_expectations[:3]:  # Limit to top 3
                    col = failed.get("column", "table")
                    exp_type = failed.get("expectation_type", "unknown")
                    blocking_issues.append(f"  - {exp_type} on {col}")

        # Calculate overall success rate
        overall_success_rate = total_passed / total_expectations if total_expectations > 0 else 1.0

        # Determine overall status
        ge_note: Optional[str] = None
        if all_passed:
            if contract_suite is not None and total_expectations == 0:
                # Never "passed" on zero applicable expectations.
                ge_status = "warning"
                ge_note = (
                    f"contract suite {contract_suite} has zero applicable expectations "
                    "for this projection"
                )
            elif overall_success_rate >= 0.95:
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
            **({"ge_validation_note": ge_note} if ge_note else {}),
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
