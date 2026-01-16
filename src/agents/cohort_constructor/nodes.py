"""Node functions for CohortConstructor LangGraph workflow.

Defines the 4-node workflow:
1. validate_config - Validate configuration and input data
2. apply_criteria - Apply inclusion/exclusion criteria
3. validate_temporal - Validate temporal eligibility
4. generate_metadata - Generate execution metadata and audit trail
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .configs import get_brand_config
from .constants import CohortErrorCode, Defaults, SLAThreshold
from .state import CohortConstructorState
from .types import (
    CohortConfig,
    Criterion,
    CriterionType,
    EligibilityLogEntry,
    Operator,
    PatientAssignment,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# NODE 1: VALIDATE CONFIG
# ==============================================================================


async def validate_config(state: CohortConstructorState) -> Dict[str, Any]:
    """Validate cohort configuration and input data.

    This node:
    1. Resolves brand configuration if not explicitly provided
    2. Validates configuration completeness
    3. Checks required fields in source data
    4. Prepares state for criteria application

    Args:
        state: Current pipeline state

    Returns:
        Updated state with validated configuration
    """
    start_time = time.perf_counter()
    config_errors: List[str] = []
    missing_fields: List[str] = []

    try:
        logger.info("Node 1: Validating cohort configuration")

        # Get or resolve configuration
        config_dict = state.get("config")
        brand = state.get("brand")
        indication = state.get("indication")

        cohort_config: Optional[CohortConfig] = None

        if config_dict:
            # Explicit configuration provided
            try:
                cohort_config = CohortConfig.from_dict(config_dict)
                logger.info(f"Using explicit configuration: {cohort_config.cohort_name}")
            except Exception as e:
                config_errors.append(f"Invalid configuration format: {str(e)}")

        elif brand:
            # Use pre-built brand configuration
            try:
                cohort_config = get_brand_config(brand, indication)
                logger.info(f"Using brand configuration: {cohort_config.cohort_name}")
            except ValueError as e:
                config_errors.append(str(e))

        else:
            config_errors.append("No configuration or brand specified")

        # Validate configuration if we have one
        if cohort_config:
            # Check for empty criteria
            if not cohort_config.inclusion_criteria:
                config_errors.append("No inclusion criteria defined")

            # Validate operator types
            for criterion in cohort_config.inclusion_criteria + cohort_config.exclusion_criteria:
                if not isinstance(criterion.operator, Operator):
                    config_errors.append(
                        f"Invalid operator for criterion '{criterion.field}': {criterion.operator}"
                    )

        # Check required fields in source population if provided
        source_population = state.get("source_population")
        if source_population and cohort_config:
            available_columns = source_population.get("columns", [])
            for required_field in cohort_config.required_fields:
                if required_field not in available_columns:
                    missing_fields.append(required_field)

        # Determine validation result
        config_valid = len(config_errors) == 0 and cohort_config is not None
        required_fields_present = len(missing_fields) == 0

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        if not config_valid:
            logger.warning(f"Configuration validation failed: {config_errors}")
            return {
                "validated_config": None,
                "config_valid": False,
                "config_errors": config_errors,
                "required_fields_present": required_fields_present,
                "missing_fields": missing_fields,
                "validate_config_ms": elapsed_ms,
                "current_phase": "validating_config",
                "status": "failed",
                "error": config_errors[0] if config_errors else "Validation failed",
                "error_code": CohortErrorCode.CC_001.value,
                "error_category": "INVALID_CONFIG",
            }

        # Success - return validated configuration as dict
        return {
            "validated_config": cohort_config.to_dict() if cohort_config else None,
            "config_valid": True,
            "config_errors": [],
            "required_fields_present": required_fields_present,
            "missing_fields": missing_fields,
            "validate_config_ms": elapsed_ms,
            "current_phase": "applying_criteria",
            "status": "processing",
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        return {
            "config_valid": False,
            "config_errors": [f"Validation error: {str(e)}"],
            "validate_config_ms": elapsed_ms,
            "current_phase": "validating_config",
            "status": "failed",
            "error": str(e),
            "error_code": CohortErrorCode.CC_001.value,
            "error_category": "INVALID_CONFIG",
        }


# ==============================================================================
# NODE 2: APPLY CRITERIA
# ==============================================================================


def _evaluate_criterion(df: pd.DataFrame, criterion: Criterion) -> pd.Series:
    """Evaluate a single criterion against DataFrame.

    Args:
        df: Patient DataFrame
        criterion: Criterion to evaluate

    Returns:
        Boolean Series indicating which rows match the criterion
    """
    field = criterion.field
    op = criterion.operator
    value = criterion.value

    if field not in df.columns:
        raise ValueError(f"Field '{field}' not found in data")

    column = df[field]

    if op == Operator.EQUAL:
        return column == value
    elif op == Operator.NOT_EQUAL:
        return column != value
    elif op == Operator.GREATER:
        return column > value
    elif op == Operator.GREATER_EQUAL:
        return column >= value
    elif op == Operator.LESS:
        return column < value
    elif op == Operator.LESS_EQUAL:
        return column <= value
    elif op == Operator.IN:
        return column.isin(value)
    elif op == Operator.NOT_IN:
        return ~column.isin(value)
    elif op == Operator.BETWEEN:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"BETWEEN requires [min, max] tuple, got: {value}")
        return (column >= value[0]) & (column <= value[1])
    elif op == Operator.CONTAINS:
        return column.astype(str).str.contains(str(value), na=False)
    else:
        raise ValueError(f"Unsupported operator: {op}")


async def apply_criteria(state: CohortConstructorState) -> Dict[str, Any]:
    """Apply inclusion and exclusion criteria to patient population.

    This node:
    1. Loads patient data from source population
    2. Applies inclusion criteria (AND logic - keep if ALL match)
    3. Applies exclusion criteria (AND NOT logic - remove if ANY match)
    4. Creates eligibility log entries for audit trail

    Args:
        state: Current pipeline state with validated configuration

    Returns:
        Updated state with filtered population and eligibility logs
    """
    start_time = time.perf_counter()
    inclusion_log: List[Dict[str, Any]] = []
    exclusion_log: List[Dict[str, Any]] = []

    try:
        logger.info("Node 2: Applying eligibility criteria")

        # Get validated configuration
        config_dict = state.get("validated_config")
        if not config_dict:
            return {
                "apply_criteria_ms": int((time.perf_counter() - start_time) * 1000),
                "current_phase": "applying_criteria",
                "status": "failed",
                "error": "No validated configuration available",
                "error_code": CohortErrorCode.CC_001.value,
                "error_category": "INVALID_CONFIG",
            }

        config = CohortConfig.from_dict(config_dict)

        # Get patient data (from source_population DataFrame reference)
        source_population = state.get("source_population")
        if not source_population:
            return {
                "apply_criteria_ms": int((time.perf_counter() - start_time) * 1000),
                "current_phase": "applying_criteria",
                "status": "failed",
                "error": "No source population data provided",
                "error_code": CohortErrorCode.CC_002.value,
                "error_category": "MISSING_FIELDS",
            }

        # Expect source_population to contain DataFrame directly or path
        df = source_population.get("dataframe")
        if df is None:
            return {
                "apply_criteria_ms": int((time.perf_counter() - start_time) * 1000),
                "status": "failed",
                "error": "Source population DataFrame not provided",
                "error_code": CohortErrorCode.CC_002.value,
                "error_category": "MISSING_FIELDS",
            }

        initial_count = len(df)
        logger.info(f"Starting with {initial_count} patients")

        # Apply INCLUSION criteria (AND logic)
        eligible_mask = pd.Series([True] * len(df), index=df.index)

        for i, criterion in enumerate(config.inclusion_criteria):
            try:
                criterion_mask = _evaluate_criterion(df, criterion)
                before_count = eligible_mask.sum()
                eligible_mask = eligible_mask & criterion_mask
                after_count = eligible_mask.sum()
                removed_count = before_count - after_count

                log_entry = EligibilityLogEntry(
                    criterion_name=criterion.field,
                    criterion_type="inclusion",
                    criterion_order=i + 1,
                    operator=criterion.operator.value,
                    value=criterion.value,
                    removed_count=removed_count,
                    remaining_count=after_count,
                    description=criterion.description,
                    clinical_rationale=criterion.clinical_rationale,
                )
                inclusion_log.append(log_entry.to_dict())

                logger.debug(
                    f"Inclusion criterion '{criterion.field}': "
                    f"removed {removed_count}, remaining {after_count}"
                )

            except Exception as e:
                logger.error(f"Error applying inclusion criterion '{criterion.field}': {e}")
                return {
                    "apply_criteria_ms": int((time.perf_counter() - start_time) * 1000),
                    "status": "failed",
                    "error": f"Inclusion criterion error: {str(e)}",
                    "error_code": CohortErrorCode.CC_004.value,
                    "error_category": "DATA_VALIDATION",
                }

        post_inclusion_count = eligible_mask.sum()
        logger.info(f"After inclusion criteria: {post_inclusion_count} patients")

        # Apply EXCLUSION criteria (AND NOT logic - remove if ANY match)
        for i, criterion in enumerate(config.exclusion_criteria):
            try:
                exclusion_mask = _evaluate_criterion(df, criterion)
                before_count = eligible_mask.sum()
                eligible_mask = eligible_mask & ~exclusion_mask
                after_count = eligible_mask.sum()
                removed_count = before_count - after_count

                log_entry = EligibilityLogEntry(
                    criterion_name=criterion.field,
                    criterion_type="exclusion",
                    criterion_order=i + 1,
                    operator=criterion.operator.value,
                    value=criterion.value,
                    removed_count=removed_count,
                    remaining_count=after_count,
                    description=criterion.description,
                    clinical_rationale=criterion.clinical_rationale,
                )
                exclusion_log.append(log_entry.to_dict())

                logger.debug(
                    f"Exclusion criterion '{criterion.field}': "
                    f"removed {removed_count}, remaining {after_count}"
                )

            except Exception as e:
                logger.error(f"Error applying exclusion criterion '{criterion.field}': {e}")
                return {
                    "apply_criteria_ms": int((time.perf_counter() - start_time) * 1000),
                    "status": "failed",
                    "error": f"Exclusion criterion error: {str(e)}",
                    "error_code": CohortErrorCode.CC_004.value,
                    "error_category": "DATA_VALIDATION",
                }

        post_exclusion_count = eligible_mask.sum()
        logger.info(f"After exclusion criteria: {post_exclusion_count} patients")

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Store the eligible mask and intermediate counts
        return {
            "post_inclusion_count": int(post_inclusion_count),
            "inclusion_log": inclusion_log,
            "post_exclusion_count": int(post_exclusion_count),
            "exclusion_log": exclusion_log,
            "apply_criteria_ms": elapsed_ms,
            "current_phase": "validating_temporal",
            "status": "processing",
            # Store the mask as list of eligible indices for next node
            "_eligible_indices": df.index[eligible_mask].tolist(),
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Criteria application error: {e}", exc_info=True)
        return {
            "apply_criteria_ms": elapsed_ms,
            "status": "failed",
            "error": str(e),
            "error_code": CohortErrorCode.CC_004.value,
            "error_category": "DATA_VALIDATION",
        }


# ==============================================================================
# NODE 3: VALIDATE TEMPORAL
# ==============================================================================


async def validate_temporal(state: CohortConstructorState) -> Dict[str, Any]:
    """Validate temporal eligibility requirements.

    This node:
    1. Checks lookback period (sufficient historical data before index date)
    2. Checks followup period (sufficient outcome data after index date)
    3. Excludes patients with insufficient temporal coverage
    4. Creates temporal eligibility log entries

    Args:
        state: Current pipeline state with criteria-filtered population

    Returns:
        Updated state with temporal validation results
    """
    start_time = time.perf_counter()
    temporal_log: List[Dict[str, Any]] = []

    try:
        logger.info("Node 3: Validating temporal eligibility")

        # Get configuration and data
        config_dict = state.get("validated_config")
        if not config_dict:
            return {
                "validate_temporal_ms": int((time.perf_counter() - start_time) * 1000),
                "status": "failed",
                "error": "No validated configuration",
                "error_code": CohortErrorCode.CC_001.value,
            }

        config = CohortConfig.from_dict(config_dict)
        temporal_req = config.temporal_requirements

        source_population = state.get("source_population")
        if not source_population:
            return {
                "validate_temporal_ms": int((time.perf_counter() - start_time) * 1000),
                "status": "failed",
                "error": "No source population data",
                "error_code": CohortErrorCode.CC_002.value,
            }

        df = source_population.get("dataframe")
        if df is None:
            return {
                "validate_temporal_ms": int((time.perf_counter() - start_time) * 1000),
                "status": "failed",
                "error": "No DataFrame in source population",
                "error_code": CohortErrorCode.CC_002.value,
            }

        # Get eligible indices from previous node
        eligible_indices = state.get("_eligible_indices", [])
        if not eligible_indices:
            # No patients passed criteria - this is allowed but logged
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "post_temporal_count": 0,
                "temporal_log": [],
                "lookback_exclusions": 0,
                "followup_exclusions": 0,
                "total_temporal_exclusions": 0,
                "validate_temporal_ms": elapsed_ms,
                "current_phase": "generating_metadata",
                "status": "processing",
                "_temporal_eligible_indices": [],
            }

        # Filter to eligible patients
        eligible_df = df.loc[eligible_indices]
        initial_count = len(eligible_df)

        # Get temporal columns
        index_date_field = temporal_req.index_date_field
        first_obs_field = "first_observation_date"
        last_obs_field = "last_observation_date"

        # Check if temporal columns exist
        required_temporal = [index_date_field, first_obs_field, last_obs_field]
        missing_temporal = [f for f in required_temporal if f not in eligible_df.columns]

        if missing_temporal:
            logger.warning(f"Missing temporal fields: {missing_temporal}, skipping temporal validation")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "post_temporal_count": initial_count,
                "temporal_log": [],
                "lookback_exclusions": 0,
                "followup_exclusions": 0,
                "total_temporal_exclusions": 0,
                "validate_temporal_ms": elapsed_ms,
                "current_phase": "generating_metadata",
                "status": "processing",
                "_temporal_eligible_indices": eligible_indices,
                "warnings": [f"Temporal validation skipped: missing fields {missing_temporal}"],
            }

        # Convert dates
        try:
            index_dates = pd.to_datetime(eligible_df[index_date_field])
            first_obs = pd.to_datetime(eligible_df[first_obs_field])
            last_obs = pd.to_datetime(eligible_df[last_obs_field])
        except Exception as e:
            logger.error(f"Date conversion error: {e}")
            return {
                "validate_temporal_ms": int((time.perf_counter() - start_time) * 1000),
                "status": "failed",
                "error": f"Date conversion error: {str(e)}",
                "error_code": CohortErrorCode.CC_005.value,
                "error_category": "TEMPORAL_VALIDATION",
            }

        # Calculate lookback and followup
        lookback_days = (index_dates - first_obs).dt.days
        followup_days = (last_obs - index_dates).dt.days

        # Apply lookback requirement
        lookback_mask = lookback_days >= temporal_req.lookback_days
        lookback_exclusions = (~lookback_mask).sum()

        log_entry = EligibilityLogEntry(
            criterion_name="lookback_period",
            criterion_type="temporal",
            criterion_order=1,
            operator=">=",
            value=temporal_req.lookback_days,
            removed_count=int(lookback_exclusions),
            remaining_count=int(lookback_mask.sum()),
            description=f"Lookback period >= {temporal_req.lookback_days} days",
            clinical_rationale="Sufficient historical data before index date",
        )
        temporal_log.append(log_entry.to_dict())

        # Apply followup requirement
        temporal_mask = lookback_mask.copy()
        followup_mask = followup_days >= temporal_req.followup_days
        before_followup = temporal_mask.sum()
        temporal_mask = temporal_mask & followup_mask
        followup_exclusions = before_followup - temporal_mask.sum()

        log_entry = EligibilityLogEntry(
            criterion_name="followup_period",
            criterion_type="temporal",
            criterion_order=2,
            operator=">=",
            value=temporal_req.followup_days,
            removed_count=int(followup_exclusions),
            remaining_count=int(temporal_mask.sum()),
            description=f"Followup period >= {temporal_req.followup_days} days",
            clinical_rationale="Sufficient outcome observation after index date",
        )
        temporal_log.append(log_entry.to_dict())

        post_temporal_count = temporal_mask.sum()
        total_temporal_exclusions = int(lookback_exclusions) + int(followup_exclusions)

        logger.info(
            f"Temporal validation: {initial_count} -> {post_temporal_count} "
            f"(lookback: -{lookback_exclusions}, followup: -{followup_exclusions})"
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "post_temporal_count": int(post_temporal_count),
            "temporal_log": temporal_log,
            "lookback_exclusions": int(lookback_exclusions),
            "followup_exclusions": int(followup_exclusions),
            "total_temporal_exclusions": total_temporal_exclusions,
            "validate_temporal_ms": elapsed_ms,
            "current_phase": "generating_metadata",
            "status": "processing",
            "_temporal_eligible_indices": eligible_df.index[temporal_mask].tolist(),
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Temporal validation error: {e}", exc_info=True)
        return {
            "validate_temporal_ms": elapsed_ms,
            "status": "failed",
            "error": str(e),
            "error_code": CohortErrorCode.CC_005.value,
            "error_category": "TEMPORAL_VALIDATION",
        }


# ==============================================================================
# NODE 4: GENERATE METADATA
# ==============================================================================


async def generate_metadata(state: CohortConstructorState) -> Dict[str, Any]:
    """Generate execution metadata and finalize cohort results.

    This node:
    1. Extracts final eligible patient IDs
    2. Calculates eligibility statistics
    3. Creates patient assignment records (if enabled)
    4. Generates execution metadata with timing breakdown
    5. Prepares handoff context for data_preparer

    Args:
        state: Current pipeline state with temporal validation complete

    Returns:
        Final state with complete cohort execution results
    """
    start_time = time.perf_counter()

    try:
        logger.info("Node 4: Generating execution metadata")

        # Get configuration
        config_dict = state.get("validated_config")
        config = CohortConfig.from_dict(config_dict) if config_dict else None

        # Get data
        source_population = state.get("source_population")
        df = source_population.get("dataframe") if source_population else None

        # Get eligible indices
        eligible_indices = state.get("_temporal_eligible_indices", [])
        eligible_patient_count = len(eligible_indices)

        # Extract patient IDs
        eligible_patient_ids: List[str] = []
        if df is not None and eligible_indices:
            patient_id_field = "patient_journey_id"
            if patient_id_field in df.columns:
                eligible_df = df.loc[eligible_indices]
                eligible_patient_ids = eligible_df[patient_id_field].astype(str).tolist()
            else:
                # Use index as IDs
                eligible_patient_ids = [str(idx) for idx in eligible_indices]

        # Check for empty cohort
        if eligible_patient_count == 0:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            total_latency_ms = (
                state.get("validate_config_ms", 0) +
                state.get("apply_criteria_ms", 0) +
                state.get("validate_temporal_ms", 0) +
                elapsed_ms
            )

            return {
                "cohort_id": None,
                "eligible_patient_ids": [],
                "eligible_patient_count": 0,
                "generate_metadata_ms": elapsed_ms,
                "total_latency_ms": total_latency_ms,
                "current_phase": "complete",
                "status": "failed",
                "error": "Empty cohort: all patients excluded by eligibility criteria",
                "error_code": CohortErrorCode.CC_003.value,
                "error_category": "EMPTY_COHORT",
                "pipeline_blocked": True,
                "block_reason": "No eligible patients for analysis",
                "suggested_next_agent": None,
            }

        # Generate cohort ID
        cohort_id = config.cohort_id if config else f"cohort_{uuid.uuid4().hex[:12]}"
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        # Calculate statistics
        total_input = len(df) if df is not None else 0
        excluded_count = total_input - eligible_patient_count
        exclusion_rate = excluded_count / total_input if total_input > 0 else 0.0

        # Combine all eligibility logs
        all_logs = (
            state.get("inclusion_log", []) +
            state.get("exclusion_log", []) +
            state.get("temporal_log", [])
        )

        eligibility_stats = {
            "total_input_patients": total_input,
            "eligible_patient_count": eligible_patient_count,
            "excluded_patient_count": excluded_count,
            "exclusion_rate": round(exclusion_rate, 4),
            "eligibility_log": all_logs,
            "temporal_validation": {
                "lookback_exclusions": state.get("lookback_exclusions", 0),
                "followup_exclusions": state.get("followup_exclusions", 0),
                "total_temporal_exclusions": state.get("total_temporal_exclusions", 0),
            },
        }

        # Timing
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        total_latency_ms = (
            state.get("validate_config_ms", 0) +
            state.get("apply_criteria_ms", 0) +
            state.get("validate_temporal_ms", 0) +
            elapsed_ms
        )

        # SLA compliance check
        sla_target_ms = state.get("sla_target_ms", SLAThreshold.TOTAL_EXECUTION_MS)
        sla_compliant = total_latency_ms <= sla_target_ms

        if not sla_compliant:
            logger.warning(
                f"SLA exceeded: {total_latency_ms}ms > {sla_target_ms}ms target"
            )

        # Execution metadata
        execution_metadata = {
            "execution_id": execution_id,
            "execution_timestamp": datetime.now().isoformat(),
            "execution_time_ms": total_latency_ms,
            "environment": state.get("environment", Defaults.DEFAULT_ENVIRONMENT),
            "executed_by": state.get("executed_by"),
            "validate_config_ms": state.get("validate_config_ms", 0),
            "apply_criteria_ms": state.get("apply_criteria_ms", 0),
            "validate_temporal_ms": state.get("validate_temporal_ms", 0),
            "generate_metadata_ms": elapsed_ms,
            "database_records": {},  # Will be populated by database integration
        }

        # Prepare context for next agent (data_preparer)
        context_for_next_agent = {
            "cohort_id": cohort_id,
            "eligible_patient_ids": eligible_patient_ids,
            "cohort_spec": config_dict,
            "quality_checks_required": [
                "temporal_completeness",
                "feature_availability",
                "data_freshness",
            ],
        }

        # Key findings
        key_findings = [
            f"Identified {eligible_patient_count:,} eligible patients from {total_input:,} total",
            f"Exclusion rate: {exclusion_rate:.1%}",
            f"Execution time: {total_latency_ms}ms (SLA: {'compliant' if sla_compliant else 'exceeded'})",
        ]

        logger.info(
            f"Cohort construction complete: {eligible_patient_count} eligible patients, "
            f"execution time {total_latency_ms}ms"
        )

        return {
            "cohort_id": cohort_id,
            "eligible_patient_ids": eligible_patient_ids,
            "eligible_patient_count": eligible_patient_count,
            "eligibility_stats": eligibility_stats,
            "execution_metadata": execution_metadata,
            "generate_metadata_ms": elapsed_ms,
            "total_latency_ms": total_latency_ms,
            "sla_compliant": sla_compliant,
            "current_phase": "complete",
            "status": "completed",
            "context_for_next_agent": context_for_next_agent,
            "suggested_next_agent": "data_preparer",
            "pipeline_blocked": False,
            "key_findings": key_findings,
            "confidence": 1.0 if sla_compliant else 0.9,
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Metadata generation error: {e}", exc_info=True)
        return {
            "generate_metadata_ms": elapsed_ms,
            "status": "failed",
            "error": str(e),
            "error_code": CohortErrorCode.CC_006.value,
            "error_category": "DATABASE_ERROR",
            "pipeline_blocked": True,
            "block_reason": f"Metadata generation failed: {str(e)}",
        }
