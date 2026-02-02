"""Core CohortConstructor implementation.

Implements the 4-node workflow for explicit rule-based patient cohort construction:
1. validate_config - Validate configuration and input data
2. apply_criteria - Apply inclusion/exclusion criteria
3. validate_temporal - Validate temporal eligibility
4. generate_metadata - Generate execution metadata and audit trail

SLA: <120 seconds for 100K patients
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import pandas as pd

from .constants import (
    CohortErrorCode,
    Defaults,
    SLAThreshold,
)
from .types import (
    CohortConfig,
    CohortExecutionResult,
    Criterion,
    CriterionType,
    EligibilityLogEntry,
    Operator,
    PatientAssignment,
)

logger = logging.getLogger(__name__)


class CohortConstructor:
    """Constructs eligible patient cohorts based on explicit clinical criteria.

    Implements a 4-node workflow:
    1. validate_config - Validate configuration and required fields
    2. apply_criteria - Apply inclusion/exclusion criteria
    3. validate_temporal - Validate temporal eligibility (lookback/followup)
    4. generate_metadata - Generate execution metadata and audit trail

    SLA: <120 seconds for 100K patients

    Example:
        config = CohortConfig.from_brand("remibrutinib")
        constructor = CohortConstructor(config)
        eligible_df, result = constructor.construct_cohort(patient_df)
    """

    def __init__(self, config: CohortConfig):
        """Initialize CohortConstructor with configuration.

        Args:
            config: CohortConfig with criteria and temporal requirements
        """
        self.config = config
        self.eligibility_log: List[EligibilityLogEntry] = []
        self.patient_assignments: List[PatientAssignment] = []
        self.validation_errors: List[Dict[str, Any]] = []

        # Node latencies (milliseconds)
        self._validate_config_ms: int = 0
        self._apply_criteria_ms: int = 0
        self._validate_temporal_ms: int = 0
        self._generate_metadata_ms: int = 0

        # Criterion counter for ordering
        self._criterion_order: int = 0

    def construct_cohort(
        self, df: pd.DataFrame, track_assignments: bool = True
    ) -> Tuple[pd.DataFrame, CohortExecutionResult]:
        """Main cohort construction pipeline.

        Executes the 4-node workflow:
        1. Validate configuration and required fields
        2. Apply inclusion criteria
        3. Apply exclusion criteria
        4. Validate temporal eligibility
        5. Generate metadata

        Args:
            df: Patient DataFrame with required fields
            track_assignments: Whether to track individual patient eligibility

        Returns:
            Tuple of (eligible_df, CohortExecutionResult)

        Raises:
            ValueError: If required fields are missing (CC_002)
        """
        start_time = time.time()
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        logger.info("=" * 60)
        logger.info("COHORT CONSTRUCTION")
        logger.info("=" * 60)
        logger.info(f"Brand: {self.config.brand}")
        logger.info(f"Indication: {self.config.indication}")
        logger.info(f"Initial Population: {len(df):,}")

        # Reset state
        self._reset_state()
        initial_count = len(df)

        try:
            # NODE 1: Validate configuration and required fields
            node1_start = time.time()
            self._validate_required_fields(df)
            self._validate_config_ms = int((time.time() - node1_start) * 1000)

            # NODE 2: Apply criteria
            node2_start = time.time()
            eligible = self._apply_inclusion_criteria(df)
            eligible = self._apply_exclusion_criteria(eligible)
            self._apply_criteria_ms = int((time.time() - node2_start) * 1000)

            # NODE 3: Validate temporal eligibility
            node3_start = time.time()
            eligible = self._validate_temporal_eligibility(eligible)
            self._validate_temporal_ms = int((time.time() - node3_start) * 1000)

            # Track patient assignments if requested
            if track_assignments:
                self._track_patient_assignments(df, eligible)

            # NODE 4: Generate metadata
            node4_start = time.time()
            result = self._create_execution_result(
                initial_df=df,
                eligible_df=eligible,
                execution_id=execution_id,
                start_time=start_time,
            )
            self._generate_metadata_ms = int((time.time() - node4_start) * 1000)

            logger.info(f"\n{'=' * 60}")
            logger.info("COHORT CONSTRUCTION COMPLETE")
            logger.info(f"{'=' * 60}")
            logger.info(f"Initial Population: {initial_count:,}")
            logger.info(f"Eligible Population: {len(eligible):,}")
            logger.info(f"Exclusion Rate: {result.eligibility_stats.get('exclusion_rate', 0):.1%}")

            return eligible, result

        except ValueError as e:
            # Handle validation errors
            total_ms = int((time.time() - start_time) * 1000)
            result = self._create_error_result(
                execution_id=execution_id,
                initial_count=initial_count,
                error_code=CohortErrorCode.CC_002,
                error_message=str(e),
                total_ms=total_ms,
            )
            return pd.DataFrame(), result

        except Exception as e:
            # Handle unexpected errors
            total_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"Unexpected error during cohort construction: {e}")
            result = self._create_error_result(
                execution_id=execution_id,
                initial_count=initial_count,
                error_code=CohortErrorCode.CC_004,
                error_message=str(e),
                total_ms=total_ms,
            )
            return pd.DataFrame(), result

    def _reset_state(self) -> None:
        """Reset internal state for new construction."""
        self.eligibility_log = []
        self.patient_assignments = []
        self.validation_errors = []
        self._criterion_order = 0
        self._validate_config_ms = 0
        self._apply_criteria_ms = 0
        self._validate_temporal_ms = 0
        self._generate_metadata_ms = 0

    def _validate_required_fields(self, df: pd.DataFrame) -> None:
        """Validate that required fields are present.

        Args:
            df: Patient DataFrame

        Raises:
            ValueError: If required fields are missing (CC_002)
        """
        missing_fields = [f for f in self.config.required_fields if f not in df.columns]

        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            logger.error(error_msg)
            self.validation_errors.append(
                {
                    "type": "missing_fields",
                    "fields": missing_fields,
                    "error_code": CohortErrorCode.CC_002.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            raise ValueError(error_msg)

        logger.info(f"All {len(self.config.required_fields)} required fields present")

    def _apply_inclusion_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inclusion criteria to filter eligible patients.

        Args:
            df: Patient DataFrame

        Returns:
            Filtered DataFrame with only patients meeting inclusion criteria
        """
        logger.info(
            f"\nApplying Inclusion Criteria ({len(self.config.inclusion_criteria)} criteria)..."
        )
        logger.info(f"   Starting population: {len(df):,}")

        eligible = df.copy()

        for criterion in self.config.inclusion_criteria:
            initial_count = len(eligible)
            eligible = self._apply_criterion(eligible, criterion, CriterionType.INCLUSION)
            removed = initial_count - len(eligible)

            logger.info(f"   {criterion.field}: {removed:,} excluded ({len(eligible):,} remaining)")

            self._criterion_order += 1
            self.eligibility_log.append(
                EligibilityLogEntry(
                    criterion_name=criterion.field,
                    criterion_type="inclusion",
                    criterion_order=self._criterion_order,
                    operator=criterion.operator.value,
                    value=criterion.value,
                    removed_count=removed,
                    remaining_count=len(eligible),
                    description=criterion.description,
                    clinical_rationale=criterion.clinical_rationale,
                )
            )

        return eligible

    def _apply_exclusion_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exclusion criteria to remove ineligible patients.

        Args:
            df: Patient DataFrame

        Returns:
            Filtered DataFrame with excluded patients removed
        """
        logger.info(
            f"\nApplying Exclusion Criteria ({len(self.config.exclusion_criteria)} criteria)..."
        )

        eligible = df.copy()

        for criterion in self.config.exclusion_criteria:
            if criterion.field not in eligible.columns:
                logger.warning(f"   Field {criterion.field} not found, skipping")
                continue

            initial_count = len(eligible)
            eligible = self._apply_criterion(eligible, criterion, CriterionType.EXCLUSION)
            removed = initial_count - len(eligible)

            logger.info(f"   {criterion.field}: {removed:,} excluded ({len(eligible):,} remaining)")

            self._criterion_order += 1
            self.eligibility_log.append(
                EligibilityLogEntry(
                    criterion_name=criterion.field,
                    criterion_type="exclusion",
                    criterion_order=self._criterion_order,
                    operator=criterion.operator.value,
                    value=criterion.value,
                    removed_count=removed,
                    remaining_count=len(eligible),
                    description=criterion.description,
                    clinical_rationale=criterion.clinical_rationale,
                )
            )

        return eligible

    def _apply_criterion(
        self,
        df: pd.DataFrame,
        criterion: Criterion,
        criterion_type: CriterionType,
    ) -> pd.DataFrame:
        """Apply a single criterion using the appropriate operator.

        Args:
            df: Patient DataFrame
            criterion: Criterion to apply
            criterion_type: Whether inclusion or exclusion

        Returns:
            Filtered DataFrame
        """
        if criterion.field not in df.columns:
            return df

        field_data = df[criterion.field]
        mask = self._compute_mask(field_data, criterion.operator, criterion.value)

        # Apply based on criterion type
        if criterion_type == CriterionType.INCLUSION:
            # Keep only those who meet inclusion criteria
            return df[mask]
        else:  # EXCLUSION
            # Remove those who meet exclusion criteria
            return df[~mask]

    def _compute_mask(self, field_data: pd.Series, operator: Operator, value: Any) -> pd.Series:
        """Compute boolean mask for operator and value.

        Args:
            field_data: Series of field values
            operator: Comparison operator
            value: Value to compare against

        Returns:
            Boolean Series mask

        Raises:
            ValueError: If operator is not supported
        """
        if operator == Operator.EQUAL:
            return field_data == value
        elif operator == Operator.NOT_EQUAL:
            return field_data != value
        elif operator == Operator.GREATER:
            return field_data > value
        elif operator == Operator.GREATER_EQUAL:
            return field_data >= value
        elif operator == Operator.LESS:
            return field_data < value
        elif operator == Operator.LESS_EQUAL:
            return field_data <= value
        elif operator == Operator.IN:
            return field_data.isin(value)
        elif operator == Operator.NOT_IN:
            return ~field_data.isin(value)
        elif operator == Operator.BETWEEN:
            return field_data.between(value[0], value[1])
        elif operator == Operator.CONTAINS:
            return field_data.astype(str).str.contains(value, na=False)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _validate_temporal_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate temporal eligibility (lookback and followup periods).

        Args:
            df: Patient DataFrame

        Returns:
            DataFrame with only temporally eligible patients
        """
        temporal = self.config.temporal_requirements
        logger.info("\nValidating Temporal Eligibility...")
        logger.info(f"   Lookback required: {temporal.lookback_days} days")
        logger.info(f"   Follow-up required: {temporal.followup_days} days")

        eligible = df.copy()
        len(eligible)

        # Define index date
        if temporal.index_date_field in eligible.columns:
            eligible["_index_date"] = pd.to_datetime(
                eligible[temporal.index_date_field], errors="coerce"
            )
        else:
            logger.warning(f"   Index date field '{temporal.index_date_field}' not found")
            return eligible

        # Check lookback completeness
        if "journey_start_date" in eligible.columns:
            eligible["_journey_start"] = pd.to_datetime(
                eligible["journey_start_date"], errors="coerce"
            )
            eligible["_lookback_days"] = (
                eligible["_index_date"] - eligible["_journey_start"]
            ).dt.days

            lookback_mask = eligible["_lookback_days"] >= temporal.lookback_days
            pre_lookback_count = len(eligible)
            eligible = eligible[lookback_mask]
            lookback_removed = pre_lookback_count - len(eligible)

            logger.info(
                f"   Lookback validation: {lookback_removed:,} excluded ({len(eligible):,} remaining)"
            )

            self._criterion_order += 1
            self.eligibility_log.append(
                EligibilityLogEntry(
                    criterion_name="lookback_days",
                    criterion_type="temporal",
                    criterion_order=self._criterion_order,
                    operator=">=",
                    value=temporal.lookback_days,
                    removed_count=lookback_removed,
                    remaining_count=len(eligible),
                    description="Sufficient historical data",
                    clinical_rationale="Lookback period for baseline characteristics",
                )
            )

        # Check follow-up completeness
        if "follow_up_days" in eligible.columns:
            followup_mask = eligible["follow_up_days"] >= temporal.followup_days
            pre_followup_count = len(eligible)
            eligible = eligible[followup_mask]
            followup_removed = pre_followup_count - len(eligible)

            logger.info(
                f"   Follow-up validation: {followup_removed:,} excluded ({len(eligible):,} remaining)"
            )

            self._criterion_order += 1
            self.eligibility_log.append(
                EligibilityLogEntry(
                    criterion_name="followup_days",
                    criterion_type="temporal",
                    criterion_order=self._criterion_order,
                    operator=">=",
                    value=temporal.followup_days,
                    removed_count=followup_removed,
                    remaining_count=len(eligible),
                    description="Sufficient follow-up period",
                    clinical_rationale="Follow-up period for outcome observation",
                )
            )

        # Clean up temporary columns
        temp_cols = [c for c in eligible.columns if c.startswith("_")]
        eligible = eligible.drop(columns=temp_cols, errors="ignore")

        return eligible

    def _track_patient_assignments(
        self, initial_df: pd.DataFrame, eligible_df: pd.DataFrame
    ) -> None:
        """Track individual patient eligibility assignments.

        Args:
            initial_df: Initial patient DataFrame
            eligible_df: Eligible patient DataFrame
        """
        if "patient_journey_id" not in initial_df.columns:
            return

        eligible_ids = set(eligible_df["patient_journey_id"].tolist())

        for _, row in initial_df.iterrows():
            patient_id = row.get("patient_journey_id", "")
            is_eligible = patient_id in eligible_ids

            # Determine failed criteria for ineligible patients
            failed_criteria = []
            if not is_eligible:
                failed_criteria = self._determine_failed_criteria(row)

            self.patient_assignments.append(
                PatientAssignment(
                    patient_journey_id=patient_id,
                    is_eligible=is_eligible,
                    failed_criteria=failed_criteria,
                    index_date=str(row.get(self.config.temporal_requirements.index_date_field, "")),
                    journey_start_date=str(row.get("journey_start_date", "")),
                    journey_end_date=str(row.get("journey_end_date", "")),
                )
            )

    def _determine_failed_criteria(self, row: pd.Series) -> List[str]:
        """Determine which criteria a patient failed.

        Args:
            row: Patient data row

        Returns:
            List of failed criterion names
        """
        failed = []

        # Check inclusion criteria
        for criterion in self.config.inclusion_criteria:
            if criterion.field not in row.index:
                continue
            value = row[criterion.field]
            if pd.isna(value):
                failed.append(criterion.field)
                continue
            mask = self._compute_mask(pd.Series([value]), criterion.operator, criterion.value)
            if not mask.iloc[0]:
                failed.append(criterion.field)

        # Check exclusion criteria
        for criterion in self.config.exclusion_criteria:
            if criterion.field not in row.index:
                continue
            value = row[criterion.field]
            if pd.isna(value):
                continue
            mask = self._compute_mask(pd.Series([value]), criterion.operator, criterion.value)
            if mask.iloc[0]:
                failed.append(f"exclusion:{criterion.field}")

        return failed

    def _create_execution_result(
        self,
        initial_df: pd.DataFrame,
        eligible_df: pd.DataFrame,
        execution_id: str,
        start_time: float,
    ) -> CohortExecutionResult:
        """Create execution result with statistics and metadata.

        Args:
            initial_df: Initial patient DataFrame
            eligible_df: Eligible patient DataFrame
            execution_id: Unique execution ID
            start_time: Execution start time

        Returns:
            CohortExecutionResult with full metadata
        """
        total_ms = int((time.time() - start_time) * 1000)
        initial_count = len(initial_df)
        eligible_count = len(eligible_df)
        excluded_count = initial_count - eligible_count
        exclusion_rate = excluded_count / initial_count if initial_count > 0 else 0.0

        # Extract eligible patient IDs
        eligible_ids = []
        if "patient_journey_id" in eligible_df.columns:
            eligible_ids = eligible_df["patient_journey_id"].tolist()

        # Build eligibility stats
        eligibility_stats = {
            "total_input_patients": initial_count,
            "eligible_patient_count": eligible_count,
            "excluded_patient_count": excluded_count,
            "exclusion_rate": exclusion_rate,
        }

        # Build execution metadata
        execution_metadata = {
            "execution_id": execution_id,
            "execution_timestamp": datetime.now().isoformat(),
            "execution_time_ms": total_ms,
            "environment": Defaults.DEFAULT_ENVIRONMENT,
            "validate_config_ms": self._validate_config_ms,
            "apply_criteria_ms": self._apply_criteria_ms,
            "validate_temporal_ms": self._validate_temporal_ms,
            "generate_metadata_ms": self._generate_metadata_ms,
            "sla_target_ms": SLAThreshold.TOTAL_EXECUTION_MS,
            "sla_compliant": total_ms <= SLAThreshold.TOTAL_EXECUTION_MS,
            "config": self.config.to_dict(),
        }

        return CohortExecutionResult(
            cohort_id=self.config.cohort_id,
            execution_id=execution_id,
            eligible_patient_ids=eligible_ids,
            eligibility_stats=eligibility_stats,
            eligibility_log=self.eligibility_log,
            patient_assignments=self.patient_assignments,
            execution_metadata=execution_metadata,
            status="success",
        )

    def _create_error_result(
        self,
        execution_id: str,
        initial_count: int,
        error_code: CohortErrorCode,
        error_message: str,
        total_ms: int,
    ) -> CohortExecutionResult:
        """Create error result for failed execution.

        Args:
            execution_id: Unique execution ID
            initial_count: Initial patient count
            error_code: Error code
            error_message: Error message
            total_ms: Total execution time

        Returns:
            CohortExecutionResult with error status
        """
        return CohortExecutionResult(
            cohort_id=self.config.cohort_id,
            execution_id=execution_id,
            eligible_patient_ids=[],
            eligibility_stats={
                "total_input_patients": initial_count,
                "eligible_patient_count": 0,
                "excluded_patient_count": initial_count,
                "exclusion_rate": 1.0,
            },
            eligibility_log=self.eligibility_log,
            patient_assignments=[],
            execution_metadata={
                "execution_id": execution_id,
                "execution_timestamp": datetime.now().isoformat(),
                "execution_time_ms": total_ms,
            },
            status="failed",
            error_code=error_code.value,
            error_message=error_message,
        )

    def is_eligible(self, patient_row: pd.Series) -> bool:
        """Check if a single patient is eligible.

        Args:
            patient_row: Series representing a patient record

        Returns:
            True if eligible, False otherwise
        """
        try:
            patient_df = pd.DataFrame([patient_row])
            eligible_df, _ = self.construct_cohort(patient_df, track_assignments=False)
            return len(eligible_df) > 0
        except Exception as e:
            logger.error(f"Error checking eligibility: {e}")
            return False

    def summary_report(self, result: CohortExecutionResult) -> str:
        """Generate human-readable summary report.

        Args:
            result: CohortExecutionResult from construct_cohort

        Returns:
            Formatted summary report string
        """
        stats = result.eligibility_stats
        metadata = result.execution_metadata

        report = f"""
{"=" * 60}
COHORT CONSTRUCTION SUMMARY
{"=" * 60}

Cohort: {self.config.cohort_name}
Brand: {self.config.brand.upper()}
Indication: {self.config.indication.upper()}
Version: {self.config.version}
Cohort ID: {result.cohort_id}

POPULATION STATISTICS
{"=" * 60}
Initial Population:     {stats.get("total_input_patients", 0):>10,}
Eligible Population:    {stats.get("eligible_patient_count", 0):>10,}
Excluded Population:    {stats.get("excluded_patient_count", 0):>10,}
Exclusion Rate:         {stats.get("exclusion_rate", 0):>10.1%}

CRITERIA APPLIED
{"=" * 60}
Inclusion Criteria:     {len(self.config.inclusion_criteria):>10}
Exclusion Criteria:     {len(self.config.exclusion_criteria):>10}
Temporal Validation:    Lookback {self.config.temporal_requirements.lookback_days}d, Follow-up {self.config.temporal_requirements.followup_days}d

PERFORMANCE
{"=" * 60}
Total Execution Time:   {metadata.get("execution_time_ms", 0):>10,} ms
SLA Target:             {metadata.get("sla_target_ms", 120000):>10,} ms
SLA Compliant:          {metadata.get("sla_compliant", True)}

ELIGIBILITY LOG
{"=" * 60}
"""
        for entry in result.eligibility_log:
            report += f"\n{entry.criterion_order}. [{entry.criterion_type.upper()}] {entry.criterion_name}"
            report += f"\n   Operator: {entry.operator}"
            report += f"\n   Value: {entry.value}"
            report += (
                f"\n   Removed: {entry.removed_count:,} | Remaining: {entry.remaining_count:,}"
            )
            if entry.description:
                report += f"\n   Description: {entry.description}"

        report += f"\n\n{'=' * 60}\n"

        return report


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def create_cohort_quick(
    brand: str, indication: str, df: pd.DataFrame
) -> Tuple[pd.DataFrame, CohortExecutionResult]:
    """Quick cohort construction with pre-built configuration.

    Args:
        brand: Drug brand (e.g., 'remibrutinib')
        indication: Indication (e.g., 'csu')
        df: Patient DataFrame

    Returns:
        Tuple of (eligible_df, CohortExecutionResult)
    """
    from .configs import get_brand_config

    config = get_brand_config(brand, indication)
    constructor = CohortConstructor(config)
    return constructor.construct_cohort(df)


def compare_cohorts(df: pd.DataFrame, configs: List[CohortConfig]) -> pd.DataFrame:
    """Compare multiple cohort definitions on same dataset.

    Args:
        df: Patient DataFrame
        configs: List of CohortConfig objects

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for config in configs:
        constructor = CohortConstructor(config)
        eligible_df, result = constructor.construct_cohort(df, track_assignments=False)
        stats = result.eligibility_stats

        results.append(
            {
                "cohort_name": config.cohort_name,
                "brand": config.brand,
                "indication": config.indication,
                "initial_population": stats.get("total_input_patients", 0),
                "eligible_population": stats.get("eligible_patient_count", 0),
                "exclusion_rate": stats.get("exclusion_rate", 0),
                "inclusion_criteria": len(config.inclusion_criteria),
                "exclusion_criteria": len(config.exclusion_criteria),
            }
        )

    return pd.DataFrame(results)
