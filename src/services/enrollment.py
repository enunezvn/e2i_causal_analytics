"""
Enrollment Service for A/B Testing.

Phase 15: A/B Testing Infrastructure

Manages experiment enrollment lifecycle:
- Eligibility validation
- Unit enrollment and assignment tracking
- Consent management
- Withdrawal handling
- Protocol deviation tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class EnrollmentStatus(str, Enum):
    """Enrollment status types."""

    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    EXCLUDED = "excluded"
    COMPLETED = "completed"
    LOST_TO_FOLLOWUP = "lost_to_followup"


class WithdrawalInitiator(str, Enum):
    """Who initiated the withdrawal."""

    SUBJECT = "subject"
    INVESTIGATOR = "investigator"
    SPONSOR = "sponsor"
    SYSTEM = "system"


class ConsentMethod(str, Enum):
    """Consent collection methods."""

    EMAIL = "email"
    PHONE = "phone"
    IN_PERSON = "in_person"
    IMPLIED = "implied"
    DIGITAL = "digital"


class DeviationSeverity(str, Enum):
    """Protocol deviation severity levels."""

    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class EligibilityCriteria:
    """Criteria for enrollment eligibility."""

    # Minimum requirements
    min_rx_history_months: int = 0
    min_patient_panel_size: int = 0
    active_in_territory: bool = True

    # Exclusion criteria
    not_in_concurrent_study: bool = True
    no_recent_protocol_violations: bool = True

    # Custom criteria (key: description, value: required)
    custom_criteria: Dict[str, bool] = field(default_factory=dict)


@dataclass
class EligibilityResult:
    """Result of eligibility check."""

    is_eligible: bool
    criteria_results: Dict[str, bool]
    failed_criteria: List[str]
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None


@dataclass
class EnrollmentRecord:
    """Enrollment record for a unit."""

    id: UUID
    assignment_id: UUID
    enrolled_at: datetime
    enrollment_status: EnrollmentStatus
    eligibility_criteria_met: Dict[str, bool]
    consent_timestamp: Optional[datetime] = None
    consent_method: Optional[ConsentMethod] = None
    consent_version: Optional[str] = None
    withdrawal_timestamp: Optional[datetime] = None
    withdrawal_reason: Optional[str] = None
    withdrawal_initiated_by: Optional[WithdrawalInitiator] = None
    protocol_deviations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProtocolDeviation:
    """Protocol deviation record."""

    date: datetime
    deviation_type: str
    severity: DeviationSeverity
    description: str
    corrective_action: Optional[str] = None


@dataclass
class EnrollmentStats:
    """Enrollment statistics for an experiment."""

    experiment_id: UUID
    total_assigned: int
    total_enrolled: int
    active_count: int
    withdrawn_count: int
    excluded_count: int
    completed_count: int
    lost_to_followup_count: int
    enrollment_rate: float
    withdrawal_rate: float
    by_variant: Dict[str, Dict[str, int]]
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnrollmentConfig:
    """Configuration for enrollment service."""

    # Consent settings
    require_explicit_consent: bool = True
    default_consent_version: str = "1.0"

    # Eligibility defaults
    default_criteria: EligibilityCriteria = field(default_factory=EligibilityCriteria)

    # Protocol settings
    max_minor_deviations: int = 3
    auto_exclude_on_major_deviation: bool = True


class EnrollmentService:
    """
    Service for managing experiment enrollments.

    Handles the complete enrollment lifecycle from eligibility
    checking through completion or withdrawal.
    """

    def __init__(self, config: Optional[EnrollmentConfig] = None):
        """
        Initialize enrollment service.

        Args:
            config: Enrollment configuration
        """
        self.config = config or EnrollmentConfig()

    async def check_eligibility(
        self,
        experiment_id: UUID,
        unit: Dict[str, Any],
        criteria: Optional[EligibilityCriteria] = None,
    ) -> EligibilityResult:
        """
        Check if a unit meets eligibility criteria.

        Args:
            experiment_id: Experiment UUID
            unit: Unit data dictionary
            criteria: Optional criteria override

        Returns:
            Eligibility result with detailed criteria evaluation
        """
        criteria = criteria or self.config.default_criteria
        results: Dict[str, bool] = {}
        failed: List[str] = []

        # Check rx history (handle None values)
        rx_months = unit.get("rx_history_months") or 0
        results["min_rx_history"] = rx_months >= criteria.min_rx_history_months
        if not results["min_rx_history"]:
            failed.append(
                f"Insufficient Rx history: {rx_months} < {criteria.min_rx_history_months} months"
            )

        # Check patient panel size (handle None values)
        panel_size = unit.get("patient_panel_size") or 0
        results["min_patient_panel"] = panel_size >= criteria.min_patient_panel_size
        if not results["min_patient_panel"]:
            failed.append(
                f"Insufficient patient panel: {panel_size} < {criteria.min_patient_panel_size}"
            )

        # Check territory activity
        if criteria.active_in_territory:
            is_active = unit.get("active_in_territory", True)
            results["active_in_territory"] = is_active
            if not results["active_in_territory"]:
                failed.append("Unit not active in territory")

        # Check concurrent study participation
        if criteria.not_in_concurrent_study:
            in_study = unit.get("in_concurrent_study", False)
            results["not_in_concurrent_study"] = not in_study
            if not results["not_in_concurrent_study"]:
                failed.append("Unit enrolled in concurrent study")

        # Check recent protocol violations
        if criteria.no_recent_protocol_violations:
            has_violations = unit.get("recent_protocol_violations", False)
            results["no_recent_violations"] = not has_violations
            if not results["no_recent_violations"]:
                failed.append("Unit has recent protocol violations")

        # Check custom criteria
        for criterion_name, required in criteria.custom_criteria.items():
            met = unit.get(criterion_name, not required)  # Default to passing if not specified
            results[criterion_name] = met if required else True
            if required and not met:
                failed.append(f"Custom criterion not met: {criterion_name}")

        is_eligible = len(failed) == 0

        logger.info(
            f"Eligibility check for unit {unit.get('id', 'unknown')} in experiment "
            f"{experiment_id}: {'eligible' if is_eligible else 'not eligible'}"
        )

        return EligibilityResult(
            is_eligible=is_eligible,
            criteria_results=results,
            failed_criteria=failed,
        )

    async def enroll_unit(
        self,
        assignment_id: UUID,
        eligibility_result: EligibilityResult,
        consent_timestamp: Optional[datetime] = None,
        consent_method: Optional[ConsentMethod] = None,
        consent_version: Optional[str] = None,
    ) -> EnrollmentRecord:
        """
        Enroll a unit that has been assigned to an experiment.

        Args:
            assignment_id: Assignment UUID
            eligibility_result: Result from eligibility check
            consent_timestamp: When consent was obtained
            consent_method: How consent was obtained
            consent_version: Version of consent form

        Returns:
            Enrollment record

        Raises:
            ValueError: If unit is not eligible
        """
        if not eligibility_result.is_eligible:
            raise ValueError(
                f"Cannot enroll ineligible unit. Failed criteria: {eligibility_result.failed_criteria}"
            )

        # Handle consent
        if self.config.require_explicit_consent and consent_timestamp is None:
            consent_timestamp = datetime.now(timezone.utc)
            consent_method = consent_method or ConsentMethod.IMPLIED

        consent_version = consent_version or self.config.default_consent_version

        # Create enrollment via repository
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        enrollment = await repo.create_enrollment(
            assignment_id=assignment_id,
            eligibility_criteria_met=eligibility_result.criteria_results,
            consent_timestamp=consent_timestamp,
            consent_method=consent_method.value if consent_method else None,
            consent_version=consent_version,
        )

        logger.info(f"Enrolled unit with assignment {assignment_id}")

        return enrollment

    async def withdraw_unit(
        self,
        enrollment_id: UUID,
        reason: str,
        initiated_by: WithdrawalInitiator = WithdrawalInitiator.SUBJECT,
    ) -> EnrollmentRecord:
        """
        Withdraw a unit from an experiment.

        Args:
            enrollment_id: Enrollment UUID
            reason: Reason for withdrawal
            initiated_by: Who initiated the withdrawal

        Returns:
            Updated enrollment record
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        enrollment = await repo.update_enrollment_status(
            enrollment_id=enrollment_id,
            status=EnrollmentStatus.WITHDRAWN.value,
            withdrawal_timestamp=datetime.now(timezone.utc),
            withdrawal_reason=reason,
            withdrawal_initiated_by=initiated_by.value,
        )

        logger.info(
            f"Withdrew enrollment {enrollment_id}: {reason} (initiated by {initiated_by.value})"
        )

        return enrollment

    async def mark_completed(
        self,
        enrollment_id: UUID,
    ) -> EnrollmentRecord:
        """
        Mark an enrollment as completed.

        Args:
            enrollment_id: Enrollment UUID

        Returns:
            Updated enrollment record
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        enrollment = await repo.update_enrollment_status(
            enrollment_id=enrollment_id,
            status=EnrollmentStatus.COMPLETED.value,
        )

        logger.info(f"Marked enrollment {enrollment_id} as completed")
        return enrollment

    async def mark_excluded(
        self,
        enrollment_id: UUID,
        reason: str,
    ) -> EnrollmentRecord:
        """
        Mark an enrollment as excluded (protocol violation).

        Args:
            enrollment_id: Enrollment UUID
            reason: Exclusion reason

        Returns:
            Updated enrollment record
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        enrollment = await repo.update_enrollment_status(
            enrollment_id=enrollment_id,
            status=EnrollmentStatus.EXCLUDED.value,
            withdrawal_reason=reason,
            withdrawal_initiated_by=WithdrawalInitiator.INVESTIGATOR.value,
        )

        logger.info(f"Excluded enrollment {enrollment_id}: {reason}")
        return enrollment

    async def mark_lost_to_followup(
        self,
        enrollment_id: UUID,
    ) -> EnrollmentRecord:
        """
        Mark an enrollment as lost to follow-up.

        Args:
            enrollment_id: Enrollment UUID

        Returns:
            Updated enrollment record
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        enrollment = await repo.update_enrollment_status(
            enrollment_id=enrollment_id,
            status=EnrollmentStatus.LOST_TO_FOLLOWUP.value,
        )

        logger.info(f"Marked enrollment {enrollment_id} as lost to follow-up")
        return enrollment

    async def record_protocol_deviation(
        self,
        enrollment_id: UUID,
        deviation: ProtocolDeviation,
    ) -> EnrollmentRecord:
        """
        Record a protocol deviation for an enrollment.

        Args:
            enrollment_id: Enrollment UUID
            deviation: Protocol deviation details

        Returns:
            Updated enrollment record
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()

        # Get current enrollment
        enrollment = await repo.get_enrollment(enrollment_id)
        if not enrollment:
            raise ValueError(f"Enrollment {enrollment_id} not found")

        # Add deviation
        deviations = enrollment.protocol_deviations or []
        deviations.append(
            {
                "date": deviation.date.isoformat(),
                "type": deviation.deviation_type,
                "severity": deviation.severity.value,
                "description": deviation.description,
                "corrective_action": deviation.corrective_action,
            }
        )

        # Update enrollment
        enrollment = await repo.update_protocol_deviations(
            enrollment_id=enrollment_id,
            protocol_deviations=deviations,
        )

        logger.info(
            f"Recorded {deviation.severity.value} protocol deviation for "
            f"enrollment {enrollment_id}: {deviation.deviation_type}"
        )

        # Auto-exclude on major deviation if configured
        if self.config.auto_exclude_on_major_deviation and deviation.severity in (
            DeviationSeverity.MAJOR,
            DeviationSeverity.CRITICAL,
        ):
            enrollment = await self.mark_excluded(
                enrollment_id,
                f"Auto-excluded due to {deviation.severity.value} deviation: {deviation.deviation_type}",
            )

        # Check minor deviation threshold
        minor_count = sum(
            1 for d in deviations if d.get("severity") == DeviationSeverity.MINOR.value
        )
        if minor_count >= self.config.max_minor_deviations:
            logger.warning(
                f"Enrollment {enrollment_id} has {minor_count} minor deviations "
                f"(threshold: {self.config.max_minor_deviations})"
            )

        return enrollment

    async def get_enrollment_stats(
        self,
        experiment_id: UUID,
    ) -> EnrollmentStats:
        """
        Get enrollment statistics for an experiment.

        Args:
            experiment_id: Experiment UUID

        Returns:
            Enrollment statistics
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()

        # Get all assignments and enrollments
        assignments = await repo.get_assignments(experiment_id)
        total_assigned = len(assignments)

        # Initialize counts
        status_counts = dict.fromkeys(EnrollmentStatus, 0)
        by_variant: Dict[str, Dict[str, int]] = {}

        total_enrolled = 0
        for assignment in assignments:
            variant = assignment.variant
            if variant not in by_variant:
                by_variant[variant] = {
                    "assigned": 0,
                    "enrolled": 0,
                    "active": 0,
                    "withdrawn": 0,
                }
            by_variant[variant]["assigned"] += 1

            # Get enrollment for this assignment
            enrollment = await repo.get_enrollment_by_assignment(assignment.id)
            if enrollment:
                total_enrolled += 1
                by_variant[variant]["enrolled"] += 1

                status = EnrollmentStatus(enrollment.enrollment_status)
                status_counts[status] += 1

                if status == EnrollmentStatus.ACTIVE:
                    by_variant[variant]["active"] += 1
                elif status == EnrollmentStatus.WITHDRAWN:
                    by_variant[variant]["withdrawn"] += 1

        # Calculate rates
        enrollment_rate = total_enrolled / total_assigned if total_assigned > 0 else 0.0
        withdrawal_rate = (
            status_counts[EnrollmentStatus.WITHDRAWN] / total_enrolled
            if total_enrolled > 0
            else 0.0
        )

        return EnrollmentStats(
            experiment_id=experiment_id,
            total_assigned=total_assigned,
            total_enrolled=total_enrolled,
            active_count=status_counts[EnrollmentStatus.ACTIVE],
            withdrawn_count=status_counts[EnrollmentStatus.WITHDRAWN],
            excluded_count=status_counts[EnrollmentStatus.EXCLUDED],
            completed_count=status_counts[EnrollmentStatus.COMPLETED],
            lost_to_followup_count=status_counts[EnrollmentStatus.LOST_TO_FOLLOWUP],
            enrollment_rate=enrollment_rate,
            withdrawal_rate=withdrawal_rate,
            by_variant=by_variant,
        )

    async def batch_enroll(
        self,
        assignments: List[Dict[str, Any]],
        eligibility_criteria: Optional[EligibilityCriteria] = None,
        auto_consent: bool = False,
    ) -> Dict[str, Any]:
        """
        Batch enroll multiple units.

        Args:
            assignments: List of assignment dictionaries with 'assignment_id' and unit data
            eligibility_criteria: Optional criteria override
            auto_consent: Whether to auto-apply implied consent

        Returns:
            Summary of enrollment results
        """
        results = {
            "enrolled": 0,
            "ineligible": 0,
            "errors": 0,
            "details": [],
        }

        for assignment_data in assignments:
            assignment_id = assignment_data.get("assignment_id")
            unit_data = assignment_data.get("unit", {})

            try:
                # Check eligibility
                eligibility = await self.check_eligibility(
                    experiment_id=assignment_data.get("experiment_id"),
                    unit=unit_data,
                    criteria=eligibility_criteria,
                )

                if eligibility.is_eligible:
                    consent_timestamp = datetime.now(timezone.utc) if auto_consent else None
                    consent_method = ConsentMethod.IMPLIED if auto_consent else None

                    await self.enroll_unit(
                        assignment_id=UUID(assignment_id),
                        eligibility_result=eligibility,
                        consent_timestamp=consent_timestamp,
                        consent_method=consent_method,
                    )
                    results["enrolled"] += 1
                    results["details"].append(
                        {
                            "assignment_id": assignment_id,
                            "status": "enrolled",
                        }
                    )
                else:
                    results["ineligible"] += 1
                    results["details"].append(
                        {
                            "assignment_id": assignment_id,
                            "status": "ineligible",
                            "reasons": eligibility.failed_criteria,
                        }
                    )

            except Exception as e:
                results["errors"] += 1
                results["details"].append(
                    {
                        "assignment_id": assignment_id,
                        "status": "error",
                        "error": str(e),
                    }
                )
                logger.error(f"Error enrolling assignment {assignment_id}: {e}")

        logger.info(
            f"Batch enrollment complete: {results['enrolled']} enrolled, "
            f"{results['ineligible']} ineligible, {results['errors']} errors"
        )

        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_enrollment_service(
    config: Optional[EnrollmentConfig] = None,
) -> EnrollmentService:
    """Get enrollment service instance."""
    return EnrollmentService(config)
