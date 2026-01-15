"""
A/B Experiment Repository.

Phase 15: A/B Testing Infrastructure

Data access layer for:
- Experiment assignments
- Enrollment records
- Interim analyses
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Assignment:
    """Experiment assignment record."""

    id: UUID
    experiment_id: UUID
    unit_id: str
    unit_type: str
    variant: str
    assigned_at: datetime
    randomization_method: str
    stratification_key: Optional[Dict[str, Any]] = None
    block_id: Optional[str] = None
    assignment_hash: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class Enrollment:
    """Enrollment record."""

    id: UUID
    assignment_id: UUID
    enrolled_at: datetime
    enrollment_status: str
    eligibility_criteria_met: Dict[str, bool] = field(default_factory=dict)
    eligibility_check_timestamp: Optional[datetime] = None
    consent_timestamp: Optional[datetime] = None
    consent_method: Optional[str] = None
    consent_version: Optional[str] = None
    withdrawal_timestamp: Optional[datetime] = None
    withdrawal_reason: Optional[str] = None
    withdrawal_initiated_by: Optional[str] = None
    protocol_deviations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class InterimAnalysis:
    """Interim analysis record."""

    id: UUID
    experiment_id: UUID
    analysis_number: int
    performed_at: datetime
    analysis_type: str
    information_fraction: float
    sample_size_at_analysis: int
    target_sample_size: Optional[int] = None
    spending_function: str = "obrien_fleming"
    alpha_spent: float = 0.0
    cumulative_alpha_spent: float = 0.0
    adjusted_alpha: float = 0.05
    test_statistic: Optional[float] = None
    standard_error: Optional[float] = None
    p_value: Optional[float] = None
    effect_estimate: Optional[float] = None
    effect_ci_lower: Optional[float] = None
    effect_ci_upper: Optional[float] = None
    conditional_power: Optional[float] = None
    predictive_probability: Optional[float] = None
    decision: str = "continue"
    decision_rationale: Optional[str] = None
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    performed_by: Optional[str] = None
    approved_by: Optional[str] = None
    created_at: Optional[datetime] = None


# =============================================================================
# REPOSITORY
# =============================================================================


class ABExperimentRepository(BaseRepository):
    """
    Repository for A/B experiment data access.

    Handles assignments, enrollments, and interim analyses.
    """

    table_name = "ab_experiment_assignments"
    model_class = Assignment

    def __init__(self, supabase_client=None):
        """Initialize repository with Supabase client."""
        super().__init__(supabase_client)
        self._ensure_client()

    def _ensure_client(self):
        """Ensure we have a Supabase client."""
        if self.client is None:
            try:
                from src.repositories import get_supabase_client
                self.client = get_supabase_client()
            except ImportError:
                logger.warning("Supabase client not available, running in mock mode")

    # =========================================================================
    # ASSIGNMENT OPERATIONS
    # =========================================================================

    async def create_assignment(
        self,
        experiment_id: UUID,
        unit_id: str,
        unit_type: str,
        variant: str,
        randomization_method: str,
        stratification_key: Optional[Dict[str, Any]] = None,
        block_id: Optional[str] = None,
        assignment_hash: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Assignment:
        """
        Create a new experiment assignment.

        Args:
            experiment_id: Experiment UUID
            unit_id: Unit identifier
            unit_type: Type of unit (hcp, patient, territory, account)
            variant: Assigned variant
            randomization_method: Method used for randomization
            stratification_key: Optional stratification variables
            block_id: Optional block identifier
            assignment_hash: Optional deterministic hash
            created_by: User who created the assignment

        Returns:
            Created assignment record
        """
        if not self.client:
            # Return mock assignment for testing
            return Assignment(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                experiment_id=experiment_id,
                unit_id=unit_id,
                unit_type=unit_type,
                variant=variant,
                assigned_at=datetime.now(timezone.utc),
                randomization_method=randomization_method,
                stratification_key=stratification_key,
                block_id=block_id,
                assignment_hash=assignment_hash,
                created_by=created_by,
            )

        data = {
            "experiment_id": str(experiment_id),
            "unit_id": unit_id,
            "unit_type": unit_type,
            "variant": variant,
            "randomization_method": randomization_method,
            "stratification_key": stratification_key or {},
            "block_id": block_id,
            "assignment_hash": assignment_hash,
            "created_by": created_by,
        }

        result = self.client.table(self.table_name).insert(data).execute()

        if result.data:
            return self._to_assignment(result.data[0])

        raise RuntimeError("Failed to create assignment")

    async def create_assignments_batch(
        self,
        assignments: List[Dict[str, Any]],
    ) -> List[Assignment]:
        """
        Create multiple assignments in batch.

        Args:
            assignments: List of assignment dictionaries

        Returns:
            List of created assignments
        """
        if not self.client:
            return []

        # Prepare data
        data = []
        for a in assignments:
            data.append({
                "experiment_id": str(a["experiment_id"]),
                "unit_id": a["unit_id"],
                "unit_type": a["unit_type"],
                "variant": a["variant"],
                "randomization_method": a["randomization_method"],
                "stratification_key": a.get("stratification_key", {}),
                "block_id": a.get("block_id"),
                "assignment_hash": a.get("assignment_hash"),
                "created_by": a.get("created_by"),
            })

        result = self.client.table(self.table_name).insert(data).execute()

        return [self._to_assignment(row) for row in result.data]

    async def get_assignment(
        self,
        assignment_id: UUID,
    ) -> Optional[Assignment]:
        """Get assignment by ID."""
        if not self.client:
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", str(assignment_id))
            .execute()
        )

        return self._to_assignment(result.data[0]) if result.data else None

    async def get_assignments(
        self,
        experiment_id: UUID,
        variant: Optional[str] = None,
        unit_type: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[Assignment]:
        """
        Get assignments for an experiment.

        Args:
            experiment_id: Experiment UUID
            variant: Optional variant filter
            unit_type: Optional unit type filter
            limit: Maximum records to return
            offset: Pagination offset

        Returns:
            List of assignments
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("experiment_id", str(experiment_id))
        )

        if variant:
            query = query.eq("variant", variant)
        if unit_type:
            query = query.eq("unit_type", unit_type)

        query = query.limit(limit).offset(offset)
        result = query.execute()

        return [self._to_assignment(row) for row in result.data]

    async def get_assignment_by_unit(
        self,
        experiment_id: UUID,
        unit_id: str,
    ) -> Optional[Assignment]:
        """
        Get assignment for a specific unit in an experiment.

        Args:
            experiment_id: Experiment UUID
            unit_id: Unit identifier

        Returns:
            Assignment if found, None otherwise
        """
        if not self.client:
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .eq("unit_id", unit_id)
            .execute()
        )

        return self._to_assignment(result.data[0]) if result.data else None

    async def get_assignment_counts(
        self,
        experiment_id: UUID,
    ) -> Dict[str, int]:
        """
        Get assignment counts by variant.

        Args:
            experiment_id: Experiment UUID

        Returns:
            Dictionary of variant -> count
        """
        if not self.client:
            return {}

        # Use SQL function if available, otherwise count manually
        result = (
            self.client.table(self.table_name)
            .select("variant")
            .eq("experiment_id", str(experiment_id))
            .execute()
        )

        counts: Dict[str, int] = {}
        for row in result.data:
            variant = row["variant"]
            counts[variant] = counts.get(variant, 0) + 1

        return counts

    def _to_assignment(self, data: Dict[str, Any]) -> Assignment:
        """Convert database row to Assignment."""
        return Assignment(
            id=UUID(data["id"]),
            experiment_id=UUID(data["experiment_id"]),
            unit_id=data["unit_id"],
            unit_type=data["unit_type"],
            variant=data["variant"],
            assigned_at=datetime.fromisoformat(data["assigned_at"].replace("Z", "+00:00")),
            randomization_method=data["randomization_method"],
            stratification_key=data.get("stratification_key"),
            block_id=data.get("block_id"),
            assignment_hash=data.get("assignment_hash"),
            created_by=data.get("created_by"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else None,
        )

    # =========================================================================
    # ENROLLMENT OPERATIONS
    # =========================================================================

    async def create_enrollment(
        self,
        assignment_id: UUID,
        eligibility_criteria_met: Dict[str, bool],
        consent_timestamp: Optional[datetime] = None,
        consent_method: Optional[str] = None,
        consent_version: Optional[str] = None,
    ) -> Enrollment:
        """
        Create an enrollment record.

        Args:
            assignment_id: Assignment UUID
            eligibility_criteria_met: Criteria check results
            consent_timestamp: When consent was obtained
            consent_method: How consent was obtained
            consent_version: Version of consent form

        Returns:
            Created enrollment record
        """
        if not self.client:
            return Enrollment(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                assignment_id=assignment_id,
                enrolled_at=datetime.now(timezone.utc),
                enrollment_status="active",
                eligibility_criteria_met=eligibility_criteria_met,
                consent_timestamp=consent_timestamp,
                consent_method=consent_method,
                consent_version=consent_version,
            )

        data = {
            "assignment_id": str(assignment_id),
            "eligibility_criteria_met": eligibility_criteria_met,
            "eligibility_check_timestamp": datetime.now(timezone.utc).isoformat(),
            "consent_timestamp": consent_timestamp.isoformat() if consent_timestamp else None,
            "consent_method": consent_method,
            "consent_version": consent_version,
        }

        result = (
            self.client.table("ab_experiment_enrollments")
            .insert(data)
            .execute()
        )

        if result.data:
            return self._to_enrollment(result.data[0])

        raise RuntimeError("Failed to create enrollment")

    async def get_enrollment(
        self,
        enrollment_id: UUID,
    ) -> Optional[Enrollment]:
        """Get enrollment by ID."""
        if not self.client:
            return None

        result = (
            self.client.table("ab_experiment_enrollments")
            .select("*")
            .eq("id", str(enrollment_id))
            .execute()
        )

        return self._to_enrollment(result.data[0]) if result.data else None

    async def get_enrollment_by_assignment(
        self,
        assignment_id: UUID,
    ) -> Optional[Enrollment]:
        """Get enrollment for an assignment."""
        if not self.client:
            return None

        result = (
            self.client.table("ab_experiment_enrollments")
            .select("*")
            .eq("assignment_id", str(assignment_id))
            .execute()
        )

        return self._to_enrollment(result.data[0]) if result.data else None

    async def update_enrollment_status(
        self,
        enrollment_id: UUID,
        status: str,
        withdrawal_timestamp: Optional[datetime] = None,
        withdrawal_reason: Optional[str] = None,
        withdrawal_initiated_by: Optional[str] = None,
    ) -> Enrollment:
        """
        Update enrollment status.

        Args:
            enrollment_id: Enrollment UUID
            status: New status
            withdrawal_timestamp: When withdrawn (if applicable)
            withdrawal_reason: Reason for withdrawal
            withdrawal_initiated_by: Who initiated withdrawal

        Returns:
            Updated enrollment
        """
        if not self.client:
            return Enrollment(
                id=enrollment_id,
                assignment_id=UUID("00000000-0000-0000-0000-000000000000"),
                enrolled_at=datetime.now(timezone.utc),
                enrollment_status=status,
            )

        updates = {
            "enrollment_status": status,
        }

        if withdrawal_timestamp:
            updates["withdrawal_timestamp"] = withdrawal_timestamp.isoformat()
        if withdrawal_reason:
            updates["withdrawal_reason"] = withdrawal_reason
        if withdrawal_initiated_by:
            updates["withdrawal_initiated_by"] = withdrawal_initiated_by

        result = (
            self.client.table("ab_experiment_enrollments")
            .update(updates)
            .eq("id", str(enrollment_id))
            .execute()
        )

        if result.data:
            return self._to_enrollment(result.data[0])

        raise RuntimeError(f"Failed to update enrollment {enrollment_id}")

    async def update_protocol_deviations(
        self,
        enrollment_id: UUID,
        protocol_deviations: List[Dict[str, Any]],
    ) -> Enrollment:
        """
        Update protocol deviations for an enrollment.

        Args:
            enrollment_id: Enrollment UUID
            protocol_deviations: Updated deviations list

        Returns:
            Updated enrollment
        """
        if not self.client:
            return Enrollment(
                id=enrollment_id,
                assignment_id=UUID("00000000-0000-0000-0000-000000000000"),
                enrolled_at=datetime.now(timezone.utc),
                enrollment_status="active",
                protocol_deviations=protocol_deviations,
            )

        result = (
            self.client.table("ab_experiment_enrollments")
            .update({"protocol_deviations": protocol_deviations})
            .eq("id", str(enrollment_id))
            .execute()
        )

        if result.data:
            return self._to_enrollment(result.data[0])

        raise RuntimeError(f"Failed to update protocol deviations for {enrollment_id}")

    async def get_enrollments_by_experiment(
        self,
        experiment_id: UUID,
        status: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Enrollment]:
        """
        Get enrollments for an experiment.

        Args:
            experiment_id: Experiment UUID
            status: Optional status filter
            limit: Maximum records

        Returns:
            List of enrollments
        """
        if not self.client:
            return []

        # Join with assignments to filter by experiment
        query = (
            self.client.table("ab_experiment_enrollments")
            .select("*, ab_experiment_assignments!inner(experiment_id)")
            .eq("ab_experiment_assignments.experiment_id", str(experiment_id))
        )

        if status:
            query = query.eq("enrollment_status", status)

        query = query.limit(limit)
        result = query.execute()

        return [self._to_enrollment(row) for row in result.data]

    def _to_enrollment(self, data: Dict[str, Any]) -> Enrollment:
        """Convert database row to Enrollment."""
        return Enrollment(
            id=UUID(data["id"]),
            assignment_id=UUID(data["assignment_id"]),
            enrolled_at=datetime.fromisoformat(data["enrolled_at"].replace("Z", "+00:00")),
            enrollment_status=data["enrollment_status"],
            eligibility_criteria_met=data.get("eligibility_criteria_met", {}),
            eligibility_check_timestamp=(
                datetime.fromisoformat(data["eligibility_check_timestamp"].replace("Z", "+00:00"))
                if data.get("eligibility_check_timestamp")
                else None
            ),
            consent_timestamp=(
                datetime.fromisoformat(data["consent_timestamp"].replace("Z", "+00:00"))
                if data.get("consent_timestamp")
                else None
            ),
            consent_method=data.get("consent_method"),
            consent_version=data.get("consent_version"),
            withdrawal_timestamp=(
                datetime.fromisoformat(data["withdrawal_timestamp"].replace("Z", "+00:00"))
                if data.get("withdrawal_timestamp")
                else None
            ),
            withdrawal_reason=data.get("withdrawal_reason"),
            withdrawal_initiated_by=data.get("withdrawal_initiated_by"),
            protocol_deviations=data.get("protocol_deviations", []),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
                if data.get("updated_at")
                else None
            ),
        )

    # =========================================================================
    # INTERIM ANALYSIS OPERATIONS
    # =========================================================================

    async def record_interim_analysis(
        self,
        experiment_id: UUID,
        analysis_number: int,
        information_fraction: float,
        sample_size_at_analysis: int,
        target_sample_size: Optional[int] = None,
        spending_function: str = "obrien_fleming",
        alpha_spent: float = 0.0,
        cumulative_alpha_spent: float = 0.0,
        adjusted_alpha: float = 0.05,
        test_statistic: Optional[float] = None,
        standard_error: Optional[float] = None,
        p_value: Optional[float] = None,
        effect_estimate: Optional[float] = None,
        effect_ci_lower: Optional[float] = None,
        effect_ci_upper: Optional[float] = None,
        conditional_power: Optional[float] = None,
        predictive_probability: Optional[float] = None,
        decision: str = "continue",
        decision_rationale: Optional[str] = None,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
        analysis_type: str = "scheduled",
        performed_by: Optional[str] = None,
    ) -> InterimAnalysis:
        """
        Record an interim analysis.

        Args:
            experiment_id: Experiment UUID
            analysis_number: Analysis sequence number
            information_fraction: Proportion of data collected
            sample_size_at_analysis: Current sample size
            target_sample_size: Target sample size
            spending_function: Alpha spending function used
            alpha_spent: Alpha spent in this analysis
            cumulative_alpha_spent: Total alpha spent
            adjusted_alpha: Adjusted significance threshold
            test_statistic: Test statistic value
            standard_error: Standard error of estimate
            p_value: P-value
            effect_estimate: Effect size estimate
            effect_ci_lower: CI lower bound
            effect_ci_upper: CI upper bound
            conditional_power: Conditional power
            predictive_probability: Predictive probability of success
            decision: Decision (continue, stop_efficacy, stop_futility, etc.)
            decision_rationale: Explanation of decision
            metrics_snapshot: All metrics at this point
            analysis_type: Type of analysis
            performed_by: Who performed the analysis

        Returns:
            Recorded interim analysis
        """
        if not self.client:
            return InterimAnalysis(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                experiment_id=experiment_id,
                analysis_number=analysis_number,
                performed_at=datetime.now(timezone.utc),
                analysis_type=analysis_type,
                information_fraction=information_fraction,
                sample_size_at_analysis=sample_size_at_analysis,
                decision=decision,
            )

        data = {
            "experiment_id": str(experiment_id),
            "analysis_number": analysis_number,
            "analysis_type": analysis_type,
            "information_fraction": information_fraction,
            "sample_size_at_analysis": sample_size_at_analysis,
            "target_sample_size": target_sample_size,
            "spending_function": spending_function,
            "alpha_spent": alpha_spent,
            "cumulative_alpha_spent": cumulative_alpha_spent,
            "adjusted_alpha": adjusted_alpha,
            "test_statistic": test_statistic,
            "standard_error": standard_error,
            "p_value": p_value,
            "effect_estimate": effect_estimate,
            "effect_ci_lower": effect_ci_lower,
            "effect_ci_upper": effect_ci_upper,
            "conditional_power": conditional_power,
            "predictive_probability": predictive_probability,
            "decision": decision,
            "decision_rationale": decision_rationale,
            "metrics_snapshot": metrics_snapshot or {},
            "performed_by": performed_by,
        }

        result = (
            self.client.table("ab_interim_analyses")
            .insert(data)
            .execute()
        )

        if result.data:
            return self._to_interim_analysis(result.data[0])

        raise RuntimeError("Failed to record interim analysis")

    async def get_interim_analyses(
        self,
        experiment_id: UUID,
    ) -> List[InterimAnalysis]:
        """
        Get all interim analyses for an experiment.

        Args:
            experiment_id: Experiment UUID

        Returns:
            List of interim analyses ordered by analysis number
        """
        if not self.client:
            return []

        result = (
            self.client.table("ab_interim_analyses")
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .order("analysis_number")
            .execute()
        )

        return [self._to_interim_analysis(row) for row in result.data]

    async def get_latest_interim_analysis(
        self,
        experiment_id: UUID,
    ) -> Optional[InterimAnalysis]:
        """
        Get the most recent interim analysis.

        Args:
            experiment_id: Experiment UUID

        Returns:
            Latest interim analysis or None
        """
        if not self.client:
            return None

        result = (
            self.client.table("ab_interim_analyses")
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .order("analysis_number", desc=True)
            .limit(1)
            .execute()
        )

        return self._to_interim_analysis(result.data[0]) if result.data else None

    def _to_interim_analysis(self, data: Dict[str, Any]) -> InterimAnalysis:
        """Convert database row to InterimAnalysis."""
        return InterimAnalysis(
            id=UUID(data["id"]),
            experiment_id=UUID(data["experiment_id"]),
            analysis_number=data["analysis_number"],
            performed_at=datetime.fromisoformat(data["performed_at"].replace("Z", "+00:00")),
            analysis_type=data["analysis_type"],
            information_fraction=data["information_fraction"],
            sample_size_at_analysis=data["sample_size_at_analysis"],
            target_sample_size=data.get("target_sample_size"),
            spending_function=data.get("spending_function", "obrien_fleming"),
            alpha_spent=data.get("alpha_spent", 0.0),
            cumulative_alpha_spent=data.get("cumulative_alpha_spent", 0.0),
            adjusted_alpha=data.get("adjusted_alpha", 0.05),
            test_statistic=data.get("test_statistic"),
            standard_error=data.get("standard_error"),
            p_value=data.get("p_value"),
            effect_estimate=data.get("effect_estimate"),
            effect_ci_lower=data.get("effect_ci_lower"),
            effect_ci_upper=data.get("effect_ci_upper"),
            conditional_power=data.get("conditional_power"),
            predictive_probability=data.get("predictive_probability"),
            decision=data.get("decision", "continue"),
            decision_rationale=data.get("decision_rationale"),
            metrics_snapshot=data.get("metrics_snapshot", {}),
            performed_by=data.get("performed_by"),
            approved_by=data.get("approved_by"),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_ab_experiment_repository(
    supabase_client=None,
) -> ABExperimentRepository:
    """Get A/B experiment repository instance."""
    return ABExperimentRepository(supabase_client)
