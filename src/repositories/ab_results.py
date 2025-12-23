"""
A/B Results Repository.

Phase 15: A/B Testing Infrastructure

Data access layer for:
- Experiment results
- SRM checks
- Fidelity comparisons
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
class ExperimentResultRecord:
    """Experiment result database record."""

    id: UUID
    experiment_id: UUID
    analysis_type: str
    analysis_method: str
    computed_at: datetime
    primary_metric: str
    control_mean: float
    treatment_mean: float
    effect_estimate: float
    effect_ci_lower: float
    effect_ci_upper: float
    p_value: float
    sample_size_control: int
    sample_size_treatment: int
    statistical_power: float
    is_significant: bool
    secondary_metrics: List[Dict[str, Any]] = field(default_factory=list)
    segment_results: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class SRMCheckRecord:
    """SRM check database record."""

    id: UUID
    experiment_id: UUID
    checked_at: datetime
    expected_ratio: Dict[str, float]
    actual_counts: Dict[str, int]
    chi_squared_statistic: float
    p_value: float
    is_srm_detected: bool
    severity: str
    investigation_notes: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class FidelityComparisonRecord:
    """Fidelity comparison database record."""

    id: UUID
    experiment_id: UUID
    twin_simulation_id: UUID
    comparison_timestamp: datetime
    predicted_effect: float
    actual_effect: float
    prediction_error: float
    prediction_error_percent: float
    predicted_ci_lower: float
    predicted_ci_upper: float
    ci_coverage: bool
    fidelity_score: float
    fidelity_grade: str
    calibration_adjustment: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


# =============================================================================
# REPOSITORY
# =============================================================================


class ABResultsRepository(BaseRepository):
    """
    Repository for A/B experiment results.

    Handles results, SRM checks, and fidelity comparisons.
    """

    table_name = "ab_experiment_results"
    model_class = ExperimentResultRecord

    def __init__(self, supabase_client=None):
        """Initialize repository with Supabase client."""
        super().__init__(supabase_client)
        self._ensure_client()

    def _ensure_client(self):
        """Ensure we have a Supabase client."""
        if self.client is None:
            try:
                from src.db.supabase_client import get_supabase_client
                self.client = get_supabase_client()
            except ImportError:
                logger.warning("Supabase client not available, running in mock mode")

    # =========================================================================
    # RESULTS OPERATIONS
    # =========================================================================

    async def save_results(
        self,
        results,  # ExperimentResults from results_analysis.py
    ) -> ExperimentResultRecord:
        """
        Save experiment results.

        Args:
            results: ExperimentResults object

        Returns:
            Created result record
        """
        if not self.client:
            return ExperimentResultRecord(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                experiment_id=results.experiment_id,
                analysis_type=results.analysis_type.value,
                analysis_method=results.analysis_method.value,
                computed_at=results.computed_at,
                primary_metric=results.primary_metric,
                control_mean=results.control_mean,
                treatment_mean=results.treatment_mean,
                effect_estimate=results.effect_estimate,
                effect_ci_lower=results.effect_ci_lower,
                effect_ci_upper=results.effect_ci_upper,
                p_value=results.p_value,
                sample_size_control=results.sample_size_control,
                sample_size_treatment=results.sample_size_treatment,
                statistical_power=results.statistical_power,
                is_significant=results.is_significant,
                secondary_metrics=results.secondary_metrics,
                segment_results=results.segment_results,
            )

        data = {
            "experiment_id": str(results.experiment_id),
            "analysis_type": results.analysis_type.value,
            "analysis_method": results.analysis_method.value,
            "primary_metric": results.primary_metric,
            "control_mean": results.control_mean,
            "treatment_mean": results.treatment_mean,
            "effect_estimate": results.effect_estimate,
            "effect_ci_lower": results.effect_ci_lower,
            "effect_ci_upper": results.effect_ci_upper,
            "p_value": results.p_value,
            "sample_size_control": results.sample_size_control,
            "sample_size_treatment": results.sample_size_treatment,
            "statistical_power": results.statistical_power,
            "is_significant": results.is_significant,
            "secondary_metrics": results.secondary_metrics,
            "segment_results": results.segment_results,
        }

        result = self.client.table(self.table_name).insert(data).execute()

        if result.data:
            return self._to_result_record(result.data[0])

        raise RuntimeError("Failed to save results")

    async def get_results(
        self,
        experiment_id: UUID,
        analysis_type: Optional[str] = None,
        analysis_method: Optional[str] = None,
    ) -> List[ExperimentResultRecord]:
        """
        Get results for an experiment.

        Args:
            experiment_id: Experiment UUID
            analysis_type: Optional type filter
            analysis_method: Optional method filter

        Returns:
            List of result records
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("experiment_id", str(experiment_id))
        )

        if analysis_type:
            query = query.eq("analysis_type", analysis_type)
        if analysis_method:
            query = query.eq("analysis_method", analysis_method)

        query = query.order("computed_at", desc=True)
        result = query.execute()

        return [self._to_result_record(row) for row in result.data]

    async def get_latest_results(
        self,
        experiment_id: UUID,
        analysis_method: str = "itt",
    ) -> Optional[ExperimentResultRecord]:
        """
        Get most recent results for an experiment.

        Args:
            experiment_id: Experiment UUID
            analysis_method: Analysis method

        Returns:
            Latest result record or None
        """
        if not self.client:
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .eq("analysis_method", analysis_method)
            .order("computed_at", desc=True)
            .limit(1)
            .execute()
        )

        return self._to_result_record(result.data[0]) if result.data else None

    def _to_result_record(self, data: Dict[str, Any]) -> ExperimentResultRecord:
        """Convert database row to ExperimentResultRecord."""
        return ExperimentResultRecord(
            id=UUID(data["id"]),
            experiment_id=UUID(data["experiment_id"]),
            analysis_type=data["analysis_type"],
            analysis_method=data["analysis_method"],
            computed_at=datetime.fromisoformat(data["computed_at"].replace("Z", "+00:00")),
            primary_metric=data["primary_metric"],
            control_mean=data["control_mean"],
            treatment_mean=data["treatment_mean"],
            effect_estimate=data["effect_estimate"],
            effect_ci_lower=data["effect_ci_lower"],
            effect_ci_upper=data["effect_ci_upper"],
            p_value=data["p_value"],
            sample_size_control=data["sample_size_control"],
            sample_size_treatment=data["sample_size_treatment"],
            statistical_power=data["statistical_power"],
            is_significant=data["is_significant"],
            secondary_metrics=data.get("secondary_metrics", []),
            segment_results=data.get("segment_results"),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )

    # =========================================================================
    # SRM CHECK OPERATIONS
    # =========================================================================

    async def save_srm_check(
        self,
        check,  # SRMCheckResult from results_analysis.py
    ) -> SRMCheckRecord:
        """
        Save SRM check result.

        Args:
            check: SRMCheckResult object

        Returns:
            Created SRM check record
        """
        if not self.client:
            return SRMCheckRecord(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                experiment_id=check.experiment_id,
                checked_at=check.checked_at,
                expected_ratio=check.expected_ratio,
                actual_counts=check.actual_counts,
                chi_squared_statistic=check.chi_squared_statistic,
                p_value=check.p_value,
                is_srm_detected=check.is_srm_detected,
                severity=check.severity.value,
                investigation_notes=check.investigation_notes,
            )

        data = {
            "experiment_id": str(check.experiment_id),
            "expected_ratio": check.expected_ratio,
            "actual_counts": check.actual_counts,
            "chi_squared_statistic": check.chi_squared_statistic,
            "p_value": check.p_value,
            "is_srm_detected": check.is_srm_detected,
            "severity": check.severity.value,
            "investigation_notes": check.investigation_notes,
        }

        result = self.client.table("ab_srm_checks").insert(data).execute()

        if result.data:
            return self._to_srm_record(result.data[0])

        raise RuntimeError("Failed to save SRM check")

    async def get_srm_history(
        self,
        experiment_id: UUID,
        limit: int = 100,
    ) -> List[SRMCheckRecord]:
        """
        Get SRM check history for an experiment.

        Args:
            experiment_id: Experiment UUID
            limit: Maximum records

        Returns:
            List of SRM check records
        """
        if not self.client:
            return []

        result = (
            self.client.table("ab_srm_checks")
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .order("checked_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._to_srm_record(row) for row in result.data]

    async def get_latest_srm_check(
        self,
        experiment_id: UUID,
    ) -> Optional[SRMCheckRecord]:
        """Get most recent SRM check."""
        if not self.client:
            return None

        result = (
            self.client.table("ab_srm_checks")
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .order("checked_at", desc=True)
            .limit(1)
            .execute()
        )

        return self._to_srm_record(result.data[0]) if result.data else None

    def _to_srm_record(self, data: Dict[str, Any]) -> SRMCheckRecord:
        """Convert database row to SRMCheckRecord."""
        return SRMCheckRecord(
            id=UUID(data["id"]),
            experiment_id=UUID(data["experiment_id"]),
            checked_at=datetime.fromisoformat(data["checked_at"].replace("Z", "+00:00")),
            expected_ratio=data["expected_ratio"],
            actual_counts=data["actual_counts"],
            chi_squared_statistic=data["chi_squared_statistic"],
            p_value=data["p_value"],
            is_srm_detected=data["is_srm_detected"],
            severity=data["severity"],
            investigation_notes=data.get("investigation_notes"),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )

    # =========================================================================
    # FIDELITY COMPARISON OPERATIONS
    # =========================================================================

    async def save_fidelity_comparison(
        self,
        comparison,  # FidelityComparison from results_analysis.py
    ) -> FidelityComparisonRecord:
        """
        Save fidelity comparison result.

        Args:
            comparison: FidelityComparison object

        Returns:
            Created fidelity comparison record
        """
        if not self.client:
            return FidelityComparisonRecord(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                experiment_id=comparison.experiment_id,
                twin_simulation_id=comparison.twin_simulation_id,
                comparison_timestamp=comparison.comparison_timestamp,
                predicted_effect=comparison.predicted_effect,
                actual_effect=comparison.actual_effect,
                prediction_error=comparison.prediction_error,
                prediction_error_percent=comparison.prediction_error_percent,
                predicted_ci_lower=comparison.predicted_ci_lower,
                predicted_ci_upper=comparison.predicted_ci_upper,
                ci_coverage=comparison.ci_coverage,
                fidelity_score=comparison.fidelity_score,
                fidelity_grade=comparison.fidelity_grade,
                calibration_adjustment=comparison.calibration_adjustment,
            )

        data = {
            "experiment_id": str(comparison.experiment_id),
            "twin_simulation_id": str(comparison.twin_simulation_id),
            "predicted_effect": comparison.predicted_effect,
            "actual_effect": comparison.actual_effect,
            "prediction_error": comparison.prediction_error,
            "prediction_error_percent": comparison.prediction_error_percent,
            "predicted_ci_lower": comparison.predicted_ci_lower,
            "predicted_ci_upper": comparison.predicted_ci_upper,
            "ci_coverage": comparison.ci_coverage,
            "fidelity_score": comparison.fidelity_score,
            "fidelity_grade": comparison.fidelity_grade,
            "calibration_adjustment": comparison.calibration_adjustment,
        }

        result = self.client.table("ab_fidelity_comparisons").insert(data).execute()

        if result.data:
            return self._to_fidelity_record(result.data[0])

        raise RuntimeError("Failed to save fidelity comparison")

    async def get_fidelity_comparisons(
        self,
        experiment_id: UUID,
    ) -> List[FidelityComparisonRecord]:
        """
        Get fidelity comparisons for an experiment.

        Args:
            experiment_id: Experiment UUID

        Returns:
            List of fidelity comparison records
        """
        if not self.client:
            return []

        result = (
            self.client.table("ab_fidelity_comparisons")
            .select("*")
            .eq("experiment_id", str(experiment_id))
            .order("comparison_timestamp", desc=True)
            .execute()
        )

        return [self._to_fidelity_record(row) for row in result.data]

    async def get_fidelity_by_twin(
        self,
        twin_simulation_id: UUID,
    ) -> List[FidelityComparisonRecord]:
        """
        Get fidelity comparisons for a specific Digital Twin simulation.

        Args:
            twin_simulation_id: Twin simulation UUID

        Returns:
            List of fidelity comparison records
        """
        if not self.client:
            return []

        result = (
            self.client.table("ab_fidelity_comparisons")
            .select("*")
            .eq("twin_simulation_id", str(twin_simulation_id))
            .order("comparison_timestamp", desc=True)
            .execute()
        )

        return [self._to_fidelity_record(row) for row in result.data]

    async def get_average_fidelity_score(
        self,
        experiment_id: Optional[UUID] = None,
    ) -> float:
        """
        Get average fidelity score.

        Args:
            experiment_id: Optional experiment filter

        Returns:
            Average fidelity score (0-1)
        """
        if not self.client:
            return 0.0

        query = self.client.table("ab_fidelity_comparisons").select("fidelity_score")

        if experiment_id:
            query = query.eq("experiment_id", str(experiment_id))

        result = query.execute()

        if result.data:
            scores = [row["fidelity_score"] for row in result.data]
            return sum(scores) / len(scores)

        return 0.0

    def _to_fidelity_record(self, data: Dict[str, Any]) -> FidelityComparisonRecord:
        """Convert database row to FidelityComparisonRecord."""
        return FidelityComparisonRecord(
            id=UUID(data["id"]),
            experiment_id=UUID(data["experiment_id"]),
            twin_simulation_id=UUID(data["twin_simulation_id"]),
            comparison_timestamp=datetime.fromisoformat(data["comparison_timestamp"].replace("Z", "+00:00")),
            predicted_effect=data["predicted_effect"],
            actual_effect=data["actual_effect"],
            prediction_error=data["prediction_error"],
            prediction_error_percent=data.get("prediction_error_percent", 0.0),
            predicted_ci_lower=data["predicted_ci_lower"],
            predicted_ci_upper=data["predicted_ci_upper"],
            ci_coverage=data["ci_coverage"],
            fidelity_score=data["fidelity_score"],
            fidelity_grade=data["fidelity_grade"],
            calibration_adjustment=data.get("calibration_adjustment", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                if data.get("created_at")
                else None
            ),
        )

    # =========================================================================
    # SUMMARY OPERATIONS
    # =========================================================================

    async def get_experiment_summary(
        self,
        experiment_id: UUID,
    ) -> Dict[str, Any]:
        """
        Get comprehensive summary for an experiment.

        Args:
            experiment_id: Experiment UUID

        Returns:
            Summary dictionary
        """
        # Get latest results
        latest_results = await self.get_latest_results(experiment_id)

        # Get SRM status
        latest_srm = await self.get_latest_srm_check(experiment_id)

        # Get fidelity comparisons
        fidelity_records = await self.get_fidelity_comparisons(experiment_id)

        return {
            "experiment_id": str(experiment_id),
            "has_results": latest_results is not None,
            "latest_results": {
                "effect_estimate": latest_results.effect_estimate if latest_results else None,
                "p_value": latest_results.p_value if latest_results else None,
                "is_significant": latest_results.is_significant if latest_results else None,
                "computed_at": latest_results.computed_at.isoformat() if latest_results else None,
            } if latest_results else None,
            "srm_status": {
                "is_srm_detected": latest_srm.is_srm_detected if latest_srm else False,
                "severity": latest_srm.severity if latest_srm else None,
                "checked_at": latest_srm.checked_at.isoformat() if latest_srm else None,
            } if latest_srm else None,
            "fidelity_summary": {
                "num_comparisons": len(fidelity_records),
                "average_score": (
                    sum(r.fidelity_score for r in fidelity_records) / len(fidelity_records)
                    if fidelity_records else None
                ),
                "latest_grade": fidelity_records[0].fidelity_grade if fidelity_records else None,
            } if fidelity_records else None,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_ab_results_repository(
    supabase_client=None,
) -> ABResultsRepository:
    """Get A/B results repository instance."""
    return ABResultsRepository(supabase_client)
