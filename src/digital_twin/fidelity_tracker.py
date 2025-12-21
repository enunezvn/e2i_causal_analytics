"""
Fidelity Tracker
================

Tracks and validates digital twin prediction accuracy against
real-world experiment outcomes. This addresses the "Validation Paradox"
by systematically recording how well simulations predict actual results.

Key Functions:
    - Record simulation predictions
    - Link to real experiment outcomes
    - Calculate fidelity metrics
    - Detect model degradation
    - Generate retraining alerts
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from .models.simulation_models import (
    FidelityGrade,
    FidelityRecord,
    SimulationResult,
)

logger = logging.getLogger(__name__)


class FidelityTracker:
    """
    Tracks digital twin prediction fidelity against real outcomes.

    The fidelity tracker maintains a record of simulation predictions
    and compares them against actual experiment results to:
    1. Assess model accuracy over time
    2. Detect degradation requiring retraining
    3. Calibrate confidence in future predictions

    Attributes:
        records: List of fidelity tracking records
        model_fidelity_cache: Cached model-level fidelity scores

    Example:
        >>> tracker = FidelityTracker()
        >>> tracker.record_prediction(simulation_result)
        >>> # After real experiment completes:
        >>> tracker.validate(simulation_id, actual_ate=0.08, actual_ci=(0.05, 0.11))
        >>> report = tracker.get_model_fidelity_report(model_id)
    """

    # Thresholds for fidelity grades
    GRADE_THRESHOLDS = {
        FidelityGrade.EXCELLENT: 0.10,  # < 10% error
        FidelityGrade.GOOD: 0.20,  # < 20% error
        FidelityGrade.FAIR: 0.35,  # < 35% error
        FidelityGrade.POOR: float("inf"),  # >= 35% error
    }

    # Degradation detection thresholds
    DEGRADATION_LOOKBACK_DAYS = 90
    DEGRADATION_THRESHOLD = 0.10  # 10% increase in error
    MIN_VALIDATIONS_FOR_ALERT = 5

    def __init__(self, repository=None):
        """
        Initialize fidelity tracker.

        Args:
            repository: Optional TwinRepository for persistence
        """
        self.repository = repository
        self.records: Dict[UUID, FidelityRecord] = {}
        self.model_fidelity_cache: Dict[UUID, Dict[str, Any]] = {}

        logger.info("Initialized FidelityTracker")

    def record_prediction(
        self,
        simulation_result: SimulationResult,
    ) -> FidelityRecord:
        """
        Record a simulation prediction for future validation.

        Args:
            simulation_result: Completed simulation result

        Returns:
            FidelityRecord tracking this prediction
        """
        record = FidelityRecord(
            simulation_id=simulation_result.simulation_id,
            simulated_ate=simulation_result.simulated_ate,
            simulated_ci_lower=simulation_result.simulated_ci_lower,
            simulated_ci_upper=simulation_result.simulated_ci_upper,
        )

        self.records[record.tracking_id] = record

        logger.info(
            f"Recorded prediction for simulation {simulation_result.simulation_id}: "
            f"ATE={simulation_result.simulated_ate:.4f}"
        )

        if self.repository:
            self.repository.save_fidelity_record(record)

        return record

    def validate(
        self,
        simulation_id: UUID,
        actual_ate: float,
        actual_ci: Optional[Tuple[float, float]] = None,
        actual_sample_size: Optional[int] = None,
        actual_experiment_id: Optional[UUID] = None,
        notes: Optional[str] = None,
        confounding_factors: Optional[List[str]] = None,
        validated_by: Optional[str] = None,
    ) -> FidelityRecord:
        """
        Validate a simulation prediction against actual results.

        Args:
            simulation_id: ID of the simulation to validate
            actual_ate: Actual Average Treatment Effect from experiment
            actual_ci: Optional (lower, upper) confidence interval
            actual_sample_size: Sample size of actual experiment
            actual_experiment_id: Optional ID of the real experiment
            notes: Optional validation notes
            confounding_factors: Optional list of confounding factors
            validated_by: Optional validator identifier

        Returns:
            Updated FidelityRecord with validation metrics

        Raises:
            ValueError: If simulation_id not found
        """
        # Find the record
        record = self._find_record_by_simulation(simulation_id)
        if record is None:
            raise ValueError(f"No fidelity record found for simulation {simulation_id}")

        # Update with actuals
        record.actual_ate = actual_ate
        if actual_ci:
            record.actual_ci_lower = actual_ci[0]
            record.actual_ci_upper = actual_ci[1]
        record.actual_sample_size = actual_sample_size
        record.actual_experiment_id = actual_experiment_id
        record.validation_notes = notes
        record.confounding_factors = confounding_factors or []
        record.validated_by = validated_by

        # Calculate fidelity metrics
        record.calculate_fidelity()

        logger.info(
            f"Validated simulation {simulation_id}: "
            f"predicted={record.simulated_ate:.4f}, actual={actual_ate:.4f}, "
            f"error={record.prediction_error:.2%}, grade={record.fidelity_grade.value}"
        )

        if self.repository:
            self.repository.update_fidelity_record(record)

        # Invalidate model cache
        self._invalidate_model_cache(simulation_id)

        return record

    def get_record(self, tracking_id: UUID) -> Optional[FidelityRecord]:
        """Get fidelity record by tracking ID."""
        return self.records.get(tracking_id)

    def get_simulation_record(self, simulation_id: UUID) -> Optional[FidelityRecord]:
        """Get fidelity record by simulation ID."""
        return self._find_record_by_simulation(simulation_id)

    def get_model_fidelity_report(
        self,
        model_id: UUID,
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Generate fidelity report for a model.

        Args:
            model_id: ID of the twin generator model
            lookback_days: Days to look back for validations

        Returns:
            Report with fidelity metrics and trends
        """
        # Check cache
        cache_key = f"{model_id}_{lookback_days}"
        if cache_key in self.model_fidelity_cache:
            cached = self.model_fidelity_cache[cache_key]
            if (datetime.now(timezone.utc) - cached["computed_at"]).seconds < 3600:
                return cached

        # Gather validated records for this model
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        validated_records = [
            r
            for r in self.records.values()
            if r.validated_at and r.validated_at >= cutoff and r.actual_ate is not None
        ]

        if not validated_records:
            return {
                "model_id": str(model_id),
                "validation_count": 0,
                "message": "No validated records in timeframe",
                "computed_at": datetime.now(timezone.utc),
            }

        # Calculate metrics
        errors = [abs(r.prediction_error) for r in validated_records if r.prediction_error]
        ci_coverages = [r.ci_coverage for r in validated_records if r.ci_coverage is not None]

        # Grade distribution
        grade_counts = {}
        for r in validated_records:
            grade = r.fidelity_grade.value
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        report = {
            "model_id": str(model_id),
            "lookback_days": lookback_days,
            "validation_count": len(validated_records),
            "metrics": {
                "mean_absolute_error": float(np.mean(errors)) if errors else None,
                "median_absolute_error": float(np.median(errors)) if errors else None,
                "max_error": float(max(errors)) if errors else None,
                "ci_coverage_rate": float(np.mean(ci_coverages)) if ci_coverages else None,
            },
            "grade_distribution": grade_counts,
            "fidelity_score": self._calculate_fidelity_score(errors, ci_coverages),
            "degradation_alert": self._check_degradation(validated_records),
            "computed_at": datetime.now(timezone.utc),
        }

        self.model_fidelity_cache[cache_key] = report
        return report

    def check_degradation_alerts(
        self,
        model_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if model shows fidelity degradation.

        Args:
            model_id: Model to check

        Returns:
            Alert dict if degradation detected, None otherwise
        """
        report = self.get_model_fidelity_report(model_id)

        if report.get("degradation_alert"):
            return {
                "model_id": str(model_id),
                "alert_type": "fidelity_degradation",
                "message": "Model fidelity has degraded significantly",
                "current_error": report["metrics"]["mean_absolute_error"],
                "recommendation": "Consider retraining twin model",
                "generated_at": datetime.now(timezone.utc),
            }

        return None

    def get_grade(self, prediction_error: float) -> FidelityGrade:
        """Determine fidelity grade from prediction error."""
        abs_error = abs(prediction_error)

        if abs_error < self.GRADE_THRESHOLDS[FidelityGrade.EXCELLENT]:
            return FidelityGrade.EXCELLENT
        elif abs_error < self.GRADE_THRESHOLDS[FidelityGrade.GOOD]:
            return FidelityGrade.GOOD
        elif abs_error < self.GRADE_THRESHOLDS[FidelityGrade.FAIR]:
            return FidelityGrade.FAIR
        else:
            return FidelityGrade.POOR

    def _find_record_by_simulation(self, simulation_id: UUID) -> Optional[FidelityRecord]:
        """Find fidelity record by simulation ID."""
        for record in self.records.values():
            if record.simulation_id == simulation_id:
                return record

        # Try repository if available
        if self.repository:
            return self.repository.get_fidelity_by_simulation(simulation_id)

        return None

    def _calculate_fidelity_score(
        self,
        errors: List[float],
        ci_coverages: List[bool],
    ) -> float:
        """Calculate composite fidelity score (0-1)."""
        if not errors:
            return 0.5  # Unknown

        # Error component (lower is better)
        mean_error = np.mean(errors)
        error_score = max(0, 1 - mean_error)  # Cap at 0

        # CI coverage component
        coverage_score = np.mean(ci_coverages) if ci_coverages else 0.5

        # Weighted combination
        fidelity_score = 0.7 * error_score + 0.3 * coverage_score

        return float(min(1.0, max(0.0, fidelity_score)))

    def _check_degradation(
        self,
        records: List[FidelityRecord],
    ) -> bool:
        """Check if recent validations show degradation vs historical."""
        if len(records) < self.MIN_VALIDATIONS_FOR_ALERT * 2:
            return False

        # Sort by validation date
        sorted_records = sorted(records, key=lambda r: r.validated_at or datetime.min, reverse=True)

        # Compare recent vs older
        half = len(sorted_records) // 2
        recent = sorted_records[:half]
        older = sorted_records[half:]

        recent_errors = [abs(r.prediction_error) for r in recent if r.prediction_error]
        older_errors = [abs(r.prediction_error) for r in older if r.prediction_error]

        if not recent_errors or not older_errors:
            return False

        recent_mean = np.mean(recent_errors)
        older_mean = np.mean(older_errors)

        # Check if recent error is significantly higher
        increase = (recent_mean - older_mean) / older_mean if older_mean > 0 else 0

        return increase > self.DEGRADATION_THRESHOLD

    def _invalidate_model_cache(self, simulation_id: UUID) -> None:
        """Invalidate model cache when new validation added."""
        # In a real implementation, would need to map simulation to model
        # For now, clear all cache
        self.model_fidelity_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall tracking statistics."""
        total_records = len(self.records)
        validated = sum(1 for r in self.records.values() if r.actual_ate is not None)

        grades = {}
        for r in self.records.values():
            if r.fidelity_grade != FidelityGrade.UNVALIDATED:
                grade = r.fidelity_grade.value
                grades[grade] = grades.get(grade, 0) + 1

        return {
            "total_predictions": total_records,
            "validated_predictions": validated,
            "validation_rate": validated / total_records if total_records > 0 else 0,
            "grade_distribution": grades,
        }
