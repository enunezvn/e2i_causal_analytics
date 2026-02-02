"""
Twin Retraining Service
=======================

Automatic retraining triggers for digital twin models based on
fidelity degradation. This service monitors twin prediction accuracy
and triggers retraining when models fall below performance thresholds.

Integration Points:
    - FidelityTracker (for fidelity monitoring)
    - TwinGenerator (for model retraining)
    - TwinRepository (for model persistence)
    - MLflow (for experiment tracking)

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class TwinTriggerReason(str, Enum):
    """Reasons for triggering twin model retraining."""

    FIDELITY_DEGRADATION = "fidelity_degradation"
    PREDICTION_ERROR = "prediction_error"
    CI_COVERAGE_DROP = "ci_coverage_drop"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA_AVAILABLE = "new_data_available"


class TwinRetrainingStatus(str, Enum):
    """Status of twin retraining job."""

    PENDING = "pending"
    APPROVED = "approved"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TwinRetrainingConfig:
    """Configuration for twin model retraining triggers."""

    # Fidelity thresholds
    fidelity_threshold: float = 0.70  # Below this triggers retraining
    min_validations_for_decision: int = 5  # Min validations before triggering

    # Error thresholds
    max_mean_absolute_error: float = 0.25  # 25% mean error triggers retraining
    min_ci_coverage_rate: float = 0.80  # 80% CI coverage required

    # Cooldown settings
    cooldown_hours: int = 24  # Hours between retraining attempts
    max_retraining_attempts: int = 3  # Max retries before manual intervention

    # Approval settings
    auto_approve_threshold: float = 0.50  # Auto-approve if fidelity drops below this
    require_approval: bool = True

    # Training configuration
    min_training_samples: int = 1000
    max_training_data_age_days: int = 90


@dataclass
class TwinRetrainingDecision:
    """Decision about whether to trigger twin model retraining."""

    should_retrain: bool
    reason: Optional[TwinTriggerReason] = None
    confidence: float = 0.0
    fidelity_score: float = 0.0
    mean_absolute_error: float = 0.0
    ci_coverage_rate: float = 0.0
    validation_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = True
    recommended_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwinRetrainingJob:
    """Twin model retraining job details."""

    job_id: str
    model_id: str
    new_model_id: Optional[str] = None
    trigger_reason: TwinTriggerReason = TwinTriggerReason.MANUAL
    status: TwinRetrainingStatus = TwinRetrainingStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    fidelity_before: float = 0.0
    fidelity_after: Optional[float] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class TwinRetrainingService:
    """
    Service for managing automatic twin model retraining.

    Monitors twin prediction fidelity and triggers retraining when
    models show degraded performance. Integrates with FidelityTracker
    for monitoring and TwinGenerator for retraining.

    Example:
        service = TwinRetrainingService()

        # Check if model needs retraining
        decision = await service.evaluate_retraining_need(model_id)
        if decision.should_retrain:
            job = await service.trigger_retraining(
                model_id, decision.reason
            )
    """

    def __init__(
        self,
        config: Optional[TwinRetrainingConfig] = None,
        repository=None,
    ):
        """
        Initialize twin retraining service.

        Args:
            config: Retraining configuration
            repository: Optional TwinRepository for persistence
        """
        self.config = config or TwinRetrainingConfig()
        self.repository = repository
        self._pending_jobs: Dict[str, TwinRetrainingJob] = {}

        logger.info(
            f"Initialized TwinRetrainingService "
            f"(fidelity_threshold={self.config.fidelity_threshold}, "
            f"auto_approve_threshold={self.config.auto_approve_threshold})"
        )

    async def evaluate_retraining_need(
        self,
        model_id: UUID,
        fidelity_report: Optional[Dict[str, Any]] = None,
    ) -> TwinRetrainingDecision:
        """
        Evaluate whether a twin model needs retraining.

        Checks fidelity scores, prediction errors, and CI coverage
        against configured thresholds.

        Args:
            model_id: Twin model ID to evaluate
            fidelity_report: Optional pre-computed fidelity report

        Returns:
            Retraining decision with details
        """
        # Get fidelity report if not provided
        if fidelity_report is None:
            if self.repository:
                fidelity_report = await self._get_fidelity_report(model_id)
            else:
                return TwinRetrainingDecision(
                    should_retrain=False,
                    details={"error": "No fidelity data available"},
                )

        # Extract metrics
        validation_count = fidelity_report.get("validation_count", 0)
        metrics = fidelity_report.get("metrics", {})
        fidelity_score = fidelity_report.get("fidelity_score", 0.5)
        mean_abs_error = metrics.get("mean_absolute_error", 0.0) or 0.0
        ci_coverage = metrics.get("ci_coverage_rate", 1.0) or 1.0

        # Check if enough validations for decision
        if validation_count < self.config.min_validations_for_decision:
            return TwinRetrainingDecision(
                should_retrain=False,
                fidelity_score=fidelity_score,
                mean_absolute_error=mean_abs_error,
                ci_coverage_rate=ci_coverage,
                validation_count=validation_count,
                details={
                    "blocked_reason": "insufficient_validations",
                    "required": self.config.min_validations_for_decision,
                    "current": validation_count,
                },
            )

        # Check cooldown period
        cooldown_check = await self._check_cooldown(model_id)
        if cooldown_check:
            return TwinRetrainingDecision(
                should_retrain=False,
                fidelity_score=fidelity_score,
                mean_absolute_error=mean_abs_error,
                ci_coverage_rate=ci_coverage,
                validation_count=validation_count,
                details={
                    "blocked_reason": "cooldown_period",
                    "last_retraining": cooldown_check.isoformat(),
                    "cooldown_hours": self.config.cooldown_hours,
                },
            )

        # Evaluate triggers
        should_retrain = False
        trigger_reason = None
        confidence = 0.0

        # Check fidelity threshold
        if fidelity_score < self.config.fidelity_threshold:
            should_retrain = True
            trigger_reason = TwinTriggerReason.FIDELITY_DEGRADATION
            confidence = 1.0 - fidelity_score

        # Check mean absolute error
        elif mean_abs_error > self.config.max_mean_absolute_error:
            should_retrain = True
            trigger_reason = TwinTriggerReason.PREDICTION_ERROR
            confidence = min(1.0, mean_abs_error / self.config.max_mean_absolute_error)

        # Check CI coverage
        elif ci_coverage < self.config.min_ci_coverage_rate:
            should_retrain = True
            trigger_reason = TwinTriggerReason.CI_COVERAGE_DROP
            confidence = 1.0 - ci_coverage

        # Determine approval requirement
        requires_approval = self.config.require_approval
        if fidelity_score < self.config.auto_approve_threshold:
            requires_approval = False  # Auto-approve critical degradation

        # Build recommended training config
        recommended_config = self._build_training_config(
            trigger_reason, fidelity_score, mean_abs_error
        )

        return TwinRetrainingDecision(
            should_retrain=should_retrain,
            reason=trigger_reason,
            confidence=confidence,
            fidelity_score=fidelity_score,
            mean_absolute_error=mean_abs_error,
            ci_coverage_rate=ci_coverage,
            validation_count=validation_count,
            details={
                "fidelity_threshold": self.config.fidelity_threshold,
                "error_threshold": self.config.max_mean_absolute_error,
                "coverage_threshold": self.config.min_ci_coverage_rate,
                "grade_distribution": fidelity_report.get("grade_distribution", {}),
                "degradation_alert": fidelity_report.get("degradation_alert", False),
            },
            requires_approval=requires_approval,
            recommended_config=recommended_config,
        )

    async def trigger_retraining(
        self,
        model_id: UUID,
        reason: TwinTriggerReason,
        config_overrides: Optional[Dict[str, Any]] = None,
        approved_by: Optional[str] = None,
    ) -> TwinRetrainingJob:
        """
        Trigger twin model retraining.

        Creates a retraining job and queues it for execution.

        Args:
            model_id: Model ID to retrain
            reason: Reason for triggering retraining
            config_overrides: Optional training config overrides
            approved_by: User who approved (if manual approval)

        Returns:
            Created retraining job
        """
        import uuid

        # Get current fidelity
        fidelity_before = 0.0
        if self.repository:
            report = await self._get_fidelity_report(model_id)
            fidelity_before = report.get("fidelity_score", 0.0)

        # Build training config
        training_config = self._build_training_config(reason, fidelity_before, 0.0)
        if config_overrides:
            training_config.update(config_overrides)
        training_config["approved_by"] = approved_by
        training_config["triggered_at"] = datetime.now(timezone.utc).isoformat()

        # Create job
        job = TwinRetrainingJob(
            job_id=str(uuid.uuid4()),
            model_id=str(model_id),
            trigger_reason=reason,
            status=TwinRetrainingStatus.PENDING,
            fidelity_before=fidelity_before,
            training_config=training_config,
        )

        self._pending_jobs[job.job_id] = job

        logger.info(
            f"Triggered twin retraining for model {model_id}, "
            f"reason: {reason.value}, job_id: {job.job_id}"
        )

        # Queue retraining task (if Celery is available)
        try:
            from src.tasks.ab_testing_tasks import execute_twin_retraining

            task = execute_twin_retraining.delay(
                retraining_job_id=job.job_id,
                model_id=str(model_id),
                training_config=training_config,
            )
            logger.info(f"Queued retraining task: {task.id}")
        except ImportError:
            logger.warning("Celery tasks not available, retraining job created but not queued")
        except Exception as e:
            logger.error(f"Failed to queue retraining task: {e}")

        return job

    async def check_and_trigger_retraining(
        self,
        model_id: UUID,
        fidelity_report: Optional[Dict[str, Any]] = None,
        auto_approve: bool = False,
    ) -> Optional[TwinRetrainingJob]:
        """
        Evaluate and automatically trigger retraining if needed.

        Args:
            model_id: Model ID to check
            fidelity_report: Optional pre-computed fidelity report
            auto_approve: Skip approval check if True

        Returns:
            Retraining job if triggered, None otherwise
        """
        decision = await self.evaluate_retraining_need(model_id, fidelity_report)

        if not decision.should_retrain:
            logger.debug(f"No retraining needed for model {model_id}")
            return None

        if decision.requires_approval and not auto_approve:
            logger.info(
                f"Retraining recommended for model {model_id} but requires approval. "
                f"Reason: {decision.reason}, confidence: {decision.confidence:.2f}, "
                f"fidelity: {decision.fidelity_score:.2f}"
            )
            return None

        return await self.trigger_retraining(
            model_id=model_id,
            reason=decision.reason,
            approved_by="auto_approved" if auto_approve else None,
        )

    async def get_job_status(self, job_id: str) -> Optional[TwinRetrainingJob]:
        """Get status of a retraining job."""
        return self._pending_jobs.get(job_id)

    async def complete_retraining(
        self,
        job_id: str,
        new_model_id: str,
        fidelity_after: float,
        success: bool = True,
    ) -> Optional[TwinRetrainingJob]:
        """
        Mark retraining job as complete.

        Args:
            job_id: Retraining job ID
            new_model_id: ID of the newly trained model
            fidelity_after: Fidelity score of new model
            success: Whether retraining was successful

        Returns:
            Updated job or None
        """
        job = self._pending_jobs.get(job_id)
        if not job:
            logger.warning(f"Retraining job not found: {job_id}")
            return None

        job.new_model_id = new_model_id
        job.fidelity_after = fidelity_after
        job.completed_at = datetime.now(timezone.utc)
        job.status = TwinRetrainingStatus.COMPLETED if success else TwinRetrainingStatus.FAILED

        logger.info(
            f"Completed retraining job {job_id}: "
            f"fidelity {job.fidelity_before:.2f} -> {fidelity_after:.2f}, "
            f"success={success}"
        )

        return job

    async def cancel_retraining(
        self,
        job_id: str,
        reason: str,
    ) -> Optional[TwinRetrainingJob]:
        """Cancel a pending retraining job."""
        job = self._pending_jobs.get(job_id)
        if not job:
            return None

        if job.status not in (
            TwinRetrainingStatus.PENDING,
            TwinRetrainingStatus.APPROVED,
        ):
            logger.warning(f"Cannot cancel job {job_id} in status {job.status}")
            return None

        job.status = TwinRetrainingStatus.CANCELLED
        job.error_message = f"Cancelled: {reason}"

        logger.info(f"Cancelled retraining job {job_id}: {reason}")
        return job

    async def _get_fidelity_report(self, model_id: UUID) -> Dict[str, Any]:
        """Get fidelity report from repository."""
        try:
            if self.repository:
                records = await self.repository.get_model_fidelity_records(
                    model_id, validated_only=True, limit=50
                )

                if not records:
                    return {"validation_count": 0, "fidelity_score": 0.5}

                # Calculate metrics
                errors = [
                    abs(r.prediction_error) for r in records if r.prediction_error is not None
                ]
                coverages = [r.ci_coverage for r in records if r.ci_coverage is not None]

                import numpy as np

                mean_error = float(np.mean(errors)) if errors else 0.0
                coverage_rate = float(np.mean(coverages)) if coverages else 1.0

                # Calculate fidelity score
                error_score = max(0, 1 - mean_error)
                fidelity_score = 0.7 * error_score + 0.3 * coverage_rate

                # Grade distribution
                grades = {}
                for r in records:
                    grade = r.fidelity_grade.value
                    grades[grade] = grades.get(grade, 0) + 1

                return {
                    "validation_count": len(records),
                    "metrics": {
                        "mean_absolute_error": mean_error,
                        "ci_coverage_rate": coverage_rate,
                    },
                    "fidelity_score": fidelity_score,
                    "grade_distribution": grades,
                }
        except Exception as e:
            logger.error(f"Failed to get fidelity report: {e}")

        return {"validation_count": 0, "fidelity_score": 0.5}

    async def _check_cooldown(self, model_id: UUID) -> Optional[datetime]:
        """Check if model is in cooldown period."""
        # Check pending jobs
        for job in self._pending_jobs.values():
            if job.model_id == str(model_id):
                if job.status in (
                    TwinRetrainingStatus.PENDING,
                    TwinRetrainingStatus.TRAINING,
                ):
                    return job.created_at
                if job.completed_at:
                    cooldown_end = job.completed_at + timedelta(hours=self.config.cooldown_hours)
                    if datetime.now(timezone.utc) < cooldown_end:
                        return job.completed_at

        return None

    def _build_training_config(
        self,
        trigger_reason: Optional[TwinTriggerReason],
        fidelity_score: float,
        mean_error: float,
    ) -> Dict[str, Any]:
        """Build recommended training configuration."""
        config = {
            "min_samples": self.config.min_training_samples,
            "data_window_days": self.config.max_training_data_age_days,
            "validation_split": 0.2,
            "cv_folds": 5,
            "hyperparameter_tuning": True,
        }

        # Adjust based on trigger reason
        if trigger_reason == TwinTriggerReason.FIDELITY_DEGRADATION:
            config["retrain_full"] = True
            config["increase_training_samples"] = True

        elif trigger_reason == TwinTriggerReason.PREDICTION_ERROR:
            config["focus_on_high_error_segments"] = True
            config["adjust_feature_weights"] = True

        elif trigger_reason == TwinTriggerReason.CI_COVERAGE_DROP:
            config["recalibrate_uncertainty"] = True
            config["increase_ensemble_size"] = True

        # Adjust for severe degradation
        if fidelity_score < 0.5:
            config["retrain_full"] = True
            config["extended_tuning_budget"] = True
            config["data_window_days"] = min(30, self.config.max_training_data_age_days)

        return config

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        status_counts = {}
        for job in self._pending_jobs.values():
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_jobs": len(self._pending_jobs),
            "status_distribution": status_counts,
            "config": {
                "fidelity_threshold": self.config.fidelity_threshold,
                "auto_approve_threshold": self.config.auto_approve_threshold,
                "cooldown_hours": self.config.cooldown_hours,
            },
        }


def get_twin_retraining_service(
    config: Optional[TwinRetrainingConfig] = None,
    repository=None,
) -> TwinRetrainingService:
    """
    Get twin retraining service instance.

    Args:
        config: Optional configuration
        repository: Optional TwinRepository

    Returns:
        TwinRetrainingService instance
    """
    return TwinRetrainingService(config, repository)
