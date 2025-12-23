"""Retraining Trigger Service.

Phase 14: Model Monitoring & Drift Detection

Automatic model retraining triggers based on:
- Drift score thresholds (data, model, concept)
- Performance degradation (accuracy, precision, recall)
- Scheduled retraining windows
- Manual triggers

Integration Points:
- Drift Monitor Agent (Tier 3)
- Model Trainer Agent (Tier 0)
- MLflow for experiment tracking
- Celery for async job execution
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TriggerReason(str, Enum):
    """Reasons for triggering retraining."""

    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FEATURE_CHANGE = "feature_change"
    DATA_VOLUME = "data_volume"


class RetrainingStatus(str, Enum):
    """Status of retraining job."""

    PENDING = "pending"
    APPROVED = "approved"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class RetrainingTriggerConfig:
    """Configuration for retraining triggers."""

    # Drift thresholds
    data_drift_threshold: float = 0.5  # Overall drift score threshold
    model_drift_threshold: float = 0.4
    concept_drift_threshold: float = 0.3

    # Performance thresholds
    accuracy_min_threshold: float = 0.7
    performance_drop_threshold: float = 0.1  # 10% relative drop

    # Cooldown settings
    min_hours_between_retraining: int = 24
    max_retraining_attempts: int = 3

    # Approval settings
    require_approval: bool = True
    auto_approve_threshold: float = 0.8  # Auto-approve if drift exceeds this

    # Data requirements
    min_samples_for_retraining: int = 1000
    max_training_data_age_days: int = 90


@dataclass
class RetrainingDecision:
    """Decision about whether to trigger retraining."""

    should_retrain: bool
    reason: Optional[TriggerReason] = None
    confidence: float = 0.0
    drift_score: float = 0.0
    performance_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = True
    recommended_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingJob:
    """Retraining job details."""

    job_id: str
    model_version: str
    new_model_version: str
    trigger_reason: TriggerReason
    status: RetrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    drift_score_before: float = 0.0
    performance_before: float = 0.0
    performance_after: Optional[float] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class RetrainingTriggerService:
    """Service for managing automatic model retraining triggers."""

    def __init__(self, config: Optional[RetrainingTriggerConfig] = None):
        self.config = config or RetrainingTriggerConfig()

    async def evaluate_retraining_need(
        self,
        model_version: str,
    ) -> RetrainingDecision:
        """
        Evaluate whether a model needs retraining.

        Checks drift scores, performance metrics, and cooldown periods.

        Args:
            model_version: Model version/ID to evaluate

        Returns:
            Retraining decision with details
        """
        from src.repositories.drift_monitoring import (
            DriftHistoryRepository,
            RetrainingHistoryRepository,
        )
        from src.services.performance_tracking import get_performance_tracker

        # Get drift status
        drift_repo = DriftHistoryRepository()
        drift_records = await drift_repo.get_latest_drift_status(model_version, limit=50)

        # Calculate drift scores by type
        data_drift_scores = []
        model_drift_scores = []
        concept_drift_scores = []

        for record in drift_records:
            severity_to_score = {
                "none": 0.0,
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "critical": 1.0,
            }
            score = severity_to_score.get(record.severity, 0.0)

            if record.drift_type == "data":
                data_drift_scores.append(score)
            elif record.drift_type == "model":
                model_drift_scores.append(score)
            elif record.drift_type == "concept":
                concept_drift_scores.append(score)

        # Calculate average drift scores
        avg_data_drift = sum(data_drift_scores) / len(data_drift_scores) if data_drift_scores else 0.0
        avg_model_drift = sum(model_drift_scores) / len(model_drift_scores) if model_drift_scores else 0.0
        avg_concept_drift = sum(concept_drift_scores) / len(concept_drift_scores) if concept_drift_scores else 0.0
        overall_drift = max(avg_data_drift, avg_model_drift, avg_concept_drift)

        # Get performance metrics
        tracker = get_performance_tracker()
        try:
            perf_trend = await tracker.get_performance_trend(model_version, "accuracy")
            current_performance = perf_trend.current_value
            baseline_performance = perf_trend.baseline_value
            performance_drop = (
                (baseline_performance - current_performance) / baseline_performance
                if baseline_performance > 0
                else 0.0
            )
        except Exception:
            current_performance = 1.0
            baseline_performance = 1.0
            performance_drop = 0.0

        # Check cooldown period
        retrain_repo = RetrainingHistoryRepository()
        recent_retraining = await self._check_cooldown(model_version, retrain_repo)
        if recent_retraining:
            return RetrainingDecision(
                should_retrain=False,
                reason=None,
                confidence=0.0,
                drift_score=overall_drift,
                performance_score=current_performance,
                details={
                    "blocked_reason": "cooldown_period",
                    "last_retraining": recent_retraining.isoformat(),
                    "cooldown_hours": self.config.min_hours_between_retraining,
                },
            )

        # Evaluate triggers
        trigger_reason = None
        should_retrain = False
        confidence = 0.0

        # Check drift thresholds
        if avg_data_drift >= self.config.data_drift_threshold:
            should_retrain = True
            trigger_reason = TriggerReason.DATA_DRIFT
            confidence = avg_data_drift
        elif avg_model_drift >= self.config.model_drift_threshold:
            should_retrain = True
            trigger_reason = TriggerReason.MODEL_DRIFT
            confidence = avg_model_drift
        elif avg_concept_drift >= self.config.concept_drift_threshold:
            should_retrain = True
            trigger_reason = TriggerReason.CONCEPT_DRIFT
            confidence = avg_concept_drift

        # Check performance thresholds
        if current_performance < self.config.accuracy_min_threshold:
            should_retrain = True
            trigger_reason = TriggerReason.PERFORMANCE_DEGRADATION
            confidence = max(confidence, 1.0 - current_performance)
        elif performance_drop >= self.config.performance_drop_threshold:
            should_retrain = True
            trigger_reason = TriggerReason.PERFORMANCE_DEGRADATION
            confidence = max(confidence, performance_drop)

        # Determine approval requirement
        requires_approval = self.config.require_approval
        if overall_drift >= self.config.auto_approve_threshold:
            requires_approval = False  # Auto-approve critical cases

        # Build recommended training config
        recommended_config = self._build_training_config(
            trigger_reason, overall_drift, current_performance
        )

        return RetrainingDecision(
            should_retrain=should_retrain,
            reason=trigger_reason,
            confidence=confidence,
            drift_score=overall_drift,
            performance_score=current_performance,
            details={
                "data_drift_score": avg_data_drift,
                "model_drift_score": avg_model_drift,
                "concept_drift_score": avg_concept_drift,
                "performance_current": current_performance,
                "performance_baseline": baseline_performance,
                "performance_drop": performance_drop,
                "features_with_drift": len([r for r in drift_records if r.drift_detected]),
            },
            requires_approval=requires_approval,
            recommended_config=recommended_config,
        )

    async def trigger_retraining(
        self,
        model_version: str,
        reason: TriggerReason,
        config_overrides: Optional[Dict[str, Any]] = None,
        approved_by: Optional[str] = None,
    ) -> RetrainingJob:
        """
        Trigger model retraining.

        Creates a retraining job and queues it for execution.

        Args:
            model_version: Model version to retrain
            reason: Reason for triggering retraining
            config_overrides: Optional training config overrides
            approved_by: User who approved (if manual approval)

        Returns:
            Created retraining job
        """
        import uuid

        from src.repositories.drift_monitoring import (
            DriftHistoryRepository,
            RetrainingHistoryRepository,
        )
        from src.services.performance_tracking import get_performance_tracker

        # Get current metrics
        drift_repo = DriftHistoryRepository()
        drift_records = await drift_repo.get_latest_drift_status(model_version, limit=20)
        drift_score = max(
            (self._severity_to_score(r.severity) for r in drift_records), default=0.0
        )

        tracker = get_performance_tracker()
        try:
            perf_trend = await tracker.get_performance_trend(model_version, "accuracy")
            performance_before = perf_trend.current_value
        except Exception:
            performance_before = 0.0

        # Generate new model version
        base_version = model_version.rsplit("_", 1)[0] if "_" in model_version else model_version
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        new_version = f"{base_version}_retrained_{timestamp}"

        # Build training config
        training_config = self._build_training_config(reason, drift_score, performance_before)
        if config_overrides:
            training_config.update(config_overrides)
        training_config["approved_by"] = approved_by

        # Record retraining trigger
        retrain_repo = RetrainingHistoryRepository()
        record = await retrain_repo.trigger_retraining(
            old_model_version=model_version,
            new_model_version=new_version,
            trigger_reason=reason.value,
            drift_score_before=drift_score,
            performance_before=performance_before,
            training_config=training_config,
        )

        # Queue retraining task
        from src.tasks.drift_monitoring_tasks import execute_model_retraining

        task = execute_model_retraining.delay(
            retraining_id=record.id,
            model_version=model_version,
            new_version=new_version,
            training_config=training_config,
        )

        logger.info(
            f"Triggered retraining for {model_version} -> {new_version}, "
            f"reason: {reason.value}, task_id: {task.id}"
        )

        return RetrainingJob(
            job_id=record.id,
            model_version=model_version,
            new_model_version=new_version,
            trigger_reason=reason,
            status=RetrainingStatus.PENDING,
            created_at=record.created_at,
            drift_score_before=drift_score,
            performance_before=performance_before,
            training_config=training_config,
        )

    async def check_and_trigger_retraining(
        self,
        model_version: str,
        auto_approve: bool = False,
    ) -> Optional[RetrainingJob]:
        """
        Evaluate and automatically trigger retraining if needed.

        Args:
            model_version: Model version to check
            auto_approve: Skip approval check if True

        Returns:
            Retraining job if triggered, None otherwise
        """
        decision = await self.evaluate_retraining_need(model_version)

        if not decision.should_retrain:
            logger.info(f"No retraining needed for {model_version}")
            return None

        if decision.requires_approval and not auto_approve:
            logger.info(
                f"Retraining recommended for {model_version} but requires approval. "
                f"Reason: {decision.reason}, confidence: {decision.confidence:.2f}"
            )
            # Would send notification for approval here
            return None

        return await self.trigger_retraining(
            model_version=model_version,
            reason=decision.reason,
            approved_by="auto_approved" if auto_approve else None,
        )

    async def get_retraining_status(
        self,
        job_id: str,
    ) -> Optional[RetrainingJob]:
        """
        Get status of a retraining job.

        Args:
            job_id: Retraining job UUID

        Returns:
            Job details or None if not found
        """
        from src.repositories.drift_monitoring import RetrainingHistoryRepository

        repo = RetrainingHistoryRepository()
        record = await repo.get_by_id(job_id)

        if not record:
            return None

        return RetrainingJob(
            job_id=record.id,
            model_version=record.old_model_version,
            new_model_version=record.new_model_version,
            trigger_reason=TriggerReason(record.trigger_reason),
            status=RetrainingStatus(record.status),
            created_at=record.created_at,
            completed_at=record.completed_at,
            drift_score_before=record.drift_score_before,
            performance_before=record.performance_before,
            performance_after=record.performance_after,
            training_config=record.training_config,
        )

    async def complete_retraining(
        self,
        job_id: str,
        performance_after: float,
        success: bool = True,
    ) -> Optional[RetrainingJob]:
        """
        Mark retraining job as complete.

        Args:
            job_id: Retraining job UUID
            performance_after: Performance metric after retraining
            success: Whether retraining was successful

        Returns:
            Updated job or None
        """
        from src.repositories.drift_monitoring import RetrainingHistoryRepository

        repo = RetrainingHistoryRepository()
        record = await repo.complete_retraining(job_id, performance_after, success)

        if not record:
            return None

        return await self.get_retraining_status(job_id)

    async def rollback_retraining(
        self,
        job_id: str,
        reason: str,
    ) -> Optional[RetrainingJob]:
        """
        Rollback a completed retraining (revert to old model).

        Args:
            job_id: Retraining job UUID
            reason: Reason for rollback

        Returns:
            Updated job or None
        """
        from src.repositories.drift_monitoring import RetrainingHistoryRepository

        repo = RetrainingHistoryRepository()
        record = await repo.rollback_retraining(job_id)

        if not record:
            return None

        logger.warning(f"Rolled back retraining {job_id}: {reason}")

        return await self.get_retraining_status(job_id)

    async def _check_cooldown(
        self,
        model_version: str,
        repo,
    ) -> Optional[datetime]:
        """Check if model is in cooldown period."""
        try:
            # Get recent retraining for this model
            result = await repo.client.table("ml_retraining_history").select(
                "completed_at"
            ).eq("old_model_version", model_version).eq("status", "completed").order(
                "completed_at", desc=True
            ).limit(1).execute()

            if result.data:
                last_completed = datetime.fromisoformat(result.data[0]["completed_at"])
                cooldown_end = last_completed + timedelta(
                    hours=self.config.min_hours_between_retraining
                )
                if datetime.now(timezone.utc) < cooldown_end:
                    return last_completed

        except Exception as e:
            logger.warning(f"Failed to check cooldown: {e}")

        return None

    def _severity_to_score(self, severity: str) -> float:
        """Convert severity string to numeric score."""
        mapping = {
            "none": 0.0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0,
        }
        return mapping.get(severity, 0.0)

    def _build_training_config(
        self,
        trigger_reason: Optional[TriggerReason],
        drift_score: float,
        current_performance: float,
    ) -> Dict[str, Any]:
        """Build recommended training configuration."""
        config = {
            "training_strategy": "incremental",
            "data_window_days": self.config.max_training_data_age_days,
            "min_samples": self.config.min_samples_for_retraining,
            "hyperparameter_tuning": True,
            "validation_split": 0.2,
            "early_stopping": True,
        }

        # Adjust based on trigger reason
        if trigger_reason == TriggerReason.DATA_DRIFT:
            config["training_strategy"] = "retrain_full"
            config["focus_on_drifted_features"] = True

        elif trigger_reason == TriggerReason.CONCEPT_DRIFT:
            config["training_strategy"] = "retrain_full"
            config["adjust_feature_weights"] = True
            config["recalibrate"] = True

        elif trigger_reason == TriggerReason.PERFORMANCE_DEGRADATION:
            config["hyperparameter_tuning"] = True
            config["extended_tuning_budget"] = True

        # Adjust based on severity
        if drift_score >= 0.7:
            config["training_strategy"] = "retrain_full"
            config["data_window_days"] = min(30, self.config.max_training_data_age_days)

        return config


# =============================================================================
# FACTORY
# =============================================================================


def get_retraining_trigger_service(
    config: Optional[RetrainingTriggerConfig] = None,
) -> RetrainingTriggerService:
    """Get retraining trigger service instance."""
    return RetrainingTriggerService(config)


# =============================================================================
# CELERY TASK
# =============================================================================


async def evaluate_and_trigger_retraining(
    model_version: str,
    auto_approve: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate and trigger retraining for a model (Celery task helper).

    Args:
        model_version: Model version/ID
        auto_approve: Skip approval requirement

    Returns:
        Evaluation and trigger results
    """
    service = get_retraining_trigger_service()

    # Evaluate
    decision = await service.evaluate_retraining_need(model_version)

    result = {
        "model_version": model_version,
        "should_retrain": decision.should_retrain,
        "reason": decision.reason.value if decision.reason else None,
        "confidence": decision.confidence,
        "drift_score": decision.drift_score,
        "performance_score": decision.performance_score,
        "requires_approval": decision.requires_approval,
        "details": decision.details,
    }

    # Trigger if appropriate
    if decision.should_retrain and (not decision.requires_approval or auto_approve):
        job = await service.trigger_retraining(
            model_version=model_version,
            reason=decision.reason,
            approved_by="auto" if auto_approve else None,
        )
        result["retraining_triggered"] = True
        result["job_id"] = job.job_id
        result["new_model_version"] = job.new_model_version
    else:
        result["retraining_triggered"] = False

    return result
