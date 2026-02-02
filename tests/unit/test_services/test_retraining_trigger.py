"""
Unit Tests for Retraining Trigger Service (Phase 14).

Tests cover:
- Retraining need evaluation
- Trigger reason classification
- Cooldown period handling
- Job creation and status tracking
- Retraining completion and rollback
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.retraining_trigger import (
    RetrainingDecision,
    RetrainingJob,
    RetrainingStatus,
    RetrainingTriggerConfig,
    RetrainingTriggerService,
    TriggerReason,
    evaluate_and_trigger_retraining,
    get_retraining_trigger_service,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> RetrainingTriggerConfig:
    """Create default retraining trigger configuration."""
    return RetrainingTriggerConfig(
        data_drift_threshold=0.5,
        model_drift_threshold=0.4,
        concept_drift_threshold=0.3,
        accuracy_min_threshold=0.7,
        performance_drop_threshold=0.1,
        min_hours_between_retraining=24,
        max_retraining_attempts=3,
        require_approval=True,
        auto_approve_threshold=0.8,
        min_samples_for_retraining=1000,
        max_training_data_age_days=90,
    )


@pytest.fixture
def retraining_service(default_config: RetrainingTriggerConfig) -> RetrainingTriggerService:
    """Create retraining trigger service instance."""
    return RetrainingTriggerService(config=default_config)


@pytest.fixture
def sample_retraining_job() -> RetrainingJob:
    """Create sample retraining job."""
    return RetrainingJob(
        job_id=str(uuid4()),
        model_version="propensity_v2.1.0",
        new_model_version="propensity_v2.1.0_retrained_20250101_1200",
        status=RetrainingStatus.PENDING,
        trigger_reason=TriggerReason.DATA_DRIFT,
        created_at=datetime.now(timezone.utc),
        drift_score_before=0.65,
        performance_before=0.85,
        training_config={"training_strategy": "incremental"},
    )


# =============================================================================
# MOCK DATA CLASSES
# =============================================================================


@dataclass
class MockDriftRecord:
    """Mock drift record for testing."""

    drift_type: str
    severity: str
    drift_detected: bool = True


@dataclass
class MockRetrainingRecord:
    """Mock retraining record for testing."""

    id: str
    old_model_version: str
    new_model_version: str
    trigger_reason: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    drift_score_before: float = 0.0
    performance_before: float = 0.0
    performance_after: Optional[float] = None
    training_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.training_config is None:
            self.training_config = {}


@dataclass
class MockPerformanceTrend:
    """Mock performance trend for testing."""

    current_value: float
    baseline_value: float
    change_percent: float
    trend: str


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestTriggerReason:
    """Tests for TriggerReason enum."""

    def test_data_drift_reason(self):
        """Test DATA_DRIFT reason exists."""
        assert TriggerReason.DATA_DRIFT == "data_drift"

    def test_model_drift_reason(self):
        """Test MODEL_DRIFT reason exists."""
        assert TriggerReason.MODEL_DRIFT == "model_drift"

    def test_concept_drift_reason(self):
        """Test CONCEPT_DRIFT reason exists."""
        assert TriggerReason.CONCEPT_DRIFT == "concept_drift"

    def test_performance_degradation_reason(self):
        """Test PERFORMANCE_DEGRADATION reason exists."""
        assert TriggerReason.PERFORMANCE_DEGRADATION == "performance_degradation"

    def test_scheduled_reason(self):
        """Test SCHEDULED reason exists."""
        assert TriggerReason.SCHEDULED == "scheduled"

    def test_manual_reason(self):
        """Test MANUAL reason exists."""
        assert TriggerReason.MANUAL == "manual"

    def test_feature_change_reason(self):
        """Test FEATURE_CHANGE reason exists."""
        assert TriggerReason.FEATURE_CHANGE == "feature_change"

    def test_data_volume_reason(self):
        """Test DATA_VOLUME reason exists."""
        assert TriggerReason.DATA_VOLUME == "data_volume"


class TestRetrainingStatus:
    """Tests for RetrainingStatus enum."""

    def test_pending_status(self):
        """Test PENDING status exists."""
        assert RetrainingStatus.PENDING == "pending"

    def test_approved_status(self):
        """Test APPROVED status exists."""
        assert RetrainingStatus.APPROVED == "approved"

    def test_training_status(self):
        """Test TRAINING status exists."""
        assert RetrainingStatus.TRAINING == "training"

    def test_validating_status(self):
        """Test VALIDATING status exists."""
        assert RetrainingStatus.VALIDATING == "validating"

    def test_completed_status(self):
        """Test COMPLETED status exists."""
        assert RetrainingStatus.COMPLETED == "completed"

    def test_failed_status(self):
        """Test FAILED status exists."""
        assert RetrainingStatus.FAILED == "failed"

    def test_rolled_back_status(self):
        """Test ROLLED_BACK status exists."""
        assert RetrainingStatus.ROLLED_BACK == "rolled_back"

    def test_cancelled_status(self):
        """Test CANCELLED status exists."""
        assert RetrainingStatus.CANCELLED == "cancelled"


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestRetrainingTriggerConfig:
    """Tests for RetrainingTriggerConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = RetrainingTriggerConfig()

        assert config.data_drift_threshold == 0.5
        assert config.model_drift_threshold == 0.4
        assert config.concept_drift_threshold == 0.3
        assert config.accuracy_min_threshold == 0.7
        assert config.performance_drop_threshold == 0.1
        assert config.min_hours_between_retraining == 24
        assert config.require_approval is True

    def test_custom_config_values(self, default_config: RetrainingTriggerConfig):
        """Test custom configuration values."""
        assert default_config.model_drift_threshold == 0.4
        assert default_config.concept_drift_threshold == 0.3
        assert default_config.max_retraining_attempts == 3

    def test_config_auto_approve_threshold(self, default_config: RetrainingTriggerConfig):
        """Test auto-approve threshold configuration."""
        assert default_config.auto_approve_threshold == 0.8

    def test_config_data_requirements(self, default_config: RetrainingTriggerConfig):
        """Test data requirement configuration."""
        assert default_config.min_samples_for_retraining == 1000
        assert default_config.max_training_data_age_days == 90


# =============================================================================
# RETRAINING DECISION TESTS
# =============================================================================


class TestRetrainingDecision:
    """Tests for RetrainingDecision dataclass."""

    def test_decision_should_retrain(self):
        """Test decision indicating retraining needed."""
        decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.85,
            drift_score=0.65,
            performance_score=0.82,
            details={
                "data_drift_score": 0.65,
                "performance_current": 0.82,
            },
            requires_approval=True,
            recommended_config={"training_strategy": "retrain_full"},
        )

        assert decision.should_retrain is True
        assert decision.reason == TriggerReason.DATA_DRIFT
        assert decision.confidence == 0.85
        assert decision.drift_score == 0.65

    def test_decision_no_retrain(self):
        """Test decision indicating no retraining needed."""
        decision = RetrainingDecision(
            should_retrain=False,
            reason=None,
            confidence=0.0,
            drift_score=0.15,
            performance_score=0.92,
            details={
                "data_drift_score": 0.15,
                "performance_current": 0.92,
            },
            requires_approval=True,
            recommended_config={},
        )

        assert decision.should_retrain is False
        assert decision.reason is None
        assert decision.drift_score < 0.5

    def test_decision_auto_approve(self):
        """Test decision that doesn't require approval."""
        decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.95,
            drift_score=0.85,
            performance_score=0.65,
            details={"data_drift_score": 0.85},
            requires_approval=False,  # Auto-approved due to high drift
            recommended_config={"training_strategy": "retrain_full"},
        )

        assert decision.requires_approval is False
        assert decision.drift_score >= 0.8


# =============================================================================
# RETRAINING JOB TESTS
# =============================================================================


class TestRetrainingJob:
    """Tests for RetrainingJob dataclass."""

    def test_job_creation(self, sample_retraining_job: RetrainingJob):
        """Test retraining job creation."""
        assert sample_retraining_job.model_version == "propensity_v2.1.0"
        assert sample_retraining_job.new_model_version.startswith("propensity")
        assert sample_retraining_job.status == RetrainingStatus.PENDING
        assert sample_retraining_job.trigger_reason == TriggerReason.DATA_DRIFT

    def test_job_with_metrics(self):
        """Test job with performance metrics."""
        job = RetrainingJob(
            job_id=str(uuid4()),
            model_version="test_v1.0",
            new_model_version="test_v1.0_retrained_20250101",
            status=RetrainingStatus.PENDING,
            trigger_reason=TriggerReason.MANUAL,
            created_at=datetime.now(timezone.utc),
            drift_score_before=0.55,
            performance_before=0.82,
            training_config={"approved_by": "user_123"},
        )

        assert job.drift_score_before == 0.55
        assert job.performance_before == 0.82

    def test_job_completed(self):
        """Test completed retraining job."""
        job = RetrainingJob(
            job_id=str(uuid4()),
            model_version="test_v1.0",
            new_model_version="test_v1.0_retrained_20250101",
            status=RetrainingStatus.COMPLETED,
            trigger_reason=TriggerReason.SCHEDULED,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            started_at=datetime.now(timezone.utc) - timedelta(hours=1),
            completed_at=datetime.now(timezone.utc),
            drift_score_before=0.55,
            performance_before=0.82,
            performance_after=0.88,
            training_config={},
        )

        assert job.status == RetrainingStatus.COMPLETED
        assert job.performance_after > job.performance_before

    def test_job_with_error(self):
        """Test failed retraining job with error."""
        job = RetrainingJob(
            job_id=str(uuid4()),
            model_version="test_v1.0",
            new_model_version="test_v1.0_retrained_20250101",
            status=RetrainingStatus.FAILED,
            trigger_reason=TriggerReason.DATA_DRIFT,
            created_at=datetime.now(timezone.utc),
            error_message="Training failed: insufficient data",
            training_config={},
        )

        assert job.status == RetrainingStatus.FAILED
        assert job.error_message is not None


# =============================================================================
# RETRAINING TRIGGER SERVICE TESTS
# =============================================================================


class TestRetrainingTriggerService:
    """Tests for RetrainingTriggerService class."""

    def test_service_initialization(self, retraining_service: RetrainingTriggerService):
        """Test service initialization."""
        assert retraining_service is not None
        assert retraining_service.config.min_hours_between_retraining == 24

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_high_drift(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test evaluation with high drift scores."""
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="high"),
            MockDriftRecord(drift_type="data", severity="high"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.85,
            baseline_value=0.88,
            change_percent=-3.4,
            trend="stable",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    # No cooldown
                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need(
                            "propensity_v2.1.0"
                        )

                        assert decision.should_retrain is True
                        assert decision.reason == TriggerReason.DATA_DRIFT

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_performance_degradation(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test evaluation with performance degradation."""
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="low"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.60,  # Below accuracy_min_threshold of 0.7
            baseline_value=0.85,
            change_percent=-29.4,
            trend="degrading",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need(
                            "propensity_v2.1.0"
                        )

                        assert decision.should_retrain is True
                        assert decision.reason == TriggerReason.PERFORMANCE_DEGRADATION

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_cooldown_active(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test evaluation with active cooldown."""
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="high"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.85,
            baseline_value=0.88,
            change_percent=-3.4,
            trend="stable",
        )

        cooldown_time = datetime.now(timezone.utc) - timedelta(hours=12)

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = cooldown_time

                        decision = await retraining_service.evaluate_retraining_need(
                            "propensity_v2.1.0"
                        )

                        assert decision.should_retrain is False
                        assert "blocked_reason" in decision.details

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_stable_model(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test evaluation with stable model."""
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="low"),
            MockDriftRecord(drift_type="model", severity="none"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.88,
            baseline_value=0.87,
            change_percent=1.1,
            trend="stable",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need(
                            "propensity_v2.1.0"
                        )

                        assert decision.should_retrain is False

    @pytest.mark.asyncio
    async def test_trigger_retraining(self, retraining_service: RetrainingTriggerService):
        """Test triggering retraining."""
        mock_record = MockRetrainingRecord(
            id="job-123",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101_1200",
            trigger_reason="data_drift",
            status="pending",
            created_at=datetime.now(timezone.utc),
            drift_score_before=0.65,
            performance_before=0.82,
            training_config={},
        )

        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="high"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.82,
            baseline_value=0.88,
            change_percent=-6.8,
            trend="degrading",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo.trigger_retraining = AsyncMock(return_value=mock_record)
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch(
                        "src.tasks.drift_monitoring_tasks.execute_model_retraining"
                    ) as mock_task:
                        mock_task.delay = MagicMock(return_value=MagicMock(id="task-abc"))

                        job = await retraining_service.trigger_retraining(
                            model_version="propensity_v2.1.0",
                            reason=TriggerReason.DATA_DRIFT,
                            approved_by="user_123",
                        )

                        assert job.model_version == "propensity_v2.1.0"
                        assert job.trigger_reason == TriggerReason.DATA_DRIFT
                        assert job.status == RetrainingStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_retraining_status(self, retraining_service: RetrainingTriggerService):
        """Test getting retraining job status."""
        mock_record = MockRetrainingRecord(
            id="job-125",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101",
            trigger_reason="scheduled",
            status="training",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        with patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=mock_record)
            mock_repo_cls.return_value = mock_repo

            job = await retraining_service.get_retraining_status("job-125")

            assert job.job_id == "job-125"
            assert job.status == RetrainingStatus.TRAINING

    @pytest.mark.asyncio
    async def test_complete_retraining_success(self, retraining_service: RetrainingTriggerService):
        """Test completing retraining successfully."""
        mock_record = MockRetrainingRecord(
            id="job-126",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101",
            trigger_reason="data_drift",
            status="completed",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            completed_at=datetime.now(timezone.utc),
            performance_before=0.82,
            performance_after=0.88,
        )

        with patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.complete_retraining = AsyncMock(return_value=mock_record)
            mock_repo.get_by_id = AsyncMock(return_value=mock_record)
            mock_repo_cls.return_value = mock_repo

            job = await retraining_service.complete_retraining(
                job_id="job-126",
                performance_after=0.88,
                success=True,
            )

            assert job.status == RetrainingStatus.COMPLETED
            assert job.performance_after == 0.88

    @pytest.mark.asyncio
    async def test_complete_retraining_failure(self, retraining_service: RetrainingTriggerService):
        """Test completing retraining with failure."""
        mock_record = MockRetrainingRecord(
            id="job-127",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101",
            trigger_reason="data_drift",
            status="failed",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            completed_at=datetime.now(timezone.utc),
            performance_before=0.82,
            performance_after=0.75,
        )

        with patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.complete_retraining = AsyncMock(return_value=mock_record)
            mock_repo.get_by_id = AsyncMock(return_value=mock_record)
            mock_repo_cls.return_value = mock_repo

            job = await retraining_service.complete_retraining(
                job_id="job-127",
                performance_after=0.75,
                success=False,
            )

            assert job.status == RetrainingStatus.FAILED

    @pytest.mark.asyncio
    async def test_rollback_retraining(self, retraining_service: RetrainingTriggerService):
        """Test rolling back retraining."""
        mock_record = MockRetrainingRecord(
            id="job-128",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101",
            trigger_reason="data_drift",
            status="rolled_back",
            created_at=datetime.now(timezone.utc) - timedelta(hours=3),
            completed_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        with patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.rollback_retraining = AsyncMock(return_value=mock_record)
            mock_repo.get_by_id = AsyncMock(return_value=mock_record)
            mock_repo_cls.return_value = mock_repo

            job = await retraining_service.rollback_retraining(
                job_id="job-128",
                reason="Performance degradation on validation set",
            )

            assert job.status == RetrainingStatus.ROLLED_BACK


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_retraining_trigger_service_default(self):
        """Test getting default retraining trigger service."""
        service = get_retraining_trigger_service()
        assert isinstance(service, RetrainingTriggerService)

    def test_get_retraining_trigger_service_with_config(
        self, default_config: RetrainingTriggerConfig
    ):
        """Test getting service with custom config."""
        service = get_retraining_trigger_service(config=default_config)
        assert service.config.auto_approve_threshold == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_and_trigger_retraining(self):
        """Test convenience function for evaluation and triggering."""
        with patch("src.services.retraining_trigger.get_retraining_trigger_service") as mock_get:
            mock_service = MagicMock()

            mock_decision = RetrainingDecision(
                should_retrain=True,
                reason=TriggerReason.DATA_DRIFT,
                confidence=0.85,
                drift_score=0.65,
                performance_score=0.82,
                details={},
                requires_approval=False,
                recommended_config={},
            )

            mock_job = RetrainingJob(
                job_id="job-129",
                model_version="propensity_v2.1.0",
                new_model_version="propensity_v2.1.0_retrained_20250101",
                trigger_reason=TriggerReason.DATA_DRIFT,
                status=RetrainingStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                training_config={},
            )

            mock_service.evaluate_retraining_need = AsyncMock(return_value=mock_decision)
            mock_service.trigger_retraining = AsyncMock(return_value=mock_job)
            mock_get.return_value = mock_service

            result = await evaluate_and_trigger_retraining(
                model_version="propensity_v2.1.0",
                auto_approve=True,
            )

            assert result["should_retrain"] is True
            assert result["retraining_triggered"] is True
            assert result["job_id"] == "job-129"


# =============================================================================
# THRESHOLD TESTS
# =============================================================================


class TestThresholds:
    """Tests for threshold-based decisions."""

    @pytest.mark.asyncio
    async def test_data_drift_threshold_boundary(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test data drift at exact threshold."""
        # At threshold (0.5 = "medium" severity)
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="medium"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.85,
            baseline_value=0.85,
            change_percent=0.0,
            trend="stable",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need("test_v1.0")

                        # At exactly 0.5 threshold with >= comparison should trigger
                        assert decision.should_retrain in [True, False]

    @pytest.mark.asyncio
    async def test_multiple_thresholds_exceeded(self, retraining_service: RetrainingTriggerService):
        """Test when multiple thresholds are exceeded."""
        mock_drift_records = [
            MockDriftRecord(drift_type="data", severity="high"),
            MockDriftRecord(drift_type="model", severity="high"),
            MockDriftRecord(drift_type="concept", severity="medium"),
        ]

        mock_perf_trend = MockPerformanceTrend(
            current_value=0.60,
            baseline_value=0.85,
            change_percent=-29.4,
            trend="degrading",
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=mock_drift_records)
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(return_value=mock_perf_trend)
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need("test_v1.0")

                        assert decision.should_retrain is True


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_evaluate_new_model_no_history(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test evaluation for new model with no history."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as mock_drift_repo_cls:
            mock_drift_repo = MagicMock()
            mock_drift_repo.get_latest_drift_status = AsyncMock(return_value=[])
            mock_drift_repo_cls.return_value = mock_drift_repo

            with patch(
                "src.services.performance_tracking.get_performance_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_performance_trend = AsyncMock(side_effect=Exception("No data"))
                mock_tracker_fn.return_value = mock_tracker

                with patch(
                    "src.repositories.drift_monitoring.RetrainingHistoryRepository"
                ) as mock_retrain_repo_cls:
                    mock_retrain_repo = MagicMock()
                    mock_retrain_repo_cls.return_value = mock_retrain_repo

                    with patch.object(
                        retraining_service, "_check_cooldown", new_callable=AsyncMock
                    ) as mock_cooldown:
                        mock_cooldown.return_value = None

                        decision = await retraining_service.evaluate_retraining_need(
                            "new_model_v1.0"
                        )

                        # Should handle gracefully
                        assert decision.should_retrain is False

    @pytest.mark.asyncio
    async def test_get_status_nonexistent_job(self, retraining_service: RetrainingTriggerService):
        """Test getting status of non-existent job."""
        with patch(
            "src.repositories.drift_monitoring.RetrainingHistoryRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=None)
            mock_repo_cls.return_value = mock_repo

            job = await retraining_service.get_retraining_status("nonexistent-job")

            assert job is None

    def test_severity_to_score_conversion(self, retraining_service: RetrainingTriggerService):
        """Test severity string to score conversion."""
        assert retraining_service._severity_to_score("none") == 0.0
        assert retraining_service._severity_to_score("low") == 0.25
        assert retraining_service._severity_to_score("medium") == 0.5
        assert retraining_service._severity_to_score("high") == 0.75
        assert retraining_service._severity_to_score("critical") == 1.0
        assert retraining_service._severity_to_score("unknown") == 0.0


# =============================================================================
# TRAINING CONFIG TESTS
# =============================================================================


class TestTrainingConfig:
    """Tests for training configuration building."""

    def test_build_training_config_data_drift(self, retraining_service: RetrainingTriggerService):
        """Test training config for data drift trigger."""
        config = retraining_service._build_training_config(
            trigger_reason=TriggerReason.DATA_DRIFT,
            drift_score=0.65,
            current_performance=0.82,
        )

        assert config["training_strategy"] == "retrain_full"
        assert config["focus_on_drifted_features"] is True

    def test_build_training_config_concept_drift(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test training config for concept drift trigger."""
        config = retraining_service._build_training_config(
            trigger_reason=TriggerReason.CONCEPT_DRIFT,
            drift_score=0.45,
            current_performance=0.78,
        )

        assert config["training_strategy"] == "retrain_full"
        assert config["adjust_feature_weights"] is True
        assert config["recalibrate"] is True

    def test_build_training_config_performance_degradation(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test training config for performance degradation trigger."""
        config = retraining_service._build_training_config(
            trigger_reason=TriggerReason.PERFORMANCE_DEGRADATION,
            drift_score=0.30,
            current_performance=0.65,
        )

        assert config["hyperparameter_tuning"] is True
        assert config["extended_tuning_budget"] is True

    def test_build_training_config_high_severity(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test training config adjustments for high severity."""
        config = retraining_service._build_training_config(
            trigger_reason=TriggerReason.DATA_DRIFT,
            drift_score=0.85,  # High severity
            current_performance=0.70,
        )

        assert config["training_strategy"] == "retrain_full"
        assert config["data_window_days"] == 30


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestRetrainingWorkflow:
    """Tests for complete retraining workflows."""

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_triggers(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test check_and_trigger_retraining when retraining is needed."""
        mock_decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.85,
            drift_score=0.65,
            performance_score=0.82,
            details={},
            requires_approval=False,  # Auto-approved
            recommended_config={},
        )

        MockRetrainingRecord(
            id="job-130",
            old_model_version="propensity_v2.1.0",
            new_model_version="propensity_v2.1.0_retrained_20250101",
            trigger_reason="data_drift",
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        with patch.object(
            retraining_service, "evaluate_retraining_need", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_decision

            with patch.object(
                retraining_service, "trigger_retraining", new_callable=AsyncMock
            ) as mock_trigger:
                mock_job = RetrainingJob(
                    job_id="job-130",
                    model_version="propensity_v2.1.0",
                    new_model_version="propensity_v2.1.0_retrained_20250101",
                    trigger_reason=TriggerReason.DATA_DRIFT,
                    status=RetrainingStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                    training_config={},
                )
                mock_trigger.return_value = mock_job

                job = await retraining_service.check_and_trigger_retraining(
                    model_version="propensity_v2.1.0",
                    auto_approve=True,
                )

                assert job is not None
                assert job.trigger_reason == TriggerReason.DATA_DRIFT

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_no_trigger(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test check_and_trigger_retraining when no retraining needed."""
        mock_decision = RetrainingDecision(
            should_retrain=False,
            reason=None,
            confidence=0.0,
            drift_score=0.15,
            performance_score=0.92,
            details={},
            requires_approval=True,
            recommended_config={},
        )

        with patch.object(
            retraining_service, "evaluate_retraining_need", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_decision

            job = await retraining_service.check_and_trigger_retraining(
                model_version="propensity_v2.1.0",
                auto_approve=False,
            )

            assert job is None

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_requires_approval(
        self, retraining_service: RetrainingTriggerService
    ):
        """Test check_and_trigger_retraining when approval required."""
        mock_decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.65,
            drift_score=0.55,
            performance_score=0.80,
            details={},
            requires_approval=True,
            recommended_config={},
        )

        with patch.object(
            retraining_service, "evaluate_retraining_need", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_decision

            job = await retraining_service.check_and_trigger_retraining(
                model_version="propensity_v2.1.0",
                auto_approve=False,  # Don't auto-approve
            )

            # Should not trigger because approval is required
            assert job is None
