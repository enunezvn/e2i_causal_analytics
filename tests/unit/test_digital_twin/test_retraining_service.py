"""
Unit tests for digital_twin/retraining_service.py

Tests cover:
- TwinRetrainingService initialization
- Retraining need evaluation
- Trigger retraining
- Job management (status, complete, cancel)
- Cooldown logic
- Training config generation
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.digital_twin.retraining_service import (
    TwinRetrainingConfig,
    TwinRetrainingDecision,
    TwinRetrainingJob,
    TwinRetrainingService,
    TwinRetrainingStatus,
    TwinTriggerReason,
    get_twin_retraining_service,
)


@pytest.fixture
def config():
    """Retraining config with default thresholds."""
    return TwinRetrainingConfig(
        fidelity_threshold=0.70,
        min_validations_for_decision=5,
        max_mean_absolute_error=0.25,
        min_ci_coverage_rate=0.80,
        cooldown_hours=24,
        max_retraining_attempts=3,
        auto_approve_threshold=0.50,
        require_approval=True,
        min_training_samples=1000,
        max_training_data_age_days=90,
    )


@pytest.fixture
def mock_repository():
    """Mock TwinRepository."""
    repo = MagicMock()
    repo.get_model_fidelity_records = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def service(config, mock_repository):
    """TwinRetrainingService instance."""
    return TwinRetrainingService(config=config, repository=mock_repository)


class TestTwinRetrainingConfig:
    """Tests for TwinRetrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = TwinRetrainingConfig()
        assert cfg.fidelity_threshold == 0.70
        assert cfg.min_validations_for_decision == 5
        assert cfg.max_mean_absolute_error == 0.25
        assert cfg.min_ci_coverage_rate == 0.80
        assert cfg.cooldown_hours == 24
        assert cfg.max_retraining_attempts == 3
        assert cfg.auto_approve_threshold == 0.50
        assert cfg.require_approval is True
        assert cfg.min_training_samples == 1000
        assert cfg.max_training_data_age_days == 90

    def test_custom_config(self):
        """Test custom configuration."""
        cfg = TwinRetrainingConfig(
            fidelity_threshold=0.60,
            cooldown_hours=12,
            auto_approve_threshold=0.40,
        )
        assert cfg.fidelity_threshold == 0.60
        assert cfg.cooldown_hours == 12
        assert cfg.auto_approve_threshold == 0.40


class TestTwinRetrainingDecision:
    """Tests for TwinRetrainingDecision dataclass."""

    def test_decision_defaults(self):
        """Test default decision values."""
        decision = TwinRetrainingDecision(should_retrain=False)
        assert decision.should_retrain is False
        assert decision.reason is None
        assert decision.confidence == 0.0
        assert decision.fidelity_score == 0.0
        assert decision.mean_absolute_error == 0.0
        assert decision.ci_coverage_rate == 0.0
        assert decision.validation_count == 0
        assert decision.details == {}
        assert decision.requires_approval is True
        assert decision.recommended_config == {}

    def test_decision_with_values(self):
        """Test decision with all values."""
        decision = TwinRetrainingDecision(
            should_retrain=True,
            reason=TwinTriggerReason.FIDELITY_DEGRADATION,
            confidence=0.85,
            fidelity_score=0.65,
            mean_absolute_error=0.15,
            ci_coverage_rate=0.90,
            validation_count=10,
            details={"test": "data"},
            requires_approval=False,
            recommended_config={"retrain_full": True},
        )
        assert decision.should_retrain is True
        assert decision.reason == TwinTriggerReason.FIDELITY_DEGRADATION
        assert decision.confidence == 0.85
        assert decision.fidelity_score == 0.65
        assert decision.mean_absolute_error == 0.15
        assert decision.ci_coverage_rate == 0.90
        assert decision.validation_count == 10
        assert decision.details == {"test": "data"}
        assert decision.requires_approval is False
        assert decision.recommended_config == {"retrain_full": True}


class TestTwinRetrainingJob:
    """Tests for TwinRetrainingJob dataclass."""

    def test_job_defaults(self):
        """Test default job values."""
        job = TwinRetrainingJob(job_id="test-123", model_id="model-456")
        assert job.job_id == "test-123"
        assert job.model_id == "model-456"
        assert job.new_model_id is None
        assert job.trigger_reason == TwinTriggerReason.MANUAL
        assert job.status == TwinRetrainingStatus.PENDING
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
        assert job.fidelity_before == 0.0
        assert job.fidelity_after is None
        assert job.training_config == {}
        assert job.error_message is None

    def test_job_with_values(self):
        """Test job with all values."""
        now = datetime.now(timezone.utc)
        job = TwinRetrainingJob(
            job_id="test-789",
            model_id="model-abc",
            new_model_id="model-def",
            trigger_reason=TwinTriggerReason.PREDICTION_ERROR,
            status=TwinRetrainingStatus.COMPLETED,
            created_at=now,
            started_at=now + timedelta(minutes=5),
            completed_at=now + timedelta(minutes=35),
            fidelity_before=0.68,
            fidelity_after=0.82,
            training_config={"cv_folds": 5},
            error_message=None,
        )
        assert job.new_model_id == "model-def"
        assert job.trigger_reason == TwinTriggerReason.PREDICTION_ERROR
        assert job.status == TwinRetrainingStatus.COMPLETED
        assert job.fidelity_before == 0.68
        assert job.fidelity_after == 0.82


class TestTwinRetrainingService:
    """Tests for TwinRetrainingService class."""

    def test_initialization(self, config):
        """Test service initialization."""
        service = TwinRetrainingService(config=config, repository=None)
        assert service.config == config
        assert service.repository is None
        assert service._pending_jobs == {}

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        service = TwinRetrainingService()
        assert isinstance(service.config, TwinRetrainingConfig)
        assert service.config.fidelity_threshold == 0.70

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_no_repository(self, service):
        """Test evaluation without repository returns no retrain."""
        service.repository = None
        model_id = uuid4()

        decision = await service.evaluate_retraining_need(model_id)

        assert decision.should_retrain is False
        assert "No fidelity data available" in decision.details.get("error", "")

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_insufficient_validations(
        self, service, mock_repository
    ):
        """Test evaluation with insufficient validations."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 3,
            "metrics": {"mean_absolute_error": 0.15, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.75,
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is False
        assert decision.details["blocked_reason"] == "insufficient_validations"
        assert decision.details["required"] == 5
        assert decision.details["current"] == 3
        assert decision.validation_count == 3

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_fidelity_degradation(self, service):
        """Test evaluation triggers on fidelity degradation."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.15, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.65,  # Below 0.70 threshold
            "grade_distribution": {"good": 5, "fair": 3, "poor": 2},
            "degradation_alert": True,
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is True
        assert decision.reason == TwinTriggerReason.FIDELITY_DEGRADATION
        assert decision.confidence == 1.0 - 0.65  # 0.35
        assert decision.fidelity_score == 0.65
        assert decision.validation_count == 10
        assert decision.recommended_config["retrain_full"] is True

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_prediction_error(self, service):
        """Test evaluation triggers on high prediction error."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 8,
            "metrics": {"mean_absolute_error": 0.30, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.75,
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is True
        assert decision.reason == TwinTriggerReason.PREDICTION_ERROR
        assert decision.mean_absolute_error == 0.30
        assert decision.recommended_config["focus_on_high_error_segments"] is True

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_ci_coverage_drop(self, service):
        """Test evaluation triggers on CI coverage drop."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 7,
            "metrics": {"mean_absolute_error": 0.15, "ci_coverage_rate": 0.75},
            "fidelity_score": 0.80,
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is True
        assert decision.reason == TwinTriggerReason.CI_COVERAGE_DROP
        assert decision.ci_coverage_rate == 0.75
        assert decision.recommended_config["recalibrate_uncertainty"] is True

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_auto_approve_critical(self, service):
        """Test auto-approval for critical degradation."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.15, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.45,  # Below auto_approve_threshold (0.50)
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is True
        assert decision.requires_approval is False  # Auto-approved

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_no_trigger(self, service):
        """Test evaluation when no triggers are met."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.15, "ci_coverage_rate": 0.90},
            "fidelity_score": 0.85,
        }

        decision = await service.evaluate_retraining_need(model_id, fidelity_report)

        assert decision.should_retrain is False
        assert decision.reason is None

    @pytest.mark.asyncio
    async def test_trigger_retraining(self, service):
        """Test triggering retraining creates job."""
        model_id = uuid4()

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = UUID("12345678-1234-5678-1234-567812345678")

            job = await service.trigger_retraining(
                model_id, TwinTriggerReason.FIDELITY_DEGRADATION, approved_by="admin"
            )

        assert job.job_id == "12345678-1234-5678-1234-567812345678"
        assert job.model_id == str(model_id)
        assert job.trigger_reason == TwinTriggerReason.FIDELITY_DEGRADATION
        assert job.status == TwinRetrainingStatus.PENDING
        assert job.training_config["approved_by"] == "admin"
        assert "triggered_at" in job.training_config

    @pytest.mark.asyncio
    async def test_trigger_retraining_with_config_overrides(self, service):
        """Test triggering retraining with config overrides."""
        model_id = uuid4()
        overrides = {"cv_folds": 10, "custom_param": "test"}

        job = await service.trigger_retraining(
            model_id, TwinTriggerReason.MANUAL, config_overrides=overrides
        )

        assert job.training_config["cv_folds"] == 10
        assert job.training_config["custom_param"] == "test"

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_no_need(self, service):
        """Test check and trigger when no retraining needed."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.10, "ci_coverage_rate": 0.95},
            "fidelity_score": 0.90,
        }

        job = await service.check_and_trigger_retraining(model_id, fidelity_report)

        assert job is None

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_requires_approval(self, service):
        """Test check and trigger when approval required."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.20, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.65,
        }

        job = await service.check_and_trigger_retraining(
            model_id, fidelity_report, auto_approve=False
        )

        assert job is None  # Requires manual approval

    @pytest.mark.asyncio
    async def test_check_and_trigger_retraining_auto_approve(self, service):
        """Test check and trigger with auto-approval."""
        model_id = uuid4()
        fidelity_report = {
            "validation_count": 10,
            "metrics": {"mean_absolute_error": 0.20, "ci_coverage_rate": 0.85},
            "fidelity_score": 0.65,
        }

        job = await service.check_and_trigger_retraining(
            model_id, fidelity_report, auto_approve=True
        )

        assert job is not None
        assert job.trigger_reason == TwinTriggerReason.FIDELITY_DEGRADATION
        assert job.training_config["approved_by"] == "auto_approved"

    @pytest.mark.asyncio
    async def test_get_job_status(self, service):
        """Test getting job status."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

        retrieved_job = await service.get_job_status(job.job_id)

        assert retrieved_job == job
        assert retrieved_job.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, service):
        """Test getting status for non-existent job."""
        job = await service.get_job_status("nonexistent-job")
        assert job is None

    @pytest.mark.asyncio
    async def test_complete_retraining_success(self, service):
        """Test completing retraining job successfully."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.FIDELITY_DEGRADATION)
        new_model_id = str(uuid4())

        completed_job = await service.complete_retraining(
            job.job_id, new_model_id, fidelity_after=0.85, success=True
        )

        assert completed_job is not None
        assert completed_job.new_model_id == new_model_id
        assert completed_job.fidelity_after == 0.85
        assert completed_job.status == TwinRetrainingStatus.COMPLETED
        assert completed_job.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_retraining_failure(self, service):
        """Test completing retraining job with failure."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        new_model_id = str(uuid4())

        completed_job = await service.complete_retraining(
            job.job_id, new_model_id, fidelity_after=0.60, success=False
        )

        assert completed_job.status == TwinRetrainingStatus.FAILED

    @pytest.mark.asyncio
    async def test_complete_retraining_job_not_found(self, service):
        """Test completing non-existent job."""
        result = await service.complete_retraining("nonexistent", "model", 0.8)
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_retraining(self, service):
        """Test canceling pending retraining job."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

        cancelled_job = await service.cancel_retraining(job.job_id, "Not needed")

        assert cancelled_job is not None
        assert cancelled_job.status == TwinRetrainingStatus.CANCELLED
        assert "Not needed" in cancelled_job.error_message

    @pytest.mark.asyncio
    async def test_cancel_retraining_wrong_status(self, service):
        """Test canceling job in non-cancelable status."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        job.status = TwinRetrainingStatus.TRAINING

        result = await service.cancel_retraining(job.job_id, "Test")

        assert result is None  # Cannot cancel training job

    @pytest.mark.asyncio
    async def test_check_cooldown_pending_job(self, service):
        """Test cooldown check with pending job."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

        cooldown = await service._check_cooldown(model_id)

        assert cooldown == job.created_at

    @pytest.mark.asyncio
    async def test_check_cooldown_recent_completion(self, service):
        """Test cooldown check with recently completed job."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        job.status = TwinRetrainingStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc) - timedelta(hours=12)

        cooldown = await service._check_cooldown(model_id)

        assert cooldown == job.completed_at

    @pytest.mark.asyncio
    async def test_check_cooldown_expired(self, service):
        """Test cooldown check with expired cooldown."""
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        job.status = TwinRetrainingStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc) - timedelta(hours=25)  # > 24 hours

        cooldown = await service._check_cooldown(model_id)

        assert cooldown is None

    def test_build_training_config_fidelity_degradation(self, service):
        """Test training config for fidelity degradation."""
        config = service._build_training_config(TwinTriggerReason.FIDELITY_DEGRADATION, 0.65, 0.15)

        assert config["retrain_full"] is True
        assert config["increase_training_samples"] is True
        assert config["min_samples"] == 1000
        assert config["validation_split"] == 0.2

    def test_build_training_config_prediction_error(self, service):
        """Test training config for prediction error."""
        config = service._build_training_config(TwinTriggerReason.PREDICTION_ERROR, 0.75, 0.30)

        assert config["focus_on_high_error_segments"] is True
        assert config["adjust_feature_weights"] is True

    def test_build_training_config_ci_coverage_drop(self, service):
        """Test training config for CI coverage drop."""
        config = service._build_training_config(TwinTriggerReason.CI_COVERAGE_DROP, 0.75, 0.15)

        assert config["recalibrate_uncertainty"] is True
        assert config["increase_ensemble_size"] is True

    def test_build_training_config_severe_degradation(self, service):
        """Test training config for severe degradation."""
        config = service._build_training_config(TwinTriggerReason.FIDELITY_DEGRADATION, 0.45, 0.30)

        assert config["retrain_full"] is True
        assert config["extended_tuning_budget"] is True
        assert config["data_window_days"] == 30

    @pytest.mark.asyncio
    async def test_get_statistics(self, service):
        """Test getting service statistics."""
        # Add some jobs
        model_id = uuid4()
        await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        await service.trigger_retraining(model_id, TwinTriggerReason.FIDELITY_DEGRADATION)

        stats = service.get_statistics()

        assert stats["total_jobs"] == 2
        assert "pending" in stats["status_distribution"]
        assert stats["config"]["fidelity_threshold"] == 0.70


class TestGetTwinRetrainingService:
    """Tests for factory function."""

    def test_get_twin_retraining_service_default(self):
        """Test getting service with defaults."""
        service = get_twin_retraining_service()
        assert isinstance(service, TwinRetrainingService)
        assert isinstance(service.config, TwinRetrainingConfig)

    def test_get_twin_retraining_service_custom_config(self):
        """Test getting service with custom config."""
        config = TwinRetrainingConfig(fidelity_threshold=0.60)
        service = get_twin_retraining_service(config=config)
        assert service.config.fidelity_threshold == 0.60
