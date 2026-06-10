"""Tests for registry_manager nodes (register_model, validate_promotion, promote_stage)."""

from unittest.mock import patch

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    promote_stage,
    register_model,
    validate_promotion,
)


class TestRegisterModel:
    """Test register_model node."""

    @pytest.mark.asyncio
    async def test_register_model_success(self):
        """Test successful model registration with mocked MLflow."""
        state = {
            "model_uri": "mlflow://models/test_model/1",
            "deployment_name": "test_deployment",
            "experiment_id": "exp_123",
        }

        # Mock the MLflow registration to return predictable values
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.registry_manager._register_model_mlflow",
            return_value=("test_deployment", 1, "None"),
        ):
            result = await register_model(state)

        assert result["registration_successful"] is True
        assert result["registered_model_name"] == "test_deployment"
        assert result["model_version"] == 1
        assert result["current_stage"] == "None"
        assert "registration_timestamp" in result
        # The register path must also propagate deployment identity + status so the
        # runner's deployment manifest reflects reality (not empty-string/"pending").
        assert result["deployment_id"] == "test_deployment:v1"
        assert result["deployment_status"] == "healthy"
        assert "deployed_at" in result

    @pytest.mark.asyncio
    async def test_register_model_sets_deployment_id_and_status(self):
        """register_model must set deployment_id + deployment_status='healthy' on success.

        Regression test for the bug where the register-only path left deployment_id=''
        and deployment_status='pending', causing Tier-0 validation checks to falsely pass
        (empty string bypassed `!= "N/A"`) and the status check to fail against an
        unreachable 'deployed' sentinel.
        """
        state = {
            "model_uri": "runs:/abc/model",
            "deployment_name": "rwd_model",
            "experiment_id": "exp_42",
        }

        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.registry_manager._register_model_mlflow",
            return_value=("rwd_model", 3, "None"),
        ):
            result = await register_model(state)

        assert result["deployment_id"] == "rwd_model:v3"
        assert result["deployment_id"] != ""
        assert result["deployment_status"] == "healthy"
        assert result["deployment_status"] in {
            "pending",
            "deploying",
            "healthy",
            "unhealthy",
            "failed",
        }

    @pytest.mark.asyncio
    async def test_register_model_missing_model_uri(self):
        """Test registration with missing model_uri."""
        state = {
            "deployment_name": "test_deployment",
            "experiment_id": "exp_123",
        }

        result = await register_model(state)

        assert result["registration_successful"] is False
        assert result["error"] == "Missing model_uri for registration"
        assert result["error_type"] == "missing_model_uri"

    @pytest.mark.asyncio
    async def test_register_model_missing_deployment_name(self):
        """Test registration with missing deployment_name."""
        state = {
            "model_uri": "mlflow://models/test_model/1",
            "experiment_id": "exp_123",
        }

        result = await register_model(state)

        assert result["registration_successful"] is False
        assert result["error"] == "Missing deployment_name for registration"
        assert result["error_type"] == "missing_deployment_name"


class TestValidatePromotion:
    """Test validate_promotion node."""

    @pytest.mark.asyncio
    async def test_validate_promotion_none_to_staging(self):
        """Test promotion from None to Staging (allowed)."""
        state = {
            "current_stage": "None",
            "target_environment": "staging",
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is True
        assert result["promotion_target_stage"] == "Staging"
        assert "promotion_reason" in result

    @pytest.mark.asyncio
    async def test_validate_promotion_staging_to_shadow(self):
        """Test promotion from Staging to Shadow (allowed)."""
        state = {
            "current_stage": "Staging",
            "target_environment": "shadow",
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is True
        assert result["promotion_target_stage"] == "Shadow"

    @pytest.mark.asyncio
    async def test_validate_promotion_shadow_to_production_valid(self):
        """Test promotion from Shadow to Production with valid shadow metrics."""
        state = {
            "current_stage": "Shadow",
            "target_environment": "production",
            "shadow_mode_duration_hours": 25,
            "shadow_mode_requests": 1500,
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 120,
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is True
        assert result["promotion_target_stage"] == "Production"
        assert result["shadow_mode_validated"] is True

    @pytest.mark.asyncio
    async def test_validate_promotion_shadow_to_production_insufficient_duration(self):
        """Test promotion from Shadow to Production with insufficient duration."""
        state = {
            "current_stage": "Shadow",
            "target_environment": "production",
            "shadow_mode_duration_hours": 12,  # Less than 24
            "shadow_mode_requests": 1500,
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 120,
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is False
        assert result["shadow_mode_validated"] is False
        assert "duration_hours" in result["validation_failures"][0]

    @pytest.mark.asyncio
    async def test_validate_promotion_shadow_to_production_insufficient_requests(self):
        """Test promotion from Shadow to Production with insufficient requests."""
        state = {
            "current_stage": "Shadow",
            "target_environment": "production",
            "shadow_mode_duration_hours": 25,
            "shadow_mode_requests": 500,  # Less than 1000
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 120,
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is False
        assert result["shadow_mode_validated"] is False
        assert "requests" in result["validation_failures"][0]

    @pytest.mark.asyncio
    async def test_validate_promotion_shadow_to_production_high_error_rate(self):
        """Test promotion from Shadow to Production with high error rate."""
        state = {
            "current_stage": "Shadow",
            "target_environment": "production",
            "shadow_mode_duration_hours": 25,
            "shadow_mode_requests": 1500,
            "shadow_mode_error_rate": 0.02,  # Greater than 0.01
            "shadow_mode_latency_p99_ms": 120,
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is False
        assert result["shadow_mode_validated"] is False
        assert "error_rate" in result["validation_failures"][0]

    @pytest.mark.asyncio
    async def test_validate_promotion_shadow_to_production_high_latency(self):
        """Test promotion from Shadow to Production with high latency."""
        state = {
            "current_stage": "Shadow",
            "target_environment": "production",
            "shadow_mode_duration_hours": 25,
            "shadow_mode_requests": 1500,
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 200,  # Greater than 150
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is False
        assert result["shadow_mode_validated"] is False
        assert "latency_p99_ms" in result["validation_failures"][0]

    @pytest.mark.asyncio
    async def test_validate_promotion_none_to_production_without_shadow_metrics(self):
        """Test initial deployment to production fails without valid shadow metrics.

        Initial deployments (None stage) to production ARE allowed as a path,
        but require valid shadow mode metrics. Without providing any shadow
        metrics, this should fail shadow validation.
        """
        state = {
            "current_stage": "None",
            "target_environment": "production",
            # No shadow mode metrics provided - will use defaults (0, 0, 1.0, 999)
        }

        result = await validate_promotion(state)

        # Path is allowed, but shadow validation fails
        assert result["promotion_allowed"] is False
        assert result["shadow_mode_validated"] is False
        assert "error" in result
        assert "Shadow mode validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_promotion_invalid_path_staging_to_production(self):
        """Test invalid promotion path (Staging to Production without Shadow)."""
        state = {
            "current_stage": "Staging",
            "target_environment": "production",
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is False
        assert "Invalid promotion path" in result["promotion_denial_reason"]

    @pytest.mark.asyncio
    async def test_validate_promotion_production_to_archived(self):
        """Test promotion from Production to Archived (allowed)."""
        state = {
            "current_stage": "Production",
            "target_environment": "archived",
        }

        result = await validate_promotion(state)

        assert result["promotion_allowed"] is True
        assert result["promotion_target_stage"] == "Archived"


class TestPromoteStage:
    """Test promote_stage node."""

    @pytest.fixture(autouse=True)
    def _mlflow_transition_succeeds(self):
        """These tests verify promotion LOGIC (stage, reason, metrics) on the
        success path, so mock the real MLflow stage transition to succeed.

        F4 (audit): the prior tests called promote_stage with no MLflow mock —
        the real transition fails for a non-registered model, and the code USED
        to hardcode ``promotion_successful=True`` (the fail-OPEN bug). Now that
        promotion fails closed on a failed transition, the logic tests mock the
        MLflow boundary to genuinely succeed. (Fail-closed-on-simulation is
        covered by test_f4_fail_closed_simulation.py.)"""
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes."
            "registry_manager._transition_stage_mlflow",
            return_value=True,
        ):
            yield

    @pytest.mark.asyncio
    async def test_promote_stage_success(self):
        """Test successful stage promotion."""
        state = {
            "registered_model_name": "test_deployment",
            "model_version": 1,
            "current_stage": "None",
            "promotion_target_stage": "Staging",
            "promotion_reason": "Initial deployment",
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is True
        assert result["current_stage"] == "Staging"
        assert result["previous_stage"] == "None"
        assert "promotion_timestamp" in result

    @pytest.mark.asyncio
    async def test_promote_stage_missing_model_name(self):
        """Test promotion with missing registered_model_name."""
        state = {
            "model_version": 1,
            "current_stage": "None",
            "promotion_target_stage": "Staging",
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is False
        assert result["error"] == "Missing registered_model_name for promotion"
        assert result["error_type"] == "missing_model_name"

    @pytest.mark.asyncio
    async def test_promote_stage_missing_target_stage(self):
        """Test promotion with missing promotion_target_stage."""
        state = {
            "registered_model_name": "test_deployment",
            "model_version": 1,
            "current_stage": "None",
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is False
        assert result["error"] == "Missing promotion_target_stage for promotion"
        assert result["error_type"] == "missing_target_stage"

    @pytest.mark.asyncio
    async def test_promote_stage_with_custom_reason(self):
        """Test promotion with custom reason."""
        state = {
            "registered_model_name": "test_deployment",
            "model_version": 1,
            "current_stage": "Staging",
            "promotion_target_stage": "Shadow",
            "promotion_reason": "Validation tests passed",
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is True
        assert result["promotion_reason"] == "Validation tests passed"

    @pytest.mark.asyncio
    async def test_promote_stage_reads_roc_auc_from_validation_metrics(self):
        """D2.0 regression guard: ``metrics_at_promotion["test_auc"]`` must
        read from ``validation_metrics["roc_auc"]`` (the canonical key
        emitted by model_trainer's ``_compute_split_classification_metrics``),
        not the transposed ``auc_roc``.

        Pre-D2.0 the lookup at registry_manager.py:358 used ``auc_roc`` and
        silently returned the 0.0 default for every promotion — every
        production promotion recorded ``test_auc=0.0`` regardless of the
        actual model AUC. Surfaced by Phase-1 D2 investigation
        (.claude/state/d2_investigation_20260505.md, field #4).
        """
        state = {
            "registered_model_name": "test_deployment",
            "model_version": 1,
            "current_stage": "Staging",
            "promotion_target_stage": "Shadow",
            "validation_metrics": {
                "roc_auc": 0.85,
                "precision": 0.78,
                "recall": 0.72,
                "f1_score": 0.75,
            },
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is True
        assert "metrics_at_promotion" in result
        # The critical assertion — pre-D2.0 this would have been 0.0.
        assert result["metrics_at_promotion"]["test_auc"] == 0.85, (
            "metrics_at_promotion['test_auc'] should read from "
            "validation_metrics['roc_auc'], not the transposed 'auc_roc'"
        )
        # Companion assertions that other fields flow correctly (these
        # were not affected by the bug; pinning them so a future refactor
        # of metrics_at_promotion's shape doesn't silently regress).
        assert result["metrics_at_promotion"]["test_precision"] == 0.78
        assert result["metrics_at_promotion"]["test_recall"] == 0.72
        assert result["metrics_at_promotion"]["test_f1"] == 0.75

    @pytest.mark.asyncio
    async def test_promote_stage_returns_zero_test_auc_when_validation_metrics_empty(self):
        """D2.0: when validation_metrics is missing or has no roc_auc,
        ``test_auc`` must default to 0.0 (matches the pre-D2.0 behavior
        for the missing-metrics case; confirms we did not break the
        default fallback while fixing the typo).
        """
        state = {
            "registered_model_name": "test_deployment",
            "model_version": 1,
            "current_stage": "Staging",
            "promotion_target_stage": "Shadow",
            "validation_metrics": {"precision": 0.78},  # no roc_auc
        }

        result = await promote_stage(state)

        assert result["promotion_successful"] is True
        assert result["metrics_at_promotion"]["test_auc"] == 0.0
        assert result["metrics_at_promotion"]["test_precision"] == 0.78
