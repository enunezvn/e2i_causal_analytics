"""Tests for registry_manager nodes (register_model, validate_promotion, promote_stage)."""

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
        """Test successful model registration."""
        state = {
            "model_uri": "mlflow://models/test_model/1",
            "deployment_name": "test_deployment",
            "experiment_id": "exp_123",
        }

        result = await register_model(state)

        assert result["registration_successful"] is True
        assert result["registered_model_name"] == "test_deployment"
        assert result["model_version"] == 1
        assert result["current_stage"] == "None"
        assert "registration_timestamp" in result

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
