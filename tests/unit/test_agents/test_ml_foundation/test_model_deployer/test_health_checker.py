"""Tests for health_checker node (check_health)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.model_deployer.nodes.health_checker import check_health


class TestCheckHealth:
    """Test check_health node."""

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Test successful health check with mocked HTTP."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
            "deployment_status": "healthy",
        }

        # Mock the HTTP health check to return success
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(True, 200, None),
        ):
            result = await check_health(state)

        assert result["health_check_passed"] is True
        assert result["health_check_url"] == "https://api.e2i.com/v1/test-staging/predict/health"
        assert result["metrics_url"] == "https://api.e2i.com/v1/test-staging/predict/metrics"
        assert "health_check_response_time_ms" in result
        assert result["health_check_response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_health_missing_endpoint_url(self):
        """Test health check with missing endpoint_url."""
        state = {
            "deployment_status": "healthy",
        }

        result = await check_health(state)

        assert result["health_check_passed"] is False
        assert result["error"] == "Missing endpoint_url for health check"
        assert result["error_type"] == "missing_endpoint_url"

    @pytest.mark.asyncio
    async def test_check_health_deployment_failed(self):
        """Test health check when deployment failed."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
            "deployment_status": "failed",
        }

        result = await check_health(state)

        assert result["health_check_passed"] is False
        assert result["health_check_error"] == "Deployment failed, skipping health check"

    @pytest.mark.asyncio
    async def test_check_health_unhealthy_deployment(self):
        """Test health check for unhealthy deployment with mocked HTTP failure."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
            "deployment_status": "unhealthy",
        }

        # Mock the HTTP health check to return failure
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(False, 503, "Service unavailable"),
        ):
            result = await check_health(state)

        assert result["health_check_passed"] is False
        assert result["health_check_url"] == "https://api.e2i.com/v1/test-staging/predict/health"
        assert result["metrics_url"] == "https://api.e2i.com/v1/test-staging/predict/metrics"
        assert "health_check_error" in result

    @pytest.mark.asyncio
    async def test_check_health_pending_deployment(self):
        """Test health check for pending deployment with mocked HTTP."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
            "deployment_status": "pending",
        }

        # Mock the HTTP health check to return failure (pending deployment)
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(False, None, "Connection refused"),
        ):
            result = await check_health(state)

        assert result["health_check_passed"] is False
        assert "health_check_error" in result

    @pytest.mark.asyncio
    async def test_check_health_production_endpoint(self):
        """Test health check for production endpoint with mocked HTTP."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-production/predict",
            "deployment_status": "healthy",
        }

        # Mock the HTTP health check to return success
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(True, 200, None),
        ):
            result = await check_health(state)

        assert result["health_check_passed"] is True
        assert result["health_check_url"] == "https://api.e2i.com/v1/test-production/predict/health"
        assert result["metrics_url"] == "https://api.e2i.com/v1/test-production/predict/metrics"

    @pytest.mark.asyncio
    async def test_check_health_shadow_endpoint(self):
        """Test health check for shadow endpoint with mocked HTTP."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-shadow/predict",
            "deployment_status": "healthy",
        }

        # Mock the HTTP health check to return success
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(True, 200, None),
        ):
            result = await check_health(state)

        assert result["health_check_passed"] is True
        assert result["health_check_url"] == "https://api.e2i.com/v1/test-shadow/predict/health"

    @pytest.mark.asyncio
    async def test_check_health_response_time_measurement(self):
        """Test that response time is measured."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
            "deployment_status": "healthy",
        }

        # Mock the HTTP health check to return success
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(True, 200, None),
        ):
            result = await check_health(state)

        assert "health_check_response_time_ms" in result
        assert isinstance(result["health_check_response_time_ms"], (int, float))
        assert result["health_check_response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_health_with_default_deployment_status(self):
        """Test health check with default deployment_status and mocked HTTP."""
        state = {
            "endpoint_url": "https://api.e2i.com/v1/test-staging/predict",
        }

        # Mock the HTTP health check to return failure
        with patch(
            "src.agents.ml_foundation.model_deployer.nodes.health_checker._perform_http_health_check",
            new_callable=AsyncMock,
            return_value=(False, None, "Connection refused"),
        ):
            result = await check_health(state)

        # Should treat as pending (default) and fail
        assert result["health_check_passed"] is False
