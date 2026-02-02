"""
Tests for Predictions API endpoints.

Phase 2D of API Audit - Model Predictions API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2D.1: Inference (POST /api/models/predict/{model}, POST /api/models/predict/{model}/batch, GET /api/models/{model}/info)
- Batch 2D.2: Health (GET /api/models/{model}/health, GET /api/models/status)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies.bentoml_client import get_bentoml_client
from src.api.main import app

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_prediction_result():
    """Mock prediction result from BentoML."""
    return {
        "prediction": 0.85,
        "confidence": 0.92,
        "probabilities": {"high": 0.85, "low": 0.15},
        "prediction_interval": {"lower": 0.78, "upper": 0.92},
        "feature_importance": {"feature_a": 0.4, "feature_b": 0.3, "feature_c": 0.3},
        "model_version": "v2.1.0",
        "_metadata": {
            "model_name": "churn_model",
            "latency_ms": 15.5,
            "endpoint": "http://localhost:3000/churn_model",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def mock_batch_result():
    """Mock batch prediction result from BentoML."""
    return {
        "predictions": [
            {
                "prediction": 0.85,
                "confidence": 0.92,
                "model_version": "v2.1.0",
                "latency_ms": 10.0,
            },
            {
                "prediction": 0.42,
                "confidence": 0.88,
                "model_version": "v2.1.0",
                "latency_ms": 12.0,
            },
        ]
    }


@pytest.fixture
def mock_health_result():
    """Mock health check result."""
    return {
        "status": "healthy",
        "endpoint": "http://localhost:3000/churn_model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_model_info():
    """Mock model info result."""
    return {
        "name": "churn_model",
        "version": "v2.1.0",
        "framework": "sklearn",
        "created_at": "2024-01-15T00:00:00Z",
        "features": ["feature_a", "feature_b", "feature_c"],
        "target": "churn",
        "metrics": {"accuracy": 0.92, "auc": 0.95},
    }


@pytest.fixture
def mock_bentoml_client(
    mock_prediction_result, mock_batch_result, mock_health_result, mock_model_info
):
    """Mock BentoMLClient instance."""
    client_mock = MagicMock()
    client_mock.predict = AsyncMock(return_value=mock_prediction_result)
    client_mock.predict_batch = AsyncMock(return_value=mock_batch_result)
    client_mock.health_check = AsyncMock(return_value=mock_health_result)
    client_mock.get_model_info = AsyncMock(return_value=mock_model_info)
    return client_mock


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


# =============================================================================
# BATCH 2D.1 - INFERENCE TESTS
# =============================================================================


class TestSinglePrediction:
    """Tests for POST /api/models/predict/{model_name}."""

    def test_predict_success(self, mock_bentoml_client):
        """Should return prediction result."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model",
            json={
                "features": {"hcp_id": "HCP001", "territory": "Northeast"},
                "time_horizon": "short_term",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "churn_model"
        assert "prediction" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "timestamp" in data

    def test_predict_with_probabilities(self, mock_bentoml_client):
        """Should return class probabilities when requested."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model",
            json={
                "features": {"hcp_id": "HCP001"},
                "return_probabilities": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["probabilities"] is not None
        assert "high" in data["probabilities"]

    def test_predict_with_intervals(self, mock_bentoml_client):
        """Should return prediction intervals when requested."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/regression_model",
            json={
                "features": {"feature_a": 0.5},
                "return_intervals": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction_interval"] is not None
        assert "lower" in data["prediction_interval"]
        assert "upper" in data["prediction_interval"]

    def test_predict_with_entity_id(self, mock_bentoml_client):
        """Should accept entity_id for feature store lookup."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model",
            json={
                "features": {},
                "entity_id": "HCP-NE-12345",
            },
        )

        assert response.status_code == 200
        # Verify entity_id was passed
        call_args = mock_bentoml_client.predict.call_args
        assert "entity_id" in call_args[0][1]

    def test_predict_circuit_breaker_open(self, mock_bentoml_client):
        """Should return 503 when circuit breaker is open."""
        mock_bentoml_client.predict = AsyncMock(side_effect=RuntimeError("Circuit breaker open"))

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/failing_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 503
        assert "Circuit breaker" in response.json()["detail"]

    def test_predict_internal_error(self, mock_bentoml_client):
        """Should return 500 for other prediction failures."""
        mock_bentoml_client.predict = AsyncMock(side_effect=Exception("Model inference failed"))

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/broken_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestBatchPrediction:
    """Tests for POST /api/models/predict/{model_name}/batch."""

    def test_batch_predict_success(self, mock_bentoml_client):
        """Should process batch predictions."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model/batch",
            json={
                "instances": [
                    {"features": {"hcp_id": "HCP001"}},
                    {"features": {"hcp_id": "HCP002"}},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "churn_model"
        assert data["total_count"] == 2
        assert "predictions" in data
        assert "success_count" in data
        assert "failed_count" in data
        assert "total_latency_ms" in data

    def test_batch_predict_partial_failure(self, mock_bentoml_client):
        """Should handle partial failures gracefully."""
        mock_bentoml_client.predict_batch = AsyncMock(
            return_value={
                "predictions": [
                    {"prediction": 0.8, "confidence": 0.9},
                    {"error": "Invalid features"},
                ]
            }
        )

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model/batch",
            json={
                "instances": [
                    {"features": {"x": 1}},
                    {"features": {"invalid": "data"}},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["failed_count"] >= 1

    def test_batch_predict_empty_request(self):
        """Should reject empty batch request."""
        response = client.post(
            "/api/models/predict/churn_model/batch",
            json={"instances": []},
        )

        assert response.status_code == 422  # Validation error

    def test_batch_predict_error(self, mock_bentoml_client):
        """Should return 500 for batch failures."""
        mock_bentoml_client.predict_batch = AsyncMock(
            side_effect=Exception("Batch processing failed")
        )

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/broken_model/batch",
            json={"instances": [{"features": {"x": 1}}]},
        )

        assert response.status_code == 500


class TestModelInfo:
    """Tests for GET /api/models/{model_name}/info."""

    def test_get_model_info_success(self, mock_bentoml_client):
        """Should return model metadata."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/churn_model/info")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "churn_model"
        assert "version" in data
        assert "framework" in data

    def test_get_model_info_not_found(self, mock_bentoml_client):
        """Should return 404 for unknown model."""
        mock_bentoml_client.get_model_info = AsyncMock(side_effect=Exception("Model not found"))

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/nonexistent_model/info")

        assert response.status_code == 404
        # Check for error detail in response body
        data = response.json()
        error_text = data.get("detail", str(data)).lower()
        assert "not found" in error_text or "unavailable" in error_text


# =============================================================================
# BATCH 2D.2 - HEALTH TESTS
# =============================================================================


class TestModelHealth:
    """Tests for GET /api/models/{model_name}/health."""

    def test_health_check_healthy(self, mock_bentoml_client):
        """Should return healthy status."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/churn_model/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "churn_model"
        assert data["status"] == "healthy"
        assert "endpoint" in data
        assert "last_check" in data

    def test_health_check_unhealthy(self, mock_bentoml_client):
        """Should return unhealthy status when model is down."""
        mock_bentoml_client.health_check = AsyncMock(
            return_value={
                "status": "unhealthy",
                "endpoint": "http://localhost:3000/broken_model",
                "error": "Connection refused",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/broken_model/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["error"] is not None


class TestModelsStatus:
    """Tests for GET /api/models/status."""

    def test_models_status_success(self, mock_bentoml_client):
        """Should return status of all models."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/status")

        assert response.status_code == 200
        data = response.json()
        assert "total_models" in data
        assert "healthy_count" in data
        assert "unhealthy_count" in data
        assert "models" in data
        assert "timestamp" in data

    def test_models_status_with_filter(self, mock_bentoml_client):
        """Should filter to specific models when provided."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get(
            "/api/models/status",
            params={"models": ["churn_model", "conversion_model"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 2

    def test_models_status_mixed_health(self, mock_bentoml_client):
        """Should report mixed health status correctly."""
        call_count = [0]

        async def alternating_health(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return {
                    "status": "unhealthy",
                    "endpoint": "http://localhost:3000/model",
                    "error": "Down",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return {
                "status": "healthy",
                "endpoint": "http://localhost:3000/model",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        mock_bentoml_client.health_check = alternating_health

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.get("/api/models/status")

        assert response.status_code == 200
        data = response.json()
        # With 3 default models: healthy_count + unhealthy_count = total_models
        assert data["healthy_count"] + data["unhealthy_count"] == data["total_models"]
