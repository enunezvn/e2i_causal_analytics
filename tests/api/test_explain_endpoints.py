"""
Tests for Explain API endpoints.

Phase 2C of API Audit - Model Interpretability API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2C.1: SHAP Core (POST /explain/predict, POST /explain/predict/batch, GET /explain/history/{patient_id})
- Batch 2C.2: Infrastructure (GET /explain/models, GET /explain/health)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.explain import FeatureContribution

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_shap_result():
    """Mock SHAP computation result."""
    return {
        "base_value": 0.42,
        "contributions": [
            FeatureContribution(
                feature_name="days_since_last_hcp_visit",
                feature_value=45,
                shap_value=0.15,
                contribution_direction="positive",
                contribution_rank=1,
            ),
            FeatureContribution(
                feature_name="total_hcp_interactions_90d",
                feature_value=12,
                shap_value=0.12,
                contribution_direction="positive",
                contribution_rank=2,
            ),
            FeatureContribution(
                feature_name="therapy_adherence_score",
                feature_value=0.72,
                shap_value=-0.08,
                contribution_direction="negative",
                contribution_rank=3,
            ),
        ],
        "shap_sum": 0.19,
        "explainer_type": "TreeExplainer",
        "computation_time_ms": 127.5,
    }


@pytest.fixture
def mock_prediction():
    """Mock prediction result."""
    return {
        "prediction_class": "high_propensity",
        "prediction_probability": 0.78,
        "model_version_id": "v2.3.1-prod",
    }


@pytest.fixture
def mock_features():
    """Mock feature values."""
    return {
        "days_since_last_hcp_visit": 45,
        "total_hcp_interactions_90d": 12,
        "therapy_adherence_score": 0.72,
        "lab_value_trend": 0.15,
        "prior_brand_experience": 1,
    }


@pytest.fixture
def mock_shap_service(mock_shap_result, mock_prediction, mock_features):
    """Mock RealTimeSHAPService instance."""
    service = MagicMock()
    service.get_features = AsyncMock(return_value=mock_features)
    service.get_prediction = AsyncMock(return_value=mock_prediction)
    service.compute_shap = AsyncMock(return_value=mock_shap_result)
    service.generate_narrative = AsyncMock(
        return_value="This patient shows high propensity (confidence: 78%)."
    )
    service.store_audit_record = AsyncMock(return_value=True)
    service._ensure_initialized = AsyncMock()

    # Mock explainer for cache stats
    service.shap_explainer = MagicMock()
    service.shap_explainer.get_cache_stats = MagicMock(
        return_value={"hits": 10, "misses": 5, "size": 3}
    )

    # Mock dependency status
    service.bentoml_client = MagicMock()
    service.feast_client = MagicMock()
    service.shap_repo = MagicMock()
    service.shap_repo.client = MagicMock()

    return service


# =============================================================================
# BATCH 2C.1 - SHAP CORE TESTS
# =============================================================================


class TestExplainPrediction:
    """Tests for POST /explain/predict."""

    def test_explain_prediction_success(self, mock_shap_service):
        """Should return prediction with SHAP explanation."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "top_k": 5,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "explanation_id" in data
        assert data["patient_id"] == "PAT-2024-001234"
        assert data["model_type"] == "propensity"
        assert "prediction_class" in data
        assert "prediction_probability" in data
        assert "top_features" in data
        assert "shap_sum" in data

    def test_explain_prediction_with_features(self, mock_shap_service):
        """Should use provided features instead of fetching."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "features": {
                        "days_since_last_hcp_visit": 30,
                        "therapy_adherence_score": 0.85,
                    },
                },
            )

        assert response.status_code == 200
        # Should not call get_features when features are provided
        mock_shap_service.get_features.assert_not_called()

    def test_explain_prediction_with_narrative(self, mock_shap_service):
        """Should generate narrative when format=narrative."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "format": "narrative",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["narrative_explanation"] is not None
        mock_shap_service.generate_narrative.assert_called_once()

    def test_explain_prediction_with_hcp_context(self, mock_shap_service):
        """Should accept HCP context."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "hcp_id": "HCP-NE-5678",
                    "model_type": "propensity",
                },
            )

        assert response.status_code == 200

    def test_explain_prediction_all_model_types(self, mock_shap_service):
        """Should support all model types."""
        model_types = ["propensity", "risk_stratification", "next_best_action", "churn_prediction"]

        for model_type in model_types:
            with patch(
                "src.api.routes.explain.get_shap_service",
                new=AsyncMock(return_value=mock_shap_service),
            ):
                response = client.post(
                    "/explain/predict",
                    json={
                        "patient_id": "PAT-2024-001234",
                        "model_type": model_type,
                    },
                )
            assert response.status_code == 200, f"Failed for model_type: {model_type}"

    def test_explain_prediction_stores_audit(self, mock_shap_service):
        """Should store audit record when requested."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "store_for_audit": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["audit_stored"] is True


class TestBatchExplanation:
    """Tests for POST /explain/predict/batch."""

    def test_batch_explanation_success(self, mock_shap_service):
        """Should process multiple patients."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict/batch",
                json={
                    "requests": [
                        {"patient_id": "PAT-001", "model_type": "propensity"},
                        {"patient_id": "PAT-002", "model_type": "propensity"},
                    ],
                    "parallel": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["total_requests"] == 2
        assert "successful" in data
        assert "failed" in data
        assert "explanations" in data
        assert "total_time_ms" in data

    def test_batch_explanation_sequential(self, mock_shap_service):
        """Should process sequentially when parallel=False."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict/batch",
                json={
                    "requests": [
                        {"patient_id": "PAT-001", "model_type": "propensity"},
                    ],
                    "parallel": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] >= 0

    def test_batch_explanation_handles_errors(self, mock_shap_service):
        """Should handle partial failures gracefully."""
        # Make second request fail
        call_count = [0]

        async def failing_compute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise Exception("Simulated failure")
            return {
                "base_value": 0.42,
                "contributions": [],
                "shap_sum": 0.0,
            }

        mock_shap_service.compute_shap = failing_compute

        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/explain/predict/batch",
                json={
                    "requests": [
                        {"patient_id": "PAT-001", "model_type": "propensity"},
                        {"patient_id": "PAT-002", "model_type": "propensity"},
                    ],
                    "parallel": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        # Should have some errors
        assert "errors" in data


class TestExplanationHistory:
    """Tests for GET /explain/history/{patient_id}."""

    def test_get_history_success(self, mock_shap_service):
        """Should return explanation history."""
        mock_repo = MagicMock()
        mock_repo.client = MagicMock()
        mock_repo.table_name = "ml_shap_analyses"

        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "uuid-1",
                "experiment_id": "EXPL-001",
                "computed_at": "2024-01-01T00:00:00Z",
            }
        ]
        mock_repo.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch("src.api.routes.explain.get_shap_analysis_repository", return_value=mock_repo):
            response = client.get("/explain/history/PAT-2024-001234")

        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "PAT-2024-001234"
        assert "total_explanations" in data
        assert "explanations" in data

    def test_get_history_with_limit(self, mock_shap_service):
        """Should respect limit parameter."""
        mock_repo = MagicMock()
        mock_repo.client = MagicMock()
        mock_repo.table_name = "ml_shap_analyses"

        mock_result = MagicMock()
        mock_result.data = []
        mock_repo.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch("src.api.routes.explain.get_shap_analysis_repository", return_value=mock_repo):
            response = client.get("/explain/history/PAT-2024-001234?limit=5")

        assert response.status_code == 200

    def test_get_history_no_db_connection(self, mock_shap_service):
        """Should handle missing database gracefully."""
        mock_repo = MagicMock()
        mock_repo.client = None

        with patch("src.api.routes.explain.get_shap_analysis_repository", return_value=mock_repo):
            response = client.get("/explain/history/PAT-2024-001234")

        assert response.status_code == 200
        data = response.json()
        assert data["total_explanations"] == 0
        assert "message" in data


# =============================================================================
# BATCH 2C.2 - INFRASTRUCTURE TESTS
# =============================================================================


class TestListExplainableModels:
    """Tests for GET /explain/models."""

    def test_list_models_success(self, mock_shap_service):
        """Should list all explainable models."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/models")

        assert response.status_code == 200
        data = response.json()
        assert "supported_models" in data
        assert "total_models" in data
        assert data["total_models"] >= 4  # 4 model types defined

    def test_list_models_includes_explainer_type(self, mock_shap_service):
        """Should indicate explainer type for each model."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/models")

        assert response.status_code == 200
        data = response.json()
        for model in data["supported_models"]:
            assert "model_type" in model
            assert "explainer_type" in model
            assert model["explainer_type"] in ["TreeExplainer", "KernelExplainer"]

    def test_list_models_includes_cache_stats(self, mock_shap_service):
        """Should include cache statistics."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/models")

        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data


class TestExplainHealthCheck:
    """Tests for GET /explain/health."""

    def test_health_check_healthy(self, mock_shap_service):
        """Should return healthy status."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "real-time-shap-api"
        assert "version" in data
        assert "timestamp" in data
        assert "dependencies" in data

    def test_health_check_dependencies(self, mock_shap_service):
        """Should report dependency status."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/health")

        assert response.status_code == 200
        data = response.json()
        deps = data["dependencies"]
        assert "bentoml" in deps
        assert "feast" in deps
        assert "shap_explainer" in deps
        assert "ml_shap_analyses_db" in deps

    def test_health_check_degraded_no_shap(self, mock_shap_service):
        """Should return degraded when SHAP not loaded."""
        mock_shap_service.shap_explainer = None

        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/explain/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
