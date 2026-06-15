"""
Tests for Explain API endpoints.

Phase 2C of API Audit - Model Interpretability API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2C.1: SHAP Core (POST /explain/predict, POST /explain/predict/batch, GET /explain/history/{patient_id})
- Batch 2C.2: Infrastructure (GET /explain/models, GET /explain/health)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.explain import FeatureContribution
from src.api.utils.data_masking import mask_identifier

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
    """Mock prediction result.

    Includes the canonical ``model_features`` dict the endpoint requires —
    get_prediction is contracted to return the strictly-validated, model-ordered
    {name: float} features that feed SHAP + the audit record.
    """
    return {
        "prediction_class": "high_propensity",
        "prediction_probability": 0.78,
        "model_version_id": "v2.3.1-prod",
        "model_features": {
            "days_since_last_hcp_visit": 45.0,
            "total_hcp_interactions_90d": 12.0,
            "therapy_adherence_score": 0.72,
        },
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
                "/api/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "top_k": 5,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "explanation_id" in data
        # patient_id is PII-masked in the response (security enhancement); the
        # raw value is preserved only in audit records. (This assertion silently
        # rotted while tests/api/ was unwired from CI.)
        assert data["patient_id"] == mask_identifier("PAT-2024-001234")
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
                "/api/explain/predict",
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

    def test_endpoint_feeds_canonical_features_to_shap_and_audit(self, mock_shap_service):
        """compute_shap + store_audit_record must receive the canonical
        ``model_features`` from get_prediction (validated, model-ordered), NOT
        the raw request features (which may carry extra/non-numeric fields)."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/api/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "features": {
                        "days_since_last_hcp_visit": 30,
                        "therapy_adherence_score": 0.85,
                        "extra_string_field": "north",  # must NOT reach SHAP/audit
                    },
                    "store_for_audit": True,
                },
            )

        assert response.status_code == 200, response.text
        # compute_shap got the canonical dict from get_prediction, not the raw
        # request features (no extra_string_field).
        shap_kwargs = mock_shap_service.compute_shap.call_args.kwargs
        assert "extra_string_field" not in shap_kwargs["features"]
        assert shap_kwargs["features"] == {
            "days_since_last_hcp_visit": 45.0,
            "total_hcp_interactions_90d": 12.0,
            "therapy_adherence_score": 0.72,
        }
        audit_kwargs = mock_shap_service.store_audit_record.call_args.kwargs
        assert "extra_string_field" not in audit_kwargs["features"]

    def test_endpoint_fails_closed_when_prediction_lacks_model_features(
        self, mock_shap_service, mock_prediction
    ):
        """If get_prediction returns no validated model_features, the endpoint
        must FAIL CLOSED (500) rather than run SHAP/audit over raw features."""
        broken = dict(mock_prediction)
        broken.pop("model_features")
        mock_shap_service.get_prediction = AsyncMock(return_value=broken)
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/api/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "features": {"days_since_last_hcp_visit": 30},
                },
            )
        assert response.status_code == 500, response.text
        mock_shap_service.compute_shap.assert_not_called()

    def test_explain_prediction_with_narrative(self, mock_shap_service):
        """Should generate narrative when format=narrative."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/api/explain/predict",
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
                "/api/explain/predict",
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
                    "/api/explain/predict",
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
                "/api/explain/predict",
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
                "/api/explain/predict/batch",
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
                "/api/explain/predict/batch",
                json={
                    "requests": [
                        {"patient_id": "PAT-001", "model_type": "propensity"},
                    ],
                    "parallel": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        # One request, mocked service succeeds -> exactly one successful explanation.
        assert data["total_requests"] == 1
        assert data["successful"] == 1
        assert data["failed"] == 0

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
                "/api/explain/predict/batch",
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
        # First request succeeds; the second fails in compute_shap -> 1 ok / 1 error.
        assert data["total_requests"] == 2
        assert data["successful"] == 1
        assert data["failed"] == 1
        assert len(data["errors"]) == 1


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
            response = client.get("/api/explain/history/PAT-2024-001234")

        assert response.status_code == 200
        data = response.json()
        # patient_id is PII-masked in the response. (Rotted while unwired from CI.)
        assert data["patient_id"] == mask_identifier("PAT-2024-001234")
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
            response = client.get("/api/explain/history/PAT-2024-001234?limit=5")

        assert response.status_code == 200

    def test_get_history_no_db_connection(self, mock_shap_service):
        """Should handle missing database gracefully."""
        mock_repo = MagicMock()
        mock_repo.client = None

        with patch("src.api.routes.explain.get_shap_analysis_repository", return_value=mock_repo):
            response = client.get("/api/explain/history/PAT-2024-001234")

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
            response = client.get("/api/explain/models")

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
            response = client.get("/api/explain/models")

        assert response.status_code == 200
        data = response.json()
        for model in data["supported_models"]:
            assert "model_type" in model
            assert "explainer_type" in model
            # LinearExplainer added for the gold-standard calibrated-LR cohort
            # models (initiation/persistence/discontinuation/hcp_adoption) now
            # enumerated by /api/explain/models — SHAP uses LinearExplainer for
            # linear/logistic estimators (see explain.py _explainer_label).
            assert model["explainer_type"] in [
                "TreeExplainer",
                "KernelExplainer",
                "LinearExplainer",
            ]

    def test_list_models_includes_cache_stats(self, mock_shap_service):
        """Should include cache statistics."""
        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.get("/api/explain/models")

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
            response = client.get("/api/explain/health")

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
            response = client.get("/api/explain/health")

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
            response = client.get("/api/explain/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


# =============================================================================
# #532 - FAIL-LOUD ON FEAST UNAVAILABILITY (no silent fabricated features)
# =============================================================================


class TestExplainFailLoud:
    """The explain route must fail loud when server-side feature retrieval is
    unavailable, NEVER fabricate plausible feature values into a real SHAP
    explanation / regulatory-audit record (#532 silent-degradation contract)."""

    def test_explain_predict_fails_loud_when_features_unavailable(self, mock_shap_service):
        """When the server-side feature fetch fails (503), the route surfaces 503
        and does NOT compute SHAP or store an audit record over fabricated data."""
        from fastapi import HTTPException

        mock_shap_service.get_features = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Feature store lookup failed")
        )

        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=mock_shap_service),
        ):
            response = client.post(
                "/api/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "store_for_audit": True,
                },
            )

        assert response.status_code == 503
        # No explanation computed and no audit record persisted over fake inputs.
        mock_shap_service.compute_shap.assert_not_called()
        mock_shap_service.store_audit_record.assert_not_called()

    async def test_service_get_features_fails_loud_on_feast_error(self):
        """RealTimeSHAPService.get_features raises 503 (it does NOT return the
        fabricated _get_default_features) when the Feast online lookup fails.

        Lives here (tests/api/, CI-collected) rather than the CI-ignored unit
        tests/unit/.../test_explain.py, so the #532 anti-silent-degradation
        contract for the explain path is actually enforced in CI.
        """
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType, RealTimeSHAPService

        feast = MagicMock()
        feast.get_online_features = AsyncMock(side_effect=Exception("sidecar down"))
        service = RealTimeSHAPService(feast_client=feast, shap_explainer=MagicMock())
        service._initialized = True

        with pytest.raises(HTTPException) as exc_info:
            await service.get_features("PAT-2024-001234", ModelType.PROPENSITY)

        assert exc_info.value.status_code == 503

    async def test_service_get_features_fails_loud_when_all_features_null(self):
        """#576: get_features must raise 503 when Feast returns a 200 whose
        required feature values are ALL null (the composite-key-absent /
        empty-store trap, verified live for PAT_000191 single-key) — NOT return
        a null vector that would drive a real SHAP / regulatory-audit record.

        This is distinct from the feast-error case above: here the lookup
        SUCCEEDS but yields nulls, which the prior fail-loud guard (exceptions
        only) did not catch.
        """
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType, RealTimeSHAPService

        feast = MagicMock()
        feast.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-001234"],
                "days_since_last_hcp_visit": [None],
                "total_hcp_interactions_90d": [None],
                "therapy_adherence_score": [None],
            }
        )
        service = RealTimeSHAPService(feast_client=feast, shap_explainer=MagicMock())
        service._initialized = True

        with pytest.raises(HTTPException) as exc_info:
            await service.get_features("PAT-2024-001234", ModelType.PROPENSITY)

        assert exc_info.value.status_code == 503

    async def test_service_get_features_succeeds_when_features_present(self):
        """Honest success path at the service level: a fully-present non-null
        Feast response is returned as-is (the guard must not over-fire)."""
        from src.api.routes.explain import ModelType, RealTimeSHAPService

        feast = MagicMock()
        feast.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-001234"],
                "days_since_last_hcp_visit": [12.0],
                "total_hcp_interactions_90d": [5.0],
                "therapy_adherence_score": [0.83],
            }
        )
        service = RealTimeSHAPService(feast_client=feast, shap_explainer=MagicMock())
        service._initialized = True

        features = await service.get_features("PAT-2024-001234", ModelType.PROPENSITY)
        assert features["days_since_last_hcp_visit"] == 12.0
        assert features["therapy_adherence_score"] == 0.83

    def test_explain_predict_success_through_real_get_features_guard(
        self, mock_shap_result, mock_prediction
    ):
        """End-to-end happy path through a REAL RealTimeSHAPService: the real
        get_features guard passes on a non-null Feast response, then prediction,
        SHAP, and the audit record proceed. Only the heavy external collaborators
        (BentoML prediction, SHAP compute, audit repo) are mocked — get_features
        and its #576 guard run for real. Locks the integration the guard fronts
        (codex LOW: prior success test was service-level only)."""
        from src.api.routes.explain import RealTimeSHAPService

        feast = MagicMock()
        feast.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-001234"],
                "days_since_last_hcp_visit": [12.0],
                "total_hcp_interactions_90d": [5.0],
                "therapy_adherence_score": [0.83],
            }
        )
        service = RealTimeSHAPService(feast_client=feast, shap_explainer=MagicMock())
        service._initialized = True
        # Replace ONLY the heavy external collaborators; get_features stays REAL.
        service.get_prediction = AsyncMock(return_value=mock_prediction)
        service.compute_shap = AsyncMock(return_value=mock_shap_result)
        service.store_audit_record = AsyncMock(return_value=True)

        with patch(
            "src.api.routes.explain.get_shap_service",
            new=AsyncMock(return_value=service),
        ):
            response = client.post(
                "/api/explain/predict",
                json={
                    "patient_id": "PAT-2024-001234",
                    "model_type": "propensity",
                    "store_for_audit": True,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["audit_stored"] is True
        # The REAL get_features guard ran against the (non-null) Feast response.
        feast.get_online_features.assert_awaited_once()
        # The prediction + SHAP legs actually executed (not bypassed) ...
        service.get_prediction.assert_awaited_once()
        service.compute_shap.assert_awaited_once()
        # ... and the SHAP/prediction results flow into the response (guards
        # against a 200 that skips compute_shap and returns a placeholder shape).
        assert body["shap_sum"] == mock_shap_result["shap_sum"]
        assert len(body["top_features"]) == len(mock_shap_result["contributions"])
        assert body["prediction_class"] == mock_prediction["prediction_class"]
        assert body["prediction_probability"] == mock_prediction["prediction_probability"]
        # The audit record was scheduled and executed (TestClient runs bg tasks).
        service.store_audit_record.assert_awaited_once()


# =============================================================================
# get_prediction: real vectorization + fail-closed (no fabricated 0.78)
# =============================================================================


@pytest.mark.unit
class TestSHAPGetPredictionFailClosed:
    """The SHAP prediction feeds audit-grade output, so it must vectorize via
    the model's real feature order and FAIL CLOSED — never fabricate 0.78."""

    def _service(self, bentoml_client):
        from src.api.routes.explain import RealTimeSHAPService

        svc = RealTimeSHAPService(bentoml_client=bentoml_client)
        svc._ensure_initialized = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_vectorizes_in_model_feature_order(self):
        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(
            return_value={"feature_columns": ["a", "b", "c"], "model_id": "m1"}
        )
        bento.predict = AsyncMock(
            return_value={"predictions": [1.0], "probabilities": [0.9], "model_id": "m1"}
        )
        svc = self._service(bento)

        out = await svc.get_prediction(
            features={"c": 3.0, "a": 1.0, "b": 2.0},  # deliberately unordered
            model_type=ModelType.PROPENSITY,
        )
        sent = bento.predict.call_args.kwargs["input_data"]
        assert sent["features"] == [[1.0, 2.0, 3.0]]  # reordered to a,b,c
        assert out["prediction_probability"] == 0.9
        assert out["model_version_id"] == "m1"

    @pytest.mark.asyncio
    async def test_no_client_fails_closed(self):
        from src.api.routes.explain import ModelType

        svc = self._service(None)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(features={"a": 1.0}, model_type=ModelType.PROPENSITY)
        assert ei.value.status_code == 503

    @pytest.mark.asyncio
    async def test_no_feature_order_fails_closed(self):
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"model_id": "m1"})  # no columns
        bento.predict = AsyncMock()
        svc = self._service(bento)

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(features={"a": 1.0}, model_type=ModelType.PROPENSITY)
        assert ei.value.status_code == 503
        bento.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_feature_fails_closed(self):
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"feature_columns": ["a", "b"]})
        bento.predict = AsyncMock()
        svc = self._service(bento)

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(features={"a": 1.0}, model_type=ModelType.PROPENSITY)
        assert ei.value.status_code == 422
        bento.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_null_feature_fails_closed_not_zero_filled(self):
        """A required feature present as null must 422, NOT be fabricated as 0.0."""
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"feature_columns": ["a", "b"]})
        bento.predict = AsyncMock()
        svc = self._service(bento)

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(
                features={"a": 1.0, "b": None}, model_type=ModelType.PROPENSITY
            )
        assert ei.value.status_code == 422
        bento.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_numeric_feature_fails_closed_not_hash_encoded(self):
        """A required string feature must 422, NOT be silently hash-encoded."""
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"feature_columns": ["a", "b"]})
        bento.predict = AsyncMock()
        svc = self._service(bento)

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(
                features={"a": 1.0, "b": "Northeast"}, model_type=ModelType.PROPENSITY
            )
        assert ei.value.status_code == 422
        bento.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_probabilities_fails_closed_not_fabricated(self):
        """A response with no probabilities must 502, NOT fabricate a 0.0
        audit-grade probability from class predictions."""
        from fastapi import HTTPException

        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"feature_columns": ["a"]})
        bento.predict = AsyncMock(return_value={"predictions": [0.0]})  # no probabilities
        svc = self._service(bento)

        with pytest.raises(HTTPException) as ei:
            await svc.get_prediction(features={"a": 1.0}, model_type=ModelType.PROPENSITY)
        assert ei.value.status_code == 502

    @pytest.mark.asyncio
    async def test_returns_canonical_features_ignoring_extras(self):
        """get_prediction returns a canonical model_features dict (model order,
        validated) that EXCLUDES extra non-model request fields, so SHAP/audit
        never receive a fabricated value for an off-contract key."""
        from src.api.routes.explain import ModelType

        bento = MagicMock()
        bento.get_model_info = AsyncMock(return_value={"feature_columns": ["a", "b"]})
        bento.predict = AsyncMock(
            return_value={"predictions": [1.0], "probabilities": [0.9], "model_id": "m1"}
        )
        svc = self._service(bento)

        out = await svc.get_prediction(
            features={"a": 1.0, "b": 2.0, "extra_string": "north", "extra_obj": {"x": 1}},
            model_type=ModelType.PROPENSITY,
        )
        # Only the model's two features, in order, as floats — extras dropped.
        assert out["model_features"] == {"a": 1.0, "b": 2.0}
