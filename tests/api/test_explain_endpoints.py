"""
Tests for Explain API endpoints.

Phase 2C of API Audit - Model Interpretability API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2C.1: SHAP Core (POST /explain/predict, POST /explain/predict/batch, GET /explain/history/{patient_id})
- Batch 2C.2: Infrastructure (GET /explain/models, GET /explain/health)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.explain import FeatureContribution
from src.api.utils.data_masking import mask_identifier

client = TestClient(app)


# =============================================================================
# HELPERS FOR GLOBAL / SAMPLE-ENTITIES TESTS (#39 — option 2)
# =============================================================================


class _FakeSlot:
    """No-op stand-in for the heavy-compute slot async context manager."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeQuery:
    """Fluent stand-in for the async Supabase client query builder.

    Every builder method returns ``self``; ``execute()`` pops the next queued
    result (a ``SimpleNamespace(data=...)``), so a single fake can serve several
    sequential queries in order.
    """

    def __init__(self, results):
        self._results = list(results)

    def table(self, *a, **k):
        return self

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def like(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    async def execute(self):
        return self._results.pop(0)


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


# =============================================================================
# COHORT-LEVEL (GLOBAL) FEATURE IMPORTANCE + SAMPLE ENTITIES (#39 — option 2)
# =============================================================================


def _ns_resp(base_value, feats):
    """Minimal ExplainResponse-like object the aggregator reads."""
    return SimpleNamespace(
        base_value=base_value,
        top_features=[
            FeatureContribution(
                feature_name=name,
                feature_value=val,
                shap_value=shap,
                contribution_direction="positive" if shap >= 0 else "negative",
                contribution_rank=i + 1,
            )
            for i, (name, val, shap) in enumerate(feats)
        ],
    )


class TestComputeGlobalImportance:
    """Pure-logic tests for the cohort aggregation (no HTTP, no DB, no SHAP)."""

    async def test_aggregates_mean_abs_signed_and_value_and_skips_failures(self):
        from src.api.routes import explain as explain_mod

        # Two good explanations + one that 503s in the middle (must be skipped).
        resp1 = _ns_resp(-0.7, [("disease_severity", 5.0, 0.6), ("region_ne", 1.0, -0.2)])
        resp3 = _ns_resp(-0.9, [("disease_severity", 7.0, 0.8), ("region_ne", 3.0, 0.1)])

        with (
            patch.object(
                explain_mod,
                "_sample_entity_ids",
                new=AsyncMock(return_value=["e1", "e2", "e3"]),
            ),
            patch.object(
                explain_mod,
                "explain_prediction",
                new=AsyncMock(side_effect=[resp1, HTTPException(status_code=503), resp3]),
            ),
            patch(
                "src.api.dependencies.compute.heavy_compute_slot",
                new=lambda *a, **k: _FakeSlot(),
            ),
        ):
            agg = await explain_mod._compute_global_importance(
                explain_mod.ModelType.INITIATION,
                "Remibrutinib",
                sample_size=2,
                background_tasks=None,
            )

        # The failing entity is skipped — honest n_succeeded over the 2 that worked.
        assert agg["sample_size"] == 2
        feats = {f["feature_name"]: f for f in agg["features"]}
        # disease_severity: mean|shap| = (0.6+0.8)/2 = 0.7 ; mean_shap = 0.7 ; mean_val = 6.0
        assert feats["disease_severity"]["mean_abs_shap"] == pytest.approx(0.7)
        assert feats["disease_severity"]["mean_shap"] == pytest.approx(0.7)
        assert feats["disease_severity"]["mean_feature_value"] == pytest.approx(6.0)
        # region_ne: mean|shap| = (0.2+0.1)/2 = 0.15 ; mean_shap = (-0.2+0.1)/2 = -0.05
        assert feats["region_ne"]["mean_abs_shap"] == pytest.approx(0.15)
        assert feats["region_ne"]["mean_shap"] == pytest.approx(-0.05)
        # Ranked by mean|shap| desc.
        assert agg["features"][0]["feature_name"] == "disease_severity"
        assert agg["features"][0]["contribution_rank"] == 1
        # Base value averaged across the successful explanations.
        assert agg["base_value"] == pytest.approx(-0.8)
        # Real per-entity points retained for the beeswarm distribution.
        assert len(agg["points"]["disease_severity"]) == 2

    async def test_sparse_features_divide_by_full_sample_not_per_feature_count(self):
        """A feature present in only some entities must be averaged over the FULL
        cohort (n_succeeded), treating its absence as ~0 — not inflated by dividing
        by the count of entities where it surfaced."""
        from src.api.routes import explain as explain_mod

        # shared_feat in both; only_in_1 / only_in_2 each in exactly one entity.
        resp1 = _ns_resp(-0.5, [("shared_feat", 5.0, 0.4), ("only_in_1", 1.0, 0.6)])
        resp2 = _ns_resp(-0.5, [("shared_feat", 5.0, 0.4), ("only_in_2", 1.0, 0.6)])

        with (
            patch.object(
                explain_mod, "_sample_entity_ids", new=AsyncMock(return_value=["e1", "e2"])
            ),
            patch.object(
                explain_mod, "explain_prediction", new=AsyncMock(side_effect=[resp1, resp2])
            ),
            patch(
                "src.api.dependencies.compute.heavy_compute_slot",
                new=lambda *a, **k: _FakeSlot(),
            ),
        ):
            agg = await explain_mod._compute_global_importance(
                explain_mod.ModelType.INITIATION,
                "Remibrutinib",
                sample_size=2,
                background_tasks=None,
            )

        feats = {f["feature_name"]: f for f in agg["features"]}
        # Divided by n_succeeded=2 (NOT by feat_n=1) -> 0.6/2 = 0.3, not 0.6.
        assert feats["only_in_1"]["mean_abs_shap"] == pytest.approx(0.3)
        assert feats["only_in_2"]["mean_abs_shap"] == pytest.approx(0.3)
        # Shared feature surfaced in both -> 0.4 either way.
        assert feats["shared_feat"]["mean_abs_shap"] == pytest.approx(0.4)
        # And the shared feature outranks the sparse ones.
        assert agg["features"][0]["feature_name"] == "shared_feat"

    async def test_raises_503_when_no_entity_explains(self):
        from src.api.routes import explain as explain_mod

        with (
            patch.object(
                explain_mod, "_sample_entity_ids", new=AsyncMock(return_value=["e1", "e2"])
            ),
            patch.object(
                explain_mod,
                "explain_prediction",
                new=AsyncMock(side_effect=HTTPException(status_code=503)),
            ),
            patch(
                "src.api.dependencies.compute.heavy_compute_slot",
                new=lambda *a, **k: _FakeSlot(),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await explain_mod._compute_global_importance(
                    explain_mod.ModelType.INITIATION,
                    "Kisqali",
                    sample_size=2,
                    background_tasks=None,
                )
        assert exc.value.status_code == 503


class TestRowToGlobalAgg:
    """Parsing the stored JSONB global row back into the aggregate shape."""

    def test_parses_rich_jsonb_and_sorts(self):
        from src.api.routes.explain import _row_to_global_agg

        row = {
            "global_importance": {
                "region_ne": {
                    "mean_abs_shap": 0.15,
                    "mean_shap": -0.05,
                    "mean_feature_value": 2.0,
                    "contribution_rank": 2,
                    "points": [{"s": -0.2, "v": 1.0}, {"s": 0.1, "v": 3.0}],
                },
                "disease_severity": {
                    "mean_abs_shap": 0.7,
                    "mean_shap": 0.7,
                    "mean_feature_value": 6.0,
                    "contribution_rank": 1,
                    "points": [{"s": 0.6, "v": 5.0}],
                },
            },
            "base_value": -0.8,
            "sample_size": 2,
            "computation_method": "LinearExplainer",
            "computed_at": "2026-06-15T22:00:00+00:00",
        }
        agg = _row_to_global_agg(row)
        assert [f["feature_name"] for f in agg["features"]] == ["disease_severity", "region_ne"]
        assert agg["sample_size"] == 2
        assert agg["base_value"] == pytest.approx(-0.8)
        assert len(agg["points"]["region_ne"]) == 2

    def test_handles_legacy_bare_number_shape(self):
        from src.api.routes.explain import _row_to_global_agg

        agg = _row_to_global_agg({"global_importance": {"age": 0.3}, "sample_size": 100})
        assert agg["features"][0]["feature_name"] == "age"
        assert agg["features"][0]["mean_abs_shap"] == pytest.approx(0.3)
        assert agg["points"]["age"] == []


class TestGlobalEndpointGuards:
    """HTTP guards on GET /api/explain/global."""

    def test_400_for_legacy_non_goldstd_model(self):
        resp = client.get("/api/explain/global", params={"model_type": "propensity"})
        assert resp.status_code == 400

    def test_422_for_unknown_brand(self):
        resp = client.get(
            "/api/explain/global",
            params={"model_type": "initiation", "brand": "NotARealBrand"},
        )
        assert resp.status_code == 422

    def test_cached_read_returns_ranked_features(self):
        stored_row = {
            "global_importance": {
                "disease_severity": {
                    "mean_abs_shap": 0.81,
                    "mean_shap": 0.8,
                    "mean_feature_value": 5.2,
                    "contribution_rank": 1,
                    "points": [{"s": 0.7, "v": 5.0}, {"s": 0.9, "v": 6.0}],
                },
                "region_ne": {
                    "mean_abs_shap": 0.14,
                    "mean_shap": -0.14,
                    "mean_feature_value": 0.3,
                    "contribution_rank": 2,
                    "points": [{"s": -0.1, "v": 0.0}],
                },
            },
            "base_value": -0.72,
            "sample_size": 30,
            "computation_method": "LinearExplainer",
            "computed_at": "2026-06-15T22:00:00+00:00",
        }
        # resolve registry id (1 execute) then load global row (2nd execute)
        fake = _FakeQuery(
            [SimpleNamespace(data=[{"id": "reg-1"}]), SimpleNamespace(data=[stored_row])]
        )
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=fake),
        ):
            resp = client.get(
                "/api/explain/global",
                params={"model_type": "initiation", "brand": "Remibrutinib", "max_points": 10},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cached"] is True
        assert body["model_name"] == "initiation_remibrutinib_goldstd_lr_v1"
        assert body["sample_size"] == 30
        assert body["features"][0]["feature_name"] == "disease_severity"
        assert body["features"][0]["mean_abs_shap"] == pytest.approx(0.81)
        # Real beeswarm points surfaced for the top feature.
        ds_points = [p for p in body["points"] if p["feature_name"] == "disease_severity"]
        assert len(ds_points) == 2


class TestSampleEntities:
    """GET /api/explain/sample-entities."""

    def test_patient_grain(self):
        fake = _FakeQuery(
            [SimpleNamespace(data=[{"patient_id": "scvpt_000000"}, {"patient_id": "scvpt_000001"}])]
        )
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=fake),
        ):
            resp = client.get("/api/explain/sample-entities", params={"model_type": "initiation"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["grain"] == "patient"
        assert body["id_field"] == "patient_id"
        assert body["entities"] == ["scvpt_000000", "scvpt_000001"]

    def test_hcp_grain(self):
        fake = _FakeQuery([SimpleNamespace(data=[{"hcp_id": "scvhcp_00000"}])])
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=fake),
        ):
            resp = client.get("/api/explain/sample-entities", params={"model_type": "hcp_adoption"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["grain"] == "hcp"
        assert body["id_field"] == "hcp_id"
        assert body["entities"] == ["scvhcp_00000"]


# Valid members of the ``data_split_type`` enum (database/core/
# e2i_ml_complete_v3_schema.sql). The durable global-importance cache row writes
# this column; any value outside this set is rejected by Postgres.
_VALID_DATA_SPLIT = {"train", "validation", "test", "holdout", "unassigned"}


class _CapturingInsertQuery:
    """Records the dict passed to ``.insert(...)`` (async ``.execute()``)."""

    def __init__(self, sink):
        self._sink = sink

    def insert(self, record):
        self._sink["record"] = record
        return self

    async def execute(self):
        return SimpleNamespace(data=[self._sink["record"]])

    def table(self, *a, **k):
        return self


class TestGlobalImportanceCachePersists:
    """Regression: the durable global-importance cache must actually persist.

    ``_store_global_importance_row`` previously wrote ``data_split="synthetic"``,
    which is NOT a member of the ``data_split_type`` enum, so every INSERT was
    rejected by Postgres and silently swallowed by the function's broad
    ``except`` — the cache never persisted and ``/explain/global`` recomputed the
    full ~25-entity SHAP aggregate (under the heavy-compute slot) on every load.
    The stored row must use a VALID enum value so the cache works.
    """

    @pytest.mark.asyncio
    async def test_stored_row_uses_valid_data_split_enum(self):
        from src.api.routes.explain import ModelType, _store_global_importance_row

        sink: dict = {}

        async def _fake_client():
            return _CapturingInsertQuery(sink)

        agg = {
            "features": [
                {
                    "feature_name": "disease_severity",
                    "mean_abs_shap": 0.77,
                    "mean_shap": 0.77,
                    "mean_feature_value": 4.6,
                    "contribution_rank": 1,
                }
            ],
            "points": {"disease_severity": [(0.7, 5.0), (0.9, 6.0)]},
            "base_value": -0.91,
            "sample_size": 5,
            "computation_method": "LinearExplainer",
        }
        with patch("src.memory.services.factories.get_async_supabase_client", new=_fake_client):
            await _store_global_importance_row("reg-1", ModelType.INITIATION, agg)

        record = sink.get("record")
        assert record is not None, "no row was inserted (store path broke)"
        assert record["data_split"] in _VALID_DATA_SPLIT, (
            f"data_split={record['data_split']!r} is not a valid data_split_type "
            f"enum member -> Postgres rejects the insert -> cache never persists"
        )
