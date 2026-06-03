"""
Tests for Predictions API endpoints.

Phase 2D of API Audit - Model Predictions API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2D.1: Inference (POST /api/models/predict/{model}, POST /api/models/predict/{model}/batch, GET /api/models/{model}/info)
- Batch 2D.2: Health (GET /api/models/{model}/health, GET /api/models/status)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def mock_feast_client(monkeypatch):
    """Mock FeastClient and patch the route's resolver to return it.

    The default ``get_online_features`` return mirrors Feast's column-oriented
    output for a single entity (one value per feature). Tests override
    ``side_effect`` / ``return_value`` when they need different shapes or
    failure modes.

    Patches ``src.api.routes.predictions._resolve_feast_client`` directly
    rather than going through ``app.dependency_overrides`` because the route
    intentionally fetches Feast lazily inside the function body (to avoid a
    FastAPI body-vs-dependency disambiguation issue with multiple Pydantic
    parameters). See the ``_resolve_feast_client`` docstring for context.
    """
    feast_mock = MagicMock()
    feast_mock.get_online_features = AsyncMock(
        return_value={
            "days_since_last_hcp_visit": [12.0],
            "total_hcp_interactions_90d": [5.0],
            "therapy_adherence_score": [0.83],
        }
    )

    async def _fake_resolver():
        return feast_mock

    monkeypatch.setattr("src.api.routes.predictions._resolve_feast_client", _fake_resolver)
    return feast_mock


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

    def test_predict_preserves_falsy_zero_prediction(self, mock_bentoml_client):
        """A legitimate prediction value of 0.0 must NOT be dropped by an ``or``.

        Regression guard: the route built the response with
        ``result.get("prediction") or result.get("predictions", [None])[0]``.
        With ``or``, a falsy-but-valid scalar (0, 0.0, False) short-circuits to
        the ``predictions`` fallback (or None), corrupting the response. A
        binary classifier emitting class 0, or a regressor emitting exactly
        0.0, is a real, common case for pharma uplift/propensity models.
        """
        mock_bentoml_client.predict = AsyncMock(
            return_value={
                "prediction": 0.0,
                # If the buggy ``or`` runs, it falls through to predictions[0]
                # which would yield the WRONG value (0.99), proving the bug.
                "predictions": [0.99],
                "confidence": 0.5,
                "_metadata": {"latency_ms": 1.0},
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.0

    def test_predict_falls_back_to_predictions_when_prediction_absent(self, mock_bentoml_client):
        """When ``prediction`` key is missing entirely, fall back to predictions[0]."""
        mock_bentoml_client.predict = AsyncMock(
            return_value={
                "predictions": [0.73],
                "confidence": 0.8,
                "_metadata": {"latency_ms": 1.0},
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/churn_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 200
        assert response.json()["prediction"] == 0.73

    def test_predict_with_entity_id(self, mock_bentoml_client, mock_feast_client):
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
        # Verify entity_id was passed through to BentoML for downstream telemetry
        call_args = mock_bentoml_client.predict.call_args
        assert "entity_id" in call_args[0][1]

    def test_predictions_uses_online_features_when_entity_id_present(
        self, mock_bentoml_client, mock_feast_client
    ):
        """When entity_id is present, the route fetches features from Feast.

        Asserts:
          - FeastClient.get_online_features is called once with the request's
            entity_id (mapped under 'patient_id') and a non-empty feature_refs
            list.
          - The fetched features replace the request body's ``features`` dict
            in the BentoML payload (the dict sent to BentoML carries the Feast
            values, not the original empty dict).
          - The response carries ``feature_source='feast_online'`` and the
            BentoML input also carries that telemetry tag.
        """
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["feature_source"] == "feast_online"

        # Feast was called with the patient_id mapping + a real feature_refs list.
        mock_feast_client.get_online_features.assert_awaited_once()
        feast_kwargs = mock_feast_client.get_online_features.await_args.kwargs
        assert feast_kwargs["entity_rows"] == [{"patient_id": "PAT-2024-0001"}]
        assert isinstance(feast_kwargs["feature_refs"], list)
        assert len(feast_kwargs["feature_refs"]) > 0
        assert feast_kwargs["full_feature_names"] is False

        # BentoML received the resolved Feast features, not the empty dict.
        bento_payload = mock_bentoml_client.predict.call_args[0][1]
        assert bento_payload["feature_source"] == "feast_online"
        assert bento_payload["features"] == {
            "days_since_last_hcp_visit": 12.0,
            "total_hcp_interactions_90d": 5.0,
            "therapy_adherence_score": 0.83,
        }

    def test_predictions_user_provided_when_no_entity_id(
        self, mock_bentoml_client, mock_feast_client
    ):
        """Without entity_id, Feast is NOT called and tag is 'user_provided'."""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/churn_model",
            json={"features": {"hcp_id": "HCP001"}},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["feature_source"] == "user_provided"

        # Feast must not be invoked when entity_id is absent.
        mock_feast_client.get_online_features.assert_not_called()

        bento_payload = mock_bentoml_client.predict.call_args[0][1]
        assert bento_payload["features"] == {"hcp_id": "HCP001"}
        assert bento_payload["feature_source"] == "user_provided"

    def test_predictions_feast_wins_when_both_features_and_entity_id_provided(
        self, mock_bentoml_client, mock_feast_client
    ):
        """3A-M-5: route docstring says ``Both set -> Feast wins``.

        When the caller supplies BOTH ``features`` (a non-empty dict) AND
        ``entity_id``, the route MUST honour the documented precedence:
        Feast lookup happens, the supplied features dict is OVERWRITTEN
        with the Feast values, and the response is tagged
        ``feature_source='feast_online'``. This test pins that contract.
        """
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={
                # Caller-supplied features that should be ignored in favour
                # of Feast lookup.
                "features": {
                    "days_since_last_hcp_visit": 99999.0,
                    "stale_value": "ignored",
                },
                "entity_id": "PAT-2024-0042",
            },
        )

        assert response.status_code == 200
        body = response.json()
        # Feast wins — response telemetry tag reflects Feast invocation.
        assert body["feature_source"] == "feast_online"

        # Feast.get_online_features was actually invoked.
        mock_feast_client.get_online_features.assert_awaited_once()

        # The payload sent to BentoML carries the Feast-resolved features,
        # NOT the caller-supplied ones — proving the override happened.
        bento_payload = mock_bentoml_client.predict.call_args[0][1]
        assert bento_payload["feature_source"] == "feast_online"
        assert bento_payload["features"] == {
            "days_since_last_hcp_visit": 12.0,
            "total_hcp_interactions_90d": 5.0,
            "therapy_adherence_score": 0.83,
        }
        # Defensive: ensure the stale caller key is gone.
        assert "stale_value" not in bento_payload["features"]

    def test_predictions_feature_source_route_is_source_of_truth(
        self, mock_bentoml_client, mock_feast_client
    ):
        """3A-I-3: route, not BentoML, owns ``feature_source``.

        Even if BentoML returns a value for ``feature_source`` in its
        response (e.g. the BentoML container falls back to a different
        path for any reason), the route's response MUST report what the
        route did — which here is ``'feast_online'`` because the request
        carried an ``entity_id`` and Feast was invoked.
        """
        # Mock BentoML returning a CONTRADICTORY feature_source in its
        # response — the route must ignore it.
        mock_bentoml_client.predict = AsyncMock(
            return_value={
                "prediction": 0.5,
                "model_version": "v1.0",
                "feature_source": "user_provided",  # contradicts the route
                "_metadata": {"latency_ms": 5.0, "timestamp": "2026-04-26T00:00:00+00:00"},
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 200
        body = response.json()
        # Route says feast_online — wins over BentoML's "user_provided".
        assert body["feature_source"] == "feast_online"

    def test_predictions_feast_failure_returns_503(self, mock_bentoml_client, mock_feast_client):
        """Feast errors surface as 503 — the route does not silently swallow them.

        The exact error envelope shape comes from the project's custom
        exception handlers; we assert the status code and that BentoML was
        never invoked (the truly load-bearing behavior — no silent fallback
        to 'user_provided' on Feast failure).
        """
        mock_feast_client.get_online_features = AsyncMock(
            side_effect=Exception("Feast connection refused")
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 503
        # BentoML must not have been called when Feast fails.
        mock_bentoml_client.predict.assert_not_called()

    def test_predict_circuit_breaker_open(self, mock_bentoml_client):
        """Should return 503 when circuit breaker is open."""
        mock_bentoml_client.predict = AsyncMock(side_effect=RuntimeError("Circuit breaker open"))

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/failing_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 503
        # The app's exception handler maps the route's 503 (RuntimeError / open
        # circuit breaker) into a generic DependencyError envelope
        # ({"error": "DependencyError", "message": "Dependency 'service' is
        # unavailable", ...}); it does NOT surface the "Circuit breaker open"
        # detail, and uses an E2IError envelope, not FastAPI's {"detail": ...}.
        # Assert the load-bearing contract: 503 + service-unavailable. (This
        # assertion silently rotted while tests/api/ was unwired from CI.)
        body = response.json()
        assert "unavailable" in str(body.get("message") or body.get("detail") or body).lower()

    def test_predict_internal_error(self, mock_bentoml_client):
        """Should return 500 for other prediction failures."""
        mock_bentoml_client.predict = AsyncMock(side_effect=Exception("Model inference failed"))

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client
        response = client.post(
            "/api/models/predict/broken_model",
            json={"features": {"x": 1}},
        )

        assert response.status_code == 500
        body = response.json()
        assert "prediction failed" in str(body.get("message") or body.get("detail") or body).lower()

    def test_predict_does_not_label_feast_online_on_fallback_error(
        self, mock_bentoml_client, mock_feast_client
    ):
        """#532 route-intrinsic honesty: when the FeastClient refuses to serve a
        custom-store fallback (it raises FeastFallbackError — the production guard
        for the embedded online fallback), the route surfaces 503 and must NOT tag
        the response feature_source='feast_online'. The route never labels data it
        did not actually fetch from the Feast online store.
        """
        from src.feature_store.feast_client import FeastFallbackError

        mock_feast_client.get_online_features = AsyncMock(
            side_effect=FeastFallbackError(
                "Feast online features unavailable; fallback is forbidden in production."
            )
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 503
        mock_bentoml_client.predict.assert_not_called()
        assert "feast_online" not in str(response.json()).lower()

    def test_predict_entity_id_fails_loud_when_sidecar_unreachable(
        self, monkeypatch, mock_bentoml_client
    ):
        """#532 end-to-end on the no-feast app image: with ``import feast``
        unavailable and a remote FEAST_URL configured, an entity_id prediction
        whose sidecar fetch fails must return 503 — never a silent fallback
        mislabeled feast_online, and BentoML must not be invoked.

        Exercises the REAL FeastClient remote path (only the httpx transport is
        mocked) — the route->client->fail-loud seam that the importorskip-gated
        live smoke (tests/integration/test_feast_remote_online_smoke.py) cannot
        cover in CI.
        """
        import builtins

        import httpx

        from src.feature_store.feast_client import FeastClient, FeastConfig

        # Faithfully simulate the production app image: feast cannot be imported.
        real_import = builtins.__import__

        def _no_feast_import(name, *args, **kwargs):
            if name == "feast" or name.startswith("feast."):
                raise ImportError("No module named 'feast' (app image)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_feast_import)

        # Real remote-mode client: records FEAST_URL, never imports feast.
        real_client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))

        async def _resolver():
            return real_client

        monkeypatch.setattr("src.api.routes.predictions._resolve_feast_client", _resolver)

        # Sidecar unreachable: the POST raises a transport error (mirrors the
        # proven mock in test_feast_client.py::test_remote_fails_loud_on_sidecar_error).
        mock_httpx_client = MagicMock()
        mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        cm.__aexit__ = AsyncMock(return_value=None)

        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            response = client.post(
                "/api/models/predict/propensity",
                json={"features": {}, "entity_id": "PAT-2024-0001"},
            )

        assert response.status_code == 503
        mock_bentoml_client.predict.assert_not_called()
        assert "feast_online" not in str(response.json()).lower()


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


# =============================================================================
# #576 - ANTI-NULL-TRAP GUARD (predictions route)
#
# A Feast feature-server 200 response can carry PRESENT-but-null values
# (verified LIVE against the prod sidecar: a single-key patient lookup against
# patient_journey_features returns status=PRESENT with value=null when the
# composite key is absent or the online store is empty). Labeling such a
# response feature_source='feast_online' feeds null features to the model while
# presenting them as real — the exact #532 harm. The route MUST fail loud (503)
# instead of mislabeling. A real 0/0.0 is a legitimate value and is NOT a
# violation (the COALESCE-0 source-masking concern is a data-layer issue).
# =============================================================================


class TestPredictFeastNullGuard:
    """POST /api/models/predict/{model} must not label a null Feast response feast_online."""

    def test_predict_feast_all_null_returns_503(self, mock_bentoml_client, mock_feast_client):
        """All required Feast feature values null -> 503, never feast_online,
        BentoML never invoked (no prediction over a null vector)."""
        mock_feast_client.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-0001"],  # entity key echoed back; must be ignored
                "days_since_last_hcp_visit": [None],
                "total_hcp_interactions_90d": [None],
                "therapy_adherence_score": [None],
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 503
        mock_bentoml_client.predict.assert_not_called()

    def test_predict_feast_partial_null_required_field_returns_503(
        self, mock_bentoml_client, mock_feast_client
    ):
        """A REQUIRED served feature being null (while others are present) is
        still fail-loud: a partial/mixed null vector must not be labeled real.
        Pins the live PAT_000191 finding (days_on_therapy real, adherence_rate null)."""
        mock_feast_client.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-0001"],
                "days_since_last_hcp_visit": [12.0],  # present + real
                "total_hcp_interactions_90d": [None],  # REQUIRED but null
                "therapy_adherence_score": [0.83],
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 503
        mock_bentoml_client.predict.assert_not_called()

    def test_predict_feast_all_present_nonnull_sets_feast_online(
        self, mock_bentoml_client, mock_feast_client
    ):
        """The honest success path: all required features present + non-null ->
        200, feature_source='feast_online', prediction served. (default mock_feast_client
        returns the three propensity fields non-null.)"""
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 200
        assert response.json()["feature_source"] == "feast_online"
        mock_bentoml_client.predict.assert_called_once()

    def test_predict_feast_zero_value_is_not_treated_as_null(
        self, mock_bentoml_client, mock_feast_client
    ):
        """A real 0.0 is a legitimate feature value, NOT a null -> the guard must
        NOT fire on it (else we'd 503 on perfectly valid zero-valued features)."""
        mock_feast_client.get_online_features = AsyncMock(
            return_value={
                "patient_id": ["PAT-2024-0001"],
                "days_since_last_hcp_visit": [0.0],
                "total_hcp_interactions_90d": [0.0],
                "therapy_adherence_score": [0.0],
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: mock_bentoml_client

        response = client.post(
            "/api/models/predict/propensity",
            json={"features": {}, "entity_id": "PAT-2024-0001"},
        )

        assert response.status_code == 200
        assert response.json()["feature_source"] == "feast_online"


class TestOnlineFeaturePresenceHelper:
    """Pure helper that backs the #576 anti-null-trap guard."""

    def test_required_feature_fields_strips_view_prefix(self):
        from src.feature_store.online_feature_presence import required_feature_fields

        refs = [
            "patient_journey_features:days_on_therapy",
            "patient_journey_features:adherence_rate",
        ]
        assert required_feature_fields(refs) == ["days_on_therapy", "adherence_rate"]

    def test_required_feature_fields_skips_wildcard(self):
        from src.feature_store.online_feature_presence import required_feature_fields

        assert required_feature_fields(["v:*"]) == []

    def test_missing_or_null_flags_null_and_absent_not_zero(self):
        from src.feature_store.online_feature_presence import missing_or_null_feature_fields

        refs = ["v:a", "v:b", "v:c", "v:d"]
        # a present+real, b null, c absent, d real-zero. patient_id echo ignored.
        payload = {"patient_id": "X", "a": 1.0, "b": None, "d": 0.0}
        assert set(missing_or_null_feature_fields(payload, refs)) == {"b", "c"}

    def test_all_present_nonnull_returns_empty(self):
        from src.feature_store.online_feature_presence import missing_or_null_feature_fields

        refs = ["v:a", "v:b"]
        assert missing_or_null_feature_fields({"a": 1.0, "b": 0.0}, refs) == []
