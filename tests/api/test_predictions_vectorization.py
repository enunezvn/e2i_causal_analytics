"""Red-first tests for feature-dict -> ordered numeric vector conversion.

Why this file exists
--------------------
The live BentoML service's ``PredictionInput.features`` is a POSITIONAL numeric
matrix (``List[List[float]]``), ordered by the model's own ``feature_columns``
(verified live: the bundled tier0_df99c7ba model carries
``feature_columns = ['brand','geographic_region','prior_treatments',
'age_group','hcp_visits','data_quality_score']``). The FastAPI predict routes
previously forwarded a feature *dict* (``{"features": {...}}``) which the
service rejects (it expects a 2D array).

The fix: the route resolves the model's authoritative feature ORDER from the
service's ``/model_info`` (which now exposes ``feature_columns``) and builds the
ordered numeric row. A missing required feature fails CLOSED with a 4xx — no
silent zero-fill, no fabricated vector.

These tests pin:
  1. A feature dict is vectorized into the correct ordered 2D row.
  2. A missing required feature -> honest 422 (not a zero-filled vector).
  3. Extra features in the dict are ignored (only contract columns used).
  4. When the service exposes no feature order, the route fails closed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies.bentoml_client import get_bentoml_client
from src.api.main import app

client = TestClient(app)

# The model's authoritative order (verified live from the bundled model).
TIER0_FEATURE_COLUMNS = [
    "brand",
    "geographic_region",
    "prior_treatments",
    "age_group",
    "hcp_visits",
    "data_quality_score",
]


@pytest.fixture
def vectorizing_client():
    """A BentoML client mock that reports the tier0 feature order via model_info
    and echoes back a successful flat-contract prediction."""
    m = MagicMock()
    m.get_model_info = AsyncMock(
        return_value={
            "model_id": "tier0_df99c7ba:abc",
            "model_loaded": True,
            "feature_columns": TIER0_FEATURE_COLUMNS,
        }
    )
    m.predict = AsyncMock(
        return_value={
            "predictions": [0.0],
            "probabilities": [0.24],
            "model_id": "tier0_df99c7ba:abc",
            "prediction_time_ms": 5.0,
            "is_mock": False,
        }
    )
    return m


@pytest.mark.unit
class TestPredictVectorization:
    def test_feature_dict_is_vectorized_into_ordered_row(self, vectorizing_client):
        """A user-provided feature dict becomes a 2D ordered row in the model's
        feature_columns order."""
        app.dependency_overrides[get_bentoml_client] = lambda: vectorizing_client
        try:
            resp = client.post(
                "/api/models/predict/tier0_df99c7ba",
                json={
                    "features": {
                        "brand": 1.0,
                        "geographic_region": 2.0,
                        "prior_treatments": 3.0,
                        "age_group": 4.0,
                        "hcp_visits": 5.0,
                        "data_quality_score": 6.0,
                    }
                },
            )
            assert resp.status_code == 200, resp.text
            sent = vectorizing_client.predict.call_args[0][1]
            # The route must send a 2D positional matrix in feature_columns order.
            assert sent["features"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
            assert sent.get("model_type") == "classification"
        finally:
            app.dependency_overrides.clear()

    def test_missing_feature_fails_closed_4xx(self, vectorizing_client):
        """A missing required feature must 4xx, NOT zero-fill silently."""
        app.dependency_overrides[get_bentoml_client] = lambda: vectorizing_client
        try:
            resp = client.post(
                "/api/models/predict/tier0_df99c7ba",
                json={
                    "features": {
                        "brand": 1.0,
                        # geographic_region MISSING
                        "prior_treatments": 3.0,
                        "age_group": 4.0,
                        "hcp_visits": 5.0,
                        "data_quality_score": 6.0,
                    }
                },
            )
            assert resp.status_code == 422, resp.text
            assert "geographic_region" in resp.text
            # Must not have called predict over a fabricated vector.
            vectorizing_client.predict.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_extra_features_are_ignored(self, vectorizing_client):
        """Extra keys not in the contract are dropped; only ordered columns sent."""
        app.dependency_overrides[get_bentoml_client] = lambda: vectorizing_client
        try:
            resp = client.post(
                "/api/models/predict/tier0_df99c7ba",
                json={
                    "features": {
                        "brand": 1.0,
                        "geographic_region": 2.0,
                        "prior_treatments": 3.0,
                        "age_group": 4.0,
                        "hcp_visits": 5.0,
                        "data_quality_score": 6.0,
                        "irrelevant_extra": 999.0,
                    }
                },
            )
            assert resp.status_code == 200, resp.text
            sent = vectorizing_client.predict.call_args[0][1]
            assert sent["features"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        finally:
            app.dependency_overrides.clear()

    def test_no_feature_order_available_fails_closed(self, vectorizing_client):
        """If the service exposes no feature order, the route fails closed
        rather than guessing a positional order."""
        vectorizing_client.get_model_info = AsyncMock(
            return_value={"model_id": "x", "model_loaded": True}  # no feature_columns
        )
        app.dependency_overrides[get_bentoml_client] = lambda: vectorizing_client
        try:
            resp = client.post(
                "/api/models/predict/tier0_df99c7ba",
                json={"features": {"brand": 1.0}},
            )
            assert resp.status_code in (422, 503), resp.text
            vectorizing_client.predict.assert_not_called()
        finally:
            app.dependency_overrides.clear()


@pytest.mark.unit
class TestBatchVectorization:
    def test_batch_rows_vectorized_in_order(self, vectorizing_client):
        """Each batch instance's feature dict is vectorized into an ordered row."""
        vectorizing_client.predict_batch = AsyncMock(
            return_value={
                "batch_id": "b1",
                "total_samples": 2,
                "predictions": [0.0, 1.0],
                "processing_time_ms": 10.0,
                "is_mock": False,
                "model_id": "tier0_df99c7ba:abc",
            }
        )
        app.dependency_overrides[get_bentoml_client] = lambda: vectorizing_client
        try:
            full = {
                "brand": 1.0,
                "geographic_region": 2.0,
                "prior_treatments": 3.0,
                "age_group": 4.0,
                "hcp_visits": 5.0,
                "data_quality_score": 6.0,
            }
            resp = client.post(
                "/api/models/predict/tier0_df99c7ba/batch",
                json={"instances": [{"features": full}, {"features": full}]},
            )
            assert resp.status_code == 200, resp.text
            sent = vectorizing_client.predict_batch.call_args[0][1]
            assert sent["features"] == [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        finally:
            app.dependency_overrides.clear()
