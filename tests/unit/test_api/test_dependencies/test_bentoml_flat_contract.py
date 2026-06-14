"""Red-first contract tests for the BentoML client flat single-model contract.

Why this file exists
--------------------
A live audit found the FastAPI BentoML client calling **per-model** paths
``{base_url}/{model_name}/healthz|metadata|predict`` which 404 against the
actually-deployed service. The running container (``e2i_bentoml_dev``) serves a
**flat single-model** contract for the one bundled model (``tier0_df99c7ba``):

Verified live (2026-06-14) by probing ``e2i_bentoml_dev`` on port 3000:

  * ``GET  /healthz``      -> 200, empty body                (liveness)
  * ``POST /health``       -> 200, {status, model_loaded, model_tag, ...}
  * ``POST /model_info``   -> 200, {model_id, supported_endpoints, ...}
  * ``POST /predict``      body {"input_data": {"features": [[...]],
                                                "model_type": "classification"}}
                            -> 200, {predictions, probabilities, model_id,
                                     prediction_time_ms, is_mock}
  * ``POST /predict_batch``body {"input_data": {"batch_id": str,
                                                "features": [[...]]}}
                            -> 200, {batch_id, total_samples, predictions,
                                     processing_time_ms, is_mock}

There is **no** ``/{model_name}`` path prefix and **no** ``GET /metadata``
endpoint. These tests pin the client to the real flat contract so the 6+
dashboard panels that depend on it (PredictiveAnalytics, ModelPerformance,
FeatureImportance/SHAP, Monitoring runs/alerts/health) stop 404-ing.

These tests assert the *transport contract* (HTTP verb + path) the client uses,
which is what was broken; they do not exercise the live model (covered by the
opt-in integration test ``test_bentoml_flat_contract_live``).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.api.dependencies.bentoml_client import BentoMLClient, BentoMLClientConfig


@pytest.mark.unit
class TestBentoMLFlatContract:
    """Pin the client to the verified flat single-model contract."""

    def test_endpoint_url_is_flat_base_not_per_model(self):
        """get_endpoint_url must return the flat base, never base/{model_name}.

        The live service serves ONE model at the root; appending the model name
        produces a 404 path. The model name is retained only for tracing /
        circuit-breaker keying, not for URL routing.
        """
        config = BentoMLClientConfig(base_url="http://localhost:3000")
        assert config.get_endpoint_url("csu_treatment_initiation_lr_balanced_v1") == (
            "http://localhost:3000"
        )
        assert config.get_endpoint_url("churn_model") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_predict_posts_flat_predict_path(self):
        """predict() must POST to {base}/predict (flat), not {base}/{model}/predict."""
        client = BentoMLClient(BentoMLClientConfig(base_url="http://localhost:3000"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": [0.0], "probabilities": [0.24]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.predict(
                "csu_treatment_initiation_lr_balanced_v1",
                {"features": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], "model_type": "classification"},
            )
            url = mock_post.call_args.args[0]
            assert url == "http://localhost:3000/predict", url

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_gets_flat_healthz_path(self):
        """health_check() must GET {base}/healthz (flat), not {base}/{model}/healthz."""
        client = BentoMLClient(BentoMLClientConfig(base_url="http://localhost:3000"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            await client.health_check(model_name="csu_treatment_initiation_lr_balanced_v1")
            url = mock_get.call_args.args[0]
            assert url == "http://localhost:3000/healthz", url

        await client.close()

    @pytest.mark.asyncio
    async def test_get_model_info_posts_flat_model_info_path(self):
        """get_model_info() must POST {base}/model_info, not GET {base}/{model}/metadata.

        The live service exposes ``POST /model_info`` (GET /model_info -> 405,
        GET /metadata -> 404). The previous GET /metadata call could never
        succeed against the deployed service.
        """
        client = BentoMLClient(BentoMLClientConfig(base_url="http://localhost:3000"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_id": "tier0_df99c7ba:mxf4urr73odle5xl",
            "supported_endpoints": ["/predict", "/predict_batch", "/health", "/metrics"],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.get_model_info("csu_treatment_initiation_lr_balanced_v1")
            url = mock_post.call_args.args[0]
            assert url == "http://localhost:3000/model_info", url
            assert "supported_endpoints" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_predict_batch_wraps_input_data_for_flat_contract(self):
        """predict_batch() must POST {base}/predict_batch wrapped as input_data.

        The live contract is ``{"input_data": {"batch_id": ..., "features": ...}}``
        — NOT the previous ``{"instances": [...]}`` shape, which fails Pydantic
        validation (422/400) on the deployed service.
        """
        client = BentoMLClient(BentoMLClientConfig(base_url="http://localhost:3000"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "batch_id": "b1",
            "total_samples": 1,
            "predictions": [0.0],
            "processing_time_ms": 10.0,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.predict_batch(
                "csu_treatment_initiation_lr_balanced_v1",
                {"batch_id": "b1", "features": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]},
            )
            url = mock_post.call_args.args[0]
            sent = mock_post.call_args.kwargs["json"]
            assert url == "http://localhost:3000/predict_batch", url
            assert "input_data" in sent, sent
            assert "instances" not in sent, sent

        await client.close()


@pytest.mark.integration
class TestBentoMLFlatContractLive:
    """Opt-in test against the real running e2i_bentoml_dev service.

    Runs only when ``E2I_BENTOML_INTEGRATION=1`` so CI (no live service) skips
    it. Proves the client talks to the actually-deployed flat contract and does
    NOT break the currently-working ``tier0_df99c7ba`` serving.
    """

    @pytest.mark.asyncio
    async def test_live_health_model_info_and_predict(self):
        import os

        if os.environ.get("E2I_BENTOML_INTEGRATION") != "1":
            pytest.skip("set E2I_BENTOML_INTEGRATION=1 to run against the live service")

        client = BentoMLClient(
            BentoMLClientConfig(
                base_url=os.environ.get("BENTOML_SERVICE_URL", "http://localhost:3000")
            )
        )
        try:
            health = await client.health_check()
            assert health["status"] == "healthy", health

            info = await client.get_model_info("tier0_df99c7ba")
            assert info.get("model_loaded") is True, info
            assert "tier0_df99c7ba" in info.get("model_id", ""), info

            # 6-feature vector matches the bundled RandomForestClassifier.
            result = await client.predict(
                "tier0_df99c7ba",
                {"features": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], "model_type": "classification"},
            )
            assert "predictions" in result, result
            assert result.get("is_mock") is False, result
        finally:
            await client.close()
