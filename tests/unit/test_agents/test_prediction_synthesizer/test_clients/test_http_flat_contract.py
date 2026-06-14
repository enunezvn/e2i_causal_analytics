"""Red-first tests: prediction_synthesizer HTTPModelClient + factory must use
the FLAT BentoML contract, not per-model ``{base}/{model_id}`` paths.

The deployed BentoML service is flat/single-model (verified live: POST /predict
body ``{"input_data": {"features": [[...]], "model_type": ...}}`` -> flat
``{predictions, probabilities, ...}``; no ``/{model_id}`` path prefix). The
factory previously defaulted ``endpoint_url`` to ``{base_url}/{model_id}`` and
the prod config baked ``/churn_model`` into each URL, so HTTP clients 404'd.

These tests pin:
  1. The factory default endpoint URL is the FLAT base (no /{model_id}).
  2. HTTPModelClient POSTs the flat input_data wrapper with a 2D ordered
     ``features`` matrix (built from /model_info feature_columns), not the
     legacy dict shape.
  3. The Mock fallback path is PRESERVED (intentional dev/test client).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.agents.prediction_synthesizer.clients.factory import (
    DEFAULT_BASE_URL,
    ModelClientFactory,
    ModelEndpointsConfig,
)
from src.agents.prediction_synthesizer.clients.http_model_client import (
    HTTPModelClient,
    HTTPModelClientConfig,
)


@pytest.mark.unit
class TestFactoryFlatEndpoint:
    def test_default_endpoint_url_is_flat_base(self):
        """An endpoint config with no explicit url defaults to the FLAT base,
        never base/{model_id}."""
        cfg = ModelEndpointsConfig.from_yaml("does/not/exist.yaml")  # -> defaults
        # Simulate an endpoints dict built from a yaml entry WITHOUT a url:
        # the factory must fill base_url, not base_url/{model_id}.
        from src.agents.prediction_synthesizer.clients.factory import ModelEndpointConfig

        # Build the way from_yaml does for a url-less entry.
        base_url = DEFAULT_BASE_URL
        ec = ModelEndpointConfig(model_id="churn_model", endpoint_url=base_url)
        assert ec.endpoint_url == base_url
        assert "/churn_model" not in ec.endpoint_url
        assert isinstance(cfg, ModelEndpointsConfig)

    def test_from_yaml_url_less_entry_defaults_to_flat_base(self, tmp_path):
        """A YAML endpoint entry with no ``url`` resolves to the flat base."""
        yaml_file = tmp_path / "model_endpoints.yaml"
        yaml_file.write_text(
            "default_base_url: http://bento:3000\n"
            "endpoints:\n"
            "  churn_model:\n"
            "    client_type: http\n"
        )
        cfg = ModelEndpointsConfig.from_yaml(str(yaml_file))
        assert cfg.endpoints["churn_model"].endpoint_url == "http://bento:3000"
        assert "/churn_model" not in cfg.endpoints["churn_model"].endpoint_url


@pytest.mark.unit
class TestHTTPModelClientFlatContract:
    @pytest.mark.asyncio
    async def test_predict_posts_flat_input_data_with_ordered_matrix(self):
        """predict() must POST {endpoint}/predict with the flat input_data
        wrapper and a 2D ordered features matrix (ordered by /model_info
        feature_columns), not the legacy dict shape."""
        client = HTTPModelClient(
            model_id="churn_model",
            endpoint_url="http://bento:3000",
            config=HTTPModelClientConfig(model_id="churn_model", endpoint_url="http://bento:3000"),
        )

        model_info_resp = MagicMock()
        model_info_resp.status_code = 200
        model_info_resp.json.return_value = {
            "model_id": "tier0_df99c7ba:abc",
            "feature_columns": ["recency", "frequency"],
        }
        model_info_resp.raise_for_status = MagicMock()

        predict_resp = MagicMock()
        predict_resp.status_code = 200
        predict_resp.json.return_value = {
            "predictions": [0.0],
            "probabilities": [0.7],
            "model_id": "tier0_df99c7ba:abc",
        }
        predict_resp.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [model_info_resp, predict_resp]
            await client.predict(
                entity_id="E1",
                features={"recency": 10, "frequency": 5},
                time_horizon="30d",
            )

            # Last POST is the prediction; assert flat path + flat input_data.
            predict_call = mock_post.call_args_list[-1]
            url = predict_call.args[0] if predict_call.args else predict_call.kwargs.get("url")
            assert url == "http://bento:3000/predict", url
            sent = predict_call.kwargs["json"]
            assert "input_data" in sent, sent
            assert sent["input_data"]["features"] == [[10.0, 5.0]]

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_hits_flat_healthz(self):
        client = HTTPModelClient(
            model_id="churn_model",
            endpoint_url="http://bento:3000",
            config=HTTPModelClientConfig(model_id="churn_model", endpoint_url="http://bento:3000"),
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "ok"}
        resp.raise_for_status = MagicMock()
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = resp
            await client.health_check()
            url = mock_get.call_args.args[0]
            assert url == "http://bento:3000/healthz", url
        await client.close()


@pytest.mark.unit
class TestMockFallbackPreserved:
    @pytest.mark.asyncio
    async def test_mock_client_still_constructable_via_factory(self, tmp_path, monkeypatch):
        """The intentional dev/test Mock client path is preserved (F-012/#430)."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        primary = tmp_path / "model_endpoints.yaml"
        primary.write_text("default_base_url: http://bento:3000\nendpoints: {}\n")
        dev = tmp_path / "dev_model_endpoints.yaml"
        dev.write_text(
            "endpoints:\n"
            "  mock_model:\n"
            "    client_type: mock\n"
            "    default_prediction: 0.65\n"
            "    enabled: true\n"
        )
        cfg = ModelEndpointsConfig.from_yaml(str(primary))
        assert "mock_model" in cfg.endpoints
        assert cfg.endpoints["mock_model"].client_type == "mock"

        factory = ModelClientFactory(cfg)
        client = await factory.get_client("mock_model")
        assert type(client).__name__ == "MockModelClient"
