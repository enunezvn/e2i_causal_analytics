"""Tests for F-012 (#430): orchestrator and factory fail-closed on missing mocks.

The orchestrator previously fell silently through to ``_create_mock_prediction``
when ``self.clients.get(model_id)`` returned None — returning randomized values
with no error signal. The factory previously merged ``mock_model`` from
``config/model_endpoints.yaml`` in all environments, including production.

These tests pin the new contract:

* No client + ``ALLOW_MOCK_MODEL`` unset → ValueError raised.
* No client + ``ALLOW_MOCK_MODEL=1`` → legacy mock prediction returned.
* Dev YAML merged when ENVIRONMENT != production.
* Dev YAML NOT merged when ENVIRONMENT == production.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.prediction_synthesizer.clients.factory import (
    ModelClientFactory,
    ModelEndpointConfig,
    ModelEndpointsConfig,
)
from src.agents.prediction_synthesizer.nodes.model_orchestrator import (
    ModelOrchestratorNode,
)


@pytest.fixture
def base_state():
    """Minimal PredictionSynthesizerState compatible dict."""
    return {
        "entity_id": "hcp_123",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "features": {"f1": 1.0},
        "time_horizon": "30d",
        "models_to_use": [],
        "errors": [],
        "warnings": [],
    }


class TestOrchestratorFailClosed:
    @pytest.mark.asyncio
    async def test_raises_value_error_when_client_missing_and_mock_disabled(self, base_state):
        """Default (ALLOW_MOCK_MODEL unset) → ValueError, not silent mock."""
        base_state["models_to_use"] = ["unregistered_model"]
        node = ModelOrchestratorNode(model_clients={})

        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("ALLOW_MOCK_MODEL", None)
            result = await node.execute(base_state)

        # Orchestrator catches inner exceptions and reports failure in state
        # rather than re-raising. Verify the failure surface carries the
        # explicit ValueError message — not a fabricated mock prediction.
        assert result["models_failed"] == 1
        assert result["models_succeeded"] == 0
        warnings = result.get("warnings", [])
        assert any(
            "unregistered_model" in w and "no registered client" in w.lower() for w in warnings
        ), f"Expected ValueError surface in warnings, got: {warnings}"

    @pytest.mark.asyncio
    async def test_allows_mock_when_env_gate_set(self, base_state):
        """ALLOW_MOCK_MODEL=1 → legacy mock prediction path remains available."""
        base_state["models_to_use"] = ["mock_model"]
        node = ModelOrchestratorNode(model_clients={})

        with patch.dict("os.environ", {"ALLOW_MOCK_MODEL": "1"}, clear=False):
            result = await node.execute(base_state)

        assert result["models_succeeded"] == 1
        assert result["models_failed"] == 0
        predictions = result["individual_predictions"]
        assert len(predictions) == 1
        assert predictions[0]["model_id"] == "mock_model"
        assert predictions[0]["model_type"] == "mock"

    @pytest.mark.asyncio
    async def test_no_clients_no_models_to_use_no_silent_mock(self, base_state):
        """Empty clients + empty models_to_use → graceful failure, no mock."""
        base_state["models_to_use"] = []
        node = ModelOrchestratorNode(model_clients={})

        result = await node.execute(base_state)

        assert result["status"] == "failed"
        # No fabricated predictions
        assert len(result.get("individual_predictions", [])) == 0


class TestFactoryConfigSplit:
    """Verify the dev YAML is loaded conditionally on ENVIRONMENT."""

    def _write_configs(self, tmpdir: Path) -> Path:
        """Create prod + dev yaml fixtures in a temp directory."""
        prod_yaml = tmpdir / "model_endpoints.yaml"
        prod_yaml.write_text(
            """default_base_url: http://localhost:3000
default_timeout: 5.0
default_max_retries: 3
endpoints:
  churn_model:
    url: http://localhost:3000/churn_model
    client_type: http
    enabled: true
"""
        )
        dev_yaml = tmpdir / "dev_model_endpoints.yaml"
        dev_yaml.write_text(
            """endpoints:
  mock_model:
    client_type: mock
    default_prediction: 0.65
    default_confidence: 0.85
    enabled: true
"""
        )
        return prod_yaml

    def test_dev_yaml_loaded_in_development(self):
        """ENVIRONMENT=development → dev_model_endpoints.yaml IS merged."""
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            with patch.dict("os.environ", {"ENVIRONMENT": "development"}, clear=False):
                config = ModelEndpointsConfig.from_yaml(str(prod_path))

        # Both prod (churn_model) and dev (mock_model) entries present
        assert "churn_model" in config.endpoints
        assert "mock_model" in config.endpoints
        assert config.endpoints["mock_model"].client_type == "mock"

    @pytest.mark.parametrize("value", ["development", "dev", "test", "testing", "local"])
    def test_full_allowlist_matrix(self, value):
        """Codex iter-2 M2: pin full _KNOWN_DEV_ENVIRONMENTS matrix to
        prevent silent drift in this site relative to agent_import_guard
        and cate_estimator.
        """
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            with patch.dict("os.environ", {"ENVIRONMENT": value}, clear=False):
                config = ModelEndpointsConfig.from_yaml(str(prod_path))
        assert "mock_model" in config.endpoints

    def test_dev_yaml_skipped_in_production(self):
        """ENVIRONMENT=production → dev_model_endpoints.yaml NOT merged.

        This is the core F-012 fix: production planners that name
        model_id='mock_model' must hit the fail-closed path in the
        orchestrator instead of being silently routed to MockModelClient.
        """
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=False):
                config = ModelEndpointsConfig.from_yaml(str(prod_path))

        assert "churn_model" in config.endpoints
        assert "mock_model" not in config.endpoints

    def test_dev_yaml_skipped_when_environment_unset(self):
        """Codex iter-1 H1: ENVIRONMENT unset MUST NOT merge dev yaml.

        Missing deployment metadata must not silently enable mock clients
        in what is potentially a production environment.
        """
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            with patch.dict("os.environ", {}, clear=False):
                import os as _os

                _os.environ.pop("ENVIRONMENT", None)
                config = ModelEndpointsConfig.from_yaml(str(prod_path))

        assert "churn_model" in config.endpoints
        assert "mock_model" not in config.endpoints

    def test_dev_yaml_skipped_when_environment_misspelled(self):
        """Codex iter-1 H1: misspelled ENVIRONMENT does NOT merge dev yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            with patch.dict("os.environ", {"ENVIRONMENT": "develoment"}, clear=False):  # typo
                config = ModelEndpointsConfig.from_yaml(str(prod_path))
        assert "mock_model" not in config.endpoints

    def test_dev_yaml_missing_is_safe(self):
        """When dev_model_endpoints.yaml absent in dev, loader still succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = Path(tmp) / "model_endpoints.yaml"
            prod_path.write_text(
                """default_base_url: http://localhost:3000
default_timeout: 5.0
default_max_retries: 3
endpoints:
  churn_model:
    url: http://localhost:3000/churn_model
    client_type: http
    enabled: true
"""
            )
            with patch.dict("os.environ", {"ENVIRONMENT": "development"}, clear=False):
                config = ModelEndpointsConfig.from_yaml(str(prod_path))

        assert "churn_model" in config.endpoints
        assert "mock_model" not in config.endpoints


class TestFactoryRequiresExplicitEndpoint:
    """Codex iter-1 H2: factory.get_client must NOT synthesize an HTTP URL
    for undeclared model_ids. Previously, asking for an unknown model_id
    silently constructed ``HTTPModelClient(endpoint_url=base/model_id)``
    — defeating the F-012 config split because removing ``mock_model``
    from the prod yaml didn't actually prevent that name from being
    served.
    """

    @pytest.mark.asyncio
    async def test_raises_on_undeclared_model_id(self):
        config = ModelEndpointsConfig(
            endpoints={
                "churn_model": ModelEndpointConfig(
                    model_id="churn_model",
                    endpoint_url="http://localhost:3000/churn_model",
                    client_type="http",
                )
            },
        )
        factory = ModelClientFactory(config)

        with pytest.raises(ValueError) as exc_info:
            await factory.get_client("mock_model")
        assert "no endpoint declaration" in str(exc_info.value).lower()
        assert "mock_model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_client_for_declared_model(self):
        """Declared model still works."""
        config = ModelEndpointsConfig(
            endpoints={
                "ok_model": ModelEndpointConfig(
                    model_id="ok_model",
                    endpoint_url="http://localhost:3000/ok_model",
                    client_type="mock",
                    default_prediction=0.5,
                )
            },
        )
        factory = ModelClientFactory(config)
        client = await factory.get_client("ok_model")
        assert client is not None

    @pytest.mark.asyncio
    async def test_get_clients_re_raises_undeclared_value_error(self):
        """Codex iter-2 M1: get_clients() must NOT swallow ValueError from
        an undeclared model_id at the batch entry point. Previously the
        broad except suppressed the fail-loud signal we just installed
        in get_client().
        """
        config = ModelEndpointsConfig(
            endpoints={
                "ok_model": ModelEndpointConfig(
                    model_id="ok_model",
                    endpoint_url="http://localhost:3000/ok_model",
                    client_type="mock",
                    default_prediction=0.5,
                )
            },
        )
        factory = ModelClientFactory(config)
        # Asking for one good + one undeclared should still fail loud.
        with pytest.raises(ValueError):
            await factory.get_clients(["ok_model", "undeclared_model"])
