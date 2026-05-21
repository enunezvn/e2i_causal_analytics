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

from src.agents.prediction_synthesizer.clients.factory import ModelEndpointsConfig
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

    def test_dev_yaml_loaded_when_environment_unset(self):
        """ENVIRONMENT unset defaults to development → dev yaml IS merged."""
        with tempfile.TemporaryDirectory() as tmp:
            prod_path = self._write_configs(Path(tmp))
            # Remove ENVIRONMENT to test default behavior
            with patch.dict("os.environ", {}, clear=False):
                import os as _os

                _os.environ.pop("ENVIRONMENT", None)
                config = ModelEndpointsConfig.from_yaml(str(prod_path))

        assert "mock_model" in config.endpoints

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
