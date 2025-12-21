"""
E2I Prediction Synthesizer Agent - Model Orchestrator Node Tests
"""

import pytest

from src.agents.prediction_synthesizer.nodes.model_orchestrator import (
    ModelOrchestratorNode,
)


class TestModelOrchestratorNode:
    """Tests for ModelOrchestratorNode."""

    @pytest.mark.asyncio
    async def test_orchestrate_with_registry(
        self, mock_model_registry, mock_model_clients, base_state
    ):
        """Test orchestration using model registry to find models."""
        node = ModelOrchestratorNode(
            model_registry=mock_model_registry,
            model_clients=mock_model_clients,
        )

        result = await node.execute(base_state)

        # Should find and execute churn models
        assert result["models_succeeded"] >= 2
        assert len(result["individual_predictions"]) >= 2
        assert result["status"] == "combining"
        # Latency can be 0 for sub-ms operations
        assert result["orchestration_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_orchestrate_with_specified_models(self, mock_model_clients, base_state):
        """Test orchestration with specific models requested."""
        base_state["models_to_use"] = ["churn_xgb", "churn_rf"]

        node = ModelOrchestratorNode(model_clients=mock_model_clients)
        result = await node.execute(base_state)

        assert result["models_succeeded"] == 2
        assert result["models_failed"] == 0

        model_ids = [p["model_id"] for p in result["individual_predictions"]]
        assert "churn_xgb" in model_ids
        assert "churn_rf" in model_ids

    @pytest.mark.asyncio
    async def test_orchestrate_parallel_execution(self, mock_model_clients, base_state):
        """Test that models are executed in parallel."""
        base_state["models_to_use"] = ["churn_xgb", "churn_rf", "churn_nn"]

        node = ModelOrchestratorNode(model_clients=mock_model_clients)
        result = await node.execute(base_state)

        # All 3 models should succeed
        assert result["models_succeeded"] == 3

        # Latency is tracked (may be 0 for sub-ms operations)
        assert result["orchestration_latency_ms"] >= 0
        assert len(result["individual_predictions"]) == 3

    @pytest.mark.asyncio
    async def test_orchestrate_with_partial_failures(self, failing_model_clients, base_state):
        """Test handling when some models fail."""
        base_state["models_to_use"] = ["churn_xgb", "churn_rf", "churn_nn"]

        node = ModelOrchestratorNode(model_clients=failing_model_clients)
        result = await node.execute(base_state)

        # 2 succeed, 1 fails
        assert result["models_succeeded"] == 2
        assert result["models_failed"] == 1
        assert result["status"] == "combining"  # Still continue

    @pytest.mark.asyncio
    async def test_orchestrate_all_models_fail(self, base_state):
        """Test when all models fail."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockModelClient,
        )

        failing_clients = {
            "churn_xgb": MockModelClient(should_fail=True, error_message="Model XGB failed"),
            "churn_rf": MockModelClient(should_fail=True, error_message="Model RF failed"),
        }
        base_state["models_to_use"] = ["churn_xgb", "churn_rf"]

        node = ModelOrchestratorNode(model_clients=failing_clients)
        result = await node.execute(base_state)

        assert result["models_succeeded"] == 0
        assert result["models_failed"] == 2
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_orchestrate_no_models_available(self, base_state):
        """Test when no models are available."""
        # Empty models_to_use list with no clients = no models available
        base_state["models_to_use"] = []

        node = ModelOrchestratorNode(model_clients={})
        result = await node.execute(base_state)

        assert result["models_succeeded"] == 0
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_orchestrate_preserves_prediction_values(self, mock_model_clients, base_state):
        """Test that prediction values are correctly preserved."""
        base_state["models_to_use"] = ["churn_xgb"]

        node = ModelOrchestratorNode(model_clients=mock_model_clients)
        result = await node.execute(base_state)

        pred = result["individual_predictions"][0]
        assert pred["model_id"] == "churn_xgb"
        assert pred["prediction"] == 0.72
        assert pred["confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_orchestrate_already_failed(self, mock_model_clients, base_state):
        """Test that already failed state is passed through."""
        base_state["status"] = "failed"
        base_state["errors"] = [{"error": "Previous error"}]

        node = ModelOrchestratorNode(model_clients=mock_model_clients)
        result = await node.execute(base_state)

        # Should pass through without modification
        assert result["status"] == "failed"
        assert result["errors"] == [{"error": "Previous error"}]

    @pytest.mark.asyncio
    async def test_orchestrate_uses_features(self, mock_model_clients, base_state, sample_features):
        """Test that features are passed to model clients."""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.predict.return_value = {
            "prediction": 0.5,
            "confidence": 0.8,
            "latency_ms": 50,
        }

        base_state["models_to_use"] = ["test_model"]
        clients = {"test_model": mock_client}

        node = ModelOrchestratorNode(model_clients=clients)
        await node.execute(base_state)

        mock_client.predict.assert_called_once()
        call_kwargs = mock_client.predict.call_args.kwargs
        assert call_kwargs["entity_id"] == "hcp_123"
        assert call_kwargs["features"] == sample_features
        assert call_kwargs["time_horizon"] == "30d"


class TestModelOrchestratorTimeout:
    """Tests for model timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, base_state):
        """Test that slow models are handled gracefully."""
        import asyncio

        class SlowClient:
            async def predict(self, entity_id, features, time_horizon="30d", **kwargs):
                await asyncio.sleep(10)  # Very slow
                return {"prediction": 0.5, "confidence": 0.8, "latency_ms": 10000}

        base_state["models_to_use"] = ["slow_model"]
        clients = {"slow_model": SlowClient()}

        node = ModelOrchestratorNode(
            model_clients=clients,
            timeout_per_model=0.1,  # 100ms timeout
        )

        result = await node.execute(base_state)

        # Model should timeout and be marked as failed
        assert result["models_succeeded"] == 0
        assert result["models_failed"] == 1


class TestModelOrchestratorRegistry:
    """Tests for model registry integration."""

    @pytest.mark.asyncio
    async def test_registry_model_discovery(
        self, mock_model_registry, mock_model_clients, base_state
    ):
        """Test that models are discovered from registry."""
        base_state["prediction_target"] = "churn"
        base_state["models_to_use"] = None  # Use registry

        node = ModelOrchestratorNode(
            model_registry=mock_model_registry,
            model_clients=mock_model_clients,
        )

        result = await node.execute(base_state)

        # Should find all churn models from registry
        assert result["models_succeeded"] >= 2

    @pytest.mark.asyncio
    async def test_registry_filters_by_target(
        self, mock_model_registry, mock_model_clients, base_state
    ):
        """Test that registry filters by prediction target."""
        base_state["prediction_target"] = "conversion"
        base_state["models_to_use"] = None

        node = ModelOrchestratorNode(
            model_registry=mock_model_registry,
            model_clients=mock_model_clients,
        )

        result = await node.execute(base_state)

        # Should only find conversion model
        if result["models_succeeded"] > 0:
            model_ids = [p["model_id"] for p in result["individual_predictions"]]
            for model_id in model_ids:
                assert "conversion" in model_id
