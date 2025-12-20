"""
E2I Prediction Synthesizer Agent - Integration Tests
"""

import pytest
from src.agents.prediction_synthesizer import (
    PredictionSynthesizerAgent,
    PredictionSynthesizerInput,
    PredictionSynthesizerOutput,
    synthesize_predictions,
    build_prediction_synthesizer_graph,
)


class TestPredictionSynthesizerAgent:
    """Integration tests for PredictionSynthesizerAgent."""

    @pytest.mark.asyncio
    async def test_full_synthesis_pipeline(
        self,
        mock_model_registry,
        mock_model_clients,
        mock_context_store,
        mock_feature_store,
        sample_features,
    ):
        """Test complete prediction synthesis pipeline."""
        agent = PredictionSynthesizerAgent(
            model_registry=mock_model_registry,
            model_clients=mock_model_clients,
            context_store=mock_context_store,
            feature_store=mock_feature_store,
        )

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf"],
            ensemble_method="weighted",
            include_context=True,
        )

        assert isinstance(result, PredictionSynthesizerOutput)
        assert result.status == "completed"
        assert result.models_succeeded == 2
        assert result.models_failed == 0
        assert result.ensemble_prediction is not None
        assert result.prediction_context is not None
        # Latency can be 0 if operations complete in sub-milliseconds
        assert result.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_quick_predict(
        self, mock_model_clients, sample_features
    ):
        """Test quick prediction without context."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        result = await agent.quick_predict(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
        )

        assert result.status in ["completed", "failed"]
        # Context should not be populated for quick predict
        assert result.prediction_context is None

    @pytest.mark.asyncio
    async def test_synthesis_with_partial_failures(
        self, failing_model_clients, sample_features
    ):
        """Test synthesis with some model failures."""
        agent = PredictionSynthesizerAgent(model_clients=failing_model_clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf", "churn_nn"],
            include_context=False,
        )

        # Should complete with partial results
        assert result.models_succeeded == 2
        assert result.models_failed == 1
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_handoff_generation(
        self, mock_model_clients, sample_features
    ):
        """Test handoff generation for orchestrator."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf"],
            include_context=False,
        )

        handoff = agent.get_handoff(result)

        assert handoff["agent"] == "prediction_synthesizer"
        assert handoff["analysis_type"] == "prediction"
        assert "key_findings" in handoff
        assert "prediction" in handoff["key_findings"]
        assert "models" in handoff
        assert handoff["models"]["succeeded"] == 2

    @pytest.mark.asyncio
    async def test_handoff_with_low_confidence(
        self, sample_features
    ):
        """Test handoff recommendations for low confidence."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockModelClient,
        )

        clients = {
            "model_1": MockModelClient(prediction=0.9, confidence=0.3),
            "model_2": MockModelClient(prediction=0.1, confidence=0.2),
        }

        agent = PredictionSynthesizerAgent(model_clients=clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["model_1", "model_2"],
            include_context=False,
        )

        handoff = agent.get_handoff(result)

        # Should have recommendations for low confidence
        assert handoff["requires_further_analysis"] is True
        assert any("confidence" in r.lower() or "agreement" in r.lower()
                   for r in handoff["recommendations"])


class TestPredictionSynthesizerGraph:
    """Tests for LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_full_graph_execution(
        self, mock_model_registry, mock_model_clients, mock_context_store
    ):
        """Test full graph execution."""
        graph = build_prediction_synthesizer_graph(
            model_registry=mock_model_registry,
            model_clients=mock_model_clients,
            context_store=mock_context_store,
        )

        initial_state = {
            "query": "What is the churn risk?",
            "entity_id": "hcp_123",
            "entity_type": "hcp",
            "prediction_target": "churn",
            "features": {"call_frequency": 10},
            "time_horizon": "30d",
            "models_to_use": ["churn_xgb"],
            "ensemble_method": "weighted",
            "confidence_level": 0.95,
            "include_context": True,
            "individual_predictions": None,
            "models_succeeded": 0,
            "models_failed": 0,
            "ensemble_prediction": None,
            "prediction_summary": None,
            "prediction_context": None,
            "orchestration_latency_ms": 0,
            "ensemble_latency_ms": 0,
            "total_latency_ms": 0,
            "timestamp": "",
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await graph.ainvoke(initial_state)

        assert result["status"] == "completed"
        assert result["models_succeeded"] >= 1
        assert result["ensemble_prediction"] is not None

    @pytest.mark.asyncio
    async def test_error_handling_path(self):
        """Test error handling path in graph when no models available."""
        from src.agents.prediction_synthesizer.graph import (
            build_simple_prediction_graph,
        )

        # No clients and no models specified - should fail with no models available
        graph = build_simple_prediction_graph(model_clients={})

        initial_state = {
            "query": "What is the churn risk?",
            "entity_id": "hcp_123",
            "entity_type": "hcp",
            "prediction_target": "churn",
            "features": {},
            "time_horizon": "30d",
            "models_to_use": [],  # Empty list - no models available
            "ensemble_method": "weighted",
            "confidence_level": 0.95,
            "include_context": False,
            "individual_predictions": None,
            "models_succeeded": 0,
            "models_failed": 0,
            "ensemble_prediction": None,
            "prediction_summary": None,
            "prediction_context": None,
            "orchestration_latency_ms": 0,
            "ensemble_latency_ms": 0,
            "total_latency_ms": 0,
            "timestamp": "",
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await graph.ainvoke(initial_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_synthesize_predictions_function(
        self, mock_model_clients, sample_features
    ):
        """Test synthesize_predictions convenience function."""
        result = await synthesize_predictions(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            model_clients=mock_model_clients,
        )

        assert isinstance(result, PredictionSynthesizerOutput)


class TestInputOutputContracts:
    """Tests for Pydantic contracts."""

    def test_input_contract_defaults(self):
        """Test input contract default values."""
        input_data = PredictionSynthesizerInput(
            entity_id="hcp_123",
            prediction_target="churn",
        )

        assert input_data.entity_type == "hcp"
        assert input_data.time_horizon == "30d"
        assert input_data.ensemble_method == "weighted"
        assert input_data.confidence_level == 0.95
        assert input_data.include_context is True

    def test_input_contract_custom_values(self):
        """Test input contract with custom values."""
        input_data = PredictionSynthesizerInput(
            entity_id="territory_456",
            entity_type="territory",
            prediction_target="conversion",
            features={"market_share": 0.15},
            time_horizon="90d",
            models_to_use=["conversion_model"],
            ensemble_method="average",
            confidence_level=0.99,
            include_context=False,
        )

        assert input_data.entity_id == "territory_456"
        assert input_data.entity_type == "territory"
        assert input_data.ensemble_method == "average"
        assert input_data.confidence_level == 0.99

    def test_output_contract_defaults(self):
        """Test output contract default values."""
        output_data = PredictionSynthesizerOutput()

        assert output_data.ensemble_prediction is None
        assert output_data.individual_predictions == []
        assert output_data.models_succeeded == 0
        assert output_data.status == "pending"

    def test_output_contract_serialization(self):
        """Test output contract JSON serialization."""
        output_data = PredictionSynthesizerOutput(
            status="completed",
            models_succeeded=2,
            prediction_summary="Test summary",
        )

        json_data = output_data.model_dump()

        assert json_data["status"] == "completed"
        assert json_data["models_succeeded"] == 2
        assert json_data["prediction_summary"] == "Test summary"


class TestEnsembleMethods:
    """Tests for different ensemble methods."""

    @pytest.mark.asyncio
    async def test_average_method(self, mock_model_clients, sample_features):
        """Test average ensemble method."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf"],
            ensemble_method="average",
            include_context=False,
        )

        assert result.ensemble_prediction["ensemble_method"] == "average"

    @pytest.mark.asyncio
    async def test_weighted_method(self, mock_model_clients, sample_features):
        """Test weighted ensemble method."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf"],
            ensemble_method="weighted",
            include_context=False,
        )

        assert result.ensemble_prediction["ensemble_method"] == "weighted"

    @pytest.mark.asyncio
    async def test_voting_method(self, mock_model_clients, sample_features):
        """Test voting ensemble method."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        result = await agent.synthesize(
            entity_id="hcp_123",
            prediction_target="churn",
            features=sample_features,
            models_to_use=["churn_xgb", "churn_rf"],
            ensemble_method="voting",
            include_context=False,
        )

        assert result.ensemble_prediction["ensemble_method"] == "voting"


class TestLazyLoading:
    """Tests for lazy graph loading."""

    def test_graph_lazy_loading(self, mock_model_clients):
        """Test that graphs are lazily loaded."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        # Graphs should not be built yet
        assert agent._full_graph is None
        assert agent._simple_graph is None

    def test_full_graph_builds_on_access(self, mock_model_clients):
        """Test full graph builds when accessed."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        # Access the graph
        _ = agent.full_graph

        assert agent._full_graph is not None

    def test_simple_graph_builds_on_access(self, mock_model_clients):
        """Test simple graph builds when accessed."""
        agent = PredictionSynthesizerAgent(model_clients=mock_model_clients)

        # Access the graph
        _ = agent.simple_graph

        assert agent._simple_graph is not None
