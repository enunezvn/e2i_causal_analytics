"""Unit tests for ModelInferenceTool.

Tests cover:
- Tool initialization
- Prediction with direct features
- Prediction with Feast feature retrieval
- Error handling
- Registry integration
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tool_registry.tools.model_inference import (
    ModelInferenceInput,
    ModelInferenceOutput,
    ModelInferenceTool,
    model_inference,
    register_model_inference_tool,
)

# =============================================================================
# INPUT/OUTPUT SCHEMA TESTS
# =============================================================================


class TestModelInferenceInput:
    """Tests for ModelInferenceInput schema."""

    def test_minimal_input(self):
        """Test minimal valid input."""
        inp = ModelInferenceInput(model_name="test_model")
        assert inp.model_name == "test_model"
        assert inp.features == {}
        assert inp.entity_id is None
        assert inp.time_horizon == "short_term"

    def test_full_input(self):
        """Test full input with all fields."""
        inp = ModelInferenceInput(
            model_name="churn_model",
            features={"recency": 10, "frequency": 5},
            entity_id="HCP001",
            time_horizon="medium_term",
            return_probabilities=True,
            return_explanation=True,
            trace_context={"trace_id": "abc123"},
        )
        assert inp.model_name == "churn_model"
        assert inp.features["recency"] == 10
        assert inp.entity_id == "HCP001"
        assert inp.return_probabilities is True


class TestModelInferenceOutput:
    """Tests for ModelInferenceOutput schema."""

    def test_minimal_output(self):
        """Test minimal valid output."""
        out = ModelInferenceOutput(
            model_name="test_model",
            prediction=0.85,
            latency_ms=50.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert out.prediction == 0.85
        assert out.confidence is None
        assert out.warnings == []

    def test_full_output(self):
        """Test output with all fields."""
        out = ModelInferenceOutput(
            model_name="churn_model",
            prediction=0.85,
            confidence=0.92,
            probabilities={"churn": 0.85, "no_churn": 0.15},
            explanation={"recency": 0.3, "frequency": 0.5},
            model_version="1.0.0",
            latency_ms=45.0,
            timestamp="2024-01-01T00:00:00Z",
            trace_id="abc123",
            features_used={"recency": 10},
            warnings=["Minor drift detected"],
        )
        assert out.probabilities["churn"] == 0.85
        assert out.explanation["frequency"] == 0.5


# =============================================================================
# TOOL TESTS
# =============================================================================


class TestModelInferenceTool:
    """Tests for ModelInferenceTool."""

    @pytest.fixture
    def tool(self):
        """Create a fresh tool instance."""
        return ModelInferenceTool(
            bentoml_base_url="http://test:3000",
            feast_enabled=False,
            opik_enabled=False,
        )

    @pytest.fixture
    def mock_bentoml_client(self):
        """Create a mock BentoML client."""
        client = AsyncMock()
        client.predict = AsyncMock(
            return_value={
                "prediction": 0.85,
                "confidence": 0.92,
                "model_version": "1.0.0",
                "_metadata": {
                    "latency_ms": 45.0,
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            }
        )
        return client

    @pytest.mark.asyncio
    async def test_invoke_with_features(self, tool, mock_bentoml_client):
        """Test prediction with direct features."""
        tool._client = mock_bentoml_client

        result = await tool.invoke(
            {
                "model_name": "churn_model",
                "features": {"recency": 10, "frequency": 5},
            }
        )

        assert result.model_name == "churn_model"
        assert result.prediction == 0.85
        assert result.confidence == 0.92
        assert result.latency_ms == 45.0

        mock_bentoml_client.predict.assert_called_once()
        call_args = mock_bentoml_client.predict.call_args
        assert call_args.kwargs["model_name"] == "churn_model"

    @pytest.mark.asyncio
    async def test_invoke_with_pydantic_input(self, tool, mock_bentoml_client):
        """Test prediction with Pydantic input model."""
        tool._client = mock_bentoml_client

        input_model = ModelInferenceInput(
            model_name="conversion_model",
            features={"engagement_score": 0.8},
            return_probabilities=True,
        )

        result = await tool.invoke(input_model)

        assert result.model_name == "conversion_model"
        assert result.prediction == 0.85

    @pytest.mark.asyncio
    async def test_invoke_handles_circuit_breaker_open(self, tool):
        """Test handling when circuit breaker is open."""
        mock_client = AsyncMock()
        mock_client.predict = AsyncMock(
            side_effect=RuntimeError("Circuit breaker open for model 'test_model'")
        )
        tool._client = mock_client

        result = await tool.invoke({"model_name": "test_model", "features": {}})

        assert result.prediction is None
        assert len(result.warnings) > 0
        assert "unavailable" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_invoke_handles_general_error(self, tool):
        """Test handling of general errors."""
        mock_client = AsyncMock()
        mock_client.predict = AsyncMock(side_effect=Exception("Connection refused"))
        tool._client = mock_client

        result = await tool.invoke({"model_name": "test_model", "features": {}})

        assert result.prediction is None
        assert len(result.warnings) > 0
        assert "failed" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_invoke_with_feast_features(self, tool, mock_bentoml_client):
        """Test prediction with Feast feature retrieval."""
        tool._client = mock_bentoml_client
        tool.feast_enabled = True

        # Mock Feast store
        mock_feast = AsyncMock()
        mock_feast_result = MagicMock()
        mock_feast_result.to_dict.return_value = {"feast_feature": 100}
        mock_feast.get_online_features = AsyncMock(return_value=mock_feast_result)
        tool._feast_store = mock_feast

        result = await tool.invoke(
            {
                "model_name": "churn_model",
                "features": {"manual_feature": 50},
                "entity_id": "HCP001",
            }
        )

        assert result.prediction == 0.85
        # Features should be merged
        call_args = mock_bentoml_client.predict.call_args
        features = call_args.kwargs["input_data"]["features"]
        assert features["feast_feature"] == 100
        assert features["manual_feature"] == 50

    @pytest.mark.asyncio
    async def test_feast_error_graceful_fallback(self, tool, mock_bentoml_client):
        """Test graceful fallback when Feast fails."""
        tool._client = mock_bentoml_client
        tool.feast_enabled = True

        # Mock Feast store that raises error
        mock_feast = AsyncMock()
        mock_feast.get_online_features = AsyncMock(side_effect=Exception("Feast down"))
        tool._feast_store = mock_feast

        # Patch _fetch_feast_features to raise an exception that gets caught

        async def failing_fetch(*args, **kwargs):
            raise Exception("Feast down")

        tool._fetch_feast_features = failing_fetch

        result = await tool.invoke(
            {
                "model_name": "churn_model",
                "features": {"manual_feature": 50},
                "entity_id": "HCP001",
            }
        )

        # Should still succeed with warning about Feast
        assert result.prediction == 0.85
        assert len(result.warnings) > 0
        assert "Feast" in result.warnings[0]


# =============================================================================
# FUNCTION INTERFACE TESTS
# =============================================================================


class TestModelInferenceFunction:
    """Tests for the model_inference function."""

    @pytest.mark.asyncio
    async def test_function_interface(self):
        """Test the function interface returns correct structure."""
        mock_client = AsyncMock()
        mock_client.predict = AsyncMock(
            return_value={
                "prediction": 0.75,
                "confidence": 0.88,
                "_metadata": {
                    "latency_ms": 30.0,
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            }
        )

        # Reset singleton for clean test
        import src.tool_registry.tools.model_inference as module

        module._tool_instance = None

        # Patch the import of get_bentoml_client inside the module
        with patch(
            "src.api.dependencies.bentoml_client.get_bentoml_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await model_inference(
                model_name="test_model",
                features={"x": 1, "y": 2},
            )

            assert isinstance(result, dict)
            assert result["prediction"] == 0.75
            assert result["confidence"] == 0.88

        # Cleanup
        module._tool_instance = None


# =============================================================================
# REGISTRY INTEGRATION TESTS
# =============================================================================


class TestToolRegistration:
    """Tests for tool registry integration."""

    def test_tool_is_registered(self):
        """Test that tool is registered in the registry."""
        from src.tool_registry.registry import get_registry

        # Re-register to ensure it's there
        register_model_inference_tool()

        registry = get_registry()
        assert registry.validate_tool_exists("model_inference")

    def test_tool_schema_complete(self):
        """Test that tool schema has all required fields."""
        from src.tool_registry.registry import get_registry

        register_model_inference_tool()

        registry = get_registry()
        schema = registry.get_schema("model_inference")

        assert schema is not None
        assert schema.name == "model_inference"
        assert schema.source_agent == "prediction_synthesizer"
        assert schema.tier == 4
        assert len(schema.input_parameters) >= 2  # At least model_name and features

    def test_tool_callable_exists(self):
        """Test that tool callable is registered."""
        from src.tool_registry.registry import get_registry

        register_model_inference_tool()

        registry = get_registry()
        callable_func = registry.get_callable("model_inference")

        assert callable_func is not None
        assert callable(callable_func)

    def test_tool_in_agent_list(self):
        """Test that tool appears in agent's tool list."""
        from src.tool_registry.registry import get_registry

        register_model_inference_tool()

        registry = get_registry()
        agent_tools = registry.list_by_agent("prediction_synthesizer")

        assert "model_inference" in agent_tools

    def test_tool_in_tier_list(self):
        """Test that tool appears in tier 4 tool list."""
        from src.tool_registry.registry import get_registry

        register_model_inference_tool()

        registry = get_registry()
        tier4_tools = registry.list_by_tier(4)

        assert "model_inference" in tier4_tools
