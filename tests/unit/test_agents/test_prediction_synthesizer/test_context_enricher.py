"""
E2I Prediction Synthesizer Agent - Context Enricher Node Tests
"""

import pytest
from src.agents.prediction_synthesizer.nodes.context_enricher import (
    ContextEnricherNode,
)


class TestContextEnricherNode:
    """Tests for ContextEnricherNode."""

    @pytest.mark.asyncio
    async def test_enrich_with_context(
        self, mock_context_store, mock_feature_store, state_with_ensemble
    ):
        """Test context enrichment with all components."""
        node = ContextEnricherNode(
            context_store=mock_context_store,
            feature_store=mock_feature_store,
        )

        result = await node.execute(state_with_ensemble)

        assert result["prediction_context"] is not None
        context = result["prediction_context"]

        assert "similar_cases" in context
        assert "feature_importance" in context
        assert "historical_accuracy" in context
        assert "trend_direction" in context
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_enrich_similar_cases(
        self, mock_context_store, state_with_ensemble
    ):
        """Test similar cases enrichment."""
        node = ContextEnricherNode(context_store=mock_context_store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert len(context["similar_cases"]) > 0
        assert context["similar_cases"][0]["entity_id"] == "hcp_100"

    @pytest.mark.asyncio
    async def test_enrich_feature_importance(
        self, mock_context_store, mock_feature_store, state_with_ensemble
    ):
        """Test feature importance enrichment."""
        node = ContextEnricherNode(
            context_store=mock_context_store,
            feature_store=mock_feature_store,
        )

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert "call_frequency" in context["feature_importance"]
        assert "prescription_count" in context["feature_importance"]

    @pytest.mark.asyncio
    async def test_enrich_historical_accuracy(
        self, mock_context_store, state_with_ensemble
    ):
        """Test historical accuracy enrichment."""
        node = ContextEnricherNode(context_store=mock_context_store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["historical_accuracy"] == 0.82

    @pytest.mark.asyncio
    async def test_enrich_trend_increasing(
        self, mock_context_store, state_with_ensemble
    ):
        """Test trend detection for increasing values."""
        # History has increasing values
        node = ContextEnricherNode(context_store=mock_context_store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["trend_direction"] == "increasing"

    @pytest.mark.asyncio
    async def test_enrich_trend_decreasing(self, state_with_ensemble):
        """Test trend detection for decreasing values."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockContextStore,
        )

        store = MockContextStore(
            history=[
                {"prediction": 0.90, "timestamp": "2024-01-01"},
                {"prediction": 0.75, "timestamp": "2024-02-01"},
                {"prediction": 0.60, "timestamp": "2024-03-01"},
                {"prediction": 0.45, "timestamp": "2024-04-01"},
                {"prediction": 0.30, "timestamp": "2024-05-01"},
            ]
        )

        node = ContextEnricherNode(context_store=store)
        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["trend_direction"] == "decreasing"

    @pytest.mark.asyncio
    async def test_enrich_trend_stable(self, state_with_ensemble):
        """Test trend detection for stable values."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockContextStore,
        )

        store = MockContextStore(
            history=[
                {"prediction": 0.70, "timestamp": "2024-01-01"},
                {"prediction": 0.71, "timestamp": "2024-02-01"},
                {"prediction": 0.69, "timestamp": "2024-03-01"},
                {"prediction": 0.70, "timestamp": "2024-04-01"},
                {"prediction": 0.70, "timestamp": "2024-05-01"},
            ]
        )

        node = ContextEnricherNode(context_store=store)
        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["trend_direction"] == "stable"

    @pytest.mark.asyncio
    async def test_enrich_no_context_requested(self, state_with_ensemble):
        """Test skipping enrichment when not requested."""
        state_with_ensemble["include_context"] = False

        node = ContextEnricherNode()
        result = await node.execute(state_with_ensemble)

        assert result["prediction_context"] is None
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_enrich_without_stores(self, state_with_ensemble):
        """Test enrichment with no stores configured."""
        node = ContextEnricherNode()

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["similar_cases"] == []
        assert context["feature_importance"] == {}
        assert context["historical_accuracy"] == 0.0
        assert context["trend_direction"] == "stable"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_enrich_already_failed(self, state_with_ensemble):
        """Test that already failed state passes through."""
        state_with_ensemble["status"] = "failed"

        node = ContextEnricherNode()
        result = await node.execute(state_with_ensemble)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_enrich_already_completed(self, state_with_ensemble):
        """Test that already completed state passes through."""
        state_with_ensemble["status"] = "completed"

        node = ContextEnricherNode()
        result = await node.execute(state_with_ensemble)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_enrich_total_latency(
        self, mock_context_store, state_with_ensemble
    ):
        """Test total latency calculation."""
        node = ContextEnricherNode(context_store=mock_context_store)

        result = await node.execute(state_with_ensemble)

        # Total should include orchestration + ensemble + context
        assert result["total_latency_ms"] >= (
            state_with_ensemble["orchestration_latency_ms"]
            + state_with_ensemble["ensemble_latency_ms"]
        )


class TestContextEnricherErrorHandling:
    """Error handling tests for ContextEnricherNode."""

    @pytest.mark.asyncio
    async def test_store_error_graceful(self, state_with_ensemble):
        """Test graceful handling of store errors."""

        class FailingStore:
            async def find_similar(self, *args, **kwargs):
                raise RuntimeError("Database error")

            async def get_accuracy(self, *args, **kwargs):
                raise RuntimeError("Database error")

            async def get_prediction_history(self, *args, **kwargs):
                raise RuntimeError("Database error")

        node = ContextEnricherNode(context_store=FailingStore())

        result = await node.execute(state_with_ensemble)

        # Should complete despite errors (non-fatal)
        assert result["status"] == "completed"
        context = result["prediction_context"]
        # Fallbacks should be used
        assert context["similar_cases"] == []
        assert context["historical_accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_partial_store_failure(self, state_with_ensemble):
        """Test handling of partial store failures."""

        class PartialFailStore:
            async def find_similar(self, *args, **kwargs):
                return [{"entity_id": "hcp_100", "prediction": 0.7}]

            async def get_accuracy(self, *args, **kwargs):
                raise RuntimeError("Accuracy error")

            async def get_prediction_history(self, *args, **kwargs):
                return []

        node = ContextEnricherNode(context_store=PartialFailStore())

        result = await node.execute(state_with_ensemble)

        assert result["status"] == "completed"
        context = result["prediction_context"]
        # Working parts should be populated
        assert len(context["similar_cases"]) == 1
        # Failed parts should have fallbacks
        assert context["historical_accuracy"] == 0.0


class TestFeatureImportanceAggregation:
    """Tests for feature importance aggregation."""

    @pytest.mark.asyncio
    async def test_importance_aggregation_multiple_models(
        self, mock_feature_store, state_with_ensemble
    ):
        """Test feature importance aggregation across models."""
        # Add more predictions
        state_with_ensemble["individual_predictions"] = [
            {"model_id": "churn_xgb", "prediction": 0.72, "confidence": 0.88, "latency_ms": 50},
            {"model_id": "churn_rf", "prediction": 0.68, "confidence": 0.82, "latency_ms": 60},
            {"model_id": "churn_nn", "prediction": 0.75, "confidence": 0.85, "latency_ms": 80},
        ]

        node = ContextEnricherNode(feature_store=mock_feature_store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        importance = context["feature_importance"]

        # Should have aggregated importance from all models
        assert "call_frequency" in importance
        assert "prescription_count" in importance

    @pytest.mark.asyncio
    async def test_importance_top_10_limit(
        self, state_with_ensemble
    ):
        """Test that feature importance is limited to top 10."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockFeatureStore,
        )

        # Create feature store with many features
        store = MockFeatureStore(
            importance={
                "model_1": {f"feature_{i}": 1.0 / (i + 1) for i in range(20)}
            }
        )

        state_with_ensemble["individual_predictions"] = [
            {"model_id": "model_1", "prediction": 0.7, "confidence": 0.8, "latency_ms": 50}
        ]

        node = ContextEnricherNode(feature_store=store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert len(context["feature_importance"]) <= 10


class TestTrendCalculation:
    """Tests for trend calculation."""

    @pytest.mark.asyncio
    async def test_trend_insufficient_history(self, state_with_ensemble):
        """Test trend with insufficient history."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockContextStore,
        )

        store = MockContextStore(
            history=[
                {"prediction": 0.70, "timestamp": "2024-01-01"},
            ]
        )

        node = ContextEnricherNode(context_store=store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["trend_direction"] == "stable"

    @pytest.mark.asyncio
    async def test_trend_empty_history(self, state_with_ensemble):
        """Test trend with empty history."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockContextStore,
        )

        store = MockContextStore(history=[])

        node = ContextEnricherNode(context_store=store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["trend_direction"] == "stable"
