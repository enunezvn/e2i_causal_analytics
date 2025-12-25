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
    async def test_enrich_similar_cases(self, mock_context_store, state_with_ensemble):
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
    async def test_enrich_historical_accuracy(self, mock_context_store, state_with_ensemble):
        """Test historical accuracy enrichment."""
        node = ContextEnricherNode(context_store=mock_context_store)

        result = await node.execute(state_with_ensemble)

        context = result["prediction_context"]
        assert context["historical_accuracy"] == 0.82

    @pytest.mark.asyncio
    async def test_enrich_trend_increasing(self, mock_context_store, state_with_ensemble):
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
    async def test_enrich_total_latency(self, mock_context_store, state_with_ensemble):
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
    async def test_importance_top_10_limit(self, state_with_ensemble):
        """Test that feature importance is limited to top 10."""
        from tests.unit.test_agents.test_prediction_synthesizer.conftest import (
            MockFeatureStore,
        )

        # Create feature store with many features
        store = MockFeatureStore(
            importance={"model_1": {f"feature_{i}": 1.0 / (i + 1) for i in range(20)}}
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


class TestFeastOnlineFeatureIntegration:
    """Tests for Feast online feature integration in context enricher."""

    @pytest.mark.asyncio
    async def test_online_features_fetched(self, state_with_ensemble):
        """Test that online features are fetched from Feast."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(
            return_value={
                "call_frequency": 25.0,
                "prescription_count": 100.0,
            }
        )
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={"fresh": True, "stale_features": []}
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        assert result["status"] == "completed"
        assert "feast_online_features" in result
        assert result["feast_online_features"]["call_frequency"] == 25.0

    @pytest.mark.asyncio
    async def test_online_features_merged_with_input(self, state_with_ensemble):
        """Test that online features are merged with input features."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(
            return_value={"new_feature": 42.0}
        )
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={"fresh": True, "stale_features": []}
        )

        state_with_ensemble["entity_id"] = "hcp_123"
        state_with_ensemble["features"] = {"existing_feature": 10.0}

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        # Both existing and online features should be present
        assert result["features"]["existing_feature"] == 10.0
        assert result["features"]["new_feature"] == 42.0

    @pytest.mark.asyncio
    async def test_online_features_override_input(self, state_with_ensemble):
        """Test that online features override stale input features."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(
            return_value={"call_frequency": 50.0}  # Updated value
        )
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={"fresh": True, "stale_features": []}
        )

        state_with_ensemble["entity_id"] = "hcp_123"
        state_with_ensemble["features"] = {"call_frequency": 10.0}  # Old value

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        # Online feature should override input
        assert result["features"]["call_frequency"] == 50.0

    @pytest.mark.asyncio
    async def test_online_features_disabled(self, state_with_ensemble):
        """Test that online features can be disabled."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(
            return_value={"feature": 100.0}
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(
            feature_store=mock_feast_store,
            enable_online_features=False,
        )
        result = await node.execute(state_with_ensemble)

        # Online features should not be fetched
        mock_feast_store.get_online_features.assert_not_called()
        assert "feast_online_features" not in result

    @pytest.mark.asyncio
    async def test_online_features_no_entity_id(self, state_with_ensemble):
        """Test graceful handling when no entity_id in state."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock()

        # Remove entity_id
        state_with_ensemble.pop("entity_id", None)

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        # Should complete without error
        assert result["status"] == "completed"
        # Online features not called without entity
        mock_feast_store.get_online_features.assert_not_called()

    @pytest.mark.asyncio
    async def test_freshness_check_included(self, state_with_ensemble):
        """Test that freshness check results are included."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(return_value={"f1": 1.0})
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={
                "fresh": True,
                "stale_features": [],
                "last_updated": "2024-12-25T00:00:00Z",
            }
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        assert "feast_freshness" in result
        assert result["feast_freshness"]["fresh"] is True

    @pytest.mark.asyncio
    async def test_stale_features_warning(self, state_with_ensemble):
        """Test warning generated for stale features."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(return_value={"f1": 1.0})
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={
                "fresh": False,
                "stale_features": ["feature1", "feature2", "feature3"],
            }
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        assert result["warnings"]
        assert any("Stale features" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_online_features_error_graceful(self, state_with_ensemble):
        """Test graceful handling of online feature errors."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(
            side_effect=Exception("Feast unavailable")
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(feature_store=mock_feast_store)
        result = await node.execute(state_with_ensemble)

        # Should still complete (non-fatal)
        assert result["status"] == "completed"
        # Warning should be added
        assert any("Online feature retrieval failed" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_custom_staleness_hours(self, state_with_ensemble):
        """Test custom max staleness hours configuration."""
        from unittest.mock import AsyncMock, MagicMock

        mock_feast_store = MagicMock()
        mock_feast_store.get_importance = AsyncMock(return_value={})
        mock_feast_store.get_online_features = AsyncMock(return_value={})
        mock_feast_store.check_feature_freshness = AsyncMock(
            return_value={"fresh": True, "stale_features": []}
        )

        state_with_ensemble["entity_id"] = "hcp_123"

        node = ContextEnricherNode(
            feature_store=mock_feast_store,
            max_staleness_hours=48.0,
        )
        await node.execute(state_with_ensemble)

        # Verify custom staleness was passed
        mock_feast_store.check_feature_freshness.assert_called_once()
        call_kwargs = mock_feast_store.check_feature_freshness.call_args[1]
        assert call_kwargs["max_staleness_hours"] == 48.0

    @pytest.mark.asyncio
    async def test_feature_store_without_online_features_method(
        self, mock_feature_store, state_with_ensemble
    ):
        """Test with feature store that doesn't support online features."""
        state_with_ensemble["entity_id"] = "hcp_123"

        # mock_feature_store doesn't have get_online_features
        node = ContextEnricherNode(feature_store=mock_feature_store)
        result = await node.execute(state_with_ensemble)

        # Should complete normally
        assert result["status"] == "completed"
        # No Feast metadata
        assert "feast_online_features" not in result
