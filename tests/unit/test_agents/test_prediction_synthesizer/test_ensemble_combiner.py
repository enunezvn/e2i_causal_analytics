"""
E2I Prediction Synthesizer Agent - Ensemble Combiner Node Tests
"""

import pytest

from src.agents.prediction_synthesizer.nodes.ensemble_combiner import (
    EnsembleCombinerNode,
)


class TestEnsembleCombinerNode:
    """Tests for EnsembleCombinerNode."""

    @pytest.mark.asyncio
    async def test_combine_weighted_average(self, state_with_predictions):
        """Test weighted average ensemble combination."""
        state_with_predictions["ensemble_method"] = "weighted"

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        assert result["ensemble_prediction"] is not None
        ensemble = result["ensemble_prediction"]

        # Weighted average: (0.72*0.88 + 0.68*0.82) / (0.88 + 0.82) = 0.7018...
        assert 0.69 < ensemble["point_estimate"] < 0.71
        assert ensemble["ensemble_method"] == "weighted"
        assert result["status"] == "enriching"

    @pytest.mark.asyncio
    async def test_combine_simple_average(self, state_with_predictions):
        """Test simple average ensemble combination."""
        state_with_predictions["ensemble_method"] = "average"

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Simple average: (0.72 + 0.68) / 2 = 0.70
        assert abs(ensemble["point_estimate"] - 0.70) < 0.01
        assert ensemble["ensemble_method"] == "average"

    @pytest.mark.asyncio
    async def test_combine_voting(self, state_with_predictions):
        """Test voting ensemble combination."""
        state_with_predictions["ensemble_method"] = "voting"
        # Make predictions binary-like
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.8, "confidence": 0.9, "latency_ms": 50},
            {"model_id": "m2", "prediction": 0.7, "confidence": 0.85, "latency_ms": 50},
            {"model_id": "m3", "prediction": 0.3, "confidence": 0.8, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 3

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Voting: round(0.8)=1, round(0.7)=1, round(0.3)=0 -> majority is 1
        assert ensemble["point_estimate"] == 1.0
        assert ensemble["ensemble_method"] == "voting"

    @pytest.mark.asyncio
    async def test_combine_stacking(self, state_with_predictions):
        """Test stacking ensemble (falls back to average)."""
        state_with_predictions["ensemble_method"] = "stacking"

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Stacking uses average for now
        assert abs(ensemble["point_estimate"] - 0.70) < 0.01
        assert ensemble["ensemble_method"] == "stacking"

    @pytest.mark.asyncio
    async def test_prediction_interval(self, state_with_predictions):
        """Test prediction interval calculation."""
        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Should have lower and upper bounds
        assert ensemble["prediction_interval_lower"] < ensemble["point_estimate"]
        assert ensemble["prediction_interval_upper"] > ensemble["point_estimate"]

        # 95% CI should use z=1.96
        assert ensemble["prediction_interval_lower"] is not None
        assert ensemble["prediction_interval_upper"] is not None

    @pytest.mark.asyncio
    async def test_model_agreement_high(self, state_with_predictions):
        """Test model agreement calculation with similar predictions."""
        # Very similar predictions
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.70, "confidence": 0.85, "latency_ms": 50},
            {"model_id": "m2", "prediction": 0.71, "confidence": 0.85, "latency_ms": 50},
            {"model_id": "m3", "prediction": 0.69, "confidence": 0.85, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 3

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # High agreement (>0.8) for similar predictions
        assert ensemble["model_agreement"] > 0.8

    @pytest.mark.asyncio
    async def test_model_agreement_low(self, state_with_predictions):
        """Test model agreement calculation with divergent predictions."""
        # Very different predictions
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.90, "confidence": 0.85, "latency_ms": 50},
            {"model_id": "m2", "prediction": 0.10, "confidence": 0.85, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 2

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Low agreement for divergent predictions
        assert ensemble["model_agreement"] < 0.5

    @pytest.mark.asyncio
    async def test_ensemble_confidence(self, state_with_predictions):
        """Test ensemble confidence calculation."""
        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Confidence should be mean_confidence * agreement
        # Mean: (0.88 + 0.82) / 2 = 0.85
        # Agreement: high due to similar predictions
        assert 0.5 < ensemble["confidence"] < 1.0

    @pytest.mark.asyncio
    async def test_prediction_summary_generation(self, state_with_predictions):
        """Test prediction summary generation."""
        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        summary = result["prediction_summary"]

        assert "Prediction:" in summary
        assert "CI:" in summary
        assert "Confidence:" in summary
        assert "Model agreement:" in summary
        assert "2 models" in summary

    @pytest.mark.asyncio
    async def test_no_predictions_fails(self, base_state):
        """Test that empty predictions causes failure."""
        base_state["individual_predictions"] = []

        node = EnsembleCombinerNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "No predictions" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_already_failed_passthrough(self, state_with_predictions):
        """Test that already failed state passes through."""
        state_with_predictions["status"] = "failed"
        state_with_predictions["errors"] = [{"error": "Previous error"}]

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        assert result["status"] == "failed"
        assert result["errors"] == [{"error": "Previous error"}]

    @pytest.mark.asyncio
    async def test_single_prediction(self, state_with_predictions):
        """Test ensemble with single prediction."""
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.72, "confidence": 0.88, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 1

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]

        # Single prediction should be the estimate
        assert ensemble["point_estimate"] == 0.72
        # CRITICAL SAFETY: Single model confidence capped at 30%
        assert ensemble["confidence"] <= 0.30
        # CRITICAL: Single model has NO agreement (cannot validate without diversity)
        assert ensemble["model_agreement"] == 0.0

    @pytest.mark.asyncio
    async def test_ensemble_latency_tracked(self, state_with_predictions):
        """Test that ensemble latency is tracked."""
        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        # Latency can be 0 if operation completes in <1ms
        assert result["ensemble_latency_ms"] >= 0
        assert "ensemble_latency_ms" in result

    @pytest.mark.asyncio
    async def test_status_with_context(self, state_with_predictions):
        """Test status is enriching when context is requested."""
        state_with_predictions["include_context"] = True

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        assert result["status"] == "enriching"

    @pytest.mark.asyncio
    async def test_status_without_context(self, state_with_predictions):
        """Test status is completed when no context requested."""
        state_with_predictions["include_context"] = False

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        assert result["status"] == "completed"


class TestEnsembleCombinerEdgeCases:
    """Edge case tests for EnsembleCombinerNode."""

    @pytest.mark.asyncio
    async def test_zero_confidence_predictions(self, state_with_predictions):
        """Test handling of zero confidence predictions."""
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.70, "confidence": 0.0, "latency_ms": 50},
            {"model_id": "m2", "prediction": 0.80, "confidence": 0.0, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 2

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        # Should fall back to simple average
        ensemble = result["ensemble_prediction"]
        assert abs(ensemble["point_estimate"] - 0.75) < 0.01

    @pytest.mark.asyncio
    async def test_identical_predictions(self, state_with_predictions):
        """Test handling of identical predictions."""
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.50, "confidence": 0.80, "latency_ms": 50},
            {"model_id": "m2", "prediction": 0.50, "confidence": 0.80, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 2

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]
        assert ensemble["point_estimate"] == 0.50
        assert ensemble["model_agreement"] == 1.0

    @pytest.mark.asyncio
    async def test_extreme_predictions(self, state_with_predictions):
        """Test handling of extreme (0 and 1) predictions."""
        state_with_predictions["individual_predictions"] = [
            {"model_id": "m1", "prediction": 0.0, "confidence": 0.85, "latency_ms": 50},
            {"model_id": "m2", "prediction": 1.0, "confidence": 0.85, "latency_ms": 50},
        ]
        state_with_predictions["models_succeeded"] = 2

        node = EnsembleCombinerNode()
        result = await node.execute(state_with_predictions)

        ensemble = result["ensemble_prediction"]
        # Average of 0 and 1 should be 0.5
        assert abs(ensemble["point_estimate"] - 0.50) < 0.01
        # Agreement should be very low
        assert ensemble["model_agreement"] < 0.3
