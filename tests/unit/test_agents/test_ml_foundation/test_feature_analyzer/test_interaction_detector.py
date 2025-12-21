"""Tests for interaction detection node."""

import numpy as np
import pytest

from src.agents.ml_foundation.feature_analyzer.nodes.interaction_detector import (
    _compute_correlation_interactions,
    _extract_top_interactions,
    detect_interactions,
)


@pytest.mark.asyncio
class TestDetectInteractions:
    """Test interaction detection node."""

    async def test_detects_interactions(self):
        """Should detect feature interactions from SHAP values."""
        # Create SHAP values with some correlated features
        shap_values = np.array(
            [
                [0.5, 0.4, -0.2, 0.1],
                [0.6, 0.5, -0.3, 0.15],
                [0.4, 0.3, -0.1, 0.05],
                [0.7, 0.6, -0.4, 0.2],
            ]
        )
        # feat_0 and feat_1 are highly correlated (positive)
        # feat_0 and feat_2 are negatively correlated

        state = {
            "shap_values": shap_values,
            "feature_names": ["feat_0", "feat_1", "feat_2", "feat_3"],
            "compute_interactions": True,
        }

        result = await detect_interactions(state)

        assert "error" not in result
        assert "interaction_matrix" in result
        assert "top_interactions_raw" in result
        assert result["interaction_method"] == "correlation"

    async def test_skips_when_compute_interactions_false(self):
        """Should skip interaction detection when compute_interactions is False."""
        state = {
            "shap_values": np.random.rand(10, 5),
            "feature_names": ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
            "compute_interactions": False,
        }

        result = await detect_interactions(state)

        assert "error" not in result
        assert result["interaction_matrix"] == {}
        assert result["top_interactions_raw"] == []
        assert result["interaction_method"] == "skipped"

    async def test_error_when_missing_shap_values(self):
        """Should return error when SHAP values are missing."""
        state = {
            "feature_names": ["feat_0", "feat_1"],
            "compute_interactions": True,
        }

        result = await detect_interactions(state)

        assert "error" in result
        assert result["error_type"] == "missing_shap_data"
        assert result["status"] == "failed"

    async def test_error_when_missing_feature_names(self):
        """Should return error when feature names are missing."""
        state = {
            "shap_values": np.random.rand(10, 3),
            "feature_names": [],
            "compute_interactions": True,
        }

        result = await detect_interactions(state)

        assert "error" in result
        assert result["error_type"] == "missing_shap_data"

    async def test_returns_top_10_interactions(self):
        """Should return top 10 interactions by strength."""
        # Create SHAP values with many features
        shap_values = np.random.rand(100, 15)
        feature_names = [f"feat_{i}" for i in range(15)]

        state = {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "compute_interactions": True,
        }

        result = await detect_interactions(state)

        assert "top_interactions_raw" in result
        assert len(result["top_interactions_raw"]) <= 10  # Top 10

    async def test_records_computation_time(self):
        """Should record computation time."""
        state = {
            "shap_values": np.random.rand(50, 8),
            "feature_names": [f"feat_{i}" for i in range(8)],
            "compute_interactions": True,
        }

        result = await detect_interactions(state)

        assert "interaction_computation_time_seconds" in result
        assert result["interaction_computation_time_seconds"] >= 0


class TestComputeCorrelationInteractions:
    """Test correlation-based interaction computation."""

    def test_computes_correlation_matrix(self):
        """Should compute correlation-based interaction matrix."""
        # Create SHAP values with clear correlations
        shap_values = np.array(
            [
                [1.0, 0.9, -0.8, 0.0],
                [2.0, 1.8, -1.6, 0.1],
                [1.5, 1.4, -1.2, -0.1],
                [2.5, 2.3, -2.0, 0.05],
            ]
        )

        feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]

        interaction_matrix, method = _compute_correlation_interactions(shap_values, feature_names)

        assert method == "correlation"
        assert isinstance(interaction_matrix, dict)
        assert "feat_0" in interaction_matrix

        # feat_0 and feat_1 should be highly positively correlated
        if "feat_1" in interaction_matrix["feat_0"]:
            assert interaction_matrix["feat_0"]["feat_1"] > 0.8

    def test_skips_self_interactions(self):
        """Should skip self-interactions (diagonal)."""
        shap_values = np.random.rand(20, 3)
        feature_names = ["feat_0", "feat_1", "feat_2"]

        interaction_matrix, _ = _compute_correlation_interactions(shap_values, feature_names)

        # No feature should have interaction with itself
        for feature in feature_names:
            if feature in interaction_matrix:
                assert feature not in interaction_matrix[feature]

    def test_filters_weak_correlations(self):
        """Should filter out weak correlations (abs < 0.1)."""
        # Create SHAP values with very weak correlations
        np.random.seed(42)
        shap_values = np.random.rand(100, 4)

        feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]

        interaction_matrix, _ = _compute_correlation_interactions(shap_values, feature_names)

        # Check that only strong correlations are stored
        for _feature_i, interactions in interaction_matrix.items():
            for _feature_j, strength in interactions.items():
                assert abs(strength) > 0.1  # Threshold


class TestExtractTopInteractions:
    """Test top interaction extraction."""

    def test_extracts_top_k_interactions(self):
        """Should extract top k interactions by absolute strength."""
        interaction_matrix = {
            "feat_0": {"feat_1": 0.9, "feat_2": -0.3},
            "feat_1": {"feat_0": 0.9, "feat_3": 0.5},
            "feat_2": {"feat_0": -0.3, "feat_3": -0.7},
            "feat_3": {"feat_1": 0.5, "feat_2": -0.7},
        }

        top_interactions = _extract_top_interactions(interaction_matrix, top_k=3)

        assert len(top_interactions) == 3

        # Sorted by absolute strength
        # feat_0 x feat_1: 0.9
        # feat_2 x feat_3: -0.7
        # feat_1 x feat_3: 0.5
        assert abs(top_interactions[0][2]) >= abs(top_interactions[1][2])
        assert abs(top_interactions[1][2]) >= abs(top_interactions[2][2])

    def test_avoids_duplicate_pairs(self):
        """Should avoid duplicate pairs (feat_0-feat_1 and feat_1-feat_0)."""
        interaction_matrix = {
            "feat_0": {"feat_1": 0.8},
            "feat_1": {"feat_0": 0.8},
        }

        top_interactions = _extract_top_interactions(interaction_matrix, top_k=10)

        # Should only have one entry for feat_0-feat_1 pair
        assert len(top_interactions) == 1

    def test_handles_empty_matrix(self):
        """Should handle empty interaction matrix."""
        interaction_matrix = {}

        top_interactions = _extract_top_interactions(interaction_matrix, top_k=5)

        assert len(top_interactions) == 0

    def test_limits_to_top_k(self):
        """Should limit results to top k interactions."""
        # Create matrix with many interactions
        interaction_matrix = {
            f"feat_{i}": {f"feat_{j}": 0.5 - abs(i - j) * 0.1 for j in range(20) if i != j}
            for i in range(20)
        }

        top_interactions = _extract_top_interactions(interaction_matrix, top_k=5)

        assert len(top_interactions) <= 5
