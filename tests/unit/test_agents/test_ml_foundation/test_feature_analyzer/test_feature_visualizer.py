"""Unit tests for feature_visualizer node.

Tests visualization generation:
- Feature importance bar charts
- Selection funnel charts
- Statistics tables
- SHAP summary plots
"""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.agents.ml_foundation.feature_analyzer.nodes.feature_visualizer import (
    generate_visualizations,
    _create_importance_chart,
    _create_selection_funnel,
    _create_statistics_table,
    _create_shap_summary,
    _fig_to_base64,
    _get_output_path,
)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestCreateImportanceChart:
    """Tests for feature importance chart creation."""

    def test_creates_importance_chart(self):
        """Should create importance chart with correct structure."""
        feature_importance = {
            "feature_a": 0.35,
            "feature_b": 0.25,
            "feature_c": 0.20,
            "feature_d": 0.15,
            "feature_e": 0.05,
        }

        result = _create_importance_chart(feature_importance)

        assert result["type"] == "importance_bar_chart"
        assert len(result["features"]) == 5
        assert len(result["importances"]) == 5
        assert "image_base64" in result

    def test_respects_top_n(self):
        """Should limit to top_n features."""
        feature_importance = {f"feat_{i}": 1.0 / (i + 1) for i in range(30)}

        result = _create_importance_chart(feature_importance, top_n=10)

        assert len(result["features"]) == 10

    def test_features_sorted_by_importance(self):
        """Should sort features by importance descending."""
        feature_importance = {
            "low": 0.1,
            "high": 0.9,
            "medium": 0.5,
        }

        result = _create_importance_chart(feature_importance)

        assert result["features"][0] == "high"
        assert result["features"][-1] == "low"

    def test_saves_to_file_when_path_provided(self):
        """Should save chart to file when output path provided."""
        feature_importance = {"feat_1": 0.5, "feat_2": 0.3}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_chart.png"
            result = _create_importance_chart(feature_importance, output_path=output_path)

            assert result["path"] == str(output_path)
            assert os.path.exists(output_path)
            assert "image_base64" not in result

    def test_handles_empty_importance(self):
        """Should handle empty feature importance."""
        result = _create_importance_chart({})

        assert "error" in result

    def test_handles_single_feature(self):
        """Should handle single feature."""
        result = _create_importance_chart({"single": 1.0})

        assert len(result["features"]) == 1


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestCreateSelectionFunnel:
    """Tests for selection funnel chart creation."""

    def test_creates_funnel_chart(self):
        """Should create funnel chart with correct structure."""
        selection_history = [
            {"step": "variance_filter", "features_before": 100, "features_after": 80},
            {"step": "correlation_filter", "features_before": 80, "features_after": 50},
            {"step": "vif_filter", "features_before": 50, "features_after": 40},
        ]

        result = _create_selection_funnel(selection_history, original_count=100)

        assert result["type"] == "selection_funnel"
        assert len(result["steps"]) == 4  # Original + 3 steps
        assert result["steps"][0] == "Original"
        assert result["counts"][0] == 100
        assert result["counts"][-1] == 40

    def test_calculates_reduction_percentage(self):
        """Should calculate final reduction percentage."""
        selection_history = [
            {"step": "filter", "features_after": 50},
        ]

        result = _create_selection_funnel(selection_history, original_count=100)

        assert result["final_reduction"] == "50.0%"

    def test_handles_empty_history(self):
        """Should handle empty selection history."""
        result = _create_selection_funnel([], original_count=100)

        assert "error" in result

    def test_saves_to_file(self):
        """Should save funnel to file when path provided."""
        selection_history = [
            {"step": "filter", "features_after": 80},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "funnel.png"
            result = _create_selection_funnel(
                selection_history, original_count=100, output_path=output_path
            )

            assert os.path.exists(output_path)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestCreateStatisticsTable:
    """Tests for statistics table creation."""

    def test_creates_statistics_table(self):
        """Should create statistics table with correct structure."""
        feature_statistics = {
            "feature_a": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0, "null_pct": 0.0},
            "feature_b": {"mean": 100, "std": 25, "min": 50, "max": 200, "null_pct": 0.05},
        }

        result = _create_statistics_table(feature_statistics)

        assert result["type"] == "statistics_table"
        assert result["features_shown"] == 2
        assert "image_base64" in result

    def test_respects_top_n(self):
        """Should limit to top_n features."""
        feature_statistics = {
            f"feat_{i}": {"mean": i, "std": 1, "min": 0, "max": 10, "null_pct": 0}
            for i in range(30)
        }

        result = _create_statistics_table(feature_statistics, top_n=15)

        assert result["features_shown"] == 15
        assert result["total_features"] == 30

    def test_handles_empty_statistics(self):
        """Should handle empty statistics."""
        result = _create_statistics_table({})

        assert "error" in result

    def test_truncates_long_feature_names(self):
        """Should truncate long feature names."""
        feature_statistics = {
            "this_is_a_very_long_feature_name_that_exceeds_thirty_characters": {
                "mean": 0.5, "std": 0.1, "min": 0, "max": 1, "null_pct": 0
            },
        }

        result = _create_statistics_table(feature_statistics)

        # Should not fail
        assert result["features_shown"] == 1


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestCreateShapSummary:
    """Tests for SHAP summary chart creation."""

    def test_creates_shap_summary(self):
        """Should create SHAP summary chart."""
        np.random.seed(42)
        shap_values = np.random.rand(100, 5)
        feature_names = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]

        result = _create_shap_summary(shap_values, feature_names)

        assert result["type"] == "shap_summary"
        assert len(result["features"]) == 5
        assert len(result["mean_shap_values"]) == 5
        assert "image_base64" in result

    def test_respects_top_n(self):
        """Should limit to top_n features."""
        np.random.seed(42)
        shap_values = np.random.rand(100, 20)
        feature_names = [f"feat_{i}" for i in range(20)]

        result = _create_shap_summary(shap_values, feature_names, top_n=10)

        assert len(result["features"]) == 10

    def test_features_sorted_by_mean_shap(self):
        """Should sort features by mean absolute SHAP value."""
        np.random.seed(42)
        # Create SHAP values with known ordering
        shap_values = np.array([
            [0.1, 0.5, 0.3],  # feat_0 has low, feat_1 high, feat_2 medium
        ] * 100)
        feature_names = ["low", "high", "medium"]

        result = _create_shap_summary(shap_values, feature_names)

        assert result["features"][0] == "high"
        assert result["features"][-1] == "low"

    def test_handles_none_shap_values(self):
        """Should handle None SHAP values."""
        result = _create_shap_summary(None, ["feat_1"])

        assert "error" in result

    def test_handles_empty_feature_names(self):
        """Should handle empty feature names."""
        result = _create_shap_summary(np.random.rand(10, 3), [])

        assert "error" in result


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestFigToBase64:
    """Tests for base64 encoding."""

    def test_encodes_figure(self):
        """Should encode matplotlib figure to base64."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        result = _fig_to_base64(fig)

        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        # Should be PNG
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'

        plt.close(fig)


class TestGetOutputPath:
    """Tests for output path helper."""

    def test_returns_path_when_dir_provided(self):
        """Should return path when directory provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_output_path(tmpdir, "test.png")

            assert result is not None
            assert result.name == "test.png"
            assert str(result.parent) == tmpdir

    def test_creates_directory_if_not_exists(self):
        """Should create directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_subdir = os.path.join(tmpdir, "new_dir")
            result = _get_output_path(new_subdir, "test.png")

            assert os.path.exists(new_subdir)
            assert result is not None

    def test_returns_none_when_no_dir(self):
        """Should return None when no directory provided."""
        result = _get_output_path(None, "test.png")

        assert result is None


@pytest.mark.asyncio
@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestGenerateVisualizationsNode:
    """Integration tests for generate_visualizations node."""

    async def test_generates_all_visualizations(self):
        """Should generate all available visualizations."""
        np.random.seed(42)
        state = {
            "feature_importance": {
                "feat_1": 0.35,
                "feat_2": 0.25,
                "feat_3": 0.20,
            },
            "selection_history": [
                {"step": "variance", "features_after": 80},
                {"step": "correlation", "features_after": 50},
            ],
            "original_feature_count": 100,
            "feature_statistics": {
                "feat_1": {"mean": 0.5, "std": 0.1, "min": 0, "max": 1, "null_pct": 0},
            },
            "shap_values": np.random.rand(100, 3),
            "feature_names": ["feat_1", "feat_2", "feat_3"],
        }

        result = await generate_visualizations(state)

        assert result["visualizations_generated"] is True
        assert result["visualization_count"] == 4
        assert "importance_chart" in result["visualizations"]
        assert "selection_funnel" in result["visualizations"]
        assert "statistics_table" in result["visualizations"]
        assert "shap_summary" in result["visualizations"]

    async def test_generates_only_available_visualizations(self):
        """Should only generate visualizations for available data."""
        state = {
            "feature_importance": {
                "feat_1": 0.5,
            },
        }

        result = await generate_visualizations(state)

        assert result["visualizations_generated"] is True
        assert "importance_chart" in result["visualizations"]
        assert "selection_funnel" not in result["visualizations"]

    async def test_saves_to_output_directory(self):
        """Should save visualizations to output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "feature_importance": {"feat_1": 0.5},
                "visualization_output_dir": tmpdir,
            }

            result = await generate_visualizations(state)

            assert result["visualizations_generated"] is True
            assert os.path.exists(os.path.join(tmpdir, "feature_importance.png"))

    async def test_handles_empty_state(self):
        """Should handle empty state gracefully."""
        result = await generate_visualizations({})

        # No visualizations but should not error
        assert result["visualization_count"] == 0

    async def test_handles_visualization_error(self):
        """Should handle visualization errors gracefully."""
        state = {
            "feature_importance": "invalid_data",  # Should be dict
        }

        result = await generate_visualizations(state)

        # Should fail gracefully
        assert "visualization_error" in result or result["visualizations_generated"] is False


@pytest.mark.skipif(MATPLOTLIB_AVAILABLE, reason="Test when matplotlib unavailable")
class TestWithoutMatplotlib:
    """Tests for behavior when matplotlib is not available."""

    @pytest.mark.asyncio
    async def test_returns_error_without_matplotlib(self):
        """Should return error when matplotlib not available."""
        with patch.dict('sys.modules', {'matplotlib': None}):
            state = {"feature_importance": {"feat_1": 0.5}}

            result = await generate_visualizations(state)

            assert result["visualizations_generated"] is False
            assert "matplotlib" in result.get("visualization_error", "").lower()
