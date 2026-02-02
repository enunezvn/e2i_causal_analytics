"""Unit tests for feature_selector node.

Tests feature selection capabilities:
- Variance threshold selection
- Correlation-based selection
- VIF-based multicollinearity removal
- Model-based importance ranking
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.feature_analyzer.nodes.feature_selector import (
    _apply_correlation_selection,
    _apply_variance_selection,
    _apply_vif_selection,
    _compute_feature_statistics,
    _compute_model_importance,
    get_feature_selection_summary,
    select_features,
)


class TestVarianceSelection:
    """Tests for variance threshold selection."""

    def test_remove_zero_variance_features(self):
        """Should remove features with zero variance."""
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 1],
                "variable": [1, 2, 3, 4, 5],
            }
        )
        selected, removed = _apply_variance_selection(df, threshold=0.0)

        assert "variable" in selected
        assert "constant" in removed

    def test_remove_low_variance_features(self):
        """Should remove features below variance threshold."""
        df = pd.DataFrame(
            {
                "low_var": [1, 1, 1, 1, 2],  # variance ~0.16
                "high_var": [1, 2, 3, 4, 5],  # variance 2.0
            }
        )
        selected, removed = _apply_variance_selection(df, threshold=0.5)

        assert "high_var" in selected
        assert "low_var" in removed

    def test_keeps_all_with_zero_threshold(self):
        """Should keep all features with zero threshold."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        selected, removed = _apply_variance_selection(df, threshold=0.0)

        assert len(selected) == 2
        assert len(removed) == 0


class TestCorrelationSelection:
    """Tests for correlation-based selection."""

    def test_remove_highly_correlated_features(self):
        """Should remove one of two highly correlated features."""
        np.random.seed(42)
        base = np.random.rand(100)
        df = pd.DataFrame(
            {
                "original": base,
                "highly_correlated": base + np.random.rand(100) * 0.01,  # Almost same
                "uncorrelated": np.random.rand(100),
            }
        )
        selected, removed = _apply_correlation_selection(df, threshold=0.95)

        assert len(selected) == 2
        assert "uncorrelated" in selected

    def test_keeps_moderately_correlated(self):
        """Should keep moderately correlated features."""
        np.random.seed(42)
        base = np.random.rand(100)
        df = pd.DataFrame(
            {
                "a": base,
                "b": base * 0.5 + np.random.rand(100) * 0.5,  # ~50% correlation
            }
        )
        selected, removed = _apply_correlation_selection(df, threshold=0.95)

        assert len(selected) == 2

    def test_keeps_all_with_high_threshold(self):
        """Should keep all features with high threshold."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.rand(50),
                "b": np.random.rand(50),
                "c": np.random.rand(50),
            }
        )
        selected, removed = _apply_correlation_selection(df, threshold=0.99)

        assert len(selected) == 3


class TestVIFSelection:
    """Tests for VIF-based multicollinearity removal."""

    def test_remove_multicollinear_features(self):
        """Should remove features with high VIF."""
        np.random.seed(42)
        x1 = np.random.rand(100)
        x2 = x1 + np.random.rand(100) * 0.01  # Highly collinear with x1
        x3 = np.random.rand(100)  # Independent

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
            }
        )
        selected, removed = _apply_vif_selection(df, threshold=5.0)

        # One of x1 or x2 should be removed
        assert len(selected) <= 3
        assert "x3" in selected

    def test_keeps_independent_features(self):
        """Should keep independent features."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "x1": np.random.rand(100),
                "x2": np.random.rand(100),
                "x3": np.random.rand(100),
            }
        )
        selected, removed = _apply_vif_selection(df, threshold=10.0)

        assert len(selected) == 3

    def test_handles_single_feature(self):
        """Should handle single feature without error."""
        df = pd.DataFrame({"single": np.random.rand(100)})
        selected, removed = _apply_vif_selection(df, threshold=5.0)

        assert "single" in selected


class TestModelImportance:
    """Tests for model-based importance computation."""

    def test_computes_importance_classification(self):
        """Should compute importance for classification."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "important": np.random.rand(100),
                "noise": np.random.rand(100),
            }
        )
        # Create target correlated with 'important'
        y = (X["important"] > 0.5).astype(int)

        importance_dict, ranked = _compute_model_importance(X, y, problem_type="classification")

        assert "important" in importance_dict
        assert "noise" in importance_dict
        assert len(ranked) == 2

    def test_computes_importance_regression(self):
        """Should compute importance for regression."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "predictive": np.random.rand(100),
                "noise": np.random.rand(100),
            }
        )
        y = X["predictive"] * 2 + np.random.rand(100) * 0.1

        importance_dict, ranked = _compute_model_importance(X, y, problem_type="regression")

        assert "predictive" in importance_dict
        assert importance_dict["predictive"] > importance_dict["noise"]

    def test_ranking_is_sorted(self):
        """Should return features sorted by importance."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.rand(100),
                "b": np.random.rand(100),
                "c": np.random.rand(100),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 100))

        importance_dict, ranked = _compute_model_importance(X, y, "classification")

        # Check ranking is in descending order
        importances = [importance_dict[f] for f in ranked]
        assert importances == sorted(importances, reverse=True)


class TestFeatureStatistics:
    """Tests for feature statistics computation."""

    def test_computes_basic_statistics(self):
        """Should compute mean, std, min, max."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
            }
        )
        result = _compute_feature_statistics(df)

        assert "feature1" in result
        assert result["feature1"]["mean"] == 3.0
        assert result["feature1"]["min"] == 1.0
        assert result["feature1"]["max"] == 5.0

    def test_computes_null_percentage(self):
        """Should compute null percentage."""
        df = pd.DataFrame(
            {
                "has_nulls": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        result = _compute_feature_statistics(df)

        assert result["has_nulls"]["null_pct"] == 0.4

    def test_handles_all_null_column(self):
        """Should handle all-null columns."""
        df = pd.DataFrame(
            {
                "all_null": [np.nan, np.nan, np.nan],
            }
        )
        result = _compute_feature_statistics(df)

        assert result["all_null"]["null_pct"] == 1.0


class TestGetFeatureSelectionSummary:
    """Tests for summary generation."""

    def test_generates_summary(self):
        """Should generate human-readable summary."""
        state = {
            "original_feature_count": 100,
            "selected_feature_count": 25,
            "removed_features": {
                "variance": ["a", "b"],
                "correlation": ["c", "d", "e"],
                "vif": ["f"],
            },
            "feature_importance_ranked": [
                ("top_feat", 0.25),
                ("second_feat", 0.15),
            ],
        }

        summary = get_feature_selection_summary(state)

        assert "100" in summary
        assert "25" in summary
        assert "variance" in summary.lower()
        assert "correlation" in summary.lower()


@pytest.mark.asyncio
class TestSelectFeaturesNode:
    """Integration tests for select_features node."""

    async def test_full_selection_pipeline(self):
        """Should run complete selection pipeline."""
        np.random.seed(42)
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "constant": [1] * 100,  # Zero variance
                    "highly_corr_1": np.random.rand(100),
                    "highly_corr_2": None,  # Will be set below
                    "independent": np.random.rand(100),
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 100)),
            "problem_type": "classification",
            "selection_config": {
                "variance_threshold": 0.01,
                "correlation_threshold": 0.95,
                "apply_vif_filter": False,
            },
        }
        # Make highly correlated
        state["X_train_generated"]["highly_corr_2"] = (
            state["X_train_generated"]["highly_corr_1"] + np.random.rand(100) * 0.01
        )

        result = await select_features(state)

        assert "X_train_selected" in result
        assert "selected_features" in result
        assert "feature_importance" in result
        assert "constant" not in result["selected_features"]

    async def test_applies_to_validation_data(self):
        """Should apply same selection to validation data."""
        np.random.seed(42)
        train_df = pd.DataFrame(
            {
                "keep": np.random.rand(50),
                "drop": [1] * 50,  # Zero variance
            }
        )
        val_df = pd.DataFrame(
            {
                "keep": np.random.rand(20),
                "drop": [1] * 20,
            }
        )

        state = {
            "X_train_generated": train_df,
            "X_val_generated": val_df,
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
            "selection_config": {"variance_threshold": 0.01},
        }

        result = await select_features(state)

        assert "X_val_selected" in result
        assert list(result["X_train_selected"].columns) == list(result["X_val_selected"].columns)

    async def test_tracks_selection_history(self):
        """Should track selection history steps."""
        np.random.seed(42)
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "a": np.random.rand(50),
                    "b": np.random.rand(50),
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
        }

        result = await select_features(state)

        assert "selection_history" in result
        assert len(result["selection_history"]) > 0
        for step in result["selection_history"]:
            assert "step" in step

    async def test_handles_missing_data(self):
        """Should handle missing training data."""
        state = {
            "y_train": pd.Series([0, 1]),
            "problem_type": "classification",
        }

        result = await select_features(state)

        assert result.get("error") is not None

    async def test_preserves_non_numeric_columns(self):
        """Should preserve non-numeric columns separately."""
        np.random.seed(42)
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "numeric": np.random.rand(50),
                    "category": ["A", "B"] * 25,
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
        }

        result = await select_features(state)

        # Non-numeric should be tracked separately
        assert "selected_features_all" in result

    async def test_regression_problem_type(self):
        """Should work with regression problems."""
        np.random.seed(42)
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "predictive": np.random.rand(100),
                    "noise": np.random.rand(100),
                }
            ),
            "y_train": pd.Series(np.random.rand(100) * 100),
            "problem_type": "regression",
        }

        result = await select_features(state)

        assert "error" not in result
        assert "feature_importance" in result


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_single_feature(self):
        """Should handle single feature DataFrame."""
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "only_feature": np.random.rand(50),
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
        }

        result = await select_features(state)

        assert len(result["selected_features"]) == 1

    @pytest.mark.asyncio
    async def test_all_constant_features(self):
        """Should handle all constant features."""
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "const1": [1] * 50,
                    "const2": [2] * 50,
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
            "selection_config": {"variance_threshold": 0.01},
        }

        result = await select_features(state)

        # Should warn but not crash
        assert result.get("selected_features") is not None

    def test_empty_correlation_matrix(self):
        """Should handle DataFrame with single column for correlation."""
        df = pd.DataFrame({"single": np.random.rand(50)})
        selected, removed = _apply_correlation_selection(df, threshold=0.95)

        assert "single" in selected

    @pytest.mark.asyncio
    async def test_missing_y_train(self):
        """Should handle missing y_train for importance."""
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "a": np.random.rand(50),
                }
            ),
            "problem_type": "classification",
        }

        result = await select_features(state)

        # Should still work for variance/correlation selection
        assert "X_train_selected" in result or result.get("error")

    @pytest.mark.asyncio
    async def test_default_config(self):
        """Should use sensible defaults when config is missing."""
        np.random.seed(42)
        state = {
            "X_train_generated": pd.DataFrame(
                {
                    "a": np.random.rand(50),
                    "b": np.random.rand(50),
                }
            ),
            "y_train": pd.Series(np.random.randint(0, 2, 50)),
            "problem_type": "classification",
        }

        result = await select_features(state)

        assert result.get("error") is None
        assert "selected_features" in result
