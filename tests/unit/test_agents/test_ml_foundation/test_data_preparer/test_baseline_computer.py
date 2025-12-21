"""Unit tests for baseline_computer node."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.baseline_computer import (
    compute_baseline_metrics,
)


@pytest.fixture
def mock_state_regression():
    """Create mock state for regression problem."""
    train_df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": ["cat1", "cat2", "cat1", "cat2"] * 25,
            "target": np.random.randn(100),
        }
    )

    return {
        "experiment_id": "exp_regression_123",
        "train_df": train_df,
        "scope_spec": {
            "experiment_id": "exp_regression_123",
            "required_features": ["feature1", "feature2", "feature3"],
            "prediction_target": "target",
        },
    }


@pytest.fixture
def mock_state_classification():
    """Create mock state for binary classification problem."""
    np.random.seed(42)
    train_df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.binomial(1, 0.3, 100),
        }
    )

    return {
        "experiment_id": "exp_classification_123",
        "train_df": train_df,
        "scope_spec": {
            "experiment_id": "exp_classification_123",
            "required_features": ["feature1", "feature2"],
            "prediction_target": "target",
        },
    }


@pytest.mark.asyncio
async def test_compute_baseline_metrics_numerical_features(mock_state_regression):
    """Test baseline computation for numerical features."""
    result = await compute_baseline_metrics(mock_state_regression)

    # Check feature stats computed
    assert "feature_stats" in result
    feature_stats = result["feature_stats"]

    # Check feature1 stats (numerical)
    assert "feature1" in feature_stats
    stats = feature_stats["feature1"]
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "p25" in stats
    assert "p50" in stats
    assert "p75" in stats
    assert stats["dtype"] == "numerical"

    # Check feature3 stats (categorical)
    assert "feature3" in feature_stats
    cat_stats = feature_stats["feature3"]
    assert "unique_count" in cat_stats
    assert "most_common" in cat_stats
    assert cat_stats["dtype"] == "categorical"


@pytest.mark.asyncio
async def test_compute_baseline_metrics_binary_target(mock_state_classification):
    """Test baseline computation for binary classification target."""
    result = await compute_baseline_metrics(mock_state_classification)

    # Check target rate computed
    assert "target_rate" in result
    assert result["target_rate"] is not None
    assert 0.0 <= result["target_rate"] <= 1.0

    # Check target distribution
    assert "target_distribution" in result
    target_dist = result["target_distribution"]
    assert target_dist["type"] == "binary"
    assert "positive_rate" in target_dist
    assert "negative_rate" in target_dist
    assert "positive_count" in target_dist
    assert "negative_count" in target_dist


@pytest.mark.asyncio
async def test_compute_baseline_metrics_correlation_matrix(mock_state_regression):
    """Test correlation matrix computation."""
    result = await compute_baseline_metrics(mock_state_regression)

    # Check correlation matrix computed for numerical features
    assert "correlation_matrix" in result
    corr_matrix = result["correlation_matrix"]

    # Should have correlations for numerical features only
    assert "feature1" in corr_matrix
    assert "feature2" in corr_matrix
    assert "feature3" not in corr_matrix  # Categorical, excluded

    # Check correlation values in valid range
    for _feature, correlations in corr_matrix.items():
        for _corr_feature, corr_value in correlations.items():
            assert -1.0 <= corr_value <= 1.0


@pytest.mark.asyncio
async def test_compute_baseline_training_samples_count(mock_state_regression):
    """Test that training sample count is recorded."""
    result = await compute_baseline_metrics(mock_state_regression)

    assert "training_samples" in result
    assert result["training_samples"] == 100  # From fixture


@pytest.mark.asyncio
async def test_compute_baseline_missing_train_df():
    """Test baseline computation with missing train_df."""
    state = {
        "experiment_id": "exp_test_123",
        # Missing train_df
        "scope_spec": {
            "required_features": ["feature1"],
        },
    }

    result = await compute_baseline_metrics(state)

    # Should handle error gracefully
    assert "error" in result
    assert result["error_type"] == "baseline_computation_error"


@pytest.mark.asyncio
async def test_compute_baseline_only_from_train_split(mock_state_regression):
    """CRITICAL: Test that baseline is ONLY computed from train split.

    This test ensures we never leak validation/test/holdout data into baselines.
    """
    # Add other splits to state
    state = mock_state_regression.copy()
    state["validation_df"] = pd.DataFrame({"feature1": [999, 999, 999]})
    state["test_df"] = pd.DataFrame({"feature1": [888, 888, 888]})

    result = await compute_baseline_metrics(state)

    # Baseline should only reflect train data
    feature_stats = result["feature_stats"]["feature1"]

    # If validation/test data leaked, mean would be very different
    # Train data has ~N(0,1), so mean should be close to 0
    assert abs(feature_stats["mean"]) < 0.5  # Should be near 0, not 999 or 888

    # Training samples should match train_df only
    assert result["training_samples"] == 100


@pytest.mark.asyncio
async def test_compute_baseline_timestamp(mock_state_regression):
    """Test that computed_at timestamp is generated."""
    result = await compute_baseline_metrics(mock_state_regression)

    assert "computed_at" in result
    # Should be valid ISO timestamp
    datetime.fromisoformat(result["computed_at"])
