"""Tests for resampling application node.

Tests the apply_resampling function and its helper functions.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.apply_resampling import (
    _apply_strategy,
    _ensure_numpy,
    apply_resampling,
)

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def smote_state():
    """Create state for active SMOTE resampling."""
    np.random.seed(42)
    return {
        "recommended_strategy": "smote",
        "imbalance_detected": True,
        "X_train_preprocessed": np.random.rand(100, 5),
        "train_data": {"y": np.array([0] * 80 + [1] * 20)},
        "class_distribution": {0: 80, 1: 20},
    }


@pytest.fixture
def no_resampling_state():
    """Create state where resampling should be skipped."""
    np.random.seed(42)
    return {
        "recommended_strategy": "none",
        "imbalance_detected": False,
        "X_train_preprocessed": np.random.rand(100, 5),
        "train_data": {"y": np.array([0] * 50 + [1] * 50)},
        "class_distribution": {0: 50, 1: 50},
    }


@pytest.fixture
def extreme_state():
    """Create state with extreme imbalance for adaptive k_neighbors."""
    np.random.seed(42)
    return {
        "recommended_strategy": "combined",
        "imbalance_detected": True,
        "X_train_preprocessed": np.random.rand(100, 5),
        "train_data": {"y": np.array([0] * 97 + [1] * 3)},
        "class_distribution": {0: 97, 1: 3},
    }


# ============================================================================
# Test _ensure_numpy
# ============================================================================


class TestEnsureNumpy:
    """Test numpy conversion utility."""

    def test_none_returns_none(self):
        """Should return None for None input."""
        assert _ensure_numpy(None) is None

    def test_ndarray_unchanged(self):
        """Should return same ndarray object."""
        arr = np.array([1, 2, 3])
        assert _ensure_numpy(arr) is arr

    def test_converts_dataframe(self):
        """Should convert DataFrame to ndarray."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _ensure_numpy(df)
        assert isinstance(result, np.ndarray)

    def test_converts_series(self):
        """Should convert Series to ndarray."""
        s = pd.Series([1, 2, 3])
        result = _ensure_numpy(s)
        assert isinstance(result, np.ndarray)

    def test_converts_list(self):
        """Should convert list to ndarray."""
        result = _ensure_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_converts_tuple(self):
        """Should convert tuple to ndarray."""
        result = _ensure_numpy((1, 2, 3))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])


# ============================================================================
# Test _apply_strategy
# ============================================================================


class TestApplyStrategyDispatch:
    """Test strategy dispatch function."""

    def test_smote_increases_samples(self):
        """SMOTE should increase total sample count."""
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_out, y_out = _apply_strategy(X, y, "smote", 20)
        assert X_out.shape[0] > 100

    def test_random_oversample_balances(self):
        """Random oversampling should increase minority count."""
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        _, y_out = _apply_strategy(X, y, "random_oversample", 20)
        minority_count = np.sum(y_out == 1)
        assert minority_count > 20

    def test_random_undersample_reduces(self):
        """Random undersampling should reduce total sample count."""
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_out, _ = _apply_strategy(X, y, "random_undersample", 20)
        assert X_out.shape[0] < 100

    def test_smote_tomek_resamples(self):
        """SMOTE-Tomek should change dataset shape."""
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_out, _ = _apply_strategy(X, y, "smote_tomek", 20)
        assert X_out.shape[0] != 100

    def test_combined_partial_oversample(self):
        """Combined strategy should increase samples less than full balance."""
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        X_out, _ = _apply_strategy(X, y, "combined", 20)
        assert X_out.shape[0] > 100
        # Full balance would be 160 (80+80), combined targets 50% ratio
        assert X_out.shape[0] < 160

    def test_none_returns_original(self):
        """Strategy 'none' should return original arrays."""
        X = np.random.rand(10, 3)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X_out, y_out = _apply_strategy(X, y, "none", 5)
        assert X_out is X

    def test_class_weight_returns_original(self):
        """Strategy 'class_weight' should return original arrays."""
        X = np.random.rand(10, 3)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X_out, y_out = _apply_strategy(X, y, "class_weight", 5)
        assert X_out is X


# ============================================================================
# Test apply_resampling (main function)
# ============================================================================


@pytest.mark.asyncio
class TestApplyResampling:
    """Test main resampling application function."""

    # ---- Skip-path tests ----

    async def test_skips_when_no_imbalance(self, no_resampling_state):
        """Should skip resampling when no imbalance detected."""
        result = await apply_resampling(no_resampling_state)
        assert result["resampling_applied"] is False

    async def test_skips_for_none_strategy(self, smote_state):
        """Should skip resampling when strategy is 'none'."""
        smote_state["recommended_strategy"] = "none"
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is False

    async def test_skips_for_class_weight(self, smote_state):
        """Should skip resampling when strategy is 'class_weight'."""
        smote_state["recommended_strategy"] = "class_weight"
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is False

    async def test_skips_when_x_is_none(self, smote_state):
        """Should skip resampling when X_train is None."""
        smote_state["X_train_preprocessed"] = None
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is False

    async def test_skips_when_y_is_none(self, smote_state):
        """Should skip resampling when y_train is None."""
        smote_state["train_data"] = {"y": None}
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is False

    # ---- Strategy tests ----

    async def test_smote_applied(self, smote_state):
        """Should apply SMOTE and increase sample count."""
        pytest.importorskip("imblearn")
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is True
        assert result["resampled_train_shape"][0] > result["original_train_shape"][0]

    async def test_random_oversample_applied(self, smote_state):
        """Should apply random oversampling."""
        pytest.importorskip("imblearn")
        smote_state["recommended_strategy"] = "random_oversample"
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is True

    async def test_random_undersample_applied(self, smote_state):
        """Should apply random undersampling and reduce sample count."""
        pytest.importorskip("imblearn")
        smote_state["recommended_strategy"] = "random_undersample"
        result = await apply_resampling(smote_state)
        assert result["resampling_applied"] is True
        assert result["resampled_train_shape"][0] < result["original_train_shape"][0]

    async def test_returns_all_output_keys(self, smote_state):
        """Should return all expected output keys."""
        pytest.importorskip("imblearn")
        result = await apply_resampling(smote_state)
        expected_keys = {
            "X_train_resampled",
            "y_train_resampled",
            "resampling_applied",
            "resampling_strategy",
            "original_train_shape",
            "resampled_train_shape",
            "original_distribution",
            "resampled_distribution",
        }
        assert set(result.keys()) == expected_keys

    async def test_distributions_tracked(self, smote_state):
        """Should track different distributions after resampling."""
        pytest.importorskip("imblearn")
        result = await apply_resampling(smote_state)
        assert result["original_distribution"] != result["resampled_distribution"]

    # ---- Error tests ----

    async def test_handles_import_error(self, smote_state):
        """Should handle missing imblearn dependency."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.apply_resampling._apply_strategy",
            side_effect=ImportError("No module named 'imblearn'"),
        ):
            result = await apply_resampling(smote_state)
        assert result["error_type"] == "missing_dependency"

    async def test_handles_resampling_error(self, smote_state):
        """Should handle resampling failures."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.apply_resampling._apply_strategy",
            side_effect=ValueError("Resampling failed"),
        ):
            result = await apply_resampling(smote_state)
        assert result["error_type"] == "resampling_error"
