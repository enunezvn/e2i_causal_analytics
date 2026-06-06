"""Tests for class imbalance detection node.

Tests the deterministic matrix-driven `detect_class_imbalance` and its
helper functions (Block 6A — replaces the prior LLM-based path).
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance import (
    SEVERITY_THRESHOLDS,
    VALID_STRATEGIES,
    _calculate_imbalance_metrics,
    _load_imbalance_config,
    _lookup_strategy,
    detect_class_imbalance,
)

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def balanced_state():
    """Create state with balanced class distribution."""
    np.random.seed(42)
    return {
        "train_data": {"y": np.array([0] * 50 + [1] * 50)},
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
    }


@pytest.fixture
def moderate_imbalance_state():
    """Create state with moderate class imbalance (75/25)."""
    np.random.seed(42)
    return {
        "train_data": {"y": np.array([0] * 75 + [1] * 25)},
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
    }


@pytest.fixture
def severe_imbalance_state():
    """Create state with severe class imbalance (90/10)."""
    np.random.seed(42)
    return {
        "train_data": {"y": np.array([0] * 90 + [1] * 10)},
        "algorithm_name": "LogisticRegression",
        "problem_type": "binary_classification",
    }


@pytest.fixture
def extreme_imbalance_state():
    """Create state with extreme class imbalance (97/3)."""
    np.random.seed(42)
    return {
        "train_data": {"y": np.array([0] * 97 + [1] * 3)},
        "algorithm_name": "LogisticRegression",
        "problem_type": "binary_classification",
    }


# ============================================================================
# Test _calculate_imbalance_metrics
# ============================================================================


class TestCalculateImbalanceMetrics:
    """Test imbalance metrics calculation."""

    def test_balanced_returns_severity_none(self):
        """Should return severity='none' for balanced data."""
        y = np.array([0] * 50 + [1] * 50)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "none"
        assert metrics["minority_ratio"] == 0.5

    def test_moderate_imbalance_severity(self):
        """Should return severity='moderate' for 70/30 split."""
        y = np.array([0] * 70 + [1] * 30)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "moderate"

    def test_severe_imbalance_severity(self):
        """Should return severity='severe' for 90/10 split."""
        y = np.array([0] * 90 + [1] * 10)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "severe"

    def test_extreme_imbalance_severity(self):
        """Should return severity='extreme' for 98/2 split."""
        y = np.array([0] * 98 + [1] * 2)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "extreme"

    def test_boundary_at_40_percent_is_none(self):
        """Should return severity='none' when minority is exactly 40%."""
        y = np.array([0] * 60 + [1] * 40)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "none"

    def test_boundary_below_40_percent_is_moderate(self):
        """Should return severity='moderate' when minority is 39%."""
        y = np.array([0] * 61 + [1] * 39)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["severity"] == "moderate"

    def test_class_distribution_dict(self):
        """Should return correct class distribution dictionary."""
        y = np.array([0] * 80 + [1] * 20)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["class_distribution"] == {0: 80, 1: 20}

    def test_imbalance_ratio(self):
        """Should compute correct imbalance ratio."""
        y = np.array([0] * 80 + [1] * 20)
        metrics = _calculate_imbalance_metrics(y)
        assert metrics["imbalance_ratio"] == 4.0

    def test_severity_thresholds_match_default_config(self):
        """The legacy SEVERITY_THRESHOLDS module dict must mirror the
        bands declared in `config/imbalance_strategy.yaml`."""
        config = _load_imbalance_config()
        assert SEVERITY_THRESHOLDS["none"] == config["severity_bands"]["none"]
        assert SEVERITY_THRESHOLDS["moderate"] == config["severity_bands"]["moderate"]
        assert SEVERITY_THRESHOLDS["severe"] == config["severity_bands"]["severe"]
        assert SEVERITY_THRESHOLDS["extreme"] == 0.0


# ============================================================================
# Test _lookup_strategy (deterministic matrix replacement for the old
# `_heuristic_strategy` + LLM path)
# ============================================================================


class TestLookupStrategy:
    """Test deterministic matrix-driven strategy selection."""

    def test_none_severity_returns_none(self):
        """Should return 'none' for no imbalance."""
        metrics = {"severity": "none", "minority_count": 50, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "RandomForest", "binary_classification")
        assert strategy == "none"

    def test_moderate_tree_returns_class_weight(self):
        """Should return 'class_weight' for moderate imbalance with tree model."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "XGBoost", "binary_classification")
        assert strategy == "class_weight"

    def test_moderate_non_tree_returns_oversample(self):
        """Should return 'random_oversample' for moderate imbalance with non-tree model."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
        assert strategy == "random_oversample"

    def test_severe_non_tree_returns_class_weight(self):
        """Re-tune 2026-06-06: severe non-tree now uses class_weight (not SMOTE).

        At severe imbalance the HPO objective is average_precision (PR-AUC), and
        synthetic oversampling degrades PR-AUC for linear models; class_weight is
        the cost-sensitive alternative regardless of minority count.
        """
        for mc in (10, 5):
            metrics = {"severity": "severe", "minority_count": mc, "total_samples": 100}
            strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
            assert strategy == "class_weight"

    def test_extreme_non_tree_returns_class_weight(self):
        """Re-tune 2026-06-06: extreme non-tree now uses class_weight (not 'combined').

        The prior SMOTE-based 'combined' measured at/below the no-resampling
        PR-AUC baseline on the feature-bound Optum mart; class_weight only.
        """
        for mc in (15, 7, 3):
            metrics = {"severity": "extreme", "minority_count": mc, "total_samples": 100}
            strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
            assert strategy == "class_weight"

    def test_extreme_tiny_minority_returns_class_weight(self):
        """Should return 'class_weight' for extreme imbalance with very few samples."""
        metrics = {"severity": "extreme", "minority_count": 3, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
        assert strategy == "class_weight"

    @pytest.mark.parametrize(
        "model_name",
        [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "GradientBoosting",
            "CausalForest",
        ],
    )
    def test_all_tree_models_moderate(self, model_name):
        """Should return 'class_weight' for all tree models with moderate imbalance."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, model_name, "binary_classification")
        assert strategy == "class_weight"

    @pytest.mark.parametrize(
        "model_name",
        [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "GradientBoosting",
            "CausalForest",
        ],
    )
    def test_severe_tree_returns_class_weight(self, model_name):
        """Tree models should get class_weight instead of SMOTE at severe imbalance."""
        metrics = {"severity": "severe", "minority_count": 15, "total_samples": 100}
        strategy, rationale = _lookup_strategy(metrics, model_name, "binary_classification")
        assert strategy == "class_weight"
        assert "memorization" in rationale.lower() or "synthetic" in rationale.lower()

    @pytest.mark.parametrize(
        "model_name",
        [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "GradientBoosting",
            "CausalForest",
        ],
    )
    def test_extreme_tree_returns_class_weight(self, model_name):
        """Tree models should get class_weight instead of combined at extreme imbalance."""
        metrics = {"severity": "extreme", "minority_count": 15, "total_samples": 1000}
        strategy, rationale = _lookup_strategy(metrics, model_name, "binary_classification")
        assert strategy == "class_weight"
        assert "memorization" in rationale.lower() or "synthetic" in rationale.lower()

    def test_severe_non_tree_returns_class_weight_high_count(self):
        """Re-tune 2026-06-06: severe non-tree uses class_weight even at high
        minority counts (was SMOTE) — cost-sensitive over synthetic oversampling."""
        metrics = {"severity": "severe", "minority_count": 15, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
        assert strategy == "class_weight"

    def test_extreme_non_tree_returns_class_weight_high_count(self):
        """Re-tune 2026-06-06: extreme non-tree uses class_weight even at high
        minority counts (was 'combined') — no synthetic minority generation."""
        metrics = {"severity": "extreme", "minority_count": 15, "total_samples": 100}
        strategy, _ = _lookup_strategy(metrics, "LogisticRegression", "binary_classification")
        assert strategy == "class_weight"

    def test_strategy_in_valid_strategies(self):
        """Every matrix branch must yield a strategy in VALID_STRATEGIES."""
        for severity in ("none", "moderate", "severe", "extreme"):
            for mc in (50, 25, 15, 10, 7, 5, 3, 0):
                for alg in ("XGBoost", "LogisticRegression"):
                    metrics = {
                        "severity": severity,
                        "minority_count": mc,
                        "total_samples": 100,
                    }
                    strategy, _ = _lookup_strategy(metrics, alg, "binary_classification")
                    assert strategy in VALID_STRATEGIES, (
                        f"strategy={strategy!r} for severity={severity}, mc={mc}, alg={alg}"
                    )


# ============================================================================
# Determinism guarantee (Block 6A explicit ask, Finding #16)
# ============================================================================


class TestDeterministicStrategyMatrix:
    """Two calls with the same inputs must produce byte-identical outputs."""

    def test_deterministic_strategy_matrix(self):
        """The matrix lookup must be byte-identical across repeated calls."""
        metrics = {"severity": "extreme", "minority_count": 15, "total_samples": 100}
        s1, r1 = _lookup_strategy(metrics, "XGBoost", "binary_classification")
        s2, r2 = _lookup_strategy(metrics, "XGBoost", "binary_classification")
        assert s1 == s2
        assert r1 == r2

    @pytest.mark.parametrize(
        "severity,minority_count,algorithm",
        [
            ("none", 50, "RandomForest"),
            ("moderate", 25, "XGBoost"),
            ("moderate", 25, "LogisticRegression"),
            ("severe", 15, "RandomForest"),
            ("severe", 10, "LogisticRegression"),
            ("severe", 5, "LogisticRegression"),
            ("extreme", 15, "XGBoost"),
            ("extreme", 15, "LogisticRegression"),
            ("extreme", 7, "LogisticRegression"),
            ("extreme", 3, "LogisticRegression"),
        ],
    )
    def test_repeated_calls_identical(self, severity, minority_count, algorithm):
        """Determinism across the full default-matrix coverage table."""
        metrics = {
            "severity": severity,
            "minority_count": minority_count,
            "total_samples": 100,
        }
        results = [_lookup_strategy(metrics, algorithm, "binary_classification") for _ in range(5)]
        # All five tuples must compare equal.
        assert all(r == results[0] for r in results), (
            f"non-deterministic output for ({severity}, mc={minority_count}, {algorithm}): "
            f"{results}"
        )

    @pytest.mark.asyncio
    async def test_detect_class_imbalance_is_deterministic(self, severe_imbalance_state):
        """Top-level node must produce identical dicts across runs."""
        # Use deepcopy-style dicts to avoid any state mutation surprise.
        import copy

        s1 = copy.deepcopy(severe_imbalance_state)
        s2 = copy.deepcopy(severe_imbalance_state)
        r1 = await detect_class_imbalance(s1)
        r2 = await detect_class_imbalance(s2)
        assert r1 == r2
        assert r1["recommended_strategy"] == r2["recommended_strategy"]
        assert r1["strategy_rationale"] == r2["strategy_rationale"]


# ============================================================================
# Test detect_class_imbalance (main function)
# ============================================================================


@pytest.mark.asyncio
class TestDetectClassImbalance:
    """Test main class imbalance detection function."""

    async def test_safe_defaults_when_no_labels(self):
        """Should return safe defaults when no training labels available."""
        state = {"train_data": {"y": None}}
        result = await detect_class_imbalance(state)
        assert result["imbalance_detected"] is False
        assert result["imbalance_severity"] == "unknown"

    async def test_not_applicable_for_regression(self):
        """Should return not_applicable for regression problems."""
        state = {
            "train_data": {"y": np.array([0.1, 0.5, 0.9])},
            "problem_type": "regression",
        }
        result = await detect_class_imbalance(state)
        assert result["imbalance_severity"] == "not_applicable"

    async def test_not_applicable_for_continuous(self):
        """Should return not_applicable for continuous problems."""
        state = {
            "train_data": {"y": np.array([0.1, 0.5, 0.9])},
            "problem_type": "continuous",
        }
        result = await detect_class_imbalance(state)
        assert result["imbalance_severity"] == "not_applicable"

    async def test_degenerate_single_class(self):
        """Should return degenerate severity for single-class data."""
        state = {
            "train_data": {"y": np.array([0] * 100)},
            "problem_type": "binary_classification",
        }
        result = await detect_class_imbalance(state)
        assert result["imbalance_severity"] == "degenerate"

    async def test_balanced_no_imbalance(self, balanced_state):
        """Should detect no imbalance in balanced data."""
        result = await detect_class_imbalance(balanced_state)
        assert result["imbalance_detected"] is False
        assert result["imbalance_severity"] == "none"

    async def test_balanced_returns_none_strategy(self, balanced_state):
        """Balanced data must skip the matrix lookup and emit strategy='none'.

        Replaces the legacy `test_balanced_skips_llm` — same observable
        outcome (the deterministic lookup is bypassed for balanced data),
        without depending on the LLM helper that no longer exists.
        """
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._lookup_strategy"
        ) as mock_lookup:
            result = await detect_class_imbalance(balanced_state)
        mock_lookup.assert_not_called()
        assert result["recommended_strategy"] == "none"
        assert result["imbalance_detected"] is False

    async def test_detects_moderate_imbalance(self, moderate_imbalance_state):
        """Should detect moderate imbalance and pick the matrix-default strategy.

        With a tree-model algorithm (RandomForest) the deterministic
        matrix returns class_weight at moderate severity.
        """
        result = await detect_class_imbalance(moderate_imbalance_state)
        assert result["imbalance_detected"] is True
        assert result["imbalance_severity"] == "moderate"
        assert result["recommended_strategy"] == "class_weight"
        assert result["recommended_strategy"] in VALID_STRATEGIES

    async def test_detects_severe_imbalance(self, severe_imbalance_state):
        """Should detect severe imbalance and pick the matrix-default strategy.

        Re-tune 2026-06-06: non-tree algorithm (LogisticRegression) at severe
        imbalance → class_weight (was smote) — cost-sensitive over resampling.
        """
        result = await detect_class_imbalance(severe_imbalance_state)
        assert result["imbalance_severity"] == "severe"
        assert result["recommended_strategy"] == "class_weight"

    async def test_detects_extreme_imbalance(self, extreme_imbalance_state):
        """Should detect extreme imbalance and pick the matrix-default strategy.

        Non-tree algorithm with minority_count=3 → class_weight per the
        default matrix (too few samples for resampling).
        """
        result = await detect_class_imbalance(extreme_imbalance_state)
        assert result["imbalance_severity"] == "extreme"
        assert result["recommended_strategy"] == "class_weight"
        assert result["recommended_strategy"] in VALID_STRATEGIES

    async def test_returns_all_output_keys(self, severe_imbalance_state):
        """Should return all expected output keys."""
        result = await detect_class_imbalance(severe_imbalance_state)
        expected_keys = {
            "imbalance_detected",
            "imbalance_ratio",
            "minority_ratio",
            "imbalance_severity",
            "class_distribution",
            "recommended_strategy",
            "strategy_rationale",
        }
        assert set(result.keys()) == expected_keys

    async def test_lookup_invoked_for_imbalanced_data(self, severe_imbalance_state):
        """Imbalanced data must dispatch to the deterministic matrix lookup."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._lookup_strategy",
            return_value=("smote", "test rationale"),
        ) as mock_lookup:
            result = await detect_class_imbalance(severe_imbalance_state)
        mock_lookup.assert_called_once()
        # The patched lookup is sync; ensure no AsyncMock is needed.
        assert not isinstance(mock_lookup, AsyncMock)
        assert result["recommended_strategy"] == "smote"
        assert result["strategy_rationale"] == "test rationale"
