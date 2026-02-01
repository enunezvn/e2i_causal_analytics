"""Tests for class imbalance detection node.

Tests the detect_class_imbalance function and its helper functions.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance import (
    detect_class_imbalance,
    _calculate_imbalance_metrics,
    _get_llm_recommendation,
    _heuristic_strategy,
    SEVERITY_THRESHOLDS,
    VALID_STRATEGIES,
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


@pytest.fixture
def mock_anthropic_client():
    """Factory for creating mock anthropic module with configurable response."""
    def _create(response_text):
        mock_module = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_text)]
        mock_module.Anthropic.return_value.messages.create.return_value = mock_message
        return mock_module

    return _create


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


# ============================================================================
# Test _heuristic_strategy
# ============================================================================


class TestHeuristicStrategy:
    """Test heuristic strategy selection."""

    def test_none_severity_returns_none(self):
        """Should return 'none' for no imbalance."""
        metrics = {"severity": "none", "minority_count": 50, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "RandomForest")
        assert strategy == "none"

    def test_moderate_tree_returns_class_weight(self):
        """Should return 'class_weight' for moderate imbalance with tree model."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "XGBoost")
        assert strategy == "class_weight"

    def test_moderate_non_tree_returns_oversample(self):
        """Should return 'random_oversample' for moderate imbalance with non-tree model."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "random_oversample"

    def test_severe_enough_minority_returns_smote(self):
        """Should return 'smote' for severe imbalance with enough minority samples."""
        metrics = {"severity": "severe", "minority_count": 10, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "smote"

    def test_severe_few_minority_returns_oversample(self):
        """Should return 'random_oversample' for severe imbalance with few minority samples."""
        metrics = {"severity": "severe", "minority_count": 5, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "random_oversample"

    def test_extreme_large_minority_returns_combined(self):
        """Should return 'combined' for extreme imbalance with enough samples."""
        metrics = {"severity": "extreme", "minority_count": 15, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "combined"

    def test_extreme_medium_minority_returns_oversample(self):
        """Should return 'random_oversample' for extreme imbalance with medium minority."""
        metrics = {"severity": "extreme", "minority_count": 7, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "random_oversample"

    def test_extreme_tiny_minority_returns_class_weight(self):
        """Should return 'class_weight' for extreme imbalance with very few samples."""
        metrics = {"severity": "extreme", "minority_count": 3, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, "LogisticRegression")
        assert strategy == "class_weight"

    @pytest.mark.parametrize("model_name", [
        "XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "CausalForest",
    ])
    def test_all_tree_models_moderate(self, model_name):
        """Should return 'class_weight' for all tree models with moderate imbalance."""
        metrics = {"severity": "moderate", "minority_count": 25, "total_samples": 100}
        strategy, _ = _heuristic_strategy(metrics, model_name)
        assert strategy == "class_weight"


# ============================================================================
# Test _get_llm_recommendation
# ============================================================================


@pytest.mark.asyncio
class TestGetLLMRecommendation:
    """Test LLM-based strategy recommendation."""

    async def test_parses_valid_llm_response(self, mock_anthropic_client):
        """Should parse valid STRATEGY/RATIONALE response."""
        mock_module = mock_anthropic_client("STRATEGY: smote\nRATIONALE: good for this case")
        metrics = _calculate_imbalance_metrics(np.array([0] * 90 + [1] * 10))

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            strategy, rationale = await _get_llm_recommendation(
                metrics, "RandomForest", "binary_classification"
            )

        assert strategy == "smote"
        assert "good" in rationale

    async def test_falls_back_on_api_exception(self):
        """Should fall back to heuristic when API call fails."""
        mock_module = MagicMock()
        mock_module.Anthropic.return_value.messages.create.side_effect = Exception(
            "API error"
        )
        metrics = _calculate_imbalance_metrics(np.array([0] * 90 + [1] * 10))

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            strategy, rationale = await _get_llm_recommendation(
                metrics, "RandomForest", "binary_classification"
            )

        assert strategy in VALID_STRATEGIES

    async def test_falls_back_on_invalid_strategy(self, mock_anthropic_client):
        """Should fall back to heuristic when LLM returns invalid strategy."""
        mock_module = mock_anthropic_client(
            "STRATEGY: bogus\nRATIONALE: invalid strategy"
        )
        metrics = _calculate_imbalance_metrics(np.array([0] * 90 + [1] * 10))

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            strategy, rationale = await _get_llm_recommendation(
                metrics, "RandomForest", "binary_classification"
            )

        assert strategy in VALID_STRATEGIES

    async def test_parses_multiline_response(self, mock_anthropic_client):
        """Should parse STRATEGY/RATIONALE from multi-line response."""
        response = (
            "Some preamble text\n"
            "STRATEGY: combined\n"
            "Some explanation\n"
            "RATIONALE: extreme imbalance needs combined approach"
        )
        mock_module = mock_anthropic_client(response)
        metrics = _calculate_imbalance_metrics(np.array([0] * 97 + [1] * 3))

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            strategy, rationale = await _get_llm_recommendation(
                metrics, "LogisticRegression", "binary_classification"
            )

        assert strategy == "combined"
        assert rationale == "extreme imbalance needs combined approach"


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

    async def test_balanced_skips_llm(self, balanced_state):
        """LLM should not be called when data is balanced."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._get_llm_recommendation",
            new_callable=AsyncMock,
        ) as mock_llm:
            await detect_class_imbalance(balanced_state)
            mock_llm.assert_not_called()

    async def test_detects_moderate_imbalance(self, moderate_imbalance_state):
        """Should detect moderate imbalance."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._get_llm_recommendation",
            new_callable=AsyncMock,
            return_value=("class_weight", "Moderate imbalance"),
        ):
            result = await detect_class_imbalance(moderate_imbalance_state)
        assert result["imbalance_detected"] is True
        assert result["imbalance_severity"] == "moderate"

    async def test_detects_severe_imbalance(self, severe_imbalance_state):
        """Should detect severe imbalance."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._get_llm_recommendation",
            new_callable=AsyncMock,
            return_value=("smote", "Severe imbalance"),
        ):
            result = await detect_class_imbalance(severe_imbalance_state)
        assert result["imbalance_severity"] == "severe"

    async def test_detects_extreme_imbalance(self, extreme_imbalance_state):
        """Should detect extreme imbalance."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._get_llm_recommendation",
            new_callable=AsyncMock,
            return_value=("combined", "Extreme imbalance"),
        ):
            result = await detect_class_imbalance(extreme_imbalance_state)
        assert result["imbalance_severity"] == "extreme"

    async def test_returns_all_output_keys(self, severe_imbalance_state):
        """Should return all expected output keys."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance._get_llm_recommendation",
            new_callable=AsyncMock,
            return_value=("smote", "Severe imbalance"),
        ):
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
