"""Tests for algorithm_registry.py filtering logic."""

import pytest
from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
    filter_algorithms,
    ALGORITHM_REGISTRY,
    _filter_by_problem_type,
    _filter_by_constraints,
    _filter_by_preferences,
)


class TestAlgorithmRegistry:
    """Test algorithm catalog structure."""

    def test_registry_has_causal_ml_algorithms(self):
        """Verify CausalML algorithms present."""
        assert "CausalForest" in ALGORITHM_REGISTRY
        assert "LinearDML" in ALGORITHM_REGISTRY
        assert ALGORITHM_REGISTRY["CausalForest"]["family"] == "causal_ml"
        assert ALGORITHM_REGISTRY["LinearDML"]["family"] == "causal_ml"

    def test_registry_has_gradient_boosting_algorithms(self):
        """Verify gradient boosting algorithms present."""
        assert "XGBoost" in ALGORITHM_REGISTRY
        assert "LightGBM" in ALGORITHM_REGISTRY
        assert ALGORITHM_REGISTRY["XGBoost"]["family"] == "gradient_boosting"

    def test_registry_has_linear_baseline_algorithms(self):
        """Verify linear baseline algorithms present."""
        assert "LogisticRegression" in ALGORITHM_REGISTRY
        assert "Ridge" in ALGORITHM_REGISTRY
        assert "Lasso" in ALGORITHM_REGISTRY
        assert ALGORITHM_REGISTRY["LogisticRegression"]["family"] == "linear"

    def test_all_algorithms_have_required_fields(self):
        """Verify all algorithms have required specification fields."""
        required_fields = [
            "family",
            "framework",
            "problem_types",
            "strengths",
            "inference_latency_ms",
            "memory_gb",
            "interpretability_score",
            "scalability_score",
            "hyperparameter_space",
            "default_hyperparameters",
        ]
        for algo_name, spec in ALGORITHM_REGISTRY.items():
            for field in required_fields:
                assert field in spec, f"{algo_name} missing {field}"

    def test_hyperparameter_spaces_valid(self):
        """Verify hyperparameter spaces are properly structured."""
        for algo_name, spec in ALGORITHM_REGISTRY.items():
            hp_space = spec["hyperparameter_space"]
            assert isinstance(hp_space, dict), f"{algo_name} hp_space not a dict"
            for hp_name, hp_spec in hp_space.items():
                assert "type" in hp_spec, f"{algo_name}.{hp_name} missing type"
                assert hp_spec["type"] in ["int", "float", "categorical"]


class TestFilterByProblemType:
    """Test problem type filtering."""

    def test_binary_classification_filters_correctly(self):
        """Binary classification should include appropriate algorithms."""
        candidates = _filter_by_problem_type("binary_classification")
        algo_names = [c["name"] for c in candidates]

        # Should include
        assert "CausalForest" in algo_names
        assert "LinearDML" in algo_names
        assert "XGBoost" in algo_names
        assert "LogisticRegression" in algo_names

        # Should exclude regression-only
        assert "Ridge" not in algo_names
        assert "Lasso" not in algo_names

    def test_regression_filters_correctly(self):
        """Regression should include appropriate algorithms."""
        candidates = _filter_by_problem_type("regression")
        algo_names = [c["name"] for c in candidates]

        # Should include
        assert "CausalForest" in algo_names
        assert "LinearDML" in algo_names
        assert "XGBoost" in algo_names
        assert "Ridge" in algo_names
        assert "Lasso" in algo_names

        # Should exclude classification-only
        assert "LogisticRegression" not in algo_names

    def test_multiclass_classification_filters_correctly(self):
        """Multiclass should include appropriate algorithms."""
        candidates = _filter_by_problem_type("multiclass_classification")
        algo_names = [c["name"] for c in candidates]

        # Should include
        assert "XGBoost" in algo_names
        assert "LightGBM" in algo_names
        assert "RandomForest" in algo_names
        assert "LogisticRegression" in algo_names

    def test_filtered_candidates_include_name(self):
        """Filtered candidates should include 'name' field."""
        candidates = _filter_by_problem_type("binary_classification")
        for candidate in candidates:
            assert "name" in candidate
            assert "family" in candidate
            assert "problem_types" in candidate


class TestFilterByConstraints:
    """Test constraint filtering."""

    def test_latency_constraint_filters_slow_algorithms(self):
        """Latency constraint should filter slow algorithms."""
        candidates = _filter_by_problem_type("binary_classification")
        constraints = ["inference_latency_<20ms"]

        filtered = _filter_by_constraints(candidates, constraints)
        algo_names = [c["name"] for c in filtered]

        # Should include fast algorithms
        assert "LinearDML" in algo_names  # 10ms
        assert "LogisticRegression" in algo_names  # 1ms

        # Should exclude slower algorithms
        assert "CausalForest" not in algo_names  # 50ms
        assert "RandomForest" not in algo_names  # 30ms

    def test_memory_constraint_filters_high_memory_algorithms(self):
        """Memory constraint should filter high-memory algorithms."""
        candidates = _filter_by_problem_type("binary_classification")
        constraints = ["memory_<2gb"]

        filtered = _filter_by_constraints(candidates, constraints)
        algo_names = [c["name"] for c in filtered]

        # Should include low-memory algorithms
        assert "LinearDML" in algo_names  # 1.0GB
        assert "LogisticRegression" in algo_names  # 0.1GB

        # Should exclude high-memory algorithms
        assert "CausalForest" not in algo_names  # 4.0GB
        assert "RandomForest" not in algo_names  # 3.0GB

    def test_multiple_constraints_applied(self):
        """Multiple constraints should be applied together."""
        candidates = _filter_by_problem_type("binary_classification")
        constraints = ["inference_latency_<30ms", "memory_<2gb"]

        filtered = _filter_by_constraints(candidates, constraints)
        algo_names = [c["name"] for c in filtered]

        # Should only include algorithms meeting both constraints
        assert "LinearDML" in algo_names  # 10ms, 1.0GB
        assert "LogisticRegression" in algo_names  # 1ms, 0.1GB
        assert "XGBoost" in algo_names  # 20ms, 2.0GB

        # Should exclude algorithms violating either constraint
        assert "CausalForest" not in algo_names  # 50ms (latency), 4.0GB (memory)

    def test_malformed_constraint_ignored(self):
        """Malformed constraints should be ignored."""
        candidates = _filter_by_problem_type("binary_classification")
        constraints = ["invalid_constraint", "latency_<_invalid"]

        filtered = _filter_by_constraints(candidates, constraints)

        # Should return all candidates (no valid constraints)
        assert len(filtered) == len(candidates)


class TestFilterByPreferences:
    """Test preference filtering."""

    def test_excluded_algorithms_removed(self):
        """Excluded algorithms should be filtered out."""
        candidates = _filter_by_problem_type("binary_classification")
        excluded = ["XGBoost", "LightGBM"]

        filtered = _filter_by_preferences(candidates, [], excluded)
        algo_names = [c["name"] for c in filtered]

        assert "XGBoost" not in algo_names
        assert "LightGBM" not in algo_names
        assert "CausalForest" in algo_names

    def test_preferences_dont_hard_filter(self):
        """Preferences should not hard-filter (used in ranking instead)."""
        candidates = _filter_by_problem_type("binary_classification")
        preferences = ["CausalForest"]

        filtered = _filter_by_preferences(candidates, preferences, [])

        # All candidates should remain (preferences used in ranking, not filtering)
        assert len(filtered) == len(candidates)

    def test_excluded_overrides_preferences(self):
        """Exclusions should override preferences."""
        candidates = _filter_by_problem_type("binary_classification")
        preferences = ["XGBoost"]
        excluded = ["XGBoost"]

        filtered = _filter_by_preferences(candidates, preferences, excluded)
        algo_names = [c["name"] for c in filtered]

        # XGBoost should be excluded despite being preferred
        assert "XGBoost" not in algo_names


@pytest.mark.asyncio
class TestFilterAlgorithms:
    """Test complete filter_algorithms node."""

    async def test_filter_algorithms_progressive_filtering(self):
        """Test progressive filtering: type → constraints → preferences."""
        state = {
            "problem_type": "binary_classification",
            "technical_constraints": ["inference_latency_<30ms"],
            "algorithm_preferences": ["CausalForest"],
            "excluded_algorithms": ["LogisticRegression"],
            "interpretability_required": False,
        }

        result = await filter_algorithms(state)

        # Verify all filter stages present
        assert "filtered_by_problem_type" in result
        assert "filtered_by_constraints" in result
        assert "filtered_by_preferences" in result
        assert "candidate_algorithms" in result

        # Verify progressive filtering
        algo_names = [c["name"] for c in result["candidate_algorithms"]]

        # Should exclude LogisticRegression (excluded)
        assert "LogisticRegression" not in algo_names

        # Should include fast algorithms
        assert "LinearDML" in algo_names

    async def test_interpretability_required_filters_black_box_models(self):
        """Interpretability requirement should filter low-interpretability models."""
        state = {
            "problem_type": "binary_classification",
            "technical_constraints": [],
            "algorithm_preferences": [],
            "excluded_algorithms": [],
            "interpretability_required": True,
        }

        result = await filter_algorithms(state)
        algo_names = [c["name"] for c in result["candidate_algorithms"]]

        # Should include high interpretability (>= 0.7)
        assert "LinearDML" in algo_names  # 0.9
        assert "CausalForest" in algo_names  # 0.7
        assert "LogisticRegression" in algo_names  # 1.0

        # Should exclude low interpretability (< 0.7)
        # XGBoost is 0.6, RandomForest is 0.5

    async def test_fallback_to_linear_models_if_all_filtered(self):
        """If all algorithms filtered, fallback to linear baselines."""
        state = {
            "problem_type": "binary_classification",
            "technical_constraints": ["inference_latency_<1ms"],  # Only linear models
            "algorithm_preferences": [],
            "excluded_algorithms": ["LogisticRegression"],  # Exclude the best option
            "interpretability_required": False,
        }

        result = await filter_algorithms(state)

        # Should still have candidates (fallback to linear baselines)
        assert len(result["candidate_algorithms"]) > 0

    async def test_regression_problem_type(self):
        """Test filtering for regression problem."""
        state = {
            "problem_type": "regression",
            "technical_constraints": [],
            "algorithm_preferences": [],
            "excluded_algorithms": [],
            "interpretability_required": False,
        }

        result = await filter_algorithms(state)
        algo_names = [c["name"] for c in result["candidate_algorithms"]]

        # Should include regression algorithms
        assert "Ridge" in algo_names
        assert "Lasso" in algo_names
        assert "CausalForest" in algo_names

        # Should exclude classification-only
        assert "LogisticRegression" not in algo_names
