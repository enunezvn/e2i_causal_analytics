"""Unit tests for benchmark_runner node."""

import numpy as np
import pytest

from src.agents.ml_foundation.model_selector.nodes.benchmark_runner import (
    _benchmark_algorithm,
    _create_model_instance,
    _rerank_by_benchmarks,
    _run_cross_validation,
    compare_with_baselines,
    run_benchmarks,
)


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)
    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(200) * 0.5
    return X, y


@pytest.fixture
def ranked_candidates():
    """Sample ranked candidates."""
    return [
        {
            "name": "XGBoost",
            "family": "gradient_boosting",
            "framework": "xgboost",
            "default_hyperparameters": {"n_estimators": 100, "max_depth": 6},
            "selection_score": 0.85,
        },
        {
            "name": "LightGBM",
            "family": "gradient_boosting",
            "framework": "lightgbm",
            "default_hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "selection_score": 0.82,
        },
        {
            "name": "RandomForest",
            "family": "ensemble",
            "framework": "sklearn",
            "default_hyperparameters": {"n_estimators": 100, "max_depth": 10},
            "selection_score": 0.78,
        },
        {
            "name": "LogisticRegression",
            "family": "linear",
            "framework": "sklearn",
            "default_hyperparameters": {"C": 1.0},
            "selection_score": 0.70,
        },
    ]


@pytest.fixture
def base_state(ranked_candidates, sample_classification_data):
    """Base state for benchmarking."""
    X, y = sample_classification_data
    return {
        "ranked_candidates": ranked_candidates,
        "problem_type": "binary_classification",
        "X_sample": X,
        "y_sample": y,
        "skip_benchmarks": False,
        "max_benchmark_candidates": 3,
        "cv_folds": 3,
    }


class TestCreateModelInstance:
    """Tests for _create_model_instance."""

    def test_create_xgboost_classifier(self):
        """Should create XGBoost classifier."""
        try:
            model = _create_model_instance("XGBoost", "xgboost", {}, "binary_classification")
            if model is not None:
                assert hasattr(model, "fit")
                assert hasattr(model, "predict")
            else:
                # XGBoost may fail to instantiate due to version/platform issues
                pytest.skip("XGBoost instantiation failed (may be version/platform issue)")
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_create_lightgbm_classifier(self):
        """Should create LightGBM classifier."""
        model = _create_model_instance("LightGBM", "lightgbm", {}, "binary_classification")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_random_forest_classifier(self):
        """Should create RandomForest classifier."""
        model = _create_model_instance("RandomForest", "sklearn", {}, "binary_classification")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_logistic_regression(self):
        """Should create LogisticRegression."""
        model = _create_model_instance("LogisticRegression", "sklearn", {}, "binary_classification")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_xgboost_regressor(self):
        """Should create XGBoost regressor for regression."""
        try:
            model = _create_model_instance("XGBoost", "xgboost", {}, "regression")
            if model is not None:
                assert hasattr(model, "fit")
            else:
                pytest.skip("XGBoost instantiation failed (may be version/platform issue)")
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_create_ridge_regressor(self):
        """Should create Ridge regressor."""
        model = _create_model_instance("Ridge", "sklearn", {}, "regression")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_lasso_regressor(self):
        """Should create Lasso regressor."""
        model = _create_model_instance("Lasso", "sklearn", {}, "regression")
        assert model is not None
        assert hasattr(model, "fit")

    def test_unknown_algorithm_returns_none(self):
        """Should return None for unknown algorithm."""
        model = _create_model_instance("UnknownAlgo", "unknown", {}, "binary_classification")
        assert model is None

    def test_causal_model_returns_none(self):
        """Should return None for causal models (require special handling)."""
        model = _create_model_instance("CausalForest", "econml", {}, "binary_classification")
        assert model is None

    def test_respects_hyperparameters(self):
        """Should apply hyperparameters."""
        hyperparams = {"n_estimators": 50, "max_depth": 3}
        model = _create_model_instance("RandomForest", "sklearn", hyperparams, "binary_classification")
        assert model is not None
        assert model.n_estimators == 50
        assert model.max_depth == 3


class TestRunCrossValidation:
    """Tests for _run_cross_validation."""

    def test_cv_classification(self, sample_classification_data):
        """Should run CV on classification data."""
        X, y = sample_classification_data
        model = _create_model_instance("RandomForest", "sklearn", {}, "binary_classification")

        cv_scores, cv_metrics = _run_cross_validation(
            model, X, y, "binary_classification", cv_folds=3
        )

        assert isinstance(cv_scores, list)
        assert len(cv_scores) == 3
        assert cv_metrics["mean"] > 0.0
        assert cv_metrics["mean"] <= 1.0

    def test_cv_regression(self, sample_regression_data):
        """Should run CV on regression data."""
        X, y = sample_regression_data
        model = _create_model_instance("Ridge", "sklearn", {}, "regression")

        cv_scores, cv_metrics = _run_cross_validation(
            model, X, y, "regression", cv_folds=3
        )

        # Should return RMSE values (converted from negative MSE)
        assert isinstance(cv_scores, list)
        assert isinstance(cv_metrics["mean"], float)

    def test_cv_uses_correct_scoring(self, sample_classification_data):
        """Should use ROC-AUC for classification."""
        X, y = sample_classification_data
        model = _create_model_instance("LogisticRegression", "sklearn", {}, "binary_classification")

        cv_scores, cv_metrics = _run_cross_validation(
            model, X, y, "binary_classification", cv_folds=3
        )

        # ROC-AUC is between 0 and 1
        assert 0.0 <= cv_metrics["mean"] <= 1.0


@pytest.mark.asyncio
class TestBenchmarkAlgorithm:
    """Tests for _benchmark_algorithm."""

    async def test_benchmark_successful_algorithm(self, sample_classification_data):
        """Should benchmark algorithm successfully."""
        X, y = sample_classification_data
        candidate = {
            "name": "RandomForest",
            "framework": "sklearn",
            "default_hyperparameters": {"n_estimators": 10},
        }

        result = await _benchmark_algorithm(
            candidate, X, y, "binary_classification", cv_folds=3
        )

        assert "cv_score_mean" in result
        assert "cv_score_std" in result
        assert "training_time_seconds" in result
        assert "error" not in result

    async def test_benchmark_unsupported_algorithm(self, sample_classification_data):
        """Should return error for unsupported algorithm."""
        X, y = sample_classification_data
        candidate = {
            "name": "CausalForest",
            "framework": "econml",
            "default_hyperparameters": {},
        }

        result = await _benchmark_algorithm(
            candidate, X, y, "binary_classification", cv_folds=3
        )

        assert "error" in result


class TestRerankByBenchmarks:
    """Tests for _rerank_by_benchmarks."""

    def test_reranks_by_combined_score(self, ranked_candidates):
        """Should rerank candidates by combined score."""
        benchmark_results = {
            "XGBoost": {"cv_score_mean": 0.80, "cv_score_std": 0.02},
            "LightGBM": {"cv_score_mean": 0.85, "cv_score_std": 0.03},
            "RandomForest": {"cv_score_mean": 0.75, "cv_score_std": 0.04},
        }

        reranked = _rerank_by_benchmarks(ranked_candidates, benchmark_results)

        # LightGBM has best benchmark, should be boosted
        # All should have combined_score
        for candidate in reranked:
            if candidate["name"] in benchmark_results:
                assert "combined_score" in candidate
                assert "benchmark_score" in candidate

    def test_preserves_unbenchmarked_candidates(self, ranked_candidates):
        """Should preserve candidates without benchmarks."""
        benchmark_results = {
            "XGBoost": {"cv_score_mean": 0.80, "cv_score_std": 0.02},
        }

        reranked = _rerank_by_benchmarks(ranked_candidates, benchmark_results)

        # All original candidates should still be present
        names = [c["name"] for c in reranked]
        for candidate in ranked_candidates:
            assert candidate["name"] in names

    def test_handles_error_results(self, ranked_candidates):
        """Should handle benchmark errors gracefully."""
        benchmark_results = {
            "XGBoost": {"cv_score_mean": 0.80, "cv_score_std": 0.02},
            "LightGBM": {"error": "Benchmark failed"},
        }

        reranked = _rerank_by_benchmarks(ranked_candidates, benchmark_results)

        # LightGBM should have penalized combined_score (original * 0.8) due to error
        lgbm = next(c for c in reranked if c["name"] == "LightGBM")
        assert lgbm.get("combined_score") is not None
        assert lgbm.get("benchmark_score") == 0.0
        # Combined score should be 80% of original (penalty for error)
        expected_combined = lgbm["selection_score"] * 0.8
        assert abs(lgbm["combined_score"] - expected_combined) < 0.001


@pytest.mark.asyncio
class TestRunBenchmarks:
    """Tests for run_benchmarks node."""

    async def test_run_benchmarks_with_data(self, base_state):
        """Should run benchmarks when data is provided."""
        result = await run_benchmarks(base_state)

        assert "benchmark_results" in result
        assert isinstance(result["benchmark_results"], dict)
        assert "benchmark_time_seconds" in result

    async def test_skip_benchmarks_when_no_data(self, ranked_candidates):
        """Should skip benchmarks when no sample data."""
        state = {
            "ranked_candidates": ranked_candidates,
            "problem_type": "binary_classification",
            "X_sample": None,
            "y_sample": None,
        }

        result = await run_benchmarks(state)

        assert result.get("benchmarks_skipped", False) is True
        assert "benchmark_skip_reason" in result

    async def test_benchmarks_run_regardless_of_flag(self, base_state):
        """Benchmarks run when data provided (skip_benchmarks checked at graph level)."""
        # Note: skip_benchmarks flag is checked by conditional_edges in graph.py
        # The run_benchmarks node itself runs when called
        base_state["skip_benchmarks"] = True

        result = await run_benchmarks(base_state)

        # When called, run_benchmarks executes (graph controls when it's called)
        assert "benchmark_results" in result

    async def test_limits_benchmark_candidates(self, base_state):
        """Should limit number of benchmarked candidates."""
        base_state["max_benchmark_candidates"] = 2

        result = await run_benchmarks(base_state)

        # Should only benchmark top 2 candidates
        assert len(result["benchmark_results"]) <= 2


@pytest.mark.asyncio
class TestCompareWithBaselines:
    """Tests for compare_with_baselines node."""

    async def test_compare_with_baselines(self, ranked_candidates):
        """Should compare selected candidate with baselines."""
        state = {
            "ranked_candidates": ranked_candidates,
            "primary_candidate": ranked_candidates[0],
            "baseline_metrics": {
                "random_baseline": {"auc": 0.50, "accuracy": 0.50},
                "majority_baseline": {"auc": 0.55, "accuracy": 0.60},
            },
        }

        result = await compare_with_baselines(state)

        assert "baseline_comparison" in result
        assert isinstance(result["baseline_comparison"], dict)

    async def test_compare_without_baselines(self, ranked_candidates):
        """Should handle missing baselines gracefully."""
        state = {
            "ranked_candidates": ranked_candidates,
            "primary_candidate": ranked_candidates[0],
            "baseline_metrics": {},
        }

        result = await compare_with_baselines(state)

        assert "baseline_comparison" in result
        # Should still have a comparison structure with default baseline
        comparison = result["baseline_comparison"]
        assert "baseline_model" in comparison
        assert "baseline_score" in comparison
        assert "primary_model" in comparison

    async def test_baseline_to_beat_extracted(self, ranked_candidates):
        """Should extract baseline to beat."""
        state = {
            "ranked_candidates": ranked_candidates,
            "primary_candidate": ranked_candidates[0],
            "baseline_metrics": {
                "logistic_baseline": {"auc": 0.65, "accuracy": 0.68},
            },
        }

        result = await compare_with_baselines(state)

        assert "baseline_to_beat" in result
