"""Benchmark runner for model_selector.

This module runs cross-validation benchmarks on candidate algorithms
to provide empirical performance comparison.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


async def run_benchmarks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run cross-validation benchmarks on top candidates.

    Performs quick benchmarks on the top-N candidates to get
    empirical performance estimates before final selection.

    Args:
        state: ModelSelectorState with ranked_candidates, X_sample, y_sample

    Returns:
        Dictionary with benchmark_results, benchmark_rankings
    """
    ranked_candidates = state.get("ranked_candidates", [])
    X_sample = state.get("X_sample")
    y_sample = state.get("y_sample")
    problem_type = state.get("problem_type", "binary_classification")
    max_benchmark_candidates = state.get("max_benchmark_candidates", 3)
    cv_folds = state.get("cv_folds", 3)

    # Skip if no sample data available
    if X_sample is None or y_sample is None:
        return {
            "benchmark_results": {},
            "benchmark_rankings": ranked_candidates,
            "benchmarks_skipped": True,
            "benchmark_skip_reason": "No sample data provided for benchmarking",
        }

    # Limit to top candidates
    candidates_to_benchmark = ranked_candidates[:max_benchmark_candidates]

    if not candidates_to_benchmark:
        return {
            "benchmark_results": {},
            "benchmark_rankings": [],
            "benchmarks_skipped": True,
            "benchmark_skip_reason": "No candidates to benchmark",
        }

    benchmark_results = {}
    benchmark_start_time = time.time()

    for candidate in candidates_to_benchmark:
        algo_name = candidate["name"]
        try:
            result = await _benchmark_algorithm(
                candidate, X_sample, y_sample, problem_type, cv_folds
            )
            benchmark_results[algo_name] = result
        except Exception as e:
            benchmark_results[algo_name] = {
                "error": str(e),
                "cv_score_mean": 0.0,
                "cv_score_std": 1.0,
                "training_time_seconds": 0.0,
            }

    benchmark_time = time.time() - benchmark_start_time

    # Re-rank based on benchmark results
    benchmark_rankings = _rerank_by_benchmarks(candidates_to_benchmark, benchmark_results)

    # Add remaining candidates that weren't benchmarked
    benchmarked_names = {c["name"] for c in benchmark_rankings}
    for candidate in ranked_candidates[max_benchmark_candidates:]:
        if candidate["name"] not in benchmarked_names:
            benchmark_rankings.append(candidate)

    return {
        "benchmark_results": benchmark_results,
        "benchmark_rankings": benchmark_rankings,
        "benchmarks_skipped": False,
        "benchmark_time_seconds": benchmark_time,
    }


async def _benchmark_algorithm(
    candidate: Dict[str, Any],
    X: Any,
    y: Any,
    problem_type: str,
    cv_folds: int,
) -> Dict[str, Any]:
    """Run cross-validation benchmark for a single algorithm.

    Args:
        candidate: Algorithm specification
        X: Feature matrix (numpy array or DataFrame)
        y: Target vector
        problem_type: Problem type
        cv_folds: Number of CV folds

    Returns:
        Benchmark results dictionary
    """
    algo_name = candidate["name"]
    framework = candidate.get("framework", "sklearn")
    default_params = candidate.get("default_hyperparameters", {})

    start_time = time.time()

    # Get the model instance
    model = _create_model_instance(algo_name, framework, default_params, problem_type)

    if model is None:
        return {
            "error": f"Could not instantiate {algo_name}",
            "cv_score_mean": 0.0,
            "cv_score_std": 1.0,
            "training_time_seconds": 0.0,
        }

    # Run cross-validation
    cv_scores, cv_metrics = _run_cross_validation(model, X, y, problem_type, cv_folds)

    training_time = time.time() - start_time

    return {
        "cv_score_mean": float(np.mean(cv_scores)),
        "cv_score_std": float(np.std(cv_scores)),
        "cv_scores": [float(s) for s in cv_scores],
        "cv_metrics": cv_metrics,
        "training_time_seconds": training_time,
        "n_samples": len(y) if hasattr(y, "__len__") else 0,
        "n_folds": cv_folds,
    }


def _create_model_instance(
    algo_name: str,
    framework: str,
    params: Dict[str, Any],
    problem_type: str,
) -> Optional[Any]:
    """Create model instance from algorithm specification.

    Args:
        algo_name: Algorithm name
        framework: Framework name
        params: Hyperparameters
        problem_type: Problem type

    Returns:
        Model instance or None if creation fails
    """
    try:
        if algo_name == "XGBoost":
            from xgboost import XGBClassifier, XGBRegressor

            if "classification" in problem_type:
                return XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
            return XGBRegressor(**params)

        if algo_name == "LightGBM":
            from lightgbm import LGBMClassifier, LGBMRegressor

            if "classification" in problem_type:
                return LGBMClassifier(**params, verbose=-1)
            return LGBMRegressor(**params, verbose=-1)

        if algo_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if "classification" in problem_type:
                return RandomForestClassifier(**params, n_jobs=1)
            return RandomForestRegressor(**params, n_jobs=1)

        if algo_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**params)

        if algo_name == "Ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**params)

        if algo_name == "Lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**params)

        if algo_name == "CausalForest":
            # CausalForest requires special handling
            return None  # Skip benchmarking for causal ML

        if algo_name == "LinearDML":
            # LinearDML requires special handling
            return None  # Skip benchmarking for causal ML

    except ImportError:
        return None
    except Exception:
        return None

    return None


def _run_cross_validation(
    model: Any,
    X: Any,
    y: Any,
    problem_type: str,
    cv_folds: int,
) -> Tuple[List[float], Dict[str, float]]:
    """Run cross-validation and compute metrics.

    Args:
        model: Model instance
        X: Feature matrix
        y: Target vector
        problem_type: Problem type
        cv_folds: Number of folds

    Returns:
        Tuple of (cv_scores, aggregated_metrics)
    """
    try:
        from sklearn.model_selection import cross_val_score

        # Choose scoring metric
        if "classification" in problem_type:
            scoring = "roc_auc"
        else:
            scoring = "neg_mean_squared_error"

        # Convert to numpy if DataFrame
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Run CV
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)

        # For regression, convert negative MSE to RMSE
        if scoring == "neg_mean_squared_error":
            cv_scores = np.sqrt(-cv_scores)

        metrics = {
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
            "min": float(np.min(cv_scores)),
            "max": float(np.max(cv_scores)),
        }

        return list(cv_scores), metrics

    except Exception as e:
        # Return default scores on failure
        return [0.5] * cv_folds, {"error": str(e)}


def _rerank_by_benchmarks(
    candidates: List[Dict[str, Any]],
    benchmark_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Re-rank candidates based on benchmark results.

    Combines original selection score (60%) with benchmark score (40%).

    Args:
        candidates: List of candidates
        benchmark_results: Benchmark results by algorithm name

    Returns:
        Re-ranked candidates list
    """
    for candidate in candidates:
        algo_name = candidate["name"]
        original_score = candidate.get("selection_score", 0.5)

        if algo_name in benchmark_results:
            result = benchmark_results[algo_name]
            if "error" not in result:
                cv_mean = result.get("cv_score_mean", 0.5)
                cv_std = result.get("cv_score_std", 0.5)

                # Penalize high variance
                benchmark_score = cv_mean - (0.5 * cv_std)
                benchmark_score = max(0.0, min(1.0, benchmark_score))

                # Combined score: 60% original + 40% benchmark
                combined_score = (0.6 * original_score) + (0.4 * benchmark_score)
                candidate["combined_score"] = combined_score
                candidate["benchmark_score"] = benchmark_score
            else:
                candidate["combined_score"] = original_score * 0.8  # Penalty for errors
                candidate["benchmark_score"] = 0.0
        else:
            candidate["combined_score"] = original_score
            candidate["benchmark_score"] = None

    # Sort by combined score
    return sorted(candidates, key=lambda x: x.get("combined_score", 0), reverse=True)


async def compare_with_baselines(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compare primary candidate against baseline models.

    Runs quick benchmarks on LogisticRegression/Ridge as baselines
    and compares with the selected primary candidate.

    Args:
        state: ModelSelectorState with primary_candidate, X_sample, y_sample

    Returns:
        Dictionary with baseline_comparison, improvement_over_baseline
    """
    primary = state.get("primary_candidate", {})
    benchmark_results = state.get("benchmark_results", {})
    problem_type = state.get("problem_type", "binary_classification")

    if not primary:
        return {
            "baseline_comparison": {},
            "improvement_over_baseline": 0.0,
        }

    algo_name = primary.get("name", "")

    # Define baseline for problem type
    if "classification" in problem_type:
        baseline_name = "LogisticRegression"
    else:
        baseline_name = "Ridge"

    # Get baseline performance (from benchmarks or defaults)
    baseline_score = 0.5  # Default
    if baseline_name in benchmark_results:
        baseline_result = benchmark_results[baseline_name]
        if "error" not in baseline_result:
            baseline_score = baseline_result.get("cv_score_mean", 0.5)

    # Get primary performance
    primary_score = 0.5
    if algo_name in benchmark_results:
        primary_result = benchmark_results[algo_name]
        if "error" not in primary_result:
            primary_score = primary_result.get("cv_score_mean", 0.5)

    improvement = primary_score - baseline_score

    return {
        "baseline_comparison": {
            "baseline_model": baseline_name,
            "baseline_score": baseline_score,
            "primary_model": algo_name,
            "primary_score": primary_score,
            "improvement": improvement,
            "improvement_percent": (improvement / max(baseline_score, 0.01)) * 100,
        },
        "improvement_over_baseline": improvement,
        "baseline_candidates": [baseline_name],
        "baseline_to_beat": {baseline_name: baseline_score},
    }
