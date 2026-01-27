"""Candidate ranking logic for model_selector.

This module ranks algorithms by composite scoring based on multiple factors.
"""

from typing import Any, Dict, List


async def rank_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rank candidate algorithms by composite score.

    Scoring factors (weights):
    - Historical success rate: 40%
    - Inference speed: 20%
    - Memory efficiency: 15%
    - Interpretability: 15%
    - Causal ML preference (E2I): 10% - ONLY if problem requires causal inference

    Args:
        state: ModelSelectorState with candidate_algorithms, historical_success_rates,
               algorithm_preferences

    Returns:
        Dictionary with ranked_candidates, selection_scores
    """
    candidates = state.get("candidate_algorithms", [])
    success_rates = state.get("historical_success_rates", {})
    algorithm_preferences = state.get("algorithm_preferences", [])
    row_count = state.get("row_count", 1000)

    # Determine if causal inference is appropriate for this problem
    scope_spec = state.get("scope_spec", {})
    problem_type = scope_spec.get("problem_type", "classification")
    requires_causal = _requires_causal_inference(problem_type, scope_spec)

    if not candidates:
        return {
            "ranked_candidates": [],
            "selection_scores": {},
            "error": "No candidate algorithms available",
            "error_type": "no_candidates_error",
        }

    selection_scores = {}

    for candidate in candidates:
        algo_name = candidate["name"]
        score = _compute_selection_score(
            candidate, success_rates, algorithm_preferences, row_count, requires_causal
        )
        selection_scores[algo_name] = score
        candidate["selection_score"] = score

    # Sort by selection score descending
    ranked = sorted(candidates, key=lambda x: x["selection_score"], reverse=True)

    return {
        "ranked_candidates": ranked,
        "selection_scores": selection_scores,
        "requires_causal_inference": requires_causal,
    }


def _requires_causal_inference(problem_type: str, scope_spec: Dict[str, Any]) -> bool:
    """Determine if the problem requires causal inference models.

    Causal inference is appropriate when:
    - Problem type explicitly mentions 'causal' or 'treatment_effect'
    - Business objective involves understanding 'why' or 'effect'
    - There's a treatment variable defined
    - The task is about intervention optimization

    Args:
        problem_type: Type of ML problem
        scope_spec: Scope specification from scope_definer

    Returns:
        True if causal models should be preferred
    """
    # Check problem type
    causal_problem_types = [
        "causal_inference",
        "treatment_effect",
        "uplift_modeling",
        "heterogeneous_treatment_effect",
        "intervention_optimization",
    ]

    if problem_type.lower() in causal_problem_types:
        return True

    # Check business objective for causal keywords
    business_objective = scope_spec.get("business_objective", "").lower()
    causal_keywords = ["treatment effect", "causal", "intervention", "uplift", "what-if", "counterfactual"]
    if any(keyword in business_objective for keyword in causal_keywords):
        return True

    # Check if treatment variable is defined
    if scope_spec.get("treatment_column") or scope_spec.get("treatment_variable"):
        return True

    # Standard classification/regression don't need causal models
    standard_types = ["binary_classification", "classification", "regression", "multiclass"]
    if problem_type.lower() in standard_types:
        return False

    return False


def _compute_selection_score(
    candidate: Dict[str, Any],
    success_rates: Dict[str, float],
    preferences: List[str],
    data_size: int,
    requires_causal: bool = False,
) -> float:
    """Compute composite selection score for a candidate.

    Args:
        candidate: Algorithm specification
        success_rates: Historical success rates by algorithm name
        preferences: User algorithm preferences
        data_size: Number of training samples
        requires_causal: Whether the problem requires causal inference

    Returns:
        Float score in range [0, 1]
    """
    score = 0.0
    algo_name = candidate["name"]
    algo_family = candidate.get("family", "")

    # Factor 1: Historical success rate (40%)
    historical_rate = success_rates.get(algo_name, 0.5)  # Default 50% for new algos
    score += historical_rate * 0.4

    # Factor 2: Inference speed (20%)
    # Lower latency is better
    latency_ms = candidate.get("inference_latency_ms", 100)
    latency_score = max(0, 1 - (latency_ms / 200))  # Normalize to [0, 1]
    score += latency_score * 0.2

    # Factor 3: Memory efficiency (15%)
    # Lower memory is better
    memory_gb = candidate.get("memory_gb", 8)
    memory_score = max(0, 1 - (memory_gb / 16))  # Normalize to [0, 1]
    score += memory_score * 0.15

    # Factor 4: Interpretability (15%)
    interpretability = candidate.get("interpretability_score", 0.5)
    score += interpretability * 0.15

    # Factor 5: Causal ML preference for E2I (10%)
    # ONLY apply bonus if problem actually requires causal inference
    if requires_causal and algo_family == "causal_ml":
        score += 0.10
    elif not requires_causal and algo_family == "causal_ml":
        # PENALTY: Causal models are inappropriate for standard classification
        # They use effect() not predict(), causing issues with standard eval
        score -= 0.15

    # Bonus: User preference (adds up to 0.1)
    if preferences and algo_name in preferences:
        score += 0.10

    # Penalty: Poor scalability with large data (deduct up to 0.1)
    if data_size > 100000:  # > 100k samples
        scalability = candidate.get("scalability_score", 0.7)
        if scalability < 0.5:
            score -= 0.1

    # Ensure score in [0, 1]
    return max(0.0, min(1.0, score))


async def select_primary_candidate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select primary candidate and alternatives.

    Args:
        state: ModelSelectorState with ranked_candidates

    Returns:
        Dictionary with primary_candidate, alternative_candidates
    """
    ranked = state.get("ranked_candidates", [])

    if not ranked:
        return {
            "error": "No ranked candidates available",
            "error_type": "no_ranked_candidates_error",
        }

    # Select top candidate
    primary = ranked[0]

    # Select top 2-3 alternatives
    alternatives = ranked[1:4] if len(ranked) > 1 else []

    # Extract key fields for primary candidate
    return {
        "primary_candidate": primary,
        "alternative_candidates": alternatives,
        "algorithm_name": primary["name"],
        "algorithm_family": primary["family"],
        "algorithm_class": _get_algorithm_class(primary),
        "default_hyperparameters": primary.get("default_hyperparameters", {}),
        "hyperparameter_search_space": primary.get("hyperparameter_space", {}),
        "estimated_inference_latency_ms": primary.get("inference_latency_ms", 0),
        "memory_requirement_gb": primary.get("memory_gb", 0),
        "interpretability_score": primary.get("interpretability_score", 0.5),
        "scalability_score": primary.get("scalability_score", 0.7),
        "selection_score": primary.get("selection_score", 0.5),
    }


def _get_algorithm_class(candidate: Dict[str, Any]) -> str:
    """Get Python class path for the algorithm.

    Args:
        candidate: Algorithm specification

    Returns:
        Python class path string
    """
    framework = candidate.get("framework", "sklearn")
    algo_name = candidate["name"]

    # Map to Python class paths
    class_map = {
        # Causal ML
        "CausalForest": "econml.dml.CausalForestDML",
        "LinearDML": "econml.dml.LinearDML",
        # Gradient Boosting
        "XGBoost": "xgboost.XGBClassifier",
        "LightGBM": "lightgbm.LGBMClassifier",
        # Random Forest
        "RandomForest": "sklearn.ensemble.RandomForestClassifier",
        # Linear Models
        "LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "Ridge": "sklearn.linear_model.Ridge",
        "Lasso": "sklearn.linear_model.Lasso",
    }

    return class_map.get(algo_name, f"{framework}.{algo_name}")
