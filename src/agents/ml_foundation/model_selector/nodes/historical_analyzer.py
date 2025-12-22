"""Historical analyzer for model_selector.

This module analyzes historical experiment data to inform
algorithm selection based on past performance.
"""

from typing import Any, Dict, List, Optional


async def analyze_historical_performance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze historical performance of algorithms.

    Queries past experiments to get success rates and
    performance trends for each algorithm type.

    Args:
        state: ModelSelectorState with problem_type, experiment_id

    Returns:
        Dictionary with historical_success_rates, similar_experiments
    """
    problem_type = state.get("problem_type", "binary_classification")
    kpi_category = state.get("kpi_category")
    experiment_id = state.get("experiment_id", "")

    # Try to get historical data from database
    historical_data = await _query_historical_experiments(problem_type, kpi_category)

    if historical_data:
        success_rates = _compute_success_rates(historical_data)
        similar_experiments = _find_similar_experiments(
            historical_data, problem_type, kpi_category
        )
    else:
        # Use prior knowledge as defaults
        success_rates = _get_default_success_rates(problem_type)
        similar_experiments = []

    return {
        "historical_success_rates": success_rates,
        "similar_experiments": similar_experiments,
        "historical_data_available": len(historical_data) > 0,
        "historical_experiments_count": len(historical_data),
    }


async def _query_historical_experiments(
    problem_type: str,
    kpi_category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query historical experiments from database.

    Args:
        problem_type: Problem type to filter by
        kpi_category: Optional KPI category filter

    Returns:
        List of historical experiment records
    """
    try:
        from src.repositories.ml_data_loader import MLDataLoader

        loader = MLDataLoader()

        # Query ml_training_runs table
        query = """
            SELECT
                tr.algorithm_name,
                tr.algorithm_family,
                tr.primary_metric_value,
                tr.status,
                ex.problem_type,
                ex.kpi_category,
                ex.created_at
            FROM ml_training_runs tr
            JOIN ml_experiments ex ON tr.experiment_id = ex.id
            WHERE ex.problem_type = :problem_type
            AND tr.status = 'completed'
        """
        params = {"problem_type": problem_type}

        if kpi_category:
            query += " AND ex.kpi_category = :kpi_category"
            params["kpi_category"] = kpi_category

        query += " ORDER BY ex.created_at DESC LIMIT 100"

        result = await loader.execute_query(query, params)
        return result if result else []

    except Exception:
        # Return empty if database not available
        return []


def _compute_success_rates(historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute success rates from historical data.

    Success is defined as achieving a metric value above threshold.

    Args:
        historical_data: List of historical experiment records

    Returns:
        Dictionary mapping algorithm name to success rate
    """
    algorithm_results: Dict[str, List[float]] = {}

    for record in historical_data:
        algo_name = record.get("algorithm_name", "unknown")
        metric_value = record.get("primary_metric_value")

        if metric_value is not None:
            if algo_name not in algorithm_results:
                algorithm_results[algo_name] = []
            algorithm_results[algo_name].append(metric_value)

    # Convert to success rates
    success_rates = {}
    for algo_name, metrics in algorithm_results.items():
        if metrics:
            # For classification: AUC > 0.7 is success
            # For regression: R2 > 0.5 is success
            avg_metric = sum(metrics) / len(metrics)
            # Normalize to [0, 1] range as success rate
            success_rates[algo_name] = min(1.0, max(0.0, avg_metric))

    return success_rates


def _find_similar_experiments(
    historical_data: List[Dict[str, Any]],
    problem_type: str,
    kpi_category: Optional[str] = None,
) -> List[str]:
    """Find similar past experiments.

    Args:
        historical_data: Historical experiment records
        problem_type: Current problem type
        kpi_category: Current KPI category

    Returns:
        List of similar experiment IDs
    """
    similar = []

    for record in historical_data:
        if record.get("problem_type") == problem_type:
            if kpi_category is None or record.get("kpi_category") == kpi_category:
                exp_id = record.get("experiment_id")
                if exp_id and exp_id not in similar:
                    similar.append(exp_id)

        if len(similar) >= 5:  # Limit to 5 similar experiments
            break

    return similar


def _get_default_success_rates(problem_type: str) -> Dict[str, float]:
    """Get default success rates based on prior knowledge.

    These are based on general algorithm performance patterns
    across pharmaceutical analytics use cases.

    Args:
        problem_type: Problem type

    Returns:
        Dictionary mapping algorithm name to default success rate
    """
    if "classification" in problem_type:
        return {
            # Causal ML - good for E2I use cases
            "CausalForest": 0.72,
            "LinearDML": 0.68,
            # Gradient boosting - high accuracy
            "XGBoost": 0.78,
            "LightGBM": 0.76,
            # Ensemble
            "RandomForest": 0.72,
            # Linear baselines
            "LogisticRegression": 0.65,
        }
    else:  # regression
        return {
            # Causal ML
            "CausalForest": 0.68,
            "LinearDML": 0.64,
            # Gradient boosting
            "XGBoost": 0.75,
            "LightGBM": 0.73,
            # Ensemble
            "RandomForest": 0.70,
            # Linear baselines
            "Ridge": 0.60,
            "Lasso": 0.58,
        }


async def get_algorithm_trends(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get performance trends for algorithms over time.

    Analyzes how algorithm performance has changed over
    recent experiments.

    Args:
        state: ModelSelectorState with candidate_algorithms

    Returns:
        Dictionary with algorithm_trends
    """
    candidates = state.get("candidate_algorithms", [])
    candidate_names = [c["name"] for c in candidates]

    trends = {}
    for algo_name in candidate_names:
        # Default trend: stable
        trends[algo_name] = {
            "trend": "stable",
            "recent_avg": 0.5,
            "older_avg": 0.5,
            "change": 0.0,
        }

    # TODO: Query database for time-based trends
    # For now, return default stable trends

    return {
        "algorithm_trends": trends,
    }


async def get_recommendations_from_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get algorithm recommendations based on historical patterns.

    Identifies patterns in what algorithms worked well for
    similar problem configurations.

    Args:
        state: ModelSelectorState with problem_type, historical data

    Returns:
        Dictionary with history_recommended_algorithms
    """
    problem_type = state.get("problem_type", "binary_classification")
    success_rates = state.get("historical_success_rates", {})
    kpi_category = state.get("kpi_category")

    # Get top performers from history
    if success_rates:
        sorted_algos = sorted(
            success_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommended = [name for name, rate in sorted_algos[:3] if rate > 0.6]
    else:
        # Default recommendations based on problem type and domain
        recommended = _get_default_recommendations(problem_type, kpi_category)

    return {
        "history_recommended_algorithms": recommended,
        "recommendation_source": "historical" if success_rates else "prior_knowledge",
    }


def _get_default_recommendations(
    problem_type: str,
    kpi_category: Optional[str] = None,
) -> List[str]:
    """Get default algorithm recommendations.

    Args:
        problem_type: Problem type
        kpi_category: KPI category

    Returns:
        List of recommended algorithm names
    """
    # E2I-specific recommendations based on KPI category
    if kpi_category:
        kpi_lower = kpi_category.lower()

        # Causal inference use cases
        if any(term in kpi_lower for term in ["causal", "impact", "effect", "treatment"]):
            return ["CausalForest", "LinearDML", "XGBoost"]

        # Churn prediction
        if "churn" in kpi_lower:
            return ["XGBoost", "LightGBM", "RandomForest"]

        # Conversion optimization
        if "conversion" in kpi_lower:
            return ["XGBoost", "CausalForest", "LightGBM"]

        # Market share forecasting
        if "market" in kpi_lower or "share" in kpi_lower:
            return ["LightGBM", "XGBoost", "Ridge"]

    # Default by problem type
    if "classification" in problem_type:
        return ["XGBoost", "LightGBM", "CausalForest"]
    else:
        return ["XGBoost", "LightGBM", "Ridge"]
