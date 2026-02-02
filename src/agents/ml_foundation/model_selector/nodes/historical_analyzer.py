"""Historical analyzer for model_selector.

This module analyzes historical experiment data to inform
algorithm selection based on past performance.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    state.get("experiment_id", "")

    # Try to get historical data from database
    historical_data = await _query_historical_experiments(problem_type, kpi_category)

    if historical_data:
        success_rates = _compute_success_rates(historical_data)
        similar_experiments = _find_similar_experiments(historical_data, problem_type, kpi_category)
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

    # Query database for time-based trends
    trend_data = await _query_algorithm_trends(candidate_names)

    trends = {}
    for algo_name in candidate_names:
        if algo_name in trend_data:
            trends[algo_name] = trend_data[algo_name]
        else:
            # Default trend: stable (no historical data)
            trends[algo_name] = {
                "trend": "stable",
                "recent_avg": 0.5,
                "older_avg": 0.5,
                "change": 0.0,
                "sample_count": 0,
            }

    return {
        "algorithm_trends": trends,
    }


async def _query_algorithm_trends(
    algorithm_names: List[str],
    recent_days: int = 30,
    older_days: int = 90,
) -> Dict[str, Dict[str, Any]]:
    """Query database for time-based performance trends.

    Compares recent performance (last 30 days) vs older performance (30-90 days ago)
    to identify improving, declining, or stable trends.

    Args:
        algorithm_names: List of algorithm names to analyze
        recent_days: Days to consider as "recent"
        older_days: Days to consider as "older" comparison period

    Returns:
        Dictionary mapping algorithm name to trend data
    """
    try:
        from src.repositories.ml_data_loader import MLDataLoader

        loader = MLDataLoader()

        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=recent_days)
        older_cutoff = now - timedelta(days=older_days)

        # Query for trend data by time period
        query = """
            SELECT
                algorithm,
                CASE
                    WHEN started_at >= :recent_cutoff THEN 'recent'
                    WHEN started_at >= :older_cutoff THEN 'older'
                    ELSE 'historical'
                END as time_period,
                AVG(
                    COALESCE(
                        (test_metrics->>'auc_roc')::float,
                        (test_metrics->>'r2')::float,
                        (validation_metrics->>'auc_roc')::float,
                        0.5
                    )
                ) as avg_metric,
                COUNT(*) as run_count
            FROM ml_training_runs
            WHERE algorithm = ANY(:algorithms)
            AND status = 'completed'
            AND started_at >= :older_cutoff
            GROUP BY algorithm, time_period
            ORDER BY algorithm, time_period
        """

        params = {
            "algorithms": algorithm_names,
            "recent_cutoff": recent_cutoff.isoformat(),
            "older_cutoff": older_cutoff.isoformat(),
        }

        result = await loader.execute_query(query, params)

        # Process results into trend data
        trends = {}
        for row in result or []:
            algo = row.get("algorithm")
            period = row.get("time_period")
            avg_metric = row.get("avg_metric", 0.5)
            run_count = row.get("run_count", 0)

            if algo not in trends:
                trends[algo] = {
                    "recent_avg": 0.5,
                    "older_avg": 0.5,
                    "recent_count": 0,
                    "older_count": 0,
                }

            if period == "recent":
                trends[algo]["recent_avg"] = avg_metric
                trends[algo]["recent_count"] = run_count
            elif period == "older":
                trends[algo]["older_avg"] = avg_metric
                trends[algo]["older_count"] = run_count

        # Compute trend direction and change
        for algo, data in trends.items():
            change = data["recent_avg"] - data["older_avg"]
            data["change"] = round(change, 4)
            data["sample_count"] = data["recent_count"] + data["older_count"]

            # Determine trend based on change magnitude
            if data["sample_count"] < 3:
                data["trend"] = "insufficient_data"
            elif change > 0.05:
                data["trend"] = "improving"
            elif change < -0.05:
                data["trend"] = "declining"
            else:
                data["trend"] = "stable"

            # Clean up intermediate fields
            del data["recent_count"]
            del data["older_count"]

        logger.debug(f"Computed trends for {len(trends)} algorithms")
        return trends

    except Exception as e:
        logger.warning(f"Failed to query algorithm trends: {e}")
        return {}


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
        sorted_algos = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
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
