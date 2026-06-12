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
        client = loader.client
        if client is None:
            logger.warning(
                "Historical experiment query unavailable: no Supabase client; model "
                "selection will fall back to default success rates."
            )
            return []

        # F9: query via the real PostgREST path. The prior code called a
        # non-existent ``MLDataLoader.execute_query`` whose AttributeError was
        # silently swallowed, so the historical query never actually ran. PostgREST
        # cannot express the ml_training_runs JOIN ml_experiments, so we run two
        # simple filtered queries and join in Python.
        from src.repositories.provenance import apply_provenance_filter

        # #894: both tables are is_synthetic-tagged (migration 069) — planted
        # experiments/runs must not skew historical success rates that drive
        # real model selection.
        exp_query = (
            client.table("ml_experiments")
            .select("id,problem_type,kpi_category,created_at")
            .eq("problem_type", problem_type)
        )
        if kpi_category:
            exp_query = exp_query.eq("kpi_category", kpi_category)
        exp_query = apply_provenance_filter(exp_query)
        # The original SQL applied ORDER BY ex.created_at DESC LIMIT 100 to the JOINED
        # rows, so we must NOT cap the per-table fetches at 100 (capping each side
        # independently would drop valid completed runs of slightly-older experiments).
        # Fetch experiments newest-first with a cap of 1000 — that is >= the entire
        # ml_experiments table (621 rows) and far exceeds any single problem_type's
        # subset, so it is functionally unbounded for this dataset while staying bounded
        # for safety. The 100-row cap + ordering are applied to the JOINED result below.
        experiments = exp_query.order("created_at", desc=True).limit(1000).execute().data or []
        if not experiments:
            return []

        exp_by_id = {e["id"]: e for e in experiments if e.get("id") is not None}
        if not exp_by_id:
            return []

        runs_query = (
            client.table("ml_training_runs")
            .select("algorithm_name,algorithm_family,primary_metric_value,status,experiment_id")
            .in_("experiment_id", list(exp_by_id.keys()))
            .eq("status", "completed")
        )
        runs = apply_provenance_filter(runs_query).execute().data or []

        records: List[Dict[str, Any]] = []
        for run in runs:
            exp = exp_by_id.get(run.get("experiment_id"))
            if exp is None:
                continue
            records.append(
                {
                    "algorithm_name": run.get("algorithm_name"),
                    "algorithm_family": run.get("algorithm_family"),
                    "primary_metric_value": run.get("primary_metric_value"),
                    "status": run.get("status"),
                    "experiment_id": run.get("experiment_id"),
                    "problem_type": exp.get("problem_type"),
                    "kpi_category": exp.get("kpi_category"),
                    "created_at": exp.get("created_at"),
                }
            )
        # Replicate ORDER BY ex.created_at DESC LIMIT 100 on the joined result
        # (ISO timestamp strings sort lexicographically == chronologically).
        records.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return records[:100]

    except Exception as e:
        # F9: log instead of silently swallowing — a bare except previously hid that
        # execute_query did not exist, so a broken query looked like "no data".
        logger.warning(
            "Historical experiment query failed (%s); falling back to default success rates.",
            e,
        )
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


def _parse_started_at(value: Any) -> Optional[datetime]:
    """Parse a started_at value (ISO string or datetime) into an aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _extract_trend_metric(row: Dict[str, Any]) -> float:
    """COALESCE the trend metric: test auc_roc -> test r2 -> validation auc_roc -> 0.5.

    Mirrors the prior SQL ``COALESCE((test_metrics->>'auc_roc')::float, ...)``.
    """
    test = row.get("test_metrics") or {}
    val = row.get("validation_metrics") or {}
    for source, key in ((test, "auc_roc"), (test, "r2"), (val, "auc_roc")):
        if isinstance(source, dict) and source.get(key) is not None:
            try:
                return float(source[key])
            except (TypeError, ValueError):
                continue
    return 0.5


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
    if not algorithm_names:
        # PostgREST .in_() with an empty list is undefined/client-dependent — guard it.
        return {}

    try:
        from src.repositories.ml_data_loader import MLDataLoader

        loader = MLDataLoader()
        client = loader.client
        if client is None:
            logger.warning("Algorithm trend query unavailable: no Supabase client.")
            return {}

        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=recent_days)
        older_cutoff = now - timedelta(days=older_days)

        # F9: query via the real PostgREST path (prior code called the non-existent
        # MLDataLoader.execute_query). PostgREST cannot do the CASE-bucketing /
        # JSONB-AVG / GROUP BY, so fetch the completed runs in the window and
        # aggregate per (algorithm, time_period) in Python — producing the same
        # {algorithm, time_period, avg_metric, run_count} shape the SQL returned.
        runs = (
            client.table("ml_training_runs")
            .select("algorithm,started_at,test_metrics,validation_metrics,status")
            .in_("algorithm", algorithm_names)
            .eq("status", "completed")
            .gte("started_at", older_cutoff.isoformat())
            .execute()
            .data
            or []
        )

        metrics_by_group: Dict[tuple, List[float]] = {}
        for run in runs:
            algo = run.get("algorithm")
            started = _parse_started_at(run.get("started_at"))
            if algo is None or started is None:
                continue
            if started >= recent_cutoff:
                period = "recent"
            elif started >= older_cutoff:
                period = "older"
            else:
                period = "historical"
            metrics_by_group.setdefault((algo, period), []).append(_extract_trend_metric(run))

        result = [
            {
                "algorithm": algo,
                "time_period": period,
                "avg_metric": (sum(vals) / len(vals)) if vals else 0.5,
                "run_count": len(vals),
            }
            for (algo, period), vals in metrics_by_group.items()
        ]

        # Process results into trend data
        trends: Dict[str, Dict[str, Any]] = {}
        for row in result or []:
            # Fresh names + narrow to str: the dict ``.get`` returns Any|None, but
            # ``period`` was already str-typed by the grouping loop above and
            # ``algo`` indexes the str-keyed ``trends``. Skip malformed rows rather
            # than key a trend on a missing algorithm/period (in practice these are
            # always present — ``result`` is built from the grouping above).
            row_algo = row.get("algorithm")
            row_period = row.get("time_period")
            if not isinstance(row_algo, str) or not isinstance(row_period, str):
                continue
            avg_metric = row.get("avg_metric", 0.5)
            run_count = row.get("run_count", 0)

            if row_algo not in trends:
                trends[row_algo] = {
                    "recent_avg": 0.5,
                    "older_avg": 0.5,
                    "recent_count": 0,
                    "older_count": 0,
                }

            if row_period == "recent":
                trends[row_algo]["recent_avg"] = avg_metric
                trends[row_algo]["recent_count"] = run_count
            elif row_period == "older":
                trends[row_algo]["older_avg"] = avg_metric
                trends[row_algo]["older_count"] = run_count

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
