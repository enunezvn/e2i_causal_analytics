"""Nodes for model_selector agent.

This module exports all node functions used in the model_selector LangGraph.
"""

from .algorithm_registry import filter_algorithms
from .benchmark_runner import compare_with_baselines, run_benchmarks
from .candidate_ranker import rank_candidates, select_primary_candidate
from .historical_analyzer import (
    analyze_historical_performance,
    get_algorithm_trends,
    get_recommendations_from_history,
)
from .mlflow_registrar import (
    create_selection_summary,
    log_benchmark_comparison,
    register_selection_in_mlflow,
)
from .rationale_generator import generate_rationale

__all__ = [
    # Algorithm filtering
    "filter_algorithms",
    # Benchmarking
    "run_benchmarks",
    "compare_with_baselines",
    # Ranking and selection
    "rank_candidates",
    "select_primary_candidate",
    # Historical analysis
    "analyze_historical_performance",
    "get_algorithm_trends",
    "get_recommendations_from_history",
    # MLflow registration
    "register_selection_in_mlflow",
    "log_benchmark_comparison",
    "create_selection_summary",
    # Rationale generation
    "generate_rationale",
]
