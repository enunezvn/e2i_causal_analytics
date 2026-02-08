"""LangGraph workflow for model_selector agent.

This module assembles the model_selector graph with 8 nodes:

Core Pipeline:
1. analyze_historical - Get historical performance data
2. filter_algorithms - Filter by problem type, constraints, preferences
3. rank_candidates - Rank by composite score
4. run_benchmarks - Cross-validation benchmarks (if data provided)
5. select_primary_candidate - Select top candidate and alternatives
6. compare_baselines - Compare against baseline models
7. generate_rationale - Generate selection explanation
8. register_mlflow - Register selection in MLflow
"""

from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    analyze_historical_performance,
    compare_with_baselines,
    create_selection_summary,
    filter_algorithms,
    generate_rationale,
    rank_candidates,
    register_selection_in_mlflow,
    run_benchmarks,
    select_primary_candidate,
)
from .state import ModelSelectorState


def create_model_selector_graph() -> CompiledStateGraph:
    """Create model_selector LangGraph workflow.

    Full Pipeline:
        START → analyze_historical → filter → rank → benchmark
              → select → baseline_compare → rationale → mlflow → END

    Conditional branches:
    - Benchmarks are skipped if no sample data is provided
    - MLflow registration is skipped if connection fails

    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(ModelSelectorState)

    # Add all nodes
    workflow.add_node("analyze_historical", analyze_historical_performance)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("filter_algorithms", filter_algorithms)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("rank_candidates", rank_candidates)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("run_benchmarks", run_benchmarks)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("select_primary_candidate", select_primary_candidate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("compare_baselines", compare_with_baselines)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("generate_rationale", generate_rationale)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("register_mlflow", register_selection_in_mlflow)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("create_summary", create_selection_summary)  # type: ignore[type-var,arg-type,call-overload]

    # Define main flow
    workflow.set_entry_point("analyze_historical")
    workflow.add_edge("analyze_historical", "filter_algorithms")
    workflow.add_edge("filter_algorithms", "rank_candidates")
    workflow.add_edge("rank_candidates", "run_benchmarks")
    workflow.add_edge("run_benchmarks", "select_primary_candidate")
    workflow.add_edge("select_primary_candidate", "compare_baselines")
    workflow.add_edge("compare_baselines", "generate_rationale")
    workflow.add_edge("generate_rationale", "register_mlflow")
    workflow.add_edge("register_mlflow", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()


def create_simple_selector_graph() -> CompiledStateGraph:
    """Create simplified model_selector graph (no benchmarks/MLflow).

    Simple Pipeline:
        START → filter → rank → select → rationale → END

    Use this when:
    - Quick selection is needed
    - No sample data available for benchmarking
    - MLflow tracking not required

    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(ModelSelectorState)

    # Add core nodes only
    workflow.add_node("filter_algorithms", filter_algorithms)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("rank_candidates", rank_candidates)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("select_primary_candidate", select_primary_candidate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("generate_rationale", generate_rationale)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges
    workflow.set_entry_point("filter_algorithms")
    workflow.add_edge("filter_algorithms", "rank_candidates")
    workflow.add_edge("rank_candidates", "select_primary_candidate")
    workflow.add_edge("select_primary_candidate", "generate_rationale")
    workflow.add_edge("generate_rationale", END)

    return workflow.compile()


def should_run_benchmarks(state: Dict[str, Any]) -> str:
    """Conditional edge: determine if benchmarks should run.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    has_data = state.get("X_sample") is not None and state.get("y_sample") is not None
    skip_benchmarks = state.get("skip_benchmarks", False)

    if has_data and not skip_benchmarks:
        return "run_benchmarks"
    return "select_primary_candidate"


def should_register_mlflow(state: Dict[str, Any]) -> str:
    """Conditional edge: determine if MLflow registration should run.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    skip_mlflow = state.get("skip_mlflow", False)
    has_error = state.get("error") is not None

    if not skip_mlflow and not has_error:
        return "register_mlflow"
    return "create_summary"


def create_conditional_selector_graph() -> CompiledStateGraph:
    """Create model_selector graph with conditional branches.

    Conditional Pipeline:
        START → analyze_historical → filter → rank
              → [benchmark if data] → select → baseline
              → rationale → [mlflow if enabled] → summary → END

    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(ModelSelectorState)

    # Add all nodes
    workflow.add_node("analyze_historical", analyze_historical_performance)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("filter_algorithms", filter_algorithms)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("rank_candidates", rank_candidates)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("run_benchmarks", run_benchmarks)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("select_primary_candidate", select_primary_candidate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("compare_baselines", compare_with_baselines)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("generate_rationale", generate_rationale)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("register_mlflow", register_selection_in_mlflow)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("create_summary", create_selection_summary)  # type: ignore[type-var,arg-type,call-overload]

    # Define flow with conditional branches
    workflow.set_entry_point("analyze_historical")
    workflow.add_edge("analyze_historical", "filter_algorithms")
    workflow.add_edge("filter_algorithms", "rank_candidates")

    # Conditional: run benchmarks if data available
    workflow.add_conditional_edges(
        "rank_candidates",
        should_run_benchmarks,
        {
            "run_benchmarks": "run_benchmarks",
            "select_primary_candidate": "select_primary_candidate",
        },
    )

    workflow.add_edge("run_benchmarks", "select_primary_candidate")
    workflow.add_edge("select_primary_candidate", "compare_baselines")
    workflow.add_edge("compare_baselines", "generate_rationale")

    # Conditional: register in MLflow if enabled
    workflow.add_conditional_edges(
        "generate_rationale",
        should_register_mlflow,
        {
            "register_mlflow": "register_mlflow",
            "create_summary": "create_summary",
        },
    )

    workflow.add_edge("register_mlflow", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()
