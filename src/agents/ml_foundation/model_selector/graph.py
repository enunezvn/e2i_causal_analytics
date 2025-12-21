"""LangGraph workflow for model_selector agent.

This module assembles the model_selector graph with 4 core nodes:
1. filter_algorithms - Filter by problem type, constraints, preferences
2. rank_candidates - Rank by composite score
3. select_primary_candidate - Select top candidate and alternatives
4. generate_rationale - Generate selection explanation
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    filter_algorithms,
    generate_rationale,
    rank_candidates,
    select_primary_candidate,
)
from .state import ModelSelectorState


def create_model_selector_graph() -> StateGraph:
    """Create model_selector LangGraph workflow.

    Pipeline:
        START → filter → rank → select → rationale → END

    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(ModelSelectorState)

    # Add nodes
    workflow.add_node("filter_algorithms", filter_algorithms)
    workflow.add_node("rank_candidates", rank_candidates)
    workflow.add_node("select_primary_candidate", select_primary_candidate)
    workflow.add_node("generate_rationale", generate_rationale)

    # Define edges
    workflow.set_entry_point("filter_algorithms")
    workflow.add_edge("filter_algorithms", "rank_candidates")
    workflow.add_edge("rank_candidates", "select_primary_candidate")
    workflow.add_edge("select_primary_candidate", "generate_rationale")
    workflow.add_edge("generate_rationale", END)

    return workflow.compile()


# TODO: Add conditional edges for error handling
# TODO: Add baseline_comparator node (compare to baseline models)
# TODO: Add historical_analyzer node (fetch historical success rates from Supabase)
# TODO: Add mlflow_registrar node (register selected model in MLflow)
