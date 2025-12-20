"""LangGraph workflow for feature_analyzer agent.

Hybrid pipeline:
  Node 1 (NO LLM): SHAP Computation
    ↓
  Node 2 (NO LLM): Interaction Detection
    ↓
  Node 3 (WITH LLM): NL Interpretation
"""

from langgraph.graph import StateGraph, END
from .state import FeatureAnalyzerState
from .nodes import (
    compute_shap,
    detect_interactions,
    narrate_importance,
)


def create_feature_analyzer_graph() -> StateGraph:
    """Create feature_analyzer LangGraph workflow.

    Pipeline:
        START
          ↓
        compute_shap (SHAP values computation - NO LLM)
          ↓
        [SHAP computed successfully?]
          ↓ YES
        detect_interactions (Feature interaction detection - NO LLM)
          ↓
        narrate_importance (Natural language interpretation - LLM)
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(FeatureAnalyzerState)

    # Add nodes
    workflow.add_node("compute_shap", compute_shap)
    workflow.add_node("detect_interactions", detect_interactions)
    workflow.add_node("narrate_importance", narrate_importance)

    # Define edges
    workflow.set_entry_point("compute_shap")

    # Conditional edge after SHAP computation
    workflow.add_conditional_edges(
        "compute_shap",
        _should_continue_after_shap,
        {
            "detect_interactions": "detect_interactions",
            "end": END
        }
    )

    # Continue to interpretation after interaction detection
    workflow.add_edge("detect_interactions", "narrate_importance")

    # End after interpretation
    workflow.add_edge("narrate_importance", END)

    return workflow.compile()


def _should_continue_after_shap(state: dict) -> str:
    """Determine if pipeline should continue after SHAP computation.

    Args:
        state: Current state

    Returns:
        "detect_interactions" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    return "detect_interactions"
