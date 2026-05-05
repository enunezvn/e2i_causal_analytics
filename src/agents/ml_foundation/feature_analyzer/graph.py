"""LangGraph workflow for feature_analyzer agent.

Extended hybrid pipeline with 5 nodes:
  Node 1 (NO LLM): Feature Generation
    ↓
  Node 2 (NO LLM): Feature Selection
    ↓
  Node 3 (NO LLM): SHAP Computation (optional, requires model_uri)
    ↓
  Node 4 (NO LLM): Interaction Detection
    ↓
  Node 5 (WITH LLM): NL Interpretation
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    compute_shap,
    detect_interactions,
    generate_features,
    narrate_importance,
    select_features,
)
from .state import FeatureAnalyzerState


def create_feature_analyzer_graph() -> CompiledStateGraph:
    """Create feature_analyzer LangGraph workflow.

    Full Pipeline:
        START
          ↓
        generate_features (Feature engineering - NO LLM)
          ↓
        [Generation successful?]
          ↓ YES
        select_features (Feature selection - NO LLM)
          ↓
        [Model URI provided?]
          ↓ YES                    ↓ NO
        compute_shap          narrate_importance
          ↓
        detect_interactions
          ↓
        narrate_importance
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(FeatureAnalyzerState)

    # Add all nodes
    workflow.add_node("generate_features", generate_features)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("select_features", select_features)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("compute_shap", compute_shap)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("detect_interactions", detect_interactions)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("narrate_importance", narrate_importance)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges
    workflow.set_entry_point("generate_features")

    # Conditional edge after feature generation
    workflow.add_conditional_edges(
        "generate_features",
        _should_continue_after_generation,
        {"select_features": "select_features", "end": END},
    )

    # Conditional edge after feature selection
    workflow.add_conditional_edges(
        "select_features",
        _should_compute_shap,
        {
            "compute_shap": "compute_shap",
            "narrate_importance": "narrate_importance",
            "end": END,
        },
    )

    # Conditional edge after SHAP computation
    workflow.add_conditional_edges(
        "compute_shap",
        _should_continue_after_shap,
        {"detect_interactions": "detect_interactions", "end": END},
    )

    # Continue to interpretation after interaction detection
    workflow.add_edge("detect_interactions", "narrate_importance")

    # End after interpretation
    workflow.add_edge("narrate_importance", END)

    # GUARD: If you add ``checkpointer=...`` to this ``workflow.compile()``
    # call, ALSO resolve sub-shard D5 of the TypedDict-to-Pydantic migration:
    # ``FeatureAnalyzerState.shap_values: Optional[np.ndarray]`` does NOT
    # round-trip through JSON (no ``@field_serializer``; closed won't-fix
    # 2026-05-05). Adding a checkpointer here would silently fail or bloat
    # checkpoints with multi-MB SHAP arrays. See ``state.py`` shap_values
    # docstring for full rationale.
    return workflow.compile()


def create_feature_engineering_graph() -> CompiledStateGraph:
    """Create feature engineering-only workflow (no SHAP).

    Simplified Pipeline:
        START
          ↓
        generate_features
          ↓
        select_features
          ↓
        END

    Returns:
        Compiled StateGraph for feature engineering only
    """
    workflow = StateGraph(FeatureAnalyzerState)

    # Add nodes
    workflow.add_node("generate_features", generate_features)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("select_features", select_features)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges
    workflow.set_entry_point("generate_features")

    workflow.add_conditional_edges(
        "generate_features",
        _should_continue_after_generation,
        {"select_features": "select_features", "end": END},
    )

    workflow.add_edge("select_features", END)

    # GUARD: If you add ``checkpointer=...`` to this ``workflow.compile()``
    # call, ALSO resolve sub-shard D5 of the TypedDict-to-Pydantic migration:
    # ``FeatureAnalyzerState.shap_values: Optional[np.ndarray]`` does NOT
    # round-trip through JSON (no ``@field_serializer``; closed won't-fix
    # 2026-05-05). Even though this engineering-only graph never sets
    # ``shap_values``, the State class is shared across all three
    # feature_analyzer graphs — a checkpointer here pulls in the constraint.
    return workflow.compile()


def create_shap_analysis_graph() -> CompiledStateGraph:
    """Create SHAP analysis-only workflow (assumes features already selected).

    Simplified Pipeline:
        START
          ↓
        compute_shap
          ↓
        detect_interactions
          ↓
        narrate_importance
          ↓
        END

    Returns:
        Compiled StateGraph for SHAP analysis only
    """
    workflow = StateGraph(FeatureAnalyzerState)

    # Add nodes
    workflow.add_node("compute_shap", compute_shap)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("detect_interactions", detect_interactions)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("narrate_importance", narrate_importance)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges
    workflow.set_entry_point("compute_shap")

    workflow.add_conditional_edges(
        "compute_shap",
        _should_continue_after_shap,
        {"detect_interactions": "detect_interactions", "end": END},
    )

    workflow.add_edge("detect_interactions", "narrate_importance")
    workflow.add_edge("narrate_importance", END)

    # GUARD: If you add ``checkpointer=...`` to this ``workflow.compile()``
    # call, ALSO resolve sub-shard D5 of the TypedDict-to-Pydantic migration:
    # ``FeatureAnalyzerState.shap_values: Optional[np.ndarray]`` does NOT
    # round-trip through JSON (no ``@field_serializer``; closed won't-fix
    # 2026-05-05). This SHAP-only graph WILL populate ``shap_values`` —
    # adding a checkpointer here without a serializer raises
    # ``PydanticSerializationError`` at the first checkpoint write.
    return workflow.compile()


def _should_continue_after_generation(state: dict) -> str:
    """Determine if pipeline should continue after feature generation.

    Args:
        state: Current state

    Returns:
        "select_features" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    return "select_features"


def _should_compute_shap(state: dict) -> str:
    """Determine if pipeline should compute SHAP values.

    SHAP is computed if:
    - No error occurred during selection
    - model_uri is provided in state

    Args:
        state: Current state

    Returns:
        "compute_shap" if model available, "narrate_importance" if not, "end" if error
    """
    if state.get("error"):
        return "end"

    # If model_uri is provided, compute SHAP
    if state.get("model_uri"):
        return "compute_shap"

    # Otherwise, skip to interpretation (feature selection results only)
    return "narrate_importance"


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
