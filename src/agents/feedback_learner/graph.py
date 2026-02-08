"""
E2I Feedback Learner Agent - Graph Assembly
Version: 4.2
Purpose: LangGraph assembly for feedback learning workflow

DSPy Integration:
- Cognitive context injection at entry point
- Training signal collection throughout pipeline
- Memory contribution generation on completion
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import create_workflow_initializer
from src.utils.audit_chain import AgentTier

from .dspy_integration import (
    FeedbackLearnerCognitiveContext,
    FeedbackLearnerTrainingSignal,
)
from .nodes.feedback_collector import FeedbackCollectorNode
from .nodes.knowledge_updater import KnowledgeUpdaterNode
from .nodes.learning_extractor import LearningExtractorNode
from .nodes.pattern_analyzer import PatternAnalyzerNode
from .nodes.rubric_node import RubricNode
from .state import FeedbackLearnerState

logger = logging.getLogger(__name__)


def build_feedback_learner_graph(
    feedback_store: Optional[Any] = None,
    outcome_store: Optional[Any] = None,
    knowledge_stores: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    llm: Optional[Any] = None,
    cognitive_rag: Optional[Any] = None,
    db_client: Optional[Any] = None,
    enable_rubric_evaluation: bool = True,
):
    """
    Build the Feedback Learner agent graph with DSPy integration.

    Architecture (with rubric evaluation enabled):
        [audit_init] → [enrich] → [collect] → [analyze] → [rubric] → [extract] → [update] → [finalize] → END

    Architecture (without rubric evaluation):
        [audit_init] → [enrich] → [collect] → [analyze] → [extract] → [update] → [finalize] → END

    Args:
        feedback_store: Store for user feedback
        outcome_store: Store for outcome data
        knowledge_stores: Dictionary of knowledge stores by type
        use_llm: Whether to use LLM for analysis
        llm: Optional LLM instance
        cognitive_rag: Optional CognitiveRAG instance for context enrichment
        db_client: Optional database client for storing rubric evaluations
        enable_rubric_evaluation: Whether to include rubric evaluation node (default: True)

    Returns:
        Compiled LangGraph workflow
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("feedback_learner", AgentTier.SELF_IMPROVEMENT)

    # Initialize nodes
    collector = FeedbackCollectorNode(feedback_store, outcome_store)
    analyzer = PatternAnalyzerNode(use_llm=use_llm, llm=llm)
    rubric_node = RubricNode(db_client=db_client) if enable_rubric_evaluation else None
    extractor = LearningExtractorNode(use_llm=use_llm, llm=llm)
    updater = KnowledgeUpdaterNode(knowledge_stores)

    # Build graph
    workflow = StateGraph(FeedbackLearnerState)

    # Create cognitive enricher with bound cognitive_rag
    async def enrich_node(state: FeedbackLearnerState) -> FeedbackLearnerState:
        return await _cognitive_context_enricher(state, cognitive_rag)

    # Add nodes
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain
    workflow.add_node("enrich", enrich_node)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("collect", collector.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("analyze", analyzer.execute)  # type: ignore[type-var,arg-type,call-overload]
    if rubric_node:
        workflow.add_node("rubric", rubric_node.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("extract", extractor.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("update", updater.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("finalize", _finalize_training_signal)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("error_handler", _error_handler_node)  # type: ignore[type-var,arg-type,call-overload]

    # Flow - start with audit initialization
    workflow.set_entry_point("audit_init")

    # Audit init proceeds to cognitive enrichment
    workflow.add_edge("audit_init", "enrich")

    # Enrich always proceeds to collect
    workflow.add_edge("enrich", "collect")

    # Conditional edges with error handling
    workflow.add_conditional_edges(
        "collect",
        lambda s: "error" if s.get("status") == "failed" else "analyze",
        {"analyze": "analyze", "error": "error_handler"},
    )

    # Analyze proceeds to rubric (if enabled) or extract
    if rubric_node:
        workflow.add_conditional_edges(
            "analyze",
            lambda s: "error" if s.get("status") == "failed" else "rubric",
            {"rubric": "rubric", "error": "error_handler"},
        )

        workflow.add_conditional_edges(
            "rubric",
            lambda s: "error" if s.get("status") == "failed" else "extract",
            {"extract": "extract", "error": "error_handler"},
        )
    else:
        workflow.add_conditional_edges(
            "analyze",
            lambda s: "error" if s.get("status") == "failed" else "extract",
            {"extract": "extract", "error": "error_handler"},
        )

    workflow.add_conditional_edges(
        "extract",
        lambda s: "error" if s.get("status") == "failed" else "update",
        {"update": "update", "error": "error_handler"},
    )

    # Update proceeds to finalize for training signal collection
    workflow.add_edge("update", "finalize")
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


def build_simple_feedback_learner_graph():
    """
    Build a simple feedback learner graph without external stores.

    Returns:
        Compiled LangGraph workflow
    """
    return build_feedback_learner_graph(
        feedback_store=None,
        outcome_store=None,
        knowledge_stores=None,
        use_llm=False,
        llm=None,
    )


async def _cognitive_context_enricher(
    state: FeedbackLearnerState,
    cognitive_rag: Optional[Any] = None,
) -> FeedbackLearnerState:
    """
    Enrich state with cognitive context from CognitiveRAG.

    This node calls the CognitiveRAG 4-phase cycle to retrieve:
    - Historical patterns from episodic memory
    - Agent baselines from semantic memory
    - Prior learnings and optimization examples

    Args:
        state: Current pipeline state
        cognitive_rag: Optional CognitiveRAG instance

    Returns:
        State enriched with cognitive context
    """
    if cognitive_rag is None:
        logger.debug("No CognitiveRAG provided, skipping cognitive enrichment")
        return {
            **state,
            "cognitive_context": None,  # type: ignore[typeddict-item]
        }

    try:
        # Build query for CognitiveRAG
        query = f"Feedback analysis for agents: {state.get('focus_agents', 'all')} "
        query += f"from {state.get('time_range_start')} to {state.get('time_range_end')}"

        # Execute 4-phase cognitive cycle
        cognitive_result = await cognitive_rag.process(query)

        # Extract relevant context for feedback learning
        cognitive_context: FeedbackLearnerCognitiveContext = {
            "synthesized_summary": cognitive_result.get("summary", ""),
            "historical_patterns": cognitive_result.get("patterns", []),
            "optimization_examples": cognitive_result.get("examples", []),
            "agent_baselines": cognitive_result.get("baselines", {}),
            "prior_learnings": cognitive_result.get("learnings", []),
            "correlation_insights": cognitive_result.get("correlations", []),
            "evidence_confidence": cognitive_result.get("confidence", 0.0),
        }

        logger.info(
            f"Cognitive context enriched: {len(cognitive_context['historical_patterns'])} "
            f"patterns, confidence={cognitive_context['evidence_confidence']:.2f}"
        )

        return {
            **state,
            "cognitive_context": cognitive_context,
        }

    except Exception as e:
        logger.warning(f"Cognitive enrichment failed: {e}, continuing without context")
        return {
            **state,
            "cognitive_context": None,  # type: ignore[typeddict-item]
            "warnings": (state.get("warnings") or []) + [f"Cognitive enrichment skipped: {str(e)}"],
        }


async def _finalize_training_signal(state: FeedbackLearnerState) -> FeedbackLearnerState:
    """
    Finalize training signal for MIPROv2 optimization.

    Collects metrics from the completed pipeline run and creates
    a training signal that can be used for prompt optimization.

    Args:
        state: Completed pipeline state

    Returns:
        State with finalized training signal
    """
    patterns = state.get("detected_patterns") or []
    recommendations = state.get("learning_recommendations") or []
    applied_updates = state.get("applied_updates") or []
    feedback_items = state.get("feedback_items") or []

    # Calculate metrics for training signal
    pattern_accuracy = 0.85 if patterns else 0.0  # Placeholder - would be validated
    recommendation_actionability = min(len(recommendations) / 5.0, 1.0) if recommendations else 0.0
    update_effectiveness = len(applied_updates) / max(len(state.get("proposed_updates") or []), 1)
    min(1.0, 5000 / max(state.get("total_latency_ms", 1), 1))  # Target < 5s
    min(len(patterns) / max(len(feedback_items), 1), 1.0) if feedback_items else 0.0

    # Get rubric evaluation metrics if available
    rubric_weighted_score = state.get("rubric_weighted_score")
    rubric_decision = state.get("rubric_decision")
    rubric_pattern_flags = state.get("rubric_pattern_flags") or []

    training_signal = FeedbackLearnerTrainingSignal(
        batch_id=state.get("batch_id", ""),
        feedback_count=len(feedback_items),
        time_range_start=state.get("time_range_start", ""),
        time_range_end=state.get("time_range_end", ""),
        focus_agents=state.get("focus_agents") or [],
        cognitive_context=state.get("cognitive_context"),  # type: ignore[arg-type]
        patterns_detected=len(patterns),
        recommendations_generated=len(recommendations),
        updates_applied=len(applied_updates),
        pattern_accuracy=pattern_accuracy,
        recommendation_actionability=recommendation_actionability,
        update_effectiveness=update_effectiveness,
        rubric_weighted_score=rubric_weighted_score,
        rubric_decision=rubric_decision,
        rubric_pattern_flags=len(rubric_pattern_flags),
        collection_latency_ms=state.get("collection_latency_ms", 0),
        analysis_latency_ms=state.get("analysis_latency_ms", 0),
        extraction_latency_ms=state.get("extraction_latency_ms", 0),
        update_latency_ms=state.get("update_latency_ms", 0),
        total_latency_ms=state.get("total_latency_ms", 0),
        model_used=state.get("model_used") or "deterministic",
    )

    rubric_info = ""
    if rubric_weighted_score is not None:
        rubric_info = (
            f", rubric_score={rubric_weighted_score:.2f}, rubric_decision={rubric_decision}"
        )

    logger.info(
        f"Training signal finalized: reward={training_signal.compute_reward():.3f}, "
        f"patterns={len(patterns)}, recommendations={len(recommendations)}{rubric_info}"
    )

    return {
        **state,
        "training_signal": training_signal,
        "status": "completed",
    }


async def _error_handler_node(state: FeedbackLearnerState) -> FeedbackLearnerState:
    """Handle errors in the pipeline."""
    errors = state.get("errors") or []
    error_messages = [e.get("error", "Unknown error") for e in errors]

    logger.error(f"Feedback learning pipeline failed: {error_messages}")

    # Still create a training signal for failed runs (for learning from failures)
    training_signal = FeedbackLearnerTrainingSignal(
        batch_id=state.get("batch_id", ""),
        feedback_count=len(state.get("feedback_items") or []),
        time_range_start=state.get("time_range_start", ""),
        time_range_end=state.get("time_range_end", ""),
        focus_agents=state.get("focus_agents") or [],
        cognitive_context=None,
        patterns_detected=0,
        recommendations_generated=0,
        updates_applied=0,
        pattern_accuracy=0.0,
        recommendation_actionability=0.0,
        update_effectiveness=0.0,
        collection_latency_ms=state.get("collection_latency_ms", 0),
        analysis_latency_ms=state.get("analysis_latency_ms", 0),
        extraction_latency_ms=state.get("extraction_latency_ms", 0),
        update_latency_ms=state.get("update_latency_ms", 0),
        total_latency_ms=state.get("total_latency_ms", 0),
        model_used="deterministic",
    )

    return {
        **state,
        "learning_summary": f"Learning cycle failed: {'; '.join(error_messages)}",
        "training_signal": training_signal,
        "status": "failed",
    }
