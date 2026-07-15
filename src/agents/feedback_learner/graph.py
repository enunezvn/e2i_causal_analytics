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
from functools import partial
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
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
    prefer_optimized: bool = True,
    persist_signals: bool = True,
    persist_client: Optional[Any] = None,
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
        prefer_optimized: Whether the pattern analyzer should prefer the latest
            optimized DSPy module (closes the self-improvement loop; default: True)
        persist_signals: When True (default), persist the finalized training signal
            to ``dspy_agent_training_signals`` from the finalize node so every
            caller of the graph (including the API route) persists exactly once.
            Best-effort: a DB error never fails the node.
        persist_client: Optional Supabase client for persistence. When None and
            ``persist_signals`` is True, the default factory client is used.

    Returns:
        Compiled LangGraph workflow
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("feedback_learner", AgentTier.SELF_IMPROVEMENT)

    # Initialize nodes
    collector = FeedbackCollectorNode(feedback_store, outcome_store)
    analyzer = PatternAnalyzerNode(use_llm=use_llm, llm=llm, prefer_optimized=prefer_optimized)
    rubric_node = RubricNode(db_client=db_client) if enable_rubric_evaluation else None
    extractor = LearningExtractorNode(use_llm=use_llm, llm=llm)
    updater = KnowledgeUpdaterNode(knowledge_stores)

    # Build graph
    workflow = StateGraph(FeedbackLearnerState)

    # Create cognitive enricher with bound cognitive_rag
    async def enrich_node(state: FeedbackLearnerState) -> FeedbackLearnerState:
        return await _cognitive_context_enricher(state, cognitive_rag)

    # Create finalize node closure that optionally persists the training signal.
    # Mirrors the enrich_node pattern — captures build-time args so every
    # caller of the graph (including the API route) persists exactly once.
    async def finalize_node(state: FeedbackLearnerState) -> FeedbackLearnerState:
        result = await _finalize_training_signal(state)
        if persist_signals:
            training_signal = result.get("training_signal")
            # Defensive guard (mirrors the old learn() guard): the finalize node
            # always builds a real FeedbackLearnerTrainingSignal, but guard against
            # a non-standard type so we never call persist on something unexpected.
            if training_signal is not None and hasattr(training_signal, "compute_reward"):
                try:
                    from .signal_store import persist_training_signal

                    await persist_training_signal(training_signal, client=persist_client)
                except Exception as exc:  # noqa: BLE001 - best-effort
                    logger.warning(
                        "finalize_node: failed to persist training signal batch=%s: %s",
                        getattr(training_signal, "batch_id", "?"),
                        exc,
                    )
        return result

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node, agent_name="feedback_learner", agent_tier=AgentTier.SELF_IMPROVEMENT
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "enrich", enrich_node)
    timed(workflow, "collect", collector.execute)
    timed(workflow, "analyze", analyzer.execute)
    if rubric_node:
        timed(workflow, "rubric", rubric_node.execute)
    timed(workflow, "extract", extractor.execute)
    timed(workflow, "update", updater.execute)
    timed(workflow, "finalize", finalize_node)
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
    # `applied_updates` in state is a list of applied update_id STRINGS (see
    # KnowledgeUpdaterNode); the full update dicts live in `proposed_updates`.
    applied_updates = state.get("applied_updates") or []
    proposed_updates = state.get("proposed_updates") or []
    applied_update_ids = set(applied_updates)
    applied_update_records = [
        dict(u)
        for u in proposed_updates
        if isinstance(u, dict) and u.get("update_id") in applied_update_ids
    ]
    feedback_items = state.get("feedback_items") or []

    # Calculate metrics for training signal.
    #
    # F-015 (issue #424): `pattern_accuracy` requires ground-truth-validated
    # pattern labels to be computed honestly. The labeling infrastructure does
    # not yet exist (see #426 / F-015-PhaseB). Until it lands, propagate `None`
    # so downstream consumers (`compute_reward`, MIPROv2 optimization) skip the
    # accuracy term rather than anchor on a fabricated value. Setting this to a
    # constant like 0.85 would silently bias the self-improvement loop.
    pattern_accuracy: float | None = None
    recommendation_actionability = min(len(recommendations) / 5.0, 1.0) if recommendations else 0.0
    # F15 (audit): update_effectiveness is only measurable when a real
    # knowledge_store apply-backend is wired AND updates were proposed.
    # Otherwise applied_updates is structurally empty regardless of feedback, so
    # we emit None (cf. pattern_accuracy above) — compute_reward then SKIPS the
    # term and redistributes its weight, rather than anchoring the
    # self-improvement reward on a misleading 0.0. A real knowledge_stores
    # backend is a separate feature (see F15 follow-up).
    # ... AND application was actually attempted: with auto_apply=False the
    # updater withholds every apply, so applied/proposed would fabricate a
    # 0.0 "ineffective" when effectiveness is simply unmeasurable this cycle.
    _proposed = state.get("proposed_updates") or []
    update_effectiveness: float | None
    if state.get("update_backend_wired") and _proposed and state.get("auto_apply"):
        update_effectiveness = len(applied_updates) / len(_proposed)
    else:
        update_effectiveness = None

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
        # F6: carry bounded real content so signal->Example conversion has input.
        feedback_batch=[dict(fb) for fb in feedback_items[:20]],
        patterns=[dict(p) for p in patterns[:20]],
        recommendations=[dict(r) for r in recommendations[:20]],
        applied_updates=applied_update_records[:20],
        learning_summary=state.get("learning_summary") or "",
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
        # F-015 (issue #424): use None for "no measurement", not fabricated 0.0.
        # See dspy_integration.FeedbackLearnerTrainingSignal.pattern_accuracy.
        pattern_accuracy=None,
        recommendation_actionability=0.0,
        # F15: unmeasurable on the error/empty path -> None (skipped in reward).
        update_effectiveness=None,
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
