"""
E2I Feedback Learner Agent - Main Agent Class
Version: 4.2
Purpose: Self-improvement from user feedback

DSPy Integration:
- CognitiveRAG context enrichment at pipeline entry
- Training signal collection for MIPROv2 optimization
- Memory contribution helpers for system learning
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .state import (
    FeedbackLearnerState,
    DetectedPattern,
    LearningRecommendation,
    KnowledgeUpdate,
)
from .graph import build_feedback_learner_graph, build_simple_feedback_learner_graph
from .dspy_integration import FeedbackLearnerTrainingSignal, DSPY_AVAILABLE

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONTRACTS
# ============================================================================


class FeedbackLearnerInput(BaseModel):
    """Input contract for Feedback Learner agent."""

    batch_id: str = ""
    time_range_start: str = ""
    time_range_end: str = ""
    focus_agents: Optional[List[str]] = None


class FeedbackLearnerOutput(BaseModel):
    """Output contract for Feedback Learner agent."""

    batch_id: str = ""
    detected_patterns: List[DetectedPattern] = Field(default_factory=list)
    learning_recommendations: List[LearningRecommendation] = Field(default_factory=list)
    priority_improvements: List[str] = Field(default_factory=list)
    proposed_updates: List[KnowledgeUpdate] = Field(default_factory=list)
    applied_updates: List[str] = Field(default_factory=list)
    learning_summary: str = ""
    feedback_count: int = 0
    pattern_count: int = 0
    recommendation_count: int = 0
    total_latency_ms: int = 0
    model_used: str = ""
    timestamp: str = ""
    status: str = "pending"
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # DSPy Integration fields
    training_reward: Optional[float] = None
    cognitive_context_used: bool = False
    dspy_available: bool = DSPY_AVAILABLE


# ============================================================================
# AGENT CLASS
# ============================================================================


class FeedbackLearnerAgent:
    """
    Tier 5 Feedback Learner Agent.

    Responsibilities:
    - Process user feedback batches
    - Detect systematic patterns
    - Generate improvement recommendations
    - Update organizational knowledge

    DSPy Integration:
    - Accepts CognitiveRAG for 4-phase cognitive enrichment
    - Collects training signals for MIPROv2 optimization
    - Generates memory contributions for system learning
    """

    def __init__(
        self,
        feedback_store: Optional[Any] = None,
        outcome_store: Optional[Any] = None,
        knowledge_stores: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
        llm: Optional[Any] = None,
        cognitive_rag: Optional[Any] = None,
    ):
        """
        Initialize Feedback Learner agent.

        Args:
            feedback_store: Store for user feedback
            outcome_store: Store for outcome data
            knowledge_stores: Dictionary of knowledge stores by type
            use_llm: Whether to use LLM for analysis
            llm: Optional LLM instance
            cognitive_rag: Optional CognitiveRAG instance for context enrichment
        """
        self._feedback_store = feedback_store
        self._outcome_store = outcome_store
        self._knowledge_stores = knowledge_stores
        self._use_llm = use_llm
        self._llm = llm
        self._cognitive_rag = cognitive_rag
        self._graph = None

    @property
    def graph(self):
        """Lazy-load the feedback learning graph with DSPy integration."""
        if self._graph is None:
            self._graph = build_feedback_learner_graph(
                feedback_store=self._feedback_store,
                outcome_store=self._outcome_store,
                knowledge_stores=self._knowledge_stores,
                use_llm=self._use_llm,
                llm=self._llm,
                cognitive_rag=self._cognitive_rag,
            )
        return self._graph

    async def learn(
        self,
        time_range_start: str,
        time_range_end: str,
        batch_id: Optional[str] = None,
        focus_agents: Optional[List[str]] = None,
    ) -> FeedbackLearnerOutput:
        """
        Process a batch of feedback to learn and improve.

        Args:
            time_range_start: Start of time range (ISO format)
            time_range_end: End of time range (ISO format)
            batch_id: Optional batch identifier
            focus_agents: Optional list of agents to focus on

        Returns:
            FeedbackLearnerOutput with learning results
        """
        if not batch_id:
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        initial_state: FeedbackLearnerState = {
            "batch_id": batch_id,
            "time_range_start": time_range_start,
            "time_range_end": time_range_end,
            "focus_agents": focus_agents,
            # DSPy Integration fields
            "cognitive_context": None,
            "training_signal": None,
            # Feedback data
            "feedback_items": None,
            "feedback_summary": None,
            "detected_patterns": None,
            "pattern_clusters": None,
            "learning_recommendations": None,
            "priority_improvements": None,
            "proposed_updates": None,
            "applied_updates": None,
            "learning_summary": None,
            "metrics_before": None,
            "metrics_after": None,
            "collection_latency_ms": 0,
            "analysis_latency_ms": 0,
            "extraction_latency_ms": 0,
            "update_latency_ms": 0,
            "total_latency_ms": 0,
            "model_used": None,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        logger.info(
            f"Starting learning cycle: batch={batch_id}, "
            f"range={time_range_start} to {time_range_end}"
        )

        result = await self.graph.ainvoke(initial_state)

        feedback_items = result.get("feedback_items") or []
        patterns = result.get("detected_patterns") or []
        recommendations = result.get("learning_recommendations") or []
        training_signal = result.get("training_signal")
        cognitive_context = result.get("cognitive_context")

        # Extract training reward if available
        training_reward = None
        if training_signal is not None and hasattr(training_signal, "compute_reward"):
            training_reward = training_signal.compute_reward()

        return FeedbackLearnerOutput(
            batch_id=batch_id,
            detected_patterns=patterns,
            learning_recommendations=recommendations,
            priority_improvements=result.get("priority_improvements") or [],
            proposed_updates=result.get("proposed_updates") or [],
            applied_updates=result.get("applied_updates") or [],
            learning_summary=result.get("learning_summary") or "",
            feedback_count=len(feedback_items),
            pattern_count=len(patterns),
            recommendation_count=len(recommendations),
            total_latency_ms=result.get("total_latency_ms", 0),
            model_used=result.get("model_used") if isinstance(result.get("model_used"), str) else "deterministic",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=result.get("status", "failed"),
            errors=result.get("errors") or [],
            warnings=result.get("warnings") or [],
            # DSPy Integration fields
            training_reward=training_reward,
            cognitive_context_used=cognitive_context is not None,
            dspy_available=DSPY_AVAILABLE,
        )

    async def process_feedback(
        self, feedback_items: List[Dict[str, Any]]
    ) -> FeedbackLearnerOutput:
        """
        Process a specific list of feedback items.

        Args:
            feedback_items: List of feedback items to process

        Returns:
            FeedbackLearnerOutput with learning results
        """
        # Convert to proper format and call learn with mock store
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Create a simple in-memory mock store
        class MockStore:
            def __init__(self, items):
                self._items = items

            async def get_feedback(self, **kwargs):
                return self._items

        mock_store = MockStore(feedback_items)

        # Temporarily replace store
        original_store = self._feedback_store
        self._feedback_store = mock_store
        self._graph = None  # Reset graph to use new store

        try:
            return await self.learn(
                time_range_start=now,
                time_range_end=now,
                batch_id=batch_id,
            )
        finally:
            self._feedback_store = original_store
            self._graph = None

    def get_handoff(self, output: FeedbackLearnerOutput) -> Dict[str, Any]:
        """
        Generate handoff for orchestrator.

        Args:
            output: Learning output

        Returns:
            Handoff dictionary for other agents
        """
        patterns = output.detected_patterns or []
        recommendations = output.learning_recommendations or []

        return {
            "agent": "feedback_learner",
            "analysis_type": "learning_cycle",
            "key_findings": {
                "feedback_processed": output.feedback_count,
                "patterns_detected": output.pattern_count,
                "recommendations": output.recommendation_count,
                "updates_applied": len(output.applied_updates),
            },
            "patterns": [
                {
                    "type": p.get("pattern_type"),
                    "severity": p.get("severity"),
                    "affected_agents": p.get("affected_agents", []),
                }
                for p in patterns[:3]
            ],
            "top_recommendations": output.priority_improvements[:3],
            "summary": output.learning_summary,
            "requires_further_analysis": output.status == "failed",
            "suggested_next_agent": "experiment_designer" if output.status == "completed" else None,
            # DSPy Integration
            "dspy_integration": {
                "training_reward": output.training_reward,
                "cognitive_context_used": output.cognitive_context_used,
                "dspy_available": output.dspy_available,
            },
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def process_feedback_batch(
    time_range_start: str,
    time_range_end: str,
    focus_agents: Optional[List[str]] = None,
) -> FeedbackLearnerOutput:
    """
    Convenience function for processing feedback batches.

    Args:
        time_range_start: Start of time range (ISO format)
        time_range_end: End of time range (ISO format)
        focus_agents: Optional list of agents to focus on

    Returns:
        FeedbackLearnerOutput
    """
    agent = FeedbackLearnerAgent()
    return await agent.learn(
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        focus_agents=focus_agents,
    )
