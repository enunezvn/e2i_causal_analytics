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

from .dspy_integration import DSPY_AVAILABLE
from .graph import build_feedback_learner_graph
from .state import (
    DetectedPattern,
    FeedbackLearnerState,
    KnowledgeUpdate,
    LearningRecommendation,
)

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
        persist_client: Optional[Any] = None,
        persist_signals: bool = True,
        db_client: Optional[Any] = None,
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
            persist_client: Optional Supabase client used to persist the
                finalized training signal (audit F5). When None, the default
                factory client is used at persist time.
            persist_signals: When True (default), persist the finalized signal to
                ``dspy_agent_training_signals`` so it is durable + readable by the
                optimizer. Best-effort: a DB error never fails a learning cycle.
            db_client: Optional async Supabase client for the rubric node's
                ``learning_signals`` persistence (#883 deferred item). When
                None the rubric node derives no context and skips — production
                triggers pass the shared client from
                :func:`build_production_feedback_stores`.
        """
        self._feedback_store = feedback_store
        self._outcome_store = outcome_store
        self._knowledge_stores = knowledge_stores
        self._use_llm = use_llm
        self._llm = llm
        self._cognitive_rag = cognitive_rag
        self._persist_client = persist_client
        self._persist_signals = persist_signals
        self._db_client = db_client
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
                db_client=self._db_client,
                persist_signals=self._persist_signals,
                persist_client=self._persist_client,
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

        # NotRequired fields are omitted - populated during graph execution
        initial_state: FeedbackLearnerState = {
            "batch_id": batch_id,
            "time_range_start": time_range_start,
            "time_range_end": time_range_end,
            "focus_agents": focus_agents or [],
            # Required output fields with initial values
            "learning_summary": "",
            "collection_latency_ms": 0,
            "analysis_latency_ms": 0,
            "extraction_latency_ms": 0,
            "update_latency_ms": 0,
            "total_latency_ms": 0,
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

        # Note: persistence is now owned by the finalize_node closure inside the
        # graph (graph.py) so every caller — including the API route — persists
        # exactly once. Do NOT re-persist here.

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
            model_used=(
                result.get("model_used")
                if isinstance(result.get("model_used"), str)
                else "deterministic"
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=result.get("status", "failed"),
            errors=result.get("errors") or [],
            warnings=result.get("warnings") or [],
            # DSPy Integration fields
            training_reward=training_reward,
            cognitive_context_used=cognitive_context is not None,
            dspy_available=DSPY_AVAILABLE,
        )

    async def process_feedback(self, feedback_items: List[Dict[str, Any]]) -> FeedbackLearnerOutput:
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


class _CompositeFeedbackStore:
    """Fan ``get_feedback`` out to several stores and concatenate the results.

    Production composes explicit chat thumbs (``chatbot_message_feedback``)
    with the always-flowing cognitive reward stream (``learning_signals``), so
    learning cycles have material even when thumbs volume is low. Each store
    fails closed to ``[]`` independently — a broken source degrades coverage
    without killing the cycle.
    """

    def __init__(self, stores: List[Any]):
        self._stores = [s for s in stores if s is not None]

    async def get_feedback(self, **kwargs: Any) -> List[Dict[str, Any]]:
        import asyncio

        results = await asyncio.gather(
            *(s.get_feedback(**kwargs) for s in self._stores),
            return_exceptions=True,
        )
        items: List[Dict[str, Any]] = []
        for store, res in zip(self._stores, results, strict=True):
            if isinstance(res, BaseException):
                logger.warning(
                    "feedback_learner: store %s failed, continuing without it: %s",
                    type(store).__name__,
                    res,
                )
                continue
            items.extend(res or [])
        return items


async def build_production_feedback_stores() -> tuple[
    Optional[Any], Optional[Dict[str, Any]], Optional[Any]
]:
    """Build the REAL ``(feedback_store, knowledge_stores, db_client)`` for every
    production learning-cycle trigger (the Celery task, the ``/feedback/learn``
    route, and :func:`process_feedback_batch`) from one async Supabase client.

    The third element is the shared async client itself, injected into the
    graph build as the rubric node's ``learning_signals`` sink (#883 deferred
    item: ``graph.py`` plumbed ``db_client`` but no production site armed it,
    leaving the rubric persistence path structurally dead).

    FAIL-CLOSED: returns ``(None, None, None)`` when the client is unavailable
    (SUPABASE_URL unset, CI / offline) so the cycle runs the HONEST unwired path —
    ``update_backend_wired`` False, ``update_effectiveness`` None (the F15
    contract), never a fabricated 0.0 — and the rubric node skips (it never
    constructs a client-less write path, the #845 convention).

    NOTE on the orchestrator-DISPATCHED path: it never reaches this builder OR the
    learning cycle. The agent registry holds a pre-built ``FeedbackLearnerAgent``
    (constructed synchronously at startup via the generic factory), and the
    dispatcher splats the generic dispatch payload into ``learn(**kwargs)``.
    ``learn`` has a narrow signature (``time_range_start``/``end``, ``batch_id``,
    ``focus_agents``) and there is NO ``feedback_learner`` ``INPUT_RESOLVER``, so
    that call FAILS CLOSED with a kwargs-mismatch ``TypeError`` (success=False)
    BEFORE ``KnowledgeUpdaterNode`` runs — ``update_effectiveness`` is never
    computed, never fabricated. That is a PRE-EXISTING dispatch-wiring gap (a real
    ``feedback_learner`` input resolver belongs to the resolver registry, #839),
    not something #837 introduces or worsens. The measurable learning-cycle
    triggers are the async entry points wired through this builder.
    """
    try:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.chatbot_feedback import get_chatbot_feedback_repository
        from src.repositories.learning_signals_feedback import (
            get_learning_signals_feedback_store,
        )

        from .knowledge_stores import build_knowledge_stores

        client = await get_async_supabase_client()
        # Explicit chat thumbs + the per-turn cognitive reward stream. The
        # composite keeps cycles fed on real data even when nobody clicks the
        # thumbs (the rating UI ships with this change; volume will ramp).
        feedback_store = _CompositeFeedbackStore(
            [
                get_chatbot_feedback_repository(supabase_client=client),
                get_learning_signals_feedback_store(supabase_client=client),
            ]
        )
        return (
            feedback_store,
            build_knowledge_stores(client),
            client,
        )
    except Exception as exc:  # pragma: no cover - degraded path
        logger.warning(
            "feedback_learner: could not build production feedback/knowledge stores "
            "(%s); learning from empty, update_effectiveness stays None, "
            "rubric persistence disarmed",
            exc,
        )
        return None, None, None


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
    # #837: wire the real feedback + knowledge stores so update_effectiveness is
    # measurable on this public convenience path too (fail-closed → unwired/None).
    # #883 deferred: the shared client also arms the rubric persistence path.
    # Post-auto_apply-gate note: `learn()` (below) has no `auto_apply` parameter
    # and never threads one into `initial_state`, so this path is unconditionally
    # propose-only — update_effectiveness is honestly None here regardless of
    # knowledge_stores wiring, until a future change (see API route auto_apply
    # threading) adds a way to opt in.
    feedback_store, knowledge_stores, db_client = await build_production_feedback_stores()
    agent = FeedbackLearnerAgent(
        feedback_store=feedback_store,
        knowledge_stores=knowledge_stores,
        db_client=db_client,
    )
    return await agent.learn(
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        focus_agents=focus_agents,
    )
