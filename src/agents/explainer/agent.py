"""
E2I Explainer Agent - Main Agent Class
Version: 4.2
Purpose: Natural language explanations for complex analyses

Memory Integration:
- Working Memory (Redis): Session caching, conversation context
- Episodic Memory (Supabase): Historical explanations with embeddings
- Semantic Memory (FalkorDB): Entity relationships and knowledge graph
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .graph import build_explainer_graph
from .state import ExplainerState, Insight, NarrativeSection

if TYPE_CHECKING:
    from .memory_hooks import ExplanationMemoryHooks

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONTRACTS
# ============================================================================


class ExplainerInput(BaseModel):
    """Input contract for Explainer agent."""

    query: str = ""
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
    output_format: Literal["narrative", "structured", "presentation", "brief"] = "narrative"
    focus_areas: Optional[List[str]] = None

    # Memory integration fields
    session_id: Optional[str] = None  # For memory correlation
    memory_config: Optional[Dict[str, Any]] = None  # Memory configuration (brand, region)


class ExplainerOutput(BaseModel):
    """Output contract for Explainer agent."""

    executive_summary: str = ""
    detailed_explanation: str = ""
    narrative_sections: List[NarrativeSection] = Field(default_factory=list)
    extracted_insights: List[Insight] = Field(default_factory=list)
    key_themes: List[str] = Field(default_factory=list)
    visual_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    total_latency_ms: int = 0
    model_used: str = ""
    timestamp: str = ""
    status: str = "pending"
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# AGENT CLASS
# ============================================================================


class ExplainerAgent:
    """
    Tier 5 Explainer Agent.

    Responsibilities:
    - Synthesize complex analyses into clear narratives
    - Adapt explanations to user expertise level
    - Generate actionable insights
    - Suggest visualizations and follow-up questions

    Memory Integration:
    - Working Memory (Redis): Session caching with 24h TTL
    - Episodic Memory (Supabase): Historical explanations
    - Semantic Memory (FalkorDB): Entity relationships
    """

    def __init__(
        self,
        conversation_store: Optional[Any] = None,
        use_llm: bool = False,
        llm: Optional[Any] = None,
    ):
        """
        Initialize Explainer agent.

        Args:
            conversation_store: Optional store for conversation history
            use_llm: Whether to use LLM for enhanced reasoning
            llm: Optional LLM instance to use
        """
        self._conversation_store = conversation_store
        self._use_llm = use_llm
        self._llm = llm
        self._graph = None
        self._memory_hooks = None

    @property
    def graph(self):
        """Lazy-load the explanation graph."""
        if self._graph is None:
            self._graph = build_explainer_graph(
                conversation_store=self._conversation_store,
                use_llm=self._use_llm,
                llm=self._llm,
            )
        return self._graph

    @property
    def memory_hooks(self) -> Optional["ExplanationMemoryHooks"]:
        """Lazy-load memory hooks for tri-memory integration."""
        if self._memory_hooks is None:
            try:
                from .memory_hooks import get_explanation_memory_hooks

                self._memory_hooks = get_explanation_memory_hooks()
                logger.debug("Memory hooks initialized for Explainer agent")
            except Exception as e:
                logger.warning(f"Failed to initialize memory hooks: {e}")
                return None
        return self._memory_hooks

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for memory correlation."""
        return f"explainer_{uuid.uuid4().hex[:12]}"

    async def explain(
        self,
        analysis_results: List[Dict[str, Any]],
        query: str = "",
        user_expertise: str = "analyst",
        output_format: str = "narrative",
        focus_areas: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> ExplainerOutput:
        """
        Generate natural language explanation for analysis results.

        Args:
            analysis_results: Results from upstream agents
            query: Original user query
            user_expertise: Target audience expertise level
            output_format: Desired output format
            focus_areas: Specific areas to focus on
            session_id: Session ID for memory correlation
            memory_config: Memory configuration (brand, region)

        Returns:
            ExplainerOutput with narrative explanation
        """
        # Generate session ID if not provided
        effective_session_id = session_id or self._generate_session_id()

        initial_state: ExplainerState = {
            "query": query,
            "analysis_results": analysis_results,
            "user_expertise": user_expertise,
            "output_format": output_format,
            "focus_areas": focus_areas,
            # Memory integration fields
            "session_id": effective_session_id,
            "memory_config": memory_config or {},
            "episodic_context": None,
            "semantic_context": None,
            "working_memory_messages": None,
            # Context fields
            "analysis_context": None,
            "user_context": None,
            "conversation_history": None,
            "extracted_insights": None,
            "narrative_structure": None,
            "key_themes": None,
            "executive_summary": None,
            "detailed_explanation": None,
            "narrative_sections": None,
            "visual_suggestions": None,
            "follow_up_questions": None,
            "related_analyses": None,
            "assembly_latency_ms": 0,
            "reasoning_latency_ms": 0,
            "generation_latency_ms": 0,
            "total_latency_ms": 0,
            "model_used": None,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        logger.info(
            f"Starting explanation: {len(analysis_results)} results, "
            f"expertise={user_expertise}, format={output_format}, "
            f"session={effective_session_id}"
        )

        # Provide config with thread_id for checkpointer (if enabled)
        config = {"configurable": {"thread_id": effective_session_id}}
        result = await self.graph.ainvoke(initial_state, config=config)

        return ExplainerOutput(
            executive_summary=result.get("executive_summary", ""),
            detailed_explanation=result.get("detailed_explanation", ""),
            narrative_sections=result.get("narrative_sections") or [],
            extracted_insights=result.get("extracted_insights") or [],
            key_themes=result.get("key_themes") or [],
            visual_suggestions=result.get("visual_suggestions") or [],
            follow_up_questions=result.get("follow_up_questions") or [],
            total_latency_ms=result.get("total_latency_ms", 0),
            model_used=result.get("model_used") or "deterministic",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=result.get("status", "failed"),
            errors=result.get("errors") or [],
            warnings=result.get("warnings") or [],
        )

    async def summarize(
        self, analysis_results: List[Dict[str, Any]], query: str = ""
    ) -> ExplainerOutput:
        """
        Generate brief summary of analysis results.

        Args:
            analysis_results: Results from upstream agents
            query: Original user query

        Returns:
            ExplainerOutput with brief summary
        """
        return await self.explain(
            analysis_results=analysis_results,
            query=query,
            user_expertise="executive",
            output_format="brief",
        )

    async def explain_for_audience(
        self,
        analysis_results: List[Dict[str, Any]],
        audience: str,
        query: str = "",
    ) -> ExplainerOutput:
        """
        Generate explanation tailored to specific audience.

        Args:
            analysis_results: Results from upstream agents
            audience: Target audience type
            query: Original user query

        Returns:
            ExplainerOutput tailored to audience
        """
        return await self.explain(
            analysis_results=analysis_results,
            query=query,
            user_expertise=audience,
            output_format="narrative",
        )

    def get_handoff(self, output: ExplainerOutput) -> Dict[str, Any]:
        """
        Generate handoff for orchestrator.

        Args:
            output: Explanation output

        Returns:
            Handoff dictionary for other agents
        """
        insights = output.extracted_insights or []
        finding_count = len([i for i in insights if i.get("category") == "finding"])
        rec_count = len([i for i in insights if i.get("category") == "recommendation"])

        return {
            "agent": "explainer",
            "analysis_type": "explanation",
            "key_findings": {
                "insight_count": len(insights),
                "finding_count": finding_count,
                "recommendation_count": rec_count,
                "themes": output.key_themes[:3] if output.key_themes else [],
            },
            "outputs": {
                "executive_summary": "available" if output.executive_summary else "unavailable",
                "detailed_explanation": (
                    "available" if output.detailed_explanation else "unavailable"
                ),
                "sections": len(output.narrative_sections),
            },
            "suggestions": {
                "visuals": [v.get("type") for v in output.visual_suggestions[:3]],
                "follow_ups": output.follow_up_questions[:3],
            },
            "requires_further_analysis": output.status == "failed",
            "suggested_next_agent": "feedback_learner" if output.status == "completed" else None,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def explain_analysis(
    analysis_results: List[Dict[str, Any]],
    query: str = "",
    user_expertise: str = "analyst",
    output_format: str = "narrative",
) -> ExplainerOutput:
    """
    Convenience function for generating explanations.

    Args:
        analysis_results: Results from upstream agents
        query: Original user query
        user_expertise: Target audience expertise level
        output_format: Desired output format

    Returns:
        ExplainerOutput
    """
    agent = ExplainerAgent()
    return await agent.explain(
        analysis_results=analysis_results,
        query=query,
        user_expertise=user_expertise,
        output_format=output_format,
    )
