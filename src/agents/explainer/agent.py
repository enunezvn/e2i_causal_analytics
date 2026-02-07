"""
E2I Explainer Agent - Main Agent Class
Version: 4.3
Purpose: Natural language explanations for complex analyses

Memory Integration:
- Working Memory (Redis): Session caching, conversation context
- Episodic Memory (Supabase): Historical explanations with embeddings
- Semantic Memory (FalkorDB): Entity relationships and knowledge graph

Smart LLM Mode Selection (v4.3):
- Auto-detects complexity to decide LLM vs deterministic mode
- Configurable threshold and scoring weights
- Considers: result count, query complexity, causal discovery, expertise level
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.agents.base import SkillsMixin

from .config import ComplexityScorer, ExplainerConfig, get_default_config
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


class ExplainerAgent(SkillsMixin):
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

    Smart LLM Mode Selection (v4.3):
    - use_llm=None: Auto-detect based on input complexity
    - use_llm=True: Always use LLM reasoning
    - use_llm=False: Always use deterministic mode

    Skills Integration:
        - causal-inference/dowhy-workflow.md: Causal explanation context
        - pharma-commercial/brand-analytics.md: Brand-specific terminology
    """

    def __init__(
        self,
        conversation_store: Optional[Any] = None,
        use_llm: Optional[bool] = None,
        llm: Optional[Any] = None,
        config: Optional[ExplainerConfig] = None,
    ):
        """
        Initialize Explainer agent.

        Args:
            conversation_store: Optional store for conversation history
            use_llm: LLM mode selection:
                - None (default): Auto-detect based on complexity
                - True: Always use LLM reasoning
                - False: Always use deterministic mode
            llm: Optional LLM instance to use
            config: Optional configuration for complexity scoring
        """
        self._conversation_store = conversation_store
        self._use_llm = use_llm  # None = auto mode
        self._llm = llm
        self._config = config or get_default_config()
        self._complexity_scorer = ComplexityScorer(self._config)
        self._graph = None
        self._graph_use_llm = False  # Track which mode graph was built with
        self._memory_hooks = None

    def _get_graph(self, use_llm: bool):
        """
        Get or build the explanation graph for the specified LLM mode.

        Args:
            use_llm: Whether to use LLM mode

        Returns:
            The compiled LangGraph
        """
        # Rebuild graph if LLM mode changed
        if self._graph is None or self._graph_use_llm != use_llm:
            _testing = os.environ.get("E2I_TESTING_MODE", "").lower() in ("true", "1", "yes")
            self._graph = build_explainer_graph(
                conversation_store=self._conversation_store,
                use_llm=use_llm,
                llm=self._llm,
                use_default_checkpointer=not _testing,
            )
            self._graph_use_llm = use_llm
        return self._graph

    @property
    def graph(self):
        """Lazy-load the explanation graph with default LLM mode."""
        # For backward compatibility, use False if use_llm is None
        effective_use_llm = self._use_llm if self._use_llm is not None else False
        return self._get_graph(effective_use_llm)

    def _should_use_llm(
        self,
        analysis_results: List[Dict[str, Any]],
        query: str,
        user_expertise: str,
    ) -> tuple[bool, float, str]:
        """
        Determine if LLM should be used based on input complexity.

        Args:
            analysis_results: Analysis results to explain
            query: User's query
            user_expertise: Target audience expertise level

        Returns:
            Tuple of (should_use_llm, complexity_score, reason)
        """
        # Check for causal discovery data in results
        has_causal_discovery = any(
            any(
                key in result
                for key in [
                    "discovered_dag",
                    "causal_graph",
                    "dag_adjacency",
                    "discovery_gate_decision",
                ]
            )
            for result in analysis_results
        )

        return self._complexity_scorer.should_use_llm(
            analysis_results=analysis_results,
            query=query,
            user_expertise=user_expertise,
            has_causal_discovery=has_causal_discovery,
        )

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

    async def _load_explanation_skills(
        self,
        analysis_results: List[Dict[str, Any]],
        memory_config: Optional[Dict[str, Any]],
    ) -> None:
        """Load relevant skills for explanation generation.

        Loads domain-specific procedural knowledge based on the analysis context.
        Skills are optional - explanation proceeds without them if unavailable.

        Args:
            analysis_results: The analysis results to explain.
            memory_config: Optional memory configuration with brand info.
        """
        try:
            # Check if results include causal analysis
            has_causal = any(
                any(key in result for key in ["causal_effect", "treatment_effect", "ate", "cate"])
                for result in analysis_results
            )

            if has_causal:
                await self.load_skill("causal-inference/dowhy-workflow.md")

            # Load brand-specific context if brand is specified
            brand = memory_config.get("brand") if memory_config else None
            if brand:
                await self.load_skill("pharma-commercial/brand-analytics.md")

            loaded_names = self.get_loaded_skill_names()
            if loaded_names:
                logger.info(f"Loaded {len(loaded_names)} explanation skills: {loaded_names}")
        except Exception as e:
            # Skills are optional - log warning and proceed without
            logger.warning(f"Failed to load explanation skills (proceeding without): {e}")

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
        # Clear loaded skills from previous invocation
        self.clear_loaded_skills()

        # Load relevant domain skills for explanation
        await self._load_explanation_skills(analysis_results, memory_config)

        # Generate session ID if not provided
        effective_session_id = session_id or self._generate_session_id()

        # Determine LLM mode
        if self._use_llm is not None:
            # Explicit mode set
            effective_use_llm = self._use_llm
            complexity_score = None
            llm_reason = "explicit_setting"
        else:
            # Auto-detect based on complexity
            effective_use_llm, complexity_score, llm_reason = self._should_use_llm(
                analysis_results, query, user_expertise
            )
            logger.info(
                f"Auto LLM selection: use_llm={effective_use_llm}, "
                f"complexity={complexity_score:.2f}, reason={llm_reason}"
            )

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
            f"use_llm={effective_use_llm}, session={effective_session_id}"
        )

        # Get graph with appropriate LLM mode
        graph = self._get_graph(effective_use_llm)

        # Provide config with thread_id for checkpointer (if enabled)
        config = {"configurable": {"thread_id": effective_session_id}}
        result = await graph.ainvoke(initial_state, config=config)

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
