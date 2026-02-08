"""
Causal Impact Agent Memory Hooks
================================

Memory integration hooks for the Causal Impact agent's tri-memory architecture.

The Causal Impact agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past causal analyses)
3. Query semantic memory (FalkorDB - causal paths, entity relationships)
4. Store causal analysis results for future retrieval and RAG

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CausalAnalysisContext:
    """Context retrieved from all memory systems for causal analysis."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CausalAnalysisRecord:
    """Record of a causal analysis for storage in episodic memory."""

    session_id: str
    query: str
    treatment_var: str
    outcome_var: str
    ate_estimate: float
    confidence_interval: tuple
    refutation_passed: bool
    confidence: float
    executive_summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalPathRecord:
    """Record of a causal path for storage in semantic memory."""

    path_id: str
    treatment_var: str
    outcome_var: str
    confounders: List[str]
    ate_estimate: float
    effect_size: str
    confidence: float
    refutation_passed: bool


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class CausalImpactMemoryHooks:
    """
    Memory integration hooks for the Causal Impact agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache causal analysis results in working memory (24h TTL)
    - Store causal analyses in episodic memory for future retrieval
    - Store causal paths in semantic memory for knowledge graph
    - Query existing causal relationships from semantic memory
    """

    # Cache TTL in seconds (24 hours)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None
        self._semantic_memory = None

    # =========================================================================
    # LAZY-LOADED MEMORY CLIENTS
    # =========================================================================

    @property
    def working_memory(self):
        """Lazy-load Redis working memory (port 6382)."""
        if self._working_memory is None:
            try:
                from src.memory.working_memory import get_working_memory

                self._working_memory = get_working_memory()
                logger.debug("Working memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize working memory: {e}")
                self._working_memory = None
        return self._working_memory

    @property
    def semantic_memory(self):
        """Lazy-load FalkorDB semantic memory (port 6381)."""
        if self._semantic_memory is None:
            try:
                from src.memory.semantic_memory import get_semantic_memory

                self._semantic_memory = get_semantic_memory()
                logger.debug("Semantic memory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
                self._semantic_memory = None
        return self._semantic_memory

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        query: str,
        treatment_var: Optional[str] = None,
        outcome_var: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> CausalAnalysisContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            treatment_var: Optional treatment variable for filtering
            outcome_var: Optional outcome variable for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            CausalAnalysisContext with data from all memory systems
        """
        context = CausalAnalysisContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past causal analyses)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (existing causal paths, entity relationships)
        context.semantic_context = await self._get_semantic_context(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_causal_paths={len(context.semantic_context.get('causal_paths', []))}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation from working memory."""
        if not self.working_memory:
            return []

        try:
            messages = await self.working_memory.get_messages(session_id, limit=limit)
            return cast(List[Dict[str, Any]], messages)
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

    async def _get_episodic_context(
        self,
        query: str,
        treatment_var: Optional[str] = None,
        outcome_var: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past causal analyses."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="causal_analysis_completed",
                agent_name="causal_impact",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=True,
            )

            # Further filter by treatment/outcome if specified
            if treatment_var or outcome_var:
                filtered_results = []
                for result in results:
                    content = result.get("raw_content", {})
                    if treatment_var and content.get("treatment_var") != treatment_var:
                        continue
                    if outcome_var and content.get("outcome_var") != outcome_var:
                        continue
                    filtered_results.append(result)
                return filtered_results

            return results
        except Exception as e:
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    async def _get_semantic_context(
        self,
        treatment_var: Optional[str] = None,
        outcome_var: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query semantic memory for existing causal paths and entity relationships."""
        if not self.semantic_memory:
            return {"entities": [], "relationships": [], "causal_paths": []}

        try:
            result: Dict[str, Any] = {
                "entities": [],
                "relationships": [],
                "causal_paths": [],
                "prior_analyses": [],
            }

            # Get causal paths for treatment-outcome relationship
            if treatment_var and outcome_var:
                causal_paths = await self._get_causal_paths(treatment_var, outcome_var)
                result["causal_paths"] = causal_paths

            # Get paths impacting the outcome KPI
            if outcome_var:
                kpi_paths = await self._get_kpi_causal_paths(outcome_var)
                result["kpi_paths"] = kpi_paths

            # Get graph statistics for context
            stats = self.semantic_memory.get_graph_stats()
            result["graph_stats"] = stats

            return result
        except Exception as e:
            logger.warning(f"Failed to query semantic memory: {e}")
            return {"entities": [], "relationships": [], "causal_paths": []}

    async def _get_causal_paths(
        self,
        treatment_var: str,
        outcome_var: str,
    ) -> List[Dict[str, Any]]:
        """Get existing causal paths between treatment and outcome variables."""
        if not self.semantic_memory:
            return []

        try:
            # Query for causal path nodes
            query = """
            MATCH (t:Variable {name: $treatment})-[r:CAUSES*1..3]->(o:Variable {name: $outcome})
            RETURN t, r, o
            LIMIT 10
            """

            results = self.semantic_memory.query(
                query,
                {"treatment": treatment_var, "outcome": outcome_var},
            )

            return cast(List[Dict[str, Any]], results)
        except Exception as e:
            logger.warning(f"Failed to get causal paths: {e}")
            return []

    async def _get_kpi_causal_paths(
        self,
        kpi_name: str,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get causal paths impacting a KPI from semantic memory."""
        if not self.semantic_memory:
            return []

        try:
            paths = self.semantic_memory.find_causal_paths_for_kpi(
                kpi_name=kpi_name,
                min_confidence=min_confidence,
            )
            return cast(List[Dict[str, Any]], paths[:10])  # Limit to top 10 paths
        except Exception as e:
            logger.warning(f"Failed to get KPI causal paths: {e}")
            return []

    # =========================================================================
    # CAUSAL ANALYSIS CACHING (Working Memory)
    # =========================================================================

    async def cache_causal_analysis(
        self,
        session_id: str,
        analysis_result: Dict[str, Any],
    ) -> bool:
        """
        Cache causal analysis result in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            analysis_result: Causal analysis output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"causal_impact:cache:{session_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(analysis_result, default=str),
            )

            logger.debug(f"Cached causal analysis for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache causal analysis: {e}")
            return False

    async def get_cached_causal_analysis(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached causal analysis from working memory.

        Args:
            session_id: Session identifier

        Returns:
            Cached analysis or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"causal_impact:cache:{session_id}"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached causal analysis: {e}")
            return None

    # =========================================================================
    # CAUSAL ANALYSIS STORAGE (Episodic Memory)
    # =========================================================================

    async def store_causal_analysis(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store causal analysis in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            result: Causal analysis output to store
            state: Causal impact state with analysis details
            brand: Optional brand context
            region: Optional region context

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                E2IEntityReferences,
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Build description for embedding
            treatment_var = state.get("treatment_var", "unknown")
            outcome_var = state.get("outcome_var", "unknown")
            ate = result.get("ate_estimate", 0)
            confidence = result.get("confidence", 0)
            refutation = result.get("refutation_passed", False)

            description = (
                f"Causal analysis: {treatment_var} -> {outcome_var}, "
                f"ATE={ate:.3f}, confidence={confidence:.2f}, "
                f"refutation_passed={refutation}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="causal_analysis_completed",
                event_subtype="ate_estimation",
                description=description,
                raw_content={
                    "treatment_var": treatment_var,
                    "outcome_var": outcome_var,
                    "confounders": state.get("confounders", []),
                    "ate_estimate": ate,
                    "confidence_interval": result.get("confidence_interval"),
                    "refutation_passed": refutation,
                    "effect_size": result.get("effect_size"),
                    "model_used": result.get("model_used"),
                    "executive_summary": result.get("executive_summary", "")[:500],
                },
                entities=None,
                outcome_type="causal_analysis_delivered",
                agent_name="causal_impact",
                importance_score=0.85,  # Causal analyses are high value
                e2i_refs=E2IEntityReferences(
                    brand=brand,
                    region=region,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{state.get('query', '')} {description}",
                session_id=session_id,
            )

            logger.info(f"Stored causal analysis in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store causal analysis in episodic memory: {e}")
            return None

    # =========================================================================
    # CAUSAL PATH STORAGE (Semantic Memory)
    # =========================================================================

    async def store_causal_path(
        self,
        treatment_var: str,
        outcome_var: str,
        confounders: List[str],
        ate_estimate: float,
        confidence: float,
        refutation_passed: bool,
        effect_size: str,
    ) -> bool:
        """
        Store discovered causal path in semantic memory.

        This contributes to the knowledge graph for future causal queries
        and RAG retrieval.

        Args:
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            confounders: List of confounding variables
            ate_estimate: Average treatment effect
            confidence: Analysis confidence
            refutation_passed: Whether refutation tests passed
            effect_size: Effect size category

        Returns:
            True if successfully stored
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available for causal path storage")
            return False

        try:
            # Create or update treatment variable node
            self.semantic_memory.add_e2i_entity(
                entity_type="Variable",
                entity_id=f"var:{treatment_var}",
                properties={
                    "name": treatment_var,
                    "role": "treatment",
                    "agent": "causal_impact",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create or update outcome variable node
            self.semantic_memory.add_e2i_entity(
                entity_type="Variable",
                entity_id=f"var:{outcome_var}",
                properties={
                    "name": outcome_var,
                    "role": "outcome",
                    "agent": "causal_impact",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create causal relationship
            self.semantic_memory.add_relationship(
                from_entity_id=f"var:{treatment_var}",
                to_entity_id=f"var:{outcome_var}",
                relationship_type="CAUSES",
                properties={
                    "ate_estimate": ate_estimate,
                    "confidence": confidence,
                    "effect_size": effect_size,
                    "refutation_passed": refutation_passed,
                    "confounders": json.dumps(confounders),
                    "agent": "causal_impact",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(f"Stored causal path: {treatment_var} -> {outcome_var}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store causal path: {e}")
            return False

    # =========================================================================
    # PRIOR CAUSAL ANALYSES (For DSPy Training)
    # =========================================================================

    async def get_prior_analyses(
        self,
        treatment_var: Optional[str] = None,
        outcome_var: Optional[str] = None,
        min_confidence: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get prior causal analyses for DSPy training signals.

        Args:
            treatment_var: Optional filter by treatment variable
            outcome_var: Optional filter by outcome variable
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return

        Returns:
            List of prior causal analyses
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"causal analysis {treatment_var or ''} {outcome_var or ''}"

            filters = EpisodicSearchFilters(
                event_type="causal_analysis_completed",
                agent_name="causal_impact",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit * 2,  # Fetch more to filter
                min_similarity=0.5,
                include_entity_context=False,
            )

            # Filter by confidence
            filtered = [
                r
                for r in results
                if r.get("raw_content", {}).get("confidence", 0) >= min_confidence
            ]

            return filtered[:limit]
        except Exception as e:
            logger.warning(f"Failed to get prior analyses: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[CausalImpactMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute causal analysis results to CognitiveRAG's memory systems.

    This is the primary interface for storing causal impact
    results in the tri-memory architecture.

    Args:
        result: CausalImpactOutput dictionary
        state: CausalImpactState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (uses state value if not provided)
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if analysis stored, 0 otherwise
        - semantic_stored: 1 if causal path stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_causal_impact_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if analysis failed
    if result.get("status") == "failed":
        logger.info("Skipping memory storage for failed analysis")
        return counts

    # 1. Cache in working memory
    cached = await memory_hooks.cache_causal_analysis(session_id, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_causal_analysis(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand or state.get("brand"),
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store causal path in semantic memory (only if refutation passed)
    treatment_var = state.get("treatment_var")
    outcome_var = state.get("outcome_var")
    confounders = state.get("confounders", [])

    if treatment_var and outcome_var and result.get("refutation_passed"):
        stored = await memory_hooks.store_causal_path(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            confounders=confounders,
            ate_estimate=result.get("ate_estimate", 0),
            confidence=result.get("confidence", 0),
            refutation_passed=True,
            effect_size=result.get("effect_size", "unknown"),
        )
        if stored:
            counts["semantic_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"semantic={counts['semantic_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[CausalImpactMemoryHooks] = None


def get_causal_impact_memory_hooks() -> CausalImpactMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = CausalImpactMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
