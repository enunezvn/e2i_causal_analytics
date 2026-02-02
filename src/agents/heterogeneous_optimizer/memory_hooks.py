"""
Heterogeneous Optimizer Agent Memory Hooks
==========================================

Memory integration hooks for the Heterogeneous Optimizer agent's tri-memory architecture.

The Heterogeneous Optimizer uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past CATE analyses)
3. Query semantic memory (FalkorDB - entity relationships, causal paths)
4. Store CATE analysis results and segment profiles for future retrieval

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CATEAnalysisContext:
    """Context retrieved from all memory systems for CATE analysis."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CATEAnalysisRecord:
    """Record of a CATE analysis for storage in episodic memory."""

    session_id: str
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]
    overall_ate: float
    heterogeneity_score: float
    high_responders_count: int
    low_responders_count: int
    executive_summary: str
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentEffectRecord:
    """Record of a segment effect for storage in semantic memory."""

    segment_id: str
    segment_name: str
    segment_value: str
    responder_type: str  # "high", "low", "average"
    cate_estimate: float
    defining_features: List[Dict[str, Any]]
    recommendation: str
    confidence: float


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class HeterogeneousOptimizerMemoryHooks:
    """
    Memory integration hooks for the Heterogeneous Optimizer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache CATE analysis results in working memory (24h TTL)
    - Store CATE analyses in episodic memory for future retrieval
    - Store segment profiles in semantic memory for knowledge graph
    - Query causal paths and entity relationships from semantic memory
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
    ) -> CATEAnalysisContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            treatment_var: Optional treatment variable for filtering
            outcome_var: Optional outcome variable for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            CATEAnalysisContext with data from all memory systems
        """
        context = CATEAnalysisContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past CATE analyses)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (causal paths, entity relationships)
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
            return messages
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
        """Search episodic memory for similar past CATE analyses."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="cate_analysis_completed",
                agent_name="heterogeneous_optimizer",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=False,
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
        """Query semantic memory for causal paths and entity relationships."""
        if not self.semantic_memory:
            return {"entities": [], "relationships": [], "causal_paths": []}

        try:
            result = {
                "entities": [],
                "relationships": [],
                "causal_paths": [],
                "segment_profiles": [],
            }

            # Get causal paths for treatment-outcome relationship
            if treatment_var and outcome_var:
                causal_paths = await self._get_causal_paths(treatment_var, outcome_var)
                result["causal_paths"] = causal_paths

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
        """Get causal paths between treatment and outcome variables."""
        if not self.semantic_memory:
            return []

        try:
            paths = self.semantic_memory.find_causal_paths_for_kpi(
                kpi_name=outcome_var,
                min_confidence=0.5,
            )

            # Filter paths involving the treatment variable
            filtered = [p for p in paths if treatment_var.lower() in str(p).lower()]

            return filtered
        except Exception as e:
            logger.warning(f"Failed to get causal paths: {e}")
            return []

    # =========================================================================
    # CATE ANALYSIS CACHING (Working Memory)
    # =========================================================================

    async def cache_cate_analysis(
        self,
        session_id: str,
        analysis_result: Dict[str, Any],
    ) -> bool:
        """
        Cache CATE analysis result in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            analysis_result: CATE analysis output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"heterogeneous_optimizer:cache:{session_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(analysis_result),
            )

            logger.debug(f"Cached CATE analysis for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache CATE analysis: {e}")
            return False

    async def get_cached_cate_analysis(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached CATE analysis from working memory.

        Args:
            session_id: Session identifier

        Returns:
            Cached analysis or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"heterogeneous_optimizer:cache:{session_id}"

            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached CATE analysis: {e}")
            return None

    # =========================================================================
    # CATE ANALYSIS STORAGE (Episodic Memory)
    # =========================================================================

    async def store_cate_analysis(
        self,
        session_id: str,
        analysis_result: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store CATE analysis in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            analysis_result: CATE analysis output to store
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
            treatment_var = analysis_result.get("treatment_var", "unknown")
            outcome_var = analysis_result.get("outcome_var", "unknown")
            heterogeneity = analysis_result.get("heterogeneity_score", 0)
            high_count = len(analysis_result.get("high_responders", []))
            low_count = len(analysis_result.get("low_responders", []))

            description = (
                f"CATE analysis: {treatment_var} -> {outcome_var}, "
                f"heterogeneity={heterogeneity:.2f}, "
                f"high_responders={high_count}, low_responders={low_count}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="cate_analysis_completed",
                event_subtype="segment_analysis",
                description=description,
                raw_content=analysis_result,
                entities=None,
                outcome_type="cate_analysis_delivered",
                agent_name="heterogeneous_optimizer",
                importance_score=0.8,  # CATE analyses are high value
                e2i_refs=E2IEntityReferences(
                    brand=brand,
                    region=region,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=description,
                session_id=session_id,
            )

            logger.info(f"Stored CATE analysis in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store CATE analysis in episodic memory: {e}")
            return None

    # =========================================================================
    # SEGMENT PROFILE STORAGE (Semantic Memory)
    # =========================================================================

    async def store_segment_profiles(
        self,
        high_responders: List[Dict[str, Any]],
        low_responders: List[Dict[str, Any]],
        treatment_var: str,
        outcome_var: str,
    ) -> int:
        """
        Store segment profiles in semantic memory for knowledge graph.

        This contributes CATE analysis results to the Cognitive RAG's
        semantic memory for future retrieval and causal reasoning.

        Args:
            high_responders: List of high responder segment profiles
            low_responders: List of low responder segment profiles
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name

        Returns:
            Number of profiles stored successfully
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available for segment profile storage")
            return 0

        stored_count = 0

        try:
            # Store top 5 high responders
            for profile in high_responders[:5]:
                success = await self._store_segment_effect(
                    profile=profile,
                    responder_type="high",
                    treatment_var=treatment_var,
                    outcome_var=outcome_var,
                )
                if success:
                    stored_count += 1

            # Store top 5 low responders
            for profile in low_responders[:5]:
                success = await self._store_segment_effect(
                    profile=profile,
                    responder_type="low",
                    treatment_var=treatment_var,
                    outcome_var=outcome_var,
                )
                if success:
                    stored_count += 1

            logger.info(f"Stored {stored_count} segment profiles in semantic memory")
            return stored_count
        except Exception as e:
            logger.warning(f"Failed to store segment profiles: {e}")
            return stored_count

    async def _store_segment_effect(
        self,
        profile: Dict[str, Any],
        responder_type: str,
        treatment_var: str,
        outcome_var: str,
    ) -> bool:
        """Store a single segment effect in semantic memory."""
        if not self.semantic_memory:
            return False

        try:
            segment_id = profile.get("segment_id", "unknown")
            cate_estimate = profile.get("cate_estimate", 0.0)
            defining_features = profile.get("defining_features", [])
            recommendation = profile.get("recommendation", "")

            # Build node properties
            node_properties = {
                "segment_id": segment_id,
                "responder_type": responder_type,
                "cate_estimate": cate_estimate,
                "treatment_var": treatment_var,
                "outcome_var": outcome_var,
                "defining_features": json.dumps(defining_features),
                "recommendation": recommendation,
                "agent": "heterogeneous_optimizer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store as segment effect entity in graph
            # This creates a node that can be linked to KPIs, treatments, etc.
            self.semantic_memory.add_e2i_entity(
                entity_type="SegmentEffect",
                entity_id=f"segment_effect:{segment_id}:{treatment_var}:{outcome_var}",
                properties=node_properties,
            )

            return True
        except Exception as e:
            logger.warning(f"Failed to store segment effect: {e}")
            return False

    # =========================================================================
    # CAUSAL PATH QUERIES
    # =========================================================================

    async def get_causal_context(
        self,
        kpi_name: str,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Get causal paths impacting a KPI from semantic memory.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx")
            min_confidence: Minimum confidence threshold

        Returns:
            List of causal paths with effect sizes
        """
        if not self.semantic_memory:
            return []

        try:
            paths = self.semantic_memory.find_causal_paths_for_kpi(
                kpi_name=kpi_name,
                min_confidence=min_confidence,
            )
            return paths
        except Exception as e:
            logger.warning(f"Failed to get causal context: {e}")
            return []

    async def get_prior_segment_effects(
        self,
        treatment_var: str,
        outcome_var: str,
    ) -> List[Dict[str, Any]]:
        """
        Get prior segment effects for a treatment-outcome pair.

        Useful for priming CATE analysis with historical segment data.

        Args:
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name

        Returns:
            List of prior segment effect records
        """
        if not self.semantic_memory:
            return []

        try:
            # Query for segment effect nodes with matching treatment/outcome
            query = """
            MATCH (s:SegmentEffect)
            WHERE s.treatment_var = $treatment_var AND s.outcome_var = $outcome_var
            RETURN s
            ORDER BY s.timestamp DESC
            LIMIT 20
            """

            results = self.semantic_memory.query(
                query,
                {"treatment_var": treatment_var, "outcome_var": outcome_var},
            )

            return [
                {
                    "segment_id": r.get("segment_id"),
                    "responder_type": r.get("responder_type"),
                    "cate_estimate": r.get("cate_estimate"),
                    "defining_features": json.loads(r.get("defining_features", "[]")),
                    "recommendation": r.get("recommendation"),
                    "timestamp": r.get("timestamp"),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Failed to get prior segment effects: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[HeterogeneousOptimizerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute CATE analysis results to CognitiveRAG's memory systems.

    This is the primary interface for storing heterogeneous optimizer
    results in the tri-memory architecture.

    Args:
        result: HeterogeneousOptimizerOutput dictionary
        state: HeterogeneousOptimizerState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if analysis stored, 0 otherwise
        - semantic_stored: Number of segment profiles stored
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_heterogeneous_optimizer_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

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
    cached = await memory_hooks.cache_cate_analysis(session_id, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_cate_analysis(
        session_id=session_id,
        analysis_result=result,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store segment profiles in semantic memory
    high_responders = result.get("high_responders", [])
    low_responders = result.get("low_responders", [])
    treatment_var = state.get("treatment_var", "unknown")
    outcome_var = state.get("outcome_var", "unknown")

    if high_responders or low_responders:
        stored = await memory_hooks.store_segment_profiles(
            high_responders=high_responders,
            low_responders=low_responders,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        counts["semantic_stored"] = stored

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

_memory_hooks: Optional[HeterogeneousOptimizerMemoryHooks] = None


def get_heterogeneous_optimizer_memory_hooks() -> HeterogeneousOptimizerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = HeterogeneousOptimizerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
