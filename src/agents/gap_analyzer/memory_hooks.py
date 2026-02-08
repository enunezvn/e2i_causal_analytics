"""
Gap Analyzer Agent Memory Hooks
===============================

Memory integration hooks for the Gap Analyzer agent's memory architecture.

The Gap Analyzer uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past gap analyses, ROI estimates)
3. Store gap analysis results and ROI calculations for future retrieval

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
class GapAnalysisContext:
    """Context retrieved from memory systems for gap analysis."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GapAnalysisRecord:
    """Record of a gap analysis for storage in episodic memory."""

    session_id: str
    query: str
    brand: str
    metrics: List[str]
    segments: List[str]
    gaps_count: int
    total_addressable_value: float
    quick_wins_count: int
    confidence: float
    executive_summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpportunityRecord:
    """Record of a prioritized opportunity for storage."""

    opportunity_id: str
    brand: str
    segment: str
    segment_value: str
    metric: str
    gap_size: float
    expected_roi: float
    risk_adjusted_roi: float
    recommended_action: str
    difficulty: str
    time_to_impact: str


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class GapAnalyzerMemoryHooks:
    """
    Memory integration hooks for the Gap Analyzer agent.

    Provides methods to:
    - Retrieve context from working and episodic memory
    - Cache gap analysis results in working memory (24h TTL)
    - Store gap analyses in episodic memory for future retrieval
    - Search for similar past analyses to inform ROI estimates
    """

    # Cache TTL in seconds (24 hours)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None

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

    # =========================================================================
    # CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        query: str,
        brand: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
        max_episodic_results: int = 5,
    ) -> GapAnalysisContext:
        """
        Retrieve context from working and episodic memory.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            brand: Optional brand for filtering
            metrics: Optional metrics for filtering
            segments: Optional segments for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            GapAnalysisContext with data from memory systems
        """
        context = GapAnalysisContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past gap analyses)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            brand=brand,
            metrics=metrics,
            segments=segments,
            limit=max_episodic_results,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}"
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
        brand: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past gap analyses."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="gap_analysis_completed",
                agent_name="gap_analyzer",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=False,
            )

            # Further filter by brand/metrics/segments if specified
            if brand or metrics or segments:
                filtered_results = []
                for result in results:
                    content = result.get("raw_content", {})

                    # Filter by brand
                    if brand and content.get("brand") != brand:
                        continue

                    # Filter by metrics (any overlap)
                    if metrics:
                        result_metrics = set(content.get("metrics", []))
                        if not result_metrics.intersection(set(metrics)):
                            continue

                    # Filter by segments (any overlap)
                    if segments:
                        result_segments = set(content.get("segments", []))
                        if not result_segments.intersection(set(segments)):
                            continue

                    filtered_results.append(result)
                return filtered_results

            return results
        except Exception as e:
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    # =========================================================================
    # GAP ANALYSIS CACHING (Working Memory)
    # =========================================================================

    async def cache_gap_analysis(
        self,
        session_id: str,
        analysis_result: Dict[str, Any],
    ) -> bool:
        """
        Cache gap analysis result in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            analysis_result: Gap analysis output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"gap_analyzer:cache:{session_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(analysis_result, default=str),
            )

            logger.debug(f"Cached gap analysis for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache gap analysis: {e}")
            return False

    async def get_cached_gap_analysis(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached gap analysis from working memory.

        Args:
            session_id: Session identifier

        Returns:
            Cached analysis or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"gap_analyzer:cache:{session_id}"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached gap analysis: {e}")
            return None

    # =========================================================================
    # GAP ANALYSIS STORAGE (Episodic Memory)
    # =========================================================================

    async def store_gap_analysis(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store gap analysis in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            result: Gap analysis output to store
            state: Gap analyzer state with analysis details
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
            brand = state.get("brand", "unknown")
            metrics = state.get("metrics", [])
            segments = state.get("segments", [])
            gaps_count = len(result.get("prioritized_opportunities", []))
            total_value = result.get("total_addressable_value", 0)

            description = (
                f"Gap analysis for {brand}: "
                f"metrics={','.join(metrics[:3])}, "
                f"segments={','.join(segments[:3])}, "
                f"gaps_found={gaps_count}, "
                f"total_addressable_value=${total_value:,.0f}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="gap_analysis_completed",
                event_subtype="roi_prioritization",
                description=description,
                raw_content={
                    "brand": brand,
                    "metrics": metrics,
                    "segments": segments,
                    "gaps_count": gaps_count,
                    "total_addressable_value": total_value,
                    "quick_wins_count": len(result.get("quick_wins", [])),
                    "strategic_bets_count": len(result.get("strategic_bets", [])),
                    "confidence": result.get("confidence", 0),
                    "executive_summary": result.get("executive_summary", "")[:500],
                    "key_insights": result.get("key_insights", [])[:5],
                },
                entities=None,
                outcome_type="gap_analysis_delivered",
                agent_name="gap_analyzer",
                importance_score=0.8,  # Gap analyses are high value
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

            logger.info(f"Stored gap analysis in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store gap analysis in episodic memory: {e}")
            return None

    # =========================================================================
    # HISTORICAL ROI DATA (For DSPy Training)
    # =========================================================================

    async def get_historical_roi_data(
        self,
        brand: Optional[str] = None,
        metric: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get historical ROI data for DSPy training signals.

        Used to collect training signals for ROI estimation optimization.

        Args:
            brand: Optional filter by brand
            metric: Optional filter by metric
            limit: Maximum results to return

        Returns:
            List of historical gap analyses with ROI outcomes
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"gap analysis ROI {brand or ''} {metric or ''}"

            filters = EpisodicSearchFilters(
                event_type="gap_analysis_completed",
                agent_name="gap_analyzer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.4,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get historical ROI data: {e}")
            return []

    async def get_opportunity_benchmarks(
        self,
        segment: str,
        metric: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark data for similar opportunities.

        Used to calibrate ROI estimates based on past performance.

        Args:
            segment: Segment type (e.g., "region", "specialty")
            metric: Metric being analyzed
            limit: Maximum results to return

        Returns:
            List of similar past opportunities with outcomes
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"opportunity {segment} {metric} ROI"

            filters = EpisodicSearchFilters(
                event_type="gap_analysis_completed",
                agent_name="gap_analyzer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get opportunity benchmarks: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[GapAnalyzerMemoryHooks] = None,
    session_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute gap analysis results to CognitiveRAG's memory systems.

    This is the primary interface for storing gap analyzer
    results in the memory architecture.

    Args:
        result: GapAnalyzerOutput dictionary
        state: GapAnalyzerState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if analysis stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_gap_analyzer_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if analysis failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed analysis")
        return counts

    # 1. Cache in working memory
    cached = await memory_hooks.cache_gap_analysis(session_id, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_gap_analysis(
        session_id=session_id,
        result=result,
        state=state,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[GapAnalyzerMemoryHooks] = None


def get_gap_analyzer_memory_hooks() -> GapAnalyzerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = GapAnalyzerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
