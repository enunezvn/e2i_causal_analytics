"""
Health Score Agent Memory Hooks
================================

Memory integration hooks for the Health Score agent's memory architecture.

The Health Score agent uses these hooks to:
1. Retrieve context from working memory (Redis - cached health checks)
2. Search episodic memory (Supabase - historical health trends)
3. Cache health check results for fast retrieval

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
class HealthCheckContext:
    """Context retrieved from memory systems for health checks."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    cached_health: Optional[Dict[str, Any]] = None
    historical_trends: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HealthRecord:
    """Record of a health check for storage."""

    session_id: str
    overall_score: float
    health_grade: str
    component_score: float
    model_score: float
    pipeline_score: float
    agent_score: float
    critical_issues_count: int
    warnings_count: int
    check_scope: str
    total_latency_ms: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class HealthScoreMemoryHooks:
    """
    Memory integration hooks for the Health Score agent.

    Provides methods to:
    - Retrieve cached health checks from working memory
    - Cache health check results with short TTL (for dashboard refresh)
    - Store significant health events in episodic memory
    - Track health trends over time
    """

    # Cache TTL in seconds (5 minutes for quick checks)
    QUICK_CACHE_TTL = 300
    # Cache TTL for full checks (15 minutes)
    FULL_CACHE_TTL = 900

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
        check_scope: str = "full",
        include_history: bool = False,
    ) -> HealthCheckContext:
        """
        Retrieve context from working and episodic memory.

        Args:
            session_id: Session identifier for working memory lookup
            check_scope: Scope of health check (quick, full, models, etc.)
            include_history: Whether to include historical trends

        Returns:
            HealthCheckContext with data from memory systems
        """
        context = HealthCheckContext(session_id=session_id)

        # 1. Get working memory (session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Check for cached health check
        context.cached_health = await self._get_cached_health(check_scope)

        # 3. Get historical trends if requested
        if include_history:
            context.historical_trends = await self._get_health_trends()

        logger.debug(
            f"Retrieved health context for session {session_id}: "
            f"cached={context.cached_health is not None}, "
            f"history={len(context.historical_trends)}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 5,
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

    async def _get_cached_health(
        self,
        check_scope: str,
    ) -> Optional[Dict[str, Any]]:
        """Get recently cached health check result."""
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"health_score:cache:{check_scope}"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached health: {e}")
            return None

    async def _get_health_trends(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get historical health check trends."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = "system health check score grade"

            filters = EpisodicSearchFilters(
                event_type="health_check_completed",
                agent_name="health_score",
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
            logger.warning(f"Failed to get health trends: {e}")
            return []

    # =========================================================================
    # HEALTH CHECK CACHING (Working Memory)
    # =========================================================================

    async def cache_health_check(
        self,
        check_scope: str,
        health_result: Dict[str, Any],
    ) -> bool:
        """
        Cache health check result in working memory.

        Args:
            check_scope: Scope of health check
            health_result: Health check output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"health_score:cache:{check_scope}"

            # Use appropriate TTL based on scope
            ttl = self.QUICK_CACHE_TTL if check_scope == "quick" else self.FULL_CACHE_TTL

            await redis.setex(
                cache_key,
                ttl,
                json.dumps(health_result, default=str),
            )

            logger.debug(f"Cached health check for scope {check_scope}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache health check: {e}")
            return False

    async def invalidate_cache(
        self,
        check_scope: Optional[str] = None,
    ) -> bool:
        """
        Invalidate cached health checks.

        Args:
            check_scope: Specific scope to invalidate, or None for all

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            if check_scope:
                cache_key = f"health_score:cache:{check_scope}"
                await redis.delete(cache_key)
            else:
                # Invalidate all scopes
                for scope in ["quick", "full", "models", "pipelines", "agents"]:
                    cache_key = f"health_score:cache:{scope}"
                    await redis.delete(cache_key)

            return True
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
            return False

    # =========================================================================
    # HEALTH EVENT STORAGE (Episodic Memory)
    # =========================================================================

    async def store_health_check(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store significant health check in episodic memory.

        Only stores health checks with:
        - Critical issues
        - Grade changes
        - Significant score drops (>10%)

        Args:
            session_id: Session identifier
            result: Health check output
            state: Health score state

        Returns:
            Memory ID if stored, None otherwise
        """
        # Only store significant events
        if not self._is_significant_health_event(result):
            logger.debug("Health check not significant enough to store")
            return None

        try:
            from src.memory.episodic_memory import (
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Extract key fields
            overall_score = result.get("overall_health_score", 0)
            health_grade = result.get("health_grade", "?")
            critical_issues = result.get("critical_issues", [])
            check_scope = state.get("check_scope", "full")

            # Build description
            issues_str = f", {len(critical_issues)} critical issues" if critical_issues else ""
            description = (
                f"System health check ({check_scope}): "
                f"Grade {health_grade}, Score {overall_score:.1f}/100{issues_str}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="health_check_completed",
                event_subtype="significant_health_event",
                description=description,
                raw_content={
                    "overall_score": overall_score,
                    "health_grade": health_grade,
                    "component_score": result.get("component_health_score", 0),
                    "model_score": result.get("model_health_score", 0),
                    "pipeline_score": result.get("pipeline_health_score", 0),
                    "agent_score": result.get("agent_health_score", 0),
                    "critical_issues": critical_issues[:5],  # Limit size
                    "warnings_count": len(result.get("warnings", [])),
                    "check_scope": check_scope,
                    "total_latency_ms": result.get("total_latency_ms", 0),
                },
                entities=None,
                outcome_type="health_assessment_delivered",
                agent_name="health_score",
                importance_score=self._calculate_importance(result),
                e2i_refs=None,
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{state.get('query', '')} {description}",
                session_id=session_id,
            )

            logger.info(f"Stored health check in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store health check in episodic memory: {e}")
            return None

    def _is_significant_health_event(
        self,
        result: Dict[str, Any],
    ) -> bool:
        """Determine if health check is significant enough to store."""
        # Critical issues always significant
        if result.get("critical_issues"):
            return True

        # Failing grades are significant
        grade = result.get("health_grade", "A")
        if grade in ("D", "F"):
            return True

        # Low scores are significant
        score = result.get("overall_health_score", 100)
        if score < 70:
            return True

        return False

    def _calculate_importance(
        self,
        result: Dict[str, Any],
    ) -> float:
        """Calculate importance score for episodic memory."""
        score = float(result.get("overall_health_score", 100))
        critical_count = len(result.get("critical_issues", []))

        # Base importance from score (lower score = higher importance)
        base_importance = max(0.3, 1.0 - (score / 100))

        # Boost for critical issues
        issue_boost = min(0.3, critical_count * 0.1)

        return float(min(1.0, base_importance + issue_boost))

    # =========================================================================
    # TREND ANALYSIS (For DSPy Recipients)
    # =========================================================================

    async def get_score_history(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get historical health scores for trend analysis.

        Used by recipients to analyze health trends
        and inform recommendations.

        Args:
            hours: Time window in hours
            limit: Maximum results to return

        Returns:
            List of historical health scores
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = "health check overall score trend"

            filters = EpisodicSearchFilters(
                event_type="health_check_completed",
                agent_name="health_score",
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
            logger.warning(f"Failed to get score history: {e}")
            return []

    async def get_component_issues(
        self,
        component: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get historical issues for a specific component.

        Args:
            component: Component name (database, model, pipeline, etc.)
            limit: Maximum results to return

        Returns:
            List of historical issues for the component
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"health issue {component} critical warning"

            filters = EpisodicSearchFilters(
                event_type="health_check_completed",
                agent_name="health_score",
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
            logger.warning(f"Failed to get component issues: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[HealthScoreMemoryHooks] = None,
    session_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute health check results to CognitiveRAG's memory systems.

    This is the primary interface for storing health score
    results in the memory architecture.

    Args:
        result: HealthScoreOutput dictionary
        state: HealthScoreState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if check stored (significant events only), 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_health_score_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if check failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed health check")
        return counts

    check_scope = state.get("check_scope", "full")

    # 1. Always cache in working memory
    cached = await memory_hooks.cache_health_check(check_scope, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory (only significant events)
    memory_id = await memory_hooks.store_health_check(
        session_id=session_id,
        result=result,
        state=state,
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

_memory_hooks: Optional[HealthScoreMemoryHooks] = None


def get_health_score_memory_hooks() -> HealthScoreMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = HealthScoreMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
