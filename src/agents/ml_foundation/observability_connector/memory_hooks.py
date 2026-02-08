"""
Observability Connector Agent Memory Hooks
==========================================

Memory integration hooks for the Observability Connector agent's tri-memory architecture.

The Observability Connector agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past observability events)
3. Query semantic memory (FalkorDB - system health patterns, anomalies)
4. Store observability metrics for future retrieval and RAG

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
class ObservabilityContext:
    """Context retrieved from all memory systems for observability."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ObservabilityRecord:
    """Record of observability metrics for storage in episodic memory."""

    session_id: str
    time_window: str
    overall_success_rate: float
    overall_p95_latency_ms: float
    quality_score: Optional[float]
    error_rate_by_agent: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ObservabilityConnectorMemoryHooks:
    """
    Memory integration hooks for the Observability Connector agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache observability metrics in working memory (24h TTL)
    - Store observability events in episodic memory for future retrieval
    - Store system health patterns in semantic memory for knowledge graph
    """

    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        """Initialize memory hooks with lazy-loaded clients."""
        self._working_memory = None
        self._semantic_memory = None

    @property
    def working_memory(self):
        """Lazy-load Redis working memory."""
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
        """Lazy-load FalkorDB semantic memory."""
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
        time_window: str = "24h",
        agent_name_filter: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> ObservabilityContext:
        """Retrieve context from all three memory systems."""
        context = ObservabilityContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            time_window=time_window,
            agent_name_filter=agent_name_filter,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            agent_name_filter=agent_name_filter,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_anomalies={len(context.semantic_context.get('anomalies', []))}"
        )

        return context

    async def _get_working_memory_context(
        self, session_id: str, limit: int = 10
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
        time_window: str,
        agent_name_filter: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar observability events."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"observability metrics {time_window} {agent_name_filter or ''}"

            filters = EpisodicSearchFilters(
                event_type="observability_metrics_collected",
                agent_name="observability_connector",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=True,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get episodic context: {e}")
            return []

    async def _get_semantic_context(
        self,
        agent_name_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get semantic memory context for system health patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context: Dict[str, Any] = {
                "health_snapshots": [],
                "anomalies": [],
                "agent_patterns": [],
            }

            # Query recent health snapshots
            snapshots = self.semantic_memory.query(
                "MATCH (h:HealthSnapshot) RETURN h ORDER BY h.timestamp DESC LIMIT 10"
            )
            context["health_snapshots"] = snapshots

            # Query anomalies
            anomalies = self.semantic_memory.query(
                "MATCH (a:Anomaly) RETURN a ORDER BY a.timestamp DESC LIMIT 10"
            )
            context["anomalies"] = anomalies

            # Query agent-specific patterns
            if agent_name_filter:
                patterns = self.semantic_memory.query(
                    f"MATCH (p:AgentPattern {{agent_name: '{agent_name_filter}'}}) "
                    f"RETURN p ORDER BY p.timestamp DESC LIMIT 5"
                )
                context["agent_patterns"] = patterns

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_metrics(
        self,
        session_id: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """Cache observability metrics in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"observability_connector:metrics:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(metrics),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached metrics for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache metrics: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_observability_event(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store observability event in episodic memory."""
        try:
            from src.memory.episodic_memory import insert_episodic_memory

            content = {
                "time_window": state.get("time_window"),
                "overall_success_rate": result.get("overall_success_rate"),
                "overall_p95_latency_ms": result.get("overall_p95_latency_ms"),
                "overall_p99_latency_ms": result.get("overall_p99_latency_ms"),
                "quality_score": result.get("quality_score"),
                "total_spans_analyzed": result.get("total_spans_analyzed"),
                "error_rate_by_agent": result.get("error_rate_by_agent", {}),
                "latency_by_agent": result.get("latency_by_agent", {}),
                "status_distribution": result.get("status_distribution", {}),
                "fallback_invocation_rate": result.get("fallback_invocation_rate"),
            }

            success_rate = result.get("overall_success_rate", 0)
            quality_score = result.get("quality_score", 0)
            summary = (
                f"Observability: {state.get('time_window', 'unknown')} window. "
                f"Success rate: {success_rate:.1%}. "
                f"Quality score: {quality_score:.2f}. "
                f"Spans: {result.get('total_spans_analyzed', 0)}."
            )

            memory_id = await insert_episodic_memory(  # type: ignore[call-arg]
                session_id=session_id,
                event_type="observability_metrics_collected",
                agent_name="observability_connector",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored observability event in episodic memory: {memory_id}")
            return str(memory_id) if memory_id else None
        except Exception as e:
            logger.warning(f"Failed to store observability event: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_health_snapshot(
        self,
        time_window: str,
        overall_success_rate: float,
        overall_p95_latency_ms: float,
        quality_score: Optional[float],
        error_rate_by_agent: Dict[str, float],
        anomalies_detected: List[str],
    ) -> bool:
        """Store system health snapshot in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            snapshot_id = f"health:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            # Create health snapshot node
            self.semantic_memory.add_e2i_entity(
                entity_type="HealthSnapshot",
                entity_id=snapshot_id,
                properties={
                    "time_window": time_window,
                    "overall_success_rate": overall_success_rate,
                    "overall_p95_latency_ms": overall_p95_latency_ms,
                    "quality_score": quality_score or 0,
                    "agent": "observability_connector",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Store anomalies if detected
            for anomaly in anomalies_detected[:10]:
                anomaly_id = (
                    f"anomaly:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}:{anomaly[:20]}"
                )
                self.semantic_memory.add_e2i_entity(
                    entity_type="Anomaly",
                    entity_id=anomaly_id,
                    properties={
                        "description": anomaly,
                        "agent": "observability_connector",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.semantic_memory.add_relationship(
                    from_entity_id=snapshot_id,
                    to_entity_id=anomaly_id,
                    relationship_type="DETECTED",
                    properties={"agent": "observability_connector"},
                )

            # Store agent patterns for high-error agents
            for agent_name, error_rate in error_rate_by_agent.items():
                if error_rate > 0.1:  # Store patterns for agents with >10% error rate
                    pattern_id = f"pattern:{agent_name}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    self.semantic_memory.add_e2i_entity(
                        entity_type="AgentPattern",
                        entity_id=pattern_id,
                        properties={
                            "agent_name": agent_name,
                            "error_rate": error_rate,
                            "pattern_type": "high_error_rate",
                            "agent": "observability_connector",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    self.semantic_memory.add_relationship(
                        from_entity_id=snapshot_id,
                        to_entity_id=pattern_id,
                        relationship_type="INCLUDES",
                        properties={"agent": "observability_connector"},
                    )

            logger.info(f"Stored health snapshot: {snapshot_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store health snapshot: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ObservabilityConnectorMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute observability results to memory systems."""
    import uuid

    if memory_hooks is None:
        memory_hooks = get_observability_connector_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if error
    if state.get("error"):
        logger.info("Skipping memory storage due to error")
        return counts

    # 1. Cache in working memory
    metrics = {
        "overall_success_rate": result.get("overall_success_rate"),
        "overall_p95_latency_ms": result.get("overall_p95_latency_ms"),
        "quality_score": result.get("quality_score"),
        "total_spans_analyzed": result.get("total_spans_analyzed"),
    }
    cached = await memory_hooks.cache_metrics(session_id, metrics)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_observability_event(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store health snapshot in semantic memory
    stored = await memory_hooks.store_health_snapshot(
        time_window=state.get("time_window", "24h"),
        overall_success_rate=result.get("overall_success_rate", 0),
        overall_p95_latency_ms=result.get("overall_p95_latency_ms", 0),
        quality_score=result.get("quality_score"),
        error_rate_by_agent=result.get("error_rate_by_agent", {}),
        anomalies_detected=_detect_anomalies(result),
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


def _detect_anomalies(result: Dict[str, Any]) -> List[str]:
    """Detect anomalies from observability results."""
    anomalies = []

    # Check overall success rate
    success_rate = result.get("overall_success_rate", 1.0)
    if success_rate < 0.95:
        anomalies.append(f"Low success rate: {success_rate:.1%}")

    # Check latency
    p95_latency = result.get("overall_p95_latency_ms", 0)
    if p95_latency > 5000:
        anomalies.append(f"High P95 latency: {p95_latency:.0f}ms")

    # Check quality score
    quality_score = result.get("quality_score")
    if quality_score and quality_score < 0.7:
        anomalies.append(f"Low quality score: {quality_score:.2f}")

    # Check agent error rates
    error_rates = result.get("error_rate_by_agent", {})
    for agent, rate in error_rates.items():
        if rate > 0.2:
            anomalies.append(f"High error rate for {agent}: {rate:.1%}")

    return anomalies


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[ObservabilityConnectorMemoryHooks] = None


def get_observability_connector_memory_hooks() -> ObservabilityConnectorMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ObservabilityConnectorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
