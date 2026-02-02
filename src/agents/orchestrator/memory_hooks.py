"""
Orchestrator Agent Memory Hooks
===============================

Memory integration hooks for the Orchestrator agent's tri-memory architecture.

The Orchestrator uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - conversation history, past interactions)
3. Query semantic memory (FalkorDB - entity relationships, causal paths)
4. Store orchestration results and routing decisions for future retrieval

As the Hub agent, the Orchestrator coordinates memory access across all agents
and provides context enrichment for downstream agent execution.

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
class OrchestrationContext:
    """Context retrieved from all memory systems for orchestration."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrchestrationRecord:
    """Record of an orchestration for storage in episodic memory."""

    session_id: str
    query_id: str
    query: str
    intent: str
    agents_dispatched: List[str]
    response_summary: str
    confidence: float
    total_latency_ms: int
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecisionRecord:
    """Record of a routing decision for learning optimization."""

    query_pattern: str
    intent_classified: str
    agents_selected: List[str]
    execution_mode: str  # "sequential" or "parallel"
    success: bool
    latency_ms: int
    confidence: float


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class OrchestratorMemoryHooks:
    """
    Memory integration hooks for the Orchestrator agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache orchestration results in working memory (24h TTL)
    - Store orchestration history in episodic memory
    - Query entity relationships and causal paths from semantic memory
    - Track routing decisions for optimization

    As the Hub agent, the Orchestrator has READ access to all memory types
    and WRITE access to Working and Episodic memory.
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
    # CONTEXT RETRIEVAL (Hub Coordination)
    # =========================================================================

    async def get_context(
        self,
        session_id: str,
        query: str,
        entities: Optional[Dict[str, List[str]]] = None,
        max_episodic_results: int = 5,
    ) -> OrchestrationContext:
        """
        Retrieve context from all three memory systems.

        As the Hub agent, the Orchestrator retrieves comprehensive context
        to enrich downstream agent execution.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            entities: Optional extracted entities for semantic lookup
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            OrchestrationContext with data from all memory systems
        """
        context = OrchestrationContext(session_id=session_id)

        # 1. Get working memory (recent session context, conversation history)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past orchestrations)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (entity relationships, causal paths)
        context.semantic_context = await self._get_semantic_context(
            entities=entities,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_entities={len(context.semantic_context.get('entities', []))}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation and context from working memory."""
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
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past orchestrations."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="orchestration_completed",
                agent_name="orchestrator",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=True,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    async def _get_semantic_context(
        self,
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Query semantic memory for entity relationships."""
        if not self.semantic_memory:
            return {"entities": [], "relationships": [], "causal_paths": []}

        try:
            result = {
                "entities": [],
                "relationships": [],
                "causal_paths": [],
            }

            # Get graph statistics for context
            stats = self.semantic_memory.get_graph_stats()
            result["graph_stats"] = stats

            # If entities provided, lookup their relationships
            if entities:
                # Lookup KPI relationships
                kpis = entities.get("kpi", [])
                for kpi in kpis[:3]:  # Limit to top 3 KPIs
                    causal_paths = await self._get_causal_paths_for_kpi(kpi)
                    result["causal_paths"].extend(causal_paths)

                # Lookup brand relationships
                brands = entities.get("brand", [])
                for brand in brands[:2]:  # Limit to top 2 brands
                    brand_context = await self._get_brand_context(brand)
                    result["entities"].extend(brand_context)

            return result
        except Exception as e:
            logger.warning(f"Failed to query semantic memory: {e}")
            return {"entities": [], "relationships": [], "causal_paths": []}

    async def _get_causal_paths_for_kpi(
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
            return paths[:10]  # Limit to top 10 paths
        except Exception as e:
            logger.warning(f"Failed to get causal paths: {e}")
            return []

    async def _get_brand_context(
        self,
        brand_name: str,
    ) -> List[Dict[str, Any]]:
        """Get brand-related entities and relationships."""
        if not self.semantic_memory:
            return []

        try:
            # Query for brand-related entities
            query = """
            MATCH (b:Brand {name: $brand_name})-[r]->(e)
            RETURN e, type(r) as relationship
            LIMIT 20
            """

            results = self.semantic_memory.query(
                query,
                {"brand_name": brand_name},
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get brand context: {e}")
            return []

    # =========================================================================
    # CONVERSATION HISTORY (Working Memory)
    # =========================================================================

    async def store_conversation_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        intent: Optional[str] = None,
        agents_used: Optional[List[str]] = None,
    ) -> bool:
        """
        Store a conversation turn in working memory.

        Args:
            session_id: Session identifier
            query: User query
            response: System response
            intent: Classified intent
            agents_used: Agents dispatched

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.working_memory.add_message(session_id, message)

            response_message = {
                "role": "assistant",
                "content": response,
                "intent": intent,
                "agents_used": agents_used or [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.working_memory.add_message(session_id, response_message)

            logger.debug(f"Stored conversation turn for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {e}")
            return False

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history from working memory.

        Args:
            session_id: Session identifier
            limit: Maximum messages to retrieve

        Returns:
            List of conversation messages
        """
        if not self.working_memory:
            return []

        try:
            messages = await self.working_memory.get_messages(session_id, limit=limit)
            return messages
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
            return []

    # =========================================================================
    # ORCHESTRATION CACHING (Working Memory)
    # =========================================================================

    async def cache_orchestration_result(
        self,
        session_id: str,
        query_id: str,
        result: Dict[str, Any],
    ) -> bool:
        """
        Cache orchestration result in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            query_id: Query identifier
            result: Orchestration output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"orchestrator:cache:{session_id}:{query_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(result, default=str),
            )

            logger.debug(f"Cached orchestration result for query {query_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache orchestration result: {e}")
            return False

    async def get_cached_orchestration(
        self,
        session_id: str,
        query_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached orchestration from working memory.

        Args:
            session_id: Session identifier
            query_id: Query identifier

        Returns:
            Cached result or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"orchestrator:cache:{session_id}:{query_id}"

            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached orchestration: {e}")
            return None

    # =========================================================================
    # ORCHESTRATION HISTORY (Episodic Memory)
    # =========================================================================

    async def store_orchestration(
        self,
        session_id: str,
        result: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store orchestration in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            result: Orchestration output to store
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
            query = result.get("query", "")
            intent = result.get("intent_classified", "unknown")
            agents = result.get("agents_dispatched", [])
            response_snippet = result.get("response_text", "")[:200]

            description = (
                f"Orchestration: intent={intent}, agents={','.join(agents)}, query: {query[:100]}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="orchestration_completed",
                event_subtype="query_response",
                description=description,
                raw_content={
                    "query": query,
                    "intent": intent,
                    "agents_dispatched": agents,
                    "response_summary": response_snippet,
                    "confidence": result.get("response_confidence", 0.0),
                    "latency_ms": result.get("total_latency_ms", 0),
                    "status": result.get("status", "unknown"),
                },
                entities=None,
                outcome_type="response_delivered",
                agent_name="orchestrator",
                importance_score=0.7,
                e2i_refs=E2IEntityReferences(
                    brand=brand,
                    region=region,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{query} {description}",
                session_id=session_id,
            )

            logger.info(f"Stored orchestration in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store orchestration in episodic memory: {e}")
            return None

    # =========================================================================
    # ROUTING DECISION TRACKING (For DSPy Optimization)
    # =========================================================================

    async def track_routing_decision(
        self,
        session_id: str,
        query_pattern: str,
        intent: str,
        agents_selected: List[str],
        execution_mode: str,
        success: bool,
        latency_ms: int,
        confidence: float,
    ) -> bool:
        """
        Track routing decision for DSPy optimization learning.

        The Orchestrator as Hub agent collects routing signals
        for the AgentRoutingSignature DSPy optimization.

        Args:
            session_id: Session identifier
            query_pattern: Query pattern/type
            intent: Classified intent
            agents_selected: Agents chosen for dispatch
            execution_mode: Sequential or parallel
            success: Whether orchestration succeeded
            latency_ms: Total latency
            confidence: Intent confidence

        Returns:
            True if successfully tracked
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            # Store routing decision for batch processing
            decision = {
                "session_id": session_id,
                "query_pattern": query_pattern,
                "intent": intent,
                "agents_selected": agents_selected,
                "execution_mode": execution_mode,
                "success": success,
                "latency_ms": latency_ms,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add to routing decisions list
            await redis.lpush(
                "orchestrator:routing_decisions",
                json.dumps(decision),
            )

            # Trim to keep only last 1000 decisions
            await redis.ltrim("orchestrator:routing_decisions", 0, 999)

            logger.debug(f"Tracked routing decision for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to track routing decision: {e}")
            return False

    async def get_routing_decisions(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent routing decisions for DSPy optimization.

        Args:
            limit: Maximum decisions to retrieve

        Returns:
            List of routing decisions
        """
        if not self.working_memory:
            return []

        try:
            redis = await self.working_memory.get_client()
            decisions_raw = await redis.lrange(
                "orchestrator:routing_decisions",
                0,
                limit - 1,
            )

            decisions = [json.loads(d) for d in decisions_raw]
            return decisions
        except Exception as e:
            logger.warning(f"Failed to get routing decisions: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[OrchestratorMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute orchestration results to CognitiveRAG's memory systems.

    This is the primary interface for storing orchestrator
    results in the tri-memory architecture.

    Args:
        result: OrchestratorOutput dictionary
        state: OrchestratorState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (uses state value if not provided)
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if orchestration stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
        - conversation_stored: 1 if conversation turn stored, 0 otherwise
        - routing_tracked: 1 if routing decision tracked, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_orchestrator_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
        "conversation_stored": 0,
        "routing_tracked": 0,
    }

    # Skip storage if orchestration failed
    status = result.get("status", "failed")
    if status == "failed":
        logger.info("Skipping memory storage for failed orchestration")
        return counts

    query_id = result.get("query_id", "unknown")

    # 1. Cache in working memory
    cached = await memory_hooks.cache_orchestration_result(
        session_id=session_id,
        query_id=query_id,
        result=result,
    )
    if cached:
        counts["working_cached"] = 1

    # 2. Store conversation turn
    query = state.get("query", "")
    response = result.get("response_text", "")
    intent = result.get("intent_classified")
    agents = result.get("agents_dispatched", [])

    if query and response:
        stored = await memory_hooks.store_conversation_turn(
            session_id=session_id,
            query=query,
            response=response,
            intent=intent,
            agents_used=agents,
        )
        if stored:
            counts["conversation_stored"] = 1

    # 3. Store in episodic memory
    # Merge query into result for storage
    result_with_query = {**result, "query": query}
    memory_id = await memory_hooks.store_orchestration(
        session_id=session_id,
        result=result_with_query,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 4. Track routing decision for DSPy optimization
    state.get("dispatch_plan", [])
    execution_mode = "parallel" if state.get("parallel_groups") else "sequential"

    tracked = await memory_hooks.track_routing_decision(
        session_id=session_id,
        query_pattern=query[:50] if query else "",
        intent=intent or "unknown",
        agents_selected=agents,
        execution_mode=execution_mode,
        success=(status == "completed"),
        latency_ms=result.get("total_latency_ms", 0),
        confidence=result.get("intent_confidence", 0.0),
    )
    if tracked:
        counts["routing_tracked"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"working_cached={counts['working_cached']}, "
        f"conversation={counts['conversation_stored']}, "
        f"routing_tracked={counts['routing_tracked']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[OrchestratorMemoryHooks] = None


def get_orchestrator_memory_hooks() -> OrchestratorMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = OrchestratorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
