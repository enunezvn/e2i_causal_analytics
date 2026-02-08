"""
Tool Composer Agent Memory Hooks
================================

Memory integration hooks for the Tool Composer agent's memory architecture.

The Tool Composer uses these hooks to:
1. Retrieve context from working memory (Redis - execution state during composition)
2. Search episodic memory (Supabase - similar past compositions for plan optimization)
3. Store composition patterns and execution plans for procedural learning

As a Hybrid DSPy agent, Tool Composer both generates and consumes training signals
for VisualizationConfigSignature optimization.

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# =============================================================================
# UUID CONVERSION HELPER
# =============================================================================


def _ensure_uuid(value: str) -> str:
    """Convert a non-UUID string to a deterministic UUID v5."""
    if not value:
        return str(_uuid.uuid4())
    try:
        _uuid.UUID(value)
        return value  # Already a valid UUID
    except ValueError:
        return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, value))


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CompositionContext:
    """Context retrieved from memory systems for tool composition."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    procedural_patterns: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CompositionRecord:
    """Record of a composition for storage in episodic memory."""

    session_id: str
    composition_id: str
    query: str
    sub_questions_count: int
    tools_executed: int
    tools_succeeded: int
    total_latency_ms: int
    confidence: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionPattern:
    """Pattern of successful compositions for procedural memory."""

    pattern_id: str
    query_intent: str
    decomposition_strategy: str
    tool_sequence: List[str]
    parallel_groups: List[List[str]]
    success_rate: float
    avg_latency_ms: float


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ToolComposerMemoryHooks:
    """
    Memory integration hooks for the Tool Composer agent.

    Provides methods to:
    - Retrieve context from working and episodic memory
    - Cache composition results in working memory (24h TTL)
    - Store composition history in episodic memory
    - Store successful patterns in procedural memory for optimization
    - Retrieve similar past compositions for plan optimization
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
        max_episodic_results: int = 5,
    ) -> CompositionContext:
        """
        Retrieve context from working, episodic, and procedural memory.

        Args:
            session_id: Session identifier for working memory lookup
            query: Query text for episodic similarity search
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            CompositionContext with data from memory systems
        """
        context = CompositionContext(session_id=session_id)

        # 1. Get working memory (current execution state)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past compositions)
        context.episodic_context = await self._get_episodic_context(
            query=query,
            limit=max_episodic_results,
        )

        # 3. Get procedural memory (learned composition patterns)
        context.procedural_patterns = await self._get_procedural_patterns(query)

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"procedural={len(context.procedural_patterns)}"
        )

        return context

    async def _get_working_memory_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve current session context from working memory."""
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
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past compositions."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="composition_completed",
                agent_name="tool_composer",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit,
                min_similarity=0.65,  # Higher threshold for composition matching
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    async def _get_procedural_patterns(
        self,
        query: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve successful composition patterns from procedural memory."""
        try:
            from src.memory.procedural_memory import find_relevant_procedures_by_text

            # Search for successful composition patterns
            patterns = await find_relevant_procedures_by_text(
                query_text=query,
                procedure_type="tool_composition",
                limit=limit,
            )

            return patterns
        except Exception as e:
            logger.warning(f"Failed to get procedural patterns: {e}")
            return []

    # =========================================================================
    # COMPOSITION CACHING (Working Memory)
    # =========================================================================

    async def cache_composition_result(
        self,
        session_id: str,
        composition_id: str,
        result: Dict[str, Any],
    ) -> bool:
        """
        Cache composition result in working memory with 24h TTL.

        Args:
            session_id: Session identifier
            composition_id: Composition identifier
            result: Composition result to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"tool_composer:cache:{session_id}:{composition_id}"

            # Store as JSON with TTL
            await redis.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(result, default=str),
            )

            logger.debug(f"Cached composition result for {composition_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache composition result: {e}")
            return False

    async def get_cached_composition(
        self,
        session_id: str,
        composition_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached composition from working memory.

        Args:
            session_id: Session identifier
            composition_id: Composition identifier

        Returns:
            Cached result or None if not found
        """
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"tool_composer:cache:{session_id}:{composition_id}"

            cached = await redis.get(cache_key)
            if cached:
                return cast(Dict[str, Any], json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached composition: {e}")
            return None

    # =========================================================================
    # EXECUTION STATE TRACKING (Working Memory)
    # =========================================================================

    async def store_execution_state(
        self,
        session_id: str,
        composition_id: str,
        phase: str,
        step_id: Optional[str] = None,
        step_status: Optional[str] = None,
        step_output: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store execution state during composition for recovery and debugging.

        Args:
            session_id: Session identifier
            composition_id: Composition identifier
            phase: Current phase (decompose, plan, execute, synthesize)
            step_id: Optional step ID being executed
            step_status: Optional step status
            step_output: Optional step output

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            state_key = f"tool_composer:state:{session_id}:{composition_id}"

            state = {
                "phase": phase,
                "step_id": step_id,
                "step_status": step_status,
                "step_output": step_output,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store with shorter TTL (1 hour) as this is transient state
            await redis.setex(
                state_key,
                3600,
                json.dumps(state, default=str),
            )

            logger.debug(f"Stored execution state: {phase}/{step_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store execution state: {e}")
            return False

    # =========================================================================
    # COMPOSITION HISTORY (Episodic Memory)
    # =========================================================================

    async def store_composition(
        self,
        session_id: str,
        result: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store composition in episodic memory for future retrieval.

        Args:
            session_id: Session identifier
            result: Composition result to store
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

            # Extract key metrics
            composition_id = result.get("composition_id", "unknown")
            query = result.get("query", "")
            sub_questions = result.get("decomposition", {}).get("sub_questions", [])
            execution = result.get("execution", {})
            response = result.get("response", {})

            description = (
                f"Tool composition: "
                f"sub_questions={len(sub_questions)}, "
                f"tools_executed={execution.get('tools_executed', 0)}, "
                f"success={result.get('success', False)}, "
                f"confidence={response.get('confidence', 0):.2f}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="composition_completed",
                event_subtype="multi_faceted_query",
                description=description,
                raw_content={
                    "composition_id": composition_id,
                    "query": query[:500],
                    "sub_questions_count": len(sub_questions),
                    "tools_executed": execution.get("tools_executed", 0),
                    "tools_succeeded": execution.get("tools_succeeded", 0),
                    "total_duration_ms": result.get("total_duration_ms", 0),
                    "success": result.get("success", False),
                    "confidence": response.get("confidence", 0),
                    "tool_sequence": [
                        s.get("tool_name") for s in result.get("plan", {}).get("steps", [])
                    ],
                },
                entities=None,
                outcome_type="composition_delivered",
                agent_name="tool_composer",
                importance_score=0.75,
                e2i_refs=E2IEntityReferences(
                    brand=brand,
                    region=region,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{query} {description}",
                session_id=_ensure_uuid(session_id),
            )

            logger.info(f"Stored composition in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store composition in episodic memory: {e}")
            return None

    # =========================================================================
    # PROCEDURAL PATTERN STORAGE (Procedural Memory)
    # =========================================================================

    async def store_composition_pattern(
        self,
        result: Dict[str, Any],
    ) -> bool:
        """
        Store successful composition pattern in procedural memory.

        This enables learning of effective tool sequences for similar queries.
        Only stores patterns from successful compositions with high confidence.

        Args:
            result: Successful composition result

        Returns:
            True if successfully stored
        """
        # Only store successful, high-confidence patterns
        if not result.get("success", False):
            return False

        response = result.get("response", {})
        if response.get("confidence", 0) < 0.7:
            return False

        try:
            from src.memory.procedural_memory import ProceduralMemoryInput, insert_procedural_memory
            from src.memory.services.factories import get_embedding_service

            # Extract pattern
            decomposition = result.get("decomposition", {})
            plan = result.get("plan", {})

            pattern = {
                "query_intent": decomposition.get("decomposition_reasoning", "")[:200],
                "decomposition_strategy": decomposition.get("decomposition_reasoning", "")[:100],
                "tool_sequence": [s.get("tool_name") for s in plan.get("steps", [])],
                "parallel_groups": plan.get("parallel_groups", []),
            }

            # Build procedure input
            procedure = ProceduralMemoryInput(
                procedure_name=f"composition_{result.get('composition_id', 'unknown')}",
                tool_sequence=plan.get("steps", []),
                procedure_type="tool_composition",
                trigger_pattern=pattern.get("query_intent", ""),
                applicable_agents=["tool_composer"],
            )

            # Generate embedding and store
            embedding_service = get_embedding_service()
            trigger_embedding = await embedding_service.embed(
                pattern.get("query_intent", "tool composition")
            )
            await insert_procedural_memory(
                procedure=procedure,
                trigger_embedding=trigger_embedding,
            )

            logger.info("Stored composition pattern in procedural memory")
            return True
        except Exception as e:
            logger.warning(f"Failed to store composition pattern: {e}")
            return False

    # =========================================================================
    # SIMILAR COMPOSITION LOOKUP (For Plan Optimization)
    # =========================================================================

    async def find_similar_compositions(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past compositions for plan optimization.

        Used during the PLAN phase to leverage successful past compositions.

        Args:
            query: Query to find similar compositions for
            limit: Maximum results to return

        Returns:
            List of similar past compositions with their plans
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="composition_completed",
                agent_name="tool_composer",
            )

            results = await search_episodic_by_text(
                query_text=query,
                filters=filters,
                limit=limit * 2,  # Get more to filter by success
                min_similarity=0.7,
                include_entity_context=False,
            )

            # Filter to successful compositions only
            successful = [r for r in results if r.get("raw_content", {}).get("success", False)]

            return successful[:limit]
        except Exception as e:
            logger.warning(f"Failed to find similar compositions: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    session_id: Optional[str] = None,
    memory_hooks: Optional[ToolComposerMemoryHooks] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute composition results to CognitiveRAG's memory systems.

    This is the primary interface for storing tool composer
    results in the memory architecture.

    Args:
        result: CompositionResult dictionary
        session_id: Session identifier (generates UUID if not provided)
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if composition stored, 0 otherwise
        - procedural_stored: 1 if pattern stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_tool_composer_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "procedural_stored": 0,
        "working_cached": 0,
    }

    composition_id = result.get("composition_id", "unknown")

    # 1. Cache in working memory
    cached = await memory_hooks.cache_composition_result(
        session_id=session_id,
        composition_id=composition_id,
        result=result,
    )
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_composition(
        session_id=session_id,
        result=result,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store successful pattern in procedural memory
    if result.get("success", False):
        stored = await memory_hooks.store_composition_pattern(result)
        if stored:
            counts["procedural_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"procedural={counts['procedural_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[ToolComposerMemoryHooks] = None


def get_tool_composer_memory_hooks() -> ToolComposerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ToolComposerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
