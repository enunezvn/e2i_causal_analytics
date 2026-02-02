"""
Resource Optimizer Agent Memory Hooks
======================================

Memory integration hooks for the Resource Optimizer agent's memory architecture.

The Resource Optimizer uses these hooks to:
1. Retrieve context from working memory (Redis - cached optimizations)
2. Store/retrieve optimization patterns from procedural memory
3. Cache optimization results for scenario comparisons

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

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
class OptimizationContext:
    """Context retrieved from memory systems for optimization."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    cached_optimization: Optional[Dict[str, Any]] = None
    similar_optimizations: List[Dict[str, Any]] = field(default_factory=list)
    learned_patterns: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationPattern:
    """Learned optimization pattern for procedural memory."""

    pattern_id: str
    resource_type: str
    objective: str
    constraint_signature: str  # Hash of constraint types
    solver_type: str
    avg_solve_time_ms: int
    success_rate: float
    common_adjustments: List[Dict[str, Any]]
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationRecord:
    """Record of an optimization for storage."""

    session_id: str
    resource_type: str
    objective: str
    objective_value: float
    projected_roi: float
    entities_optimized: int
    solver_type: str
    solve_time_ms: int
    solver_status: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ResourceOptimizerMemoryHooks:
    """
    Memory integration hooks for the Resource Optimizer agent.

    Provides methods to:
    - Retrieve context from working and procedural memory
    - Cache optimization results for scenario comparisons
    - Store/retrieve learned optimization patterns
    - Track optimization performance for DSPy recipients
    """

    # Cache TTL in seconds (1 hour for scenario comparisons)
    CACHE_TTL_SECONDS = 3600
    # Pattern learning TTL (30 days)
    PATTERN_TTL_SECONDS = 2592000

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
        resource_type: str,
        objective: str,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationContext:
        """
        Retrieve context from working and procedural memory.

        Args:
            session_id: Session identifier for working memory lookup
            resource_type: Type of resource being optimized
            objective: Optimization objective
            constraints: List of constraints for pattern matching

        Returns:
            OptimizationContext with data from memory systems
        """
        context = OptimizationContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Check for cached optimization in this session
        context.cached_optimization = await self._get_cached_optimization(session_id)

        # 3. Get similar past optimizations
        context.similar_optimizations = await self._get_similar_optimizations(
            resource_type=resource_type,
            objective=objective,
        )

        # 4. Get learned patterns from procedural memory
        context.learned_patterns = await self._get_optimization_patterns(
            resource_type=resource_type,
            objective=objective,
            constraints=constraints,
        )

        logger.info(
            f"Retrieved optimization context for session {session_id}: "
            f"cached={context.cached_optimization is not None}, "
            f"similar={len(context.similar_optimizations)}, "
            f"patterns={len(context.learned_patterns)}"
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

    async def _get_cached_optimization(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization result from current session."""
        if not self.working_memory:
            return None

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"resource_optimizer:session:{session_id}"

            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached optimization: {e}")
            return None

    async def _get_similar_optimizations(
        self,
        resource_type: str,
        objective: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get similar past optimizations from episodic memory."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"resource optimization {resource_type} {objective}"

            filters = EpisodicSearchFilters(
                event_type="optimization_completed",
                agent_name="resource_optimizer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.6,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get similar optimizations: {e}")
            return []

    async def _get_optimization_patterns(
        self,
        resource_type: str,
        objective: str,
        constraints: Optional[List[Dict[str, Any]]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Get learned optimization patterns from procedural memory."""
        try:
            from src.memory.procedural_memory import (
                ProceduralSearchFilters,
                search_procedures_by_text,
            )

            # Build query from optimization parameters
            query_text = f"optimization pattern {resource_type} {objective}"
            if constraints:
                constraint_types = [c.get("constraint_type", "") for c in constraints[:3]]
                query_text += f" constraints: {' '.join(constraint_types)}"

            filters = ProceduralSearchFilters(
                procedure_type="optimization_pattern",
                agent_name="resource_optimizer",
            )

            results = await search_procedures_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
            )

            return results
        except ImportError:
            # Procedural memory may not be fully implemented yet
            logger.debug("Procedural memory not available, skipping pattern retrieval")
            return []
        except Exception as e:
            logger.warning(f"Failed to get optimization patterns: {e}")
            return []

    # =========================================================================
    # OPTIMIZATION CACHING (Working Memory)
    # =========================================================================

    async def cache_optimization(
        self,
        session_id: str,
        optimization_result: Dict[str, Any],
        scenario_name: Optional[str] = None,
    ) -> bool:
        """
        Cache optimization result in working memory.

        Args:
            session_id: Session identifier
            optimization_result: Optimization output to cache
            scenario_name: Optional scenario name for multi-scenario caching

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            # Cache by session
            session_key = f"resource_optimizer:session:{session_id}"
            await redis.setex(
                session_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(optimization_result, default=str),
            )

            # Also cache by scenario if provided
            if scenario_name:
                scenario_key = f"resource_optimizer:scenario:{session_id}:{scenario_name}"
                await redis.setex(
                    scenario_key,
                    self.CACHE_TTL_SECONDS,
                    json.dumps(optimization_result, default=str),
                )

            logger.debug(f"Cached optimization for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache optimization: {e}")
            return False

    async def get_scenario_comparison(
        self,
        session_id: str,
        scenario_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get cached scenarios for comparison.

        Args:
            session_id: Session identifier
            scenario_names: List of scenario names to retrieve

        Returns:
            Dictionary mapping scenario names to results
        """
        if not self.working_memory:
            return {}

        try:
            redis = await self.working_memory.get_client()
            results = {}

            for name in scenario_names:
                scenario_key = f"resource_optimizer:scenario:{session_id}:{name}"
                cached = await redis.get(scenario_key)
                if cached:
                    results[name] = json.loads(cached)

            return results
        except Exception as e:
            logger.warning(f"Failed to get scenario comparison: {e}")
            return {}

    # =========================================================================
    # PATTERN LEARNING (Procedural Memory)
    # =========================================================================

    async def store_optimization_pattern(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store successful optimization pattern in procedural memory.

        Only stores patterns from successful optimizations with good ROI.

        Args:
            session_id: Session identifier
            result: Optimization output
            state: Optimization state

        Returns:
            Pattern ID if stored, None otherwise
        """
        # Only learn from successful optimizations
        if state.get("status") != "completed":
            return None

        solver_status = result.get("solver_status", "")
        if solver_status not in ("optimal", "feasible"):
            return None

        # Only learn from optimizations with positive ROI
        projected_roi = result.get("projected_roi", 0)
        if projected_roi < 1.0:
            return None

        try:
            from src.memory.procedural_memory import (
                ProceduralMemoryInput,
                insert_procedural_memory,
            )

            resource_type = state.get("resource_type", "unknown")
            objective = state.get("objective", "unknown")
            solver_type = state.get("solver_type", "linear")
            constraints = state.get("constraints", [])

            # Build constraint signature
            constraint_types = sorted([c.get("constraint_type", "") for c in constraints])
            constraint_signature = "|".join(constraint_types)

            # Extract common adjustments from allocations
            allocations = result.get("optimal_allocations", [])
            adjustments = []
            for alloc in allocations[:5]:  # Top 5 adjustments
                if abs(alloc.get("change_percentage", 0)) > 10:
                    adjustments.append({
                        "entity_type": alloc.get("entity_type", "unknown"),
                        "direction": "increase" if alloc.get("change", 0) > 0 else "decrease",
                        "magnitude": abs(alloc.get("change_percentage", 0)),
                    })

            # Build description
            description = (
                f"Optimization pattern: {resource_type} / {objective} "
                f"with constraints [{constraint_signature}], "
                f"solver={solver_type}, ROI={projected_roi:.2f}"
            )

            # Create procedural memory input
            memory_input = ProceduralMemoryInput(
                procedure_type="optimization_pattern",
                description=description,
                raw_content={
                    "resource_type": resource_type,
                    "objective": objective,
                    "constraint_signature": constraint_signature,
                    "solver_type": solver_type,
                    "projected_roi": projected_roi,
                    "solve_time_ms": result.get("solve_time_ms", 0),
                    "common_adjustments": adjustments,
                    "entities_optimized": len(allocations),
                },
                agent_name="resource_optimizer",
                success_count=1,
                last_success=datetime.now(timezone.utc).isoformat(),
            )

            # Insert pattern
            pattern_id = await insert_procedural_memory(
                memory=memory_input,
                text_to_embed=description,
            )

            logger.info(f"Stored optimization pattern: {pattern_id}")
            return pattern_id
        except ImportError:
            logger.debug("Procedural memory not available, skipping pattern storage")
            return None
        except Exception as e:
            logger.warning(f"Failed to store optimization pattern: {e}")
            return None

    # =========================================================================
    # OPTIMIZATION STORAGE (Episodic Memory)
    # =========================================================================

    async def store_optimization(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store optimization in episodic memory for future reference.

        Args:
            session_id: Session identifier
            result: Optimization output
            state: Optimization state

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Extract key fields
            resource_type = state.get("resource_type", "unknown")
            objective = state.get("objective", "unknown")
            objective_value = result.get("objective_value", 0)
            projected_roi = result.get("projected_roi", 0)
            allocations = result.get("optimal_allocations", [])

            # Build description
            description = (
                f"Resource optimization: {resource_type} / {objective}, "
                f"objective={objective_value:.2f}, ROI={projected_roi:.2f}, "
                f"entities={len(allocations)}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="optimization_completed",
                event_subtype="resource_allocation",
                description=description,
                raw_content={
                    "resource_type": resource_type,
                    "objective": objective,
                    "objective_value": objective_value,
                    "projected_roi": projected_roi,
                    "projected_total_outcome": result.get("projected_total_outcome", 0),
                    "entities_optimized": len(allocations),
                    "solver_type": state.get("solver_type", "linear"),
                    "solver_status": result.get("solver_status", "unknown"),
                    "solve_time_ms": result.get("solve_time_ms", 0),
                    "recommendations": result.get("recommendations", [])[:5],
                },
                entities=None,
                outcome_type="optimization_delivered",
                agent_name="resource_optimizer",
                importance_score=0.7,
                e2i_refs=None,
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{state.get('query', '')} {description}",
                session_id=_ensure_uuid(session_id),
            )

            logger.info(f"Stored optimization in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store optimization in episodic memory: {e}")
            return None


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ResourceOptimizerMemoryHooks] = None,
    session_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute optimization results to CognitiveRAG's memory systems.

    This is the primary interface for storing resource optimizer
    results in the memory architecture.

    Args:
        result: ResourceOptimizerOutput dictionary
        state: ResourceOptimizerState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if optimization stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
        - pattern_learned: 1 if pattern stored, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_resource_optimizer_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
        "pattern_learned": 0,
    }

    # Skip storage if optimization failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed optimization")
        return counts

    # 1. Cache in working memory
    cached = await memory_hooks.cache_optimization(session_id, result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_optimization(
        session_id=session_id,
        result=result,
        state=state,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Learn pattern in procedural memory (if successful)
    pattern_id = await memory_hooks.store_optimization_pattern(
        session_id=session_id,
        result=result,
        state=state,
    )
    if pattern_id:
        counts["pattern_learned"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"working_cached={counts['working_cached']}, "
        f"pattern_learned={counts['pattern_learned']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[ResourceOptimizerMemoryHooks] = None


def get_resource_optimizer_memory_hooks() -> ResourceOptimizerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ResourceOptimizerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
