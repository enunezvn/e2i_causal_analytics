"""
Model Selector Agent Memory Hooks
==================================

Memory integration hooks for the Model Selector agent's tri-memory architecture.

The Model Selector agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past model selections)
3. Query semantic memory (FalkorDB - algorithm success patterns)
4. Store model selection rationale for future retrieval and RAG

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
class ModelSelectionContext:
    """Context retrieved from all memory systems for model selection."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelSelectionRecord:
    """Record of model selection for storage in episodic memory."""

    session_id: str
    experiment_id: str
    algorithm_name: str
    algorithm_family: str
    selection_score: float
    selection_rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ModelSelectorMemoryHooks:
    """
    Memory integration hooks for the Model Selector agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache model selections in working memory (24h TTL)
    - Store model selections in episodic memory for future retrieval
    - Store algorithm patterns in semantic memory for knowledge graph
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
        problem_type: str,
        kpi_category: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> ModelSelectionContext:
        """Retrieve context from all three memory systems."""
        context = ModelSelectionContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            problem_type=problem_type,
            kpi_category=kpi_category,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            problem_type=problem_type,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_algorithms={len(context.semantic_context.get('algorithms', []))}"
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
            return messages
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

    async def _get_episodic_context(
        self,
        problem_type: str,
        kpi_category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar model selections."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"model selection algorithm {problem_type} {kpi_category or ''}"

            filters = EpisodicSearchFilters(
                event_type="model_selection_completed",
                agent_name="model_selector",
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
        problem_type: str,
    ) -> Dict[str, Any]:
        """Get semantic memory context for algorithm patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context = {
                "algorithms": [],
                "success_rates": {},
                "problem_type_algorithms": [],
            }

            # Query algorithms suited for this problem type
            algorithms = self.semantic_memory.query(
                f"MATCH (a:Algorithm)-[:SUITED_FOR]->(p:ProblemType {{name: '{problem_type}'}}) "
                f"RETURN a ORDER BY a.success_rate DESC LIMIT 10"
            )
            context["problem_type_algorithms"] = algorithms

            # Query historical algorithm success rates
            success_rates = self.semantic_memory.query(
                "MATCH (a:Algorithm)-[u:USED_IN]->(e:Experiment) "
                "WHERE u.success = true "
                "RETURN a.name, count(*) as successes ORDER BY successes DESC LIMIT 10"
            )
            context["success_rates"] = success_rates

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_model_selection(
        self,
        session_id: str,
        selection: Dict[str, Any],
    ) -> bool:
        """Cache model selection in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"model_selector:selection:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(selection),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached model selection for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache model selection: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_model_selection(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store model selection in episodic memory."""
        try:
            from src.memory.episodic_memory import store_episodic_memory

            content = {
                "experiment_id": state.get("experiment_id"),
                "algorithm_name": result.get("algorithm_name"),
                "algorithm_family": result.get("algorithm_family"),
                "algorithm_class": result.get("algorithm_class"),
                "selection_score": result.get("selection_score"),
                "selection_rationale": result.get("selection_rationale"),
                "interpretability_score": result.get("interpretability_score"),
                "scalability_score": result.get("scalability_score"),
                "expected_performance": result.get("expected_performance", {}),
                "alternative_candidates": result.get("alternative_candidates", []),
                "benchmark_results": state.get("benchmark_results", {}),
            }

            summary = (
                f"Model Selection: {result.get('algorithm_name', 'unknown')} "
                f"({result.get('algorithm_family', 'unknown')}). "
                f"Score: {result.get('selection_score', 0):.2f}. "
                f"Reason: {result.get('primary_reason', 'N/A')}"
            )

            memory_id = await store_episodic_memory(
                session_id=session_id,
                event_type="model_selection_completed",
                agent_name="model_selector",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored model selection in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store model selection: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_algorithm_pattern(
        self,
        experiment_id: str,
        algorithm_name: str,
        algorithm_family: str,
        problem_type: str,
        selection_score: float,
        benchmark_results: Dict[str, Any],
    ) -> bool:
        """Store algorithm selection pattern in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            # Create algorithm node
            self.semantic_memory.add_e2i_entity(
                entity_type="Algorithm",
                entity_id=f"algo:{algorithm_name}",
                properties={
                    "name": algorithm_name,
                    "family": algorithm_family,
                    "agent": "model_selector",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create problem type relationship
            self.semantic_memory.add_relationship(
                from_entity_id=f"algo:{algorithm_name}",
                to_entity_id=f"ptype:{problem_type}",
                relationship_type="SUITED_FOR",
                properties={
                    "selection_score": selection_score,
                    "agent": "model_selector",
                },
            )

            # Create usage relationship to experiment
            self.semantic_memory.add_relationship(
                from_entity_id=f"algo:{algorithm_name}",
                to_entity_id=f"exp:{experiment_id}",
                relationship_type="USED_IN",
                properties={
                    "selection_score": selection_score,
                    "benchmark_score": benchmark_results.get("score"),
                    "agent": "model_selector",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(f"Stored algorithm pattern: {algorithm_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store algorithm pattern: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ModelSelectorMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute model selection results to memory systems."""
    import uuid

    if memory_hooks is None:
        memory_hooks = get_model_selector_memory_hooks()

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
    selection = {
        "algorithm_name": result.get("algorithm_name"),
        "algorithm_family": result.get("algorithm_family"),
        "selection_score": result.get("selection_score"),
    }
    cached = await memory_hooks.cache_model_selection(session_id, selection)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_model_selection(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store pattern in semantic memory
    experiment_id = state.get("experiment_id")
    algorithm_name = result.get("algorithm_name")
    if experiment_id and algorithm_name:
        stored = await memory_hooks.store_algorithm_pattern(
            experiment_id=experiment_id,
            algorithm_name=algorithm_name,
            algorithm_family=result.get("algorithm_family", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            selection_score=result.get("selection_score", 0),
            benchmark_results=state.get("benchmark_results", {}),
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

_memory_hooks: Optional[ModelSelectorMemoryHooks] = None


def get_model_selector_memory_hooks() -> ModelSelectorMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ModelSelectorMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
