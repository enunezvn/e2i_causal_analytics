"""
Scope Definer Agent Memory Hooks
=================================

Memory integration hooks for the Scope Definer agent's tri-memory architecture.

The Scope Definer agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past scope definitions)
3. Query semantic memory (FalkorDB - ML experiment relationships, KPI patterns)
4. Store scope specifications for future retrieval and RAG

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
class ScopeDefinitionContext:
    """Context retrieved from all memory systems for scope definition."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScopeDefinitionRecord:
    """Record of a scope definition for storage in episodic memory."""

    session_id: str
    experiment_id: str
    experiment_name: str
    problem_type: str
    target_variable: str
    business_objective: str
    success_criteria: Dict[str, Any]
    scope_spec: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ScopeDefinerMemoryHooks:
    """
    Memory integration hooks for the Scope Definer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache scope definitions in working memory (24h TTL)
    - Store scope specs in episodic memory for future retrieval
    - Store ML experiment patterns in semantic memory for knowledge graph
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
        problem_description: str,
        problem_type: Optional[str] = None,
        target_variable: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> ScopeDefinitionContext:
        """
        Retrieve context from all three memory systems.

        Args:
            session_id: Session identifier for working memory lookup
            problem_description: Problem description for episodic similarity search
            problem_type: Optional problem type for filtering
            target_variable: Optional target variable for filtering
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            ScopeDefinitionContext with data from all memory systems
        """
        context = ScopeDefinitionContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Get episodic memory (similar past scope definitions)
        context.episodic_context = await self._get_episodic_context(
            problem_description=problem_description,
            problem_type=problem_type,
            target_variable=target_variable,
            limit=max_episodic_results,
        )

        # 3. Get semantic memory (existing ML patterns, experiment relationships)
        context.semantic_context = await self._get_semantic_context(
            problem_type=problem_type,
            target_variable=target_variable,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_experiments={len(context.semantic_context.get('experiments', []))}"
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
        problem_description: str,
        problem_type: Optional[str] = None,
        target_variable: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past scope definitions."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            filters = EpisodicSearchFilters(
                event_type="scope_definition_completed",
                agent_name="scope_definer",
            )

            results = await search_episodic_by_text(
                query_text=problem_description,
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
        problem_type: Optional[str] = None,
        target_variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get semantic memory context for ML experiments."""
        if not self.semantic_memory:
            return {}

        try:
            context: Dict[str, Any] = {
                "experiments": [],
                "problem_type_patterns": [],
                "target_variable_history": [],
            }

            # Query experiments related to problem type
            if problem_type:
                experiments = self.semantic_memory.query(
                    f"MATCH (e:Experiment)-[:HAS_TYPE]->(t:ProblemType {{name: '{problem_type}'}}) "
                    f"RETURN e LIMIT 10"
                )
                context["experiments"] = experiments

            # Query previous uses of target variable
            if target_variable:
                history = self.semantic_memory.query(
                    f"MATCH (s:ScopeSpec)-[:TARGETS]->(v:Variable {{name: '{target_variable}'}}) "
                    f"RETURN s LIMIT 10"
                )
                context["target_variable_history"] = history

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_scope_definition(
        self,
        session_id: str,
        scope_spec: Dict[str, Any],
    ) -> bool:
        """
        Cache scope definition in working memory.

        Args:
            session_id: Session identifier
            scope_spec: The scope specification to cache

        Returns:
            True if successfully cached
        """
        if not self.working_memory:
            return False

        try:
            cache_key = f"scope_definer:result:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(scope_spec),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached scope definition for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache scope definition: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_scope_definition(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store scope definition in episodic memory.

        Args:
            session_id: Session identifier
            result: ScopeDefinerState output fields
            state: Full ScopeDefinerState
            brand: Optional brand context
            region: Optional region context

        Returns:
            Memory entry ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import insert_episodic_memory

            # Build content from scope specification
            result.get("scope_spec", {})
            success_criteria = result.get("success_criteria", {})

            content = {
                "experiment_id": result.get("experiment_id"),
                "experiment_name": result.get("experiment_name"),
                "problem_type": state.get("inferred_problem_type"),
                "target_variable": state.get("inferred_target_variable"),
                "business_objective": state.get("business_objective"),
                "required_features": state.get("required_features", []),
                "success_criteria": success_criteria,
                "validation_passed": state.get("validation_passed", False),
            }

            # Build searchable summary
            summary = (
                f"ML Scope Definition: {result.get('experiment_name', 'unknown')}. "
                f"Problem type: {state.get('inferred_problem_type', 'unknown')}. "
                f"Target: {state.get('inferred_target_variable', 'unknown')}. "
                f"Objective: {state.get('business_objective', 'unknown')}"
            )

            memory_id = await insert_episodic_memory(  # type: ignore[call-arg]
                session_id=session_id,
                event_type="scope_definition_completed",
                agent_name="scope_definer",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
                kpi_category=state.get("use_case"),
            )

            logger.info(f"Stored scope definition in episodic memory: {memory_id}")
            return str(memory_id) if memory_id else None
        except Exception as e:
            logger.warning(f"Failed to store scope definition: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_experiment_pattern(
        self,
        experiment_id: str,
        experiment_name: str,
        problem_type: str,
        target_variable: str,
        features: List[str],
        success_criteria: Dict[str, Any],
    ) -> bool:
        """
        Store ML experiment pattern in semantic memory (knowledge graph).

        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
            problem_type: ML problem type
            target_variable: Target variable name
            features: List of features used
            success_criteria: Success criteria configuration

        Returns:
            True if successfully stored
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available for experiment storage")
            return False

        try:
            # Create experiment node
            self.semantic_memory.add_e2i_entity(
                entity_type="Experiment",
                entity_id=f"exp:{experiment_id}",
                properties={
                    "experiment_id": experiment_id,
                    "name": experiment_name,
                    "agent": "scope_definer",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create problem type node and relationship
            self.semantic_memory.add_e2i_entity(
                entity_type="ProblemType",
                entity_id=f"ptype:{problem_type}",
                properties={
                    "name": problem_type,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.semantic_memory.add_relationship(
                from_entity_id=f"exp:{experiment_id}",
                to_entity_id=f"ptype:{problem_type}",
                relationship_type="HAS_TYPE",
                properties={"agent": "scope_definer"},
            )

            # Create target variable node and relationship
            self.semantic_memory.add_e2i_entity(
                entity_type="Variable",
                entity_id=f"var:{target_variable}",
                properties={
                    "name": target_variable,
                    "role": "target",
                    "agent": "scope_definer",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.semantic_memory.add_relationship(
                from_entity_id=f"exp:{experiment_id}",
                to_entity_id=f"var:{target_variable}",
                relationship_type="TARGETS",
                properties={"agent": "scope_definer"},
            )

            # Create scope spec node
            self.semantic_memory.add_e2i_entity(
                entity_type="ScopeSpec",
                entity_id=f"scope:{experiment_id}",
                properties={
                    "experiment_id": experiment_id,
                    "success_criteria": json.dumps(success_criteria),
                    "feature_count": len(features),
                    "agent": "scope_definer",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.semantic_memory.add_relationship(
                from_entity_id=f"exp:{experiment_id}",
                to_entity_id=f"scope:{experiment_id}",
                relationship_type="DEFINED_BY",
                properties={"agent": "scope_definer"},
            )

            logger.info(f"Stored experiment pattern: {experiment_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store experiment pattern: {e}")
            return False

    # =========================================================================
    # PRIOR SCOPE DEFINITIONS (For DSPy Training)
    # =========================================================================

    async def get_prior_scopes(
        self,
        problem_type: Optional[str] = None,
        min_validation_score: float = 0.8,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get prior scope definitions for DSPy training signals.

        Args:
            problem_type: Optional filter by problem type
            min_validation_score: Minimum validation score threshold
            limit: Maximum results to return

        Returns:
            List of prior scope definitions
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"scope definition {problem_type or ''}"

            filters = EpisodicSearchFilters(
                event_type="scope_definition_completed",
                agent_name="scope_definer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit * 2,
                min_similarity=0.5,
                include_entity_context=False,
            )

            # Filter by validation status
            filtered = [
                r for r in results if r.get("raw_content", {}).get("validation_passed", False)
            ]

            return filtered[:limit]
        except Exception as e:
            logger.warning(f"Failed to get prior scopes: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ScopeDefinerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute scope definition results to CognitiveRAG's memory systems.

    This is the primary interface for storing scope definer results
    in the tri-memory architecture.

    Args:
        result: ScopeDefinerState output fields
        state: Full ScopeDefinerState
        memory_hooks: Optional memory hooks instance
        session_id: Session identifier
        brand: Optional brand context
        region: Optional region context

    Returns:
        Dictionary with counts of stored memories
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_scope_definer_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if validation failed
    if not state.get("validation_passed"):
        logger.info("Skipping memory storage for failed validation")
        return counts

    # 1. Cache in working memory
    scope_spec = result.get("scope_spec", {})
    cached = await memory_hooks.cache_scope_definition(session_id, scope_spec)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_scope_definition(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand or state.get("brand"),
        region=region or state.get("region"),
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store experiment pattern in semantic memory
    experiment_id = result.get("experiment_id")
    if experiment_id:
        stored = await memory_hooks.store_experiment_pattern(
            experiment_id=experiment_id,
            experiment_name=result.get("experiment_name", ""),
            problem_type=state.get("inferred_problem_type", "unknown"),
            target_variable=state.get("inferred_target_variable", "unknown"),
            features=state.get("required_features", []),
            success_criteria=result.get("success_criteria", {}),
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

_memory_hooks: Optional[ScopeDefinerMemoryHooks] = None


def get_scope_definer_memory_hooks() -> ScopeDefinerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ScopeDefinerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
