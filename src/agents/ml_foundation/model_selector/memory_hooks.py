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
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

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
# READING THE AGENT OUTPUT (#2325)
# =============================================================================

#: What a missing value reads as in the row description. A score that was not
#: recorded must never render as ``0.00``, nor a missing algorithm as ``unknown``.
NOT_RECORDED = "not recorded"


def _text(value: Any) -> Optional[str]:
    """A non-empty string, else None (``_build_output`` defaults absent names to "")."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> Optional[float]:
    """A finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def selection_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """The selection as ``ModelSelectorAgent.run()`` returns it.

    ``run()`` (``agent._build_output``) nests the chosen algorithm under
    ``model_candidate`` and the reasons under ``selection_rationale``; nothing
    about the selection is at the top level. Reading the top level recorded NULL
    on every prod row (#2325). A value the output does not carry stays None.
    """
    candidate = result.get("model_candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    rationale = result.get("selection_rationale")
    rationale = rationale if isinstance(rationale, dict) else {}
    expected = candidate.get("expected_performance")
    return {
        "algorithm_name": _text(candidate.get("algorithm_name")),
        "algorithm_family": _text(candidate.get("algorithm_family")),
        "algorithm_class": _text(candidate.get("algorithm_class")),
        "selection_score": _number(candidate.get("selection_score")),
        "interpretability_score": _number(candidate.get("interpretability_score")),
        "scalability_score": _number(candidate.get("scalability_score")),
        "expected_performance": expected if isinstance(expected, dict) else {},
        "primary_reason": _text(rationale.get("primary_reason")),
    }


def selection_summary_text(fields: Dict[str, Any]) -> str:
    """The episodic description; an absent value reads as absent."""
    score = fields.get("selection_score")
    return (
        f"Model Selection: {fields.get('algorithm_name') or 'algorithm ' + NOT_RECORDED} "
        f"({fields.get('algorithm_family') or 'family ' + NOT_RECORDED}). "
        f"Score: {f'{score:.3f}' if score is not None else NOT_RECORDED}. "
        f"Reason: {fields.get('primary_reason') or NOT_RECORDED}"
    )


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
            return cast(List[Dict[str, Any]], messages)
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
            context: Dict[str, Any] = {
                "algorithms": [],
                "success_rates": {},
                "problem_type_algorithms": [],
            }

            # Query algorithms suited for this problem type
            algorithms = self.semantic_memory.query(
                "MATCH (a:Algorithm)-[:SUITED_FOR]->(p:ProblemType {name: $problem_type}) "
                "RETURN a ORDER BY a.success_rate DESC LIMIT 10",
                {"problem_type": problem_type},
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
        session_id: Optional[str],
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store model selection in episodic memory."""
        try:
            from src.memory.episodic_memory import insert_episodic_memory

            fields = selection_fields(result)
            content = {
                "experiment_id": state.get("experiment_id"),
                # The audit chain's workflow id used to be persisted AS the session
                # (#2099). It is a real correlation handle but not a conversation,
                # so it keeps its value here. Absent when the state has none.
                **(
                    {"audit_workflow_id": str(state["audit_workflow_id"])}
                    if state.get("audit_workflow_id")
                    else {}
                ),
                **fields,
                # The whole rationale block (text, factors, alternatives considered,
                # constraint compliance), the shape every prod row already stores.
                "selection_rationale": result.get("selection_rationale"),
                # The MLflow ``model_selection_<algorithm>`` run that holds the
                # structured selection, when this run registered one.
                "mlflow_run_id": result.get("mlflow_run_id"),
                "alternative_candidates": result.get("alternative_candidates", []),
                "benchmark_results": state.get("benchmark_results", {}),
            }

            summary = selection_summary_text(fields)

            memory_id = await insert_episodic_memory(  # type: ignore[call-arg]
                session_id=session_id,
                event_type="model_selection_completed",
                agent_name="model_selector",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored model selection in episodic memory: {memory_id}")
            return str(memory_id) if memory_id else None
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
        algorithm_family: Optional[str],
        problem_type: Optional[str],
        selection_score: Optional[float],
        benchmark_results: Dict[str, Any],
    ) -> bool:
        """Store algorithm selection pattern in semantic memory.

        A value that is not known is left off the node or edge rather than written
        as a stand-in (#2325): ``family='unknown'`` would overwrite a real family on
        the MERGEd Algorithm node, a ``0.0`` score reads as a measurement, and a
        ``ptype:unknown`` endpoint does not exist, so that edge was silently dropped.
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        def _known(props: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in props.items() if v is not None}

        try:
            # Create algorithm node
            self.semantic_memory.add_e2i_entity(
                entity_type="Algorithm",
                entity_id=f"algo:{algorithm_name}",
                properties=_known(
                    {
                        "name": algorithm_name,
                        "family": algorithm_family,
                        "agent": "model_selector",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )

            # Create problem type relationship (only to a problem type we know)
            if problem_type:
                self.semantic_memory.add_relationship(
                    from_entity_id=f"algo:{algorithm_name}",
                    to_entity_id=f"ptype:{problem_type}",
                    relationship_type="SUITED_FOR",
                    properties=_known(
                        {
                            "selection_score": selection_score,
                            "agent": "model_selector",
                        }
                    ),
                )

            # Create usage relationship to experiment
            self.semantic_memory.add_relationship(
                from_entity_id=f"algo:{algorithm_name}",
                to_entity_id=f"exp:{experiment_id}",
                relationship_type="USED_IN",
                properties=_known(
                    {
                        "selection_score": selection_score,
                        "benchmark_score": benchmark_results.get("score"),
                        "agent": "model_selector",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ),
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
    if memory_hooks is None:
        memory_hooks = get_model_selector_memory_hooks()

    if session_id is None:
        # No mint (#2076). An absent session id stays None: the episodic writer
        # coerces it to an honest NULL for the nullable ``session_id`` column
        # rather than recording a uuid that belongs to no conversation. A falsy
        # state value normalises to None too, so the session-KEYED Redis writes
        # below are skipped instead of keyed on an empty string.
        session_id = state.get("session_id") or None

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if error
    if state.get("error"):
        logger.info("Skipping memory storage due to error")
        return counts

    # The selection lives under model_candidate / selection_rationale (#2325).
    fields = selection_fields(result)

    # 1. Cache in working memory
    selection = {
        "algorithm_name": fields["algorithm_name"],
        "algorithm_family": fields["algorithm_family"],
        "selection_score": fields["selection_score"],
    }
    # Skipped without a session (#2076): the cache key embeds the session id, so a
    # session-less write would land under a key no reader can ever ask for.
    if session_id is not None:
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
    algorithm_name = fields["algorithm_name"]
    if experiment_id and algorithm_name:
        stored = await memory_hooks.store_algorithm_pattern(
            experiment_id=experiment_id,
            algorithm_name=algorithm_name,
            algorithm_family=fields["algorithm_family"],
            problem_type=state.get("problem_type"),
            selection_score=fields["selection_score"],
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
