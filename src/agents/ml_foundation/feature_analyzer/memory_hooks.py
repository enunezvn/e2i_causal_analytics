"""
Feature Analyzer Agent Memory Hooks
====================================

Memory integration hooks for the Feature Analyzer agent's tri-memory architecture.

The Feature Analyzer agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past feature analyses)
3. Query semantic memory (FalkorDB - feature importance patterns, interactions)
4. Store SHAP analyses and feature insights for future retrieval and RAG

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class FeatureAnalysisContext:
    """Context retrieved from all memory systems for feature analysis."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureImportanceRecord:
    """Record of feature importance analysis for storage."""

    session_id: str
    experiment_id: str
    shap_analysis_id: str
    top_features: List[str]
    global_importance: Dict[str, float]
    interactions: List[Tuple[str, str, float]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class FeatureAnalyzerMemoryHooks:
    """
    Memory integration hooks for the Feature Analyzer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache SHAP analyses in working memory (24h TTL)
    - Store feature analyses in episodic memory for future retrieval
    - Store feature importance patterns in semantic memory
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
        experiment_id: str,
        feature_names: Optional[List[str]] = None,
        max_episodic_results: int = 5,
    ) -> FeatureAnalysisContext:
        """Retrieve context from all three memory systems."""
        context = FeatureAnalysisContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            experiment_id=experiment_id,
            feature_names=feature_names,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            feature_names=feature_names,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_features={len(context.semantic_context.get('features', []))}"
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
        experiment_id: str,
        feature_names: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar feature analyses."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            feature_str = " ".join(feature_names[:5]) if feature_names else ""
            query_text = f"feature analysis SHAP importance {experiment_id} {feature_str}"

            filters = EpisodicSearchFilters(
                event_type="feature_analysis_completed",
                agent_name="feature_analyzer",
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
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get semantic memory context for feature patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context = {
                "features": [],
                "interactions": [],
                "importance_history": [],
            }

            if feature_names:
                for feature in feature_names[:5]:
                    history = self.semantic_memory.query(
                        f"MATCH (f:Feature {{name: '{feature}'}})-[r:HAS_IMPORTANCE]->(e:Experiment) "
                        f"RETURN f, r, e LIMIT 5"
                    )
                    context["importance_history"].extend(history)

            # Query known feature interactions
            interactions = self.semantic_memory.query(
                "MATCH (f1:Feature)-[i:INTERACTS_WITH]->(f2:Feature) "
                "RETURN f1, i, f2 ORDER BY i.strength DESC LIMIT 20"
            )
            context["interactions"] = interactions

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_feature_analysis(
        self,
        session_id: str,
        analysis: Dict[str, Any],
    ) -> bool:
        """Cache feature analysis in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"feature_analyzer:analysis:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(analysis),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached feature analysis for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache feature analysis: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_feature_analysis(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store feature analysis in episodic memory."""
        try:
            from src.memory.episodic_memory import store_episodic_memory

            content = {
                "experiment_id": state.get("experiment_id"),
                "shap_analysis_id": result.get("shap_analysis_id"),
                "top_features": result.get("top_features", []),
                "global_importance": state.get("global_importance", {}),
                "selected_features": state.get("selected_features", []),
                "selected_feature_count": state.get("selected_feature_count"),
                "interaction_list": result.get("interaction_list", []),
                "executive_summary": state.get("executive_summary"),
                "key_insights": state.get("key_insights", []),
                "recommendations": state.get("recommendations", []),
            }

            top_features = result.get("top_features", [])[:3]
            summary = (
                f"Feature Analysis: {state.get('experiment_id', 'unknown')}. "
                f"Top features: {', '.join(top_features) if top_features else 'N/A'}. "
                f"Selected {state.get('selected_feature_count', 0)} features."
            )

            memory_id = await store_episodic_memory(
                session_id=session_id,
                event_type="feature_analysis_completed",
                agent_name="feature_analyzer",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored feature analysis in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store feature analysis: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_feature_importance_patterns(
        self,
        experiment_id: str,
        global_importance: Dict[str, float],
        interactions: List[Dict[str, Any]],
    ) -> bool:
        """Store feature importance patterns in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            # Store top features with importance
            for feature_name, importance in sorted(
                global_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                self.semantic_memory.add_e2i_entity(
                    entity_type="Feature",
                    entity_id=f"feat:{feature_name}",
                    properties={
                        "name": feature_name,
                        "agent": "feature_analyzer",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                # Create importance relationship to experiment
                self.semantic_memory.add_relationship(
                    from_entity_id=f"feat:{feature_name}",
                    to_entity_id=f"exp:{experiment_id}",
                    relationship_type="HAS_IMPORTANCE",
                    properties={
                        "importance": importance,
                        "agent": "feature_analyzer",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # Store feature interactions
            for interaction in interactions[:10]:
                feat1 = interaction.get("feature_1")
                feat2 = interaction.get("feature_2")
                strength = interaction.get("interaction_strength", 0)

                if feat1 and feat2:
                    self.semantic_memory.add_relationship(
                        from_entity_id=f"feat:{feat1}",
                        to_entity_id=f"feat:{feat2}",
                        relationship_type="INTERACTS_WITH",
                        properties={
                            "strength": strength,
                            "experiment_id": experiment_id,
                            "agent": "feature_analyzer",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            logger.info(f"Stored feature importance patterns: {experiment_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store feature patterns: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[FeatureAnalyzerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute feature analysis results to memory systems."""
    import uuid

    if memory_hooks is None:
        memory_hooks = get_feature_analyzer_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if failed
    if result.get("status") == "failed":
        logger.info("Skipping memory storage for failed analysis")
        return counts

    # 1. Cache in working memory
    analysis = {
        "shap_analysis_id": result.get("shap_analysis_id"),
        "top_features": result.get("top_features"),
        "selected_feature_count": state.get("selected_feature_count"),
    }
    cached = await memory_hooks.cache_feature_analysis(session_id, analysis)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_feature_analysis(
        session_id=session_id,
        result=result,
        state=state,
        brand=brand,
        region=region,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    # 3. Store patterns in semantic memory
    experiment_id = state.get("experiment_id")
    global_importance = state.get("global_importance", {})
    interactions = result.get("interaction_list", [])

    if experiment_id and global_importance:
        stored = await memory_hooks.store_feature_importance_patterns(
            experiment_id=experiment_id,
            global_importance=global_importance,
            interactions=interactions,
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

_memory_hooks: Optional[FeatureAnalyzerMemoryHooks] = None


def get_feature_analyzer_memory_hooks() -> FeatureAnalyzerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = FeatureAnalyzerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
