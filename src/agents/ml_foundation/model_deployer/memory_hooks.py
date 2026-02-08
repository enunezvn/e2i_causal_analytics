"""
Model Deployer Agent Memory Hooks
==================================

Memory integration hooks for the Model Deployer agent's tri-memory architecture.

The Model Deployer agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past deployments)
3. Query semantic memory (FalkorDB - deployment patterns, version history)
4. Store deployment manifests for future retrieval and RAG

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
class DeploymentContext:
    """Context retrieved from all memory systems for deployment."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DeploymentRecord:
    """Record of deployment for storage in episodic memory."""

    session_id: str
    experiment_id: str
    deployment_id: str
    model_version: int
    target_environment: str
    deployment_status: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ModelDeployerMemoryHooks:
    """
    Memory integration hooks for the Model Deployer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache deployment manifests in working memory (24h TTL)
    - Store deployments in episodic memory for future retrieval
    - Store deployment patterns in semantic memory for knowledge graph
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
        model_uri: str,
        target_environment: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> DeploymentContext:
        """Retrieve context from all three memory systems."""
        context = DeploymentContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            model_uri=model_uri,
            target_environment=target_environment,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            model_uri=model_uri,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_deployments={len(context.semantic_context.get('deployments', []))}"
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
        model_uri: str,
        target_environment: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar deployments."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"model deployment {model_uri} {target_environment or ''}"

            filters = EpisodicSearchFilters(
                event_type="model_deployment_completed",
                agent_name="model_deployer",
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
        model_uri: str,
    ) -> Dict[str, Any]:
        """Get semantic memory context for deployment patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context: Dict[str, Any] = {
                "deployments": [],
                "version_history": [],
                "rollback_incidents": [],
            }

            # Query past deployments
            deployments = self.semantic_memory.query(
                "MATCH (d:Deployment) RETURN d ORDER BY d.timestamp DESC LIMIT 10"
            )
            context["deployments"] = deployments

            # Query rollback incidents
            rollbacks = self.semantic_memory.query(
                "MATCH (r:Rollback) RETURN r ORDER BY r.timestamp DESC LIMIT 5"
            )
            context["rollback_incidents"] = rollbacks

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_deployment_manifest(
        self,
        session_id: str,
        manifest: Dict[str, Any],
    ) -> bool:
        """Cache deployment manifest in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"model_deployer:manifest:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(manifest),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached deployment manifest for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache deployment manifest: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_deployment(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store deployment in episodic memory."""
        try:
            from src.memory.episodic_memory import insert_episodic_memory

            content = {
                "experiment_id": state.get("experiment_id"),
                "deployment_id": result.get("deployment_id"),
                "model_uri": state.get("model_uri"),
                "model_version": result.get("model_version"),
                "target_environment": state.get("target_environment"),
                "deployment_status": result.get("deployment_status"),
                "deployment_strategy": state.get("deployment_strategy"),
                "endpoint_url": result.get("endpoint_url"),
                "bento_tag": result.get("final_bento_tag"),
                "health_check_passed": result.get("health_check_passed"),
                "deployment_duration_seconds": result.get("deployment_duration_seconds"),
            }

            summary = (
                f"Deployment: {result.get('deployment_id', 'unknown')} "
                f"to {state.get('target_environment', 'unknown')}. "
                f"Status: {result.get('deployment_status', 'unknown')}. "
                f"Health: {'PASSED' if result.get('health_check_passed') else 'FAILED'}."
            )

            memory_id = await insert_episodic_memory(  # type: ignore[call-arg]
                session_id=session_id,
                event_type="model_deployment_completed",
                agent_name="model_deployer",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored deployment in episodic memory: {memory_id}")
            return str(memory_id) if memory_id else None
        except Exception as e:
            logger.warning(f"Failed to store deployment: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_deployment_pattern(
        self,
        experiment_id: str,
        deployment_id: str,
        model_version: int,
        target_environment: str,
        deployment_status: str,
        deployment_strategy: str,
        rollback_occurred: bool = False,
    ) -> bool:
        """Store deployment pattern in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            # Create deployment node
            self.semantic_memory.add_e2i_entity(
                entity_type="Deployment",
                entity_id=f"deploy:{deployment_id}",
                properties={
                    "deployment_id": deployment_id,
                    "experiment_id": experiment_id,
                    "model_version": model_version,
                    "target_environment": target_environment,
                    "status": deployment_status,
                    "strategy": deployment_strategy,
                    "agent": "model_deployer",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create relationship to experiment
            self.semantic_memory.add_relationship(
                from_entity_id=f"deploy:{deployment_id}",
                to_entity_id=f"exp:{experiment_id}",
                relationship_type="DEPLOYS_FOR",
                properties={"agent": "model_deployer"},
            )

            # Store rollback incident if occurred
            if rollback_occurred:
                self.semantic_memory.add_e2i_entity(
                    entity_type="Rollback",
                    entity_id=f"rollback:{deployment_id}",
                    properties={
                        "deployment_id": deployment_id,
                        "agent": "model_deployer",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.semantic_memory.add_relationship(
                    from_entity_id=f"rollback:{deployment_id}",
                    to_entity_id=f"deploy:{deployment_id}",
                    relationship_type="ROLLED_BACK",
                    properties={"agent": "model_deployer"},
                )

            logger.info(f"Stored deployment pattern: {deployment_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store deployment pattern: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ModelDeployerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute deployment results to memory systems."""
    import uuid

    if memory_hooks is None:
        memory_hooks = get_model_deployer_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if failed
    if result.get("overall_status") == "failed":
        logger.info("Skipping memory storage for failed deployment")
        return counts

    # 1. Cache in working memory
    manifest = result.get("deployment_manifest", {})
    cached = await memory_hooks.cache_deployment_manifest(session_id, manifest)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_deployment(
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
    deployment_id = result.get("deployment_id")
    if experiment_id and deployment_id:
        stored = await memory_hooks.store_deployment_pattern(
            experiment_id=experiment_id,
            deployment_id=deployment_id,
            model_version=result.get("model_version", 0),
            target_environment=state.get("target_environment", "unknown"),
            deployment_status=result.get("deployment_status", "unknown"),
            deployment_strategy=state.get("deployment_strategy", "direct"),
            rollback_occurred=result.get("rollback_successful", False),
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

_memory_hooks: Optional[ModelDeployerMemoryHooks] = None


def get_model_deployer_memory_hooks() -> ModelDeployerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ModelDeployerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
