"""
Model Trainer Agent Memory Hooks
=================================

Memory integration hooks for the Model Trainer agent's tri-memory architecture.

The Model Trainer agent uses these hooks to:
1. Retrieve context from working memory (Redis - recent session data)
2. Search episodic memory (Supabase - similar past training runs)
3. Query semantic memory (FalkorDB - model performance patterns)
4. Store training results for future retrieval and RAG

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
class ModelTrainingContext:
    """Context retrieved from all memory systems for model training."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrainingResultRecord:
    """Record of training result for storage in episodic memory."""

    session_id: str
    experiment_id: str
    training_run_id: str
    algorithm_name: str
    test_metrics: Dict[str, float]
    success_criteria_met: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class ModelTrainerMemoryHooks:
    """
    Memory integration hooks for the Model Trainer agent.

    Provides methods to:
    - Retrieve context from working, episodic, and semantic memory
    - Cache training results in working memory (24h TTL)
    - Store training runs in episodic memory for future retrieval
    - Store model performance patterns in semantic memory
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
        algorithm_name: str,
        problem_type: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> ModelTrainingContext:
        """Retrieve context from all three memory systems."""
        context = ModelTrainingContext(session_id=session_id)

        context.working_memory = await self._get_working_memory_context(session_id)
        context.episodic_context = await self._get_episodic_context(
            algorithm_name=algorithm_name,
            problem_type=problem_type,
            limit=max_episodic_results,
        )
        context.semantic_context = await self._get_semantic_context(
            algorithm_name=algorithm_name,
        )

        logger.info(
            f"Retrieved context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"episodic={len(context.episodic_context)}, "
            f"semantic_runs={len(context.semantic_context.get('training_runs', []))}"
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
        algorithm_name: str,
        problem_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar training runs."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"model training {algorithm_name} {problem_type or ''}"

            filters = EpisodicSearchFilters(
                event_type="model_training_completed",
                agent_name="model_trainer",
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
        algorithm_name: str,
    ) -> Dict[str, Any]:
        """Get semantic memory context for training patterns."""
        if not self.semantic_memory:
            return {}

        try:
            context: Dict[str, Any] = {
                "training_runs": [],
                "best_hyperparameters": {},
                "performance_history": [],
            }

            # Query past training runs for this algorithm
            runs = self.semantic_memory.query(
                f"MATCH (m:Model)-[:TRAINED_WITH]->(a:Algorithm {{name: '{algorithm_name}'}}) "
                f"RETURN m ORDER BY m.test_auc DESC LIMIT 10"
            )
            context["training_runs"] = runs

            # Query best hyperparameters
            hyperparams = self.semantic_memory.query(
                f"MATCH (h:Hyperparameters)-[:USED_BY]->(m:Model)-[:TRAINED_WITH]->(a:Algorithm {{name: '{algorithm_name}'}}) "
                f"WHERE m.success_criteria_met = true "
                f"RETURN h LIMIT 5"
            )
            context["best_hyperparameters"] = hyperparams

            return context
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return {}

    # =========================================================================
    # STORAGE: WORKING MEMORY (CACHE)
    # =========================================================================

    async def cache_training_result(
        self,
        session_id: str,
        training_result: Dict[str, Any],
    ) -> bool:
        """Cache training result in working memory."""
        if not self.working_memory:
            return False

        try:
            cache_key = f"model_trainer:result:{session_id}"
            await self.working_memory.set(
                cache_key,
                json.dumps(training_result),
                ex=self.CACHE_TTL_SECONDS,
            )
            logger.debug(f"Cached training result for session {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache training result: {e}")
            return False

    # =========================================================================
    # STORAGE: EPISODIC MEMORY
    # =========================================================================

    async def store_training_result(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """Store training result in episodic memory."""
        try:
            from src.memory.episodic_memory import insert_episodic_memory

            test_metrics = result.get("test_metrics", {})

            content = {
                "experiment_id": state.get("experiment_id"),
                "training_run_id": result.get("training_run_id"),
                "model_id": result.get("model_id"),
                "algorithm_name": state.get("algorithm_name"),
                "test_metrics": test_metrics,
                "train_metrics": result.get("train_metrics", {}),
                "validation_metrics": result.get("validation_metrics", {}),
                "best_hyperparameters": state.get("best_hyperparameters", {}),
                "success_criteria_met": result.get("success_criteria_met"),
                "mlflow_run_id": result.get("mlflow_run_id"),
                "model_artifact_uri": result.get("model_artifact_uri"),
                "training_duration_seconds": result.get("total_training_duration_seconds"),
            }

            # Format metrics for summary
            auc = test_metrics.get("auc_roc") or test_metrics.get("auc")
            f1 = test_metrics.get("f1_score") or test_metrics.get("f1")
            rmse = test_metrics.get("rmse")

            metric_str = ""
            if auc:
                metric_str = f"AUC={auc:.3f}"
            elif f1:
                metric_str = f"F1={f1:.3f}"
            elif rmse:
                metric_str = f"RMSE={rmse:.3f}"

            summary = (
                f"Model Training: {state.get('algorithm_name', 'unknown')}. "
                f"{metric_str}. "
                f"Success criteria: {'MET' if result.get('success_criteria_met') else 'NOT MET'}."
            )

            memory_id = await insert_episodic_memory(  # type: ignore[call-arg]
                session_id=session_id,
                event_type="model_training_completed",
                agent_name="model_trainer",
                summary=summary,
                raw_content=content,
                brand=brand,
                region=region,
            )

            logger.info(f"Stored training result in episodic memory: {memory_id}")
            return str(memory_id) if memory_id else None
        except Exception as e:
            logger.warning(f"Failed to store training result: {e}")
            return None

    # =========================================================================
    # STORAGE: SEMANTIC MEMORY
    # =========================================================================

    async def store_model_pattern(
        self,
        experiment_id: str,
        training_run_id: str,
        algorithm_name: str,
        test_metrics: Dict[str, float],
        best_hyperparameters: Dict[str, Any],
        success_criteria_met: bool,
    ) -> bool:
        """Store model training pattern in semantic memory."""
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return False

        try:
            # Create model node
            self.semantic_memory.add_e2i_entity(
                entity_type="Model",
                entity_id=f"model:{training_run_id}",
                properties={
                    "training_run_id": training_run_id,
                    "experiment_id": experiment_id,
                    "algorithm_name": algorithm_name,
                    "test_auc": test_metrics.get("auc_roc") or test_metrics.get("auc"),
                    "test_f1": test_metrics.get("f1_score") or test_metrics.get("f1"),
                    "test_rmse": test_metrics.get("rmse"),
                    "success_criteria_met": success_criteria_met,
                    "agent": "model_trainer",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Create relationship to algorithm
            self.semantic_memory.add_relationship(
                from_entity_id=f"model:{training_run_id}",
                to_entity_id=f"algo:{algorithm_name}",
                relationship_type="TRAINED_WITH",
                properties={
                    "success": success_criteria_met,
                    "agent": "model_trainer",
                },
            )

            # Create relationship to experiment
            self.semantic_memory.add_relationship(
                from_entity_id=f"model:{training_run_id}",
                to_entity_id=f"exp:{experiment_id}",
                relationship_type="BELONGS_TO",
                properties={"agent": "model_trainer"},
            )

            # Store hyperparameters if successful
            if success_criteria_met and best_hyperparameters:
                self.semantic_memory.add_e2i_entity(
                    entity_type="Hyperparameters",
                    entity_id=f"hp:{training_run_id}",
                    properties={
                        "training_run_id": training_run_id,
                        "config": json.dumps(best_hyperparameters),
                        "agent": "model_trainer",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.semantic_memory.add_relationship(
                    from_entity_id=f"hp:{training_run_id}",
                    to_entity_id=f"model:{training_run_id}",
                    relationship_type="USED_BY",
                    properties={"agent": "model_trainer"},
                )

            logger.info(f"Stored model pattern: {training_run_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store model pattern: {e}")
            return False


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[ModelTrainerMemoryHooks] = None,
    session_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, int]:
    """Contribute training results to memory systems."""
    import uuid

    if memory_hooks is None:
        memory_hooks = get_model_trainer_memory_hooks()

    if session_id is None:
        session_id = state.get("session_id") or str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "semantic_stored": 0,
        "working_cached": 0,
    }

    # Skip if training failed
    if result.get("training_status") == "failed":
        logger.info("Skipping memory storage for failed training")
        return counts

    # 1. Cache in working memory
    training_result = {
        "training_run_id": result.get("training_run_id"),
        "model_id": result.get("model_id"),
        "success_criteria_met": result.get("success_criteria_met"),
        "test_metrics": result.get("test_metrics"),
    }
    cached = await memory_hooks.cache_training_result(session_id, training_result)
    if cached:
        counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_training_result(
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
    training_run_id = result.get("training_run_id")
    if experiment_id and training_run_id:
        stored = await memory_hooks.store_model_pattern(
            experiment_id=experiment_id,
            training_run_id=training_run_id,
            algorithm_name=state.get("algorithm_name", "unknown"),
            test_metrics=result.get("test_metrics", {}),
            best_hyperparameters=state.get("best_hyperparameters", {}),
            success_criteria_met=result.get("success_criteria_met", False),
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

_memory_hooks: Optional[ModelTrainerMemoryHooks] = None


def get_model_trainer_memory_hooks() -> ModelTrainerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = ModelTrainerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
