"""
Prediction Synthesizer Agent Memory Hooks
==========================================

Memory integration hooks for the Prediction Synthesizer agent's memory architecture.

The Prediction Synthesizer uses these hooks to:
1. Retrieve context from working memory (Redis - recent predictions, session context)
2. Search episodic memory (Supabase - similar past predictions and their accuracy)
3. Store prediction results for future reference and calibration

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PredictionMemoryContext:
    """Context retrieved from memory systems for prediction synthesis."""

    session_id: str
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    cached_predictions: List[Dict[str, Any]] = field(default_factory=list)
    model_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PredictionRecord:
    """Record of a prediction for storage in episodic memory."""

    session_id: str
    entity_id: str
    entity_type: str
    prediction_target: str
    point_estimate: float
    confidence: float
    model_agreement: float
    ensemble_method: str
    models_succeeded: int
    models_failed: int
    time_horizon: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPredictionRecord:
    """Record of an individual model's prediction for tracking."""

    model_id: str
    model_type: str
    prediction: float
    confidence: float
    latency_ms: int
    features_used: List[str]


# =============================================================================
# MEMORY HOOKS CLASS
# =============================================================================


class PredictionSynthesizerMemoryHooks:
    """
    Memory integration hooks for the Prediction Synthesizer agent.

    Provides methods to:
    - Retrieve context from working and episodic memory
    - Cache predictions in working memory (24h TTL)
    - Store predictions in episodic memory for future calibration
    - Track model performance for ensemble weighting
    """

    # Cache TTL in seconds (24 hours)
    CACHE_TTL_SECONDS = 86400
    # Prediction cache TTL (shorter for freshness)
    PREDICTION_CACHE_TTL = 3600  # 1 hour

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
        entity_id: str,
        entity_type: str,
        prediction_target: str,
        time_horizon: Optional[str] = None,
        max_episodic_results: int = 5,
    ) -> PredictionMemoryContext:
        """
        Retrieve context from working and episodic memory.

        Args:
            session_id: Session identifier for working memory lookup
            entity_id: Entity being predicted (HCP ID, territory ID, etc.)
            entity_type: Type of entity (hcp, territory, patient)
            prediction_target: What is being predicted
            time_horizon: Prediction time horizon
            max_episodic_results: Maximum episodic memories to retrieve

        Returns:
            PredictionMemoryContext with data from memory systems
        """
        context = PredictionMemoryContext(session_id=session_id)

        # 1. Get working memory (recent session context)
        context.working_memory = await self._get_working_memory_context(session_id)

        # 2. Check for cached predictions for this entity
        context.cached_predictions = await self._get_cached_predictions(
            entity_id=entity_id,
            entity_type=entity_type,
            prediction_target=prediction_target,
        )

        # 3. Get episodic memory (similar past predictions)
        context.episodic_context = await self._get_episodic_context(
            entity_type=entity_type,
            prediction_target=prediction_target,
            time_horizon=time_horizon,
            limit=max_episodic_results,
        )

        # 4. Get model performance history
        context.model_performance = await self._get_model_performance_history(
            prediction_target=prediction_target,
        )

        logger.info(
            f"Retrieved prediction context for session {session_id}: "
            f"working={len(context.working_memory)}, "
            f"cached={len(context.cached_predictions)}, "
            f"episodic={len(context.episodic_context)}"
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

    async def _get_cached_predictions(
        self,
        entity_id: str,
        entity_type: str,
        prediction_target: str,
    ) -> List[Dict[str, Any]]:
        """Get recently cached predictions for an entity."""
        if not self.working_memory:
            return []

        try:
            redis = await self.working_memory.get_client()
            cache_key = f"prediction_synthesizer:entity:{entity_type}:{entity_id}:{prediction_target}"

            cached = await redis.get(cache_key)
            if cached:
                return [json.loads(cached)]
            return []
        except Exception as e:
            logger.warning(f"Failed to get cached predictions: {e}")
            return []

    async def _get_episodic_context(
        self,
        entity_type: str,
        prediction_target: str,
        time_horizon: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for similar past predictions."""
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"prediction {entity_type} {prediction_target}"
            if time_horizon:
                query_text += f" {time_horizon}"

            filters = EpisodicSearchFilters(
                event_type="prediction_completed",
                agent_name="prediction_synthesizer",
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
            logger.warning(f"Failed to search episodic memory: {e}")
            return []

    async def _get_model_performance_history(
        self,
        prediction_target: str,
        limit: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """Get historical model performance for weighting."""
        if not self.working_memory:
            return {}

        try:
            redis = await self.working_memory.get_client()
            performance_key = f"prediction_synthesizer:model_performance:{prediction_target}"

            cached = await redis.get(performance_key)
            if cached:
                return json.loads(cached)
            return {}
        except Exception as e:
            logger.warning(f"Failed to get model performance history: {e}")
            return {}

    # =========================================================================
    # PREDICTION CACHING (Working Memory)
    # =========================================================================

    async def cache_prediction(
        self,
        session_id: str,
        entity_id: str,
        entity_type: str,
        prediction_target: str,
        prediction_result: Dict[str, Any],
    ) -> bool:
        """
        Cache prediction result in working memory.

        Args:
            session_id: Session identifier
            entity_id: Entity being predicted
            entity_type: Type of entity
            prediction_target: What was predicted
            prediction_result: Prediction output to cache

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()

            # Cache by entity for fast lookup
            entity_key = f"prediction_synthesizer:entity:{entity_type}:{entity_id}:{prediction_target}"
            await redis.setex(
                entity_key,
                self.PREDICTION_CACHE_TTL,
                json.dumps(prediction_result, default=str),
            )

            # Also cache by session
            session_key = f"prediction_synthesizer:session:{session_id}"
            await redis.setex(
                session_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(prediction_result, default=str),
            )

            logger.debug(f"Cached prediction for entity {entity_type}:{entity_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
            return False

    async def update_model_performance(
        self,
        prediction_target: str,
        model_id: str,
        accuracy: float,
        calibration_error: float,
    ) -> bool:
        """
        Update model performance tracking for future weighting.

        Args:
            prediction_target: What was predicted
            model_id: Model identifier
            accuracy: Model accuracy on this target
            calibration_error: Calibration error

        Returns:
            True if successful
        """
        if not self.working_memory:
            return False

        try:
            redis = await self.working_memory.get_client()
            performance_key = f"prediction_synthesizer:model_performance:{prediction_target}"

            # Get existing performance data
            existing = await redis.get(performance_key)
            performance = json.loads(existing) if existing else {}

            # Update model entry
            performance[model_id] = {
                "accuracy": accuracy,
                "calibration_error": calibration_error,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            # Store with TTL
            await redis.setex(
                performance_key,
                self.CACHE_TTL_SECONDS * 7,  # Keep for 1 week
                json.dumps(performance),
            )

            return True
        except Exception as e:
            logger.warning(f"Failed to update model performance: {e}")
            return False

    # =========================================================================
    # PREDICTION STORAGE (Episodic Memory)
    # =========================================================================

    async def store_prediction(
        self,
        session_id: str,
        result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store prediction in episodic memory for future reference.

        Args:
            session_id: Session identifier
            result: Prediction output to store
            state: Prediction synthesizer state with details

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            from src.memory.episodic_memory import (
                E2IEntityReferences,
                EpisodicMemoryInput,
                insert_episodic_memory_with_text,
            )

            # Extract key fields
            entity_id = state.get("entity_id", "unknown")
            entity_type = state.get("entity_type", "unknown")
            prediction_target = state.get("prediction_target", "unknown")
            time_horizon = state.get("time_horizon", "unknown")

            ensemble = result.get("ensemble_prediction", {})
            point_estimate = ensemble.get("point_estimate", 0)
            confidence = ensemble.get("confidence", 0)
            model_agreement = ensemble.get("model_agreement", 0)

            # Build description for embedding
            description = (
                f"Prediction for {entity_type} {entity_id}: "
                f"target={prediction_target}, "
                f"estimate={point_estimate:.3f}, "
                f"confidence={confidence:.2f}, "
                f"agreement={model_agreement:.2f}, "
                f"horizon={time_horizon}"
            )

            # Create episodic memory input
            memory_input = EpisodicMemoryInput(
                event_type="prediction_completed",
                event_subtype="ensemble_prediction",
                description=description,
                raw_content={
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "prediction_target": prediction_target,
                    "time_horizon": time_horizon,
                    "point_estimate": point_estimate,
                    "prediction_interval": [
                        ensemble.get("prediction_interval_lower", 0),
                        ensemble.get("prediction_interval_upper", 0),
                    ],
                    "confidence": confidence,
                    "model_agreement": model_agreement,
                    "ensemble_method": ensemble.get("ensemble_method", "unknown"),
                    "models_succeeded": result.get("models_succeeded", 0),
                    "models_failed": result.get("models_failed", 0),
                    "total_latency_ms": result.get("total_latency_ms", 0),
                },
                entities=None,
                outcome_type="prediction_delivered",
                agent_name="prediction_synthesizer",
                importance_score=0.7,  # Predictions are moderately important
                e2i_refs=E2IEntityReferences(
                    hcp_id=entity_id if entity_type == "hcp" else None,
                ),
            )

            # Insert with auto-generated embedding
            memory_id = await insert_episodic_memory_with_text(
                memory=memory_input,
                text_to_embed=f"{state.get('query', '')} {description}",
                session_id=session_id,
            )

            logger.info(f"Stored prediction in episodic memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Failed to store prediction in episodic memory: {e}")
            return None

    # =========================================================================
    # CALIBRATION DATA (For DSPy Training)
    # =========================================================================

    async def get_calibration_data(
        self,
        prediction_target: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get historical predictions with actual outcomes for calibration.

        Used to collect training signals for DSPy optimization
        of prediction confidence calibration.

        Args:
            prediction_target: Filter by prediction target
            entity_type: Optional filter by entity type
            limit: Maximum results to return

        Returns:
            List of historical predictions with outcomes
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            query_text = f"prediction calibration {prediction_target}"
            if entity_type:
                query_text += f" {entity_type}"

            filters = EpisodicSearchFilters(
                event_type="prediction_completed",
                agent_name="prediction_synthesizer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.4,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get calibration data: {e}")
            return []

    async def get_similar_predictions(
        self,
        entity_type: str,
        prediction_target: str,
        features: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get similar past predictions for context.

        Used to enrich predictions with historical context
        and improve confidence calibration.

        Args:
            entity_type: Type of entity
            prediction_target: What is being predicted
            features: Feature values for similarity matching
            limit: Maximum results to return

        Returns:
            List of similar past predictions
        """
        try:
            from src.memory.episodic_memory import (
                EpisodicSearchFilters,
                search_episodic_by_text,
            )

            # Build query from features
            feature_str = " ".join(
                f"{k}:{v}" for k, v in list(features.items())[:5]
            )
            query_text = f"prediction {entity_type} {prediction_target} {feature_str}"

            filters = EpisodicSearchFilters(
                event_type="prediction_completed",
                agent_name="prediction_synthesizer",
            )

            results = await search_episodic_by_text(
                query_text=query_text,
                filters=filters,
                limit=limit,
                min_similarity=0.5,
                include_entity_context=False,
            )

            return results
        except Exception as e:
            logger.warning(f"Failed to get similar predictions: {e}")
            return []


# =============================================================================
# MEMORY CONTRIBUTION FUNCTION (Per Specialist Document)
# =============================================================================


async def contribute_to_memory(
    result: Dict[str, Any],
    state: Dict[str, Any],
    memory_hooks: Optional[PredictionSynthesizerMemoryHooks] = None,
    session_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Contribute prediction results to CognitiveRAG's memory systems.

    This is the primary interface for storing prediction synthesizer
    results in the memory architecture.

    Args:
        result: PredictionSynthesizerOutput dictionary
        state: PredictionSynthesizerState dictionary
        memory_hooks: Optional memory hooks instance (creates new if not provided)
        session_id: Session identifier (generates UUID if not provided)

    Returns:
        Dictionary with counts of stored memories:
        - episodic_stored: 1 if prediction stored, 0 otherwise
        - working_cached: 1 if cached, 0 otherwise
    """
    import uuid

    if memory_hooks is None:
        memory_hooks = get_prediction_synthesizer_memory_hooks()

    if session_id is None:
        session_id = str(uuid.uuid4())

    counts = {
        "episodic_stored": 0,
        "working_cached": 0,
    }

    # Skip storage if prediction failed
    if state.get("status") == "failed":
        logger.info("Skipping memory storage for failed prediction")
        return counts

    # Extract entity info
    entity_id = state.get("entity_id", "")
    entity_type = state.get("entity_type", "")
    prediction_target = state.get("prediction_target", "")

    # 1. Cache in working memory
    if entity_id and entity_type and prediction_target:
        cached = await memory_hooks.cache_prediction(
            session_id=session_id,
            entity_id=entity_id,
            entity_type=entity_type,
            prediction_target=prediction_target,
            prediction_result=result,
        )
        if cached:
            counts["working_cached"] = 1

    # 2. Store in episodic memory
    memory_id = await memory_hooks.store_prediction(
        session_id=session_id,
        result=result,
        state=state,
    )
    if memory_id:
        counts["episodic_stored"] = 1

    logger.info(
        f"Memory contribution complete: "
        f"episodic={counts['episodic_stored']}, "
        f"working_cached={counts['working_cached']}"
    )

    return counts


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[PredictionSynthesizerMemoryHooks] = None


def get_prediction_synthesizer_memory_hooks() -> PredictionSynthesizerMemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = PredictionSynthesizerMemoryHooks()
    return _memory_hooks


def reset_memory_hooks() -> None:
    """Reset the memory hooks singleton (for testing)."""
    global _memory_hooks
    _memory_hooks = None
