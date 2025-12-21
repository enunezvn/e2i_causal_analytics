"""
E2I Prediction Synthesizer Agent - Context Enricher Node
Version: 4.2
Purpose: Enrich prediction with context for interpretation
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from ..state import PredictionContext, PredictionSynthesizerState

logger = logging.getLogger(__name__)


class ContextStore(Protocol):
    """Protocol for context storage"""

    async def find_similar(
        self, entity_type: str, features: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar historical cases"""
        ...

    async def get_accuracy(self, prediction_target: str, entity_type: str) -> float:
        """Get historical accuracy for prediction type"""
        ...

    async def get_prediction_history(
        self, entity_id: str, prediction_target: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get prediction history for entity"""
        ...


class FeatureStore(Protocol):
    """Protocol for feature store"""

    async def get_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for model"""
        ...


class ContextEnricherNode:
    """
    Enrich prediction with context for interpretation.
    Fetches similar cases, feature importance, and trends.
    """

    def __init__(
        self,
        context_store: Optional[ContextStore] = None,
        feature_store: Optional[FeatureStore] = None,
    ):
        """
        Initialize context enricher.

        Args:
            context_store: Store for historical context
            feature_store: Store for feature metadata
        """
        self.context_store = context_store
        self.feature_store = feature_store

    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        """Enrich prediction with context."""
        start_time = time.time()

        if state.get("status") in ["failed", "completed"]:
            return state

        if not state.get("include_context", False):
            total_time = state.get("orchestration_latency_ms", 0) + state.get(
                "ensemble_latency_ms", 0
            )
            return {
                **state,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        try:
            # Fetch context elements in parallel
            tasks = [
                self._get_similar_cases(state),
                self._get_feature_importance(state),
                self._get_historical_accuracy(state),
                self._get_trend(state),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            similar, importance, accuracy, trend = results

            context = PredictionContext(
                similar_cases=(similar if not isinstance(similar, Exception) else []),
                feature_importance=(importance if not isinstance(importance, Exception) else {}),
                historical_accuracy=(accuracy if not isinstance(accuracy, Exception) else 0.0),
                trend_direction=(trend if not isinstance(trend, Exception) else "stable"),
            )

            context_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("orchestration_latency_ms", 0)
                + state.get("ensemble_latency_ms", 0)
                + context_time
            )

            logger.info(
                f"Context enrichment complete: "
                f"similar_cases={len(context['similar_cases'])}, "
                f"duration={context_time}ms"
            )

            return {
                **state,
                "prediction_context": context,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        except Exception as e:
            logger.warning(f"Context enrichment failed: {e}")
            total_time = state.get("orchestration_latency_ms", 0) + state.get(
                "ensemble_latency_ms", 0
            )
            return {
                **state,
                "warnings": [f"Context enrichment failed: {str(e)}"],
                "total_latency_ms": total_time,
                "status": "completed",  # Non-fatal
            }

    async def _get_similar_cases(self, state: PredictionSynthesizerState) -> List[Dict[str, Any]]:
        """Find similar historical cases."""
        if not self.context_store:
            return []

        return await self.context_store.find_similar(
            entity_type=state.get("entity_type", ""),
            features=state.get("features", {}),
            limit=5,
        )

    async def _get_feature_importance(self, state: PredictionSynthesizerState) -> Dict[str, float]:
        """Get feature importance for prediction."""
        if not self.feature_store:
            return {}

        # Aggregate importance across models
        importances: Dict[str, float] = {}
        predictions = state.get("individual_predictions", [])

        for pred in predictions:
            try:
                model_importance = await self.feature_store.get_importance(pred["model_id"])
                for feature, importance in model_importance.items():
                    if feature in importances:
                        importances[feature] = (importances[feature] + importance) / 2
                    else:
                        importances[feature] = importance
            except Exception as e:
                logger.debug(f"Failed to get importance for {pred['model_id']}: {e}")

        # Return top 10 features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])

    async def _get_historical_accuracy(self, state: PredictionSynthesizerState) -> float:
        """Get historical accuracy for this prediction type."""
        if not self.context_store:
            return 0.0

        return await self.context_store.get_accuracy(
            prediction_target=state.get("prediction_target", ""),
            entity_type=state.get("entity_type", ""),
        )

    async def _get_trend(self, state: PredictionSynthesizerState) -> str:
        """Determine trend direction."""
        if not self.context_store:
            return "stable"

        history = await self.context_store.get_prediction_history(
            entity_id=state.get("entity_id", ""),
            prediction_target=state.get("prediction_target", ""),
            limit=10,
        )

        if not history or len(history) < 3:
            return "stable"

        values = [h.get("prediction", 0) for h in history]
        if len(values) < 2:
            return "stable"

        slope = (values[-1] - values[0]) / len(values)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
