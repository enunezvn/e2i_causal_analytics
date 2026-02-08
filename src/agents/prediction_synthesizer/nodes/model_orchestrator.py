"""
E2I Prediction Synthesizer Agent - Model Orchestrator Node
Version: 4.2
Purpose: Orchestrate predictions from multiple models in parallel
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from ..state import ModelPrediction, PredictionSynthesizerState

logger = logging.getLogger(__name__)


class ModelRegistry(Protocol):
    """Protocol for model registry"""

    async def get_models_for_target(self, target: str, entity_type: str) -> List[str]:
        """Get list of models for a prediction target"""
        ...


class ModelClient(Protocol):
    """Protocol for model prediction client"""

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Get prediction from model"""
        ...


class ModelOrchestratorNode:
    """
    Orchestrate predictions from multiple models in parallel.
    Handles timeouts, failures, and result aggregation.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        model_clients: Optional[Dict[str, ModelClient]] = None,
        timeout_per_model: float = 5.0,
    ):
        """
        Initialize model orchestrator.

        Args:
            model_registry: Registry of available models
            model_clients: Dict mapping model_id to client
            timeout_per_model: Timeout in seconds for each model
        """
        self.registry = model_registry
        self.clients = model_clients or {}
        self.timeout_per_model = timeout_per_model

    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        """Orchestrate predictions from multiple models."""
        # Check if already failed or completed
        if state.get("status") in ["failed", "completed"]:
            return state

        start_time = time.time()

        try:
            # Determine which models to use
            models_to_use = state.get("models_to_use")
            if not models_to_use and self.registry:
                models_to_use = await self.registry.get_models_for_target(
                    target=state.get("prediction_target", ""),
                    entity_type=state.get("entity_type", ""),
                )

            if not models_to_use:
                # No registry or no models - return mock predictions for testing
                models_to_use = list(self.clients.keys()) if self.clients else []

            if not models_to_use:
                logger.warning("No models available for prediction")
                return {
                    **state,
                    "errors": [
                        {
                            "node": "orchestrator",
                            "error": "No models available for this prediction target",
                        }
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }

            # Run predictions in parallel
            tasks = [self._get_prediction(model_id, state) for model_id in models_to_use]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            predictions: List[ModelPrediction] = []
            succeeded = 0
            failed = 0
            new_warnings = []

            for model_id, result in zip(models_to_use, results, strict=False):
                if isinstance(result, BaseException):
                    failed += 1
                    new_warnings.append(f"Model {model_id} failed: {str(result)}")
                    logger.warning(f"Model {model_id} prediction failed: {result}")
                elif result is not None:
                    predictions.append(result)
                    succeeded += 1
                else:
                    failed += 1

            orchestration_time = int((time.time() - start_time) * 1000)

            if not predictions:
                return {
                    **state,
                    "individual_predictions": [],
                    "models_succeeded": succeeded,
                    "models_failed": failed,
                    "orchestration_latency_ms": orchestration_time,
                    "errors": [{"node": "orchestrator", "error": "All models failed"}],
                    "warnings": new_warnings,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }

            logger.info(
                f"Model orchestration complete: {succeeded} succeeded, "
                f"{failed} failed, duration={orchestration_time}ms"
            )

            return {
                **state,
                "individual_predictions": predictions,
                "models_succeeded": succeeded,
                "models_failed": failed,
                "orchestration_latency_ms": orchestration_time,
                "warnings": new_warnings,
                "status": "combining",
            }

        except Exception as e:
            logger.error(f"Model orchestration failed: {e}")
            return {
                **state,
                "errors": [{"node": "orchestrator", "error": str(e)}],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
            }

    async def _get_prediction(
        self,
        model_id: str,
        state: PredictionSynthesizerState,
    ) -> Optional[ModelPrediction]:
        """Get prediction from a single model."""
        start = time.time()

        client = self.clients.get(model_id)
        if not client:
            # No client - create mock prediction for testing
            return self._create_mock_prediction(model_id, start)

        try:
            result = await asyncio.wait_for(
                client.predict(
                    entity_id=state.get("entity_id", ""),
                    features=state.get("features", {}),
                    time_horizon=state.get("time_horizon", "30d"),
                ),
                timeout=self.timeout_per_model,
            )

            latency = int((time.time() - start) * 1000)

            return ModelPrediction(
                model_id=model_id,
                model_type=result.get("model_type", "unknown"),
                prediction=result["prediction"],
                prediction_proba=result.get("proba"),
                confidence=result.get("confidence", 0.5),
                latency_ms=latency,
                features_used=result.get("features_used", []),
            )

        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Model {model_id} timed out") from e

    def _create_mock_prediction(self, model_id: str, start_time: float) -> ModelPrediction:
        """Create mock prediction for testing."""
        import random

        latency = int((time.time() - start_time) * 1000) + 10
        return ModelPrediction(
            model_id=model_id,
            model_type="mock",
            prediction=random.uniform(0.3, 0.8),
            prediction_proba=None,
            confidence=random.uniform(0.7, 0.95),
            latency_ms=latency,
            features_used=["feature_1", "feature_2"],
        )
