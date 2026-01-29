"""
E2I Prediction Synthesizer Agent - Ensemble Combiner Node
Version: 4.2
Purpose: Combine individual model predictions into ensemble
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List

from ..state import EnsemblePrediction, ModelPrediction, PredictionSynthesizerState

logger = logging.getLogger(__name__)


class EnsembleCombinerNode:
    """
    Combine individual model predictions into ensemble.
    Supports multiple aggregation methods with uncertainty quantification.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ensemble combiner.

        Args:
            confidence_level: Default confidence level for intervals
        """
        self.default_confidence_level = confidence_level

    async def execute(self, state: PredictionSynthesizerState) -> PredictionSynthesizerState:
        """Combine predictions into ensemble."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            predictions = state.get("individual_predictions", [])
            if not predictions:
                return {
                    **state,
                    "errors": [{"node": "ensemble", "error": "No predictions to combine"}],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }

            method = state.get("ensemble_method", "weighted")
            confidence_level = state.get("confidence_level", self.default_confidence_level)

            # Extract prediction values and confidences
            pred_values = [p["prediction"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]

            # Combine based on method
            point_estimate = self._combine_predictions(pred_values, confidences, method)

            # Calculate prediction interval
            interval_lower, interval_upper = self._calculate_interval(
                pred_values, point_estimate, confidence_level
            )

            # Calculate model agreement
            agreement = self._calculate_agreement(pred_values)

            # Overall confidence
            mean_confidence = sum(confidences) / len(confidences)
            ensemble_confidence = mean_confidence * agreement

            ensemble_pred = EnsemblePrediction(
                point_estimate=point_estimate,
                prediction_interval_lower=interval_lower,
                prediction_interval_upper=interval_upper,
                confidence=ensemble_confidence,
                ensemble_method=method,
                model_agreement=agreement,
            )

            # Generate summary
            summary = self._generate_summary(ensemble_pred, predictions)

            ensemble_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Ensemble combination complete: estimate={point_estimate:.3f}, "
                f"agreement={agreement:.2f}, duration={ensemble_time}ms"
            )

            return {
                **state,
                "ensemble_prediction": ensemble_pred,
                "prediction_summary": summary,
                "ensemble_latency_ms": ensemble_time,
                "status": "enriching" if state.get("include_context") else "completed",
            }

        except Exception as e:
            logger.error(f"Ensemble combination failed: {e}")
            return {
                **state,
                "errors": [{"node": "ensemble", "error": str(e)}],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
            }

    def _combine_predictions(
        self,
        predictions: List[float],
        confidences: List[float],
        method: str,
    ) -> float:
        """Combine predictions using specified method."""
        if not predictions:
            return 0.0

        if method == "average":
            return sum(predictions) / len(predictions)

        elif method == "weighted":
            # Weight by confidence
            total_conf = sum(confidences)
            if total_conf == 0:
                return sum(predictions) / len(predictions)
            weights = [c / total_conf for c in confidences]
            return sum(p * w for p, w in zip(predictions, weights, strict=False))

        elif method == "voting":
            # For classification - round to 0/1 and majority vote
            rounded = [round(p) for p in predictions]
            return max(set(rounded), key=rounded.count)

        elif method == "stacking":
            # Simple average for now - full stacking requires meta-learner
            return sum(predictions) / len(predictions)

        else:
            # Default to average
            return sum(predictions) / len(predictions)

    def _calculate_interval(
        self,
        predictions: List[float],
        point_estimate: float,
        confidence_level: float,
    ) -> tuple:
        """Calculate prediction interval."""
        if len(predictions) < 2:
            return point_estimate - 0.1, point_estimate + 0.1

        # Calculate standard deviation
        mean = sum(predictions) / len(predictions)
        variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        std = variance**0.5

        # Z-score for confidence level (approximation)
        # 0.95 -> 1.96, 0.90 -> 1.645, 0.99 -> 2.576
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)

        interval_lower = point_estimate - z * std
        interval_upper = point_estimate + z * std

        return interval_lower, interval_upper

    def _calculate_agreement(self, predictions: List[float]) -> float:
        """Calculate model agreement (1 - normalized std)."""
        if len(predictions) < 2:
            return 1.0

        mean = sum(predictions) / len(predictions)
        if mean == 0:
            return 1.0

        variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        std = variance**0.5

        # Coefficient of variation
        cv = std / abs(mean)

        # Agreement is inverse of CV, capped at 0-1
        agreement = max(0.0, min(1.0, 1 - cv))
        return agreement

    def _generate_summary(
        self,
        ensemble: EnsemblePrediction,
        individual: List[ModelPrediction],
    ) -> str:
        """Generate prediction summary."""
        pred = ensemble["point_estimate"]
        lower = ensemble["prediction_interval_lower"]
        upper = ensemble["prediction_interval_upper"]
        agreement = ensemble["model_agreement"]
        confidence = ensemble["confidence"]

        confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        agreement_desc = "strong" if agreement > 0.8 else "moderate" if agreement > 0.5 else "weak"

        summary = f"Prediction: {pred:.3f} (95% CI: [{lower:.3f}, {upper:.3f}]). "
        summary += f"Confidence: {confidence_desc}. "
        summary += f"Model agreement: {agreement_desc} across {len(individual)} models."

        return summary
