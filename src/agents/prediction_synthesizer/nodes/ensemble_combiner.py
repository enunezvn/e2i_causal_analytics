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

from typing import Any, Dict, Optional

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

            # Overall confidence with single-model safety cap
            mean_confidence = sum(confidences) / len(confidences)
            models_succeeded = len(predictions)

            # CRITICAL SAFETY: Cap confidence for single model - cannot validate
            if models_succeeded < 2:
                # Single model: cap at 30% confidence regardless of model's self-report
                ensemble_confidence = min(0.30, mean_confidence * 0.5)
                logger.warning(
                    f"SINGLE MODEL ONLY: Capping confidence at {ensemble_confidence:.0%}. "
                    f"Cannot validate prediction without model diversity."
                )
            else:
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

            # P2 Enhancement: Generate interpretation with anomaly detection
            prediction_target = state.get("prediction_target", "")
            interpretation = self._interpret_prediction(
                ensemble_pred, predictions, prediction_target
            )

            # Add anomaly warnings to state warnings
            anomaly_warnings = []
            for anomaly in interpretation.get("anomaly_flags", []):
                anomaly_warnings.append(f"[{anomaly['severity'].upper()}] {anomaly['message']}")

            # Enhanced summary with interpretation
            enhanced_summary = summary
            if interpretation.get("risk_level"):
                enhanced_summary += f"\n\nRisk Assessment: {interpretation['risk_level']}"
            if interpretation.get("recommendations"):
                top_rec = interpretation["recommendations"][0]
                enhanced_summary += f"\nRecommendation: {top_rec}"

            ensemble_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Ensemble combination complete: estimate={point_estimate:.3f}, "
                f"agreement={agreement:.2f}, reliability={interpretation.get('reliability_assessment', 'N/A')}, "
                f"anomalies={len(interpretation.get('anomaly_flags', []))}, duration={ensemble_time}ms"
            )

            return {
                **state,
                "ensemble_prediction": ensemble_pred,
                "prediction_summary": enhanced_summary,
                "prediction_interpretation": interpretation,
                "warnings": anomaly_warnings,
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
        """Calculate model agreement (1 - normalized std).

        CRITICAL: Single model CANNOT have agreement - agreement requires
        multiple viewpoints to validate. Returns 0.0 for single model.
        """
        if len(predictions) < 2:
            return 0.0  # NO agreement with single model - can't validate

        mean = sum(predictions) / len(predictions)
        if mean == 0:
            # All predictions are zero - perfect agreement on zero
            variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
            if variance == 0:
                return 1.0  # All models agree on exactly zero
            return 0.5  # Mixed predictions averaging to zero

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

    def _interpret_prediction(
        self,
        ensemble: EnsemblePrediction,
        individual: List[ModelPrediction],
        prediction_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Interpret prediction with business context and anomaly detection.

        Provides actionable interpretation of predictions including risk level,
        anomaly detection, and recommendations.

        Args:
            ensemble: The ensemble prediction result
            individual: List of individual model predictions
            prediction_target: What was being predicted (e.g., "churn", "conversion")

        Returns:
            Interpretation dictionary with risk, anomalies, recommendations
        """
        pred = ensemble["point_estimate"]
        confidence = ensemble["confidence"]
        agreement = ensemble["model_agreement"]
        lower = ensemble["prediction_interval_lower"]
        upper = ensemble["prediction_interval_upper"]

        interpretation: Dict[str, Any] = {
            "risk_level": "",
            "anomaly_flags": [],
            "recommendations": [],
            "confidence_explanation": "",
            "reliability_assessment": "",
            "action_urgency": "",
        }

        # ===== Anomaly Detection =====

        # CRITICAL: Detect single-model predictions (agreement=0.0 means single model)
        models_count = len(individual) if individual else 0
        if models_count < 2:
            interpretation["anomaly_flags"].append({
                "type": "single_model_prediction",
                "severity": "critical",
                "message": (
                    f"Only {models_count} model(s) succeeded. Cannot validate prediction "
                    f"without model diversity. Ensemble confidence capped at 30%."
                ),
            })
            interpretation["reliability_assessment"] = "UNVALIDATED"

            # Special case: single model with zero prediction
            if pred == 0.0:
                interpretation["anomaly_flags"].append({
                    "type": "single_model_zero_prediction",
                    "severity": "critical",
                    "message": (
                        "Zero prediction from single model - cannot validate. "
                        "This could indicate model failure, missing features, or data issues."
                    ),
                })
                interpretation["recommendations"] = [
                    "DO NOT act on this prediction - insufficient model diversity",
                    "Run additional models before making decisions",
                    "Investigate why other models failed",
                ]
                return interpretation  # STOP - don't generate normal recommendations

            # Single model non-zero: still unreliable
            interpretation["recommendations"].insert(
                0, "CAUTION: Single-model prediction requires validation before action"
            )

        # Check for extreme disagreement with extreme values
        elif pred == 0.0 and agreement < 0.3:
            interpretation["anomaly_flags"].append({
                "type": "extreme_disagreement",
                "severity": "critical",
                "message": (
                    f"Prediction of 0.0 with only {agreement:.0%} model agreement is anomalous. "
                    f"Models radically disagree - prediction is unreliable."
                ),
            })
            interpretation["recommendations"].append(
                "DO NOT act on this prediction. Investigate model disagreement root cause."
            )
            interpretation["reliability_assessment"] = "UNRELIABLE"

        # Check for zero prediction when models disagree significantly
        elif pred == 0.0 and agreement < 0.5:
            interpretation["anomaly_flags"].append({
                "type": "zero_with_disagreement",
                "severity": "warning",
                "message": (
                    f"Zero prediction with moderate disagreement ({agreement:.0%}) suggests "
                    f"possible data quality issues or model calibration problems."
                ),
            })
            interpretation["recommendations"].append(
                "Validate input features and check for missing data before acting on this prediction."
            )

        # Check for extremely wide prediction intervals
        interval_width = upper - lower
        if interval_width > 0.5:
            interpretation["anomaly_flags"].append({
                "type": "high_uncertainty",
                "severity": "warning",
                "message": (
                    f"Wide prediction interval ({lower:.2f} to {upper:.2f}) indicates "
                    f"significant uncertainty. Consider gathering more data."
                ),
            })

        # Check for prediction near decision boundary with low confidence
        if 0.45 <= pred <= 0.55 and confidence < 0.5:
            interpretation["anomaly_flags"].append({
                "type": "boundary_uncertainty",
                "severity": "warning",
                "message": (
                    f"Prediction {pred:.2f} near 0.5 decision boundary with low confidence "
                    f"({confidence:.0%}). Classification outcome is uncertain."
                ),
            })

        # Check for high prediction with low agreement
        if pred > 0.7 and agreement < 0.4:
            interpretation["anomaly_flags"].append({
                "type": "high_pred_low_agreement",
                "severity": "warning",
                "message": (
                    f"High prediction ({pred:.2f}) but poor model agreement ({agreement:.0%}). "
                    f"Some models may have significantly different views."
                ),
            })

        # ===== Risk Level Assessment =====
        # GUARD: Don't generate action recommendations if prediction is unreliable
        if interpretation["reliability_assessment"] in ("UNRELIABLE", "UNVALIDATED"):
            # Already have critical warnings - don't add misleading action recommendations
            interpretation["risk_level"] = "CANNOT_ASSESS"
            interpretation["action_urgency"] = "blocked"
            # Ensure we have at least one actionable recommendation
            if not interpretation["recommendations"]:
                interpretation["recommendations"].append(
                    "INSUFFICIENT DATA - No recommendation can be made safely"
                )
            return interpretation  # Don't proceed to action recommendations

        # Determine risk based on prediction value (assuming higher = higher risk for churn-like predictions)
        target_lower = (prediction_target or "").lower()
        is_negative_outcome = any(word in target_lower for word in ["churn", "discontinuation", "attrition", "loss", "risk"])

        if is_negative_outcome:
            # Higher prediction = higher risk (e.g., churn probability)
            if pred > 0.7:
                interpretation["risk_level"] = "HIGH"
                interpretation["action_urgency"] = "immediate"
                interpretation["recommendations"].append(
                    "Immediate intervention recommended - high discontinuation risk detected."
                )
            elif pred > 0.4:
                interpretation["risk_level"] = "MODERATE"
                interpretation["action_urgency"] = "short_term"
                interpretation["recommendations"].append(
                    "Monitor closely and prepare intervention if trend continues."
                )
            else:
                interpretation["risk_level"] = "LOW"
                interpretation["action_urgency"] = "routine"
                interpretation["recommendations"].append(
                    "Continue standard engagement - no immediate action needed."
                )
        else:
            # For positive outcomes (conversion, adoption), lower prediction = concern
            if pred > 0.7:
                interpretation["risk_level"] = "FAVORABLE"
                interpretation["action_urgency"] = "opportunity"
                interpretation["recommendations"].append(
                    "High probability of positive outcome - consider resource prioritization."
                )
            elif pred > 0.4:
                interpretation["risk_level"] = "MODERATE"
                interpretation["action_urgency"] = "standard"
                interpretation["recommendations"].append(
                    "Moderate probability - standard engagement approach recommended."
                )
            else:
                interpretation["risk_level"] = "LOW_PROBABILITY"
                interpretation["action_urgency"] = "evaluate"
                interpretation["recommendations"].append(
                    "Low predicted probability - evaluate if additional effort is worthwhile."
                )

        # ===== Confidence Explanation =====
        if confidence >= 0.8:
            interpretation["confidence_explanation"] = (
                f"High confidence ({confidence:.0%}) - prediction is well-supported by multiple "
                f"models with strong agreement. Suitable for decision-making."
            )
        elif confidence >= 0.5:
            interpretation["confidence_explanation"] = (
                f"Moderate confidence ({confidence:.0%}) - prediction has reasonable support "
                f"but some model disagreement exists. Use as directional guidance."
            )
        else:
            interpretation["confidence_explanation"] = (
                f"Low confidence ({confidence:.0%}) - prediction should be treated as directional only. "
                f"Consider gathering additional data or consulting domain experts."
            )

        # ===== Reliability Assessment =====
        if interpretation["reliability_assessment"] == "":  # Not already set by anomaly detection
            if confidence > 0.7 and agreement > 0.8 and not interpretation["anomaly_flags"]:
                interpretation["reliability_assessment"] = "HIGH"
            elif confidence > 0.4 and agreement > 0.5 and len(interpretation["anomaly_flags"]) <= 1:
                interpretation["reliability_assessment"] = "MODERATE"
            else:
                interpretation["reliability_assessment"] = "LOW"

        # ===== Additional Recommendations Based on Model Contributions =====
        if individual:
            # Check if any single model is an outlier
            pred_values = [p["prediction"] for p in individual]
            mean_pred = sum(pred_values) / len(pred_values)
            outliers = [
                p for p in individual
                if abs(p["prediction"] - mean_pred) > 0.3
            ]
            if outliers:
                outlier_ids = [o["model_id"] for o in outliers]
                interpretation["recommendations"].append(
                    f"Review outlier models ({', '.join(outlier_ids)}) for potential calibration issues."
                )

        return interpretation
