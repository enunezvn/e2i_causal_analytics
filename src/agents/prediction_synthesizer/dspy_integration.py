"""
E2I Prediction Synthesizer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for prediction_synthesizer Sender role

The Prediction Synthesizer agent is a DSPy Sender agent that:
1. Generates training signals from prediction aggregation executions
2. Provides EvidenceSynthesisSignature training examples
3. Routes high-quality signals to feedback_learner for optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TRAINING SIGNAL STRUCTURE
# =============================================================================


@dataclass
class PredictionSynthesisTrainingSignal:
    """
    Training signal for Prediction Synthesizer DSPy optimization.

    Captures prediction aggregation decisions and their outcomes to train:
    - EvidenceSynthesisSignature: Synthesizing multi-model evidence
    - PredictionInterpretationSignature: Explaining predictions
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    entity_id: str = ""
    entity_type: str = ""  # hcp, territory, patient
    prediction_target: str = ""
    time_horizon: str = ""

    # === Model Orchestration ===
    models_requested: int = 0
    models_succeeded: int = 0
    models_failed: int = 0
    ensemble_method: str = ""  # average, weighted, stacking, voting

    # === Ensemble Results ===
    point_estimate: float = 0.0
    prediction_interval_width: float = 0.0
    ensemble_confidence: float = 0.0
    model_agreement: float = 0.0

    # === Context Enrichment ===
    similar_cases_found: int = 0
    feature_importance_calculated: bool = False
    historical_accuracy: float = 0.0
    trend_direction: str = ""  # increasing, stable, decreasing

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    orchestration_latency_ms: float = 0.0
    ensemble_latency_ms: float = 0.0
    prediction_accuracy: Optional[float] = None  # Validated later
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - model_success_rate: 0.25 (models succeeded / requested)
        - ensemble_quality: 0.25 (confidence + agreement)
        - efficiency: 0.15 (latency)
        - context_quality: 0.15 (enrichment completeness)
        - accuracy/satisfaction: 0.20 (if available)
        """
        reward = 0.0

        # Model success rate
        if self.models_requested > 0:
            success_rate = self.models_succeeded / self.models_requested
            reward += 0.25 * success_rate

        # Ensemble quality
        ensemble_score = 0.0
        # High confidence is good
        ensemble_score += 0.5 * self.ensemble_confidence
        # High model agreement is good
        ensemble_score += 0.3 * self.model_agreement
        # Narrow prediction interval is good (relative to point estimate)
        if self.point_estimate != 0:
            relative_width = self.prediction_interval_width / abs(self.point_estimate)
            # Target: interval width < 50% of estimate
            interval_score = min(1.0, 0.5 / max(relative_width, 0.1))
            ensemble_score += 0.2 * interval_score
        reward += 0.25 * ensemble_score

        # Efficiency (target < 5s for full prediction)
        target_latency = 5000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.15 * efficiency
        else:
            reward += 0.15

        # Context quality
        context_score = 0.0
        if self.similar_cases_found > 0:
            context_score += 0.4
        if self.feature_importance_calculated:
            context_score += 0.3
        if self.trend_direction:
            context_score += 0.3
        reward += 0.15 * context_score

        # Accuracy / Satisfaction
        if self.prediction_accuracy is not None:
            # Accuracy is a measure of how close prediction was to actual
            reward += 0.20 * self.prediction_accuracy
        elif self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.20 * satisfaction_score
        else:
            reward += 0.10  # Partial credit

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"ps_{self.session_id}_{self.created_at}",
            "source_agent": "prediction_synthesizer",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "prediction_target": self.prediction_target,
                "time_horizon": self.time_horizon,
            },
            "model_orchestration": {
                "models_requested": self.models_requested,
                "models_succeeded": self.models_succeeded,
                "models_failed": self.models_failed,
                "ensemble_method": self.ensemble_method,
            },
            "ensemble_results": {
                "point_estimate": self.point_estimate,
                "prediction_interval_width": self.prediction_interval_width,
                "ensemble_confidence": self.ensemble_confidence,
                "model_agreement": self.model_agreement,
            },
            "context_enrichment": {
                "similar_cases_found": self.similar_cases_found,
                "feature_importance_calculated": self.feature_importance_calculated,
                "historical_accuracy": self.historical_accuracy,
                "trend_direction": self.trend_direction,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "orchestration_latency_ms": self.orchestration_latency_ms,
                "ensemble_latency_ms": self.ensemble_latency_ms,
                "prediction_accuracy": self.prediction_accuracy,
                "user_satisfaction": self.user_satisfaction,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class PredictionSynthesisSignature(dspy.Signature):
        """
        Synthesize multiple model predictions into ensemble.

        Given predictions from multiple models, determine the best
        way to combine them into a single prediction with uncertainty.
        """

        individual_predictions: str = dspy.InputField(desc="Predictions from each model")
        model_metadata: str = dspy.InputField(desc="Model types and historical accuracy")
        entity_context: str = dspy.InputField(desc="Context about the entity being predicted")

        ensemble_method: str = dspy.OutputField(desc="average, weighted, stacking, or voting")
        weighting_rationale: str = dspy.OutputField(desc="If weighted, why these weights")
        confidence_assessment: str = dspy.OutputField(desc="Assessment of prediction confidence")
        disagreement_notes: list = dspy.OutputField(desc="Notes on model disagreements")

    class PredictionInterpretationSignature(dspy.Signature):
        """
        Generate human-readable interpretation of prediction.

        Create actionable summary of prediction results with context.
        """

        prediction_summary: str = dspy.InputField(desc="Point estimate and interval")
        model_agreement: float = dspy.InputField(desc="How much models agree (0-1)")
        feature_importance: str = dspy.InputField(desc="Key features driving prediction")
        historical_context: str = dspy.InputField(desc="Similar past cases and outcomes")

        interpretation: str = dspy.OutputField(desc="Natural language interpretation")
        confidence_level: str = dspy.OutputField(desc="high, moderate, or low confidence")
        key_drivers: list = dspy.OutputField(desc="Top factors affecting prediction")
        recommendations: list = dspy.OutputField(desc="Suggested actions based on prediction")

    class UncertaintyQuantificationSignature(dspy.Signature):
        """
        Quantify uncertainty in ensemble predictions.

        Assess sources of uncertainty and provide calibrated intervals.
        """

        individual_predictions: str = dspy.InputField(desc="Predictions from each model")
        model_uncertainties: str = dspy.InputField(desc="Each model's uncertainty estimate")
        historical_calibration: str = dspy.InputField(desc="How well calibrated past predictions were")

        prediction_interval: str = dspy.OutputField(desc="Lower and upper bounds")
        confidence_level: float = dspy.OutputField(desc="Probability interval contains true value")
        uncertainty_sources: list = dspy.OutputField(desc="Main sources of uncertainty")
        calibration_warning: str = dspy.OutputField(desc="Any calibration concerns")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Prediction Synthesizer agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic synthesis")
    PredictionSynthesisSignature = None
    PredictionInterpretationSignature = None
    UncertaintyQuantificationSignature = None


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class PredictionSynthesizerSignalCollector:
    """
    Collects training signals from prediction synthesis executions.

    The Prediction Synthesizer agent is a Sender that generates signals
    for EvidenceSynthesisSignature optimization.
    """

    def __init__(self):
        self._signals_buffer: List[PredictionSynthesisTrainingSignal] = []
        self._buffer_limit = 100

    def collect_synthesis_signal(
        self,
        session_id: str,
        query: str,
        entity_id: str,
        entity_type: str,
        prediction_target: str,
        time_horizon: str,
        models_requested: int,
    ) -> PredictionSynthesisTrainingSignal:
        """
        Initialize training signal at synthesis start.

        Call this when starting a new prediction synthesis.
        """
        signal = PredictionSynthesisTrainingSignal(
            session_id=session_id,
            query=query,
            entity_id=entity_id,
            entity_type=entity_type,
            prediction_target=prediction_target,
            time_horizon=time_horizon,
            models_requested=models_requested,
        )
        return signal

    def update_model_orchestration(
        self,
        signal: PredictionSynthesisTrainingSignal,
        models_succeeded: int,
        models_failed: int,
        ensemble_method: str,
        orchestration_latency_ms: float,
    ) -> PredictionSynthesisTrainingSignal:
        """Update signal with model orchestration results."""
        signal.models_succeeded = models_succeeded
        signal.models_failed = models_failed
        signal.ensemble_method = ensemble_method
        signal.orchestration_latency_ms = orchestration_latency_ms
        return signal

    def update_ensemble_results(
        self,
        signal: PredictionSynthesisTrainingSignal,
        point_estimate: float,
        prediction_interval_lower: float,
        prediction_interval_upper: float,
        ensemble_confidence: float,
        model_agreement: float,
        ensemble_latency_ms: float,
    ) -> PredictionSynthesisTrainingSignal:
        """Update signal with ensemble results."""
        signal.point_estimate = point_estimate
        signal.prediction_interval_width = prediction_interval_upper - prediction_interval_lower
        signal.ensemble_confidence = ensemble_confidence
        signal.model_agreement = model_agreement
        signal.ensemble_latency_ms = ensemble_latency_ms
        return signal

    def update_context_enrichment(
        self,
        signal: PredictionSynthesisTrainingSignal,
        similar_cases_found: int,
        feature_importance_calculated: bool,
        historical_accuracy: float,
        trend_direction: str,
        total_latency_ms: float,
    ) -> PredictionSynthesisTrainingSignal:
        """Update signal with context enrichment results."""
        signal.similar_cases_found = similar_cases_found
        signal.feature_importance_calculated = feature_importance_calculated
        signal.historical_accuracy = historical_accuracy
        signal.trend_direction = trend_direction
        signal.total_latency_ms = total_latency_ms

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_accuracy(
        self,
        signal: PredictionSynthesisTrainingSignal,
        prediction_accuracy: float,
        user_satisfaction: Optional[float] = None,
    ) -> PredictionSynthesisTrainingSignal:
        """Update signal with accuracy validation (delayed feedback)."""
        signal.prediction_accuracy = prediction_accuracy
        signal.user_satisfaction = user_satisfaction
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [
            s.to_dict()
            for s in self._signals_buffer
            if s.compute_reward() >= min_reward
        ]
        return signals[-limit:]

    def get_accurate_examples(
        self,
        min_accuracy: float = 0.8,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples with high prediction accuracy (ground truth)."""
        signals = [
            s for s in self._signals_buffer
            if s.prediction_accuracy is not None and s.prediction_accuracy >= min_accuracy
        ]
        sorted_signals = sorted(signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[PredictionSynthesizerSignalCollector] = None


def get_prediction_synthesizer_signal_collector() -> PredictionSynthesizerSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = PredictionSynthesizerSignalCollector()
    return _signal_collector


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _signal_collector
    _signal_collector = None


# =============================================================================
# 5. EXPORTS
# =============================================================================

__all__ = [
    # Training Signals
    "PredictionSynthesisTrainingSignal",
    # DSPy Signatures
    "PredictionSynthesisSignature",
    "PredictionInterpretationSignature",
    "UncertaintyQuantificationSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "PredictionSynthesizerSignalCollector",
    # Access
    "get_prediction_synthesizer_signal_collector",
    "reset_dspy_integration",
]
