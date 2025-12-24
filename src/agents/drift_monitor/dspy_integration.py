"""
E2I Drift Monitor Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for drift_monitor Sender role

The Drift Monitor agent is a DSPy Sender agent that:
1. Generates training signals from drift detection executions
2. Provides HopDecisionSignature training examples
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
class DriftDetectionTrainingSignal:
    """
    Training signal for Drift Monitor DSPy optimization.

    Captures drift detection decisions and their outcomes to train:
    - HopDecisionSignature: Deciding when to escalate drift
    - DriftInterpretationSignature: Explaining drift implications
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    model_id: str = ""
    features_monitored: int = 0
    time_window: str = ""

    # === Detection Configuration ===
    check_data_drift: bool = True
    check_model_drift: bool = False
    check_concept_drift: bool = False
    psi_threshold: float = 0.1
    significance_level: float = 0.05

    # === Detection Results ===
    data_drift_count: int = 0
    model_drift_count: int = 0
    concept_drift_count: int = 0
    overall_drift_score: float = 0.0
    severity_distribution: Dict[str, int] = field(default_factory=dict)  # none, low, medium, high, critical

    # === Alert Generation ===
    alerts_generated: int = 0
    critical_alerts: int = 0
    warnings: int = 0
    recommended_actions_count: int = 0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    features_checked: int = 0
    drift_correctly_identified: Optional[bool] = None  # Validated later
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - detection_accuracy: 0.30 (drift correctly identified if validated)
        - alerting_quality: 0.25 (appropriate severity assignment)
        - efficiency: 0.20 (latency per feature)
        - actionability: 0.15 (recommendations provided)
        - user_satisfaction: 0.10 (if available)
        """
        reward = 0.0

        # Detection accuracy (if validated)
        if self.drift_correctly_identified is not None:
            reward += 0.30 if self.drift_correctly_identified else 0.0
        else:
            # Proxy: appropriate drift detection rate
            if self.features_monitored > 0:
                total_drift = self.data_drift_count + self.model_drift_count + self.concept_drift_count
                drift_rate = total_drift / self.features_monitored
                # Ideal: 5-20% drift rate (not too few, not too many)
                if 0.05 <= drift_rate <= 0.20:
                    reward += 0.30
                elif drift_rate < 0.05:
                    reward += 0.20  # May be missing drift
                else:
                    reward += 0.15  # May have too many false positives

        # Alerting quality
        alerting_score = 0.0
        total_drift = self.data_drift_count + self.model_drift_count + self.concept_drift_count
        if total_drift > 0:
            # Should generate alerts for critical drift
            alert_rate = self.alerts_generated / max(1, total_drift)
            alerting_score += 0.5 * min(1.0, alert_rate)
            # Critical alerts should be rare but not zero
            critical_rate = self.critical_alerts / max(1, self.alerts_generated) if self.alerts_generated > 0 else 0
            if 0.1 <= critical_rate <= 0.3:
                alerting_score += 0.5
            else:
                alerting_score += 0.3
        reward += 0.25 * alerting_score

        # Efficiency (target < 200ms per feature)
        if self.features_checked > 0:
            latency_per_feature = self.total_latency_ms / self.features_checked
            efficiency = min(1.0, 200 / latency_per_feature) if latency_per_feature > 0 else 1.0
            reward += 0.20 * efficiency
        else:
            reward += 0.10

        # Actionability
        if self.recommended_actions_count > 0:
            actionability = min(1.0, self.recommended_actions_count / max(1, self.alerts_generated))
            reward += 0.15 * actionability
        else:
            reward += 0.05

        # User satisfaction
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.10 * satisfaction_score
        else:
            reward += 0.05  # Partial credit

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"dm_{self.session_id}_{self.created_at}",
            "source_agent": "drift_monitor",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "model_id": self.model_id,
                "features_monitored": self.features_monitored,
                "time_window": self.time_window,
            },
            "configuration": {
                "check_data_drift": self.check_data_drift,
                "check_model_drift": self.check_model_drift,
                "check_concept_drift": self.check_concept_drift,
                "psi_threshold": self.psi_threshold,
                "significance_level": self.significance_level,
            },
            "detection_results": {
                "data_drift_count": self.data_drift_count,
                "model_drift_count": self.model_drift_count,
                "concept_drift_count": self.concept_drift_count,
                "overall_drift_score": self.overall_drift_score,
                "severity_distribution": self.severity_distribution,
            },
            "alerting": {
                "alerts_generated": self.alerts_generated,
                "critical_alerts": self.critical_alerts,
                "warnings": self.warnings,
                "recommended_actions_count": self.recommended_actions_count,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "features_checked": self.features_checked,
                "drift_correctly_identified": self.drift_correctly_identified,
                "user_satisfaction": self.user_satisfaction,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class DriftDetectionSignature(dspy.Signature):
        """
        Determine if detected drift requires action.

        Given statistical drift metrics, decide the severity level
        and whether to generate alerts.
        """

        feature_name: str = dspy.InputField(desc="Name of the feature being monitored")
        drift_statistic: float = dspy.InputField(desc="PSI or other drift statistic")
        p_value: float = dspy.InputField(desc="Statistical significance of drift")
        historical_context: str = dspy.InputField(desc="Historical drift patterns")

        drift_detected: bool = dspy.OutputField(desc="Whether meaningful drift is present")
        severity: str = dspy.OutputField(desc="none, low, medium, high, or critical")
        action_needed: bool = dspy.OutputField(desc="Whether action is recommended")
        rationale: str = dspy.OutputField(desc="Explanation of the decision")

    class HopDecisionSignature(dspy.Signature):
        """
        Decide whether to escalate drift to next investigation hop.

        Given current drift evidence, determine if further investigation
        is needed (concept drift, model retraining, etc.).
        """

        drift_summary: str = dspy.InputField(desc="Summary of detected drift")
        drift_score: float = dspy.InputField(desc="Overall drift score (0-1)")
        alerts_generated: int = dspy.InputField(desc="Number of alerts raised")
        model_context: str = dspy.InputField(desc="Model metadata and history")

        escalate: bool = dspy.OutputField(desc="Whether to escalate investigation")
        escalation_type: str = dspy.OutputField(
            desc="model_retrain, concept_drift_analysis, data_quality_review, or none"
        )
        urgency: str = dspy.OutputField(desc="immediate, scheduled, or monitor")
        next_steps: list = dspy.OutputField(desc="Recommended next actions")

    class DriftInterpretationSignature(dspy.Signature):
        """
        Generate human-readable interpretation of drift findings.

        Create actionable summary for stakeholders about drift implications.
        """

        features_with_drift: str = dspy.InputField(desc="List of drifting features")
        drift_types: str = dspy.InputField(desc="Types of drift detected")
        overall_score: float = dspy.InputField(desc="Aggregate drift score")
        model_performance: str = dspy.InputField(desc="Current model performance metrics")

        summary: str = dspy.OutputField(desc="Executive summary of drift situation")
        business_impact: str = dspy.OutputField(desc="Impact on business outcomes")
        recommended_actions: list = dspy.OutputField(desc="Prioritized action items")
        monitoring_adjustments: list = dspy.OutputField(desc="Changes to monitoring")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Drift Monitor agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic drift detection")
    DriftDetectionSignature = None
    HopDecisionSignature = None
    DriftInterpretationSignature = None


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class DriftMonitorSignalCollector:
    """
    Collects training signals from drift detection executions.

    The Drift Monitor agent is a Sender that generates signals
    for HopDecisionSignature optimization.
    """

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[DriftDetectionTrainingSignal] = []
        self._buffer_limit = 100

    def collect_detection_signal(
        self,
        session_id: str,
        query: str,
        model_id: str,
        features_monitored: int,
        time_window: str,
        check_data_drift: bool,
        check_model_drift: bool,
        check_concept_drift: bool,
    ) -> DriftDetectionTrainingSignal:
        """
        Initialize training signal at detection start.

        Call this when starting a new drift detection run.
        """
        signal = DriftDetectionTrainingSignal(
            session_id=session_id,
            query=query,
            model_id=model_id,
            features_monitored=features_monitored,
            time_window=time_window,
            check_data_drift=check_data_drift,
            check_model_drift=check_model_drift,
            check_concept_drift=check_concept_drift,
        )
        return signal

    def update_detection_results(
        self,
        signal: DriftDetectionTrainingSignal,
        data_drift_count: int,
        model_drift_count: int,
        concept_drift_count: int,
        overall_drift_score: float,
        severity_distribution: Dict[str, int],
        features_checked: int,
    ) -> DriftDetectionTrainingSignal:
        """Update signal with detection results."""
        signal.data_drift_count = data_drift_count
        signal.model_drift_count = model_drift_count
        signal.concept_drift_count = concept_drift_count
        signal.overall_drift_score = overall_drift_score
        signal.severity_distribution = severity_distribution
        signal.features_checked = features_checked
        return signal

    def update_alerting(
        self,
        signal: DriftDetectionTrainingSignal,
        alerts_generated: int,
        critical_alerts: int,
        warnings: int,
        recommended_actions_count: int,
        total_latency_ms: float,
    ) -> DriftDetectionTrainingSignal:
        """Update signal with alerting results."""
        signal.alerts_generated = alerts_generated
        signal.critical_alerts = critical_alerts
        signal.warnings = warnings
        signal.recommended_actions_count = recommended_actions_count
        signal.total_latency_ms = total_latency_ms

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_validation(
        self,
        signal: DriftDetectionTrainingSignal,
        drift_correctly_identified: bool,
        user_satisfaction: Optional[float] = None,
    ) -> DriftDetectionTrainingSignal:
        """Update signal with validation (delayed feedback)."""
        signal.drift_correctly_identified = drift_correctly_identified
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

    def get_validated_examples(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples with validated drift detection (ground truth)."""
        signals = [s for s in self._signals_buffer if s.drift_correctly_identified is not None]
        correct_signals = [s for s in signals if s.drift_correctly_identified]
        sorted_signals = sorted(correct_signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[DriftMonitorSignalCollector] = None


def get_drift_monitor_signal_collector() -> DriftMonitorSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = DriftMonitorSignalCollector()
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
    "DriftDetectionTrainingSignal",
    # DSPy Signatures
    "DriftDetectionSignature",
    "HopDecisionSignature",
    "DriftInterpretationSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "DriftMonitorSignalCollector",
    # Access
    "get_drift_monitor_signal_collector",
    "reset_dspy_integration",
]
