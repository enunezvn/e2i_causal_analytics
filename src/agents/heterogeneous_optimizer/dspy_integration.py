"""
E2I Heterogeneous Optimizer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for heterogeneous_optimizer Sender role

The Heterogeneous Optimizer agent is a DSPy Sender agent that:
1. Generates training signals from CATE analysis and policy optimization
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
class HeterogeneousOptimizationTrainingSignal:
    """
    Training signal for Heterogeneous Optimizer DSPy optimization.

    Captures CATE analysis, segment discovery, and policy recommendations
    to train:
    - EvidenceSynthesisSignature: Synthesizing segment evidence
    - PolicyRecommendationSignature: Generating allocation policies
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    treatment_var: str = ""
    outcome_var: str = ""
    segment_vars_count: int = 0
    effect_modifiers_count: int = 0

    # === CATE Estimation Phase ===
    overall_ate: float = 0.0
    heterogeneity_score: float = 0.0  # 0-1, higher = more heterogeneity
    cate_segments_count: int = 0
    significant_cate_count: int = 0  # Statistically significant

    # === Segment Discovery Phase ===
    high_responders_count: int = 0
    low_responders_count: int = 0
    responder_spread: float = 0.0  # Difference between high and low CATE

    # === Policy Learning Phase ===
    policy_recommendations_count: int = 0
    expected_total_lift: float = 0.0
    actionable_policies: int = 0

    # === Output Quality ===
    executive_summary_length: int = 0
    key_insights_count: int = 0
    visualization_data_complete: bool = False

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    estimation_latency_ms: float = 0.0
    analysis_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - heterogeneity_detection: 0.25 (meaningful heterogeneity found)
        - segment_quality: 0.25 (clear responder separation)
        - policy_quality: 0.20 (actionable recommendations)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.15 (if available)
        """
        reward = 0.0

        # Heterogeneity detection quality
        # Good analysis should detect some heterogeneity if present
        het_score = 0.0
        if self.heterogeneity_score > 0:
            # Moderate heterogeneity is most actionable (0.3-0.7 range)
            if 0.3 <= self.heterogeneity_score <= 0.7:
                het_score = 1.0
            else:
                het_score = 0.7  # Still valuable
        if self.cate_segments_count > 0:
            significance_rate = self.significant_cate_count / self.cate_segments_count
            het_score = (het_score + significance_rate) / 2
        reward += 0.25 * het_score

        # Segment quality
        segment_score = 0.0
        if self.high_responders_count > 0 and self.low_responders_count > 0:
            # Found both high and low responders
            segment_score += 0.5
            # Good separation between them
            if self.responder_spread > 0.2:  # 20% effect difference
                segment_score += 0.5 * min(1.0, self.responder_spread / 0.5)
        reward += 0.25 * segment_score

        # Policy quality
        policy_score = 0.0
        if self.policy_recommendations_count > 0:
            # Has recommendations
            policy_score += 0.3
            # Actionable recommendations ratio
            if self.actionable_policies > 0:
                policy_score += 0.4 * min(
                    1.0, self.actionable_policies / self.policy_recommendations_count
                )
            # Expected lift is positive
            if self.expected_total_lift > 0:
                policy_score += 0.3
        reward += 0.20 * policy_score

        # Efficiency (target < 12s for full analysis)
        target_latency = 12000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.15 * efficiency
        else:
            reward += 0.15

        # User satisfaction
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.15 * satisfaction_score
        else:
            reward += 0.075  # Partial credit

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"ho_{self.session_id}_{self.created_at}",
            "source_agent": "heterogeneous_optimizer",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "treatment_var": self.treatment_var,
                "outcome_var": self.outcome_var,
                "segment_vars_count": self.segment_vars_count,
                "effect_modifiers_count": self.effect_modifiers_count,
            },
            "cate_estimation": {
                "overall_ate": self.overall_ate,
                "heterogeneity_score": self.heterogeneity_score,
                "cate_segments_count": self.cate_segments_count,
                "significant_cate_count": self.significant_cate_count,
            },
            "segment_discovery": {
                "high_responders_count": self.high_responders_count,
                "low_responders_count": self.low_responders_count,
                "responder_spread": self.responder_spread,
            },
            "policy_learning": {
                "policy_recommendations_count": self.policy_recommendations_count,
                "expected_total_lift": self.expected_total_lift,
                "actionable_policies": self.actionable_policies,
            },
            "output": {
                "executive_summary_length": self.executive_summary_length,
                "key_insights_count": self.key_insights_count,
                "visualization_data_complete": self.visualization_data_complete,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "estimation_latency_ms": self.estimation_latency_ms,
                "analysis_latency_ms": self.analysis_latency_ms,
                "confidence_score": self.confidence_score,
                "user_satisfaction": self.user_satisfaction,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class CATEInterpretationSignature(dspy.Signature):
        """
        Interpret CATE results for business decision-making.

        Given CATE estimates across segments, generate actionable
        interpretation of treatment effect heterogeneity.
        """

        overall_ate: float = dspy.InputField(desc="Average treatment effect")
        cate_by_segment: str = dspy.InputField(desc="CATE for each segment")
        heterogeneity_score: float = dspy.InputField(desc="Heterogeneity measure (0-1)")
        feature_importance: str = dspy.InputField(desc="Important effect modifiers")

        interpretation: str = dspy.OutputField(desc="Business interpretation of heterogeneity")
        high_responder_description: str = dspy.OutputField(desc="Who responds best")
        low_responder_description: str = dspy.OutputField(desc="Who responds least")
        actionable_segments: list = dspy.OutputField(desc="Segments to prioritize")

    class PolicyRecommendationSignature(dspy.Signature):
        """
        Generate treatment allocation policy recommendations.

        Given segment response profiles, recommend optimal treatment
        allocation for maximizing outcomes.
        """

        high_responders: str = dspy.InputField(desc="High responder segment profiles")
        low_responders: str = dspy.InputField(desc="Low responder segment profiles")
        current_allocation: str = dspy.InputField(desc="Current treatment allocation")
        constraints: str = dspy.InputField(desc="Budget and capacity constraints")

        recommendations: list = dspy.OutputField(desc="Allocation recommendations by segment")
        expected_lift: float = dspy.OutputField(desc="Expected outcome improvement")
        implementation_priority: list = dspy.OutputField(desc="Priority order for changes")
        risk_factors: list = dspy.OutputField(desc="Risks to monitor")

    class SegmentProfileSignature(dspy.Signature):
        """
        Generate segment profiles for visualization and reporting.

        Create clear descriptions of high and low responder segments
        for stakeholder communication.
        """

        segment_data: str = dspy.InputField(desc="Segment characteristics and CATE")
        responder_type: str = dspy.InputField(desc="high or low responder")
        defining_features: str = dspy.InputField(desc="Key distinguishing features")

        profile_description: str = dspy.OutputField(desc="Human-readable segment description")
        targeting_criteria: list = dspy.OutputField(desc="Criteria for segment targeting")
        expected_response: str = dspy.OutputField(desc="Expected treatment response")
        business_recommendation: str = dspy.OutputField(desc="What to do with this segment")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Heterogeneous Optimizer agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic optimization")
    CATEInterpretationSignature = None
    PolicyRecommendationSignature = None
    SegmentProfileSignature = None


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class HeterogeneousOptimizerSignalCollector:
    """
    Collects training signals from heterogeneous optimization executions.

    The Heterogeneous Optimizer agent is a Sender that generates signals
    for segment analysis and policy optimization.
    """

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[HeterogeneousOptimizationTrainingSignal] = []
        self._buffer_limit = 100

    def collect_optimization_signal(
        self,
        session_id: str,
        query: str,
        treatment_var: str,
        outcome_var: str,
        segment_vars_count: int,
        effect_modifiers_count: int,
    ) -> HeterogeneousOptimizationTrainingSignal:
        """
        Initialize training signal at optimization start.

        Call this when starting a new heterogeneous analysis.
        """
        signal = HeterogeneousOptimizationTrainingSignal(
            session_id=session_id,
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            segment_vars_count=segment_vars_count,
            effect_modifiers_count=effect_modifiers_count,
        )
        return signal

    def update_cate_estimation(
        self,
        signal: HeterogeneousOptimizationTrainingSignal,
        overall_ate: float,
        heterogeneity_score: float,
        cate_segments_count: int,
        significant_cate_count: int,
        estimation_latency_ms: float,
    ) -> HeterogeneousOptimizationTrainingSignal:
        """Update signal with CATE estimation phase results."""
        signal.overall_ate = overall_ate
        signal.heterogeneity_score = heterogeneity_score
        signal.cate_segments_count = cate_segments_count
        signal.significant_cate_count = significant_cate_count
        signal.estimation_latency_ms = estimation_latency_ms
        return signal

    def update_segment_discovery(
        self,
        signal: HeterogeneousOptimizationTrainingSignal,
        high_responders_count: int,
        low_responders_count: int,
        responder_spread: float,
        analysis_latency_ms: float,
    ) -> HeterogeneousOptimizationTrainingSignal:
        """Update signal with segment discovery phase results."""
        signal.high_responders_count = high_responders_count
        signal.low_responders_count = low_responders_count
        signal.responder_spread = responder_spread
        signal.analysis_latency_ms = analysis_latency_ms
        return signal

    def update_policy_learning(
        self,
        signal: HeterogeneousOptimizationTrainingSignal,
        policy_recommendations_count: int,
        expected_total_lift: float,
        actionable_policies: int,
        executive_summary_length: int,
        key_insights_count: int,
        visualization_data_complete: bool,
        total_latency_ms: float,
        confidence_score: float,
    ) -> HeterogeneousOptimizationTrainingSignal:
        """Update signal with policy learning phase results."""
        signal.policy_recommendations_count = policy_recommendations_count
        signal.expected_total_lift = expected_total_lift
        signal.actionable_policies = actionable_policies
        signal.executive_summary_length = executive_summary_length
        signal.key_insights_count = key_insights_count
        signal.visualization_data_complete = visualization_data_complete
        signal.total_latency_ms = total_latency_ms
        signal.confidence_score = confidence_score

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_feedback(
        self,
        signal: HeterogeneousOptimizationTrainingSignal,
        user_satisfaction: Optional[float] = None,
    ) -> HeterogeneousOptimizationTrainingSignal:
        """Update signal with user feedback (delayed)."""
        signal.user_satisfaction = user_satisfaction
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [s.to_dict() for s in self._signals_buffer if s.compute_reward() >= min_reward]
        return signals[-limit:]

    def get_high_heterogeneity_examples(
        self,
        min_heterogeneity: float = 0.3,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples with meaningful heterogeneity (most valuable for CATE)."""
        signals = [s for s in self._signals_buffer if s.heterogeneity_score >= min_heterogeneity]
        sorted_signals = sorted(signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[HeterogeneousOptimizerSignalCollector] = None


def get_heterogeneous_optimizer_signal_collector() -> HeterogeneousOptimizerSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = HeterogeneousOptimizerSignalCollector()
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
    "HeterogeneousOptimizationTrainingSignal",
    # DSPy Signatures
    "CATEInterpretationSignature",
    "PolicyRecommendationSignature",
    "SegmentProfileSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "HeterogeneousOptimizerSignalCollector",
    # Access
    "get_heterogeneous_optimizer_signal_collector",
    "reset_dspy_integration",
]
