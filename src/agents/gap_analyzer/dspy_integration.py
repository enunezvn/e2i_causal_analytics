"""
E2I Gap Analyzer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for gap_analyzer Sender role

The Gap Analyzer agent is a DSPy Sender agent that:
1. Generates training signals from gap detection and ROI analysis
2. Provides EvidenceRelevanceSignature training examples
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
class GapAnalysisTrainingSignal:
    """
    Training signal for Gap Analyzer DSPy optimization.

    Captures gap detection, ROI calculation, and prioritization outcomes
    to train:
    - EvidenceRelevanceSignature: Scoring evidence relevance
    - GapPrioritizationSignature: Ranking opportunities
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    brand: str = ""
    metrics_analyzed: List[str] = field(default_factory=list)
    segments_analyzed: int = 0

    # === Detection Phase ===
    gaps_detected_count: int = 0
    total_gap_value: float = 0.0
    gap_types: List[str] = field(default_factory=list)  # vs_target, vs_benchmark, etc.

    # === ROI Phase ===
    roi_estimates_count: int = 0
    total_addressable_value: float = 0.0
    avg_expected_roi: float = 0.0
    high_roi_count: int = 0  # ROI > 2.0

    # === Prioritization Phase ===
    quick_wins_count: int = 0
    strategic_bets_count: int = 0
    prioritization_confidence: float = 0.0

    # === Output Quality ===
    executive_summary_length: int = 0
    key_insights_count: int = 0
    actionable_recommendations: int = 0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    roi_latency_ms: float = 0.0
    user_satisfaction: Optional[float] = None  # 1-5 rating
    recommendations_implemented: Optional[int] = None  # Delayed feedback

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - gap_detection_quality: 0.25 (gaps found relative to segments)
        - roi_quality: 0.25 (high ROI opportunities found)
        - prioritization_quality: 0.20 (quick wins + strategic bets)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.15 (if available)
        """
        reward = 0.0

        # Gap detection quality
        if self.segments_analyzed > 0:
            # Ideal: 1-3 gaps per segment
            gap_ratio = self.gaps_detected_count / self.segments_analyzed
            gap_quality = 1.0 if 1 <= gap_ratio <= 3 else max(0.5, 1.0 - abs(gap_ratio - 2) * 0.2)
            reward += 0.25 * gap_quality

        # ROI quality
        roi_score = 0.0
        if self.roi_estimates_count > 0:
            # Higher avg ROI is better (target: 2.0)
            roi_quality = (
                min(1.0, self.avg_expected_roi / 2.0) if self.avg_expected_roi > 0 else 0.0
            )
            roi_score += 0.5 * roi_quality
            # High ROI opportunities found
            high_roi_ratio = self.high_roi_count / self.roi_estimates_count
            roi_score += 0.5 * high_roi_ratio
        reward += 0.25 * roi_score

        # Prioritization quality
        prioritization_score = 0.0
        if self.gaps_detected_count > 0:
            # Should have both quick wins and strategic bets
            has_quick_wins = min(1.0, self.quick_wins_count / 3) * 0.4
            has_strategic_bets = min(1.0, self.strategic_bets_count / 3) * 0.3
            has_insights = min(1.0, self.key_insights_count / 5) * 0.3
            prioritization_score = has_quick_wins + has_strategic_bets + has_insights
        reward += 0.20 * prioritization_score

        # Efficiency (target < 8s for full analysis)
        target_latency = 8000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.15 * efficiency
        else:
            reward += 0.15

        # User satisfaction
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.15 * satisfaction_score
        elif self.recommendations_implemented is not None and self.actionable_recommendations > 0:
            # Use implementation rate as proxy
            impl_rate = min(1.0, self.recommendations_implemented / self.actionable_recommendations)
            reward += 0.15 * impl_rate
        else:
            reward += 0.075  # Partial credit

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"ga_{self.session_id}_{self.created_at}",
            "source_agent": "gap_analyzer",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "brand": self.brand,
                "metrics_analyzed": self.metrics_analyzed[:10],
                "segments_analyzed": self.segments_analyzed,
            },
            "detection": {
                "gaps_detected_count": self.gaps_detected_count,
                "total_gap_value": self.total_gap_value,
                "gap_types": self.gap_types,
            },
            "roi": {
                "roi_estimates_count": self.roi_estimates_count,
                "total_addressable_value": self.total_addressable_value,
                "avg_expected_roi": self.avg_expected_roi,
                "high_roi_count": self.high_roi_count,
            },
            "prioritization": {
                "quick_wins_count": self.quick_wins_count,
                "strategic_bets_count": self.strategic_bets_count,
                "prioritization_confidence": self.prioritization_confidence,
            },
            "output": {
                "executive_summary_length": self.executive_summary_length,
                "key_insights_count": self.key_insights_count,
                "actionable_recommendations": self.actionable_recommendations,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "detection_latency_ms": self.detection_latency_ms,
                "roi_latency_ms": self.roi_latency_ms,
                "user_satisfaction": self.user_satisfaction,
                "recommendations_implemented": self.recommendations_implemented,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class GapDetectionSignature(dspy.Signature):
        """
        Detect performance gaps in segment data.

        Given metrics and segment data, identify significant gaps
        between current performance and targets/benchmarks.
        """

        metrics: str = dspy.InputField(desc="KPIs to analyze")
        segment_data: str = dspy.InputField(desc="Performance by segment")
        benchmarks: str = dspy.InputField(desc="Target or benchmark values")
        min_gap_threshold: float = dspy.InputField(desc="Minimum gap % to report")

        gaps: list = dspy.OutputField(desc="List of detected gaps")
        gap_summary: str = dspy.OutputField(desc="Summary of gap landscape")
        priority_segments: list = dspy.OutputField(desc="Segments with largest gaps")

    class EvidenceRelevanceSignature(dspy.Signature):
        """
        Score the relevance of gap evidence for prioritization.

        Given a gap and its context, determine how relevant and
        actionable it is for business decisions.
        """

        gap_description: str = dspy.InputField(desc="Description of the performance gap")
        roi_estimate: str = dspy.InputField(desc="ROI analysis for closing the gap")
        implementation_context: str = dspy.InputField(desc="Constraints and resources")
        historical_success: str = dspy.InputField(desc="Past similar interventions")

        relevance_score: float = dspy.OutputField(desc="Relevance score 0-1")
        actionability: str = dspy.OutputField(desc="immediate, near_term, strategic")
        confidence: float = dspy.OutputField(desc="Confidence in relevance assessment")
        rationale: str = dspy.OutputField(desc="Explanation of relevance scoring")

    class GapPrioritizationSignature(dspy.Signature):
        """
        Prioritize gaps into quick wins and strategic bets.

        Given all detected gaps with ROI estimates, categorize them
        for different investment horizons.
        """

        gaps_with_roi: str = dspy.InputField(desc="Gaps with ROI estimates")
        resource_constraints: str = dspy.InputField(desc="Budget and capacity limits")
        strategic_priorities: str = dspy.InputField(desc="Business strategic focus areas")

        quick_wins: list = dspy.OutputField(desc="High ROI, low difficulty opportunities")
        strategic_bets: list = dspy.OutputField(desc="High impact, higher investment opportunities")
        key_insights: list = dspy.OutputField(desc="Strategic insights from gap analysis")
        executive_summary: str = dspy.OutputField(desc="2-3 sentence summary for leadership")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Gap Analyzer agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic prioritization")
    GapDetectionSignature = None  # type: ignore[assignment,misc]
    EvidenceRelevanceSignature = None  # type: ignore[assignment,misc]
    GapPrioritizationSignature = None  # type: ignore[assignment,misc]


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class GapAnalyzerSignalCollector:
    """
    Collects training signals from gap analysis executions.

    The Gap Analyzer agent is a Sender that generates signals
    for EvidenceRelevanceSignature optimization.
    """

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[GapAnalysisTrainingSignal] = []
        self._buffer_limit = 100

    def collect_analysis_signal(
        self,
        session_id: str,
        query: str,
        brand: str,
        metrics_analyzed: List[str],
        segments_analyzed: int,
    ) -> GapAnalysisTrainingSignal:
        """
        Initialize training signal at analysis start.

        Call this when starting a new gap analysis.
        """
        signal = GapAnalysisTrainingSignal(
            session_id=session_id,
            query=query,
            brand=brand,
            metrics_analyzed=metrics_analyzed,
            segments_analyzed=segments_analyzed,
        )
        return signal

    def update_detection(
        self,
        signal: GapAnalysisTrainingSignal,
        gaps_detected_count: int,
        total_gap_value: float,
        gap_types: List[str],
        detection_latency_ms: float,
    ) -> GapAnalysisTrainingSignal:
        """Update signal with detection phase results."""
        signal.gaps_detected_count = gaps_detected_count
        signal.total_gap_value = total_gap_value
        signal.gap_types = gap_types
        signal.detection_latency_ms = detection_latency_ms
        return signal

    def update_roi(
        self,
        signal: GapAnalysisTrainingSignal,
        roi_estimates_count: int,
        total_addressable_value: float,
        avg_expected_roi: float,
        high_roi_count: int,
        roi_latency_ms: float,
    ) -> GapAnalysisTrainingSignal:
        """Update signal with ROI phase results."""
        signal.roi_estimates_count = roi_estimates_count
        signal.total_addressable_value = total_addressable_value
        signal.avg_expected_roi = avg_expected_roi
        signal.high_roi_count = high_roi_count
        signal.roi_latency_ms = roi_latency_ms
        return signal

    def update_prioritization(
        self,
        signal: GapAnalysisTrainingSignal,
        quick_wins_count: int,
        strategic_bets_count: int,
        prioritization_confidence: float,
        executive_summary_length: int,
        key_insights_count: int,
        actionable_recommendations: int,
        total_latency_ms: float,
    ) -> GapAnalysisTrainingSignal:
        """Update signal with prioritization phase results."""
        signal.quick_wins_count = quick_wins_count
        signal.strategic_bets_count = strategic_bets_count
        signal.prioritization_confidence = prioritization_confidence
        signal.executive_summary_length = executive_summary_length
        signal.key_insights_count = key_insights_count
        signal.actionable_recommendations = actionable_recommendations
        signal.total_latency_ms = total_latency_ms

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_feedback(
        self,
        signal: GapAnalysisTrainingSignal,
        user_satisfaction: Optional[float] = None,
        recommendations_implemented: Optional[int] = None,
    ) -> GapAnalysisTrainingSignal:
        """Update signal with user feedback (delayed)."""
        signal.user_satisfaction = user_satisfaction
        signal.recommendations_implemented = recommendations_implemented
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [s.to_dict() for s in self._signals_buffer if s.compute_reward() >= min_reward]
        return signals[-limit:]

    def get_high_roi_examples(
        self,
        min_avg_roi: float = 2.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples with high average ROI (successful analyses)."""
        signals = [s for s in self._signals_buffer if s.avg_expected_roi >= min_avg_roi]
        sorted_signals = sorted(signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[GapAnalyzerSignalCollector] = None


def get_gap_analyzer_signal_collector() -> GapAnalyzerSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = GapAnalyzerSignalCollector()
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
    "GapAnalysisTrainingSignal",
    # DSPy Signatures
    "GapDetectionSignature",
    "EvidenceRelevanceSignature",
    "GapPrioritizationSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "GapAnalyzerSignalCollector",
    # Access
    "get_gap_analyzer_signal_collector",
    "reset_dspy_integration",
]
