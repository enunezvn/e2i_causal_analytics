"""Standard Agent GEPA Metric for E2I Standard Agents.

This module provides a generic GEPA metric for Standard agents in Tiers 0, 1, and 4.
These agents are tool-heavy and SLA-bound, requiring simpler optimization.

The metric optimizes for:
- SLA compliance (latency thresholds)
- Basic accuracy (output matching)
"""

from dataclasses import dataclass
from typing import Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace, ScoreWithFeedback


@dataclass
class StandardAgentGEPAMetric:
    """Generic GEPA metric for Standard agents (Tiers 0, 1, 4).

    Uses SLA compliance and basic accuracy for tool-heavy, SLA-bound operations.

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        sla_threshold_ms: Maximum acceptable latency in milliseconds (default 2000)
        sla_weight: Weight for SLA compliance scoring (default 0.4)
        accuracy_weight: Weight for accuracy scoring (default 0.6)
    """

    name: str = "standard_agent_gepa"
    description: str = "GEPA metric for Standard agents - SLA compliance and accuracy"

    sla_threshold_ms: int = 2000
    sla_weight: float = 0.4
    accuracy_weight: float = 0.6

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> ScoreWithFeedback:
        """Compute score and feedback for standard agent operations.

        Args:
            gold: Ground truth Example with expected output
            pred: Model Prediction to evaluate
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Dict with 'score' (float 0-1) and 'feedback' (str) for GEPA reflection
        """
        # SLA scoring
        latency = getattr(pred, "latency_ms", self.sla_threshold_ms)
        sla_pass = latency <= self.sla_threshold_ms
        sla_score = 1.0 if sla_pass else 0.5

        # Accuracy scoring
        accuracy_score, accuracy_fb = self._score_accuracy(pred, gold)

        total = self.sla_weight * sla_score + self.accuracy_weight * accuracy_score

        sla_fb = f"SLA={'PASS' if sla_pass else 'FAIL'} ({latency}ms)"

        return {
            "score": total,
            "feedback": f"Score: {total:.3f} | {sla_fb} | {accuracy_fb}",
        }

    def _score_accuracy(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score prediction accuracy against expected output.

        Supports exact matching and numeric similarity.

        Args:
            pred: Prediction with output attribute
            gold: Example with expected_output attribute

        Returns:
            Tuple of (score, feedback_message)
        """
        expected = getattr(gold, "expected_output", None)
        actual = getattr(pred, "output", None)

        if expected is None:
            return 0.5, "No ground truth"

        if actual == expected:
            return 1.0, "Exact match"

        # Numeric similarity
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == 0:
                return (1.0, "Match") if actual == 0 else (0.0, f"Expected 0, got {actual}")
            rel_error = abs(actual - expected) / abs(expected)
            score = max(0, 1 - rel_error)
            return score, f"Relative error: {rel_error:.2%}"

        return 0.5, "Type mismatch or partial match"


__all__ = ["StandardAgentGEPAMetric"]
