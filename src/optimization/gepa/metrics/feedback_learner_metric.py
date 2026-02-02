"""Feedback Learner GEPA Metric for E2I Tier 5 Deep Agent.

This module provides the GEPA metric for optimizing the Feedback Learner agent,
which performs meta-optimization (optimizing the optimizer itself).

The metric measures:
- Learning extraction quality
- Storage efficiency
- Downstream application success
"""

from dataclasses import dataclass
from typing import Any, Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace, ScoreWithFeedback


@dataclass
class FeedbackLearnerGEPAMetric:
    """GEPA metric for Feedback Learner Deep agent (Tier 5).

    Meta-optimization: optimizing the optimizer itself.

    Integrates with:
    - Learning extraction pipeline
    - Knowledge storage backends
    - Downstream agent performance tracking

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        extraction_weight: Weight for learning extraction quality (default 0.40)
        storage_weight: Weight for storage efficiency (default 0.20)
        application_weight: Weight for downstream application success (default 0.40)
    """

    name: str = "feedback_learner_gepa"
    description: str = (
        "GEPA metric for Tier 5 Feedback Learner agent - extraction, storage, application"
    )

    extraction_weight: float = 0.40
    storage_weight: float = 0.20
    application_weight: float = 0.40

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> ScoreWithFeedback:
        """Compute score and feedback for feedback learning.

        Args:
            gold: Ground truth Example with expected patterns
            pred: Model Prediction with extracted learnings
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Dict with 'score' (float 0-1) and 'feedback' (str) for GEPA reflection
        """
        feedback_parts = []
        scores = {}

        scores["extraction"], fb = self._score_extraction(pred, gold)
        feedback_parts.append(f"[Extraction] {fb}")

        scores["storage"], fb = self._score_storage(pred)
        feedback_parts.append(f"[Storage] {fb}")

        scores["application"], fb = self._score_application(pred, gold)
        feedback_parts.append(f"[Application] {fb}")

        total = (
            self.extraction_weight * scores["extraction"]
            + self.storage_weight * scores["storage"]
            + self.application_weight * scores["application"]
        )

        feedback = f"Score: {total:.3f}\n" + "\n".join(feedback_parts)

        return {"score": total, "feedback": feedback}

    def _score_extraction(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score learning extraction quality.

        Compares extracted learnings against expected patterns.

        Args:
            pred: Prediction with extracted_learnings attribute
            gold: Example with expected_patterns

        Returns:
            Tuple of (score, feedback_message)
        """
        learnings = getattr(pred, "extracted_learnings", None)

        if not learnings:
            return 0.0, "CRITICAL: No learnings extracted"

        expected = getattr(gold, "expected_patterns", [])

        if not expected:
            return 0.7, f"Extracted {len(learnings)} learnings (no ground truth)"

        matched = sum(1 for e in expected if any(self._pattern_match(e, l) for l in learnings))

        recall = matched / len(expected)
        spurious = max(0, len(learnings) - matched)
        penalty = min(0.3, spurious * 0.1)

        score = max(0, recall - penalty)

        return score, f"{matched}/{len(expected)} patterns, {spurious} spurious"

    def _pattern_match(self, expected: Any, learning: Any) -> bool:
        """Simple pattern matching between expected and extracted.

        Uses word overlap to determine if patterns match.

        Args:
            expected: Expected pattern
            learning: Extracted learning

        Returns:
            True if patterns match (>50% word overlap)
        """
        exp_words = set(str(expected).lower().split())
        learn_words = set(str(learning).lower().split())

        if not exp_words:
            return False

        overlap = len(exp_words & learn_words) / len(exp_words)
        return overlap > 0.5

    def _score_storage(self, pred: Prediction) -> tuple[float, str]:
        """Score storage efficiency.

        Checks for compression and indexing.

        Args:
            pred: Prediction with storage_format attribute

        Returns:
            Tuple of (score, feedback_message)
        """
        storage = getattr(pred, "storage_format", {})

        score = 0.6  # Baseline

        if storage.get("compressed"):
            score += 0.2
        if storage.get("indexed"):
            score += 0.2

        return min(
            1.0, score
        ), f"compressed={storage.get('compressed', False)}, indexed={storage.get('indexed', False)}"

    def _score_application(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score downstream application success.

        Measures performance improvement from applied learnings.

        Args:
            pred: Prediction with application_results attribute
            gold: Example (unused but required by protocol)

        Returns:
            Tuple of (score, feedback_message)
        """
        results = getattr(pred, "application_results", None)

        if not results:
            return 0.0, "No downstream application measured"

        delta = results.get("performance_delta", 0)

        if delta > 0.05:
            return 1.0, f"Strong improvement: +{delta:.1%}"
        elif delta > 0:
            return 0.7, f"Modest improvement: +{delta:.1%}"
        elif delta == 0:
            return 0.4, "No measurable impact"
        return 0.1, f"NEGATIVE impact: {delta:.1%}"


__all__ = ["FeedbackLearnerGEPAMetric"]
