"""Experiment Designer GEPA Metric for E2I Tier 3 Hybrid Agent.

This module provides the GEPA metric for optimizing the Experiment Designer agent,
which designs A/B tests with Digital Twin pre-screening and power analysis.

The metric optimizes for:
- Power analysis validity
- Design validity (randomization, controls, blinding)
- Integration with ExperimentKnowledgeStore
- Pre-registration completeness
"""

from dataclasses import dataclass
from typing import Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace, ScoreWithFeedback


@dataclass
class ExperimentDesignerGEPAMetric:
    """GEPA metric for Experiment Designer Hybrid agent (Tier 3).

    Integrates with:
    - Digital Twin pre-screening
    - ExperimentKnowledgeStore for past learnings
    - Power analysis utilities

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        power_weight: Weight for power analysis scoring (default 0.35)
        design_weight: Weight for design validity (default 0.30)
        learning_weight: Weight for past learnings integration (default 0.20)
        prereg_weight: Weight for pre-registration completeness (default 0.15)
        target_power: Minimum acceptable statistical power (default 0.80)
    """

    name: str = "experiment_designer_gepa"
    description: str = "GEPA metric for Tier 3 Experiment Designer agent - power, design, learning, pre-registration"

    power_weight: float = 0.35
    design_weight: float = 0.30
    learning_weight: float = 0.20
    prereg_weight: float = 0.15

    target_power: float = 0.80

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> ScoreWithFeedback:
        """Compute score and feedback for experiment design.

        Args:
            gold: Ground truth Example with expected design elements
            pred: Model Prediction with experiment design
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Dict with 'score' (float 0-1) and 'feedback' (str) for GEPA reflection
        """
        feedback_parts = []
        scores = {}

        scores["power"], fb = self._score_power(pred, gold)
        feedback_parts.append(f"[Power] {fb}")

        scores["design"], fb = self._score_design(pred)
        feedback_parts.append(f"[Design] {fb}")

        scores["learning"], fb = self._score_learning(pred, trace)
        feedback_parts.append(f"[Learning] {fb}")

        scores["prereg"], fb = self._score_prereg(pred)
        feedback_parts.append(f"[PreReg] {fb}")

        total = (
            self.power_weight * scores["power"]
            + self.design_weight * scores["design"]
            + self.learning_weight * scores["learning"]
            + self.prereg_weight * scores["prereg"]
        )

        feedback = f"Score: {total:.3f}\n" + "\n".join(feedback_parts)

        return {"score": total, "feedback": feedback}

    def _score_power(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score power analysis validity.

        Args:
            pred: Prediction with power_calculation attribute
            gold: Example with expected_effect_size

        Returns:
            Tuple of (score, feedback_message)
        """
        calc = getattr(pred, "power_calculation", None)

        if not calc:
            return 0.0, "CRITICAL: No power analysis"

        power = calc.get("power", 0)
        n = calc.get("required_n", 0)

        if power >= self.target_power:
            expected_effect = getattr(gold, "expected_effect_size", 0.2)
            if self._sample_reasonable(n, expected_effect):
                return (
                    1.0,
                    f"Power={power:.2f} with n={n} (appropriate for effect={expected_effect})",
                )
            return 0.7, f"Power={power:.2f} but n={n} may be over/under-estimated"

        return 0.3, f"UNDERPOWERED: {power:.2f} < {self.target_power}"

    def _sample_reasonable(self, n: int, effect: float) -> bool:
        """Check if sample size is reasonable for expected effect size.

        Args:
            n: Required sample size
            effect: Expected effect size

        Returns:
            True if sample size is appropriate
        """
        if effect < 0.1:
            return n >= 1000
        elif effect < 0.3:
            return n >= 200
        return n >= 50

    def _score_design(self, pred: Prediction) -> tuple[float, str]:
        """Score experimental design validity.

        Checks for:
        - Randomization method
        - Control group presence
        - Blinding

        Args:
            pred: Prediction with design attributes

        Returns:
            Tuple of (score, feedback_message)
        """
        score = 0.0
        issues = []

        rand = getattr(pred, "randomization_method", None)
        if rand in ["stratified", "blocked", "cluster"]:
            score += 0.4
        elif rand == "simple":
            score += 0.3
            issues.append("Consider stratified randomization")
        else:
            issues.append("No randomization specified")

        if getattr(pred, "control_group", False):
            score += 0.3
        else:
            issues.append("No control group")

        if getattr(pred, "blinding", False):
            score += 0.3
        else:
            issues.append("No blinding")

        if issues:
            return score, f"Issues: {'; '.join(issues)}"
        return score, "Design validated"

    def _score_learning(self, pred: Prediction, trace: Optional[DSPyTrace]) -> tuple[float, str]:
        """Score ExperimentKnowledgeStore integration.

        Args:
            pred: Prediction with past_learnings_applied attribute
            trace: Optional execution trace

        Returns:
            Tuple of (score, feedback_message)
        """
        learnings = getattr(pred, "past_learnings_applied", None)

        if learnings is None:
            return 0.0, "ExperimentKnowledgeStore not consulted"

        if not learnings:
            return 0.3, "No applicable past learnings found"

        applied = sum(1 for l in learnings if l.get("applied", False))

        if applied == len(learnings):
            return 1.0, f"All {applied} learnings applied"
        elif applied > 0:
            return 0.7, f"{applied}/{len(learnings)} learnings applied"
        return 0.4, "Learnings retrieved but not applied"

    def _score_prereg(self, pred: Prediction) -> tuple[float, str]:
        """Score pre-registration completeness.

        Required fields:
        - hypothesis
        - primary_outcome
        - sample_size_justification
        - analysis_plan
        - stopping_rules

        Args:
            pred: Prediction with preregistration attribute

        Returns:
            Tuple of (score, feedback_message)
        """
        prereg = getattr(pred, "preregistration", None)

        if not prereg:
            return 0.0, "No pre-registration"

        required = [
            "hypothesis",
            "primary_outcome",
            "sample_size_justification",
            "analysis_plan",
            "stopping_rules",
        ]

        present = [f for f in required if prereg.get(f)]
        missing = [f for f in required if not prereg.get(f)]

        score = len(present) / len(required)

        if missing:
            return score, f"Missing: {', '.join(missing)}"
        return score, "Pre-registration complete"


__all__ = ["ExperimentDesignerGEPAMetric"]
