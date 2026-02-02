"""Evidence Synthesis GEPA Metric for DSPy Module Optimization.

This metric evaluates the EvidenceSynthesisSignature outputs from the
CausalImpactModule DSPy module. Unlike CausalImpactGEPAMetric which
evaluates full pipeline outputs, this metric focuses on interpretation quality.

The metric optimizes for:
- Narrative clarity and completeness
- Key findings accuracy and relevance
- Confidence level calibration
- Recommendation actionability
- Limitation acknowledgment
"""

from dataclasses import dataclass
from typing import Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace


@dataclass
class EvidenceSynthesisGEPAMetric:
    """GEPA metric for Evidence Synthesis DSPy module.

    Evaluates the quality of causal interpretation outputs from
    EvidenceSynthesisSignature, matching what the CausalImpactModule
    actually produces.

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        narrative_weight: Weight for narrative quality (default 0.25)
        findings_weight: Weight for key findings (default 0.25)
        confidence_weight: Weight for confidence calibration (default 0.20)
        recommendations_weight: Weight for recommendations (default 0.20)
        limitations_weight: Weight for limitations (default 0.10)
    """

    name: str = "evidence_synthesis_gepa"
    description: str = "GEPA metric for Evidence Synthesis DSPy module"

    narrative_weight: float = 0.25
    findings_weight: float = 0.25
    confidence_weight: float = 0.20
    recommendations_weight: float = 0.20
    limitations_weight: float = 0.10

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> float:
        """Compute score for evidence synthesis quality.

        GEPA requires metrics to return a float score for aggregation.
        The score is a weighted combination of interpretation quality factors.

        Args:
            gold: Ground truth Example with expected outputs
            pred: Model Prediction with synthesis results
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Float score between 0.0 and 1.0
        """
        scores = {}

        # Component 1: Narrative Quality (25%)
        scores["narrative"] = self._score_narrative(pred, gold)

        # Component 2: Key Findings (25%)
        scores["findings"] = self._score_findings(pred, gold)

        # Component 3: Confidence Calibration (20%)
        scores["confidence"] = self._score_confidence(pred, gold)

        # Component 4: Recommendations (20%)
        scores["recommendations"] = self._score_recommendations(pred, gold)

        # Component 5: Limitations (10%)
        scores["limitations"] = self._score_limitations(pred, gold)

        # Aggregate weighted score
        total_score = (
            self.narrative_weight * scores["narrative"]
            + self.findings_weight * scores["findings"]
            + self.confidence_weight * scores["confidence"]
            + self.recommendations_weight * scores["recommendations"]
            + self.limitations_weight * scores["limitations"]
        )

        return total_score

    def _score_narrative(self, pred: Prediction, gold: Example) -> float:
        """Score narrative quality and completeness.

        Checks for:
        - Presence of narrative
        - Minimum length (meaningful content)
        - Key term coverage from input

        Args:
            pred: Prediction with narrative attribute
            gold: Example with expected narrative

        Returns:
            Score between 0.0 and 1.0
        """
        narrative = getattr(pred, "narrative", None)

        if not narrative:
            return 0.0

        score = 0.0

        # Basic presence check
        if isinstance(narrative, str) and len(narrative) > 0:
            score += 0.3

        # Length check (at least 50 chars for meaningful content)
        if isinstance(narrative, str) and len(narrative) >= 50:
            score += 0.3

        # Key term coverage - check if narrative mentions key concepts
        # from the input (estimation, effect, treatment, etc.)
        key_terms = ["effect", "causal", "treatment", "outcome", "significant"]
        if isinstance(narrative, str):
            narrative_lower = narrative.lower()
            terms_found = sum(1 for term in key_terms if term in narrative_lower)
            term_coverage = min(1.0, terms_found / 3)  # At least 3 terms for full score
            score += 0.4 * term_coverage

        return min(1.0, score)

    def _score_findings(self, pred: Prediction, gold: Example) -> float:
        """Score key findings completeness and quality.

        Checks for:
        - Presence of findings list
        - Number of findings (target: 3-5)
        - Content quality (non-empty strings)

        Args:
            pred: Prediction with key_findings attribute
            gold: Example with expected findings

        Returns:
            Score between 0.0 and 1.0
        """
        findings = getattr(pred, "key_findings", None)

        if not findings:
            return 0.0

        score = 0.0

        # Check if it's a list
        if isinstance(findings, list):
            # Count valid findings (non-empty strings)
            valid_findings = [f for f in findings if isinstance(f, str) and len(f.strip()) > 10]

            # Ideal: 3-5 findings
            count = len(valid_findings)
            if count >= 5:
                score += 0.5
            elif count >= 3:
                score += 0.4
            elif count >= 1:
                score += 0.2

            # Quality bonus for substantial findings
            if valid_findings:
                avg_length = sum(len(f) for f in valid_findings) / len(valid_findings)
                if avg_length >= 50:  # Substantial content
                    score += 0.5
                elif avg_length >= 25:
                    score += 0.3
                else:
                    score += 0.1

        return min(1.0, score)

    def _score_confidence(self, pred: Prediction, gold: Example) -> float:
        """Score confidence level calibration.

        Checks for:
        - Valid confidence level (low/medium/high)
        - Match with expected confidence from gold

        Args:
            pred: Prediction with confidence_level attribute
            gold: Example with expected confidence_level

        Returns:
            Score between 0.0 and 1.0
        """
        confidence = getattr(pred, "confidence_level", None)
        expected = getattr(gold, "confidence_level", None)

        if not confidence:
            return 0.0

        score = 0.0

        # Valid confidence level
        valid_levels = ["low", "medium", "high"]
        if isinstance(confidence, str):
            confidence_lower = confidence.lower().strip()

            # Check if valid level
            if confidence_lower in valid_levels:
                score += 0.5

                # Match with expected (if provided)
                if expected:
                    expected_lower = str(expected).lower().strip()
                    if confidence_lower == expected_lower:
                        score += 0.5
                    elif (
                        abs(
                            valid_levels.index(confidence_lower)
                            - valid_levels.index(expected_lower)
                        )
                        == 1
                    ):
                        # Adjacent level (e.g., medium vs high) - partial credit
                        score += 0.25
                else:
                    # No expected value, give partial credit for valid response
                    score += 0.25

        return min(1.0, score)

    def _score_recommendations(self, pred: Prediction, gold: Example) -> float:
        """Score recommendation quality and actionability.

        Checks for:
        - Presence of recommendations list
        - Number of recommendations (target: 3-5)
        - Actionable language (verbs indicating action)

        Args:
            pred: Prediction with recommendations attribute
            gold: Example with expected recommendations

        Returns:
            Score between 0.0 and 1.0
        """
        recommendations = getattr(pred, "recommendations", None)

        if not recommendations:
            return 0.0

        score = 0.0

        if isinstance(recommendations, list):
            # Count valid recommendations
            valid_recs = [r for r in recommendations if isinstance(r, str) and len(r.strip()) > 10]

            # Ideal: 3-5 recommendations
            count = len(valid_recs)
            if count >= 5:
                score += 0.4
            elif count >= 3:
                score += 0.3
            elif count >= 1:
                score += 0.15

            # Actionability check - look for action verbs
            action_verbs = [
                "increase",
                "decrease",
                "monitor",
                "implement",
                "consider",
                "scale",
                "optimize",
                "test",
                "analyze",
                "review",
                "continue",
                "expand",
                "reduce",
                "focus",
                "invest",
                "develop",
                "maintain",
            ]

            actionable_count = 0
            for rec in valid_recs:
                rec_lower = rec.lower()
                if any(verb in rec_lower for verb in action_verbs):
                    actionable_count += 1

            if valid_recs:
                actionability = actionable_count / len(valid_recs)
                score += 0.6 * actionability

        return min(1.0, score)

    def _score_limitations(self, pred: Prediction, gold: Example) -> float:
        """Score limitations acknowledgment.

        Checks for:
        - Presence of limitations list
        - At least some limitations mentioned
        - Meaningful content

        Args:
            pred: Prediction with limitations attribute
            gold: Example with expected limitations

        Returns:
            Score between 0.0 and 1.0
        """
        limitations = getattr(pred, "limitations", None)

        if not limitations:
            return 0.0

        score = 0.0

        if isinstance(limitations, list):
            # Count valid limitations
            valid_lims = [
                lim for lim in limitations if isinstance(lim, str) and len(lim.strip()) > 10
            ]

            # Any limitations acknowledged is good
            count = len(valid_lims)
            if count >= 3:
                score += 0.6
            elif count >= 2:
                score += 0.5
            elif count >= 1:
                score += 0.3

            # Quality bonus for thoughtful limitations
            if valid_lims:
                avg_length = sum(len(lim) for lim in valid_lims) / len(valid_lims)
                if avg_length >= 40:
                    score += 0.4
                elif avg_length >= 20:
                    score += 0.2

        return min(1.0, score)


__all__ = ["EvidenceSynthesisGEPAMetric"]
