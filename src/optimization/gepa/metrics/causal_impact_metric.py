"""Causal Impact GEPA Metric for E2I Tier 2 Hybrid Agent.

This module provides the GEPA metric for optimizing the Causal Impact agent,
which traces causal chains and estimates treatment effects using DoWhy/EconML.

The metric optimizes for:
- Refutation test pass rate (DoWhy 5-test suite)
- Sensitivity analysis robustness (E-value)
- Methodology validity (DAG approval, method selection)
- Business relevance (KPI attribution, actionability)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace


@dataclass
class CausalImpactGEPAMetric:
    """GEPA metric for Causal Impact Hybrid agent (Tier 2).

    Integrates with:
    - causal_validations table (V4.1)
    - RefutationRunner.run_suite()
    - 5-node workflow (GraphBuilder→Estimation→Refutation→Sensitivity→Interpretation)

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        refutation_weight: Weight for refutation test scoring (default 0.30)
        sensitivity_weight: Weight for sensitivity analysis (default 0.25)
        methodology_weight: Weight for methodology validation (default 0.25)
        business_weight: Weight for business relevance (default 0.20)
        refutation_tests: List of DoWhy refutation tests to run
    """

    name: str = "causal_impact_gepa"
    description: str = "GEPA metric for Tier 2 Causal Impact agent - refutation, sensitivity, methodology, business"

    refutation_weight: float = 0.30
    sensitivity_weight: float = 0.25
    methodology_weight: float = 0.25
    business_weight: float = 0.20

    # From domain_vocabulary.yaml v3.1.0
    refutation_tests: list[str] = field(
        default_factory=lambda: [
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
            "sensitivity_e_value",
        ]
    )

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> float:
        """Compute score for causal impact estimation.

        GEPA requires metrics to return a float score for aggregation.
        The score is a weighted combination of:
        - Refutation test pass rate (30%)
        - Sensitivity analysis robustness (25%)
        - Methodology validity (25%)
        - Business relevance (20%)

        Args:
            gold: Ground truth Example with expected outputs
            pred: Model Prediction with causal analysis results
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Float score between 0.0 and 1.0
        """
        scores = {}

        # Component 1: Refutation Tests (Node 3)
        scores["refutation"], _ = self._score_refutation(pred)

        # Component 2: Sensitivity Analysis (Node 4)
        scores["sensitivity"], _ = self._score_sensitivity(pred)

        # Component 3: Methodology
        scores["methodology"], _ = self._score_methodology(pred, gold, trace)

        # Component 4: Business Relevance
        scores["business"], _ = self._score_business(pred, gold)

        # Aggregate weighted score
        total_score = (
            self.refutation_weight * scores["refutation"]
            + self.sensitivity_weight * scores["sensitivity"]
            + self.methodology_weight * scores["methodology"]
            + self.business_weight * scores["business"]
        )

        return total_score

    def _score_refutation(self, pred: Prediction) -> tuple[float, str]:
        """Score refutation test results from DoWhy suite.

        Args:
            pred: Prediction with refutation_results attribute

        Returns:
            Tuple of (score, feedback_message)
        """
        results = getattr(pred, "refutation_results", None)

        if not results:
            return 0.0, "CRITICAL: No refutation tests executed. Wire RefutationRunner.run_suite()"

        passed, failed = [], []
        for test in self.refutation_tests:
            test_result = results.get(test, {})
            status = test_result.get("status", "skipped")

            if status == "passed":
                passed.append(test)
            elif status == "failed":
                failed.append(
                    {
                        "test": test,
                        "original": test_result.get("original_effect", "N/A"),
                        "refuted": test_result.get("refuted_effect", "N/A"),
                        "p_value": test_result.get("p_value", "N/A"),
                    }
                )

        score = len(passed) / len(self.refutation_tests) if self.refutation_tests else 0

        if failed:
            details = "; ".join(
                [
                    (
                        f"{f['test']}: {f['original']:.3f}→{f['refuted']:.3f} (p={f['p_value']:.4f})"
                        if isinstance(f["original"], (int, float))
                        else f"{f['test']}: FAILED"
                    )
                    for f in failed
                ]
            )
            return score, f"FAILED {len(failed)}/{len(self.refutation_tests)}: {details}"

        return score, f"All {len(self.refutation_tests)} tests passed"

    def _score_sensitivity(self, pred: Prediction) -> tuple[float, str]:
        """Score sensitivity analysis robustness via E-value.

        E-value thresholds:
        - >= 3.0: Strong robustness (score 1.0)
        - >= 2.0: Good robustness (score 0.8)
        - >= 1.5: Marginal robustness (score 0.5)
        - < 1.5: Weak robustness (score 0.2)

        Args:
            pred: Prediction with sensitivity_analysis attribute

        Returns:
            Tuple of (score, feedback_message)
        """
        sensitivity = getattr(pred, "sensitivity_analysis", None)

        if not sensitivity:
            return 0.0, "CRITICAL: No sensitivity analysis. Check Node 4 execution"

        e_value = sensitivity.get("e_value", 0)

        if e_value >= 3:
            return 1.0, f"Strong robustness (E-value={e_value:.2f} ≥ 3.0)"
        elif e_value >= 2:
            return 0.8, f"Good robustness (E-value={e_value:.2f} ≥ 2.0)"
        elif e_value >= 1.5:
            return (
                0.5,
                f"MARGINAL (E-value={e_value:.2f}). Increase sample or use stronger instruments",
            )
        else:
            return 0.2, f"WEAK (E-value={e_value:.2f} < 1.5). High risk of unobserved confounding"

    def _score_methodology(
        self,
        pred: Prediction,
        gold: Example,
        trace: Optional[DSPyTrace],
    ) -> tuple[float, str]:
        """Score methodology selection and DAG approval.

        Args:
            pred: Prediction with dag_approved and estimation_method
            gold: Example with data_characteristics
            trace: Optional execution trace

        Returns:
            Tuple of (score, feedback_message)
        """
        score = 0.0
        issues = []

        # DAG approval (from expert_reviews table)
        if getattr(pred, "dag_approved", False):
            score += 0.4
        else:
            issues.append("DAG not expert-approved")

        # Estimation method appropriateness
        method = getattr(pred, "estimation_method", None)
        data_chars = getattr(gold, "data_characteristics", {})

        if method:
            if self._method_fits_data(method, data_chars):
                score += 0.6
            else:
                score += 0.3
                issues.append(f"Method '{method}' may not match data: {data_chars}")
        else:
            issues.append("No estimation method specified")

        if issues:
            return score, f"Issues: {'; '.join(issues)}"
        return score, "Validated: DAG approved, method appropriate"

    def _method_fits_data(self, method: str, data_chars: dict[str, Any]) -> bool:
        """Check if estimation method is appropriate for data characteristics.

        Args:
            method: Estimation method name (e.g., 'CausalForest', 'LinearDML')
            data_chars: Data characteristics dict

        Returns:
            True if method fits data characteristics
        """
        method_lower = method.lower()

        if data_chars.get("heterogeneous") and "forest" in method_lower:
            return True
        if data_chars.get("high_dim_confounders") and "dml" in method_lower:
            return True
        if not data_chars.get("complex") and method_lower in ["ols", "ipw"]:
            return True
        if "causalforest" in method_lower or "lineardml" in method_lower:
            return True  # Generally good defaults

        return False

    def _score_business(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score business relevance: KPI attribution and recommendations.

        Args:
            pred: Prediction with kpi_attribution and recommendations
            gold: Example with expected_kpis

        Returns:
            Tuple of (score, feedback_message)
        """
        score = 0.0
        issues = []

        # KPI attribution
        attributed = set(getattr(pred, "kpi_attribution", []))
        expected = set(getattr(gold, "expected_kpis", []))

        if attributed and expected:
            overlap = attributed & expected
            if overlap:
                score += 0.5
            else:
                issues.append(f"KPI mismatch: expected {expected}, got {attributed}")
        elif attributed:
            score += 0.3  # Partial credit
        else:
            issues.append("No KPI attribution")

        # Actionability
        if getattr(pred, "recommendations", None):
            score += 0.5
        else:
            issues.append("No recommendations generated")

        if issues:
            return score, f"Issues: {'; '.join(issues)}"
        return score, "KPIs attributed, recommendations generated"

    def _build_feedback(
        self,
        total: float,
        scores: dict[str, float],
        parts: list[str],
        pred_name: Optional[str],
    ) -> str:
        """Build comprehensive feedback string for GEPA reflection.

        Args:
            total: Overall weighted score
            scores: Component scores dict
            parts: Feedback parts list
            pred_name: Optional predictor name

        Returns:
            Formatted feedback string
        """
        lines = [
            f"Overall: {total:.3f} (ref={scores['refutation']:.2f}, sens={scores['sensitivity']:.2f}, "
            f"meth={scores['methodology']:.2f}, biz={scores['business']:.2f})",
            "",
        ]
        lines.extend(parts)

        if pred_name:
            lines.append(f"\n[Optimizing predictor: {pred_name}]")

        return "\n".join(lines)


__all__ = ["CausalImpactGEPAMetric"]
