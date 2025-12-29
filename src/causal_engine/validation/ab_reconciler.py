"""A/B experiment reconciler for causal estimate validation.

B8.2: ABReconciler for comparing observational causal estimates
with experimental A/B test results.

Validates that DoWhy/EconML/CausalML estimates align with actual
experimental results when available.
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from src.causal_engine.validation.state import (
    ABExperimentResult,
    ABReconciliationResult,
    LibraryEffectEstimate,
)

logger = logging.getLogger(__name__)


class ABReconciler:
    """Reconcile causal estimates with A/B experiment results.

    Compares observational causal inference results (from DoWhy, EconML,
    CausalML) with ground truth from randomized A/B experiments.

    Reconciliation Status Thresholds:
    - excellent: Within CI and < 10% relative difference
    - good: Within CI and < 20% relative difference
    - acceptable: Overlapping CIs or < 30% relative difference
    - poor: > 30% relative difference but same direction
    - failed: Different direction or > 50% relative difference

    Use Cases:
    - Validate observational estimates before scaling interventions
    - Identify model misspecification or unobserved confounding
    - Build confidence in causal estimates for decision-making
    """

    # Reconciliation thresholds
    EXCELLENT_THRESHOLD: float = 0.10  # < 10% relative difference
    GOOD_THRESHOLD: float = 0.20  # < 20% relative difference
    ACCEPTABLE_THRESHOLD: float = 0.30  # < 30% relative difference
    POOR_THRESHOLD: float = 0.50  # < 50% relative difference

    def __init__(
        self,
        excellent_threshold: float = 0.10,
        good_threshold: float = 0.20,
        acceptable_threshold: float = 0.30,
    ):
        """Initialize ABReconciler.

        Args:
            excellent_threshold: Max relative difference for 'excellent' (0-1)
            good_threshold: Max relative difference for 'good' (0-1)
            acceptable_threshold: Max relative difference for 'acceptable' (0-1)
        """
        self.excellent_threshold = excellent_threshold
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold

    async def reconcile(
        self,
        experiment: ABExperimentResult,
        causal_estimates: List[LibraryEffectEstimate],
    ) -> ABReconciliationResult:
        """Reconcile causal estimates with A/B experiment results.

        Args:
            experiment: A/B experiment ground truth result
            causal_estimates: Causal effect estimates from various libraries

        Returns:
            ABReconciliationResult with reconciliation metrics
        """
        start_time = time.time()

        observed_effect = experiment.get("observed_effect", 0.0)
        observed_ci_lower = experiment.get("observed_ci_lower")
        observed_ci_upper = experiment.get("observed_ci_upper")

        # Compute consensus estimate from causal libraries
        estimated_effect = self._compute_weighted_estimate(causal_estimates)
        estimated_ci_lower, estimated_ci_upper = self._compute_aggregate_ci(causal_estimates)

        # Calculate reconciliation metrics
        gap = abs(observed_effect - estimated_effect)
        ratio = observed_effect / estimated_effect if abs(estimated_effect) > 1e-10 else float("inf")
        relative_gap = gap / max(abs(observed_effect), abs(estimated_effect), 1e-10)

        # Check if estimated is within observed CI
        within_ci = self._is_within_ci(
            estimated_effect,
            observed_ci_lower,
            observed_ci_upper,
        )

        # Calculate CI overlap
        ci_overlap = self._compute_ci_overlap(
            estimated_ci_lower,
            estimated_ci_upper,
            observed_ci_lower,
            observed_ci_upper,
        )

        # Agreement assessment
        direction_match = (observed_effect >= 0) == (estimated_effect >= 0)
        magnitude_match = relative_gap < self.acceptable_threshold
        significance_match = self._check_significance_match(
            experiment.get("is_significant", False),
            causal_estimates,
        )

        # Determine status and score
        status, score = self._determine_reconciliation_status(
            relative_gap,
            within_ci,
            ci_overlap,
            direction_match,
        )

        # Generate analysis
        discrepancy_analysis = self._generate_discrepancy_analysis(
            observed_effect,
            estimated_effect,
            relative_gap,
            direction_match,
            within_ci,
            status,
        )

        # Generate recommendations
        recommended_adjustments = self._generate_adjustments(
            status,
            direction_match,
            magnitude_match,
            relative_gap,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return ABReconciliationResult(
            experiment=experiment,
            causal_estimates=causal_estimates,
            observed_vs_estimated_gap=gap,
            observed_vs_estimated_ratio=ratio if ratio != float("inf") else 0.0,
            within_ci=within_ci,
            ci_overlap=ci_overlap,
            direction_match=direction_match,
            magnitude_match=magnitude_match,
            significance_match=significance_match,
            reconciliation_status=status,
            reconciliation_score=score,
            discrepancy_analysis=discrepancy_analysis,
            recommended_adjustments=recommended_adjustments,
            reconciliation_latency_ms=latency_ms,
        )

    def _compute_weighted_estimate(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> float:
        """Compute confidence-weighted consensus estimate.

        Args:
            estimates: List of library effect estimates

        Returns:
            Weighted average effect estimate
        """
        if not estimates:
            return 0.0

        effects = []
        weights = []

        for est in estimates:
            effect = est.get("estimate")
            confidence = est.get("confidence", 0.5)
            if effect is not None:
                effects.append(effect)
                weights.append(confidence)

        if not effects:
            return 0.0

        total_weight = sum(weights)
        if total_weight <= 0:
            return float(np.mean(effects))

        return sum(e * w for e, w in zip(effects, weights)) / total_weight

    def _compute_aggregate_ci(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute aggregate confidence interval from multiple estimates.

        Uses the widest bounds across all library estimates.

        Args:
            estimates: List of library effect estimates

        Returns:
            Tuple of (ci_lower, ci_upper)
        """
        ci_lowers = []
        ci_uppers = []

        for est in estimates:
            ci_lower = est.get("ci_lower")
            ci_upper = est.get("ci_upper")
            if ci_lower is not None:
                ci_lowers.append(ci_lower)
            if ci_upper is not None:
                ci_uppers.append(ci_upper)

        if not ci_lowers or not ci_uppers:
            return None, None

        return min(ci_lowers), max(ci_uppers)

    def _is_within_ci(
        self,
        estimate: float,
        ci_lower: Optional[float],
        ci_upper: Optional[float],
    ) -> bool:
        """Check if estimate falls within confidence interval.

        Args:
            estimate: Effect estimate to check
            ci_lower: CI lower bound
            ci_upper: CI upper bound

        Returns:
            True if estimate is within CI
        """
        if ci_lower is None or ci_upper is None:
            return False
        return ci_lower <= estimate <= ci_upper

    def _compute_ci_overlap(
        self,
        est_ci_lower: Optional[float],
        est_ci_upper: Optional[float],
        obs_ci_lower: Optional[float],
        obs_ci_upper: Optional[float],
    ) -> float:
        """Compute overlap between estimated and observed CIs.

        Args:
            est_ci_lower: Estimated CI lower bound
            est_ci_upper: Estimated CI upper bound
            obs_ci_lower: Observed CI lower bound
            obs_ci_upper: Observed CI upper bound

        Returns:
            Overlap ratio (0-1)
        """
        if any(x is None for x in [est_ci_lower, est_ci_upper, obs_ci_lower, obs_ci_upper]):
            return 0.0

        overlap_lower = max(est_ci_lower, obs_ci_lower)
        overlap_upper = min(est_ci_upper, obs_ci_upper)

        if overlap_upper <= overlap_lower:
            return 0.0

        overlap_width = overlap_upper - overlap_lower
        total_width = max(est_ci_upper, obs_ci_upper) - min(est_ci_lower, obs_ci_lower)

        if total_width <= 0:
            return 0.0

        return overlap_width / total_width

    def _check_significance_match(
        self,
        experiment_significant: bool,
        estimates: List[LibraryEffectEstimate],
    ) -> bool:
        """Check if statistical significance matches between experiment and estimates.

        Args:
            experiment_significant: Whether experiment showed significance
            estimates: Causal effect estimates

        Returns:
            True if majority of estimates agree on significance
        """
        if not estimates:
            return True

        significant_count = 0
        total_with_pvalue = 0

        for est in estimates:
            p_value = est.get("p_value")
            if p_value is not None:
                total_with_pvalue += 1
                if p_value < 0.05:
                    significant_count += 1

        if total_with_pvalue == 0:
            return True

        majority_significant = significant_count > total_with_pvalue / 2
        return majority_significant == experiment_significant

    def _determine_reconciliation_status(
        self,
        relative_gap: float,
        within_ci: bool,
        ci_overlap: float,
        direction_match: bool,
    ) -> tuple[Literal["excellent", "good", "acceptable", "poor", "failed"], float]:
        """Determine reconciliation status and score.

        Args:
            relative_gap: Relative difference between observed and estimated
            within_ci: Whether estimated is within observed CI
            ci_overlap: CI overlap ratio
            direction_match: Whether effects have same direction

        Returns:
            Tuple of (status, score)
        """
        # Failed: direction mismatch or extreme gap
        if not direction_match:
            return "failed", 0.0

        if relative_gap > self.POOR_THRESHOLD:
            return "failed", max(0.0, 0.3 - (relative_gap - self.POOR_THRESHOLD))

        # Excellent: within CI and small gap
        if within_ci and relative_gap < self.excellent_threshold:
            return "excellent", min(1.0, 0.95 + (0.05 * (1 - relative_gap / self.excellent_threshold)))

        # Good: within CI or small gap
        if within_ci or relative_gap < self.good_threshold:
            base_score = 0.75
            gap_penalty = (relative_gap / self.good_threshold) * 0.15
            ci_bonus = 0.10 if within_ci else 0.0
            return "good", min(0.95, base_score - gap_penalty + ci_bonus)

        # Acceptable: overlapping CIs or moderate gap
        if ci_overlap > 0.3 or relative_gap < self.acceptable_threshold:
            base_score = 0.55
            gap_penalty = (relative_gap / self.acceptable_threshold) * 0.15
            overlap_bonus = 0.10 * ci_overlap
            return "acceptable", max(0.4, base_score - gap_penalty + overlap_bonus)

        # Poor: large gap but same direction
        return "poor", max(0.1, 0.4 - (relative_gap - self.acceptable_threshold) * 0.5)

    def _generate_discrepancy_analysis(
        self,
        observed: float,
        estimated: float,
        relative_gap: float,
        direction_match: bool,
        within_ci: bool,
        status: str,
    ) -> str:
        """Generate human-readable discrepancy analysis.

        Args:
            observed: Observed effect from experiment
            estimated: Estimated effect from causal analysis
            relative_gap: Relative difference
            direction_match: Whether directions match
            within_ci: Whether estimated is within observed CI
            status: Reconciliation status

        Returns:
            Discrepancy analysis text
        """
        if status == "excellent":
            return (
                f"Causal estimates ({estimated:.3f}) closely match experimental "
                f"results ({observed:.3f}) with only {relative_gap:.1%} difference. "
                "High confidence in observational analysis."
            )

        if status == "good":
            ci_note = "The estimate falls within the experimental confidence interval. " if within_ci else ""
            return (
                f"Causal estimates ({estimated:.3f}) show good agreement with "
                f"experimental results ({observed:.3f}). {ci_note}"
                f"The {relative_gap:.1%} difference is within acceptable bounds."
            )

        if status == "acceptable":
            return (
                f"Causal estimates ({estimated:.3f}) show moderate agreement with "
                f"experimental results ({observed:.3f}). The {relative_gap:.1%} difference "
                "suggests some model uncertainty. Consider additional validation."
            )

        if status == "poor":
            return (
                f"Causal estimates ({estimated:.3f}) show poor agreement with "
                f"experimental results ({observed:.3f}). The {relative_gap:.1%} difference "
                "indicates potential unobserved confounding or model misspecification."
            )

        # Failed
        if not direction_match:
            return (
                f"CRITICAL: Causal estimates ({estimated:.3f}) predict opposite direction "
                f"from experimental results ({observed:.3f}). This indicates serious "
                "model misspecification or unobserved confounding. Do not use for decisions."
            )
        else:
            return (
                f"Causal estimates ({estimated:.3f}) differ significantly from "
                f"experimental results ({observed:.3f}) with {relative_gap:.1%} gap. "
                "Observational estimates should not be trusted without investigation."
            )

    def _generate_adjustments(
        self,
        status: str,
        direction_match: bool,
        magnitude_match: bool,
        relative_gap: float,
    ) -> List[str]:
        """Generate recommended adjustments based on reconciliation.

        Args:
            status: Reconciliation status
            direction_match: Whether directions match
            magnitude_match: Whether magnitudes are close
            relative_gap: Relative difference

        Returns:
            List of recommended adjustments
        """
        adjustments = []

        if status == "excellent":
            adjustments.append("Observational estimates are validated - proceed with confidence.")
            adjustments.append("Consider using causal estimates for similar future analyses.")
            return adjustments

        if status == "good":
            adjustments.append("Estimates are validated but monitor for drift in future experiments.")
            if not magnitude_match:
                adjustments.append(
                    f"Consider applying {relative_gap:.0%} adjustment factor for conservative estimates."
                )
            return adjustments

        if status == "acceptable":
            adjustments.append("Run additional robustness tests before relying on causal estimates.")
            adjustments.append("Consider expanding confounder set or using instrumental variables.")
            adjustments.append(
                f"Apply {relative_gap:.0%} uncertainty adjustment when using estimates for decisions."
            )
            return adjustments

        if status == "poor":
            adjustments.append("Investigate potential unobserved confounders.")
            adjustments.append("Review causal graph assumptions and data quality.")
            adjustments.append("Consider alternative estimation methods (IV, RDD, DiD).")
            adjustments.append("Run sensitivity analysis on confounding strength.")
            return adjustments

        # Failed
        adjustments.append("DO NOT use causal estimates for business decisions.")
        if not direction_match:
            adjustments.append(
                "CRITICAL: Direction mismatch - completely review causal model and assumptions."
            )
            adjustments.append("Check for confounders with opposite effects on treatment and outcome.")
        else:
            adjustments.append("Large discrepancy suggests severe model misspecification.")

        adjustments.append("Conduct thorough investigation before re-running analysis.")
        adjustments.append("Consider if experimental design matches observational data context.")
        return adjustments

    async def create_calibration_factor(
        self,
        reconciliation: ABReconciliationResult,
    ) -> Dict[str, Any]:
        """Create calibration factor based on reconciliation results.

        Generates a factor that can adjust future causal estimates
        based on the observed vs estimated relationship.

        Args:
            reconciliation: ABReconciliationResult from reconcile()

        Returns:
            Calibration factor dictionary
        """
        ratio = reconciliation.get("observed_vs_estimated_ratio", 1.0)
        status = reconciliation.get("reconciliation_status", "failed")

        # Only create calibration for acceptable+ reconciliations
        if status in ["failed", "poor"]:
            return {
                "applicable": False,
                "reason": f"Reconciliation status '{status}' is too poor for calibration",
                "factor": None,
            }

        # Calibration factor bounds
        if 0.5 <= ratio <= 2.0:  # Within 2x
            confidence = reconciliation.get("reconciliation_score", 0.5)
            return {
                "applicable": True,
                "factor": ratio,
                "confidence": confidence,
                "direction": "multiply" if ratio > 1 else "divide" if ratio < 1 else "none",
                "adjustment_percentage": (ratio - 1) * 100,
                "recommendation": (
                    f"Apply {abs(ratio - 1) * 100:.1f}% "
                    f"{'increase' if ratio > 1 else 'decrease'} to future estimates"
                    if abs(ratio - 1) > 0.05
                    else "No adjustment needed"
                ),
            }
        else:
            return {
                "applicable": False,
                "reason": "Ratio outside calibration bounds (0.5x - 2.0x)",
                "factor": ratio,
            }
