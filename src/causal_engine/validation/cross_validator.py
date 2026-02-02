"""Cross-library validator for DoWhy ↔ CausalML comparison.

B8.1: CrossValidator for pairwise validation between causal libraries.

Validates effect estimates by comparing:
1. DoWhy ATE vs CausalML Uplift
2. EconML CATE vs CausalML segment uplift
3. Refutation test consistency across libraries
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from src.causal_engine.validation.state import (
    CrossValidationResult,
    LibraryEffectEstimate,
    PairwiseValidation,
    RefutationValidation,
    ValidationSummary,
)

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-library validator for causal effect estimates.

    Performs pairwise validation between DoWhy, EconML, and CausalML
    to ensure consistent causal effect estimates across libraries.

    Validation Criteria:
    - Effect direction agreement (both positive or both negative)
    - Magnitude agreement (within relative tolerance)
    - Statistical significance agreement
    - Confidence interval overlap

    Thresholds:
    - AGREEMENT_THRESHOLD: 0.8 (80% agreement required to pass)
    - RELATIVE_DIFFERENCE_THRESHOLD: 0.3 (max 30% relative difference)
    - CI_OVERLAP_THRESHOLD: 0.5 (at least 50% CI overlap)
    """

    # Validation thresholds
    AGREEMENT_THRESHOLD: float = 0.8
    RELATIVE_DIFFERENCE_THRESHOLD: float = 0.3
    CI_OVERLAP_THRESHOLD: float = 0.5
    DIRECTION_WEIGHT: float = 0.4
    MAGNITUDE_WEIGHT: float = 0.3
    SIGNIFICANCE_WEIGHT: float = 0.2
    CI_OVERLAP_WEIGHT: float = 0.1

    def __init__(
        self,
        agreement_threshold: float = 0.8,
        relative_difference_threshold: float = 0.3,
        ci_overlap_threshold: float = 0.5,
    ):
        """Initialize CrossValidator.

        Args:
            agreement_threshold: Minimum overall agreement to pass (0-1)
            relative_difference_threshold: Max relative difference between estimates (0-1)
            ci_overlap_threshold: Minimum CI overlap to consider agreement (0-1)
        """
        self.agreement_threshold = agreement_threshold
        self.relative_difference_threshold = relative_difference_threshold
        self.ci_overlap_threshold = ci_overlap_threshold

    async def validate(
        self,
        treatment_var: str,
        outcome_var: str,
        estimates: List[LibraryEffectEstimate],
        run_refutation_comparison: bool = True,
    ) -> CrossValidationResult:
        """Perform cross-library validation on effect estimates.

        Args:
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            estimates: List of effect estimates from different libraries
            run_refutation_comparison: Whether to compare refutation results

        Returns:
            CrossValidationResult with pairwise comparisons and summary
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []

        if len(estimates) < 2:
            return CrossValidationResult(
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                validation_type="dowhy_causalml",
                estimates=estimates,
                pairwise_results=[],
                summary=ValidationSummary(
                    overall_status="failed",
                    overall_agreement=0.0,
                    pairwise_validations=[],
                    libraries_validated=[e.get("library", "unknown") for e in estimates],
                    consensus_effect=None,
                    consensus_confidence=0.0,
                    discrepancies=["Insufficient estimates for cross-validation (need >= 2)"],
                    recommendations=["Run analysis with multiple libraries"],
                ),
                validation_latency_ms=int((time.time() - start_time) * 1000),
                total_latency_ms=int((time.time() - start_time) * 1000),
                status="failed",
                errors=["Need at least 2 library estimates for cross-validation"],
                warnings=[],
            )

        # Perform pairwise validations
        pairwise_results: List[PairwiseValidation] = []
        libraries_validated: List[str] = []

        for i, est_a in enumerate(estimates):
            lib_a = est_a.get("library", "unknown")
            if lib_a not in libraries_validated:
                libraries_validated.append(lib_a)

            for est_b in estimates[i + 1 :]:
                lib_b = est_b.get("library", "unknown")
                if lib_b not in libraries_validated:
                    libraries_validated.append(lib_b)

                try:
                    pairwise = self._compare_estimates(est_a, est_b)
                    pairwise_results.append(pairwise)
                except Exception as e:
                    logger.warning(f"Pairwise comparison failed for {lib_a} vs {lib_b}: {e}")
                    warnings.append(f"Comparison {lib_a} vs {lib_b} failed: {str(e)}")

        # Compute summary
        summary = self._compute_summary(pairwise_results, estimates, libraries_validated)

        # Determine validation status
        status: Literal["completed", "partial", "failed"]
        if summary["overall_status"] == "failed":
            status = "failed"
        elif len(pairwise_results) < len(estimates) * (len(estimates) - 1) // 2:
            status = "partial"
        else:
            status = "completed"

        total_latency_ms = int((time.time() - start_time) * 1000)

        return CrossValidationResult(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            validation_type=self._determine_validation_type(libraries_validated),
            estimates=estimates,
            pairwise_results=pairwise_results,
            summary=summary,
            validation_latency_ms=total_latency_ms,
            total_latency_ms=total_latency_ms,
            status=status,
            errors=errors,
            warnings=warnings,
        )

    def _compare_estimates(
        self,
        est_a: LibraryEffectEstimate,
        est_b: LibraryEffectEstimate,
    ) -> PairwiseValidation:
        """Compare two library effect estimates.

        Args:
            est_a: First library estimate
            est_b: Second library estimate

        Returns:
            PairwiseValidation result
        """
        lib_a = est_a.get("library", "unknown")
        lib_b = est_b.get("library", "unknown")
        effect_a = est_a.get("estimate", 0.0)
        effect_b = est_b.get("estimate", 0.0)

        # Calculate differences
        absolute_difference = abs(effect_a - effect_b)
        max_abs = max(abs(effect_a), abs(effect_b), 1e-10)  # Avoid division by zero

        # Handle near-zero edge case: if both effects are very small, consider them in agreement
        both_near_zero = abs(effect_a) < 0.05 and abs(effect_b) < 0.05
        if both_near_zero:
            relative_difference = absolute_difference / 0.05  # Normalize to reasonable scale
        else:
            relative_difference = absolute_difference / max_abs

        # Agreement score (inverse of relative difference, capped at 1.0)
        agreement_score = max(0.0, 1.0 - relative_difference)

        # Direction agreement: same sign, or both near zero
        direction_agreement = (effect_a >= 0) == (effect_b >= 0) or both_near_zero

        # Significance agreement
        sig_a = est_a.get("p_value") is not None and est_a.get("p_value", 1.0) < 0.05
        sig_b = est_b.get("p_value") is not None and est_b.get("p_value", 1.0) < 0.05
        significance_agreement = sig_a == sig_b

        # CI overlap
        ci_overlap = self._compute_ci_overlap(
            est_a.get("ci_lower"),
            est_a.get("ci_upper"),
            est_b.get("ci_lower"),
            est_b.get("ci_upper"),
        )

        # Validation status
        validation_status = self._determine_pairwise_status(
            agreement_score,
            direction_agreement,
            significance_agreement,
            ci_overlap,
        )

        # Validation message
        validation_message = self._generate_validation_message(
            lib_a,
            lib_b,
            agreement_score,
            direction_agreement,
            significance_agreement,
            validation_status,
        )

        return PairwiseValidation(
            library_a=lib_a,
            library_b=lib_b,
            effect_a=effect_a,
            effect_b=effect_b,
            absolute_difference=absolute_difference,
            relative_difference=relative_difference,
            agreement_score=agreement_score,
            direction_agreement=direction_agreement,
            significance_agreement=significance_agreement,
            ci_overlap=ci_overlap,
            validation_status=validation_status,
            validation_message=validation_message,
        )

    def _compute_ci_overlap(
        self,
        ci_a_lower: Optional[float],
        ci_a_upper: Optional[float],
        ci_b_lower: Optional[float],
        ci_b_upper: Optional[float],
    ) -> Optional[float]:
        """Compute overlap between two confidence intervals.

        Args:
            ci_a_lower: CI lower bound for estimate A
            ci_a_upper: CI upper bound for estimate A
            ci_b_lower: CI lower bound for estimate B
            ci_b_upper: CI upper bound for estimate B

        Returns:
            Overlap ratio (0-1) or None if CIs unavailable
        """
        if any(x is None for x in [ci_a_lower, ci_a_upper, ci_b_lower, ci_b_upper]):
            return None

        # Compute overlap
        overlap_lower = max(ci_a_lower, ci_b_lower)
        overlap_upper = min(ci_a_upper, ci_b_upper)

        if overlap_upper <= overlap_lower:
            return 0.0  # No overlap

        overlap_width = overlap_upper - overlap_lower
        total_width = max(ci_a_upper, ci_b_upper) - min(ci_a_lower, ci_b_lower)

        if total_width <= 0:
            return 0.0

        return overlap_width / total_width

    def _determine_pairwise_status(
        self,
        agreement_score: float,
        direction_agreement: bool,
        significance_agreement: bool,
        ci_overlap: Optional[float],
    ) -> Literal["passed", "warning", "failed"]:
        """Determine validation status for a pairwise comparison.

        Args:
            agreement_score: Effect magnitude agreement (0-1)
            direction_agreement: Whether effects have same direction
            significance_agreement: Whether significance matches
            ci_overlap: CI overlap ratio (0-1)

        Returns:
            Validation status
        """
        # Critical failure: direction disagreement
        if not direction_agreement:
            return "failed"

        # High agreement + significance match = passed
        if agreement_score >= self.agreement_threshold and significance_agreement:
            return "passed"

        # Moderate agreement or CI overlap = warning
        if agreement_score >= 0.5:
            return "warning"

        if ci_overlap is not None and ci_overlap >= self.ci_overlap_threshold:
            return "warning"

        return "failed"

    def _generate_validation_message(
        self,
        lib_a: str,
        lib_b: str,
        agreement_score: float,
        direction_agreement: bool,
        significance_agreement: bool,
        status: Literal["passed", "warning", "failed"],
    ) -> str:
        """Generate human-readable validation message.

        Args:
            lib_a: First library name
            lib_b: Second library name
            agreement_score: Effect magnitude agreement
            direction_agreement: Whether effects agree on direction
            significance_agreement: Whether significance matches
            status: Overall validation status

        Returns:
            Human-readable message
        """
        if status == "passed":
            return (
                f"{lib_a} and {lib_b} show strong agreement "
                f"({agreement_score:.1%}) with consistent direction and significance."
            )
        elif status == "warning":
            issues = []
            if agreement_score < self.agreement_threshold:
                issues.append(f"moderate magnitude difference ({1 - agreement_score:.1%})")
            if not significance_agreement:
                issues.append("significance disagreement")
            return f"{lib_a} and {lib_b} show partial agreement with {', '.join(issues)}."
        else:
            issues = []
            if not direction_agreement:
                issues.append("opposite effect directions")
            if agreement_score < 0.5:
                issues.append(f"large magnitude difference ({1 - agreement_score:.1%})")
            return f"{lib_a} and {lib_b} show disagreement: {', '.join(issues)}."

    def _compute_summary(
        self,
        pairwise_results: List[PairwiseValidation],
        estimates: List[LibraryEffectEstimate],
        libraries_validated: List[str],
    ) -> ValidationSummary:
        """Compute overall validation summary.

        Args:
            pairwise_results: All pairwise validation results
            estimates: Original effect estimates
            libraries_validated: List of libraries that were validated

        Returns:
            ValidationSummary with overall metrics and recommendations
        """
        if not pairwise_results:
            return ValidationSummary(
                overall_status="failed",
                overall_agreement=0.0,
                pairwise_validations=pairwise_results,
                libraries_validated=libraries_validated,
                consensus_effect=None,
                consensus_confidence=0.0,
                discrepancies=["No pairwise comparisons completed"],
                recommendations=["Run validation with multiple successful library estimates"],
            )

        # Calculate overall agreement (weighted average)
        total_agreement = sum(p.get("agreement_score", 0.0) for p in pairwise_results)
        overall_agreement = total_agreement / len(pairwise_results) if pairwise_results else 0.0

        # Count statuses
        sum(1 for p in pairwise_results if p.get("validation_status") == "passed")
        sum(1 for p in pairwise_results if p.get("validation_status") == "warning")
        failed_count = sum(1 for p in pairwise_results if p.get("validation_status") == "failed")

        # Determine overall status
        if failed_count == 0 and overall_agreement >= self.agreement_threshold:
            overall_status: Literal["passed", "warning", "failed"] = "passed"
        elif failed_count <= len(pairwise_results) // 2:
            overall_status = "warning"
        else:
            overall_status = "failed"

        # Compute consensus effect (confidence-weighted)
        consensus_effect, consensus_confidence = self._compute_consensus(estimates)

        # Identify discrepancies
        discrepancies = self._identify_discrepancies(pairwise_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_status,
            pairwise_results,
            discrepancies,
        )

        return ValidationSummary(
            overall_status=overall_status,
            overall_agreement=overall_agreement,
            pairwise_validations=pairwise_results,
            libraries_validated=libraries_validated,
            consensus_effect=consensus_effect,
            consensus_confidence=consensus_confidence,
            discrepancies=discrepancies,
            recommendations=recommendations,
        )

    def _compute_consensus(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> Tuple[Optional[float], float]:
        """Compute confidence-weighted consensus effect estimate.

        Args:
            estimates: List of library effect estimates

        Returns:
            Tuple of (consensus_effect, consensus_confidence)
        """
        if not estimates:
            return None, 0.0

        # Extract effects and confidences
        effects = []
        weights = []
        for est in estimates:
            effect = est.get("estimate")
            confidence = est.get("confidence", 0.5)
            if effect is not None:
                effects.append(effect)
                weights.append(confidence)

        if not effects:
            return None, 0.0

        # Confidence-weighted average
        total_weight = sum(weights)
        if total_weight <= 0:
            return float(np.mean(effects)), 0.5

        consensus_effect = sum(e * w for e, w in zip(effects, weights, strict=False)) / total_weight

        # Consensus confidence based on agreement
        effect_std = float(np.std(effects)) if len(effects) > 1 else 0.0
        max_abs_effect = max(abs(e) for e in effects)
        if max_abs_effect > 0:
            cv = effect_std / max_abs_effect  # Coefficient of variation
            consensus_confidence = max(0.0, 1.0 - cv)
        else:
            consensus_confidence = 1.0 if len(effects) > 1 else 0.5

        return consensus_effect, min(consensus_confidence, float(np.mean(weights)))

    def _identify_discrepancies(
        self,
        pairwise_results: List[PairwiseValidation],
    ) -> List[str]:
        """Identify specific discrepancies from pairwise comparisons.

        Args:
            pairwise_results: All pairwise validation results

        Returns:
            List of discrepancy descriptions
        """
        discrepancies = []

        for result in pairwise_results:
            if result.get("validation_status") == "failed":
                lib_a = result.get("library_a", "unknown")
                lib_b = result.get("library_b", "unknown")

                if not result.get("direction_agreement", True):
                    effect_a = result.get("effect_a", 0)
                    effect_b = result.get("effect_b", 0)
                    discrepancies.append(
                        f"Direction disagreement: {lib_a} ({effect_a:+.3f}) vs {lib_b} ({effect_b:+.3f})"
                    )
                elif result.get("relative_difference", 0) > self.relative_difference_threshold:
                    rel_diff = result.get("relative_difference", 0)
                    discrepancies.append(
                        f"Magnitude disagreement: {lib_a} vs {lib_b} ({rel_diff:.1%} difference)"
                    )

            elif result.get("validation_status") == "warning":
                if not result.get("significance_agreement", True):
                    lib_a = result.get("library_a", "unknown")
                    lib_b = result.get("library_b", "unknown")
                    discrepancies.append(f"Significance disagreement: {lib_a} vs {lib_b}")

        return discrepancies

    def _generate_recommendations(
        self,
        overall_status: Literal["passed", "warning", "failed"],
        pairwise_results: List[PairwiseValidation],
        discrepancies: List[str],
    ) -> List[str]:
        """Generate actionable recommendations based on validation results.

        Args:
            overall_status: Overall validation status
            pairwise_results: All pairwise validation results
            discrepancies: Identified discrepancies

        Returns:
            List of recommendations
        """
        recommendations = []

        if overall_status == "passed":
            recommendations.append(
                "Effect estimates are consistent across libraries - high confidence in results."
            )
            recommendations.append("Consider using consensus effect for downstream analysis.")
        elif overall_status == "warning":
            recommendations.append(
                "Review effect estimates for potential methodological differences."
            )
            if any("significance" in d.lower() for d in discrepancies):
                recommendations.append(
                    "Check sample sizes and power across library implementations."
                )
            recommendations.append("Consider running additional robustness tests.")
        else:
            recommendations.append(
                "Effect estimates show significant disagreement - investigate causes."
            )

            # Direction disagreement recommendations
            if any("direction" in d.lower() for d in discrepancies):
                recommendations.append(
                    "Direction disagreement suggests potential confounding or model misspecification."
                )
                recommendations.append("Review causal graph assumptions and confounder selection.")

            # Magnitude disagreement recommendations
            if any("magnitude" in d.lower() for d in discrepancies):
                recommendations.append(
                    "Magnitude disagreement may indicate different effect definitions (ATE vs CATE)."
                )
                recommendations.append(
                    "Ensure all libraries use consistent treatment/outcome definitions."
                )

            recommendations.append(
                "Consider collecting more data or running A/B experiment to validate."
            )

        return recommendations

    def _determine_validation_type(
        self,
        libraries: List[str],
    ) -> Literal["dowhy_causalml", "econml_causalml", "full_pipeline"]:
        """Determine validation type based on libraries used.

        Args:
            libraries: List of libraries validated

        Returns:
            Validation type identifier
        """
        has_dowhy = "dowhy" in libraries
        has_econml = "econml" in libraries
        has_causalml = "causalml" in libraries
        has_networkx = "networkx" in libraries

        if has_networkx and has_dowhy and has_econml and has_causalml:
            return "full_pipeline"
        elif has_econml and has_causalml:
            return "econml_causalml"
        else:
            return "dowhy_causalml"

    async def compare_refutations(
        self,
        dowhy_refutations: Dict[str, Any],
        causalml_stability: Dict[str, Any],
    ) -> List[RefutationValidation]:
        """Compare refutation test results between DoWhy and CausalML.

        Args:
            dowhy_refutations: DoWhy refutation test results
            causalml_stability: CausalML uplift stability metrics

        Returns:
            List of refutation comparison results
        """
        validations = []

        # Compare placebo treatment
        if "placebo_treatment" in dowhy_refutations:
            dowhy_placebo = dowhy_refutations["placebo_treatment"]
            causalml_stable = causalml_stability.get("placebo_stable", False)

            validations.append(
                RefutationValidation(
                    test_name="placebo_treatment",
                    dowhy_passed=dowhy_placebo.get("passed", False),
                    dowhy_new_effect=dowhy_placebo.get("new_effect"),
                    causalml_consistent=causalml_stable,
                    causalml_uplift_stable=causalml_stability.get("placebo_uplift_std"),
                    cross_validation_passed=dowhy_placebo.get("passed", False) and causalml_stable,
                    discrepancy_reason=None
                    if dowhy_placebo.get("passed") == causalml_stable
                    else "Libraries disagree on placebo treatment stability",
                )
            )

        # Compare data subset validation
        if "data_subset" in dowhy_refutations:
            dowhy_subset = dowhy_refutations["data_subset"]
            causalml_subset_stable = causalml_stability.get("subset_stable", False)

            validations.append(
                RefutationValidation(
                    test_name="data_subset_validation",
                    dowhy_passed=dowhy_subset.get("passed", False),
                    dowhy_new_effect=dowhy_subset.get("new_effect"),
                    causalml_consistent=causalml_subset_stable,
                    causalml_uplift_stable=causalml_stability.get("subset_uplift_std"),
                    cross_validation_passed=dowhy_subset.get("passed", False)
                    and causalml_subset_stable,
                    discrepancy_reason=None
                    if dowhy_subset.get("passed") == causalml_subset_stable
                    else "Libraries disagree on data subset stability",
                )
            )

        # Compare bootstrap stability
        if "bootstrap" in dowhy_refutations or "bootstrap_stable" in causalml_stability:
            dowhy_bootstrap = dowhy_refutations.get("bootstrap", {})
            causalml_bootstrap_stable = causalml_stability.get("bootstrap_stable", False)

            validations.append(
                RefutationValidation(
                    test_name="bootstrap",
                    dowhy_passed=dowhy_bootstrap.get("passed", False) if dowhy_bootstrap else None,
                    dowhy_new_effect=dowhy_bootstrap.get("new_effect"),
                    causalml_consistent=causalml_bootstrap_stable,
                    causalml_uplift_stable=causalml_stability.get("bootstrap_uplift_std"),
                    cross_validation_passed=dowhy_bootstrap.get("passed", False)
                    if dowhy_bootstrap
                    else causalml_bootstrap_stable,
                    discrepancy_reason=None,
                )
            )

        return validations
