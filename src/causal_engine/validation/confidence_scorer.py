"""Confidence scoring based on library agreement.

B8.4: Compute confidence scores for causal effect estimates based on:
- Cross-library agreement (DoWhy, EconML, CausalML)
- Statistical measures (CI overlap, p-values)
- Refutation test results
- A/B experiment reconciliation (if available)

The confidence score is used to:
- Adjust ROI estimates in gap_analyzer
- Weight consensus effects in multi-library pipelines
- Gate downstream decisions based on reliability
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from src.causal_engine.validation.state import (
    ABReconciliationResult,
    CrossValidationResult,
    LibraryEffectEstimate,
    PairwiseValidation,
)

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Score confidence in causal effect estimates based on multi-library agreement.

    Confidence Components (weights configurable):
    - Agreement Score (40%): How well libraries agree on effect magnitude
    - Direction Consistency (25%): All libraries agree on effect direction
    - Statistical Significance (20%): Consistency of p-values across libraries
    - CI Overlap (15%): Confidence interval overlap between libraries

    Confidence Tiers:
    - HIGH (≥0.8): Effect is well-validated, suitable for decisions
    - MODERATE (≥0.6): Effect is reasonably reliable, use with bounds
    - LOW (≥0.4): Effect has significant uncertainty, use with caution
    - VERY_LOW (<0.4): Effect is unreliable, requires more validation

    Usage:
        scorer = ConfidenceScorer()
        score, tier = scorer.score_from_cross_validation(cross_val_result)
        adjusted_roi = roi_estimate * scorer.get_adjustment_factor(tier)
    """

    # Default component weights
    AGREEMENT_WEIGHT: float = 0.40
    DIRECTION_WEIGHT: float = 0.25
    SIGNIFICANCE_WEIGHT: float = 0.20
    CI_OVERLAP_WEIGHT: float = 0.15

    # Tier thresholds
    HIGH_THRESHOLD: float = 0.80
    MODERATE_THRESHOLD: float = 0.60
    LOW_THRESHOLD: float = 0.40

    # ROI adjustment factors per tier
    ADJUSTMENT_FACTORS: Dict[str, float] = {
        "high": 1.0,  # Full ROI
        "moderate": 0.85,  # 15% reduction
        "low": 0.65,  # 35% reduction
        "very_low": 0.40,  # 60% reduction
    }

    def __init__(
        self,
        agreement_weight: float = 0.40,
        direction_weight: float = 0.25,
        significance_weight: float = 0.20,
        ci_overlap_weight: float = 0.15,
    ):
        """Initialize ConfidenceScorer.

        Args:
            agreement_weight: Weight for agreement score component
            direction_weight: Weight for direction consistency component
            significance_weight: Weight for statistical significance component
            ci_overlap_weight: Weight for CI overlap component
        """
        # Normalize weights
        total = agreement_weight + direction_weight + significance_weight + ci_overlap_weight
        self.agreement_weight = agreement_weight / total
        self.direction_weight = direction_weight / total
        self.significance_weight = significance_weight / total
        self.ci_overlap_weight = ci_overlap_weight / total

    def score_from_estimates(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> Tuple[float, Literal["high", "moderate", "low", "very_low"]]:
        """Score confidence directly from library estimates.

        Args:
            estimates: List of effect estimates from different libraries

        Returns:
            Tuple of (confidence_score, confidence_tier)
        """
        if not estimates:
            return 0.0, "very_low"

        if len(estimates) == 1:
            # Single library - use its own confidence
            single_confidence = estimates[0].get("confidence", 0.5)
            return single_confidence, self._score_to_tier(single_confidence)

        # Compute components
        agreement = self._compute_agreement_score(estimates)
        direction = self._compute_direction_consistency(estimates)
        significance = self._compute_significance_consistency(estimates)
        ci_overlap = self._compute_avg_ci_overlap(estimates)

        # Weighted score
        score = (
            self.agreement_weight * agreement
            + self.direction_weight * direction
            + self.significance_weight * significance
            + self.ci_overlap_weight * ci_overlap
        )

        return score, self._score_to_tier(score)

    def score_from_cross_validation(
        self,
        cross_validation: CrossValidationResult,
    ) -> Tuple[float, Literal["high", "moderate", "low", "very_low"]]:
        """Score confidence from cross-validation result.

        Args:
            cross_validation: CrossValidationResult from CrossValidator

        Returns:
            Tuple of (confidence_score, confidence_tier)
        """
        summary = cross_validation.get("summary", {})
        pairwise = cross_validation.get("pairwise_results", [])
        estimates = cross_validation.get("estimates", [])

        if not pairwise:
            # No pairwise results - fall back to estimates
            return self.score_from_estimates(estimates)

        # Agreement from summary
        agreement = summary.get("overall_agreement", 0.0)

        # Direction consistency from pairwise
        direction_matches = [p.get("direction_agreement", False) for p in pairwise]
        direction = sum(direction_matches) / len(direction_matches) if direction_matches else 0.0

        # Significance consistency from pairwise
        sig_matches = [p.get("significance_agreement", False) for p in pairwise]
        significance = sum(sig_matches) / len(sig_matches) if sig_matches else 0.5

        # CI overlap from pairwise (average of available)
        ci_overlaps = [p.get("ci_overlap") for p in pairwise if p.get("ci_overlap") is not None]
        ci_overlap = sum(ci_overlaps) / len(ci_overlaps) if ci_overlaps else 0.5

        # Weighted score
        score = (
            self.agreement_weight * agreement
            + self.direction_weight * direction
            + self.significance_weight * significance
            + self.ci_overlap_weight * ci_overlap
        )

        return score, self._score_to_tier(score)

    def score_with_ab_reconciliation(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: ABReconciliationResult,
    ) -> Tuple[float, Literal["high", "moderate", "low", "very_low"]]:
        """Score confidence combining cross-validation and A/B reconciliation.

        A/B reconciliation carries more weight as it provides ground truth.

        Args:
            cross_validation: CrossValidationResult (optional)
            ab_reconciliation: ABReconciliationResult with experiment comparison

        Returns:
            Tuple of (confidence_score, confidence_tier)
        """
        # A/B score from reconciliation
        ab_score = ab_reconciliation.get("reconciliation_score", 0.0)
        ab_status = ab_reconciliation.get("reconciliation_status", "failed")

        # Apply direction penalty for A/B
        if not ab_reconciliation.get("direction_match", True):
            ab_score = min(ab_score, 0.2)  # Cap at 0.2 for direction mismatch

        # Cross-validation score
        if cross_validation:
            cv_score, _ = self.score_from_cross_validation(cross_validation)
        else:
            cv_score = ab_score  # Use AB score if no CV

        # Combined score: A/B gets 60% weight (ground truth is more reliable)
        combined = 0.6 * ab_score + 0.4 * cv_score

        # Apply status penalty
        status_penalties = {
            "excellent": 0.0,
            "good": 0.0,
            "acceptable": 0.05,
            "poor": 0.15,
            "failed": 0.30,
        }
        penalty = status_penalties.get(ab_status, 0.0)
        final_score = max(0.0, combined - penalty)

        return final_score, self._score_to_tier(final_score)

    def _compute_agreement_score(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> float:
        """Compute agreement score based on effect magnitude similarity.

        Args:
            estimates: List of effect estimates

        Returns:
            Agreement score (0-1)
        """
        effects = [e.get("estimate", 0.0) for e in estimates if e.get("estimate") is not None]

        if len(effects) < 2:
            return 0.5

        # Coefficient of variation (CV) - lower is better
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)

        if abs(mean_effect) < 1e-10:
            # Near-zero mean - check if all are near zero
            return 1.0 if std_effect < 0.1 else 0.5

        cv = std_effect / abs(mean_effect)

        # Convert CV to agreement score (CV of 0 = 1.0, CV of 1+ = 0.0)
        return max(0.0, 1.0 - cv)

    def _compute_direction_consistency(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> float:
        """Compute direction consistency (all positive or all negative).

        Args:
            estimates: List of effect estimates

        Returns:
            Direction consistency score (0-1)
        """
        effects = [e.get("estimate", 0.0) for e in estimates if e.get("estimate") is not None]

        if len(effects) < 2:
            return 1.0  # Single estimate = consistent

        # Count positive, negative, and near-zero
        positive = sum(1 for e in effects if e > 0.01)
        negative = sum(1 for e in effects if e < -0.01)
        zero = len(effects) - positive - negative

        # All same direction or all near-zero = 1.0
        if positive == len(effects) or negative == len(effects) or zero == len(effects):
            return 1.0

        # Mixed positive/negative = 0.0
        if positive > 0 and negative > 0:
            return 0.0

        # Some zero, rest same direction = 0.8
        return 0.8

    def _compute_significance_consistency(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> float:
        """Compute consistency of statistical significance across libraries.

        Args:
            estimates: List of effect estimates

        Returns:
            Significance consistency score (0-1)
        """
        significances = []
        for e in estimates:
            p_value = e.get("p_value")
            if p_value is not None:
                significances.append(p_value < 0.05)

        if len(significances) < 2:
            return 0.5  # Insufficient data

        # All agree = 1.0, mixed = fraction that agrees with majority
        if all(significances) or not any(significances):
            return 1.0

        majority_significant = sum(significances) > len(significances) / 2
        agreement = sum(
            1 for s in significances if s == majority_significant
        ) / len(significances)

        return agreement

    def _compute_avg_ci_overlap(
        self,
        estimates: List[LibraryEffectEstimate],
    ) -> float:
        """Compute average CI overlap across all pairs.

        Args:
            estimates: List of effect estimates

        Returns:
            Average CI overlap (0-1)
        """
        overlaps = []

        for i, est_a in enumerate(estimates):
            ci_a_lower = est_a.get("ci_lower")
            ci_a_upper = est_a.get("ci_upper")

            if ci_a_lower is None or ci_a_upper is None:
                continue

            for est_b in estimates[i + 1:]:
                ci_b_lower = est_b.get("ci_lower")
                ci_b_upper = est_b.get("ci_upper")

                if ci_b_lower is None or ci_b_upper is None:
                    continue

                overlap = self._compute_ci_overlap_pair(
                    ci_a_lower, ci_a_upper, ci_b_lower, ci_b_upper
                )
                overlaps.append(overlap)

        if not overlaps:
            return 0.5  # No CI data available

        return float(np.mean(overlaps))

    def _compute_ci_overlap_pair(
        self,
        ci_a_lower: float,
        ci_a_upper: float,
        ci_b_lower: float,
        ci_b_upper: float,
    ) -> float:
        """Compute overlap between two CIs.

        Args:
            ci_a_lower: CI lower bound for A
            ci_a_upper: CI upper bound for A
            ci_b_lower: CI lower bound for B
            ci_b_upper: CI upper bound for B

        Returns:
            Overlap ratio (0-1)
        """
        overlap_lower = max(ci_a_lower, ci_b_lower)
        overlap_upper = min(ci_a_upper, ci_b_upper)

        if overlap_upper <= overlap_lower:
            return 0.0

        overlap_width = overlap_upper - overlap_lower
        total_width = max(ci_a_upper, ci_b_upper) - min(ci_a_lower, ci_b_lower)

        if total_width <= 0:
            return 0.0

        return overlap_width / total_width

    def _score_to_tier(
        self,
        score: float,
    ) -> Literal["high", "moderate", "low", "very_low"]:
        """Convert score to confidence tier.

        Args:
            score: Confidence score (0-1)

        Returns:
            Confidence tier
        """
        if score >= self.HIGH_THRESHOLD:
            return "high"
        elif score >= self.MODERATE_THRESHOLD:
            return "moderate"
        elif score >= self.LOW_THRESHOLD:
            return "low"
        else:
            return "very_low"

    def get_adjustment_factor(
        self,
        tier: Literal["high", "moderate", "low", "very_low"],
    ) -> float:
        """Get ROI adjustment factor for confidence tier.

        Use this to adjust ROI estimates based on validation confidence.

        Args:
            tier: Confidence tier

        Returns:
            Adjustment factor (0.4 - 1.0)
        """
        return self.ADJUSTMENT_FACTORS.get(tier, 0.4)

    def compute_consensus_estimate(
        self,
        estimates: List[LibraryEffectEstimate],
        use_confidence_weighting: bool = True,
    ) -> Tuple[float, float, float]:
        """Compute confidence-weighted consensus estimate.

        Args:
            estimates: List of effect estimates
            use_confidence_weighting: Whether to weight by library confidence

        Returns:
            Tuple of (consensus_effect, consensus_std, consensus_confidence)
        """
        effects = []
        weights = []

        for est in estimates:
            effect = est.get("estimate")
            if effect is None:
                continue

            effects.append(effect)

            if use_confidence_weighting:
                confidence = est.get("confidence", 0.5)
                weights.append(confidence)
            else:
                weights.append(1.0)

        if not effects:
            return 0.0, float("inf"), 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight <= 0:
            consensus = float(np.mean(effects))
        else:
            consensus = sum(e * w for e, w in zip(effects, weights)) / total_weight

        # Standard deviation
        std = float(np.std(effects)) if len(effects) > 1 else 0.0

        # Consensus confidence (higher agreement = higher confidence)
        cv = std / abs(consensus) if abs(consensus) > 1e-10 else 0.0
        consensus_confidence = max(0.0, 1.0 - cv)

        return consensus, std, consensus_confidence

    def get_confidence_breakdown(
        self,
        cross_validation: Optional[CrossValidationResult] = None,
        ab_reconciliation: Optional[ABReconciliationResult] = None,
        estimates: Optional[List[LibraryEffectEstimate]] = None,
    ) -> Dict[str, Any]:
        """Get detailed confidence breakdown for reporting.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result
            estimates: Direct estimates (used if no cross_validation)

        Returns:
            Dictionary with confidence breakdown
        """
        breakdown: Dict[str, Any] = {
            "components": {},
            "final_score": 0.0,
            "tier": "very_low",
            "adjustment_factor": 0.4,
            "sources": [],
        }

        # Compute from available sources
        if ab_reconciliation and cross_validation:
            score, tier = self.score_with_ab_reconciliation(
                cross_validation, ab_reconciliation
            )
            breakdown["sources"] = ["cross_validation", "ab_reconciliation"]

            # Add AB-specific components
            breakdown["components"]["ab_reconciliation_score"] = ab_reconciliation.get(
                "reconciliation_score", 0.0
            )
            breakdown["components"]["ab_direction_match"] = ab_reconciliation.get(
                "direction_match", False
            )
            breakdown["components"]["ab_within_ci"] = ab_reconciliation.get(
                "within_ci", False
            )

        elif cross_validation:
            score, tier = self.score_from_cross_validation(cross_validation)
            breakdown["sources"] = ["cross_validation"]

        elif estimates:
            score, tier = self.score_from_estimates(estimates)
            breakdown["sources"] = ["direct_estimates"]

        else:
            score, tier = 0.0, "very_low"
            breakdown["sources"] = []

        # Add cross-validation components
        if cross_validation:
            summary = cross_validation.get("summary", {})
            breakdown["components"]["agreement_score"] = summary.get("overall_agreement", 0.0)
            breakdown["components"]["consensus_confidence"] = summary.get(
                "consensus_confidence", 0.0
            )
            breakdown["components"]["libraries_validated"] = len(
                summary.get("libraries_validated", [])
            )

        breakdown["final_score"] = score
        breakdown["tier"] = tier
        breakdown["adjustment_factor"] = self.get_adjustment_factor(tier)

        return breakdown


def compute_pipeline_confidence(
    library_results: Dict[str, Dict[str, Any]],
    execution_mode: Literal["sequential", "parallel"],
) -> Dict[str, Any]:
    """Compute confidence for full pipeline execution.

    Utility function for computing confidence across all four libraries
    in the causal pipeline (NetworkX → DoWhy → EconML → CausalML).

    Args:
        library_results: Dict of library name → result dict
        execution_mode: How libraries were executed

    Returns:
        Pipeline confidence metrics
    """
    scorer = ConfidenceScorer()

    # Extract estimates from library results
    estimates: List[LibraryEffectEstimate] = []

    for lib_name, result in library_results.items():
        # Standard estimate extraction
        estimate = result.get("effect_estimate") or result.get("ate") or result.get("uplift")
        if estimate is not None:
            estimates.append(
                LibraryEffectEstimate(
                    library=lib_name,  # type: ignore
                    effect_type="ate",
                    estimate=estimate,
                    standard_error=result.get("standard_error"),
                    ci_lower=result.get("ci_lower"),
                    ci_upper=result.get("ci_upper"),
                    p_value=result.get("p_value"),
                    confidence=result.get("confidence", 0.5),
                    sample_size=result.get("sample_size"),
                    method=result.get("method"),
                    latency_ms=result.get("latency_ms", 0),
                )
            )

    if not estimates:
        return {
            "pipeline_confidence": 0.0,
            "confidence_tier": "very_low",
            "libraries_included": 0,
            "consensus_effect": None,
            "consensus_std": None,
        }

    # Score
    score, tier = scorer.score_from_estimates(estimates)
    consensus, std, _ = scorer.compute_consensus_estimate(estimates)

    return {
        "pipeline_confidence": score,
        "confidence_tier": tier,
        "adjustment_factor": scorer.get_adjustment_factor(tier),
        "libraries_included": len(estimates),
        "consensus_effect": consensus,
        "consensus_std": std,
        "execution_mode": execution_mode,
        "estimates": estimates,
    }
