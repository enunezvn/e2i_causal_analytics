"""Tests for ConfidenceScorer.

B8.4: Confidence scoring based on library agreement tests.
"""

import pytest

from src.causal_engine.validation import (
    ConfidenceScorer,
    CrossValidationResult,
    LibraryEffectEstimate,
    PairwiseValidation,
    ValidationSummary,
    compute_pipeline_confidence,
)


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer."""

    @pytest.fixture
    def scorer(self) -> ConfidenceScorer:
        """Create ConfidenceScorer instance."""
        return ConfidenceScorer()

    @pytest.fixture
    def agreeing_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates with high agreement."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                ci_lower=0.10,
                ci_upper=0.20,
                p_value=0.001,
                confidence=0.85,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.14,
                ci_lower=0.08,
                ci_upper=0.20,
                p_value=0.002,
                confidence=0.82,
                latency_ms=120,
            ),
            LibraryEffectEstimate(
                library="causalml",
                effect_type="uplift",
                estimate=0.16,
                ci_lower=0.09,
                ci_upper=0.23,
                p_value=0.003,
                confidence=0.78,
                latency_ms=150,
            ),
        ]

    @pytest.fixture
    def disagreeing_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates with low agreement."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                confidence=0.80,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="causalml",
                effect_type="uplift",
                estimate=-0.05,  # Opposite direction
                confidence=0.70,
                latency_ms=150,
            ),
        ]

    def test_score_from_estimates_high_agreement(
        self,
        scorer: ConfidenceScorer,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test scoring with high agreement estimates."""
        score, tier = scorer.score_from_estimates(agreeing_estimates)

        assert 0 <= score <= 1
        assert tier in ["high", "moderate", "low", "very_low"]
        # High agreement should result in high or moderate tier
        assert tier in ["high", "moderate"]
        assert score >= 0.6

    def test_score_from_estimates_low_agreement(
        self,
        scorer: ConfidenceScorer,
        disagreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test scoring with low agreement (direction mismatch)."""
        score, tier = scorer.score_from_estimates(disagreeing_estimates)

        assert 0 <= score <= 1
        # Direction disagreement should result in low score
        assert tier in ["low", "very_low"]
        assert score < 0.6

    def test_score_from_estimates_single(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test scoring with single estimate."""
        single = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                confidence=0.80,
                latency_ms=100,
            ),
        ]

        score, tier = scorer.score_from_estimates(single)

        # Single estimate uses its own confidence
        assert score == 0.80  # Uses estimate's confidence
        assert tier in ["high", "moderate"]  # 0.80 is high confidence

    def test_score_from_estimates_empty(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test scoring with empty estimates."""
        score, tier = scorer.score_from_estimates([])

        assert score == 0.0
        assert tier == "very_low"

    def test_adjustment_factor_high(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test adjustment factor for high tier."""
        factor = scorer.get_adjustment_factor("high")
        assert factor == 1.0

    def test_adjustment_factor_moderate(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test adjustment factor for moderate tier."""
        factor = scorer.get_adjustment_factor("moderate")
        assert factor == 0.85

    def test_adjustment_factor_low(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test adjustment factor for low tier."""
        factor = scorer.get_adjustment_factor("low")
        assert factor == 0.65

    def test_adjustment_factor_very_low(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test adjustment factor for very_low tier."""
        factor = scorer.get_adjustment_factor("very_low")
        assert factor == 0.40

    def test_compute_consensus_estimate(
        self,
        scorer: ConfidenceScorer,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test consensus estimate computation."""
        consensus, ci_lower, ci_upper = scorer.compute_consensus_estimate(agreeing_estimates)

        # Consensus should be within range of individual estimates
        effects = [e.get("estimate", 0) for e in agreeing_estimates]
        assert min(effects) <= consensus <= max(effects)

        # CI bounds should be reasonable
        assert ci_lower < consensus < ci_upper

    def test_compute_consensus_estimate_empty(
        self,
        scorer: ConfidenceScorer,
    ) -> None:
        """Test consensus estimate with empty list."""
        consensus, ci_lower, ci_upper = scorer.compute_consensus_estimate([])

        # Empty list returns sentinel values
        assert consensus == 0.0
        assert ci_lower == float("inf")
        assert ci_upper == 0.0

    def test_get_confidence_breakdown(
        self,
        scorer: ConfidenceScorer,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test confidence breakdown with CrossValidationResult."""
        # Create a CrossValidationResult to pass to get_confidence_breakdown
        cross_validation = CrossValidationResult(
            treatment_var="test",
            outcome_var="test",
            validation_type="dowhy_econml",
            estimates=agreeing_estimates,
            pairwise_results=[
                PairwiseValidation(
                    library_a="dowhy",
                    library_b="econml",
                    effect_a=0.15,
                    effect_b=0.14,
                    absolute_difference=0.01,
                    relative_difference=0.067,
                    agreement_score=0.93,
                    direction_agreement=True,
                    significance_agreement=True,
                    ci_overlap=0.8,
                    validation_status="passed",
                    validation_message="Agreement validated.",
                ),
            ],
            summary=ValidationSummary(
                overall_status="passed",
                overall_agreement=0.93,
                pairwise_validations=[],
                libraries_validated=["dowhy", "econml"],
                consensus_effect=0.145,
                consensus_confidence=0.83,
                discrepancies=[],
                recommendations=[],
            ),
            validation_latency_ms=50,
            total_latency_ms=100,
            status="completed",
            errors=[],
            warnings=[],
        )

        breakdown = scorer.get_confidence_breakdown(cross_validation)

        # Should have component scores (nested under 'components')
        assert "components" in breakdown
        assert "agreement_score" in breakdown["components"]
        assert "final_score" in breakdown
        assert "tier" in breakdown
        assert "adjustment_factor" in breakdown

        # Component scores should be 0-1
        assert 0 <= breakdown["components"]["agreement_score"] <= 1
        assert 0 <= breakdown["final_score"] <= 1


class TestConfidenceScorerWithCrossValidation:
    """Test ConfidenceScorer with CrossValidationResult."""

    @pytest.fixture
    def scorer(self) -> ConfidenceScorer:
        """Create ConfidenceScorer instance."""
        return ConfidenceScorer()

    @pytest.fixture
    def cross_validation_passed(self) -> CrossValidationResult:
        """Cross-validation with passed status."""
        return CrossValidationResult(
            treatment_var="treatment",
            outcome_var="outcome",
            validation_type="dowhy_causalml",
            estimates=[],
            pairwise_results=[
                PairwiseValidation(
                    library_a="dowhy",
                    library_b="econml",
                    effect_a=0.15,
                    effect_b=0.14,
                    absolute_difference=0.01,
                    relative_difference=0.067,
                    agreement_score=0.93,
                    direction_agreement=True,
                    significance_agreement=True,
                    ci_overlap=0.8,
                    validation_status="passed",
                    validation_message="Good agreement.",
                ),
            ],
            summary=ValidationSummary(
                overall_status="passed",
                overall_agreement=0.93,
                pairwise_validations=[],
                libraries_validated=["dowhy", "econml"],
                consensus_effect=0.145,
                consensus_confidence=0.85,
                discrepancies=[],
                recommendations=[],
            ),
            validation_latency_ms=50,
            total_latency_ms=100,
            status="completed",
            errors=[],
            warnings=[],
        )

    @pytest.fixture
    def cross_validation_failed(self) -> CrossValidationResult:
        """Cross-validation with failed status."""
        return CrossValidationResult(
            treatment_var="treatment",
            outcome_var="outcome",
            validation_type="dowhy_causalml",
            estimates=[],
            pairwise_results=[
                PairwiseValidation(
                    library_a="dowhy",
                    library_b="causalml",
                    effect_a=0.15,
                    effect_b=-0.10,
                    absolute_difference=0.25,
                    relative_difference=1.67,
                    agreement_score=0.0,
                    direction_agreement=False,
                    significance_agreement=False,
                    ci_overlap=0.0,
                    validation_status="failed",
                    validation_message="Direction mismatch.",
                ),
            ],
            summary=ValidationSummary(
                overall_status="failed",
                overall_agreement=0.0,
                pairwise_validations=[],
                libraries_validated=["dowhy", "causalml"],
                consensus_effect=None,
                consensus_confidence=0.0,
                discrepancies=["Direction disagreement"],
                recommendations=["Investigate cause"],
            ),
            validation_latency_ms=50,
            total_latency_ms=100,
            status="failed",
            errors=[],
            warnings=[],
        )

    def test_score_from_cross_validation_passed(
        self,
        scorer: ConfidenceScorer,
        cross_validation_passed: CrossValidationResult,
    ) -> None:
        """Test scoring from passed cross-validation."""
        score, tier = scorer.score_from_cross_validation(cross_validation_passed)

        assert score >= 0.7
        assert tier in ["high", "moderate"]

    def test_score_from_cross_validation_failed(
        self,
        scorer: ConfidenceScorer,
        cross_validation_failed: CrossValidationResult,
    ) -> None:
        """Test scoring from failed cross-validation."""
        score, tier = scorer.score_from_cross_validation(cross_validation_failed)

        assert score < 0.5
        assert tier in ["low", "very_low"]


class TestPipelineConfidence:
    """Test compute_pipeline_confidence utility."""

    def test_pipeline_confidence_sequential(self) -> None:
        """Test pipeline confidence for sequential execution."""
        library_results = {
            "dowhy": {"status": "success", "confidence": 0.85},
            "econml": {"status": "success", "confidence": 0.80},
            "causalml": {"status": "success", "confidence": 0.75},
        }

        result = compute_pipeline_confidence(library_results, "sequential")

        assert "pipeline_confidence" in result
        assert "confidence_tier" in result
        assert "libraries_included" in result
        assert 0 <= result["pipeline_confidence"] <= 1

    def test_pipeline_confidence_parallel(self) -> None:
        """Test pipeline confidence for parallel execution."""
        library_results = {
            "dowhy": {"status": "success", "confidence": 0.85},
            "econml": {"status": "success", "confidence": 0.80},
        }

        result = compute_pipeline_confidence(library_results, "parallel")

        assert "pipeline_confidence" in result
        assert "confidence_tier" in result

    def test_pipeline_confidence_with_failure(self) -> None:
        """Test pipeline confidence with one library failure."""
        library_results = {
            "dowhy": {"status": "success", "confidence": 0.85},
            "causalml": {"status": "failed", "confidence": 0.0},
        }

        result = compute_pipeline_confidence(library_results, "sequential")

        # Should penalize for failure
        assert result["pipeline_confidence"] < 0.85
        assert result["confidence_tier"] in ["moderate", "low", "very_low"]

    def test_pipeline_confidence_empty(self) -> None:
        """Test pipeline confidence with empty results."""
        result = compute_pipeline_confidence({}, "sequential")

        assert result["pipeline_confidence"] == 0.0
        assert result["confidence_tier"] == "very_low"
