"""Tests for CrossValidator - DoWhy ↔ CausalML cross-validation.

B8.1: Cross-library validation tests.
"""

import pytest

from src.causal_engine.validation import (
    CrossValidator,
    LibraryEffectEstimate,
)


class TestCrossValidator:
    """Test suite for CrossValidator."""

    @pytest.fixture
    def validator(self) -> CrossValidator:
        """Create CrossValidator instance."""
        return CrossValidator()

    @pytest.fixture
    def agreeing_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates that agree across libraries."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                standard_error=0.03,
                ci_lower=0.09,
                ci_upper=0.21,
                p_value=0.001,
                confidence=0.85,
                sample_size=1000,
                method="CausalForestDML",
                latency_ms=150,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.14,
                standard_error=0.04,
                ci_lower=0.06,
                ci_upper=0.22,
                p_value=0.002,
                confidence=0.82,
                sample_size=1000,
                method="LinearDML",
                latency_ms=120,
            ),
            LibraryEffectEstimate(
                library="causalml",
                effect_type="uplift",
                estimate=0.16,
                standard_error=0.05,
                ci_lower=0.06,
                ci_upper=0.26,
                p_value=0.003,
                confidence=0.78,
                sample_size=1000,
                method="random_forest",
                latency_ms=200,
            ),
        ]

    @pytest.fixture
    def disagreeing_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates that disagree across libraries."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                standard_error=0.03,
                ci_lower=0.09,
                ci_upper=0.21,
                p_value=0.001,
                confidence=0.85,
                sample_size=1000,
                method="CausalForestDML",
                latency_ms=150,
            ),
            LibraryEffectEstimate(
                library="causalml",
                effect_type="uplift",
                estimate=-0.10,  # Opposite direction
                standard_error=0.05,
                ci_lower=-0.20,
                ci_upper=0.00,
                p_value=0.05,
                confidence=0.60,
                sample_size=1000,
                method="random_forest",
                latency_ms=200,
            ),
        ]

    @pytest.mark.asyncio
    async def test_validate_agreeing_estimates(
        self,
        validator: CrossValidator,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test validation with agreeing estimates."""
        result = await validator.validate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            estimates=agreeing_estimates,
        )

        assert result["status"] == "completed"
        assert result["treatment_var"] == "marketing_spend"
        assert result["outcome_var"] == "conversion_rate"
        assert len(result["pairwise_results"]) == 3  # 3 pairs from 3 estimates

        # Summary should show agreement
        summary = result["summary"]
        assert summary["overall_agreement"] > 0.7
        assert summary["overall_status"] in ["passed", "warning"]
        assert summary["consensus_effect"] is not None
        assert len(summary["libraries_validated"]) == 3

    @pytest.mark.asyncio
    async def test_validate_disagreeing_estimates(
        self,
        validator: CrossValidator,
        disagreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test validation with disagreeing estimates (direction mismatch)."""
        result = await validator.validate(
            treatment_var="price_change",
            outcome_var="sales",
            estimates=disagreeing_estimates,
        )

        # When direction disagrees, overall status becomes "failed"
        assert result["status"] == "failed"
        assert len(result["pairwise_results"]) == 1  # 1 pair from 2 estimates

        # Should detect direction disagreement
        pairwise = result["pairwise_results"][0]
        assert pairwise["direction_agreement"] is False
        assert pairwise["validation_status"] == "failed"

        # Summary should reflect failure
        summary = result["summary"]
        assert summary["overall_status"] == "failed"
        assert any("direction" in d.lower() for d in summary["discrepancies"])

    @pytest.mark.asyncio
    async def test_validate_insufficient_estimates(
        self,
        validator: CrossValidator,
    ) -> None:
        """Test validation with only one estimate."""
        single_estimate = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                confidence=0.85,
                latency_ms=150,
            )
        ]

        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=single_estimate,
        )

        assert result["status"] == "failed"
        assert "Insufficient" in result["errors"][0] or "Need at least" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_pairwise_comparison_metrics(
        self,
        validator: CrossValidator,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test pairwise comparison produces correct metrics."""
        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=agreeing_estimates[:2],  # Just DoWhy and EconML
        )

        assert len(result["pairwise_results"]) == 1
        pairwise = result["pairwise_results"][0]

        # Check all expected fields
        assert "library_a" in pairwise
        assert "library_b" in pairwise
        assert "effect_a" in pairwise
        assert "effect_b" in pairwise
        assert "absolute_difference" in pairwise
        assert "relative_difference" in pairwise
        assert "agreement_score" in pairwise
        assert "direction_agreement" in pairwise
        assert "validation_status" in pairwise
        assert "validation_message" in pairwise

        # Verify calculation correctness
        effect_a = pairwise["effect_a"]
        effect_b = pairwise["effect_b"]
        assert pairwise["absolute_difference"] == pytest.approx(abs(effect_a - effect_b), rel=0.01)

    @pytest.mark.asyncio
    async def test_ci_overlap_calculation(
        self,
        validator: CrossValidator,
    ) -> None:
        """Test CI overlap calculation."""
        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.10,
                ci_lower=0.05,
                ci_upper=0.15,
                confidence=0.8,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.12,
                ci_lower=0.08,
                ci_upper=0.16,
                confidence=0.8,
                latency_ms=100,
            ),
        ]

        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=estimates,
        )

        pairwise = result["pairwise_results"][0]
        assert pairwise["ci_overlap"] is not None
        assert 0 < pairwise["ci_overlap"] < 1  # Partial overlap expected

    @pytest.mark.asyncio
    async def test_consensus_effect_calculation(
        self,
        validator: CrossValidator,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test consensus effect is confidence-weighted."""
        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=agreeing_estimates,
        )

        summary = result["summary"]
        consensus = summary["consensus_effect"]

        # Should be within the range of individual estimates
        effects = [e["estimate"] for e in agreeing_estimates]
        assert min(effects) <= consensus <= max(effects)

    @pytest.mark.asyncio
    async def test_recommendations_generated(
        self,
        validator: CrossValidator,
        agreeing_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test recommendations are generated based on status."""
        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=agreeing_estimates,
        )

        summary = result["summary"]
        assert len(summary["recommendations"]) > 0
        # Recommendations should be actionable
        assert all(isinstance(r, str) and len(r) > 10 for r in summary["recommendations"])


class TestCrossValidatorEdgeCases:
    """Edge case tests for CrossValidator."""

    @pytest.fixture
    def validator(self) -> CrossValidator:
        """Create CrossValidator instance."""
        return CrossValidator()

    @pytest.mark.asyncio
    async def test_zero_effects(self, validator: CrossValidator) -> None:
        """Test handling of zero effect estimates."""
        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.0,
                confidence=0.7,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.001,
                confidence=0.7,
                latency_ms=100,
            ),
        ]

        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=estimates,
        )

        # Should handle without division by zero
        assert result["status"] == "completed"
        pairwise = result["pairwise_results"][0]
        # Both near zero = direction agreement
        assert pairwise["direction_agreement"] is True

    @pytest.mark.asyncio
    async def test_missing_optional_fields(self, validator: CrossValidator) -> None:
        """Test handling of estimates with missing optional fields."""
        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                confidence=0.8,
                latency_ms=100,
                # No CI, no p_value
            ),
            LibraryEffectEstimate(
                library="causalml",
                effect_type="uplift",
                estimate=0.14,
                confidence=0.75,
                latency_ms=150,
                # No CI, no p_value
            ),
        ]

        result = await validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=estimates,
        )

        assert result["status"] == "completed"
        pairwise = result["pairwise_results"][0]
        assert pairwise["ci_overlap"] is None  # No CI data

    @pytest.mark.asyncio
    async def test_custom_thresholds(self) -> None:
        """Test CrossValidator with custom thresholds."""
        strict_validator = CrossValidator(
            agreement_threshold=0.95,  # Very strict
            relative_difference_threshold=0.1,
        )

        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.15,
                confidence=0.8,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.12,  # 20% difference
                confidence=0.8,
                latency_ms=100,
            ),
        ]

        result = await strict_validator.validate(
            treatment_var="treatment",
            outcome_var="outcome",
            estimates=estimates,
        )

        # Strict thresholds should result in warning or failure
        summary = result["summary"]
        assert summary["overall_status"] in ["warning", "failed"]


class TestRefutationComparison:
    """Test refutation comparison functionality."""

    @pytest.fixture
    def validator(self) -> CrossValidator:
        """Create CrossValidator instance."""
        return CrossValidator()

    @pytest.mark.asyncio
    async def test_compare_refutations_all_pass(
        self,
        validator: CrossValidator,
    ) -> None:
        """Test refutation comparison when all tests pass."""
        dowhy_refutations = {
            "placebo_treatment": {"passed": True, "new_effect": 0.01},
            "data_subset": {"passed": True, "new_effect": 0.14},
        }

        causalml_stability = {
            "placebo_stable": True,
            "placebo_uplift_std": 0.02,
            "subset_stable": True,
            "subset_uplift_std": 0.03,
        }

        validations = await validator.compare_refutations(
            dowhy_refutations, causalml_stability
        )

        assert len(validations) == 2
        assert all(v["cross_validation_passed"] for v in validations)
        assert all(v["discrepancy_reason"] is None for v in validations)

    @pytest.mark.asyncio
    async def test_compare_refutations_disagreement(
        self,
        validator: CrossValidator,
    ) -> None:
        """Test refutation comparison with disagreement."""
        dowhy_refutations = {
            "placebo_treatment": {"passed": True, "new_effect": 0.01},
        }

        causalml_stability = {
            "placebo_stable": False,  # CausalML says unstable
            "placebo_uplift_std": 0.15,
        }

        validations = await validator.compare_refutations(
            dowhy_refutations, causalml_stability
        )

        assert len(validations) == 1
        assert validations[0]["cross_validation_passed"] is False
        assert validations[0]["discrepancy_reason"] is not None
