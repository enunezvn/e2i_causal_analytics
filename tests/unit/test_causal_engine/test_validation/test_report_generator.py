"""Tests for ValidationReportGenerator.

B8.3: Validation report generation tests.
"""

import pytest

from src.causal_engine.validation import (
    ABExperimentResult,
    ABReconciliationResult,
    CrossValidationResult,
    LibraryEffectEstimate,
    PairwiseValidation,
    ValidationReportGenerator,
    ValidationSummary,
)


class TestValidationReportGenerator:
    """Test suite for ValidationReportGenerator."""

    @pytest.fixture
    def generator(self) -> ValidationReportGenerator:
        """Create ValidationReportGenerator instance."""
        return ValidationReportGenerator()

    @pytest.fixture
    def cross_validation_result(self) -> CrossValidationResult:
        """Create sample cross-validation result."""
        return CrossValidationResult(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            validation_type="dowhy_causalml",
            estimates=[
                LibraryEffectEstimate(
                    library="dowhy",
                    effect_type="ate",
                    estimate=0.15,
                    confidence=0.85,
                    latency_ms=100,
                ),
                LibraryEffectEstimate(
                    library="econml",
                    effect_type="ate",
                    estimate=0.14,
                    confidence=0.82,
                    latency_ms=120,
                ),
            ],
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
                    validation_message="Libraries agree.",
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
                recommendations=["Proceed with confidence."],
            ),
            validation_latency_ms=50,
            total_latency_ms=220,
            status="completed",
            errors=[],
            warnings=[],
        )

    @pytest.fixture
    def ab_reconciliation_result(self) -> ABReconciliationResult:
        """Create sample A/B reconciliation result."""
        return ABReconciliationResult(
            experiment=ABExperimentResult(
                experiment_id="exp-001",
                treatment_group_size=5000,
                control_group_size=5000,
                observed_effect=0.15,
                observed_ci_lower=0.10,
                observed_ci_upper=0.20,
                observed_p_value=0.001,
                is_significant=True,
                experiment_duration_days=28,
            ),
            causal_estimates=[
                LibraryEffectEstimate(
                    library="dowhy",
                    effect_type="ate",
                    estimate=0.14,
                    confidence=0.85,
                    latency_ms=100,
                ),
            ],
            observed_vs_estimated_gap=0.01,
            observed_vs_estimated_ratio=1.07,
            within_ci=True,
            ci_overlap=0.85,
            direction_match=True,
            magnitude_match=True,
            significance_match=True,
            reconciliation_status="excellent",
            reconciliation_score=0.95,
            discrepancy_analysis="Excellent match.",
            recommended_adjustments=["Proceed with confidence."],
            reconciliation_latency_ms=30,
        )

    @pytest.mark.asyncio
    async def test_generate_with_cross_validation(
        self,
        generator: ValidationReportGenerator,
        cross_validation_result: CrossValidationResult,
    ) -> None:
        """Test report generation with cross-validation only."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            cross_validation_result=cross_validation_result,
        )

        assert report["treatment_var"] == "marketing_spend"
        assert report["outcome_var"] == "conversion_rate"
        assert report["executive_summary"] is not None
        # Check for section presence
        assert "cross_validation_section" in report
        assert report["overall_status"] in ["passed", "warning", "failed", "critical"]
        assert len(report["key_findings"]) > 0

    @pytest.mark.asyncio
    async def test_generate_with_ab_reconciliation(
        self,
        generator: ValidationReportGenerator,
        ab_reconciliation_result: ABReconciliationResult,
    ) -> None:
        """Test report generation with A/B reconciliation only."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            ab_reconciliation_result=ab_reconciliation_result,
        )

        assert report["treatment_var"] == "marketing_spend"
        # Check for AB reconciliation section
        assert "ab_reconciliation_section" in report
        assert (
            "reconciliation" in report["executive_summary"].lower()
            or "experiment" in report["executive_summary"].lower()
        )

    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(
        self,
        generator: ValidationReportGenerator,
        cross_validation_result: CrossValidationResult,
        ab_reconciliation_result: ABReconciliationResult,
    ) -> None:
        """Test report generation with both cross-validation and A/B reconciliation."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            cross_validation_result=cross_validation_result,
            ab_reconciliation_result=ab_reconciliation_result,
        )

        # Should have sections for both
        assert "cross_validation_section" in report
        assert "ab_reconciliation_section" in report

        # Comprehensive report should have more findings
        assert len(report["key_findings"]) >= 2

    @pytest.mark.asyncio
    async def test_generate_without_inputs(
        self,
        generator: ValidationReportGenerator,
    ) -> None:
        """Test report generation with no validation results."""
        report = await generator.generate(
            treatment_var="test",
            outcome_var="test",
        )

        # Should still generate a report with minimal content
        assert report["treatment_var"] == "test"
        assert report["overall_status"] in ["failed", "critical"]  # Either indicates no data
        # Executive summary should indicate lack of validation
        summary_lower = report["executive_summary"].lower()
        assert any(
            phrase in summary_lower
            for phrase in ["no validation", "insufficient", "not validated", "failed"]
        )

    @pytest.mark.asyncio
    async def test_to_markdown(
        self,
        generator: ValidationReportGenerator,
        cross_validation_result: CrossValidationResult,
    ) -> None:
        """Test markdown conversion."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            cross_validation_result=cross_validation_result,
        )

        markdown = generator.to_markdown(report)

        assert "# Validation Report" in markdown
        assert "marketing_spend" in markdown
        assert "conversion_rate" in markdown
        assert "## Executive Summary" in markdown

    @pytest.mark.asyncio
    async def test_report_recommendations(
        self,
        generator: ValidationReportGenerator,
        cross_validation_result: CrossValidationResult,
    ) -> None:
        """Test that recommendations are generated."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            cross_validation_result=cross_validation_result,
        )

        assert "recommendations" in report
        assert len(report["recommendations"]) > 0
        # Recommendations should be actionable strings
        assert all(isinstance(r, str) and len(r) > 10 for r in report["recommendations"])

    @pytest.mark.asyncio
    async def test_report_limitations(
        self,
        generator: ValidationReportGenerator,
        cross_validation_result: CrossValidationResult,
    ) -> None:
        """Test that limitations are documented."""
        report = await generator.generate(
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
            cross_validation_result=cross_validation_result,
        )

        assert "limitations" in report
        # Limitations should exist
        assert isinstance(report["limitations"], list)
