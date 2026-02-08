"""Validation report generator with discrepancy analysis.

B8.3: Generate comprehensive validation reports combining cross-validation
and A/B reconciliation results.

Produces executive summaries, detailed analysis, and actionable recommendations
for causal effect validation.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, cast

from src.causal_engine.validation.state import (
    ABReconciliationResult,
    CrossValidationResult,
    ValidationReport,
    ValidationReportSection,
)

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """Generate comprehensive validation reports.

    Combines:
    - Cross-library validation (DoWhy ↔ EconML ↔ CausalML)
    - A/B experiment reconciliation
    - Confidence scoring
    - Discrepancy analysis
    - Actionable recommendations

    Output formats: Dict (for API), Markdown (for display)
    """

    def __init__(self) -> None:
        """Initialize ValidationReportGenerator."""
        self.report_id_prefix = "VAL"

    async def generate(
        self,
        treatment_var: str,
        outcome_var: str,
        cross_validation_result: Optional[CrossValidationResult] = None,
        ab_reconciliation_result: Optional[ABReconciliationResult] = None,
    ) -> ValidationReport:
        """Generate comprehensive validation report.

        Args:
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            cross_validation_result: Cross-library validation results
            ab_reconciliation_result: A/B reconciliation results (optional)

        Returns:
            ValidationReport with all sections
        """
        start_time = time.time()
        report_id = f"{self.report_id_prefix}-{uuid.uuid4().hex[:8].upper()}"
        generated_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Generate sections
        cross_validation_section = self._generate_cross_validation_section(cross_validation_result)

        ab_reconciliation_section = (
            self._generate_ab_reconciliation_section(ab_reconciliation_result)
            if ab_reconciliation_result
            else None
        )

        confidence_section = self._generate_confidence_section(
            cross_validation_result,
            ab_reconciliation_result,
        )

        discrepancy_section = self._generate_discrepancy_section(
            cross_validation_result,
            ab_reconciliation_result,
        )

        # Overall assessment
        overall_status, overall_confidence = self._compute_overall_assessment(
            cross_validation_result,
            ab_reconciliation_result,
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            treatment_var,
            outcome_var,
            overall_status,
            overall_confidence,
            cross_validation_result,
            ab_reconciliation_result,
        )

        # Key findings and recommendations
        key_findings = self._extract_key_findings(
            cross_validation_result,
            ab_reconciliation_result,
        )

        recommendations = self._generate_recommendations(
            overall_status,
            cross_validation_result,
            ab_reconciliation_result,
        )

        limitations = self._identify_limitations(
            cross_validation_result,
            ab_reconciliation_result,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return ValidationReport(
            report_id=report_id,
            generated_at=generated_at,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            executive_summary=executive_summary,
            cross_validation_section=cross_validation_section,
            ab_reconciliation_section=ab_reconciliation_section,
            confidence_section=confidence_section,
            discrepancy_section=discrepancy_section,
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            key_findings=key_findings,
            recommendations=recommendations,
            limitations=limitations,
            cross_validation_result=cross_validation_result,
            ab_reconciliation_result=ab_reconciliation_result,
            generation_latency_ms=latency_ms,
        )

    def _generate_cross_validation_section(
        self,
        result: Optional[CrossValidationResult],
    ) -> ValidationReportSection:
        """Generate cross-validation section of report.

        Args:
            result: Cross-validation result

        Returns:
            ValidationReportSection for cross-validation
        """
        if not result:
            return ValidationReportSection(
                title="Cross-Library Validation",
                status="failed",
                summary="No cross-validation performed.",
                details=["Cross-validation was not executed or no results available."],
                metrics={},
                visualizations=None,
            )

        summary_data = result.get("summary", {})
        overall_status = summary_data.get("overall_status", "failed")
        overall_agreement = summary_data.get("overall_agreement", 0.0)
        libraries = summary_data.get("libraries_validated", [])
        pairwise_results = result.get("pairwise_results", [])

        # Build details
        details = []
        details.append(f"Libraries validated: {', '.join(libraries)}")
        details.append(f"Overall agreement score: {overall_agreement:.1%}")

        for pw in pairwise_results:
            lib_a = pw.get("library_a", "unknown")
            lib_b = pw.get("library_b", "unknown")
            status = pw.get("validation_status", "unknown")
            agreement = pw.get("agreement_score", 0.0)
            details.append(f"  • {lib_a} ↔ {lib_b}: {status} ({agreement:.1%} agreement)")

        # Metrics
        metrics = {
            "libraries_count": len(libraries),
            "pairwise_comparisons": len(pairwise_results),
            "overall_agreement": overall_agreement,
            "passed_count": sum(
                1 for p in pairwise_results if p.get("validation_status") == "passed"
            ),
            "warning_count": sum(
                1 for p in pairwise_results if p.get("validation_status") == "warning"
            ),
            "failed_count": sum(
                1 for p in pairwise_results if p.get("validation_status") == "failed"
            ),
            "consensus_effect": summary_data.get("consensus_effect"),
            "consensus_confidence": summary_data.get("consensus_confidence", 0.0),
        }

        # Summary text
        summary = (
            f"Cross-validation across {len(libraries)} libraries shows "
            f"{overall_agreement:.1%} overall agreement. "
            f"Status: {overall_status.upper()}."
        )

        return ValidationReportSection(
            title="Cross-Library Validation",
            status=overall_status,
            summary=summary,
            details=details,
            metrics=metrics,
            visualizations=None,
        )

    def _generate_ab_reconciliation_section(
        self,
        result: Optional[ABReconciliationResult],
    ) -> Optional[ValidationReportSection]:
        """Generate A/B reconciliation section of report.

        Args:
            result: A/B reconciliation result

        Returns:
            ValidationReportSection for A/B reconciliation, or None
        """
        if not result:
            return None

        status = result.get("reconciliation_status", "failed")
        score = result.get("reconciliation_score", 0.0)
        analysis = result.get("discrepancy_analysis", "")

        experiment = result.get("experiment", {})
        observed = experiment.get("observed_effect", 0.0)
        estimated_gap = result.get("observed_vs_estimated_gap", 0.0)
        ratio = result.get("observed_vs_estimated_ratio", 1.0)

        # Status mapping
        status_map = {
            "excellent": "passed",
            "good": "passed",
            "acceptable": "warning",
            "poor": "warning",
            "failed": "failed",
        }
        section_status: Literal["passed", "warning", "failed"] = cast(
            Literal["passed", "warning", "failed"], status_map.get(status, "failed")
        )

        # Build details
        details = []
        details.append(f"Observed effect: {observed:.4f}")
        details.append(f"Observed vs estimated gap: {estimated_gap:.4f}")
        details.append(f"Ratio (observed/estimated): {ratio:.2f}x")
        details.append(f"Within CI: {'Yes' if result.get('within_ci', False) else 'No'}")
        details.append(
            f"Direction match: {'Yes' if result.get('direction_match', False) else 'No'}"
        )
        details.append(
            f"Magnitude match: {'Yes' if result.get('magnitude_match', False) else 'No'}"
        )

        # Add recommendations
        adjustments = result.get("recommended_adjustments", [])
        if adjustments:
            details.append("")
            details.append("Recommended adjustments:")
            for adj in adjustments[:3]:  # Top 3
                details.append(f"  • {adj}")

        # Metrics
        metrics = {
            "reconciliation_status": status,
            "reconciliation_score": score,
            "observed_effect": observed,
            "estimated_gap": estimated_gap,
            "ratio": ratio,
            "within_ci": result.get("within_ci", False),
            "ci_overlap": result.get("ci_overlap", 0.0),
            "direction_match": result.get("direction_match", False),
            "magnitude_match": result.get("magnitude_match", False),
            "significance_match": result.get("significance_match", False),
        }

        # Summary
        summary = (
            f"A/B experiment reconciliation shows {status.upper()} agreement "
            f"(score: {score:.1%}). {analysis[:150]}..."
            if len(analysis) > 150
            else f"A/B experiment reconciliation shows {status.upper()} agreement "
            f"(score: {score:.1%}). {analysis}"
        )

        return ValidationReportSection(
            title="A/B Experiment Reconciliation",
            status=section_status,
            summary=summary,
            details=details,
            metrics=metrics,
            visualizations=None,
        )

    def _generate_confidence_section(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> ValidationReportSection:
        """Generate confidence scoring section.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            ValidationReportSection for confidence assessment
        """
        # Compute confidence components
        cv_confidence = 0.0
        ab_confidence = 0.0
        overall_confidence = 0.0

        if cross_validation:
            summary = cross_validation.get("summary", {})
            cv_confidence = summary.get("consensus_confidence", 0.0)

        if ab_reconciliation:
            ab_confidence = ab_reconciliation.get("reconciliation_score", 0.0)

        # Weighted confidence
        if cross_validation and ab_reconciliation:
            # A/B results are more reliable, weight 60/40
            overall_confidence = 0.4 * cv_confidence + 0.6 * ab_confidence
            confidence_source = "Cross-validation (40%) + A/B reconciliation (60%)"
        elif ab_reconciliation:
            overall_confidence = ab_confidence
            confidence_source = "A/B reconciliation only"
        elif cross_validation:
            overall_confidence = cv_confidence
            confidence_source = "Cross-validation only"
        else:
            overall_confidence = 0.0
            confidence_source = "No validation performed"

        # Determine status
        if overall_confidence >= 0.8:
            status: Literal["passed", "warning", "failed"] = "passed"
        elif overall_confidence >= 0.5:
            status = "warning"
        else:
            status = "failed"

        # Details
        details = []
        details.append(f"Confidence source: {confidence_source}")
        details.append(f"Cross-validation confidence: {cv_confidence:.1%}")
        if ab_reconciliation:
            details.append(f"A/B reconciliation confidence: {ab_confidence:.1%}")
        details.append(f"Combined confidence: {overall_confidence:.1%}")
        details.append("")

        # Interpretation
        if overall_confidence >= 0.8:
            details.append("HIGH CONFIDENCE: Effect estimates are well-validated and reliable.")
        elif overall_confidence >= 0.6:
            details.append(
                "MODERATE CONFIDENCE: Effect estimates are reasonably reliable with some uncertainty."
            )
        elif overall_confidence >= 0.4:
            details.append(
                "LOW CONFIDENCE: Effect estimates have significant uncertainty. Use with caution."
            )
        else:
            details.append(
                "VERY LOW CONFIDENCE: Effect estimates are unreliable. Additional validation required."
            )

        metrics = {
            "overall_confidence": overall_confidence,
            "cross_validation_confidence": cv_confidence,
            "ab_reconciliation_confidence": ab_confidence,
            "confidence_tier": (
                "high"
                if overall_confidence >= 0.8
                else "moderate"
                if overall_confidence >= 0.6
                else "low"
                if overall_confidence >= 0.4
                else "very_low"
            ),
        }

        return ValidationReportSection(
            title="Confidence Assessment",
            status=status,
            summary=f"Overall confidence: {overall_confidence:.1%} ({str(metrics['confidence_tier']).replace('_', ' ').title()})",
            details=details,
            metrics=metrics,
            visualizations=None,
        )

    def _generate_discrepancy_section(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> Optional[ValidationReportSection]:
        """Generate discrepancy analysis section.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            ValidationReportSection for discrepancies, or None if none found
        """
        discrepancies: List[str] = []
        severity_counts = {"critical": 0, "warning": 0, "info": 0}

        # Cross-validation discrepancies
        if cross_validation:
            summary = cross_validation.get("summary", {})
            cv_discrepancies = summary.get("discrepancies", [])
            for d in cv_discrepancies:
                if "direction" in d.lower():
                    discrepancies.append(f"CRITICAL: {d}")
                    severity_counts["critical"] += 1
                elif "magnitude" in d.lower():
                    discrepancies.append(f"WARNING: {d}")
                    severity_counts["warning"] += 1
                else:
                    discrepancies.append(f"INFO: {d}")
                    severity_counts["info"] += 1

        # A/B reconciliation discrepancies
        if ab_reconciliation:
            status = ab_reconciliation.get("reconciliation_status", "failed")
            if status in ["failed", "poor"]:
                analysis = ab_reconciliation.get("discrepancy_analysis", "")
                if not ab_reconciliation.get("direction_match", True):
                    discrepancies.append(f"CRITICAL: A/B direction mismatch - {analysis}")
                    severity_counts["critical"] += 1
                else:
                    discrepancies.append(f"WARNING: A/B reconciliation {status} - {analysis}")
                    severity_counts["warning"] += 1

        if not discrepancies:
            return None

        # Determine status
        if severity_counts["critical"] > 0:
            status_val: Literal["passed", "warning", "failed"] = "failed"
        elif severity_counts["warning"] > 0:
            status_val = "warning"
        else:
            status_val = "passed"

        metrics = {
            "total_discrepancies": len(discrepancies),
            "critical_count": severity_counts["critical"],
            "warning_count": severity_counts["warning"],
            "info_count": severity_counts["info"],
        }

        return ValidationReportSection(
            title="Discrepancy Analysis",
            status=status_val,
            summary=f"Found {len(discrepancies)} discrepancies: "
            f"{severity_counts['critical']} critical, "
            f"{severity_counts['warning']} warnings, "
            f"{severity_counts['info']} informational.",
            details=discrepancies,
            metrics=metrics,
            visualizations=None,
        )

    def _compute_overall_assessment(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> tuple[Literal["passed", "warning", "failed"], float]:
        """Compute overall validation assessment.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            Tuple of (overall_status, overall_confidence)
        """
        statuses = []
        confidences = []

        if cross_validation:
            summary = cross_validation.get("summary", {})
            cv_status = summary.get("overall_status", "failed")
            cv_confidence = summary.get("consensus_confidence", 0.0)
            statuses.append(cv_status)
            confidences.append(cv_confidence)

        if ab_reconciliation:
            ab_status = ab_reconciliation.get("reconciliation_status", "failed")
            ab_confidence = ab_reconciliation.get("reconciliation_score", 0.0)
            # Map AB status to passed/warning/failed
            if ab_status in ["excellent", "good"]:
                statuses.append("passed")
            elif ab_status in ["acceptable", "poor"]:
                statuses.append("warning")
            else:
                statuses.append("failed")
            confidences.append(ab_confidence)

        if not statuses:
            return "failed", 0.0

        # Overall status: worst of all
        if "failed" in statuses:
            overall_status: Literal["passed", "warning", "failed"] = "failed"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "passed"

        # Overall confidence: weighted average
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return overall_status, overall_confidence

    def _generate_executive_summary(
        self,
        treatment_var: str,
        outcome_var: str,
        overall_status: str,
        overall_confidence: float,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> str:
        """Generate executive summary for the report.

        Args:
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            overall_status: Overall validation status
            overall_confidence: Overall confidence score
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            Executive summary text
        """
        # Get consensus effect if available
        consensus_effect = None
        if cross_validation:
            summary = cross_validation.get("summary", {})
            consensus_effect = summary.get("consensus_effect")

        # Build summary
        parts = []

        # Opening
        parts.append(
            f"Validation of causal effect from {treatment_var} → {outcome_var} "
            f"resulted in **{overall_status.upper()}** status with "
            f"**{overall_confidence:.0%}** confidence."
        )

        # Effect estimate
        if consensus_effect is not None:
            direction = (
                "positive"
                if consensus_effect > 0
                else "negative"
                if consensus_effect < 0
                else "zero"
            )
            parts.append(
                f"The consensus effect estimate is {consensus_effect:.4f} ({direction} effect)."
            )

        # Cross-validation summary
        if cross_validation:
            summary = cross_validation.get("summary", {})
            libraries = summary.get("libraries_validated", [])
            agreement = summary.get("overall_agreement", 0.0)
            parts.append(
                f"Cross-validation across {len(libraries)} libraries shows "
                f"{agreement:.0%} agreement."
            )

        # A/B summary
        if ab_reconciliation:
            ab_status = ab_reconciliation.get("reconciliation_status", "unknown")
            parts.append(f"A/B experiment reconciliation shows {ab_status} agreement.")

        # Conclusion
        if overall_status == "passed":
            parts.append("Effect estimates are validated and suitable for decision-making.")
        elif overall_status == "warning":
            parts.append("Effect estimates require caution. Review recommendations before use.")
        else:
            parts.append("Effect estimates are not validated. Further investigation required.")

        return " ".join(parts)

    def _extract_key_findings(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> List[str]:
        """Extract key findings from validation results.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            List of key findings (3-5 bullet points)
        """
        findings = []

        if cross_validation:
            summary = cross_validation.get("summary", {})
            agreement = summary.get("overall_agreement", 0.0)
            consensus = summary.get("consensus_effect")
            libraries = summary.get("libraries_validated", [])

            if consensus is not None:
                direction = "positive" if consensus > 0 else "negative"
                findings.append(
                    f"All {len(libraries)} libraries agree on {direction} treatment effect "
                    f"(consensus: {consensus:.4f})"
                )

            if agreement >= 0.8:
                findings.append(
                    f"Strong cross-library agreement ({agreement:.0%}) indicates robust estimation"
                )
            elif agreement >= 0.5:
                findings.append(
                    f"Moderate cross-library agreement ({agreement:.0%}) suggests some estimation uncertainty"
                )
            else:
                findings.append(
                    f"Weak cross-library agreement ({agreement:.0%}) indicates significant estimation uncertainty"
                )

        if ab_reconciliation:
            status = ab_reconciliation.get("reconciliation_status", "unknown")
            within_ci = ab_reconciliation.get("within_ci", False)
            direction_match = ab_reconciliation.get("direction_match", False)

            if status in ["excellent", "good"]:
                findings.append("Observational estimates closely match A/B experimental results")
            elif status == "acceptable":
                findings.append("Observational estimates show moderate alignment with A/B results")

            if within_ci:
                findings.append("Estimated effect falls within experimental confidence interval")

            if not direction_match:
                findings.append(
                    "CRITICAL: Observational and experimental effects show opposite directions"
                )

        if not findings:
            findings.append("Insufficient validation data for key findings")

        return findings[:5]  # Return max 5 findings

    def _generate_recommendations(
        self,
        overall_status: str,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> List[str]:
        """Generate actionable recommendations.

        Args:
            overall_status: Overall validation status
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            List of recommendations
        """
        recommendations = []

        if overall_status == "passed":
            recommendations.append("Proceed with using effect estimates for business decisions")
            recommendations.append("Document validation results for audit trail")
            recommendations.append("Schedule periodic re-validation (quarterly recommended)")

        elif overall_status == "warning":
            recommendations.append("Use effect estimates with appropriate uncertainty bounds")
            recommendations.append("Run additional robustness tests before major decisions")

            if cross_validation:
                summary = cross_validation.get("summary", {})
                cv_recommendations = summary.get("recommendations", [])
                recommendations.extend(cv_recommendations[:2])

            if ab_reconciliation:
                ab_adjustments = ab_reconciliation.get("recommended_adjustments", [])
                recommendations.extend(ab_adjustments[:2])

        else:  # failed
            recommendations.append("DO NOT use current effect estimates for decisions")
            recommendations.append("Investigate sources of discrepancy between libraries")
            recommendations.append("Review causal model assumptions and confounder selection")
            recommendations.append("Consider running controlled A/B experiment if not available")

        return recommendations[:6]  # Return max 6 recommendations

    def _identify_limitations(
        self,
        cross_validation: Optional[CrossValidationResult],
        ab_reconciliation: Optional[ABReconciliationResult],
    ) -> List[str]:
        """Identify limitations of the validation.

        Args:
            cross_validation: Cross-validation result
            ab_reconciliation: A/B reconciliation result

        Returns:
            List of limitations
        """
        limitations = []

        # General limitations
        limitations.append(
            "Cross-library validation assumes all libraries receive identical input data"
        )

        if cross_validation:
            libraries = cross_validation.get("summary", {}).get("libraries_validated", [])
            if len(libraries) < 3:
                limitations.append(
                    f"Limited to {len(libraries)} libraries; more diverse validation recommended"
                )

            errors = cross_validation.get("errors", [])
            if errors:
                limitations.append(f"Some library comparisons failed ({len(errors)} errors)")

        if not ab_reconciliation:
            limitations.append("No A/B experiment available for ground truth validation")
        else:
            experiment = ab_reconciliation.get("experiment", {})
            duration = experiment.get("experiment_duration_days", 0)
            if duration < 14:
                limitations.append(
                    f"A/B experiment duration ({duration} days) may be insufficient for effect stabilization"
                )

        limitations.append(
            "Validation does not guarantee external validity beyond observed population"
        )

        return limitations

    def to_markdown(self, report: ValidationReport) -> str:
        """Convert validation report to Markdown format.

        Args:
            report: ValidationReport to convert

        Returns:
            Markdown formatted string
        """
        md = []

        # Header
        md.append(
            f"# Validation Report: {report.get('treatment_var', '?')} → {report.get('outcome_var', '?')}"
        )
        md.append("")
        md.append(f"**Report ID:** {report.get('report_id', 'N/A')}")
        md.append(f"**Generated:** {report.get('generated_at', 'N/A')}")
        md.append(f"**Status:** {report.get('overall_status', 'unknown').upper()}")
        md.append(f"**Confidence:** {report.get('overall_confidence', 0):.1%}")
        md.append("")

        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        md.append(report.get("executive_summary", "No summary available."))
        md.append("")

        # Key Findings
        md.append("## Key Findings")
        md.append("")
        for finding in report.get("key_findings", []):
            md.append(f"- {finding}")
        md.append("")

        # Sections
        for section_key in [
            "cross_validation_section",
            "ab_reconciliation_section",
            "confidence_section",
            "discrepancy_section",
        ]:
            section_raw = report.get(section_key)
            if section_raw:
                section = cast(Dict[str, Any], section_raw)
                md.append(f"## {section.get('title', 'Section')}")
                md.append("")
                md.append(f"**Status:** {str(section.get('status', 'unknown')).upper()}")
                md.append("")
                md.append(str(section.get("summary", "")))
                md.append("")

                details = section.get("details", [])
                if details:
                    md.append("### Details")
                    md.append("")
                    for detail in details:
                        md.append(f"- {detail}" if not detail.startswith("  ") else detail)
                    md.append("")

        # Recommendations
        md.append("## Recommendations")
        md.append("")
        for rec in report.get("recommendations", []):
            md.append(f"1. {rec}")
        md.append("")

        # Limitations
        md.append("## Limitations")
        md.append("")
        for lim in report.get("limitations", []):
            md.append(f"- {lim}")
        md.append("")

        # Footer
        md.append("---")
        md.append(f"*Generated in {report.get('generation_latency_ms', 0)}ms*")

        return "\n".join(md)
