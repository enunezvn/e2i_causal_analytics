"""State definitions for cross-library validation.

B8: Validation Loop (DoWhy ↔ CausalML cross-validation)
Defines TypedDicts for validation results and reports.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class LibraryEffectEstimate(TypedDict, total=False):
    """Effect estimate from a single library."""

    library: Literal["networkx", "dowhy", "econml", "causalml"]
    effect_type: Literal["ate", "cate", "uplift", "graph_weight"]
    estimate: float
    standard_error: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]
    confidence: float  # Library-specific confidence (0-1)
    sample_size: Optional[int]
    method: Optional[str]  # Specific method used (e.g., "CausalForestDML")
    latency_ms: int


class PairwiseValidation(TypedDict, total=False):
    """Pairwise validation result between two libraries."""

    library_a: str
    library_b: str
    effect_a: float
    effect_b: float
    absolute_difference: float
    relative_difference: float  # |a - b| / max(|a|, |b|)
    agreement_score: float  # 1 - relative_difference (0-1)
    direction_agreement: bool  # Both positive or both negative
    significance_agreement: bool  # Both significant or both not
    ci_overlap: Optional[float]  # Overlap of confidence intervals (0-1)
    validation_status: Literal["passed", "warning", "failed"]
    validation_message: str


class RefutationValidation(TypedDict, total=False):
    """Refutation test comparison across libraries."""

    test_name: str
    dowhy_passed: Optional[bool]
    dowhy_new_effect: Optional[float]
    causalml_consistent: Optional[bool]
    causalml_uplift_stable: Optional[float]  # Stability metric
    cross_validation_passed: bool
    discrepancy_reason: Optional[str]


class ValidationSummary(TypedDict, total=False):
    """Overall validation summary."""

    overall_status: Literal["passed", "warning", "failed"]
    overall_agreement: float  # Weighted average agreement (0-1)
    pairwise_validations: List[PairwiseValidation]
    refutation_validations: Optional[List[RefutationValidation]]
    libraries_validated: List[str]
    consensus_effect: Optional[float]  # Confidence-weighted consensus
    consensus_confidence: float  # Confidence in consensus (0-1)
    discrepancies: List[str]  # List of identified discrepancies
    recommendations: List[str]  # Actions based on validation


class CrossValidationResult(TypedDict, total=False):
    """Complete cross-validation result.

    Result of validating DoWhy causal effect estimates against
    CausalML uplift models and vice versa.
    """

    # Input metadata
    treatment_var: str
    outcome_var: str
    validation_type: Literal["dowhy_causalml", "econml_causalml", "full_pipeline"]

    # Library estimates
    estimates: List[LibraryEffectEstimate]

    # Pairwise validations
    pairwise_results: List[PairwiseValidation]

    # Summary
    summary: ValidationSummary

    # Timing
    validation_latency_ms: int
    total_latency_ms: int

    # Status
    status: Literal["completed", "partial", "failed"]
    errors: List[str]
    warnings: List[str]


class ABExperimentResult(TypedDict, total=False):
    """A/B experiment result for reconciliation."""

    experiment_id: str
    treatment_group_size: int
    control_group_size: int
    observed_effect: float
    observed_ci_lower: float
    observed_ci_upper: float
    observed_p_value: float
    is_significant: bool
    experiment_duration_days: int


class ABReconciliationResult(TypedDict, total=False):
    """Result of reconciling causal estimates with A/B experiment.

    Compares observational causal inference (DoWhy/EconML/CausalML)
    with experimental A/B test results.
    """

    # Input
    experiment: ABExperimentResult
    causal_estimates: List[LibraryEffectEstimate]

    # Reconciliation
    observed_vs_estimated_gap: float  # Absolute gap
    observed_vs_estimated_ratio: float  # observed / estimated
    within_ci: bool  # Is observed within causal CI?
    ci_overlap: float  # Overlap of CIs (0-1)

    # Agreement assessment
    direction_match: bool
    magnitude_match: bool  # Within 2x
    significance_match: bool

    # Overall
    reconciliation_status: Literal["excellent", "good", "acceptable", "poor", "failed"]
    reconciliation_score: float  # 0-1
    discrepancy_analysis: str
    recommended_adjustments: List[str]

    # Metadata
    reconciliation_latency_ms: int


class ValidationReportSection(TypedDict, total=False):
    """Section of a validation report."""

    title: str
    status: Literal["passed", "warning", "failed"]
    summary: str
    details: List[str]
    metrics: Dict[str, Any]
    visualizations: Optional[List[Dict[str, Any]]]


class ValidationReport(TypedDict, total=False):
    """Complete validation report with analysis and recommendations.

    Comprehensive report combining cross-validation, A/B reconciliation,
    and confidence scoring.
    """

    # Report metadata
    report_id: str
    generated_at: str  # ISO 8601
    treatment_var: str
    outcome_var: str

    # Sections
    executive_summary: str
    cross_validation_section: ValidationReportSection
    ab_reconciliation_section: Optional[ValidationReportSection]
    confidence_section: ValidationReportSection
    discrepancy_section: Optional[ValidationReportSection]

    # Overall assessment
    overall_status: Literal["passed", "warning", "failed"]
    overall_confidence: float  # 0-1
    key_findings: List[str]
    recommendations: List[str]
    limitations: List[str]

    # Raw results
    cross_validation_result: Optional[CrossValidationResult]
    ab_reconciliation_result: Optional[ABReconciliationResult]

    # Metadata
    generation_latency_ms: int
