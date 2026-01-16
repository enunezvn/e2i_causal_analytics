"""
E2I Tool Registration Examples
Version: 4.2
Purpose: Demonstrate how agents expose tools to the Tool Composer

This file shows the pattern for registering composable tools from each agent.
Each agent should call its registration function during initialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.tool_registry import (
    composable_tool,
)

# ============================================================================
# PYDANTIC MODELS FOR TOOL I/O
# ============================================================================


class EffectEstimatorInput(BaseModel):
    """Input for causal effect estimation"""

    treatment: str
    outcome: str
    confounders: List[str] = []
    method: str = "backdoor.linear_regression"


class EffectEstimate(BaseModel):
    """Output from causal effect estimation"""

    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    n_samples: int


class CATEInput(BaseModel):
    """Input for conditional average treatment effect analysis"""

    effect_estimate: EffectEstimate
    segment_variables: List[str]


class CATEResults(BaseModel):
    """Output from CATE analysis"""

    segments: List[Dict[str, Any]]
    high_responders: List[str]
    effect_by_segment: Dict[str, float]


class GapCalculatorInput(BaseModel):
    """Input for gap calculation"""

    metric: str
    entity_type: str  # region, territory, brand
    entities: List[str]


class GapAnalysis(BaseModel):
    """Output from gap analysis"""

    gap: float
    entity_values: Dict[str, float]
    top_performer: str
    bottom_performer: str


class PowerCalculatorInput(BaseModel):
    """Input for power analysis"""

    effect_size: float
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0  # Treatment/control ratio


class PowerAnalysis(BaseModel):
    """Output from power analysis"""

    required_n: int
    actual_power: float
    detectable_effect: float


class SimulatorInput(BaseModel):
    """Input for counterfactual simulation"""

    intervention: str
    target_entities: List[str]
    expected_effect: float
    duration_weeks: int = 12


class SimulationResults(BaseModel):
    """Output from counterfactual simulation"""

    predicted_lift: float
    confidence: str  # low, medium, high
    uncertainty_range: List[float]


# ============================================================================
# COHORT CONSTRUCTOR MODELS (Tier 0)
# ============================================================================


class CohortBuilderInput(BaseModel):
    """Input for cohort construction"""

    brand: str
    indication: Optional[str] = None
    inclusion_criteria: List[str] = []
    exclusion_criteria: List[str] = []
    lookback_days: int = 365
    followup_days: int = 90


class CohortBuilderOutput(BaseModel):
    """Output from cohort construction"""

    eligible_patient_ids: List[str]
    total_evaluated: int
    total_eligible: int
    eligibility_rate: float
    criteria_breakdown: Dict[str, int]
    execution_time_ms: float


class CohortValidatorInput(BaseModel):
    """Input for cohort validation"""

    cohort_result: Dict[str, Any]
    min_cohort_size: int = 100
    required_completeness: float = 0.8


class CohortValidatorOutput(BaseModel):
    """Output from cohort validation"""

    is_valid: bool
    validation_checks: List[Dict[str, Any]]
    quality_score: float
    warnings: List[str]
    recommendations: List[str]


class CohortStatisticsInput(BaseModel):
    """Input for cohort statistics"""

    cohort_result: Dict[str, Any]
    include_demographics: bool = True
    include_clinical: bool = True


class CohortStatisticsOutput(BaseModel):
    """Output from cohort statistics"""

    cohort_size: int
    demographics: Dict[str, Any]
    clinical_characteristics: Dict[str, Any]
    summary_table: List[Dict[str, Any]]


# ============================================================================
# COHORT CONSTRUCTOR AGENT TOOLS (Tier 0)
# ============================================================================


@composable_tool(
    name="cohort_builder",
    description="Constructs patient cohorts by applying inclusion/exclusion criteria based on FDA/EMA label requirements",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {"name": "brand", "type": "str", "description": "Brand name (Remibrutinib, Fabhalta, Kisqali)"},
        {"name": "indication", "type": "str", "description": "Disease indication", "required": False},
        {"name": "inclusion_criteria", "type": "List[str]", "description": "Inclusion criteria expressions", "required": False},
        {"name": "exclusion_criteria", "type": "List[str]", "description": "Exclusion criteria expressions", "required": False},
    ],
    output_schema="CohortBuilderOutput",
    avg_execution_ms=5000,
    input_model=CohortBuilderInput,
    output_model=CohortBuilderOutput,
)
def cohort_builder(
    brand: str,
    indication: Optional[str] = None,
    inclusion_criteria: Optional[List[str]] = None,
    exclusion_criteria: Optional[List[str]] = None,
    **kwargs,
) -> CohortBuilderOutput:
    """
    Build a patient cohort using CohortConstructor agent.

    This is a placeholder implementation. The real implementation
    calls the CohortConstructorAgent.
    """
    # Placeholder - real implementation calls CohortConstructorAgent
    return CohortBuilderOutput(
        eligible_patient_ids=["P001", "P002", "P003"],
        total_evaluated=100,
        total_eligible=3,
        eligibility_rate=0.03,
        criteria_breakdown={
            "age_criteria": 85,
            "diagnosis_criteria": 45,
            "exclusion_applied": 42,
        },
        execution_time_ms=1500.0,
    )


@composable_tool(
    name="cohort_validator",
    description="Validates a constructed cohort against clinical trial requirements",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {"name": "cohort_result", "type": "dict", "description": "Output from cohort_builder"},
        {"name": "min_cohort_size", "type": "int", "description": "Minimum required cohort size", "required": False, "default": 100},
    ],
    output_schema="CohortValidatorOutput",
    avg_execution_ms=1000,
    input_model=CohortValidatorInput,
    output_model=CohortValidatorOutput,
)
def cohort_validator(
    cohort_result: Dict[str, Any],
    min_cohort_size: int = 100,
    required_completeness: float = 0.8,
    **kwargs,
) -> CohortValidatorOutput:
    """Validate a cohort against quality standards."""
    total_eligible = cohort_result.get("total_eligible", 0)
    is_valid = total_eligible >= min_cohort_size

    return CohortValidatorOutput(
        is_valid=is_valid,
        validation_checks=[
            {"check": "minimum_size", "passed": is_valid, "actual": total_eligible, "required": min_cohort_size},
            {"check": "data_completeness", "passed": True, "actual": 0.95, "required": required_completeness},
        ],
        quality_score=0.92 if is_valid else 0.45,
        warnings=[] if is_valid else [f"Cohort size {total_eligible} below minimum {min_cohort_size}"],
        recommendations=["Consider relaxing age criteria to increase cohort size"] if not is_valid else [],
    )


@composable_tool(
    name="cohort_statistics",
    description="Computes descriptive statistics for a patient cohort",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {"name": "cohort_result", "type": "dict", "description": "Output from cohort_builder"},
        {"name": "include_demographics", "type": "bool", "description": "Include demographic stats", "required": False, "default": True},
    ],
    output_schema="CohortStatisticsOutput",
    avg_execution_ms=2000,
    input_model=CohortStatisticsInput,
    output_model=CohortStatisticsOutput,
)
def cohort_statistics(
    cohort_result: Dict[str, Any],
    include_demographics: bool = True,
    include_clinical: bool = True,
    **kwargs,
) -> CohortStatisticsOutput:
    """Compute statistics for a patient cohort."""
    return CohortStatisticsOutput(
        cohort_size=cohort_result.get("total_eligible", 0),
        demographics={
            "age_mean": 52.3,
            "age_std": 14.2,
            "gender_distribution": {"male": 0.48, "female": 0.52},
        } if include_demographics else {},
        clinical_characteristics={
            "disease_severity": {"mild": 0.2, "moderate": 0.5, "severe": 0.3},
            "prior_treatment": {"naive": 0.35, "experienced": 0.65},
        } if include_clinical else {},
        summary_table=[
            {"variable": "Age", "mean": 52.3, "std": 14.2, "min": 18, "max": 85},
            {"variable": "Time to diagnosis (days)", "mean": 180, "std": 90, "min": 30, "max": 730},
        ],
    )


# ============================================================================
# CAUSAL IMPACT AGENT TOOLS
# ============================================================================


@composable_tool(
    name="causal_effect_estimator",
    description="Estimate average treatment effect (ATE/ATT) using DoWhy/EconML with confidence intervals",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable name"},
        {"name": "outcome", "type": "str", "description": "Outcome variable name"},
        {
            "name": "confounders",
            "type": "List[str]",
            "description": "Confounder variables",
            "required": False,
        },
        {"name": "method", "type": "str", "description": "Estimation method", "required": False},
    ],
    output_schema="EffectEstimate",
    avg_execution_ms=2000,
    input_model=EffectEstimatorInput,
    output_model=EffectEstimate,
)
def causal_effect_estimator(
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    method: str = "backdoor.linear_regression",
    **kwargs,
) -> EffectEstimate:
    """
    Estimate causal effect using DoWhy.

    This is a placeholder implementation. The real implementation
    would use DoWhy/EconML for causal inference.
    """
    # Placeholder - real implementation calls DoWhy
    return EffectEstimate(
        ate=0.12, ci_lower=0.08, ci_upper=0.16, p_value=0.001, method=method, n_samples=10000
    )


@composable_tool(
    name="refutation_runner",
    description="Run DoWhy refutation test suite (placebo, random cause, subset, bootstrap, sensitivity)",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {
            "name": "estimate_id",
            "type": "str",
            "description": "ID of the causal estimate to refute",
        },
    ],
    output_schema="RefutationResults",
    avg_execution_ms=5000,
)
def refutation_runner(estimate_id: str, **kwargs) -> Dict[str, Any]:
    """Run refutation tests on a causal estimate."""
    return {
        "placebo_treatment": {"passed": True, "p_value": 0.45},
        "random_common_cause": {"passed": True, "p_value": 0.52},
        "data_subset": {"passed": True, "p_value": 0.03},
        "bootstrap": {"passed": True, "ci_includes_zero": False},
        "sensitivity_e_value": {"passed": True, "e_value": 2.3},
        "overall_passed": True,
        "gate_decision": "proceed",
    }


@composable_tool(
    name="sensitivity_analyzer",
    description="Compute E-values for sensitivity to unobserved confounding",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "ate", "type": "float", "description": "Estimated average treatment effect"},
        {"name": "ci_lower", "type": "float", "description": "Lower confidence bound"},
    ],
    output_schema="SensitivityReport",
    avg_execution_ms=1500,
)
def sensitivity_analyzer(ate: float, ci_lower: float, **kwargs) -> Dict[str, Any]:
    """Compute sensitivity analysis for unobserved confounding."""
    return {
        "e_value_point": 2.3,
        "e_value_ci": 1.8,
        "interpretation": "An unobserved confounder would need to be associated with both treatment and outcome by a factor of 2.3 to explain away the effect.",
        "robustness": "moderate",
    }


# ============================================================================
# HETEROGENEOUS OPTIMIZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="cate_analyzer",
    description="Estimate conditional average treatment effects (CATE) by segment using CausalML",
    source_agent="heterogeneous_optimizer",
    tier=2,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable"},
        {"name": "outcome", "type": "str", "description": "Outcome variable"},
        {"name": "segments", "type": "List[str]", "description": "Segmentation variables"},
    ],
    output_schema="CATEResults",
    avg_execution_ms=3000,
    input_model=CATEInput,
    output_model=CATEResults,
)
def cate_analyzer(treatment: str, outcome: str, segments: List[str], **kwargs) -> CATEResults:
    """Analyze heterogeneous treatment effects by segment."""
    return CATEResults(
        segments=[
            {"name": "high_volume_academic", "cate": 0.28, "n": 1200},
            {"name": "community_practice", "cate": 0.08, "n": 3500},
            {"name": "integrated_health", "cate": 0.15, "n": 2100},
        ],
        high_responders=["high_volume_academic", "integrated_health"],
        effect_by_segment={
            "high_volume_academic": 0.28,
            "community_practice": 0.08,
            "integrated_health": 0.15,
        },
    )


@composable_tool(
    name="segment_ranker",
    description="Rank segments by treatment effect magnitude and ROI potential",
    source_agent="heterogeneous_optimizer",
    tier=2,
    input_parameters=[
        {"name": "cate_results", "type": "dict", "description": "Results from CATE analysis"},
    ],
    output_schema="SegmentRanking",
    avg_execution_ms=1000,
)
def segment_ranker(cate_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Rank segments by effect magnitude."""
    return {
        "ranking": [
            {"rank": 1, "segment": "high_volume_academic", "score": 0.92},
            {"rank": 2, "segment": "integrated_health", "score": 0.71},
            {"rank": 3, "segment": "community_practice", "score": 0.34},
        ],
        "recommended_targets": ["high_volume_academic", "integrated_health"],
    }


# ============================================================================
# GAP ANALYZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="gap_calculator",
    description="Calculate performance gaps between entities (regions, territories, brands)",
    source_agent="gap_analyzer",
    tier=2,
    input_parameters=[
        {"name": "metric", "type": "str", "description": "Metric to compare"},
        {
            "name": "entity_type",
            "type": "str",
            "description": "Type of entity (region, territory, brand)",
        },
        {"name": "entities", "type": "List[str]", "description": "Entities to compare"},
    ],
    output_schema="GapAnalysis",
    avg_execution_ms=1500,
    input_model=GapCalculatorInput,
    output_model=GapAnalysis,
)
def gap_calculator(metric: str, entity_type: str, entities: List[str], **kwargs) -> GapAnalysis:
    """Calculate performance gaps between entities."""
    return GapAnalysis(
        gap=0.23,
        entity_values={"northeast": 0.67, "midwest": 0.44, "south": 0.52, "west": 0.61},
        top_performer="northeast",
        bottom_performer="midwest",
    )


@composable_tool(
    name="roi_estimator",
    description="Estimate ROI of closing identified performance gaps",
    source_agent="gap_analyzer",
    tier=2,
    input_parameters=[
        {"name": "gap_analysis", "type": "dict", "description": "Gap analysis results"},
        {"name": "investment", "type": "float", "description": "Proposed investment amount"},
    ],
    output_schema="ROIEstimate",
    avg_execution_ms=2000,
)
def roi_estimator(gap_analysis: Dict[str, Any], investment: float, **kwargs) -> Dict[str, Any]:
    """Estimate ROI of closing gaps."""
    return {
        "estimated_roi": 3.2,
        "payback_months": 8,
        "confidence_interval": [2.4, 4.1],
        "assumptions": ["Linear relationship between investment and gap closure"],
    }


# ============================================================================
# EXPERIMENT DESIGNER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="power_calculator",
    description="Calculate required sample size for statistical power in A/B tests",
    source_agent="experiment_designer",
    tier=3,
    input_parameters=[
        {"name": "effect_size", "type": "float", "description": "Expected effect size"},
        {
            "name": "alpha",
            "type": "float",
            "description": "Significance level",
            "required": False,
            "default": 0.05,
        },
        {
            "name": "power",
            "type": "float",
            "description": "Desired power",
            "required": False,
            "default": 0.8,
        },
    ],
    output_schema="PowerAnalysis",
    avg_execution_ms=500,
    input_model=PowerCalculatorInput,
    output_model=PowerAnalysis,
)
def power_calculator(
    effect_size: float, alpha: float = 0.05, power: float = 0.8, **kwargs
) -> PowerAnalysis:
    """Calculate sample size for desired power."""
    # Simplified calculation - real implementation uses statsmodels
    n = int(16 * (1.96 + 0.84) ** 2 / (effect_size**2))
    return PowerAnalysis(required_n=n, actual_power=power, detectable_effect=effect_size)


@composable_tool(
    name="counterfactual_simulator",
    description="Simulate intervention outcomes using the causal model",
    source_agent="experiment_designer",
    tier=3,
    input_parameters=[
        {"name": "intervention", "type": "str", "description": "Intervention to simulate"},
        {
            "name": "target_entities",
            "type": "List[str]",
            "description": "Entities to apply intervention to",
        },
        {
            "name": "expected_effect",
            "type": "float",
            "description": "Expected effect from prior analysis",
        },
    ],
    output_schema="SimulationResults",
    avg_execution_ms=3000,
    input_model=SimulatorInput,
    output_model=SimulationResults,
)
def counterfactual_simulator(
    intervention: str, target_entities: List[str], expected_effect: float, **kwargs
) -> SimulationResults:
    """Simulate intervention outcomes."""
    return SimulationResults(
        predicted_lift=expected_effect * 0.85,  # Adjusted for real-world factors
        confidence="medium",
        uncertainty_range=[expected_effect * 0.6, expected_effect * 1.1],
    )


# ============================================================================
# DRIFT MONITOR AGENT TOOLS
# ============================================================================


@composable_tool(
    name="psi_calculator",
    description="Calculate Population Stability Index for drift detection",
    source_agent="drift_monitor",
    tier=3,
    input_parameters=[
        {"name": "feature", "type": "str", "description": "Feature to analyze"},
        {"name": "baseline_period", "type": "str", "description": "Baseline time period"},
        {"name": "current_period", "type": "str", "description": "Current time period"},
    ],
    output_schema="DriftMetrics",
    avg_execution_ms=800,
)
def psi_calculator(
    feature: str, baseline_period: str, current_period: str, **kwargs
) -> Dict[str, Any]:
    """Calculate PSI for drift detection."""
    return {
        "psi": 0.08,
        "interpretation": "No significant drift",
        "threshold": 0.1,
        "buckets": [
            {"range": "0-0.1", "baseline_pct": 0.15, "current_pct": 0.14},
            {"range": "0.1-0.2", "baseline_pct": 0.25, "current_pct": 0.27},
        ],
    }


@composable_tool(
    name="distribution_comparator",
    description="Compare feature distributions between time periods",
    source_agent="drift_monitor",
    tier=3,
    input_parameters=[
        {"name": "features", "type": "List[str]", "description": "Features to compare"},
        {"name": "period_1", "type": "str", "description": "First time period"},
        {"name": "period_2", "type": "str", "description": "Second time period"},
    ],
    output_schema="DistributionComparison",
    avg_execution_ms=1200,
)
def distribution_comparator(
    features: List[str], period_1: str, period_2: str, **kwargs
) -> Dict[str, Any]:
    """Compare distributions across time periods."""
    return {
        "comparisons": [
            {"feature": f, "ks_statistic": 0.05, "p_value": 0.34, "drift_detected": False}
            for f in features
        ],
        "overall_drift": False,
    }


# ============================================================================
# PREDICTION SYNTHESIZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="risk_scorer",
    description="Score entities by risk/propensity using ensemble ML models",
    source_agent="prediction_synthesizer",
    tier=4,
    input_parameters=[
        {"name": "entity_type", "type": "str", "description": "Type of entity to score"},
        {
            "name": "risk_type",
            "type": "str",
            "description": "Type of risk (churn, discontinuation, etc.)",
        },
        {
            "name": "entity_ids",
            "type": "List[str]",
            "description": "Entity IDs to score",
            "required": False,
        },
    ],
    output_schema="RiskScores",
    avg_execution_ms=1500,
)
def risk_scorer(
    entity_type: str, risk_type: str, entity_ids: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """Score entities by risk."""
    return {
        "scores": [
            {"entity_id": "E001", "risk_score": 0.82, "risk_tier": "high"},
            {"entity_id": "E002", "risk_score": 0.45, "risk_tier": "medium"},
            {"entity_id": "E003", "risk_score": 0.12, "risk_tier": "low"},
        ],
        "model_version": "v2.3.1",
        "scored_at": "2024-01-15T10:30:00Z",
    }


@composable_tool(
    name="propensity_estimator",
    description="Estimate propensity scores for treatment assignment analysis",
    source_agent="prediction_synthesizer",
    tier=4,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable"},
        {"name": "covariates", "type": "List[str]", "description": "Covariate variables"},
    ],
    output_schema="PropensityScores",
    avg_execution_ms=2000,
)
def propensity_estimator(treatment: str, covariates: List[str], **kwargs) -> Dict[str, Any]:
    """Estimate propensity scores."""
    return {
        "mean_propensity": 0.35,
        "propensity_distribution": {
            "min": 0.05,
            "q25": 0.22,
            "median": 0.34,
            "q75": 0.48,
            "max": 0.92,
        },
        "overlap_assessment": "good",
        "common_support": 0.94,
    }


# ============================================================================
# REGISTRATION HELPER
# ============================================================================


def register_all_tools():
    """
    Register all composable tools.

    Call this function during application startup to ensure
    all tools are available to the Tool Composer.
    """
    # Tools are auto-registered via the @composable_tool decorator
    # This function just ensures the module is imported
    pass


# For testing: list all registered tools
if __name__ == "__main__":
    from src.tool_registry import get_registry

    registry = get_registry()
    print(f"Registered {registry.tool_count} tools from {registry.agent_count} agents:")

    for tool_name in registry.list_tools():
        schema = registry.get_schema(tool_name)
        print(f"  - {tool_name} ({schema.source_agent}, Tier {schema.tier})")
