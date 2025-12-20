# Tier 2 Contracts: Causal Inference Agents

**Version**: 1.0
**Last Updated**: 2025-12-18
**Status**: Active

## Overview

This document defines integration contracts for **Tier 2: Causal Inference** agents in the E2I Causal Analytics platform. These agents perform causal analysis, opportunity detection, and treatment effect heterogeneity analysis.

### Tier 2 Agents

| Agent | Type | Responsibility | Primary Methods |
|-------|------|----------------|-----------------|
| **Causal Impact** | Hybrid | Causal effect estimation with narrative | DoWhy, EconML, Claude Sonnet/Opus |
| **Gap Analyzer** | Standard | ROI opportunity detection | Computational, benchmarking |
| **Heterogeneous Optimizer** | Standard | Segment-level CATE analysis | EconML CausalForestDML |

---

## 1. Shared Types

### 1.1 Common Enums

```python
from typing import Literal

# User expertise levels (affects interpretation depth)
ExpertiseLevel = Literal["executive", "analyst", "data_scientist"]

# Interpretation depth modes
InterpretationDepth = Literal["none", "minimal", "standard", "deep"]

# Gap comparison types
GapType = Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]

# Responder classification
ResponderType = Literal["high", "low", "average"]

# Implementation difficulty
DifficultyLevel = Literal["low", "medium", "high"]
```

### 1.2 Common Input Fields

All Tier 2 agents accept these common fields (in addition to agent-specific fields):

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Tier2CommonInput(BaseModel):
    """Common input fields for all Tier 2 agents"""
    query: str = Field(..., description="User's natural language query")
    data_source: str = Field(..., description="Table or query reference for data")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filtering criteria")
    brand: Optional[str] = Field(None, description="Brand context for analysis")
    time_period: Optional[str] = Field("current_quarter", description="Time period for analysis")
```

### 1.3 Common Output Fields

All Tier 2 agents return these common fields:

```python
class Tier2CommonOutput(BaseModel):
    """Common output fields for all Tier 2 agents"""
    executive_summary: str = Field(..., description="High-level summary for executives")
    key_insights: List[str] = Field(..., description="Bullet-point insights")
    total_latency_ms: int = Field(..., description="Total processing time in milliseconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in results")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    requires_further_analysis: bool = Field(..., description="Whether follow-up analysis is recommended")
    suggested_next_agent: Optional[str] = Field(None, description="Recommended next agent to invoke")
```

---

## 2. Causal Impact Agent

**Agent Type**: Hybrid (Computation + Deep Reasoning)
**Primary Models**: DoWhy, EconML, Claude Sonnet 4, Claude Opus 4.5 (fallback)
**Latency**: Up to 120s (60s computation, 30s interpretation)

### 2.1 Input Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CausalImpactInput(BaseModel):
    """Input contract for Causal Impact Agent"""

    # Required fields
    query: str
    treatment_var: str = Field(..., description="Treatment variable name")
    outcome_var: str = Field(..., description="Outcome variable name")
    confounders: List[str] = Field(..., description="List of confounder variables to control for")
    data_source: str = Field(..., description="Data source identifier")

    # Optional fields
    filters: Optional[Dict[str, Any]] = None
    mediators: Optional[List[str]] = Field(None, description="Mediator variables (causal path analysis)")
    effect_modifiers: Optional[List[str]] = Field(None, description="Effect modifier variables")
    instruments: Optional[List[str]] = Field(None, description="Instrumental variables (if using IV estimation)")

    # Configuration
    interpretation_depth: InterpretationDepth = "standard"
    user_expertise: ExpertiseLevel = "analyst"
    estimation_method: str = Field(
        "backdoor.econml.dml.CausalForestDML",
        description="DoWhy estimation method"
    )
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the causal effect of HCP engagement frequency on TRx?",
                "treatment_var": "hcp_engagement_frequency",
                "outcome_var": "trx_total",
                "confounders": ["hcp_specialty", "territory", "patient_volume"],
                "data_source": "hcp_performance_metrics",
                "interpretation_depth": "standard",
                "user_expertise": "analyst"
            }
        }
    }
```

### 2.2 Output Contract

```python
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field

class CausalImpactOutput(BaseModel):
    """Output contract for Causal Impact Agent"""

    # Core results
    ate_estimate: float = Field(..., description="Average Treatment Effect")
    confidence_interval: Tuple[float, float] = Field(..., description="95% confidence interval")
    p_value: Optional[float] = Field(None, description="Statistical significance p-value")

    # Interpretation outputs
    causal_narrative: str = Field(..., description="Natural language causal explanation")
    assumption_warnings: List[str] = Field(
        ...,
        description="Causal assumption violations or risks"
    )
    actionable_recommendations: List[str] = Field(
        ...,
        description="Recommended actions based on findings"
    )
    executive_summary: Optional[str] = Field(
        None,
        description="Executive-level summary (2-3 sentences)"
    )

    # Robustness evidence
    refutation_passed: bool = Field(..., description="Whether robustness checks passed")
    sensitivity_e_value: Optional[float] = Field(
        None,
        description="E-value for unmeasured confounding sensitivity"
    )

    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    computation_latency_ms: int
    interpretation_latency_ms: int
    total_latency_ms: int
    model_used: str = Field(..., description="Model used for interpretation")

    # Common fields
    key_insights: List[str]
    warnings: List[str] = Field(default_factory=list)
    requires_further_analysis: bool
    suggested_next_agent: Optional[str] = Field(
        None,
        description="heterogeneous_optimizer|experiment_designer|gap_analyzer"
    )

    # Optional advanced outputs
    cate_by_segment: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Conditional effects by segment (if heterogeneity detected)"
    )
```

### 2.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class CausalGraphSpec(TypedDict):
    """Causal DAG specification"""
    treatment: str
    outcome: str
    confounders: List[str]
    mediators: Optional[List[str]]
    effect_modifiers: Optional[List[str]]
    instruments: Optional[List[str]]
    edges: List[Tuple[str, str]]
    dot_string: str
    description: str

class RefutationResults(TypedDict):
    """DoWhy refutation test results"""
    placebo_treatment: Dict[str, Any]
    random_common_cause: Dict[str, Any]
    data_subset: Dict[str, Any]
    unobserved_common_cause: Optional[Dict[str, Any]]

class SensitivityAnalysis(TypedDict):
    """Sensitivity to unmeasured confounding"""
    e_value: Optional[float]
    robustness_value: Optional[float]
    confounder_strength_bounds: Optional[Dict[str, float]]
    interpretation: str

class CausalImpactState(TypedDict):
    """Complete LangGraph state for Causal Impact Agent"""

    # === INPUT ===
    query: str
    treatment_var: str
    outcome_var: str
    confounders: List[str]
    data_source: str
    filters: Optional[Dict[str, Any]]

    # === CONFIGURATION ===
    interpretation_depth: Literal["none", "minimal", "standard", "deep"]
    user_expertise: Literal["executive", "analyst", "data_scientist"]
    estimation_method: str
    confidence_level: float

    # === COMPUTATION OUTPUTS ===
    causal_graph: Optional[CausalGraphSpec]
    ate_estimate: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float]
    refutation_results: Optional[RefutationResults]
    sensitivity_analysis: Optional[SensitivityAnalysis]
    cate_by_segment: Optional[Dict[str, Dict[str, float]]]

    # === INTERPRETATION OUTPUTS ===
    causal_narrative: Optional[str]
    assumption_warnings: Optional[List[str]]
    actionable_recommendations: Optional[List[str]]
    executive_summary: Optional[str]

    # === EXECUTION METADATA ===
    computation_latency_ms: int
    interpretation_latency_ms: int
    model_used: str
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    retry_count: int
    status: Literal["pending", "computing", "interpreting", "completed", "failed"]
```

### 2.4 Handoff Format

```yaml
causal_impact_handoff:
  agent: causal_impact
  analysis_type: causal_effect_estimation
  status: completed
  confidence: 0.85

  key_findings:
    ate: 0.23
    confidence_interval: [0.18, 0.28]
    p_value: 0.001
    significant: true
    direction: positive

  robustness:
    placebo_passed: true
    subset_stable: true
    random_cause_passed: true
    sensitivity_e_value: 2.1

  narrative: |
    Increasing HCP engagement frequency causes a 23% lift in prescription rates
    (95% CI: 18-28%, p < 0.001). This effect remains robust across multiple
    sensitivity checks.

  recommendations:
    - action: "Increase HCP visit frequency in high-potential territories"
      expected_impact: "+23% TRx"
      confidence: 0.85
    - action: "Investigate heterogeneous effects across HCP segments"
      expected_impact: "Potential for targeted optimization"
      confidence: 0.70

  warnings:
    - "Assumes no unmeasured confounding stronger than observed covariates"
    - "Effect may vary by HCP specialty (consider heterogeneous_optimizer)"

  requires_further_analysis: true
  suggested_next_agent: heterogeneous_optimizer
  suggested_reason: "Detected potential heterogeneity in treatment effects across segments"
```

---

## 3. Gap Analyzer Agent

**Agent Type**: Standard (Computational)
**Primary Methods**: Computational, benchmarking
**Latency**: Up to 20s

### 3.1 Input Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class GapAnalyzerInput(BaseModel):
    """Input contract for Gap Analyzer Agent"""

    # Required fields
    query: str
    metrics: List[str] = Field(..., description="KPIs to analyze for gaps")
    segments: List[str] = Field(..., description="Segmentation dimensions (region, specialty, etc.)")
    brand: str = Field(..., description="Brand identifier")

    # Optional fields
    time_period: str = Field("current_quarter", description="Analysis time period")
    filters: Optional[Dict[str, Any]] = None

    # Configuration
    gap_type: GapType = Field("vs_potential", description="Type of gap comparison")
    min_gap_threshold: float = Field(5.0, ge=0, le=100, description="Minimum gap % to report")
    max_opportunities: int = Field(10, ge=1, le=50, description="Maximum opportunities to return")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Where are the biggest ROI opportunities for Kisqali?",
                "metrics": ["trx", "market_share", "conversion_rate"],
                "segments": ["region", "specialty", "decile"],
                "brand": "kisqali",
                "gap_type": "vs_potential",
                "min_gap_threshold": 5.0,
                "max_opportunities": 10
            }
        }
    }
```

### 3.2 Output Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PerformanceGap(BaseModel):
    """Individual performance gap"""
    gap_id: str
    metric: str
    segment: str
    segment_value: str
    current_value: float
    target_value: float
    gap_size: float
    gap_percentage: float
    gap_type: GapType

class ROIEstimate(BaseModel):
    """ROI estimate for closing a gap"""
    gap_id: str
    estimated_revenue_impact: float
    estimated_cost_to_close: float
    expected_roi: float
    payback_period_months: int = Field(..., ge=1, le=24)
    confidence: float = Field(..., ge=0.0, le=1.0)
    assumptions: List[str]

class PrioritizedOpportunity(BaseModel):
    """Prioritized gap with action recommendation"""
    rank: int
    gap: PerformanceGap
    roi_estimate: ROIEstimate
    recommended_action: str
    implementation_difficulty: DifficultyLevel
    time_to_impact: str

class GapAnalyzerOutput(BaseModel):
    """Output contract for Gap Analyzer Agent"""

    # Core results
    prioritized_opportunities: List[PrioritizedOpportunity] = Field(
        ...,
        description="All opportunities ranked by ROI"
    )
    quick_wins: List[PrioritizedOpportunity] = Field(
        ...,
        description="Low difficulty, high ROI opportunities"
    )
    strategic_bets: List[PrioritizedOpportunity] = Field(
        ...,
        description="High impact, high difficulty opportunities"
    )

    # Aggregates
    total_addressable_value: float = Field(..., description="Total potential revenue impact")
    total_gap_value: float = Field(..., description="Sum of all gap sizes")
    segments_analyzed: int = Field(..., description="Number of segments analyzed")

    # Summaries
    executive_summary: str
    key_insights: List[str]

    # Metadata
    detection_latency_ms: int
    roi_latency_ms: int
    total_latency_ms: int

    # Common fields
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    requires_further_analysis: bool
    suggested_next_agent: Optional[str] = Field(
        None,
        description="resource_optimizer|experiment_designer|causal_impact"
    )
```

### 3.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class GapAnalyzerState(TypedDict):
    """Complete LangGraph state for Gap Analyzer Agent"""

    # === INPUT ===
    query: str
    metrics: List[str]
    segments: List[str]
    brand: str
    time_period: str
    filters: Optional[Dict[str, Any]]

    # === CONFIGURATION ===
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]
    min_gap_threshold: float
    max_opportunities: int

    # === DETECTION OUTPUTS ===
    gaps_detected: Optional[List[Dict[str, Any]]]  # List of PerformanceGap
    gaps_by_segment: Optional[Dict[str, List[Dict[str, Any]]]]
    total_gap_value: Optional[float]

    # === ROI OUTPUTS ===
    roi_estimates: Optional[List[Dict[str, Any]]]  # List of ROIEstimate
    total_addressable_value: Optional[float]

    # === PRIORITIZATION OUTPUTS ===
    prioritized_opportunities: Optional[List[Dict[str, Any]]]  # List of PrioritizedOpportunity
    quick_wins: Optional[List[Dict[str, Any]]]
    strategic_bets: Optional[List[Dict[str, Any]]]

    # === SUMMARY ===
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]

    # === EXECUTION METADATA ===
    detection_latency_ms: int
    roi_latency_ms: int
    total_latency_ms: int
    segments_analyzed: int

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "detecting", "calculating", "prioritizing", "completed", "failed"]
```

### 3.4 Handoff Format

```yaml
gap_analyzer_handoff:
  agent: gap_analyzer
  analysis_type: roi_opportunity_detection
  status: completed
  confidence: 0.80

  key_findings:
    total_gaps: 47
    total_addressable_value: 12500000
    segments_analyzed: 15
    top_opportunity:
      segment: "specialty=Oncology"
      metric: "market_share"
      gap_percentage: 18.5
      expected_roi: 3.2

  quick_wins:
    - action: "Increase HCP contact frequency in region=Northeast"
      expected_roi: 2.8
      payback_months: 2
      time_to_impact: "1-3 months"
      confidence: 0.85
    - action: "Deploy targeted messaging for specialty=Rheumatology"
      expected_roi: 2.5
      payback_months: 3
      time_to_impact: "1-3 months"
      confidence: 0.80

  strategic_bets:
    - action: "Competitive displacement campaign in region=West"
      expected_impact: 4500000
      payback_months: 8
      implementation_difficulty: high
      time_to_impact: "6-12 months"
      confidence: 0.70

  recommendations:
    - "Prioritize 5 quick wins with combined ROI of 2.6x"
    - "Consider causal_impact analysis to validate intervention effects"
    - "Design A/B test with experiment_designer for top opportunity"

  warnings:
    - "ROI estimates assume 5% conversion rate improvement"
    - "Benchmark data from Q3 2024 may not reflect current market"

  requires_further_analysis: true
  suggested_next_agent: experiment_designer
  suggested_reason: "Top opportunity should be validated with controlled experiment"
```

---

## 4. Heterogeneous Optimizer Agent

**Agent Type**: Standard (Computational)
**Primary Methods**: EconML CausalForestDML
**Latency**: Up to 150s

### 4.1 Input Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HeterogeneousOptimizerInput(BaseModel):
    """Input contract for Heterogeneous Optimizer Agent"""

    # Required fields
    query: str
    treatment_var: str = Field(..., description="Treatment variable name")
    outcome_var: str = Field(..., description="Outcome variable name")
    segment_vars: List[str] = Field(..., description="Variables to segment by")
    effect_modifiers: List[str] = Field(..., description="Variables that modify treatment effect")
    data_source: str = Field(..., description="Data source identifier")

    # Optional fields
    filters: Optional[Dict[str, Any]] = None

    # Configuration
    n_estimators: int = Field(100, ge=50, le=500, description="Number of trees in Causal Forest")
    min_samples_leaf: int = Field(10, ge=5, le=100, description="Minimum samples per leaf")
    significance_level: float = Field(0.05, ge=0.01, le=0.10, description="Significance level for CI")
    top_segments_count: int = Field(10, ge=5, le=50, description="Number of top segments to return")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Which HCP segments respond best to increased engagement?",
                "treatment_var": "hcp_engagement_frequency",
                "outcome_var": "trx_total",
                "segment_vars": ["hcp_specialty", "patient_volume_decile", "region"],
                "effect_modifiers": ["hcp_tenure", "competitive_pressure", "formulary_status"],
                "data_source": "hcp_performance_metrics",
                "n_estimators": 100,
                "top_segments_count": 10
            }
        }
    }
```

### 4.2 Output Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CATEResult(BaseModel):
    """CATE estimation result for a segment"""
    segment_name: str
    segment_value: str
    cate_estimate: float
    cate_ci_lower: float
    cate_ci_upper: float
    sample_size: int
    statistical_significance: bool

class SegmentProfile(BaseModel):
    """Profile of a high/low responder segment"""
    segment_id: str
    responder_type: ResponderType
    cate_estimate: float
    defining_features: List[Dict[str, Any]]
    size: int
    size_percentage: float
    recommendation: str

class PolicyRecommendation(BaseModel):
    """Treatment allocation recommendation"""
    segment: str
    current_treatment_rate: float = Field(..., ge=0.0, le=1.0)
    recommended_treatment_rate: float = Field(..., ge=0.0, le=1.0)
    expected_incremental_outcome: float
    confidence: float = Field(..., ge=0.0, le=1.0)

class HeterogeneousOptimizerOutput(BaseModel):
    """Output contract for Heterogeneous Optimizer Agent"""

    # Core results
    overall_ate: float = Field(..., description="Overall Average Treatment Effect")
    heterogeneity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score, higher = more heterogeneity"
    )

    # Segment analysis
    high_responders: List[SegmentProfile] = Field(
        ...,
        description="Top high-responder segments"
    )
    low_responders: List[SegmentProfile] = Field(
        ...,
        description="Top low-responder segments"
    )
    cate_by_segment: Dict[str, List[CATEResult]] = Field(
        ...,
        description="CATE results by segment variable"
    )

    # Policy recommendations
    policy_recommendations: List[PolicyRecommendation] = Field(
        ...,
        description="Optimal treatment allocation recommendations"
    )
    expected_total_lift: float = Field(
        ...,
        description="Expected outcome lift from optimal allocation"
    )
    optimal_allocation_summary: str = Field(
        ...,
        description="Natural language summary of optimal policy"
    )

    # Feature importance
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Importance of each effect modifier"
    )

    # Summaries
    executive_summary: str
    key_insights: List[str]

    # Metadata
    estimation_latency_ms: int
    analysis_latency_ms: int
    total_latency_ms: int

    # Common fields
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    requires_further_analysis: bool
    suggested_next_agent: Optional[str] = Field(
        None,
        description="experiment_designer|resource_optimizer"
    )
```

### 4.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class HeterogeneousOptimizerState(TypedDict):
    """Complete LangGraph state for Heterogeneous Optimizer Agent"""

    # === INPUT ===
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]
    effect_modifiers: List[str]
    data_source: str
    filters: Optional[Dict[str, Any]]

    # === CONFIGURATION ===
    n_estimators: int
    min_samples_leaf: int
    significance_level: float
    top_segments_count: int

    # === CATE OUTPUTS ===
    cate_by_segment: Optional[Dict[str, List[Dict[str, Any]]]]  # Dict of CATEResult lists
    overall_ate: Optional[float]
    heterogeneity_score: Optional[float]
    feature_importance: Optional[Dict[str, float]]

    # === SEGMENT DISCOVERY OUTPUTS ===
    high_responders: Optional[List[Dict[str, Any]]]  # List of SegmentProfile
    low_responders: Optional[List[Dict[str, Any]]]
    segment_comparison: Optional[Dict[str, Any]]

    # === POLICY OUTPUTS ===
    policy_recommendations: Optional[List[Dict[str, Any]]]  # List of PolicyRecommendation
    expected_total_lift: Optional[float]
    optimal_allocation_summary: Optional[str]

    # === VISUALIZATION DATA ===
    cate_plot_data: Optional[Dict[str, Any]]
    segment_grid_data: Optional[Dict[str, Any]]

    # === SUMMARY ===
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]

    # === EXECUTION METADATA ===
    estimation_latency_ms: int
    analysis_latency_ms: int
    total_latency_ms: int

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "estimating", "analyzing", "optimizing", "completed", "failed"]
```

### 4.4 Handoff Format

```yaml
heterogeneous_optimizer_handoff:
  agent: heterogeneous_optimizer
  analysis_type: cate_estimation
  status: completed
  confidence: 0.82

  key_findings:
    overall_ate: 0.23
    heterogeneity_score: 0.68
    high_responder_count: 8
    low_responder_count: 7
    top_high_responder: "specialty=Oncology, decile=9-10"
    top_low_responder: "specialty=Primary Care, decile=1-2"

  segment_profiles:
    high_responders:
      - segment: "specialty=Oncology"
        cate: 0.42
        sample_size: 1250
        significance: true
        recommendation: "Prioritize treatment (CATE: 1.8x ATE)"
      - segment: "decile=9-10"
        cate: 0.38
        sample_size: 980
        significance: true
        recommendation: "Increase engagement frequency"

    low_responders:
      - segment: "specialty=Primary Care"
        cate: 0.08
        sample_size: 3400
        significance: true
        recommendation: "De-prioritize, consider alternative interventions"
      - segment: "decile=1-2"
        cate: 0.11
        sample_size: 2100
        significance: false
        recommendation: "Minimal treatment allocation"

  policy_recommendations:
    - segment: "specialty=Oncology"
      action: increase
      from_rate: 0.50
      to_rate: 0.80
      expected_lift: 156.0
      confidence: 0.85
    - segment: "specialty=Primary Care"
      action: decrease
      from_rate: 0.50
      to_rate: 0.30
      expected_lift: -20.0
      confidence: 0.78

  feature_importance:
    hcp_tenure: 0.32
    competitive_pressure: 0.28
    formulary_status: 0.21
    patient_volume: 0.19

  recommendations:
    - "Reallocate resources from low-responder to high-responder segments"
    - "Expected total lift: +487 units from optimal allocation"
    - "Design experiment to validate segment-specific interventions"

  warnings:
    - "Low sample size for some segments (n<50) excluded"
    - "Effect heterogeneity may reflect unmeasured segment characteristics"

  requires_further_analysis: true
  suggested_next_agent: experiment_designer
  suggested_reason: "Validate segment-specific treatment strategies with controlled experiments"
```

---

## 5. Inter-Agent Communication

### 5.1 Orchestrator → Tier 2 Dispatch

All Tier 2 agents receive dispatches via the standard `AgentDispatchRequest` (see `orchestrator-contracts.md`):

```python
# Example dispatch to Causal Impact
dispatch = AgentDispatchRequest(
    dispatch_id="disp_001",
    target_agent="causal_impact",
    dispatch_reason="User query requires causal effect estimation",
    query="What is the causal effect of HCP engagement on TRx?",
    parsed_query={
        "intent": "causal_effect",
        "treatment": "hcp_engagement_frequency",
        "outcome": "trx_total",
        "brand": "kisqali"
    },
    execution_mode="sequential",
    upstream_results=[],
    depends_on=[]
)
```

### 5.2 Tier 2 → Orchestrator Response

All Tier 2 agents return via the standard `AgentDispatchResponse`:

```python
# Example response from Causal Impact
response = AgentDispatchResponse(
    dispatch_id="disp_001",
    agent_name="causal_impact",
    status="completed",
    agent_result={
        "ate_estimate": 0.23,
        "confidence_interval": [0.18, 0.28],
        "causal_narrative": "...",
        "executive_summary": "...",
        # ... rest of CausalImpactOutput
    },
    errors=[],
    next_agent="heterogeneous_optimizer",
    handoff_context={
        # Include handoff YAML as dict
    }
)
```

### 5.3 Tier 2 Inter-Agent Handoffs

#### Causal Impact → Heterogeneous Optimizer

When Causal Impact detects heterogeneity, it should dispatch to Heterogeneous Optimizer:

```python
# Causal Impact sets in response
next_agent = "heterogeneous_optimizer" if heterogeneity_detected else None
handoff_context = {
    "upstream_agent": "causal_impact",
    "treatment_var": state["treatment_var"],
    "outcome_var": state["outcome_var"],
    "overall_ate": state["ate_estimate"],
    "confounders": state["confounders"],
    "data_source": state["data_source"],
    "filters": state["filters"],
    "reason": "Heterogeneous effects detected - recommend segment-level analysis"
}
```

#### Gap Analyzer → Causal Impact

When Gap Analyzer identifies an opportunity requiring causal validation:

```python
# Gap Analyzer sets in response
next_agent = "causal_impact"
handoff_context = {
    "upstream_agent": "gap_analyzer",
    "top_opportunity": {
        "metric": "trx",
        "segment": "specialty=Oncology",
        "gap_size": 500,
        "expected_roi": 3.2
    },
    "reason": "Validate whether gap can be causally attributed to actionable intervention"
}
```

---

## 6. Validation Rules

### 6.1 Input Validation

**All Tier 2 Agents:**
```python
def validate_tier2_input(state: Dict[str, Any]) -> List[str]:
    """Common validation for Tier 2 agent inputs"""
    errors = []

    # Required fields
    if not state.get("query"):
        errors.append("query is required")
    if not state.get("data_source"):
        errors.append("data_source is required")

    # Brand context
    if state.get("brand") and state["brand"] not in ["kisqali", "fabhalta", "remibrutinib"]:
        errors.append(f"Invalid brand: {state['brand']}")

    return errors
```

**Causal Impact Specific:**
```python
def validate_causal_impact_input(state: CausalImpactState) -> List[str]:
    """Validation for Causal Impact inputs"""
    errors = validate_tier2_input(state)

    # Treatment and outcome
    if not state.get("treatment_var"):
        errors.append("treatment_var is required")
    if not state.get("outcome_var"):
        errors.append("outcome_var is required")
    if state.get("treatment_var") == state.get("outcome_var"):
        errors.append("treatment_var and outcome_var must be different")

    # Confounders
    if not state.get("confounders") or len(state["confounders"]) == 0:
        errors.append("At least one confounder is required")

    # Confidence level
    if state.get("confidence_level"):
        if not (0.5 <= state["confidence_level"] <= 0.99):
            errors.append("confidence_level must be between 0.5 and 0.99")

    return errors
```

**Gap Analyzer Specific:**
```python
def validate_gap_analyzer_input(state: GapAnalyzerState) -> List[str]:
    """Validation for Gap Analyzer inputs"""
    errors = validate_tier2_input(state)

    # Metrics
    if not state.get("metrics") or len(state["metrics"]) == 0:
        errors.append("At least one metric is required")

    # Segments
    if not state.get("segments") or len(state["segments"]) == 0:
        errors.append("At least one segment is required")

    # Brand required
    if not state.get("brand"):
        errors.append("brand is required for Gap Analyzer")

    # Threshold
    if state.get("min_gap_threshold"):
        if not (0 <= state["min_gap_threshold"] <= 100):
            errors.append("min_gap_threshold must be between 0 and 100")

    return errors
```

**Heterogeneous Optimizer Specific:**
```python
def validate_heterogeneous_optimizer_input(state: HeterogeneousOptimizerState) -> List[str]:
    """Validation for Heterogeneous Optimizer inputs"""
    errors = validate_tier2_input(state)

    # Treatment and outcome
    if not state.get("treatment_var"):
        errors.append("treatment_var is required")
    if not state.get("outcome_var"):
        errors.append("outcome_var is required")

    # Segments and modifiers
    if not state.get("segment_vars") or len(state["segment_vars"]) == 0:
        errors.append("At least one segment_var is required")
    if not state.get("effect_modifiers") or len(state["effect_modifiers"]) == 0:
        errors.append("At least one effect_modifier is required")

    # Model parameters
    if state.get("n_estimators"):
        if not (50 <= state["n_estimators"] <= 500):
            errors.append("n_estimators must be between 50 and 500")
    if state.get("min_samples_leaf"):
        if not (5 <= state["min_samples_leaf"] <= 100):
            errors.append("min_samples_leaf must be between 5 and 100")

    return errors
```

### 6.2 Output Validation

```python
def validate_tier2_output(output: Dict[str, Any], agent_name: str) -> List[str]:
    """Validate Tier 2 agent outputs"""
    errors = []

    # Common required fields
    required = ["executive_summary", "key_insights", "total_latency_ms", "confidence"]
    for field in required:
        if field not in output:
            errors.append(f"{agent_name} output missing required field: {field}")

    # Confidence range
    if "confidence" in output:
        if not (0.0 <= output["confidence"] <= 1.0):
            errors.append(f"{agent_name} confidence must be between 0.0 and 1.0")

    # Latency sanity check
    if "total_latency_ms" in output:
        max_latency = {
            "causal_impact": 120000,  # 120s
            "gap_analyzer": 20000,     # 20s
            "heterogeneous_optimizer": 150000  # 150s
        }
        if output["total_latency_ms"] > max_latency.get(agent_name, 180000):
            errors.append(f"{agent_name} exceeded maximum latency")

    return errors
```

---

## 7. Error Handling

### 7.1 Common Error Patterns

**Data Errors:**
```python
class DataError(Exception):
    """Raised when data fetching or quality issues occur"""
    pass

# Example usage in node
try:
    df = await data_connector.query(...)
    if df is None or len(df) < minimum_rows:
        return {
            **state,
            "errors": [{"node": node_name, "error": f"Insufficient data (need >= {minimum_rows})"}],
            "status": "failed"
        }
except Exception as e:
    return {
        **state,
        "errors": [{"node": node_name, "error": f"Data fetch failed: {str(e)}"}],
        "status": "failed"
    }
```

**Estimation Errors (Causal Impact, Heterogeneous Optimizer):**
```python
# Fallback chain for estimation
estimation_methods = [
    "backdoor.econml.dml.CausalForestDML",
    "backdoor.econml.dml.LinearDML",
    "backdoor.linear_regression",
    "backdoor.propensity_score_weighting"
]

for method in estimation_methods:
    try:
        estimate = await estimate_with_method(model, method)
        return {**state, "ate_estimate": estimate.value, "status": "computing"}
    except Exception as e:
        continue  # Try next method

# If all fail
return {
    **state,
    "errors": [{"node": "estimation", "error": "All estimation methods failed"}],
    "status": "failed"
}
```

**Timeout Handling:**
```python
import asyncio

try:
    result = await asyncio.wait_for(
        expensive_computation(),
        timeout=timeout_seconds
    )
except asyncio.TimeoutError:
    return {
        **state,
        "errors": [{"node": node_name, "error": f"Timed out after {timeout_seconds}s"}],
        "warnings": ["Consider reducing data size or complexity"],
        "status": "failed"
    }
```

### 7.2 Graceful Degradation

**Causal Impact:**
- If interpretation fails → return computation results with minimal template-based narrative
- If refutation fails → proceed with warning
- If sensitivity analysis fails → proceed without sensitivity results

**Gap Analyzer:**
- If some segments fail → proceed with successful segments, add warning
- If ROI estimation fails for some gaps → exclude those gaps, continue
- If benchmarks unavailable → use temporal comparison instead

**Heterogeneous Optimizer:**
- If CATE estimation fails → fall back to linear model
- If segment has insufficient data → exclude segment, add warning
- If policy learning fails → return segment profiles without policy recommendations

---

## 8. Performance Requirements

| Agent | Target Latency | Max Latency | Throughput |
|-------|----------------|-------------|------------|
| **Causal Impact** | 60s (computation) + 30s (interpretation) | 120s | 1 query/2min |
| **Gap Analyzer** | 15s | 20s | 3 queries/min |
| **Heterogeneous Optimizer** | 120s | 150s | 1 query/3min |

### 8.1 Latency Monitoring

```python
import time

async def measure_latency(node_fn, state: Dict[str, Any], node_name: str):
    """Wrapper to measure and log node latency"""
    start_time = time.time()
    result = await node_fn(state)
    latency_ms = int((time.time() - start_time) * 1000)

    # Log if exceeds threshold
    thresholds = {
        "graph_builder": 5000,
        "estimation": 60000,
        "interpretation": 30000,
        "gap_detector": 10000,
        "cate_estimator": 120000
    }

    if latency_ms > thresholds.get(node_name, 30000):
        logger.warning(f"{node_name} latency {latency_ms}ms exceeds threshold")

    return {**result, f"{node_name}_latency_ms": latency_ms}
```

---

## 9. Testing Requirements

### 9.1 Unit Tests

**Causal Impact:**
- DoWhy graph construction
- EconML estimation with known effects
- Refutation tests pass/fail logic
- Interpretation prompt generation
- LLM fallback mechanism

**Gap Analyzer:**
- Gap detection across segment types
- ROI calculation correctness
- Prioritization ranking
- Quick win identification
- Strategic bet identification

**Heterogeneous Optimizer:**
- CATE estimation with synthetic data
- High/low responder identification
- Policy recommendation generation
- Feature importance ranking

### 9.2 Integration Tests

```python
# Example integration test structure
async def test_tier2_agent_integration(agent_name: str):
    """Test full agent execution"""

    # Setup
    state = create_test_input(agent_name)
    graph = build_agent_graph(agent_name)

    # Execute
    result = await graph.ainvoke(state)

    # Validate output contract
    errors = validate_tier2_output(result, agent_name)
    assert len(errors) == 0, f"Output validation failed: {errors}"

    # Check required fields
    assert result["status"] == "completed"
    assert result["executive_summary"]
    assert len(result["key_insights"]) > 0
    assert 0.0 <= result["confidence"] <= 1.0
```

### 9.3 End-to-End Tests

```python
async def test_tier2_orchestration_flow():
    """Test orchestrator dispatching to Tier 2 agents"""

    # Gap Analyzer → Causal Impact → Heterogeneous Optimizer flow
    query = "Where are the biggest opportunities and what causes them?"

    # Stage 1: Gap Analyzer
    gap_result = await orchestrator.dispatch("gap_analyzer", query, {})
    assert gap_result["status"] == "completed"
    assert len(gap_result["agent_result"]["prioritized_opportunities"]) > 0

    # Stage 2: Causal Impact (triggered by handoff)
    if gap_result["next_agent"] == "causal_impact":
        causal_result = await orchestrator.dispatch(
            "causal_impact",
            query,
            gap_result["handoff_context"]
        )
        assert causal_result["status"] == "completed"
        assert causal_result["agent_result"]["ate_estimate"] is not None

        # Stage 3: Heterogeneous Optimizer (triggered if heterogeneity detected)
        if causal_result["next_agent"] == "heterogeneous_optimizer":
            hetero_result = await orchestrator.dispatch(
                "heterogeneous_optimizer",
                query,
                causal_result["handoff_context"]
            )
            assert hetero_result["status"] == "completed"
            assert hetero_result["agent_result"]["heterogeneity_score"] > 0
```

---

## 10. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-18 | Initial Tier 2 contracts | Claude |

---

## 11. Related Documents

- `base-contract.md` - Base agent structures
- `orchestrator-contracts.md` - Orchestrator communication
- `tier0-contracts.md` - ML Foundation contracts
- `agent-handoff.yaml` - Standard handoff format examples
- `.claude/specialists/Agent_Specialists_Tiers 1-5/causal-impact.md` - Causal Impact specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/gap-analyzer.md` - Gap Analyzer specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/heterogeneous-optimizer.md` - Heterogeneous Optimizer specialist
