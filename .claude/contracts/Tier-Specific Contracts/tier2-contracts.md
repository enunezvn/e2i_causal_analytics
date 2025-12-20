# Tier 2: Causal Inference Contracts

## Overview

Tier 2 contains the core causal inference agents: Causal Impact, Gap Analyzer, and Heterogeneous Optimizer. These agents perform the primary analytical work of the platform.

---

## Causal Impact Agent Contracts

### Input Contract

```python
# src/contracts/causal_impact.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .base import BaseAgentInput

class CausalImpactInput(BaseAgentInput):
    """Input contract for Causal Impact Agent"""
    
    # Required fields
    treatment: str = Field(..., description="Treatment/intervention variable")
    outcome: str = Field(..., description="Outcome variable to measure")
    
    # Optional configuration
    confounders: Optional[List[str]] = Field(None, description="Known confounders to control for")
    instruments: Optional[List[str]] = Field(None, description="Instrumental variables if available")
    effect_modifiers: Optional[List[str]] = Field(None, description="Variables that may modify effect")
    
    # Analysis configuration
    estimation_method: Optional[Literal[
        "backdoor",
        "iv",
        "frontdoor",
        "auto"
    ]] = "auto"
    
    run_refutation: bool = True
    run_sensitivity: bool = True
    
    # Interpretation
    interpretation_depth: Literal["none", "minimal", "standard", "deep"] = "standard"
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
    
    # Data scope
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    segment_filter: Optional[Dict[str, Any]] = None
```

### Output Contract

```python
class RefutationResult(BaseModel):
    """Result of a single refutation test"""
    test_name: str
    original_effect: float
    refuted_effect: float
    p_value: float
    passed: bool

class SensitivityResult(BaseModel):
    """Sensitivity analysis results"""
    e_value: float
    robustness_value: float
    confounder_strength_bound: float
    interpretation: str

class CausalImpactOutput(BaseAgentOutput):
    """Output contract for Causal Impact Agent"""
    
    # Primary results
    ate: float = Field(..., description="Average Treatment Effect")
    confidence_interval: tuple = Field(..., description="(lower, upper) bounds")
    p_value: float
    standard_error: float
    
    # Method used
    estimation_method: str
    identified_estimand: str
    
    # Robustness
    refutation_results: List[RefutationResult] = []
    all_refutations_passed: bool
    sensitivity_analysis: Optional[SensitivityResult] = None
    
    # Interpretation
    interpretation: Optional[str] = None
    confidence_level: Literal["high", "medium", "low"]
    
    # Causal graph
    causal_graph_spec: Optional[Dict[str, Any]] = None
```

### Handoff Format

```yaml
causal_impact_handoff:
  agent: causal_impact
  analysis_type: causal_effect_estimation
  
  key_findings:
    - ate: <effect size>
    - ci: [<lower>, <upper>]
    - significant: <bool>
    - robust: <bool>
  
  methodology:
    estimand: <identified estimand>
    method: <estimation method>
    confounders_controlled: [<list>]
  
  robustness:
    refutations_passed: <count>/<total>
    e_value: <value>
  
  interpretation: <natural language summary>
  confidence_level: high|medium|low
  
  requires_further_analysis: <bool>
  suggested_next_agent: heterogeneous_optimizer|explainer
```

---

## Gap Analyzer Agent Contracts

### Input Contract

```python
# src/contracts/gap_analyzer.py

class GapAnalyzerInput(BaseAgentInput):
    """Input contract for Gap Analyzer Agent"""
    
    # Required
    metrics: List[str] = Field(..., description="Metrics to analyze for gaps")
    
    # Segmentation
    segment_by: Optional[List[str]] = Field(None, description="Dimensions to segment by")
    segment_filter: Optional[Dict[str, Any]] = None
    
    # Analysis configuration
    gap_types: List[Literal[
        "vs_target",
        "vs_benchmark",
        "vs_potential",
        "temporal"
    ]] = ["vs_target", "vs_potential"]
    
    benchmark_group: Optional[str] = None
    target_values: Optional[Dict[str, float]] = None
    
    # ROI configuration
    calculate_roi: bool = True
    revenue_per_unit: Optional[Dict[str, float]] = None
    
    # Output
    max_opportunities: int = 20
    prioritization: Literal["roi", "gap_size", "difficulty"] = "roi"
```

### Output Contract

```python
class PerformanceGap(BaseModel):
    """Individual performance gap"""
    segment_id: str
    segment_name: str
    metric: str
    current_value: float
    target_value: float
    gap_absolute: float
    gap_percentage: float
    gap_type: str

class ROIEstimate(BaseModel):
    """ROI estimate for closing a gap"""
    gap_id: str
    revenue_potential: float
    estimated_cost: float
    roi: float
    confidence: float
    payback_period_months: Optional[float]

class PrioritizedOpportunity(BaseModel):
    """Prioritized opportunity for action"""
    rank: int
    segment_id: str
    segment_name: str
    gap: PerformanceGap
    roi_estimate: ROIEstimate
    difficulty: Literal["low", "medium", "high"]
    category: Literal["quick_win", "strategic_bet", "low_priority"]
    recommended_action: str

class GapAnalyzerOutput(BaseAgentOutput):
    """Output contract for Gap Analyzer Agent"""
    
    # All gaps found
    gaps: List[PerformanceGap]
    total_gap_value: float
    
    # ROI analysis
    roi_estimates: List[ROIEstimate]
    total_revenue_potential: float
    
    # Prioritized opportunities
    opportunities: List[PrioritizedOpportunity]
    quick_wins: List[PrioritizedOpportunity]
    strategic_bets: List[PrioritizedOpportunity]
    
    # Summary
    summary: str
    top_recommendations: List[str]
```

### Handoff Format

```yaml
gap_analyzer_handoff:
  agent: gap_analyzer
  analysis_type: roi_opportunity_detection
  
  key_findings:
    - total_gaps_found: <count>
    - total_revenue_potential: <amount>
    - quick_wins: <count>
    - strategic_bets: <count>
  
  top_opportunities:
    - segment: <name>
      gap: <percentage>%
      roi: <value>
      action: <recommendation>
  
  segment_summary:
    by_region: {<region>: <gap>}
    by_brand: {<brand>: <gap>}
  
  requires_further_analysis: <bool>
  suggested_next_agent: resource_optimizer|causal_impact
```

---

## Heterogeneous Optimizer Agent Contracts

### Input Contract

```python
# src/contracts/heterogeneous_optimizer.py

class HeterogeneousOptimizerInput(BaseAgentInput):
    """Input contract for Heterogeneous Optimizer Agent"""
    
    # Required
    treatment: str = Field(..., description="Treatment variable")
    outcome: str = Field(..., description="Outcome variable")
    
    # Effect modifier candidates
    effect_modifiers: List[str] = Field(..., description="Variables to test as effect modifiers")
    
    # Configuration
    estimation_method: Literal["causal_forest", "dml", "s_learner", "t_learner"] = "causal_forest"
    
    # Segmentation
    segment_threshold: float = Field(1.5, description="Multiplier of ATE for high responder")
    min_segment_size: int = 100
    
    # Policy learning
    learn_policy: bool = True
    constraint_budget: Optional[float] = None
```

### Output Contract

```python
class CATEResult(BaseModel):
    """Conditional Average Treatment Effect result"""
    ate: float
    cate_by_segment: Dict[str, float]
    heterogeneity_score: float
    feature_importance: Dict[str, float]
    confidence_intervals: Dict[str, tuple]

class SegmentProfile(BaseModel):
    """Profile of a treatment response segment"""
    segment_id: str
    segment_name: str
    size: int
    percentage: float
    cate: float
    cate_vs_ate: float  # Ratio to ATE
    defining_features: Dict[str, Any]
    recommendation: str

class PolicyRecommendation(BaseModel):
    """Optimal treatment policy recommendation"""
    segment_id: str
    recommended_treatment_rate: float
    expected_outcome_lift: float
    confidence: float

class HeterogeneousOptimizerOutput(BaseAgentOutput):
    """Output contract for Heterogeneous Optimizer Agent"""
    
    # CATE results
    cate_result: CATEResult
    
    # Segment profiles
    high_responders: List[SegmentProfile]
    low_responders: List[SegmentProfile]
    all_segments: List[SegmentProfile]
    
    # Policy recommendations
    policy_recommendations: List[PolicyRecommendation]
    expected_total_lift: float
    
    # Summary
    summary: str
    key_effect_modifiers: List[str]
```

### Handoff Format

```yaml
heterogeneous_optimizer_handoff:
  agent: heterogeneous_optimizer
  analysis_type: cate_estimation
  
  key_findings:
    - ate: <overall effect>
    - heterogeneity_score: <0-1>
    - high_responders: <count>
    - low_responders: <count>
  
  top_effect_modifiers:
    - feature: <name>
      importance: <value>
  
  segment_profiles:
    high_responders:
      - segment: <name>
        cate: <value>
        size: <percentage>%
        defining_features: {<feature>: <value>}
  
  policy_recommendation:
    expected_lift: <value>
    allocation_changes:
      - segment: <name>
        current: <rate>
        recommended: <rate>
  
  requires_further_analysis: <bool>
  suggested_next_agent: experiment_designer|explainer
```

---

## Cross-Agent Communication

### Tier 2 Agent Chaining

Common patterns for chaining Tier 2 agents:

```yaml
# Pattern 1: Causal Impact → Heterogeneous Optimizer
# "Find the effect, then optimize by segment"

chain_1:
  - agent: causal_impact
    output_mapping:
      treatment: treatment
      outcome: outcome
      ate: baseline_ate
  - agent: heterogeneous_optimizer
    input_from_previous:
      - treatment
      - outcome
    additional_input:
      effect_modifiers: [specialty, decile, region]

# Pattern 2: Gap Analyzer → Resource Optimizer
# "Find the gaps, then optimize allocation"

chain_2:
  - agent: gap_analyzer
    output_mapping:
      opportunities: allocation_targets
      roi_estimates: response_coefficients
  - agent: resource_optimizer
    input_from_previous:
      - allocation_targets
      - response_coefficients

# Pattern 3: All Tier 2 → Explainer
# "Comprehensive analysis with explanation"

chain_3:
  parallel_group_1:
    - agent: causal_impact
    - agent: gap_analyzer
    - agent: heterogeneous_optimizer
  sequential:
    - agent: explainer
      input:
        analysis_results: [causal_impact, gap_analyzer, heterogeneous_optimizer]
```
