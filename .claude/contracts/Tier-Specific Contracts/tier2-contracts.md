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

    # V4.2: Energy Score Configuration
    selection_strategy: Literal["first_success", "best_energy", "ensemble"] = "best_energy"
    energy_score_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Energy score configuration overrides: weights, n_bootstrap, etc."
    )

    # V4.4: Discovery Configuration
    auto_discover: bool = Field(False, description="Enable automatic DAG structure learning")
    discovery_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Discovery configuration: algorithms=['ges','pc'], ensemble_threshold, alpha"
    )

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

# V4.2: Energy Score Enhancement
class EnergyScoreComponents(BaseModel):
    """Components of the energy score"""
    treatment_balance: float = Field(..., ge=0, le=1, description="Covariate balance score (35% weight)")
    outcome_fit: float = Field(..., ge=0, le=1, description="DR residual fit score (45% weight)")
    propensity_calibration: float = Field(..., ge=0, le=1, description="Propensity calibration (20% weight)")

class EstimatorEvaluationResult(BaseModel):
    """Result of evaluating a single estimator with energy score"""
    estimator_type: Literal[
        "causal_forest", "linear_dml", "dml_learner", "drlearner",
        "ortho_forest", "s_learner", "t_learner", "x_learner", "ols"
    ]
    success: bool
    ate: Optional[float] = None
    ate_std: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None
    energy_score: Optional[float] = Field(None, ge=0, le=1, description="Combined energy score (lower is better)")
    energy_components: Optional[EnergyScoreComponents] = None
    energy_ci_lower: Optional[float] = None
    energy_ci_upper: Optional[float] = None
    estimation_time_ms: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None

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

    # V4.2: Energy Score Enhancement
    selected_estimator: str = Field(..., description="Estimator type that was selected")
    selection_strategy: Literal["first_success", "best_energy", "ensemble"]
    energy_score: Optional[float] = Field(None, ge=0, le=1, description="Energy score of selected estimator")
    energy_quality_tier: Optional[Literal["excellent", "good", "acceptable", "poor", "unreliable"]] = None
    energy_components: Optional[EnergyScoreComponents] = None
    estimator_evaluations: List[EstimatorEvaluationResult] = Field(
        default_factory=list,
        description="All estimator results for comparison"
    )
    energy_score_gap: Optional[float] = Field(
        None,
        description="Gap between best and second-best energy score"
    )

    # V4.4: Discovery Results
    discovery_enabled: bool = Field(False, description="Whether auto-discovery was used")
    discovery_gate_decision: Optional[Literal["accept", "augment", "review", "reject"]] = None
    discovery_confidence: Optional[float] = Field(None, ge=0, le=1, description="Ensemble confidence")
    discovery_algorithms_used: List[str] = Field(default_factory=list, description="Algorithms used")
    discovery_latency_ms: Optional[float] = None
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

  # V4.2: Energy Score Enhancement
  estimator_selection:
    selected_estimator: <causal_forest|linear_dml|drlearner|ols|...>
    selection_strategy: <first_success|best_energy|ensemble>
    energy_score: <0.0-1.0>
    energy_quality_tier: <excellent|good|acceptable|poor|unreliable>
    energy_components:
      treatment_balance: <score>
      outcome_fit: <score>
      propensity_calibration: <score>
    alternatives_evaluated:
      - estimator: <name>
        energy_score: <score>
        ate: <estimate>
    energy_score_gap: <gap between best and second-best>

  # V4.4: Discovery Results (if auto_discover=True)
  discovery:
    enabled: <bool>
    gate_decision: <accept|augment|review|reject>
    confidence: <0.0-1.0>
    algorithms_used: [<ges>, <pc>]
    discovered_edges: <count>
    augmented_edges: [<list of (source, target) tuples>]
    latency_ms: <value>

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

    # V4.4: DAG Validation Outputs
    dag_validated_segments: Optional[List[str]] = Field(
        None, description="Segments with valid causal paths in discovered DAG"
    )
    dag_invalid_segments: Optional[List[str]] = Field(
        None, description="Segments without causal paths (use with caution)"
    )
    latent_confounder_segments: Optional[List[str]] = Field(
        None, description="Segments with bidirected edges (latent confounders)"
    )
    dag_validation_warnings: Optional[List[str]] = Field(
        None, description="Warnings from DAG validation"
    )
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

  # V4.4: DAG Validation Results (if discovered DAG available)
  dag_validation:
    enabled: <bool>
    validated_segments: <count>
    invalid_segments: <count>
    latent_confounder_segments: <count>
    warnings: [<list of warnings>]

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

---

## DSPy Sender Role

### Overview

All Tier 2 agents are **DSPy Sender** agents that:
1. Generate training signals from analysis executions
2. Provide signature-specific training examples
3. Route high-quality signals to feedback_learner for optimization

```python
# dspy_type identification
dspy_type: Literal["sender"] = "sender"
```

### Training Signal Contract

All Tier 2 agents implement a TrainingSignal dataclass with:

```python
@dataclass
class Tier2TrainingSignal:
    """Base structure for Tier 2 training signals."""

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""

    # === Agent-Specific Fields ===
    # (Defined per agent)

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """Compute reward for MIPROv2 optimization."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        ...
```

### Causal Impact Training Signal

```python
@dataclass
class CausalAnalysisTrainingSignal:
    """Training signal for Causal Impact DSPy optimization."""

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    treatment_var: str = ""
    outcome_var: str = ""
    confounders_count: int = 0

    # === Graph Building Phase ===
    dag_nodes_count: int = 0
    dag_edges_count: int = 0
    adjustment_sets_found: int = 0
    graph_confidence: float = 0.0

    # === Estimation Phase ===
    estimation_method: str = ""
    ate_estimate: float = 0.0
    ate_ci_width: float = 0.0
    statistical_significance: bool = False
    effect_size: str = ""  # small, medium, large
    sample_size: int = 0

    # === Refutation Phase ===
    refutation_tests_passed: int = 0
    refutation_tests_failed: int = 0
    overall_robust: bool = False

    # === Sensitivity Phase ===
    e_value: float = 0.0
    robust_to_confounding: bool = False

    # === V4.2: Energy Score Selection Phase ===
    selection_strategy: str = ""  # first_success, best_energy, ensemble
    selected_estimator: str = ""  # causal_forest, linear_dml, etc.
    estimators_evaluated: int = 0
    estimators_succeeded: int = 0
    energy_score: float = 0.0
    energy_quality_tier: str = ""  # excellent, good, acceptable, poor, unreliable
    energy_score_components: Dict[str, float] = field(default_factory=dict)  # treatment_balance, outcome_fit, propensity_calibration
    energy_score_gap: float = 0.0  # Gap between best and second-best
    selection_time_ms: float = 0.0

    # === V4.4: Discovery Phase ===
    auto_discover: bool = False
    discovery_algorithms: List[str] = field(default_factory=list)  # ["ges", "pc"]
    discovery_gate_decision: str = ""  # accept, augment, review, reject
    discovery_confidence: float = 0.0
    discovered_edges_count: int = 0
    augmented_edges_count: int = 0
    discovery_latency_ms: float = 0.0

    # === Interpretation Phase ===
    interpretation_depth: str = ""  # none, minimal, standard, deep
    narrative_length: int = 0
    key_findings_count: int = 0
    recommendations_count: int = 0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting (V4.2 Updated):
        - refutation_robustness: 0.25 (passed / total)
        - estimation_quality: 0.20 (significance + CI width)
        - energy_score_quality: 0.15 (1 - energy_score, inverted since lower is better)
        - interpretation_quality: 0.15 (depth + findings)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.10 (if available)
        """
        ...
```

### Gap Analyzer Training Signal

```python
@dataclass
class GapAnalysisTrainingSignal:
    """Training signal for Gap Analyzer DSPy optimization."""

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    brand: str = ""
    region: str = ""
    time_period: str = ""

    # === Gap Detection Phase ===
    gaps_identified: int = 0
    gap_categories: List[str] = field(default_factory=list)
    total_opportunity_value: float = 0.0
    prioritized_gaps: int = 0

    # === ROI Analysis Phase ===
    roi_calculations_performed: int = 0
    average_roi_estimate: float = 0.0
    high_confidence_gaps: int = 0

    # === Recommendation Phase ===
    recommendations_generated: int = 0
    actionable_recommendations: int = 0
    implementation_complexity: str = ""  # low, medium, high

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - gap_detection_quality: 0.30 (gaps + prioritization)
        - roi_accuracy: 0.25 (confidence + consistency)
        - actionability: 0.20 (actionable / total recommendations)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.10 (if available)
        """
        ...
```

### Heterogeneous Optimizer Training Signal

```python
@dataclass
class HeterogeneousOptimizationTrainingSignal:
    """Training signal for Heterogeneous Optimizer DSPy optimization."""

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    treatment_var: str = ""
    outcome_var: str = ""
    effect_modifiers: List[str] = field(default_factory=list)

    # === Segmentation Phase ===
    segments_analyzed: int = 0
    significant_segments: int = 0
    segment_method: str = ""  # causal_forest, meta_learner, etc.

    # === CATE Estimation Phase ===
    cate_estimates_count: int = 0
    effect_heterogeneity: float = 0.0
    best_responder_segment: str = ""
    worst_responder_segment: str = ""

    # === V4.4: DAG Validation Phase ===
    dag_validation_enabled: bool = False
    dag_validated_segments: int = 0
    dag_invalid_segments: int = 0
    latent_confounder_segments: int = 0
    dag_gate_decision: str = ""  # accept, review, reject, augment
    dag_validation_warnings: int = 0

    # === Policy Learning Phase ===
    policy_rules_generated: int = 0
    expected_policy_value: float = 0.0
    policy_confidence: float = 0.0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting (V4.4 Updated):
        - segmentation_quality: 0.25 (significant / total segments)
        - cate_precision: 0.25 (heterogeneity + confidence)
        - dag_validation_quality: 0.10 (validated segments ratio)
        - policy_value: 0.15 (expected value + rules)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.10 (if available)
        """
        ...
```

### Signal Collector Contract

All Tier 2 agents implement a SignalCollector:

```python
class Tier2SignalCollector:
    """Signal collector for Tier 2 Sender agents."""

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[TrainingSignal] = []
        self._buffer_limit = 100

    def collect_signal(self, **kwargs) -> TrainingSignal:
        """Initialize training signal at analysis start."""
        ...

    def update_phase(self, signal: TrainingSignal, **kwargs) -> TrainingSignal:
        """Update signal with phase-specific results."""
        ...

    def finalize_signal(
        self,
        signal: TrainingSignal,
        total_latency_ms: float,
        confidence_score: float,
    ) -> TrainingSignal:
        """Finalize signal and add to buffer."""
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)
        return signal

    def update_with_feedback(
        self,
        signal: TrainingSignal,
        user_satisfaction: Optional[float] = None,
    ) -> TrainingSignal:
        """Update signal with user feedback (delayed)."""
        signal.user_satisfaction = user_satisfaction
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [
            s.to_dict()
            for s in self._signals_buffer
            if s.compute_reward() >= min_reward
        ]
        return signals[-limit:]

    def get_high_quality_examples(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get highest-quality examples for few-shot learning."""
        sorted_signals = sorted(
            self._signals_buffer,
            key=lambda s: s.compute_reward(),
            reverse=True,
        )
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()
```

### DSPy Signatures

#### Causal Impact Signatures

```python
class CausalGraphSignature(dspy.Signature):
    """Construct a causal DAG from domain knowledge and data."""

    treatment_var: str = dspy.InputField(desc="Treatment variable name")
    outcome_var: str = dspy.InputField(desc="Outcome variable name")
    available_vars: str = dspy.InputField(desc="All available variables in data")
    domain_context: str = dspy.InputField(desc="Domain knowledge context")

    confounders: list = dspy.OutputField(desc="Variables that confound treatment-outcome")
    mediators: list = dspy.OutputField(desc="Variables that mediate the effect")
    adjustment_set: list = dspy.OutputField(desc="Recommended adjustment set")
    graph_rationale: str = dspy.OutputField(desc="Explanation of graph structure")

class EvidenceSynthesisSignature(dspy.Signature):
    """Synthesize causal evidence into an interpretation."""

    estimation_result: str = dspy.InputField(desc="ATE estimate with CI and significance")
    refutation_summary: str = dspy.InputField(desc="Summary of refutation test results")
    sensitivity_summary: str = dspy.InputField(desc="E-value and sensitivity interpretation")
    user_expertise: str = dspy.InputField(desc="User expertise level")

    narrative: str = dspy.OutputField(desc="Natural language interpretation")
    key_findings: list = dspy.OutputField(desc="3-5 key findings as bullet points")
    confidence_level: str = dspy.OutputField(desc="low, medium, or high")
    recommendations: list = dspy.OutputField(desc="Actionable recommendations")
    limitations: list = dspy.OutputField(desc="Important caveats and limitations")

# V4.2: Energy Score Selection Signature
class EstimatorSelectionSignature(dspy.Signature):
    """Interpret estimator selection results and energy score quality."""

    selection_strategy: str = dspy.InputField(desc="Selection strategy used: first_success, best_energy, ensemble")
    selected_estimator: str = dspy.InputField(desc="Name of selected estimator")
    estimator_evaluations: str = dspy.InputField(desc="JSON summary of all estimator results with energy scores")
    energy_score: float = dspy.InputField(desc="Energy score of selected estimator (0-1, lower is better)")
    energy_components: str = dspy.InputField(desc="Treatment balance, outcome fit, propensity calibration scores")

    quality_tier: str = dspy.OutputField(desc="Quality tier: excellent, good, acceptable, poor, unreliable")
    selection_rationale: str = dspy.OutputField(desc="Why this estimator was selected over alternatives")
    energy_interpretation: str = dspy.OutputField(desc="Natural language interpretation of energy score components")
    confidence_assessment: str = dspy.OutputField(desc="Assessment of causal estimate confidence based on energy score")
    recommendations: list = dspy.OutputField(desc="Recommendations based on energy score quality")
```

#### Gap Analyzer Signatures

```python
class GapDetectionSignature(dspy.Signature):
    """Detect gaps and opportunities in commercial operations."""

    brand: str = dspy.InputField(desc="Brand name")
    region: str = dspy.InputField(desc="Geographic region")
    metrics_data: str = dspy.InputField(desc="Current performance metrics")
    benchmark_data: str = dspy.InputField(desc="Benchmark or target metrics")

    gaps_identified: list = dspy.OutputField(desc="List of identified gaps")
    gap_priorities: list = dspy.OutputField(desc="Priority ranking of gaps")
    opportunity_value: float = dspy.OutputField(desc="Total opportunity value")
    gap_rationale: str = dspy.OutputField(desc="Explanation of gap detection")

class ROIEstimationSignature(dspy.Signature):
    """Estimate ROI for addressing identified gaps."""

    gap_description: str = dspy.InputField(desc="Description of the gap")
    historical_data: str = dspy.InputField(desc="Historical intervention data")
    resource_constraints: str = dspy.InputField(desc="Available resources")

    roi_estimate: float = dspy.OutputField(desc="Expected ROI")
    confidence_interval: str = dspy.OutputField(desc="CI for ROI estimate")
    implementation_steps: list = dspy.OutputField(desc="Steps to capture ROI")
    assumptions: list = dspy.OutputField(desc="Key assumptions in estimate")
```

#### Heterogeneous Optimizer Signatures

```python
class SegmentIdentificationSignature(dspy.Signature):
    """Identify segments with heterogeneous treatment effects."""

    treatment_var: str = dspy.InputField(desc="Treatment variable")
    outcome_var: str = dspy.InputField(desc="Outcome variable")
    effect_modifiers: str = dspy.InputField(desc="Potential effect modifiers")
    data_summary: str = dspy.InputField(desc="Summary of available data")

    segments: list = dspy.OutputField(desc="Identified segments")
    segment_rationale: str = dspy.OutputField(desc="Why these segments matter")
    expected_heterogeneity: str = dspy.OutputField(desc="Expected effect variation")

class PolicyRecommendationSignature(dspy.Signature):
    """Generate treatment allocation policy recommendations."""

    cate_estimates: str = dspy.InputField(desc="CATE estimates by segment")
    resource_constraints: str = dspy.InputField(desc="Resource limitations")
    business_objectives: str = dspy.InputField(desc="Business goals")

    policy_rules: list = dspy.OutputField(desc="Treatment allocation rules")
    expected_value: float = dspy.OutputField(desc="Expected policy value")
    implementation_guidance: str = dspy.OutputField(desc="How to implement policy")
```

### Signal Flow to Feedback Learner

Tier 2 agents send signals to feedback_learner for optimization:

```yaml
signal_flow:
  sender: tier_2_agent
  receiver: feedback_learner
  protocol:
    - Agent collects signal during execution
    - Signal finalized with outcome metrics
    - High-quality signals (reward > 0.5) queued for optimization
    - Feedback learner batches signals for MIPROv2 training

  trigger_conditions:
    - batch_size >= 50
    - time_elapsed >= 24h
    - explicit_optimization_request

  optimization_targets:
    - Improve signature prompts
    - Refine few-shot examples
    - Adjust reward weights
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-31 | V4.4.1: Added DAG Validation outputs for Heterogeneous Optimizer |
| 2025-12-30 | V4.4: Added Discovery Phase contracts |
| 2025-12-26 | V4.2: Added Energy Score Enhancement contracts |
| 2025-12-23 | V5: Added DSPy Sender role specification |
| 2025-12-08 | V4: Added ROI calculation outputs |
| 2025-12-04 | Initial creation for V3 |

---

## V4.2: Energy Score Enhancement Summary

### New Contracts Added

1. **EnergyScoreComponents** - Component scores (treatment_balance, outcome_fit, propensity_calibration)
2. **EstimatorEvaluationResult** - Individual estimator evaluation with energy score
3. **EstimatorSelectionSignature** - DSPy signature for interpreting selection results

### Modified Contracts

1. **CausalImpactInput** - Added `selection_strategy` and `energy_score_config` fields
2. **CausalImpactOutput** - Added energy score outputs and estimator evaluation list
3. **CausalAnalysisTrainingSignal** - Added energy score selection phase fields
4. **Handoff Format** - Added `estimator_selection` block with energy score details

### Quality Tiers

| Tier | Max Score | Description |
|------|-----------|-------------|
| excellent | 0.25 | High confidence in causal estimate |
| good | 0.45 | Reasonable confidence |
| acceptable | 0.65 | Use with caution |
| poor | 0.80 | Low confidence, consider alternatives |
| unreliable | 1.00 | Results likely unreliable |

### Supported Estimators

- causal_forest (EconML)
- linear_dml (EconML)
- dml_learner (EconML)
- drlearner (EconML)
- ortho_forest (EconML)
- s_learner (CausalML)
- t_learner (CausalML)
- x_learner (CausalML)
- ols (sklearn fallback)

### Selection Strategies

- `first_success` - Legacy: use first estimator that succeeds
- `best_energy` - Default: select estimator with lowest energy score
- `ensemble` - Future: combine multiple estimators

---

## V4.4: Discovery Phase Summary

### New Contracts Added

1. **CausalImpactInput** - Added `auto_discover` and `discovery_config` fields
2. **CausalImpactOutput** - Added discovery result fields
3. **CausalAnalysisTrainingSignal** - Added discovery phase fields
4. **Handoff Format** - Added discovery section

### Discovery Configuration

```python
discovery_config: Optional[Dict[str, Any]] = {
    "algorithms": ["ges", "pc"],  # Discovery algorithms
    "ensemble_threshold": 0.5,     # Min agreement for edge inclusion
    "alpha": 0.05,                 # Significance level for CI tests
}
```

### Gate Decisions

| Decision | Confidence | Behavior |
|----------|------------|----------|
| accept | >= 0.8 | Use discovered DAG directly |
| augment | 0.5 - 0.8 | Add high-confidence edges to manual DAG |
| review | 0.3 - 0.5 | Flag for expert review |
| reject | < 0.3 | Fall back to manual DAG only |

### Supported Discovery Algorithms

- `ges` - Greedy Equivalence Search (score-based)
- `pc` - Peter-Clark (constraint-based)
- `fci` - Fast Causal Inference (future: latent confounders)
- `lingam` - Linear Non-Gaussian (future: non-Gaussian data)

### Database Schema

Discovery results stored in:
- `ml.discovered_dags` - Discovered DAG structures
- `ml.driver_rankings` - Causal vs predictive rankings
- `ml.discovery_algorithm_runs` - Algorithm execution logs
