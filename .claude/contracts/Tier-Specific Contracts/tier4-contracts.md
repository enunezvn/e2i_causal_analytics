# Tier 4: ML Prediction Contracts

## Overview

Tier 4 agents handle machine learning predictions and resource optimization. These agents consume outputs from Tier 2-3 agents and provide actionable predictions and allocations.

**Agents Covered:**
- `prediction_synthesizer` - Multi-model ensemble predictions
- `resource_optimizer` - Resource allocation optimization

**Latency Budgets:**
- Prediction Synthesizer: <15s
- Resource Optimizer: <20s

---

## Prediction Synthesizer Contract

### Purpose
Combines multiple ML models into ensemble predictions with calibrated confidence scores.

### Input Contract

```yaml
# prediction_synthesizer_input.yaml
prediction_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # Target specification
  target:
    entity_type: enum             # "hcp" | "territory" | "brand"
    entity_ids: list[string]      # Optional: specific entities, or empty for all
    prediction_type: enum         # "propensity" | "risk" | "next_best_action" | "conversion"
    
  # Context
  context:
    brand: string                 # "Remibrutinib" | "Fabhalta" | "Kisqali"
    time_horizon: string          # "30d" | "60d" | "90d" | "180d"
    segment_filter: object        # Optional segment constraints
    
  # Model preferences
  model_config:
    ensemble_method: enum         # "weighted_average" | "stacking" | "voting"
    min_models: int               # Minimum models required (default: 2)
    confidence_threshold: float   # Minimum confidence to return (default: 0.5)
    include_explanations: bool    # Include SHAP values (default: false)
```

### Output Contract

```yaml
# prediction_synthesizer_output.yaml
prediction_response:
  request_id: string              # Echo from input
  timestamp: datetime
  processing_time_ms: int
  
  # Predictions
  predictions:
    - entity_id: string
      entity_type: string
      prediction_type: string
      
      # Ensemble result
      score: float                # 0.0-1.0 probability
      confidence: float           # 0.0-1.0 confidence in prediction
      confidence_interval:
        lower: float
        upper: float
        
      # Model breakdown
      model_contributions:
        - model_name: string
          model_version: string
          individual_score: float
          weight: float
          
      # Optional explanations
      explanations:               # If include_explanations=true
        top_features:
          - feature_name: string
            shap_value: float
            direction: enum       # "positive" | "negative"
            
  # Ensemble metadata
  ensemble_metadata:
    models_used: int
    ensemble_method: string
    calibration_applied: bool
    
  # Model performance metrics
  model_metrics:
    ensemble_auc: float
    ensemble_precision: float
    ensemble_recall: float
    last_retrained: datetime
    
  # Status
  status: enum                    # "success" | "partial" | "failed"
  warnings: list[string]          # Non-fatal issues
  errors: list[object]            # Fatal issues (if status=failed)
```

### Required Output Keys (Contract 2 Compliance)

```python
REQUIRED_KEYS = ["predictions", "model_metrics"]
```

### Validation Rules

1. **Score Range**: All scores must be in [0.0, 1.0]
2. **Confidence Threshold**: Only return predictions meeting threshold
3. **Model Minimum**: Ensemble must include at least `min_models` models
4. **Explanation Consistency**: If explanations requested, all predictions must include them
5. **Calibration**: Scores should be calibrated probabilities, not raw model outputs

---

## Resource Optimizer Contract

### Purpose
Optimizes allocation of resources (budget, rep time, samples) across entities to maximize ROI.

### Input Contract

```yaml
# resource_optimizer_input.yaml
optimization_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # Resource specification
  resource:
    type: enum                    # "budget" | "rep_time" | "samples" | "calls"
    total_amount: float           # Total available resource
    unit: string                  # "USD" | "hours" | "units" | "count"
    
  # Allocation targets
  targets:
    entity_type: enum             # "hcp" | "territory" | "region"
    entity_ids: list[string]      # Entities to allocate across
    
  # Constraints
  constraints:
    min_per_entity: float         # Minimum allocation per entity
    max_per_entity: float         # Maximum allocation per entity
    fixed_allocations:            # Pre-determined allocations
      - entity_id: string
        amount: float
    exclusions: list[string]      # Entities to exclude
    
  # Optimization parameters
  optimization:
    objective: enum               # "maximize_roi" | "maximize_reach" | "balance"
    time_horizon: string          # "Q1" | "Q2" | "H1" | "FY"
    brand: string                 # Brand context
    
  # Optional inputs from other agents
  agent_inputs:
    gap_analysis: object          # From Gap Analyzer
    cate_estimates: object        # From Heterogeneous Optimizer
    predictions: object           # From Prediction Synthesizer
```

### Output Contract

```yaml
# resource_optimizer_output.yaml
optimization_response:
  request_id: string              # Echo from input
  timestamp: datetime
  processing_time_ms: int
  
  # Allocation results
  resource_allocation:
    - entity_id: string
      entity_type: string
      
      # Allocation
      allocated_amount: float
      allocation_percentage: float  # Of total
      
      # Expected impact
      expected_roi: float
      expected_lift: float          # Predicted improvement
      confidence: float
      
      # Ranking
      priority_rank: int
      priority_tier: enum           # "high" | "medium" | "low"
      
      # Rationale
      allocation_rationale: string  # Brief explanation
      
  # Summary statistics
  allocation_summary:
    total_allocated: float
    total_unallocated: float        # If constraints prevent full allocation
    entities_allocated: int
    entities_excluded: int
    
    # Distribution
    high_priority_share: float      # % to high priority
    medium_priority_share: float
    low_priority_share: float
    
  # Expected outcomes
  expected_outcomes:
    total_expected_roi: float
    total_expected_lift: float
    roi_confidence_interval:
      lower: float
      upper: float
      
  # Optimization metadata
  optimization_metadata:
    solver_used: string             # "scipy" | "cvxpy" | "or-tools"
    iterations: int
    convergence_achieved: bool
    objective_value: float
    
  # Constraint satisfaction
  constraint_satisfaction:
    all_constraints_met: bool
    violated_constraints: list[string]
    relaxed_constraints: list[string]
    
  # Status
  status: enum                      # "optimal" | "feasible" | "infeasible" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys (Contract 2 Compliance)

```python
REQUIRED_KEYS = ["resource_allocation"]
```

### Validation Rules

1. **Budget Balance**: Sum of allocations ≤ total_amount
2. **Constraint Compliance**: All hard constraints must be satisfied
3. **Non-Negative**: All allocations must be ≥ 0
4. **Entity Coverage**: All non-excluded entities must have allocation (even if 0)
5. **ROI Validity**: Expected ROI must be based on documented assumptions

---

## Inter-Agent Communication

### Tier 4 → Orchestrator Handoff

```yaml
# tier4_handoff.yaml
handoff:
  source_agent: string            # "prediction_synthesizer" | "resource_optimizer"
  source_tier: 4
  
  # Results summary
  summary:
    primary_result: string        # One-line summary
    confidence: float
    key_findings: list[string]
    
  # Full results
  analysis_results:
    # Agent-specific keys as defined above
    
  # Narrative
  narrative: string               # Natural language explanation
  
  # Metadata
  processing_time_ms: int
  models_used: list[string]       # For prediction_synthesizer
  solver_used: string             # For resource_optimizer
```

### Tier 4 ← Tier 2/3 Dependencies

```yaml
# tier4_dependencies.yaml
prediction_synthesizer:
  optional_inputs:
    - source: heterogeneous_optimizer
      data: cate_estimates
      usage: "Segment-specific prediction adjustments"
    - source: drift_monitor
      data: drift_metrics
      usage: "Model validity check before prediction"
      
resource_optimizer:
  optional_inputs:
    - source: gap_analyzer
      data: gaps, roi_estimates
      usage: "Priority scoring for allocation"
    - source: heterogeneous_optimizer
      data: cate_by_segment
      usage: "Segment-specific ROI estimation"
    - source: prediction_synthesizer
      data: predictions
      usage: "Entity-level expected response"
```

---

## Error Handling

### Prediction Synthesizer Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `PS_001` | Insufficient models available | Return partial with warning |
| `PS_002` | Model version mismatch | Use fallback model versions |
| `PS_003` | Feature data missing | Impute or exclude entity |
| `PS_004` | Calibration failed | Return uncalibrated with warning |

### Resource Optimizer Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `RO_001` | Infeasible constraints | Relax constraints and retry |
| `RO_002` | Solver timeout | Return best feasible solution |
| `RO_003` | Missing entity data | Exclude entity with warning |
| `RO_004` | Invalid ROI estimates | Use conservative defaults |

---

---

## DSPy Role Specifications

### Overview

Tier 4 agents have mixed DSPy roles:
- **Sender**: prediction_synthesizer (generates training signals)
- **Recipient**: resource_optimizer (receives optimized prompts)

---

## Prediction Synthesizer - DSPy Sender Role

### Training Signal Contract

```python
@dataclass
class PredictionSynthesisTrainingSignal:
    """Training signal for prediction synthesis optimization."""

    # Identity
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Input Context
    model_count: int = 0
    prediction_type: str = ""
    target_entity: str = ""
    time_horizon: str = ""

    # Synthesis Output
    ensemble_method: str = ""
    predictions_generated: int = 0
    confidence_intervals_computed: bool = False
    uncertainty_quantified: bool = False

    # Quality Metrics
    model_agreement: float = 0.0
    prediction_confidence: float = 0.0
    interpretation_quality: float = 0.0

    # Outcome
    total_latency_ms: float = 0.0
    prediction_accurate: Optional[bool] = None
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - synthesis_completeness: 0.25 (all models, CI, uncertainty)
        - model_agreement: 0.20 (ensemble consensus)
        - confidence_calibration: 0.20 (confidence vs accuracy)
        - efficiency: 0.15 (latency)
        - accuracy: 0.20 (if available)
        """
        ...
```

### DSPy Signatures

```python
class PredictionSynthesisSignature(dspy.Signature):
    """Synthesize predictions from multiple models."""

    model_outputs: str = dspy.InputField(desc="Individual model predictions")
    model_weights: str = dspy.InputField(desc="Model weight/importance scores")
    ensemble_method: str = dspy.InputField(desc="Ensemble method to use")

    synthesized_prediction: float = dspy.OutputField(desc="Combined prediction value")
    confidence_interval: tuple = dspy.OutputField(desc="95% confidence interval")
    model_contributions: dict = dspy.OutputField(desc="Each model's contribution")

class PredictionInterpretationSignature(dspy.Signature):
    """Generate interpretation of synthesized prediction."""

    prediction: str = dspy.InputField(desc="Synthesized prediction result")
    model_details: str = dspy.InputField(desc="Model information and features")
    business_context: str = dspy.InputField(desc="Business context for interpretation")

    interpretation: str = dspy.OutputField(desc="Human-readable interpretation")
    key_drivers: list = dspy.OutputField(desc="Key factors driving prediction")
    caveats: list = dspy.OutputField(desc="Important caveats and limitations")

class UncertaintyQuantificationSignature(dspy.Signature):
    """Quantify prediction uncertainty."""

    predictions: str = dspy.InputField(desc="Model predictions")
    model_variances: str = dspy.InputField(desc="Individual model uncertainties")
    data_quality: str = dspy.InputField(desc="Data quality indicators")

    total_uncertainty: float = dspy.OutputField(desc="Total uncertainty score")
    epistemic_uncertainty: float = dspy.OutputField(desc="Model uncertainty")
    aleatoric_uncertainty: float = dspy.OutputField(desc="Data uncertainty")
    uncertainty_sources: list = dspy.OutputField(desc="Main uncertainty sources")
```

### Signal Collector Contract

```python
class PredictionSynthesizerSignalCollector:
    """Signal collector for Prediction Synthesizer agent."""

    dspy_type: Literal["sender"] = "sender"

    def collect_synthesis_signal(
        self,
        session_id: str,
        query: str,
        model_count: int,
        prediction_type: str,
        target_entity: str,
        time_horizon: str,
    ) -> PredictionSynthesisTrainingSignal: ...

    def update_synthesis_output(
        self,
        signal: PredictionSynthesisTrainingSignal,
        ensemble_method: str,
        predictions_generated: int,
        confidence_intervals_computed: bool,
        uncertainty_quantified: bool,
    ) -> PredictionSynthesisTrainingSignal: ...

    def update_quality_metrics(
        self,
        signal: PredictionSynthesisTrainingSignal,
        model_agreement: float,
        prediction_confidence: float,
        interpretation_quality: float,
        total_latency_ms: float,
    ) -> PredictionSynthesisTrainingSignal: ...

    def update_with_outcome(
        self,
        signal: PredictionSynthesisTrainingSignal,
        prediction_accurate: bool,
        user_satisfaction: Optional[float],
    ) -> PredictionSynthesisTrainingSignal: ...

    def get_signals_for_training(self, min_reward: float = 0.0, limit: int = 50) -> List[Dict]: ...
    def clear_buffer(self) -> None: ...
```

---

## Resource Optimizer - DSPy Recipient Role

### Overview

Resource Optimizer is a **Recipient** agent that receives optimized prompts from feedback_learner
but does not generate training signals for optimization.

### DSPy Signatures

```python
class OptimizationSummarySignature(dspy.Signature):
    """Generate optimization summary."""

    optimization_results: str = dspy.InputField(desc="Optimization output")
    constraints: str = dspy.InputField(desc="Applied constraints")
    objectives: str = dspy.InputField(desc="Optimization objectives")

    summary: str = dspy.OutputField(desc="Executive summary")
    key_recommendations: list = dspy.OutputField(desc="Top recommendations")
    trade_offs: list = dspy.OutputField(desc="Key trade-offs made")

class AllocationRecommendationSignature(dspy.Signature):
    """Generate resource allocation recommendations."""

    current_allocation: str = dspy.InputField(desc="Current resource distribution")
    optimization_output: str = dspy.InputField(desc="Optimization results")
    business_priorities: str = dspy.InputField(desc="Business priority context")

    recommendations: list = dspy.OutputField(desc="Ordered allocation recommendations")
    expected_impact: dict = dspy.OutputField(desc="Expected impact per change")
    implementation_steps: list = dspy.OutputField(desc="Implementation guidance")

class ScenarioNarrativeSignature(dspy.Signature):
    """Generate narrative for optimization scenario."""

    scenario_name: str = dspy.InputField(desc="Scenario identifier")
    scenario_results: str = dspy.InputField(desc="Scenario optimization results")
    comparison_baseline: str = dspy.InputField(desc="Baseline for comparison")

    narrative: str = dspy.OutputField(desc="Scenario narrative explanation")
    pros: list = dspy.OutputField(desc="Scenario advantages")
    cons: list = dspy.OutputField(desc="Scenario disadvantages")
    recommendation: str = dspy.OutputField(desc="Scenario recommendation")
```

### Recipient Configuration

```python
class ResourceOptimizerRecipient:
    """DSPy Recipient for Resource Optimizer agent."""

    dspy_type: Literal["recipient"] = "recipient"

    # Prompt optimization settings
    prompt_refresh_interval_hours: int = 24

    # Signatures that can receive optimized prompts
    optimizable_signatures: List[str] = [
        "OptimizationSummarySignature",
        "AllocationRecommendationSignature",
        "ScenarioNarrativeSignature",
    ]

    def apply_optimized_prompt(
        self,
        signature_name: str,
        optimized_prompt: str,
        version: str,
    ) -> bool: ...

    def get_current_prompt_version(self, signature_name: str) -> str: ...

    def report_prompt_performance(
        self,
        signature_name: str,
        success_rate: float,
        latency_ms: float,
    ) -> None: ...
```

---

## Signal Flow

### Tier 4 → Feedback Learner Flow

```
prediction_synthesizer ──► feedback_learner
                                  │
                                  ▼
                            Optimization
                                  │
                                  ▼
                        resource_optimizer (receives optimized prompts)
```

---

## Validation Tests

```bash
# Prediction Synthesizer contracts
pytest tests/integration/test_tier4_contracts.py::test_prediction_synthesizer_input
pytest tests/integration/test_tier4_contracts.py::test_prediction_synthesizer_output

# Resource Optimizer contracts
pytest tests/integration/test_tier4_contracts.py::test_resource_optimizer_input
pytest tests/integration/test_tier4_contracts.py::test_resource_optimizer_output

# Inter-agent communication
pytest tests/integration/test_tier4_contracts.py::test_tier4_handoff
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-23 | V2: Added DSPy Role specifications for Tier 4 agents |
| 2025-12-08 | Initial creation |
