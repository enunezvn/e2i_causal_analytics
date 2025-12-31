# Tier 3: Design & Monitoring Contracts

## Overview

Tier 3 contains agents for experiment design and system monitoring: Experiment Designer, Drift Monitor, and Health Score.

---

## Experiment Designer Agent Contracts

### Input Contract

```python
# src/contracts/experiment_designer.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .base import BaseAgentInput

class ExperimentDesignerInput(BaseAgentInput):
    """Input contract for Experiment Designer Agent"""
    
    # Required
    hypothesis: str = Field(..., description="Hypothesis to test")
    treatment_description: str = Field(..., description="Description of the intervention")
    primary_outcome: str = Field(..., description="Primary outcome metric")
    
    # Optional outcomes
    secondary_outcomes: List[str] = []
    
    # Constraints
    max_sample_size: Optional[int] = None
    max_duration_weeks: Optional[int] = None
    budget_constraint: Optional[float] = None
    
    # Design preferences
    design_type_preference: Optional[Literal["rct", "cluster_rct", "quasi"]] = None
    
    # Power analysis parameters
    expected_effect_size: Optional[float] = None
    significance_level: float = 0.05
    power: float = 0.8
    
    # Context
    similar_past_experiments: Optional[List[str]] = None
    known_confounders: Optional[List[str]] = None
```

### Output Contract

```python
class TreatmentDefinition(BaseModel):
    """Definition of experimental treatment"""
    name: str
    description: str
    dosage_or_intensity: Optional[str]
    duration: str
    delivery_mechanism: str

class OutcomeDefinition(BaseModel):
    """Definition of outcome measurement"""
    name: str
    metric_type: Literal["continuous", "binary", "count", "time_to_event"]
    measurement_timing: str
    data_source: str

class ValidityThreat(BaseModel):
    """Identified threat to validity"""
    threat_type: Literal["internal", "external", "statistical"]
    threat_name: str
    description: str
    severity: Literal["low", "medium", "high"]
    mitigation: str
    mitigation_difficulty: Literal["low", "medium", "high"]

class ExperimentDesignerOutput(BaseAgentOutput):
    """Output contract for Experiment Designer Agent"""

    # Design specification
    design_type: Literal["rct", "cluster_rct", "regression_discontinuity", "diff_in_diff", "synthetic_control"]

    # Refined definitions
    refined_hypothesis: str
    treatment_definition: TreatmentDefinition
    outcome_definitions: List[OutcomeDefinition]

    # Sample size and power
    required_sample_size: int
    estimated_power: float
    estimated_duration_weeks: int

    # Validity
    validity_threats: List[ValidityThreat]
    overall_validity_score: Literal["strong", "moderate", "weak"]

    # Implementation
    randomization_strategy: str
    analysis_plan: str

    # Generated artifacts
    dag_specification: Dict[str, Any]
    analysis_code_template: str
    preregistration_document: str

    # V4.4: DAG-Aware Validity Validation
    dag_confounders_validated: Optional[List[str]] = Field(
        None, description="Confounders validated against discovered DAG"
    )
    dag_missing_confounders: Optional[List[str]] = Field(
        None, description="Assumed confounders NOT found in discovered DAG"
    )
    dag_latent_confounders: Optional[List[str]] = Field(
        None, description="Latent confounders detected from FCI bidirected edges"
    )
    dag_instrument_candidates: Optional[List[str]] = Field(
        None, description="Valid IV candidates identified from DAG structure"
    )
    dag_effect_modifiers: Optional[List[str]] = Field(
        None, description="Effect modifiers identified from DAG"
    )
    dag_validation_warnings: Optional[List[str]] = Field(
        None, description="Warnings from DAG-aware validity validation"
    )
```

### Handoff Format

```yaml
experiment_designer_handoff:
  agent: experiment_designer
  analysis_type: experiment_design

  key_findings:
    - design_type: <type>
    - required_sample_size: <n>
    - duration_weeks: <n>
    - validity_score: strong|moderate|weak

  design_summary:
    hypothesis: <refined hypothesis>
    treatment: <treatment description>
    outcome: <primary outcome>
    randomization: <strategy>

  validity_assessment:
    threats_identified: <count>
    mitigations_proposed: <count>
    proceed_recommendation: proceed|proceed_with_caution|redesign_needed

  # V4.4: DAG-Aware Validity Validation (if discovered DAG available)
  dag_validation:
    enabled: <bool>
    confounders_validated: <count>
    missing_confounders: [<list>]
    latent_confounders: [<list>]
    instrument_candidates: [<list>]
    effect_modifiers: [<list>]
    warnings: [<list>]

  artifacts:
    dag: <available>
    analysis_code: <available>
    preregistration: <available>

  requires_further_analysis: <bool>
  suggested_next_agent: causal_impact
```

---

## Drift Monitor Agent Contracts

### Input Contract

```python
# src/contracts/drift_monitor.py

class DriftMonitorInput(BaseAgentInput):
    """Input contract for Drift Monitor Agent"""
    
    # What to monitor
    model_id: Optional[str] = None
    features_to_monitor: List[str] = Field(..., description="Features to check for drift")
    
    # Time configuration
    time_window: str = Field("7d", description="Window for current data")
    
    # Detection configuration
    check_data_drift: bool = True
    check_model_drift: bool = True
    check_concept_drift: bool = True
    
    # Thresholds
    significance_level: float = 0.05
    psi_threshold: float = 0.1
```

### Output Contract

```python
class DriftResult(BaseModel):
    """Individual drift detection result"""
    feature: str
    drift_type: Literal["data", "model", "concept"]
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: Literal["none", "low", "medium", "high", "critical"]

class DriftAlert(BaseModel):
    """Drift alert for notification"""
    alert_id: str
    severity: Literal["warning", "critical"]
    drift_type: str
    affected_features: List[str]
    message: str
    recommended_action: str

class DriftMonitorOutput(BaseAgentOutput):
    """Output contract for Drift Monitor Agent"""
    
    # Detection results
    data_drift_results: List[DriftResult]
    model_drift_results: List[DriftResult]
    concept_drift_results: List[DriftResult]
    
    # Aggregated
    overall_drift_score: float  # 0-1
    features_with_drift: List[str]
    
    # Alerts
    alerts: List[DriftAlert]
    
    # Summary
    drift_summary: str
    recommended_actions: List[str]
    
    # Metadata
    features_checked: int
    detection_latency_ms: int
```

### Handoff Format

```yaml
drift_monitor_handoff:
  agent: drift_monitor
  analysis_type: drift_detection
  
  key_findings:
    - drift_score: <0-1>
    - features_with_drift: <count>
    - critical_alerts: <count>
    - warning_alerts: <count>
  
  drift_summary:
    data_drift: <count> features
    model_drift: <status>
    concept_drift: <status>
  
  alerts:
    - severity: critical
      type: data
      features: [<feature 1>]
      action: <recommended action>
  
  requires_further_analysis: <bool>
  suggested_next_agent: health_score|experiment_designer
```

---

## Health Score Agent Contracts

### Input Contract

```python
# src/contracts/health_score.py

class HealthScoreInput(BaseAgentInput):
    """Input contract for Health Score Agent"""
    
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"] = "full"
```

### Output Contract

```python
class ComponentStatus(BaseModel):
    """Status of a system component"""
    component_name: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    latency_ms: Optional[int]
    error_message: Optional[str]

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    accuracy: Optional[float]
    auc_roc: Optional[float]
    prediction_latency_p99_ms: Optional[int]
    error_rate: float
    status: Literal["healthy", "degraded", "unhealthy"]

class HealthScoreOutput(BaseAgentOutput):
    """Output contract for Health Score Agent"""
    
    # Overall
    overall_health_score: float  # 0-100
    health_grade: Literal["A", "B", "C", "D", "F"]
    
    # Component scores
    component_health_score: float
    model_health_score: float
    pipeline_health_score: float
    agent_health_score: float
    
    # Details
    component_statuses: List[ComponentStatus]
    model_metrics: List[ModelMetrics]
    
    # Issues
    critical_issues: List[str]
    warnings: List[str]
    
    # Summary
    health_summary: str
    
    # Metadata
    check_latency_ms: int
```

### Handoff Format

```yaml
health_score_handoff:
  agent: health_score
  analysis_type: system_health
  
  key_findings:
    - overall_score: <0-100>
    - grade: A|B|C|D|F
    - critical_issues: <count>
  
  component_scores:
    component: <0-1>
    model: <0-1>
    pipeline: <0-1>
    agent: <0-1>
  
  issues:
    critical:
      - <issue 1>
    warnings:
      - <warning 1>
  
  requires_further_analysis: <bool>
  suggested_next_agent: drift_monitor
```

---

## Cross-Tier Communication

### Tier 3 → Tier 2 Patterns

```yaml
# Pattern: Design → Execute → Analyze

experiment_lifecycle:
  phase_1_design:
    agent: experiment_designer
    output: design_spec, dag, analysis_code
    
  phase_2_execute:
    # External: Run experiment
    duration: <weeks>
    
  phase_3_analyze:
    agent: causal_impact
    input:
      - treatment: from design_spec
      - outcome: from design_spec
      - dag: from experiment_designer
```

### Monitoring → Action Patterns

```yaml
# Pattern: Monitor → Detect → Respond

monitoring_response:
  continuous_monitoring:
    agents:
      - health_score  # Every 5 minutes
      - drift_monitor  # Every hour

  on_critical_alert:
    actions:
      - notify: team
      - trigger: experiment_designer  # If retraining needed
      - trigger: feedback_learner  # Log for learning
```

---

## DSPy Role Specifications

### Overview

Tier 3 agents have mixed DSPy roles:
- **Sender**: drift_monitor, experiment_designer (generate training signals)
- **Recipient**: health_score (receives optimized prompts)

---

## Drift Monitor - DSPy Sender Role

### Training Signal Contract

```python
@dataclass
class DriftDetectionTrainingSignal:
    """Training signal for drift detection optimization."""

    # Identity
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Configuration
    model_id: str = ""
    features_monitored: int = 0
    time_window: str = ""
    check_data_drift: bool = True
    check_model_drift: bool = False
    check_concept_drift: bool = False
    psi_threshold: float = 0.1
    significance_level: float = 0.05

    # Detection Results
    data_drift_count: int = 0
    model_drift_count: int = 0
    concept_drift_count: int = 0
    overall_drift_score: float = 0.0
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    features_checked: int = 0

    # Alerting
    alerts_generated: int = 0
    critical_alerts: int = 0
    warnings: int = 0
    recommended_actions_count: int = 0

    # Outcome
    total_latency_ms: float = 0.0
    drift_correctly_identified: Optional[bool] = None
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - drift_detection_rate: 0.25 (ideal: 5-20% of features)
        - alerting_quality: 0.25 (critical alert ratio)
        - efficiency: 0.20 (latency per feature)
        - actionability: 0.15 (recommendations per alert)
        - validation: 0.15 (correct identification)
        """
        ...
```

### DSPy Signatures

```python
class DriftDetectionSignature(dspy.Signature):
    """Analyze feature distributions for drift."""

    feature_name: str = dspy.InputField(desc="Feature being analyzed")
    reference_stats: str = dspy.InputField(desc="Reference distribution statistics")
    current_stats: str = dspy.InputField(desc="Current distribution statistics")
    threshold: float = dspy.InputField(desc="Drift detection threshold")

    has_drift: bool = dspy.OutputField(desc="Whether significant drift detected")
    drift_score: float = dspy.OutputField(desc="Drift magnitude score (0-1)")
    severity: str = dspy.OutputField(desc="Severity: low, medium, high, critical")
    interpretation: str = dspy.OutputField(desc="Human-readable interpretation")

class HopDecisionSignature(dspy.Signature):
    """Decide whether to perform detailed analysis hop."""

    initial_results: str = dspy.InputField(desc="Initial drift scan results")
    drift_count: int = dspy.InputField(desc="Number of features with detected drift")
    severity_max: str = dspy.InputField(desc="Maximum severity detected")

    needs_hop: bool = dspy.OutputField(desc="Whether to perform deeper analysis")
    hop_reason: str = dspy.OutputField(desc="Reason for hop decision")
    priority_features: list = dspy.OutputField(desc="Features to analyze in depth")

class DriftInterpretationSignature(dspy.Signature):
    """Generate actionable drift interpretation."""

    drift_results: str = dspy.InputField(desc="Complete drift analysis results")
    model_context: str = dspy.InputField(desc="Model information and history")
    business_context: str = dspy.InputField(desc="Business impact context")

    summary: str = dspy.OutputField(desc="Executive summary of drift status")
    root_causes: list = dspy.OutputField(desc="Likely root causes of drift")
    recommendations: list = dspy.OutputField(desc="Prioritized action recommendations")
    urgency: str = dspy.OutputField(desc="Action urgency: immediate, soon, monitor")
```

### Signal Collector Contract

```python
class DriftMonitorSignalCollector:
    """Signal collector for Drift Monitor agent."""

    dspy_type: Literal["sender"] = "sender"

    def collect_detection_signal(
        self,
        session_id: str,
        query: str,
        model_id: str,
        features_monitored: int,
        time_window: str,
        check_data_drift: bool,
        check_model_drift: bool,
        check_concept_drift: bool,
    ) -> DriftDetectionTrainingSignal: ...

    def update_detection_results(
        self,
        signal: DriftDetectionTrainingSignal,
        data_drift_count: int,
        model_drift_count: int,
        concept_drift_count: int,
        overall_drift_score: float,
        severity_distribution: Dict[str, int],
        features_checked: int,
    ) -> DriftDetectionTrainingSignal: ...

    def update_alerting(
        self,
        signal: DriftDetectionTrainingSignal,
        alerts_generated: int,
        critical_alerts: int,
        warnings: int,
        recommended_actions_count: int,
        total_latency_ms: float,
    ) -> DriftDetectionTrainingSignal: ...

    def update_with_validation(
        self,
        signal: DriftDetectionTrainingSignal,
        drift_correctly_identified: bool,
        user_satisfaction: Optional[float] = None,
    ) -> DriftDetectionTrainingSignal: ...

    def get_signals_for_training(self, min_reward: float = 0.0, limit: int = 50) -> List[Dict]: ...
    def get_validated_examples(self, limit: int = 10) -> List[Dict]: ...
    def clear_buffer(self) -> None: ...
```

---

## Experiment Designer - DSPy Sender Role

### Training Signal Contract

```python
@dataclass
class ExperimentDesignTrainingSignal:
    """Training signal for experiment design optimization."""

    # Identity
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Input Context
    hypothesis: str = ""
    treatment_type: str = ""
    outcome_metric: str = ""
    constraints_count: int = 0

    # Design Output
    design_type: str = ""
    sample_size: int = 0
    duration_weeks: int = 0
    power_achieved: float = 0.0
    effect_size_target: float = 0.0

    # Investigation Quality
    past_experiments_found: int = 0
    confounders_identified: int = 0
    validity_threats_count: int = 0

    # V4.4: DAG-Aware Validity Validation Phase
    dag_validation_enabled: bool = False
    dag_confounders_validated: int = 0
    dag_missing_confounders: int = 0
    dag_latent_confounders: int = 0
    dag_instrument_candidates: int = 0
    dag_effect_modifiers: int = 0
    dag_gate_decision: str = ""  # accept, review, reject, augment
    dag_validation_warnings: int = 0

    # Digital Twin Integration
    used_digital_twin: bool = False
    simulation_iterations: int = 0
    predicted_success_rate: float = 0.0

    # Outcome
    total_latency_ms: float = 0.0
    design_implemented: Optional[bool] = None
    experiment_success: Optional[bool] = None
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting (V4.4 Updated):
        - power_achievement: 0.20 (power >= 0.8)
        - validity_coverage: 0.20 (threats identified)
        - dag_validation_quality: 0.10 (confounders validated ratio)
        - investigation_depth: 0.15 (past experiments, confounders)
        - efficiency: 0.15 (latency)
        - outcome_success: 0.20 (if available)
        """
        ...
```

### DSPy Signatures

```python
class DesignReasoningSignature(dspy.Signature):
    """Reason about optimal experiment design."""

    hypothesis: str = dspy.InputField(desc="Hypothesis to test")
    treatment: str = dspy.InputField(desc="Treatment description")
    outcome: str = dspy.InputField(desc="Primary outcome metric")
    constraints: str = dspy.InputField(desc="Design constraints")

    design_type: str = dspy.OutputField(desc="Recommended design type")
    sample_size: int = dspy.OutputField(desc="Required sample size")
    duration: str = dspy.OutputField(desc="Recommended duration")
    rationale: str = dspy.OutputField(desc="Design rationale")

class InvestigationPlanSignature(dspy.Signature):
    """Plan investigation for experiment design."""

    hypothesis: str = dspy.InputField(desc="Hypothesis being tested")
    domain: str = dspy.InputField(desc="Domain context")

    investigation_steps: list = dspy.OutputField(desc="Ordered investigation steps")
    queries_needed: list = dspy.OutputField(desc="Database/memory queries")
    hop_recommendation: bool = dspy.OutputField(desc="Whether deeper analysis needed")

class ValidityAssessmentSignature(dspy.Signature):
    """Assess validity threats for experiment."""

    design_spec: str = dspy.InputField(desc="Proposed experiment design")
    context: str = dspy.InputField(desc="Domain and operational context")

    internal_threats: list = dspy.OutputField(desc="Internal validity threats")
    external_threats: list = dspy.OutputField(desc="External validity threats")
    mitigations: list = dspy.OutputField(desc="Recommended mitigations")
    overall_validity: float = dspy.OutputField(desc="Validity score (0-1)")
```

### Signal Collector Contract

```python
class ExperimentDesignerSignalCollector:
    """Signal collector for Experiment Designer agent."""

    dspy_type: Literal["sender"] = "sender"

    def collect_design_signal(
        self,
        session_id: str,
        query: str,
        hypothesis: str,
        treatment_type: str,
        outcome_metric: str,
        constraints_count: int,
    ) -> ExperimentDesignTrainingSignal: ...

    def update_design_output(
        self,
        signal: ExperimentDesignTrainingSignal,
        design_type: str,
        sample_size: int,
        duration_weeks: int,
        power_achieved: float,
        effect_size_target: float,
    ) -> ExperimentDesignTrainingSignal: ...

    def update_investigation(
        self,
        signal: ExperimentDesignTrainingSignal,
        past_experiments_found: int,
        confounders_identified: int,
        validity_threats_count: int,
        used_digital_twin: bool,
        simulation_iterations: int,
        predicted_success_rate: float,
    ) -> ExperimentDesignTrainingSignal: ...

    def finalize_signal(
        self,
        signal: ExperimentDesignTrainingSignal,
        total_latency_ms: float,
    ) -> ExperimentDesignTrainingSignal: ...

    def update_with_outcome(
        self,
        signal: ExperimentDesignTrainingSignal,
        design_implemented: bool,
        experiment_success: Optional[bool],
        user_satisfaction: Optional[float],
    ) -> ExperimentDesignTrainingSignal: ...

    def get_signals_for_training(self, min_reward: float = 0.0, limit: int = 50) -> List[Dict]: ...
    def clear_buffer(self) -> None: ...
```

---

## Health Score - DSPy Recipient Role

### Overview

Health Score is a **Recipient** agent that receives optimized prompts from feedback_learner
but does not generate training signals for optimization.

### DSPy Signatures

```python
class HealthSummarySignature(dspy.Signature):
    """Generate health summary from metrics."""

    system_metrics: str = dspy.InputField(desc="Current system metrics")
    historical_baseline: str = dspy.InputField(desc="Historical baseline values")
    thresholds: str = dspy.InputField(desc="Alert thresholds")

    overall_health: float = dspy.OutputField(desc="Overall health score (0-100)")
    status: str = dspy.OutputField(desc="Status: healthy, degraded, critical")
    summary: str = dspy.OutputField(desc="Executive health summary")
    concerns: list = dspy.OutputField(desc="Areas of concern")

class HealthRecommendationSignature(dspy.Signature):
    """Generate health recommendations."""

    health_status: str = dspy.InputField(desc="Current health status")
    degraded_components: str = dspy.InputField(desc="Components showing issues")
    resource_availability: str = dspy.InputField(desc="Available resources")

    recommendations: list = dspy.OutputField(desc="Prioritized recommendations")
    immediate_actions: list = dspy.OutputField(desc="Actions needed immediately")
    monitoring_focus: list = dspy.OutputField(desc="Areas to monitor closely")
```

### Recipient Configuration

```python
class HealthScoreRecipient:
    """DSPy Recipient for Health Score agent."""

    dspy_type: Literal["recipient"] = "recipient"

    # Prompt optimization settings
    prompt_refresh_interval_hours: int = 24

    # Signatures that can receive optimized prompts
    optimizable_signatures: List[str] = [
        "HealthSummarySignature",
        "HealthRecommendationSignature",
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

### Tier 3 → Feedback Learner Flow

```
drift_monitor         ──┐
                       ├──► feedback_learner
experiment_designer   ──┘         │
                                  ▼
                            Optimization
                                  │
                                  ▼
                            health_score (receives optimized prompts)
```

### Signal Lifecycle

1. **Collection**: Sender agents collect signals at decision time
2. **Update**: Signals updated with execution outcomes
3. **Buffer**: Signals buffered (max 100 per agent)
4. **Transfer**: High-quality signals sent to feedback_learner
5. **Optimization**: feedback_learner runs MIPROv2 optimization
6. **Distribution**: Optimized prompts distributed to recipient agents

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-31 | V4.4: Added DAG-Aware Validity Validation for Experiment Designer |
| 2025-12-23 | V2: Added DSPy Role specifications for all Tier 3 agents |
| 2025-12-08 | V1: Initial creation |
