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
