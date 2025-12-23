# Tier 3 Contracts: Design & Monitoring Agents

**Version**: 1.0
**Last Updated**: 2025-12-18
**Status**: Active

## Overview

This document defines integration contracts for **Tier 3: Design & Monitoring** agents in the E2I Causal Analytics platform. These agents handle experiment design, drift detection, and system health monitoring.

### Tier 3 Agents

| Agent | Type | Responsibility | Primary Methods |
|-------|------|----------------|-----------------|
| **Experiment Designer** | Hybrid | A/B test and experiment design | Deep reasoning (Claude Sonnet/Opus) + Power analysis |
| **Drift Monitor** | Standard (Fast) | Data/model/concept drift detection | Statistical tests, PSI, KS test |
| **Health Score** | Standard (Fast) | System health monitoring | Metric aggregation, no LLM |

---

## 1. Shared Types

### 1.1 Common Enums

```python
from typing import Literal

# Design types
DesignType = Literal["rct", "cluster_rct", "quasi_did", "quasi_rdd", "quasi_iv"]

# Validity assessment
ValidityScore = Literal["strong", "moderate", "weak"]
ProceedRecommendation = Literal["proceed", "proceed_with_caution", "redesign_needed"]

# Drift severity
DriftSeverity = Literal["none", "low", "medium", "high", "critical"]
DriftType = Literal["data", "model", "concept"]

# Health statuses
HealthStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]
HealthGrade = Literal["A", "B", "C", "D", "F"]
CheckScope = Literal["full", "quick", "models", "pipelines", "agents"]
```

### 1.2 Common Input Fields

All Tier 3 agents accept these common fields:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Tier3CommonInput(BaseModel):
    """Common input fields for all Tier 3 agents"""
    query: str = Field(..., description="User's natural language query")
```

### 1.3 Common Output Fields

All Tier 3 agents return these common fields:

```python
from typing import List

class Tier3CommonOutput(BaseModel):
    """Common output fields for all Tier 3 agents"""
    total_latency_ms: int = Field(..., description="Total processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    timestamp: str = Field(..., description="ISO 8601 timestamp of completion")
```

---

## 2. Experiment Designer Agent

**Agent Type**: Hybrid (Deep Reasoning + Computation)
**Primary Models**: Claude Sonnet 4, Claude Opus 4.5 (fallback to Haiku)
**Latency**: Up to 60s

### 2.1 Input Contract

```python
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ExperimentDesignerInput(BaseModel):
    """Input contract for Experiment Designer Agent"""

    # Required fields
    business_question: str = Field(..., description="The business question the experiment should answer")
    constraints: Dict[str, Any] = Field(
        ...,
        description="Constraints: budget, timeline, ethical, operational"
    )
    available_data: Dict[str, Any] = Field(
        ...,
        description="Available variables and data sources"
    )

    # Optional configuration
    preregistration_formality: Literal["light", "medium", "heavy"] = Field(
        "medium",
        description="Level of detail for pre-registration document"
    )
    max_redesign_iterations: int = Field(2, ge=0, le=5, description="Maximum redesign loops")
    enable_validity_audit: bool = Field(True, description="Whether to run validity audit")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "business_question": "Does increasing HCP engagement frequency cause higher TRx?",
                "constraints": {
                    "budget": 500000,
                    "timeline_weeks": 12,
                    "ethical_constraints": ["No patient data", "HCP consent required"],
                    "expected_effect_size": 0.3,
                    "power": 0.80,
                    "alpha": 0.05
                },
                "available_data": {
                    "variables": ["hcp_engagement_frequency", "trx_total", "hcp_specialty", "region"],
                    "sample_size_available": 5000
                },
                "preregistration_formality": "medium"
            }
        }
    }
```

### 2.2 Output Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TreatmentDefinition(BaseModel):
    """Treatment arm specification"""
    name: str
    levels: List[str]
    operationalization: str
    dosage_description: Optional[str] = None

class OutcomeDefinition(BaseModel):
    """Outcome variable specification"""
    primary: str
    secondary: List[str]
    measurement_timing: str
    measurement_method: str
    type: Literal["continuous", "binary", "count", "time_to_event"]

class ValidityThreat(BaseModel):
    """Identified validity threat"""
    threat_type: Literal["selection_bias", "confounding", "measurement", "contamination", "temporal", "attrition"]
    description: str
    likelihood: Literal["low", "medium", "high"]
    severity: Literal["low", "medium", "high"]
    mitigation: str

class MitigationRecommendation(BaseModel):
    """Recommendation to address validity threat"""
    threat_addressed: str
    recommendation: str
    implementation_difficulty: Literal["low", "medium", "high"]

class ExperimentDesignerOutput(BaseModel):
    """Output contract for Experiment Designer Agent"""

    # Core design outputs
    refined_hypothesis: str = Field(..., description="Precise causal claim to test")
    treatment_definition: TreatmentDefinition = Field(..., description="Treatment specification")
    outcome_definition: OutcomeDefinition = Field(..., description="Outcome specification")
    design_type: DesignType = Field(..., description="Recommended experiment design type")
    design_rationale: str = Field(..., description="Why this design was chosen")

    # Design details
    stratification_vars: List[str] = Field(default_factory=list, description="Variables to stratify by")
    blocking_strategy: Optional[str] = Field(None, description="Blocking strategy if applicable")
    anticipated_confounders: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Expected confounders and how addressed"
    )

    # Power analysis
    required_sample_size: int = Field(..., description="Required sample size")
    achievable_mde: float = Field(..., description="Minimum detectable effect given constraints")
    power: float = Field(..., ge=0.5, le=0.99, description="Statistical power")
    estimated_duration_weeks: int = Field(..., description="Expected experiment duration")
    power_analysis_details: Dict[str, Any] = Field(..., description="Detailed power analysis")

    # Validity assessment
    internal_validity_threats: List[ValidityThreat] = Field(
        ...,
        description="Identified internal validity threats"
    )
    external_validity_limits: List[str] = Field(..., description="External validity limitations")
    statistical_concerns: List[str] = Field(..., description="Statistical concerns")
    mitigation_recommendations: List[MitigationRecommendation] = Field(
        ...,
        description="Recommended mitigations"
    )
    validity_score: ValidityScore = Field(..., description="Overall validity assessment")
    proceed_recommendation: ProceedRecommendation = Field(..., description="Proceed or redesign")

    # DoWhy integration
    causal_dag_spec: Dict[str, Any] = Field(..., description="DoWhy-compatible DAG specification")
    analysis_code_template: str = Field(..., description="Python code template for analysis")
    preregistration_doc: str = Field(..., description="Pre-registration document")

    # Metadata
    design_latency_ms: int
    power_latency_ms: int
    validity_latency_ms: int
    template_latency_ms: int
    total_latency_ms: int
    models_used: List[str] = Field(..., description="Models used for reasoning")
    redesign_iterations: int = Field(..., description="Number of redesign loops executed")
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 2.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class ExperimentDesignState(TypedDict):
    """Complete LangGraph state for Experiment Designer Agent"""

    # === INPUT ===
    business_question: str
    constraints: Dict[str, Any]
    available_data: Dict[str, Any]

    # === ORGANIZATIONAL CONTEXT ===
    similar_experiments: Optional[List[Dict]]
    organizational_defaults: Optional[Dict[str, Any]]
    recent_assumption_violations: Optional[List[Dict]]

    # === DESIGN OUTPUTS ===
    refined_hypothesis: Optional[str]
    treatment_definition: Optional[Dict[str, Any]]  # TreatmentDefinition
    outcome_definition: Optional[Dict[str, Any]]  # OutcomeDefinition
    design_type: Optional[Literal["rct", "cluster_rct", "quasi_did", "quasi_rdd", "quasi_iv"]]
    design_rationale: Optional[str]
    stratification_vars: Optional[List[str]]
    blocking_strategy: Optional[str]
    anticipated_confounders: Optional[List[Dict[str, str]]]

    # === POWER ANALYSIS OUTPUTS ===
    required_sample_size: Optional[int]
    achievable_mde: Optional[float]
    power: Optional[float]
    estimated_duration_weeks: Optional[int]
    power_analysis_details: Optional[Dict[str, Any]]

    # === VALIDITY AUDIT OUTPUTS ===
    internal_validity_threats: Optional[List[Dict[str, Any]]]  # ValidityThreat
    external_validity_limits: Optional[List[str]]
    statistical_concerns: Optional[List[str]]
    mitigation_recommendations: Optional[List[Dict[str, Any]]]  # MitigationRecommendation
    validity_score: Optional[Literal["strong", "moderate", "weak"]]
    proceed_recommendation: Optional[Literal["proceed", "proceed_with_caution", "redesign_needed"]]

    # === DOWHY INTEGRATION OUTPUTS ===
    causal_dag_spec: Optional[Dict[str, Any]]
    analysis_code_template: Optional[str]
    preregistration_doc: Optional[str]

    # === EXECUTION METADATA ===
    design_latency_ms: int
    power_latency_ms: int
    validity_latency_ms: int
    template_latency_ms: int
    total_latency_ms: int
    models_used: List[str]
    redesign_iterations: int
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    status: Literal["pending", "designing", "analyzing_power", "auditing", "redesigning", "generating", "completed", "failed"]
```

### 2.4 Handoff Format

```yaml
experiment_designer_handoff:
  agent: experiment_designer
  analysis_type: experiment_design
  status: completed

  design:
    hypothesis: "Increasing HCP engagement frequency causes higher TRx"
    type: cluster_rct
    rationale: "Territory-level randomization minimizes contamination"
    sample_size: 2400
    duration_weeks: 12
    power: 0.80
    mde: 0.25

  treatment:
    name: "hcp_engagement_frequency"
    levels: ["control", "2x", "3x"]
    operationalization: "Number of rep visits per month"

  outcome:
    primary: "trx_total"
    secondary: ["nrx", "market_share"]
    type: continuous
    measurement_timing: "End of 12 weeks"

  validity:
    score: strong
    proceed_recommendation: proceed
    key_threats:
      - type: contamination
        likelihood: low
        mitigation: "Territory-level randomization with buffer zones"
      - type: attrition
        likelihood: medium
        mitigation: "Engagement incentives, regular check-ins"

  outputs:
    dag_spec: available
    analysis_code: available
    preregistration: available

  recommendations:
    - "Proceed with cluster RCT design"
    - "Implement buffer zones between territories"
    - "Pre-register analysis plan before data collection"
    - "Use intent-to-treat analysis for primary outcome"

  warnings:
    - "Expected ICC of 0.05 may require larger sample if actual ICC higher"
    - "12-week duration may not capture long-term effects"

  requires_further_analysis: false
  suggested_next_agent: causal_impact
  suggested_reason: "Use for post-experiment analysis"
```

---

## 3. Drift Monitor Agent

**Agent Type**: Standard (Fast Path)
**Primary Methods**: Statistical tests (PSI, KS test, Chi-square)
**Latency**: Up to 10s

### 3.1 Input Contract

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class DriftMonitorInput(BaseModel):
    """Input contract for Drift Monitor Agent"""

    # Required fields
    query: str
    features_to_monitor: List[str] = Field(..., description="Features to check for drift")

    # Optional fields
    model_id: Optional[str] = Field(None, description="Model ID for model drift checks")
    time_window: str = Field("7d", description="Time window for comparison (e.g., '7d', '30d')")
    brand: Optional[str] = None

    # Configuration
    significance_level: float = Field(0.05, ge=0.01, le=0.10, description="Statistical significance level")
    psi_threshold: float = Field(0.1, ge=0.0, le=1.0, description="PSI warning threshold")
    check_data_drift: bool = Field(True, description="Whether to check data drift")
    check_model_drift: bool = Field(True, description="Whether to check model drift")
    check_concept_drift: bool = Field(True, description="Whether to check concept drift")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Check for drift in prediction models",
                "features_to_monitor": ["hcp_specialty", "patient_volume", "competitive_pressure"],
                "model_id": "trx_predictor_v2",
                "time_window": "7d",
                "significance_level": 0.05,
                "psi_threshold": 0.1
            }
        }
    }
```

### 3.2 Output Contract

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class DriftResult(BaseModel):
    """Individual drift detection result"""
    feature: str
    drift_type: DriftType
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: DriftSeverity
    baseline_period: str
    current_period: str

class DriftAlert(BaseModel):
    """Drift alert for notification"""
    alert_id: str
    severity: Literal["warning", "critical"]
    drift_type: DriftType
    affected_features: List[str]
    message: str
    recommended_action: str
    timestamp: str

class DriftMonitorOutput(BaseModel):
    """Output contract for Drift Monitor Agent"""

    # Detection results
    data_drift_results: List[DriftResult] = Field(..., description="Data drift detection results")
    model_drift_results: List[DriftResult] = Field(..., description="Model drift detection results")
    concept_drift_results: List[DriftResult] = Field(..., description="Concept drift detection results")

    # Aggregated outputs
    overall_drift_score: float = Field(..., ge=0.0, le=1.0, description="Composite drift score")
    features_with_drift: List[str] = Field(..., description="Features showing drift")
    alerts: List[DriftAlert] = Field(..., description="Generated alerts")

    # Summary
    drift_summary: str = Field(..., description="Human-readable summary")
    recommended_actions: List[str] = Field(..., description="Recommended actions")

    # Metadata
    detection_latency_ms: int
    features_checked: int
    baseline_timestamp: str
    current_timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 3.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class DriftMonitorState(TypedDict):
    """Complete LangGraph state for Drift Monitor Agent"""

    # === INPUT ===
    query: str
    model_id: Optional[str]
    features_to_monitor: List[str]
    time_window: str
    brand: Optional[str]

    # === CONFIGURATION ===
    significance_level: float
    psi_threshold: float
    check_data_drift: bool
    check_model_drift: bool
    check_concept_drift: bool

    # === DETECTION OUTPUTS ===
    data_drift_results: Optional[List[Dict[str, Any]]]  # DriftResult
    model_drift_results: Optional[List[Dict[str, Any]]]
    concept_drift_results: Optional[List[Dict[str, Any]]]

    # === AGGREGATED OUTPUTS ===
    overall_drift_score: Optional[float]
    features_with_drift: Optional[List[str]]
    alerts: Optional[List[Dict[str, Any]]]  # DriftAlert

    # === SUMMARY ===
    drift_summary: Optional[str]
    recommended_actions: Optional[List[str]]

    # === EXECUTION METADATA ===
    detection_latency_ms: int
    features_checked: int
    baseline_timestamp: str
    current_timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "detecting", "aggregating", "completed", "failed"]
```

### 3.4 Handoff Format

```yaml
drift_monitor_handoff:
  agent: drift_monitor
  analysis_type: drift_detection
  status: completed

  key_findings:
    overall_drift_score: 0.34
    features_checked: 15
    features_with_drift: 4
    critical_alerts: 1
    warning_alerts: 2

  drift_results:
    data_drift:
      - feature: "hcp_specialty"
        severity: critical
        psi: 0.28
        p_value: 0.001
        drift_detected: true
      - feature: "patient_volume"
        severity: high
        psi: 0.19
        p_value: 0.012
        drift_detected: true

    model_drift:
      - feature: "prediction_score"
        severity: medium
        ks_stat: 0.15
        p_value: 0.045
        drift_detected: true

    concept_drift: []

  alerts:
    - alert_id: "a1b2c3d4"
      severity: critical
      drift_type: data
      affected_features: ["hcp_specialty"]
      message: "Critical data drift detected in 1 feature(s)"
      recommended_action: "Investigate data pipeline and retrain model if necessary"
      timestamp: "2025-12-18T10:30:00Z"
    - alert_id: "e5f6g7h8"
      severity: warning
      drift_type: data
      affected_features: ["patient_volume", "competitive_pressure"]
      message: "Elevated drift detected in 2 feature(s)"
      recommended_action: "Monitor closely and prepare for potential intervention"
      timestamp: "2025-12-18T10:30:00Z"

  recommendations:
    - "URGENT: Address critical drift alerts immediately"
    - "Investigate data pipeline for schema or distribution changes"
    - "Evaluate model performance metrics and consider retraining"
    - "Schedule review of features showing elevated drift"

  warnings:
    - "Low sample size for some segments (n<50) excluded"
    - "Baseline data from 30 days ago may not reflect seasonal patterns"

  requires_further_analysis: true
  suggested_next_agent: health_score
  suggested_reason: "Check overall system health after drift detected"
```

---

## 4. Health Score Agent

**Agent Type**: Standard (Fast Path)
**Primary Methods**: Metric aggregation, no LLM
**Latency**: Up to 5s

### 4.1 Input Contract

```python
from pydantic import BaseModel, Field

class HealthScoreInput(BaseModel):
    """Input contract for Health Score Agent"""

    # Required fields
    query: str

    # Optional configuration
    check_scope: CheckScope = Field("full", description="Scope of health check")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the system health status?",
                "check_scope": "full"
            }
        }
    }
```

### 4.2 Output Contract

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class ComponentStatus(BaseModel):
    """Status of a system component"""
    component_name: str
    status: HealthStatus
    latency_ms: Optional[int]
    last_check: str
    error_message: Optional[str]

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    prediction_latency_p50_ms: Optional[int]
    prediction_latency_p99_ms: Optional[int]
    predictions_last_24h: int
    error_rate: float
    status: HealthStatus

class PipelineStatus(BaseModel):
    """Data pipeline status"""
    pipeline_name: str
    last_run: str
    last_success: str
    rows_processed: int
    freshness_hours: float
    status: Literal["healthy", "stale", "failed"]

class AgentStatus(BaseModel):
    """Agent availability status"""
    agent_name: str
    tier: int
    available: bool
    avg_latency_ms: int
    success_rate: float
    last_invocation: str

class HealthScoreOutput(BaseModel):
    """Output contract for Health Score Agent"""

    # Component health
    component_statuses: List[ComponentStatus] = Field(..., description="Status of system components")
    component_health_score: float = Field(..., ge=0.0, le=1.0)

    # Model health
    model_metrics: List[ModelMetrics] = Field(..., description="Model performance metrics")
    model_health_score: float = Field(..., ge=0.0, le=1.0)

    # Pipeline health
    pipeline_statuses: List[PipelineStatus] = Field(..., description="Data pipeline statuses")
    pipeline_health_score: float = Field(..., ge=0.0, le=1.0)

    # Agent health
    agent_statuses: List[AgentStatus] = Field(..., description="Agent availability")
    agent_health_score: float = Field(..., ge=0.0, le=1.0)

    # Composite score
    overall_health_score: float = Field(..., ge=0.0, le=100.0, description="Overall health (0-100)")
    health_grade: HealthGrade = Field(..., description="Letter grade (A-F)")

    # Issues
    critical_issues: List[str] = Field(..., description="Critical issues")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Summary
    health_summary: str = Field(..., description="Human-readable summary")

    # Metadata
    check_latency_ms: int
    timestamp: str
```

### 4.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class HealthScoreState(TypedDict):
    """Complete LangGraph state for Health Score Agent"""

    # === INPUT ===
    query: str
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"]

    # === COMPONENT HEALTH ===
    component_statuses: Optional[List[Dict[str, Any]]]  # ComponentStatus
    component_health_score: Optional[float]

    # === MODEL HEALTH ===
    model_metrics: Optional[List[Dict[str, Any]]]  # ModelMetrics
    model_health_score: Optional[float]

    # === PIPELINE HEALTH ===
    pipeline_statuses: Optional[List[Dict[str, Any]]]  # PipelineStatus
    pipeline_health_score: Optional[float]

    # === AGENT HEALTH ===
    agent_statuses: Optional[List[Dict[str, Any]]]  # AgentStatus
    agent_health_score: Optional[float]

    # === COMPOSITE SCORE ===
    overall_health_score: Optional[float]
    health_grade: Optional[Literal["A", "B", "C", "D", "F"]]

    # === ISSUES ===
    critical_issues: Optional[List[str]]
    warnings: Optional[List[str]]

    # === SUMMARY ===
    health_summary: Optional[str]

    # === EXECUTION METADATA ===
    check_latency_ms: int
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    status: Literal["pending", "checking", "completed", "failed"]
```

### 4.4 Handoff Format

```yaml
health_score_handoff:
  agent: health_score
  analysis_type: system_health
  status: completed

  key_findings:
    overall_score: 87.5
    grade: B
    critical_issues: 0
    warnings: 3

  component_scores:
    component_health: 0.95
    model_health: 0.85
    pipeline_health: 0.90
    agent_health: 0.80

  component_details:
    healthy: ["database", "cache", "vector_store", "api_gateway"]
    degraded: ["message_queue"]
    unhealthy: []

  model_details:
    healthy: ["trx_predictor_v2", "market_share_model"]
    degraded: ["churn_classifier"]
    unhealthy: []

  pipeline_details:
    healthy: ["hcp_data_pipeline", "metrics_aggregator"]
    stale: ["external_api_sync"]
    failed: []

  agent_details:
    available: 10
    unavailable: 0
    low_success_rate: ["resource_optimizer"]

  critical_issues: []

  warnings:
    - "Component 'message_queue' is degraded"
    - "Model 'churn_classifier' has accuracy below threshold"
    - "Pipeline 'external_api_sync' data is stale (48 hours)"
    - "Agent 'resource_optimizer' has low success rate (78%)"

  recommendations:
    - "Investigate message queue latency issues"
    - "Review churn_classifier model performance and consider retraining"
    - "Check external API connectivity for sync pipeline"
    - "Debug resource_optimizer agent errors"

  summary: "System health is good (Grade: B, Score: 87.5/100). All systems operational with minor issues."

  requires_further_analysis: false
  suggested_next_agent: drift_monitor
  suggested_reason: "Monitor for drift if degraded model performance persists"
```

---

## 5. Inter-Agent Communication

### 5.1 Orchestrator → Tier 3 Dispatch

All Tier 3 agents receive dispatches via the standard `AgentDispatchRequest` (see `orchestrator-contracts.md`).

### 5.2 Tier 3 → Orchestrator Response

All Tier 3 agents return via the standard `AgentDispatchResponse`.

### 5.3 Tier 3 Inter-Agent Handoffs

#### Gap Analyzer → Experiment Designer

When Gap Analyzer identifies an opportunity requiring experimental validation:

```python
# Gap Analyzer sets in response
next_agent = "experiment_designer"
handoff_context = {
    "upstream_agent": "gap_analyzer",
    "top_opportunity": {
        "metric": "trx",
        "segment": "specialty=Oncology",
        "gap_size": 500,
        "expected_roi": 3.2
    },
    "suggested_design": {
        "business_question": "Does targeted engagement increase TRx in Oncology segment?",
        "treatment": "targeted_engagement_program",
        "outcome": "trx",
        "segment": "specialty=Oncology"
    },
    "reason": "Design experiment to validate gap closure intervention"
}
```

#### Drift Monitor → Health Score

When Drift Monitor detects critical drift:

```python
# Drift Monitor sets in response
next_agent = "health_score"
handoff_context = {
    "upstream_agent": "drift_monitor",
    "drift_summary": {
        "overall_score": 0.65,
        "critical_alerts": 2,
        "features_with_drift": ["hcp_specialty", "patient_volume"]
    },
    "reason": "Check system health after critical drift detected"
}
```

#### Experiment Designer → Causal Impact

After experiment design is complete:

```python
# Experiment Designer sets in response
next_agent = "causal_impact"
handoff_context = {
    "upstream_agent": "experiment_designer",
    "experiment_design": {
        "hypothesis": "Increasing HCP engagement causes higher TRx",
        "treatment_var": "hcp_engagement_frequency",
        "outcome_var": "trx_total",
        "confounders": ["hcp_specialty", "region", "patient_volume"],
        "design_type": "cluster_rct"
    },
    "reason": "Use for post-experiment causal analysis"
}
```

---

## 6. Validation Rules

### 6.1 Input Validation

**Experiment Designer:**
```python
def validate_experiment_designer_input(state: ExperimentDesignState) -> List[str]:
    """Validation for Experiment Designer inputs"""
    errors = []

    # Required fields
    if not state.get("business_question"):
        errors.append("business_question is required")
    if not state.get("constraints"):
        errors.append("constraints are required")
    if not state.get("available_data"):
        errors.append("available_data is required")

    # Constraints validation
    constraints = state.get("constraints", {})
    if "expected_effect_size" in constraints:
        if not (0.1 <= constraints["expected_effect_size"] <= 2.0):
            errors.append("expected_effect_size must be between 0.1 and 2.0")
    if "power" in constraints:
        if not (0.5 <= constraints["power"] <= 0.99):
            errors.append("power must be between 0.5 and 0.99")
    if "alpha" in constraints:
        if not (0.001 <= constraints["alpha"] <= 0.1):
            errors.append("alpha must be between 0.001 and 0.1")

    return errors
```

**Drift Monitor:**
```python
def validate_drift_monitor_input(state: DriftMonitorState) -> List[str]:
    """Validation for Drift Monitor inputs"""
    errors = []

    # Required fields
    if not state.get("features_to_monitor") or len(state["features_to_monitor"]) == 0:
        errors.append("At least one feature is required to monitor")

    # Configuration validation
    if state.get("significance_level"):
        if not (0.01 <= state["significance_level"] <= 0.10):
            errors.append("significance_level must be between 0.01 and 0.10")
    if state.get("psi_threshold"):
        if not (0.0 <= state["psi_threshold"] <= 1.0):
            errors.append("psi_threshold must be between 0.0 and 1.0")

    # Model drift requires model_id
    if state.get("check_model_drift", True) and not state.get("model_id"):
        errors.append("model_id is required when check_model_drift is True")

    return errors
```

**Health Score:**
```python
def validate_health_score_input(state: HealthScoreState) -> List[str]:
    """Validation for Health Score inputs"""
    errors = []

    # Minimal validation
    if not state.get("query"):
        errors.append("query is required")

    # check_scope validation
    valid_scopes = ["full", "quick", "models", "pipelines", "agents"]
    if state.get("check_scope") and state["check_scope"] not in valid_scopes:
        errors.append(f"check_scope must be one of {valid_scopes}")

    return errors
```

### 6.2 Output Validation

```python
def validate_tier3_output(output: Dict[str, Any], agent_name: str) -> List[str]:
    """Validate Tier 3 agent outputs"""
    errors = []

    # Common required fields
    required = ["total_latency_ms", "warnings", "timestamp"]
    for field in required:
        if field not in output:
            errors.append(f"{agent_name} output missing required field: {field}")

    # Latency sanity check
    max_latency = {
        "experiment_designer": 60000,  # 60s
        "drift_monitor": 10000,         # 10s
        "health_score": 5000            # 5s
    }
    if "total_latency_ms" in output:
        if output["total_latency_ms"] > max_latency.get(agent_name, 60000):
            errors.append(f"{agent_name} exceeded maximum latency")

    # Agent-specific validation
    if agent_name == "experiment_designer":
        if "validity_score" in output:
            if output["validity_score"] not in ["strong", "moderate", "weak"]:
                errors.append("validity_score must be 'strong', 'moderate', or 'weak'")
    elif agent_name == "drift_monitor":
        if "overall_drift_score" in output:
            if not (0.0 <= output["overall_drift_score"] <= 1.0):
                errors.append("overall_drift_score must be between 0.0 and 1.0")
    elif agent_name == "health_score":
        if "overall_health_score" in output:
            if not (0.0 <= output["overall_health_score"] <= 100.0):
                errors.append("overall_health_score must be between 0.0 and 100.0")
        if "health_grade" in output:
            if output["health_grade"] not in ["A", "B", "C", "D", "F"]:
                errors.append("health_grade must be A, B, C, D, or F")

    return errors
```

---

## 7. Error Handling

### 7.1 Experiment Designer Error Patterns

**Design Reasoning Timeout:**
```python
try:
    response = await asyncio.wait_for(
        llm.ainvoke(design_prompt),
        timeout=120
    )
except asyncio.TimeoutError:
    # Fallback to simpler model
    response = await fallback_llm.ainvoke(simplified_prompt)
    state = {**state, "warnings": ["Design used fallback model due to timeout"]}
```

**Validity Audit Failure:**
```python
# Validity audit is non-fatal - proceed without it
if validity_audit_failed:
    return {
        **state,
        "validity_score": "unknown",
        "proceed_recommendation": "proceed_with_caution",
        "warnings": ["Validity audit failed - manual review recommended"],
        "status": "generating"
    }
```

**Redesign Loop Exceeded:**
```python
# If max redesign iterations reached, proceed anyway
if redesign_iterations >= max_redesign_iterations:
    return {
        **state,
        "warnings": [f"Max redesign iterations ({max_redesign_iterations}) reached"],
        "proceed_recommendation": "proceed_with_caution",
        "status": "generating"
    }
```

### 7.2 Drift Monitor Error Patterns

**Baseline Missing:**
```python
if baseline_data is None:
    return {
        **state,
        "warnings": ["No baseline data available - cannot detect drift"],
        "data_drift_results": [],
        "status": "completed"
    }
```

**Insufficient Sample Size:**
```python
if len(current) < 30 or len(baseline) < 30:
    return {
        **state,
        "warnings": [f"Feature '{feature}' has insufficient sample size (n<30)"],
        "data_drift_results": []  # Skip this feature
    }
```

### 7.3 Health Score Error Patterns

**Component Check Timeout:**
```python
try:
    result = await asyncio.wait_for(
        health_client.check(endpoint),
        timeout=2.0  # 2 second timeout
    )
except asyncio.TimeoutError:
    return ComponentStatus(
        component_name=component_name,
        status="unhealthy",
        latency_ms=2000,
        last_check=datetime.now().isoformat(),
        error_message="Health check timed out"
    )
```

**Metrics Unavailable:**
```python
# If metrics unavailable, mark as unknown and continue
if not metrics:
    return ModelMetrics(
        model_id=model_id,
        status="unknown",
        predictions_last_24h=0,
        error_rate=0.0,
        # ... other fields None
    )
```

---

## 8. Performance Requirements

| Agent | Target Latency | Max Latency | Throughput | LLM Calls |
|-------|----------------|-------------|------------|-----------|
| **Experiment Designer** | 30-40s | 60s | 1 query/min | 2-3 (design + validity) |
| **Drift Monitor** | 5-7s | 10s | 6 queries/min | 0 (no LLM) |
| **Health Score** | 2-3s | 5s | 12 queries/min | 0 (no LLM) |

### 8.1 Latency Breakdown

**Experiment Designer:**
- Context loading: <2s
- Design reasoning (LLM): 10-20s
- Power analysis: 1-2s
- Validity audit (LLM): 10-15s
- Template generation: 1-2s
- Total: 30-40s

**Drift Monitor:**
- Data fetch: 1-2s
- Parallel drift detection: 2-4s
- Alert aggregation: <1s
- Total: 5-7s

**Health Score:**
- Component checks (parallel): 1-2s
- Model metrics (parallel): <1s
- Score composition: <1s
- Total: 2-3s

---

## 9. Testing Requirements

### 9.1 Unit Tests

**Experiment Designer:**
- Design reasoning prompt generation
- Power analysis calculations (continuous, binary, cluster RCT)
- Validity threat identification
- DoWhy DAG generation
- Redesign loop logic

**Drift Monitor:**
- PSI calculation
- KS test execution
- Chi-square test for categorical
- Alert threshold logic
- Severity classification

**Health Score:**
- Component health scoring
- Model metric aggregation
- Composite score calculation
- Grade determination
- Issue identification

### 9.2 Integration Tests

```python
async def test_tier3_agent_integration(agent_name: str):
    """Test full agent execution"""

    # Setup
    state = create_test_input(agent_name)
    graph = build_agent_graph(agent_name)

    # Execute
    result = await graph.ainvoke(state)

    # Validate output contract
    errors = validate_tier3_output(result, agent_name)
    assert len(errors) == 0, f"Output validation failed: {errors}"

    # Check required fields
    assert result["status"] == "completed"
    assert result["total_latency_ms"] > 0
    assert result["timestamp"]
```

---

## 10. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-18 | Initial Tier 3 contracts | Claude |

---

## 11. Related Documents

- `base-contract.md` - Base agent structures
- `orchestrator-contracts.md` - Orchestrator communication
- `tier0-contracts.md` - ML Foundation contracts
- `tier2-contracts.md` - Causal Inference contracts
- `agent-handoff.yaml` - Standard handoff format examples
- `.claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md` - Experiment Designer specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md` - Drift Monitor specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/health-score.md` - Health Score specialist

---

## 12. DSPy Signal Contracts

Tier 3 agents have mixed DSPy roles (from E2I DSPy Feedback Learner Architecture V2):

| Agent | DSPy Role | Primary Signature | Behavior |
|-------|-----------|-------------------|----------|
| **Drift Monitor** | Sender | `HopDecisionSignature` | Generates signals |
| **Experiment Designer** | Sender | `InvestigationPlanSignature` | Generates signals |
| **Health Score** | Recipient | N/A | Consumes optimized prompts |

### 12.1 Sender Agents (Drift Monitor, Experiment Designer)

```python
# Drift Monitor: Sender
class DriftMonitorAgent(DSPySenderMixin):
    @property
    def agent_name(self) -> str:
        return "drift_monitor"

    @property
    def primary_signature(self) -> str:
        return "HopDecisionSignature"

    async def _decide_investigation_hop(self, state: DriftMonitorState) -> Dict:
        # ... hop decision logic ...
        result = await self._call_hop_decision(context)

        # Collect training signal
        self.collect_training_signal(
            input_data={"context": context, "drift_type": state["drift_type"]},
            output_data={"next_hop": result["hop_type"], "reasoning": result["reasoning"]},
            quality_score=result.get("quality", 0.8),
            signature_name="HopDecisionSignature",
            session_id=state.get("session_id")
        )

        return result


# Experiment Designer: Sender
class ExperimentDesignerAgent(DSPySenderMixin):
    @property
    def agent_name(self) -> str:
        return "experiment_designer"

    @property
    def primary_signature(self) -> str:
        return "InvestigationPlanSignature"
```

### 12.2 Recipient Agent (Health Score)

```python
class DSPyRecipientMixin(ABC):
    """
    Mixin for agents that receive DSPy-optimized prompts.

    Recipients don't generate training signals, but use optimized prompts
    for improved performance.
    """

    def __init__(self):
        self._optimized_prompts: Dict[str, str] = {}

    async def load_optimized_prompts(
        self,
        prompts: Dict[str, str]
    ) -> None:
        """
        Load optimized prompts from Hub.

        Called by orchestrator before agent execution.
        """
        self._optimized_prompts.update(prompts)

    def get_optimized_prompt(
        self,
        signature_name: str,
        default: str = ""
    ) -> str:
        """Get optimized prompt for a signature."""
        return self._optimized_prompts.get(signature_name, default)

    def has_optimized_prompts(self) -> bool:
        """Check if optimized prompts are available."""
        return bool(self._optimized_prompts)


class HealthScoreAgent(DSPyRecipientMixin):
    """Health Score agent receives optimized prompts."""
    pass
```

### 12.3 Signal Quality Thresholds

```python
TIER3_SIGNAL_QUALITY_THRESHOLDS = {
    "drift_monitor": {
        "min_quality": 0.6,
        "min_confidence": 0.5,
        "max_latency_ms": 30000,
        "required_fields": ["drift_detected", "drift_type"]
    },
    "experiment_designer": {
        "min_quality": 0.6,
        "min_confidence": 0.5,
        "max_latency_ms": 60000,
        "required_fields": ["experiment_design", "sample_size"]
    }
}
```
