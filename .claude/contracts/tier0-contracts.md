# Tier 0 (ML Foundation) Contracts

**Purpose**: Define integration contracts for the 7 ML Foundation agents that handle the complete ML lifecycle from problem definition to production deployment.

**Version**: 1.1
**Last Updated**: 2025-12-31 (V4.4 Causal Discovery Integration)
**Owner**: E2I Development Team

---

## Overview

Tier 0 is the ML Foundation layer consisting of 7 agents that must execute **sequentially** in a strict pipeline:

```
scope_definer → data_preparer → model_selector → model_trainer → feature_analyzer → model_deployer
                                                                                           ↓
                    observability_connector (cross-cutting, always running)
```

**Critical Constraint**: `data_preparer` implements a **QC GATE** that blocks downstream training if data quality fails.

---

## Agent Pipeline Flow

### Sequential Execution Requirements

| Step | Agent | Input From | Output To | Can Skip | Critical |
|------|-------|------------|-----------|----------|----------|
| 1 | scope_definer | User/Orchestrator | data_preparer | No | Yes |
| 2 | data_preparer | scope_definer | model_selector | No | Yes (GATE) |
| 3 | model_selector | data_preparer | model_trainer | No | Yes |
| 4 | model_trainer | model_selector | feature_analyzer | No | Yes |
| 5 | feature_analyzer | model_trainer | model_deployer | No | Yes |
| 6 | model_deployer | feature_analyzer | Tier 1-5 agents | Sometimes | Yes |
| 7 | observability_connector | All agents | Database | No | No |

---

## 1. scope_definer Contracts

### Input Contract

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class ScopeDefinerInput(BaseModel):
    """Input for scope_definer agent."""

    # === PROBLEM DESCRIPTION ===
    problem_description: str = Field(
        ...,
        description="Natural language description of ML problem"
    )

    business_objective: str = Field(
        ...,
        description="Business objective this ML model serves"
    )

    # === PROBLEM TYPE (if known) ===
    problem_type_hint: Optional[Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "causal_inference",
        "time_series"
    ]] = Field(None, description="Hint about problem type")

    # === TARGET ===
    target_variable: Optional[str] = Field(
        None,
        description="Target variable name (if known)"
    )

    # === FEATURES ===
    candidate_features: Optional[List[str]] = Field(
        None,
        description="Candidate feature list (if known)"
    )

    # === CONSTRAINTS ===
    time_budget_hours: Optional[float] = Field(
        None,
        description="Maximum training time budget"
    )

    performance_requirements: Optional[Dict[str, float]] = Field(
        None,
        description="Performance requirements (e.g., {'min_f1': 0.85})"
    )

    # === CONTEXT ===
    brand: Optional[str] = Field(None, description="Brand context")
    use_case: Optional[str] = Field(None, description="Use case category")
```

### Output Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ScopeSpec(BaseModel):
    """Scope specification output."""

    # === PROBLEM DEFINITION ===
    experiment_id: str = Field(..., description="Unique experiment identifier")
    problem_type: str = Field(..., description="Classification, regression, etc.")
    prediction_target: str = Field(..., description="Target variable name")

    # === FEATURES ===
    required_features: List[str] = Field(..., description="Features required for training")
    excluded_features: List[str] = Field(
        default_factory=list,
        description="Features to exclude (e.g., PII, leakage risks)"
    )

    # === DATA REQUIREMENTS ===
    inclusion_criteria: List[str] = Field(..., description="Data inclusion criteria")
    exclusion_criteria: List[str] = Field(..., description="Data exclusion criteria")
    minimum_samples: int = Field(..., description="Minimum required samples")

    # === SUCCESS CRITERIA ===
    success_criteria: Dict[str, float] = Field(
        ...,
        description="Success metrics and thresholds"
    )

    # === CONSTRAINTS ===
    time_budget_hours: Optional[float] = None
    cost_budget_dollars: Optional[float] = None

    # === METADATA ===
    brand: Optional[str] = None
    use_case: str

class ScopeDefinerOutput(BaseModel):
    """Complete output from scope_definer."""

    scope_spec: ScopeSpec
    validation_passed: bool
    validation_warnings: List[str] = Field(default_factory=list)
```

### Integration Contract

- **Upstream**: Orchestrator or user
- **Downstream**: data_preparer
- **Handoff**: `scope_definer_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_experiments` table
- **Blocking**: No (but data_preparer cannot proceed without valid scope)

---

## 2. data_preparer Contracts

### Input Contract

```python
from typing import Dict, Any
from pydantic import BaseModel, Field

class DataPreparerInput(BaseModel):
    """Input for data_preparer agent."""

    # === SCOPE (from scope_definer) ===
    scope_spec: ScopeSpec = Field(..., description="Scope from scope_definer")

    # === DATA SOURCE ===
    data_source: str = Field(
        ...,
        description="Data source table/view name"
    )

    split_id: Optional[str] = Field(
        None,
        description="ML split ID (if using existing split)"
    )

    # === VALIDATION ===
    validation_suite: Optional[str] = Field(
        None,
        description="Great Expectations suite name (defaults to auto-detect)"
    )

    skip_leakage_check: bool = Field(
        default=False,
        description="Skip data leakage detection (NOT RECOMMENDED)"
    )
```

### Output Contract

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class QCReport(BaseModel):
    """Data quality check report."""

    report_id: str
    experiment_id: str
    status: Literal["passed", "failed", "warning", "skipped"]
    overall_score: float = Field(..., ge=0.0, le=1.0)

    # Dimension scores
    completeness_score: float
    validity_score: float
    consistency_score: float
    uniqueness_score: float
    timeliness_score: float

    # Results
    expectation_results: List[Dict[str, Any]]
    failed_expectations: List[str]
    warnings: List[str]

    # Recommendations
    remediation_steps: List[str]
    blocking_issues: List[str]  # If non-empty, blocks training

    # Metadata
    row_count: int
    column_count: int
    validated_at: str  # ISO timestamp

class BaselineMetrics(BaseModel):
    """Baseline metrics for drift detection."""

    experiment_id: str
    split_type: Literal["train"]  # Always from training set
    feature_stats: Dict[str, Dict[str, Any]]
    target_rate: Optional[float]  # For classification
    target_distribution: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]
    computed_at: str  # ISO timestamp
    training_samples: int

class DataReadiness(BaseModel):
    """Data readiness summary."""

    experiment_id: str
    is_ready: bool

    # Counts
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    holdout_samples: int

    # Features
    available_features: List[str]
    missing_required_features: List[str]

    # Quality
    qc_passed: bool
    qc_score: float

    # Blockers
    blockers: List[str]  # Must be empty for is_ready=True

class DataPreparerOutput(BaseModel):
    """Complete output from data_preparer."""

    qc_report: QCReport
    baseline_metrics: BaselineMetrics
    data_readiness: DataReadiness
    gate_passed: bool  # Critical: blocks model_trainer if False
```

### QC Gate Contract

```python
class QCGate:
    """
    QC Gate that blocks downstream training.

    The data_preparer agent implements this gate. If QC fails,
    model_trainer MUST NOT proceed.
    """

    @staticmethod
    def check_gate(qc_report: QCReport) -> bool:
        """
        Check if QC gate allows proceeding to training.

        Returns:
            True if training can proceed, False if blocked
        """
        # Gate blocked if status is failed
        if qc_report.status == "failed":
            return False

        # Gate blocked if there are blocking issues
        if qc_report.blocking_issues:
            return False

        # Gate blocked if overall score too low
        if qc_report.overall_score < 0.80:
            return False

        # Gate passed
        return True

    @staticmethod
    def get_blocking_reason(qc_report: QCReport) -> str:
        """Get human-readable reason for gate blockage."""
        if qc_report.status == "failed":
            return f"QC status: {qc_report.status}"

        if qc_report.blocking_issues:
            return f"Blocking issues: {', '.join(qc_report.blocking_issues)}"

        if qc_report.overall_score < 0.80:
            return f"QC score too low: {qc_report.overall_score:.2f} < 0.80"

        return "Unknown blocking reason"
```

### Integration Contract

- **Upstream**: scope_definer
- **Downstream**: model_selector (ONLY if QC gate passes)
- **Handoff**: `data_preparer_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_data_quality_reports`, `ml_feature_store`
- **Blocking**: YES - Implements QC gate that blocks model_trainer
- **MLOps Tools**: Great Expectations, Feast

### Feast Integration (v4.3)

```python
# data_preparer registers features in Feast AFTER QC passes
# src/agents/ml_foundation/data_preparer/nodes/feast_registrar.py

class FeastRegistrarNode:
    """Register validated features in Feast after QC passes."""

    async def execute(self, state: DataPreparerState) -> DataPreparerState:
        if not state.get("qc_passed"):
            return state  # Skip Feast registration if QC failed

        registration = await self.feast_client.register_features(
            experiment_id=state["experiment_id"],
            features_df=state["validated_data"],
            entity_key="hcp_id",
        )

        return {**state, "feast_registration": registration}
```

**Output Addition**:
```python
class DataPreparerOutput(BaseModel):
    # ... existing fields ...
    feast_registration: Optional[FeastRegistration] = None  # NEW in v4.3
```

---

## 3. model_selector Contracts

### Input Contract

```python
from pydantic import BaseModel, Field

class ModelSelectorInput(BaseModel):
    """Input for model_selector agent."""

    # === SCOPE ===
    scope_spec: ScopeSpec = Field(..., description="Problem scope")

    # === QC RESULTS (from data_preparer) ===
    qc_report: QCReport = Field(..., description="QC report (must have passed)")
    baseline_metrics: BaselineMetrics = Field(..., description="Baseline metrics")

    # === PREFERENCES ===
    algorithm_preferences: Optional[List[str]] = Field(
        None,
        description="Preferred algorithms (if any)"
    )

    excluded_algorithms: Optional[List[str]] = Field(
        None,
        description="Algorithms to exclude"
    )

    interpretability_required: bool = Field(
        default=False,
        description="Whether model must be interpretable"
    )
```

### Output Contract

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ModelCandidate(BaseModel):
    """Model candidate recommendation."""

    algorithm_name: str
    algorithm_class: str  # Python class path
    default_hyperparameters: Dict[str, Any]
    rationale: str  # Why this algorithm was selected

    # Expected performance
    expected_performance: Dict[str, float]  # Based on similar problems

    # Characteristics
    training_time_estimate_hours: float
    interpretability_score: float = Field(ge=0.0, le=1.0)
    scalability_score: float = Field(ge=0.0, le=1.0)

    # Hyperparameter search space
    hyperparameter_search_space: Dict[str, Dict[str, Any]]

class ModelSelectorOutput(BaseModel):
    """Complete output from model_selector."""

    primary_candidate: ModelCandidate
    alternative_candidates: List[ModelCandidate] = Field(default_factory=list)
    selection_rationale: str
    baseline_to_beat: Dict[str, float]  # Baseline model performance
    registered_in_mlflow: bool
```

### Integration Contract

- **Upstream**: data_preparer (requires QC gate passed)
- **Downstream**: model_trainer
- **Handoff**: `model_selector_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_model_registry` (stage: development)
- **Blocking**: No (but model_trainer needs candidate)
- **MLOps Tools**: MLflow

---

## 4. model_trainer Contracts

### Input Contract

```python
from pydantic import BaseModel, Field

class ModelTrainerInput(BaseModel):
    """Input for model_trainer agent."""

    # === MODEL CANDIDATE (from model_selector) ===
    model_candidate: ModelCandidate = Field(..., description="Model to train")

    # === DATA (from data_preparer) ===
    experiment_id: str
    qc_report: QCReport  # MUST verify gate passed before training

    # === HYPERPARAMETER TUNING ===
    enable_hpo: bool = Field(default=True, description="Enable Optuna HPO")
    hpo_trials: int = Field(default=100, ge=1, description="Number of Optuna trials")
    hpo_timeout_hours: Optional[float] = Field(None, description="HPO timeout")

    # === TRAINING CONFIGURATION ===
    early_stopping: bool = Field(default=True)
    early_stopping_patience: int = Field(default=10)

    # === SUCCESS CRITERIA ===
    success_criteria: Dict[str, float] = Field(
        ...,
        description="From scope_spec.success_criteria"
    )
```

### Output Contract

```python
from typing import Dict, Any
from pydantic import BaseModel, Field

class TrainedModel(BaseModel):
    """Trained model metadata."""

    model_id: str
    experiment_id: str
    algorithm: str
    hyperparameters: Dict[str, Any]

    # Training info
    training_samples: int
    training_duration_seconds: float
    early_stopped: bool
    final_epoch: Optional[int]

class ValidationMetrics(BaseModel):
    """Validation set metrics."""

    # Core metrics (problem-type dependent)
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None

    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # Confusion matrix (classification only)
    confusion_matrix: Optional[Dict[str, int]] = None

    # All metrics dict
    all_metrics: Dict[str, float]

class MLflowInfo(BaseModel):
    """MLflow registration info."""

    run_id: str
    model_uri: str
    registered_model_name: str
    version: int
    stage: str  # "Staging"

class ModelTrainerOutput(BaseModel):
    """Complete output from model_trainer."""

    trained_model: TrainedModel
    validation_metrics: ValidationMetrics
    mlflow_info: MLflowInfo

    success_criteria_met: bool
    success_criteria_results: Dict[str, bool]  # metric -> passed/failed

    hpo_completed: bool
    hpo_best_trial: Optional[int] = None
```

### Training Gate Check

```python
def check_training_gate(qc_report: QCReport) -> None:
    """
    MANDATORY: Check QC gate before training.

    model_trainer MUST call this before any training.

    Raises:
        QCGateBlockedError: If QC gate is blocked
    """
    if not QCGate.check_gate(qc_report):
        reason = QCGate.get_blocking_reason(qc_report)
        raise QCGateBlockedError(
            f"Cannot train: QC gate blocked. Reason: {reason}"
        )
```

### Integration Contract

- **Upstream**: model_selector (requires model candidate + QC gate passed)
- **Downstream**: feature_analyzer
- **Handoff**: `model_trainer_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_training_runs`
- **Blocking**: YES - Must verify QC gate passed before training
- **MLOps Tools**: MLflow, Optuna, Feast

### Feast Integration (v4.3)

```python
# model_trainer uses Feast for point-in-time correct feature retrieval
# src/agents/ml_foundation/model_trainer/nodes/split_loader.py

class SplitLoaderNode:
    """Load training splits with point-in-time feature retrieval."""

    async def _load_from_feast(
        self,
        entity_df: pd.DataFrame,  # Must include event_timestamp column
        feature_refs: List[str],
    ) -> pd.DataFrame:
        """Load features from Feast with point-in-time correctness.

        CRITICAL: Uses event_timestamp for temporal joins to prevent data leakage.
        Features are retrieved as-of the event_timestamp, ensuring the model
        only sees data that was available at prediction time.
        """
        return await self.feast_client.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
```

**Point-in-Time Correctness**:
```
entity_df:
  hcp_id | event_timestamp      | target
  001    | 2024-01-15 00:00:00  | 1
  002    | 2024-01-20 00:00:00  | 0

Feast returns features AS OF event_timestamp:
  - For hcp_001: features as of 2024-01-15 (no future data)
  - For hcp_002: features as of 2024-01-20 (no future data)
```

---

## 5. feature_analyzer Contracts

### Input Contract

```python
from pydantic import BaseModel, Field

class FeatureAnalyzerInput(BaseModel):
    """Input for feature_analyzer agent."""

    # === TRAINED MODEL (from model_trainer) ===
    model_uri: str = Field(..., description="MLflow model URI")
    experiment_id: str

    # === SHAP CONFIGURATION ===
    max_samples: int = Field(default=1000, description="Max samples for SHAP")
    compute_interactions: bool = Field(default=True, description="Compute feature interactions")

    # === OUTPUT OPTIONS ===
    store_in_semantic_memory: bool = Field(
        default=True,
        description="Store feature relationships in semantic memory"
    )
```

### Output Contract

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class FeatureImportance(BaseModel):
    """Feature importance from SHAP."""

    feature: str
    importance: float
    rank: int

class FeatureInteraction(BaseModel):
    """Feature interaction detected."""

    features: List[str]  # Usually 2 features
    interaction_strength: float
    interpretation: str  # Natural language interpretation

class SHAPAnalysis(BaseModel):
    """SHAP analysis results."""

    experiment_id: str
    model_version: str
    shap_analysis_id: str

    feature_importance: List[FeatureImportance]
    interactions: List[FeatureInteraction]

    samples_analyzed: int
    computation_time_seconds: float

class FeatureAnalyzerOutput(BaseModel):
    """Complete output from feature_analyzer."""

    shap_analysis: SHAPAnalysis
    interpretation: str  # Natural language summary
    semantic_memory_updated: bool
    top_features: List[str]  # Top 5 features
    top_interactions: List[FeatureInteraction]  # Top 3 interactions
```

### Integration Contract

- **Upstream**: model_trainer
- **Downstream**: model_deployer
- **Handoff**: `feature_analyzer_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_shap_analyses`
- **Memory**: Writes to semantic memory (feature relationships)
- **Blocking**: No (but provides valuable context for deployment)
- **MLOps Tools**: SHAP, Opik (for LLM interpretation)

### V4.4 Enhancement: Causal Discovery Integration

The feature_analyzer can optionally enable causal discovery to compare causal vs predictive feature importance using the DriverRanker.

#### FeatureAnalyzerDiscoveryInput

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

class FeatureAnalyzerDiscoveryInput(BaseModel):
    """Optional discovery configuration for feature_analyzer."""

    discovery_enabled: bool = Field(
        default=False,
        description="Enable causal discovery for causal vs predictive comparison"
    )

    discovery_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""DiscoveryConfig parameters:
        - algorithms: List[str] - ['ges', 'pc', 'fci', 'lingam']
        - alpha: float - Significance level (default: 0.05)
        - ensemble_threshold: float - Min algorithm agreement (default: 0.5)
        - max_cond_vars: int - Max conditioning set size (default: 3)
        """
    )

    causal_target_variable: Optional[str] = Field(
        None,
        description="Target variable for causal path analysis (inferred from y if not set)"
    )

    divergent_threshold: int = Field(
        default=3,
        description="Rank difference threshold for divergent features"
    )
```

#### FeatureAnalyzerDiscoveryOutput

```python
from typing import Dict, Any, List, Tuple, Literal
from pydantic import BaseModel, Field

class FeatureRanking(BaseModel):
    """Individual feature ranking from DriverRanker."""

    feature_name: str
    causal_rank: int
    predictive_rank: int
    causal_score: float
    predictive_score: float
    rank_difference: int
    is_direct_cause: bool
    path_length: Optional[int] = None

class FeatureAnalyzerDiscoveryOutput(BaseModel):
    """Discovery output from feature_analyzer."""

    # Discovery result
    discovery_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Full DiscoveryResult from DiscoveryRunner"
    )

    # Gate evaluation
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]] = Field(
        None,
        description="Gate decision for discovery quality"
    )
    discovery_gate_confidence: Optional[float] = Field(
        None,
        description="Gate confidence score (0-1)"
    )
    discovery_gate_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for gate decision"
    )

    # Causal rankings
    causal_rankings: List[FeatureRanking] = Field(
        default_factory=list,
        description="Feature rankings from DriverRanker"
    )

    # Rank correlation
    rank_correlation: Optional[float] = Field(
        None,
        description="Spearman correlation between causal and predictive ranks"
    )

    # Feature categorization
    divergent_features: List[str] = Field(
        default_factory=list,
        description="Features with |rank_difference| > threshold"
    )
    causal_only_features: List[str] = Field(
        default_factory=list,
        description="Features with causal but no predictive signal"
    )
    predictive_only_features: List[str] = Field(
        default_factory=list,
        description="Features with predictive but no causal signal"
    )
    concordant_features: List[str] = Field(
        default_factory=list,
        description="Features with similar causal and predictive ranks"
    )

    # Causal importance
    causal_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Causal importance scores by feature"
    )
    causal_importance_ranked: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Sorted causal importance (feature, score)"
    )

    # Direct causes
    direct_cause_features: List[str] = Field(
        default_factory=list,
        description="Features that are direct causes of target"
    )

    # Interpretation
    causal_interpretation: Optional[str] = Field(
        None,
        description="NL explanation of causal vs predictive comparison"
    )
```

#### Discovery Integration Contract

- **Components Used**: DiscoveryRunner, DiscoveryGate, DriverRanker
- **Execution**: After SHAP computation (requires shap_values for comparison)
- **Skip Conditions**: discovery_enabled=False, missing data/SHAP values
- **Gate Decisions**:
  - `accept`: Use discovered rankings directly
  - `review`: Use rankings but flag for human review
  - `reject`: Skip causal comparison (SHAP-only)
  - `augment`: Combine discovered insights with SHAP
- **Key Insight**: Divergent features reveal where correlation ≠ causation

---

## 6. model_deployer Contracts

### Input Contract

```python
from pydantic import BaseModel, Field

class ModelDeployerInput(BaseModel):
    """Input for model_deployer agent."""

    # === MODEL (from model_trainer) ===
    model_uri: str = Field(..., description="MLflow model URI")
    experiment_id: str

    # === VALIDATION (from model_trainer) ===
    validation_metrics: ValidationMetrics
    success_criteria_met: bool

    # === SHAP ANALYSIS (from feature_analyzer) ===
    shap_analysis_id: Optional[str] = Field(
        None,
        description="SHAP analysis ID for explainability"
    )

    # === DEPLOYMENT CONFIG ===
    target_environment: Literal["staging", "shadow", "production"] = Field(
        default="staging",
        description="Target deployment environment"
    )

    deployment_name: str = Field(..., description="Deployment name")

    # === SERVING CONFIG ===
    resources: Dict[str, str] = Field(
        default={"cpu": "2", "memory": "4Gi"},
        description="Resource requirements"
    )

    max_batch_size: int = Field(default=100)
    max_latency_ms: int = Field(default=100)
```

### Output Contract

```python
from typing import Dict, Any
from pydantic import BaseModel, Field

class DeploymentManifest(BaseModel):
    """Deployment manifest."""

    deployment_id: str
    experiment_id: str
    model_version: str

    # Environment
    environment: str  # "staging", "shadow", "production"
    endpoint_url: str

    # Resources
    resources: Dict[str, str]

    # Status
    status: str  # "deploying", "healthy", "unhealthy"
    deployed_at: str  # ISO timestamp

    # Health
    health_check_url: str
    metrics_url: str

class VersionRecord(BaseModel):
    """MLflow version record."""

    registered_model_name: str
    version: int
    stage: str  # "Staging", "Production", etc.
    description: str

class ModelDeployerOutput(BaseModel):
    """Complete output from model_deployer."""

    deployment_manifest: DeploymentManifest
    version_record: VersionRecord
    bentoml_tag: str

    deployment_successful: bool
    health_check_passed: bool
    rollback_available: bool
```

### Stage Progression Rules

```python
# Model stage progression (cannot skip stages)
STAGE_PROGRESSION = {
    "None": ["Staging"],
    "Staging": ["Shadow", "Archived"],
    "Shadow": ["Production", "Archived"],
    "Production": ["Archived"],
    "Archived": []  # Terminal
}

# Shadow mode requirements
SHADOW_MODE_REQUIREMENTS = {
    "min_duration_hours": 24,
    "min_requests": 1000,
    "max_error_rate": 0.01,
    "max_latency_p99_ms": 150
}
```

### Integration Contract

- **Upstream**: feature_analyzer
- **Downstream**: Tier 1-5 agents (via prediction endpoints)
- **Handoff**: `model_deployer_handoff` (see agent-handoff.yaml)
- **Database**: Writes to `ml_deployments`, updates `ml_model_registry`
- **Blocking**: No (deployment can be deferred)
- **MLOps Tools**: MLflow, BentoML

---

## 7. observability_connector Contracts

### Input Contract

```python
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class ObservabilityEvent(BaseModel):
    """Event to be logged to Opik."""

    # === IDENTIFICATION ===
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None

    # === AGENT INFO ===
    agent_name: str
    operation: str  # "execute", "compute", "llm_call", etc.

    # === TIMING ===
    started_at: str  # ISO timestamp
    completed_at: Optional[str] = None  # ISO timestamp
    duration_ms: Optional[int] = None

    # === STATUS ===
    status: str  # "started", "completed", "failed"
    error: Optional[str] = None

    # === DATA ===
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # === QUALITY METRICS ===
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    confidence: Optional[float] = None
```

### Output Contract

```python
from pydantic import BaseModel, Field

class ObservabilityConnectorOutput(BaseModel):
    """Output from observability_connector."""

    span_ids_logged: List[str]
    trace_ids_logged: List[str]
    events_logged: int

    opik_project: str
    opik_workspace: str

    quality_metrics_computed: bool
    quality_score: Optional[float] = None
```

### Integration Contract

- **Upstream**: ALL agents (cross-cutting)
- **Downstream**: Database (ml_observability_spans)
- **Handoff**: Not applicable (cross-cutting, not in main pipeline)
- **Database**: Writes to `ml_observability_spans`
- **Blocking**: No (runs async)
- **MLOps Tools**: Opik

---

## Tier 0 Pipeline Validation

### Pipeline Integrity Check

```python
from typing import Dict, List, Any

def validate_tier0_pipeline(
    pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate Tier 0 pipeline integrity.

    Args:
        pipeline_state: State dict with all agent outputs

    Returns:
        Validation result with passed/failed status
    """
    checks = []

    # Check 1: All required agents executed
    required_agents = [
        "scope_definer",
        "data_preparer",
        "model_selector",
        "model_trainer",
        "feature_analyzer"
    ]

    for agent in required_agents:
        if agent not in pipeline_state:
            checks.append({
                "check": f"{agent}_executed",
                "passed": False,
                "message": f"Required agent {agent} did not execute"
            })
        else:
            checks.append({
                "check": f"{agent}_executed",
                "passed": True,
                "message": f"{agent} executed successfully"
            })

    # Check 2: QC gate passed
    if "data_preparer" in pipeline_state:
        qc_report = pipeline_state["data_preparer"]["qc_report"]
        gate_passed = QCGate.check_gate(qc_report)
        checks.append({
            "check": "qc_gate_passed",
            "passed": gate_passed,
            "message": "QC gate passed" if gate_passed else QCGate.get_blocking_reason(qc_report)
        })

    # Check 3: Model meets success criteria
    if "model_trainer" in pipeline_state:
        success_met = pipeline_state["model_trainer"]["success_criteria_met"]
        checks.append({
            "check": "success_criteria_met",
            "passed": success_met,
            "message": "Model meets success criteria" if success_met else "Model does not meet criteria"
        })

    # Check 4: All outputs present
    required_outputs = {
        "scope_definer": "scope_spec",
        "data_preparer": "qc_report",
        "model_selector": "primary_candidate",
        "model_trainer": "trained_model",
        "feature_analyzer": "shap_analysis"
    }

    for agent, output_field in required_outputs.items():
        if agent in pipeline_state:
            if output_field in pipeline_state[agent]:
                checks.append({
                    "check": f"{agent}_{output_field}_present",
                    "passed": True,
                    "message": f"{output_field} present"
                })
            else:
                checks.append({
                    "check": f"{agent}_{output_field}_present",
                    "passed": False,
                    "message": f"{output_field} missing"
                })

    # Aggregate results
    all_passed = all(check["passed"] for check in checks)

    return {
        "pipeline_valid": all_passed,
        "checks": checks,
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"])
    }
```

---

## Testing Requirements

All Tier 0 integrations must verify:

1. **Sequential Execution**:
   - Agents execute in correct order
   - Each agent receives correct input from upstream
   - Pipeline fails gracefully if agent fails

2. **QC Gate**:
   - QC gate blocks training when data quality fails
   - QC gate allows training when data quality passes
   - Proper error messages when gate blocks

3. **Data Flow**:
   - ScopeSpec flows correctly through pipeline
   - QC report reaches all downstream agents
   - Model artifacts accessible to downstream agents

4. **MLOps Integration**:
   - MLflow logging works correctly
   - Optuna integration functions
   - SHAP computation succeeds
   - BentoML deployment works

5. **Database Writes**:
   - All agents write to correct tables
   - Foreign key relationships maintained
   - No orphaned records

---

## Examples

### Complete Tier 0 Pipeline Execution

```python
# 1. scope_definer
scope_output = await scope_definer.run(ScopeDefinerInput(
    problem_description="Predict HCP conversion likelihood",
    business_objective="Improve targeting efficiency",
    target_variable="converted",
    ...
))

# 2. data_preparer
data_output = await data_preparer.run(DataPreparerInput(
    scope_spec=scope_output.scope_spec,
    data_source="patient_journeys",
    ...
))

# CRITICAL: Check QC gate
if not data_output.gate_passed:
    raise QCGateBlockedError(
        f"QC gate blocked: {data_output.qc_report.blocking_issues}"
    )

# 3. model_selector
selector_output = await model_selector.run(ModelSelectorInput(
    scope_spec=scope_output.scope_spec,
    qc_report=data_output.qc_report,
    baseline_metrics=data_output.baseline_metrics,
    ...
))

# 4. model_trainer
trainer_output = await model_trainer.run(ModelTrainerInput(
    model_candidate=selector_output.primary_candidate,
    experiment_id=scope_output.scope_spec.experiment_id,
    qc_report=data_output.qc_report,  # Verifies gate passed
    ...
))

# 5. feature_analyzer
analyzer_output = await feature_analyzer.run(FeatureAnalyzerInput(
    model_uri=trainer_output.mlflow_info.model_uri,
    experiment_id=scope_output.scope_spec.experiment_id,
    ...
))

# 6. model_deployer (optional)
if should_deploy:
    deployer_output = await model_deployer.run(ModelDeployerInput(
        model_uri=trainer_output.mlflow_info.model_uri,
        experiment_id=scope_output.scope_spec.experiment_id,
        validation_metrics=trainer_output.validation_metrics,
        success_criteria_met=trainer_output.success_criteria_met,
        shap_analysis_id=analyzer_output.shap_analysis.shap_analysis_id,
        ...
    ))
```

---

## Compliance Checklist

Before deploying Tier 0 pipeline:

- [ ] All 7 agents implemented
- [ ] Sequential execution enforced
- [ ] QC gate correctly blocks training
- [ ] All input/output contracts validated
- [ ] MLOps tools integrated (MLflow, Optuna, SHAP, BentoML, Great Expectations, Feast, Opik)
- [ ] Database writes correct
- [ ] Foreign keys maintained
- [ ] Pipeline validation function tested
- [ ] Error propagation works correctly
- [ ] Observability spans created
- [ ] Unit tests for each agent
- [ ] Integration tests for full pipeline
- [ ] QC gate failure test passes
- [ ] Success criteria validation works

---

**End of Tier 0 Contracts**
