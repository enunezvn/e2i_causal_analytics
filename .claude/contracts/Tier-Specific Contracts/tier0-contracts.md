# Tier 0: ML Foundation Contracts

## Overview

Tier 0 agents handle the complete ML lifecycle from problem definition through deployment. These agents operate as a sequential pipeline with the **QC Gate** as a critical checkpoint.

**Agents Covered:**
- `scope_definer` - ML problem definition and success criteria
- `data_preparer` - Data quality validation and QC gating
- `model_selector` - Algorithm selection and registry
- `model_trainer` - Training execution with split enforcement
- `feature_analyzer` - SHAP computation and interpretation (Hybrid)
- `model_deployer` - Model deployment and lifecycle
- `observability_connector` - Cross-cutting telemetry (Async)

**Pipeline Flow:**
```
scope_definer → data_preparer → model_selector → model_trainer → feature_analyzer → model_deployer
                     │                                                    
                 QC GATE                                                  
              (blocks on failure)                                         
                                                                          
                     observability_connector (cross-cutting, all agents)
```

**Latency Budgets:**
| Agent | Budget | Notes |
|-------|--------|-------|
| scope_definer | <5s | Fast problem definition |
| data_preparer | <60s | Great Expectations validation |
| model_selector | <120s | Algorithm evaluation |
| model_trainer | Variable | Depends on data size, algorithm |
| feature_analyzer | <120s | SHAP computation + LLM interpretation |
| model_deployer | <30s | BentoML packaging |
| observability_connector | <100ms | Non-blocking async |

**Model Tier:** None (computation-only), except Feature Analyzer (Sonnet for interpretation)

---

## Scope Definer Contract

### Purpose
Transforms business objectives into formal ML problem specifications with measurable success criteria.

### Input Contract

```yaml
# scope_definer_input.yaml
scope_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # Business objective
  objective:
    description: string           # "Predict HCP response to engagement"
    business_goal: string         # "Increase NRx by 15%"
    brand: string                 # "Remibrutinib" | "Fabhalta" | "Kisqali"
    
  # Target specification
  target:
    variable: string              # "converted_hcp"
    definition: string            # "HCP wrote first NRx within 90 days"
    time_horizon_days: int        # 90
    
  # Population scope
  population:
    filters: object               # {"region": "Northeast", "specialty": "Rheumatology"}
    exclusions: list[string]      # ["already_prescribing", "inactive_license"]
    
  # Constraints
  constraints:
    regulatory: list[string]      # ["no_patient_pii", "hipaa_compliant"]
    ethical: list[string]         # ["no_demographic_bias", "explainable"]
    technical: object             # {"max_latency_ms": 100, "max_memory_gb": 4}
```

### Output Contract

```yaml
# scope_definer_output.yaml
scope_response:
  request_id: string              # Echo from input
  experiment_id: string           # Generated: "exp_{brand}_{region}_{timestamp}"
  timestamp: datetime
  processing_time_ms: int
  
  # Formal problem specification
  scope_spec:
    problem_type: enum            # "binary_classification" | "regression" | "ranking"
    target_variable: string
    target_definition: string
    observation_unit: string      # "hcp" | "patient" | "territory"
    time_horizon_days: int
    
    # Population
    population:
      base_query: string          # SQL or table reference
      filters: object
      exclusions: list[string]
      estimated_size: int
      
    # Features
    feature_requirements:
      required_groups: list[string]   # ["engagement_history", "hcp_profile"]
      excluded_features: list[string] # ["patient_name", "ssn"]
      temporal_cutoff: string         # "observation_date - 1 day"
      
    # Constraints
    constraints:
      regulatory: list[string]
      ethical: list[string]
      technical: object
      
  # Success criteria
  success_criteria:
    primary_metric: string        # "auc_roc"
    minimum_threshold: float      # 0.70
    
    secondary_metrics:
      - metric: string            # "precision_at_k"
        k: int                    # 100
        minimum: float            # 0.25
        
    fairness_constraints:
      - dimension: string         # "region"
        metric: string            # "equalized_odds_gap"
        max_gap: float            # 0.05
        
    # Baseline expectations
    baseline:
      random_baseline: float      # 0.50
      heuristic_baseline: float   # 0.62
      description: string         # "Current decile-based targeting"
      
  # Metadata
  status: enum                    # "success" | "partial" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["scope_spec", "success_criteria", "experiment_id"]
```

### Validation Rules

1. **Problem Type Valid**: Must be supported type (classification, regression, ranking)
2. **Target Defined**: Target variable must have clear definition
3. **Metrics Specified**: At least one success metric required
4. **Baseline Set**: Random baseline must be defined
5. **Constraints Captured**: All regulatory constraints documented

---

## Data Preparer Contract

### Purpose
Validates data quality, computes baseline metrics, and enforces the **QC Gate** that blocks training on failure.

### Input Contract

```yaml
# data_preparer_input.yaml
data_prep_request:
  request_id: string              # UUID
  experiment_id: string           # From scope_definer
  timestamp: datetime             # ISO 8601
  
  # Scope reference
  scope_spec:
    problem_type: string
    target_variable: string
    observation_unit: string
    population: object
    feature_requirements: object
    
  # Data source
  data_source:
    table: string                 # "ml_features.hcp_engagement"
    split_column: string          # "ml_split"
    timestamp_column: string      # "observation_date"
    
  # QC configuration
  qc_config:
    suite_name: string            # "e2i_classification_suite"
    blocking_severity: enum       # "error" | "warning"
    custom_expectations: list[object]  # Additional checks
```

### Output Contract

```yaml
# data_preparer_output.yaml
data_prep_response:
  request_id: string              # Echo from input
  experiment_id: string
  timestamp: datetime
  processing_time_ms: int
  
  # QC Report (CRITICAL)
  qc_report:
    status: enum                  # "PASSED" | "FAILED" | "WARNING"
    suite_name: string
    execution_time_ms: int
    
    # Dimension scores (0-1)
    dimension_scores:
      completeness: float         # % non-null
      validity: float             # % within valid ranges
      consistency: float          # % passing cross-field rules
      uniqueness: float           # % unique where required
      timeliness: float           # % within temporal bounds
      
    # Individual expectations
    expectations:
      total: int
      passed: int
      failed: int
      
    failed_expectations:
      - expectation_type: string  # "expect_column_values_to_not_be_null"
        column: string            # "hcp_id"
        observed_value: any       # 0.02 (2% null)
        threshold: any            # 0.0
        severity: enum            # "error" | "warning"
        
    # Blocking issues (if status == FAILED)
    blocking_issues:
      - issue: string
        severity: string
        recommendation: string
        
  # Baseline metrics (computed from TRAIN split only)
  baseline_metrics:
    split: string                 # "train"
    sample_size: int
    
    target_distribution:
      positive_rate: float        # 0.23
      positive_count: int
      negative_count: int
      
    feature_distributions:
      - feature: string
        dtype: string             # "numeric" | "categorical"
        mean: float               # For numeric
        std: float
        min: float
        max: float
        null_rate: float
        unique_count: int         # For categorical
        top_values: list[object]  # For categorical
        
    correlation_matrix:
      features: list[string]
      values: list[list[float]]   # Upper triangle
      
  # Leakage detection
  leakage_checks:
    temporal_leakage:
      status: enum                # "clean" | "detected"
      suspicious_features: list[string]
      
    target_leakage:
      status: enum
      high_correlation_features:
        - feature: string
          correlation: float
          
    train_test_leakage:
      status: enum
      overlapping_ids: int
      
  # Feature registration
  feature_registration:
    registered_count: int
    feast_feature_view: string    # "e2i_hcp_features"
    
  # Data readiness summary
  data_readiness:
    ready_for_training: bool      # QC status != FAILED
    blocking_issues_count: int
    warning_count: int
    recommendations: list[string]
    
  # Status
  status: enum                    # "success" | "blocked" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["qc_report", "baseline_metrics", "data_readiness"]
```

### Validation Rules

1. **QC Gate Enforcement**: `qc_report.status == FAILED` MUST block downstream training
2. **Train-Only Baselines**: All baseline_metrics computed from train split only
3. **Leakage Detection**: All three leakage types must be checked
4. **Feature Registration**: Features must be registered in Feast
5. **Split Validation**: Splits must be properly assigned

### QC Gate Critical Rule

```python
# This is a HARD GATE - training CANNOT proceed if QC fails
if qc_report.status == QCStatus.FAILED:
    return DataPrepResponse(
        status="blocked",
        data_readiness=DataReadiness(
            ready_for_training=False,
            blocking_issues_count=len(qc_report.blocking_issues)
        )
    )
```

---

## Model Selector Contract

### Purpose
Evaluates candidate algorithms and recommends optimal architecture based on problem type and constraints.

### Input Contract

```yaml
# model_selector_input.yaml
selection_request:
  request_id: string              # UUID
  experiment_id: string           # From scope_definer
  timestamp: datetime             # ISO 8601
  
  # Scope reference
  scope_spec:
    problem_type: string
    target_variable: string
    constraints: object
    
  # Success criteria
  success_criteria:
    primary_metric: string
    minimum_threshold: float
    
  # Data characteristics
  data_characteristics:
    sample_size: int
    feature_count: int
    categorical_features: int
    numeric_features: int
    class_imbalance_ratio: float
    
  # Selection preferences
  preferences:
    prioritize: enum              # "accuracy" | "speed" | "interpretability"
    require_causal: bool          # Prefer causal ML methods
    max_complexity: enum          # "low" | "medium" | "high"
```

### Output Contract

```yaml
# model_selector_output.yaml
selection_response:
  request_id: string              # Echo from input
  experiment_id: string
  timestamp: datetime
  processing_time_ms: int
  
  # Primary recommendation
  recommended_model:
    algorithm: string             # "CausalForestDML"
    framework: string             # "econml"
    version: string               # "0.15.0"
    
    hyperparameter_space:
      n_estimators:
        type: string              # "int"
        low: int                  # 100
        high: int                 # 500
      max_depth:
        type: string              # "int"
        low: int                  # 5
        high: int                 # 20
      min_samples_leaf:
        type: string              # "int"
        low: int                  # 10
        high: int                 # 100
        
    constraints_met:
      latency: bool
      memory: bool
      interpretability: bool
      
    expected_performance:
      metric: string              # "auc_roc"
      estimated_range: list[float] # [0.72, 0.78]
      confidence: float           # 0.75
      
  # Alternative candidates
  alternative_models:
    - algorithm: string           # "LinearDML"
      framework: string
      reason_not_primary: string  # "Lower expected performance"
      constraints_met: object
      
    - algorithm: string           # "XGBoostClassifier"
      framework: string
      reason_not_primary: string  # "Not causal"
      constraints_met: object
      
  # Baseline models
  baseline_models:
    - algorithm: string           # "LogisticRegression"
      purpose: string             # "Simple baseline"
      expected_performance: float # 0.65
      
    - algorithm: string           # "RandomClassifier"
      purpose: string             # "Random baseline"
      expected_performance: float # 0.50
      
  # Selection rationale
  rationale:
    primary_reasons: list[string]
    constraint_analysis: object
    historical_success_rate: float  # From procedural memory
    similar_experiments: list[string]
    
  # Registry
  model_registered:
    registry_name: string         # MLflow model registry name
    version: string               # "1"
    stage: string                 # "development"
    
  # Status
  status: enum                    # "success" | "partial" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["recommended_model", "baseline_models", "rationale"]
```

### Validation Rules

1. **Constraint Compliance**: Recommended model must meet all technical constraints
2. **Baseline Included**: At least random and one simple baseline required
3. **Hyperparameter Space**: Must be defined for Optuna tuning
4. **Registry Entry**: Model must be registered in MLflow
5. **Rationale Provided**: Selection reasoning must be documented

---

## Model Trainer Contract

### Purpose
Executes ML training with strict split enforcement and hyperparameter optimization.

### Input Contract

```yaml
# model_trainer_input.yaml
training_request:
  request_id: string              # UUID
  experiment_id: string           # From scope_definer
  timestamp: datetime             # ISO 8601
  
  # QC verification (REQUIRED)
  qc_verification:
    qc_report_id: string          # Must reference passing QC report
    qc_status: string             # Must be "PASSED"
    
  # Model specification
  model_spec:
    algorithm: string             # From model_selector
    framework: string
    hyperparameter_space: object
    
  # Data reference
  data_reference:
    feature_view: string          # Feast feature view
    entity_df: string             # Entity DataFrame reference
    
  # Training configuration
  training_config:
    n_trials: int                 # Optuna trials (default: 50)
    timeout_seconds: int          # Per-trial timeout
    early_stopping_rounds: int    # For gradient boosting
    cross_validation_folds: int   # For validation (default: 5)
    
  # Split enforcement
  split_config:
    train_split: float            # 0.60
    validation_split: float       # 0.20
    test_split: float             # 0.15
    holdout_split: float          # 0.05
```

### Output Contract

```yaml
# model_trainer_output.yaml
training_response:
  request_id: string              # Echo from input
  experiment_id: string
  run_id: string                  # MLflow run ID
  timestamp: datetime
  processing_time_ms: int
  
  # QC gate verification
  qc_gate_verified: bool          # MUST be true
  
  # Trained model
  trained_model:
    algorithm: string
    framework: string
    
    # Best hyperparameters from Optuna
    best_hyperparameters: object
    
    # Artifact locations
    artifacts:
      model_uri: string           # MLflow model URI
      preprocessor_uri: string    # Fitted preprocessor
      feature_list_uri: string    # Feature names
      
    # Training metadata
    training_metadata:
      train_samples: int
      validation_samples: int
      test_samples: int
      training_duration_seconds: float
      optuna_trials_completed: int
      early_stopped: bool
      
  # Metrics by split
  metrics:
    train:
      auc_roc: float
      precision: float
      recall: float
      f1: float
      
    validation:
      auc_roc: float
      precision: float
      recall: float
      f1: float
      # Confidence intervals from CV
      auc_roc_ci: list[float]     # [lower, upper]
      
    test:                         # Evaluated ONCE
      auc_roc: float
      precision: float
      recall: float
      f1: float
      precision_at_k:
        - k: int
          precision: float
      confusion_matrix:
        tp: int
        fp: int
        tn: int
        fn: int
        
  # Success criteria evaluation
  criteria_evaluation:
    primary_metric_met: bool
    primary_metric_value: float
    primary_metric_threshold: float
    
    secondary_metrics_met: list[object]
    fairness_constraints_met: list[object]
    
    overall_pass: bool
    
  # Model comparison
  model_comparison:
    - model: string               # "CausalForestDML"
      test_auc: float
      is_primary: bool
      
    - model: string               # "LogisticRegression"
      test_auc: float
      is_primary: bool
      improvement_over_baseline: float
      
  # Registry update
  registry_update:
    model_name: string
    version: string
    stage: string                 # "staging" if criteria met
    
  # Status
  status: enum                    # "success" | "criteria_not_met" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["trained_model", "metrics", "criteria_evaluation"]
```

### Validation Rules

1. **QC Gate Required**: Training MUST verify QC passed before proceeding
2. **Split Enforcement**: Train/Val/Test/Holdout splits must be respected
3. **Preprocessing Isolation**: Fit on train only, transform others
4. **Test Once**: Test set metrics computed exactly once
5. **Criteria Evaluation**: All success criteria must be evaluated
6. **Baseline Comparison**: Must show improvement over baselines

### Critical Split Enforcement

```python
# MANDATORY: Check QC gate first
if not qc_verification.qc_status == "PASSED":
    raise QCGateBlockedError("Training blocked: QC not passed")

# MANDATORY: Fit preprocessing on TRAIN ONLY
preprocessor.fit(X_train)

# Transform all splits (NO refitting)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Optuna tuning on VALIDATION set
best_params = optimize_hyperparameters(X_train_processed, y_train, X_val_processed, y_val)

# Final training on TRAIN set
model.fit(X_train_processed, y_train)

# Test evaluation ONCE
test_metrics = evaluate(model, X_test_processed, y_test)
```

---

## Feature Analyzer Contract (Hybrid)

### Purpose
Computes SHAP values (computation) and generates natural language interpretations (LLM).

### Input Contract

```yaml
# feature_analyzer_input.yaml
analysis_request:
  request_id: string              # UUID
  experiment_id: string
  run_id: string                  # MLflow run ID
  timestamp: datetime             # ISO 8601
  
  # Model reference
  model_reference:
    model_uri: string             # MLflow model URI
    algorithm: string
    
  # Data reference
  data_reference:
    X: string                     # Reference to feature matrix
    sample_size: int              # For SHAP computation
    
  # Analysis configuration
  analysis_config:
    compute_global: bool          # Global feature importance
    compute_local: bool           # Individual explanations
    compute_interactions: bool    # Interaction effects
    segment_analysis: list[string] # ["region", "specialty"]
    
  # Interpretation configuration
  interpretation_config:
    audience: enum                # "executive" | "analyst" | "technical"
    include_recommendations: bool
```

### Output Contract

```yaml
# feature_analyzer_output.yaml
analysis_response:
  request_id: string              # Echo from input
  experiment_id: string
  run_id: string
  timestamp: datetime
  processing_time_ms: int
  
  # Computation results (NO LLM)
  shap_analysis:
    # Global importance
    global_importance:
      - feature: string
        importance: float         # Normalized 0-1
        direction: string         # "positive" | "negative" | "mixed"
        
    # Feature interactions
    interactions:
      - feature_a: string
        feature_b: string
        interaction_strength: float
        
    # Segment analysis
    segment_shap:
      - segment: string           # "Northeast"
        dimension: string         # "region"
        top_features:
          - feature: string
            importance: float
            
    # Expected value
    expected_value: float
    
  # Interpretation results (LLM - Hybrid node)
  interpretation:
    # Executive summary
    executive_summary: string     # 2-3 sentences
    
    # Feature explanations
    feature_explanations:
      - feature: string
        plain_language: string    # "Higher call frequency strongly predicts conversion"
        business_implication: string
        
    # Key insights
    key_insights:
      - insight: string
        confidence: enum          # "high" | "medium" | "low"
        evidence: string
        
    # Recommendations
    recommendations:
      - action: string
        expected_impact: string
        priority: enum
        
  # Semantic memory updates
  memory_updates:
    feature_relationships:
      - source: string            # "call_frequency"
        target: string            # "conversion"
        relationship: string      # "positive_causal"
        strength: float
        
  # Status
  status: enum                    # "success" | "partial" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["shap_analysis", "interpretation"]
```

### Validation Rules

1. **Computation First**: SHAP computed before interpretation
2. **No LLM in Computation**: Computation nodes are pure Python
3. **Audience Adaptation**: Interpretation matches audience level
4. **Memory Updates**: Feature relationships stored in semantic memory
5. **Segment Coverage**: All requested segments analyzed

---

## Model Deployer Contract

### Purpose
Manages model lifecycle from development through production deployment.

### Input Contract

```yaml
# model_deployer_input.yaml
deployment_request:
  request_id: string              # UUID
  experiment_id: string
  run_id: string
  timestamp: datetime             # ISO 8601
  
  # Model reference
  model_reference:
    model_uri: string
    preprocessor_uri: string
    version: string
    
  # Deployment action
  action: enum                    # "promote" | "deploy" | "rollback" | "archive"
  
  # Target stage
  target_stage: enum              # "staging" | "shadow" | "production" | "archived"
  
  # Deployment configuration
  deployment_config:
    replicas: int                 # Default: 1
    resources:
      cpu: string                 # "2"
      memory: string              # "4Gi"
    traffic_percentage: int       # For canary (0-100)
    
  # Approval (for production)
  approval:
    approved_by: string           # User ID
    approval_timestamp: datetime
    criteria_verified: bool
```

### Output Contract

```yaml
# model_deployer_output.yaml
deployment_response:
  request_id: string              # Echo from input
  experiment_id: string
  timestamp: datetime
  processing_time_ms: int
  
  # Deployment result
  deployment:
    status: enum                  # "deployed" | "promoted" | "rolled_back" | "failed"
    
    # Endpoint details
    endpoint:
      name: string                # "e2i_remib_northeast_v3"
      url: string                 # "https://api.e2i.internal/predict/remib_northeast"
      bento_tag: string
      
    # Stage transition
    stage_transition:
      from_stage: string
      to_stage: string
      timestamp: datetime
      
    # Resource allocation
    resources:
      replicas: int
      cpu: string
      memory: string
      
  # Version record
  version_record:
    model_name: string
    version: string
    stage: string
    promoted_at: datetime
    promoted_by: string
    
    # Stage history
    stage_history:
      - stage: string
        entered_at: datetime
        exited_at: datetime
        duration_hours: float
        
  # Deployment validation
  validation:
    health_check: bool
    smoke_test: bool
    latency_check:
      p50_ms: float
      p95_ms: float
      p99_ms: float
      meets_sla: bool
      
  # Rollback info
  rollback:
    available: bool
    previous_version: string
    previous_endpoint: string
    
  # Status
  status: enum                    # "success" | "validation_failed" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = ["deployment", "version_record", "validation"]
```

### Validation Rules

1. **Stage Progression**: Must follow DEVELOPMENT → STAGING → SHADOW → PRODUCTION
2. **Shadow Required**: 24+ hours in shadow before production
3. **Criteria Check**: Production requires all success criteria met
4. **Health Validation**: Endpoint must pass health checks
5. **Rollback Ready**: Previous version must be available

---

## Observability Connector Contract

### Purpose
Cross-cutting telemetry for all agents, providing trace context and quality metrics.

### Input Contract

```yaml
# observability_connector_input.yaml
telemetry_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # Operation context
  operation:
    agent_name: string            # "model_trainer"
    agent_tier: int               # 0
    operation_name: string        # "train_model"
    
  # Parent context (for distributed tracing)
  parent_context:
    trace_id: string              # Propagated trace ID
    parent_span_id: string        # Parent span ID
    baggage: object               # Additional context
    
  # Span data
  span_data:
    start_time: datetime
    end_time: datetime
    status: enum                  # "ok" | "error"
    attributes: object            # Custom attributes
    
  # LLM metrics (for Hybrid/Deep agents)
  llm_metrics:
    model: string                 # "claude-sonnet-4-20250514"
    input_tokens: int
    output_tokens: int
    latency_ms: int
```

### Output Contract

```yaml
# observability_connector_output.yaml
telemetry_response:
  request_id: string              # Echo from input
  timestamp: datetime
  processing_time_ms: int         # Should be <100ms
  
  # Created span
  span:
    span_id: string
    trace_id: string
    parent_span_id: string
    operation_name: string
    start_time: datetime
    end_time: datetime
    duration_ms: int
    status: string
    attributes: object
    
  # Context for propagation
  context:
    trace_id: string
    span_id: string
    baggage: object
    
  # Quality metrics (aggregated)
  quality_metrics:
    agent_name: string
    
    latency:
      p50_ms: float
      p95_ms: float
      p99_ms: float
      
    error_rate:
      total_operations: int
      failed_operations: int
      error_rate: float
      
    token_usage:
      total_input_tokens: int
      total_output_tokens: int
      avg_tokens_per_operation: float
      
  # Status
  status: enum                    # "success" | "failed"
  async: bool                     # True - non-blocking
```

### Required Output Keys

```python
REQUIRED_KEYS = ["span", "context"]
```

### Validation Rules

1. **Non-Blocking**: Observability never blocks agent operations
2. **Context Propagation**: trace_id flows through all operations
3. **Graceful Degradation**: Failures logged but don't break agents
4. **Sampling Support**: High-volume traces can be sampled
5. **Async Processing**: Telemetry processed asynchronously

---

## Inter-Agent Communication

### Tier 0 → Tier 0 Pipeline

```yaml
# tier0_pipeline.yaml
pipeline_flow:
  - from: scope_definer
    to: data_preparer
    data: scope_spec, success_criteria, experiment_id
    
  - from: data_preparer
    to: model_selector
    data: qc_report, baseline_metrics, data_readiness
    gate: qc_report.status == "PASSED"  # CRITICAL GATE
    
  - from: model_selector
    to: model_trainer
    data: recommended_model, baseline_models
    
  - from: model_trainer
    to: feature_analyzer
    data: trained_model, metrics, run_id
    
  - from: feature_analyzer
    to: model_deployer
    data: shap_analysis, interpretation
    
  - from: model_deployer
    to: END
    data: deployment, version_record
```

### Tier 0 → Tier 1-5 Handoffs

```yaml
# tier0_to_analytics.yaml
handoffs:
  # Data Preparer → Drift Monitor
  - from: data_preparer
    to: drift_monitor
    data:
      baseline_metrics: object    # Reference distributions
      feature_distributions: object
    usage: "Compare current vs baseline for drift detection"
    
  # Model Trainer → Prediction Synthesizer
  - from: model_trainer
    to: prediction_synthesizer
    data:
      model_uri: string
      preprocessor_uri: string
    usage: "Load model for predictions"
    
  # Feature Analyzer → Causal Impact
  - from: feature_analyzer
    to: causal_impact
    data:
      feature_relationships: object  # From semantic memory
    usage: "Inform causal graph construction"
    
  # Model Deployer → Prediction Synthesizer
  - from: model_deployer
    to: prediction_synthesizer
    data:
      endpoint_url: string
    usage: "Call deployed model endpoint"
    
  # Observability Connector → Health Score
  - from: observability_connector
    to: health_score
    data:
      quality_metrics: object
    usage: "Include in system health calculation"
```

---

## Error Handling

### Error Codes by Agent

#### Scope Definer
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `SD_001` | Invalid problem type | Suggest valid types |
| `SD_002` | Missing target definition | Request clarification |
| `SD_003` | Constraint conflict | Flag incompatible constraints |

#### Data Preparer
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `DP_001` | QC validation failed | Return blocking issues, stop pipeline |
| `DP_002` | Leakage detected | Flag features, recommend removal |
| `DP_003` | Insufficient data | Report minimum required |
| `DP_004` | Split imbalance | Recommend stratification |

#### Model Selector
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `MS_001` | No models meet constraints | Relax constraints with warning |
| `MS_002` | Unknown algorithm | Default to safe choice |
| `MS_003` | Registry unavailable | Use local fallback |

#### Model Trainer
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `MT_001` | QC gate not passed | Block training, return error |
| `MT_002` | Training timeout | Return partial model |
| `MT_003` | Criteria not met | Return model with warning |
| `MT_004` | Memory exceeded | Reduce batch size, retry |

#### Feature Analyzer
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `FA_001` | SHAP computation timeout | Use subset sampling |
| `FA_002` | Interpretation failed | Return computation only |
| `FA_003` | Segment analysis empty | Skip segment, warn |

#### Model Deployer
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `MD_001` | Invalid stage transition | Reject with valid path |
| `MD_002` | Health check failed | Rollback to previous |
| `MD_003` | Shadow period insufficient | Block promotion |

#### Observability Connector
| Error Code | Description | Recovery |
|------------|-------------|----------|
| `OC_001` | Span emission failed | Log locally, continue |
| `OC_002` | Context lost | Generate new trace |

---

## Quality Assurance

### Tier 0 Quality Checks

```python
class Tier0QualityContract:
    """Quality checks for ML Foundation agents."""
    
    @staticmethod
    def validate_qc_gate(qc_report: QCReport, action: str) -> bool:
        """Ensure QC gate is properly enforced."""
        if action == "train" and qc_report.status == "FAILED":
            return False  # Must block
        return True
    
    @staticmethod
    def validate_split_isolation(preprocessor: Preprocessor, splits: dict) -> bool:
        """Ensure preprocessing only fit on train."""
        return preprocessor.is_fitted_on == "train"
    
    @staticmethod
    def validate_test_single_use(metrics_log: list) -> bool:
        """Ensure test set evaluated exactly once."""
        test_evaluations = [m for m in metrics_log if m.split == "test"]
        return len(test_evaluations) == 1
    
    @staticmethod
    def validate_stage_progression(from_stage: str, to_stage: str) -> bool:
        """Ensure valid stage transition."""
        valid_transitions = {
            "development": ["staging"],
            "staging": ["shadow"],
            "shadow": ["production"],
            "production": ["archived"]
        }
        return to_stage in valid_transitions.get(from_stage, [])
```

---

## Validation Tests

```bash
# Scope Definer contracts
pytest tests/integration/test_tier0_contracts.py::test_scope_definer_input
pytest tests/integration/test_tier0_contracts.py::test_scope_definer_output

# Data Preparer contracts (including QC gate)
pytest tests/integration/test_tier0_contracts.py::test_data_preparer_input
pytest tests/integration/test_tier0_contracts.py::test_data_preparer_output
pytest tests/integration/test_tier0_contracts.py::test_qc_gate_blocking

# Model Selector contracts
pytest tests/integration/test_tier0_contracts.py::test_model_selector_input
pytest tests/integration/test_tier0_contracts.py::test_model_selector_output

# Model Trainer contracts (including split enforcement)
pytest tests/integration/test_tier0_contracts.py::test_model_trainer_input
pytest tests/integration/test_tier0_contracts.py::test_model_trainer_output
pytest tests/integration/test_tier0_contracts.py::test_split_enforcement
pytest tests/integration/test_tier0_contracts.py::test_preprocessing_isolation

# Feature Analyzer contracts
pytest tests/integration/test_tier0_contracts.py::test_feature_analyzer_input
pytest tests/integration/test_tier0_contracts.py::test_feature_analyzer_output

# Model Deployer contracts
pytest tests/integration/test_tier0_contracts.py::test_model_deployer_input
pytest tests/integration/test_tier0_contracts.py::test_model_deployer_output
pytest tests/integration/test_tier0_contracts.py::test_stage_progression

# Observability Connector contracts
pytest tests/integration/test_tier0_contracts.py::test_observability_input
pytest tests/integration/test_tier0_contracts.py::test_observability_output

# Pipeline integration
pytest tests/integration/test_tier0_contracts.py::test_full_pipeline
pytest tests/integration/test_tier0_contracts.py::test_qc_gate_blocks_pipeline
pytest tests/integration/test_tier0_contracts.py::test_tier0_to_tier1_handoff
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-08 | Initial creation for V4 architecture |
