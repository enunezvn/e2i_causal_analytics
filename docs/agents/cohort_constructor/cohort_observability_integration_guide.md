# CohortConstructor Observability Integration Guide

## Overview

This guide explains how CohortConstructor integrates with **MLflow** (experiment tracking) and **Opik** (agent observability) in the E2I Tier 0 ML Foundation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CohortConstructor Agent                    │
│                        (Tier 0)                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼────┐                  ┌─────▼─────┐
    │ MLflow  │                  │   Opik    │
    │         │                  │           │
    │ WHAT?   │                  │   WHY?    │
    └─────────┘                  └───────────┘
         │                             │
         │                             │
    Cohort                        Execution
    Characteristics               Traces
    - Size                        - Step-by-step
    - Demographics                - Performance
    - Balance metrics             - Errors
    - Temporal properties         - Token usage
```

### Division of Responsibilities

| Tool | Purpose | Data Logged |
|------|---------|-------------|
| **MLflow** | **Experiment tracking** - What cohorts were created | Cohort size, demographics, balance metrics, temporal characteristics |
| **Opik** | **Agent observability** - Why/how cohort was constructed | Execution traces, criterion evaluations, performance, errors |

---

## Integration Patterns

### Pattern 1: Basic Integration (Recommended)

Log to both MLflow and Opik simultaneously:

```python
from cohort_constructor import CohortConstructor, CohortConfig
from cohort_mlflow import CohortMLflowLogger
from cohort_opik import InstrumentedCohortConstructor
import pandas as pd

# Setup
config = CohortConfig.from_brand('remibrutinib', 'csu')
patient_df = load_patient_data()

# Initialize loggers
mlflow_logger = CohortMLflowLogger(experiment_name="remibrutinib_csu_cohorts")
opik_constructor = InstrumentedCohortConstructor(config)

# Construct cohort with Opik tracing
with mlflow_logger.start_run(config, run_name='baseline_cohort_v1'):
    eligible_df, metadata = opik_constructor.construct_cohort(patient_df)
    
    # Log to MLflow
    mlflow_logger.log_cohort_characteristics(eligible_df, metadata)

print(f"✅ Logged to both MLflow and Opik")
```

**When to use:** Standard cohort construction with full observability

---

### Pattern 2: Development Mode (Opik Only)

Use Opik for rapid iteration and debugging:

```python
from cohort_opik import InstrumentedCohortConstructor

constructor = InstrumentedCohortConstructor(config)
eligible_df, metadata = constructor.construct_cohort(patient_df)

# View traces in Opik dashboard to debug
```

**When to use:** Developing new cohort definitions, debugging criteria

---

### Pattern 3: Production Mode (MLflow + Opik)

Full integration with error handling:

```python
from cohort_mlflow import CohortMLflowLogger
from cohort_opik import InstrumentedCohortConstructor, OpikCohortTracer
import mlflow

mlflow_logger = CohortMLflowLogger(
    experiment_name="production_cohorts",
    tracking_uri="https://mlflow.company.com"
)

opik_tracer = OpikCohortTracer(
    project_name="e2i_production",
    tags=['production', 'tier_0']
)

constructor = InstrumentedCohortConstructor(config, tracer=opik_tracer)

with mlflow_logger.start_run(config, tags={'environment': 'production'}):
    try:
        eligible_df, metadata = constructor.construct_cohort(patient_df)
        
        # Log characteristics
        mlflow_logger.log_cohort_characteristics(eligible_df, metadata)
        
        # Log success metrics
        mlflow.log_metric('success', 1)
        
    except Exception as e:
        # Error logged automatically to Opik
        mlflow.log_metric('success', 0)
        mlflow.log_param('error_message', str(e))
        raise
```

**When to use:** Production deployments with SLA requirements

---

## Detailed Integration Examples

### Example 1: Cohort Comparison with Balance Metrics

```python
from cohort_mlflow import CohortMLflowLogger

mlflow_logger = CohortMLflowLogger(experiment_name="psm_cohorts")

# Construct treatment cohort
treatment_config = CohortConfig.from_brand('remibrutinib', 'csu')
with mlflow_logger.start_run(treatment_config, run_name='treatment_cohort'):
    treatment_df, treatment_metadata = constructor.construct_cohort(treatment_patients)
    mlflow_logger.log_cohort_characteristics(treatment_df, treatment_metadata)

# Construct control cohort
control_config = CohortConfig.from_brand('remibrutinib', 'csu')
control_config.cohort_name = 'Control Cohort'
with mlflow_logger.start_run(control_config, run_name='control_cohort'):
    control_df, control_metadata = constructor.construct_cohort(control_patients)
    mlflow_logger.log_cohort_characteristics(control_df, control_metadata)
    
    # Log balance metrics
    covariates = ['age', 'gender', 'baseline_uas7', 'ah_failures']
    balance_metrics = mlflow_logger.log_balance_metrics(
        treatment_df, 
        control_df, 
        covariates
    )
    
    print(f"Average SMD: {balance_metrics['avg_absolute_smd']:.3f}")
```

**Result in MLflow:**
- Two runs: treatment and control
- Balance metrics comparing cohorts
- CSV artifacts with full balance table

---

### Example 2: A/B Testing Cohort Definitions

```python
from cohort_opik import OpikCohortTracer

tracer = OpikCohortTracer(project_name="cohort_ab_test")

# Create experiment
experiment = tracer.create_experiment(
    experiment_name="strict_vs_lenient_criteria",
    description="Compare cohort size vs. clinical purity trade-off",
    configs=[
        {'name': 'strict', 'uas7_threshold': 28},
        {'name': 'lenient', 'uas7_threshold': 16}
    ]
)

# Test both configurations
for config_name in ['strict', 'lenient']:
    config = CohortConfig.from_brand('remibrutinib', 'csu')
    # Modify config based on test...
    
    constructor = InstrumentedCohortConstructor(config, tracer=tracer)
    eligible_df, metadata = constructor.construct_cohort(patient_df)
    
    print(f"{config_name}: {len(eligible_df)} eligible")
```

**Result in Opik:**
- Side-by-side trace comparison
- Performance metrics for each variant
- Criterion-level impact analysis

---

### Example 3: Propensity Score Matching Validation

```python
from cohort_mlflow import CohortMLflowLogger

mlflow_logger = CohortMLflowLogger(experiment_name="psm_validation")

with mlflow_logger.start_run(config, run_name='psm_cohort'):
    # Construct matched cohort
    eligible_df, metadata = constructor.construct_cohort(patient_df)
    
    # Assume PS already calculated
    eligible_df['propensity_score'] = calculate_propensity_scores(eligible_df)
    
    # Log PS distribution
    mlflow_logger.log_propensity_scores(eligible_df, ps_column='propensity_score')
    
    # Log cohort characteristics
    mlflow_logger.log_cohort_characteristics(eligible_df, metadata)
```

**Result in MLflow:**
- PS distribution histogram (PNG artifact)
- PS statistics (mean, median, range)
- Common support validation

---

## Querying and Analysis

### MLflow Queries

```python
import mlflow

# Find best-balanced cohort
runs = mlflow.search_runs(
    experiment_names=["psm_cohorts"],
    filter_string="metrics.avg_absolute_smd < 0.1",
    order_by=["metrics.eligible_population DESC"]
)

best_run = runs.iloc[0]
print(f"Best cohort: {best_run['tags.mlflow.runName']}")
print(f"  Size: {best_run['metrics.eligible_population']}")
print(f"  Avg SMD: {best_run['metrics.avg_absolute_smd']:.3f}")

# Load cohort config
config_path = mlflow.artifacts.download_artifacts(
    run_id=best_run['run_id'],
    artifact_path='config/cohort_config.json'
)
```

### Opik Queries

```python
from opik import Opik

client = Opik()

# Find slowest criterion evaluations
traces = client.get_traces(
    project_name="e2i_tier0_cohort_constructor",
    filters={'span_name': 'evaluate_inclusion_criterion'}
)

slow_traces = sorted(
    traces, 
    key=lambda t: t.metadata.get('execution_time_ms', 0),
    reverse=True
)[:5]

for trace in slow_traces:
    print(f"Slow criterion: {trace.input['criterion_field']}")
    print(f"  Time: {trace.metadata['execution_time_ms']:.1f}ms")
```

---

## Dashboard Setup

### MLflow Dashboard

Access: `http://localhost:5000` or your MLflow server

**Key Views:**
1. **Experiments** → Navigate to `remibrutinib_csu_cohorts`
2. **Runs** → Compare cohort sizes, exclusion rates
3. **Artifacts** → Download balance metrics, eligibility logs
4. **Metrics** → Plot cohort size over time

### Opik Dashboard

Access: `https://www.comet.com/opik` or self-hosted Opik

**Key Views:**
1. **Traces** → See execution flow for each cohort construction
2. **Spans** → Drill down into criterion evaluations
3. **Feedback Scores** → Track cohort size, removal impact
4. **Datasets** → Version cohorts for reproducibility

---

## Performance Monitoring

### MLflow Metrics to Track

```python
# In production, track these metrics over time
mlflow.log_metric('eligible_population', len(eligible_df))
mlflow.log_metric('exclusion_rate', metadata['exclusion_rate'])
mlflow.log_metric('execution_time_seconds', metadata['execution_time_seconds'])
mlflow.log_metric('avg_absolute_smd', balance_metrics['avg_absolute_smd'])
mlflow.log_metric('data_quality_score', data_quality_check_score)
```

**Create alerts in MLflow:**
- Alert if `eligible_population < 30` (insufficient power)
- Alert if `avg_absolute_smd > 0.2` (poor balance)
- Alert if `execution_time_seconds > 120` (SLA breach)

### Opik Metrics to Track

```python
# Opik automatically tracks:
# - Token usage (if using LLM-based criteria)
# - Latency per criterion
# - Error rates
# - Memory usage

# Add custom feedback scores:
opik.log_feedback_score(name='cohort_quality', value=quality_score)
opik.log_feedback_score(name='clinical_validity', value=validity_score)
```

---

## Debugging Workflows

### Debug Low Cohort Size

**Step 1: Check MLflow**
```python
# Find run with low cohort size
runs = mlflow.search_runs(
    experiment_names=["remibrutinib_csu_cohorts"],
    filter_string="metrics.eligible_population < 50"
)

# Download eligibility log
log_path = mlflow.artifacts.download_artifacts(
    run_id=runs.iloc[0]['run_id'],
    artifact_path='audit/eligibility_log.csv'
)

import pandas as pd
log_df = pd.read_csv(log_path)
print(log_df.sort_values('removed', ascending=False).head())
```

**Step 2: Check Opik**
```python
# Find corresponding trace
trace_id = runs.iloc[0]['tags.opik_trace_id']  # If you logged it
trace = client.get_trace(trace_id)

# Examine criterion evaluations
for span in trace.spans:
    if span.name.startswith('evaluate_'):
        print(f"{span.input['criterion_field']}: removed {span.output['removed_rows']}")
```

**Step 3: Iterate**
- Adjust overly restrictive criteria
- Re-run with new config
- Compare results in MLflow

---

### Debug Poor Balance

**Step 1: Check Balance Metrics in MLflow**
```python
# Find runs with poor balance
runs = mlflow.search_runs(
    experiment_names=["psm_cohorts"],
    filter_string="metrics.avg_absolute_smd > 0.2"
)

# Download balance table
balance_path = mlflow.artifacts.download_artifacts(
    run_id=runs.iloc[0]['run_id'],
    artifact_path='balance/balance_metrics.csv'
)

balance_df = pd.read_csv(balance_path)
print(balance_df[balance_df['smd'].abs() > 0.2])  # Imbalanced covariates
```

**Step 2: Refine Matching**
- Add imbalanced covariates to matching criteria
- Tighten caliper width
- Re-run and compare

---

## Best Practices

### ✅ DO

1. **Always log to both MLflow and Opik** for production cohorts
2. **Use consistent naming** for experiments and runs
3. **Tag runs** with brand, indication, environment
4. **Log balance metrics** for all treatment/control comparisons
5. **Save config artifacts** for reproducibility
6. **Set up alerts** for SLA breaches and quality issues
7. **Version cohorts** using Opik datasets

### ❌ DON'T

1. **Don't skip MLflow logging** - you need historical cohort characteristics
2. **Don't ignore Opik traces** - they reveal performance bottlenecks
3. **Don't log PII** - patient_id should be pseudonymized
4. **Don't create duplicate experiments** - reuse existing ones
5. **Don't delete old runs** - keep full audit trail

---

## Integration with E2I Workflow

### Tier 0 ML Foundation Flow

```
1. scope_definer
   └─> Defines analysis scope
   
2. cohort_constructor ✨ (YOU ARE HERE)
   ├─> MLflow: Log cohort characteristics
   └─> Opik: Trace criterion evaluations
   
3. data_preparer
   └─> Receives eligible_patient_ids
   
4. model_trainer
   ├─> MLflow: Log model metrics
   └─> Opik: Trace training pipeline
   
5. model_deployer
   └─> Deploys model with cohort metadata
```

### Passing Data Between Agents

```python
# CohortConstructor output
cohort_output = {
    'eligible_patient_ids': eligible_df['patient_id'].tolist(),
    'cohort_metadata': metadata,
    'mlflow_run_id': mlflow.active_run().info.run_id,
    'opik_trace_id': opik.get_current_trace_id()
}

# DataPreparer receives this
data_preparer.prepare(
    patient_ids=cohort_output['eligible_patient_ids'],
    cohort_metadata=cohort_output['cohort_metadata']
)

# DataPreparer can link back to cohort construction
mlflow.log_param('cohort_mlflow_run_id', cohort_output['mlflow_run_id'])
opik.update_current_trace(
    metadata={'cohort_trace_id': cohort_output['opik_trace_id']}
)
```

---

## Troubleshooting

### Issue: MLflow not logging

**Check:**
```python
import mlflow
print(mlflow.get_tracking_uri())  # Should show server URL
print(mlflow.get_experiment_by_name("your_experiment_name"))  # Should return experiment
```

**Fix:** Set tracking URI before logging
```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Issue: Opik traces not appearing

**Check:**
```python
import opik
print(opik.config.get_config())  # Should show API key and workspace
```

**Fix:** Configure Opik
```python
opik.configure(
    api_key="your_api_key",
    workspace="your_workspace"
)
```

### Issue: Slow cohort construction

**Debug with Opik:**
1. Go to Opik dashboard
2. Find slow trace
3. Look at span durations
4. Identify slow criteria
5. Optimize or parallelize

---

## Summary

| Tool | What It Logs | When to Check | Dashboard URL |
|------|--------------|---------------|---------------|
| **MLflow** | Cohort characteristics, balance metrics, configs | Compare cohorts, track metrics over time | http://localhost:5000 |
| **Opik** | Execution traces, performance, errors | Debug issues, optimize performance | https://www.comet.com/opik |

Both tools are essential for full observability of the CohortConstructor agent in production E2I deployments.
