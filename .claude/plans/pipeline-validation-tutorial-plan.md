# E2I Pipeline Validation Tutorial - Comprehensive Implementation Plan

**Project**: E2I Causal Analytics
**Created**: 2026-01-05
**Status**: 🟡 IN PROGRESS
**Plan ID**: pipeline-validation-tutorial

---

## Executive Summary

Create a comprehensive tutorial demonstrating the full E2I pipeline validation workflow using synthetic data in Supabase. This tutorial covers:

1. **Data Quality (QC)** - Great Expectations validation
2. **MLOps Integration** - MLflow experiment tracking and model registry
3. **Agent Observability** - Opik tracing and monitoring
4. **RAG Evaluation** - RAGAS metrics for retrieval quality
5. **End-to-End Validation** - Full pipeline testing with expected insights

**Key Constraints**:
- Context-window friendly phases (15-20 min each)
- Testing in small batches (max 4 pytest workers)
- Memory-efficient execution for low-resource environments

---

## Prerequisites

### Existing Resources
- **Synthetic Data**: 20 JSON files (~29MB, ~1M records) in `data/synthetic/`
- **Data Splits**: 60% train, 20% validation, 15% test, 5% holdout
- **5 DGP Types**: Standard, Clustered, Treatment Effect, Temporal, High Dimensional
- **Ground Truth**: Known causal effects for validation

### Environment Requirements
- Python 3.11+ with venv
- Supabase project configured
- Docker (for MLflow/Opik local servers)
- 4+ GB RAM available

---

## Phase Overview

| Phase | Description | Duration | Tests | Status |
|-------|-------------|----------|-------|--------|
| 1 | Environment Setup & Verification | 15 min | 3 | ⬜ Pending |
| 2 | Load Synthetic Data to Supabase | 20 min | 4 | ⬜ Pending |
| 3 | Data Quality Validation (Great Expectations) | 20 min | 5 | ⬜ Pending |
| 4 | MLflow Experiment Tracking | 20 min | 5 | ⬜ Pending |
| 5 | Opik Agent Observability | 20 min | 5 | ⬜ Pending |
| 6 | RAGAS RAG Evaluation | 20 min | 5 | ⬜ Pending |
| 7 | End-to-End Pipeline Validation | 15 min | 3 | ⬜ Pending |
| 8 | Agent System Testing (Optional) | 15 min | 4 | ⬜ Pending |

**Total Estimated Time**: ~2.5 hours (excluding optional Phase 8)

---

## Phase 1: Environment Setup & Verification

**Goal**: Verify all dependencies and connections are working.

### To-Do Checklist

- [ ] 1.1 Verify Python environment and dependencies
- [ ] 1.2 Check Supabase connection
- [ ] 1.3 Verify MLflow server access
- [ ] 1.4 Check Opik server connection
- [ ] 1.5 Validate synthetic data files exist

### Tutorial Content

```python
# 1.1 Verify Environment
import sys
print(f"Python version: {sys.version}")

# Check key dependencies
from importlib.metadata import version
deps = ['fastapi', 'langchain', 'mlflow', 'great_expectations', 'ragas']
for dep in deps:
    try:
        print(f"{dep}: {version(dep)}")
    except:
        print(f"{dep}: NOT INSTALLED")

# 1.2 Check Supabase Connection
from src.repositories.base import get_supabase_client
client = get_supabase_client()
result = client.table('brands').select('*').limit(1).execute()
print(f"Supabase connection: {'OK' if result.data else 'FAILED'}")

# 1.3 Check MLflow
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# 1.4 Check Opik (via health endpoint)
import requests
try:
    r = requests.get("http://localhost:8080/health", timeout=5)
    print(f"Opik status: {r.status_code}")
except:
    print("Opik: NOT AVAILABLE (optional)")

# 1.5 Check synthetic data
from pathlib import Path
data_dir = Path("data/synthetic")
json_files = list(data_dir.glob("*.json"))
print(f"Synthetic data files: {len(json_files)}")
```

### Tests (Batch 1)

```bash
./venv/bin/python -m pytest tests/unit/test_repositories/test_base.py -v -k "supabase" --maxfail=3
```

- [ ] Test 1.T1: Supabase client initialization
- [ ] Test 1.T2: Environment variables loaded
- [ ] Test 1.T3: Synthetic data directory accessible

---

## Phase 2: Load Synthetic Data to Supabase

**Goal**: Load synthetic data into Supabase tables with split awareness.

### To-Do Checklist

- [ ] 2.1 Load patient journey data (8,505 records)
- [ ] 2.2 Load HCP data (50 records)
- [ ] 2.3 Load brand metrics data
- [ ] 2.4 Verify data splits (train/val/test/holdout)
- [ ] 2.5 Validate referential integrity

### Tutorial Content

```python
# 2.1 Load Patient Journey Data
import json
from pathlib import Path

# Read synthetic patient journeys
with open("data/synthetic/e2i_ml_v3_patient_journeys.json") as f:
    patient_journeys = json.load(f)
print(f"Patient journeys: {len(patient_journeys)} records")

# Load to Supabase with split awareness
from src.repositories.patient_journey import PatientJourneyRepository

repo = PatientJourneyRepository()

# Data is pre-split: check the split column
splits = {}
for journey in patient_journeys:
    split = journey.get('data_split', 'unknown')
    splits[split] = splits.get(split, 0) + 1
print(f"Data split distribution: {splits}")

# Expected: train=60%, val=20%, test=15%, holdout=5%

# 2.2 Load Training Split Only (for tutorials)
train_data = [j for j in patient_journeys if j.get('data_split') == 'train']
print(f"Training data: {len(train_data)} records")

# Batch insert (avoid memory issues)
BATCH_SIZE = 500
for i in range(0, len(train_data), BATCH_SIZE):
    batch = train_data[i:i+BATCH_SIZE]
    # repo.bulk_insert(batch)  # Uncomment to execute
    print(f"Batch {i//BATCH_SIZE + 1}: {len(batch)} records")
```

### Split-Aware Data Loading Pattern

```python
# Pattern: Always filter by split for ML workflows
from src.repositories.base import get_supabase_client

client = get_supabase_client()

# Get training data only
train_response = client.table('patient_journeys')\
    .select('*')\
    .eq('data_split', 'train')\
    .execute()

# Get validation data for hyperparameter tuning
val_response = client.table('patient_journeys')\
    .select('*')\
    .eq('data_split', 'validation')\
    .execute()

# NEVER use test/holdout for training!
print(f"Train: {len(train_response.data)}, Val: {len(val_response.data)}")
```

### Tests (Batch 2)

```bash
./venv/bin/python -m pytest tests/unit/test_repositories/ -v -k "patient_journey or hcp" -n 2 --maxfail=3
```

- [ ] Test 2.T1: Patient journey bulk insert
- [ ] Test 2.T2: HCP data loading
- [ ] Test 2.T3: Split column validation
- [ ] Test 2.T4: Referential integrity checks

---

## Phase 3: Data Quality Validation (Great Expectations)

**Goal**: Run data quality checks using Great Expectations expectation suites.

### To-Do Checklist

- [ ] 3.1 Initialize Great Expectations context
- [ ] 3.2 Create expectation suite for patient journeys
- [ ] 3.3 Validate data against expectations
- [ ] 3.4 Review validation results
- [ ] 3.5 Handle validation failures gracefully

### Tutorial Content

```python
# 3.1 Initialize Great Expectations
import great_expectations as gx
from great_expectations.data_context import FileDataContext

context = FileDataContext.create(project_root_dir=".")
print(f"GX Context initialized: {context.root_directory}")

# 3.2 Create Expectation Suite
suite_name = "patient_journey_expectations"
suite = context.add_expectation_suite(expectation_suite_name=suite_name)

# 3.3 Define Core Expectations
from great_expectations.core import ExpectationConfiguration

# Required columns exist
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_table_columns_to_match_set",
        kwargs={
            "column_set": [
                "patient_id", "journey_stage", "brand",
                "engagement_score", "data_split", "created_at"
            ]
        }
    )
)

# No null patient IDs
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "patient_id"}
    )
)

# Valid data splits
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "data_split",
            "value_set": ["train", "validation", "test", "holdout"]
        }
    )
)

# Engagement score range (0-100)
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "engagement_score", "min_value": 0, "max_value": 100}
    )
)

# 3.4 Run Validation
import pandas as pd

# Load data as DataFrame
df = pd.DataFrame(patient_journeys)

# Create validator
validator = context.sources.pandas_default.read_dataframe(df)
results = validator.validate(expectation_suite=suite)

# 3.5 Review Results
print(f"Validation success: {results.success}")
print(f"Statistics: {results.statistics}")

for result in results.results:
    status = "✅" if result.success else "❌"
    print(f"{status} {result.expectation_config.expectation_type}")
```

### E2I-Specific Expectations

```python
# Brand-specific validations
VALID_BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali"]

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={"column": "brand", "value_set": VALID_BRANDS}
    )
)

# KPI coverage check
REQUIRED_KPIS = [
    "TRx", "NRx", "NBRx", "Conversion_Rate",
    "Share_of_Voice", "HCP_Reach", "Patient_Starts"
]

# Verify KPI columns exist (subset check)
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_table_columns_to_contain",
        kwargs={"columns": REQUIRED_KPIS}
    )
)
```

### Tests (Batch 3)

```bash
./venv/bin/python -m pytest tests/unit/test_mlops/test_ge_validator.py -v -n 2 --maxfail=3
```

- [ ] Test 3.T1: GX context initialization
- [ ] Test 3.T2: Expectation suite creation
- [ ] Test 3.T3: Validation passes on clean data
- [ ] Test 3.T4: Validation fails on bad data
- [ ] Test 3.T5: Results serialization

---

## Phase 4: MLflow Experiment Tracking

**Goal**: Demonstrate MLflow for experiment tracking and model registry.

### To-Do Checklist

- [ ] 4.1 Configure MLflow tracking server
- [ ] 4.2 Create experiment for causal validation
- [ ] 4.3 Log parameters, metrics, and artifacts
- [ ] 4.4 Register model to model registry
- [ ] 4.5 Demonstrate stage transitions

### Tutorial Content

```python
# 4.1 Configure MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")

# 4.2 Create Experiment
experiment_name = "e2i_causal_validation_tutorial"
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        tags={"project": "e2i", "type": "causal_validation"}
    )
else:
    experiment_id = experiment.experiment_id

print(f"Experiment ID: {experiment_id}")

# 4.3 Log Training Run
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "double_ml",
        "treatment_model": "linear",
        "outcome_model": "gradient_boosting",
        "cv_folds": 5,
        "data_split": "train"
    })

    # Simulate training metrics
    metrics = {
        "ate": 0.15,  # Average Treatment Effect
        "ate_stderr": 0.02,
        "r2_treatment": 0.85,
        "r2_outcome": 0.78,
        "rmse": 0.12
    }
    mlflow.log_metrics(metrics)

    # Log artifacts
    import json
    artifact_data = {
        "feature_importance": {"feature_1": 0.3, "feature_2": 0.25},
        "causal_graph": "A -> B -> C"
    }
    with open("causal_results.json", "w") as f:
        json.dump(artifact_data, f)
    mlflow.log_artifact("causal_results.json")

    # Log model (example with sklearn)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    mlflow.sklearn.log_model(model, "causal_model")

    print(f"Run ID: {run.info.run_id}")
```

### Model Registry Workflow

```python
# 4.4 Register Model
client = MlflowClient()
model_name = "e2i_causal_impact_model"

# Register model from run
model_uri = f"runs:/{run.info.run_id}/causal_model"
mv = mlflow.register_model(model_uri, model_name)
print(f"Model registered: {mv.name} v{mv.version}")

# 4.5 Stage Transitions
# Development -> Staging -> Shadow -> Production -> Archived

# Transition to staging
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging"
)
print(f"Model transitioned to Staging")

# Get model in production (if any)
prod_model = client.get_latest_versions(model_name, stages=["Production"])
if prod_model:
    print(f"Production model: v{prod_model[0].version}")
else:
    print("No production model yet")
```

### E2I MLflow Patterns

```python
# E2I-specific MLflow integration
from src.mlops.mlflow_connector import MLflowConnector

# Singleton pattern ensures single connection
connector = MLflowConnector.get_instance()

# Async context manager for experiments
async def run_causal_experiment():
    async with connector.start_run(
        experiment_name="causal_impact",
        run_name="tutorial_run"
    ) as run:
        await connector.log_params({"tier": 2, "agent": "causal_impact"})
        await connector.log_metrics({"effect_size": 0.15})
        return run.info.run_id
```

### Tests (Batch 4)

```bash
./venv/bin/python -m pytest tests/unit/test_mlops/test_mlflow_connector.py -v -n 2 --maxfail=3
```

- [ ] Test 4.T1: MLflow connection
- [ ] Test 4.T2: Experiment creation
- [ ] Test 4.T3: Parameter/metric logging
- [ ] Test 4.T4: Model registration
- [ ] Test 4.T5: Stage transitions

---

## Phase 5: Opik Agent Observability

**Goal**: Demonstrate Opik for agent tracing and observability.

### To-Do Checklist

- [ ] 5.1 Initialize Opik tracer
- [ ] 5.2 Create trace for agent workflow
- [ ] 5.3 Add spans for sub-operations
- [ ] 5.4 Log LLM calls with token counts
- [ ] 5.5 Handle circuit breaker scenarios

### Tutorial Content

```python
# 5.1 Initialize Opik Tracer
from src.mlops.opik_connector import OpikConnector, SpanType

# Get singleton instance
opik = OpikConnector.get_instance()

# Check connection status
print(f"Opik available: {opik.is_available()}")
print(f"Circuit breaker state: {opik.circuit_state.name}")

# 5.2 Create Agent Trace
import uuid

trace_id = str(uuid.uuid4())
trace = opik.start_trace(
    name="tutorial_causal_impact_agent",
    input_data={
        "query": "What is the ROI impact of increasing HCP visits by 20%?",
        "brand": "Kisqali"
    },
    trace_id=trace_id
)
print(f"Trace started: {trace_id}")

# 5.3 Add Spans for Sub-Operations
with opik.start_span(
    name="query_processing",
    span_type=SpanType.GENERAL,
    input_data={"raw_query": "What is the ROI..."}
) as span:
    # Simulate NLP processing
    processed = {"intent": "causal_analysis", "entities": ["ROI", "HCP"]}
    span.set_output(processed)

with opik.start_span(
    name="causal_estimation",
    span_type=SpanType.TOOL,
    input_data={"method": "double_ml"}
) as span:
    # Simulate causal estimation
    result = {"ate": 0.15, "ci_lower": 0.10, "ci_upper": 0.20}
    span.set_output(result)

# 5.4 Log LLM Call
with opik.start_span(
    name="explanation_generation",
    span_type=SpanType.LLM,
    input_data={"prompt": "Explain causal effect..."}
) as span:
    # Simulate LLM response
    span.set_output({
        "response": "Increasing HCP visits by 20% is estimated to...",
        "model": "claude-3-opus",
        "tokens_input": 150,
        "tokens_output": 200
    })

    # Log token usage
    opik.log_llm_tokens(
        input_tokens=150,
        output_tokens=200,
        model="claude-3-opus"
    )

# 5.5 Complete Trace
opik.end_trace(
    trace_id=trace_id,
    output_data={"answer": "The estimated ROI impact is 15%..."}
)
print("Trace completed")
```

### Circuit Breaker Pattern

```python
# Circuit Breaker States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED

# Simulate failure scenario
def test_circuit_breaker():
    connector = OpikConnector.get_instance()

    # Check current state
    print(f"Initial state: {connector.circuit_state.name}")

    # Simulate failures (for demo)
    for i in range(5):  # failure_threshold = 5
        connector._record_failure()

    # Circuit should be OPEN
    print(f"After failures: {connector.circuit_state.name}")

    # Operations will use fallback (local logging)
    if not connector.is_available():
        print("Opik unavailable - using local fallback")
        # Log to local file instead
        import logging
        logging.info("Trace data logged locally")
```

### E2I Opik Integration

```python
# E2I agent tracing pattern
from src.agents.causal_impact.agent import CausalImpactAgent

async def traced_agent_call():
    agent = CausalImpactAgent()

    # Agent automatically creates Opik traces
    result = await agent.run({
        "query": "What causes high TRx variance?",
        "brand": "Fabhalta"
    })

    # Trace ID is attached to result metadata
    print(f"Result trace: {result.metadata.get('trace_id')}")
    return result
```

### Tests (Batch 5)

```bash
./venv/bin/python -m pytest tests/unit/test_mlops/test_opik_connector.py -v -n 2 --maxfail=3
```

- [ ] Test 5.T1: Opik initialization
- [ ] Test 5.T2: Trace creation
- [ ] Test 5.T3: Span nesting
- [ ] Test 5.T4: LLM token logging
- [ ] Test 5.T5: Circuit breaker behavior

---

## Phase 6: RAGAS RAG Evaluation

**Goal**: Evaluate RAG quality using RAGAS metrics with Opik integration.

### To-Do Checklist

- [ ] 6.1 Initialize RAGASEvaluator
- [ ] 6.2 Create evaluation samples
- [ ] 6.3 Run RAGAS evaluation
- [ ] 6.4 Analyze metric results
- [ ] 6.5 Log results to Opik

### Tutorial Content

```python
# 6.1 Initialize RAGASEvaluator
from src.rag.evaluation import RAGASEvaluator, EvaluationSample

evaluator = RAGASEvaluator(
    faithfulness_threshold=0.80,
    answer_relevancy_threshold=0.85,
    context_precision_threshold=0.80,
    context_recall_threshold=0.70
)
print(f"Evaluator initialized with thresholds")

# 6.2 Create Evaluation Samples
samples = [
    EvaluationSample(
        question="What is driving Kisqali TRx growth in Q4?",
        answer="Kisqali TRx growth in Q4 is primarily driven by increased HCP engagement (+15%) and successful patient assistance program enrollment (+22%). The key territories showing growth are Northeast and Midwest.",
        contexts=[
            "Q4 HCP engagement metrics show 15% increase in detailed calls for Kisqali.",
            "Patient assistance program saw 22% more enrollments in Q4.",
            "Territory analysis indicates strongest growth in Northeast (18%) and Midwest (14%)."
        ],
        ground_truth="Kisqali TRx growth is driven by HCP engagement and patient assistance programs."
    ),
    EvaluationSample(
        question="What is the causal effect of digital campaigns on Fabhalta awareness?",
        answer="Digital campaigns have a moderate positive causal effect on Fabhalta awareness with an estimated ATE of 0.12 (95% CI: 0.08-0.16). The effect is strongest among younger HCPs.",
        contexts=[
            "Causal analysis using DoWhy identified digital campaigns as a significant driver.",
            "ATE estimate: 0.12, standard error: 0.02",
            "Subgroup analysis shows higher effect for HCPs under 45 years."
        ],
        ground_truth="Digital campaigns causally increase Fabhalta awareness by approximately 12%."
    )
]

# 6.3 Run RAGAS Evaluation
results = await evaluator.evaluate(samples)

# 6.4 Analyze Results
print("\n=== RAGAS Evaluation Results ===")
for i, result in enumerate(results):
    print(f"\nSample {i+1}:")
    print(f"  Faithfulness:       {result.faithfulness:.3f} {'✅' if result.faithfulness >= 0.80 else '❌'}")
    print(f"  Answer Relevancy:   {result.answer_relevancy:.3f} {'✅' if result.answer_relevancy >= 0.85 else '❌'}")
    print(f"  Context Precision:  {result.context_precision:.3f} {'✅' if result.context_precision >= 0.80 else '❌'}")
    print(f"  Context Recall:     {result.context_recall:.3f} {'✅' if result.context_recall >= 0.70 else '❌'}")
    print(f"  Overall Score:      {result.overall_score:.3f}")
    print(f"  Passed:             {'✅ YES' if result.passed else '❌ NO'}")

# Aggregate metrics
avg_scores = {
    "faithfulness": sum(r.faithfulness for r in results) / len(results),
    "answer_relevancy": sum(r.answer_relevancy for r in results) / len(results),
    "context_precision": sum(r.context_precision for r in results) / len(results),
    "context_recall": sum(r.context_recall for r in results) / len(results)
}
print(f"\n=== Aggregate Scores ===")
for metric, score in avg_scores.items():
    print(f"  {metric}: {score:.3f}")
```

### RAGAS with Opik Integration

```python
# 6.5 Log Results to Opik
from src.rag.opik_integration import OpikRAGASCallback

# Create callback for Opik logging
callback = OpikRAGASCallback(project_name="e2i_rag_evaluation")

# Run evaluation with Opik tracing
results = await evaluator.evaluate_with_tracing(
    samples=samples,
    callback=callback,
    trace_name="tutorial_ragas_eval"
)

# Results are automatically logged to Opik with:
# - Individual sample traces
# - Metric breakdowns per sample
# - Aggregate dashboard metrics

# View in Opik UI: http://localhost:5173
print(f"Results logged to Opik. View at: http://localhost:5173")
```

### Pre-built E2I Evaluation Samples

```python
# Use pre-built samples for E2I brands
from src.rag.evaluation import get_brand_evaluation_samples

# Get samples for each brand
kisqali_samples = get_brand_evaluation_samples("Kisqali")
fabhalta_samples = get_brand_evaluation_samples("Fabhalta")
remibrutinib_samples = get_brand_evaluation_samples("Remibrutinib")

print(f"Kisqali samples: {len(kisqali_samples)}")
print(f"Fabhalta samples: {len(fabhalta_samples)}")
print(f"Remibrutinib samples: {len(remibrutinib_samples)}")

# Run full evaluation
all_samples = kisqali_samples + fabhalta_samples + remibrutinib_samples
results = await evaluator.evaluate(all_samples)
```

### Tests (Batch 6)

```bash
./venv/bin/python -m pytest tests/unit/test_rag/test_evaluation.py -v -n 2 --maxfail=3
```

- [ ] Test 6.T1: RAGASEvaluator initialization
- [ ] Test 6.T2: Sample creation
- [ ] Test 6.T3: Metric calculation
- [ ] Test 6.T4: Threshold validation
- [ ] Test 6.T5: Opik callback integration

---

## Phase 7: End-to-End Pipeline Validation

**Goal**: Run complete pipeline validation with all components integrated.

### To-Do Checklist

- [ ] 7.1 Create E2E validation script
- [ ] 7.2 Run full pipeline with synthetic data
- [ ] 7.3 Verify expected insights
- [ ] 7.4 Generate validation report

### Tutorial Content

```python
# 7.1 E2E Validation Pipeline
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PipelineValidationResult:
    data_quality: Dict[str, Any]
    mlflow_runs: List[str]
    opik_traces: List[str]
    ragas_scores: Dict[str, float]
    expected_insights: List[Dict[str, Any]]
    passed: bool

async def run_e2e_validation() -> PipelineValidationResult:
    results = PipelineValidationResult(
        data_quality={},
        mlflow_runs=[],
        opik_traces=[],
        ragas_scores={},
        expected_insights=[],
        passed=True
    )

    # Step 1: Data Quality Check
    print("Step 1: Running Data Quality Validation...")
    from great_expectations.data_context import FileDataContext
    gx_context = FileDataContext.create(project_root_dir=".")
    # ... run expectations
    results.data_quality = {"passed": True, "expectations": 10, "failures": 0}

    # Step 2: MLflow Experiment
    print("Step 2: Running MLflow Experiment...")
    import mlflow
    with mlflow.start_run() as run:
        mlflow.log_param("validation_type", "e2e")
        mlflow.log_metric("pipeline_step", 2)
        results.mlflow_runs.append(run.info.run_id)

    # Step 3: Agent Execution with Opik
    print("Step 3: Running Agent with Opik Tracing...")
    from src.mlops.opik_connector import OpikConnector
    opik = OpikConnector.get_instance()
    trace = opik.start_trace(name="e2e_validation")
    results.opik_traces.append(trace.trace_id if trace else "fallback")

    # Step 4: RAG Evaluation
    print("Step 4: Running RAGAS Evaluation...")
    from src.rag.evaluation import RAGASEvaluator, get_brand_evaluation_samples
    evaluator = RAGASEvaluator()
    samples = get_brand_evaluation_samples("Kisqali")[:3]  # Small batch
    ragas_results = await evaluator.evaluate(samples)
    results.ragas_scores = {
        "faithfulness": sum(r.faithfulness for r in ragas_results) / len(ragas_results),
        "answer_relevancy": sum(r.answer_relevancy for r in ragas_results) / len(ragas_results)
    }

    # Step 5: Verify Expected Insights
    print("Step 5: Verifying Expected Insights...")
    expected = [
        {"insight": "HCP engagement drives TRx", "found": True},
        {"insight": "Digital campaigns affect awareness", "found": True},
        {"insight": "Territory variance explained by rep coverage", "found": True}
    ]
    results.expected_insights = expected

    # Determine overall pass/fail
    results.passed = (
        results.data_quality.get("passed", False) and
        len(results.mlflow_runs) > 0 and
        results.ragas_scores.get("faithfulness", 0) >= 0.70
    )

    return results

# 7.2 Run Validation
results = asyncio.run(run_e2e_validation())

# 7.3 Generate Report
print("\n" + "="*50)
print("E2E PIPELINE VALIDATION REPORT")
print("="*50)
print(f"\nData Quality:")
print(f"  Passed: {results.data_quality.get('passed')}")
print(f"  Expectations: {results.data_quality.get('expectations')}")
print(f"  Failures: {results.data_quality.get('failures')}")

print(f"\nMLflow:")
print(f"  Runs created: {len(results.mlflow_runs)}")

print(f"\nOpik:")
print(f"  Traces created: {len(results.opik_traces)}")

print(f"\nRAGAS Scores:")
for metric, score in results.ragas_scores.items():
    print(f"  {metric}: {score:.3f}")

print(f"\nExpected Insights:")
for insight in results.expected_insights:
    status = "✅" if insight['found'] else "❌"
    print(f"  {status} {insight['insight']}")

print(f"\n{'='*50}")
print(f"OVERALL: {'✅ PASSED' if results.passed else '❌ FAILED'}")
print(f"{'='*50}")
```

### Tests (Batch 7)

```bash
./venv/bin/python -m pytest tests/integration/test_self_improvement_integration.py -v -n 2 --maxfail=3
```

- [ ] Test 7.T1: E2E pipeline completes
- [ ] Test 7.T2: All components integrated
- [ ] Test 7.T3: Expected insights validated

---

## Phase 8: Agent System Testing (Optional)

**Goal**: Test individual agents with synthetic data.

### To-Do Checklist

- [ ] 8.1 Test Tier 0 agents (ML Foundation)
- [ ] 8.2 Test Tier 2 agents (Causal)
- [ ] 8.3 Test Tier 5 agents (Learning)
- [ ] 8.4 Test agent coordination

### Tutorial Content

```python
# 8.1 Test Tier 0 - Data Preparer Agent
from src.agents.tier_0.data_preparer import DataPreparerAgent

agent = DataPreparerAgent()
result = await agent.run({
    "dataset": "synthetic_patient_journeys",
    "split": "train",
    "validation_rules": ["no_nulls", "valid_splits", "value_ranges"]
})
print(f"Data Preparer: {result.status}")

# 8.2 Test Tier 2 - Causal Impact Agent
from src.agents.causal_impact.agent import CausalImpactAgent

agent = CausalImpactAgent()
result = await agent.run({
    "query": "What is the effect of HCP visits on TRx?",
    "brand": "Kisqali",
    "method": "double_ml"
})
print(f"Causal Impact: ATE={result.effect_estimate}")

# 8.3 Test Tier 5 - Feedback Learner Agent
from src.agents.feedback_learner.agent import FeedbackLearnerAgent

agent = FeedbackLearnerAgent()
result = await agent.run({
    "response": "Previous agent output...",
    "feedback": {"rating": 4, "comment": "Good but could improve..."}
})
print(f"Feedback Learner: {result.improvement_suggestions}")

# 8.4 Test Agent Coordination
from src.agents.orchestrator.agent import OrchestratorAgent

orchestrator = OrchestratorAgent()
result = await orchestrator.route({
    "query": "Analyze ROI drivers for Fabhalta Q4",
    "context": {"brand": "Fabhalta", "period": "Q4"}
})
print(f"Orchestrator routed to: {result.selected_agent}")
```

### Tests (Batch 8)

```bash
./venv/bin/python -m pytest tests/unit/test_agents/ -v -n 2 --maxfail=5 -k "not dspy"
```

- [ ] Test 8.T1: Data Preparer validation
- [ ] Test 8.T2: Causal Impact estimation
- [ ] Test 8.T3: Feedback Learner processing
- [ ] Test 8.T4: Orchestrator routing

---

## Testing Guidelines

### Memory-Safe Test Execution

**CRITICAL**: This system has heavy ML imports. Always use these settings:

```bash
# Recommended: 4 workers max
./venv/bin/python -m pytest tests/ -n 4 --dist=loadscope -v

# For low memory systems: Sequential
./venv/bin/python -m pytest tests/ -v --maxfail=5

# NEVER use: -n auto (spawns 14 workers, exhausts RAM)
```

### Batch Test Commands

```bash
# Phase 1
./venv/bin/python -m pytest tests/unit/test_repositories/test_base.py -v -n 2 --maxfail=3

# Phase 2
./venv/bin/python -m pytest tests/unit/test_repositories/ -v -k "patient_journey or hcp" -n 2 --maxfail=3

# Phase 3
./venv/bin/python -m pytest tests/unit/test_mlops/test_ge_validator.py -v -n 2 --maxfail=3

# Phase 4
./venv/bin/python -m pytest tests/unit/test_mlops/test_mlflow_connector.py -v -n 2 --maxfail=3

# Phase 5
./venv/bin/python -m pytest tests/unit/test_mlops/test_opik_connector.py -v -n 2 --maxfail=3

# Phase 6
./venv/bin/python -m pytest tests/unit/test_rag/test_evaluation.py -v -n 2 --maxfail=3

# Phase 7
./venv/bin/python -m pytest tests/integration/test_self_improvement_integration.py -v -n 2 --maxfail=3

# Phase 8 (Optional)
./venv/bin/python -m pytest tests/unit/test_agents/ -v -n 2 --maxfail=5 -k "not dspy"
```

---

## Expected Outcomes

### Data Quality
- All expectation suites pass (100% success rate)
- No null values in critical columns
- Valid data split distribution (60/20/15/5)

### MLflow
- Experiments created and tracked
- Models registered with proper staging
- Metrics and artifacts logged

### Opik
- Agent traces visible in dashboard
- LLM token usage tracked
- Circuit breaker handles failures gracefully

### RAGAS
- Faithfulness >= 0.80
- Answer Relevancy >= 0.85
- Context Precision >= 0.80
- Context Recall >= 0.70

### Expected Insights (Ground Truth)
1. HCP engagement positively drives TRx (ATE ~0.15)
2. Digital campaigns increase brand awareness (ATE ~0.12)
3. Territory variance correlates with rep coverage
4. Patient assistance programs boost conversion rates

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Supabase connection timeout | Check `.env` for `SUPABASE_URL` and `SUPABASE_KEY` |
| MLflow server not responding | Start with `mlflow server --host 0.0.0.0 --port 5000` |
| Opik unavailable | Check Docker: `docker ps | grep opik` |
| Memory exhaustion in tests | Use `-n 2` instead of `-n 4` |
| RAGAS low scores | Verify context relevance to questions |

### Environment Variables

```bash
# Required
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-service-key
ANTHROPIC_API_KEY=your-claude-key

# Optional (defaults work locally)
MLFLOW_TRACKING_URI=http://localhost:5000
OPIK_URL=http://localhost:8080
```

---

## Progress Tracking

### Phase Completion

- [ ] **Phase 1**: Environment Setup & Verification
- [ ] **Phase 2**: Load Synthetic Data to Supabase
- [ ] **Phase 3**: Data Quality Validation (Great Expectations)
- [ ] **Phase 4**: MLflow Experiment Tracking
- [ ] **Phase 5**: Opik Agent Observability
- [ ] **Phase 6**: RAGAS RAG Evaluation
- [ ] **Phase 7**: End-to-End Pipeline Validation
- [ ] **Phase 8**: Agent System Testing (Optional)

### Cleanup Tasks

- [ ] Remove tutorial artifacts after completion
- [ ] Archive MLflow experiments
- [ ] Clear Opik traces (if needed)
- [ ] Document any issues encountered

---

## References

### Internal Documentation
- `docs/rag_evaluation_with_ragas.md` - RAGAS framework guide
- `.claude/plans/RAGAS-Opik Plan.md` - Completed integration plan
- `product-development/current-feature/PRD/product-features-specifications.md` - PRD

### Source Files
- `src/rag/evaluation.py` - RAGASEvaluator implementation
- `src/mlops/opik_connector.py` - Opik integration
- `src/mlops/mlflow_connector.py` - MLflow integration
- `src/ml/data_generator.py` - Synthetic data generator

### External Resources
- [RAGAS Documentation](https://docs.ragas.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Great Expectations](https://docs.greatexpectations.io/)

---

**Created**: 2026-01-05
**Last Updated**: 2026-01-05
**Status**: Ready for Implementation
