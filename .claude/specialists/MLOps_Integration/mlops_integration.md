# CLAUDE.md - MLOps Integration Layer

## Overview

The MLOps layer provides standardized integrations with 7 specialized tools that support the Tier 0 ML Foundation agents. Each tool has a dedicated client module that abstracts complexity and provides consistent interfaces.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 0 AGENTS                                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ scope   │ │  data   │ │ model   │ │ model   │ │feature  │ ...       │
│  │ definer │ │preparer │ │selector │ │ trainer │ │analyzer │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       │           │           │           │           │                 │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────────────┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MLOPS INTEGRATION LAYER                                                │
│                                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ MLflow  │ │  Opik   │ │  Great  │ │  Feast  │ │ Optuna  │           │
│  │         │ │         │ │ Expect. │ │         │ │         │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│                                                                         │
│  ┌─────────┐ ┌─────────┐                                               │
│  │  SHAP   │ │BentoML  │                                               │
│  │         │ │         │                                               │
│  └─────────┘ └─────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/mlops/
├── __init__.py
├── CLAUDE.md                           # This file
│
├── mlflow_client.py                    # Experiment tracking & model registry
├── opik_connector.py                   # LLM/Agent observability
├── great_expectations_validator.py     # Data quality validation
├── feast_client.py                     # Feature store
├── optuna_tuner.py                     # Hyperparameter optimization
├── shap_explainer.py                   # Model interpretability
└── bentoml_service.py                  # Model serving
```

## Tool-Agent Mapping

| Tool | Primary Agents | Purpose |
|------|----------------|---------|
| **MLflow** | model_trainer, model_selector, model_deployer | Experiment tracking, model registry |
| **Opik** | observability_connector, all Hybrid/Deep agents | LLM/agent observability |
| **Great Expectations** | data_preparer | Data quality validation |
| **Feast** | data_preparer, model_trainer, prediction_synthesizer | Feature store |
| **Optuna** | model_trainer | Hyperparameter optimization |
| **SHAP** | feature_analyzer | Model interpretability |
| **BentoML** | model_deployer, prediction_synthesizer | Model serving |

---

## 1. MLflow Client

### Purpose
Experiment tracking, model versioning, and registry management.

### Configuration

```python
# config/mlops.yaml
mlflow:
  tracking_uri: "postgresql://mlflow:mlflow@localhost:5432/mlflow"
  artifact_root: "s3://e2i-mlflow-artifacts"
  experiment_prefix: "e2i_"
  default_tags:
    platform: "e2i"
    environment: "${ENVIRONMENT}"
```

### Implementation

```python
# src/mlops/mlflow_client.py

import mlflow
from mlflow.tracking import MlflowClient
from contextlib import contextmanager
from typing import Dict, Optional, Any

class MLflowClientWrapper:
    """MLflow integration for experiment tracking and model registry."""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        self.client = MlflowClient()
    
    # ═══════════════════════════════════════════════════════════════
    # Experiment Management
    # ═══════════════════════════════════════════════════════════════
    
    def get_or_create_experiment(self, experiment_id: str) -> str:
        """Get or create MLflow experiment."""
        experiment_name = f"{self.config.experiment_prefix}{experiment_id}"
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        
        return mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"{self.config.artifact_root}/{experiment_id}",
            tags=self.config.default_tags
        )
    
    @contextmanager
    def start_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Dict[str, str] = None
    ):
        """Context manager for MLflow runs."""
        mlflow_experiment_id = self.get_or_create_experiment(experiment_id)
        
        with mlflow.start_run(
            experiment_id=mlflow_experiment_id,
            run_name=run_name,
            tags={**self.config.default_tags, **(tags or {})}
        ) as run:
            yield MLflowRunContext(run, self)
    
    # ═══════════════════════════════════════════════════════════════
    # Logging
    # ═══════════════════════════════════════════════════════════════
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to active run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to active run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str = None
    ) -> str:
        """Log model artifact and optionally register."""
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        return model_info.model_uri
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log arbitrary artifact."""
        mlflow.log_artifact(local_path, artifact_path)
    
    # ═══════════════════════════════════════════════════════════════
    # Model Registry
    # ═══════════════════════════════════════════════════════════════
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Dict[str, str] = None
    ) -> str:
        """Register model in registry."""
        result = mlflow.register_model(model_uri, name)
        
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name, result.version, key, value
                )
        
        return result.version
    
    def transition_model_version_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """Transition model version to new stage."""
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def get_latest_version(
        self,
        name: str,
        stages: list = None
    ) -> Optional[str]:
        """Get latest model version for given stages."""
        stages = stages or ["Production", "Staging"]
        versions = self.client.get_latest_versions(name, stages)
        
        if versions:
            return versions[0].version
        return None
    
    def load_model(self, model_uri: str) -> Any:
        """Load model from MLflow."""
        return mlflow.sklearn.load_model(model_uri)


class MLflowRunContext:
    """Context object for active MLflow run."""
    
    def __init__(self, run, client: MLflowClientWrapper):
        self.run = run
        self.client = client
        self.run_id = run.info.run_id
    
    @property
    def artifact_uri(self) -> str:
        return self.run.info.artifact_uri
    
    @property
    def duration_seconds(self) -> float:
        return (self.run.info.end_time - self.run.info.start_time) / 1000
```

### Usage Example

```python
# In model_trainer:

async with self.mlflow_client.start_run(
    experiment_id="exp_remib_northeast_2025",
    run_name="run_20250115_143022",
    tags={"algorithm": "CausalForest", "brand": "Remibrutinib"}
) as mlflow_run:
    
    # Log hyperparameters
    self.mlflow_client.log_params(best_params)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    self.mlflow_client.log_metrics({
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc
    })
    
    # Log model
    model_uri = self.mlflow_client.log_model(
        model, 
        "model",
        registered_model_name=f"e2i_{experiment_id}"
    )
```

---

## 2. Opik Connector

### Purpose
LLM and agent observability, trace collection, and quality monitoring.

### Configuration

```python
# config/mlops.yaml
opik:
  api_key: "${OPIK_API_KEY}"
  project_name: "e2i-causal-analytics"
  environment: "${ENVIRONMENT}"
  sampling_rate: 1.0  # Sample all traces in dev
```

### Implementation

```python
# src/mlops/opik_connector.py

from opik import Opik, trace
from opik.integrations.anthropic import track_anthropic
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

class OpikConnector:
    """Opik integration for LLM/agent observability."""
    
    def __init__(self, config: OpikConfig):
        self.config = config
        self.opik = Opik(
            api_key=config.api_key,
            project_name=config.project_name
        )
        
        # Enable Anthropic auto-tracking
        track_anthropic()
    
    # ═══════════════════════════════════════════════════════════════
    # Tracing
    # ═══════════════════════════════════════════════════════════════
    
    @asynccontextmanager
    async def trace_agent(
        self,
        agent_name: str,
        operation: str,
        trace_id: str = None,
        parent_span_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Trace an agent operation."""
        with self.opik.trace(
            name=f"{agent_name}.{operation}",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            metadata={
                "agent_name": agent_name,
                "operation": operation,
                "environment": self.config.environment,
                **(metadata or {})
            }
        ) as span:
            yield OpikSpanContext(span)
    
    @asynccontextmanager
    async def trace_llm_call(
        self,
        model: str,
        parent_span: "OpikSpanContext",
        prompt_template: str = None
    ):
        """Trace an LLM call within an agent operation."""
        with parent_span.span.child(
            name="llm_call",
            metadata={
                "model": model,
                "prompt_template": prompt_template
            }
        ) as llm_span:
            yield OpikLLMSpanContext(llm_span)
    
    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════
    
    def log_metric(
        self,
        name: str,
        value: float,
        span: "OpikSpanContext" = None
    ):
        """Log a metric, optionally attached to a span."""
        if span:
            span.span.log_metric(name, value)
        else:
            self.opik.log_metric(name, value)
    
    def log_feedback(
        self,
        trace_id: str,
        score: float,
        feedback_type: str = "quality"
    ):
        """Log feedback for a trace (e.g., from user ratings)."""
        self.opik.log_feedback(
            trace_id=trace_id,
            score=score,
            feedback_type=feedback_type
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════
    
    async def evaluate_agent(
        self,
        agent_name: str,
        test_cases: list,
        metrics: list
    ) -> Dict[str, float]:
        """Run evaluation suite for an agent."""
        results = await self.opik.evaluate(
            name=f"{agent_name}_evaluation",
            test_cases=test_cases,
            metrics=metrics
        )
        return results.to_dict()


class OpikSpanContext:
    """Context for Opik span."""
    
    def __init__(self, span):
        self.span = span
        self.span_id = span.span_id
        self.trace_id = span.trace_id
    
    def set_attribute(self, key: str, value: Any):
        self.span.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Dict = None):
        self.span.add_event(name, attributes=attributes)
    
    def set_status(self, status: str, message: str = None):
        self.span.set_status(status, message)


class OpikLLMSpanContext(OpikSpanContext):
    """Context for LLM call span with token tracking."""
    
    def log_tokens(self, input_tokens: int, output_tokens: int):
        self.span.set_attribute("input_tokens", input_tokens)
        self.span.set_attribute("output_tokens", output_tokens)
        self.span.set_attribute("total_tokens", input_tokens + output_tokens)
```

### Usage Example

```python
# In feature_analyzer (Hybrid agent):

async with self.opik.trace_agent(
    agent_name="feature_analyzer",
    operation="interpret_shap",
    trace_id=ctx.trace_id,
    metadata={"experiment_id": experiment_id}
) as span:
    
    # Computation (no LLM)
    shap_values = self._compute_shap(model, X)
    span.add_event("shap_computed", {"samples": len(X)})
    
    # LLM interpretation
    async with self.opik.trace_llm_call(
        model="claude-sonnet-4-20250514",
        parent_span=span
    ) as llm_span:
        response = await anthropic.messages.create(...)
        llm_span.log_tokens(
            response.usage.input_tokens,
            response.usage.output_tokens
        )
```

---

## 3. Great Expectations Validator

### Purpose
Data quality validation with expectation suites.

### Configuration

```python
# config/mlops.yaml
great_expectations:
  context_root: "/home/claude/e2i/great_expectations"
  datasource_name: "e2i_supabase"
  default_suites:
    classification: "e2i_classification_suite"
    regression: "e2i_regression_suite"
```

### Implementation

```python
# src/mlops/great_expectations_validator.py

import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.checkpoint import Checkpoint
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class GEValidationResult:
    """Result of Great Expectations validation."""
    success: bool
    statistics: Dict
    results: List[Dict]
    duration_seconds: float
    
    @property
    def passed_expectations(self) -> int:
        return self.statistics.get("successful_expectations", 0)
    
    @property
    def failed_expectations(self) -> int:
        return self.statistics.get("unsuccessful_expectations", 0)


class GEValidator:
    """Great Expectations integration for data validation."""
    
    def __init__(self, config: GEConfig):
        self.config = config
        self.context = ge.get_context(context_root_dir=config.context_root)
    
    # ═══════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════
    
    async def validate(
        self,
        data_source: str,
        suite_name: str,
        split_filter: str = None,
        batch_parameters: Dict = None
    ) -> GEValidationResult:
        """Run validation against expectation suite."""
        import time
        start = time.time()
        
        # Get batch
        batch_request = {
            "datasource_name": self.config.datasource_name,
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": data_source,
            "batch_identifiers": batch_parameters or {}
        }
        
        if split_filter:
            batch_request["batch_filter_parameters"] = {"split": split_filter}
        
        # Run validation
        checkpoint = Checkpoint(
            name="validation_checkpoint",
            data_context=self.context,
            config_version=1,
            validations=[{
                "batch_request": batch_request,
                "expectation_suite_name": suite_name
            }]
        )
        
        results = checkpoint.run()
        duration = time.time() - start
        
        # Parse results
        validation_result = list(results.run_results.values())[0]
        
        return GEValidationResult(
            success=validation_result.success,
            statistics=validation_result.statistics,
            results=[
                {
                    "expectation": r.expectation_config.expectation_type,
                    "success": r.success,
                    "kwargs": r.expectation_config.kwargs,
                    "result": r.result
                }
                for r in validation_result.results
            ],
            duration_seconds=duration
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Suite Management
    # ═══════════════════════════════════════════════════════════════
    
    def create_suite(
        self,
        suite_name: str,
        expectations: List[Dict]
    ) -> ExpectationSuite:
        """Create a new expectation suite."""
        suite = self.context.create_expectation_suite(suite_name)
        
        for exp in expectations:
            suite.add_expectation(
                expectation_configuration=ge.core.ExpectationConfiguration(
                    expectation_type=exp["type"],
                    kwargs=exp["kwargs"]
                )
            )
        
        self.context.save_expectation_suite(suite)
        return suite
    
    def get_default_suite(self, problem_type: str) -> str:
        """Get default suite name for problem type."""
        return self.config.default_suites.get(problem_type, "e2i_default_suite")


# Default E2I Expectation Suites
E2I_CLASSIFICATION_EXPECTATIONS = [
    # Completeness
    {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "hcp_id"}},
    {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "brand"}},
    {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "target"}},
    
    # Validity
    {"type": "expect_column_values_to_be_in_set", 
     "kwargs": {"column": "brand", "value_set": ["Remibrutinib", "Fabhalta", "Kisqali"]}},
    {"type": "expect_column_values_to_be_between",
     "kwargs": {"column": "call_frequency", "min_value": 0, "max_value": 100}},
    {"type": "expect_column_values_to_be_in_set",
     "kwargs": {"column": "target", "value_set": [0, 1]}},
    
    # Uniqueness
    {"type": "expect_compound_columns_to_be_unique",
     "kwargs": {"column_list": ["hcp_id", "observation_date"]}},
    
    # Consistency
    {"type": "expect_column_pair_values_A_to_be_greater_than_B",
     "kwargs": {"column_A": "last_call_date", "column_B": "first_call_date",
                "or_equal": True, "ignore_row_if": "either_value_is_missing"}},
    
    # Distribution
    {"type": "expect_column_proportion_of_unique_values_to_be_between",
     "kwargs": {"column": "hcp_id", "min_value": 0.01, "max_value": 1.0}},
]
```

### Usage Example

```python
# In data_preparer:

# Validate training data
ge_results = await self.ge_validator.validate(
    data_source="patient_journeys",
    suite_name="e2i_classification_suite",
    split_filter="train"
)

if not ge_results.success:
    failed = [r for r in ge_results.results if not r["success"]]
    raise QCFailedError(f"Validation failed: {failed}")
```

---

## 4. Feast Client

### Purpose
Feature store for feature registration, retrieval, and serving.

### Implementation

```python
# src/mlops/feast_client.py

from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Int64, Float64, String
from typing import Dict, List, Optional
import pandas as pd

class FeastClient:
    """Feast feature store integration."""
    
    def __init__(self, config: FeastConfig):
        self.config = config
        self.store = FeatureStore(repo_path=config.repo_path)
    
    # ═══════════════════════════════════════════════════════════════
    # Feature Registration
    # ═══════════════════════════════════════════════════════════════
    
    def register_feature(
        self,
        feature_name: str,
        feature_group: str,
        entity: str,
        dtype: str,
        description: str = None
    ):
        """Register a feature definition."""
        # This would typically be done via feature_store.yaml
        # but can be done programmatically for dynamic features
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # Feature Retrieval
    # ═══════════════════════════════════════════════════════════════
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        full_feature_names: bool = False
    ) -> pd.DataFrame:
        """Get historical features for training."""
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=full_feature_names
        ).to_df()
    
    def get_online_features(
        self,
        entity_rows: List[Dict],
        features: List[str]
    ) -> Dict:
        """Get online features for inference."""
        return self.store.get_online_features(
            entity_rows=entity_rows,
            features=features
        ).to_dict()
    
    # ═══════════════════════════════════════════════════════════════
    # Feature Statistics
    # ═══════════════════════════════════════════════════════════════
    
    def get_feature_statistics(
        self,
        feature_view: str
    ) -> Dict:
        """Get statistics for a feature view."""
        # Query feature statistics from feature store
        pass
```

---

## 5. Optuna Tuner

### Purpose
Hyperparameter optimization with Bayesian search.

### Implementation

```python
# src/mlops/optuna_tuner.py

import optuna
from optuna.integration import MLflowCallback
from typing import Dict, Callable, Any, List
import numpy as np

class OptunaTuner:
    """Optuna hyperparameter optimization."""
    
    def __init__(self, config: OptunaConfig):
        self.config = config
        self.study_storage = config.study_storage
    
    async def optimize(
        self,
        algorithm: str,
        search_space: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        metric: str = "auc",
        direction: str = "maximize",
        timeout: int = None
    ) -> Dict:
        """Run hyperparameter optimization."""
        
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial, search_space)
            
            # Create and train model
            model = self._create_model(algorithm, params)
            model.fit(X_train, y_train)
            
            # Evaluate
            score = self._evaluate(model, X_val, y_val, metric)
            
            return score
        
        # Create study
        study = optuna.create_study(
            study_name=f"{algorithm}_tuning",
            storage=self.study_storage,
            direction=direction,
            load_if_exists=True
        )
        
        # Add MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config.mlflow_tracking_uri,
            metric_name=metric
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=True
        )
        
        return study.best_params
    
    def _sample_params(
        self,
        trial: optuna.Trial,
        search_space: Dict
    ) -> Dict:
        """Sample parameters from search space."""
        params = {}
        
        for name, space in search_space.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple):
                if isinstance(space[0], int):
                    params[name] = trial.suggest_int(name, space[0], space[1])
                else:
                    params[name] = trial.suggest_float(name, space[0], space[1])
            elif isinstance(space, dict):
                if space.get("log"):
                    params[name] = trial.suggest_float(
                        name, space["low"], space["high"], log=True
                    )
        
        return params
    
    def _create_model(self, algorithm: str, params: Dict) -> Any:
        """Create model instance with parameters."""
        from src.agents.ml_foundation.model_selector.algorithm_registry import ALGORITHM_REGISTRY
        
        algo_info = ALGORITHM_REGISTRY.get(algorithm)
        framework = algo_info["framework"]
        
        if framework == "econml":
            from econml.dml import CausalForestDML, LinearDML
            if algorithm == "CausalForest":
                return CausalForestDML(**params)
            elif algorithm == "LinearDML":
                return LinearDML(**params)
        elif framework == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        elif framework == "sklearn":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        
        raise ValueError(f"Unknown algorithm: {algorithm}")
```

---

## 6. SHAP Explainer

### Purpose
Model interpretability via SHAP values.

### Implementation

```python
# src/mlops/shap_explainer.py

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class SHAPExplainer:
    """SHAP model interpretability."""
    
    def __init__(self, config: SHAPConfig = None):
        self.config = config or SHAPConfig()
    
    def compute_shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
        algorithm: str = None
    ) -> Tuple[np.ndarray, float]:
        """Compute SHAP values for model predictions."""
        
        # Select appropriate explainer
        explainer = self._get_explainer(model, X, algorithm)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle binary classification (list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        return shap_values, expected_value
    
    def _get_explainer(
        self,
        model: Any,
        X: pd.DataFrame,
        algorithm: str
    ) -> shap.Explainer:
        """Get appropriate SHAP explainer for model type."""
        
        # Tree-based models
        if algorithm in ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]:
            return shap.TreeExplainer(model)
        
        # Linear models
        if algorithm in ["LogisticRegression", "LinearRegression", "Ridge", "Lasso"]:
            return shap.LinearExplainer(model, X)
        
        # EconML models (need KernelExplainer)
        if algorithm in ["CausalForest", "LinearDML"]:
            background = shap.sample(X, min(100, len(X)))
            return shap.KernelExplainer(model.predict, background)
        
        # Default to KernelExplainer
        background = shap.sample(X, min(100, len(X)))
        return shap.KernelExplainer(model.predict_proba, background)
    
    def compute_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute global feature importance from SHAP values."""
        importance = np.abs(shap_values).mean(axis=0)
        importance = importance / importance.sum()  # Normalize
        return dict(zip(feature_names, importance.tolist()))
    
    def compute_interactions(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """Compute SHAP interaction values."""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_interaction_values(X)
    
    def explain_instance(
        self,
        model: Any,
        X: pd.DataFrame,
        instance_idx: int
    ) -> Dict[str, float]:
        """Explain a single prediction."""
        shap_values, _ = self.compute_shap_values(model, X.iloc[[instance_idx]])
        return dict(zip(X.columns, shap_values[0].tolist()))
```

---

## 7. BentoML Service

### Purpose
Model packaging and serving endpoints.

### Implementation

```python
# src/mlops/bentoml_service.py

import bentoml
from bentoml.io import JSON, NumpyNdarray
from typing import Dict, Any, List
import numpy as np

class BentoMLService:
    """BentoML model serving integration."""
    
    def __init__(self, config: BentoMLConfig):
        self.config = config
    
    # ═══════════════════════════════════════════════════════════════
    # Model Packaging
    # ═══════════════════════════════════════════════════════════════
    
    async def package_model(
        self,
        model_uri: str,
        preprocessing_uri: str,
        name: str,
        version: str
    ) -> str:
        """Package model as BentoML service."""
        
        # Load model and preprocessor
        model = self._load_from_mlflow(model_uri)
        preprocessor = self._load_from_mlflow(preprocessing_uri)
        
        # Save to BentoML model store
        bento_model = bentoml.sklearn.save_model(
            name,
            model,
            custom_objects={"preprocessor": preprocessor},
            signatures={"predict": {"batchable": True}}
        )
        
        # Build Bento
        bento = bentoml.build(
            service=f"{name}_service:svc",
            version=version,
            labels={
                "platform": "e2i",
                "model_uri": model_uri
            }
        )
        
        return bento.tag
    
    # ═══════════════════════════════════════════════════════════════
    # Deployment
    # ═══════════════════════════════════════════════════════════════
    
    async def deploy(
        self,
        bento_tag: str,
        endpoint_name: str,
        replicas: int = 1,
        resources: Dict = None
    ) -> "Endpoint":
        """Deploy Bento to endpoint."""
        
        resources = resources or {
            "cpu": "2",
            "memory": "4Gi"
        }
        
        deployment = bentoml.deployment.create(
            name=endpoint_name,
            bento=bento_tag,
            scaling={
                "replicas": replicas,
                "min_replicas": 1,
                "max_replicas": 10
            },
            envs=[
                {"name": "E2I_ENVIRONMENT", "value": self.config.environment}
            ],
            resources=resources
        )
        
        # Wait for deployment
        endpoint_url = await self._wait_for_ready(deployment)
        
        return Endpoint(
            name=endpoint_name,
            url=endpoint_url,
            bento_tag=bento_tag,
            replicas=replicas
        )
    
    async def update_traffic(
        self,
        endpoint_name: str,
        traffic_percentage: int
    ):
        """Update traffic routing to endpoint."""
        bentoml.deployment.update(
            name=endpoint_name,
            traffic_percentage=traffic_percentage
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════════════════════════
    
    async def predict(
        self,
        endpoint_url: str,
        features: np.ndarray
    ) -> np.ndarray:
        """Call prediction endpoint."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint_url}/predict",
                json={"input": features.tolist()}
            )
            return np.array(response.json()["output"])


# Example BentoML Service Definition
# This would be generated dynamically or templated

"""
# e2i_model_service.py

import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray

runner = bentoml.sklearn.get("e2i_remib_model:latest").to_runner()
svc = bentoml.Service("e2i_remib_service", runners=[runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_array: np.ndarray) -> dict:
    # Get preprocessor
    preprocessor = runner.model.custom_objects["preprocessor"]
    
    # Preprocess
    X_processed = preprocessor.transform(input_array)
    
    # Predict
    predictions = await runner.predict.async_run(X_processed)
    probabilities = await runner.predict_proba.async_run(X_processed)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities[:, 1].tolist()
    }
"""
```

---

## Environment Setup

### Required Environment Variables

```bash
# MLflow
export MLFLOW_TRACKING_URI="postgresql://mlflow:password@localhost:5432/mlflow"
export MLFLOW_ARTIFACT_ROOT="s3://e2i-mlflow-artifacts"

# Opik
export OPIK_API_KEY="your-opik-api-key"

# Feast
export FEAST_REPO_PATH="/home/claude/e2i/feature_repo"

# BentoML
export BENTOML_HOME="/home/claude/.bentoml"

# General
export ENVIRONMENT="development"  # development | staging | production
```

### Dependencies

```
# requirements-mlops.txt
mlflow>=2.10.0
opik>=0.1.0
great-expectations>=0.18.0
feast>=0.35.0
optuna>=3.5.0
shap>=0.44.0
bentoml>=1.2.0
```

---

## Integration Testing

```python
# tests/integration/test_mlops_integration.py

class TestMLOpsIntegration:
    
    async def test_full_training_pipeline(self):
        """Test MLflow → Optuna → SHAP → BentoML pipeline."""
        
        # 1. Start MLflow run
        with mlflow_client.start_run(experiment_id, run_name) as run:
            
            # 2. Validate data with Great Expectations
            ge_result = await ge_validator.validate(data_source, suite_name)
            assert ge_result.success
            
            # 3. Get features from Feast
            features = feast_client.get_historical_features(entity_df, feature_list)
            
            # 4. Tune hyperparameters with Optuna
            best_params = await optuna_tuner.optimize(
                algorithm="XGBoost",
                search_space=search_space,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
            
            # 5. Train and log model
            model.fit(X_train, y_train)
            model_uri = mlflow_client.log_model(model, "model")
            
            # 6. Compute SHAP
            shap_values, _ = shap_explainer.compute_shap_values(model, X_val)
            
            # 7. Package with BentoML
            bento_tag = await bentoml_service.package_model(
                model_uri, preprocessing_uri, name, version
            )
        
        assert bento_tag is not None
```

---

## Key Principles

1. **Abstraction**: All tools wrapped with consistent interfaces
2. **Configuration**: All settings externalized to config files
3. **Error Handling**: Graceful degradation when tools unavailable
4. **Observability**: All operations traced via Opik
5. **Versioning**: All artifacts versioned and tracked
6. **Testing**: Integration tests for full pipelines
