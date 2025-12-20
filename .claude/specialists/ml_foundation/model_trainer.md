# CLAUDE.md - Model Trainer Agent

## Overview

The **Model Trainer** executes the ML training pipeline, enforcing data splits, running hyperparameter optimization, and producing trained model artifacts. It integrates with MLflow for experiment tracking and Optuna for hyperparameter tuning.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | Variable (depends on training) |
| **Primary Output** | TrainedModel, ValidationMetrics |
| **Database Table** | `ml_training_runs` |
| **Memory Types** | Working, Episodic, Procedural |
| **MLOps Tools** | MLflow, Optuna, Feast |

## Responsibilities

1. **Split Enforcement**: Ensure strict train/val/test/holdout separation (60/20/15/5)
2. **Preprocessing Isolation**: Fit preprocessing only on training data
3. **Hyperparameter Tuning**: Run Optuna optimization on validation set
4. **Model Training**: Train final model with best hyperparameters
5. **Metric Logging**: Log all metrics to MLflow
6. **Artifact Storage**: Store model artifacts for deployment

## Position in Pipeline

```
┌──────────────────┐
│  model_selector  │
│  (Algorithm)     │
└────────┬─────────┘
         │ ModelCandidate
         ▼
┌──────────────────┐
│  model_trainer   │ ◀── YOU ARE HERE
│  (Training)      │
└────────┬─────────┘
         │ TrainedModel
         ▼
┌──────────────────┐
│ feature_analyzer │
│  (SHAP)          │
└──────────────────┘
```

## Critical: ML Split Enforcement

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SPLITS (STRICTLY ENFORCED)                                │
├─────────────────────────────────────────────────────────────────┤
│  TRAIN (60%)      │ Fit preprocessing, train model              │
│  VALIDATION (20%) │ Hyperparameter tuning, early stopping       │
│  TEST (15%)       │ Final evaluation (NEVER used in training)   │
│  HOLDOUT (5%)     │ Future validation (LOCKED until production) │
└─────────────────────────────────────────────────────────────────┘

⚠️  PREPROCESSING MUST BE FIT ON TRAIN ONLY
⚠️  VALIDATION USED FOR TUNING, NOT TRAINING
⚠️  TEST SET TOUCHED ONLY ONCE FOR FINAL METRICS
⚠️  HOLDOUT IS LOCKED UNTIL POST-DEPLOYMENT
```

## Inputs

### From model_selector

```python
@dataclass
class ModelCandidate:
    model_version_id: str
    algorithm_name: str
    hyperparameter_space: Dict
    default_hyperparameters: Dict
```

### From data_preparer

```python
@dataclass
class QCReport:
    status: DQStatus        # MUST be PASSED or WARNING
```

## Outputs

### TrainedModel

```python
@dataclass
class TrainedModel:
    """Trained model artifact with metadata."""
    training_run_id: str
    model_version_id: str
    experiment_id: str
    
    # Algorithm
    algorithm_name: str
    framework: str
    
    # Hyperparameters
    best_hyperparameters: Dict
    tuning_trials: int
    
    # Artifacts
    model_artifact_uri: str      # MLflow artifact path
    preprocessing_artifact_uri: str
    
    # Training Metadata
    training_samples: int
    training_duration_seconds: float
    training_completed_at: datetime
```

### ValidationMetrics

```python
@dataclass
class ValidationMetrics:
    """Model performance metrics."""
    training_run_id: str
    
    # Primary Metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]       # Final evaluation
    
    # Classification Metrics
    auc: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    
    # Calibration
    brier_score: float
    calibration_error: float
    
    # Threshold Analysis
    optimal_threshold: float
    precision_at_k: Dict[int, float]     # {100: 0.35, 500: 0.28}
    
    # Confidence
    confidence_interval: Dict[str, Tuple[float, float]]
    bootstrap_samples: int
```

## Database Schema

### ml_training_runs Table

```sql
CREATE TABLE ml_training_runs (
    training_run_id TEXT PRIMARY KEY,
    model_version_id TEXT REFERENCES ml_model_registry(model_version_id),
    experiment_id TEXT REFERENCES ml_experiments(experiment_id),
    mlflow_run_id TEXT,
    
    -- Algorithm
    algorithm_name TEXT NOT NULL,
    framework TEXT,
    
    -- Hyperparameters
    hyperparameters JSONB NOT NULL,
    hyperparameter_space JSONB,
    tuning_trials INTEGER,
    
    -- Metrics
    train_metrics JSONB,
    validation_metrics JSONB,
    test_metrics JSONB,
    
    -- Splits Used
    train_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    
    -- Artifacts
    model_artifact_uri TEXT,
    preprocessing_artifact_uri TEXT,
    
    -- Status
    status TEXT DEFAULT 'running',  -- running, completed, failed
    
    -- Timing
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    training_duration_seconds NUMERIC(10,2),
    
    -- Metadata
    trained_by agent_name_enum DEFAULT 'model_trainer'
);

CREATE INDEX idx_training_experiment ON ml_training_runs(experiment_id);
CREATE INDEX idx_training_status ON ml_training_runs(status);
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.mlops.mlflow_client import MLflowClient
from src.mlops.optuna_tuner import OptunaTuner
from src.mlops.feast_client import FeastClient
from src.database.ml_split.split_registry import SplitRegistry
from .training_orchestrator import TrainingOrchestrator
from .split_enforcer import SplitEnforcer

class ModelTrainerAgent(BaseAgent):
    """
    Model Trainer: Execute ML training pipeline with strict split enforcement.
    
    CRITICAL: This agent MUST enforce preprocessing isolation and split integrity.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = None  # Variable based on training
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.mlflow_client = MLflowClient()
        self.optuna_tuner = OptunaTuner()
        self.feast_client = FeastClient()
        self.split_registry = SplitRegistry()
        self.split_enforcer = SplitEnforcer()
        self.training_orchestrator = TrainingOrchestrator()
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Train model with split enforcement.
        
        Steps:
        1. GATE CHECK: Verify QC passed
        2. Load split-aware data
        3. Fit preprocessing (train only)
        4. Run hyperparameter optimization (validation)
        5. Train final model (train)
        6. Evaluate on test set (ONCE)
        7. Log to MLflow
        8. Store artifacts
        """
        model_candidate = state.model_candidate
        experiment_id = state.experiment_id
        
        # Step 1: GATE CHECK - QC must have passed
        qc_report = state.qc_report
        if qc_report.status == DQStatus.FAILED:
            raise QCGateBlockedError(
                f"Cannot train: QC failed with score {qc_report.overall_score}"
            )
        
        # Generate training run ID
        training_run_id = f"run_{experiment_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Step 2: Load split-aware data
        splits = await self._load_splits(experiment_id)
        
        # Validate split sizes
        self.split_enforcer.validate_split_ratios(splits)
        
        # Step 3: Start MLflow run
        with self.mlflow_client.start_run(
            experiment_id=experiment_id,
            run_name=training_run_id
        ) as mlflow_run:
            
            # Step 4: Fit preprocessing on TRAIN ONLY
            preprocessor = await self._fit_preprocessing(
                X_train=splits.train.X,
                y_train=splits.train.y
            )
            
            # Transform all splits
            X_train = preprocessor.transform(splits.train.X)
            X_val = preprocessor.transform(splits.validation.X)
            X_test = preprocessor.transform(splits.test.X)
            
            # Step 5: Hyperparameter optimization (validation set)
            best_params = await self.optuna_tuner.optimize(
                algorithm=model_candidate.algorithm_name,
                search_space=model_candidate.hyperparameter_space,
                X_train=X_train,
                y_train=splits.train.y,
                X_val=X_val,
                y_val=splits.validation.y,
                n_trials=50,
                metric="auc"
            )
            
            # Step 6: Train final model with best params
            model = await self.training_orchestrator.train(
                algorithm=model_candidate.algorithm_name,
                hyperparameters=best_params,
                X_train=X_train,
                y_train=splits.train.y
            )
            
            # Step 7: Evaluate on all splits
            train_metrics = self._evaluate(model, X_train, splits.train.y)
            val_metrics = self._evaluate(model, X_val, splits.validation.y)
            test_metrics = self._evaluate(model, X_test, splits.test.y)  # FINAL
            
            # Step 8: Log to MLflow
            self.mlflow_client.log_params(best_params)
            self.mlflow_client.log_metrics({
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()}
            })
            
            # Step 9: Save artifacts
            model_uri = self.mlflow_client.log_model(model, "model")
            preproc_uri = self.mlflow_client.log_artifact(
                preprocessor, "preprocessor.pkl"
            )
        
        # Step 10: Build outputs
        trained_model = TrainedModel(
            training_run_id=training_run_id,
            model_version_id=model_candidate.model_version_id,
            experiment_id=experiment_id,
            algorithm_name=model_candidate.algorithm_name,
            framework=self._get_framework(model_candidate.algorithm_name),
            best_hyperparameters=best_params,
            tuning_trials=50,
            model_artifact_uri=model_uri,
            preprocessing_artifact_uri=preproc_uri,
            training_samples=len(X_train),
            training_duration_seconds=mlflow_run.duration_seconds,
            training_completed_at=datetime.utcnow()
        )
        
        validation_metrics = ValidationMetrics(
            training_run_id=training_run_id,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            auc=test_metrics["auc"],
            precision=test_metrics["precision"],
            recall=test_metrics["recall"],
            f1=test_metrics["f1"],
            pr_auc=test_metrics["pr_auc"],
            brier_score=test_metrics.get("brier_score"),
            calibration_error=test_metrics.get("calibration_error"),
            optimal_threshold=self._find_optimal_threshold(model, X_val, splits.validation.y),
            precision_at_k=self._compute_precision_at_k(model, X_test, splits.test.y),
            confidence_interval=self._bootstrap_confidence(model, X_test, splits.test.y),
            bootstrap_samples=1000
        )
        
        # Step 11: Persist to database
        await self.training_repo.create(trained_model, validation_metrics)
        
        # Step 12: Update procedural memory with successful config
        await self.procedural_memory.store(
            pattern_type="training_configuration",
            content={
                "algorithm": model_candidate.algorithm_name,
                "best_hyperparameters": best_params,
                "final_auc": test_metrics["auc"],
                "training_samples": len(X_train)
            }
        )
        
        return state.with_updates(
            trained_model=trained_model,
            validation_metrics=validation_metrics
        )
```

### split_enforcer.py

```python
class SplitEnforcer:
    """Enforce ML split integrity."""
    
    EXPECTED_RATIOS = {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.15,
        "holdout": 0.05
    }
    TOLERANCE = 0.02  # Allow 2% deviation
    
    def validate_split_ratios(self, splits: DataSplits) -> None:
        """Validate splits match expected ratios."""
        total = splits.total_samples
        
        for split_name, expected_ratio in self.EXPECTED_RATIOS.items():
            actual_ratio = getattr(splits, split_name).count / total
            
            if abs(actual_ratio - expected_ratio) > self.TOLERANCE:
                raise SplitRatioViolationError(
                    f"{split_name} ratio {actual_ratio:.2f} deviates from "
                    f"expected {expected_ratio:.2f} beyond tolerance"
                )
    
    def verify_no_leakage(self, splits: DataSplits) -> None:
        """Verify no patient/HCP appears in multiple splits."""
        train_ids = set(splits.train.patient_ids)
        val_ids = set(splits.validation.patient_ids)
        test_ids = set(splits.test.patient_ids)
        
        if train_ids & val_ids:
            raise DataLeakageError("Patients in both train and validation")
        if train_ids & test_ids:
            raise DataLeakageError("Patients in both train and test")
        if val_ids & test_ids:
            raise DataLeakageError("Patients in both validation and test")
```

### hyperparameter_tuner.py (Optuna Integration)

```python
# src/mlops/optuna_tuner.py

class OptunaTuner:
    """Optuna hyperparameter optimization."""
    
    async def optimize(
        self,
        algorithm: str,
        search_space: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        metric: str = "auc"
    ) -> Dict:
        """Run hyperparameter optimization."""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    params[param_name] = trial.suggest_float(
                        param_name, param_values[0], param_values[1]
                    )
            
            # Train model
            model = self._create_model(algorithm, params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            return score
        
        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
```

## Preprocessing Isolation

```python
class PreprocessingPipeline:
    """Preprocessing pipeline fitted on train only."""
    
    def __init__(self):
        self.fitted = False
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
        # Statistics (computed from train only)
        self.train_statistics = {}
    
    def fit(self, X_train: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit on training data ONLY."""
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        
        # Fit imputer and scaler on numeric columns
        self.imputer.fit(X_train[numeric_cols])
        X_imputed = self.imputer.transform(X_train[numeric_cols])
        self.scaler.fit(X_imputed)
        
        # Fit encoder on categorical columns
        if len(categorical_cols) > 0:
            self.encoder.fit(X_train[categorical_cols])
        
        # Store statistics
        self.train_statistics = {
            "numeric_means": dict(zip(numeric_cols, self.imputer.statistics_)),
            "numeric_stds": dict(zip(numeric_cols, self.scaler.scale_)),
            "categorical_categories": {
                col: list(cats) for col, cats in 
                zip(categorical_cols, self.encoder.categories_)
            } if len(categorical_cols) > 0 else {}
        }
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using fitted parameters (NO refitting)."""
        if not self.fitted:
            raise ValueError("Pipeline must be fit before transform")
        
        # Transform using statistics from training
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = self.imputer.transform(X[numeric_cols])
        X_scaled = self.scaler.transform(X_numeric)
        
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            X_encoded = self.encoder.transform(X[categorical_cols])
            return np.hstack([X_scaled, X_encoded])
        
        return X_scaled
```

## Error Handling

```python
class ModelTrainerError(AgentError):
    """Base error for model_trainer."""
    pass

class QCGateBlockedError(ModelTrainerError):
    """Training blocked due to QC failure."""
    pass

class SplitRatioViolationError(ModelTrainerError):
    """Split ratios don't match expected values."""
    pass

class DataLeakageError(ModelTrainerError):
    """Data leakage detected across splits."""
    pass

class TrainingFailedError(ModelTrainerError):
    """Model training failed."""
    pass

class HyperparameterTuningError(ModelTrainerError):
    """Optuna optimization failed."""
    pass
```

## Testing

```python
class TestModelTrainer:
    
    async def test_split_enforcement(self):
        """Test splits are correctly enforced."""
        result = await agent.execute(state)
        
        # Verify split ratios
        assert abs(result.trained_model.train_samples / total - 0.60) < 0.02
    
    async def test_preprocessing_isolation(self):
        """Test preprocessing fitted only on train."""
        # This is critical - verify no data from val/test used in preprocessing
        pass
    
    async def test_qc_gate_blocks_training(self):
        """Test training blocked when QC fails."""
        state.qc_report.status = DQStatus.FAILED
        
        with pytest.raises(QCGateBlockedError):
            await agent.execute(state)
    
    async def test_mlflow_logging(self):
        """Test all metrics logged to MLflow."""
        result = await agent.execute(state)
        
        # Verify MLflow run exists
        run = mlflow.get_run(result.trained_model.mlflow_run_id)
        assert "train_auc" in run.data.metrics
        assert "test_auc" in run.data.metrics
```

## Key Principles

1. **QC Gate**: NEVER train if QC failed
2. **Split Isolation**: Train/Val/Test/Holdout strictly separated
3. **Preprocessing Fit**: ONLY on training data
4. **Test Set Once**: Touch test set only for final evaluation
5. **Holdout Locked**: Never use until post-deployment validation
6. **Full Logging**: All params and metrics in MLflow
7. **Artifact Versioning**: Models and preprocessors stored with versions
