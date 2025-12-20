# CLAUDE.md - Model Selector Agent

## Overview

The **Model Selector** evaluates candidate algorithms against the problem scope and recommends the optimal model architecture. It compares baseline approaches, considers computational constraints, and registers candidates in the model registry.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | <120 seconds |
| **Primary Output** | ModelCandidate, SelectionRationale |
| **Database Table** | `ml_model_registry` |
| **Memory Types** | Working, Episodic, Procedural |
| **MLOps Tools** | MLflow |

## Responsibilities

1. **Algorithm Evaluation**: Assess candidate algorithms for the problem type
2. **Baseline Comparison**: Compare against simple baselines (random, heuristic)
3. **Constraint Matching**: Filter by computational/latency constraints
4. **Hyperparameter Space**: Define initial hyperparameter search space
5. **Model Registration**: Register candidates in MLflow registry

## Position in Pipeline

```
┌──────────────────┐
│  data_preparer   │
│  (QC: PASSED)    │
└────────┬─────────┘
         │ QCReport
         ▼
┌──────────────────┐
│  model_selector  │ ◀── YOU ARE HERE
│  (Algorithm)     │
└────────┬─────────┘
         │ ModelCandidate
         ▼
┌──────────────────┐
│  model_trainer   │
│  (Training)      │
└──────────────────┘
```

## Inputs

### From scope_definer

```python
@dataclass
class ScopeSpec:
    experiment_id: str
    problem_type: str                 # classification | regression | ranking
    prediction_target: str
    technical_constraints: List[str]  # ["inference_latency_<100ms"]
```

### From data_preparer

```python
@dataclass
class QCReport:
    status: DQStatus                  # Must be PASSED or WARNING
    row_count: int
    column_count: int
```

## Outputs

### ModelCandidate

```python
@dataclass
class ModelCandidate:
    """Selected algorithm with configuration."""
    model_version_id: str
    experiment_id: str
    
    # Algorithm
    algorithm_name: str               # "CausalForest", "LinearDML", "XGBoost"
    algorithm_family: str             # "causal_ml", "gradient_boosting", "linear"
    
    # Configuration
    hyperparameter_space: Dict        # Search space for Optuna
    default_hyperparameters: Dict     # Starting point
    
    # Constraints
    estimated_training_time: str      # "~30 minutes"
    estimated_inference_latency_ms: int
    memory_requirement_gb: float
    
    # Baselines
    baseline_candidates: List[str]    # ["random", "logistic_regression"]
    
    # Metadata
    selection_score: float            # 0-1 confidence
    stage: ModelStage                 # development
    created_at: datetime
```

### SelectionRationale

```python
@dataclass
class SelectionRationale:
    """Explanation for algorithm selection."""
    model_version_id: str
    
    # Reasoning
    primary_reason: str               # "Best for heterogeneous treatment effects"
    supporting_factors: List[str]
    alternatives_considered: List[AlternativeModel]
    
    # Constraints Evaluation
    constraint_compliance: Dict[str, bool]
    # {"inference_latency_<100ms": True, "memory_<8gb": True}
    
    # Similar Past Selections
    similar_experiments: List[str]    # Past experiments with similar scope
    success_rate_of_algorithm: float  # Historical success rate
```

## Database Schema

### ml_model_registry Table

```sql
CREATE TABLE ml_model_registry (
    model_version_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    experiment_id TEXT REFERENCES ml_experiments(experiment_id),
    
    -- Algorithm
    algorithm_name TEXT NOT NULL,
    algorithm_family TEXT,
    framework TEXT,                    -- "econml", "sklearn", "xgboost"
    
    -- Version
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Configuration
    hyperparameters JSONB,
    hyperparameter_space JSONB,
    
    -- Stage
    stage model_stage_enum DEFAULT 'development',
    -- development, staging, shadow, production, archived, deprecated
    
    -- Performance (populated after training)
    metrics JSONB,
    validation_metrics JSONB,
    
    -- Artifacts
    artifact_uri TEXT,                 -- MLflow artifact path
    
    -- Metadata
    registered_by agent_name_enum DEFAULT 'model_selector',
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    promoted_by agent_name_enum
);

-- Index for fast stage lookups
CREATE INDEX idx_model_stage ON ml_model_registry(model_name, stage);
```

## Algorithm Registry

### Supported Algorithms

```python
ALGORITHM_REGISTRY = {
    # Causal ML (DoWhy/EconML)
    "CausalForest": {
        "family": "causal_ml",
        "framework": "econml",
        "problem_types": ["classification", "regression"],
        "strengths": ["heterogeneous effects", "interpretability"],
        "inference_latency_ms": 50,
        "memory_gb": 4,
        "hyperparameter_space": {
            "n_estimators": [100, 500, 1000],
            "max_depth": [5, 10, 15, None],
            "min_samples_leaf": [5, 10, 20]
        }
    },
    "LinearDML": {
        "family": "causal_ml",
        "framework": "econml",
        "problem_types": ["classification", "regression"],
        "strengths": ["fast", "interpretable", "low variance"],
        "inference_latency_ms": 10,
        "memory_gb": 1,
        "hyperparameter_space": {
            "model_y": ["Ridge", "Lasso", "ElasticNet"],
            "model_t": ["LogisticRegression", "RandomForest"],
            "cv": [3, 5]
        }
    },
    
    # Gradient Boosting
    "XGBoost": {
        "family": "gradient_boosting",
        "framework": "xgboost",
        "problem_types": ["classification", "regression", "ranking"],
        "strengths": ["accuracy", "feature importance"],
        "inference_latency_ms": 20,
        "memory_gb": 2,
        "hyperparameter_space": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 0.8, 0.9]
        }
    },
    "LightGBM": {
        "family": "gradient_boosting",
        "framework": "lightgbm",
        "problem_types": ["classification", "regression", "ranking"],
        "strengths": ["speed", "large datasets"],
        "inference_latency_ms": 15,
        "memory_gb": 1.5
    },
    
    # Linear Models (Baselines)
    "LogisticRegression": {
        "family": "linear",
        "framework": "sklearn",
        "problem_types": ["classification"],
        "strengths": ["interpretable", "fast", "baseline"],
        "inference_latency_ms": 1,
        "memory_gb": 0.1
    },
    "Ridge": {
        "family": "linear",
        "framework": "sklearn",
        "problem_types": ["regression"],
        "strengths": ["interpretable", "fast", "baseline"],
        "inference_latency_ms": 1,
        "memory_gb": 0.1
    }
}
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.mlops.mlflow_client import MLflowClient
from src.database.repositories.ml_model_registry import MLModelRegistryRepository
from .algorithm_registry import ALGORITHM_REGISTRY
from .baseline_comparator import BaselineComparator

class ModelSelectorAgent(BaseAgent):
    """
    Model Selector: Evaluate and recommend algorithms.
    
    Considers problem type, constraints, and historical success rates.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 120
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.mlflow_client = MLflowClient()
        self.model_repo = MLModelRegistryRepository()
        self.baseline_comparator = BaselineComparator()
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Select optimal algorithm.
        
        Steps:
        1. Filter algorithms by problem type
        2. Filter by technical constraints
        3. Rank by historical success rate
        4. Select top candidate
        5. Define hyperparameter space
        6. Register in MLflow
        """
        scope_spec = state.scope_spec
        qc_report = state.qc_report
        
        # Step 1: Filter by problem type
        candidates = self._filter_by_problem_type(
            scope_spec.problem_type
        )
        
        # Step 2: Filter by constraints
        candidates = self._filter_by_constraints(
            candidates,
            scope_spec.technical_constraints
        )
        
        # Step 3: Retrieve historical success rates
        success_rates = await self._get_historical_success_rates(
            candidates,
            scope_spec.brand,
            scope_spec.problem_type
        )
        
        # Step 4: Rank and select
        ranked = self._rank_candidates(candidates, success_rates, qc_report)
        selected = ranked[0]
        
        # Step 5: Build model candidate
        model_candidate = self._build_candidate(
            selected,
            scope_spec.experiment_id
        )
        
        # Step 6: Generate rationale
        rationale = self._generate_rationale(
            selected=selected,
            alternatives=ranked[1:3],
            success_rates=success_rates,
            constraints=scope_spec.technical_constraints
        )
        
        # Step 7: Register in MLflow
        await self.mlflow_client.register_model(
            name=f"{scope_spec.experiment_id}_{selected['name']}",
            tags={
                "experiment_id": scope_spec.experiment_id,
                "algorithm": selected["name"],
                "stage": "development"
            }
        )
        
        # Step 8: Persist to database
        await self.model_repo.create(model_candidate)
        
        # Step 9: Update procedural memory
        await self.procedural_memory.store(
            pattern_type="algorithm_selection",
            content={
                "problem_type": scope_spec.problem_type,
                "selected_algorithm": selected["name"],
                "data_size": qc_report.row_count
            }
        )
        
        return state.with_updates(
            model_candidate=model_candidate,
            selection_rationale=rationale
        )
    
    def _filter_by_problem_type(self, problem_type: str) -> List[Dict]:
        """Filter algorithms supporting this problem type."""
        return [
            {**algo, "name": name}
            for name, algo in ALGORITHM_REGISTRY.items()
            if problem_type in algo["problem_types"]
        ]
    
    def _filter_by_constraints(
        self,
        candidates: List[Dict],
        constraints: List[str]
    ) -> List[Dict]:
        """Filter by technical constraints."""
        filtered = candidates
        
        for constraint in constraints:
            if "latency" in constraint.lower():
                # Parse "inference_latency_<100ms"
                max_latency = int(constraint.split("<")[1].replace("ms", ""))
                filtered = [c for c in filtered 
                           if c["inference_latency_ms"] <= max_latency]
            
            if "memory" in constraint.lower():
                # Parse "memory_<8gb"
                max_memory = float(constraint.split("<")[1].replace("gb", ""))
                filtered = [c for c in filtered 
                           if c["memory_gb"] <= max_memory]
        
        return filtered
    
    async def _get_historical_success_rates(
        self,
        candidates: List[Dict],
        brand: str,
        problem_type: str
    ) -> Dict[str, float]:
        """Get historical success rates from procedural memory."""
        rates = {}
        
        for candidate in candidates:
            patterns = await self.procedural_memory.retrieve(
                query=f"{candidate['name']} {problem_type}",
                filters={"brand": brand},
                limit=10
            )
            
            if patterns:
                successes = sum(1 for p in patterns if p.get("success", False))
                rates[candidate["name"]] = successes / len(patterns)
            else:
                # Default rate for new algorithms
                rates[candidate["name"]] = 0.5
        
        return rates
    
    def _rank_candidates(
        self,
        candidates: List[Dict],
        success_rates: Dict[str, float],
        qc_report: QCReport
    ) -> List[Dict]:
        """Rank candidates by composite score."""
        
        for candidate in candidates:
            # Composite scoring
            score = 0.0
            
            # Historical success (40%)
            score += success_rates.get(candidate["name"], 0.5) * 0.4
            
            # Inference speed (20%)
            latency_score = 1 - (candidate["inference_latency_ms"] / 100)
            score += max(0, latency_score) * 0.2
            
            # Memory efficiency (15%)
            memory_score = 1 - (candidate["memory_gb"] / 8)
            score += max(0, memory_score) * 0.15
            
            # Interpretability bonus (15%)
            if "interpretable" in candidate.get("strengths", []):
                score += 0.15
            
            # Causal ML bonus for E2I (10%)
            if candidate["family"] == "causal_ml":
                score += 0.10
            
            candidate["selection_score"] = score
        
        return sorted(candidates, key=lambda x: x["selection_score"], reverse=True)
    
    def _build_candidate(
        self,
        selected: Dict,
        experiment_id: str
    ) -> ModelCandidate:
        """Build ModelCandidate from selected algorithm."""
        model_version_id = f"mv_{experiment_id}_{selected['name'].lower()}_v1"
        
        return ModelCandidate(
            model_version_id=model_version_id,
            experiment_id=experiment_id,
            algorithm_name=selected["name"],
            algorithm_family=selected["family"],
            hyperparameter_space=selected.get("hyperparameter_space", {}),
            default_hyperparameters=self._get_defaults(selected),
            estimated_training_time=self._estimate_training_time(selected),
            estimated_inference_latency_ms=selected["inference_latency_ms"],
            memory_requirement_gb=selected["memory_gb"],
            baseline_candidates=self._get_baselines(selected),
            selection_score=selected["selection_score"],
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.utcnow()
        )
```

### baseline_comparator.py

```python
class BaselineComparator:
    """Compare models against baselines."""
    
    BASELINES = {
        "classification": ["random", "majority_class", "logistic_regression"],
        "regression": ["mean_predictor", "median_predictor", "ridge"],
        "ranking": ["random_ranking", "popularity_based"]
    }
    
    def get_baselines(self, problem_type: str) -> List[str]:
        """Get appropriate baselines for problem type."""
        return self.BASELINES.get(problem_type, ["random"])
    
    async def compute_baseline_metrics(
        self,
        baseline_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for a baseline model."""
        
        if baseline_name == "random":
            predictions = np.random.choice([0, 1], size=len(y_val))
        elif baseline_name == "majority_class":
            majority = y_train.mode()[0]
            predictions = np.full(len(y_val), majority)
        elif baseline_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_val)[:, 1]
        
        return {
            "auc": roc_auc_score(y_val, predictions),
            "accuracy": accuracy_score(y_val, predictions > 0.5)
        }
```

## Memory Patterns

### Procedural Memory for Selection Patterns

```python
# Store successful selection pattern
await procedural_memory.store(
    pattern_type="algorithm_selection",
    content={
        "problem_type": "classification",
        "data_size_range": "10k-100k",
        "selected_algorithm": "CausalForest",
        "final_auc": 0.82,
        "success": True
    }
)

# Retrieve patterns for similar problems
patterns = await procedural_memory.retrieve(
    query="classification algorithm selection",
    filters={
        "problem_type": "classification",
        "data_size_range": "10k-100k"
    }
)
```

## Error Handling

```python
class ModelSelectorError(AgentError):
    """Base error for model_selector."""
    pass

class NoViableCandidatesError(ModelSelectorError):
    """No algorithms meet constraints."""
    pass

class ConstraintParseError(ModelSelectorError):
    """Cannot parse technical constraint."""
    pass

class RegistrationFailedError(ModelSelectorError):
    """MLflow registration failed."""
    pass
```

## Testing

```python
class TestModelSelector:
    
    async def test_classification_selection(self):
        """Test algorithm selection for classification."""
        state = AgentState(
            scope_spec=ScopeSpec(
                problem_type="classification",
                technical_constraints=["inference_latency_<100ms"]
            )
        )
        result = await agent.execute(state)
        
        assert result.model_candidate.algorithm_name in ALGORITHM_REGISTRY
        assert result.model_candidate.estimated_inference_latency_ms <= 100
    
    async def test_constraint_filtering(self):
        """Test tight latency constraint filters to fast models."""
        state = AgentState(
            scope_spec=ScopeSpec(
                problem_type="classification",
                technical_constraints=["inference_latency_<5ms"]
            )
        )
        result = await agent.execute(state)
        
        # Only linear models should pass <5ms constraint
        assert result.model_candidate.algorithm_family == "linear"
    
    async def test_causal_ml_preference(self):
        """Test E2I preference for causal ML algorithms."""
        # Without constraints, causal ML should be preferred
        result = await agent.execute(unconstrained_state)
        assert result.model_candidate.algorithm_family == "causal_ml"
```

## Key Principles

1. **Constraint First**: Always filter by technical constraints before ranking
2. **Historical Learning**: Use procedural memory to improve selections over time
3. **Baseline Inclusion**: Always include simple baselines for comparison
4. **Causal ML Preference**: For E2I, prefer causal ML algorithms when viable
5. **MLflow Registration**: All candidates registered for tracking
