# Phase 5: MLflow Experiment Tracking

**Goal**: Integrate MLflow for experiment tracking

**Status**: Not Started

**Dependencies**: None (can start after Phase 1)

---

## Tasks

- [ ] **Task 5.1**: Create `src/mlops/mlflow_connector.py`
  - MLflow client wrapper
  - Experiment management
  - Run tracking
  - Artifact logging

- [ ] **Task 5.2**: Set up MLflow tracking server
  - Add to docker-compose.yaml
  - Configure backend store (Supabase)
  - Configure artifact store (S3/local)

- [ ] **Task 5.3**: Implement experiment logging helpers
  - Auto-log parameters
  - Auto-log metrics
  - Auto-log models
  - Custom tag support

- [ ] **Task 5.4**: Add model registry integration
  - Model versioning
  - Stage transitions (staging → production)
  - Model metadata

- [ ] **Task 5.5**: Connect to database tables
  - Sync with `ml_experiments` table
  - Sync with `ml_training_runs` table
  - Enable hybrid querying

- [ ] **Task 5.6**: Add MLflow UI documentation
  - Access instructions
  - Usage guide
  - Best practices

- [ ] **Task 5.7**: Test with sample training run
  - End-to-end logging test
  - Verify UI visibility

---

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `src/mlops/mlflow_connector.py` | Create | MLflow wrapper |
| `config/mlflow/mlflow.yaml` | Create | MLflow config |
| `docker/docker-compose.yaml` | Modify | Add MLflow service |
| `tests/unit/test_mlops/test_mlflow_connector.py` | Create | Unit tests |
| `docs/mlflow-guide.md` | Create | User guide |

---

## MLflow Connector Interface

```python
class MLflowConnector:
    """Production-ready MLflow integration."""

    async def create_experiment(self, name: str, tags: Dict) -> str:
        """Create or get existing experiment."""

    async def start_run(self, experiment_id: str, run_name: str) -> MLflowRun:
        """Start a new training run."""

    async def log_params(self, run_id: str, params: Dict) -> None:
        """Log hyperparameters."""

    async def log_metrics(self, run_id: str, metrics: Dict, step: int) -> None:
        """Log training metrics."""

    async def log_model(self, run_id: str, model: Any, artifact_path: str) -> None:
        """Log trained model."""

    async def register_model(self, run_id: str, model_name: str) -> ModelVersion:
        """Register model in registry."""
```

---

## Docker Configuration

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://...
      - MLFLOW_ARTIFACT_ROOT=/mlartifacts
    volumes:
      - mlflow_artifacts:/mlartifacts
```

---

## Progress Log

| Date | Update |
|------|--------|
| 2024-12-22 | Phase created |

---

## Blockers

None currently.
