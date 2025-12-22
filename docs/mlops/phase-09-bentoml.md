# Phase 9: BentoML Model Serving

**Goal**: Set up model deployment infrastructure

**Status**: Not Started

**Dependencies**: Phase 8 (need trained models)

---

## Tasks

- [ ] **Task 9.1**: Create `src/mlops/bentoml_service.py`
  - BentoML service wrapper
  - Model loading utilities
  - Prediction interface

- [ ] **Task 9.2**: Define Bento service templates
  - Classification service template
  - Regression service template
  - Causal inference service template

- [ ] **Task 9.3**: Create model packaging utilities
  - Package model with dependencies
  - Include preprocessing pipeline
  - Version management

- [ ] **Task 9.4**: Set up containerized deployment
  - Dockerfile for Bento services
  - Docker Compose integration
  - Kubernetes manifests (optional)

- [ ] **Task 9.5**: Add health checks and monitoring
  - Liveness probe
  - Readiness probe
  - Prometheus metrics

- [ ] **Task 9.6**: Document deployment workflow
  - Step-by-step deployment guide
  - Rollback procedures
  - Troubleshooting

---

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `src/mlops/bentoml_service.py` | Create | BentoML wrapper |
| `src/mlops/bentoml_templates/` | Create | Service templates |
| `docker/bentoml/Dockerfile` | Create | Container config |
| `scripts/deploy_model.py` | Create | Deployment script |
| `docs/deployment-guide.md` | Create | Documentation |

---

## BentoML Service Interface

```python
import bentoml

@bentoml.service(
    resources={"cpu": "1", "memory": "2Gi"},
    traffic={"timeout": 60}
)
class E2IPredictionService:
    """BentoML prediction service for E2I models."""

    def __init__(self):
        self.model = bentoml.models.get("e2i-model:latest")
        self.preprocessor = load_preprocessor()

    @bentoml.api
    async def predict(self, data: PredictionInput) -> PredictionOutput:
        """Run prediction on input data."""
        processed = self.preprocessor.transform(data)
        prediction = self.model.predict(processed)
        return PredictionOutput(prediction=prediction)

    @bentoml.api
    async def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────┐
│                 Load Balancer               │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼───────┐   ┌───────▼───────┐
│   Bento Pod   │   │   Bento Pod   │
│   (replica 1) │   │   (replica 2) │
└───────────────┘   └───────────────┘
```

---

## Progress Log

| Date | Update |
|------|--------|
| 2024-12-22 | Phase created |

---

## Blockers

- Depends on Phase 8 completion (trained models)
