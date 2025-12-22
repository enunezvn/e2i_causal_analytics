# Phase 12: End-to-End Integration

**Goal**: Integrate all Tier 0 agents into unified pipeline

**Status**: Not Started

**Dependencies**: All previous phases (1-11)

---

## Tasks

- [ ] **Task 12.1**: Create orchestration flow for Tier 0 agents
  - Define agent calling sequence
  - Manage state handoffs
  - Handle failures gracefully

- [ ] **Task 12.2**: Define handoff protocols between agents
  - Standardize output contracts
  - Validate inputs at each stage
  - Log transitions

- [ ] **Task 12.3**: Add end-to-end integration tests
  - Test full pipeline
  - Test partial pipelines
  - Test error recovery

- [ ] **Task 12.4**: Create sample training pipeline (churn model)
  - Real data flow
  - Production-like conditions
  - Document the process

- [ ] **Task 12.5**: Document complete ML workflow
  - Step-by-step guide
  - Troubleshooting
  - Best practices

- [ ] **Task 12.6**: Performance testing and optimization
  - Benchmark pipeline duration
  - Identify bottlenecks
  - Optimize critical paths

---

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `src/agents/tier_0/pipeline.py` | Create | Orchestration |
| `src/agents/tier_0/handoffs.py` | Create | Handoff protocols |
| `tests/e2e/test_ml_pipeline/` | Create | E2E tests |
| `docs/ml-workflow.md` | Create | Documentation |

---

## Pipeline Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Scope     │ ──▶ │    Data      │ ──▶ │   Feature    │
│   Definer    │     │   Preparer   │     │   Analyzer   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Model     │ ◀── │    Model     │ ◀── │    Model     │
│   Deployer   │     │   Trainer    │     │   Selector   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Observability│
                                          │  Connector   │
                                          └──────────────┘
```

---

## Handoff Protocol

```python
class AgentHandoff(TypedDict):
    """Standard handoff between Tier 0 agents."""

    from_agent: str
    to_agent: str
    timestamp: datetime
    status: Literal["success", "partial", "failed"]
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: List[str]

    # Validation
    output_validated: bool
    schema_version: str
```

---

## Sample Pipeline: Churn Prediction

```python
async def run_churn_prediction_pipeline():
    """End-to-end churn prediction training."""

    # 1. Define scope
    scope = await scope_definer.run({
        "request": "Train churn prediction model for Kisqali HCPs"
    })

    # 2. Prepare data
    data = await data_preparer.run({
        "scope": scope,
        "brand": "Kisqali"
    })

    # 3. Engineer features
    features = await feature_analyzer.run({
        "data": data,
        "scope": scope
    })

    # 4. Select model
    model_selection = await model_selector.run({
        "features": features,
        "problem_type": scope["problem_type"]
    })

    # 5. Train model
    trained_model = await model_trainer.run({
        "features": features,
        "model_config": model_selection
    })

    # 6. Deploy model
    deployment = await model_deployer.run({
        "model": trained_model,
        "strategy": "blue-green"
    })

    # 7. Connect observability
    await observability_connector.run({
        "deployment": deployment
    })

    return deployment
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pipeline completion rate | >95% | Successful runs / Total runs |
| End-to-end latency | <30 min | Time from scope to deployment |
| Model quality | AUC >0.8 | Test set performance |
| Deployment success | >99% | Healthy deployments / Total |

---

## Progress Log

| Date | Update |
|------|--------|
| 2024-12-22 | Phase created |

---

## Blockers

- Depends on all previous phases (1-11)
