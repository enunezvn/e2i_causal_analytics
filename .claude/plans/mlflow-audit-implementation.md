# MLflow Integration Audit - Implementation Plan

**Date**: 2025-12-25
**Scope**: Experiment tracking & model registry across 3 Tier 0 agents
**Agents**: model_trainer, model_selector, model_deployer

---

## Executive Summary

### Audit Findings

| Agent | Status | Issues |
|-------|--------|--------|
| **model_trainer** | ✅ WORKING | Correctly uses MLflowConnector async context manager |
| **model_selector** | ❌ BROKEN | 2 critical bugs - calls non-existent method, logs outside context |
| **model_deployer** | ⚠️ INCONSISTENT | Bypasses MLflowConnector, uses raw sync MLflow API |

### Critical Issues Identified

1. **model_selector `end_run()` bug** (Line 77)
   - Calls `await connector.end_run()` but method doesn't exist on MLflowConnector
   - Will cause AttributeError at runtime

2. **model_selector context bug** (Lines 66, 69)
   - Logging methods called outside `async with start_run()` block
   - Violates async context manager pattern

3. **model_deployer wrapper bypass**
   - Uses raw MLflow API: `mlflow.register_model()`, `MlflowClient().transition_model_version_stage()`
   - No circuit breaker protection, no async support
   - Inconsistent with other agents

4. **Spec-Implementation mismatch**
   - Spec document describes `MLflowClientWrapper` (sync)
   - Actual implementation is `MLflowConnector` (async singleton with circuit breaker)

---

## Implementation Plan

### Phase 1: Fix Critical model_selector Bugs
**Priority**: CRITICAL
**Estimated Files**: 1
**Testing**: Unit tests only

#### Tasks
- [ ] 1.1 Read current model_selector/nodes/mlflow_registrar.py implementation
- [ ] 1.2 Remove non-existent `end_run()` call (line 77)
- [ ] 1.3 Refactor to use proper async context manager pattern
- [ ] 1.4 Move logging calls inside `async with start_run()` block
- [ ] 1.5 Run unit tests for model_selector agent

#### Files to Modify
- `src/agents/ml_foundation/model_selector/nodes/mlflow_registrar.py`

#### Test Command
```bash
# Uses memory-safe defaults from pyproject.toml (-n 4 --dist=loadscope)
pytest tests/unit/test_agents/test_ml_foundation/test_model_selector/ -v
```

---

### Phase 2: Standardize model_deployer MLflow Integration
**Priority**: HIGH
**Estimated Files**: 1-2
**Testing**: Unit tests + integration test

#### Tasks
- [ ] 2.1 Read current model_deployer/nodes/registry_manager.py implementation
- [ ] 2.2 Replace raw `mlflow.register_model()` with MLflowConnector method
- [ ] 2.3 Replace raw `MlflowClient().transition_model_version_stage()` with connector
- [ ] 2.4 Add async support to registry operations
- [ ] 2.5 Ensure circuit breaker protection is applied
- [ ] 2.6 Run unit tests for model_deployer agent
- [ ] 2.7 Run integration test with MLflow

#### Files to Modify
- `src/agents/ml_foundation/model_deployer/nodes/registry_manager.py`
- `src/mlops/mlflow_connector.py` (if new methods needed)

#### Test Commands
```bash
# Uses memory-safe defaults from pyproject.toml (-n 4 --dist=loadscope)
pytest tests/unit/test_agents/test_ml_foundation/test_model_deployer/ -v
pytest tests/integration/test_mlflow_integration.py -v
```

---

### Phase 3: Add Missing MLflowConnector Methods (if needed)
**Priority**: MEDIUM
**Estimated Files**: 1
**Testing**: Unit tests for connector

#### Tasks
- [ ] 3.1 Review MLflowConnector for model registry methods
- [ ] 3.2 Add `register_model()` async method if missing
- [ ] 3.3 Add `transition_model_version_stage()` async method if missing
- [ ] 3.4 Add `get_model_version()` async method if missing
- [ ] 3.5 Ensure circuit breaker wraps new methods
- [ ] 3.6 Run MLflowConnector unit tests

#### Files to Modify
- `src/mlops/mlflow_connector.py`

#### Test Command
```bash
# Uses memory-safe defaults from pyproject.toml (-n 4 --dist=loadscope)
pytest tests/unit/test_mlops/test_mlflow_connector.py -v
```

---

### Phase 4: Update Documentation
**Priority**: LOW
**Estimated Files**: 1
**Testing**: None (docs only)

#### Tasks
- [ ] 4.1 Update mlops_integration.md spec to reflect MLflowConnector (async)
- [ ] 4.2 Remove references to MLflowClientWrapper (sync)
- [ ] 4.3 Document correct async context manager usage pattern
- [ ] 4.4 Add code examples showing proper usage

#### Files to Modify
- `.claude/specialists/MLOps_Integration/mlops_integration.md`

---

### Phase 5: End-to-End Validation
**Priority**: HIGH (after fixes)
**Estimated Files**: 0 (tests only)
**Testing**: Full Tier 0 pipeline test

#### Tasks
- [ ] 5.1 Run all Tier 0 agent unit tests
- [ ] 5.2 Run MLflow integration tests
- [ ] 5.3 Test model_selector → model_trainer → model_deployer flow
- [ ] 5.4 Verify MLflow experiment/run creation
- [ ] 5.5 Verify model registry operations

#### Test Commands
```bash
# All Tier 0 tests (uses memory-safe defaults from pyproject.toml)
pytest tests/unit/test_agents/test_ml_foundation/ -v

# MLflow integration
pytest tests/integration/test_mlflow*.py -v

# Full pipeline (if exists)
pytest tests/e2e/test_tier0_pipeline.py -v
```

---

## Reference: Correct MLflow Usage Pattern

From model_trainer (working example):

```python
from src.mlops.mlflow_connector import MLflowConnector

async def log_experiment(state: ModelTrainerState) -> ModelTrainerState:
    connector = MLflowConnector()

    async with connector.start_run(
        experiment_name=state.experiment_name,
        run_name=state.run_name,
        tags={"agent": "model_trainer"}
    ) as run:
        # All logging MUST be inside this block
        await run.log_params(state.hyperparameters)
        await run.log_metrics(state.metrics)
        await run.log_artifact(state.model_path)
        await run.log_model(state.model, "model")

        # Run auto-ends when exiting context

    return state
```

---

## File Paths Reference

| File | Purpose |
|------|---------|
| `src/mlops/mlflow_connector.py` | Core MLflow integration (async singleton) |
| `src/agents/ml_foundation/model_trainer/nodes/mlflow_logger.py` | ✅ Working reference |
| `src/agents/ml_foundation/model_selector/nodes/mlflow_registrar.py` | ❌ Needs fixes |
| `src/agents/ml_foundation/model_deployer/nodes/registry_manager.py` | ⚠️ Needs standardization |
| `.claude/specialists/MLOps_Integration/mlops_integration.md` | Spec (needs update) |

---

## Progress Tracking

### Phase 1: model_selector Fixes
- [x] Started
- [x] Completed
- [x] Tests Passing (191 passed, 2 skipped)

### Phase 2: model_deployer Standardization
- [ ] Started
- [ ] Completed
- [ ] Tests Passing

### Phase 3: Connector Methods
- [ ] Started
- [ ] Completed
- [ ] Tests Passing

### Phase 4: Documentation
- [ ] Started
- [ ] Completed

### Phase 5: End-to-End Validation
- [ ] Started
- [ ] All Tests Passing
- [ ] Ready for Review
