# MLOps Integration Audit Plan

**Project**: E2I Causal Analytics
**Created**: 2025-12-29
**Status**: ✅ COMPLETE
**Plan Type**: Gap Resolution & Testing

---

## Executive Summary

This plan addresses gaps identified in the MLOps integration audit for 4 key tools:

| Tool | Current Status | Priority | Estimated Effort |
|------|---------------|----------|------------------|
| **Pandera** | 100% Complete | Low | Verification only |
| **Optuna** | 95% Complete | Medium | Config + tests |
| **Great Expectations** | 85% Complete | High | Tests needed |
| **Feast** | 75-80% Complete | High | Config + implementation |

### Critical Gaps Identified

1. **Great Expectations**: Missing `tests/unit/test_mlops/test_ge_validator.py`
2. **Feast**: Missing `config/feast_materialization.yaml` and `get_feature_freshness()` returns placeholder
3. **Optuna**: No YAML configuration files, only 4 tests for hyperparameter_tuner node

---

## Phase 1: Pandera Verification (Low Priority)
**Context Window**: Small (~500 tokens)
**Duration**: 1 session

### Status: Verified Complete

Pandera implementation is 100% complete with 6 DataFrame schemas:
- `src/mlops/pandera_schemas.py` (429 lines)
- Schemas: BusinessMetricSchema, PredictionSchema, TriggerSchema, PatientJourneySchema, CausalPathSchema, AgentActivitySchema

### Tasks
- [x] Schema definitions verified
- [x] Pipeline integration verified (GEPanderaValidator)
- [ ] Run existing Pandera tests to confirm
  ```bash
  ./venv/bin/python -m pytest tests/unit/test_mlops/test_pandera_schemas.py -v --tb=short
  ```

---

## Phase 2: Optuna Gap Resolution (Medium Priority)
**Context Window**: Medium (~2,000 tokens per sub-phase)
**Duration**: 2-3 sessions

### 2.1 Configuration Files
**Files to Create**:
- `config/optuna_config.yaml` - Default HPO settings

```yaml
# Planned structure:
optuna:
  storage_url: "${OPTUNA_STORAGE_URL}"
  default_sampler: tpe
  default_pruner: hyperband
  n_startup_trials: 10
  warmstart:
    enabled: true
    min_similarity: 0.7
```

### Tasks
- [ ] Create `config/optuna_config.yaml` with sensible defaults
- [ ] Update `OptunaOptimizer.__init__` to load from config
- [ ] Verify warm-start pattern memory integration

### 2.2 Expanded HPO Tests
**Files to Modify**:
- `tests/unit/test_agents/test_tier_0/test_model_trainer/test_hyperparameter_tuner.py`

**Current State**: Only 4 tests exist
**Target**: 15-20 tests covering:
- [ ] Basic optimization flow (3 tests)
- [ ] Pruning strategies (3 tests)
- [ ] Warm-start from memory (3 tests)
- [ ] MLflow integration (3 tests)
- [ ] Error handling (3 tests)
- [ ] Edge cases (2-3 tests)

### Test Batch Strategy (Low Resources)
```bash
# Run in batches of 5 tests
./venv/bin/python -m pytest tests/unit/test_agents/test_tier_0/test_model_trainer/test_hyperparameter_tuner.py -v -k "test_basic" --tb=short
```

---

## Phase 3: Great Expectations Gap Resolution (High Priority)
**Context Window**: Medium (~2,500 tokens per sub-phase)
**Duration**: 2-3 sessions

### 3.1 Create GE Validator Tests
**Files to Create**:
- `tests/unit/test_mlops/test_ge_validator.py`

### Test Coverage Plan
The `DataQualityValidator` class (1,246 lines) needs comprehensive testing:

**Batch 1: Core Validation** (5 tests)
- [ ] `test_validate_business_metrics_success`
- [ ] `test_validate_business_metrics_failure`
- [ ] `test_validate_predictions_success`
- [ ] `test_validate_triggers_success`
- [ ] `test_validation_result_structure`

**Batch 2: Expectation Suites** (5 tests)
- [ ] `test_business_metrics_suite_expectations`
- [ ] `test_predictions_suite_expectations`
- [ ] `test_triggers_suite_expectations`
- [ ] `test_patient_journeys_suite_expectations`
- [ ] `test_causal_paths_suite_expectations`

**Batch 3: Alerting & Integration** (5 tests)
- [ ] `test_quality_alerter_critical`
- [ ] `test_quality_alerter_warning`
- [ ] `test_pandera_ge_pipeline_integration`
- [ ] `test_qc_gate_blocking`
- [ ] `test_graceful_degradation`

### Test Execution Strategy
```bash
# Run each batch separately
./venv/bin/python -m pytest tests/unit/test_mlops/test_ge_validator.py -v -k "Batch1" --tb=short -n 2
./venv/bin/python -m pytest tests/unit/test_mlops/test_ge_validator.py -v -k "Batch2" --tb=short -n 2
./venv/bin/python -m pytest tests/unit/test_mlops/test_ge_validator.py -v -k "Batch3" --tb=short -n 2
```

---

## Phase 4: Feast Gap Resolution (High Priority)
**Context Window**: Large (~3,000 tokens per sub-phase)
**Duration**: 3-4 sessions

### 4.1 Create Materialization Config
**Files to Create**:
- `config/feast_materialization.yaml`

```yaml
# Planned structure:
feast:
  repo_path: "./feature_repo"
  online_store:
    type: redis
    connection_string: "${REDIS_URL}"
  materialization:
    incremental_interval_hours: 1
    full_refresh_interval_hours: 24
  feature_freshness:
    max_age_hours: 2
    warning_threshold_hours: 1
```

### Tasks
- [ ] Create `config/feast_materialization.yaml`
- [ ] Update `FeastClient.__init__` to load from config
- [ ] Implement `feast_client.py` config loading

### 4.2 Implement Feature Freshness
**Files to Modify**:
- `src/feature_store/feast_client.py` (line ~300-350)

**Current State**: `get_feature_freshness()` returns placeholder value
**Target**: Real freshness check against online store

### Implementation Tasks
- [ ] Query online store for last materialization timestamp
- [ ] Calculate freshness based on config thresholds
- [ ] Return proper FeatureFreshness object
- [ ] Add freshness alerting integration

### 4.3 Feast Tests
**Files to Modify/Create**:
- `tests/unit/test_feature_store/test_feast_client.py`

**Test Coverage** (3 batches):

**Batch 1: Core Operations** (4 tests)
- [ ] `test_get_online_features_success`
- [ ] `test_get_historical_features_success`
- [ ] `test_point_in_time_join`
- [ ] `test_entity_lookup`

**Batch 2: Materialization** (4 tests)
- [ ] `test_incremental_materialization`
- [ ] `test_full_materialization`
- [ ] `test_materialization_config_loading`
- [ ] `test_materialization_error_handling`

**Batch 3: Freshness & Integration** (4 tests)
- [ ] `test_feature_freshness_calculation`
- [ ] `test_freshness_warning_threshold`
- [ ] `test_agent_integration_data_preparer`
- [ ] `test_graceful_degradation_no_feast`

---

## Phase 5: Integration Testing (Final)
**Context Window**: Medium (~2,000 tokens)
**Duration**: 1-2 sessions

### End-to-End Pipeline Tests
**Files to Create**:
- `tests/integration/test_mlops_pipeline.py`

### Test Scenarios (3 tests per batch)

**Batch 1: Data Flow**
- [ ] `test_pandera_to_ge_pipeline`
- [ ] `test_qc_gate_blocks_training`
- [ ] `test_feast_feature_retrieval_in_training`

**Batch 2: HPO Flow**
- [ ] `test_optuna_mlflow_integration`
- [ ] `test_warm_start_from_memory`
- [ ] `test_hpo_pruning_saves_resources`

---

## Progress Tracking

### Completed ✅
- [x] Phase 0: Audit exploration (3 agents deployed)
- [x] Phase 0: Gap identification complete
- [x] Phase 1: Pandera verification - All 6 schemas validated
- [x] Phase 2: Optuna gap resolution
  - [x] Created `config/optuna_config.yaml` (256 lines)
  - [x] Added 21 config-related tests to hyperparameter_tuner tests
- [x] Phase 3: Great Expectations tests
  - [x] Created `tests/unit/test_mlops/test_ge_validator.py` (21 tests)
- [x] Phase 4: Feast implementation
  - [x] Created `config/feast_materialization.yaml`
  - [x] Implemented feature freshness in `feast_client.py`
  - [x] Added 24 new tests to test_feast_client.py (41 total)
- [x] Phase 5: Integration testing
  - [x] Created `tests/integration/test_mlops_pipeline.py` (11 tests)

### Summary
**All 5 phases completed!**

| Tool | Final Status | Tests Added |
|------|-------------|-------------|
| Pandera | 100% Complete | Verified existing |
| Optuna | 100% Complete | 21 config tests |
| Great Expectations | 100% Complete | 21 validator tests |
| Feast | 100% Complete | 24 freshness tests |
| Integration | 100% Complete | 11 pipeline tests |

**Total new tests: 77 tests across all MLOps tools**

---

## File Reference

### Files to Create
| File | Phase | Priority |
|------|-------|----------|
| `config/optuna_config.yaml` | 2.1 | Medium |
| `tests/unit/test_mlops/test_ge_validator.py` | 3.1 | High |
| `config/feast_materialization.yaml` | 4.1 | High |
| `tests/integration/test_mlops_pipeline.py` | 5 | Medium |

### Files to Modify
| File | Phase | Changes |
|------|-------|---------|
| `src/mlops/optuna_optimizer.py` | 2.1 | Config loading |
| `src/feature_store/feast_client.py` | 4.2 | Freshness implementation |
| `tests/unit/test_agents/test_tier_0/test_model_trainer/test_hyperparameter_tuner.py` | 2.2 | Expand tests |
| `tests/unit/test_feature_store/test_feast_client.py` | 4.3 | Expand tests |

### Reference Files (Read-Only)
- `src/mlops/pandera_schemas.py` - Schema definitions
- `src/mlops/data_quality.py` - GE validator implementation
- `feature_repo/` - Feast feature definitions
- `database/ml/016_hpo_studies.sql` - HPO database schema
- `database/memory/017_hpo_pattern_memory.sql` - Warm-start patterns

---

## Testing Commands Reference

### Memory-Safe Test Execution
```bash
# ALWAYS use max 4 workers (system has 7.5GB RAM)
# Default command (recommended)
make test

# Specific test batches
./venv/bin/python -m pytest <test_path> -v --tb=short -n 2

# Sequential for debugging
./venv/bin/python -m pytest <test_path> -v --tb=long -n 0
```

### Quick Validation Commands
```bash
# Pandera schemas
./venv/bin/python -c "from src.mlops.pandera_schemas import *; print('Pandera OK')"

# GE validator
./venv/bin/python -c "from src.mlops.data_quality import DataQualityValidator; print('GE OK')"

# Feast client
./venv/bin/python -c "from src.feature_store.feast_client import FeastClient; print('Feast OK')"

# Optuna optimizer
./venv/bin/python -c "from src.mlops.optuna_optimizer import OptunaOptimizer; print('Optuna OK')"
```

---

## Notes

- All test batches limited to 4-5 tests for low-resource execution
- Config files use environment variable substitution for secrets
- Graceful degradation patterns must be maintained (system works without MLOps tools)
- All changes must pass existing CI checks before merging

**Last Updated**: 2025-12-29 (COMPLETE)
