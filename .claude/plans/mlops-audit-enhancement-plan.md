# MLOps Integration Audit & Enhancement Plan

**Project**: E2I Causal Analytics
**Created**: 2025-12-29
**Status**: IN PROGRESS
**Plan Type**: Audit Verification & Gap Resolution

---

## Executive Summary

### Audit Result: ~98% MLOps Complete (Verified)

A comprehensive audit of the MLOps integration confirms the ~98% completion status. Key findings:

| Tool | Status | Code (LOC) | Tests | Verified |
|------|--------|------------|-------|----------|
| **MLflow** | 100% Complete | 1,261 | 38 | Yes |
| **Opik** | 100% Complete | 1,223 | 30 | Yes |
| **Great Expectations** | 100% Complete | 1,246 | 44 | Yes |
| **Optuna** | 100% Complete | 1,048 | 81 | Yes |
| **SHAP** | 100% Complete | 908 | 9 | Yes |
| **BentoML** | 95% Complete | 2,431 | 62 | Yes |
| **Feast** | 93% Complete | 3,151 | 41 | Yes |

**Total MLOps Tests**: 430 unit tests + 24 integration tests = **454 tests**

### Actual Gaps Identified

| Gap | Priority | Impact | Effort |
|-----|----------|--------|--------|
| BentoML database tables missing | Medium | Audit trail incomplete | 2h |
| Feast-specific schema missing | Medium | Using generic ml_feature_store | 2h |
| SHAP test coverage limited (9 tests) | Medium | Error handling untested | 3h |
| GEPA coverage partial (5/18 agents) | Low | Standard metrics fallback | 4h |
| Deprecation warnings (pyiceberg, bentoml) | Medium | Future compatibility | 1h |

---

## Phase 1: Verification & Quick Wins
**Context Window**: Small (~800 tokens)
**Duration**: 1 session

### 1.1 Import Verification (COMPLETED)

```bash
# All imports verified working:
./venv/bin/python -c "from src.mlops.opik_connector import OpikConnector; print('OK')"
./venv/bin/python -c "from src.mlops.optuna_optimizer import OptunaOptimizer; print('OK')"
./venv/bin/python -c "from src.mlops.data_quality import DataQualityValidator; print('OK')"
./venv/bin/python -c "from src.feature_store.feast_client import FeastClient; print('OK')"
./venv/bin/python -c "from src.mlops.bentoml_service import BentoMLModelManager; print('OK')"
./venv/bin/python -c "from src.mlops.shap_explainer_realtime import RealTimeSHAPExplainer; print('OK')"
```

**Result**: All imports successful

### 1.2 Test Suite Verification

```bash
# Run MLOps tests in small batches (4 workers max)
./venv/bin/python -m pytest tests/unit/test_mlops/ -v --tb=short -n 4 -x
```

### Tasks
- [ ] Run full MLOps test suite
- [ ] Document any failing tests
- [ ] Verify 430 tests pass

---

## Phase 2: BentoML Database Integration
**Context Window**: Medium (~1,500 tokens)
**Duration**: 1-2 sessions
**Priority**: Medium

### 2.1 Problem Statement

BentoML deployments are tracked in MLflow but lack dedicated database tables for:
- Service health tracking
- Deployment versioning
- Serving metrics persistence

### 2.2 Implementation

**File**: `database/ml/024_bentoml_tables.sql`

```sql
-- Create BentoML-specific tables
CREATE TABLE IF NOT EXISTS ml_bentoml_services (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bento_tag VARCHAR(255) NOT NULL,
    bento_version VARCHAR(100) NOT NULL,
    model_id UUID REFERENCES ml_model_registry(id),
    service_name VARCHAR(255) NOT NULL,
    serving_endpoint TEXT,
    container_image TEXT,
    status deployment_status_enum DEFAULT 'pending',
    health_status VARCHAR(50) DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_bentoml_serving_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_id UUID REFERENCES ml_bentoml_services(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    requests_per_second FLOAT,
    avg_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    error_rate FLOAT,
    memory_mb FLOAT,
    cpu_percent FLOAT
);

CREATE INDEX idx_bentoml_services_tag ON ml_bentoml_services(bento_tag);
CREATE INDEX idx_bentoml_metrics_service ON ml_bentoml_serving_metrics(service_id, timestamp);
```

### 2.3 Repository Implementation

**File**: `src/repositories/bentoml_service_repository.py`

### Tasks
- [ ] Create migration file `024_bentoml_tables.sql`
- [ ] Create `BentoMLServiceRepository` class
- [ ] Update `bentoml_service.py` to use repository
- [ ] Add 10 unit tests for repository
- [ ] Run migration on Supabase

---

## Phase 3: SHAP Test Coverage Enhancement
**Context Window**: Medium (~1,200 tokens)
**Duration**: 1 session
**Priority**: Medium

### 3.1 Current State

- **Existing tests**: 9 tests (limited)
- **Missing coverage**:
  - Error handling (timeout, memory limits)
  - Large feature sets (100+ features)
  - Batch explanation edge cases
  - Model type auto-detection

### 3.2 Test Enhancement

**File**: `tests/unit/test_mlops/test_shap_explainer.py` (create/enhance)

**Test Cases to Add**:
```python
# Error handling tests
test_explain_timeout_handling()
test_explain_memory_limit_handling()
test_explain_invalid_model_type()
test_explain_missing_feature_names()

# Edge case tests
test_explain_high_dimensional_data_100_features()
test_explain_batch_with_partial_failures()
test_explain_with_nan_values()

# Model detection tests
test_auto_detect_xgboost_model()
test_auto_detect_random_forest_model()
test_auto_detect_linear_model()
test_auto_detect_deep_learning_model()

# Background data tests
test_background_data_caching()
test_background_data_from_feast_fallback()
```

### Tasks
- [ ] Create `test_shap_explainer.py` with 15+ new tests
- [ ] Add error handling tests (5 tests)
- [ ] Add edge case tests (5 tests)
- [ ] Add model detection tests (5 tests)
- [ ] Run tests: `pytest tests/unit/test_mlops/test_shap_explainer.py -v -n 2`

---

## Phase 4: Feast Schema Enhancement
**Context Window**: Small (~800 tokens)
**Duration**: 1 session
**Priority**: Medium (User requested)

### 4.1 Current State

- `ml_feature_store` table exists (generic)
- No Feast-specific tracking for:
  - Feature views
  - Materialization jobs
  - Feature freshness by view

### 4.2 Implementation

**File**: `database/ml/025_feast_tracking_tables.sql`

```sql
-- Feast-specific tracking
CREATE TABLE IF NOT EXISTS ml_feast_feature_views (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    entities JSONB,
    features JSONB,
    ttl_seconds INTEGER,
    online_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_feast_materialization_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_view_id UUID REFERENCES ml_feast_feature_views(id),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    status VARCHAR(50),
    rows_materialized INTEGER,
    duration_seconds FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Tasks
- [ ] Create migration file `025_feast_tracking_tables.sql`
- [ ] Create `FeastTrackingRepository` class
- [ ] Update FeastClient to log materializations
- [ ] Add 8 tests for Feast tracking
- [ ] Run migration on Supabase

---

## Phase 5: Documentation & Cleanup
**Context Window**: Small (~500 tokens)
**Duration**: 1 session
**Priority**: Medium

### 5.1 Update Implementation Status

**Files to Update**:
- `.claude/context/implementation-status.md` - Mark audit complete
- `README.md` - Verify MLOps section current

### 5.2 Address Deprecation Warnings (Investigate Fixes)

**Warnings Found**:
1. `pyiceberg` - `@model_validator` deprecation (Pydantic V2.12)
2. `bentoml` - `bentoml.io` deprecation (BentoML v1.4)

**Investigation Tasks**:

#### PyIceberg Deprecation
- **Source**: `pyiceberg/table/metadata.py`
- **Issue**: Using `@model_validator(mode='after')` on classmethod
- **Fix Options**:
  1. Pin pyiceberg to older version (temporary)
  2. Wait for pyiceberg update to fix this
  3. Check if Feast can use alternative library

#### BentoML Deprecation
- **Source**: `bentoml/io.py`
- **Issue**: `bentoml.io` deprecated since v1.4
- **Fix Options**:
  1. Update `src/mlops/bentoml_service.py` to use new IO types
  2. Check BentoML migration guide for v1.4+ patterns
  3. Update imports from `bentoml.io` to new style

### Tasks
- [ ] Update implementation-status.md with audit results
- [ ] Research pyiceberg deprecation workaround
- [ ] Update BentoML code to use new IO types (if feasible)
- [ ] Document findings in CHANGELOG
- [ ] Update this plan file with completion status

---

## Progress Tracking

### Completed
- [x] Phase 0: Deep audit (3 Explore agents deployed)
- [x] Import verification (all 6 MLOps tools import successfully)
- [x] Test count verification (430 unit tests + 24 integration)
- [x] HPO pattern memory table verified (database/memory/017_hpo_pattern_memory.sql)

### In Progress
- [ ] Phase 1: Full test suite verification

### Pending
- [ ] Phase 2: BentoML database integration
- [ ] Phase 3: SHAP test coverage enhancement
- [ ] Phase 4: Feast schema enhancement
- [ ] Phase 5: Documentation & cleanup

---

## Testing Strategy (Resource-Constrained)

Due to limited system resources, run tests in small batches:

```bash
# Batch 1: Core connectors (estimated: 2 min)
./venv/bin/python -m pytest tests/unit/test_mlops/test_mlflow_connector.py tests/unit/test_mlops/test_opik_connector.py -v -n 2

# Batch 2: HPO & data quality (estimated: 3 min)
./venv/bin/python -m pytest tests/unit/test_mlops/test_optuna_optimizer.py tests/unit/test_mlops/test_data_quality.py -v -n 2

# Batch 3: BentoML (estimated: 2 min)
./venv/bin/python -m pytest tests/unit/test_mlops/test_bentoml/ -v -n 2

# Batch 4: Feature store (estimated: 2 min)
./venv/bin/python -m pytest tests/unit/test_feature_store/ -v -n 2

# Batch 5: Integration (estimated: 3 min, sequential)
./venv/bin/python -m pytest tests/integration/test_mlops_pipeline.py -v -n 1
```

---

## Files to Modify

### New Files
| File | Purpose |
|------|---------|
| `database/ml/024_bentoml_tables.sql` | BentoML database schema |
| `database/ml/025_feast_tracking_tables.sql` | Feast tracking schema |
| `src/repositories/bentoml_service_repository.py` | BentoML persistence |
| `src/repositories/feast_tracking_repository.py` | Feast materialization tracking |
| `tests/unit/test_mlops/test_shap_explainer.py` | Enhanced SHAP tests (15+ new) |
| `tests/unit/test_repositories/test_bentoml_repository.py` | BentoML repo tests |
| `tests/unit/test_repositories/test_feast_tracking.py` | Feast tracking tests |

### Existing Files to Update
| File | Change |
|------|--------|
| `src/mlops/bentoml_service.py` | Add repository integration, update IO types |
| `src/feature_store/feast_client.py` | Add materialization tracking |
| `.claude/context/implementation-status.md` | Update audit results |
| `CHANGELOG.md` | Document deprecation findings |

---

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Verification | 30 min | High |
| Phase 2: BentoML DB | 2 hours | Medium |
| Phase 3: SHAP Tests | 3 hours | Medium |
| Phase 4: Feast Schema | 2 hours | Medium |
| Phase 5: Documentation + Deprecations | 1.5 hours | Medium |

**Total**: ~9 hours

---

## Success Criteria

1. All 430+ MLOps tests pass
2. BentoML deployments persist to database with new tables
3. SHAP test coverage increases from 9 to 24+ tests
4. Feast feature views and materializations tracked in dedicated tables
5. BentoML deprecation warnings resolved (new IO types)
6. PyIceberg deprecation documented with recommended action
7. Documentation reflects accurate ~98% status
8. No new failing tests introduced

---

**Last Updated**: 2025-12-29
**Next Review**: After Phase 1 completion
