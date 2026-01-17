# Technical Debt Remediation: Deprecated APIs

**Created:** 2026-01-14
**Status:** Planning
**Estimated Effort:** ~3 hours total (7 phases, 30 min each)

---

## Executive Summary

Address 5 categories of deprecated API usage across 18 files with 67+ instances. Plan structured in 7 context-window-friendly phases with droplet testing.

## Priority Matrix

| Priority | Category | Files | Instances | Risk |
|----------|----------|-------|-----------|------|
| P0 | BentoML `bentoml.io` | 2 | 2 | CRITICAL |
| P1 | Pydantic v2 (RAG API) | 1 | 3 | HIGH |
| P2 | datetime.utcnow() Core | 6 | 7 | MEDIUM |
| P3 | datetime.utcnow() Repos | 4 | 9 | MEDIUM |
| P4 | Pydantic v2 + datetime (Observability) | 1 | 8 | MEDIUM |
| P5 | MLflow set_tracking_uri | 3 | 3 | LOW |
| P6 | Pandera schemas | 1 | 0 | SKIPPED |

---

## Phase 1: BentoML Deprecations (CRITICAL)

**Files:** 2 | **Time:** 15 min

Remove unused deprecated `bentoml.io` imports (deprecated since v1.4).

### Changes

| File | Line | Action |
|------|------|--------|
| `src/mlops/bentoml_service.py` | 25 | Remove `from bentoml.io import JSON, NumpyNdarray` |
| `src/mlops/bentoml_templates/classification_service.py` | 22 | Remove `from bentoml.io import JSON, NumpyNdarray` |

### Verification
```bash
python -c "from src.mlops.bentoml_service import BENTOML_AVAILABLE; print('OK')"
pytest tests/unit/test_mlops/ -v -k "bentoml" --tb=short -n 2
```

---

## Phase 2: Pydantic v2 - RAG API (HIGH)

**Files:** 1 | **Time:** 20 min

Replace `class Config:` with `model_config = ConfigDict(...)`.

### Changes

| File | Class | Lines |
|------|-------|-------|
| `src/api/routes/rag.py` | SearchResultItem | ~85-94 |
| `src/api/routes/rag.py` | SearchRequest | ~125-134 |
| `src/api/routes/rag.py` | SearchResponse | ~157-174 |

### Pattern
```python
# BEFORE
class Config:
    json_schema_extra = {"example": {...}}

# AFTER
from pydantic import ConfigDict
model_config = ConfigDict(json_schema_extra={"example": {...}})
```

### Verification
```bash
python -c "from src.api.routes.rag import SearchRequest; print(SearchRequest.model_json_schema())"
pytest tests/unit/test_api/ -v -k "rag" --tb=short -n 2
```

---

## Phase 3: datetime.utcnow() - Core API/KPI

**Files:** 6 | **Time:** 25 min

Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`.

### Changes

| File | Occurrences | Lines |
|------|-------------|-------|
| `src/api/routes/copilotkit.py` | 2 | 2820, 2821 |
| `src/api/routes/chatbot_graph.py` | 1 | 270 |
| `src/api/routes/chatbot_tools.py` | 1 | 246 |
| `src/kpi/cache.py` | 1 | 145 |
| `src/kpi/calculator.py` | 1 | 258 |
| `src/causal_engine/validation/report_generator.py` | 1 | 63 |

### Pattern
```python
# BEFORE
from datetime import datetime
now = datetime.utcnow()

# AFTER
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
```

### Verification
```bash
pytest tests/unit/test_api/ tests/unit/test_kpi/ -v --tb=short -n 2
```

---

## Phase 4: datetime.utcnow() - Repository Layer

**Files:** 4 | **Time:** 20 min

### Changes

| File | Occurrences | Lines |
|------|-------------|-------|
| `src/repositories/chatbot_analytics.py` | 6 | 109, 164, 199, 201, 345, 385 |
| `src/repositories/chatbot_feedback.py` | 1 | 233 |
| `src/repositories/chatbot_user_profile.py` | 1 | 136 |
| `src/digital_twin/models/simulation_models.py` | 1 | 302 |

### Verification
```bash
pytest tests/unit/test_repositories/ -v --tb=short -n 2
```

---

## Phase 5: Observability Connector (Pydantic + datetime)

**Files:** 1 | **Time:** 30 min

### File: `src/agents/ml_foundation/observability_connector/models.py`

**Pydantic Changes:**

| Class | Line | Change |
|-------|------|--------|
| SpanEvent | ~102 | `class Config: frozen=True` → `model_config = ConfigDict(frozen=True)` |
| ObservabilitySpan | ~305 | `class Config: from_attributes=True` → `model_config = ConfigDict(from_attributes=True)` |
| QualityMetrics | ~458 | `class Config: from_attributes=True` → `model_config = ConfigDict(from_attributes=True)` |

**datetime Changes:**

| Line | Change |
|------|--------|
| ~96 | `default_factory=datetime.utcnow` → `default_factory=lambda: datetime.now(timezone.utc)` |
| ~274 | `datetime.utcnow()` → `datetime.now(timezone.utc)` |
| ~369 | `default_factory=datetime.utcnow` → `default_factory=lambda: datetime.now(timezone.utc)` |
| ~499 | `datetime.utcnow()` → `datetime.now(timezone.utc)` |
| ~535 | `datetime.utcnow()` → `datetime.now(timezone.utc)` |

### Verification
```bash
pytest tests/unit/test_agents/test_ml_foundation/test_observability_connector/ -v --tb=short -n 2
pytest tests/integration/test_observability_integration.py -v --tb=short
```

---

## Phase 6: MLflow Deprecations

**Files:** 3 | **Time:** 25 min

Replace `mlflow.set_tracking_uri()` with environment variable pattern.

### Changes

| File | Line | Action |
|------|------|--------|
| `src/mlops/mlflow_connector.py` | 473 | Use `os.environ["MLFLOW_TRACKING_URI"]` instead |
| `src/feature_store/client.py` | 104 | Use `os.environ["MLFLOW_TRACKING_URI"]` instead |
| `src/causal_engine/energy_score/mlflow_tracker.py` | ~105 | Use `os.environ["MLFLOW_TRACKING_URI"]` instead |

### Pattern
```python
# BEFORE
mlflow.set_tracking_uri(self.tracking_uri)

# AFTER
import os
os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
# MlflowClient already takes tracking_uri param (correct pattern)
```

### Verification
```bash
pytest tests/unit/test_mlops/ -v -k "mlflow" --tb=short -n 2
```

---

## Phase 7: Pandera Schemas - SKIPPED

**Reason:** Pandera's `class Config` is NOT Pydantic's deprecated pattern. Pandera has its own configuration API using `class Config` with attributes like `name`, `strict`, `coerce`. No changes needed.

---

## Droplet Testing Protocol

```bash
# Per-phase testing on droplet
ssh root@159.89.180.27
cd /root/Projects/e2i_causal_analytics
source .venv/bin/activate

# After each phase:
make lint
pytest <phase-specific-tests> -v --tb=short -n 2

# Final validation:
make test-fast
uvicorn src.api.main:app --port 8001 &
curl http://localhost:8001/health
pkill -f uvicorn
```

---

## Rollback Procedure

```bash
# Per-file rollback
git checkout HEAD -- <file_path>

# Full rollback
git reset --hard HEAD~1
```

---

# Progress Tracking Checklist

## Phase 1: BentoML Deprecations (CRITICAL)
- [ ] Remove bentoml.io import from `src/mlops/bentoml_service.py`
- [ ] Remove bentoml.io import from `src/mlops/bentoml_templates/classification_service.py`
- [ ] Verify imports work
- [ ] Run BentoML tests on droplet

## Phase 2: Pydantic v2 - RAG API (HIGH)
- [ ] Update SearchResultItem in `src/api/routes/rag.py`
- [ ] Update SearchRequest in `src/api/routes/rag.py`
- [ ] Update SearchResponse in `src/api/routes/rag.py`
- [ ] Run RAG API tests on droplet

## Phase 3: datetime.utcnow() - Core API/KPI
- [ ] Fix `src/api/routes/copilotkit.py` (2 occurrences)
- [ ] Fix `src/api/routes/chatbot_graph.py` (1 occurrence)
- [ ] Fix `src/api/routes/chatbot_tools.py` (1 occurrence)
- [ ] Fix `src/kpi/cache.py` (1 occurrence)
- [ ] Fix `src/kpi/calculator.py` (1 occurrence)
- [ ] Fix `src/causal_engine/validation/report_generator.py` (1 occurrence)
- [ ] Run API and KPI tests on droplet

## Phase 4: datetime.utcnow() - Repository Layer
- [ ] Fix `src/repositories/chatbot_analytics.py` (6 occurrences)
- [ ] Fix `src/repositories/chatbot_feedback.py` (1 occurrence)
- [ ] Fix `src/repositories/chatbot_user_profile.py` (1 occurrence)
- [ ] Fix `src/digital_twin/models/simulation_models.py` (1 occurrence)
- [ ] Run repository tests on droplet

## Phase 5: Pydantic v2 + datetime - Observability
- [ ] Update SpanEvent model_config in `models.py`
- [ ] Update ObservabilitySpan model_config in `models.py`
- [ ] Update QualityMetrics model_config in `models.py`
- [ ] Fix 5 datetime.utcnow() occurrences in `models.py`
- [ ] Run observability tests on droplet

## Phase 6: MLflow Deprecations
- [ ] Update `src/mlops/mlflow_connector.py`
- [ ] Update `src/feature_store/client.py`
- [ ] Update `src/causal_engine/energy_score/mlflow_tracker.py`
- [ ] Run MLflow tests on droplet

## Phase 7: Pandera Schemas
- [x] SKIPPED - Not Pydantic deprecation (Pandera's own Config pattern)

## Final Validation
- [ ] Full test suite passes (`make test`)
- [ ] API starts and health checks pass
- [ ] No deprecation warnings in logs
- [ ] Create commit with all changes
- [ ] Push to repository

---

## Commit Message Template

```
fix(deprecation): Address deprecated APIs across codebase

Pydantic v2:
- Update RAG API models to use model_config (rag.py)
- Update observability models to use model_config

datetime.utcnow() -> datetime.now(timezone.utc):
- Fix 20 occurrences across API, KPI, repository layers
- Ensures Python 3.12+ compatibility

BentoML:
- Remove deprecated bentoml.io imports (unused since v1.4)

MLflow:
- Use environment variable for tracking URI

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
