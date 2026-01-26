# Backend Test Failures Fix Plan

**Created**: 2026-01-26
**Status**: Planning
**Total Failures**: ~84 tests across 6 categories

---

## Summary of Failure Categories

| Category | Count | Root Cause | Priority |
|----------|-------|-----------|----------|
| MLflow Permission | 41 | `/mlflow` permission denied - incomplete error handling | High |
| E2E Auth | 16 | 401 Unauthorized - E2I_TESTING_MODE timing | High |
| Missing Module | 10 | sentence_transformers not installed (install required) | Medium |
| Data Quality | 4 | GE validator expectation assertions | Medium |
| Event Loop | 4 | run_async() conflicts with pytest-asyncio | High |
| Misc | 9 | String assertions, edge cases | Low |

---

## Phase 1: MLflow Permission Fixes (41 tests)

**Goal**: Add graceful degradation for permission errors in MLflow trackers

### Files to Modify
- `src/agents/drift_monitor/mlflow_tracker.py`
- `src/agents/experiment_designer/mlflow_tracker.py`

### Changes Required

1. **Extend error handling in `_get_mlflow()`** (both files):
```python
def _get_mlflow(self):
    """Lazy load MLflow to avoid import errors if not installed."""
    if self._mlflow is None:
        try:
            import mlflow
            self._mlflow = mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
        except (ImportError, OSError, PermissionError) as e:
            logger.warning(f"MLflow tracking unavailable ({type(e).__name__}): {e}")
            return None
    return self._mlflow
```

2. **Wrap artifact logging operations** with try/except:
```python
def _log_artifacts(self, artifacts: Dict[str, Any]) -> None:
    mlflow = self._get_mlflow()
    if mlflow is None:
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # ... existing code ...
            mlflow.log_artifact(artifact_path)
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to log artifacts: {e}")
```

### Verification
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_drift_monitor/ \
  tests/unit/test_agents/test_experiment_designer/ \
  -v -n 2 --timeout=60"
```

---

## Phase 2: E2E Auth Fixes (16 tests)

**Goal**: Ensure E2I_TESTING_MODE is set before app imports

### Files to Review/Modify
- `tests/integration/api/conftest.py` - Add E2I_TESTING_MODE
- `tests/e2e/` - Check for proper auth setup
- Any test importing `src.api.main` directly

### Changes Required

1. **Add E2I_TESTING_MODE to integration conftest**:
```python
# tests/integration/api/conftest.py - at TOP of file
import os
os.environ["E2I_TESTING_MODE"] = "1"

# Then other imports...
```

2. **Review E2E tests** for auth patterns:
   - Use `dependency_overrides` pattern for role testing
   - Or ensure TESTING_MODE is set before imports

### Verification
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/integration/api/ tests/e2e/ \
  -v -n 2 --timeout=60 -k 'auth or 401'"
```

---

## Phase 3: Missing Module Fixes (10 tests)

**Goal**: Install sentence_transformers on droplet (reranker functionality is essential)

### Action Required
Install `sentence_transformers` package in the production venv on the droplet.

### Installation Command
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "/opt/e2i_causal_analytics/.venv/bin/pip install sentence-transformers"
```

### Notes
- This is an exception to the "avoid pip install on droplet" rule
- Reranker functionality is essential for RAG system
- The package provides CrossEncoder model for document reranking

### Verification
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/rag/test_reranker.py -v --timeout=60"
```

---

## Phase 4: Event Loop Fixes (4 tests)

**Goal**: Fix run_async() to work with pytest-asyncio auto mode

### Files to Modify
- `src/tasks/drift_monitoring_tasks.py` (lines 72-83)
- `src/tasks/ab_testing_tasks.py` (lines 97-108)
- `src/tasks/feedback_loop_tasks.py` (lines 84-95)

### Changes Required

1. **Replace run_async() with safer pattern**:
```python
def run_async(coro):
    """Helper to run async coroutine in sync context.

    Compatible with pytest-asyncio auto mode.
    """
    try:
        # Check if we're in an existing event loop (pytest-asyncio)
        loop = asyncio.get_running_loop()
        # We're in a running loop - use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)
```

2. **Alternative: Convert tests to async**:
```python
@pytest.mark.asyncio
async def test_task_queries_drift_alerts(self, mock_get_client, mock_drift_alerts_data):
    result = await analyze_concept_drift_from_truth_async()
    assert result is not None
```

### Verification
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_tasks/ -v -n 2 --timeout=60"
```

---

## Phase 5: Data Quality Fixes (4 tests)

**Goal**: Fix GE validator expectation assertion failures

### Files to Review
- `tests/unit/test_mlops/test_ge_validator.py`
- `src/mlops/data_quality.py`

### Investigation Steps

1. **Check GE_AVAILABLE flag**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "/opt/e2i_causal_analytics/.venv/bin/python -c 'from src.mlops.data_quality import GE_AVAILABLE; print(f\"GE_AVAILABLE={GE_AVAILABLE}\")'"
```

2. **Run failing tests with verbose output**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_mlops/test_ge_validator.py -v --tb=long --timeout=60"
```

3. **Likely fixes**:
   - Update assertions to match actual expectation types
   - Handle GE unavailable case properly

### Verification
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_mlops/test_ge_validator.py -v --timeout=60"
```

---

## Phase 6: Misc Fixes (9 tests)

**Goal**: Fix string assertions and error handling edge cases

### Investigation Steps

1. **Get list of remaining failures**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/ -v --tb=line -n 4 --timeout=60 2>&1 | grep -E 'FAILED|ERROR'"
```

2. **Group by failure type and fix individually**

### Common Patterns to Look For
- String comparison issues (exact vs contains)
- Exception type mismatches
- Mock return value issues
- Timeout issues

---

## Execution Order

| Phase | Tests | Batch Size | Est. Iterations |
|-------|-------|-----------|-----------------|
| 1 | 41 MLflow | 2 test files | 1-2 |
| 2 | 16 E2E Auth | 2 directories | 1-2 |
| 3 | 10 Missing Module | pip install | 1 |
| 4 | 4 Event Loop | 3 source files | 2-3 |
| 5 | 4 Data Quality | 1 test file | 1-2 |
| 6 | 9 Misc | Individual | 2-3 |

---

## Progress Tracking

### Phase 1: MLflow Permission ✅ COMPLETED
- [x] Review drift_monitor/mlflow_tracker.py
- [x] Review experiment_designer/mlflow_tracker.py
- [x] Add PermissionError handling to _get_mlflow()
- [x] Wrap artifact logging with try/except
- [x] Run verification tests (656 passed, 1 failed - latency test)
- [x] Commit changes

### Phase 2: E2E Auth ✅ COMPLETED
- [x] Review integration/api/conftest.py
- [x] Add E2I_TESTING_MODE to conftest (both root and integration/api)
- [x] Review E2E test patterns
- [x] Run verification tests (29 passed, 93 skipped)
- [x] Commit changes

### Phase 3: Missing Module ✅ COMPLETED
- [x] Install sentence-transformers on droplet (CPU-only torch)
- [x] Fixed huggingface_hub version conflict (1.2.4 -> 1.3.3)
- [x] Verify CrossEncoder import works
- [x] Fixed conftest.py autouse skip issue (synced local to droplet)
- [x] Run verification tests (17 passed)

### Phase 4: Event Loop ✅ COMPLETED
- [x] Review drift_monitoring_tasks.py run_async()
- [x] Review ab_testing_tasks.py run_async()
- [x] Review feedback_loop_tasks.py run_async()
- [x] Implement safer run_async() pattern (reuse thread-local event loop instead of closing)
- [x] Run verification tests (78 passed)
- [ ] Commit changes (pending with other phases)

### Phase 5: Data Quality ✅ COMPLETED
- [x] Check GE_AVAILABLE status on droplet (GE_AVAILABLE=True)
- [x] Run failing tests with verbose output
- [x] Tests already passing (44 passed)
- [x] No fixes needed - tests pass

### Phase 6: Misc ✅ COMPLETED
- [x] Get list of remaining failures
- [x] Fixed: Added load_dotenv() to root conftest.py to load .env API keys
- [x] Fixed: Experiment designer conftest.py API key handling
- [x] Query logger tests: 42 passed (xdist errors were intermittent)
- [x] Experiment designer tests: 43 passed
- [x] Run full test suite: 10536 passed, 46 skipped
- [ ] Commit changes

---

## Final Verification

After all phases complete:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest \
  tests/ -v -n 4 --timeout=60 --tb=short"
```

**Success Criteria**: All 84 previously failing tests pass (or are properly skipped with reason).
