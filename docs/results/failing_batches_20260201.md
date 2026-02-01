# Failing Test Batches — 2026-02-01

**Test Run**: Batched unit test suite (43 batches, 12,174 passed, 98.8% pass rate)
**Date**: 2026-02-01 21:40–22:14 UTC
**RAM**: Stable 7.1–7.4 GB throughout, zero swap usage

---

## Batch 29: agents/experiment_designer — 44 failures

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: LLM calls not mocked — tests hit real OpenAI/Anthropic APIs with exhausted quota
**Fix**: Added autouse fixtures in conftest.py to mock get_chat_llm, get_fast_llm, get_llm_provider, _get_validity_llm, and _get_mlflow_tracker. 394/394 passing.

### Failed Tests
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_basic`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_returns_treatments`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_returns_outcomes`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_returns_randomization`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_returns_stratification`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_with_constraints`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_with_historical_context`
- `test_design_reasoning.py::TestDesignReasoningNode::test_execute_records_latency`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_basic`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_with_constraints`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_with_available_data`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_with_brand`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_heavy_preregistration`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_light_preregistration`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_without_validity_audit`
- `test_experiment_designer_agent.py::TestExperimentDesignerAgent::test_run_multiple_redesign_iterations`
- `test_experiment_designer_agent.py::TestExperimentDesignerOutput::test_output_structure`
- `test_experiment_designer_agent.py::TestExperimentDesignerOutput::test_output_validity_score_range`
- `test_experiment_designer_agent.py::TestExperimentDesignerOutput::test_output_treatments_structure`
- `test_experiment_designer_agent.py::TestExperimentDesignerOutput::test_output_outcomes_structure`
- *(+ 24 more in same files)*

---

## Batch 9: unit/test_api — 57 failures

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: Three distinct issues:
1. **Auth import-time caching** (27 failures): `TESTING_MODE` in `auth.py` was a module-level constant evaluated at import time. Under xdist parallel workers, `E2I_TESTING_MODE` env var wasn't set before auth.py was imported, causing all protected routes to return 401.
2. **Session expiry hour overflow** (1 failure): `cognitive.py` used `datetime.replace(hour=hour+1)` which overflows at UTC hour 23.
3. **FalkorDB env var caching** (1 failure): `falkordb_client.py` read env vars at module-level import time, so `patch.dict("os.environ")` had no effect.

**Fix**:
- `src/api/dependencies/auth.py`: Changed `require_auth`, `is_testing_mode`, and `is_auth_enabled` to check `os.environ` at runtime instead of using cached `TESTING_MODE` constant.
- `src/api/routes/cognitive.py`: Replaced `datetime.replace(hour=hour+1)` with `datetime.now(timezone.utc) + timedelta(hours=1)`.
- `src/api/dependencies/falkordb_client.py`: Moved env var reads inside `init_falkordb()` so they're evaluated at call time.
- `tests/unit/test_api/test_dependencies/test_auth.py`: Updated `test_require_auth_success` to use `patch.dict("os.environ")` instead of patching the stale `TESTING_MODE` constant.

**Result**: 1525/1525 passing (0 failed, 2 skipped).

---

## Batch 28: agents/drift_monitor — 24 failures

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: Three distinct issues:
1. **Field name mismatch** (20 failures): Tests used `detection_latency_ms` / `current_timestamp` but source renamed to `total_latency_ms` / `timestamp`.
2. **MLflow tracking URI** (agent tests): `_get_mlflow_tracker()` created real MLflow tracker with file:// backend, which fails with `mlflow-artifacts://` URI.
3. **Supabase UUID errors** (node tests): Tests used string feature names ("feature1") but real Supabase connector expects UUID format.

**Fix**:
- Replaced `detection_latency_ms` → `total_latency_ms` across 8 test files.
- Replaced `current_timestamp` → `timestamp` in test_data_drift.py and test_drift_monitor_agent.py.
- Created `conftest.py` with autouse fixtures to mock MLflow tracker (`_get_mlflow_tracker` → None) and force mock data connector (`DRIFT_MONITOR_CONNECTOR=mock`).

**Result**: 263/263 passing.

---

## Batch 22: agents/ml_foundation — 11 failures

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: Test expectations out of sync with source code changes across 5 sub-agents:
1. **Feature analyzer**: `model_uri` changed from required to optional (graceful degradation)
2. **Candidate ranker**: Causal ML penalty (-0.15) for non-causal problems wasn't accounted for in tests
3. **Leakage detector**: Hash-based detection replaced index-based; test fixtures had index-only overlap
4. **Model deployer**: BentoML available in test env, tag format differs from simulated
5. **MLflow registrar**: Real MLflow server available, import-error test didn't actually simulate error

**Fix**: Updated test expectations across 7 test files — all tests, not source code.
- Updated model_uri tests to match optional behavior (skip not error)
- Added `requires_causal=True` and used same family for fair score comparisons
- Fixed leakage fixtures to use actual data overlap (not just index overlap)
- Added `@patch(BENTOML_AVAILABLE, False)` for deployment tests
- Used `sys.modules[...] = None` to properly simulate import error

**Result**: 1218/1218 passing (6 skipped).

---

## Batch 20: unit/test_digital_twin — 2 failures + 3 errors

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: Three distinct issues:
1. **geographic_scope type mismatch** (3 errors): `TwinModelConfig.geographic_scope` changed from `List[str]` to `str`, but fixture still passed `["US"]`.
2. **Supabase `.not_.is_()` mock chain** (1 failure): `client.not_` returned a flat `MagicMock` without `.is_()` wired to return the client, so `await query.execute()` hit a non-async mock.
3. **uuid patch target** (1 failure): `import uuid` is inside the function (not module-level), so `patch("...retraining_service.uuid.uuid4")` raised `AttributeError`. Fixed to `patch("uuid.uuid4")`.

**Fix**:
- `test_twin_repository.py`: Changed `geographic_scope=["US"]` → `geographic_scope="US"` in fixture. Restructured `.not_` mock to wire `.is_()` back to client.
- `test_retraining_service.py`: Changed patch target from `src.digital_twin.retraining_service.uuid.uuid4` to `uuid.uuid4`.

**Result**: 321/321 passing.

### Failed Tests (before fix)
- `test_retraining_service.py::TestTwinRetrainingService::test_trigger_retraining`
- `test_twin_repository.py::TestFidelityRepository::test_get_model_fidelity_records`

### Errors (before fix)
- `test_twin_repository.py::TestTwinModelRepository::test_save_model`
- `test_twin_repository.py::TestTwinModelRepository::test_save_model_no_client`
- `test_twin_repository.py::TestTwinRepository::test_save_model_delegation`

---

## Batch 18: unit/test_mlops — 3 failures

**Status**: [x] FIXED (2026-02-01)
**Root Cause**: Tests expected old template class names (`ClassificationServiceTemplate`, `RegressionServiceTemplate`, `CausalInferenceServiceTemplate`) but the refactored `generate_service_file()` generates self-contained service classes named `{ServiceName}Service` with type-specific input classes (`ClassificationInput`, `RegressionInput`, `CausalInput`).

**Fix**: Updated assertions to check for actual generated content:
- `ClassificationServiceTemplate` → `Classification Service` + `ClassificationInput`
- `RegressionServiceTemplate` → `Regression Service` + `RegressionInput`
- `CausalInferenceServiceTemplate` → `Causal Inference Service` + `CausalInput`

**Result**: 514/514 passing (entire test_mlops batch).

### Failed Tests (before fix)
- `test_bentoml/test_bentoml_packaging.py::TestGenerateServiceFile::test_generate_classification_service`
- `test_bentoml/test_bentoml_packaging.py::TestGenerateServiceFile::test_generate_regression_service`
- `test_bentoml/test_bentoml_packaging.py::TestGenerateServiceFile::test_generate_causal_service`

---

## Fix Priority

1. ~~**experiment_designer** (44 failures)~~ FIXED
2. ~~**test_api** (57 failures)~~ FIXED
3. ~~**drift_monitor** (24 failures)~~ FIXED
4. ~~**ml_foundation** (11 failures)~~ FIXED
5. ~~**digital_twin** (5 failures)~~ FIXED
6. ~~**test_mlops** (3 failures)~~ FIXED

**All 6 batches resolved. 144 failures → 0.**
