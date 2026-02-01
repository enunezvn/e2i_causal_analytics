# Plan: Add BentoML Model Serving to Tier0 Test

## Status: COMPLETED

**Implementation Date**: ~2026-01-30
**Prediction Fix Date**: 2026-02-01
**Last Successful Run**: 2026-02-01 (`docs/results/tier0_pipeline_run_20260201_020908.md`)
**Result**: All 8 steps pass, prediction test passes (9.2ms latency)

---

## Summary

Add BentoML model serving capability to the tier0 test on the droplet, using the **REAL trained model** from the tier0 workflow (not a mock service). This enables end-to-end validation of the full ML pipeline including model deployment and prediction serving.

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `--include-bentoml` CLI flag | **Done** | `scripts/run_tier0_test.py:3510` |
| `step_7_model_deployer` params (`trained_model`, `include_bentoml`, `fitted_preprocessor`) | **Done** | `scripts/run_tier0_test.py:2398-2415` |
| BentoML registration in step 7 (with preprocessor) | **Done** | `scripts/run_tier0_test.py:2495-2608` |
| `start_bentoml_service()` (with preprocessor + framework) | **Done** | `scripts/run_tier0_test.py:1050-1170` |
| `verify_bentoml_predictions()` (multi-format, error logging) | **Done** | `scripts/run_tier0_test.py:1197-1260` |
| `stop_bentoml_service()` | **Done** | `scripts/run_tier0_test.py:1275-1290` |
| `deploy_to_persistent_service()` | **Done** | `scripts/run_tier0_test.py:1260-1274` |
| Finally-block cleanup | **Done** | `scripts/run_tier0_test.py:3452-3461` |
| BentoML infrastructure (`bentoml_service.py`) | **Done** | `src/mlops/bentoml_service.py` |
| BentoML packaging (`bentoml_packaging.py`) | **Done** | `src/mlops/bentoml_packaging.py` |
| BentoML client (`bentoml_client.py`) | **Done** | `src/api/dependencies/bentoml_client.py` |
| Prediction Synthesizer HTTP client | **Done** | `src/agents/prediction_synthesizer/clients/http_model_client.py` |
| Model inference tool | **Done** | `src/tool_registry/tools/model_inference.py` |
| BentoML dependency | **Done** | `requirements.txt` (bentoml==1.4.30), `docker/bentoml/requirements-bentoml.txt` |

## Prediction Verification Fix (2026-02-01)

### Bugs Found and Fixed

| Bug | Root Cause | Fix | File |
|-----|-----------|-----|------|
| Preprocessor always `None` | `agent.py` read `final_state.get("fitted_preprocessor")` but LangGraph state key is `"preprocessor"` | Changed to `final_state.get("preprocessor")` | `src/agents/ml_foundation/model_trainer/agent.py:281` |
| Ephemeral service had no preprocessing | Generated service code hardcoded `bentoml.sklearn.load_model()` with no preprocessor | Added `preprocessor` and `framework` params; saves preprocessor via joblib; generates service code that loads and applies it | `scripts/run_tier0_test.py:1050-1170` |
| Preprocessor not wired through pipeline | Step 5 result had preprocessor but it was never stored in state or passed to step 7 | Store `fitted_preprocessor` in state after step 5; pass to step 7 and onward to `register_model_for_serving()` and `start_bentoml_service()` | `scripts/run_tier0_test.py` |
| Persistent service preprocessor loading broken | `model_ref.info.custom_objects` doesn't exist in BentoML 1.4.30 | Try `model_ref.custom_objects` first, fallback to `.info` | `src/mlops/bentoml_service.py:460-466` |
| Preprocessor `.transform()` crashes on numpy arrays | `ColumnTransformer` expects DataFrame with named columns, not raw numpy | Convert to DataFrame with feature names from preprocessor before calling `.transform()` | `src/mlops/bentoml_service.py` (predict + predict_proba) |
| JSON payload format mismatch | Ephemeral uses `{"features": ...}`, persistent uses `{"input_data": {"features": ...}}` | Verify function tries both payload formats, stops on first non-400 | `scripts/run_tier0_test.py:1227-1243` |
| Non-200 errors silently swallowed | Only `status_code == 200` was handled; failures produced no diagnostic output | Added `else` branch capturing HTTP status + response body; display in step 7 output | `scripts/run_tier0_test.py:1244-1249, 2634-2641` |
| 1D array edge case | `arr.shape[1]` raises `IndexError` on 1D arrays | Added `ndim == 1` reshape guard | `scripts/run_tier0_test.py`, `src/mlops/bentoml_service.py` |

### Verified Result (2026-02-01)

```
BentoML Serving Verification:
  health_check: ✓ healthy
  prediction_test: ✓ passed
  predictions: [0.0]
  probabilities: [1.8290592589130572e-07]
  latency_ms: 9.2

✅ Real model deployed and serving verified via BentoML (persistent)
```

---

## Who Consumes BentoML-Served Models?

The **Prediction Synthesizer Agent (Tier 4)** is the primary consumer:

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 4: Prediction Synthesizer Agent                       │
│  - Uses HTTPModelClient (src/agents/prediction_synthesizer/)│
│  - Circuit breaker + exponential backoff                    │
│  - Multi-model orchestration                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────────────────────┐
        │  BentoML HTTP Client                      │
        │  (src/api/dependencies/bentoml_client.py) │
        │  - Connection pooling (20 connections)    │
        │  - Opik tracing                           │
        └─────────────┬─────────────────────────────┘
                      │
        ┌─────────────▼─────────────────────────────┐
        │  BentoML HTTP Services (Docker)           │
        │  - :3001 Classification (churn, conversion)│
        │  - :3002 Regression (ltv)                 │
        │  - :3003 Causal (treatment effects)       │
        └───────────────────────────────────────────┘
```

Key consumer code paths:
- `src/api/routes/predictions.py` - REST API endpoints (`POST /api/models/predict/{model_name}`)
- `src/agents/prediction_synthesizer/clients/http_model_client.py` - Agent's HTTP client
- `src/tool_registry/tools/model_inference.py` - Tool for any agent to call predictions

## Architecture Notes

### Dual Deployment Modes

The implementation supports two deployment strategies:
- **Ephemeral**: Temporary `bentoml serve` subprocess for test validation, cleaned up in finally block
- **Persistent**: Deploys to `e2i-bentoml.service` systemd unit for production use

### Preprocessing Pipeline

The ephemeral service now handles preprocessing end-to-end:
1. Preprocessor saved to `/tmp/tier0_bentoml_preprocessor.pkl` via joblib
2. Generated service loads it at startup
3. On predict: converts numpy array to DataFrame with correct column names, then calls `.transform()`
4. Model receives preprocessed features matching training dimensions

### Self-Contained Services

Generated BentoML service files have **no project imports** -- critical for BentoML's isolated execution model.

## Test Execution

All commands run **ON the droplet** (not locally):

```bash
# On droplet: Navigate and activate venv
cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Run full tier0 test with BentoML serving
python scripts/run_tier0_test.py --include-bentoml

# Run specific steps with BentoML (requires step 4+5 to train model first)
python scripts/run_tier0_test.py --step 4,5,7 --include-bentoml

# Run steps 4-8 (recommended for full flow validation)
python scripts/run_tier0_test.py --step 4,5,6,7,8 --include-bentoml
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Port 3001 in use | Check port availability, use alternative port |
| BentoML not installed | Already in venv (bentoml==1.4.30 per requirements.txt) |
| Model registration fails | Fallback to regular deployment, log error |
| Service startup timeout | 30-second wait with retries, configurable |
| Process cleanup | Use try/finally to ensure cleanup on error |
| Payload format mismatch | Verify function tries multiple formats automatically |
| Preprocessor missing | Fallback to raw features if preprocessor unavailable |

## Constraints Respected

- No `pip install` on droplet - uses existing venv
- Uses REAL trained model from tier0 workflow (not mock)
- Uses existing BentoML infrastructure (`src/mlops/bentoml_service.py`)
- Minimal changes to existing code
- Optional via `--include-bentoml` flag (doesn't break existing test)
- Actual predictions from the dynamically-selected trained model
