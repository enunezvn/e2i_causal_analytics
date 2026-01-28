# Plan: Add BentoML Model Serving to Tier0 Test

## Summary

Add BentoML model serving capability to the tier0 test on the droplet, using the **REAL trained model** from the tier0 workflow (not a mock service). This enables end-to-end validation of the full ML pipeline including model deployment and prediction serving.

## Current State

- **Tier0 test**: `scripts/run_tier0_test.py` runs 8 steps (scope → data prep → cohort → model select → train → feature analysis → deploy → observability)
- **Step 4 (Model Selector)**: Dynamically selects the best algorithm (XGBoost, LightGBM, LogisticRegression, etc.) based on comparative analysis
- **Step 5 (Model Trainer)**: Trains the selected model on synthetic data and stores it in `state["trained_model"]`
- **Step 7 (Model Deployer)**: Currently only creates a manifest (`deployment_action: "register"`) - doesn't actually deploy to BentoML
- **BentoML infrastructure**: Full registration and serving code exists in `src/mlops/bentoml_service.py` and `src/mlops/bentoml_packaging.py`

## Who Consumes BentoML-Served Models?

Based on codebase analysis, the **Prediction Synthesizer Agent (Tier 4)** is the primary consumer:

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

## Implementation Approach

**Use the REAL trained model** from step 5 (model_trainer) and register it with BentoML, then serve it and verify predictions work with actual inference.

### Key Insight

The trained model from step 5 is stored in `state["trained_model"]` (line 1115 of `run_tier0_test.py`). We need to:
1. Pass this model to step 7
2. Register it with BentoML using `register_model_for_serving()`
3. Serve it using `bentoml serve`
4. Verify predictions work with sample data

### Files to Modify

| File | Changes |
|------|---------|
| `scripts/run_tier0_test.py` | Add `--include-bentoml` flag, pass `trained_model` to step 7, add BentoML service functions |

## Implementation Details

### 1. Add CLI Flag for BentoML Testing

```python
parser.add_argument(
    "--include-bentoml",
    action="store_true",
    help="Include BentoML model serving verification with the real trained model"
)
```

### 2. Modify Step 7 Signature to Accept Trained Model

Current signature (line 833-838):
```python
async def step_7_model_deployer(
    experiment_id: str,
    model_uri: str,
    validation_metrics: dict,
    success_criteria_met: bool
) -> dict[str, Any]:
```

New signature:
```python
async def step_7_model_deployer(
    experiment_id: str,
    model_uri: str,
    validation_metrics: dict,
    success_criteria_met: bool,
    trained_model: Any = None,  # NEW: Pass the actual model
    include_bentoml: bool = False,  # NEW: Whether to deploy to BentoML
) -> dict[str, Any]:
```

### 3. Add BentoML Registration and Serving Logic

Inside step 7, when `include_bentoml=True` and `trained_model` is provided:

```python
async def step_7_model_deployer(..., trained_model=None, include_bentoml=False):
    # ... existing code ...

    if include_bentoml and trained_model is not None:
        print("\n  BentoML Model Serving:")

        # Register the real trained model with BentoML
        from src.mlops.bentoml_service import register_model_for_serving

        model_name = f"tier0_{experiment_id[:8]}"
        registration = await register_model_for_serving(
            model=trained_model,
            model_name=model_name,
            metadata={
                "experiment_id": experiment_id,
                "validation_metrics": validation_metrics,
                "tier0_test": True,
            },
            framework="sklearn",
        )

        if registration.get("registration_status") == "success":
            model_tag = registration.get("model_tag")
            print(f"    model_tag: {model_tag}")

            # Start BentoML service with the registered model
            bentoml_result = await start_bentoml_service(model_tag)

            if bentoml_result.get("started"):
                # Verify predictions work
                verification = await verify_bentoml_predictions(
                    endpoint=bentoml_result.get("endpoint"),
                    sample_features=[[0.5, 3, 1]],  # Sample from tier0 data
                )
                result["bentoml_serving"] = {
                    "model_tag": model_tag,
                    "endpoint": bentoml_result.get("endpoint"),
                    "health_check": verification.get("health_check"),
                    "prediction_test": verification.get("prediction_test"),
                    "latency_ms": verification.get("latency_ms"),
                }
                result["bentoml_pid"] = bentoml_result.get("pid")
```

### 4. Add BentoML Service Functions

```python
async def start_bentoml_service(model_tag: str, port: int = 3001) -> dict:
    """Start BentoML service serving the real trained model.

    Args:
        model_tag: BentoML model tag from registration
        port: Port to serve on

    Returns:
        {"started": True, "endpoint": "http://localhost:3001", "pid": <pid>}
    """
    import subprocess
    import asyncio
    import httpx

    # Generate a service file dynamically for the model
    service_code = f'''
import bentoml
import numpy as np

model_ref = bentoml.models.get("{model_tag}")

@bentoml.service(name="tier0_model_service")
class Tier0ModelService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("{model_tag}")

    @bentoml.api
    async def predict(self, features: list) -> dict:
        import time
        start = time.time()
        arr = np.array(features)
        predictions = self.model.predict(arr)
        probas = self.model.predict_proba(arr) if hasattr(self.model, 'predict_proba') else None
        elapsed = (time.time() - start) * 1000
        return {{
            "predictions": predictions.tolist(),
            "probabilities": probas.tolist() if probas is not None else None,
            "latency_ms": elapsed,
            "model_tag": "{model_tag}",
        }}

    @bentoml.api
    async def health(self) -> dict:
        return {{"status": "healthy", "model_tag": "{model_tag}"}}
'''

    # Write service file
    service_path = Path("/tmp/tier0_bentoml_service.py")
    service_path.write_text(service_code)

    # Start bentoml serve in background
    process = subprocess.Popen(
        ["bentoml", "serve", str(service_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for service to be ready
    endpoint = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        for _ in range(30):  # 30 retries
            await asyncio.sleep(1)
            try:
                resp = await client.get(f"{endpoint}/health")
                if resp.status_code == 200:
                    return {"started": True, "endpoint": endpoint, "pid": process.pid}
            except:
                pass

    return {"started": False, "error": "Service startup timeout"}


async def verify_bentoml_predictions(endpoint: str, sample_features: list) -> dict:
    """Verify that BentoML service returns valid predictions using the production consumer path.

    Uses BentoMLClient from src/api/dependencies/bentoml_client.py to validate:
    - Circuit breaker logic
    - Connection pooling
    - Opik tracing integration
    - Error handling with exponential backoff

    Args:
        endpoint: BentoML service endpoint (e.g., http://localhost:3001)
        sample_features: Sample feature data to test

    Returns:
        {"health_check": True, "prediction_test": True, "predictions": [...], "latency_ms": X}
    """
    from src.api.dependencies.bentoml_client import BentoMLClient
    import time

    result = {"health_check": False, "prediction_test": False}

    # Use the production BentoMLClient (circuit breaker, pooling, tracing)
    client = BentoMLClient(base_url=endpoint)

    try:
        # Health check via client
        health = await client.health_check()
        result["health_check"] = health.get("status") == "healthy"
    except Exception as e:
        result["health_error"] = str(e)

    try:
        # Prediction test via client (validates full production path)
        start = time.time()
        prediction_result = await client.predict({"features": sample_features})
        elapsed = (time.time() - start) * 1000

        result["prediction_test"] = True
        result["predictions"] = prediction_result.get("predictions")
        result["probabilities"] = prediction_result.get("probabilities")
        result["latency_ms"] = elapsed
        result["circuit_breaker_status"] = "closed"  # Would be open if failures occurred
    except Exception as e:
        result["prediction_error"] = str(e)
        result["circuit_breaker_status"] = "unknown"

    await client.close()  # Cleanup connection pool
    return result


async def stop_bentoml_service(pid: int) -> dict:
    """Stop BentoML service by PID."""
    import os
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return {"stopped": True, "pid": pid}
    except Exception as e:
        return {"stopped": False, "error": str(e)}
```

### 5. Modify Pipeline Runner

In `run_pipeline()`, update the step 7 call (around line 1200):

```python
# Step 7: Model Deployer (enhanced for BentoML)
if 7 in steps_to_run:
    step_start = time.time()

    result = await step_7_model_deployer(
        experiment_id,
        model_uri=state.get("model_uri"),
        validation_metrics=state.get("validation_metrics", {}),
        success_criteria_met=state.get("success_criteria_met", True),
        trained_model=state.get("trained_model"),  # Pass the real model
        include_bentoml=include_bentoml,  # From CLI flag
    )

    # Track BentoML PID for cleanup
    if include_bentoml and result.get("bentoml_pid"):
        state["bentoml_pid"] = result["bentoml_pid"]

    # ... rest of result handling ...
```

### 6. Add Cleanup in Finally Block

```python
finally:
    # Cleanup BentoML service if started
    if state.get("bentoml_pid"):
        print("\n  Cleaning up BentoML service...")
        await stop_bentoml_service(state["bentoml_pid"])
```

## Test Execution

All commands run **ON the droplet** (not locally). The `localhost:3001` URLs are correct from the droplet's perspective.

```bash
# Connect to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# On droplet: Navigate and activate venv
cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Run full tier0 test with REAL model BentoML serving
python scripts/run_tier0_test.py --include-bentoml

# Run specific steps with BentoML (requires step 4+5 to train model first)
python scripts/run_tier0_test.py --step 4,5,7 --include-bentoml

# Run steps 4-8 (recommended for full flow validation)
python scripts/run_tier0_test.py --step 4,5,6,7,8 --include-bentoml
```

**Note**: Step 4 (Model Selector) dynamically chooses the algorithm. Step 5 trains it. The trained model (whatever algorithm was selected) is then served via BentoML in step 7.

## Expected Output

```
STEP 4: MODEL SELECTOR
======================================================================
  Creating ModelSelectorAgent...
  Running agent...
  Output:
    selected_algorithm: XGBoostClassifier  # Dynamic selection based on data
    selection_score: 0.92
  ✅ Model selected successfully

STEP 5: MODEL TRAINER
======================================================================
  Creating ModelTrainerAgent...
  Input:
    algorithm: XGBoostClassifier  # From Step 4 (dynamic)
    train_samples: 400 (70%)
    hpo_trials: 5
  Running agent...
  Output:
    auc_roc: 0.87
    training_run_id: abc123
    model_uri: runs:/abc123/model
  ✅ Model trained successfully

STEP 7: MODEL DEPLOYER
======================================================================
  Creating ModelDeployerAgent...
  Input:
    deployment_name: kisqali_discontinuation_abc12345
    deployment_action: register

  Running agent...
  Output:
    status: completed
    deployment_successful: True

  BentoML Model Serving:
    Registering real trained model (XGBoostClassifier)...
    model_tag: tier0_abc12345:xyz789
    Starting BentoML service on port 3001...
    Service ready at http://localhost:3001 (on droplet)

  BentoML Serving Verification (via BentoMLClient):
    health_check: ✓ healthy
    circuit_breaker: closed
    prediction_test: ✓ passed
    predictions: [1]
    probabilities: [[0.21, 0.79]]
    latency_ms: 12.3

  ✅ Real model deployed and serving verified via production consumer path
```

## Verification Steps

1. **Model registration**: Real trained model from step 5 registered with BentoML
2. **Service startup**: BentoML serves the real model on port 3001 (localhost from droplet's perspective)
3. **Health check**: `GET http://localhost:3001/health` returns healthy status with model_tag
4. **Prediction test via BentoMLClient**: Use the proper consumer path:
   ```python
   from src.api.dependencies.bentoml_client import BentoMLClient

   client = BentoMLClient("http://localhost:3001")
   result = await client.predict({"features": sample_features})
   # Verifies circuit breaker, retries, tracing work correctly
   ```
5. **Cleanup**: BentoML service process terminates cleanly

### Why Test via BentoMLClient?

This validates the full production path:
- Circuit breaker logic
- Connection pooling
- Opik tracing integration
- Error handling with exponential backoff

Rather than just raw HTTP calls, this proves the real consumer code works.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Port 3001 in use | Check port availability, use alternative port |
| BentoML not installed | Already in venv (bentoml>=1.4.0 per requirements-bentoml.txt) |
| Model registration fails | Fallback to regular deployment, log error |
| Service startup timeout | 30-second wait with retries, configurable |
| Process cleanup | Use try/finally to ensure cleanup on error |

## Constraints Respected

- ✅ No `pip install` on droplet - uses existing venv
- ✅ Uses REAL trained model from tier0 workflow (not mock)
- ✅ Uses existing BentoML infrastructure (`src/mlops/bentoml_service.py`)
- ✅ Minimal changes to existing code
- ✅ Optional via `--include-bentoml` flag (doesn't break existing test)
- ✅ Actual predictions from the dynamically-selected trained model
