# BentoML Implementation Audit & Integration Plan

**Created**: 2025-12-25
**Scope**: Model serving endpoints for `model_deployer` and `prediction_synthesizer` agents
**Status**: In Progress

---

## Executive Summary

### Audit Findings

| Component | Status | Completeness |
|-----------|--------|--------------|
| BentoML Core (`/src/mlops/bentoml_*.py`) | ✅ Functional | 95% |
| model_deployer Agent | ✅ Functional | 100% |
| prediction_synthesizer Agent | ⚠️ Gap | 70% |
| API Layer Integration | ❌ Missing | 0% |
| ModelInferenceTool | ❌ Missing | 0% |
| Docker/Container Setup | ✅ Functional | 100% |
| Tests | ✅ Exists | 62 tests |

### Key Gaps Identified

1. **API Layer**: BentoML client initialization commented out in `main.py`
2. **ModelInferenceTool**: No HTTP-based tool for agents to call model endpoints
3. **prediction_synthesizer HTTP Client**: Uses in-process `ModelClient` Protocol, not HTTP endpoints
4. **Tool Registry Integration**: BentoML tools not registered for agent use

---

## Phase 1: Verification of Existing Implementation
**Estimated Context**: ~2,000 tokens per task

### Phase 1.1: Core BentoML Module Tests
- [ ] 1.1.1 Run existing bentoml_service tests
- [ ] 1.1.2 Run existing bentoml_packaging tests
- [ ] 1.1.3 Run existing bentoml_monitoring tests
- [ ] 1.1.4 Document any test failures

**Test Command**:
```bash
./venv/bin/python -m pytest tests/unit/test_mlops/test_bentoml/ -v --tb=short -n 2
```

### Phase 1.2: Service Template Tests
- [ ] 1.2.1 Run classification_service template tests
- [ ] 1.2.2 Run regression_service template tests
- [ ] 1.2.3 Run causal_service template tests

**Test Command**:
```bash
./venv/bin/python -m pytest tests/unit/test_mlops/test_bentoml/test_templates/ -v --tb=short -n 2
```

### Phase 1.3: model_deployer Agent Tests
- [ ] 1.3.1 Run model_deployer graph tests
- [ ] 1.3.2 Run deployment_orchestrator node tests
- [ ] 1.3.3 Run deployment_planner node tests
- [ ] 1.3.4 Verify BentoML integration in deployment flow

**Test Command**:
```bash
./venv/bin/python -m pytest tests/unit/test_agents/test_ml_foundation/test_model_deployer/ -v --tb=short -n 2
```

### Phase 1.4: prediction_synthesizer Agent Tests
- [ ] 1.4.1 Run prediction_synthesizer graph tests
- [ ] 1.4.2 Run model_orchestrator node tests
- [ ] 1.4.3 Verify ModelClient Protocol implementation

**Test Command**:
```bash
./venv/bin/python -m pytest tests/unit/test_agents/test_prediction_synthesizer/ -v --tb=short -n 2
```

---

## Phase 2: API Layer Integration
**Estimated Context**: ~3,000 tokens per task

### Phase 2.1: BentoML Client Setup in FastAPI
- [ ] 2.1.1 Create BentoML client configuration in `/src/api/dependencies/`
- [ ] 2.1.2 Implement async BentoML client wrapper
- [ ] 2.1.3 Add client initialization to `main.py` lifespan
- [ ] 2.1.4 Add health check endpoint for BentoML services

**Files to Create/Modify**:
- `src/api/dependencies/bentoml_client.py` (NEW)
- `src/api/main.py` (MODIFY - remove TODO)

### Phase 2.2: Prediction Route Integration
- [ ] 2.2.1 Update `/src/api/routes/explain.py` to use BentoML client
- [ ] 2.2.2 Add model prediction route to API
- [ ] 2.2.3 Add batch prediction route

**Files to Modify**:
- `src/api/routes/explain.py`
- `src/api/routes/predictions.py` (NEW or existing)

### Phase 2.3: API Integration Tests
- [ ] 2.3.1 Write unit tests for BentoML client wrapper
- [ ] 2.3.2 Write integration tests for prediction routes
- [ ] 2.3.3 Test error handling and retry logic

**Test Files**:
- `tests/unit/test_api/test_bentoml_client.py` (NEW)
- `tests/integration/test_api/test_prediction_routes.py` (NEW)

---

## Phase 3: ModelInferenceTool Implementation
**Estimated Context**: ~2,500 tokens per task

### Phase 3.1: Tool Definition
- [ ] 3.1.1 Define ModelInferenceTool interface following tool registry patterns
- [ ] 3.1.2 Implement HTTP client for BentoML endpoints
- [ ] 3.1.3 Add Feast feature retrieval integration
- [ ] 3.1.4 Add Opik tracing for tool calls

**Files to Create**:
- `src/tool_registry/tools/model_inference_tool.py` (NEW)

**Contract Reference**:
```python
class ModelInferenceInput(BaseModel):
    model_name: str
    entity_id: str
    features: Optional[Dict[str, Any]] = None  # If None, fetch from Feast
    time_horizon: str = "short_term"

class ModelInferenceOutput(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    feature_values: Dict[str, Any]
    latency_ms: float
```

### Phase 3.2: Tool Registry Integration
- [ ] 3.2.1 Register ModelInferenceTool in tool registry
- [ ] 3.2.2 Add tool documentation for agent discovery
- [ ] 3.2.3 Configure tool access permissions

**Files to Modify**:
- `src/tool_registry/__init__.py`
- `src/tool_registry/registry.py`

### Phase 3.3: ModelInferenceTool Tests
- [ ] 3.3.1 Write unit tests with mocked BentoML endpoints
- [ ] 3.3.2 Write unit tests for Feast integration
- [ ] 3.3.3 Write integration tests with actual endpoints (optional)

**Test Files**:
- `tests/unit/test_tool_registry/test_model_inference_tool.py` (NEW)

---

## Phase 4: prediction_synthesizer HTTP Integration
**Estimated Context**: ~2,000 tokens per task

### Phase 4.1: HTTP ModelClient Implementation
- [ ] 4.1.1 Create HTTPModelClient implementing ModelClient Protocol
- [ ] 4.1.2 Add connection pooling and retry logic
- [ ] 4.1.3 Add circuit breaker for failing endpoints
- [ ] 4.1.4 Add Opik tracing for predictions

**Files to Create**:
- `src/agents/prediction_synthesizer/clients/http_model_client.py` (NEW)

**Contract**:
```python
class HTTPModelClient:
    """HTTP-based ModelClient for BentoML endpoints."""

    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
    ):
        ...

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Call BentoML endpoint for prediction."""
        ...
```

### Phase 4.2: Client Factory and Configuration
- [ ] 4.2.1 Create ModelClientFactory for client instantiation
- [ ] 4.2.2 Add configuration for endpoint URLs
- [ ] 4.2.3 Support both in-process and HTTP clients

**Files to Create/Modify**:
- `src/agents/prediction_synthesizer/clients/__init__.py` (NEW)
- `src/agents/prediction_synthesizer/clients/factory.py` (NEW)
- `config/model_endpoints.yaml` (NEW)

### Phase 4.3: prediction_synthesizer HTTP Client Tests
- [ ] 4.3.1 Write unit tests for HTTPModelClient
- [ ] 4.3.2 Write unit tests for ModelClientFactory
- [ ] 4.3.3 Test failover between HTTP and in-process clients

**Test Files**:
- `tests/unit/test_agents/test_prediction_synthesizer/test_http_model_client.py` (NEW)

---

## Phase 5: End-to-End Integration Testing
**Estimated Context**: ~1,500 tokens per task

### Phase 5.1: Local Integration Tests
- [ ] 5.1.1 Start BentoML services locally via Docker Compose
- [ ] 5.1.2 Run end-to-end prediction flow test
- [ ] 5.1.3 Test model_deployer → BentoML → prediction_synthesizer flow
- [ ] 5.1.4 Verify Opik traces capture full flow

**Test Command**:
```bash
docker compose -f docker/bentoml/docker-compose.yaml up -d
./venv/bin/python -m pytest tests/e2e/test_model_serving/ -v --tb=short
```

### Phase 5.2: Performance Validation
- [ ] 5.2.1 Measure prediction latency (target: <100ms p95)
- [ ] 5.2.2 Test batch prediction throughput
- [ ] 5.2.3 Verify Prometheus metrics collection
- [ ] 5.2.4 Check Grafana dashboards render correctly

### Phase 5.3: Documentation Update
- [ ] 5.3.1 Update API documentation with prediction routes
- [ ] 5.3.2 Update agent documentation with ModelInferenceTool
- [ ] 5.3.3 Create BentoML operations runbook

---

## Progress Tracking

### Overall Progress
- [ ] Phase 1: Verification (0/4 sub-phases)
- [ ] Phase 2: API Layer (0/3 sub-phases)
- [ ] Phase 3: ModelInferenceTool (0/3 sub-phases)
- [ ] Phase 4: HTTP Client (0/3 sub-phases)
- [ ] Phase 5: E2E Testing (0/3 sub-phases)

### Test Results Log

| Phase | Tests Run | Passed | Failed | Notes |
|-------|-----------|--------|--------|-------|
| 1.1   | -         | -      | -      | -     |
| 1.2   | -         | -      | -      | -     |
| 1.3   | -         | -      | -      | -     |
| 1.4   | -         | -      | -      | -     |

---

## Key Files Reference

### Existing Implementation
| File | Lines | Description |
|------|-------|-------------|
| `src/mlops/bentoml_service.py` | 866 | Core model management |
| `src/mlops/bentoml_packaging.py` | 770 | Bento bundle creation |
| `src/mlops/bentoml_monitoring.py` | 717 | Prometheus + Opik |
| `src/agents/ml_foundation/model_deployer/nodes/deployment_orchestrator.py` | 750 | BentoML deployment |
| `src/agents/prediction_synthesizer/nodes/model_orchestrator.py` | ~400 | ModelClient Protocol |

### Files to Create
| File | Purpose |
|------|---------|
| `src/api/dependencies/bentoml_client.py` | FastAPI BentoML client |
| `src/tool_registry/tools/model_inference_tool.py` | Agent tool for model inference |
| `src/agents/prediction_synthesizer/clients/http_model_client.py` | HTTP-based ModelClient |
| `config/model_endpoints.yaml` | Endpoint configuration |

---

## Rollback Plan

If implementation causes issues:
1. All new code is in separate files (no breaking changes to existing)
2. prediction_synthesizer supports both in-process and HTTP clients via factory
3. API routes can be disabled via feature flag
4. Docker Compose allows running without BentoML services

---

## Notes

- **Memory Constraint**: Use max 4 pytest workers (`-n 4`) to avoid memory exhaustion
- **Context Window**: Each phase designed for ~2,000-3,000 tokens
- **Testing Strategy**: Test in small batches, document results before proceeding
- **Dependencies**: Phase 3 depends on Phase 2.1 (client wrapper)
