# Data Source Validation for Tier 1-5 Test Framework

## Problem Statement

The tier1-5 test framework has agents that silently use mock/hardcoded data instead of Supabase synthetic data:

| Agent | Issue |
|-------|-------|
| `health_score` | Returns 100% health because `_create_mock_status()` always returns "healthy" when no health_client injected |
| `gap_analyzer` | `use_mock=True` by default in `create_gap_analyzer_graph()` |
| `heterogeneous_optimizer` | Silently falls back to `MockDataConnector` if Supabase unavailable |
| `resource_optimizer` | Passes because purely computational (acceptable) |

## Solution Overview

1. **Create DataSourceValidator** - Detect mock vs real data usage
2. **Tighten health_score quality gate** - Reject perfect 100% scores
3. **Inject real Supabase connectors** - For agents that need external data
4. **Track data lineage** - Add observability to agent outputs

---

## Implementation Plan

### Phase 1: Create DataSourceValidator

**New file**: `src/testing/data_source_validator.py`

```python
class DataSourceType(Enum):
    SUPABASE = "supabase"
    MOCK = "mock"
    TIER0_PASSTHROUGH = "tier0"
    COMPUTATIONAL = "computational"
    UNKNOWN = "unknown"

class DataSourceValidator:
    """Validates that agents use appropriate data sources."""

    AGENT_DATA_SOURCE_REQUIREMENTS = {
        "health_score": {"acceptable": [DataSourceType.SUPABASE], "reject_mock": True},
        "gap_analyzer": {"acceptable": [DataSourceType.SUPABASE], "reject_mock": True},
        "heterogeneous_optimizer": {"acceptable": [DataSourceType.SUPABASE], "reject_mock": True},
        "resource_optimizer": {"acceptable": [DataSourceType.COMPUTATIONAL]},
        # Others: accept TIER0_PASSTHROUGH or SUPABASE
    }

    def validate(self, agent_name, agent_output, ...) -> DataSourceValidationResult:
        # Detect data source type from output patterns
        # Check if acceptable for agent
        # Return validation result
```

Key detection logic:
- `health_score`: If `overall_health_score == 100.0` → MOCK
- `gap_analyzer`/`heterogeneous_optimizer`: Check logs for "MockDataConnector"
- `resource_optimizer`: Always COMPUTATIONAL (acceptable)

### Phase 2: Tighten Quality Gates

**Modify**: `src/testing/agent_quality_gates.py`

```python
"health_score": {
    "required_output_fields": ["overall_health_score"],
    "min_required_fields_pct": 0.4,
    "data_quality_checks": {
        "overall_health_score": {
            "type": "float",
            "not_null": True,
            "must_not_be": 100.0,  # NEW: Reject perfect scores (mock indicator)
        },
        "component_health_score": {
            "must_not_be": 1.0,  # NEW: Reject perfect component scores
        },
    },
    "data_source_requirement": {  # NEW
        "reject_mock": True,
        "acceptable_sources": ["supabase"],
    },
},
```

### Phase 3: Create Real Health Client

**New file**: `src/agents/health_score/health_client.py`

```python
from src.api.dependencies.supabase_client import get_supabase, supabase_health_check

class SupabaseHealthClient:
    """Real health client that checks actual system components."""

    async def check(self, endpoint: str) -> dict:
        if endpoint == "/health/db":
            result = await supabase_health_check()
            return {"ok": result.get("status") == "healthy"}
        # ... other endpoints

def get_health_client_for_testing() -> SupabaseHealthClient:
    return SupabaseHealthClient()
```

### Phase 4: Update Agent Data Connector Defaults

**Modify**: `src/agents/gap_analyzer/graph.py` (line 42)
```python
def create_gap_analyzer_graph(
    ...
    use_mock: bool = False,  # CHANGED from True
):
```

**Modify**: `src/agents/heterogeneous_optimizer/nodes/cate_estimator.py`
```python
class CATEEstimatorNode:
    def __init__(self, data_connector=None, require_real_data: bool = False):
        self.require_real_data = require_real_data
        self.data_connector = data_connector or _get_default_data_connector()
        if self.require_real_data and "Mock" in type(self.data_connector).__name__:
            raise ValueError("Requires real data but no Supabase connector available")
```

### Phase 5: Update Test Framework

**Modify**: `scripts/run_tier1_5_test.py`

```python
# Add agent-specific kwargs injection
def _get_agent_kwargs(agent_name: str) -> dict:
    if agent_name == "health_score":
        from src.agents.health_score.health_client import get_health_client_for_testing
        return {"health_client": get_health_client_for_testing()}
    if agent_name == "gap_analyzer":
        return {"use_mock": False}
    if agent_name == "heterogeneous_optimizer":
        return {"require_real_data": True}
    return {}

# In test_agent():
agent_kwargs = _get_agent_kwargs(agent_name)
agent = agent_class(**agent_kwargs)

# Integrate DataSourceValidator into quality gate validation
quality_result = quality_validator.validate(
    agent_name=agent_name,
    output=result.agent_output,
    agent_instance=agent,
    execution_logs=captured_logs,
)
```

---

## Files to Modify

| File | Action | Changes |
|------|--------|---------|
| `src/testing/data_source_validator.py` | CREATE | DataSourceValidator class with detection logic |
| `src/testing/agent_quality_gates.py` | MODIFY | Add `must_not_be` checks for health_score, add `data_source_requirement` |
| `src/testing/quality_gate_validator.py` | MODIFY | Integrate DataSourceValidator |
| `src/testing/__init__.py` | MODIFY | Export new classes |
| `src/agents/health_score/health_client.py` | CREATE | SupabaseHealthClient factory |
| `src/agents/gap_analyzer/graph.py` | MODIFY | Change `use_mock=True` → `use_mock=False` |
| `src/agents/heterogeneous_optimizer/nodes/cate_estimator.py` | MODIFY | Add `require_real_data` parameter |
| `scripts/run_tier1_5_test.py` | MODIFY | Add `_get_agent_kwargs()`, integrate data source validation |
| `tests/unit/test_data_source_validator.py` | CREATE | Unit tests |

---

## Expected Results After Implementation

| Agent | Before | After | Reason |
|-------|--------|-------|--------|
| `health_score` | PASS (100%) | FAIL | `must_not_be: 100.0` check fails on mock data |
| `gap_analyzer` | FAIL (error) | PASS or FAIL based on Supabase | Uses real Supabase connector |
| `heterogeneous_optimizer` | PASS (mock fallback) | FAIL if no Supabase | `require_real_data=True` enforced |
| `resource_optimizer` | PASS | PASS | Computational, no change needed |

---

## Verification

```bash
# Run full tier1-5 test to verify data source validation
python scripts/run_tier1_5_test.py --skip-observability --verbose

# Test health_score specifically - should FAIL with mock data
python scripts/run_tier1_5_test.py --agents health_score --verbose

# Verify gap_analyzer uses Supabase
python scripts/run_tier1_5_test.py --agents gap_analyzer --verbose
```

---

## Implementation Order

1. Create `DataSourceValidator` class
2. Update `agent_quality_gates.py` with health_score tightening
3. Create `SupabaseHealthClient` for health_score
4. Change `gap_analyzer` default `use_mock=False`
5. Add `require_real_data` to `heterogeneous_optimizer`
6. Integrate into `run_tier1_5_test.py`
7. Add unit tests
8. Run full verification
