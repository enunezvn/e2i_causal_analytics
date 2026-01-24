# Digital Twin System Improvement Plan
## Non-Breaking Enhancements for Performance, Testing, and Reliability

**Created**: 2026-01-24
**Status**: Planned
**Target**: `src/digital_twin/`

---

## Executive Summary

This plan outlines improvements to the digital twin system across 6 phases, organized by dependencies and logical grouping. All changes are non-breaking and follow existing patterns in the codebase.

---

## Phase 1: Enhanced Logging and Observability Foundation
**Dependencies**: None (enables debugging for all subsequent phases)
**Complexity**: Low

### 1.1 DEBUG-Level Logging for Effect Modifier Calculations

**Location**: `src/digital_twin/simulation_engine.py`

**Changes Required**:
- Add DEBUG-level logging in `_calculate_individual_effect()` method
- Log each modifier calculation step:
  - Decile multiplier calculation
  - Engagement multiplier
  - Adoption stage multiplier
  - Intensity multiplier
  - Duration factor
  - Channel-specific adjustments
  - Propensity weighting

**Pattern to Follow**: Existing logging patterns in `src/agents/causal_impact/mlflow_tracker.py`

**Implementation Example**:
```python
logger.debug(
    f"Effect modifiers for twin {twin.twin_id}: "
    f"decile_mult={effect_multiplier:.4f}, "
    f"engagement_base={engagement:.4f}"
)
```

**Validation Criteria**:
- All modifier calculations logged at DEBUG level
- Log output includes twin identifier for traceability
- No impact on INFO/WARNING level logs
- Performance impact < 1ms per simulation when DEBUG disabled

---

## Phase 2: Unit Tests for Edge Cases and Data Leakage Prevention
**Dependencies**: Phase 1 (logging helps debug test failures)
**Complexity**: Medium

### 2.1 Edge Case Tests for Effect Modifiers

**Location**: `tests/unit/test_digital_twin/test_simulation_engine.py`

**New Test Class**: `TestEffectModifierEdgeCases`

| Test Name | Description | Expected Behavior |
|-----------|-------------|-------------------|
| `test_extreme_low_decile` | Decile = 0 (boundary) | Uses minimum valid decile or raises |
| `test_extreme_high_decile` | Decile = 11 (above max) | Uses maximum valid decile or raises |
| `test_zero_engagement_score` | engagement = 0.0 | Applies minimum multiplier (0.8) |
| `test_max_engagement_score` | engagement = 1.0 | Applies maximum multiplier (1.2) |
| `test_negative_engagement` | engagement = -0.5 | Clamps to 0.0 or raises |
| `test_invalid_adoption_stage` | adoption = "unknown" | Falls back to default (1.0) |
| `test_zero_intensity_multiplier` | intensity = 0.0 | Produces near-zero effect |
| `test_extreme_intensity` | intensity = 10.0 | Effect capped appropriately |
| `test_zero_duration` | duration_weeks = 0 | Handled gracefully |
| `test_combined_extreme_modifiers` | All modifiers at extremes | No overflow/underflow |

### 2.2 Data Leakage Prevention Tests

**Location**: `tests/unit/test_digital_twin/test_twin_generator.py`

**New Test Class**: `TestDataLeakagePrevention`

| Test Name | Description | Validation |
|-----------|-------------|------------|
| `test_no_future_features_in_training` | Verify feature columns don't leak future info | Check feature timestamps |
| `test_temporal_ordering_preserved` | Training data respects temporal boundaries | Validate date ordering |
| `test_target_not_in_features` | Target column excluded from features | Assert target not in X |
| `test_no_train_test_contamination` | Training twins don't leak into test | Check twin IDs unique |
| `test_feature_statistics_from_train_only` | Feature stats computed from training only | Validate stat source |

**Pattern to Follow**: `src/agents/ml_foundation/data_preparer/nodes/leakage_detector.py`

### 2.3 Boundary Condition Tests

**Location**: `tests/unit/test_digital_twin/test_simulation_engine.py`

**New Test Class**: `TestBoundaryConditions`

| Test Name | Description |
|-----------|-------------|
| `test_exactly_100_twins` | Minimum viable population |
| `test_99_twins_fails` | Below threshold fails gracefully |
| `test_empty_population` | Zero twins handled |
| `test_single_twin` | Degenerate case |
| `test_ci_bounds_equal` | ATE with zero variance |
| `test_negative_ate` | Negative treatment effects |
| `test_confidence_level_extremes` | 80% and 99% confidence levels |

---

## Phase 3: Performance and Load Testing Infrastructure
**Dependencies**: Phase 2 (unit tests ensure correctness before stress testing)
**Complexity**: Medium-High

### 3.1 Load Tests for Large Twin Populations

**Location**: `tests/performance/test_digital_twin_performance.py` (new file)

| Test Name | Population Size | Timeout | Memory Limit |
|-----------|-----------------|---------|--------------|
| `test_10k_twins_simulation` | 10,000 | 30s | 512MB |
| `test_50k_twins_simulation` | 50,000 | 120s | 2GB |
| `test_100k_twins_generation` | 100,000 | 300s | 4GB |
| `test_scaling_behavior` | 1K -> 50K | N/A | Track growth |

**Test Markers**:
```python
@pytest.mark.slow
@pytest.mark.xdist_group(name="digital_twin_performance")
```

### 3.2 Memory Profiling Tests

**Location**: `tests/performance/test_digital_twin_memory.py` (new file)

| Test Name | Description |
|-----------|-------------|
| `test_memory_growth_linear` | Memory scales linearly with population |
| `test_no_memory_leak_repeated_simulations` | Memory stable across multiple runs |
| `test_gc_releases_twin_objects` | Garbage collection works correctly |
| `test_peak_memory_under_limit` | Peak memory within configured bounds |

**Tools**: Use `tracemalloc` (stdlib) for memory profiling

### 3.3 Timeout Scenario Tests

**Location**: `tests/unit/test_digital_twin/test_simulation_engine.py`

**New Test Class**: `TestTimeoutScenarios`

| Test Name | Description |
|-----------|-------------|
| `test_simulation_timeout_handling` | Graceful timeout |
| `test_partial_results_on_timeout` | Return partial results if available |
| `test_timeout_cleanup` | Resources released on timeout |

---

## Phase 4: Concurrency and Error Recovery Tests
**Dependencies**: Phase 3 (performance baseline established)
**Complexity**: High

### 4.1 Concurrency Tests

**Location**: `tests/integration/test_digital_twin_concurrency.py` (new file)

| Test Name | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| `test_concurrent_cache_reads` | Multiple readers | No race conditions |
| `test_concurrent_cache_writes` | Multiple writers | Data integrity preserved |
| `test_cache_invalidation_during_read` | Read during invalidation | Consistent state |
| `test_concurrent_simulations` | Parallel simulations | Independent results |
| `test_transaction_rollback` | DB error mid-operation | Clean rollback |

**Test Markers**:
```python
@pytest.mark.requires_redis
@pytest.mark.xdist_group(name="digital_twin_concurrency")
```

### 4.2 Error Recovery Tests

**Location**: `tests/unit/test_digital_twin/test_error_recovery.py` (new file)

| Test Name | Error Type | Recovery Expectation |
|-----------|------------|---------------------|
| `test_malformed_intervention_config` | Invalid JSON | ValidationError with details |
| `test_missing_required_fields` | Incomplete config | Clear error message |
| `test_database_connection_failure` | DB unavailable | Graceful degradation |
| `test_mlflow_unavailable` | MLflow down | Simulation continues |
| `test_redis_connection_timeout` | Redis timeout | Fall back to no-cache |
| `test_partial_twin_data` | Incomplete features | Default values applied |

---

## Phase 5: Result Caching Implementation
**Dependencies**: Phase 4 (concurrency patterns established)
**Complexity**: Medium

### 5.1 Simulation Result Cache

**Location**: `src/digital_twin/simulation_cache.py` (new file)

**Pattern to Follow**: `src/repositories/data_cache.py`

**Proposed Interface**:
```python
@dataclass
class SimulationCacheConfig:
    """Configuration for simulation result caching."""
    ttl_seconds: int = 1800  # 30 minutes
    prefix: str = "simulation"
    enabled: bool = True

class SimulationCache:
    """Cache for simulation results."""

    async def get_cached_result(
        self,
        intervention_config: InterventionConfig,
        population_filter: PopulationFilter,
        model_id: UUID,
    ) -> Optional[SimulationResult]:
        """Get cached simulation result if available."""
        ...

    async def cache_result(
        self,
        result: SimulationResult,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache a simulation result."""
        ...

    async def invalidate_model_cache(self, model_id: UUID) -> int:
        """Invalidate all cached results for a model."""
        ...
```

### 5.2 Integration with SimulationEngine

**Location**: `src/digital_twin/simulation_engine.py`

**Non-Breaking Modification**:
```python
def simulate(
    self,
    intervention_config: InterventionConfig,
    population_filter: Optional[PopulationFilter] = None,
    confidence_level: float = 0.95,
    calculate_heterogeneity: bool = True,
    use_cache: bool = True,  # NEW - default True maintains behavior
    cache: Optional[SimulationCache] = None,  # NEW - optional injection
) -> SimulationResult:
```

### 5.3 Cache Tests

**Location**: `tests/unit/test_digital_twin/test_simulation_cache.py` (new file)

| Test Name | Description |
|-----------|-------------|
| `test_cache_hit_returns_result` | Valid cache returns result |
| `test_cache_miss_executes_simulation` | Miss triggers execution |
| `test_cache_key_uniqueness` | Different configs produce different keys |
| `test_cache_expiration` | TTL honored |
| `test_cache_invalidation_on_model_update` | Model change clears cache |

---

## Phase 6: Automatic Retraining Integration
**Dependencies**: Phase 2, Phase 5 (fidelity tracking and caching in place)
**Complexity**: High

### 6.1 Fidelity-Triggered Retraining Service

**Location**: `src/digital_twin/retraining_service.py` (new file)

**Pattern to Follow**: `src/services/retraining_trigger.py`

**Proposed Interface**:
```python
@dataclass
class TwinRetrainingConfig:
    """Configuration for twin model retraining triggers."""
    fidelity_threshold: float = 0.70
    min_validations_for_decision: int = 5
    cooldown_hours: int = 24
    auto_approve_threshold: float = 0.50  # Very poor fidelity

class TwinRetrainingService:
    """Service for automatic twin model retraining."""

    async def evaluate_retraining_need(
        self,
        model_id: UUID,
    ) -> RetrainingDecision:
        """Evaluate if model needs retraining based on fidelity."""
        ...

    async def trigger_retraining(
        self,
        model_id: UUID,
        reason: TriggerReason,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> RetrainingJob:
        """Trigger model retraining."""
        ...
```

### 6.2 Integration with FidelityTracker

**Location**: `src/digital_twin/fidelity_tracker.py`

**Non-Breaking Modification**:
```python
def __init__(
    self,
    repository=None,
    retraining_service=None,  # NEW - optional injection
    auto_trigger_retraining: bool = False,  # NEW - opt-in behavior
):
```

### 6.3 End-to-End Workflow Tests

**Location**: `tests/integration/test_digital_twin_e2e.py` (new file)

| Test Name | Workflow | Validation |
|-----------|----------|------------|
| `test_full_workflow_deploy` | Generate -> Simulate -> Deploy -> Validate | All steps complete |
| `test_full_workflow_skip` | Generate -> Simulate -> Skip (low effect) | Early exit logged |
| `test_fidelity_triggers_retrain` | Validate with poor fidelity -> Retrain | Retraining triggered |
| `test_retrained_model_improved` | Retrain -> New simulation | Better fidelity score |

---

## Implementation Sequence

```
Phase 1: Logging ─────────────────────────────────────────────────────────────►
                  │
Phase 2: Unit Tests ─────────────────────────────────────────────────────────►
                     │
                     └──► Phase 3: Performance Tests ─────────────────────────►
                                                       │
                                                       └──► Phase 4: Concurrency
                                                                              │
                                                                              └──► Phase 5: Caching
                                                                                                  │
                                                                                                  └──► Phase 6: Retraining
```

---

## New Files Summary

| Phase | New Files |
|-------|-----------|
| 2 | - |
| 3 | `tests/performance/test_digital_twin_performance.py`, `tests/performance/test_digital_twin_memory.py` |
| 4 | `tests/integration/test_digital_twin_concurrency.py`, `tests/unit/test_digital_twin/test_error_recovery.py` |
| 5 | `src/digital_twin/simulation_cache.py`, `tests/unit/test_digital_twin/test_simulation_cache.py` |
| 6 | `src/digital_twin/retraining_service.py`, `tests/integration/test_digital_twin_e2e.py` |

---

## Configuration Additions

Add to `config/digital_twin_config.yaml`:

```yaml
# Result Caching (Phase 5)
simulation_cache:
  enabled: true
  ttl_seconds: 1800           # 30 minutes
  max_cached_results: 1000    # LRU eviction threshold
  key_prefix: "twin_sim:"

# Automatic Retraining (Phase 6)
auto_retraining:
  enabled: false              # Opt-in for safety
  fidelity_threshold: 0.70
  cooldown_hours: 24
  auto_approve_threshold: 0.50
  notification_webhook: null
```

---

## Validation Criteria Summary

| Phase | Success Criteria |
|-------|-----------------|
| 1 | DEBUG logs show all modifier calculations; no performance impact when disabled |
| 2 | All edge cases have explicit tests; zero data leakage in training pipeline |
| 3 | 50K twin simulation completes in < 2 min; memory growth linear |
| 4 | No race conditions in 10 concurrent operations; clean rollbacks |
| 5 | Cache hit rate > 50% for repeated simulations; < 10ms cache lookup |
| 6 | Retraining triggers automatically when fidelity < 0.70; new model improves score |

---

## Testing Strategy

### Memory-Safe Test Execution

Per CLAUDE.md requirements:
- Maximum 4 workers: `-n 4`
- Use `--dist=loadscope` for module grouping
- 30s timeout per test
- Heavy tests marked with `@pytest.mark.xdist_group`

### Test Directory Structure

```
tests/
├── unit/test_digital_twin/
│   ├── test_simulation_engine.py      # Extended with edge cases
│   ├── test_twin_generator.py         # Extended with leakage tests
│   ├── test_fidelity_tracker.py       # Extended with retraining tests
│   ├── test_error_recovery.py         # NEW
│   └── test_simulation_cache.py       # NEW
├── integration/
│   ├── test_digital_twin_workflow.py  # Extended
│   ├── test_digital_twin_concurrency.py  # NEW
│   └── test_digital_twin_e2e.py       # NEW
└── performance/
    ├── test_digital_twin_performance.py  # NEW
    └── test_digital_twin_memory.py       # NEW
```

### Pytest Markers

```python
pytest.mark.slow  # Performance tests
pytest.mark.requires_redis  # Cache tests
pytest.mark.xdist_group(name="digital_twin_heavy")  # Memory-heavy tests
pytest.mark.xdist_group(name="digital_twin_performance")  # Load tests
pytest.mark.xdist_group(name="digital_twin_concurrency")  # Concurrency tests
```

---

## Critical Files Reference

| File | Modification Type |
|------|-------------------|
| `src/digital_twin/simulation_engine.py` | Extend (logging, caching) |
| `src/digital_twin/fidelity_tracker.py` | Extend (retraining hooks) |
| `src/digital_twin/simulation_cache.py` | New |
| `src/digital_twin/retraining_service.py` | New |
| `tests/unit/test_digital_twin/test_simulation_engine.py` | Extend (edge cases) |
| `tests/unit/test_digital_twin/test_twin_generator.py` | Extend (leakage tests) |
| `config/digital_twin_config.yaml` | Extend (new sections) |
