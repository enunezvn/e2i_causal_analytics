# Causal Discovery Enhancements Implementation Plan

**Status**: ✅ COMPLETE
**Created**: 2025-12-30
**Based On**: causal-discovery-integration-complete.md "Next Steps"

---

## Executive Summary

Enhance the existing causal discovery module with:
1. **FCI Algorithm** - Handle latent confounders (bidirected edges)
2. **LiNGAM Variants** - Support non-Gaussian data (DirectLiNGAM, ICA-LiNGAM)
3. **Discovery Caching** - Avoid redundant computation on identical data
4. **Opik Dashboard** - Discovery-specific observability metrics

**Estimated Scope**: ~15 new files, ~2,000 lines of code, ~60 new tests

---

## Phase Breakdown (Context-Window Friendly)

### Phase 1: FCI Algorithm Wrapper ✅ [COMPLETE]
**Scope**: Add FCI (Fast Causal Inference) for latent confounder detection
**Files**: 2 new, 2 modified
**Tests**: ~15 unit tests (run in 1 batch)

### Phase 2: LiNGAM Variants ✅ [COMPLETE]
**Scope**: Add DirectLiNGAM and ICA-LiNGAM for non-Gaussian data
**Files**: 2 new, 2 modified
**Tests**: ~20 unit tests (run in 2 batches of 10)

### Phase 3: Discovery Caching ✅ [COMPLETE]
**Scope**: Redis + in-memory caching layer with data hashing
**Files**: 3 new, 2 modified
**Tests**: ~15 unit tests (run in 1 batch)

### Phase 4: Opik Dashboard Integration ✅ [COMPLETE]
**Scope**: Discovery-specific metrics and traces
**Files**: 1 new (`observability.py`), 3 modified (`runner.py`, `gate.py`, `observability.yaml`)
**Tests**: 23 unit tests (all passing)

---

## Phase 1: FCI Algorithm Wrapper

### 1.1 Objective
Add FCI (Fast Causal Inference) algorithm support to detect latent confounders. FCI outputs:
- Directed edges: `X -> Y` (definite causal direction)
- Bidirected edges: `X <-> Y` (latent confounder present)
- Circle marks: `X o-> Y` (uncertain endpoint)

### 1.2 Files to Create

#### `src/causal_engine/discovery/algorithms/fci_wrapper.py`
```python
"""
FCI (Fast Causal Inference) algorithm wrapper.

FCI extends PC to handle latent confounders by:
1. Running modified PC with possible ancestors
2. Orienting edges using FCI orientation rules
3. Outputting PAG (Partial Ancestral Graph) with bidirected edges

Key output: Bidirected edges indicate latent confounders (L -> X, L -> Y)
"""
```

**Implementation Details**:
- Import: `from causallearn.search.ConstraintBased.FCI import fci`
- Input: DataFrame, alpha, max_cond_vars
- Output: AlgorithmResult with adjacency matrix (handle PAG marks)
- Special: `supports_latent_confounders() -> True`

### 1.3 Files to Modify

#### `src/causal_engine/discovery/algorithms/__init__.py`
- Add: `from .fci_wrapper import FCIAlgorithm`
- Export: `__all__ = ["GESAlgorithm", "PCAlgorithm", "FCIAlgorithm"]`

#### `src/causal_engine/discovery/runner.py`
- Add to `ALGORITHM_REGISTRY`: `DiscoveryAlgorithmType.FCI: FCIAlgorithm`

### 1.4 Tests to Create

#### `tests/unit/test_causal_engine/test_discovery/test_fci_wrapper.py`
```python
# Test cases (15 tests):
# 1. test_fci_basic_discovery - Simple DAG recovery
# 2. test_fci_latent_confounder_detection - X <-> Y edges
# 3. test_fci_supports_latent_confounders - Protocol compliance
# 4. test_fci_empty_data_error - Validation
# 5. test_fci_missing_values_error - Validation
# 6. test_fci_single_variable_error - Validation
# 7. test_fci_alpha_parameter - Significance level
# 8. test_fci_max_cond_vars - Conditioning set limit
# 9. test_fci_convergence_metadata - Runtime tracking
# 10. test_fci_edge_type_detection - Directed vs bidirected
# 11. test_fci_import_error_handling - causal-learn missing
# 12. test_fci_algorithm_failure_handling - Exception handling
# 13. test_fci_adjacency_matrix_format - Output validation
# 14. test_fci_edge_list_format - Output validation
# 15. test_fci_integration_with_runner - Ensemble participation
```

### 1.5 Test Execution
```bash
# Run Phase 1 tests only (single batch)
pytest tests/unit/test_causal_engine/test_discovery/test_fci_wrapper.py -v -x --tb=short
```

---

## Phase 2: LiNGAM Variants

### 2.1 Objective
Add LiNGAM algorithms for non-Gaussian, linear causal discovery:
- **DirectLiNGAM**: Fast, direct estimation of causal order
- **ICA-LiNGAM**: Uses Independent Component Analysis

LiNGAM assumes: Linear relationships + non-Gaussian error terms

### 2.2 Files to Create

#### `src/causal_engine/discovery/algorithms/lingam_wrapper.py`
```python
"""
LiNGAM (Linear Non-Gaussian Acyclic Model) wrappers.

Two variants:
1. DirectLiNGAM: Iteratively identifies root variables
2. ICA-LiNGAM: Uses ICA to estimate mixing matrix

Assumptions:
- Linear causal relationships
- Non-Gaussian error terms (enables identifiability)
- Acyclic structure
"""

class DirectLiNGAMAlgorithm(BaseDiscoveryAlgorithm):
    # Import: from causallearn.search.FCMBased.lingam import DirectLiNGAM
    pass

class ICALiNGAMAlgorithm(BaseDiscoveryAlgorithm):
    # Import: from causallearn.search.FCMBased.lingam import ICALiNGAM
    pass
```

**Implementation Details**:
- DirectLiNGAM: Uses regression-based pruning
- ICA-LiNGAM: Uses FastICA for unmixing
- Both: Return adjacency matrix with continuous weights -> threshold to binary
- Config: Use `assume_gaussian: False` in DiscoveryConfig to enable

### 2.3 Files to Modify

#### `src/causal_engine/discovery/algorithms/__init__.py`
- Add imports for DirectLiNGAMAlgorithm, ICALiNGAMAlgorithm
- Update `__all__`

#### `src/causal_engine/discovery/runner.py`
- Add to `ALGORITHM_REGISTRY`:
  - `DiscoveryAlgorithmType.DIRECT_LINGAM: DirectLiNGAMAlgorithm`
  - `DiscoveryAlgorithmType.ICA_LINGAM: ICALiNGAMAlgorithm`

### 2.4 Tests to Create

#### `tests/unit/test_causal_engine/test_discovery/test_lingam_wrapper.py`

**Batch 1** (10 tests - DirectLiNGAM):
```python
# 1. test_direct_lingam_basic_discovery
# 2. test_direct_lingam_causal_order
# 3. test_direct_lingam_adjacency_weights
# 4. test_direct_lingam_supports_latent_confounders_false
# 5. test_direct_lingam_empty_data_error
# 6. test_direct_lingam_missing_values_error
# 7. test_direct_lingam_convergence_metadata
# 8. test_direct_lingam_import_error_handling
# 9. test_direct_lingam_algorithm_failure_handling
# 10. test_direct_lingam_integration_with_runner
```

**Batch 2** (10 tests - ICA-LiNGAM):
```python
# 11. test_ica_lingam_basic_discovery
# 12. test_ica_lingam_mixing_matrix
# 13. test_ica_lingam_adjacency_weights
# 14. test_ica_lingam_supports_latent_confounders_false
# 15. test_ica_lingam_empty_data_error
# 16. test_ica_lingam_missing_values_error
# 17. test_ica_lingam_convergence_metadata
# 18. test_ica_lingam_import_error_handling
# 19. test_ica_lingam_algorithm_failure_handling
# 20. test_ica_lingam_integration_with_runner
```

### 2.5 Test Execution
```bash
# Batch 1: DirectLiNGAM tests
pytest tests/unit/test_causal_engine/test_discovery/test_lingam_wrapper.py -v -x --tb=short -k "direct_lingam"

# Batch 2: ICA-LiNGAM tests
pytest tests/unit/test_causal_engine/test_discovery/test_lingam_wrapper.py -v -x --tb=short -k "ica_lingam"
```

---

## Phase 3: Discovery Caching

### 3.1 Objective
Implement caching to avoid redundant discovery runs:
- **Data Hash**: SHA-256 of DataFrame content + config
- **Redis Cache**: Primary cache with TTL
- **In-Memory Cache**: Fallback when Redis unavailable
- **Cache Key**: `discovery:{data_hash}:{config_hash}`

### 3.2 Files to Create

#### `src/causal_engine/discovery/cache.py`
```python
"""
Discovery result caching with Redis + in-memory fallback.

Features:
- Data-aware hashing (DataFrame content + column order)
- Config-aware hashing (algorithm selection, thresholds)
- TTL-based expiration (default 1 hour)
- LRU eviction for in-memory cache
- Cache statistics tracking

Usage:
    cache = DiscoveryCache(redis_url="redis://localhost:6379")

    # Check cache
    cached = await cache.get(data, config)
    if cached:
        return cached

    # Run discovery and cache result
    result = await runner.discover_dag(data, config)
    await cache.set(data, config, result)
"""

@dataclass
class CacheConfig:
    redis_url: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour
    max_memory_items: int = 100
    enable_redis: bool = True
    enable_memory: bool = True

class DiscoveryCache:
    async def get(self, data: pd.DataFrame, config: DiscoveryConfig) -> Optional[DiscoveryResult]: ...
    async def set(self, data: pd.DataFrame, config: DiscoveryConfig, result: DiscoveryResult) -> None: ...
    def invalidate(self, data_hash: Optional[str] = None) -> int: ...
    def get_stats(self) -> CacheStats: ...
```

#### `src/causal_engine/discovery/hasher.py`
```python
"""
Data and config hashing utilities.

Produces deterministic hashes for:
- DataFrame content (values, dtypes, column order)
- DiscoveryConfig (algorithm list, parameters)
- Combined cache keys
"""

def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 hash of DataFrame content."""
    ...

def hash_config(config: DiscoveryConfig) -> str:
    """SHA-256 hash of config parameters."""
    ...

def make_cache_key(data_hash: str, config_hash: str) -> str:
    """Combine hashes into cache key."""
    return f"discovery:{data_hash}:{config_hash}"
```

### 3.3 Files to Modify

#### `src/causal_engine/discovery/runner.py`
- Add `cache: Optional[DiscoveryCache]` to `__init__`
- Add cache check at start of `discover_dag`
- Add cache set after successful discovery
- Add `use_cache: bool = True` parameter

#### `src/causal_engine/discovery/__init__.py`
- Export: `DiscoveryCache`, `CacheConfig`

### 3.4 Database Migration (Optional)

#### `database/ml/027_discovery_cache_tables.sql`
```sql
-- Optional: Persistent cache for long-term result storage
CREATE TABLE IF NOT EXISTS ml.discovery_cache (
    cache_key VARCHAR(128) PRIMARY KEY,
    data_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    result_json JSONB NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_discovery_cache_expires ON ml.discovery_cache(expires_at);
CREATE INDEX idx_discovery_cache_data_hash ON ml.discovery_cache(data_hash);
```

### 3.5 Tests to Create

#### `tests/unit/test_causal_engine/test_discovery/test_cache.py`
```python
# Test cases (15 tests):
# 1. test_cache_miss_returns_none
# 2. test_cache_hit_returns_result
# 3. test_cache_ttl_expiration
# 4. test_cache_different_data_different_key
# 5. test_cache_different_config_different_key
# 6. test_cache_same_data_reordered_columns_same_key
# 7. test_cache_invalidation_by_data_hash
# 8. test_cache_invalidation_all
# 9. test_cache_stats_tracking
# 10. test_cache_memory_fallback_when_redis_down
# 11. test_cache_lru_eviction
# 12. test_hash_dataframe_deterministic
# 13. test_hash_config_deterministic
# 14. test_runner_uses_cache
# 15. test_runner_skips_cache_when_disabled
```

### 3.6 Test Execution
```bash
# Run Phase 3 tests (single batch)
pytest tests/unit/test_causal_engine/test_discovery/test_cache.py -v -x --tb=short
```

---

## Phase 4: Opik Dashboard Integration ✅ [COMPLETE]

### 4.1 Objective
Add discovery-specific observability metrics to Opik:
- **Trace discovery runs** with algorithm details
- **Log gate decisions** (ACCEPT/REVIEW/REJECT/AUGMENT)
- **Track cache performance** (hit rate, latency savings)
- **Monitor ensemble agreement** across algorithms

### 4.2 Files to Create

#### `src/causal_engine/discovery/observability.py`
```python
"""
Opik integration for causal discovery observability.

Traces:
- discovery_run: Overall discovery operation
  - algorithm_execution: Per-algorithm timing
  - ensemble_voting: Edge voting process
  - gate_evaluation: Decision making
  - cache_lookup: Cache performance

Metrics:
- discovery_latency_ms: Total discovery time
- algorithm_agreement_ratio: Edge consensus
- gate_decision_distribution: ACCEPT/REVIEW/REJECT/AUGMENT counts
- cache_hit_rate: Cache effectiveness
- edges_discovered: Edge count per run
"""

from src.mlops.opik_connector import OpikConnector

class DiscoveryTracer:
    def __init__(self, opik: Optional[OpikConnector] = None):
        self.opik = opik or OpikConnector()

    @asynccontextmanager
    async def trace_discovery(
        self,
        session_id: Optional[UUID] = None,
        algorithms: List[str] = None,
    ) -> AsyncGenerator[DiscoverySpan, None]:
        """Trace a complete discovery run."""
        ...

    async def log_algorithm_result(
        self,
        parent_span: DiscoverySpan,
        result: AlgorithmResult,
    ) -> None:
        """Log individual algorithm execution."""
        ...

    async def log_gate_decision(
        self,
        parent_span: DiscoverySpan,
        decision: GateDecision,
        confidence: float,
        reasons: List[str],
    ) -> None:
        """Log gate evaluation result."""
        ...

    async def log_cache_event(
        self,
        parent_span: DiscoverySpan,
        hit: bool,
        latency_ms: float,
    ) -> None:
        """Log cache lookup result."""
        ...
```

### 4.3 Files to Modify

#### `src/causal_engine/discovery/runner.py`
- Add `tracer: Optional[DiscoveryTracer]` to `__init__`
- Wrap `discover_dag` with tracing context
- Log algorithm results, gate decisions, cache events

#### `src/causal_engine/discovery/gate.py`
- Add tracing hooks for gate evaluation
- Log confidence scores and decision reasons

#### `config/observability.yaml`
- Add discovery-specific metric definitions
- Add dashboard configuration

### 4.4 Tests to Create

#### `tests/unit/test_causal_engine/test_discovery/test_observability.py`
```python
# Test cases (10 tests):
# 1. test_tracer_creates_span_for_discovery
# 2. test_tracer_logs_algorithm_results
# 3. test_tracer_logs_gate_decisions
# 4. test_tracer_logs_cache_events
# 5. test_tracer_graceful_degradation_when_opik_down
# 6. test_tracer_span_hierarchy_correct
# 7. test_tracer_metadata_included
# 8. test_tracer_latency_metrics_accurate
# 9. test_tracer_integration_with_runner
# 10. test_tracer_integration_with_gate
```

### 4.5 Test Execution
```bash
# Run Phase 4 tests (single batch)
pytest tests/unit/test_causal_engine/test_discovery/test_observability.py -v -x --tb=short
```

---

## Testing Strategy

### Memory-Safe Execution
All tests run with `-n 4` max workers (per CLAUDE.md guidelines):
```bash
# Full discovery test suite
pytest tests/unit/test_causal_engine/test_discovery/ -v -n 4 --dist=loadscope
```

### Batch Sizes
- Phase 1: 15 tests (1 batch)
- Phase 2: 20 tests (2 batches of 10)
- Phase 3: 15 tests (1 batch)
- Phase 4: 10 tests (1 batch)
- **Total**: 60 new tests

### Integration Tests
After all phases, run integration tests:
```bash
pytest tests/integration/test_graph_builder_discovery.py -v -n 2
```

---

## Progress Tracking

### Phase 1: FCI Algorithm ✅
- [x] Create `fci_wrapper.py`
- [x] Update `algorithms/__init__.py`
- [x] Update `runner.py` registry
- [x] Create `test_fci_wrapper.py`
- [x] Run tests (15/15 passing)

### Phase 2: LiNGAM Variants ✅
- [x] Create `lingam_wrapper.py`
- [x] Update `algorithms/__init__.py`
- [x] Update `runner.py` registry
- [x] Create `test_lingam_wrapper.py`
- [x] Run DirectLiNGAM tests (10/10 passing)
- [x] Run ICA-LiNGAM tests (10/10 passing)

### Phase 3: Discovery Caching ✅
- [x] Create `hasher.py`
- [x] Create `cache.py`
- [x] Update `runner.py` with cache support
- [x] Update `__init__.py` exports
- [x] Create `test_cache.py`
- [x] Run tests (15/15 passing)

### Phase 4: Opik Dashboard ✅ [COMPLETE]
- [x] Create `observability.py` with DiscoveryTracer, DiscoverySpan, DiscoverySpanMetadata
- [x] Update `runner.py` with tracing integration (trace_discovery, log_algorithm_result, log_ensemble_result)
- [x] Update `gate.py` with tracing hooks (evaluate_with_tracing, set_tracer)
- [x] Update `config/observability.yaml` with causal_discovery metrics section
- [x] Create `test_observability.py` with 23 tests (all passing)

---

## Rollback Plan

If issues arise:
1. Each phase is independently reversible
2. New algorithms can be unregistered from `ALGORITHM_REGISTRY`
3. Cache can be disabled via `use_cache=False`
4. Tracing can be disabled via `tracer=None`

---

## Dependencies

### causal-learn Features Used
```python
# FCI
from causallearn.search.ConstraintBased.FCI import fci

# LiNGAM
from causallearn.search.FCMBased import lingam
# - lingam.DirectLiNGAM
# - lingam.ICALiNGAM
```

### Redis (for caching)
- Already available in project (`redis>=5.0.0` in pyproject.toml)
- Default: `redis://localhost:6379`

### Opik (for observability)
- Already integrated via `src/mlops/opik_connector.py`
- Dashboard: `http://localhost:5173`

---

## References

- Original completion doc: `.claude/plans/causal-discovery-integration-complete.md`
- causal-learn FCI: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html
- causal-learn LiNGAM: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Functional%20causal%20model-based%20methods/LiNGAM.html
- Opik docs: https://www.comet.com/docs/opik/
