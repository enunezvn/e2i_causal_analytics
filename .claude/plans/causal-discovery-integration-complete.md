# Causal Discovery Integration Plan

**Status**: COMPLETED
**Date Completed**: 2025-12-30
**Plan File**: streamed-gliding-unicorn.md (original)

---

## Executive Summary

Successfully integrated causal discovery capabilities into E2I Causal Analytics using the `causal-learn` library via the existing Tool Registry pattern (NOT MCP as originally proposed in PyCausalSim).

**Key Decisions Implemented**:
1. ✅ Added Causal Discovery - GES and PC algorithms with ensemble voting
2. ✅ Added Driver Ranking - Causal vs predictive importance comparison
3. ✅ Used Tool Registry pattern (NOT MCP)
4. ✅ Used causal-learn library (NOT PyCausalSim - too new/risky)
5. ✅ Preserved existing validation/uplift systems (no replacement)

---

## Implementation Status

| Task | Status | Details |
|------|--------|---------|
| Add causal-learn dependency | ✅ Complete | Added to pyproject.toml |
| Create discovery module structure | ✅ Complete | `src/causal_engine/discovery/` |
| Implement algorithm wrappers (GES, PC) | ✅ Complete | `algorithms/ges_wrapper.py`, `pc_wrapper.py` |
| Implement DiscoveryRunner | ✅ Complete | Multi-algorithm ensemble orchestration |
| Create database migration | ✅ Complete | `026_causal_discovery_tables.sql` |
| Implement DiscoveryGate | ✅ Complete | ACCEPT/REVIEW/REJECT/AUGMENT decisions |
| Implement DriverRanker | ✅ Complete | Causal vs predictive comparison |
| Integrate with GraphBuilderNode | ✅ Complete | `auto_discover` option in state |
| Register tools in Tool Registry | ✅ Complete | `discover_dag`, `rank_drivers` tools |
| Unit tests | ✅ Complete | 77 tests passing |
| Integration tests | ✅ Complete | 21 tests passing |

**Total Tests**: 98 passing

---

## Module Architecture

```
src/causal_engine/discovery/
├── __init__.py                    # Exports DiscoveryRunner, DriverRanker, Gate
├── base.py                        # Enums, dataclasses, protocols
│   ├── DiscoveryAlgorithmType     # GES, PC, FCI, LINGAM enums
│   ├── GateDecision               # ACCEPT, REVIEW, REJECT, AUGMENT
│   ├── EdgeType                   # DIRECTED, UNDIRECTED, BIDIRECTED
│   ├── DiscoveredEdge             # Edge with confidence, votes
│   ├── DiscoveryConfig            # Algorithm configuration
│   ├── AlgorithmResult            # Per-algorithm output
│   └── DiscoveryResult            # Ensemble result with DAG
├── runner.py                      # DiscoveryRunner class
│   ├── discover_dag()             # Async ensemble discovery
│   ├── discover_dag_sync()        # Sync wrapper
│   ├── _run_algorithms()          # Parallel algorithm execution
│   ├── _build_ensemble()          # Edge voting and confidence
│   └── _remove_cycles()           # DAG enforcement
├── gate.py                        # DiscoveryGate class
│   ├── GateConfig                 # Threshold configuration
│   ├── GateEvaluation             # Decision with reasons
│   ├── evaluate()                 # Main gate logic
│   ├── should_accept()            # Quick accept check
│   └── get_augmentation_edges()   # High-confidence edges for manual DAG
├── driver_ranker.py               # DriverRanker class
│   ├── ImportanceType             # CAUSAL, PREDICTIVE, COMBINED
│   ├── FeatureRanking             # Per-feature ranking data
│   ├── DriverRankingResult        # Full ranking output
│   ├── rank_drivers()             # Main ranking logic
│   └── rank_from_discovery_result() # Convenience wrapper
└── algorithms/
    ├── __init__.py
    ├── base.py                    # DiscoveryAlgorithm protocol
    ├── ges_wrapper.py             # GES (Greedy Equivalence Search)
    └── pc_wrapper.py              # PC (Peter-Clark)
```

---

## Database Schema

File: `database/ml/026_causal_discovery_tables.sql`

```sql
-- Discovered DAG structures
CREATE TABLE IF NOT EXISTS ml.discovered_dags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES ml.user_sessions(id),
    algorithms JSONB NOT NULL,           -- ["ges", "pc"]
    edge_list JSONB NOT NULL,            -- [{source, target, confidence, votes}]
    ensemble_threshold FLOAT,
    gate_decision gate_decision_type,    -- ACCEPT, REVIEW, REJECT, AUGMENT
    gate_confidence FLOAT,
    gate_reasons JSONB,
    n_nodes INTEGER,
    n_edges INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Driver rankings (causal vs predictive)
CREATE TABLE IF NOT EXISTS ml.driver_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID REFERENCES ml.discovered_dags(id),
    target_variable VARCHAR(255) NOT NULL,
    rankings JSONB NOT NULL,             -- [{feature, causal_rank, predictive_rank, ...}]
    rank_correlation FLOAT,
    causal_only_features JSONB,
    predictive_only_features JSONB,
    concordant_features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Algorithm execution logs
CREATE TABLE IF NOT EXISTS ml.discovery_algorithm_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID REFERENCES ml.discovered_dags(id),
    algorithm discovery_algorithm_type NOT NULL,
    runtime_seconds FLOAT,
    converged BOOLEAN DEFAULT TRUE,
    score FLOAT,
    n_edges_found INTEGER,
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Integration Points

### 1. GraphBuilderNode (`src/agents/causal_impact/nodes/graph_builder.py`)

Added `auto_discover` option to state:

```python
# State with auto-discovery enabled
state = {
    "query": "What is the effect of HCP engagement on conversion?",
    "treatment_var": "hcp_engagement_level",
    "outcome_var": "patient_conversion_rate",
    "auto_discover": True,                    # Enable discovery
    "data_cache": {"data": dataframe},        # Required for discovery
    "discovery_algorithms": ["ges", "pc"],    # Optional, defaults to both
    "discovery_ensemble_threshold": 0.5,      # Optional
}

result = await graph_builder_node.execute(state)

# Result includes:
# - causal_graph: DAG with discovery_enabled, discovery_gate_decision
# - discovery_result: Full DiscoveryResult
# - discovery_gate_evaluation: GateEvaluation with confidence, reasons
# - discovery_latency_ms: Performance metric
```

### 2. Tool Registry

Registered tools:
- `discover_dag` - Run structure learning with ensemble
- `rank_drivers` - Compare causal vs predictive importance

### 3. Observability

Integrated with existing Opik tracing via `ml_observability_spans` table.

---

## Test Coverage

### Unit Tests (77 tests)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_base.py` | 18 | Enums, dataclasses, serialization |
| `test_runner.py` | 25 | DiscoveryRunner, ensemble, cycles |
| `test_gate.py` | 18 | GateConfig, GateEvaluation, decisions |
| `test_driver_ranker.py` | 16 | Rankings, correlation, categorization |

### Integration Tests (21 tests)

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestGraphBuilderNodeBasic | 4 | Basic execution without discovery |
| TestGraphBuilderNodeDiscoveryIntegration | 5 | Discovery with real algorithms |
| TestGraphBuilderGateDecisions | 4 | ACCEPT, AUGMENT, REVIEW, REJECT |
| TestGraphBuilderDiscoveryErrorHandling | 2 | Exceptions, empty data |
| TestBuildCausalGraphFunction | 1 | Standalone function |
| TestVariableInference | 3 | Query-based inference |
| TestKnownCausalRelationships | 2 | Domain knowledge edges |

---

## Gate Decision Logic

| Decision | Confidence Threshold | Behavior |
|----------|---------------------|----------|
| **ACCEPT** | >= 0.8 | Use discovered DAG directly |
| **AUGMENT** | 0.5 - 0.8 | Add high-confidence edges to manual DAG |
| **REVIEW** | 0.3 - 0.5 | Flag for expert review, use manual DAG |
| **REJECT** | < 0.3 | Discovery failed, use manual DAG only |

Factors considered:
- Average edge confidence
- Algorithm agreement (edge voting)
- Structure validity (DAG, connected)
- Edge count relative to node count

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `src/causal_engine/discovery/__init__.py` | Module exports |
| `src/causal_engine/discovery/base.py` | Types and dataclasses |
| `src/causal_engine/discovery/runner.py` | DiscoveryRunner |
| `src/causal_engine/discovery/gate.py` | DiscoveryGate |
| `src/causal_engine/discovery/driver_ranker.py` | DriverRanker |
| `src/causal_engine/discovery/algorithms/__init__.py` | Algorithm exports |
| `src/causal_engine/discovery/algorithms/base.py` | Protocol |
| `src/causal_engine/discovery/algorithms/ges_wrapper.py` | GES |
| `src/causal_engine/discovery/algorithms/pc_wrapper.py` | PC |
| `database/ml/026_causal_discovery_tables.sql` | Schema |
| `tests/unit/test_causal_engine/test_discovery/test_base.py` | Unit tests |
| `tests/unit/test_causal_engine/test_discovery/test_runner.py` | Unit tests |
| `tests/unit/test_causal_engine/test_discovery/test_gate.py` | Unit tests |
| `tests/unit/test_causal_engine/test_discovery/test_driver_ranker.py` | Unit tests |
| `tests/integration/test_graph_builder_discovery.py` | Integration tests |

### Modified Files

| File | Changes |
|------|---------|
| `pyproject.toml` | Added causal-learn dependency |
| `src/agents/causal_impact/nodes/graph_builder.py` | Added auto_discover integration |
| `src/tool_registry/registry.py` | Registered discovery tools |

---

## What Was NOT Implemented (By Design)

Per the original plan evaluation:

1. **MCP Server** - Used Tool Registry instead (simpler, existing pattern)
2. **PyCausalSim dependency** - Used causal-learn (stable, PyPI package)
3. **causal_validate tool** - Redundant with existing RefutationRunner
4. **causal_uplift_segment tool** - Redundant with existing uplift module
5. **causal_simulate_intervention tool** - Mostly covered by EstimatorSelector
6. **mcp_tool_calls table** - Used existing observability infrastructure

---

## Usage Example

```python
from src.causal_engine.discovery import (
    DiscoveryRunner,
    DiscoveryConfig,
    DiscoveryAlgorithmType,
    DiscoveryGate,
    DriverRanker,
)

# Run discovery
runner = DiscoveryRunner()
config = DiscoveryConfig(
    algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
    ensemble_threshold=0.5,
)
result = await runner.discover_dag(data, config)

# Evaluate with gate
gate = DiscoveryGate()
evaluation = gate.evaluate(result)
print(f"Decision: {evaluation.decision}")  # ACCEPT, REVIEW, REJECT, AUGMENT

# Rank drivers (if SHAP values available)
ranker = DriverRanker()
ranking = ranker.rank_from_discovery_result(result, target="outcome", shap_values=shap)
for r in ranking.get_by_causal_rank(top_k=5):
    print(f"{r.feature_name}: causal={r.causal_rank}, predictive={r.predictive_rank}")
```

---

## Next Steps (Optional Enhancements)

1. **Add FCI algorithm** - For latent confounders
2. **Add LiNGAM variants** - For non-Gaussian data
3. **Add discovery caching** - Avoid re-running on same data
4. **Add Opik dashboard** - Discovery-specific metrics
5. **Add CLI command** - `python -m src.causal_engine.discovery --data file.csv`

---

## References

- Original Plan: `.claude/plans/streamed-gliding-unicorn.md`
- causal-learn Docs: https://causal-learn.readthedocs.io/
- E2I Tool Registry: `src/tool_registry/registry.py`
- GraphBuilderNode: `src/agents/causal_impact/nodes/graph_builder.py`
