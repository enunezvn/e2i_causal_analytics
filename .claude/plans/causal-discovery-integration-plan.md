# Causal Discovery Integration Plan

**Status**: APPROVED FOR IMPLEMENTATION
**Date**: 2025-12-30
**Evaluator**: Claude Code

---

## Executive Summary

After critical evaluation of the PyCausalSim MCP proposal, we recommend **selective adoption** of causal discovery capabilities using established libraries (causal-learn) integrated via the existing Tool Registry pattern—NOT MCP.

**Key Decisions**:
1. ✅ **Add Causal Discovery** - Critical gap identified (GES, PC, FCI, LiNGAM algorithms)
2. ✅ **Add Driver Ranking** - Causal vs predictive importance comparison
3. ❌ **Do NOT use MCP** - Use existing Tool Registry pattern instead
4. ❌ **Do NOT depend on PyCausalSim** - Too new (5 weeks, 6 commits, single maintainer)
5. ❌ **Do NOT replace existing validation/uplift** - 90-95% overlap with RefutationRunner

---

## 1. Architectural Analysis

### 1.1 Current E2I Tool Architecture

| Aspect | E2I Current State | PyCausalSim Proposal |
|--------|-------------------|---------------------|
| **Tool Registration** | Custom Tool Registry (singleton, decorator-based) | MCP Server |
| **Tool Discovery** | Database-driven (`tool_registry` table) | MCP resource endpoint |
| **Tool Composition** | 4-phase Tool Composer (Decompose→Plan→Execute→Synthesize) | Direct tool calls |
| **Observability** | Opik integration with circuit breaker | Custom `mcp_tool_calls` table |
| **Transport** | In-process Python calls | stdio/HTTP (separate process) |

**Critical Issue**: E2I's Tool Composer is NOT MCP-based. Adding an MCP server creates a **parallel integration pathway** that bypasses:
- Existing tool dependency management
- Composition episode logging
- Performance tracking infrastructure
- Vector similarity search for plan reuse

### 1.2 MCP Usage in E2I

Current `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "supabase": { "url": "..." }
  }
}
```

MCP is used **only for Claude Code environment** (Supabase access), NOT for agent tool invocation. The proposal assumes MCP is the primary tool integration mechanism—this is incorrect.

---

## 2. Functional Overlap Analysis

### 2.1 Tool-by-Tool Comparison

| PyCausalSim Tool | E2I Equivalent | Overlap % | Notes |
|------------------|----------------|-----------|-------|
| `causal_validate` | `RefutationRunner` | **95%** | Same 4 refutation tests (placebo, random cause, subset, bootstrap) |
| `causal_uplift_segment` | `src/causal_engine/uplift/` | **90%** | CausalML integration already exists with AUUC, Qini metrics |
| `causal_simulate_intervention` | `EstimatorSelector` + agents | **70%** | ATE/CATE estimation exists; counterfactual framing is presentation |
| `causal_rank_drivers` | **Partial** | **40%** | Feature importance via SHAP exists, but not causal-vs-predictive ranking |
| `causal_discover_graph` | **MISSING** | **0%** | Structure learning (GES/PC/FCI/LiNGAM) NOT in current system |

### 2.2 Refutation Test Comparison

| Test | E2I RefutationRunner | PyCausalSim |
|------|---------------------|-------------|
| Placebo Treatment | ✅ DoWhy | ✅ |
| Random Common Cause | ✅ DoWhy | ✅ |
| Data Subset | ✅ DoWhy | ✅ |
| Bootstrap | ✅ DoWhy | ✅ |
| Sensitivity (E-value) | ✅ DoWhy | ❌ Not mentioned |

E2I has **MORE** validation capability (E-value sensitivity analysis).

### 2.3 Unique Value: Causal Discovery

The ONLY capability PyCausalSim adds that E2I lacks:

**Structure Learning Algorithms**:
- GES (Greedy Equivalence Search)
- PC (Peter-Clark)
- FCI (Fast Causal Inference)
- LiNGAM (Linear Non-Gaussian Acyclic Model)

This IS a gap in E2I. Currently, DAGs are manually constructed in `graph_builder` nodes.

---

## 3. Database Schema Conflicts

### 3.1 Proposed Tables vs. Existing

| PyCausalSim Table | E2I Equivalent | Conflict |
|-------------------|----------------|----------|
| `mcp_tool_calls` | `tool_performance` + `composition_steps` | Duplicate observability |
| `causal_graphs` | `causal_paths` (different structure) | Schema mismatch |
| `causal_validations` (new) | `causal_validations` (existing) | **NAME COLLISION** |
| `driver_rankings` | None | New (acceptable) |
| `intervention_simulations` | None | New (acceptable) |

**Critical**: The proposed `causal_validations` table has the **same name** as the existing table in `database/ml/010_causal_validation_tables.sql` but different schema.

### 3.2 Observability Duplication

E2I already tracks tool calls via:
- `tool_performance` table (per-call metrics)
- `composition_steps` table (multi-tool workflows)
- `ml_observability_spans` table (Opik persistence)
- Opik dashboard integration

Adding `mcp_tool_calls` creates a **third observability silo**.

---

## 4. Dependency Concerns

### 4.1 PyCausalSim Library

```
pycausalsim @ git+https://github.com/Bodhi8/pycausalsim.git
```

**Concerns**:
- GitHub dependency (not PyPI) - less stable
- Repository: 5 weeks old, 6 commits, single maintainer - **too risky**
- No version pinning in requirements
- Adds ~4 new transitive dependencies

### 4.2 Existing Causal Stack

E2I already depends on:
- `dowhy>=0.11.0` - Core causal inference
- `econml>=0.15.0` - ML-based causal methods
- `causalml>=0.13.0` - Uplift modeling
- `networkx>=3.0` - Graph algorithms

PyCausalSim may introduce **version conflicts** with these established dependencies.

---

## 5. Integration Complexity

### 5.1 MCP Server Deployment

PyCausalSim requires:
1. Separate Python process (HTTP on port 8000)
2. Docker container management
3. Health check integration
4. Load balancing for production

E2I tools are currently **in-process**, avoiding network latency and deployment complexity.

### 5.2 Agent Integration

Current agent pattern:
```python
# src/agents/causal_impact/nodes/refutation.py
from src.causal_engine.refutation_runner import RefutationRunner

runner = RefutationRunner(...)
result = await runner.run_all_tests(...)
```

With MCP:
```python
# Would require MCP client, HTTP calls, error handling
async with mcp_client.connect("pycausalsim://localhost:8000") as client:
    result = await client.call_tool("causal_validate", {...})
```

This adds **network latency**, **failure modes**, and **complexity** without clear benefit.

---

## 6. Recommendations

### 6.1 Do NOT Implement As Designed

The MCP server approach conflicts with E2I's established Tool Registry pattern and creates unnecessary complexity.

### 6.2 Selective Adoption Path

**Phase 1: Causal Discovery Only**
- Extract structure learning algorithms (GES, PC, FCI, LiNGAM)
- Integrate into `src/causal_engine/discovery/` module
- Register as Tool Registry tool, NOT MCP
- Add to `graph_builder` node as optional auto-discovery

**Phase 2: Driver Ranking (Optional)**
- Implement `causal_rank_drivers` logic in `src/causal_engine/`
- Compare with existing SHAP rankings
- Expose via Tool Registry

**Phase 3: Vocabulary Integration**
- Merge useful terms into `config/domain_vocabulary_v3.1.0.yaml`
- Document in `.claude/context/` files

### 6.3 What to Discard

- MCP server architecture
- `mcp_tool_calls` table (use existing observability)
- `causal_validate` tool (redundant with RefutationRunner)
- `causal_uplift_segment` tool (redundant with existing uplift module)
- `causal_simulate_intervention` tool (mostly covered by existing estimators)
- HTTP/stdio transport complexity

---

## 7. User Clarifications (Resolved)

Questions asked and answered:

| Question | User Response |
|----------|---------------|
| Causal Discovery Priority | **"Critical gap to fill"** - automatic DAG structure learning is important |
| MCP vs Tool Registry | **"No strong preference"** - follow existing patterns |
| PyCausalSim Library Status | Evaluated: 5 weeks old, 6 commits, single maintainer - **too risky** |
| Handle Overlap | **"Extract only unique features"** - don't replace existing systems |

---

## 8. Implementation Plan

### 8.1 Recommended Library: causal-learn

Use `causal-learn` from the py-why ecosystem (same org as DoWhy):
- PyPI package (stable)
- Active maintenance
- Supports GES, PC, FCI, LiNGAM algorithms
- No version conflicts with existing DoWhy/EconML stack

### 8.2 Module Architecture

```
src/causal_engine/discovery/
├── __init__.py                    # Exports DiscoveryRunner, DriverRanker
├── runner.py                      # DiscoveryRunner class
├── driver_ranker.py               # DriverRanker class
├── algorithms/
│   ├── __init__.py
│   ├── base.py                    # DiscoveryAlgorithm protocol
│   ├── ges_wrapper.py             # GES implementation
│   ├── pc_wrapper.py              # PC implementation
│   ├── fci_wrapper.py             # FCI implementation
│   └── lingam_wrapper.py          # LiNGAM implementation
└── gate.py                        # DiscoveryGate decision logic
```

### 8.3 Key Classes

**DiscoveryRunner** (similar to RefutationRunner):
```python
class DiscoveryRunner:
    """Structure learning with multi-algorithm ensemble"""

    async def discover_dag(
        self,
        data: pd.DataFrame,
        algorithms: list[str] = ["GES", "PC"],
        alpha: float = 0.05,
        ensemble_threshold: float = 0.6,
    ) -> DiscoveryResult:
        """Run structure learning, return DAG with confidence scores"""
```

**DriverRanker**:
```python
class DriverRanker:
    """Compare causal vs predictive feature importance"""

    def rank_drivers(
        self,
        discovered_dag: nx.DiGraph,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> DriverRankingResult:
        """Return ranked features with causal/predictive comparison"""
```

**DiscoveryGate**:
```python
class DiscoveryGate:
    """Gating decisions for discovered structures"""

    def evaluate(self, result: DiscoveryResult) -> GateDecision:
        # ACCEPT: High confidence, consistent across algorithms
        # REVIEW: Medium confidence, requires expert validation
        # REJECT: Low confidence, use manual DAG instead
        # AUGMENT: Supplement manual DAG with high-confidence edges
```

### 8.4 Database Migration

New file: `database/ml/025_causal_discovery_tables.sql`

```sql
-- Discovered DAG structures
CREATE TABLE IF NOT EXISTS ml.discovered_dags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES ml.user_sessions(id),
    algorithm VARCHAR(50) NOT NULL,
    edge_list JSONB NOT NULL,
    confidence_scores JSONB NOT NULL,
    parameters JSONB,
    gate_decision VARCHAR(20),  -- ACCEPT, REVIEW, REJECT, AUGMENT
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Driver rankings
CREATE TABLE IF NOT EXISTS ml.driver_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id UUID REFERENCES ml.discovered_dags(id),
    feature_name VARCHAR(255) NOT NULL,
    causal_rank INTEGER,
    predictive_rank INTEGER,
    causal_score FLOAT,
    predictive_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 8.5 Integration Points

1. **GraphBuilderNode** (`src/agents/causal_impact/nodes/graph_builder.py`):
   - Add optional `auto_discover=True` parameter
   - Fall back to manual DAG if discovery confidence is low

2. **Tool Registry**:
   - Register `discover_dag` as composable tool
   - Register `rank_drivers` as composable tool

3. **Observability**:
   - Integrate with existing Opik tracing
   - Log to `ml_observability_spans` table

### 8.6 Implementation Phases

| Phase | Deliverables |
|-------|--------------|
| **Phase 1** | Algorithm wrappers (GES, PC), DiscoveryRunner base |
| **Phase 2** | DiscoveryGate, integration with GraphBuilderNode |
| **Phase 3** | DriverRanker, SHAP comparison |
| **Phase 4** | Tool Registry integration, tests, documentation |

### 8.7 Test Strategy

- Unit tests for each algorithm wrapper
- Integration tests with synthetic DAGs
- Property-based tests for edge detection consistency
- Performance benchmarks for large graphs

---

## 9. Files to Modify/Create

### New Files
| File | Purpose |
|------|---------|
| `src/causal_engine/discovery/__init__.py` | Module exports |
| `src/causal_engine/discovery/runner.py` | DiscoveryRunner class |
| `src/causal_engine/discovery/driver_ranker.py` | DriverRanker class |
| `src/causal_engine/discovery/gate.py` | DiscoveryGate decision logic |
| `src/causal_engine/discovery/algorithms/*.py` | Algorithm wrappers |
| `database/ml/025_causal_discovery_tables.sql` | Schema migration |
| `tests/unit/test_causal_engine/test_discovery/` | Unit tests |

### Files to Modify
| File | Change |
|------|--------|
| `src/agents/causal_impact/nodes/graph_builder.py` | Add auto_discover option |
| `src/tool_registry/registry.py` | Register new tools |
| `pyproject.toml` | Add causal-learn dependency |
| `config/domain_vocabulary_v3.1.0.yaml` | Merge useful PyCausalSim terms |

---

## 10. Files Analyzed

| File | Purpose |
|------|---------|
| `PyCausalSim/pycausalsim_mcp.py` | MCP server implementation |
| `PyCausalSim/pycausalsim_schema.sql` | Database schema |
| `PyCausalSim/PyCausalSim README.md` | Overview documentation |
| `PyCausalSim/PyCausalSim Vocabulary Extension.yml` | Terminology definitions |
| `src/causal_engine/refutation_runner.py` | Existing validation (1,168 lines) |
| `src/causal_engine/validation_outcome_store.py` | Learning integration (934 lines) |
| `src/tool_registry/registry.py` | Existing tool registration |
| `database/ml/010_causal_validation_tables.sql` | Existing causal schema |
| `database/ml/013_tool_composer_tables.sql` | Tool composer schema |
| `.claude/mcp.json` | Current MCP configuration |
