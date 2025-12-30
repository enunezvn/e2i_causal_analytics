# Audit Chain Implementation Plan

**Project**: E2I Causal Analytics - Audit Chain Integration
**Created**: 2025-12-30
**Status**: Ready for Implementation

---

## Executive Summary

The Audit Chain design document (`docs/E2I_Audit_Chain_Design.html`) specifies a tamper-evident logging system using SHA-256 hash chains. **The foundation is 100% complete** (SQL schema + Python service), but **agent integration is 0% complete**.

### Current State
| Component | Status | Location |
|-----------|--------|----------|
| SQL Schema | ✅ Complete | `database/audit/011_audit_chain_tables.sql` |
| Python Service | ✅ Complete | `src/utils/audit_chain.py` |
| API Endpoints | ❌ Missing | `src/api/routes/audit.py` |
| Agent Integration | ❌ Missing | All 18 agents (Tiers 0-5) |
| Frontend Badges | ⏸️ Skipped | User scope: Backend + API only |

---

## Phase 0: Foundation & Utilities
**Goal**: Create reusable mixin and decorators for agent integration
**Estimated Tests**: 5 unit tests

### 0.1 Create Audit Chain Mixin
**File**: `src/agents/base/audit_chain_mixin.py` (NEW)

```python
# Creates AuditChainMixin class with:
# - start_audit_workflow() -> UUID
# - record_node_entry(node_name, input_hash, output_hash, ...)
# - complete_audit_workflow(status)
# - get_workflow_chain() -> List[AuditChainEntry]
```

**Tasks**:
- [ ] Create `src/agents/base/` directory if not exists
- [ ] Implement `AuditChainMixin` class
- [ ] Add `audited_traced_node` decorator (wraps existing `traced_node`)
- [ ] Export from `src/agents/base/__init__.py`

### 0.2 State Extension Protocol
**Pattern**: Add `audit_workflow_id: NotRequired[UUID]` to each agent's state

**Files to modify** (state.py in each agent):
- [ ] `src/agents/orchestrator/state.py`
- [ ] `src/agents/tool_composer/state.py`
- [ ] `src/agents/causal_impact/state.py`
- [ ] `src/agents/gap_analyzer/state.py`
- [ ] `src/agents/heterogeneous_optimizer/state.py`
- [ ] `src/agents/drift_monitor/state.py`
- [ ] `src/agents/experiment_designer/state.py`
- [ ] `src/agents/health_score/state.py`
- [ ] `src/agents/prediction_synthesizer/state.py`
- [ ] `src/agents/resource_optimizer/state.py`
- [ ] `src/agents/explainer/state.py`
- [ ] `src/agents/feedback_learner/state.py`
- [ ] Tier 0 agents (7 files in `src/agents/tier_0/`)

### Phase 0 Tests
**File**: `tests/unit/test_agents/test_base/test_audit_chain_mixin.py`
- [ ] `test_start_audit_workflow_creates_genesis`
- [ ] `test_record_node_entry_chains_hash`
- [ ] `test_complete_audit_workflow_sets_status`
- [ ] `test_audited_traced_node_decorator`
- [ ] `test_mixin_integration_with_state`

---

## Phase 1: API Endpoints
**Goal**: REST endpoints for audit chain verification
**Estimated Tests**: 6 unit tests

### 1.1 Create Audit Routes
**File**: `src/api/routes/audit.py` (NEW)

**Endpoints**:
```python
GET  /api/v1/audit/workflow/{workflow_id}  # Get workflow chain
GET  /api/v1/audit/workflow/{workflow_id}/verify  # Verify chain integrity
GET  /api/v1/audit/agent/{agent_name}/recent  # Recent agent audits
POST /api/v1/audit/verify-all  # Verify all chains (admin)
GET  /api/v1/audit/stats  # Daily audit statistics
```

**Tasks**:
- [ ] Create `src/api/routes/audit.py`
- [ ] Add Pydantic schemas for responses
- [ ] Register routes in `src/api/main.py`
- [ ] Add authentication/authorization decorators

### Phase 1 Tests
**File**: `tests/unit/test_api/test_routes/test_audit.py`
- [ ] `test_get_workflow_chain_success`
- [ ] `test_get_workflow_chain_not_found`
- [ ] `test_verify_chain_integrity_valid`
- [ ] `test_verify_chain_integrity_tampered`
- [ ] `test_get_recent_agent_audits`
- [ ] `test_verify_all_requires_admin`

---

## Phase 2a: Tier 2 Agent Integration (Causal Analytics)
**Goal**: Integrate audit chain into Causal Impact, Gap Analyzer, Heterogeneous Optimizer
**Estimated Tests**: 9 unit tests (3 per agent)

### 2a.1 Causal Impact Agent
**Files**:
- `src/agents/causal_impact/state.py` - Add `audit_workflow_id`
- `src/agents/causal_impact/graph.py` - Wrap nodes with audit decorator

**Nodes to wrap**:
- [ ] `graph_builder_node`
- [ ] `estimation_node`
- [ ] `refutation_node`
- [ ] `sensitivity_node`
- [ ] `interpretation_node`

**Special**: Record `RefutationResults` in audit entries (placebo, random_common_cause, data_subset, unobserved_common_cause)

### 2a.2 Gap Analyzer Agent
**Files**:
- `src/agents/gap_analyzer/state.py` - Add `audit_workflow_id`
- `src/agents/gap_analyzer/graph.py` - Wrap nodes

**Nodes to wrap**:
- [ ] `data_loading_node`
- [ ] `gap_detection_node`
- [ ] `roi_calculation_node`
- [ ] `recommendation_node`

### 2a.3 Heterogeneous Optimizer Agent
**Files**:
- `src/agents/heterogeneous_optimizer/state.py` - Add `audit_workflow_id`
- `src/agents/heterogeneous_optimizer/graph.py` - Wrap nodes

**Nodes to wrap**:
- [ ] `segmentation_node`
- [ ] `cate_estimation_node`
- [ ] `optimization_node`

### Phase 2a Tests
**File**: `tests/unit/test_agents/test_causal_impact/test_audit_integration.py`
- [ ] `test_causal_impact_creates_audit_chain`
- [ ] `test_causal_impact_records_refutation_results`
- [ ] `test_causal_impact_chain_verifiable`

**File**: `tests/unit/test_agents/test_gap_analyzer/test_audit_integration.py`
- [ ] `test_gap_analyzer_creates_audit_chain`
- [ ] `test_gap_analyzer_records_roi_calculations`
- [ ] `test_gap_analyzer_chain_verifiable`

**File**: `tests/unit/test_agents/test_heterogeneous_optimizer/test_audit_integration.py`
- [ ] `test_hetero_opt_creates_audit_chain`
- [ ] `test_hetero_opt_records_segments`
- [ ] `test_hetero_opt_chain_verifiable`

---

## Phase 2b: Tier 1 Agent Integration (Coordination)
**Goal**: Integrate audit chain into Orchestrator and Tool Composer
**Estimated Tests**: 6 unit tests (3 per agent)

### 2b.1 Orchestrator Agent
**Files**:
- `src/agents/orchestrator/state.py` - Add `audit_workflow_id`
- `src/agents/orchestrator/graph.py` - Wrap nodes

**Critical**: Orchestrator creates parent workflow_id and passes to child agents

**Nodes to wrap**:
- [ ] `query_analysis_node`
- [ ] `agent_selection_node`
- [ ] `dispatch_node`
- [ ] `aggregation_node`

### 2b.2 Tool Composer Agent
**Files**:
- `src/agents/tool_composer/state.py` - Add `audit_workflow_id`
- `src/agents/tool_composer/graph.py` - Wrap nodes

**Nodes to wrap**:
- [ ] `decomposition_node`
- [ ] `tool_selection_node`
- [ ] `execution_node`
- [ ] `synthesis_node`

### Phase 2b Tests
**File**: `tests/unit/test_agents/test_orchestrator/test_audit_integration.py`
- [ ] `test_orchestrator_creates_parent_workflow`
- [ ] `test_orchestrator_passes_workflow_to_children`
- [ ] `test_orchestrator_chain_verifiable`

**File**: `tests/unit/test_agents/test_tool_composer/test_audit_integration.py`
- [ ] `test_tool_composer_creates_audit_chain`
- [ ] `test_tool_composer_records_tool_selections`
- [ ] `test_tool_composer_chain_verifiable`

---

## Phase 3a: Tier 3 Agent Integration (Monitoring)
**Goal**: Integrate audit chain into Drift Monitor, Experiment Designer, Health Score
**Estimated Tests**: 9 unit tests (3 per agent)

### 3a.1 Drift Monitor Agent
**Files**:
- `src/agents/drift_monitor/state.py`
- `src/agents/drift_monitor/graph.py`

### 3a.2 Experiment Designer Agent
**Files**:
- `src/agents/experiment_designer/state.py`
- `src/agents/experiment_designer/graph.py`

### 3a.3 Health Score Agent
**Files**:
- `src/agents/health_score/state.py`
- `src/agents/health_score/graph.py`

### Phase 3a Tests
**Files**: `tests/unit/test_agents/test_*/test_audit_integration.py` (3 files)

---

## Phase 3b: Tier 4 Agent Integration (ML Predictions)
**Goal**: Integrate audit chain into Prediction Synthesizer, Resource Optimizer
**Estimated Tests**: 6 unit tests (3 per agent)

### 3b.1 Prediction Synthesizer Agent
**Files**:
- `src/agents/prediction_synthesizer/state.py`
- `src/agents/prediction_synthesizer/graph.py`

### 3b.2 Resource Optimizer Agent
**Files**:
- `src/agents/resource_optimizer/state.py`
- `src/agents/resource_optimizer/graph.py`

### Phase 3b Tests
**Files**: `tests/unit/test_agents/test_*/test_audit_integration.py` (2 files)

---

## Phase 4: Tier 5 Agent Integration (Self-Improvement)
**Goal**: Integrate audit chain into Explainer, Feedback Learner
**Estimated Tests**: 6 unit tests (3 per agent)

### 4.1 Explainer Agent
**Files**:
- `src/agents/explainer/state.py`
- `src/agents/explainer/graph.py`

### 4.2 Feedback Learner Agent
**Files**:
- `src/agents/feedback_learner/state.py`
- `src/agents/feedback_learner/graph.py`

### Phase 4 Tests
**Files**: `tests/unit/test_agents/test_*/test_audit_integration.py` (2 files)

---

## Phase 5: Tier 0 Agent Integration (ML Foundation)
**Goal**: Integrate audit chain into 7 ML Foundation agents
**Estimated Tests**: 21 unit tests (3 per agent)

### Agents (in order):
1. [ ] `scope_definer`
2. [ ] `data_preparer`
3. [ ] `feature_analyzer`
4. [ ] `model_selector`
5. [ ] `model_trainer`
6. [ ] `model_deployer`
7. [ ] `observability_connector`

**Files per agent**:
- `src/agents/tier_0/{agent_name}/state.py`
- `src/agents/tier_0/{agent_name}/graph.py`

### Phase 5 Tests
**Files**: `tests/unit/test_agents/test_tier_0/test_*/test_audit_integration.py` (7 files)

---

## Phase 6: Integration Testing Suite
**Goal**: End-to-end audit chain verification
**Estimated Tests**: 10 integration tests

### 6.1 Cross-Agent Chain Tests
**File**: `tests/integration/test_audit_chain/test_cross_agent.py`
- [ ] `test_orchestrator_to_child_chain_continuity`
- [ ] `test_multi_agent_workflow_full_chain`
- [ ] `test_parallel_agent_chains_independent`

### 6.2 Database Verification Tests
**File**: `tests/integration/test_audit_chain/test_database_verification.py`
- [ ] `test_verify_chain_integrity_sql_function`
- [ ] `test_verify_all_chains_function`
- [ ] `test_chain_tamper_detection`
- [ ] `test_genesis_block_validation`

### 6.3 API Integration Tests
**File**: `tests/integration/test_audit_chain/test_api.py`
- [ ] `test_api_workflow_retrieval`
- [ ] `test_api_verification_endpoint`
- [ ] `test_api_stats_aggregation`

---

## Phase 7: Documentation & Migration Verification
**Goal**: Ensure documentation and database are production-ready
**Estimated Tests**: 3 verification scripts

### 7.1 Database Migration Verification
**Tasks**:
- [ ] Verify `011_audit_chain_tables.sql` is applied to Supabase
- [ ] Test `compute_entry_hash` SQL function
- [ ] Test `verify_chain_integrity` SQL function
- [ ] Test `v_audit_chain_summary` view

### 7.2 Documentation Updates
**Files**:
- [ ] Update `README.md` with audit chain section
- [ ] Update `docs/ARCHITECTURE.md` (if exists)
- [ ] Create `docs/AUDIT_CHAIN_USAGE.md`

### 7.3 Contract Updates
**Files**:
- [ ] Update `.claude/contracts/base-structures.md`
- [ ] Update relevant tier contracts

---

## Test Execution Strategy

Due to low resources, run tests in small batches:

```bash
# Phase 0 tests only
pytest tests/unit/test_agents/test_base/test_audit_chain_mixin.py -v

# Phase 1 tests only
pytest tests/unit/test_api/test_routes/test_audit.py -v

# Phase 2a tests (one agent at a time)
pytest tests/unit/test_agents/test_causal_impact/test_audit_integration.py -v
pytest tests/unit/test_agents/test_gap_analyzer/test_audit_integration.py -v
pytest tests/unit/test_agents/test_heterogeneous_optimizer/test_audit_integration.py -v

# Continue pattern for remaining phases...
```

**Memory-safe settings** (from CLAUDE.md):
- Max 4 workers: `-n 4`
- Scope distribution: `--dist=loadscope`
- Never use `-n auto`

---

## Implementation Checklist

### Phase 0: Foundation
- [ ] `src/agents/base/audit_chain_mixin.py`
- [ ] `src/agents/base/__init__.py`
- [ ] Tests passing

### Phase 1: API
- [ ] `src/api/routes/audit.py`
- [ ] Routes registered in main.py
- [ ] Tests passing

### Phase 2a: Tier 2
- [ ] Causal Impact integrated
- [ ] Gap Analyzer integrated
- [ ] Heterogeneous Optimizer integrated
- [ ] Tests passing

### Phase 2b: Tier 1
- [ ] Orchestrator integrated
- [ ] Tool Composer integrated
- [ ] Tests passing

### Phase 3a: Tier 3
- [ ] Drift Monitor integrated
- [ ] Experiment Designer integrated
- [ ] Health Score integrated
- [ ] Tests passing

### Phase 3b: Tier 4
- [ ] Prediction Synthesizer integrated
- [ ] Resource Optimizer integrated
- [ ] Tests passing

### Phase 4: Tier 5
- [ ] Explainer integrated
- [ ] Feedback Learner integrated
- [ ] Tests passing

### Phase 5: Tier 0
- [ ] All 7 ML Foundation agents integrated
- [ ] Tests passing

### Phase 6: Integration
- [ ] Cross-agent tests passing
- [ ] Database verification passing
- [ ] API integration passing

### Phase 7: Documentation
- [ ] Database migration verified
- [ ] Documentation updated
- [ ] Contracts updated

---

## Key Files Reference

### Existing (Complete)
- `database/audit/011_audit_chain_tables.sql` - SQL schema
- `src/utils/audit_chain.py` - Python service (AuditChainService)

### To Create
- `src/agents/base/audit_chain_mixin.py` - Reusable mixin
- `src/api/routes/audit.py` - REST endpoints
- 18+ test files for agent integration

### To Modify
- 18 agent `state.py` files (add `audit_workflow_id`)
- 18 agent `graph.py` files (add audit decorators)
- `src/api/main.py` (register audit routes)

---

## Notes

1. **Existing Pattern**: Use `traced_node` decorator in `causal_impact/graph.py` as template
2. **Hash Computation**: Use SHA-256 via `src/utils/audit_chain.py:compute_hash()`
3. **Genesis Block**: First entry in workflow has `previous_hash=None`, uses "GENESIS" marker
4. **Parent-Child Chains**: Orchestrator's `workflow_id` becomes parent for child agents
