# Tier 1 Orchestration Agents - Non-Breaking Improvement Plan

**Created**: 2026-01-24
**Completed**: 2026-01-24
**Status**: Complete
**Priority**: Low-Medium

---

## Executive Summary

This plan addresses non-breaking improvements identified during the evaluation of Tier 1 Orchestration agents (Orchestrator and Tool Composer). All improvements are additive and do not modify existing behavior.

---

## Improvement Items

### 1. Add Tool Composer Contract Documentation

**Priority**: Low
**Effort**: 2 hours
**Impact**: Documentation completeness

**Current State**:
- `orchestrator-contracts.md` exists and is comprehensive
- Tool Composer lacks formal contract documentation in `.claude/contracts/`
- Contract validation is only in `src/agents/tool_composer/CONTRACT_VALIDATION.md`

**Target State**:
- Create `.claude/contracts/tier1-tool-composer-contracts.md`
- Include: 4-phase pipeline contracts, CompositionResult, Tool Registry interface
- Follow same format as `orchestrator-contracts.md`

**Files to Create**:
- `.claude/contracts/tier1-tool-composer-contracts.md`

---

### 2. Centralize Timeout/Retry Configurations in YAML

**Priority**: Low
**Effort**: 3 hours
**Impact**: Configuration maintainability

**Current State**:
- Timeouts scattered across code:
  - `agent.py`: `sla_seconds = 2` (Orchestrator), `sla_seconds = 180` (Tool Composer)
  - `router.py`: Per-agent timeouts (20s-60s)
  - `executor.py`: `ExponentialBackoff` defaults (1s base, 30s max)
  - `composer.py`: Phase-specific configs

**Target State**:
- Add `tier1_orchestration` section to `config/agent_config.yaml`
- Centralize: orchestrator timeouts, tool composer phases, retry policies

**Files to Modify**:
- `config/agent_config.yaml` - Add new section

---

### 3. Add Integration Tests for CognitiveRAG Routing

**Priority**: Medium
**Effort**: 4 hours
**Impact**: Test coverage

**Current State**:
- Specialist doc describes CognitiveRAG integration (`orchestrator-agent.md:961-1159`)
- No dedicated integration tests for CognitiveRAG → Orchestrator flow
- Tests exist for individual nodes but not the integration path

**Target State**:
- Create `tests/integration/test_cognitive_rag_orchestrator_routing.py`
- Test: routing acceptance, low-confidence override, training signal emission

**Files to Create**:
- `tests/integration/test_cognitive_rag_orchestrator_routing.py`

---

### 4. Add Circuit Breaker Metrics to Opik Tracing

**Priority**: Low
**Effort**: 2 hours
**Impact**: Observability completeness

**Current State**:
- Circuit breaker implemented in `executor.py:85-100`
- `opik_tracer.py` traces phases but not circuit breaker events
- No visibility into circuit state transitions or trip counts

**Target State**:
- Add circuit breaker span events to `ToolComposerOpikTracer`
- Track: state transitions, trip counts, recovery attempts

**Files to Modify**:
- `src/agents/tool_composer/opik_tracer.py`
- `src/agents/tool_composer/executor.py` (add tracer hooks)

---

## Implementation Order

1. **Tool Composer Contract** (documentation first)
2. **Centralize Configurations** (foundation for future changes)
3. **Circuit Breaker Opik Metrics** (code change)
4. **CognitiveRAG Integration Tests** (validation)

---

## Success Criteria

| Item | Success Criteria |
|------|------------------|
| Tool Composer Contract | File exists with 4-phase contracts documented |
| Centralize Configs | All timeout/retry values in YAML, code reads from config |
| Circuit Breaker Opik | Span events visible in Opik dashboard |
| CognitiveRAG Tests | 5+ integration tests passing |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Config migration breaks runtime | Low | Medium | Keep defaults in code as fallback |
| Opik integration adds latency | Low | Low | Make tracing optional via config |
| Tests require CognitiveRAG running | Medium | Low | Use mocks for unit portion |

---

## Verification

After implementation:
1. Run existing Tier 1 tests: `pytest tests/unit/test_agents/test_orchestrator/ tests/unit/test_agents/test_tool_composer/ -v`
2. Run new integration tests: `pytest tests/integration/test_cognitive_rag_orchestrator_routing.py -v`
3. Validate configs: `python -c "from src.agents.orchestrator.config import load_tier1_config; print(load_tier1_config())"`

---

## Appendix: File Inventory

### Files to Create
1. `.claude/contracts/tier1-tool-composer-contracts.md`
2. `tests/integration/test_cognitive_rag_orchestrator_routing.py`

### Files to Modify
1. `config/agent_config.yaml`
2. `src/agents/tool_composer/opik_tracer.py`
3. `src/agents/tool_composer/executor.py`

### No Changes Required
- `src/agents/orchestrator/agent.py` (already well-structured)
- `src/agents/tool_composer/agent.py` (already well-structured)
- Existing tests (all pass, no modifications)
