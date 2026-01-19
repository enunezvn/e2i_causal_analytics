# Agent Tier Reclassification Plan: health_score & resource_optimizer

## Executive Summary

**Issue**: `health_score` and `resource_optimizer` agents were incorrectly placed in the `legacy` section of `domain_vocabulary.yaml`. This contradicts the official E2I MLOps Implementation Plan v1.1 which defines:

| Agent | Correct Tier | Current Location | Status |
|-------|--------------|------------------|--------|
| `health_score` | **Tier 3 (Monitoring)** | legacy | WRONG |
| `resource_optimizer` | **Tier 4 (ML & Predictions)** | legacy | WRONG |

**Impact**: This misclassification affects:
- Ontology vocabulary registry (source of truth)
- Database ENUM definitions (migration 029)
- Integration test fixtures and expectations
- Potential routing/classification inconsistencies

**Good News**: Both agents are **fully implemented and functional**:
- `health_score`: 5 LangGraph nodes, 76 tests passing, orchestrator-integrated
- `resource_optimizer`: 4 LangGraph nodes, 62 tests passing, orchestrator-integrated

---

## Official Classification Reference

From `docs/e2i_mlops_implementation_plan_v1.1.html`:

### Tier 3: Monitoring & Quality (3 agents)
| Agent | Type | SLA |
|-------|------|-----|
| drift_monitor | Standard | <60s |
| experiment_designer | Hybrid | <120s |
| **health_score** | **Standard** | **<60s** |

### Tier 4: ML & Predictions (2 agents)
| Agent | Type | SLA |
|-------|------|-----|
| prediction_synthesizer | Hybrid | <30s |
| **resource_optimizer** | **Standard** | **<20s** |

---

## Files Requiring Changes

### 1. Source of Truth
- [ ] `config/domain_vocabulary.yaml` - Move agents to correct tiers

### 2. Database Layer
- [ ] `database/core/030_fix_agent_tier_classification.sql` - New migration to correct ENUMs

### 3. Test Fixtures
- [ ] `tests/integration/test_ontology/conftest.py` - Update `db_enum_values` fixture

### 4. Test Expectations
- [ ] `tests/integration/test_ontology/test_vocabulary_enum_sync.py` - Update tier tests

### 5. Documentation (Verification)
- [ ] Verify `CLAUDE.md` agent tier reference table
- [ ] Verify `.claude/specialists/` agent documentation

---

## Implementation Phases

### Phase 1: Fix domain_vocabulary.yaml (Source of Truth)
**Estimated tokens**: ~500
**Risk**: Low (YAML config change)

#### Tasks:
- [ ] 1.1 Read current `config/domain_vocabulary.yaml` agents section
- [ ] 1.2 Move `health_score` from `legacy` to `tier_3_monitoring`
- [ ] 1.3 Move `resource_optimizer` from `legacy` to `tier_4_prediction`
- [ ] 1.4 Remove empty `legacy` section if no other agents remain
- [ ] 1.5 Verify YAML syntax is valid

#### Expected Result:
```yaml
agents:
  tier_3_monitoring:
    - drift_monitor
    - data_quality_monitor
    - health_score          # ADDED
  tier_4_prediction:
    - prediction_synthesizer
    - risk_assessor
    - resource_optimizer    # ADDED
  # legacy section removed or empty
```

---

### Phase 2: Create Database Migration
**Estimated tokens**: ~800
**Risk**: Medium (database schema change)

#### Tasks:
- [ ] 2.1 Review current migration 029 ENUM definitions
- [ ] 2.2 Create migration 030 to update `agent_tier_mapping` table
- [ ] 2.3 Migration should:
  - Update health_score tier from 'legacy' to 'tier_3'
  - Update resource_optimizer tier from 'legacy' to 'tier_4'
  - Use UPDATE statements (not DROP/CREATE)
- [ ] 2.4 Test migration locally (if possible) or document for droplet

#### Migration Template:
```sql
-- Migration: 030_fix_agent_tier_classification.sql
-- Purpose: Correct tier classification for health_score and resource_optimizer

BEGIN;

-- Update health_score to Tier 3 (Monitoring)
UPDATE agent_registry
SET tier = 'tier_3'
WHERE agent_name = 'health_score';

-- Update resource_optimizer to Tier 4 (ML & Predictions)
UPDATE agent_registry
SET tier = 'tier_4'
WHERE agent_name = 'resource_optimizer';

-- Verify changes
SELECT agent_name, tier FROM agent_registry
WHERE agent_name IN ('health_score', 'resource_optimizer');

COMMIT;
```

---

### Phase 3: Update Test Fixtures
**Estimated tokens**: ~400
**Risk**: Low (test configuration)

#### Tasks:
- [ ] 3.1 Update `tests/integration/test_ontology/conftest.py`
- [ ] 3.2 Fix `db_enum_values` fixture to reflect correct tier assignments:
  - Move health_score to tier_3 agent list
  - Move resource_optimizer to tier_4 agent list
  - Remove from tier_5 list (if present)
- [ ] 3.3 Verify fixture syntax

#### Expected Change:
```python
@pytest.fixture
def db_enum_values():
    return {
        "agent_tier": ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5"],
        "agent_name": [
            # Tier 0-2 unchanged...
            # Tier 3: Monitoring (FIXED)
            "experiment_designer", "drift_monitor", "data_quality_monitor", "health_score",
            # Tier 4: Prediction (FIXED)
            "prediction_synthesizer", "risk_assessor", "resource_optimizer",
            # Tier 5: Self-Improvement
            "explainer", "feedback_learner"
        ],
        # ...
    }
```

---

### Phase 4: Update Test Expectations
**Estimated tokens**: ~600
**Risk**: Low (test assertions)

#### Tasks:
- [ ] 4.1 Update `test_vocabulary_has_all_tier_3_agents` to include health_score
- [ ] 4.2 Update `test_vocabulary_has_all_tier_4_agents` to include resource_optimizer
- [ ] 4.3 If tier_5 test exists, ensure it only expects explainer + feedback_learner
- [ ] 4.4 Add explicit tests for health_score and resource_optimizer tier membership

#### New Test Cases:
```python
def test_health_score_in_tier_3(self, vocabulary_registry):
    """Test that health_score is classified in Tier 3 (Monitoring)."""
    agents = vocabulary_registry.get_agents()
    tier_3_agents = agents.get("tier_3_monitoring", [])
    assert "health_score" in tier_3_agents, \
        "health_score should be in tier_3_monitoring, not legacy"

def test_resource_optimizer_in_tier_4(self, vocabulary_registry):
    """Test that resource_optimizer is classified in Tier 4 (Prediction)."""
    agents = vocabulary_registry.get_agents()
    tier_4_agents = agents.get("tier_4_prediction", [])
    assert "resource_optimizer" in tier_4_agents, \
        "resource_optimizer should be in tier_4_prediction, not legacy"
```

---

### Phase 5: Run Tests on Droplet (Small Batches)
**Estimated tokens**: ~300 per batch
**Risk**: Low (verification only)

#### Batch 1: Vocabulary Tests
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/unit/test_ontology/test_vocabulary_registry.py -v -x"
```

#### Batch 2: Integration Tests
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/integration/test_ontology/test_vocabulary_enum_sync.py -v -x"
```

#### Batch 3: Agent-Specific Tests
```bash
# health_score tests
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/unit/test_agents/test_health_score/ -v -n 2"

# resource_optimizer tests
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/unit/test_agents/test_resource_optimizer/ -v -n 2"
```

#### Batch 4: Full Ontology Test Suite
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/unit/test_ontology/ tests/integration/test_ontology/ -v -n 4"
```

---

### Phase 6: Verify Orchestrator Routing
**Estimated tokens**: ~400
**Risk**: Low (read-only verification)

#### Tasks:
- [ ] 6.1 Verify health_score routing in `src/agents/orchestrator/nodes/router.py`
- [ ] 6.2 Verify resource_optimizer routing
- [ ] 6.3 Confirm intent mappings are correct:
  - health_score: intent=system_health, priority=critical
  - resource_optimizer: intent=resource_allocation, priority=critical
- [ ] 6.4 Run orchestrator routing tests

#### Verification Command:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/pytest \
   tests/unit/test_agents/test_orchestrator/ -v -n 2 -k 'route'"
```

---

### Phase 7: Documentation Verification
**Estimated tokens**: ~200
**Risk**: None (documentation)

#### Tasks:
- [ ] 7.1 Verify CLAUDE.md Agent Tier Reference table is correct
- [ ] 7.2 Check `.claude/specialists/` for agent tier references
- [ ] 7.3 Update any documentation that mentions legacy classification

---

## Progress Tracking

### Current Status: NOT STARTED

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Fix vocabulary YAML | [ ] Pending | |
| Phase 2: Create DB migration | [ ] Pending | |
| Phase 3: Update test fixtures | [ ] Pending | |
| Phase 4: Update test expectations | [ ] Pending | |
| Phase 5: Run droplet tests | [ ] Pending | |
| Phase 6: Verify orchestrator | [ ] Pending | |
| Phase 7: Documentation | [ ] Pending | |

---

## Rollback Plan

If issues are discovered:
1. Revert `domain_vocabulary.yaml` changes
2. Do NOT apply migration 030 to production
3. Revert test fixture/expectation changes
4. All changes are isolated and reversible

---

## Success Criteria

- [ ] `health_score` appears in `tier_3_monitoring` in vocabulary
- [ ] `resource_optimizer` appears in `tier_4_prediction` in vocabulary
- [ ] All ontology unit tests pass (321 tests)
- [ ] All ontology integration tests pass (94 tests)
- [ ] health_score agent tests pass (76 tests)
- [ ] resource_optimizer agent tests pass (62 tests)
- [ ] Orchestrator routing tests pass
- [ ] No agents in `legacy` section (unless intentionally deprecated)

---

## Notes

- Both agents are **fully functional** - this is only a classification fix
- The orchestrator already routes to these agents correctly
- Database migration is optional if only using vocabulary as source of truth
- Always sync code to droplet before running tests

---

**Created**: 2026-01-19
**Author**: Claude Code
**Related Issue**: Agent tier misclassification discovered during ontology testing
