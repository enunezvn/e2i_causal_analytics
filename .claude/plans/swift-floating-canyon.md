# CohortConstructor Tier 0 Agent Implementation Plan

**Agent**: CohortConstructor
**Tier**: 0 (ML Foundation)
**Position**: scope_definer → **cohort_constructor** → data_preparer
**Type**: Standard (tool-heavy, SLA-bound, no LLM)
**SLA**: <120 seconds for 100K patients

---

## Implementation Overview

This plan integrates the CohortConstructor agent into the E2I Causal Analytics Tier 0 pipeline. The implementation is divided into 8 small phases to maintain context-window efficiency and enable incremental testing on the droplet.

---

## Phase 1: Database Schema Migration
**Estimated Files**: 1 SQL file
**Testing**: Run migration on droplet, verify table creation

### Tasks
- [ ] 1.1 Create migration file `database/ml/012_cohort_constructor_tables.sql`
- [ ] 1.2 Define `ml_cohort_definitions` table (versioned cohort configs)
- [ ] 1.3 Define `ml_cohort_executions` table (execution metadata)
- [ ] 1.4 Define `ml_cohort_eligibility_log` table (step-by-step exclusions)
- [ ] 1.5 Define `ml_patient_cohort_assignments` table (patient eligibility)
- [ ] 1.6 Add indexes for performance (cohort_id, brand, indication)
- [ ] 1.7 Run migration on droplet and verify

### Schema Reference (from cohort_schema.sql)
```sql
-- ml_cohort_definitions: Stores versioned cohort configurations
-- ml_cohort_executions: Tracks each cohort construction run
-- ml_cohort_eligibility_log: Records per-criterion exclusions
-- ml_patient_cohort_assignments: Individual patient eligibility
```

---

## Phase 2: Core Agent Module Structure
**Estimated Files**: 4-5 Python files
**Testing**: Import tests only (no execution)

### Tasks
- [ ] 2.1 Create `src/agents/cohort_constructor/` directory
- [ ] 2.2 Create `src/agents/cohort_constructor/__init__.py` with exports
- [ ] 2.3 Create `src/agents/cohort_constructor/types.py` (Operator, CriterionType, Criterion, CohortConfig)
- [ ] 2.4 Create `src/agents/cohort_constructor/state.py` (CohortConstructorState TypedDict)
- [ ] 2.5 Create `src/agents/cohort_constructor/constants.py` (error codes CC_001-CC_007)
- [ ] 2.6 Verify imports on droplet

### Type Definitions (from cohort_constructor.py)
```python
class Operator(Enum):
    EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, IN, NOT_IN, BETWEEN, CONTAINS

class CriterionType(Enum):
    INCLUSION, EXCLUSION

@dataclass
class Criterion:
    field: str
    operator: Operator
    value: Any
    criterion_type: CriterionType
    description: str = ""
    clinical_rationale: str = ""
```

---

## Phase 3: Core Constructor Implementation
**Estimated Files**: 1 Python file (~400 lines)
**Testing**: Unit tests for operators and criteria application

### Tasks
- [ ] 3.1 Create `src/agents/cohort_constructor/constructor.py`
- [ ] 3.2 Implement `CohortConstructor.__init__()` with config validation
- [ ] 3.3 Implement `_apply_operator()` for all 10 operators
- [ ] 3.4 Implement `_apply_inclusion_criteria()` with logging
- [ ] 3.5 Implement `_apply_exclusion_criteria()` with logging
- [ ] 3.6 Implement `_validate_temporal_eligibility()` for lookback/followup
- [ ] 3.7 Implement `construct_cohort()` main pipeline
- [ ] 3.8 Run unit tests on droplet

### Core Method Signature
```python
def construct_cohort(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Main pipeline:
    1. Validate required fields
    2. Apply inclusion criteria
    3. Apply exclusion criteria
    4. Validate temporal eligibility
    5. Generate metadata
    """
```

---

## Phase 4: Brand Configuration Factory
**Estimated Files**: 1 Python file (~200 lines)
**Testing**: Factory method tests for all 3 brands

### Tasks
- [ ] 4.1 Create `src/agents/cohort_constructor/configs.py`
- [ ] 4.2 Implement `CohortConfig.from_brand("remibrutinib")` - CSU criteria
- [ ] 4.3 Implement `CohortConfig.from_brand("fabhalta")` - PNH/C3G criteria
- [ ] 4.4 Implement `CohortConfig.from_brand("kisqali")` - HR+/HER2- criteria
- [ ] 4.5 Implement `CohortConfig.to_json()` / `from_json()` for serialization
- [ ] 4.6 Run config tests on droplet

### Brand Criteria Summary
| Brand | Indication | Key Inclusion | Key Exclusion |
|-------|------------|---------------|---------------|
| Remibrutinib | CSU | Age≥18, UAS7≥16, AH failure | Pregnancy, immunodeficiency |
| Fabhalta | PNH/C3G | Diagnosis codes, lab markers | Active infection |
| Kisqali | HR+/HER2- BC | ER+/PR+, HER2-, Stage II-IV | Prior CDK4/6i |

---

## Phase 5: Agent Integration Layer
**Estimated Files**: 2 Python files
**Testing**: Integration with scope_definer output format

### Tasks
- [ ] 5.1 Create `src/agents/cohort_constructor/agent.py` (CohortConstructorAgent)
- [ ] 5.2 Implement `run()` method following E2I agent contract
- [ ] 5.3 Implement input validation from scope_definer handoff
- [ ] 5.4 Implement output formatting for data_preparer handoff
- [ ] 5.5 Create `src/agents/cohort_constructor/repository.py` (Supabase storage)
- [ ] 5.6 Implement `store_execution()`, `store_eligibility_log()`, `store_patient_assignments()`
- [ ] 5.7 Run integration tests on droplet

### Agent Contract (from cohort-constructor-contract.md)
```python
class CohortConstructorAgent:
    def run(self, input_data: CohortConstructorInput) -> CohortConstructorOutput:
        # Input: brand, indication, source_population, config_override
        # Output: cohort_id, eligible_patients, eligibility_stats, execution_metadata
```

---

## Phase 6: Observability Integration
**Estimated Files**: 2 Python files
**Testing**: MLflow logging and Opik tracing verification

### Tasks
- [ ] 6.1 Move `cohort_mlflow.py` → `src/agents/cohort_constructor/mlflow_logger.py`
- [ ] 6.2 Update MLflow logger imports and paths
- [ ] 6.3 Move `cohort_opik.py` → `src/agents/cohort_constructor/opik_tracer.py`
- [ ] 6.4 Update Opik tracer imports and paths
- [ ] 6.5 Implement `InstrumentedCohortConstructor` wrapper
- [ ] 6.6 Test MLflow experiment logging on droplet
- [ ] 6.7 Test Opik distributed tracing on droplet

### Observability Stack
- **MLflow**: Experiment tracking (cohort_size, exclusion_rates, execution_time)
- **Opik**: Distributed tracing with span hierarchy (run → validate → apply → temporal)

---

## Phase 7: Test Suite Creation
**Estimated Files**: 3-4 test files
**Testing**: Run in batches on droplet (max 4 workers)

### Tasks
- [ ] 7.1 Create `tests/unit/test_agents/test_cohort_constructor/` directory
- [ ] 7.2 Create `test_types.py` - Operator and Criterion tests
- [ ] 7.3 Create `test_constructor.py` - Core logic tests (~20 tests)
- [ ] 7.4 Create `test_configs.py` - Brand configuration tests
- [ ] 7.5 Create `test_agent.py` - Agent integration tests
- [ ] 7.6 Create `conftest.py` with fixtures (mock patients, configs)
- [ ] 7.7 Run batch 1: `test_types.py` on droplet
- [ ] 7.8 Run batch 2: `test_constructor.py` on droplet
- [ ] 7.9 Run batch 3: `test_configs.py` + `test_agent.py` on droplet

### Test Execution Command
```bash
# Memory-safe execution (4 workers max)
pytest tests/unit/test_agents/test_cohort_constructor/ -n 4 --dist=loadscope
```

---

## Phase 8: Documentation & Folder Reorganization
**Estimated Files**: 5-6 file moves/updates
**Testing**: Verify imports still work after reorganization

### Tasks
- [ ] 8.1 Create `src/agents/cohort_constructor/CLAUDE.md` (specialist doc)
- [ ] 8.2 Update `.claude/specialists/Agent_Specialists_Tier 0/` with cohort_constructor.md
- [ ] 8.3 Move remaining docs to appropriate locations:
  - `cohort-constructor-contract.md` → `.claude/contracts/tier0/cohort_constructor.md`
  - `cohort-constructor-handoff.yaml` → `.claude/contracts/tier0/cohort_constructor_handoff.yaml`
  - `cohort-constructor-data-contract.md` → `.claude/contracts/tier0/cohort_constructor_data.md`
  - `cohort-constructor-specialist.md` → `.claude/specialists/Agent_Specialists_Tier 0/cohort_constructor.md`
- [ ] 8.4 Move vocabulary files:
  - `cohort_vocabulary.yaml` → `config/cohort_vocabulary.yaml`
  - `cohort_ontology_update_workflow.md` → `docs/cohort_ontology_workflow.md`
  - `sync_clinical_codes.py` → `scripts/sync_clinical_codes.py`
- [ ] 8.5 Move remaining implementation files:
  - `test_cohort_constructor.py` → `tests/unit/test_agents/test_cohort_constructor/test_original.py`
  - `e2i_tier0_integration.py` → `src/agents/cohort_constructor/tier0_integration.py`
- [ ] 8.6 Move documentation:
  - `README_CohortConstructor.md` → `docs/agents/cohort_constructor.md`
  - `cohort_observability_integration_guide.md` → `docs/observability/cohort_constructor.md`
  - `CohortConstructor_vs_CohortNet_Comparison.md` → `docs/architecture/cohort_constructor_vs_cohortnet.md`
- [ ] 8.7 Update AGENT-INDEX-V4.md to include cohort_constructor
- [ ] 8.8 Delete or archive CohortConstructor/ folder
- [ ] 8.9 Run full import verification on droplet
- [ ] 8.10 Final test suite run on droplet

### Final Project Structure
```
src/agents/cohort_constructor/
├── __init__.py
├── CLAUDE.md
├── agent.py
├── configs.py
├── constants.py
├── constructor.py
├── mlflow_logger.py
├── opik_tracer.py
├── repository.py
├── state.py
├── tier0_integration.py
└── types.py

.claude/
├── contracts/tier0/
│   ├── cohort_constructor.md
│   ├── cohort_constructor_data.md
│   └── cohort_constructor_handoff.yaml
└── specialists/Agent_Specialists_Tier 0/
    └── cohort_constructor.md

config/
└── cohort_vocabulary.yaml

docs/
├── agents/cohort_constructor.md
├── architecture/cohort_constructor_vs_cohortnet.md
└── observability/cohort_constructor.md

scripts/
└── sync_clinical_codes.py

tests/unit/test_agents/test_cohort_constructor/
├── conftest.py
├── test_agent.py
├── test_configs.py
├── test_constructor.py
├── test_original.py
└── test_types.py
```

---

## Execution Checklist

### Phase Completion Tracking

| Phase | Status | Tests Passed | Notes |
|-------|--------|--------------|-------|
| 1. Database Migration | [ ] Pending | N/A | |
| 2. Module Structure | [ ] Pending | [ ] Imports | |
| 3. Core Constructor | [ ] Pending | [ ] Unit | |
| 4. Brand Configs | [ ] Pending | [ ] Factory | |
| 5. Agent Integration | [ ] Pending | [ ] Integration | |
| 6. Observability | [ ] Pending | [ ] MLflow/Opik | |
| 7. Test Suite | [ ] Pending | [ ] Full Suite | |
| 8. Reorganization | [ ] Pending | [ ] Final | |

### Droplet Testing Commands

```bash
# SSH to droplet
ssh root@159.89.180.27

# Navigate to project
cd /root/e2i_causal_analytics

# Run specific phase tests (memory-safe)
pytest tests/unit/test_agents/test_cohort_constructor/test_types.py -v
pytest tests/unit/test_agents/test_cohort_constructor/test_constructor.py -n 4
pytest tests/unit/test_agents/test_cohort_constructor/ -n 4 --dist=loadscope

# Verify MLflow
mlflow ui --host 0.0.0.0 --port 5000

# Verify Opik
curl http://localhost:5173/api/health
```

---

## Risk Mitigation

1. **Context Window**: Each phase is self-contained (~200-400 lines)
2. **Memory on Droplet**: Max 4 pytest workers, loadscope distribution
3. **Rollback**: Each phase can be reverted independently
4. **Dependencies**: Phases 1-4 can run in parallel; 5-8 are sequential

---

## References

- Source files: `CohortConstructor/` folder
- Existing Tier 0 agents: `src/agents/tier_0/`
- Agent contracts: `.claude/contracts/`
- Test patterns: `.claude/.agent_docs/testing-patterns.md`
