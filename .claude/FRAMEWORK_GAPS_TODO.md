# Claude Code Framework - Gaps & Enhancements TODO

**Project**: E2I Causal Analytics
**Created**: 2025-12-18
**Purpose**: Track completion of framework gaps and enhancements identified in audit

---

## Progress Overview

| Category | Total Tasks | Completed | In Progress | Not Started |
|----------|-------------|-----------|-------------|-------------|
| 🔴 Critical (Blockers) | 12 | 0 | 0 | 12 |
| 🟠 High Priority | 6 | 0 | 0 | 6 |
| 🟡 Medium Priority | 8 | 0 | 0 | 8 |
| 🟢 Low Priority | 4 | 0 | 0 | 4 |
| **TOTAL** | **30** | **0** | **0** | **30** |

**Overall Completion**: 0% (0/30)

---

## 🔴 CRITICAL PRIORITY (Blockers for Implementation)

### Contract Files Creation

These files are REQUIRED before any agent implementation can begin.

#### ✅ Foundation Contracts (Days 1-2)

- [ ] **TASK-001**: Create `base-contract.md`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Define `AgentState` base structure
    - [ ] Define `AgentConfig` base structure
    - [ ] Define `BaseAgent` interface contract
    - [ ] Define `AgentResult` return structure
    - [ ] Define error handling contracts
    - [ ] Include Pydantic model definitions
    - [ ] Include validation rules
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: All agent specialist files

- [ ] **TASK-002**: Create `orchestrator-contracts.md`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 3 hours
  - **Dependencies**: TASK-001 (base-contract.md)
  - **Acceptance Criteria**:
    - [ ] Define Orchestrator → Agent dispatch format
    - [ ] Define Agent → Orchestrator result format
    - [ ] Define agent selection criteria
    - [ ] Define result aggregation contract
    - [ ] Define error propagation rules
    - [ ] Include routing decision logic
    - [ ] Include parallel execution contracts
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: `orchestrator-agent.md`

- [ ] **TASK-003**: Create `agent-handoff.yaml`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 1.5 hours
  - **Dependencies**: TASK-001, TASK-002
  - **Acceptance Criteria**:
    - [ ] Define standard handoff YAML schema
    - [ ] Include required fields specification
    - [ ] Include optional fields specification
    - [ ] Include examples for all 18 agents
    - [ ] Define success/failure indicators
    - [ ] Define context preservation rules
    - [ ] Include validation schema
  - **Template Source**: Handoff sections from specialist files
  - **Reference Files**: All agent specialist files

#### ✅ Tier Contracts (Days 3-7)

- [ ] **TASK-004**: Create `tier0-contracts.md` (ML Foundation)
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 3 hours
  - **Dependencies**: TASK-001
  - **Acceptance Criteria**:
    - [ ] Define scope_definer → data_preparer contract
    - [ ] Define data_preparer → model_selector contract (with QC gate)
    - [ ] Define model_selector → model_trainer contract
    - [ ] Define model_trainer → feature_analyzer contract
    - [ ] Define feature_analyzer → model_deployer contract
    - [ ] Define model_deployer → observability_connector contract
    - [ ] Include QC gate blocking logic
    - [ ] Include MLOps tool integration contracts
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: `ml_foundation/*.md` files

- [ ] **TASK-005**: Create `tier2-contracts.md` (Causal Inference)
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-001
  - **Acceptance Criteria**:
    - [ ] Define causal_impact agent contracts
    - [ ] Define gap_analyzer agent contracts
    - [ ] Define heterogeneous_optimizer agent contracts
    - [ ] Define inter-agent communication within tier
    - [ ] Define causal engine integration
    - [ ] Include DoWhy/EconML result formats
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: `causal-impact.md`, `gap-analyzer.md`, `heterogeneous-optimizer.md`

- [ ] **TASK-006**: Create `tier3-contracts.md` (Design & Monitoring)
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-001
  - **Acceptance Criteria**:
    - [ ] Define experiment_designer agent contracts
    - [ ] Define drift_monitor agent contracts
    - [ ] Define health_score agent contracts
    - [ ] Define drift detection result format
    - [ ] Define experiment design validation
    - [ ] Include baseline metrics integration
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: Tier 3 agent specialist files

- [ ] **TASK-007**: Create `tier4-contracts.md` (ML Predictions)
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-001, TASK-004
  - **Acceptance Criteria**:
    - [ ] Define prediction_synthesizer agent contracts
    - [ ] Define resource_optimizer agent contracts
    - [ ] Define model endpoint integration
    - [ ] Define prediction aggregation rules
    - [ ] Include deployed model consumption contracts
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: Tier 4 agent specialist files

- [ ] **TASK-008**: Create `tier5-contracts.md` (Self-Improvement)
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-001
  - **Acceptance Criteria**:
    - [ ] Define explainer agent contracts
    - [ ] Define feedback_learner agent contracts
    - [ ] Define explanation format standards
    - [ ] Define feedback collection format
    - [ ] Define DSPy optimization contracts
    - [ ] Include RAG weight update format
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: `explainer.md`, `feedback-learner.md`

#### ✅ Integration Contracts (Days 8-10)

- [ ] **TASK-009**: Create `integration-contracts.md`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 3 hours
  - **Dependencies**: TASK-001, TASK-002
  - **Acceptance Criteria**:
    - [ ] Define API ↔ Orchestrator contracts
    - [ ] Define Orchestrator ↔ NLP contracts
    - [ ] Define Agents ↔ RAG contracts
    - [ ] Define Agents ↔ Database contracts
    - [ ] Define Agents ↔ Causal Engine contracts
    - [ ] Define Frontend ↔ API contracts
    - [ ] Define WebSocket streaming contracts
    - [ ] Include authentication/authorization contracts
  - **Template Source**: `data-contracts.md`
  - **Reference Files**: System specialist files (`api.md`, `nlp.md`, `rag.md`, etc.)

- [ ] **TASK-010**: Create `orchestrator-dispatch.yaml`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-002
  - **Acceptance Criteria**:
    - [ ] Define routing rules from orchestrator to agents
    - [ ] Include intent → agent mapping
    - [ ] Include KPI → agent mapping
    - [ ] Define priority rules for multi-agent queries
    - [ ] Define parallel vs sequential execution rules
    - [ ] Include fallback routing
    - [ ] Define routing decision tree
  - **Template Source**: `agent_config.yaml` routing section
  - **Reference Files**: `orchestrator-agent.md`

- [ ] **TASK-011**: Create `inter-agent.yaml`
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-003, TASK-004-008
  - **Acceptance Criteria**:
    - [ ] Define cross-tier communication patterns
    - [ ] Define agent dependency graph
    - [ ] Define data flow between tiers
    - [ ] Include Tier 0 → Tier 1-5 contracts
    - [ ] Define context propagation rules
    - [ ] Include example multi-agent workflows
  - **Template Source**: `agent-handoff.yaml`
  - **Reference Files**: All tier contract files

- [ ] **TASK-012**: Validate All Contract Files
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-001 through TASK-011
  - **Acceptance Criteria**:
    - [ ] Cross-reference all contracts for consistency
    - [ ] Validate Pydantic models compile
    - [ ] Check all agent integrations are covered
    - [ ] Verify no circular dependencies
    - [ ] Ensure all 18 agents have complete contracts
    - [ ] Create contract validation test suite
    - [ ] Generate contract compatibility matrix

---

## 🟠 HIGH PRIORITY (Should Complete Before Implementation)

### Contract Enhancement

- [ ] **TASK-013**: Extract Contracts from Specialist Files
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 4 hours
  - **Dependencies**: TASK-001 through TASK-011
  - **Acceptance Criteria**:
    - [ ] Extract all "Integration Contracts" sections from specialist files
    - [ ] Compare extracted contracts with created contract files
    - [ ] Identify discrepancies
    - [ ] Update contract files to match specialist files
    - [ ] Add cross-references in specialist files to contract files
  - **Purpose**: Ensure consistency between embedded and centralized contracts

- [ ] **TASK-014**: Create Contract Testing Framework
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 3 hours
  - **Dependencies**: TASK-012
  - **Acceptance Criteria**:
    - [ ] Create Pydantic models for all contracts
    - [ ] Add unit tests for contract validation
    - [ ] Create CI/CD validation script
    - [ ] Add contract compatibility tests
    - [ ] Generate contract documentation
  - **Output**: `tests/unit/test_contracts/` directory

- [ ] **TASK-015**: Create Contract Change Management Process
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 1 hour
  - **Dependencies**: TASK-012
  - **Acceptance Criteria**:
    - [ ] Define contract versioning scheme
    - [ ] Create breaking change checklist
    - [ ] Define deprecation process
    - [ ] Create contract changelog template
    - [ ] Add contract review process
  - **Output**: `CONTRACTS_CHANGELOG.md`

### Configuration Validation

- [ ] **TASK-016**: Validate agent_config.yaml References
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Verify all agents in config match specialist files
    - [ ] Verify all tools referenced exist in MLOps tools
    - [ ] Verify all memory types are valid
    - [ ] Verify all SLA values are reasonable
    - [ ] Check tier assignments match documentation
    - [ ] Validate routing configuration
  - **Output**: Validated `config/agent_config.yaml`

- [ ] **TASK-017**: Validate domain_vocabulary.yaml
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 1 hour
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Verify all brands have vocabulary entries
    - [ ] Verify all KPIs have vocabulary entries
    - [ ] Check for typos and inconsistencies
    - [ ] Validate against kpi-dictionary.md
    - [ ] Add missing entries
  - **Output**: Validated vocabulary files

- [ ] **TASK-018**: Validate Database Schema vs. Contracts
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-012
  - **Acceptance Criteria**:
    - [ ] Compare database tables with contract specifications
    - [ ] Verify all contract fields have corresponding DB columns
    - [ ] Check data types match
    - [ ] Validate foreign key relationships
    - [ ] Identify missing tables or columns
    - [ ] Generate schema migration if needed
  - **Output**: Schema validation report

---

## 🟡 MEDIUM PRIORITY (Nice to Have, Improves Quality)

### Specialist File Enhancements

- [ ] **TASK-019**: Update Model Versions in Specialist Files
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 1 hour
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Identify all model references in specialist files
    - [ ] Update to latest model versions
    - [ ] Verify model capabilities match usage
    - [ ] Update fallback model chains
  - **Files Affected**: All agent specialist files

- [ ] **TASK-020**: Add "Related Specialists" Cross-References
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Add "Related Specialists" section to each file
    - [ ] Link agents that commonly work together
    - [ ] Link system specialists to agent specialists
    - [ ] Create bidirectional references
  - **Files Affected**: All specialist files

- [ ] **TASK-021**: Enhance AGENT-INDEX-V4.md with Quick Start
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 1 hour
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Add "Quick Start" section for new developers
    - [ ] Add "Which Specialist Do I Need?" decision tree
    - [ ] Add common task → specialist mapping
    - [ ] Add troubleshooting section
  - **Output**: Enhanced `AGENT-INDEX-V4.md`

- [ ] **TASK-022**: Create Visual Architecture Diagrams
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 3 hours
  - **Dependencies**: TASK-012
  - **Acceptance Criteria**:
    - [ ] Create tier architecture diagram (Mermaid)
    - [ ] Create data flow diagram
    - [ ] Create contract relationship diagram
    - [ ] Create agent dependency graph
    - [ ] Add diagrams to relevant specialist files
  - **Output**: `docs/architecture/diagrams/`

### Context File Enhancements

- [ ] **TASK-023**: Update summary-v4.md "Open Decisions"
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 30 minutes
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Review all open decisions
    - [ ] Move resolved items to "Recently Resolved"
    - [ ] Add new pending decisions
    - [ ] Update decision owners
    - [ ] Add target resolution dates
  - **Output**: Updated `summary-v4.md`

- [ ] **TASK-024**: Expand experiment-history.md
  - **Status**: Not Started
  - **Assignee**: User (domain knowledge required)
  - **Estimated Time**: Variable
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] Add more historical experiments if available
    - [ ] Validate existing experiments
    - [ ] Add learnings from completed experiments
    - [ ] Update effect size calibration
  - **Output**: Updated `experiment-history.md`
  - **Note**: Ongoing as experiments complete

- [ ] **TASK-025**: Create "Common Patterns" Context File
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: Implementation experience
  - **Acceptance Criteria**:
    - [ ] Document common agent patterns
    - [ ] Document common integration patterns
    - [ ] Document common error handling patterns
    - [ ] Add code examples
    - [ ] Link to relevant specialist files
  - **Output**: `.claude/context/common-patterns.md`

- [ ] **TASK-026**: Create "Troubleshooting Guide" Context File
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: Implementation experience
  - **Acceptance Criteria**:
    - [ ] Common errors and solutions
    - [ ] Integration troubleshooting
    - [ ] Performance troubleshooting
    - [ ] Debugging strategies
    - [ ] Tool-specific issues
  - **Output**: `.claude/context/troubleshooting-guide.md`

---

## 🟢 LOW PRIORITY (Polish, Can Do Anytime)

### Documentation Polish

- [ ] **TASK-027**: Add Examples to All Specialist Files
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 4 hours
  - **Dependencies**: Implementation
  - **Acceptance Criteria**:
    - [ ] Add "Example Usage" section to each specialist
    - [ ] Include end-to-end example workflows
    - [ ] Add common query → agent → result examples
    - [ ] Include error handling examples
  - **Files Affected**: All specialist files

- [ ] **TASK-028**: Create Quick Reference Cards
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: None
  - **Acceptance Criteria**:
    - [ ] 1-page quick reference per agent
    - [ ] 1-page quick reference per system layer
    - [ ] Include most common operations
    - [ ] Include key contracts
  - **Output**: `docs/quick-reference/`

- [ ] **TASK-029**: Create Onboarding Guide
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: TASK-012
  - **Acceptance Criteria**:
    - [ ] Framework overview for new developers
    - [ ] Step-by-step setup guide
    - [ ] First agent implementation tutorial
    - [ ] Common pitfalls and solutions
  - **Output**: `ONBOARDING.md`

- [ ] **TASK-030**: Generate API Documentation
  - **Status**: Not Started
  - **Assignee**: Claude
  - **Estimated Time**: 2 hours
  - **Dependencies**: Implementation
  - **Acceptance Criteria**:
    - [ ] Auto-generate docs from Pydantic models
    - [ ] Create OpenAPI spec for API endpoints
    - [ ] Document all agent interfaces
    - [ ] Add interactive API explorer
  - **Output**: `docs/api/`

---

## Execution Plan

### Phase 1: Critical Foundation (Days 1-10)
**Goal**: Create all contract files
- Days 1-2: Foundation contracts (TASK-001, TASK-002, TASK-003)
- Days 3-7: Tier contracts (TASK-004 through TASK-008)
- Days 8-9: Integration contracts (TASK-009, TASK-010, TASK-011)
- Day 10: Validation (TASK-012)

### Phase 2: High Priority Enhancements (Days 11-15)
**Goal**: Ensure quality and consistency
- Days 11-12: Contract extraction and testing (TASK-013, TASK-014, TASK-015)
- Days 13-15: Configuration validation (TASK-016, TASK-017, TASK-018)

### Phase 3: Medium Priority Improvements (Days 16-20)
**Goal**: Polish and enhance usability
- Days 16-17: Specialist file enhancements (TASK-019, TASK-020, TASK-021)
- Days 18-19: Visual diagrams (TASK-022)
- Day 20: Context file updates (TASK-023, TASK-024, TASK-025, TASK-026)

### Phase 4: Low Priority Polish (Ongoing)
**Goal**: Continuous improvement
- As implementation progresses: TASK-027 through TASK-030

---

## Success Criteria

### Minimum Viable Framework (MVF)
To start implementation, we need:
- ✅ All 12 contract files created (TASK-001 through TASK-011)
- ✅ Contract validation passing (TASK-012)
- ✅ Configuration validated (TASK-016, TASK-017)

**MVF Completion**: 15 of 30 tasks (50%)

### Production-Ready Framework (PRF)
For production deployment, we need:
- ✅ MVF complete
- ✅ Contract testing framework (TASK-014)
- ✅ Database schema validated (TASK-018)
- ✅ Visual diagrams (TASK-022)
- ✅ Troubleshooting guide (TASK-026)

**PRF Completion**: 20 of 30 tasks (67%)

### Excellent Framework (EXF)
For best-in-class framework, complete all 30 tasks.

**EXF Completion**: 30 of 30 tasks (100%)

---

## Notes

### Change Log
| Date | Change | Tasks Affected |
|------|--------|----------------|
| 2025-12-18 | Initial creation | All |

### Blocked Tasks
None currently

### Risk Items
- TASK-024: Requires domain knowledge - user input needed
- TASK-027 through TASK-030: Require implementation to be complete

### Dependencies Graph
```
TASK-001 (base-contract)
  ├── TASK-002 (orchestrator-contracts)
  │   ├── TASK-009 (integration-contracts)
  │   └── TASK-010 (orchestrator-dispatch)
  ├── TASK-003 (agent-handoff)
  ├── TASK-004 (tier0-contracts)
  ├── TASK-005 (tier2-contracts)
  ├── TASK-006 (tier3-contracts)
  ├── TASK-007 (tier4-contracts)
  └── TASK-008 (tier5-contracts)
      └── TASK-011 (inter-agent)
          └── TASK-012 (validation)
              ├── TASK-013 (extraction)
              ├── TASK-014 (testing)
              ├── TASK-015 (change mgmt)
              └── TASK-018 (schema validation)
```

---

## Ready for Option 1

**Status**: ✅ READY

All gaps have been cataloged. We are ready to proceed with:
- **Option 1**: Generate all 12 missing contract files based on specialist file content

**Estimated Time for Option 1**: 20-24 hours of work (can parallelize)

**Execution Strategy**:
1. Generate files in dependency order (foundation → tier → integration)
2. Validate each file before proceeding to next
3. Cross-reference with specialist files
4. Run final validation suite

**Waiting for**: User approval to begin Option 1 execution
