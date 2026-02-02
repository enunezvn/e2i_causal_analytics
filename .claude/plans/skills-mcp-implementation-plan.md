# Skills & MCP Implementation Plan

**Created**: 2025-01-26
**Updated**: 2026-02-01
**Status**: Phases 1-4 Complete, Phase 5 Deferred (MCP connectors unavailable), Phase 6 Partially Pending
**Priority**: High
**Reference**: `.claude/PRPs/skills-vs-mcp-evaluation.md`, `.claude/PRPs/mcp-skills-implementation-plan.md`, `.claude/skills/SKILL_INTEGRATION.md`

---

## Executive Summary

This plan implements the hybrid Skills + MCP approach with **Skills first** (zero infrastructure) followed by **MCP Connectors** (requires verification). This ordering minimizes friction with the existing E2I workflow.

### Why Skills First?

| Approach | Infrastructure | Dependencies | Risk |
|----------|---------------|--------------|------|
| **Skills** | None | Pure Python | Zero - just markdown files |
| **MCP Connectors** | Anthropic API | Subscription tier, connector availability | Medium - requires verification |

### Implementation Order

1. **Phase 1**: Skills Framework (Zero Infrastructure)
2. **Phase 2**: Agent-Skill Integration
3. **Phase 3**: MCP Connector Gateway (After Verification)
4. **Phase 4**: Full Agent Integration (Skills + MCP)

---

## Progress Tracker

### Phase 1: Skills Framework Core (Zero Infrastructure) ✅ COMPLETE
- [x] **1.1** Create skills directory structure
- [x] **1.2** Implement SkillLoader class
- [x] **1.3** Implement SkillMatcher class
- [x] **1.4** Create master skill index (SKILL.md)
- [x] **1.5** Unit tests for SkillLoader
- [x] **1.6** Droplet deployment test (34/34 tests passed)

### Phase 2: Domain Skills Content ✅ COMPLETE
- [x] **2.1** Create pharma-commercial skills (kpi-calculation, brand-analytics, patient-journey)
- [x] **2.2** Create causal-inference skills (confounder-identification, dowhy-workflow)
- [x] **2.3** Create experiment-design skills (validity-threats, power-analysis)
- [x] **2.4** Create gap-analysis skills (roi-estimation, opportunity-sizing)
- [x] **2.5** Skill content validation tests (test_skill_content.py)

### Phase 3: Agent-Skill Integration ✅ COMPLETE
- [x] **3.1** Implement SkillsMixin (src/agents/base/skills_mixin.py — 316 lines)
- [x] **3.2** Integrate skills with `causal_impact` agent
- [x] **3.3** Integrate skills with `experiment_designer` agent
- [x] **3.4** Integrate skills with `gap_analyzer` agent
- [x] **3.5** Integrate skills with `explainer` agent
- [x] **3.6** Integration tests (tests/integration/test_agent_skills.py — 22+ test cases)

### Phase 4: MCP Connector Verification ✅ COMPLETE (Result: Connectors Unavailable)
- [x] **4.1** Verify Anthropic connector API access — **FAILED (401 auth error)**
- [x] **4.2** Test ChEMBL connector availability — **FAILED (401 auth error)**
- [x] **4.3** Test ClinicalTrials.gov connector availability — **FAILED (401 auth error)**
- [x] **4.4** Test PubMed connector availability — **FAILED (401 auth error)**
- [ ] **4.5** Document connector access status in PRP (results in `scripts/mcp_connector_results.json`, PRP not updated)

### Phase 5: MCP Gateway Implementation — ⏸️ DEFERRED (Connectors Unavailable)
- [ ] ~~**5.1** Create MCP connector gateway~~ — Blocked: no available connectors
- [ ] ~~**5.2** Implement permission checking~~ — Blocked
- [ ] ~~**5.3** Create connector-specific helpers~~ — Blocked
- [ ] ~~**5.4** Unit tests for gateway~~ — Blocked
- [ ] ~~**5.5** Droplet deployment test~~ — Blocked

### Phase 6: Full Integration — ⏸️ PARTIALLY PENDING
- [ ] **6.1** Update `tool_composer` with skills-only orchestration (MCP deferred)
- [ ] ~~**6.2** Create MCP config (config/mcp_config.yaml)~~ — Deferred with MCP
- [ ] **6.3** End-to-end integration tests (skills-only path)
- [ ] **6.4** Performance monitoring setup (skills metrics)

---

## Phase 1: Skills Framework Core

### Workflow Impact Summary (Phase 1)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `.claude/skills/` | New directory tree | **Addition** |
| `src/skills/__init__.py` | New module | **Addition** |
| `src/skills/loader.py` | New file | **Addition** |
| `src/skills/matcher.py` | New file | **Addition** |
| `tests/unit/test_skills/` | New test directory | **Addition** |
| Existing agents | **No changes** | None |
| Existing API | **No changes** | None |

**Breaking Changes**: None - this phase is purely additive.

---

### 1.1 Create Skills Directory Structure

**Status**: [x] Complete

**Task**: Create the skills directory hierarchy

```bash
.claude/skills/
├── SKILL.md                           # Master skill index
├── pharma-commercial/
│   ├── SKILL.md                       # Category index
│   ├── kpi-calculation.md
│   ├── brand-analytics.md
│   └── patient-journey.md
├── causal-inference/
│   ├── SKILL.md
│   ├── confounder-identification.md
│   └── dowhy-workflow.md
├── experiment-design/
│   ├── SKILL.md
│   ├── validity-threats.md
│   └── power-analysis.md
└── gap-analysis/
    ├── SKILL.md
    └── roi-estimation.md
```

**Sub-tasks**:
1. Create root `.claude/skills/` directory
2. Create category subdirectories (4 directories)
3. Create placeholder SKILL.md in each category
4. Verify structure matches SKILL_INTEGRATION.md template

**Workflow Impact**: None - creates new directory tree only

**Verification**:
```bash
ls -la .claude/skills/
ls -la .claude/skills/*/
```

---

### 1.2 Implement SkillLoader Class

**Status**: [x] Complete

**Files to Create**:
- `src/skills/__init__.py` - Module exports
- `src/skills/loader.py` - SkillLoader implementation

**Key Components**:
- `SkillMetadata` dataclass (name, version, description, triggers, agents)
- `Skill` dataclass (metadata, content, sections)
- `SkillLoader` class with caching

**Sub-tasks**:
1. Create `src/skills/` directory
2. Create `__init__.py` with exports (`SkillLoader`, `SkillMatcher`, `Skill`, `SkillMetadata`)
3. Implement `SkillMetadata` dataclass with YAML frontmatter fields
4. Implement `Skill` dataclass with `get_section(name)` method
5. Implement `SkillLoader.load(path)` async method
6. Implement `SkillLoader.load_section(path, section)` for token efficiency
7. Add LRU caching decorator for loaded skills
8. Handle FileNotFoundError gracefully

**Dependencies**:
- `pyyaml` (already in requirements.txt)
- Standard library only

**Workflow Impact**:
- Creates new `src/skills/` module
- No changes to existing code
- Agents will import from this module in Phase 3

**Verification**:
```python
# Quick test
from src.skills import SkillLoader
skill = await SkillLoader.load("pharma-commercial/kpi-calculation.md")
print(skill.metadata.name)
print(skill.get_section("TRx (Total Prescriptions)"))
```

---

### 1.3 Implement SkillMatcher Class

**Status**: [x] Complete

**File**: `src/skills/matcher.py`

**Key Components**:
- `SkillMatch` dataclass (skill_path, score, matched_triggers)
- `SkillMatcher` class with keyword matching
- Score-based ranking

**Sub-tasks**:
1. Create `src/skills/matcher.py`
2. Implement `SkillMatch` dataclass with fields:
   - `skill_path: str`
   - `score: float` (0.0 to 1.0)
   - `matched_triggers: list[str]`
3. Implement keyword extraction from skill YAML frontmatter `triggers` field
4. Build keyword index on initialization (load all skill metadata)
5. Implement `find_matches(query: str) -> list[SkillMatch]`
6. Implement scoring algorithm (weighted keyword overlap)
7. Return sorted matches (highest score first)

**Keyword Categories** (extracted from skill triggers):
- KPI terms: trx, nrx, prescription, market share, conversion
- Brand terms: kisqali, fabhalta, remibrutinib
- Causal terms: confounder, dowhy, ate, cate
- Experiment terms: validity, power, sample size

**Workflow Impact**:
- New file only, no changes to existing code
- Will be used by `BaseAgent.find_relevant_skills()` in Phase 3

**Verification**:
```python
matcher = SkillMatcher()
matches = await matcher.find_matches("calculate TRx for Kisqali")
# Should return: pharma-commercial/kpi-calculation.md with high score
```

---

### 1.4 Create Master Skill Index

**Status**: [x] Complete

**File**: `.claude/skills/SKILL.md`

**Content**: YAML frontmatter with:
- name: E2I Pharma Commercial Analytics Skills
- version: 1.0
- categories list
- triggers list
- When to load guidance

---

### 1.5 Unit Tests for SkillLoader

**Status**: [x] Complete

**File**: `tests/unit/test_skills/test_loader.py`

**Test Cases**:
- `test_load_skill_with_frontmatter`
- `test_skill_caching`
- `test_section_extraction`
- `test_skill_not_found`
- `test_trigger_matching`

---

### 1.6 Droplet Deployment Test

**Status**: [x] Complete (34/34 tests passed)

**Test Command**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_skills/ -v --tb=short"
```

**Acceptance Criteria**:
- All skill loader tests pass
- Skills load correctly from `.claude/skills/`
- Caching works as expected

---

## Phase 2: Domain Skills Content ✅ COMPLETE

### Workflow Impact Summary (Phase 2)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `.claude/skills/pharma-commercial/*.md` | 3 new files | **Addition** |
| `.claude/skills/causal-inference/*.md` | 3 new files | **Addition** |
| `.claude/skills/experiment-design/*.md` | 3 new files | **Addition** |
| `.claude/skills/gap-analysis/*.md` | 2 new files | **Addition** |
| `tests/unit/test_skills/test_skill_content.py` | New test file | **Addition** |
| Existing agents | **No changes** | None |
| Existing config | **No changes** | None |

**Breaking Changes**: None - this phase is purely additive.

**Content Source**: All skill content derived from `SKILL_INTEGRATION.md` templates.

---

### 2.1 Create Pharma-Commercial Skills

**Status**: [x] Complete (kpi-calculation.md, brand-analytics.md, patient-journey.md)

**Files to Create**:
1. `.claude/skills/pharma-commercial/SKILL.md` - Category index
2. `.claude/skills/pharma-commercial/kpi-calculation.md` - Full KPI procedures
3. `.claude/skills/pharma-commercial/brand-analytics.md` - Brand context

**Sub-tasks**:
1. Create `SKILL.md` category index with:
   - Category name and description
   - List of skills in category
   - Agent mapping (which agents use this category)
2. Create `kpi-calculation.md` with YAML frontmatter:
   - `name: KPI Calculation Procedures`
   - `triggers: [trx, nrx, prescription, market share, kpi]`
   - `agents: [gap_analyzer, health_score, orchestrator]`
3. Create `brand-analytics.md` with brand-specific context:
   - Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2-)
   - Brand-specific KPI benchmarks
   - Historical performance context

**Content Source**: Use templates from `SKILL_INTEGRATION.md`

**Key Sections for kpi-calculation.md**:
- Prescription Volume Metrics (TRx, NRx, NBRx)
- Engagement Metrics (Conversion Rate, HCP Reach)
- Adherence Metrics (PDC, Persistence)
- ROI Calculation formulas

**Workflow Impact**: None - creates skill content files only

---

### 2.2 Create Causal-Inference Skills

**Status**: [x] Complete (confounder-identification.md, dowhy-workflow.md)

**Files to Create**:
1. `.claude/skills/causal-inference/SKILL.md`
2. `.claude/skills/causal-inference/confounder-identification.md`
3. `.claude/skills/causal-inference/dowhy-workflow.md`

**Key Sections for confounder-identification.md**:
- HCP Targeting → Prescription Impact confounders
- Patient Journey → Outcome Analysis confounders
- Trigger → Conversion Analysis confounders
- Instrumental Variables list

**Key Sections for dowhy-workflow.md**:
- Phase 1: DAG Construction
- Phase 2: Estimation with Fallback Chain
- Phase 3: Refutation Testing
- Phase 4: Sensitivity Analysis
- Phase 5: Interpretation by Audience

---

### 2.3 Create Experiment-Design Skills

**Status**: [x] Complete (validity-threats.md, power-analysis.md)

**Files to Create**:
1. `.claude/skills/experiment-design/SKILL.md`
2. `.claude/skills/experiment-design/validity-threats.md`
3. `.claude/skills/experiment-design/power-analysis.md`

**Key Sections for validity-threats.md**:
- 6-Threat Taxonomy (Selection Bias, Confounding, Measurement Error, Contamination, Temporal Effects, Attrition)
- Detection Methods for each threat
- Mitigations for each threat
- Validity Scoring Framework

---

### 2.4 Create Gap-Analysis Skills

**Status**: [x] Complete (roi-estimation.md, opportunity-sizing.md)

**Files to Create**:
1. `.claude/skills/gap-analysis/SKILL.md`
2. `.claude/skills/gap-analysis/roi-estimation.md`

**Key Sections for roi-estimation.md**:
- Revenue Impact Calculation (multipliers)
- Cost-to-Close Calculation
- ROI Formula
- Payback Period
- Opportunity Categorization (Quick Wins, Strategic Bets)

---

### 2.5 Skill Content Validation Tests

**Status**: [x] Complete (tests/unit/test_skills/test_skill_content.py)

**File**: `tests/unit/test_skills/test_skill_content.py`

**Test Cases**:
- All skill files parse without errors
- All required sections exist
- Triggers are properly defined
- Agent mappings are correct

---

## Phase 3: Agent-Skill Integration ✅ COMPLETE

### Workflow Impact Summary (Phase 3)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `src/agents/base.py` | Add 4 skill methods | **Modification** |
| `src/agents/causal_impact/agent.py` | Add skill loading calls | **Modification** |
| `src/agents/experiment_designer/agent.py` | Add skill loading calls | **Modification** |
| `src/agents/gap_analyzer/agent.py` | Add skill loading calls | **Modification** |
| `src/agents/explainer/agent.py` | Add skill loading calls | **Modification** |
| `tests/integration/test_agent_skills.py` | New integration tests | **Addition** |

**Breaking Changes**:
- `BaseAgent` interface expands (backward compatible - new methods only)
- Agents remain functional without skills (graceful degradation)

**Risk Mitigation**:
- Add feature flag `SKILLS_ENABLED` in config (default: True)
- Wrap skill loading in try/except for graceful fallback
- Skills enhance but don't replace existing agent logic

---

### 3.1 Implement SkillsMixin (Approach Changed)

**Status**: [x] Complete — Implemented as `src/agents/base/skills_mixin.py` (316 lines) instead of modifying `base.py`

**File**: `src/agents/base/skills_mixin.py` (new — mixin pattern)

**Methods to Add**:
```python
async def load_skill(self, skill_path: str) -> Skill
async def load_skill_section(self, skill_path: str, section: str) -> str
async def find_relevant_skills(self, query: str) -> list[Skill]
def get_skill_context(self) -> str
```

**Sub-tasks**:
1. Add imports: `from src.skills import SkillLoader, SkillMatcher, Skill`
2. Add `_skill_loader: SkillLoader` instance variable (lazy init)
3. Add `_skill_matcher: SkillMatcher` instance variable (lazy init)
4. Implement `load_skill()` - delegates to SkillLoader
5. Implement `load_skill_section()` - loads specific section only
6. Implement `find_relevant_skills()` - delegates to SkillMatcher
7. Implement `get_skill_context()` - returns loaded skills as formatted string
8. Add `_loaded_skills: list[Skill]` to track loaded skills per invocation

**Workflow Impact**:
- Existing agents unchanged until they call new methods
- BaseAgent interface expands (backward compatible)
- No changes to agent registration or routing

**Dependencies**:
- Import `SkillLoader`, `SkillMatcher` from `src.skills`

---

### 3.2 Integrate with `causal_impact` Agent

**Status**: [x] Complete

**File**: `src/agents/causal_impact/agent.py`

**Integration Points**:
- `build_dag()`: Load `confounder-identification.md` for standard confounders
- `estimate_effect()`: Load `dowhy-workflow.md` for estimation procedures
- `interpret_results()`: Load interpretation templates by audience

**Sub-tasks**:
1. In `build_dag()` method:
   - Call `await self.load_skill_section("causal-inference/confounder-identification.md", "HCP Targeting → Prescription Impact")`
   - Use loaded confounders as checklist for DAG construction
   - Log which confounders were considered
2. In `estimate_effect()` method:
   - Call `await self.load_skill("causal-inference/dowhy-workflow.md")`
   - Follow estimation fallback chain from skill
   - Use skill's refutation test recommendations
3. In `interpret_results()` method:
   - Load audience-specific interpretation templates
   - Apply pharma-specific language from skill

**Code Change Example**:
```python
async def build_dag(self, ...):
    # NEW: Load standard confounders from skill
    confounder_skill = await self.load_skill_section(
        "causal-inference/confounder-identification.md",
        "HCP Targeting → Prescription Impact"
    )
    # Existing DAG building logic continues...
```

**Workflow Impact**:
- Agent gains procedural knowledge from skills
- Existing functionality preserved (skills enhance, don't replace)

---

### 3.3 Integrate with `experiment_designer` Agent

**Status**: [x] Complete

**File**: `src/agents/experiment_designer/agent.py`

**Integration Points**:
- `audit_validity()`: Load `validity-threats.md` for threat taxonomy
- `calculate_power()`: Load `power-analysis.md` for procedures
- Brand context loading for historical benchmarks

**Sub-tasks**:
1. In `audit_validity()` method:
   - Call `await self.load_skill("experiment-design/validity-threats.md")`
   - Apply 6-threat taxonomy systematically
   - Use skill's detection methods and mitigations
   - Calculate validity score using skill's framework
2. In `calculate_power()` method:
   - Load power analysis procedures
   - Use brand-specific effect size benchmarks
3. In design generation:
   - Load brand context for historical benchmarks
   - Apply pharma-specific design patterns

**Code Change Example**:
```python
async def audit_validity(self, experiment_config: dict) -> ValidityReport:
    # NEW: Load validity threat taxonomy
    validity_skill = await self.load_skill("experiment-design/validity-threats.md")
    threats = validity_skill.get_section("6-Threat Taxonomy")
    # Apply each threat check systematically...
```

**Workflow Impact**:
- Standardizes validity assessment across experiments
- Adds pharma-specific threat detection

---

### 3.4 Integrate with `gap_analyzer` Agent

**Status**: [x] Complete

**File**: `src/agents/gap_analyzer/agent.py`

**Integration Points**:
- `estimate_roi()`: Load `roi-estimation.md` for multipliers
- `categorize_opportunity()`: Load opportunity categorization rules
- KPI definitions for gap detection

**Sub-tasks**:
1. In `estimate_roi()` method:
   - Call `await self.load_skill("gap-analysis/roi-estimation.md")`
   - Apply revenue impact multipliers from skill
   - Use cost-to-close formulas
   - Calculate payback period using skill's method
2. In `categorize_opportunity()` method:
   - Load categorization rules (Quick Wins, Strategic Bets, etc.)
   - Apply ROI thresholds from skill
3. In gap detection:
   - Load `pharma-commercial/kpi-calculation.md` for KPI definitions
   - Use standard formulas for gap calculation

**Code Change Example**:
```python
async def estimate_roi(self, gap: Gap) -> ROIEstimate:
    # NEW: Load ROI estimation procedures
    roi_skill = await self.load_skill("gap-analysis/roi-estimation.md")
    multipliers = roi_skill.get_section("Revenue Impact Calculation")
    # Apply multipliers to gap...
```

**Workflow Impact**:
- Standardizes ROI calculations across analyses
- Ensures consistent opportunity categorization

---

### 3.5 Integrate with `explainer` Agent

**Status**: [x] Complete

**File**: `src/agents/explainer/agent.py`

**Integration Points**:
- Brand context loading for explanations
- Interpretation templates by audience level
- Patient journey context for funnel explanations

**Sub-tasks**:
1. In explanation generation:
   - Call `await self.load_skill("pharma-commercial/brand-analytics.md")`
   - Use brand-specific terminology and context
   - Apply audience-appropriate language
2. For causal explanations:
   - Load `causal-inference/dowhy-workflow.md` section "Interpretation by Audience"
   - Use executive vs technical explanation templates
3. For funnel explanations:
   - Load patient journey context
   - Apply pharma-specific funnel terminology

**Code Change Example**:
```python
async def explain_causal_result(self, result: CausalResult, audience: str) -> str:
    # NEW: Load audience-specific templates
    workflow_skill = await self.load_skill("causal-inference/dowhy-workflow.md")
    template = workflow_skill.get_section(f"Interpretation for {audience}")
    # Generate explanation using template...
```

**Workflow Impact**:
- Ensures consistent explanation quality
- Adapts explanations to audience level automatically

---

### 3.6 Integration Tests on Droplet

**Status**: [x] Complete (tests/integration/test_agent_skills.py — 22+ test cases)

**Test Command**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/integration/test_agent_skills.py -v --tb=short -n 2"
```

**Test Cases**:
- CausalImpactAgent loads confounder skills correctly
- ExperimentDesignerAgent uses validity threat taxonomy
- GapAnalyzerAgent applies ROI formulas from skills

---

## Phase 4: MCP Connector Verification ✅ COMPLETE (Result: All Connectors Unavailable)

### Workflow Impact Summary (Phase 4)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `scripts/verify_mcp_connectors.py` | New verification script | **Addition** |
| `.claude/PRPs/mcp-skills-implementation-plan.md` | Update with findings | **Modification** |
| Existing agents | **No changes** | None |
| Existing API | **No changes** | None |

**Breaking Changes**: None - this is a verification phase only.

**Decision Point**: Results determine whether to proceed with Phase 5 (MCP Gateway) or pivot to alternatives.

---

### 4.1-4.4 Verify Connector Availability

**Status**: [x] Complete — All 3 connectors returned 401 auth errors (2026-01-26)

**Critical Question**: Do Anthropic's hosted connectors (ChEMBL, ClinicalTrials.gov, PubMed) actually work via the API?

**Sub-tasks**:
1. Create `scripts/verify_mcp_connectors.py` verification script
2. Test ChEMBL connector with ribociclib query
3. Test ClinicalTrials.gov connector with breast cancer query
4. Test PubMed connector with causal inference query
5. Record response content and latency
6. Determine if responses contain real external data vs Claude's knowledge

**Verification Script**:
```python
import anthropic
import time

client = anthropic.Anthropic()

connectors = [
    ("ChEMBL", "Using the ChEMBL connector, search for ribociclib (Kisqali) and return its molecular properties"),
    ("ClinicalTrials.gov", "Using the ClinicalTrials.gov connector, find active trials for HR+/HER2- breast cancer"),
    ("PubMed", "Using the PubMed connector, search for recent papers on causal inference in pharmaceutical marketing"),
]

results = {}
for name, prompt in connectors:
    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start
    results[name] = {
        "response": response.content[0].text,
        "latency_ms": elapsed * 1000,
        "has_external_data": "VERIFY_MANUALLY"  # Check if response has real data
    }

# Output results for manual verification
for name, data in results.items():
    print(f"\n=== {name} ===")
    print(f"Latency: {data['latency_ms']:.0f}ms")
    print(f"Response preview: {data['response'][:500]}...")
```

**Expected Outcomes**:
1. **Connectors work** → Proceed to Phase 5
2. **Connectors don't work** → Document findings, consider alternatives:
   - Direct API integration (ChEMBL REST API, ClinicalTrials.gov API)
   - ToolUniverse or similar MCP server hosting
   - Skills-only approach (no external data enrichment)

### 4.5 Document Connector Access Status

**Status**: [ ] Partially Complete — Results in `scripts/mcp_connector_results.json`, PRP not yet updated

**Sub-tasks**:
1. Run verification script on droplet
2. Analyze responses for external data indicators
3. Update `.claude/PRPs/mcp-skills-implementation-plan.md` with findings
4. If connectors unavailable, document alternative approaches
5. Update this plan's Phase 5 status accordingly

**Documentation Template**:
```markdown
## MCP Connector Verification Results

**Date**: YYYY-MM-DD
**Environment**: Production droplet (138.197.4.36)

### ChEMBL Connector
- **Status**: ✅ Working / ❌ Not Available
- **Evidence**: [describe response]
- **Latency**: Xms

### ClinicalTrials.gov Connector
- **Status**: ✅ Working / ❌ Not Available
- **Evidence**: [describe response]
- **Latency**: Xms

### PubMed Connector
- **Status**: ✅ Working / ❌ Not Available
- **Evidence**: [describe response]
- **Latency**: Xms

### Decision
- [ ] Proceed to Phase 5 (MCP Gateway)
- [ ] Pivot to direct API integration
- [ ] Skills-only approach (defer MCP)
```

**Workflow Impact**: Determines whether Phase 5-6 proceed as planned

---

## Phase 5: MCP Gateway Implementation ⏸️ DEFERRED

> **Note**: This phase is CONDITIONAL on Phase 4 results. Phase 4 showed all connectors unavailable (401 auth errors on 2026-01-26). **This phase is deferred** until MCP connector access is resolved.

### Workflow Impact Summary (Phase 5)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `src/mcp/__init__.py` | New module | **Addition** |
| `src/mcp/connector_gateway.py` | New gateway class | **Addition** |
| `src/mcp/connectors/*.py` | Connector implementations | **Addition** |
| `config/mcp_config.yaml` | New configuration | **Addition** |
| `tests/unit/test_mcp/` | New test directory | **Addition** |
| Existing agents | **No changes yet** | None |

**Breaking Changes**: None - new module only

**Prerequisite**: Phase 4 verification shows connectors are available

---

### 5.1 Create MCP Module Structure

**Status**: [ ] Deferred (connectors unavailable)

**Files to Create**:
- `src/mcp/__init__.py`
- `src/mcp/connector_gateway.py`
- `src/mcp/connectors/__init__.py`
- `src/mcp/connectors/chembl.py`
- `src/mcp/connectors/clinicaltrials.py`
- `src/mcp/connectors/pubmed.py`

**Sub-tasks**:
1. Create `src/mcp/` directory structure
2. Create `__init__.py` with exports (`ConnectorGateway`, connector classes)
3. Define `ConnectorGateway` base class with:
   - `query(connector_name: str, query: str) -> ConnectorResult`
   - `list_available_connectors() -> list[str]`
   - Rate limiting and caching

### 5.2 Implement ConnectorGateway

**Status**: [ ] Deferred (connectors unavailable)

**Sub-tasks**:
1. Implement `ConnectorGateway` class with Anthropic client
2. Add connector dispatch logic
3. Implement rate limiting (respect API limits)
4. Add response caching (configurable TTL)
5. Add error handling with graceful fallbacks
6. Log all connector calls for observability

**Code Pattern**:
```python
class ConnectorGateway:
    def __init__(self, config: MCPConfig):
        self.client = anthropic.Anthropic()
        self.config = config
        self._cache = TTLCache(maxsize=100, ttl=config.cache_ttl)

    async def query(self, connector: str, query: str) -> ConnectorResult:
        cache_key = f"{connector}:{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Make API call with connector
        result = await self._call_connector(connector, query)
        self._cache[cache_key] = result
        return result
```

### 5.3 Implement Connector-Specific Helpers

**Status**: [ ] Deferred (connectors unavailable)

**Sub-tasks**:
1. `chembl.py`: ChEMBL-specific query formatting and response parsing
2. `clinicaltrials.py`: ClinicalTrials.gov query formatting
3. `pubmed.py`: PubMed search formatting and citation parsing
4. Each connector implements `ConnectorInterface` protocol

### 5.4 Unit Tests for Gateway

**Status**: [ ] Deferred (connectors unavailable)

**Files to Create**:
- `tests/unit/test_mcp/test_connector_gateway.py`
- `tests/unit/test_mcp/test_connectors.py`

**Test Cases**:
- `test_gateway_initialization`
- `test_connector_dispatch`
- `test_rate_limiting`
- `test_caching_behavior`
- `test_error_handling`

### 5.5 Droplet Deployment Test

**Status**: [ ] Deferred (connectors unavailable)

**Test Command**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_mcp/ -v --tb=short"
```

---

## Phase 6: Full Integration ⏸️ PARTIALLY PENDING (Skills-Only Path)

### Workflow Impact Summary (Phase 6)

| Component | Impact | Change Type |
|-----------|--------|-------------|
| `src/agents/tool_composer/agent.py` | Add MCP orchestration | **Modification** |
| `config/mcp_config.yaml` | New configuration file | **Addition** |
| `tests/integration/test_skills_mcp_e2e.py` | End-to-end tests | **Addition** |
| `src/monitoring/skills_metrics.py` | Performance monitoring | **Addition** |
| API endpoints | **No changes** | None |

**Breaking Changes**: None - existing functionality preserved

**Prerequisites**: Phases 1-3 complete (Skills), Phase 5 complete (MCP, if available)

---

### 6.1 Update Tool Composer with Skills Orchestration

**Status**: [ ] Pending (skills-only path — MCP deferred)

**File**: `src/agents/tool_composer/agent.py`

**Sub-tasks**:
1. Add MCP gateway import and initialization
2. Update `compose_tools()` to include MCP connectors
3. Implement intelligent tool selection (skills vs MCP vs both)
4. Add fallback logic when MCP unavailable

**Integration Pattern**:
```python
class ToolComposer:
    def __init__(self, ...):
        self.skill_loader = SkillLoader()
        self.mcp_gateway = ConnectorGateway(config) if MCP_ENABLED else None

    async def compose_for_query(self, query: str) -> ComposedToolset:
        # 1. Find relevant skills
        skills = await self.find_relevant_skills(query)

        # 2. Determine if external data needed
        if self._needs_external_data(query) and self.mcp_gateway:
            connectors = self._select_connectors(query)
        else:
            connectors = []

        return ComposedToolset(skills=skills, connectors=connectors)
```

### 6.2 Create MCP Configuration

**Status**: [ ] Deferred (MCP connectors unavailable)

**File**: `config/mcp_config.yaml`

**Sub-tasks**:
1. Create configuration file with:
   - Connector enablement flags
   - Rate limiting settings
   - Cache TTL settings
   - Timeout configurations
2. Add environment variable overrides
3. Document all configuration options

**Configuration Template**:
```yaml
mcp:
  enabled: true
  cache_ttl_seconds: 300
  rate_limit:
    requests_per_minute: 60
    burst_size: 10

  connectors:
    chembl:
      enabled: true
      timeout_seconds: 30
    clinicaltrials:
      enabled: true
      timeout_seconds: 30
    pubmed:
      enabled: true
      timeout_seconds: 30

skills:
  enabled: true
  base_path: ".claude/skills"
  cache_enabled: true
```

### 6.3 End-to-End Integration Tests

**Status**: [ ] Pending (skills-only e2e tests)

**File**: `tests/integration/test_skills_mcp_e2e.py`

**Test Scenarios**:
1. Query requiring skills only → Skills loaded, no MCP calls
2. Query requiring MCP only → Connector called, no skills loaded
3. Query requiring both → Skills + MCP combined
4. MCP failure fallback → Graceful degradation to skills-only
5. Full agent pipeline → Orchestrator → Agent → Skills/MCP → Response

**Test Command**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/integration/test_skills_mcp_e2e.py -v --tb=short -n 2"
```

### 6.4 Performance Monitoring Setup

**Status**: [ ] Pending (skill load time + match accuracy metrics)

**File**: `src/monitoring/skills_metrics.py`

**Sub-tasks**:
1. Create metrics collection for:
   - Skill load times
   - Skill match accuracy
   - MCP connector latency
   - Cache hit rates
2. Integrate with existing Opik tracing
3. Add Prometheus metrics endpoints
4. Create Grafana dashboard (optional)

**Metrics to Track**:
| Metric | Type | Target |
|--------|------|--------|
| `skill_load_duration_ms` | Histogram | p99 < 50ms |
| `skill_match_accuracy` | Gauge | > 80% |
| `mcp_connector_latency_ms` | Histogram | p99 < 2000ms |
| `skill_cache_hit_rate` | Gauge | > 70% |
| `mcp_cache_hit_rate` | Gauge | > 50% |

---

## Testing Strategy

### Unit Tests (Run First)
```bash
# Local - quick validation
pytest tests/unit/test_skills/ -v -n 4
```

### Integration Tests (Run on Droplet)
```bash
# Small batches to avoid memory issues
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/integration/test_agent_skills.py -v -n 2 --tb=short"
```

### Memory-Safe Execution
- Max 4 workers for unit tests
- Max 2 workers for integration tests
- Use `--dist=loadscope` to group tests

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MCP connectors unavailable | Medium | Low | Skills provide core value without MCP |
| Skill content errors | Low | Medium | Review process, validation tests |
| Token budget exceeded | Low | Low | Section-based loading, strict budgets |
| Agent integration breaks | Low | High | Feature flags, gradual rollout |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Skill load time | < 50ms | Profiling |
| Skill match accuracy | > 80% | Manual evaluation |
| Agent context efficiency | < 3000 tokens/skill | Token counting |
| All unit tests pass | 100% | CI/CD |

---

## Timeline Estimate

| Phase | Duration | Dependency |
|-------|----------|------------|
| Phase 1: Skills Framework | 2-3 days | None |
| Phase 2: Skills Content | 2-3 days | Phase 1 |
| Phase 3: Agent Integration | 3-4 days | Phase 2 |
| Phase 4: MCP Verification | 1 day | None (parallel) |
| Phase 5: MCP Gateway | 2-3 days | Phase 4 (if positive) |
| Phase 6: Full Integration | 2-3 days | Phases 3, 5 |

**Total**: ~2 weeks (skills-only: ~1 week)

---

## Workflow Impact Summary (All Phases)

### Files Changed by Phase

| Phase | New Files | Modified Files | Breaking Changes |
|-------|-----------|----------------|------------------|
| **Phase 1** | 4 files (`src/skills/*.py`) | None | None |
| **Phase 2** | 11 skill files (`.claude/skills/**/*.md`) | None | None |
| **Phase 3** | 1 test file | 5 agent files | None (additive) |
| **Phase 4** | 1 script | 1 PRP doc | None |
| **Phase 5** | 6 files (`src/mcp/*.py`) | None | None |
| **Phase 6** | 3 files | 1 agent file | None |

### Existing File Modifications Detail

**Phase 3 (Agent Integration)**:
```
src/agents/base.py                    → Add 4 skill methods
src/agents/causal_impact/agent.py     → Add skill loading calls
src/agents/experiment_designer/agent.py → Add skill loading calls
src/agents/gap_analyzer/agent.py      → Add skill loading calls
src/agents/explainer/agent.py         → Add skill loading calls
```

**Phase 6 (Full Integration)**:
```
src/agents/tool_composer/agent.py     → Add MCP orchestration
```

### Execution Order and Dependencies

```
Phase 1 (Skills Framework) ──────────────────────────────────┐
         │                                                    │
         ▼                                                    │
Phase 2 (Skills Content) ────────────────────────────────────┤
         │                                                    │
         ▼                                                    │
Phase 3 (Agent Integration) ─────────────────────────────────┤
         │                                                    │
         │    Phase 4 (MCP Verification) ◄── Run in parallel │
         │              │                                     │
         │              ▼                                     │
         │    [Decision Point]                                │
         │        │         │                                 │
         │        ▼         ▼                                 │
         │    Phase 5   Skip Phase 5                          │
         │    (MCP)     (if unavailable)                      │
         │        │         │                                 │
         └────────┴─────────┴─────────────────────────────────┘
                           │
                           ▼
                  Phase 6 (Full Integration)
```

---

## Next Steps

### Actionable Items (3 remaining)

1. **4.5 — Document connector status in PRP**: Update `.claude/PRPs/mcp-skills-implementation-plan.md` with Phase 4 verification results (all connectors returned 401 auth errors). Record decision to defer MCP.

2. **6.1 — Tool Composer skills-only integration**: Update `src/agents/tool_composer/agent.py` to use SkillsMixin for skill-based orchestration (without MCP). This is unblocked now that Phases 1-3 are complete.

3. **6.3 — Skills-only e2e tests**: Create `tests/integration/test_skills_e2e.py` covering the full pipeline: Orchestrator → Agent → Skills → Response.

4. **6.4 — Skills performance monitoring**: Create `src/monitoring/skills_metrics.py` with skill load time histograms, match accuracy gauges, and cache hit rate tracking. Integrate with Opik.

### Deferred Items (revisit if MCP access resolves)

- Phase 5 (entire) — MCP Gateway implementation
- 6.2 — MCP config YAML
- Re-run `scripts/verify_mcp_connectors.py` with valid API key if one becomes available

---

## Session Log

### 2025-01-26 - Plan Created
- Analyzed three source documents
- Determined Skills-first approach minimizes friction
- Created phased implementation plan with small, testable chunks
- Prioritized zero-infrastructure Skills Framework
- Made MCP Connectors conditional on verification

### 2025-01-26 - Plan Expanded
- Added detailed sub-tasks for each phase item
- Added workflow impact summaries per phase
- Documented files created vs modified
- Added code examples for agent integration
- Created execution order diagram
- Confirmed no breaking changes in any phase

### 2025-01-26 - Phase 1 Complete ✅
- Created skills directory structure (`.claude/skills/` with 4 categories)
- Implemented `src/skills/loader.py` (SkillLoader, SkillMetadata, Skill classes)
- Implemented `src/skills/matcher.py` (SkillMatcher with keyword scoring)
- Created master SKILL.md index and category indices
- Created domain skill files:
  - `pharma-commercial/kpi-calculation.md` (full KPI procedures)
  - `causal-inference/confounder-identification.md` (standard confounders)
- Created unit tests: `tests/unit/test_skills/` (34 tests)
- Deployed to droplet and verified: **34/34 tests passed**

### Post-Phase 1 — Phases 2-4 Completed (date unrecorded in plan)
- **Phase 2**: All domain skill content created (pharma-commercial, causal-inference, experiment-design, gap-analysis) including `opportunity-sizing.md` and `patient-journey.md` beyond original scope
- **Phase 3**: Agent integration via `SkillsMixin` pattern (316 lines) — cleaner than original `base.py` modification plan. Integration tests created (22+ test cases) for causal_impact, experiment_designer, gap_analyzer, explainer agents.
- **Phase 4**: MCP connector verification completed. All 3 connectors (ChEMBL, ClinicalTrials.gov, PubMed) returned **401 authentication errors**. Results saved to `scripts/mcp_connector_results.json`. Decision: defer MCP gateway until access resolves.

### 2026-02-01 - Plan Review & Status Update
- Updated all phase statuses to reflect actual implementation state
- Marked Phase 5 as DEFERRED (connectors unavailable)
- Scoped Phase 6 to skills-only path (3 actionable items remain)
- Identified 1 documentation gap: PRP not updated with Phase 4 findings (item 4.5)
