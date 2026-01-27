# Skills & MCP Implementation Plan

**Created**: 2025-01-26
**Status**: In Progress
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
- [x] **2.1** Create pharma-commercial skills (kpi-calculation, brand-analytics)
- [x] **2.2** Create causal-inference skills (confounder-identification, dowhy-workflow)
- [x] **2.3** Create experiment-design skills (validity-threats, power-analysis)
- [x] **2.4** Create gap-analysis skills (roi-estimation)
- [x] **2.5** Skill content validation tests

### Phase 3: Agent-Skill Integration ✅ COMPLETE
- [x] **3.1** Update BaseAgent with skill methods (created SkillsMixin)
- [x] **3.2** Integrate skills with `causal_impact` agent
- [x] **3.3** Integrate skills with `experiment_designer` agent
- [x] **3.4** Integrate skills with `gap_analyzer` agent
- [x] **3.5** Integrate skills with `explainer` agent
- [x] **3.6** Integration tests on droplet

### Phase 4: MCP Connector Verification ✅ COMPLETE
- [x] **4.1** Verify Anthropic connector API access
- [x] **4.2** Test ChEMBL connector availability
- [x] **4.3** Test ClinicalTrials.gov connector availability
- [x] **4.4** Test PubMed connector availability
- [x] **4.5** Document connector access status

**Result**: Pharmaceutical connectors (ChEMBL, ClinicalTrials.gov, PubMed) are NOT available as Anthropic-hosted MCP servers. These would require custom implementation. **Recommendation: Proceed with Skills-only approach** (Phase 5-6 deferred).

### Phase 5: MCP Gateway Implementation ⏸️ DEFERRED
- [ ] **5.1** Create MCP connector gateway
- [ ] **5.2** Implement permission checking
- [ ] **5.3** Create connector-specific helpers
- [ ] **5.4** Unit tests for gateway
- [ ] **5.5** Droplet deployment test

**Status**: Deferred - pharmaceutical MCP connectors not available (see Phase 4 results)

### Phase 6: Full Integration ⏸️ DEFERRED
- [ ] **6.1** Update `tool_composer` with MCP orchestration
- [ ] **6.2** Create MCP config (config/mcp_config.yaml)
- [ ] **6.3** End-to-end integration tests
- [ ] **6.4** Performance monitoring setup

**Status**: Deferred - depends on Phase 5

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

**File**: `.claude/skills/SKILL.md`

**Content**: YAML frontmatter with:
- name: E2I Pharma Commercial Analytics Skills
- version: 1.0
- categories list
- triggers list
- When to load guidance

---

### 1.5 Unit Tests for SkillLoader

**Status**: [ ] Not Started

**File**: `tests/unit/test_skills/test_loader.py`

**Test Cases**:
- `test_load_skill_with_frontmatter`
- `test_skill_caching`
- `test_section_extraction`
- `test_skill_not_found`
- `test_trigger_matching`

---

### 1.6 Droplet Deployment Test

**Status**: [ ] Not Started

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

## Phase 2: Domain Skills Content

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

**File**: `tests/unit/test_skills/test_skill_content.py`

**Test Cases**:
- All skill files parse without errors
- All required sections exist
- Triggers are properly defined
- Agent mappings are correct

---

## Phase 3: Agent-Skill Integration

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

### 3.1 Update BaseAgent with Skill Methods

**Status**: [ ] Not Started

**File**: `src/agents/base.py` (modify existing)

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

## Phase 4: MCP Connector Verification

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

**Status**: [ ] Not Started

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

**Status**: [ ] Not Started

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

## MCP Connector Verification Results

**Date**: 2026-01-26
**Environment**: Local development (WSL2)

### Key Finding: MCP Architecture Misconception

The original plan assumed ChEMBL, ClinicalTrials.gov, and PubMed were available as "Anthropic-hosted MCP connectors" that could be accessed via the messages API. This assumption was **incorrect**.

**How MCP Actually Works**:
- MCP (Model Context Protocol) servers are **separate services** that provide tools to Claude instances
- MCP servers expose tools through a standardized protocol, NOT through prompt-based requests
- Available Anthropic/community MCP servers include: GitHub, Supabase, Brave Search, Google Maps, etc.
- **Pharmaceutical data connectors (ChEMBL, ClinicalTrials.gov, PubMed) are NOT part of Anthropic's standard MCP offerings**

### ChEMBL Connector
- **Status**: ❌ Not Available as MCP Server
- **Evidence**: No MCP server exists; would require custom implementation via ChEMBL REST API
- **Alternative**: Direct API integration (https://www.ebi.ac.uk/chembl/api/data/)

### ClinicalTrials.gov Connector
- **Status**: ❌ Not Available as MCP Server
- **Evidence**: No MCP server exists; would require custom implementation
- **Alternative**: Direct API integration (https://clinicaltrials.gov/api/)

### PubMed Connector
- **Status**: ❌ Not Available as MCP Server
- **Evidence**: No MCP server exists; would require custom implementation via NCBI E-utilities
- **Alternative**: Direct API integration (https://www.ncbi.nlm.nih.gov/books/NBK25501/)

### Decision

**Selected**: ✅ **Skills-only approach (defer MCP)**

**Rationale**:
1. Skills Framework (Phases 1-3) provides immediate value with zero infrastructure
2. Building custom MCP servers for pharmaceutical data requires significant effort
3. Direct API integration can be added later as a separate feature
4. The E2I system's core value proposition (causal analytics) doesn't require real-time external data enrichment

**Deferred Work**:
- Phase 5 (MCP Gateway Implementation) - Deferred
- Phase 6 (Full MCP Integration) - Deferred
- May revisit when/if Anthropic adds pharmaceutical data MCP servers

### Verification Script

Created `scripts/verify_mcp_connectors.py` for documentation purposes. The script confirmed the API approach doesn't work (as expected - MCP is tool-based, not prompt-based).

---

## Phase 5: MCP Gateway Implementation

> **Note**: This phase is CONDITIONAL on Phase 4 results. If connectors unavailable, skip to alternative approach documentation.

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

**Status**: [ ] Blocked (awaiting Phase 4)

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

**Status**: [ ] Blocked (awaiting Phase 4)

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

**Status**: [ ] Blocked (awaiting Phase 4)

**Sub-tasks**:
1. `chembl.py`: ChEMBL-specific query formatting and response parsing
2. `clinicaltrials.py`: ClinicalTrials.gov query formatting
3. `pubmed.py`: PubMed search formatting and citation parsing
4. Each connector implements `ConnectorInterface` protocol

### 5.4 Unit Tests for Gateway

**Status**: [ ] Blocked (awaiting Phase 4)

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

**Status**: [ ] Blocked (awaiting Phase 4)

**Test Command**:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && \
  /opt/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_mcp/ -v --tb=short"
```

---

## Phase 6: Full Integration

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

### 6.1 Update Tool Composer with MCP Orchestration

**Status**: [ ] Blocked (awaiting earlier phases)

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

**Status**: [ ] Blocked (awaiting earlier phases)

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

**Status**: [ ] Blocked (awaiting earlier phases)

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

**Status**: [ ] Blocked (awaiting earlier phases)

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

1. **Start Phase 1.1**: Create skills directory structure
2. **Parallel**: Run MCP verification (Phase 4.1) to unblock/confirm Phase 5
3. **Focus on Skills First**: They provide immediate value with zero dependencies

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

### 2026-01-26 - Phase 4 Complete ✅
- Created verification script: `scripts/verify_mcp_connectors.py`
- Discovered critical architecture misconception:
  - MCP servers are separate services providing tools, NOT prompt-accessible connectors
  - Pharmaceutical data sources (ChEMBL, ClinicalTrials.gov, PubMed) are NOT available as Anthropic MCP servers
- Documented findings with alternatives (direct API integration)
- **Decision**: Proceed with Skills-only approach, defer MCP phases (5-6)
- Rationale: Skills Framework provides core value without external data dependencies

### 2026-01-26 - Workflow Integration Tests Created ✅
- Created `tests/integration/test_agent_workflow_skills.py` with 16 comprehensive tests
- Tests validate skills are properly loaded during actual agent workflow execution:
  - `TestCausalImpactWorkflowSkills` (5 tests): Core skill loading, brand-specific skills, clearing between runs
  - `TestGapAnalyzerWorkflowSkills` (3 tests): ROI estimation skill loading, brand conditional loading
  - `TestExplainerWorkflowSkills` (4 tests): Brand context loading, causal result skill loading
  - `TestSkillWorkflowRobustness` (2 tests): Graceful handling when skill loading fails
  - `TestSkillGuidanceMethod` (2 tests): Skill guidance method availability
- Tests use graph mocking to verify skill state without making LLM API calls
- Fixed skill name assertions to match actual skill frontmatter:
  - "Brand-Specific Analytics" (not "Brand Analytics Context")
  - "Confounder Identification for Pharma Analytics" (not "Confounder Identification Procedures")
- All 31 tests pass (16 new workflow tests + 15 existing skill tests)
- Run command: `pytest tests/integration/test_agent_skills.py tests/integration/test_agent_workflow_skills.py -v`
