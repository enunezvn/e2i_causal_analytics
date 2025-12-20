# E2I Dashboard Gap Resolution - Granular TODO List

**Status**: COMPLETE
**Last Updated**: 2025-12-15
**Total Gaps**: 14 (6 Critical, 8 Moderate)

---

## PROGRESS TRACKER

| # | Gap | Status | Artifact Created | Notes |
|---|-----|--------|------------------|-------|
| 1 | Filter → Query Flow | ✅ DONE | config/filter_mapping.yaml, docs/filter_query_flow.md | Complete flow with NLP integration |
| 2 | Tab → Agent Routing | ✅ DONE | agent_config.yaml (updated) | Tab routing, cache, prefetch added |
| 3 | Agent → Viz Mapping | ✅ DONE | visualization_rules_v4_1.yaml | Complete mapping created |
| 4 | Tier 0 UI Visibility | ✅ DONE | E2I_Causal_Dashboard_V3.html | ML Foundation tab added |
| 5 | Validation Badges | ✅ DONE | E2I_Causal_Dashboard_V3.html | proceed/review/block badges |
| 6 | Chat Interface | ✅ DONE | E2I_Causal_Dashboard_V3.html | Chat panel with streaming |
| 7 | Data Sources Schema | ✅ DONE | e2i_mlops/012_data_sources.sql | Schema + views + 7 sources seeded |
| 8 | Brand KPIs | ✅ DONE | kpi_definitions.yaml (existing) | BR-001 to BR-005 exist |
| 9 | ROI Calculation | ✅ DONE | docs/roi_methodology.md | Complete methodology with formulas |
| 10 | Time Granularity | ✅ DONE | domain_vocabulary_v3.1.0.yaml (updated) | Fiscal calendar, aggregation rules |
| 11 | Alert System | ✅ DONE | config/alert_config.yaml | 25+ alerts, routing, suppression |
| 12 | Confidence Logic | ✅ DONE | config/confidence_logic.yaml, docs/confidence_methodology.md | 4-component confidence scoring |
| 13 | Experiment Lifecycle | ✅ DONE | config/experiment_lifecycle.yaml | State machine, interim analysis, stopping rules |
| 14 | Viz Library Clarification | ✅ DONE | visualization_rules_v4_1.yaml | Library selection rules added |

**Progress**: 14/14 complete (100%) ✅

---

## DETAILED TASK BREAKDOWN

### 🔴 CRITICAL: Gap 1 - Filter → Query Flow

**Goal**: Map dashboard filter controls to NLP query pipeline

**Tasks**:
- [ ] 1.1 Define FilterState interface in TypeScript
  ```typescript
  interface FilterState {
    brand: string | null;
    region: string | null;
    startDate: string;
    endDate: string;
  }
  ```
- [ ] 1.2 Create filter injection logic in `query_processor.py`
  - If user typed query: inject filters into ExtractedEntities
  - If no query (just filter change): generate SQL WHERE clause
- [ ] 1.3 Add filter_injection section to domain_vocabulary.yaml
- [ ] 1.4 Update dashboard to dispatch filter changes to backend

**Files to Update**:
- `src/nlp/query_processor.py`
- `config/domain_vocabulary.yaml`
- `frontend/src/store/filterSlice.ts`
- `frontend/src/api/queries.ts`

**Estimated Effort**: 4-6 hours

---

### 🔴 CRITICAL: Gap 2 - Tab → Agent Routing

**Goal**: Define which agents serve data for each dashboard tab

**Tasks**:
- [ ] 2.1 Add tab_agent_mapping to agent_config.yaml (see visualization_rules_v4_1.yaml)
- [ ] 2.2 Create tab router middleware in orchestrator
- [ ] 2.3 Implement tab-specific cache strategies
- [ ] 2.4 Add prefetch logic for adjacent tabs

**Files to Update**:
- `config/agent_config.yaml`
- `src/agents/orchestrator/router.py`

**Estimated Effort**: 2-3 hours

---

### ✅ COMPLETE: Gap 3 - Agent → Viz Mapping

**Artifact**: `visualization_rules_v4_1.yaml`

**What was created**:
- Library selection rules (Chart.js vs Plotly vs D3.js)
- Complete Tab → Agent → Visualization mapping for all 10 tabs
- Agent output schemas with required fields
- Filter injection rules
- Validation Badge component spec
- Cache/refresh strategies per tab

---

### 🔴 CRITICAL: Gap 4 - Tier 0 UI Visibility

**Goal**: Add ML Foundation visibility to dashboard

**Tasks**:
- [ ] 4.1 Design "🔧 ML Foundation" tab layout
- [ ] 4.2 Create components:
  - ScopeDefinitionStatus
  - QCGateIndicator (pass/block)
  - ModelTrainingProgress
  - FeatureImportanceChart
  - DeploymentStatusTable
  - ObservabilityMetrics
- [ ] 4.3 Wire components to Tier 0 agents
- [ ] 4.4 Add to tab routing

**Files to Create**:
- `frontend/src/components/MLFoundation/`
- `frontend/src/pages/MLFoundationTab.tsx`

**Estimated Effort**: 6-8 hours

---

### 🔴 CRITICAL: Gap 5 - Validation Badges

**Goal**: Add V4.1 validation status to causal insights

**Tasks**:
- [ ] 5.1 Create ValidationBadge React component
  ```tsx
  <ValidationBadge 
    gateDecision="proceed|review|block"
    testsPassed={4}
    testsTotal={5}
    confidence={0.87}
    eValue={2.3}
  />
  ```
- [ ] 5.2 Style according to visualization_rules_v4_1.yaml spec
- [ ] 5.3 Add expand-on-click for test details
- [ ] 5.4 Integrate into CausalInsightCard component
- [ ] 5.5 Wire to causal_validations API endpoint

**Files to Create**:
- `frontend/src/components/ValidationBadge.tsx`
- `frontend/src/components/ValidationBadge.css`

**Estimated Effort**: 3-4 hours

---

### 🔴 CRITICAL: Gap 6 - Chat Interface

**Goal**: Add NLP query input to dashboard

**Tasks**:
- [ ] 6.1 Design chat UI (sidebar or modal)
- [ ] 6.2 Create ChatInput component with streaming support
- [ ] 6.3 Implement WebSocket connection for streaming responses
- [ ] 6.4 Create AgentBadge component (tier-colored)
- [ ] 6.5 Add chat history panel
- [ ] 6.6 Integrate with existing filter state (inject filters into queries)
- [ ] 6.7 Add inline citations to causal_paths table

**Files to Create**:
- `frontend/src/components/Chat/`
  - `ChatPanel.tsx`
  - `ChatInput.tsx`
  - `ChatMessage.tsx`
  - `AgentBadge.tsx`
  - `StreamingIndicator.tsx`
- `frontend/src/hooks/useWebSocket.ts`

**Estimated Effort**: 8-12 hours

---

### 🟡 MODERATE: Gap 7 - Data Sources Schema

**Goal**: Formalize data source tracking

**Tasks**:
- [ ] 7.1 Create data_sources reference table
  ```sql
  CREATE TABLE data_sources (
      source_id TEXT PRIMARY KEY,
      source_name TEXT NOT NULL,
      source_type TEXT,
      coverage_percent DECIMAL(5,2),
      completeness_score DECIMAL(5,4),
      freshness_days INTEGER,
      match_rate DECIMAL(5,4)
  );
  ```
- [ ] 7.2 Seed with IQVIA, HealthVerity, Komodo, Veeva
- [ ] 7.3 Update data_source_tracking FK reference

**Files to Update**:
- `supabase/migrations/011_data_sources.sql`

**Estimated Effort**: 1-2 hours

---

### ✅ COMPLETE: Gap 8 - Brand KPIs

**Existing Coverage in kpi_definitions.yaml**:
- BR-001: Remi - AH Uncontrolled %
- BR-002: Remi - Intent-to-Prescribe Δ
- BR-003: Fabhalta - % PNH Tested
- BR-004: Kisqali - Dx Adoption
- BR-005: Kisqali - Oncologist Reach

**Note**: Dashboard mock shows slightly different KPI names. May need UI label mapping.

---

### 🟡 MODERATE: Gap 9 - ROI Calculation

**Goal**: Document ROI methodology

**Tasks**:
- [ ] 9.1 Create ROI methodology document
- [ ] 9.2 Add value_drivers to gap_analyzer agent config:
  ```yaml
  roi_calculation:
    trx_lift_value: 850
    patient_identification_value: 1200
    action_rate_value: 45
  ```
- [ ] 9.3 Add confidence intervals (bootstrap sampling)

**Files to Create**:
- `docs/roi_methodology.md`
- Update `config/agent_config.yaml` (gap_analyzer section)

**Estimated Effort**: 2-3 hours

---

### 🟡 MODERATE: Gap 10 - Time Granularity

**Goal**: Specify aggregation rules

**Tasks**:
- [ ] 10.1 Add time_granularity section to domain_vocabulary.yaml
- [ ] 10.2 Define fiscal quarter mapping
- [ ] 10.3 Add business calendar configuration

**Estimated Effort**: 1 hour

---

### 🟡 MODERATE: Gap 11 - Alert System

**Goal**: Define alert generation and delivery

**Tasks**:
- [ ] 11.1 Create alert_config.yaml
- [ ] 11.2 Define alert priority levels and triggers
- [ ] 11.3 Add notification delivery options
- [ ] 11.4 Create AlertBanner component

**Estimated Effort**: 3-4 hours

---

### 🟡 MODERATE: Gap 12 - Confidence Logic

**Goal**: Specify confidence calculation

**Tasks**:
- [ ] 12.1 Document confidence formula
- [ ] 12.2 Define heterogeneous effects → confidence reduction rules
- [ ] 12.3 Add to thresholds.yaml

**Estimated Effort**: 1-2 hours

---

### 🟡 MODERATE: Gap 13 - Experiment Lifecycle

**Goal**: Define experiment status workflow

**Tasks**:
- [ ] 13.1 Create experiment_lifecycle.yaml:
  ```yaml
  states: [proposed, approved, active, completed, archived]
  transitions:
    proposed → approved: requires_power_analysis
    active → completed: requires_min_duration
  ```
- [ ] 13.2 Add interim analysis rules
- [ ] 13.3 Define early stopping criteria

**Estimated Effort**: 2 hours

---

### ✅ COMPLETE: Gap 14 - Viz Library Clarification

**Resolved in visualization_rules_v4_1.yaml**:
- Chart.js: Simple bar, line, doughnut, sparklines
- Plotly: Sankey, heatmap, radar, forest plots
- D3.js: Causal DAG (force-directed)
- CSS: Causal chains, badges, status indicators

---

## RECOMMENDED WORK ORDER

### Phase 1: Core Infrastructure (Gaps 1, 2)
1. Filter → Query Flow
2. Tab → Agent Routing

### Phase 2: V4.1 Features (Gaps 5, 6)
3. Validation Badges
4. Chat Interface

### Phase 3: Dashboard Enhancement (Gap 4)
5. Tier 0 UI Visibility

### Phase 4: Polish (Gaps 7, 9-13)
6. Data Sources Schema
7. ROI Calculation
8. Time Granularity
9. Alert System
10. Confidence Logic
11. Experiment Lifecycle

---

## SESSION CONTINUITY NOTES

**Files Created This Session**:
1. `e2i_query_flow_diagram.jsx` - React component for flow visualization
2. `e2i_query_flow_documentation.md` - Detailed Mermaid sequence + data handoff table
3. `e2i_query_flow.mermaid` - Standalone Mermaid diagram
4. `e2i_gap_analysis.md` - Full gap analysis with 14 gaps
5. `visualization_rules_v4_1.yaml` - Agent → Viz mapping (addresses Gap 3, 14)
6. `e2i_gap_todo.md` - This file

**Memory Updated**:
- Entry #9: E2I GAP TODO summary
- Entry #13: E2I GAP WORK tracking

**To Resume Work**:
1. Reference this TODO file
2. Check memory for current status
3. Pick next gap from priority list
4. Create/update artifacts
5. Update progress tracker above

---

*Last updated: 2025-12-14 | E2I Causal Analytics V4.1*
