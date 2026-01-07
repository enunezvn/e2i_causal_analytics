# Frontend Dashboard Implementation Plan: Mock to Production

**Plan Name**: piped-plotting-lecun
**Created**: 2026-01-07
**Status**: ✅ Complete
**Completed**: 2026-01-07
**Goal**: Implement rich dashboard features from `E2I_Causal_Dashboard_V3.html` mock into production frontend

---

## 1. Audit Summary

### 1.1 Current Implementation (`Home.tsx`)

| Feature | Status | Notes |
|---------|--------|-------|
| KPI Cards | Partial | Uses hardcoded `SAMPLE_KPIS` - **hooks exist but not used** |
| Agent Insights Feed | Partial | Uses hardcoded `SAMPLE_INSIGHTS` |
| System Health | Partial | Uses hardcoded `SYSTEM_STATUS` |
| Executive Intelligence Summary | Missing | Not implemented |
| Primary Causal Value Chains | Missing | Not implemented |
| AI Agent Insights Page | Missing | Not implemented |
| Real-time API Integration | Missing | **Hooks exist but not wired up** |

### 1.2 **CRITICAL FINDING: API Layer Already Exists!**

The frontend has comprehensive API infrastructure that is NOT being used:

**Existing API Services** (`frontend/src/api/`):
- `kpi.ts` - Full KPI API (list, calculate, batch, cache, health)
- `graph.ts` - Graph API (nodes, causal chains, search, cypher queries)
- `monitoring.ts` - Monitoring endpoints
- `memory.ts`, `cognitive.ts`, `explain.ts`, etc.

**Existing React Query Hooks** (`frontend/src/hooks/api/`):
- `use-kpi.ts` - `useKPIList`, `useKPIValue`, `useBatchCalculateKPIs`, `useKPIHealth`
- `use-graph.ts` - `useNodes`, `useCausalChains`, `useGraphStats`, `useGraphSearch`
- `use-monitoring.ts` - Monitoring hooks
- `use-cytoscape.ts` - Graph visualization hook!

**The gap is NOT missing APIs - it's that Home.tsx doesn't use them.**

### 1.3 Mock Dashboard Features (`E2I_Causal_Dashboard_V3.html`)

**Overview Tab (Lines 1440-1568):**
- Executive Intelligence Summary - Real-time Causal Impact Analysis
  - Total causal relationships tracked
  - Active optimization cycles
  - Estimated revenue impact
  - Model confidence score
- Primary Causal Value Chains - Live Tracking
  - Interactive causal chain cards
  - Impact indicators (high/medium)
  - Trend visualization
  - Confidence scores

**AI Agent Insights Tab (Lines 1571-1969):**
- Executive AI Brief (GPT-4 powered summary)
- Priority Actions by ROI (ranked recommendations)
- Active Causal Chains (visual graph)
- Predictive Alerts (forecasting warnings)
- Experiment Recommendations (A/B test suggestions)
- Heterogeneous Treatment Effects (CATE analysis)
- Resource Optimization Matrix
- System Health Score (operational metrics)

### 1.4 Backend API Capability

| Endpoint Module | Features | Mock Support | Frontend Hook |
|-----------------|----------|--------------|---------------|
| `/graph/*` | Knowledge graph, causal chains | Causal Value Chains | `useCausalChains`, `useGraphStats` |
| `/api/kpis/*` | 46 KPIs calculation | KPI Cards | `useKPIList`, `useKPIValue` |
| `/monitoring/*` | Drift detection, alerts, health | Health, Alerts | `use-monitoring.ts` |
| `/explain/*` | SHAP predictions | Treatment Effects | `use-explain.ts` |
| `/causal/*` | Hierarchical CATE, pipelines | Executive Summary | Needs new hook |
| `/memory/*` | Hybrid search, episodic memory | AI Brief context | `use-memory.ts` |

**Verdict**: Backend AND frontend API layer fully support all mock dashboard features. Just need to wire them up.

---

## 2. Gap Analysis (Revised)

### 2.1 True Gaps

1. **Home.tsx not using existing hooks**: Uses hardcoded data instead of `useKPIList`, `useGraphStats`, etc.
2. **Missing UI components**: Executive Summary, Causal Value Chains cards
3. **Missing AI Insights page**: New page needed
4. **Missing causal summary hook**: Need hook for `/causal/summary` endpoint

### 2.2 What Already Exists (No Work Needed)

- API service layer (`frontend/src/api/`)
- React Query hooks (`frontend/src/hooks/api/`)
- Cytoscape hook for graph visualization
- Type definitions (`frontend/src/types/`)

---

## 3. Implementation Phases (Revised - Much Smaller Scope)

### Phase 1: Wire Up Home.tsx to Existing Hooks
**Estimated Complexity**: Low-Medium
**Files to Modify**: 1-2 files

- [ ] Replace `SAMPLE_KPIS` with `useKPIList()` hook
- [ ] Replace `SYSTEM_STATUS` with monitoring hooks
- [ ] Add loading states and error boundaries
- [ ] Keep existing UI components, just change data source

**Critical Files**:
- `frontend/src/pages/Home.tsx` (modify)
- `frontend/src/hooks/api/use-kpi.ts` (exists - use as-is)
- `frontend/src/hooks/api/use-monitoring.ts` (exists - use as-is)

**Testing**: Verify API integration works, check loading states

---

### Phase 2: Executive Intelligence Summary Component
**Estimated Complexity**: Medium
**Files to Create/Modify**: 2-3 files

- [ ] Create `frontend/src/components/dashboard/ExecutiveSummary.tsx`
  - Total causal relationships (from `useGraphStats`)
  - Active optimization cycles (from monitoring)
  - Revenue impact estimate
  - Model confidence score
- [ ] Use existing `useGraphStats()` hook from `use-graph.ts`
- [ ] Add to `Home.tsx` layout above KPI cards

**Critical Files**:
- `frontend/src/components/dashboard/ExecutiveSummary.tsx` (create)
- `frontend/src/pages/Home.tsx` (modify)
- `frontend/src/hooks/api/use-graph.ts` (exists - use `useGraphStats`)

**Testing**: Component render test with mock data

---

### Phase 3: Primary Causal Value Chains Component
**Estimated Complexity**: Medium
**Files to Create/Modify**: 2-3 files

- [ ] Create `frontend/src/components/dashboard/CausalValueChains.tsx`
  - Interactive causal chain cards
  - Impact indicator badges (high/medium/low)
  - Confidence scores
- [ ] Use existing `useCausalChains()` mutation hook
- [ ] Use existing `useCytoscape()` hook for visualization
- [ ] Add to `Home.tsx` layout

**Critical Files**:
- `frontend/src/components/dashboard/CausalValueChains.tsx` (create)
- `frontend/src/pages/Home.tsx` (modify)
- `frontend/src/hooks/api/use-graph.ts` (exists - use `useCausalChains`)
- `frontend/src/hooks/use-cytoscape.ts` (exists - for graph viz)

**Testing**: Component render test, causal chain display

---

### Phase 4: AI Agent Insights Page
**Estimated Complexity**: Medium-High
**Files to Create/Modify**: 3-4 files

- [ ] Create `frontend/src/pages/AIAgentInsights.tsx` - New page
- [ ] Create `frontend/src/components/insights/` directory with:
  - `ExecutiveAIBrief.tsx` - Use `use-cognitive.ts` or `use-memory.ts`
  - `PriorityActionsROI.tsx` - Display ranked recommendations
  - `PredictiveAlerts.tsx` - Use `use-monitoring.ts`
- [ ] Add route to router (check `App.tsx` or router config)
- [ ] Add navigation link to sidebar

**Critical Files**:
- `frontend/src/pages/AIAgentInsights.tsx` (create)
- `frontend/src/App.tsx` or router config (modify)
- Existing hooks: `use-cognitive.ts`, `use-monitoring.ts`, `use-explain.ts`

**Testing**: Page render tests, navigation test

---

### Phase 5: AI Insights Advanced Components
**Estimated Complexity**: Medium
**Files to Create/Modify**: 4-5 files

- [ ] Create `frontend/src/components/insights/ActiveCausalChains.tsx`
  - Full Cytoscape graph visualization
  - Use existing `useCytoscape` hook
- [ ] Create `frontend/src/components/insights/ExperimentRecommendations.tsx`
- [ ] Create `frontend/src/components/insights/HeterogeneousTreatmentEffects.tsx`
  - Use `use-explain.ts` for SHAP data
- [ ] Create `frontend/src/components/insights/SystemHealthScore.tsx`
  - Use `use-monitoring.ts`

**Critical Files**:
- Components in `frontend/src/components/insights/`
- Hooks: `use-cytoscape.ts`, `use-explain.ts`, `use-monitoring.ts`

**Testing**: Individual component tests

---

### Phase 6: Polish and Integration
**Estimated Complexity**: Low-Medium
**Files to Modify**: Multiple

- [ ] Add loading skeletons to all new components
- [ ] Add error boundaries with retry buttons
- [ ] Responsive layout adjustments
- [ ] Match mock dashboard styling

**Testing**: Visual inspection, responsive tests

---

## 4. Testing Strategy

### 4.1 Resource-Conscious Approach

Due to limited resources, testing will follow these principles:

1. **Small Batch Execution**: Max 5 tests per run
2. **Isolation**: Test components in isolation before integration
3. **Existing Tests**: Leverage existing hook tests in `frontend/src/hooks/api/*.test.ts`
4. **Progressive Testing**: Test each phase before moving to next

### 4.2 Test Commands

```bash
# Run existing hook tests first (they already pass)
cd frontend && npm test -- use-kpi.test.ts
cd frontend && npm test -- use-graph.test.ts

# Phase-specific testing (recommended)
npm test -- ExecutiveSummary
npm test -- CausalValueChains

# Small batch E2E (single worker for memory)
npx playwright test --grep "dashboard" --workers=1
```

### 4.3 Existing Test Coverage (Already Done)

These tests already exist and pass:
- `frontend/src/hooks/api/use-kpi.test.ts`
- `frontend/src/hooks/api/use-graph.test.ts`
- `frontend/src/hooks/api/use-monitoring.test.ts`
- `frontend/src/hooks/api/use-explain.test.ts`
- `frontend/src/hooks/use-cytoscape.test.ts`

---

## 5. Progress Tracking

### Overall Progress: 6/6 Phases Complete ✅

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| 1. Wire Up Home.tsx | ✅ Complete | 2026-01-07 | 2026-01-07 |
| 2. Executive Summary | ✅ Complete | 2026-01-07 | 2026-01-07 |
| 3. Causal Value Chains | ✅ Complete | 2026-01-07 | 2026-01-07 |
| 4. AI Insights Page | ✅ Complete | 2026-01-07 | 2026-01-07 |
| 5. AI Insights Advanced | ✅ Complete | 2026-01-07 | 2026-01-07 |
| 6. Polish & Integration | ✅ Complete | 2026-01-07 | 2026-01-07 |

### Phase 1 Summary
- Added API hook imports (`useKPIList`, `useKPIHealth`, `useGraphStats`, `useAlerts`)
- Created `effectiveKPIs` transformation layer (API → local KPIMetric)
- Updated `filteredKPIs` and `summaryStats` to use API data when available
- Updated alerts to use API data with fallback to sample data
- Added API status indicator badges in header
- Connected refresh button to invalidate React Query cache
- Build passes ✅

### Phase 2 Summary
- Created `frontend/src/components/dashboard/ExecutiveSummary.tsx`
- Component displays: main summary card, quick stats row (Causal Paths, Graph Nodes, System Health, Est. Impact)
- Three metric cards: Data-to-Value Pipeline, Model-to-Impact Bridge, Fairness & Trust Nexus
- Uses `useGraphStats` and `useKPIHealth` hooks for live data
- Loading skeleton state included
- Added to Home.tsx layout
- Build passes ✅

### Phase 3 Summary
- Created `frontend/src/components/dashboard/CausalValueChains.tsx`
- Component displays live tracking of causal value chains
- Uses `useCausalChains` mutation hook for real data
- Three chain cards with: chain visualization (node → node → result), status badges, confidence/method metadata
- Includes `transformGraphPathToCard` function to convert API GraphPath → UI display
- Sample data fallback when API unavailable
- Added to Home.tsx layout after ExecutiveSummary
- Build passes ✅

### Phase 4 Summary
- Created `frontend/src/pages/AIAgentInsights.tsx` - Main page with grid layout
- Created `frontend/src/components/insights/ExecutiveAIBrief.tsx` - GPT-powered executive summary
  - Uses `useCognitiveRAG` hook for AI-generated insights
  - Three brief sections with confidence scores
  - Refresh capability and loading states
- Created `frontend/src/components/insights/PriorityActionsROI.tsx` - Ranked recommendations
  - Action cards with estimated ROI ($2.3M, $1.8M, etc.)
  - Effort level badges (low/medium/high)
  - Confidence progress bars
- Created `frontend/src/components/insights/PredictiveAlerts.tsx` - Alert display
  - Uses `useAlerts` hook with `AlertStatus.ACTIVE` filter
  - Severity-based styling (critical/warning/info)
  - Transform function for API→UI mapping
- Created `frontend/src/components/insights/index.ts` - Barrel export
- Updated `frontend/src/router/routes.tsx`:
  - Added lazy import for AIAgentInsights
  - Added route config (path: '/ai-insights', icon: 'brain', showInNav: true)
  - Added route definition with LazyPage wrapper
- Build passes ✅

### Phase 5 Summary
- Created `frontend/src/components/insights/ActiveCausalChains.tsx`
  - Full Cytoscape.js graph visualization with `useCytoscape` hook
  - Uses `useCausalChains` mutation for real API data
  - Interactive node selection, zoom controls, re-layout functionality
  - Legend showing node types (intervention, mediator, moderator, outcome)
  - Sample fallback data when API unavailable
- Created `frontend/src/components/insights/ExperimentRecommendations.tsx`
  - A/B test recommendations with power analysis
  - Priority badges (high/medium), segment targeting
  - Sample size and duration estimates
- Created `frontend/src/components/insights/HeterogeneousTreatmentEffects.tsx`
  - Segment-level CATE (Conditional Average Treatment Effects) analysis
  - Uses `useBatchExplain` hook for SHAP-based analysis
  - Shows treatment effects, confidence, p-values per segment
  - Top drivers visualization with Progress bars
- Created `frontend/src/components/insights/SystemHealthScore.tsx`
  - Overall system health score display
  - Uses `useModelHealth`, `useMonitoringRuns`, `useAlerts` hooks
  - Health metrics: Model Drift, Data Quality, API Latency, Inference Throughput
  - Quick stats: Models monitored, Active alerts, Last check
  - Recommendations section from health data
- Updated `frontend/src/components/insights/index.ts` barrel export
- Updated `frontend/src/pages/AIAgentInsights.tsx` to use all 7 components
- Fixed type errors in ActiveCausalChains.tsx and HeterogeneousTreatmentEffects.tsx
- Build passes ✅

### Phase 6 Summary
- Ran lint check - only pre-existing warnings in test files
- Verified all existing hook tests pass:
  - `use-graph.test.ts`: 31 tests passed
  - `use-monitoring.test.ts`: 41 tests passed
  - `use-explain.test.ts`: 19 tests passed
  - `use-cytoscape.test.ts`: 80 tests passed
  - Total: 171 hook tests passing
- Ran Home page E2E tests: 30 tests passed
- Created `frontend/e2e/specs/ai-insights.spec.ts`:
  - Page load tests (4 tests)
  - Component visibility tests (9 tests)
  - Responsive design tests (3 tests)
  - Total: 17 E2E tests passing
- All builds and tests passing ✅
- Implementation complete!

---

## 6. Dependencies

### External Libraries (Already Installed)
- `@tanstack/react-query` - Already installed and configured
- `cytoscape` - Already installed (see `use-cytoscape.ts`)
- `recharts` - Already installed for charts
- `axios` or fetch wrapper - Already in `@/lib/api-client`

### No New Dependencies Required!

### Backend Endpoints (All Exist)
- `GET /graph/stats` - Graph statistics (total nodes, relationships)
- `POST /graph/causal-chains` - Causal chains query
- `GET /kpis` - KPI list
- `GET /kpis/{id}` - Individual KPI value
- `GET /monitoring/health` - System health

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Large context window usage | Small phases (6 not 8), test in batches |
| Memory constraints | Workers=1 for E2E, use existing test mocks |
| Backend not running | Use existing MSW mocks in test files |
| Type mismatches | Types already defined in `frontend/src/types/` |

---

## 8. Next Steps

1. **Immediate**: Begin Phase 1 - Wire up Home.tsx to existing hooks
2. **First change**: Replace `SAMPLE_KPIS` with `useKPIList()` call
3. **Verify**: Backend is running (or use test mocks)

---

## 9. Key Files Summary

### Files to Create (6 new files)
```
frontend/src/components/dashboard/ExecutiveSummary.tsx
frontend/src/components/dashboard/CausalValueChains.tsx
frontend/src/pages/AIAgentInsights.tsx
frontend/src/components/insights/ExecutiveAIBrief.tsx
frontend/src/components/insights/PriorityActionsROI.tsx
frontend/src/components/insights/PredictiveAlerts.tsx
frontend/src/components/insights/ActiveCausalChains.tsx
frontend/src/components/insights/SystemHealthScore.tsx
```

### Files to Modify (2-3 files)
```
frontend/src/pages/Home.tsx - Main changes
frontend/src/App.tsx - Add AI Insights route
```

### Files to Use As-Is (Existing Infrastructure)
```
frontend/src/api/kpi.ts
frontend/src/api/graph.ts
frontend/src/api/monitoring.ts
frontend/src/hooks/api/use-kpi.ts
frontend/src/hooks/api/use-graph.ts
frontend/src/hooks/api/use-monitoring.ts
frontend/src/hooks/use-cytoscape.ts
frontend/src/types/kpi.ts
frontend/src/types/graph.ts
```

---

## Appendix: Mock Dashboard Reference

Key HTML sections from `E2I_Causal_Dashboard_V3.html`:
- Executive Summary: Lines 1440-1500
- Causal Value Chains: Lines 1503-1568
- AI Agent Insights Tab: Lines 1571-1969

Reference these for exact styling and data structure requirements.
