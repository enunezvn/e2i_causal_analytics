# E2I Causal Analytics Frontend Audit & Implementation Plan

**Created**: 2026-01-02
**Updated**: 2026-01-03
**Status**: ✅ ALL PHASES COMPLETE (Frontend Audit Implementation Done)
**Estimated Duration**: ~12 working days (2.5 weeks)

---

## Current Progress (as of 2026-01-03)

### Implementation Status: ALL CODE COMPLETE ✅

| Phase | Code | Tests | Status |
|-------|------|-------|--------|
| Phase 0: Foundation | ✅ | ✅ 3 files | COMPLETE |
| Phase 1: Viz Components | ✅ 14 components | ❌ Missing | Code complete, needs tests |
| Phase 2: P1 Pages | ✅ 3 pages | ❌ Missing | Code complete, needs tests |
| Phase 3: Monitoring/DQ | ✅ 2 pages | ❌ Missing | Code complete, needs tests |
| Phase 4: Analytics | ✅ 3 pages | ❌ Missing | Code complete, needs tests |
| Phase 5: CopilotKit | ✅ | ✅ 5 files | COMPLETE |
| Phase 6: Home | ✅ | ❌ Missing | Code complete, needs tests |
| Phase 7: Digital Twin | ✅ | ✅ 3 files | COMPLETE |

### Test Summary
- **Passing**: 331 tests across 15 test files
- **Complete**: Batches 1, 2, 11, 13
- **Missing**: ~76 tests across 8 batches (Batches 3-10)

### Batch 2 Completed (175 tests)
- Dashboard: 47 tests (KPICard, StatusBadge, ProgressRing, AlertCard)
- SHAP: 40 tests (SHAPBarChart, SHAPBeeswarm, SHAPForcePlot, SHAPWaterfall)
- Charts: 51 tests (MetricTrend, ROCCurve, ConfusionMatrix, MultiAxisLineChart)
- Agents: 37 tests (AgentTierBadge, TierOverview, AgentInsightCard)

### Resolved Issues
- ✅ Fixed CSS import error in Vitest (katex.min.css from CopilotKit)
- ✅ Added CopilotKit mocks in test setup.ts

---

## Executive Summary

The E2I frontend is a React 18 + TypeScript + Vite application with:
- **3 fully implemented pages** (Home, KnowledgeGraph, CausalDiscovery)
- **8 placeholder pages** requiring full implementation
- **Dead code** (App.tsx, App.css) to remove
- **Missing API hooks** for monitoring, KPIs, predictions
- **No unit tests** for components

This plan breaks implementation into **7 context-window friendly phases** with **13 small test batches**.

---

## Current State

### Tech Stack (Complete)
- React 18.3.1, TypeScript 5.6.2, Vite 6.0.5
- State: Zustand + TanStack React Query
- UI: Shadcn (16 components) + Tailwind CSS + Radix UI
- Viz: Cytoscape.js, D3.js 7.9.0, Recharts, Plotly.js

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Home Page | COMPLETE | Navigation grid |
| KnowledgeGraph Page | COMPLETE | Cytoscape + stats + filters |
| CausalDiscovery Page | COMPLETE | D3 DAG + effects table |
| API Layer | COMPLETE | 5 modules (rag, graph, memory, cognitive, explain) |
| React Query Hooks | COMPLETE | 5 modules, 1,718 LOC |
| App.tsx | DEAD CODE | Default Vite template - DELETE |
| 8 Placeholder Pages | PLACEHOLDER | 15 LOC each, "coming soon" |

### Placeholder Pages to Implement

| Page | Priority | Complexity | Backend API |
|------|----------|------------|-------------|
| ModelPerformance | P0 | Medium | `/monitoring/performance/*` |
| FeatureImportance | P0 | High | `/explain/*` |
| DataQuality | P1 | Medium | `/api/kpis/*` |
| SystemHealth | P1 | Low | `/monitoring/health/*` |
| Monitoring | P1 | Medium | `/monitoring/alerts`, `/monitoring/runs` |
| TimeSeries | P2 | Medium | `/api/kpis/*` (time-based) |
| InterventionImpact | P2 | High | `/causal/*` |
| PredictiveAnalytics | P2 | Medium | `/api/models/predict/*` |

---

## Phase 0: Cleanup & Foundation

**Goal**: Remove dead code, add missing API infrastructure
**Duration**: 0.5 days

### Files to DELETE
```
frontend/src/App.tsx           # Dead code - main.tsx uses AppRouter
frontend/src/App.css           # Unused styles
```

### Files to CREATE

**API Clients:**
```
frontend/src/api/monitoring.ts     # ~200 LOC - health, alerts, drift
frontend/src/api/kpi.ts            # ~150 LOC - KPI endpoints
frontend/src/api/predictions.ts    # ~100 LOC - model predictions
```

**React Query Hooks:**
```
frontend/src/hooks/api/use-monitoring.ts   # ~300 LOC
frontend/src/hooks/api/use-kpi.ts          # ~200 LOC
frontend/src/hooks/api/use-predictions.ts  # ~150 LOC
```

**TypeScript Types:**
```
frontend/src/types/monitoring.ts   # ~200 LOC
frontend/src/types/kpi.ts          # ~150 LOC
frontend/src/types/predictions.ts  # ~100 LOC
```

### Files to MODIFY
```
frontend/src/mocks/handlers.ts     # Add monitoring, KPI, predictions handlers
frontend/src/types/index.ts        # Export new types
frontend/src/hooks/api/index.ts    # Export new hooks (create if needed)
```

### Test Batch 1 (15 tests)
```
frontend/src/__tests__/api/monitoring.test.ts
frontend/src/__tests__/api/kpi.test.ts
frontend/src/__tests__/hooks/use-monitoring.test.ts
```

---

## Phase 1: Core Visualization Components

**Goal**: Build reusable components needed across multiple pages
**Duration**: 1.5 days
**Dependencies**: Phase 0

### 1A. SHAP Visualization Components
```
frontend/src/components/visualizations/shap/
├── SHAPBarChart.tsx       # ~200 LOC - Horizontal importance bar
├── SHAPBeeswarm.tsx       # ~300 LOC - D3 beeswarm plot
├── SHAPForcePlot.tsx      # ~250 LOC - Force plot
├── SHAPWaterfall.tsx      # ~200 LOC - Waterfall chart
└── index.ts
```

### 1B. Chart Components
```
frontend/src/components/visualizations/charts/
├── MultiAxisLineChart.tsx # ~250 LOC - Recharts multi-axis
├── ConfusionMatrix.tsx    # ~200 LOC - Heatmap matrix
├── ROCCurve.tsx           # ~250 LOC - D3 ROC/PR curves
├── MetricTrend.tsx        # ~150 LOC - Sparkline + trend
└── index.ts
```

### 1C. Dashboard Components
```
frontend/src/components/dashboard/
├── KPICard.tsx            # ~150 LOC - Metric card with trend
├── StatusBadge.tsx        # ~80 LOC - Health/status badge
├── ProgressRing.tsx       # ~100 LOC - Circular progress
├── AlertCard.tsx          # ~120 LOC - Alert display
└── index.ts
```

### 1D. Agent Components (from V3 inspiration)
```
frontend/src/components/agents/
├── AgentInsightCard.tsx   # ~200 LOC - 4 agent card variants
├── AgentTierBadge.tsx     # ~80 LOC - Tier color badges
└── index.ts
```

### Test Batch 2 (15 tests)
```
frontend/src/__tests__/components/shap/SHAPBarChart.test.tsx
frontend/src/__tests__/components/charts/MultiAxisLineChart.test.tsx
frontend/src/__tests__/components/dashboard/KPICard.test.tsx
```

---

## Phase 2: P1 Priority Pages

**Goal**: Implement SystemHealth, ModelPerformance, FeatureImportance
**Duration**: 2.5 days
**Dependencies**: Phase 1

### 2A. SystemHealth Page (0.5 day)
**File**: `frontend/src/pages/SystemHealth.tsx` (~300 LOC)

**Features**:
- Service status grid (API, Database, Redis, FalkorDB, BentoML)
- Model health cards with health scores
- Active alerts list
- Auto-refresh every 30s

**Pattern**: Follow `KnowledgeGraph.tsx` (stats cards + main content)

### 2B. ModelPerformance Page (1 day)
**File**: `frontend/src/pages/ModelPerformance.tsx` (~450 LOC)

**Features**:
- Model selector dropdown
- Performance metrics cards (Accuracy, Precision, Recall, F1, AUC)
- Confusion matrix heatmap
- ROC/PR curve visualization
- Performance trend over time

**Pattern**: Follow `CausalDiscovery.tsx` (controls + visualization + details)

### 2C. FeatureImportance Page (1 day)
**File**: `frontend/src/pages/FeatureImportance.tsx` (~400 LOC)

**Features**:
- Model/Entity selector
- SHAP summary bar chart (global importance)
- SHAP beeswarm plot
- Individual force plot for selected instance
- Narrative explanation display

**Pattern**: Follow `CausalDiscovery.tsx` (tabs for different views)

### Test Batches 3-5 (28 tests)
```
# Batch 3: SystemHealth (8 tests)
frontend/src/__tests__/pages/SystemHealth.test.tsx

# Batch 4: ModelPerformance (10 tests)
frontend/src/__tests__/pages/ModelPerformance.test.tsx

# Batch 5: FeatureImportance (10 tests)
frontend/src/__tests__/pages/FeatureImportance.test.tsx
```

---

## Phase 3: Monitoring & Data Quality Pages

**Goal**: Implement DataQuality and Monitoring pages
**Duration**: 1 day
**Dependencies**: Phase 1

### 3A. DataQuality Page (0.5 day)
**File**: `frontend/src/pages/DataQuality.tsx` (~350 LOC)

**Features**:
- Data quality KPI cards (completeness, freshness, accuracy)
- Validation rules status grid
- Data profiling summary
- Quality trend charts
- Issue alerts

### 3B. Monitoring Page (0.5 day)
**File**: `frontend/src/pages/Monitoring.tsx` (~400 LOC)

**Features**:
- Active alerts table with actions (acknowledge, resolve)
- Drift detection results grid
- Monitoring runs history
- Retraining triggers display
- Alert severity filters

### Test Batches 6-7 (18 tests)
```
# Batch 6: DataQuality (8 tests)
frontend/src/__tests__/pages/DataQuality.test.tsx

# Batch 7: Monitoring (10 tests)
frontend/src/__tests__/pages/Monitoring.test.tsx
```

---

## Phase 4: Analytics Pages

**Goal**: Implement TimeSeries, InterventionImpact, PredictiveAnalytics
**Duration**: 3 days
**Dependencies**: Phase 1

### 4A. TimeSeries Page (1 day)
**File**: `frontend/src/pages/TimeSeries.tsx` (~400 LOC)

**Features**:
- KPI selector for time series
- Date range picker
- Multi-metric line chart
- Trend decomposition
- Forecast with confidence intervals
- Anomaly detection highlights

### 4B. InterventionImpact Page (1.5 days)
**File**: `frontend/src/pages/InterventionImpact.tsx` (~500 LOC)

**New Components**:
```
frontend/src/components/visualizations/causal/
├── TreatmentEffectChart.tsx   # ~250 LOC - CATE visualization
└── HeterogeneousHeatmap.tsx   # ~200 LOC - Segment effects
```

**Features**:
- Treatment/Outcome selector
- ATE display with confidence intervals
- CATE by segment (heatmap)
- Refutation test results
- Counterfactual scenarios

### 4C. PredictiveAnalytics Page (0.5 day)
**File**: `frontend/src/pages/PredictiveAnalytics.tsx` (~350 LOC)

**Features**:
- Model selector (churn, conversion, propensity)
- Entity lookup form
- Prediction result card
- Confidence score display
- Feature contribution summary

### Test Batches 8-10 (30 tests)
```
# Batch 8: TimeSeries (10 tests)
frontend/src/__tests__/pages/TimeSeries.test.tsx

# Batch 9: InterventionImpact (12 tests)
frontend/src/__tests__/pages/InterventionImpact.test.tsx

# Batch 10: PredictiveAnalytics (8 tests)
frontend/src/__tests__/pages/PredictiveAnalytics.test.tsx
```

---

## Phase 5: CopilotKit Chat Interface

**Goal**: Natural Language Query interface using CopilotKit + Supabase memory
**Duration**: 1.5 days
**Dependencies**: Phase 0

### Pre-requisite: Database Migration
```bash
# Run the chat memory tables migration
psql -h $SUPABASE_HOST -U postgres -d postgres \
  -f "Chatbot memory/008_chatbot_memory_tables.sql"
```

This creates:
- `chat_threads` - Conversation threads with topic embeddings
- `chat_messages` - Messages with content embeddings & feedback
- `user_preferences` - User-level preferences (detail_level, default_brand, etc.)

### Install Dependencies
```bash
cd frontend
npm install @copilotkit/react-core @copilotkit/react-ui framer-motion
```

### Files to COPY/ADAPT from "Chatbot memory/"
```
# Copy and adapt these files:
Chatbot memory/E2ICopilotProvider.tsx → frontend/src/providers/E2ICopilotProvider.tsx
Chatbot memory/usage.tsx             → frontend/src/components/chat/examples.tsx (reference)
```

### Files to CREATE
```
frontend/src/providers/
├── E2ICopilotProvider.tsx   # From Chatbot memory folder (~400 LOC)
└── index.ts

frontend/src/components/chat/
├── E2IChatSidebar.tsx       # Sidebar wrapper using CopilotKit (~150 LOC)
├── E2IChatPopup.tsx         # Popup variant with ⌘/ shortcut (~100 LOC)
├── AgentStatusPanel.tsx     # Shows 18 agents status (~200 LOC)
├── ValidationBadge.tsx      # PROCEED/REVIEW/BLOCK badges (~80 LOC)
└── index.ts

frontend/src/hooks/
├── useE2IFilters.ts         # Dashboard filter context hook (~100 LOC)
├── useE2IHighlights.ts      # Causal path highlighting hook (~80 LOC)
├── useE2IValidation.ts      # Validation state hook (~80 LOC)
└── useUserPreferences.ts    # Supabase preferences hook (~120 LOC)
```

### Files to MODIFY
```
frontend/src/main.tsx                    # Wrap with E2ICopilotProvider
frontend/src/components/layout/Layout.tsx # Add E2IChatSidebar
```

### Backend API Endpoint (if not exists)
```
src/api/routes/copilotkit.py  # POST /api/copilotkit - LangGraph adapter
```

**CopilotKit Features**:
- **useCopilotReadable**: Expose dashboard filters, agents, preferences to AI
- **useCopilotAction**: Let agents trigger UI changes (highlights, navigation)
- **E2IChatSidebar**: Sliding panel with streaming responses
- **E2IChatPopup**: Modal chat (⌘/ shortcut)
- **Agent tier badges**: Color-coded by tier (0-5)
- **Validation badges**: PROCEED (green), REVIEW (orange), BLOCK (red)
- **Supabase persistence**: Chat history with semantic search

**Key Integration Points**:
```tsx
// main.tsx
<E2ICopilotProvider
  runtimeUrl="/api/copilotkit"
  initialFilters={{ brand: 'Remibrutinib' }}
  userRole="analyst"
>
  <QueryClientProvider client={queryClient}>
    <AppRouter />
  </QueryClientProvider>
</E2ICopilotProvider>

// Layout.tsx
<E2IChatSidebar defaultOpen={false} />
```

### Test Batch 11 (13 tests)
```
frontend/src/__tests__/providers/E2ICopilotProvider.test.tsx
frontend/src/__tests__/components/chat/E2IChatSidebar.test.tsx
frontend/src/__tests__/hooks/useUserPreferences.test.tsx
```

---

## Phase 6: Enhanced Home Page

**Goal**: Transform Home into KPI Executive Dashboard
**Duration**: 1 day
**Dependencies**: Phases 1, 2, 3

### File to MODIFY
```
frontend/src/pages/Home.tsx  # Rewrite ~600 LOC
```

**Features**:
- KPI Dashboard with 46+ metrics in categories
- Quick stats cards (TRx, NRx, conversion rates)
- Brand selector (Remibrutinib, Fabhalta, Kisqali)
- Recent agent insights feed
- System health summary
- Quick action shortcuts

### Test Batch 12 (12 tests)
```
frontend/src/__tests__/pages/Home.test.tsx
```

---

## Phase 7: Digital Twin Integration

**Goal**: Digital Twin simulation visualization (P0 PRD requirement)
**Duration**: 1 day
**Dependencies**: Phase 4B

### Files to CREATE
```
frontend/src/components/digital-twin/
├── SimulationPanel.tsx       # ~200 LOC - Controls
├── ScenarioResults.tsx       # ~250 LOC - Results display
├── RecommendationCards.tsx   # ~150 LOC - Action recommendations
└── index.ts

frontend/src/api/digital-twin.ts          # ~100 LOC
frontend/src/hooks/api/use-digital-twin.ts # ~150 LOC
frontend/src/types/digital-twin.ts        # ~100 LOC
```

### Files to MODIFY
```
frontend/src/pages/InterventionImpact.tsx  # Add Digital Twin tab
```

**Features**:
- Simulation controls (intervention type, sample size)
- Results display (ATE, ROI projection)
- Fidelity metrics
- Recommendation cards (Deploy/Skip/Refine)

### Test Batch 13 (8 tests)
```
frontend/src/__tests__/components/digital-twin/SimulationPanel.test.tsx
```

---

## Dependency Graph

```
Phase 0 (Foundation) ─────────────────────────────────────────┐
    │                                                         │
    └──→ Phase 1 (Viz Components) ──┬──→ Phase 2A (SystemHealth)
                                    │
                                    ├──→ Phase 2B (ModelPerformance)
                                    │
                                    ├──→ Phase 2C (FeatureImportance)
                                    │
                                    ├──→ Phase 3A (DataQuality)
                                    │
                                    ├──→ Phase 3B (Monitoring)
                                    │
                                    ├──→ Phase 4A (TimeSeries)
                                    │
                                    ├──→ Phase 4B (InterventionImpact) ──→ Phase 7 (Digital Twin)
                                    │
                                    ├──→ Phase 4C (PredictiveAnalytics)
                                    │
                                    └──→ Phase 5 (Chat) ← Independent

Phases 2-4 ──→ Phase 6 (Enhanced Home)
```

---

## Testing Strategy

### Memory-Efficient Execution
Due to low resource constraints, run tests in small batches:

```bash
# Run one batch at a time
cd frontend
npm run test -- --run src/__tests__/api/
npm run test -- --run src/__tests__/components/shap/
npm run test -- --run src/__tests__/pages/SystemHealth.test.tsx
# etc.
```

### Test Distribution
| Batch | Tests | Files | Phase |
|-------|-------|-------|-------|
| 1 | 15 | 3 | Phase 0 |
| 2 | 15 | 3 | Phase 1 |
| 3 | 8 | 1 | Phase 2A |
| 4 | 10 | 1 | Phase 2B |
| 5 | 10 | 1 | Phase 2C |
| 6 | 8 | 1 | Phase 3A |
| 7 | 10 | 1 | Phase 3B |
| 8 | 10 | 1 | Phase 4A |
| 9 | 12 | 1 | Phase 4B |
| 10 | 8 | 1 | Phase 4C |
| 11 | 13 | 2 | Phase 5 |
| 12 | 12 | 1 | Phase 6 |
| 13 | 8 | 1 | Phase 7 |
| **Total** | **~139** | **18** | |

---

## Critical Reference Files

**Patterns to Follow:**
- `frontend/src/pages/KnowledgeGraph.tsx` - Page structure with API hooks
- `frontend/src/components/visualizations/CausalDiscovery.tsx` - Complex viz with D3
- `frontend/src/hooks/api/use-explain.ts` - React Query hook patterns
- `frontend/src/stores/ui-store.ts` - Zustand store pattern

**Backend API Reference:**
- `src/api/routes/monitoring.py` - Monitoring endpoints
- `src/api/routes/explain.py` - SHAP/explain endpoints
- `src/api/routes/kpis.py` - KPI endpoints

**V3 Inspiration:**
- `frontend/E2I_Causal_Dashboard_V3.html` - Design patterns

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0: Foundation | 0.5 days | 0.5 days |
| Phase 1: Viz Components | 1.5 days | 2 days |
| Phase 2: P1 Pages | 2.5 days | 4.5 days |
| Phase 3: Monitoring/DQ | 1 day | 5.5 days |
| Phase 4: Analytics | 3 days | 8.5 days |
| Phase 5: Chat | 1.5 days | 10 days |
| Phase 6: Home | 1 day | 11 days |
| Phase 7: Digital Twin | 1 day | 12 days |

**Total: ~12 working days**

---

## TODO Checklist

**Last Verified**: 2026-01-03 (156 tests passing across 11 test files)

### Phase 0: Foundation ✅ COMPLETE
- [x] Delete `frontend/src/App.tsx` ✅
- [x] Delete `frontend/src/App.css` ✅
- [x] Create `frontend/src/api/monitoring.ts` ✅ (15.7KB)
- [x] Create `frontend/src/api/kpi.ts` ✅ (6.6KB)
- [x] Create `frontend/src/api/predictions.ts` ✅ (4.6KB)
- [x] Create `frontend/src/hooks/api/use-monitoring.ts` ✅ (23.1KB)
- [x] Create `frontend/src/hooks/api/use-kpi.ts` ✅ (12.4KB)
- [x] Create `frontend/src/hooks/api/use-predictions.ts` ✅ (9.6KB)
- [x] Create `frontend/src/types/monitoring.ts` ✅ (11.1KB)
- [x] Create `frontend/src/types/kpi.ts` ✅ (7.5KB)
- [x] Create `frontend/src/types/predictions.ts` ✅ (4.6KB)
- [x] Update MSW handlers ✅
- [x] Run Test Batch 1 ✅ (monitoring.test.ts, kpi.test.ts, predictions.test.ts)

### Phase 1: Visualization Components ✅ CODE COMPLETE (Tests Missing)
- [x] Create SHAP components (4 files) ✅ (SHAPBarChart, SHAPBeeswarm, SHAPForcePlot, SHAPWaterfall)
- [x] Create Chart components (4 files) ✅ (MultiAxisLineChart, ConfusionMatrix, ROCCurve, MetricTrend)
- [x] Create Dashboard components (4 files) ✅ (KPICard, StatusBadge, ProgressRing, AlertCard)
- [x] Create Agent components (2 files) ✅ (AgentInsightCard, AgentTierBadge)
- [ ] **Run Test Batch 2** ⏳ NOT STARTED

### Phase 2: P1 Priority Pages ✅ CODE COMPLETE (Tests Missing)
- [x] Implement SystemHealth page ✅ (17.9KB)
- [ ] **Run Test Batch 3** ⏳ NOT STARTED
- [x] Implement ModelPerformance page ✅ (19.2KB)
- [ ] **Run Test Batch 4** ⏳ NOT STARTED
- [x] Implement FeatureImportance page ✅ (21.4KB)
- [ ] **Run Test Batch 5** ⏳ NOT STARTED

### Phase 3: Monitoring & Data Quality ✅ CODE COMPLETE (Tests Missing)
- [x] Implement DataQuality page ✅ (30.1KB)
- [ ] **Run Test Batch 6** ⏳ NOT STARTED
- [x] Implement Monitoring page ✅ (34.8KB)
- [ ] **Run Test Batch 7** ⏳ NOT STARTED

### Phase 4: Analytics Pages ✅ CODE COMPLETE (Tests Missing)
- [x] Implement TimeSeries page ✅ (43.6KB)
- [ ] **Run Test Batch 8** ⏳ NOT STARTED
- [x] Implement InterventionImpact page ✅ (42KB)
- [x] Create causal visualization components ✅
- [ ] **Run Test Batch 9** ⏳ NOT STARTED
- [x] Implement PredictiveAnalytics page ✅ (37.3KB)
- [ ] **Run Test Batch 10** ⏳ NOT STARTED

### Phase 5: CopilotKit Chat Interface ✅ COMPLETE
- [x] Run database migration (008_chatbot_memory_tables.sql) ✅
- [x] Install CopilotKit packages ✅
- [x] Copy/adapt E2ICopilotProvider ✅ (15.4KB)
- [x] Create E2IChatSidebar component ✅ (8.5KB)
- [x] Create E2IChatPopup component ✅ (7.1KB)
- [x] Create AgentStatusPanel component ✅ (9KB)
- [x] Create ValidationBadge component ✅ (5.7KB)
- [x] Create useE2IFilters hook ✅ (4.2KB)
- [x] Create useE2IHighlights hook ✅ (4.1KB)
- [x] Create useE2IValidation hook ✅ (5.5KB)
- [x] Create useUserPreferences hook ✅ (5.1KB)
- [x] Wrap main.tsx with E2ICopilotProvider ✅
- [x] Integrate E2IChatSidebar into Layout.tsx ✅
- [x] Create backend endpoint (POST /api/copilotkit) ✅ (21.6KB)
- [x] Run Test Batch 11 ✅ (5 test files: 4 hook tests + ValidationBadge.test.tsx)

### Phase 6: Enhanced Home ✅ CODE COMPLETE (Tests Missing)
- [x] Redesign Home page as KPI Dashboard ✅ (28KB)
- [ ] **Run Test Batch 12** ⏳ NOT STARTED

### Phase 7: Digital Twin ✅ COMPLETE
- [x] Create digital-twin API & hooks ✅ (digital-twin.ts: 4.3KB, use-digital-twin.ts: 6.7KB)
- [x] Create SimulationPanel component ✅ (11.4KB + test)
- [x] Create ScenarioResults component ✅ (15.6KB + test)
- [x] Create RecommendationCards component ✅ (10.1KB + test)
- [x] Integrate into InterventionImpact page ✅
- [x] Run Test Batch 13 ✅ (3 component tests)

---

## Outstanding Work: Test Batches

All frontend code is implemented. **Missing test batches:**

| Batch | Phase | Target | Tests Needed |
|-------|-------|--------|--------------|
| 2 | Phase 1 | SHAP, Charts, Dashboard, Agent components | ~15 tests |
| 3 | Phase 2A | SystemHealth page | ~8 tests |
| 4 | Phase 2B | ModelPerformance page | ~10 tests |
| 5 | Phase 2C | FeatureImportance page | ~10 tests |
| 6 | Phase 3A | DataQuality page | ~8 tests |
| 7 | Phase 3B | Monitoring page | ~10 tests |
| 8 | Phase 4A | TimeSeries page | ~10 tests |
| 9 | Phase 4B | InterventionImpact page | ~12 tests |
| 10 | Phase 4C | PredictiveAnalytics page | ~8 tests |
| 12 | Phase 6 | Home page | ~12 tests |

**Estimated remaining tests: ~103 tests across 10 batches**

---

## Success Criteria

- [x] All 8 placeholder pages fully implemented ✅
- [x] Dead code removed (App.tsx, App.css) ✅
- [x] CopilotKit chat interface functional with streaming & Supabase memory ✅
- [ ] All 13 test batches passing (~139 tests) ⏳ **11/13 batches complete (156 tests passing)**
- [x] Home page transformed into KPI dashboard ✅
- [x] Digital Twin integration complete ✅
- [x] All pages follow consistent patterns ✅
- [x] useCopilotReadable exposes dashboard context to AI ✅
- [x] useCopilotAction enables agent-triggered UI changes ✅

**Current Status**: 156 tests passing, ~103 tests remaining across 10 batches
