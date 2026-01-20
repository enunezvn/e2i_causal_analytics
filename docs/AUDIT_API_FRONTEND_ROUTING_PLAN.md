# API-Frontend Routing Audit Plan

**Audit Date:** 2026-01-19
**Branch:** `claude/audit-api-frontend-routing-OhAkA`
**Auditor:** Claude Code
**Scope:** Backend API completeness, frontend integration, agent output routing

---

## Executive Summary

This audit examines the data flow from the 18-agent tiered architecture through the backend API to the frontend UI. The goal is to ensure all agent outputs have a clear path to users, enabling complete visibility into system capabilities.

### Key Findings

| Metric | Value |
|--------|-------|
| Total Backend API Route Files | 20 |
| Total Backend Endpoints | 105+ |
| Frontend API Service Files | 9 |
| Frontend-Connected Endpoints | ~45 (43%) |
| **Unconnected Endpoints** | **60+ (57%)** |
| Total Agents | 18 |
| Agents with Full Frontend Exposure | 4 (22%) |
| **Agents with No Frontend Exposure** | **5 (28%)** |
| Agents with Partial Exposure | 9 (50%) |

### Risk Assessment

| Risk Level | Finding |
|------------|---------|
| **HIGH** | 5 agents produce outputs that never reach the frontend |
| **HIGH** | 57% of backend endpoints have no frontend integration |
| **MEDIUM** | Causal inference capabilities largely inaccessible to users |
| **MEDIUM** | A/B testing system (24 endpoints) completely disconnected |
| **LOW** | Audit chain (compliance/traceability) not exposed to users |

---

## 1. Backend API Coverage

### 1.1 Backend Route Files

| Route File | Prefix | Endpoints | Frontend Client | Status |
|------------|--------|-----------|-----------------|--------|
| `agents.py` | `/api/agents` | 1 | None | **MISSING** |
| `audit.py` | `/api/audit` | 4 | None | **MISSING** |
| `causal.py` | `/api/causal` | 10 | None | **MISSING** |
| `chatbot_graph.py` | internal | - | CopilotKit | Internal |
| `cognitive.py` | `/api/cognitive` | 5 | `cognitive.ts` | Partial |
| `copilotkit.py` | `/api/copilotkit` | 6 | CopilotKit SDK | SDK-only |
| `digital_twin.py` | `/api/digital-twin` | 8 | `digital-twin.ts` | Partial |
| `experiments.py` | `/api/experiments` | 24 | None | **MISSING** |
| `explain.py` | `/api/explain` | 5 | `explain.ts` | Complete |
| `graph.py` | `/api/graph` | 11 | `graph.ts` | Complete |
| `kpi.py` | `/api/kpis` | 8 | `kpi.ts` | Complete |
| `memory.py` | `/api/memory` | 6 | `memory.ts` | Complete |
| `monitoring.py` | `/api/monitoring` | 22 | `monitoring.ts` | Complete |
| `predictions.py` | `/api/models` | 5 | `predictions.ts` | Complete |
| `rag.py` | `/api/rag` | 6 | `rag.ts` | Complete |

### 1.2 Backend Endpoints Without Frontend Clients

#### Critical Gap: Causal Inference API (`/api/causal/*`)

| Endpoint | Method | Purpose | Impact |
|----------|--------|---------|--------|
| `/causal/hierarchical/analyze` | POST | Hierarchical CATE analysis | Users cannot trigger structured causal analysis |
| `/causal/route` | POST | Route query to appropriate library | Library selection opaque to users |
| `/causal/pipeline/sequential` | POST | Sequential multi-library pipeline | Advanced pipeline unavailable |
| `/causal/pipeline/parallel` | POST | Parallel multi-library analysis | Parallel analysis unavailable |
| `/causal/validate` | POST | Cross-library validation | Validation results hidden |
| `/causal/estimators` | GET | List available estimators | Estimator options unknown to users |
| `/causal/health` | GET | Component health check | Causal system health not visible |

#### Critical Gap: Experiments API (`/api/experiments/*`)

| Endpoint Group | Count | Purpose | Impact |
|----------------|-------|---------|--------|
| Randomization | 3 | Simple/stratified/block randomization | A/B test setup impossible from UI |
| Enrollment | 4 | Unit enrollment management | Participant tracking unavailable |
| Analysis | 4 | Interim/final analysis | Results inaccessible |
| Monitoring | 5 | SRM detection, fidelity checks | Experiment health hidden |
| Results | 4 | Final results, exports | Outcomes not displayable |
| Digital Twin Validation | 4 | Pre-screen experiments | Simulation validation hidden |

#### Medium Gap: Audit Chain API (`/api/audit/*`)

| Endpoint | Method | Purpose | Impact |
|----------|--------|---------|--------|
| `/audit/workflow/{id}` | GET | Workflow audit entries | Compliance trail inaccessible |
| `/audit/workflow/{id}/verify` | GET | Cryptographic verification | Tamper detection unavailable |
| `/audit/workflow/{id}/summary` | GET | Workflow metrics | Aggregated audit data hidden |
| `/audit/recent` | GET | Recent workflows | Activity monitoring unavailable |

#### Low Gap: Agent Status API (`/api/agents/*`)

| Endpoint | Method | Purpose | Impact |
|----------|--------|---------|--------|
| `/agents/status` | GET | 18-agent tier status | Agent health monitoring unavailable |

---

## 2. Agent Output Routing Analysis

### 2.1 Agent Tier Coverage Summary

```
TIER 0 - ML Foundation (7 agents)     → INTERNAL ONLY (acceptable)
TIER 1 - Orchestrator (2 agents)      → FULLY EXPOSED via CopilotKit
TIER 2 - Causal Analytics (3 agents)  → 1/3 PARTIAL, 2/3 MISSING
TIER 3 - Monitoring (3 agents)        → 2/3 EXPOSED, 1/3 MISSING
TIER 4 - ML Predictions (2 agents)    → 1/2 EXPOSED, 1/2 MISSING
TIER 5 - Self-Improvement (2 agents)  → 0/2 MISSING (critical)
```

### 2.2 Agents with No Frontend Path

#### Gap Analyzer (Tier 2) - **HIGH PRIORITY**

**State File:** `src/agents/gap_analyzer/state.py`

| Output Field | Type | Business Value |
|--------------|------|----------------|
| `prioritized_opportunities` | `List[PrioritizedOpportunity]` | ROI opportunities ranked by value |
| `quick_wins` | `List[Opportunity]` | Low-effort, high-impact actions |
| `strategic_bets` | `List[Opportunity]` | High-effort, high-reward initiatives |
| `total_addressable_value` | `float` | Total potential value identified |
| `executive_summary` | `str` | Natural language summary |

**Missing Infrastructure:**
- No `/api/gaps/*` endpoint
- No `frontend/src/api/gaps.ts` client
- No `useGapAnalysis()` hook
- No UI component to display opportunities

**Business Impact:** Users cannot see ROI optimization recommendations.

---

#### Heterogeneous Optimizer (Tier 2) - **HIGH PRIORITY**

**State File:** `src/agents/heterogeneous_optimizer/state.py`

| Output Field | Type | Business Value |
|--------------|------|----------------|
| `cate_by_segment` | `Dict[str, CATEResult]` | Treatment effects by segment |
| `policy_recommendations` | `List[PolicyRec]` | Targeting recommendations |
| `high_responders` | `List[Segment]` | Best treatment targets |
| `low_responders` | `List[Segment]` | Avoid targeting these |
| `targeting_efficiency` | `float` | Efficiency metric |

**Missing Infrastructure:**
- No `/api/heterogeneous/*` endpoint
- No `frontend/src/api/heterogeneous.ts` client
- No `useHeterogeneousOptimization()` hook
- No segment analysis UI

**Business Impact:** Users cannot see which segments respond best to treatments.

---

#### Resource Optimizer (Tier 4) - **HIGH PRIORITY**

**State File:** `src/agents/resource_optimizer/state.py`

| Output Field | Type | Business Value |
|--------------|------|----------------|
| `optimal_allocations` | `List[AllocationResult]` | Budget allocation recommendations |
| `projected_roi` | `float` | Expected return on investment |
| `scenario_results` | `List[ScenarioResult]` | What-if analysis results |
| `recommendations` | `List[Recommendation]` | Actionable next steps |

**Missing Infrastructure:**
- No `/api/resource-optimize/*` endpoint
- No `frontend/src/api/resource-optimizer.ts` client
- No `useResourceOptimizer()` hook
- No allocation optimization UI

**Business Impact:** Users cannot access budget optimization recommendations.

---

#### Feedback Learner (Tier 5) - **MEDIUM PRIORITY**

**State File:** `src/agents/feedback_learner/state.py`

| Output Field | Type | Business Value |
|--------------|------|----------------|
| `learning_recommendations` | `List[LearningRec]` | System improvement suggestions |
| `detected_patterns` | `List[Pattern]` | Usage patterns identified |
| `proposed_updates` | `List[Update]` | Proposed system changes |
| `applied_updates` | `List[Update]` | Changes already applied |

**Missing Infrastructure:**
- No `/api/feedback-learning/*` endpoint
- No frontend client
- No learning insights UI

**Business Impact:** Self-improvement loop invisible to users; no transparency into system learning.

---

#### Health Score Agent (Tier 3) - **MEDIUM PRIORITY**

**State File:** `src/agents/health_score/state.py`

| Output Field | Type | Business Value |
|--------------|------|----------------|
| `overall_health_score` | `float (0-100)` | System health metric |
| `health_grade` | `str (A-F)` | Letter grade |
| `component_statuses` | `List[ComponentStatus]` | Per-component health |
| `critical_issues` | `List[Issue]` | Urgent problems |

**Missing Infrastructure:**
- No public `/api/health-score/*` endpoint
- No frontend client
- No system health dashboard

**Business Impact:** Users cannot monitor overall system health.

---

### 2.3 Agents with Partial Exposure

#### Causal Impact (Tier 2)

**Current State:**
- Outputs accessible only through CopilotKit chat streaming
- No dedicated structured API endpoint
- Results not reusable/exportable

**Missing:**
- Dedicated `/api/causal-impact/*` endpoint for structured results
- Frontend hook for direct access
- Export functionality

---

#### Experiment Designer (Tier 3)

**Current State:**
- Experiment execution endpoints exist (`/api/experiments/*`)
- Design output (templates, power analysis) not separately exposed

**Missing:**
- `/api/experiments/design` endpoint for design phase
- Frontend integration for experiment design workflow

---

#### Explainer Agent vs SHAP Explanations (Tier 5)

**Current Confusion:**
- `/api/explain` endpoint serves **SHAP explanations** (feature importance)
- Explainer Agent produces **natural language explanations**
- These are different capabilities with the same name

**Missing:**
- Separate `/api/nl-explain/*` endpoint for Explainer Agent
- Frontend distinction between SHAP and NL explanations

---

## 3. Frontend Integration Gaps

### 3.1 Missing Frontend API Clients

| Missing Client | Backend Route | Priority | Reason |
|----------------|---------------|----------|--------|
| `agents.ts` | `agents.py` | Low | Agent status monitoring |
| `audit.ts` | `audit.py` | Medium | Compliance visibility |
| `causal.ts` | `causal.py` | **High** | Core causal inference capability |
| `experiments.ts` | `experiments.py` | **High** | A/B testing capability |
| `gaps.ts` | None (missing) | **High** | Gap analysis results |
| `heterogeneous.ts` | None (missing) | **High** | Segment optimization |
| `resource-optimizer.ts` | None (missing) | **High** | Budget optimization |
| `feedback-learning.ts` | None (missing) | Medium | Learning transparency |
| `health-score.ts` | None (missing) | Medium | System health |

### 3.2 Incomplete Frontend Implementations

#### `cognitive.ts` - Partial

| Endpoint | Implemented | Status |
|----------|-------------|--------|
| `/cognitive/rag` | Yes | Working |
| `/cognitive/sessions` (create/list) | Yes | Working |
| `/cognitive/sessions/{id}` (get) | **No** | Missing |
| `/cognitive/sessions/{id}` (delete) | **No** | Missing |
| `/cognitive/query` (primary) | **No** | Missing |

#### `digital-twin.ts` - Partial

| Endpoint | Implemented | Status |
|----------|-------------|--------|
| `/digital-twin/simulate` | Yes | Working |
| `/digital-twin/simulations` | Yes | Working |
| `/digital-twin/compare` | **No** | Missing |
| `/digital-twin/models/*` | **No** | Missing (6 endpoints) |
| `/digital-twin/validate` | **No** | Missing |

### 3.3 Unused Response Fields

Several backend responses include fields that the frontend ignores:

| Endpoint | Unused Field | Value |
|----------|--------------|-------|
| KPI Results | `confidence_interval` | Statistical confidence |
| KPI Results | `causal_library_used` | Provenance info |
| Monitoring | `refutation_results` | Robustness tests |
| Monitoring | `recommendations` | Suggested actions |
| Graph Search | `search_relevance_scores` | Result quality |
| Causal | `warnings`, `errors` | Important alerts |

---

## 4. Recommendations

### 4.1 High Priority (Complete Agent Coverage)

#### Recommendation 1: Create Gap Analysis API & Frontend

**Backend:**
```
src/api/routes/gaps.py
├── POST /api/gaps/analyze          # Run gap analysis
├── GET  /api/gaps/{analysis_id}    # Get results
├── GET  /api/gaps/opportunities    # List opportunities
└── GET  /api/gaps/health           # Service health
```

**Frontend:**
```
frontend/src/api/gaps.ts            # API client
frontend/src/hooks/api/use-gaps.ts  # React Query hooks
frontend/src/pages/GapAnalysis.tsx  # Results page
```

**Effort:** 3-5 days

---

#### Recommendation 2: Create Heterogeneous Optimization API & Frontend

**Backend:**
```
src/api/routes/heterogeneous.py
├── POST /api/segments/analyze      # Run segment analysis
├── GET  /api/segments/{id}         # Get results
├── GET  /api/segments/policies     # Get recommendations
└── GET  /api/segments/health       # Service health
```

**Frontend:**
```
frontend/src/api/segments.ts
frontend/src/hooks/api/use-segments.ts
frontend/src/pages/SegmentAnalysis.tsx
```

**Effort:** 3-5 days

---

#### Recommendation 3: Create Resource Optimization API & Frontend

**Backend:**
```
src/api/routes/resource_optimizer.py
├── POST /api/optimize/allocations  # Run optimization
├── POST /api/optimize/scenarios    # What-if analysis
├── GET  /api/optimize/{id}         # Get results
└── GET  /api/optimize/health       # Service health
```

**Frontend:**
```
frontend/src/api/resource-optimizer.ts
frontend/src/hooks/api/use-resource-optimizer.ts
frontend/src/pages/ResourceOptimization.tsx
```

**Effort:** 3-5 days

---

#### Recommendation 4: Create Causal Inference Frontend Client

**Frontend only (backend exists):**
```
frontend/src/api/causal.ts
├── hierarchicalAnalyze()
├── routeQuery()
├── runSequentialPipeline()
├── runParallelPipeline()
├── validateAcrossLibraries()
├── listEstimators()
└── getCausalHealth()

frontend/src/hooks/api/use-causal.ts
frontend/src/pages/CausalAnalysis.tsx
```

**Effort:** 2-3 days

---

#### Recommendation 5: Create Experiments Frontend Client

**Frontend only (backend exists):**
```
frontend/src/api/experiments.ts
├── createExperiment()
├── randomize()
├── enroll()
├── getAssignments()
├── runInterimAnalysis()
├── getResults()
├── checkSRM()
└── ... (24 endpoints)

frontend/src/hooks/api/use-experiments.ts
frontend/src/pages/Experiments.tsx
```

**Effort:** 3-4 days

---

### 4.2 Medium Priority (Visibility & Compliance)

#### Recommendation 6: Create Feedback Learning API & Frontend

Expose self-improvement insights to users for transparency.

**Effort:** 2-3 days

---

#### Recommendation 7: Create Health Score Dashboard

Create a system health dashboard showing all agent statuses.

**Effort:** 2-3 days

---

#### Recommendation 8: Create Audit Chain Frontend

Enable compliance officers to view workflow audit trails.

**Effort:** 2-3 days

---

### 4.3 Low Priority (Refinements)

#### Recommendation 9: Complete Partial Implementations

- Finish `cognitive.ts` (get/delete session, primary query)
- Finish `digital-twin.ts` (compare, models, validate)

**Effort:** 1-2 days

---

#### Recommendation 10: Expose Unused Response Fields

Add UI elements to display:
- Confidence intervals on KPI results
- Recommendations from monitoring alerts
- Relevance scores in search results
- Warnings/errors from causal analysis

**Effort:** 1-2 days

---

#### Recommendation 11: Distinguish Explainer Agent from SHAP

Create separate endpoint/UI for Explainer Agent's natural language explanations vs SHAP feature importance.

**Effort:** 1-2 days

---

## 5. Implementation Roadmap

### Phase 1: Critical Agent Exposure (Weeks 1-2)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Gap Analysis API + Frontend | High | 4d | None |
| Heterogeneous Optimization API + Frontend | High | 4d | None |
| Resource Optimization API + Frontend | High | 4d | None |
| Causal Frontend Client | High | 3d | None |

**Outcome:** All Tier 2 and Tier 4 agent outputs accessible to users.

### Phase 2: Experiment & Compliance (Weeks 3-4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Experiments Frontend Client | High | 4d | None |
| Feedback Learning API + Frontend | Medium | 3d | None |
| Audit Chain Frontend | Medium | 3d | None |

**Outcome:** A/B testing accessible; compliance visibility enabled.

### Phase 3: Polish & Refinement (Week 5)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Health Score Dashboard | Medium | 2d | None |
| Complete cognitive.ts | Low | 1d | None |
| Complete digital-twin.ts | Low | 1d | None |
| Expose unused response fields | Low | 2d | Phase 1-2 |
| Explainer/SHAP distinction | Low | 1d | None |

**Outcome:** Full coverage; all agent outputs visible.

---

## 6. Validation Checklist

After implementation, verify:

- [ ] All 18 agents have at least indirect frontend exposure
- [ ] All backend endpoints have corresponding frontend clients
- [ ] All response fields are consumed or intentionally excluded
- [ ] Agent status monitoring available (`/agents/status`)
- [ ] Audit trail accessible for compliance
- [ ] Experiment lifecycle manageable from UI
- [ ] Causal inference tools accessible without chat interface
- [ ] System health dashboard operational
- [ ] Self-improvement insights visible to users

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                          (React + TypeScript)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  kpi.ts     │ │cognitive.ts │ │monitoring.ts│ │ predict.ts  │       │
│  │     ✅      │ │     ⚠️      │ │     ✅      │ │     ✅      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  graph.ts   │ │  memory.ts  │ │  explain.ts │ │   rag.ts    │       │
│  │     ✅      │ │     ✅      │ │     ✅      │ │     ✅      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ digital-    │ │  causal.ts  │ │experiments  │ │  gaps.ts    │       │
│  │  twin.ts ⚠️ │ │     ❌      │ │   .ts ❌    │ │     ❌      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ segments.ts │ │ resource-   │ │ feedback-   │ │ health-     │       │
│  │     ❌      │ │optimizer ❌ │ │learning ❌  │ │  score ❌   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  Legend: ✅ Complete  ⚠️ Partial  ❌ Missing                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           BACKEND API                                    │
│                       (FastAPI + Pydantic)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   kpi.py    │ │cognitive.py │ │monitoring.py│ │predictions  │       │
│  │     ✅      │ │     ✅      │ │     ✅      │ │   .py ✅    │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  graph.py   │ │  memory.py  │ │  explain.py │ │   rag.py    │       │
│  │     ✅      │ │     ✅      │ │     ✅      │ │     ✅      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │digital_twin │ │  causal.py  │ │experiments  │ │  audit.py   │       │
│  │   .py ✅    │ │     ✅      │ │   .py ✅    │ │     ✅      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  agents.py  │ │copilotkit.py│ │  gaps.py    │ │ segments.py │       │
│  │     ✅      │ │     ✅      │ │     ❌      │ │     ❌      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                                          │
│  ┌─────────────┐ ┌─────────────┐                                        │
│  │ resource_   │ │ feedback_   │                                        │
│  │optimizer ❌ │ │learning ❌  │                                        │
│  └─────────────┘ └─────────────┘                                        │
│                                                                          │
│  Legend: ✅ Exists  ❌ Missing                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          AGENT LAYER                                     │
│                  (18 Agents in 6 Tiers)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TIER 0 - ML Foundation (7)        TIER 1 - Orchestrator (2)           │
│  ┌───────────────────────┐         ┌───────────────────────┐           │
│  │ Internal pipeline     │         │ Orchestrator      ✅  │           │
│  │ (no frontend needed)  │         │ Tool Composer     ✅  │           │
│  └───────────────────────┘         └───────────────────────┘           │
│                                                                          │
│  TIER 2 - Causal Analytics (3)     TIER 3 - Monitoring (3)             │
│  ┌───────────────────────┐         ┌───────────────────────┐           │
│  │ Causal Impact     ⚠️  │         │ Drift Monitor     ✅  │           │
│  │ Gap Analyzer      ❌  │         │ Experiment Design ⚠️  │           │
│  │ Heterogeneous Opt ❌  │         │ Health Score      ❌  │           │
│  └───────────────────────┘         └───────────────────────┘           │
│                                                                          │
│  TIER 4 - ML Predictions (2)       TIER 5 - Self-Improvement (2)       │
│  ┌───────────────────────┐         ┌───────────────────────┐           │
│  │ Prediction Synth  ✅  │         │ Explainer         ⚠️  │           │
│  │ Resource Optimizer❌  │         │ Feedback Learner  ❌  │           │
│  └───────────────────────┘         └───────────────────────┘           │
│                                                                          │
│  Legend: ✅ Exposed  ⚠️ Partial  ❌ Not Exposed                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Appendix

### A. File Inventory

**Backend Route Files (20):**
- `src/api/routes/agents.py`
- `src/api/routes/audit.py`
- `src/api/routes/causal.py`
- `src/api/routes/chatbot_dspy.py`
- `src/api/routes/chatbot_graph.py`
- `src/api/routes/chatbot_state.py`
- `src/api/routes/chatbot_tools.py`
- `src/api/routes/chatbot_tracer.py`
- `src/api/routes/cognitive.py`
- `src/api/routes/copilotkit.py`
- `src/api/routes/digital_twin.py`
- `src/api/routes/experiments.py`
- `src/api/routes/explain.py`
- `src/api/routes/graph.py`
- `src/api/routes/kpi.py`
- `src/api/routes/memory.py`
- `src/api/routes/monitoring.py`
- `src/api/routes/predictions.py`
- `src/api/routes/rag.py`

**Frontend API Clients (9):**
- `frontend/src/api/cognitive.ts`
- `frontend/src/api/digital-twin.ts`
- `frontend/src/api/explain.ts`
- `frontend/src/api/graph.ts`
- `frontend/src/api/kpi.ts`
- `frontend/src/api/memory.ts`
- `frontend/src/api/monitoring.ts`
- `frontend/src/api/predictions.ts`
- `frontend/src/api/rag.ts`

### B. Agent State Files

- `src/agents/orchestrator/state.py`
- `src/agents/causal_impact/state.py`
- `src/agents/gap_analyzer/state.py`
- `src/agents/heterogeneous_optimizer/state.py`
- `src/agents/drift_monitor/state.py`
- `src/agents/experiment_designer/state.py`
- `src/agents/health_score/state.py`
- `src/agents/prediction_synthesizer/state.py`
- `src/agents/resource_optimizer/state.py`
- `src/agents/explainer/state.py`
- `src/agents/feedback_learner/state.py`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Status:** Draft - Pending Review
