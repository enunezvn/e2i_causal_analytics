# Frontend Integration Implementation Plan

**Date:** 2026-01-19
**Branch:** `claude/audit-api-frontend-routing-OhAkA`
**Based On:** `docs/AUDIT_API_FRONTEND_ROUTING_PLAN.md`
**Status:** Recommended Changes with Testing Strategy

---

## Executive Summary

After comprehensive review of the audit findings against user personas, business value, and product completeness, this document identifies which frontend integrations are **truly merited** versus **nice-to-have**.

### Key Determination Criteria

| Criterion | Weight | Question |
|-----------|--------|----------|
| **Broken Workflow** | Critical | Does a UI page exist that needs data but can't get it? |
| **Core Value Proposition** | High | Does the gap undermine E2I's differentiation? |
| **User Decision Support** | High | Does the data help users make commercial decisions? |
| **Alternative Access** | Medium | Can users get this via chat, MLflow, or other means? |
| **Compliance/Audit** | Medium | Is this required for regulatory or audit purposes? |

### Verdict Summary

| Gap | Verdict | Rationale |
|-----|---------|-----------|
| **Gap Analyzer** | **MUST IMPLEMENT** | Core value prop - ROI identification |
| **Experiments API** | **MUST IMPLEMENT** | Broken workflow - A/B test lifecycle |
| **Causal API Frontend** | **MUST IMPLEMENT** | Core value prop - structured causal access |
| **Resource Optimizer** | **MUST IMPLEMENT** | Core value prop - budget allocation |
| **Heterogeneous Optimizer** | **MUST IMPLEMENT** | Core value prop - segment targeting |
| **Health Score Dashboard** | **SHOULD IMPLEMENT** | Operational visibility gap |
| Feedback Learner UI | Nice-to-have | Transparency only, not decision-critical |
| Audit Chain UI | Conditional | Only if compliance requires it |
| Tier 0 ML Pipeline UI | Do Not Implement | Internal MLOps, use MLflow directly |

---

## Part 1: Merited Frontend Changes

### 1.1 Gap Analyzer Integration - **MUST IMPLEMENT**

#### Why It's Merited

**Business Case:**
- Gap Analyzer identifies **ROI opportunities** - the core reason pharma commercial teams use analytics
- Outputs: `prioritized_opportunities`, `quick_wins`, `strategic_bets`, `total_addressable_value`
- Users cannot see WHERE to invest marketing dollars without this

**User Persona Impact:**
- Brand Manager: Needs quick wins for quarterly targets
- Commercial Strategy: Needs strategic bets for annual planning
- Field Operations: Needs prioritized HCP lists

**Current State:**
- Agent exists with 132 tests passing
- Backend agent fully implemented
- **No API endpoint** - backend capability completely hidden
- **No frontend page** - users cannot access gap analysis at all

**Alternative Access:** None. Chat interface is not sufficient for exporting prioritized opportunity lists.

#### Implementation Scope

**Backend (New):**
```
src/api/routes/gaps.py
├── POST /api/gaps/analyze          # Trigger gap analysis
│   Input: { brand, region, date_range, min_opportunity_value }
│   Output: { analysis_id, status }
│
├── GET  /api/gaps/{analysis_id}    # Get analysis results
│   Output: GapAnalysisResult (full state)
│
├── GET  /api/gaps/opportunities    # List opportunities (paginated)
│   Query: brand, category, min_value, sort_by
│   Output: List[PrioritizedOpportunity]
│
├── GET  /api/gaps/quick-wins       # Filter to quick wins only
│   Output: List[Opportunity] where effort=low, impact=high
│
└── GET  /api/gaps/health           # Service health
```

**Frontend (New):**
```
frontend/src/api/gaps.ts                    # API client
frontend/src/hooks/api/use-gaps.ts          # React Query hooks
frontend/src/pages/GapAnalysis.tsx          # Main page
frontend/src/components/gaps/
├── OpportunityCard.tsx                     # Individual opportunity
├── OpportunityTable.tsx                    # Sortable/filterable table
├── QuickWinsPanel.tsx                      # Quick wins highlight
├── ROISummary.tsx                          # Total addressable value
└── OpportunityFilters.tsx                  # Brand/region/value filters
```

**Navigation:** Add to Analytics section in sidebar

#### Testing Strategy

**Unit Tests:**
- `tests/unit/api/test_gaps_routes.py` - Endpoint input validation, response schemas
- `tests/unit/frontend/gaps.test.tsx` - Component rendering, filter interactions

**Integration Tests:**
- `tests/integration/test_gaps_api.py` - Full flow: trigger → poll → retrieve
- Test pagination, filtering, sorting

**E2E Tests:**
- `tests/e2e/test_gap_analysis_workflow.py`
  1. Navigate to Gap Analysis page
  2. Select brand filter (Kisqali)
  3. Trigger analysis
  4. Verify loading state
  5. Verify opportunity cards render
  6. Click "Quick Wins" tab
  7. Verify filtered results
  8. Export to CSV

**Acceptance Criteria:**
- [ ] User can trigger gap analysis for any brand
- [ ] Opportunities display with value, effort, impact scores
- [ ] Quick wins tab shows low-effort, high-impact items
- [ ] Total addressable value displays prominently
- [ ] Results exportable to CSV
- [ ] Loading/error states handled gracefully

---

### 1.2 Experiments API Frontend - **MUST IMPLEMENT**

#### Why It's Merited

**Business Case:**
- A/B testing is how pharma proves what works (HCP targeting, messaging, etc.)
- 24 backend endpoints exist - **completely inaccessible from UI**
- This is a **BROKEN WORKFLOW**: Digital Twin page exists for simulation but users cannot design/manage actual experiments

**User Persona Impact:**
- Brand Manager: Needs to prove intervention ROI for budget defense
- Commercial Strategy: Needs to design targeting experiments
- Analytics Team: Needs to monitor experiment health (SRM, enrollment)

**Current State:**
- Full backend exists: `src/api/routes/experiments.py` (24 endpoints)
- Experiment Designer agent exists with 209 tests
- Digital Twin page exists but only for simulation preview
- **No frontend client for experiments** - cannot design, enroll, or analyze A/B tests

**Alternative Access:** None. Cannot run experiments via chat.

#### Implementation Scope

**Frontend (New) - Backend Already Exists:**
```
frontend/src/api/experiments.ts             # API client (24 endpoints)
frontend/src/hooks/api/use-experiments.ts   # React Query hooks
frontend/src/pages/Experiments.tsx          # Experiment management
frontend/src/pages/ExperimentDetail.tsx     # Single experiment view
frontend/src/components/experiments/
├── ExperimentWizard.tsx                    # Multi-step creation
├── RandomizationConfig.tsx                 # Stratification settings
├── EnrollmentTracker.tsx                   # Real-time enrollment
├── SRMAlert.tsx                            # Sample ratio mismatch warning
├── ResultsChart.tsx                        # Effect estimate visualization
├── PowerAnalysis.tsx                       # Power/sample size display
└── ExperimentTimeline.tsx                  # Lifecycle visualization
```

**Navigation:** Add to System section or new "Experimentation" section

#### Testing Strategy

**Unit Tests:**
- `tests/unit/frontend/experiments.test.tsx` - Wizard steps, validation
- Mock API responses for all 24 endpoints

**Integration Tests:**
- `tests/integration/test_experiment_lifecycle.py`
  1. Create experiment with randomization
  2. Enroll units
  3. Check assignments
  4. Trigger interim analysis
  5. Detect SRM if present
  6. Get final results

**E2E Tests:**
- `tests/e2e/test_experiment_workflow.py`
  1. Create experiment via wizard
  2. Configure stratification
  3. Start enrollment
  4. Monitor enrollment dashboard
  5. Run interim analysis
  6. View results with confidence intervals

**Acceptance Criteria:**
- [ ] User can create experiment via step-by-step wizard
- [ ] Randomization options: simple, stratified, blocked
- [ ] Real-time enrollment tracking
- [ ] SRM detection with actionable alerts
- [ ] Interim and final analysis accessible
- [ ] Results show effect estimate + CI + p-value
- [ ] Integration with Digital Twin for pre-screening

---

### 1.3 Causal API Frontend - **MUST IMPLEMENT**

#### Why It's Merited

**Business Case:**
- Causal inference is E2I's **primary value proposition**
- 10 backend endpoints exist for structured causal analysis
- Currently only accessible via chat - cannot export, compare, or reuse results

**User Persona Impact:**
- All users: Need structured causal results, not just chat responses
- Analytics Team: Need to compare analyses across brands/time periods
- Brand Manager: Need to export causal evidence for leadership

**Current State:**
- Backend: `src/api/routes/causal.py` (10 endpoints including hierarchical CATE, multi-library routing)
- Causal Discovery page exists but only shows DAG visualization
- **No frontend client** - structured causal analysis completely hidden

**Alternative Access:** Chat provides results but:
- Cannot export structured data
- Cannot compare multiple analyses
- Cannot trigger specific analysis types (hierarchical, parallel)

#### Implementation Scope

**Frontend (New) - Backend Already Exists:**
```
frontend/src/api/causal.ts                  # API client
frontend/src/hooks/api/use-causal.ts        # React Query hooks
frontend/src/pages/CausalAnalysis.tsx       # New dedicated page
frontend/src/components/causal/
├── AnalysisSelector.tsx                    # Choose analysis type
├── HierarchicalCATEView.tsx                # CATE by segment
├── LibraryComparison.tsx                   # DoWhy vs EconML vs CausalML
├── RefutationPanel.tsx                     # Robustness test results
├── EstimatorSelector.tsx                   # Choose estimator
├── EffectSizeChart.tsx                     # Visualization
└── CausalExport.tsx                        # Export to CSV/JSON
```

**Enhance Existing:**
- Causal Discovery page: Add link to structured analysis
- Knowledge Graph: Add causal analysis actions on nodes

#### Testing Strategy

**Unit Tests:**
- `tests/unit/frontend/causal.test.tsx` - Analysis type selection, result rendering

**Integration Tests:**
- `tests/integration/test_causal_frontend_backend.py`
  1. Call hierarchical analysis endpoint
  2. Verify response schema matches frontend expectations
  3. Call parallel pipeline
  4. Verify multi-library results render

**E2E Tests:**
- `tests/e2e/test_causal_analysis_workflow.py`
  1. Navigate to Causal Analysis
  2. Select analysis type (hierarchical CATE)
  3. Configure parameters (treatment, outcome, segments)
  4. Run analysis
  5. View results by segment
  6. Compare DoWhy vs EconML
  7. Export results

**Acceptance Criteria:**
- [ ] User can select analysis type (hierarchical, parallel, sequential)
- [ ] Available estimators displayed
- [ ] Results show ATE, CATE by segment, confidence intervals
- [ ] Refutation tests displayed
- [ ] Library comparison available
- [ ] Results exportable

---

### 1.4 Resource Optimizer Integration - **MUST IMPLEMENT**

#### Why It's Merited

**Business Case:**
- Resource Optimizer answers: "How should I allocate my budget?"
- Core commercial decision - directly affects marketing ROI
- Outputs: `optimal_allocations`, `projected_roi`, `scenario_results`

**User Persona Impact:**
- Brand Manager: Needs to justify budget allocation
- Commercial Strategy: Needs scenario planning for budget cycles
- Finance: Needs projected ROI for investments

**Current State:**
- Agent exists with 75 tests passing
- State file: `src/agents/resource_optimizer/state.py`
- **No API endpoint** - cannot access optimization
- **No frontend page** - budget allocation invisible

**Alternative Access:** None. Too complex for chat interface.

#### Implementation Scope

**Backend (New):**
```
src/api/routes/resource_optimizer.py
├── POST /api/optimize/allocations      # Run optimization
│   Input: { brand, budget, channels, constraints }
│   Output: { optimization_id, status }
│
├── GET  /api/optimize/{id}             # Get results
│   Output: ResourceOptimizationResult
│
├── POST /api/optimize/scenarios        # What-if analysis
│   Input: { base_allocation, scenarios[] }
│   Output: List[ScenarioResult]
│
└── GET  /api/optimize/health           # Service health
```

**Frontend (New):**
```
frontend/src/api/resource-optimizer.ts
frontend/src/hooks/api/use-resource-optimizer.ts
frontend/src/pages/ResourceOptimization.tsx
frontend/src/components/optimizer/
├── AllocationInput.tsx                 # Budget/channel configuration
├── ConstraintBuilder.tsx               # Set constraints
├── AllocationChart.tsx                 # Sankey or bar chart
├── ROIProjection.tsx                   # Projected returns
├── ScenarioComparison.tsx              # Compare scenarios
└── RecommendationList.tsx              # Actionable suggestions
```

**Navigation:** Add to Analytics section

#### Testing Strategy

**Unit Tests:**
- `tests/unit/api/test_optimizer_routes.py` - Input validation, response schemas
- `tests/unit/frontend/optimizer.test.tsx` - Allocation inputs, constraint builder

**Integration Tests:**
- `tests/integration/test_optimizer_api.py`
  1. Submit optimization request
  2. Poll for completion
  3. Verify allocation sums to budget
  4. Run scenario comparison

**E2E Tests:**
- `tests/e2e/test_resource_optimization_workflow.py`
  1. Navigate to Resource Optimization
  2. Enter budget amount
  3. Select channels (rep visits, digital, events)
  4. Set constraints (min/max per channel)
  5. Run optimization
  6. View recommended allocation
  7. Compare 3 scenarios
  8. Export recommendation

**Acceptance Criteria:**
- [ ] User can input budget and channel options
- [ ] Constraints configurable (min/max, business rules)
- [ ] Optimal allocation displayed visually
- [ ] Projected ROI shown with confidence
- [ ] Scenario comparison available
- [ ] Recommendations actionable

---

### 1.5 Heterogeneous Optimizer Integration - **MUST IMPLEMENT**

#### Why It's Merited

**Business Case:**
- Answers: "Which segments respond best to treatment?"
- Core pharma commercial question - HCP targeting depends on this
- Outputs: `cate_by_segment`, `policy_recommendations`, `high_responders`, `targeting_efficiency`

**User Persona Impact:**
- Field Operations: Needs high-responder HCP lists
- Commercial Strategy: Needs segment-level targeting policy
- Brand Manager: Needs to justify segment prioritization

**Current State:**
- Agent exists with 100+ tests passing
- State file: `src/agents/heterogeneous_optimizer/state.py`
- **No API endpoint**
- **No frontend page**

**Alternative Access:** Chat can provide summary, but:
- Cannot export segment-level CATE
- Cannot get high/low responder lists
- Cannot compare policies

#### Implementation Scope

**Backend (New):**
```
src/api/routes/segments.py
├── POST /api/segments/analyze          # Run segment analysis
│   Input: { treatment, outcome, segment_vars, brand }
│   Output: { analysis_id, status }
│
├── GET  /api/segments/{id}             # Get full results
│   Output: HeterogeneousOptimizerResult
│
├── GET  /api/segments/{id}/high-responders  # Get high responder list
│   Output: List[Segment] with CATE > threshold
│
├── GET  /api/segments/{id}/policy      # Get targeting policy
│   Output: PolicyRecommendation
│
└── GET  /api/segments/health           # Service health
```

**Frontend (New):**
```
frontend/src/api/segments.ts
frontend/src/hooks/api/use-segments.ts
frontend/src/pages/SegmentAnalysis.tsx
frontend/src/components/segments/
├── SegmentSelector.tsx                 # Choose segmentation variables
├── CATEHeatmap.tsx                     # CATE by segment visualization
├── ResponderList.tsx                   # High/low responder lists
├── PolicyDisplay.tsx                   # Targeting recommendations
├── UpliftCurve.tsx                     # Cumulative uplift curve
└── SegmentExport.tsx                   # Export to CRM
```

**Navigation:** Add to Analytics section

#### Testing Strategy

**Unit Tests:**
- `tests/unit/api/test_segments_routes.py`
- `tests/unit/frontend/segments.test.tsx`

**Integration Tests:**
- `tests/integration/test_segments_api.py`
  1. Submit segment analysis
  2. Verify CATE values per segment
  3. Verify high-responders filter works
  4. Verify policy recommendations generated

**E2E Tests:**
- `tests/e2e/test_segment_analysis_workflow.py`
  1. Navigate to Segment Analysis
  2. Select treatment (HCP visit)
  3. Select outcome (NRx)
  4. Choose segment variables (specialty, volume)
  5. Run analysis
  6. View CATE heatmap
  7. Click "High Responders" tab
  8. Export list to CSV

**Acceptance Criteria:**
- [ ] User can select treatment, outcome, segments
- [ ] CATE displayed per segment with CI
- [ ] High/low responder lists filterable
- [ ] Targeting policy displayed
- [ ] Uplift curve visualization
- [ ] Export to CSV for CRM integration

---

### 1.6 Health Score Dashboard - **SHOULD IMPLEMENT**

#### Why It's Merited

**Business Case:**
- Users need visibility into system reliability
- Currently: System Health page exists but shows infrastructure, not agent health
- Health Score agent produces `overall_health_score`, `component_statuses`, `critical_issues`

**User Persona Impact:**
- All users: Need confidence system is working
- Analytics Team: Need to know if agents are healthy before analysis

**Current State:**
- Agent exists with 95 tests
- State file: `src/agents/health_score/state.py`
- System Health page exists but doesn't show agent health
- **Partial gap** - page exists, data missing

#### Implementation Scope

**Backend Enhancement:**
```
src/api/routes/health_score.py (or add to existing health route)
├── GET  /api/health/agents             # All agent health scores
├── GET  /api/health/agents/{tier}      # Agents by tier
├── GET  /api/health/overall            # Overall system health
└── GET  /api/health/issues             # Critical issues
```

**Frontend Enhancement:**
```
frontend/src/pages/SystemHealth.tsx     # Enhance existing
frontend/src/components/health/
├── AgentHealthGrid.tsx                 # Grid of agent health cards
├── HealthScoreGauge.tsx                # Overall score visualization
├── CriticalIssuesAlert.tsx             # Prominent issue display
└── HealthTrend.tsx                     # Health over time
```

#### Testing Strategy

**Integration Tests:**
- `tests/integration/test_health_api.py`
  1. Call health endpoint
  2. Verify all 18 agents represented
  3. Verify score in valid range (0-100)

**E2E Tests:**
- `tests/e2e/test_health_dashboard.py`
  1. Navigate to System Health
  2. Verify overall health score displayed
  3. Click into tier view
  4. Verify agent statuses shown
  5. Trigger unhealthy state (if possible)
  6. Verify alert displays

**Acceptance Criteria:**
- [ ] Overall health score (0-100, A-F grade) displayed
- [ ] Each tier shows agent status
- [ ] Critical issues prominently displayed
- [ ] Health trend over time visible

---

## Part 2: Conditional/Nice-to-Have Changes

### 2.1 Audit Chain UI - **CONDITIONAL**

**Verdict:** Implement only if compliance requires it

**Business Case:**
- Audit trail for compliance/regulatory purposes
- Backend exists: `src/api/routes/audit.py` (4 endpoints)

**When to Implement:**
- Required: If SOX, FDA 21 CFR Part 11, or internal audit requires UI access
- Not Required: If audit access via database queries is sufficient

**If Implemented:**
```
frontend/src/pages/AuditTrail.tsx
frontend/src/api/audit.ts
```

---

### 2.2 Feedback Learner UI - **NICE-TO-HAVE**

**Verdict:** Low priority - transparency benefit only

**Business Case:**
- Shows how system learns from feedback
- Users don't make decisions based on this
- Alternative: Feedback collection already in chat; learning is automatic

**If Implemented:**
```
frontend/src/pages/SystemLearning.tsx
├── LearningRecommendations.tsx
├── DetectedPatterns.tsx
└── KnowledgeUpdates.tsx
```

**Priority:** Defer to future release

---

### 2.3 Tier 0 ML Pipeline UI - **DO NOT IMPLEMENT**

**Verdict:** Internal tooling, not user-facing

**Rationale:**
- Tier 0 agents (scope_definer, data_preparer, model_trainer, etc.) are internal ML pipeline
- Target user: MLOps team, not commercial users
- **Alternative Access:** MLflow UI already exists for experiment tracking

**Action:** Document that MLflow at `http://138.197.4.36:5000` is the interface for ML pipeline monitoring.

---

## Part 3: Implementation Phases

### Phase 1: Core Commercial Value (Must-Have)

| Item | Backend | Frontend | Tests | Days |
|------|---------|----------|-------|------|
| Gap Analyzer API | New | New | 15+ | 4 |
| Experiments Frontend | Exists | New | 20+ | 5 |
| **Phase 1 Total** | | | | **9** |

**Rationale:** These unblock the most critical user workflows.

### Phase 2: Analytical Completeness (Must-Have)

| Item | Backend | Frontend | Tests | Days |
|------|---------|----------|-------|------|
| Causal API Frontend | Exists | New | 12+ | 3 |
| Resource Optimizer | New | New | 15+ | 4 |
| Heterogeneous Optimizer | New | New | 15+ | 4 |
| **Phase 2 Total** | | | | **11** |

**Rationale:** These complete the analytical toolkit.

### Phase 3: Operational Polish (Should-Have)

| Item | Backend | Frontend | Tests | Days |
|------|---------|----------|-------|------|
| Health Score Dashboard | Enhance | Enhance | 8+ | 2 |
| Partial Implementation Fixes | - | - | 5+ | 1 |
| **Phase 3 Total** | | | | **3** |

### Total Implementation Estimate

| Phase | Focus | Days |
|-------|-------|------|
| Phase 1 | Core Commercial Value | 9 |
| Phase 2 | Analytical Completeness | 11 |
| Phase 3 | Operational Polish | 3 |
| **Total** | | **23** |

---

## Part 4: Testing Strategy Summary

### Test Coverage Requirements

| Category | Minimum Coverage | Focus Areas |
|----------|------------------|-------------|
| Unit Tests | 80% | Input validation, component rendering |
| Integration Tests | 70% | API contract, data flow |
| E2E Tests | Key workflows | Happy path + error states |

### Test Naming Convention

```
tests/
├── unit/
│   ├── api/
│   │   ├── test_gaps_routes.py
│   │   ├── test_segments_routes.py
│   │   └── test_optimizer_routes.py
│   └── frontend/
│       ├── gaps.test.tsx
│       ├── experiments.test.tsx
│       ├── causal.test.tsx
│       ├── segments.test.tsx
│       └── optimizer.test.tsx
├── integration/
│   ├── test_gaps_api.py
│   ├── test_segments_api.py
│   ├── test_optimizer_api.py
│   ├── test_causal_frontend_backend.py
│   └── test_experiment_lifecycle.py
└── e2e/
    ├── test_gap_analysis_workflow.py
    ├── test_experiment_workflow.py
    ├── test_causal_analysis_workflow.py
    ├── test_segment_analysis_workflow.py
    ├── test_resource_optimization_workflow.py
    └── test_health_dashboard.py
```

### CI/CD Integration

```yaml
# .github/workflows/frontend-integration.yml
name: Frontend Integration Tests

on:
  push:
    paths:
      - 'frontend/src/api/**'
      - 'frontend/src/pages/**'
      - 'src/api/routes/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Backend Tests
        run: |
          pytest tests/unit/api/ -v -n 4
          pytest tests/integration/ -v -n 4

      - name: Frontend Tests
        run: |
          cd frontend
          npm run test:coverage

      - name: E2E Tests
        run: |
          # Start services
          docker-compose up -d
          # Run Playwright tests
          npx playwright test tests/e2e/
```

### Regression Test Matrix

| Feature | Pre-Integration Test | Post-Integration Test |
|---------|---------------------|----------------------|
| Gap Analyzer | Agent unit tests pass | API → Frontend renders |
| Experiments | 24 endpoints work | Wizard creates experiment |
| Causal | Analysis runs | Results export correctly |
| Segments | CATE calculates | High-responder list accurate |
| Optimizer | Optimization converges | Allocation sums to budget |

---

## Part 5: Acceptance Checklist

### Pre-Implementation

- [ ] Reviewed audit findings
- [ ] Agreed on priority order
- [ ] Assigned ownership per phase
- [ ] Set up feature flags (if gradual rollout)

### Per-Feature Acceptance

**Gap Analyzer:**
- [ ] API endpoints created and documented
- [ ] Frontend page renders with mock data
- [ ] Integration tests pass
- [ ] E2E workflow completes
- [ ] Product review approved

**Experiments:**
- [ ] Frontend client wraps all 24 endpoints
- [ ] Wizard creates valid experiments
- [ ] SRM detection works
- [ ] Results display correctly
- [ ] Product review approved

**Causal:**
- [ ] All analysis types accessible
- [ ] Export to CSV works
- [ ] Multi-library comparison displays
- [ ] Product review approved

**Segments:**
- [ ] CATE heatmap renders
- [ ] High-responder export works
- [ ] Policy recommendations display
- [ ] Product review approved

**Optimizer:**
- [ ] Allocation inputs work
- [ ] Constraints enforced
- [ ] Scenario comparison works
- [ ] Product review approved

**Health Dashboard:**
- [ ] Agent health grid shows all 18 agents
- [ ] Critical issues alert works
- [ ] Product review approved

### Post-Implementation

- [ ] All tests passing (unit, integration, E2E)
- [ ] Documentation updated (API docs, user guide)
- [ ] Feature flags removed (if used)
- [ ] Performance benchmarks met
- [ ] User training completed

---

## Part 6: Risk Assessment

### Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Backend agent state incompatible with API schema | Medium | High | Review state files before implementation |
| Frontend performance with large datasets | Medium | Medium | Implement pagination, virtualization |
| Experiment feature complexity | High | Medium | Phase wizard steps, start simple |
| Cross-feature dependencies | Low | Medium | Test integration points explicitly |

### Technical Debt to Address

1. **Inconsistent API patterns**: Some routes use `/api/v1/`, others use `/api/`. Standardize during implementation.

2. **Missing error schemas**: Define standard error response format for all new endpoints.

3. **Frontend API client inconsistencies**: Use consistent hook patterns (React Query) for all new clients.

---

## Conclusion

### What to Implement

| Priority | Item | Justification |
|----------|------|---------------|
| **P0** | Gap Analyzer API + Frontend | Core value prop, revenue impact |
| **P0** | Experiments Frontend | Broken workflow, A/B testing |
| **P1** | Causal API Frontend | Core value prop, structured access |
| **P1** | Resource Optimizer API + Frontend | Budget allocation decisions |
| **P1** | Heterogeneous Optimizer API + Frontend | Segment targeting |
| **P2** | Health Score Dashboard | Operational visibility |

### What NOT to Implement

| Item | Reason |
|------|--------|
| Tier 0 ML Pipeline UI | Use MLflow directly |
| Feedback Learner UI | Nice-to-have, not decision-critical |
| Low-level causal endpoints | Research-level, not for business users |

### Next Steps

1. Review this plan with product/engineering
2. Finalize priority order
3. Create JIRA tickets per feature
4. Begin Phase 1 implementation
5. Schedule user testing for Phase 1 completion

---

**Document Version:** 1.2
**Last Updated:** 2026-01-20
**Status:** ✅ Implementation Complete (Pending Deployment)

---

## Part 7: Implementation Progress Tracking

### Audit Findings (2026-01-20)

**Surprising Discovery:** Backend APIs and Frontend API clients were already implemented!

| Component | Gap Analyzer | Experiments | Causal | Resource Opt | Segments | Health Score |
|-----------|--------------|-------------|--------|--------------|----------|--------------|
| Backend API | ✅ `gaps.py` (25KB) | ✅ `experiments.py` (41KB) | ✅ `causal.py` (42KB) | ✅ `resource_optimizer.py` (28KB) | ✅ `segments.py` (30KB) | ✅ `health_score.py` (32KB) |
| Frontend API Client | ✅ `gaps.ts` (8KB) | ✅ `experiments.ts` (18KB) | ✅ `causal.ts` (16KB) | ✅ `resources.ts` (9KB) | ✅ `segments.ts` (9KB) | ✅ `health-score.ts` (13KB) |
| TypeScript Types | ✅ `gaps.ts` | ✅ `experiments.ts` | ✅ `causal.ts` | ✅ `resources.ts` | ✅ `segments.ts` | ✅ `health-score.ts` |
| React Query Hooks | ✅ `use-gaps.ts` | ✅ `use-experiments.ts` | ✅ `use-causal.ts` | ✅ `use-resources.ts` | ✅ `use-segments.ts` | ✅ `use-health-score.ts` |
| Page Component | ✅ `GapAnalysis.tsx` | ✅ `Experiments.tsx` | ✅ `CausalAnalysis.tsx` | ✅ `ResourceOptimization.tsx` | ✅ `SegmentAnalysis.tsx` | ✅ Enhanced |
| Route Entry | ✅ `/gap-analysis` | ✅ `/experiments` | ✅ `/causal-analysis` | ✅ `/resource-optimization` | ✅ `/segment-analysis` | ✅ `/system-health` |
| Sidebar Nav | ✅ Added | ✅ Added | ✅ Added | ✅ Added | ✅ Added | ✅ Exists |

**Revised Scope:** Only need to implement:
1. React Query hooks (6 files)
2. Page components (5 new + 1 enhancement)
3. Route entries (5 new routes)
4. Sidebar navigation updates

### Implementation Log

#### Session: 2026-01-20 21:53 UTC

**Status:** ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Audit existing code | ✅ Complete | Backend + API clients already exist |
| Create `use-gaps.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-gaps.ts` |
| Create `use-experiments.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-experiments.ts` |
| Create `use-causal.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-causal.ts` |
| Create `use-resources.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-resources.ts` |
| Create `use-segments.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-segments.ts` |
| Create `use-health-score.ts` hook | ✅ Complete | `frontend/src/hooks/api/use-health-score.ts` |
| Create `GapAnalysis.tsx` page | ✅ Complete | `frontend/src/pages/GapAnalysis.tsx` |
| Create `Experiments.tsx` page | ✅ Complete | `frontend/src/pages/Experiments.tsx` |
| Create `CausalAnalysis.tsx` page | ✅ Complete | `frontend/src/pages/CausalAnalysis.tsx` |
| Create `ResourceOptimization.tsx` page | ✅ Complete | `frontend/src/pages/ResourceOptimization.tsx` |
| Create `SegmentAnalysis.tsx` page | ✅ Complete | `frontend/src/pages/SegmentAnalysis.tsx` |
| Enhance `SystemHealth.tsx` | ✅ Complete | Added Health Score API integration with 5-tab layout |
| Update `routes.tsx` | ✅ Complete | All 5 new routes added |
| Update `Sidebar.tsx` icons | ✅ Complete | Added flask, calculator, users icons |
| Deploy to droplet | 🔲 Pending | |
| Verify in browser | 🔲 Pending | |

#### Session Summary - 2026-01-20

**Completed Items:**

1. **React Query Hooks (6 files):**
   - `use-gaps.ts` - useGapOpportunities, useQuickWins, useGapAnalysis, etc.
   - `use-experiments.ts` - useExperiments, useExperimentDetails, useRunExperiment, etc.
   - `use-causal.ts` - useCausalAnalysis, useHierarchicalCATE, useMultiLibraryAnalysis, etc.
   - `use-resources.ts` - useResourceOptimization, useOptimalAllocations, useScenarioComparison, etc.
   - `use-segments.ts` - useSegmentAnalysis, useCATEBySegment, useHighResponders, etc.
   - `use-health-score.ts` - useQuickHealthCheck, useAgentHealth, usePipelineHealth, useHealthHistory, etc.

2. **Page Components (5 new + 1 enhanced):**
   - `GapAnalysis.tsx` - ROI opportunities with quick wins, strategic bets, filtering
   - `Experiments.tsx` - A/B testing with wizard, enrollment tracking, SRM alerts
   - `CausalAnalysis.tsx` - Multi-library causal inference with hierarchical CATE
   - `ResourceOptimization.tsx` - Budget allocation with scenario comparison
   - `SegmentAnalysis.tsx` - Heterogeneous treatment effects with CATE heatmap
   - `SystemHealth.tsx` (enhanced) - Agent health by tier, pipelines, health history

3. **Routes & Navigation:**
   - Added 5 new route entries in `routes.tsx`
   - All routes protected with ProtectedRoute wrapper
   - Route configs include path, title, description, icon, showInNav
   - Added missing icons to Sidebar.tsx: flask, calculator, users

**Files Modified/Created:**
- `frontend/src/hooks/api/use-gaps.ts` (new)
- `frontend/src/hooks/api/use-experiments.ts` (new)
- `frontend/src/hooks/api/use-causal.ts` (new)
- `frontend/src/hooks/api/use-resources.ts` (new)
- `frontend/src/hooks/api/use-segments.ts` (new)
- `frontend/src/hooks/api/use-health-score.ts` (new)
- `frontend/src/pages/GapAnalysis.tsx` (new)
- `frontend/src/pages/Experiments.tsx` (new)
- `frontend/src/pages/CausalAnalysis.tsx` (new)
- `frontend/src/pages/ResourceOptimization.tsx` (new)
- `frontend/src/pages/SegmentAnalysis.tsx` (new)
- `frontend/src/pages/SystemHealth.tsx` (enhanced)
- `frontend/src/router/routes.tsx` (modified)
- `frontend/src/components/layout/Sidebar.tsx` (modified)

**TypeScript Verification:** ✅ `npx tsc --noEmit` passes without errors

**Next Steps:**
- Deploy frontend changes to droplet
- Verify all pages render correctly in browser
- Run E2E tests to validate complete workflows
