# Data Flow Integrity & Agent Observability Improvement Plan

**Date:** 2026-01-25
**Status:** Ready for Implementation
**Scope:** Full Implementation (Phases A + B + C) + Audit API
**Current Scores:** Data Flow 80/100 (B), Agent Observability 90/100 (A)
**Target Scores:** Data Flow 90/100 (A), Agent Observability 97/100 (A+)
**Estimated Effort:** 12-16 days

---

## Executive Summary

The comprehensive system evaluation identified these two areas as having room for improvement. While both have strong foundations, specific gaps prevent them from reaching A/A+ grades.

---

## Part 1: Data Flow Integrity (80 → 90)

### Current Strengths (Why it's at 80)
- **API Client Architecture (95/100)**: Excellent Axios wrapper with interceptors, auth headers, correlation IDs
- **Response Validation (95/100)**: Comprehensive Zod schemas in `frontend/src/lib/api-schemas.ts`
- **Route Configuration (100/100)**: All paths consistent, no mismatches
- **Type Safety (90/100)**: Strong TypeScript usage throughout

### Critical Gaps (Why it's not 90+)

#### Gap 1: Budget Input Validation Missing
**Severity:** HIGH | **Location:** `frontend/src/components/digital-twin/SimulationPanel.tsx:253-268`

```typescript
// Current - accepts any value including negative
<Input type="number" value={formValues.budget ?? ''} />

// Missing: min, max, step constraints
```

**Problems:**
- Accepts negative budget values
- No maximum constraint
- No validation before form submission

**Fix:** Add HTML5 constraints + Zod validation

---

#### Gap 2: Mutation Error Handling Missing
**Severity:** HIGH | **Location:** `frontend/src/pages/DigitalTwin.tsx:321-327`

```typescript
// Current - no onError callback
const { mutate: runSim } = useRunSimulation({
  onSuccess: () => { refetchHistory(); setActiveTab('history'); },
  // onError: MISSING - silent failure
});
```

**Problems:**
- User sees spinner disappear with no feedback on failure
- No error toast/notification
- No retry mechanism visible

**Fix:** Add onError callback with toast notification

---

#### Gap 3: Direct fetch() Bypasses API Client
**Severity:** CRITICAL | **Location:** `frontend/src/pages/AgentOrchestration.tsx:304`

```typescript
// Current - bypasses all apiClient benefits
const response = await fetch('/api/agents/status');

// Missing: auth header, correlation ID, error transformation, response validation
```

**Problems:**
- No Authorization header injection
- No X-Correlation-ID for tracing
- No typed error handling
- No response validation

**Fix:** Replace with `getValidated(AgentStatusSchema, '/agents/status')`

---

#### Gap 4: No Request Body Validation Schemas
**Severity:** MEDIUM | **Location:** `frontend/src/lib/api-schemas.ts`

- Response validation: Comprehensive
- Request validation: Missing

**Fix:** Add Zod schemas for POST/PUT request bodies

---

#### Gap 5: Form-Level Validation Missing
**Severity:** MEDIUM | **Location:** `frontend/src/components/digital-twin/SimulationPanel.tsx:270-285`

- `use-e2i-validation.ts` exists but not integrated
- No form-level error summary shown

**Fix:** Integrate validation hook, add error display

---

### Data Flow Remediation Summary

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Budget validation | 0.5d | Data integrity |
| P0 | Mutation error handling | 0.5d | User feedback |
| P0 | Replace direct fetch() | 0.5d | Security/tracing |
| P1 | Request body schemas | 1d | Data integrity |
| P1 | Form-level validation | 1d | UX |
| P2 | Stale data indicators | 0.5d | UX |

**Total Estimated Effort:** 4 days → Raises score from 80 to 90

---

## Part 2: Agent Observability (90 → 97)

### Current Strengths (Why it's at 90)

**ChatResponse Model** (`src/api/routes/copilotkit.py:2755`):
- `agents_dispatched`: List of agents called
- `execution_time_ms`: Total execution time
- `routing_rationale`: Why agent was selected
- `response_confidence`: Orchestrator confidence score
- `intent` + `intent_confidence`: Query classification

**Tracing Infrastructure:**
- Opik integration with circuit breaker pattern
- MLflow experiment tracking
- Audit chain with tamper-evident hashing
- Latency breakdown (classification, RAG, routing, dispatch, synthesis)

**Partial Failure Handling:**
- `successful_agents` / `failed_agents` lists
- `failure_details` with agent_name, error, latency_ms
- User-friendly warning messages

**Frontend Components:**
- AgentStatusPanel with real-time status indicators
- AgentProgressRenderer with progress bars
- SSE streaming with dispatch_info events

### Critical Gaps (Why it's not 97+)

#### Gap 1: No User-Facing Observability Dashboard
**Severity:** HIGH | **Impact:** 15 points

**Missing:**
- No dedicated dashboard showing agent metrics over time
- No visualization of latency breakdown
- No per-agent performance trending (p50/p95/p99)
- Historical query metrics not accessible

**Fix:** Create `/analytics` page with metrics visualization

---

#### Gap 2: Trace/Request ID Not Visible to Users
**Severity:** MEDIUM | **Impact:** 10 points

**Missing:**
- Trace ID not displayed in UI
- No copy-to-clipboard for support tickets
- Cannot correlate UI events with backend logs

**Fix:** Add trace ID display with copy button in chat footer

---

#### Gap 3: Routing Rationale Not Displayed
**Severity:** MEDIUM | **Impact:** 10 points

**Missing:**
- `routing_rationale` exists but not shown prominently
- Candidate agents not displayed
- Confidence score not contextualized

**Fix:** Add expandable "Why this agent?" section in chat messages

---

#### Gap 4: Audit Trail API Missing
**Severity:** HIGH | **Impact:** 10 points (compliance)

**Current State:**
- Backend audit chain fully implemented
- Database tables exist: `audit_chain_entries`, `v_audit_chain_summary`
- No API endpoints exposed
- No UI for audit history

**Fix:** Create audit API endpoints + simple history view

---

#### Gap 5: Error Messages Lack Remediation Guidance
**Severity:** MEDIUM | **Impact:** 10 points

**Current:**
```
"One analysis component (causal_impact) took too long..."
```

**Should Be:**
```
"One analysis component (causal_impact) took too long. Try again in 30 seconds or simplify your query."
```

**Fix:** Enhance failure messages with actionable suggestions

---

### Observability Remediation Summary

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Audit trail API + UI | 2-3d | Compliance |
| P1 | Metrics dashboard | 3-4d | Transparency |
| P1 | Routing rationale display | 1-2d | Trust |
| P2 | Trace ID display | 1d | Support |
| P2 | Error remediation guidance | 1-2d | UX |

**Total Estimated Effort:** 8-12 days → Raises score from 90 to 97

---

## Decisions Made

1. **Scope:** Full Implementation (Phases A + B + C)
2. **Audit API:** Yes - Include audit trail API for compliance
3. **Metrics Dashboard:** Yes - Include in Phase C

---

## Implementation Sequence

### Week 1: Phase A - Quick Wins (3-4 days)

**Day 1-2: Data Flow Fixes**
1. `SimulationPanel.tsx` - Add budget validation (min=0, max=999999999, step=1)
2. `DigitalTwin.tsx` - Add onError callback with toast notification
3. `AgentOrchestration.tsx` - Replace direct fetch() with apiClient

**Day 3-4: Observability UI**
4. `CustomAssistantMessage.tsx` - Add routing rationale expandable section
5. Chat footer - Add trace ID display with copy button

### Week 2: Phase B - Core Improvements (5-7 days)

**Day 5-6: Data Flow Enhancement**
6. `api-schemas.ts` - Add request body Zod schemas
7. `SimulationPanel.tsx` - Integrate form-level validation

**Day 7-9: Audit Trail API**
8. Create `src/api/routes/audit.py` with endpoints:
   - `GET /api/audit/workflow/{workflow_id}` - Get audit entries
   - `GET /api/audit/verify/{workflow_id}` - Verify chain integrity
   - `GET /api/audit/history` - Paginated history
9. Create simple audit history UI component

**Day 10-11: Error Remediation**
10. Enhance error messages with actionable suggestions
11. Add retry recommendations to failure states

### Week 3: Phase C - Advanced Features (4-5 days)

**Day 12-15: Metrics Dashboard**
12. Create `/analytics` page with:
    - Query execution metrics over time
    - Latency breakdown charts (classification, RAG, routing, dispatch)
    - Per-agent success/failure rates
    - p50/p95/p99 latency percentiles

**Day 16: Polish**
13. Add stale data indicators to query hooks
14. Final testing and integration verification

---

## Critical Files to Modify

### Data Flow Integrity (6 files)
```
frontend/src/components/digital-twin/SimulationPanel.tsx  # Budget validation + form validation
frontend/src/pages/DigitalTwin.tsx                        # Mutation error handling
frontend/src/pages/AgentOrchestration.tsx                 # Replace fetch() with apiClient
frontend/src/lib/api-schemas.ts                           # Request body schemas
frontend/src/hooks/api/use-digital-twin.ts                # Stale data indicators
```

### Agent Observability (7 files)
```
frontend/src/components/chat/CustomAssistantMessage.tsx   # Routing rationale display
frontend/src/components/chat/ChatFooter.tsx (or similar)  # Trace ID display
src/api/routes/audit.py                                   # NEW - Audit API endpoints
frontend/src/pages/Analytics.tsx                          # NEW - Metrics dashboard
frontend/src/components/audit/AuditHistory.tsx            # NEW - Audit history UI
src/api/errors.py                                         # Error remediation messages
frontend/src/router/routes.tsx                            # Add /analytics route
```

---

## Verification Checklist

### After Phase A (Day 4)
- [ ] Budget input rejects negative values and values > 999,999,999
- [ ] Simulation failure shows toast notification
- [ ] No `await fetch(` in AgentOrchestration.tsx
- [ ] Routing rationale visible in chat messages
- [ ] Trace ID displayed with copy button

### After Phase B (Day 11)
- [ ] Request bodies validated with Zod before submission
- [ ] Form shows inline validation errors
- [ ] `GET /api/audit/workflow/{id}` returns audit entries
- [ ] `GET /api/audit/verify/{id}` returns verification status
- [ ] Error messages include actionable suggestions

### After Phase C (Day 16)
- [ ] `/analytics` page loads with metrics charts
- [ ] Latency breakdown visible per query type
- [ ] Agent success rates displayed
- [ ] Stale data shows "Last updated X ago" indicator

### Test Commands
```bash
# Data Flow tests
npm run test -- src/components/digital-twin/SimulationPanel.test.tsx
npm run test -- src/pages/DigitalTwin.test.tsx
grep -r "await fetch(" frontend/src/pages/ --include="*.tsx"  # Should return nothing

# Observability tests
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest tests/unit/api/routes/test_audit.py -v -n 2"

# Full validation
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/.venv/bin/pytest tests/ -v -n 4 --dist=loadscope"
```

---

## Success Criteria

### Data Flow Integrity (Target: 90/100)
- [ ] Budget input has min=0, max=999999999 constraints
- [ ] All mutations have onError callbacks with user feedback
- [ ] Zero direct fetch() calls in pages (all use apiClient)
- [ ] Request body Zod schemas for all POST endpoints
- [ ] Form validation errors displayed inline

### Agent Observability (Target: 97/100)
- [ ] Routing rationale displayed in expandable UI
- [ ] Trace ID visible with copy-to-clipboard
- [ ] Audit API endpoints: GET /api/audit/workflow/{id}, /api/audit/verify/{id}
- [ ] Error messages include remediation suggestions
- [ ] Metrics dashboard at /analytics

---

**Document Version:** 1.0
**Created:** 2026-01-25
**Next Action:** Begin Phase A - Day 1 (Budget validation in SimulationPanel.tsx)
