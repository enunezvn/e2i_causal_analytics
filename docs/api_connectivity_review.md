# Backend ↔ Frontend API Connectivity Review

**Date:** 2026-05-16
**Branch:** `claude/plan-api-integration-review-JQCVn`
**Scope:** Verify every backend FastAPI route is reachable from the React frontend and that pages render real data rather than mocks.

---

## 1. Executive Summary

| Metric | Count |
|---|---:|
| Backend routes exposed (incl. health/metrics) | **161** |
| Distinct frontend HTTP calls | **138** |
| Routes correctly wired (frontend ↔ backend agree on method+path) | **122** |
| Routes exposed but unused by the frontend | **39** |
| Frontend calls that hit no backend route (will 404) | **16** |
| Pages with real backend data | **7** |
| Pages that only call `/health` then render hard-coded `SAMPLE_*` | **5** |
| Pages with zero API integration (pure mocks) | **8** |
| Pages with dead wiring (mutation runs, response ignored) | **1** |

**Headline issues**

1. **RAG entire family broken** — frontend `*_BASE = '/rag'` but the backend router is mounted at `/api/v1/rag` (`src/api/main.py:924`). All 6 RAG endpoints 404. No page currently calls them, so the bug is latent.
2. **Cognitive session endpoints broken** — frontend uses `/cognitive/sessions/*` (plural), backend serves `/cognitive/session/*` (singular). 4 endpoints 404. Plus `/cognitive/status` referenced by the client doesn't exist on the backend at all.
3. **`/api/explain[/batch]` broken** — frontend posts to `/api/explain` and `/api/explain/batch`; backend exposes `/api/explain/predict` and `/api/explain/predict/batch`. This **is hit at runtime**: `AIAgentInsights.tsx` renders `HeterogeneousTreatmentEffects.tsx:18`, which calls `useBatchExplain`. The AIAgentInsights page will show error toasts when that section loads.
4. **Memory verb mismatches** — frontend GET `/memory/episodic` and POST `/memory/semantic/paths`; backend has POST `/memory/episodic` (list-by-search) and GET `/memory/semantic/paths`. Both 405.
5. **Digital-twin orphan endpoints** — frontend calls `/digital-twin/simulations/compare` and `/digital-twin/simulations/history`, neither exists server-side.
6. **8 pages have no backend wiring at all** (Monitoring, AIAgentInsights — except embedded components, CausalDiscovery, PredictiveAnalytics, DataQuality, TimeSeries, FeatureImportance, ModelPerformance). They render purely from `SAMPLE_*` / `MOCK_*` constants defined in the file.
7. **ResourceOptimization.tsx** imports `useRunOptimization` and calls `.mutate(...)` on button click but `optimizationResult` is unconditionally assigned to `sampleOptimizationResult` (`pages/ResourceOptimization.tsx:406`) — the mutation's response is never read.
8. **Three backend routers have no frontend client at all** — `executive-insights`, `sentinels`, `agents` (the last has one inline call from `AgentOrchestration.tsx:307`).

---

## 2. Architecture under review

```
┌────────────────────────────┐    ┌─────────────────────────────────────┐
│ Page (pages/*.tsx)         │    │ Backend route (src/api/routes/*.py) │
│  └─ Hook (hooks/api/use-*) │    │   └─ APIRouter(prefix="…")          │
│      └─ Client (api/*.ts)  │ →  │       └─ @router.<verb>("/path")    │
│          └─ apiClient      │    │                                     │
│              baseURL=/api  │    │ mounted in src/api/main.py with     │
│                            │    │   include_router(r, prefix="/api")  │
└────────────────────────────┘    └─────────────────────────────────────┘
```

`Effective URL = env.apiUrl ("/api") + module *_BASE + per-call path`
(see `frontend/src/config/env.ts:80`, `frontend/src/lib/api-client.ts:222`).

A connection is **healthy** only when method + full path agree across all four layers.

---

## 3. URL prefix audit

| Backend router | Built-in prefix | Mount prefix in `main.py` | Frontend `*_BASE` | Effective frontend URL | Match |
|---|---|---|---|---|---|
| `analytics.py` | `/analytics` | `/api` (`main.py:963`) | `/analytics` (`analytics.ts:40` via `apiClient.get`) | `/api/analytics/*` | ✓ |
| `audit.py` | `/audit` | `/api` (`main.py:960`) | `/audit` (`audit.ts:34`) | `/api/audit/*` | ✓ |
| `causal.py` | `/causal` | `/api` (`main.py:957`) | `/causal` (`causal.ts:40`) | `/api/causal/*` | ✓ |
| `cognitive.py` | `/cognitive` | `/api` (`main.py:918`) | `/cognitive` (`cognitive.ts:32`) | `/api/cognitive/*` | ✓ for `query`, `rag`; **✗ for `session*`, `status`** |
| `digital_twin.py` | `/digital-twin` | `/api` (`main.py:948`) | `/digital-twin` (`digital-twin.ts:47`) | `/api/digital-twin/*` | ✓ for most; ✗ `simulations/compare`, `simulations/history` |
| `experiments.py` | `/experiments` | `/api` (`main.py:930`) | `/experiments` (`experiments.ts:52`) | `/api/experiments/*` | ✓ |
| `explain.py` | `/explain` | `/api` (`main.py:912`) | `/explain` (`explain.ts:34`) | `/api/explain/*` | **✗ path mismatch (`/predict` missing)** |
| `feedback.py` | `/feedback` | `/api` (`main.py:942`) | `/feedback` (`feedback.ts:36`) | `/api/feedback/*` | ✓ for endpoints frontend uses |
| `gaps.py` | `/gaps` | `/api` (`main.py:933`) | `/gaps` (`gaps.ts:29`) | `/api/gaps/*` | ✓ |
| `graph.py` | `/graph` | `/api` (`main.py:921`) | `/graph` (`graph.ts:46`) | `/api/graph/*` | ✓ |
| `health_score.py` | `/health-score` | `/api` (`main.py:945`) | `/health-score` (`health-score.ts:33`) | `/api/health-score/*` | ✓ |
| `kpi.py` | **`/api/kpis`** | none (`main.py:954`) | `/kpis` (`kpi.ts:37`) | `/api/kpis/*` | ✓ |
| `memory.py` | `/memory` | `/api` (`main.py:915`) | `/memory` (`memory.ts:35`) | `/api/memory/*` | ✓ for `search`,`stats`; **✗ verb mismatch on `/episodic` and `/semantic/paths`** |
| `monitoring.py` | `/monitoring` | `/api` (`main.py:927`) | `/monitoring` (`monitoring.ts:51`) | `/api/monitoring/*` | ✓ |
| `predictions.py` | **`/api/models`** | none (`main.py:951`) | `/models` (`predictions.ts:31`) | `/api/models/*` | ✓ |
| `rag.py` | **`/api/v1/rag`** | none (`main.py:924`) | `/rag` (`rag.ts:35`) | `/api/rag/*` | **✗ — every call 404s** |
| `resource_optimizer.py` | `/resources` | `/api` (`main.py:939`) | `/resources` (`resources.ts:29`) | `/api/resources/*` | ✓ |
| `segments.py` | `/segments` | `/api` (`main.py:936`) | `/segments` (`segments.ts:29`) | `/api/segments/*` | ✓ |
| `agents.py` | `/agents` | `/api` (`main.py:972`) | — (no client file) | inline only | ✓ for `/status` only |
| `sentinels.py` | `/sentinels` | `/api` (`main.py:975`) | — | — | no frontend client |
| `executive_insights.py` | `/executive-insights` | `/api` (`main.py:979`) | — | — | no frontend client |
| `copilotkit.py` | `/copilotkit` | `/api` (`main.py:966`) | — (CopilotKit SDK handles `/api/copilotkit/{action}`) | `/api/copilotkit/*` | runtime endpoints consumed by `@copilotkit/react-core`; status/feedback/analytics endpoints have no consumer |
| `metrics.py` | _(none, mounts at root)_ | none (`main.py:982`) | — | `/metrics` | Prometheus-only, intentional |
| `chatbot_dspy/graph/state/tools/tracer.py` | — | — | — | — | **Not route modules** — these files define helper functions/classes consumed by `copilotkit.py`; the `routes/` folder location is a misnomer. No action needed. |

---

## 4. Critical mismatches (frontend calls that 404 or 405 at runtime)

### 4.1 RAG — entire family broken (6 endpoints)
Backend prefix `/api/v1/rag` vs frontend prefix `/api/rag`.

| Frontend (`rag.ts`) | Backend (`rag.py`) |
|---|---|
| `POST /api/rag/search` | `POST /api/v1/rag/search` |
| `GET /api/rag/health` | `GET /api/v1/rag/health` |
| `GET /api/rag/stats` | `GET /api/v1/rag/stats` |
| `GET /api/rag/entities` | `GET /api/v1/rag/entities` |
| `GET /api/rag/graph/subgraph/{entity}` | `GET /api/v1/rag/graph/{entity}` |
| `GET /api/rag/graph/paths` | `GET /api/v1/rag/causal-path` |

The last two also have **different paths** beyond the prefix issue. No page currently imports `use-rag`, so users won't see the failure today — but the `RAG_BASE` constant must be fixed before any RAG UI work.

**Fix** (single line): `frontend/src/api/rag.ts:35` → `const RAG_BASE = '/v1/rag';` and update the two diverging paths (`/graph/subgraph/{entity}` → `/graph/{entity}`, `/graph/paths` → `/causal-path`).

### 4.2 Cognitive — session endpoints and status

| Frontend (`cognitive.ts`) | Backend (`cognitive.py`) |
|---|---|
| `GET /api/cognitive/status` (L85) | _(no such endpoint)_ |
| `POST /api/cognitive/sessions` (L113) | `POST /api/cognitive/session` (L515) |
| `GET /api/cognitive/sessions/{sessionId}` (L135) | `GET /api/cognitive/session/{session_id}` (L441) |
| `DELETE /api/cognitive/sessions/{sessionId}` (L158) | `DELETE /api/cognitive/session/{session_id}` (L559) |
| `GET /api/cognitive/sessions` (L177, list) | _(no list endpoint)_ |

**Fix:** either rename frontend paths to `/session/*` and remove `/status` + the list call, or add a `/sessions` plural alias and a `/status` endpoint to the backend. The route filenames in the backend already use singular consistently, so renaming the frontend is the smaller change.

### 4.3 Explain — `/predict` segment missing on the frontend
- `frontend/src/api/explain.ts:62` posts to `${EXPLAIN_BASE}` (i.e. `/api/explain`) → no such route. Backend is `POST /api/explain/predict` (`explain.py:533`).
- `frontend/src/api/explain.ts:88` posts to `${EXPLAIN_BASE}/batch` (i.e. `/api/explain/batch`) → no such route. Backend is `POST /api/explain/predict/batch` (`explain.py:643`).

**User impact:** `pages/AIAgentInsights.tsx:80` renders `HeterogeneousTreatmentEffects`, which (`components/insights/HeterogeneousTreatmentEffects.tsx:18`) calls `useBatchExplain` → `getBatchExplanations` → broken URL. This section will surface a 404 toast every time the page loads.

**Fix:** change both calls in `explain.ts` to `${EXPLAIN_BASE}/predict` and `${EXPLAIN_BASE}/predict/batch`.

### 4.4 Memory — verb mismatches

| Frontend (`memory.ts`) | Backend (`memory.py`) |
|---|---|
| `GET /api/memory/episodic` (L118) | only `POST /api/memory/episodic` (`memory.py:337`, list-by-search semantic) |
| `POST /api/memory/semantic/paths` (L197) | only `GET /api/memory/semantic/paths` (`memory.py:483`) |

Result: 405 Method Not Allowed at runtime. Either change the frontend verb or add the missing handler.

### 4.5 Digital-Twin — orphan frontend paths

- `frontend/src/api/digital-twin.ts:634-650` calls `/api/digital-twin/simulations/history` (GET) and `/api/digital-twin/simulations/compare` (POST). Neither exists in `digital_twin.py`.
- The closest backend equivalent for history is `GET /api/digital-twin/simulations` with query filters (`digital_twin.py:548`).

**Fix:** drop the two helper functions (`getSimulationHistory`, `compareScenarios` in `digital-twin.ts`) or add the corresponding backend endpoints.

---

## 5. Pages on mock data — UI vs reality

```
Live (backend-backed):  Home, Analytics, KnowledgeGraph, MemoryArchitecture,
                        KPIDictionary, SystemHealth

Health-only + samples:  CausalAnalysis, SegmentAnalysis, GapAnalysis,
                        FeedbackLearning, AuditChain

Dead wiring:            ResourceOptimization (mutation fires, result ignored)

Mock-only (no API):     Monitoring, AIAgentInsights*, CausalDiscovery,
                        PredictiveAnalytics, DataQuality, TimeSeries,
                        FeatureImportance, ModelPerformance, DigitalTwin†

(*) AIAgentInsights itself imports no hook, but the components it renders do —
    that's the only thing pulling live data there.
(†) DigitalTwin uses useRunSimulation + useSimulationHistory + useDigitalTwinHealth
    but the listing falls back to SAMPLE_SIMULATION at L319.
```

| Page | Hook(s) imported | Endpoint(s) called | What renders | Notes |
|---|---|---|---|---|
| `Home.tsx` | `useKPIList`, `useKPIHealth`, `useGraphStats`, `useAlerts` | `/api/kpis`, `/api/kpis/health`, `/api/graph/stats`, `/api/monitoring/alerts` | Live | Plus mounts `<ExecutiveSummary/>`, `<CausalValueChains/>` |
| `Analytics.tsx` | `useAnalyticsDashboard` | `/api/analytics/dashboard` | Live | |
| `KnowledgeGraph.tsx` | `useNodes`, `useRelationships`, `useGraphStats` | `/api/graph/{nodes,relationships,stats}` | Live | |
| `MemoryArchitecture.tsx` | `useMemoryStats`, `useEpisodicMemories` | `/api/memory/stats`, `/api/memory/episodic` | **Partly broken** — `useEpisodicMemories` is GET to an endpoint that's only POST on the backend (see §4.4) |
| `KPIDictionary.tsx` | `useKPIList`, `useWorkstreams`, `useKPIHealth` | `/api/kpis*` | Live | |
| `SystemHealth.tsx` | `useAlerts`, `useMonitoringRuns` | `/api/monitoring/alerts`, `/api/monitoring/runs` | Live | |
| `CausalAnalysis.tsx` | `useCausalHealth`, `useRunHierarchicalAnalysis` | `/api/causal/health` only (mutation imported but result never bound to charts) | Sample data | L243 health probe; L59+ sample tables |
| `SegmentAnalysis.tsx` | `useSegmentHealth`, `useRunSegmentAnalysis`, `usePolicies` | `/api/segments/health`, `/api/segments/policies` | Sample data | `sampleAnalysisResult` at L59 |
| `GapAnalysis.tsx` | `useGapHealth` | `/api/gaps/health` | Sample data | L275 health probe; L278 fallback to SAMPLE_GAPS |
| `FeedbackLearning.tsx` | `useFeedbackHealth` | `/api/feedback/health` | Sample data | L253 health probe |
| `AuditChain.tsx` | `useRecentWorkflows` | `/api/audit/recent` | Live + sample fallback | Renders `<AuditHistory/>` separately, which fetches workflow details inline (`components/audit/AuditHistory.tsx:136-162`). **AuditHistory is not imported by AuditChain.tsx** — it's a dead component. |
| `ResourceOptimization.tsx` | `useResourceHealth`, `useScenarios`, `useRunOptimization` | `/api/resources/health`, `/api/resources/scenarios`, `POST /api/resources/optimize` | **Dead wiring** | L403-431: mutation runs but L406 `optimizationResult = sampleOptimizationResult` hard-codes the displayed value |
| `Experiments.tsx` | `useTriggerMonitoring` | `POST /api/experiments/monitor` | Live with sample fallback | L285 fires monitor sweep; L289 derives experiments from result, else SAMPLE_EXPERIMENTS |
| `DigitalTwin.tsx` | `useDigitalTwinHealth`, `useSimulationHistory`, `useRunSimulation` | `/api/digital-twin/{health,simulations,simulate}` | Live with sample fallback | L319 `selectedSimulation = SAMPLE_SIMULATION`; `/intervention-impact` retired (T10) — redirects to `/causal-analysis`, simulate served only here |
| `AgentOrchestration.tsx` | _none — inline `useQuery`_ | `GET /api/agents/status` | Live | Bypasses the hook layer; should be refactored into `frontend/src/api/agents.ts` + a `use-agents.ts` hook for consistency |
| `Monitoring.tsx` | — | — | Mock | Defines `ApiMetric`, `EndpointStats`, `UserActivity`, `ErrorLog` types locally; populates with hard-coded arrays |
| `AIAgentInsights.tsx` | — | — | Composite | Only renders insight components, which themselves use hooks. Inherits the broken explain call (§4.3) |
| `CausalDiscovery.tsx` | — | — | Mock | |
| `PredictiveAnalytics.tsx` | — | — | Mock | |
| `DataQuality.tsx` | — | — | Mock | |
| `TimeSeries.tsx` | — | — | Mock | |
| `FeatureImportance.tsx` | — | — | Mock | Backend has `POST /api/explain/predict` available, but the page never calls it |
| `ModelPerformance.tsx` | — | — | Mock | Backend `monitoring.py` performance endpoints unused |

---

## 6. Backend endpoints with no frontend caller

39 routes. Bucketed by intent:

**Intentionally serverside-only** (5): `/metrics`, `/metrics/health` (Prometheus scrape), `POST /api/copilotkit/chat`, `POST /api/copilotkit/chat/stream`, `POST /api/copilotkit/feedback` (handled by the CopilotKit SDK, not by our `api/*.ts`).

**Routers with no frontend client at all** (10):
- `executive-insights` (3): `GET /api/executive-insights`, `GET /api/executive-insights/{id}`, `POST /api/executive-insights/crystallize`
- `sentinels` (5): `GET/POST /api/sentinels`, `GET/PATCH/DELETE /api/sentinels/{id}`
- `copilotkit` admin (6 unused): `/copilotkit/{status, feedback/stats, analytics/{usage,agents,errors,hourly}}`
- Existing `chatbot_*.py` files are helper modules, not routers — no action needed.

**Wrapped on the frontend but no page consumes them** (24):
- Causal: `GET /api/causal/pipeline/{id}` (no polling UI for parallel/sequential pipelines)
- Cognitive: `GET/POST/DELETE /api/cognitive/session*` (broken paths — see §4.2)
- Experiments: `POST /experiments/{id}/srm-check`, `POST /experiments/{id}/fidelity/{twin_simulation_id}`
- Explain: `/explain/predict` and `/explain/predict/batch` (broken — see §4.3)
- Feedback: `POST /feedback/trace`, `POST /feedback/updates/{id}/rollback`, `GET /feedback/agent/{name}/stats`, `…/signals`, `…/gepa-batch`
- Memory: `GET /memory/semantic/paths`, `POST /memory/episodic` (both have verb-mismatched frontend counterparts)
- RAG: all 6 backend endpoints (the frontend wrapper is broken — see §4.1)

---

## 7. Verification

### 7.1 Inventory data
The inventory was produced by parsing source files (no fastapi runtime available in the review environment). Reproducer:

```bash
# Backend inventory (parses src/api/main.py mounts + per-router @router decorators)
python3 -c "import ..."   # see git history of this commit; output at /tmp/backend_routes.tsv

# Frontend inventory (parses frontend/src/api/*.ts and inline calls in pages/components/hooks)
python3 -c "import ..."   # see git history; output at /tmp/frontend_calls.tsv
```

### 7.2 Live verification
Step 6 from the plan (booting `uvicorn src.api.main:app` and curl-checking each broken path) was **not run** because `fastapi` is not installed in the review container. The static analysis is authoritative because:

- Every backend path is grep-able from `src/api/routes/*.py` `@router.<verb>("...")` literals (no decorators are wrapped in conditionals).
- Every frontend path resolves at compile time via `const *_BASE = '/...'` + string-literal suffix; no runtime path computation that would invalidate the diff.

Anyone with a working dev environment can rerun the diff after installing requirements and confirm by:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/rag/search          # expect 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/v1/rag/search       # expect 405 (it's POST)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/cognitive/sessions  # expect 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/cognitive/session   # expect 405 (it's POST)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/explain             # expect 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/explain/predict     # expect 405 (it's POST)
```

---

## 8. Recommended next actions (priority order)

1. **P0** — Fix `frontend/src/api/explain.ts` paths (§4.3). This is the only broken endpoint currently surfaced on a user-visible page.
2. **P1** — Decide direction (frontend rename vs backend rename) and fix the `cognitive/sessions` mismatch (§4.2). The `useSessions`/`useCreateSession` hooks aren't reached today, but the same diff will block any cognitive-session UI work.
3. **P1** — Fix `RAG_BASE` to `/v1/rag` and update the two diverging paths in `rag.ts` (§4.1). Same rationale — needed before any RAG UI lands.
4. **P1** — Fix memory verb mismatches (§4.4) or align backend semantics. `MemoryArchitecture.tsx` will silently show empty results until then.
5. **P2** — Remove the hard-coded `optimizationResult = sampleOptimizationResult` line in `ResourceOptimization.tsx:406` so the mutation's response actually renders.
6. **P2** — Either drop or back-fill `getSimulationHistory` / `compareScenarios` in `digital-twin.ts` (§4.5).
7. **P2** — Replace the inline `useQuery` in `AgentOrchestration.tsx:307` with a proper `useAgentStatus` hook in `frontend/src/hooks/api/use-agents.ts` backed by `frontend/src/api/agents.ts`, matching the pattern used elsewhere.
8. **P3** — Wire the 8 mock-only pages: each has a corresponding backend router. Start with `ModelPerformance` (already has `monitoring.py` performance endpoints fully exposed) and `FeatureImportance` (already has `POST /api/explain/predict` after the §4.3 fix).
9. **P3** — Wire executive-insights and sentinels into pages, or remove the unused routers.
10. **P3** — Delete the orphan `AuditHistory.tsx` component or import it from `AuditChain.tsx`.

---

## Appendix A — Full backend route inventory

See `/tmp/backend_routes.tsv` (regenerable). 161 rows: `METHOD<TAB>PATH<TAB>MODULE`.

## Appendix B — Full frontend call inventory

See `/tmp/frontend_calls.tsv` (regenerable). 142 rows: `METHOD<TAB>PATH<TAB>SOURCE_FILE`.

## Appendix C — Reviewer's note on chatbot_* files

The files `src/api/routes/chatbot_dspy.py`, `chatbot_graph.py`, `chatbot_state.py`, `chatbot_tools.py`, and `chatbot_tracer.py` contain **no `@router.<verb>` decorators**. They are helper modules (DSPy classifier, LangGraph state graph, intent classification utilities) imported by `routes/copilotkit.py`. Placing them in `routes/` is misleading naming but not a bug. Consider moving them to `src/agents/chatbot/` for clarity in a future cleanup.
