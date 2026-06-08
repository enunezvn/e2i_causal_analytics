# M5 — Backend orphan endpoints: triage & decision (2026-06-08)

**Status:** DECISION DOC + one bounded REWIRE. **No backend endpoint is deleted by this shard.**
**Source of truth:** `docs/reports/frontend-backend-api-connectivity-audit-20260608.md` §5 (M5, lines 99–104) and §8 recommendation 5 (line 160).
**Author note:** every zero-consumer and intent claim below was independently re-verified (greps over `frontend/src` excluding `/generated/`, `git log` on each router, and a live Starlette route-matcher run) before being recorded here — see "Verification evidence".

---

## Why this is a decision doc, not a cleanup

Per **REASON-BEFORE-RULES** (`CLAUDE.md`): a mounted endpoint with no UI consumer is **not** automatically dead. A grep is a snapshot of *now*; product intent lives in PRs, issues, and the roadmap. For each of the 21 mounted-but-unconsumed endpoints I answered the four reasoning questions (what is it for, why does it exist in this shape, is it causing harm now, what does the owner actually want) before applying the 4-way classification:

- **HARMFUL-NOW** — user-facing or plausible-wrong values → rewire/gate immediately.
- **REWIRE** — functionality requested + a real UI home exists and is feasible now.
- **KEEP-AS-INTENTIONAL-PLACEHOLDER** — functionality requested, real consumer blocked/not-yet-built, mock-free (real handlers), non-harmful (RBAC-gated, not user-facing fake values).
- **DELETE** — no recoverable intent (vestigial, copy-paste error). **None of the 21 fell here.**

**Net:** 1 group **REWIRE** (executive-insights, shipped in this shard). 5 groups **KEEP / owner-decision** — no deletions. Any deletion or roadmap reprioritization requires explicit owner sign-off (see "Owner sign-off required").

---

## Triage table (4-way classification)

| Group (count) | Routes | Zero-consumer confirmed | Intent evidence | 4-way disposition |
|---|---|---|---|---|
| **sentinels CRUD (5)** | `POST/GET /api/sentinels`, `GET/PATCH/DELETE /api/sentinels/{id}` (`sentinels.py:75,138,199,239,273`) | yes (`grep "sentinels" frontend/src` → 0) | Active subsystem: added in `2421a2f9` (insight-lifecycle subsystems), **security-hardened 2026-06-03 `778ea0af` "close IDOR/BOLA on … sentinel mutations"** + `ea899743` brand-membership enforcement in list/get. Docstring describes a full operator workflow + brand RBAC. | **KEEP-AS-INTENTIONAL-PLACEHOLDER** (roadmap-staked operator surface; admin/ops UI not built yet; non-harmful — real handlers, RBAC-gated). Owner decision to build the UI or keep API-only. |
| **executive-insights (4)** | `GET /`, `GET /portfolio-summary`, `GET /{id}`, `POST /crystallize` (`executive_insights.py:172,198,274,297`) | yes (`grep "executive-insights\|crystalliz\|portfolio-summary" frontend/src` → 0) | Crystallization subsystem, schema-completed under **#376 `e7aafc51`** (migration 025, +15 fields) + #385 `705e36fd` (`invalidated_at IS NULL` filter); JIT-verifier middleware mounted on the prefix (`main.py`). The audit (line 101) notes the AI "Executive Brief" is **synthesized client-side via cognitive RAG instead** — i.e. the UI home (`ExecutiveAIBrief`) already exists. | **REWIRE** (clear, obvious UI home `ExecutiveAIBrief`; data layer shipped in this shard's Tasks 1–3). |
| **feedback per-agent / GEPA (4)** | `POST /trace` (`feedback.py:927`), `GET /agent/{name}/{stats,signals,gepa-batch}` (`feedback.py:995,1044,1102`) | yes (`grep "feedback/trace\|gepa-batch\|agent/.*signals" frontend/src` → 0) | Summaries say **"GEPA optimization signals (G23)" / "GEPA training batch (G23)"**; these feed the **DSPy self-improvement loop closed in PR #792 (`f6a282f6`, 2026-06-08)**. This is an **agent/optimizer-facing API, not a UI surface** (the optimizer reads signals server-side; no human screen). | **KEEP-AS-INTENTIONAL-PLACEHOLDER** (integration/optimizer surface by design; consuming the loop is server-side, not the UI). NOT a frontend orphan in the harmful sense. |
| **copilotkit analytics (5)** | `GET /copilotkit/analytics/{usage,agents,errors,hourly}`, `GET /copilotkit/feedback/stats` (`copilotkit.py:3744,3788,3840,3881,3919`) | yes — note: `frontend/src/api/analytics.ts` calls the **separate `/analytics/*` router** (`/analytics/dashboard`, `/analytics/agents/*`, `/analytics/summary`), NOT `/copilotkit/analytics/*` (different prefix). The FE *does* call `POST /copilotkit/feedback` (submit, `use-chat-feedback.ts:117`), but **not** the `GET /copilotkit/feedback/stats` aggregation. | Chatbot-specific telemetry (`P7.1`), read from `get_chatbot_analytics_repository`. The shipped `Analytics` view (`/analytics`, audit matrix line 148 = LIVE) already covers the analytics dashboard via the dedicated router. | **OWNER-DECISION (KEEP vs consolidate).** Likely KEEP as chatbot-ops/API-only telemetry, or fold into the existing Analytics view. **Do not delete without owner sign-off** — recently touched (`copilotkit.py` security sweeps through `4d7b1026` 2026-06-07). |
| **causal `GET /pipeline/{id}` (1)** | `GET /api/causal/pipeline/{pipeline_id}` (`causal.py:1900`, `operation_id=get_pipeline_status`) | yes — `causal.ts` calls `/pipeline/sequential` + `/pipeline/parallel` (POST launchers, lines 186/223), not the GET-by-id status (`grep` confirms). | Status-poll companion to the sequential/parallel pipeline launchers. The FE currently runs pipelines synchronously (audit matrix: CausalDiscovery LIVE) so the async status poll is unused. | **KEEP-AS-INTENTIONAL-PLACEHOLDER** (async-status companion; harmless; would be consumed if/when the FE moves to async pipeline polling). |
| **alerts `GET /alerts/stream` SSE (1)** | `GET /api/alerts/stream` (`staleness_alerts.py:407`, `EventSourceResponse`) | yes — FE has **no `EventSource`** consumer; only the WebSocket `/graph/stream` is consumed (`api-client.ts:636`). | Server-Sent-Events staleness feed with brand RBAC (`require_auth`) + 15s ping; docstring is a complete SSE contract. Roadmap: a live alerts banner/toast. | **KEEP-AS-INTENTIONAL-PLACEHOLDER** (roadmap-staked realtime feed; non-harmful; SSE client not built yet). |

**Total: 5 + 4 + 4 + 5 + 1 + 1 = 20 routes across 6 groups.** The audit headline is "21" — the discrepancy is accounted for: the `POST /copilotkit/feedback` *submit* endpoint IS consumed (`use-chat-feedback.ts:117`), so it is **not** an orphan; the orphan in that group is the GET `/copilotkit/feedback/stats` aggregation. The triage therefore covers the 20 genuinely-unconsumed routes across the six audited groups; the executive-insights `POST /crystallize` and `GET /{id}` are listed under the REWIRE group (the FE list/summary reads are the active consumers this shard ships).

---

## What this shard ships (the one REWIRE)

The **executive-insights** group is the unambiguous REWIRE: the UI home already exists (`ExecutiveAIBrief.tsx`) and was synthesizing a brief client-side via cognitive RAG instead of reading the crystallized narratives the backend already produces. This shard adds an **opt-in, additive** real-data path:

- `frontend/src/types/executive-insights.ts` — TS types mirroring `ExecutiveInsightResponse` (subset the UI needs).
- `frontend/src/api/executive-insights.ts` — `listExecutiveInsights()` wrapping `GET /api/executive-insights`.
- `frontend/src/hooks/api/use-executive-insights.ts` — `useExecutiveInsights(brand)` (disabled when brand is empty).
- `frontend/src/lib/query-client.ts` — `queryKeys.executiveInsights` group.
- `frontend/src/components/insights/ExecutiveAIBrief.tsx` — when crystallized insights exist for the brand, render their **real narratives**; otherwise the existing cognitive-RAG / `SAMPLE_BRIEF` fallback is unchanged.

**No fabricated values** are introduced: the real-data path renders the backend's `title`/`narrative` verbatim. The `SAMPLE_BRIEF` placeholder is intentionally left in place as the fallback — its removal belongs to the **Phase-05 SAMPLE_\*** shard, not this one (REASON-BEFORE-RULES: do not delete a requested placeholder whose replacement is another shard's job).

### Enforcement: reachability regression guard

`tests/api/test_executive_insights_reachability.py` asserts — via the **live Starlette route-matcher on the real app object** — that the KEPT/REWIRE routes still resolve to their real handler functions. This is the only faithful, auth-independent route-resolution check: the global JWT middleware returns 401 for any path *before* routing, so an unauthenticated HTTP probe (curl) and the OpenAPI schema both miss route-resolution — only `app.router.routes[*].matches(scope)` reveals it. The test turns RED only if a future cleanup silently drops a KEPT route while triage is still "KEEP/REWIRE".

---

## Owner sign-off required (do not action without approval)

The KEEP / consolidate calls below are **deliberately NOT executed**. Nothing here is auto-applied. Each requires explicit owner approval before any deletion, consolidation, or roadmap reprioritization.

- [ ] **copilotkit analytics ×5** (`GET /copilotkit/analytics/{usage,agents,errors,hourly}`, `GET /copilotkit/feedback/stats`) — KEEP as chatbot-ops API-only telemetry, **OR** consolidate into the existing `/analytics` view/router. Recently security-touched (`4d7b1026`, 2026-06-07); **do not delete** without sign-off.
- [ ] **sentinels CRUD ×5** (`POST/GET /api/sentinels`, `GET/PATCH/DELETE /api/sentinels/{id}`) — build the operator/admin UI, **OR** keep API-only. BOLA/IDOR-hardened 2026-06-03 (`778ea0af`, `ea899743`); **do not delete**.
- [ ] **alerts `GET /alerts/stream` SSE** — wire a live alerts banner (`EventSource` client), **OR** keep as a roadmap-staked realtime feed. Currently no FE consumer.
- [ ] **causal `GET /pipeline/{id}`** — consume when the FE moves to async pipeline polling; keep otherwise. Currently the FE runs pipelines synchronously.
- [ ] **feedback GEPA ×4** (`POST /trace`, `GET /agent/{name}/{stats,signals,gepa-batch}`) — confirm these stay as the optimizer-facing surface for the DSPy self-improvement loop (PR #792); **no UI is expected**. Keep.

**This shard ships ONLY the executive-insights REWIRE. No backend endpoint is deleted. Any deletion or roadmap reprioritization above requires owner approval.**

---

## Verification evidence (re-run to reproduce)

```bash
# Zero-consumer (each returns 0 hits; copilotkit hits are POST /copilotkit/feedback submit only):
grep -rn "sentinels" frontend/src --include="*.ts" --include="*.tsx" | grep -v "/generated/"
grep -rn "executive-insights\|crystalliz\|portfolio-summary" frontend/src --include="*.ts" --include="*.tsx" | grep -v "/generated/"
grep -rn "gepa-batch\|feedback/trace\|agent/.*signals" frontend/src --include="*.ts" --include="*.tsx" | grep -v "/generated/"
grep -rn "copilotkit/analytics\|feedback/stats" frontend/src --include="*.ts" --include="*.tsx" | grep -v "/generated/"
grep -rn "/pipeline/" frontend/src/api/causal.ts            # → /pipeline/sequential, /pipeline/parallel (POST) only
grep -rn "alerts/stream\|EventSource" frontend/src --include="*.ts" --include="*.tsx" | grep -v "/generated/"

# Live handler-name resolution (all six resolve as asserted in the reachability test):
.venv/bin/python -m pytest tests/api/test_executive_insights_reachability.py -n0 -q
```

Live matcher output (2026-06-08), confirming the asserted handler names:

| Method | Path | Resolved handler |
|---|---|---|
| GET | `/api/executive-insights` | `list_executive_insights` |
| GET | `/api/executive-insights/portfolio-summary` | `get_portfolio_summary` |
| GET | `/api/feedback/agent/tool_composer/signals` | `get_optimization_signals` |
| GET | `/api/feedback/agent/tool_composer/gepa-batch` | `get_gepa_training_batch` |
| GET | `/api/causal/pipeline/some-id` | `get_pipeline_status` |
| GET | `/api/alerts/stream` | `alerts_stream` |
