# Chat & Routing-Classifier API Reference

**Version**: 1.0 | **Last Updated**: 2026-07-31 | **Closes #1346**

Source of truth: `src/api/routes/copilotkit.py` (endpoints, handler, schemas),
`src/api/routes/chat_bridge.py` (bridge), `src/api/routes/chat.py` (suggestion
pills), `src/api/middleware/auth_middleware.py` (auth allowlist),
`src/agents/orchestrator/nodes/intent_classifier.py` +
`src/agents/orchestrator/nodes/router.py` (classifier-mode semantics). Every
claim below was verified against those files on 2026-07-31; symbol names are
cited so the reference survives line drift.

---

## 1. The two chat surfaces

| Surface | Endpoint | Brain | Notes |
|---|---|---|---|
| Real copilot UI | `POST /api/copilotkit/agent/default` (CopilotKit/AG-UI protocol) | `chat_node` + bound tools + `synthesize_node` | What the browser sidebar speaks; answers are tool-grounded; scriptable via `scripts/demos/copilot_agui_runner.py` |
| Scripted chat API | `POST /api/copilotkit/chat/stream` (SSE) | `classify → orchestrator → generate` (`src/api/routes/chatbot_graph.py`) | Deterministic routing instrument; fails closed on conversational queries by design (#883), with the #1336 bridge fallback (§6) |
| Non-streaming variant | `POST /api/copilotkit/chat` | same as `/chat/stream` | Single JSON `ChatResponse` (§5) |

The demo/measurement implications of the split are documented in
`docs/demos/COPILOT_CHAT_DEMO_SCENARIOS_V2.md`.

---

## 2. CopilotKit / AG-UI endpoints

Routes are registered by `add_copilotkit_routes()` (`copilotkit.py`) as a base
route `/api/copilotkit` plus a catch-all `/api/copilotkit/{path:path}` — they
are `include_in_schema=False`, so they do **not** appear in the OpenAPI schema
(this file is their reference).

### 2.1 Discovery (public)

- `GET /api/copilotkit` or `GET /api/copilotkit/info` — SDK info response,
  transformed to frontend v1.x format by `transform_info_response()`:
  `{"actions": [...], "agents": {"default": {"description": ...}}, "version": ...}`.
- `POST /api/copilotkit` with an empty body, `{}`, `{"action": "getInfo"}` or
  `{"method": "info"}` — same info response.
- `GET /api/copilotkit/status` — integration status (agent/action counts, LLM
  provider). GET-only public; POST to `/status` requires auth.

### 2.2 Execution (JWT required)

The CopilotKit JSON-RPC protocol mixes discovery and execution under the same
paths via the body `method` field, so auth is **body-aware**
(`copilotkit_custom_handler` → `_require_auth_for_copilotkit_execution`):
execution-shaped POSTs to the middleware-public base path (`agent/run`,
`agent/connect`, `action/run`, SDK fallback) are gated in-handler (401 without
a valid `Authorization: Bearer` Supabase JWT). Every other CopilotKit sub-path
(`/agent/{name}`, `/action/{name}`, `/agents/execute`, …) requires JWT at the
middleware (`auth_middleware.py PUBLIC_PATHS` — only the base, `/status` GET,
and `/info` are public).

**`POST /api/copilotkit/agent/default`** — run the `default` agent (the only
registered agent). Body (AG-UI protocol; the runner script shows the minimal
form):

```json
{
  "threadId": "<conversation id>",
  "state": {},
  "messages": [ {"id": "...", "role": "user", "content": "..."} ],
  "actions": []
}
```

- **Full history resent each turn**: the frontend sends the whole message list
  every run; the graph has no server-side cross-run memory of its own (the
  LangGraph checkpointer thread is deliberately **fresh per request** —
  `LangGraphAgent.execute` regenerates `thread_id` to defeat the SDK's
  regenerate mode, v1.9.4 fix).
- **`threadId` ≡ DB `session_id`**: the *original* frontend `threadId` is
  carried as `persistent_session_id` into graph state and the
  `_session_id_context` contextvar; message persistence, analytics and
  learning signals key on it. CopilotKit threadIds are bare UUIDs, so chat
  attribution falls back to the verified JWT identity
  (`set_authenticated_user` in `_require_auth_for_copilotkit_execution`).
- **Response**: SSE (`text/event-stream`), `data: {...}\n\n` frames carrying
  AG-UI protocol events (`RUN_STARTED`, `TEXT_MESSAGE_*`, `MESSAGES_SNAPSHOT`,
  …; PascalCase-typed, serialized by `agent.execute`). Stream errors are
  emitted as `{"type": "RUN_ERROR", "message": ..., "code": "STREAM_ERROR"}`.

The same execution can be reached as `POST /api/copilotkit` with
`{"method": "agent/run", "body": {"threadId": ..., "messages": [...]}}` (the
custom handler's streaming branch); `{"method": "agent/connect"}` is
acknowledged with `{"status": "connected"}`.

---

## 3. `POST /api/copilotkit/chat/stream` (SSE)

Auth: `Depends(require_viewer)` (any authenticated role). Request body is
`ChatRequest`:

| Field | Type | Notes |
|---|---|---|
| `query` | str, required | User query text |
| `user_id` | str, required | **NON-AUTHORITATIVE** (kept for backward compatibility). Identity always comes from the JWT; a mismatching body value is rejected 403 (`_resolve_chat_identity`, IDOR Finding 1) |
| `request_id` | str? | Falls back to the `X-Request-ID` header |
| `session_id` | str? | Generated as `{user_id}~{uuid4}` when empty |
| `brand_context` | str? | Validated against the caller's brand grants — out-of-grant values are rejected 403 (`_resolve_chat_brand`, H1/#694 write-poisoning guard) |
| `region_context` | str? | Region filter |

Response: `text/event-stream` with `X-Request-ID` echoed. **Framing is
`data: {json}\n\n` lines only — there are no `event:` lines.** Frame shapes
(`_stream_chat_response`):

```text
data: {"type": "session_id", "data": "<session id>"}
data: {"type": "text", "data": "<incremental text chunk>"}
data: {"type": "conversation_title", "data": "<title>"}
data: {"type": "tool_call", "data": "..."}            ← declared in the frame contract; no producer in the current generator body
data: {"type": "dispatch_info", "data": { ...see below... }}
data: {"type": "done", "data": ""}
data: {"type": "error", "data": "<generic client-safe message>"}
```

Order: `session_id` first; `text` / `conversation_title` as produced;
`dispatch_info` once, immediately before `done`. On any exception the stream
emits a single `error` frame with a generic message (internal detail is logged
server-side only, Finding 3).

### `dispatch_info` payload

```json
{
  "orchestrator_used": false,
  "agents_dispatched": [],
  "routed_agent": null,
  "response_confidence": null,
  "intent": null,
  "intent_confidence": null,
  "routing_rationale": null,
  "routing_pattern": null,
  "classification_latency_ms": null,
  "used_llm_layer": null,
  "execution_time_ms": 1234.56
}
```

The last three (before `execution_time_ms`) are the 4-stage
ClassificationPipeline observability fields added in PR #1330; they stay
`null` when `ORCHESTRATOR_CLASSIFIER_MODE=off` or when the orchestrator was
not consulted for the turn.

---

## 4. `POST /api/copilotkit/chat` (non-streaming)

Same auth, identity and brand rules as `/chat/stream`; same `ChatRequest`
body. Returns a single `ChatResponse` JSON object. Errors return HTTP 200 with
`success=false` and a generic `error` string (403 identity/brand rejections
propagate as real 403s).

## 5. `ChatResponse` schema

Defined in `copilotkit.py` (`class ChatResponse`); mirrored in
`frontend/src/types/generated/api.ts` and `frontend/src/lib/api-schemas.ts`.
**Any change triggers the verify-types triple** (`make generate-types` →
commit `api.ts`, update zod mirror + fixture) in the same PR.

| Field | Type | Meaning |
|---|---|---|
| `success` | bool | |
| `session_id` | str | |
| `response` | str | Answer text |
| `conversation_title` | str? | |
| `agent_name` | str? | Answering agent; `"chat_bridge"` when the bridge authored the answer (§6) |
| `error` | str? | Generic client-safe message |
| `orchestrator_used` | bool | Dispatch observability (Phase 1) |
| `agents_dispatched` | list[str] | |
| `routed_agent` | str? | Router's choice — preserved even when the bridge answers |
| `response_confidence` | float? | 0.0–1.0 |
| `execution_time_ms` | float? | |
| `intent` | str? | Legacy classified intent |
| `intent_confidence` | float? | |
| `routing_rationale` | str? | Phase 4 routing transparency |
| `routing_pattern` | str? | **PR #1330** — 4-stage pipeline decision (`SINGLE_AGENT` / `PARALLEL_DELEGATION` / `TOOL_COMPOSER` / `CLARIFICATION_NEEDED`); `null` when mode=off or orchestrator not consulted |
| `classification_latency_ms` | float? | **PR #1330** — pipeline latency (measured median 0.72 ms) |
| `used_llm_layer` | bool? | **PR #1330** — whether the pipeline's LLM stage ran (currently hard-disabled: `_get_classification_pipeline()` constructs with `enable_llm_layer=False` pending the async stage-3 implementation) |

---

## 6. `ORCHESTRATOR_CLASSIFIER_MODE` (off / shadow / active)

Read lazily per call (`intent_classifier._classifier_mode`; flip the droplet
`.env` + restart, no rebuild):

| Mode | Pipeline runs? | Routing authority | `classification_logs` write |
|---|---|---|---|
| `off` | no | legacy only | no |
| `shadow` (default) | yes | legacy only — decision surfaced in `dispatch_info`/`ChatResponse` | yes (fire-and-forget, fail-open; suppressed under `E2I_TESTING_MODE`) |
| `active` | yes | pipeline **iff confident**, else legacy | yes |

**Active-mode abstention** (`RouterNode._dispatch_from_classification`): the
pipeline takes routing authority only when its pattern is dispatchable AND
`confidence ≥ 0.5` (`RouterNode.MIN_ACTIVE_CONFIDENCE`). It abstains — legacy
intent routing proceeds unchanged — on `CLARIFICATION_NEEDED`, confidence
< 0.5, empty targets, or an unknown pattern. Measured on the 2026-07-29
active subset: 2/10 engaged, 8/10 abstained, zero unsafe dispatches
(`docs/demos/results/2026-07-29_copilot_chat_perf/SUMMARY.md`).

Note: abstention means **CLARIFICATION_NEEDED never reaches the user as a
clarify flow** — the legacy path cannot produce it either (open issue #1407).

### Chat bridge fallback (#1336, PR #1394)

`src/api/routes/chat_bridge.py run_conversational_bridge()`, called from the
orchestrator node in `chatbot_graph.py` **only on complete orchestrator
failure** (zero successful agents): the turn is re-run through the AG-UI brain
and its grounded answer is streamed behind the honest preamble *"The full
analysis pipeline couldn't complete for this question, so here's what I can
tell you from the data available:"*. Properties (all verified in source):

- Fires only on complete failure; partial/full successes untouched.
- Fails open to the status quo: any bridge error/timeout returns `None` and
  the caller keeps the original fail-closed summary. Never raises.
- Kill switch `E2I_CHAT_BRIDGE_ENABLED` (default `true`); timeout
  `E2I_CHAT_BRIDGE_TIMEOUT_S` (default 90 s); history capped at 8 messages.
- Runs under a shadow session `"{session_id}~bridge"` so bridged turns don't
  double-write the real session's history; `ChatResponse.agent_name` becomes
  `"chat_bridge"` while `routed_agent` keeps the router's choice.

---

## 7. Classifier telemetry persistence

Every shadow/active pipeline decision is written (fire-and-forget, fail-open)
to `classification_logs` via
`src/repositories/classification_log.py record_classification()`; the nightly
labeler (`src/tasks/routing_label_tasks.py`, Celery beat
`routing-label-nightly`, 04:30 UTC) fills `was_correct`/`correct_pattern`/
`feedback_notes`, and per-run safety telemetry snapshots to
`routing_classifier_metrics` (`src/tasks/routing_metrics.py`). Column
reference: `docs/data/07-SUPPORTING-SCHEMAS.md` §"Routing Classifier Schema".

## 8. Related chat endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/chat/suggestions` | `require_auth` | Conversation-/page-adaptive suggestion pills (one fast-tier LLM call; 502 on failure → frontend falls back to static pills). `src/api/routes/chat.py` |
| `POST /api/copilotkit/feedback` | `require_viewer` | Per-message thumbs feedback (consumed by the #1341 labeler as its strongest signal) |
| `GET /api/copilotkit/feedback/stats`, `GET /api/copilotkit/analytics/usage`, `GET /api/copilotkit/analytics/agents` | `require_viewer` | Feedback/usage/agent analytics |
| `GET /api/copilotkit/kpis/summary` | public | Real `business_metrics` KPI rollup (Home QUICK_STATS) |
