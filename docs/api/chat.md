# Chat & Routing-Classifier API Reference

**Version**: 1.1 | **Last Updated**: 2026-09-07 | **Closes #1346**

Source of truth: `src/api/routes/copilotkit.py` (endpoints, handler, schemas,
AG-UI channels), `src/api/routes/chatbot_graph.py` (`/chat/stream` graph,
clarification), `src/api/routes/chatbot_tools.py` (bound tools),
`src/api/routes/chat_bridge.py` (bridge), `src/api/routes/chat.py` (suggestion
pills), `src/services/chat_capability_catalog.py` (pill catalog + validator),
`src/api/utils/sse_keepalive.py` (SSE keepalive), `src/services/enum_labels.py`
(region resolution), `src/api/middleware/auth_middleware.py` (auth allowlist),
`src/agents/orchestrator/nodes/intent_classifier.py` +
`src/agents/orchestrator/nodes/router.py` (classifier-mode semantics). Every
claim below was verified against those files on 2026-09-07; symbol names are
cited so the reference survives line drift.

Reference convention: a bare `#N` is a **GitHub issue**; `PR #N` is a **merged
pull request**. They are different numbers for the same work.

---

## 1. The two chat surfaces

| Surface | Endpoint | Brain | Notes |
|---|---|---|---|
| Real copilot UI | `POST /api/copilotkit/agent/default` (CopilotKit/AG-UI protocol) | `chat_node` + bound tools (§2.3) + `synthesize_node` | What the browser sidebar speaks; answers are tool-grounded; sees the page via `state.filters` + `context` readables (§2.2); scriptable via `scripts/demos/copilot_agui_runner.py` |
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
  "state": {"filters": { "brand": "Remibrutinib", "region": "All US" }},
  "messages": [ {"id": "...", "role": "user", "content": "..."} ],
  "actions": [],
  "context": [ {"description": "<what this readable is>", "value": "<JSON or prose>"} ]
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

#### `state.filters` — the dashboard-filter channel (PR #1724)

The UI's active filters ride CoAgent shared state (`useCoAgent` in
`frontend/src/components/chat/AgentFiltersBridge.tsx`) and arrive as
`state.filters`. `filters` is declared as its own channel on the graph state
(`E2IAgentState.filters` in `copilotkit.py`) — **LangGraph drops input keys
that are not declared state channels**, so an undeclared key would vanish
silently. Shape mirrors the UI's `E2IFilters`:

| Key | Type | Notes |
|---|---|---|
| `brand` | str | `"All"` is the **no-selection sentinel**, not a brand constraint |
| `region` | str | `"All US"` (and `"All"`) is the no-selection sentinel — PR #1724, extended to Region by #1753 |
| `dateRange` | `{start, end}` | Both ISO strings, or the field is ignored |
| `territory` | str | Passed through verbatim |
| `hcpSegment` | str | Passed through verbatim |

`_filters_context_note()` renders the non-sentinel values into an **ACTIVE
DASHBOARD FILTERS** system-prompt suffix used by both `chat_node` and
`build_synthesis_prompt`. Its instruction is **resolve-don't-ask**: when the
user's message does not name a brand, region, period, territory or segment,
the model resolves it from the filters and says which value it used, instead of
asking "which brand?". Explicit user wording in the message always wins over
the filter. When no non-sentinel filter is present the note is `""`, so
filter-less runs keep the prompt byte-identical.

> The typed channel exists **because** a sentinel-aware reading is impossible
> from the raw readables below: a readable can report `brand: "All"` but not
> that `"All"` means *no constraint*.

#### `context` — the AG-UI readables channel (PR #1818)

Every `useCopilotReadable` on the page arrives in the `agent/run` body as
`context`, a list of `{"description": str, "value": str}` (the client
JSON-stringifies `value`). `_coerce_agui_context()` normalises it, the SDK
merges it to `state["copilotkit"]["context"]`, and `_readables_context_note()`
renders it into an **ON-SCREEN APP CONTEXT** system-prompt block used by
`chat_node` and `build_synthesis_prompt`.

- Instruction: answer questions about "the data on the page / screen / GUI"
  from this context **first**, computing counts and ranks directly from it, and
  say which on-screen values were used. Call tools only for data not on screen.
- Budget: `_READABLE_ITEM_MAX_CHARS = 12_000` per readable,
  `_READABLES_TOTAL_MAX_CHARS = 32_000` total; over-budget items are replaced
  by an `[N readable(s) omitted: context budget exceeded]` line.
- A readable whose value is not JSON is treated as a **prose page summary**;
  the prompt then adds a rule forbidding the model from presenting its figures
  as a tool result. Pages that publish no prose summary keep the pre-summary
  prompt byte-identical.
- Historical note: comments claiming readables "never leave the browser" were
  reading the `RUN_STARTED` echo of a `context` the route had itself zeroed.
  Measured 2026-08-26, they **do** arrive in the request body.

#### Empty-delta invariant (PR #1724)

**A `TEXT_MESSAGE_CONTENT` event is never emitted with an empty `delta`.** The
browser's `@ag-ui/core` Zod schema refines that field with `s.length > 0`, and
a single invalid event aborts the **entire** CopilotKit run client-side while
the backend 200s, finishes the run and persists the answer. `ag-ui-protocol`
0.1.18 (the image's pin) no longer validates `delta` server-side, so the
boundary must: `_is_zod_fatal_empty_content()` drops such events on every
serialization branch (str, pydantic v1, pydantic v2).

#### SSE keepalive and proxy timeouts

Both `/agent/default` and `/chat/stream` wrap their body in
`with_sse_keepalive()` (`src/api/utils/sse_keepalive.py`; #1659 for
`/chat/stream`, #1669 for the AG-UI surface) and set
`X-Accel-Buffering: no`.

- `SSE_KEEPALIVE_INTERVAL_SECONDS = 15` — after 15 s with no upstream frame the
  wrapper emits `: keepalive\n\n`, an **SSE comment**. Per the EventSource spec
  a line beginning with `:` is ignored by conforming parsers, so it resets
  nginx's read timer without appearing as an event to any consumer.
- `PROXY_READ_TIMEOUT_SECONDS = 300` mirrors `proxy_read_timeout` in
  `docker/nginx/host-nginx.conf` for the `/api/` and `/copilotkit/` locations;
  `tests/unit/test_tests_meta/test_proxy_ceiling_coherence_1659.py` parses the
  nginx file and fails if the constant drifts from it.
- Why it is needed: `proxy_read_timeout` bounds the gap *between* reads, not
  total duration. Every `/chat/stream` frame originates from a LangGraph
  node-completion update, and the orchestrator is one node, so the silent
  window is the whole turn (measured on production: 34 395.7 ms of silence
  between `session_id` and the first `text` frame). The keepalive replaces
  "total turn wall time < proxy ceiling" with a relation between two constants
  in one module.
- What it does **not** fix: a keepalive is a coroutine, so compute that blocks
  the event loop still starves it.

### 2.3 Bound tools

`chat_node` binds `E2I_CHATBOT_TOOLS` (`src/api/routes/chatbot_tools.py`) plus
any frontend `useCopilotAction` schemas that arrived on
`state["copilotkit"]["actions"]`. The backend set is ten `@tool` functions:

| Tool | Purpose |
|---|---|
| `e2i_data_query_tool` | Query platform data (brand / region / metric filters) |
| `kpi_calculate_tool` | Compute a KPI from `business_metrics` |
| `causal_analysis_tool` | Run a causal effect estimate |
| `clinical_context_tool` | Brand-faithful sourced clinical context |
| `agent_routing_tool` | Describe / choose the agent for an ask |
| `conversation_memory_tool` | Recall prior turns for this session |
| `document_retrieval_tool` | RAG document lookup |
| `orchestrator_tool` | Dispatch the multi-agent orchestrator |
| `tool_composer_tool` | Compose a multi-step tool plan |
| `predict_hcp_segment_likelihood_tool` | Per-HCP-segment likelihood-to-prescribe (PR #1399) |

`E2I_TOOL_MAP` maps tool name → function, and the graph's `ToolNode` is
constructed from the same list, so the bound set and the executable set cannot
diverge.

#### Region resolution and clarification

Region-taking tools resolve their `region` argument through
`resolve_region_label(..., allow_synonyms=True)`
(`src/services/enum_labels.py`). Resolvable: the four US census regions
(`northeast`, `south`, `midwest`, `west`) in any casing, separator variants,
and every synonym the platform's entity extraction recognises (`NE`,
`new england`, `pacific`, `west coast`). Since #1565 a leading `the` and a
trailing `region` / `area` are stripped at lookup, so *"the Northeast region"*
resolves.

Genuinely ambiguous phrasings — *"East"*, *"East Coast"*, *"central coast"* —
span more than one census region and deliberately do **not** resolve (#1572).
An unresolvable region does not silently produce a national figure: the tool
returns zero rows plus a clarify hint (`_REGION_CLARIFY_HINT`) so the model
asks which census region was meant. The KPI response's `region_status` says
whether the figure is region-scoped at all — only `applied` means it is;
`not_applicable` means the KPI has no region variant and the value is global.

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

Initialised with every key present (nulls, not omissions), so a consumer can
read any field without a presence check:

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
  "routing_authority": null,
  "classification_latency_ms": null,
  "used_llm_layer": null,

  "node_wall_ms": null,
  "graph_total_ms": null,
  "untimed_overhead_ms": null,
  "first_request_in_worker": null,
  "worker_pid": null,
  "orchestrator_stage_ms": null,
  "orchestrator_run_ms": null,
  "orchestrator_untimed_ms": null,
  "rag_stage_ms": null,
  "rag_meta": null,

  "empty_response_fallback": false,
  "execution_time_ms": 1234.56
}
```

| Group | Keys | Source |
|---|---|---|
| Dispatch (Phase 1 / Phase 4) | `orchestrator_used`, `agents_dispatched`, `routed_agent`, `response_confidence`, `intent`, `intent_confidence`, `routing_rationale` | Orchestrator node output |
| 4-stage classifier | `routing_pattern`, `classification_latency_ms`, `used_llm_layer` (PR #1330); `routing_authority` (PR #1596) | Orchestrator node output; `null` when `ORCHESTRATOR_CLASSIFIER_MODE=off` or the orchestrator was not consulted |
| Latency span (#1454, PR #1471/#1474) | `node_wall_ms` (per-node dict), `graph_total_ms`, `untimed_overhead_ms`, `first_request_in_worker`, `worker_pid` | A synthetic `__latency_span__` stream item — observability only, never rendered as answer text |
| Orchestrator-internal (#1475) | `orchestrator_stage_ms`, `orchestrator_run_ms`, `orchestrator_untimed_ms` | Same span item |
| RAG-internal (#1484) | `rag_stage_ms`, `rag_meta` | Same span item |
| Zero-char guard (#1561) | `empty_response_fallback` | `true` when the turn produced no text and the fallback envelope was emitted (see below) |
| Timing | `execution_time_ms` | Wall time of the whole request, stamped last |

**`empty_response_fallback`** — a zero-char completion must never close as a
silent HTTP 200. The guard checks the **total** response at stream end
(individual empty chunks are benign); when it is empty it sets the flag, logs
at ERROR, and emits one `text` frame carrying `_EMPTY_STREAM_FALLBACK` ("I
wasn't able to produce a response for this question — the analysis ran but
returned no text…"). The guard and the §6 bridge are mutually exclusive by
construction: bridge-authored text makes the guard inert.

**`worker_pid` / `first_request_in_worker`** name which gunicorn worker served
the request and whether it was that worker's first — the pair that made the
#1454 cold-start (~68 s of DSPy classifier + agent-registry init) legible
without scraping logs.

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
| `routing_authority` | str? | **PR #1596** (#1582) — which subsystem actually produced this turn's dispatch plan: `"pipeline"` \| `"legacy"` \| `"explicit_target"` (PR #1723, #1714). Distinct from `routing_pattern`, which is the *pipeline's* decision and is emitted in shadow mode too — so an abstaining pattern beside a real `agents_dispatched` no longer reads as a routing regression. Set by `RouterNode` |
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

Abstention means the *pipeline's* `CLARIFICATION_NEEDED` pattern never takes
routing authority. It no longer means the user gets no clarification.

### Multi-turn clarification ask-back (#1407, PR #1441)

`/chat/stream` now asks back. An underspecified analytical ask routes to
`clarify_node` (`src/api/routes/chatbot_graph.py`) instead of paying a full
~13–17 s orchestrator dispatch that fails closed:

- **Eligible intents only**: `CLARIFY_INTENTS = {KPI_QUERY, CAUSAL_ANALYSIS}`.
  Greeting, help, agent_status, general, search, recommendation and cohort
  never clarify — they need no slots or have their own handling.
- **Trigger**: no brand **and** no metric **and** no prior referent in
  conversation context. A brand counts as filled when the query names one of
  the tracked brands (`_REAL_BRANDS`).
- **Output**: 1–3 clarifying questions (LLM-generated, canned fallback on
  failure), with the pending ask persisted to `chatbot_conversations.metadata`
  so the **next** turn resumes on a merged query. A detected pivot drops the
  pending ask instead of merging.
- **Knobs**: `CHATBOT_CLARIFY` (default `true`; set `false` to force the legacy
  `retrieve_rag` path), `CHATBOT_CLARIFY_TTL_MINUTES` (default `30`) after
  which a pending ask is dropped and the next turn is treated as new.

### Chat bridge fallback (#1336, PR #1394)

`src/api/routes/chat_bridge.py run_conversational_bridge()`, called from the
orchestrator node in `chatbot_graph.py` **only on complete orchestrator
failure** (zero successful agents): the turn is re-run through the AG-UI brain
and its answer is streamed behind an honest preamble. Properties (all verified
in source):

- **Two preamble variants, selected by evidence** (#1451/#1458 —
  `build_bridge_preamble()`). `BridgeAnswer.tool_grounded` is true only when
  the AG-UI run executed a tool **and** at least one result was not a
  fail-closed `{"success": false, …}` envelope:
  - grounded → *"Answered from live platform data, pulled through the analytics
    tools just now. The deeper multi-agent analysis did not run for this
    question."*
  - ungrounded → *"Answered directly by the platform assistant. The deeper
    multi-agent analysis did not run for this question."* — claiming live
    platform data here would be a fabricated provenance claim.

  When the dispatcher wrote one, the primary failed agent's `user_action`
  invitation is appended (only ever one — two invitations for one turn
  contradict each other). The pre-#1451 wording ("The full analysis pipeline
  couldn't complete for this question…") led with the internal pipeline's
  outcome and buried a correct answer under an apology; it is gone.
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

Labeler knobs (read per run from the environment,
`src/tasks/routing_label_tasks.py`):

| Env var | Default | Effect |
|---|---|---|
| `ROUTING_LABEL_MIN_NEW_ROWS` | `10` | The run no-ops below this many unlabeled rows (a `force=True` argument overrides) |
| `ROUTING_LABEL_JUDGE_CAP` | `50` | Max LLM-judge calls per run — token spend and droplet capacity (a `judge_cap` argument overrides) |
| `ROUTING_LABEL_JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Judge model id |
| `ROUTING_LABEL_LOOKBACK_DAYS` | `30` | How far back the run considers rows |

## 8. Related chat endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/chat/suggestions` | `require_auth` | Conversation-/page-adaptive suggestion pills. See below. `src/api/routes/chat.py` |
| `POST /api/copilotkit/feedback` | `require_viewer` | Per-message thumbs feedback (consumed by the #1341 labeler as its strongest signal) |
| `GET /api/copilotkit/feedback/stats` | `require_viewer` | Feedback rollup; `agent_name?`, `days` (1–90, default 30) |
| `GET /api/copilotkit/analytics/usage` | `require_viewer` | Usage analytics; `days` (1–90, default 7) |
| `GET /api/copilotkit/analytics/agents` | `require_viewer` | Per-agent analytics; `agent_name?`, `days` (1–90, default 30) |
| `GET /api/copilotkit/analytics/errors` | `require_viewer` | Recent chat errors; `limit` (1–200, default 20) |
| `GET /api/copilotkit/analytics/hourly` | `require_viewer` | Usage distribution by hour of day for capacity planning; `days` (1–90, default 7) |
| `GET /api/copilotkit/kpis/summary` | public | Real `business_metrics` KPI rollup (Home QUICK_STATS); `brand` (default `All`), `region?` |

### `POST /api/chat/suggestions` — catalog-grounded pills (PR #1900)

One fast-tier LLM call (`max_tokens=600`, 8 s timeout) returning up to four
pills. Non-empty `messages` → follow-ups from the recent transcript; empty
`messages` (opener mode) → openers grounded in `page_context`.

The prompt is a **template**, not a fixed string: `{capability_catalog}` and
`{route_hint}` are filled per request from
`src/services/chat_capability_catalog.py` (KPI registry, history coverage and
the other capability sources, derived from code and data — #1638). Proposing an
analysis, grain, axis or metric outside that catalog is a defect, so the reply
is then **post-filtered**: `filter_unsupported_pills()` matches each pill's
title + message against the catalog's rules and drops the unsupported ones,
logging each drop at INFO (`rule=…`) — the drop rate is the production
measurement of how often the prompt still proposes an unanswerable pill.

`502` on any generation or parsing failure, **and** when the post-filter drops
every pill; the frontend falls back to its static context-aware pills.
