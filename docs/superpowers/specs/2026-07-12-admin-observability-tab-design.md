# Admin Observability Tab — LLM Usage, Tokens, and Cost

**Date:** 2026-07-12
**Status:** Approved design (Approach A — unified usage-events table with contextvar attribution)
**Scope decision (user):** "Chat + platform totals" — per-user/per-session attribution for chatbot
usage, plus a separate "Platform LLM usage" section showing non-chat spend (insights, agents,
RAG) as aggregates explicitly NOT attributed to users.

## 1. Problem

The `/admin` page has two tabs (Users, Activity). The user wants a third **Observability** tab
showing which LLMs are used, tokens consumed, and $ cost — per user and per session.

**Verified state of the world (live prod DB, 2026-07-12):**

- `chatbot_messages` already has `model_used TEXT`, `tokens_used INTEGER`, `latency_ms INTEGER`
  columns — but across all 320 rows (119 assistant), **zero** have `tokens_used` or `model_used`
  populated. 112/119 assistant rows carry `metadata->>'model_used'` (e.g.
  `anthropic:claude-sonnet-4-6`), but **no token counts exist anywhere**.
- `chatbot_graph.py::_execute_finalize()` stamps `model_used` from an env var (not the actual
  model) and never passes tokens. `copilotkit.py::_persist_message_sync()` writes neither column.
- Non-chat LLM traffic (insights, Executive Brief, RAG, DSPy agents) is not recorded at all.
- `src/mlops/agent_cost_tracker.py` is an unwired G25 scaffold (zero production consumers,
  in-memory, stale pricing). Per reason-before-rules it is a per-agent *budget* scaffold on a
  different axis — **left untouched**. We reuse its pricing-table *pattern* only.

Conclusion: this feature is two layers — a **capture layer** (nothing usable exists today) and a
**display layer**.

## 2. Data model — migration `database/migrations/104_llm_usage_events.sql`

One append-only table:

```sql
CREATE TABLE IF NOT EXISTS llm_usage_events (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider      TEXT NOT NULL,            -- 'anthropic' | 'openai'
    model         TEXT NOT NULL,            -- exact model id from the RESPONSE, not the request
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    surface       TEXT NOT NULL,            -- 'chat' | 'insights' | 'rag' | 'agent' | 'other'
    component     TEXT,                     -- finer label when known (e.g. 'exec_brief')
    user_id       UUID,                     -- NULL => platform-level call
    session_id    VARCHAR,                  -- chat session ('<user_id>~<uuid>' format)
    request_id    TEXT                      -- correlation with a chat run (run_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_user    ON llm_usage_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_session ON llm_usage_events (session_id);
```

- **No `cost` column.** Cost is computed at read time from tokens × pricing table so pricing
  corrections apply retroactively; no stale baked values.
- No rollup tables — expected volume is hundreds of rows/day; aggregate at query time.
- Migration contains **no BEGIN/COMMIT** (established migration convention); applied on the
  droplet via the standard `docker exec supabase-db psql` flow; dry-run first with
  `BEGIN; \i file; ROLLBACK;`.
- 104 verified as the next free migration number (103_model_health_named_metrics.sql is highest).

## 3. Capture layer

Two hooks cover every LLM call site in the codebase; both are **fail-open** (a capture failure
logs a warning and never breaks the request).

### 3.1 Attribution contextvar — `src/utils/llm_attribution.py` (new)

A single contextvar holding `{user_id, session_id, surface, component, request_id}` plus a
mutable per-run token accumulator:

- The CopilotKit chat entrypoint sets it per run (same two-channel pattern already proven for
  `_run_id_context`; user_id derived from the session_id prefix, anonymous UUID
  `00000000-…-0000` treated as NULL user).
- Non-chat callers may optionally set `{surface, component}` (e.g. insights endpoints); when
  unset, hooks fall back to `surface='other'` and NULL user/session — an honest platform-level
  row, never a guessed attribution.

### 3.2 LangChain hook — callback attached in `llm_factory`

- A `UsageRecorderCallback` (LangChain `BaseCallbackHandler`, `on_llm_end`) reads
  `usage_metadata` / `response_metadata` from the result and enqueues one event.
- Attached inside `_create_anthropic_llm()` / `_create_openai_llm()` via the `callbacks=`
  constructor arg — every one of the ~11 factory consumers is covered with zero call-site edits;
  construction-time callbacks fire on `.astream()` too (verified against langchain-anthropic
  1.3.1 / langchain-openai 1.1.14).
- `ChatOpenAI` gains `stream_usage=True` so streamed OpenAI calls report usage (Anthropic streams
  usage by default).
- The recorded `model` comes from the response metadata; fall back to the requested model name if
  absent.

### 3.3 DSPy/litellm hook — global success callback

- One global `litellm.success_callback` registered once at FastAPI startup (lifespan in
  `src/api/main.py`). Covers ALL `dspy.LM` traffic regardless of instantiation site
  (`src/optimization/dspy_lm.py`, `chatbot_dspy.py`, `cognitive_rag_dspy.py`, `causal_rag.py`,
  `causal_role_classifier_loader.py`) — verified litellm 1.84.0 success callbacks carry the
  usage object.
- Reads the same attribution contextvar.

### 3.4 Recorder — `src/services/llm_usage_recorder.py` (new)

- Bounded in-memory queue + background flush task writing batches to `llm_usage_events` via the
  existing Supabase client patterns.
- Bounded queue ⇒ a DB outage drops events (logged) instead of growing memory unboundedly.
- Both hooks call `recorder.enqueue(event)`; nothing in the request path ever awaits the DB
  write.

### 3.5 Stamp `chatbot_messages` at persist time

The run accumulator (sum of input+output tokens across all LLM calls in the run; last model
used) is flushed into the existing `chatbot_messages.tokens_used` / `model_used` columns when
the assistant message is persisted (`copilotkit.py::_persist_message_sync` — the live chat
path — gains the two fields; `chatbot_graph.py::_execute_finalize`'s env-var `model_used`
stamping is replaced with the same accumulator read so both persist paths agree). Historical
rows stay NULL — displayed as "untracked", never estimated.

## 4. Pricing — `src/services/llm_pricing.py` (new)

Static table, per-1M-token USD rates (pattern borrowed from `agent_cost_tracker.MODEL_PRICING`,
values current as of 2026-07):

| model | input /1M | output /1M |
|---|---|---|
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5-20251001 | $1.00 | $5.00 |
| claude-opus-4-5 | $5.00 | $25.00 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |

- `cost_usd(model, input_tokens, output_tokens) -> Optional[float]` — returns **None** for
  unknown models (surfaced as "unpriced" in API/UI; never silently costed at a default).
- Module exposes `PRICING_VERSION` string so the UI can show provenance.
- Matching is prefix-tolerant (e.g. `anthropic/claude-sonnet-4-6` and dated variants resolve to
  the same row).

## 5. API — extend `src/api/routes/admin.py`

`GET /api/admin/observability/llm-usage?days=30` (`days: int = Query(default=30, ge=1, le=365)`),
gated by the existing `Depends(require_admin)`; DB work via `asyncio.to_thread` like siblings.
Lives on the existing `/admin` router — no new router/tag, so no `openapi_tags`/sentinel-drift
updates needed. Aggregation logic goes in a small service module
(`src/services/llm_observability_service.py`) so the route stays thin, mirroring
`AdminUserService`.

Response payload (one call renders the whole tab):

```jsonc
{
  "summary":  { "total_cost_usd", "input_tokens", "output_tokens", "calls",
                "distinct_users", "days", "tracking_since" },   // earliest event timestamp
  "daily":    [ { "date", "chat_cost_usd", "platform_cost_usd", "tokens" } ],
  "by_user":  [ { "user_id", "email", "sessions", "calls", "input_tokens",
                  "output_tokens", "cost_usd", "models": ["…"] } ],
  "sessions": { "<user_id>": [ { "session_id", "title", "started_at", "calls",
                  "tokens", "cost_usd", "models": ["…"] } ] },
  "platform": [ { "surface", "component", "model", "calls",
                  "input_tokens", "output_tokens", "cost_usd" } ],
  "pricing_version": "…",
  "unpriced_models": ["…"]   // models seen in-window missing from the pricing table
}
```

- Emails joined from the existing admin user listing (`AdminUserService.list_users`).
- Session titles joined from `chatbot_conversations` where available.
- All costs computed at read time via `llm_pricing`; events with unpriced models contribute
  tokens/calls but `null` cost, and the model id is listed in `unpriced_models`.

## 6. Frontend — Observability tab

- `Admin.tsx`: `type Tab = 'users' | 'activity' | 'observability'`; third tablist button.
- New `frontend/src/components/admin/ObservabilityTab.tsx` mirroring `ActivityTab` conventions:
  recharts, `useLlmUsage(days)` hook added to `frontend/src/hooks/api/use-admin.ts`, types in
  `frontend/src/types/admin.ts`, window dropdown (7/30/90 days).

Layout, top to bottom:

1. **Stat cards** — Total cost · Total tokens (in/out) · LLM calls · Active users.
2. **Cost over time** — daily stacked bar chart, chat vs platform series.
3. **Per-user table** — email, sessions, calls, tokens in/out, cost, model chips; rows expand to
   a per-session breakdown (session title, started, calls, tokens, cost, models). Display-only,
   no navigation.
4. **Platform LLM usage (non-chat)** — table by surface/component × model with calls, tokens,
   cost. Clearly labeled as not attributable to users.
5. **Honest states** — banner "Usage tracking began \<tracking_since\>; earlier sessions are
   untracked." when the window predates the first event; unpriced models render cost as "—" with
   a tooltip naming the model; empty window shows an explicit empty state, never zeros dressed
   as data.

## 7. Error handling

- Capture hooks: try/except around everything; failures log a warning and drop the event
  (fail-open — usage logging must never break a chat response or an insight generation).
- Recorder: bounded queue; on overflow or DB failure, drop + warn.
- API: malformed/absent rows aggregate to zeros; unknown models → `unpriced_models`, not
  exceptions.
- Frontend: standard TanStack Query error state matching ActivityTab.

## 8. Testing

**Cheapest-disproof-first (implementation step 1, before any UI work):** a one-shot script run
on the droplet venv making one tiny `get_chat_llm(...)` call and one tiny `dspy.LM(...)` call,
asserting both hooks produce a `llm_usage_events` row with real token counts (~$0.001). This
validates the single load-bearing assumption (both hooks capture usage in the real environment)
before the rest is built. If it fails, redesign the capture layer before proceeding.

Backend (targeted runs locally; CI is the arbiter):

- `llm_pricing`: known models, unknown → None, prefix matching.
- Recorder: enqueue/flush/overflow/fail-open (mock Supabase).
- Attribution contextvar: set/unset propagation, anonymous-user → NULL.
- Callback shape tests: extract usage from langchain `on_llm_end` payloads (Anthropic + OpenAI
  shapes) and litellm success-callback kwargs.
- Endpoint: aggregation, day-window filter, unpriced handling (mock Supabase) — placed under the
  existing `tests/unit/` tree (already in the `backend-tests.yml` allowlist; no new top-level
  test dir).

Frontend: ObservabilityTab renders stat cards/tables from fixture payload, row expansion,
empty state, unpriced "—" display, window dropdown.

**Post-deploy verification (faithful):** send one chat message in prod, confirm a row lands in
`llm_usage_events` attributed to the sending user, and the tab renders it; confirm a scheduled
insight run produces platform-level (NULL-user) rows.

## 9. Out of scope

- Budgets/alerts/quotas (that is `agent_cost_tracker`'s eventual axis).
- Backfilling token counts for historical messages (impossible — data was never captured).
- Per-message cost drill-down UI (message-level columns are stamped, but the tab stops at
  session granularity).
- Prometheus/OTel export.
- Embedding-model usage (no embedding traffic runs through the two chokepoints today; can be
  added later as another enqueue site).
