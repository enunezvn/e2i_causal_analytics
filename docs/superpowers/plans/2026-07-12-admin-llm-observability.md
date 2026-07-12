# Admin Observability Tab (LLM usage / tokens / cost) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Observability tab to `/admin` showing LLM model, tokens, and $ cost per user and per session (chat), plus aggregate platform (non-chat) LLM spend — backed by a new capture layer, since NO token data exists today.

**Architecture:** Append-only `llm_usage_events` table (migration 104). Two fail-open capture hooks cover every LLM call site: a LangChain callback attached at model construction inside `src/utils/llm_factory.py`, and one global litellm `CustomLogger` registered at FastAPI startup (covers all `dspy.LM` traffic). A contextvar (`src/utils/llm_attribution.py`) set by the CopilotKit chat entrypoint attributes chat calls to user/session; everything else lands as platform-level NULL-user rows. Cost is computed at READ time from tokens × a static pricing table. One admin endpoint returns the whole tab's payload; the React tab mirrors `ActivityTab` conventions.

**Tech Stack:** Python 3.12, FastAPI, Supabase (postgres), langchain-anthropic 1.3.1 / langchain-openai 1.1.14, litellm 1.84.0 (via dspy 3.1.0), React 18 + TypeScript + TanStack Query + recharts, vitest.

**Spec:** `docs/superpowers/specs/2026-07-12-admin-observability-tab-design.md` (approved 2026-07-12).

---

## Conventions & environment facts (read first)

- **PROD == DEV == this host** (droplet). Live Supabase: `docker exec supabase-db psql -U postgres -d postgres`. Backend venv: `.venv/` at repo root — run python/pytest as `.venv/bin/python` / `.venv/bin/pytest` from the repo root.
- **Do NOT run whole-tree mypy or whole-tree pytest on this box** — CI is the arbiter. Scope checks to changed files only.
- **Lint gate is BOTH** `ruff check` AND `ruff format --check` (ruff pinned 0.14.10: `.venv/bin/ruff`).
- **Frontend typecheck:** `npx tsc -p tsconfig.app.json --noEmit` from `frontend/` (bare `npx tsc --noEmit` is a FALSE GREEN). No prettier gate — do not `--write` unrelated files.
- **Migrations:** no `BEGIN`/`COMMIT` in the file (the runner wraps files). Dry-run first with an explicit BEGIN/ROLLBACK wrapper (Task 1).
- **CI test allowlist:** `backend-tests.yml` runs an explicit list of `tests/unit/` subdirectories. All new backend tests in this plan go in ALREADY-ALLOWLISTED dirs (`tests/unit/test_services/`, `tests/unit/test_utils/`, `tests/unit/test_api/`) — do NOT create a new top-level test dir.
- **Git:** branch `feat/admin-llm-observability` already exists (spec committed). Before any push: `git config --global http.https://github.com.proxy ""`. Never squash-merge.
- **Commit trailer:** end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File map

| File | Action | Responsibility |
|---|---|---|
| `database/migrations/104_llm_usage_events.sql` | Create | usage-events table + RLS |
| `src/services/llm_pricing.py` | Create | read-time $ pricing (pure) |
| `src/utils/llm_attribution.py` | Create | per-run attribution contextvar + token accumulator |
| `src/services/llm_usage_recorder.py` | Create | bounded queue + background flush to DB (fail-open) |
| `src/utils/llm_usage_callback.py` | Create | LangChain `on_llm_end` capture |
| `src/utils/llm_factory.py` | Modify | attach callback at construction; `stream_usage=True` for OpenAI |
| `src/utils/litellm_usage_logger.py` | Create | global litellm CustomLogger (dspy coverage) |
| `src/api/main.py` | Modify | register litellm logger in lifespan |
| `scripts/verify_llm_usage_capture.py` | Create | faithful capture-proof script (GATE) |
| `src/api/routes/copilotkit.py` | Modify | set chat attribution per run; stamp tokens/model on persist |
| `src/api/routes/chatbot_graph.py` | Modify | replace env-var `model_used` fabrication with honest drain |
| `src/insights/common.py` | Modify | tag page-insight LLM calls `surface='insights'` |
| `src/services/llm_observability_service.py` | Create | aggregation for the endpoint |
| `src/api/routes/admin.py` | Modify | `GET /admin/observability/llm-usage` |
| `frontend/src/types/admin.ts` | Modify | response types |
| `frontend/src/api/admin.ts` | Modify | `getLlmUsage()` |
| `frontend/src/lib/query-client.ts` | Modify | `queryKeys.admin.llmUsage` |
| `frontend/src/hooks/api/use-admin.ts` | Modify | `useLlmUsage()` |
| `frontend/src/components/admin/ObservabilityTab.tsx` (+`.test.tsx`) | Create | the tab |
| `frontend/src/components/admin/index.ts` | Modify | export |
| `frontend/src/pages/Admin.tsx` | Modify | third tab |
| Tests | Create | `tests/unit/test_services/test_llm_pricing.py`, `tests/unit/test_utils/test_llm_attribution.py`, `tests/unit/test_services/test_llm_usage_recorder.py`, `tests/unit/test_utils/test_llm_usage_capture.py`, `tests/unit/test_services/test_llm_observability_service.py`, `tests/unit/test_api/test_routes/test_admin_llm_usage.py`, `tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py` |

---

### Task 1: Migration 104 — `llm_usage_events`

**Files:**
- Create: `database/migrations/104_llm_usage_events.sql`

- [ ] **Step 1: Write the migration**

```sql
-- Migration 104: llm_usage_events — per-call LLM usage capture (admin
-- observability, spec 2026-07-12).
--
-- One row per completed LLM call, written fail-open by the backend capture
-- hooks (llm_factory LangChain callback + global litellm logger). user_id /
-- session_id are NULL for platform-level (non-chat) calls — attribution is
-- honest-only, never guessed. No cost column: cost is computed at read time
-- from tokens x the pricing table (src/services/llm_pricing.py) so pricing
-- corrections apply retroactively.
--
-- Grants: mirror 101 — RLS on, admins read, service_role (recorder) bypasses.
-- NOTE: no BEGIN/COMMIT here — the migration runner wraps files itself.

CREATE TABLE IF NOT EXISTS llm_usage_events (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    surface       TEXT NOT NULL DEFAULT 'other',
    component     TEXT,
    user_id       UUID,
    session_id    VARCHAR,
    request_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_user    ON llm_usage_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_session ON llm_usage_events (session_id);

ALTER TABLE llm_usage_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS llm_usage_admin_read ON llm_usage_events;
CREATE POLICY llm_usage_admin_read ON llm_usage_events
    FOR SELECT TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM chatbot_user_profiles p
            WHERE p.id = auth.uid() AND p.role = 'admin'
        )
    );
-- service_role bypasses RLS; the recorder writes with the service-role client.
```

- [ ] **Step 2: Dry-run against the live DB**

Run:
```bash
(echo "BEGIN;"; cat database/migrations/104_llm_usage_events.sql; echo "ROLLBACK;") \
  | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1
```
Expected: `CREATE TABLE`, `CREATE INDEX` ×3, `ALTER TABLE`, `CREATE POLICY`, ends `ROLLBACK` — no errors.

- [ ] **Step 3: Apply for real**

Run:
```bash
cat database/migrations/104_llm_usage_events.sql \
  | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1
docker exec supabase-db psql -U postgres -d postgres -c "\d llm_usage_events" | head -20
```
Expected: table described with all 11 columns.

- [ ] **Step 4: Commit**

```bash
git add database/migrations/104_llm_usage_events.sql
git commit -m "feat(observability): migration 104 llm_usage_events table

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Pricing module (read-time cost)

**Files:**
- Create: `src/services/llm_pricing.py`
- Test: `tests/unit/test_services/test_llm_pricing.py`

- [ ] **Step 1: Write the failing tests**

```python
"""llm_pricing: read-time cost from tokens x static rates; unknown models
return None (surfaced as 'unpriced'), NEVER a default cost."""

from src.services.llm_pricing import (
    PRICING_VERSION,
    cost_usd,
    normalize_model,
    resolve_pricing_key,
)


def test_known_model_cost_per_million():
    # claude-sonnet-4-6: $3/1M in, $15/1M out
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_small_call_cost():
    # 1000 in + 500 out on gpt-4o-mini: 1000*0.15/1M + 500*0.60/1M
    assert cost_usd("gpt-4o-mini", 1000, 500) == (1000 * 0.15 + 500 * 0.60) / 1_000_000


def test_dated_variant_resolves_to_base_key():
    assert resolve_pricing_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"


def test_provider_prefixes_stripped():
    assert normalize_model("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert normalize_model("openai/gpt-4o") == "gpt-4o"
    # copilotkit metadata format uses a colon
    assert normalize_model("anthropic:claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_gpt4o_mini_does_not_match_gpt4o():
    assert resolve_pricing_key("gpt-4o-mini") == "gpt-4o-mini"
    assert resolve_pricing_key("gpt-4o") == "gpt-4o"


def test_unknown_model_returns_none_not_zero():
    assert cost_usd("mistral-large", 1000, 1000) is None
    assert resolve_pricing_key("mistral-large") is None


def test_zero_tokens_known_model_is_zero_cost():
    assert cost_usd("gpt-4o", 0, 0) == 0.0


def test_pricing_version_is_a_date_string():
    assert len(PRICING_VERSION) == 10  # YYYY-MM-DD
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.llm_pricing'`

- [ ] **Step 3: Implement `src/services/llm_pricing.py`**

```python
"""Read-time LLM pricing (admin observability, spec 2026-07-12).

No cost is ever baked into llm_usage_events rows — consumers call cost_usd()
at read time so pricing corrections apply retroactively. Unknown models
return None (rendered as 'unpriced'), never a silent default.

Rate-table pattern borrowed from src/mlops/agent_cost_tracker.MODEL_PRICING
(which stays untouched — it is an unwired per-agent budget scaffold on a
different axis).
"""

from typing import Optional

# Bump whenever rates change; surfaced in the API payload for provenance.
PRICING_VERSION = "2026-07-12"

# USD per 1M tokens: normalized model key -> (input_rate, output_rate).
# Keys are prefixes: resolve_pricing_key strips provider prefixes and picks
# the LONGEST matching key, so dated variants (claude-haiku-4-5-20251001)
# resolve and gpt-4o-mini never falls into gpt-4o.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-5": (5.00, 25.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

_KEYS_LONGEST_FIRST = sorted(MODEL_PRICING, key=len, reverse=True)

_PROVIDER_PREFIXES = ("anthropic", "openai", "azure")


def normalize_model(model: str) -> str:
    """Strip provider prefixes: 'anthropic/x', 'openai:y' -> bare model id."""
    m = (model or "").strip().lower()
    for sep in ("/", ":"):
        if sep in m:
            head, tail = m.split(sep, 1)
            if head in _PROVIDER_PREFIXES:
                m = tail
    return m


def resolve_pricing_key(model: str) -> Optional[str]:
    m = normalize_model(model)
    for key in _KEYS_LONGEST_FIRST:
        if m == key or m.startswith(key + "-"):
            return key
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD cost, or None for unknown models — callers surface those as
    'unpriced' rather than costing them at a fabricated rate."""
    key = resolve_pricing_key(model)
    if key is None:
        return None
    in_rate, out_rate = MODEL_PRICING[key]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_pricing.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/llm_pricing.py tests/unit/test_services/test_llm_pricing.py
git commit -m "feat(observability): read-time LLM pricing table

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Attribution contextvar + run accumulator

**Files:**
- Create: `src/utils/llm_attribution.py`
- Test: `tests/unit/test_utils/test_llm_attribution.py`

- [ ] **Step 1: Write the failing tests**

```python
"""llm_attribution: per-run contextvar both capture hooks read, plus the
drain-on-persist token accumulator (drain = read-and-reset so sums across a
session's assistant rows never double-count)."""

from src.utils.llm_attribution import (
    ANONYMOUS_USER_ID,
    clear_attribution,
    drain_run_usage,
    get_attribution,
    record_usage,
    set_chat_attribution,
    set_platform_attribution,
    user_id_from_session,
)

USER = "11111111-1111-1111-1111-111111111111"


def setup_function(_fn):
    clear_attribution()


def test_user_id_from_session_shapes():
    assert user_id_from_session(f"{USER}~abc-123") == USER
    assert user_id_from_session(f"{ANONYMOUS_USER_ID}~abc") is None  # honest NULL
    assert user_id_from_session("not-a-uuid~abc") is None
    assert user_id_from_session("no-tilde-here") is None
    assert user_id_from_session(None) is None


def test_chat_attribution_set_and_get():
    set_chat_attribution(f"{USER}~s1", request_id="run-9")
    attr = get_attribution()
    assert attr is not None
    assert attr.user_id == USER
    assert attr.session_id == f"{USER}~s1"
    assert attr.surface == "chat"
    assert attr.request_id == "run-9"


def test_platform_attribution():
    set_platform_attribution("insights", component="ExecutiveBrief")
    attr = get_attribution()
    assert attr.user_id is None
    assert attr.surface == "insights"
    assert attr.component == "ExecutiveBrief"


def test_record_usage_noop_without_attribution():
    record_usage("gpt-4o", 10, 5)  # must not raise
    assert drain_run_usage() is None


def test_record_and_drain_resets():
    set_chat_attribution(f"{USER}~s1")
    record_usage("gpt-4o", 10, 5)
    record_usage("claude-sonnet-4-6", 100, 50)
    drained = drain_run_usage()
    assert drained.input_tokens == 110
    assert drained.output_tokens == 55
    assert drained.last_model == "claude-sonnet-4-6"
    # drained: second read is empty — no double-counting across persists
    assert drain_run_usage() is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_attribution.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/utils/llm_attribution.py`**

```python
"""Per-run LLM attribution contextvar (admin observability, spec 2026-07-12).

Chat entrypoints call set_chat_attribution() at run start; both capture hooks
(the llm_factory LangChain callback and the global litellm logger) read it to
attribute usage rows to a user/session. Unset => honest platform-level rows
(NULL user/session, surface fallback 'other'), never a guessed attribution.

Also carries the per-run token accumulator that message persistence drains
into chatbot_messages.tokens_used / model_used. Drain = read-and-reset, so
each assistant row carries tokens accrued since the previous drained row and
sums across a session never double-count.
"""

import contextvars
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional

ANONYMOUS_USER_ID = "00000000-0000-0000-0000-000000000000"


@dataclass
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    last_model: Optional[str] = None


@dataclass
class LLMAttribution:
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    surface: str = "other"
    component: Optional[str] = None
    request_id: Optional[str] = None
    usage: RunUsage = field(default_factory=RunUsage)


_attribution: contextvars.ContextVar[Optional[LLMAttribution]] = contextvars.ContextVar(
    "llm_attribution", default=None
)


def user_id_from_session(session_id: Optional[str]) -> Optional[str]:
    """'<user_id>~<uuid>' -> user_id. Anonymous, malformed, or non-UUID
    prefixes -> None (honest NULL, never fabricated)."""
    if not session_id or "~" not in session_id:
        return None
    prefix = session_id.split("~", 1)[0]
    try:
        _uuid.UUID(prefix)
    except ValueError:
        return None
    return None if prefix == ANONYMOUS_USER_ID else prefix


def set_chat_attribution(session_id: str, request_id: Optional[str] = None) -> LLMAttribution:
    attr = LLMAttribution(
        user_id=user_id_from_session(session_id),
        session_id=session_id,
        surface="chat",
        request_id=request_id,
    )
    _attribution.set(attr)
    return attr


def set_platform_attribution(surface: str, component: Optional[str] = None) -> LLMAttribution:
    attr = LLMAttribution(surface=surface, component=component)
    _attribution.set(attr)
    return attr


def get_attribution() -> Optional[LLMAttribution]:
    return _attribution.get()


def clear_attribution() -> None:
    _attribution.set(None)


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Accumulate into the current run; no-op when no attribution is set."""
    attr = _attribution.get()
    if attr is None:
        return
    attr.usage.input_tokens += input_tokens
    attr.usage.output_tokens += output_tokens
    attr.usage.last_model = model


def drain_run_usage() -> Optional[RunUsage]:
    """Return-and-reset the run accumulator; None when nothing was recorded."""
    attr = _attribution.get()
    if attr is None or (attr.usage.input_tokens == 0 and attr.usage.output_tokens == 0):
        return None
    drained = attr.usage
    attr.usage = RunUsage()
    return drained
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_attribution.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/llm_attribution.py tests/unit/test_utils/test_llm_attribution.py
git commit -m "feat(observability): llm attribution contextvar + run accumulator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Usage recorder (bounded queue, fail-open flush)

**Files:**
- Create: `src/services/llm_usage_recorder.py`
- Test: `tests/unit/test_services/test_llm_usage_recorder.py`

- [ ] **Step 1: Write the failing tests**

```python
"""llm_usage_recorder: never blocks, never raises; bounded queue drops on
overflow; batch insert failures log-and-drop (fail-open)."""

import queue
from types import SimpleNamespace

import src.services.llm_usage_recorder as recorder
from src.services.llm_usage_recorder import LLMUsageEvent


def _event(**over):
    base = dict(provider="openai", model="gpt-4o", input_tokens=10, output_tokens=5)
    base.update(over)
    return LLMUsageEvent(**base)


def test_to_row_shape():
    row = _event(surface="chat", user_id="u1", session_id="u1~s", request_id="r1").to_row()
    assert row == {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 10,
        "output_tokens": 5,
        "surface": "chat",
        "component": None,
        "user_id": "u1",
        "session_id": "u1~s",
        "request_id": "r1",
    }


def test_enqueue_drops_when_full(monkeypatch):
    monkeypatch.setattr(recorder, "_ensure_flusher", lambda: None)
    monkeypatch.setattr(recorder, "_queue", queue.Queue(maxsize=2))
    assert recorder.enqueue(_event()) is True
    assert recorder.enqueue(_event()) is True
    assert recorder.enqueue(_event()) is False  # dropped, no exception


def test_insert_batch_success():
    inserted = []

    class _Client:
        def table(self, name):
            assert name == "llm_usage_events"
            return SimpleNamespace(
                insert=lambda rows: SimpleNamespace(
                    execute=lambda: inserted.append(rows) or SimpleNamespace(data=rows)
                )
            )

    assert recorder._insert_batch([_event(), _event()], _Client()) is True
    assert len(inserted[0]) == 2


def test_insert_batch_failure_is_swallowed():
    class _Boom:
        def table(self, name):
            raise RuntimeError("db down")

    assert recorder._insert_batch([_event()], _Boom()) is False  # no raise


def test_drain_batch_respects_max(monkeypatch):
    q = queue.Queue()
    for _ in range(60):
        q.put(_event())
    monkeypatch.setattr(recorder, "_queue", q)
    batch = recorder._drain_batch()
    assert len(batch) == recorder._BATCH_MAX
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_usage_recorder.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/services/llm_usage_recorder.py`**

```python
"""Fail-open, non-blocking writer for llm_usage_events (spec 2026-07-12).

Capture hooks call enqueue() from the request path; a lazily-started daemon
thread batches rows into Supabase. Failure policy is drop-and-warn at every
stage: a DB outage or full queue loses usage telemetry but can never break an
LLM call or grow memory unboundedly. In-flight events may be lost on process
shutdown — accepted (telemetry, not billing records).
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAX = 1000
_BATCH_MAX = 50
_POLL_SECONDS = 2.0


@dataclass
class LLMUsageEvent:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    surface: str = "other"
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "surface": self.surface,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
        }


_queue: "queue.Queue[LLMUsageEvent]" = queue.Queue(maxsize=_QUEUE_MAX)
_flusher_started = False
_flusher_lock = threading.Lock()


def enqueue(event: LLMUsageEvent) -> bool:
    """Never blocks, never raises. False = queue full, event dropped."""
    try:
        _queue.put_nowait(event)
    except queue.Full:
        logger.warning("llm_usage_recorder: queue full, dropping usage event")
        return False
    _ensure_flusher()
    return True


def _ensure_flusher() -> None:
    global _flusher_started
    if _flusher_started:
        return
    with _flusher_lock:
        if _flusher_started:
            return
        thread = threading.Thread(target=_flush_loop, name="llm-usage-flusher", daemon=True)
        thread.start()
        _flusher_started = True


def _drain_batch() -> List[LLMUsageEvent]:
    events: List[LLMUsageEvent] = []
    try:
        events.append(_queue.get(timeout=_POLL_SECONDS))
    except queue.Empty:
        return events
    while len(events) < _BATCH_MAX:
        try:
            events.append(_queue.get_nowait())
        except queue.Empty:
            break
    return events


def _insert_batch(events: List[LLMUsageEvent], client: Any) -> bool:
    """Separated from the loop so unit tests exercise it with a fake client."""
    if not events:
        return True
    try:
        client.table("llm_usage_events").insert([e.to_row() for e in events]).execute()
        return True
    except Exception as e:
        logger.warning(
            "llm_usage_recorder: batch insert failed, dropping %d event(s): %s",
            len(events),
            e,
        )
        return False


def _flush_loop() -> None:
    from src.api.dependencies.supabase_client import get_supabase

    while True:
        events = _drain_batch()
        if not events:
            continue
        try:
            client = get_supabase()
        except Exception as e:
            logger.warning("llm_usage_recorder: no client, dropping %d: %s", len(events), e)
            continue
        if client is None:
            logger.warning("llm_usage_recorder: Supabase unavailable, dropping %d", len(events))
            continue
        _insert_batch(events, client)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_usage_recorder.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/llm_usage_recorder.py tests/unit/test_services/test_llm_usage_recorder.py
git commit -m "feat(observability): fail-open llm usage recorder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: LangChain capture — callback + llm_factory wiring

**Files:**
- Create: `src/utils/llm_usage_callback.py`
- Modify: `src/utils/llm_factory.py` (`_create_anthropic_llm` ~line 137, `_create_openai_llm` ~line 167)
- Test: `tests/unit/test_utils/test_llm_usage_capture.py` (LangChain half; Task 6 adds the litellm half to the same file)

- [ ] **Step 1: Write the failing tests**

```python
"""Capture hooks: extract usage from LangChain LLMResult shapes and litellm
success payloads; enqueue with current attribution; zero-usage => no event."""

from types import SimpleNamespace

import src.utils.llm_usage_callback as cb_mod
from src.utils.llm_attribution import clear_attribution, drain_run_usage, set_chat_attribution
from src.utils.llm_usage_callback import UsageRecorderCallback, _extract_usage

USER = "11111111-1111-1111-1111-111111111111"


def setup_function(_fn):
    clear_attribution()


def _anthropic_stream_result():
    # langchain-anthropic 1.3.x aggregated stream: usage on message.usage_metadata
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 11, "output_tokens": 7},
        response_metadata={"model_name": "claude-sonnet-4-6"},
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None)


def _openai_llm_output_result():
    # langchain-openai fallback shape: usage in llm_output.token_usage
    return SimpleNamespace(
        generations=[[SimpleNamespace(message=None)]],
        llm_output={
            "token_usage": {"prompt_tokens": 9, "completion_tokens": 3},
            "model_name": "gpt-4o",
        },
    )


def test_extract_usage_message_metadata():
    model, i, o = _extract_usage(_anthropic_stream_result(), "claude-sonnet-4-6")
    assert (model, i, o) == ("claude-sonnet-4-6", 11, 7)


def test_extract_usage_llm_output_fallback():
    model, i, o = _extract_usage(_openai_llm_output_result(), "gpt-4o")
    assert (model, i, o) == ("gpt-4o", 9, 3)


def test_extract_usage_empty_result_is_zero():
    empty = SimpleNamespace(generations=[], llm_output=None)
    model, i, o = _extract_usage(empty, "gpt-4o")
    assert (i, o) == (0, 0)
    assert model == "gpt-4o"


def test_callback_enqueues_with_attribution(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    set_chat_attribution(f"{USER}~s1", request_id="run-1")

    cb = UsageRecorderCallback(provider="anthropic", default_model="claude-sonnet-4-6")
    cb.on_llm_end(_anthropic_stream_result())

    assert len(events) == 1
    ev = events[0]
    assert ev.user_id == USER
    assert ev.session_id == f"{USER}~s1"
    assert ev.surface == "chat"
    assert ev.request_id == "run-1"
    assert (ev.input_tokens, ev.output_tokens) == (11, 7)
    # accumulator updated for persist-time stamping
    drained = drain_run_usage()
    assert drained.input_tokens == 11 and drained.last_model == "claude-sonnet-4-6"


def test_callback_without_attribution_is_platform_row(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(_openai_llm_output_result())
    assert events[0].user_id is None
    assert events[0].session_id is None
    assert events[0].surface == "other"


def test_callback_zero_usage_no_event(monkeypatch):
    events = []
    monkeypatch.setattr(cb_mod, "enqueue", lambda e: events.append(e) or True)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(SimpleNamespace(generations=[], llm_output=None))
    assert events == []  # never fabricate


def test_callback_never_raises(monkeypatch):
    def _boom(_e):
        raise RuntimeError("recorder exploded")

    monkeypatch.setattr(cb_mod, "enqueue", _boom)
    cb = UsageRecorderCallback(provider="openai", default_model="gpt-4o")
    cb.on_llm_end(_openai_llm_output_result())  # must not raise


def test_factory_attaches_callback_and_stream_usage(monkeypatch):
    """llm_factory must construct models with the capture callback (and
    stream_usage=True for OpenAI) — the whole point of the chokepoint."""
    import src.utils.llm_factory as factory

    captured = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    factory._create_openai_llm("gpt-4o", 100, 0.3, None)
    assert captured["stream_usage"] is True
    assert any(isinstance(c, UsageRecorderCallback) for c in captured["callbacks"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_usage_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.llm_usage_callback'`

- [ ] **Step 3: Implement `src/utils/llm_usage_callback.py`**

```python
"""LangChain usage-capture callback (admin observability, spec 2026-07-12).

Attached at model construction in src/utils/llm_factory.py, so every factory
consumer is covered with zero call-site edits; construction-time callbacks
fire on .invoke() AND .astream() (verified langchain-anthropic 1.3.1 /
langchain-openai 1.1.14). Fail-open: capture must never break an LLM call.
"""

import logging
from typing import Any, Tuple

from langchain_core.callbacks import BaseCallbackHandler

from src.services.llm_usage_recorder import LLMUsageEvent, enqueue
from src.utils.llm_attribution import get_attribution, record_usage

logger = logging.getLogger(__name__)


def _extract_usage(response: Any, default_model: str) -> Tuple[str, int, int]:
    """(model, input_tokens, output_tokens) across the LLMResult shapes both
    providers emit for invoke and aggregated streams. Zeros when the provider
    reported no usage — the caller then records nothing (never fabricates)."""
    model = default_model
    input_tokens = 0
    output_tokens = 0

    message = None
    try:
        message = getattr(response.generations[0][0], "message", None)
    except (IndexError, AttributeError, TypeError):
        pass

    if message is not None:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model") or model

    if input_tokens == 0 and output_tokens == 0:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        model = llm_output.get("model_name") or llm_output.get("model") or model

    return model, input_tokens, output_tokens


class UsageRecorderCallback(BaseCallbackHandler):
    """One instance per constructed model (carries provider + requested model
    as fallbacks when the response omits them)."""

    def __init__(self, provider: str, default_model: str) -> None:
        self._provider = provider
        self._default_model = default_model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            model, input_tokens, output_tokens = _extract_usage(response, self._default_model)
            if input_tokens == 0 and output_tokens == 0:
                return
            record_usage(model, input_tokens, output_tokens)
            attr = get_attribution()
            enqueue(
                LLMUsageEvent(
                    provider=self._provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    surface=attr.surface if attr else "other",
                    component=attr.component if attr else None,
                    user_id=attr.user_id if attr else None,
                    session_id=attr.session_id if attr else None,
                    request_id=attr.request_id if attr else None,
                )
            )
        except Exception as e:  # fail-open by contract
            logger.warning("UsageRecorderCallback failed (non-blocking): %s", e)
```

- [ ] **Step 4: Wire into `src/utils/llm_factory.py`**

Add the import after `logger = logging.getLogger(__name__)` (line 43):

```python
from src.utils.llm_usage_callback import UsageRecorderCallback
```

In `_create_anthropic_llm`, change the kwargs dict (lines 137-141) to:

```python
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Usage capture (spec 2026-07-12): construction-time callbacks fire on
        # invoke AND astream, covering every factory consumer.
        "callbacks": [UsageRecorderCallback(provider="anthropic", default_model=model)],
    }
```

In `_create_openai_llm`, change the kwargs dict (lines 167-171) to:

```python
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # stream_usage: OpenAI omits usage on streamed responses unless asked
        # (Anthropic streams usage by default).
        "stream_usage": True,
        "callbacks": [UsageRecorderCallback(provider="openai", default_model=model)],
    }
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_usage_capture.py -v`
Expected: 8 PASS

- [ ] **Step 6: Commit**

```bash
git add src/utils/llm_usage_callback.py src/utils/llm_factory.py tests/unit/test_utils/test_llm_usage_capture.py
git commit -m "feat(observability): langchain usage capture via llm_factory callback

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: litellm capture (all dspy traffic) + startup registration

**Files:**
- Create: `src/utils/litellm_usage_logger.py`
- Modify: `src/api/main.py` (lifespan, insert after the Supabase-init block ending line 295)
- Test: append to `tests/unit/test_utils/test_llm_usage_capture.py`

- [ ] **Step 1: Append failing tests to `tests/unit/test_utils/test_llm_usage_capture.py`**

```python
# ---------------------------------------------------------------- litellm ---

from src.utils.litellm_usage_logger import record_litellm_success


def _litellm_response(model="gpt-4o", prompt=9, completion=4):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def test_litellm_success_enqueues(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o"}, _litellm_response())
    assert len(events) == 1
    assert events[0].provider == "openai"
    assert (events[0].input_tokens, events[0].output_tokens) == (9, 4)


def test_litellm_anthropic_provider_detection(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success(
        {"model": "anthropic/claude-sonnet-4-6"},
        _litellm_response(model="claude-sonnet-4-6"),
    )
    assert events[0].provider == "anthropic"


def test_litellm_cache_hit_skipped(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o", "cache_hit": True}, _litellm_response())
    assert events == []  # cached replay spent no tokens


def test_litellm_zero_usage_skipped(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.services.llm_usage_recorder.enqueue", lambda e: events.append(e) or True
    )
    record_litellm_success({"model": "gpt-4o"}, SimpleNamespace(model="gpt-4o", usage=None))
    assert events == []


def test_litellm_never_raises(monkeypatch):
    def _boom(_e):
        raise RuntimeError("recorder exploded")

    monkeypatch.setattr("src.services.llm_usage_recorder.enqueue", _boom)
    record_litellm_success({"model": "gpt-4o"}, _litellm_response())  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_usage_capture.py -v`
Expected: new tests FAIL — `ModuleNotFoundError: No module named 'src.utils.litellm_usage_logger'`

- [ ] **Step 3: Implement `src/utils/litellm_usage_logger.py`**

```python
"""Global litellm usage logger (admin observability, spec 2026-07-12).

DSPy rides litellm, so ONE logger registered at API startup covers every
dspy.LM call site (dspy_lm.py, chatbot_dspy.py, cognitive_rag_dspy.py,
causal_rag.py, causal_role_classifier_loader.py) regardless of where the LM
was instantiated. litellm is imported lazily inside register so this module
stays cheap to import. Fail-open everywhere.
"""

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_registered = False


def _model_and_provider(kwargs: dict, response_obj: Any) -> Tuple[str, str]:
    model = getattr(response_obj, "model", None) or kwargs.get("model") or "unknown"
    lowered = f"{kwargs.get('custom_llm_provider') or ''} {model}".lower()
    provider = "anthropic" if ("anthropic" in lowered or "claude" in lowered) else "openai"
    return str(model), provider


def _usage_tokens(response_obj: Any) -> Tuple[int, int]:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return 0, 0

    def _get(name: str) -> int:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return int(value or 0)

    return _get("prompt_tokens"), _get("completion_tokens")


def record_litellm_success(kwargs: dict, response_obj: Any) -> None:
    """Shared body for the sync and async success hooks. Never raises."""
    try:
        # Late imports keep module import free of recorder/attribution cost
        # and let tests monkeypatch the source modules.
        from src.services.llm_usage_recorder import LLMUsageEvent, enqueue
        from src.utils.llm_attribution import get_attribution, record_usage

        if kwargs.get("cache_hit"):
            return  # cached replay: no tokens were spent
        input_tokens, output_tokens = _usage_tokens(response_obj)
        if input_tokens == 0 and output_tokens == 0:
            return
        model, provider = _model_and_provider(kwargs, response_obj)
        record_usage(model, input_tokens, output_tokens)
        attr = get_attribution()
        enqueue(
            LLMUsageEvent(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                surface=attr.surface if attr else "other",
                component=attr.component if attr else None,
                user_id=attr.user_id if attr else None,
                session_id=attr.session_id if attr else None,
                request_id=attr.request_id if attr else None,
            )
        )
    except Exception as e:  # fail-open by contract
        logger.warning("litellm usage logging failed (non-blocking): %s", e)


def register_litellm_usage_logger() -> bool:
    """Idempotent. False when litellm is unavailable (capture disabled)."""
    global _registered
    if _registered:
        return True
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError:
        logger.warning("litellm not installed; dspy usage capture disabled")
        return False

    class _UsageLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_litellm_success(kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_litellm_success(kwargs, response_obj)

    litellm.callbacks.append(_UsageLogger())
    _registered = True
    logger.info("litellm usage logger registered")
    return True
```

- [ ] **Step 4: Register in `src/api/main.py` lifespan**

Insert directly after the Supabase-init `except` block (after line 295, before the MLflow block):

```python
    # LLM usage capture (admin observability, spec 2026-07-12): one global
    # litellm logger covers all dspy.LM traffic; LangChain traffic is captured
    # per-instance inside llm_factory. Fail-open — never blocks startup.
    try:
        from src.utils.litellm_usage_logger import register_litellm_usage_logger

        register_litellm_usage_logger()
    except Exception as e:  # noqa: BLE001 - never block startup on this
        logger.warning(f"litellm usage logger registration failed (non-critical): {e}")
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_utils/test_llm_usage_capture.py -v`
Expected: 13 PASS

- [ ] **Step 6: Commit**

```bash
git add src/utils/litellm_usage_logger.py src/api/main.py tests/unit/test_utils/test_llm_usage_capture.py
git commit -m "feat(observability): global litellm usage logger for dspy traffic

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: 🚧 GATE — faithful capture verification (cheapest-disproof-first)

**The load-bearing assumption of this whole feature is that both hooks capture real token counts in the real environment.** Prove it with ~$0.001 of API spend BEFORE building attribution wiring, the endpoint, or any UI. If this fails, STOP and redesign the capture layer.

**Files:**
- Create: `scripts/verify_llm_usage_capture.py`

- [ ] **Step 1: Write the script**

```python
"""One-shot faithful verification of both LLM usage capture hooks (spec
2026-07-12). Run ON THE DROPLET from the repo root with the prod .env:

    PYTHONPATH=. .venv/bin/python scripts/verify_llm_usage_capture.py

Makes one tiny LangChain call via llm_factory (STREAMED — the copilotkit
path) and one tiny dspy/litellm call (~$0.001 total), then asserts both
landed in llm_usage_events with nonzero tokens. Exits non-zero on failure.
"""

import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.api.dependencies.supabase_client import get_supabase  # noqa: E402
from src.utils.litellm_usage_logger import register_litellm_usage_logger  # noqa: E402


async def _stream_langchain_call() -> str:
    from src.utils.llm_factory import get_chat_llm

    llm = get_chat_llm(model_tier="fast", max_tokens=16)
    chunks = []
    async for chunk in llm.astream("Reply with the single word OK."):
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            chunks.append(content)
    return "".join(chunks)


def main() -> int:
    client = get_supabase()
    if client is None:
        print("FAIL: no Supabase client (check .env)")
        return 1
    before = client.table("llm_usage_events").select("id", count="exact").execute().count or 0

    print("langchain (streamed):", asyncio.run(_stream_langchain_call()))

    register_litellm_usage_logger()
    import dspy

    from src.optimization.dspy_lm import get_default_dspy_model

    lm = dspy.LM(get_default_dspy_model(), max_tokens=16, cache=False)
    print("dspy:", lm("Reply with the single word OK."))

    time.sleep(8)  # background flusher polls every 2s

    after = client.table("llm_usage_events").select("id", count="exact").execute().count or 0
    new_count = after - before
    rows = (
        client.table("llm_usage_events")
        .select("provider, model, input_tokens, output_tokens, surface, user_id")
        .order("id", desc=True)
        .limit(max(new_count, 1))
        .execute()
        .data
        or []
    )
    print(f"rows before={before} after={after}")
    for row in rows:
        print(row)

    if new_count < 2:
        print(f"FAIL: expected >=2 new llm_usage_events rows, got {new_count}")
        return 1
    zero_rows = [r for r in rows if not (r["input_tokens"] or r["output_tokens"])]
    if zero_rows:
        print(f"FAIL: rows with zero tokens: {zero_rows}")
        return 1
    print("PASS: both capture hooks recorded real token usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it (faithful: droplet, prod .env, real keys)**

Run: `PYTHONPATH=. .venv/bin/python scripts/verify_llm_usage_capture.py`
Expected: two `OK`-ish responses printed, then `PASS: both capture hooks recorded real token usage`, exit 0. Rows should show `surface='other'`, `user_id=None` (attribution wiring comes in Task 8).

**If it FAILS:** stop the plan. Read the printed rows, check API logs, fix the capture layer, re-run. Do not proceed until PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_llm_usage_capture.py
git commit -m "feat(observability): faithful capture verification script (gate passed)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Chat attribution + persist-time stamping (copilotkit)

**Files:**
- Modify: `src/api/routes/copilotkit.py` (import block ~line 324; run setup ~line 751; `_persist_message_sync` ~line 1202)
- Test: `tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py`

- [ ] **Step 1: Write the failing test**

```python
"""_persist_message_sync stamps the run's captured tokens/model onto
assistant rows (drain = read-and-reset — no double-count across persists),
and leaves user rows and drained runs unstamped (honest NULL)."""

from types import SimpleNamespace

import pytest

from src.utils.llm_attribution import clear_attribution, record_usage, set_chat_attribution

USER = "11111111-1111-1111-1111-111111111111"
SESSION = f"{USER}~conv-1"


@pytest.fixture()
def captured(monkeypatch):
    rows = []

    class _Table:
        def insert(self, data):
            rows.append(data)
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"id": len(rows)}]))

    class _Client:
        def table(self, name):
            assert name == "chatbot_messages"
            return _Table()

    monkeypatch.setattr(
        "src.api.dependencies.supabase_client.get_supabase", lambda: _Client()
    )
    clear_attribution()
    return rows


def test_assistant_row_stamped_then_drained(captured):
    from src.api.routes.copilotkit import _persist_message_sync

    set_chat_attribution(SESSION, request_id="run-1")
    record_usage("claude-sonnet-4-6", 100, 50)

    _persist_message_sync(SESSION, "assistant", "answer one")
    assert captured[0]["tokens_used"] == 150
    assert captured[0]["model_used"] == "claude-sonnet-4-6"

    # accumulator drained: next assistant row must NOT repeat the tokens
    _persist_message_sync(SESSION, "assistant", "answer two")
    assert "tokens_used" not in captured[1]
    assert "model_used" not in captured[1]


def test_user_row_never_stamped(captured):
    from src.api.routes.copilotkit import _persist_message_sync

    set_chat_attribution(SESSION)
    record_usage("gpt-4o", 10, 5)
    _persist_message_sync(SESSION, "user", "question")
    assert "tokens_used" not in captured[0]
    # user persist must not consume the accumulator either
    _persist_message_sync(SESSION, "assistant", "answer")
    assert captured[1]["tokens_used"] == 15
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py -v`
Expected: FAIL — `KeyError: 'tokens_used'` / assertion errors (columns not yet written).

- [ ] **Step 3: Implement in `src/api/routes/copilotkit.py`**

(a) Extend the existing factory import (line 324) area with:

```python
from src.utils.llm_attribution import drain_run_usage, set_chat_attribution
```

(b) After `_run_id_context.set(run_id)` (line 751), add:

```python
        # Attribute this run's LLM usage to the chat user/session (admin
        # observability, spec 2026-07-12). Both capture hooks read this
        # contextvar; the user_id is derived from the session prefix and the
        # anonymous UUID maps to NULL — attribution is honest-only.
        set_chat_attribution(persistent_session_id, run_id)
```

(c) In `_persist_message_sync`, after the `message_data = {...}` dict (ends line 1208), add:

```python
        if role == "assistant":
            # Drain the run's token accumulator into the row (read-and-reset:
            # sums across a session's assistant rows never double-count).
            # None when nothing was captured — honest NULL, never fabricated.
            drained = drain_run_usage()
            if drained:
                message_data["tokens_used"] = drained.input_tokens + drained.output_tokens
                message_data["model_used"] = drained.last_model
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/copilotkit.py tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py
git commit -m "feat(observability): chat attribution + token stamping on persisted messages

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Honest stamping in chatbot_graph + insights surface tag

**Files:**
- Modify: `src/api/routes/chatbot_graph.py` (`_execute_finalize`, lines 1639-1651)
- Modify: `src/insights/common.py` (`run_signature`, ~line 29)

- [ ] **Step 1: Replace the env-var fabrication in `chatbot_graph.py`**

Current code (lines 1639-1651) stamps `model_used = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")` — an env default that has NEVER matched the actual model. Replace with the accumulator drain (same helper as copilotkit; None when nothing captured):

```python
                # Save assistant message with full context. tokens/model come
                # from the run's capture accumulator (drain = read-and-reset);
                # None when nothing was captured — honest NULL, never the
                # env-var fabrication this replaces (spec 2026-07-12).
                drained = drain_run_usage()
                await msg_repo.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=response_text,
                    agent_name=agent_name,
                    agent_tier=agent_tier,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    rag_context=rag_context,
                    rag_sources=rag_sources,
                    model_used=drained.last_model if drained else None,
                    tokens_used=(drained.input_tokens + drained.output_tokens)
                    if drained
                    else None,
                    metadata={
                        "request_id": state.get("request_id"),
                        "intent": state.get("intent"),
                        "brand_context": state.get("brand_context"),
                        "region_context": state.get("region_context"),
                    },
                )
```

Add to the import section at the top of `chatbot_graph.py` (alongside the other `from src....` imports):

```python
from src.utils.llm_attribution import drain_run_usage
```

Then delete the now-unused `model_used = os.getenv(...)` line (1640).

- [ ] **Step 2: Tag page insights with their surface in `src/insights/common.py`**

`run_signature()` is the single chokepoint every page-insight module calls. At the top of its `try:` block (right before `import dspy`, line 44), add:

```python
        # Tag this generation's litellm calls as platform-level insights usage
        # (admin observability, spec 2026-07-12): NULL user/session, but a
        # meaningful surface/component in the Platform LLM usage table.
        from src.utils.llm_attribution import set_platform_attribution

        set_platform_attribution("insights", component=signature_cls.__name__)
```

- [ ] **Step 3: Verify no fabricated fallback remains + targeted tests still pass**

Run:
```bash
grep -n "ANTHROPIC_MODEL" src/api/routes/chatbot_graph.py
.venv/bin/ruff check src/api/routes/chatbot_graph.py src/insights/common.py
.venv/bin/pytest tests/unit/test_insights/ -x -q
```
Expected: no `os.getenv("ANTHROPIC_MODEL"...)` hit in the finalize path; ruff clean; insights tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/api/routes/chatbot_graph.py src/insights/common.py
git commit -m "feat(observability): honest model/token stamping in graph finalize; tag insights surface

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Aggregation service

**Files:**
- Create: `src/services/llm_observability_service.py`
- Test: `tests/unit/test_services/test_llm_observability_service.py`

- [ ] **Step 1: Write the failing tests**

```python
"""LLMObservabilityService.llm_usage aggregation: summary, daily buckets,
per-user + per-session rollups (chat), platform grouping (NULL-user rows),
unpriced-model honesty."""

from types import SimpleNamespace

from src.services.llm_observability_service import LLMObservabilityService

U1 = "11111111-1111-1111-1111-111111111111"
U2 = "22222222-2222-2222-2222-222222222222"
S1 = f"{U1}~conv-a"
S2 = f"{U1}~conv-b"
S3 = f"{U2}~conv-c"

USERS = [
    {"id": U1, "email": "alice@x.com"},
    {"id": U2, "email": "bob@x.com"},
]


def _ev(model, i, o, user=None, session=None, surface="chat", component=None, day="2026-07-10"):
    return {
        "created_at": f"{day}T12:00:00+00:00",
        "provider": "anthropic" if "claude" in model else "openai",
        "model": model,
        "input_tokens": i,
        "output_tokens": o,
        "surface": surface if user is None else "chat",
        "component": component,
        "user_id": user,
        "session_id": session,
    }


class _Query:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, _name):
        def _chain(*_a, **_k):
            return self

        return _chain

    def execute(self):
        return SimpleNamespace(data=self._data)


class _Client:
    """Scripted per-table responses, popped in call order."""

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}

    def table(self, name):
        return _Query(self._script[name].pop(0))


def _service(events, conversations=None, first_event=None):
    script = {
        "llm_usage_events": [
            events,  # _fetch_events page 1 (< page size => single page)
            [first_event] if first_event else [],  # _tracking_since
        ],
        "chatbot_conversations": [conversations or []],
    }
    return LLMObservabilityService(client=_Client(script))


def test_aggregation_end_to_end():
    events = [
        _ev("claude-sonnet-4-6", 1000, 500, user=U1, session=S1),
        _ev("claude-sonnet-4-6", 2000, 1000, user=U1, session=S1, day="2026-07-11"),
        _ev("gpt-4o", 500, 250, user=U1, session=S2),
        _ev("gpt-4o-mini", 100, 50, user=U2, session=S3),
        _ev("gpt-4o", 4000, 2000, surface="insights", component="ExecutiveBrief"),
    ]
    convs = [{"session_id": S1, "title": "Kisqali TRx", "created_at": "2026-07-10T11:59:00+00:00"}]
    svc = _service(events, conversations=convs, first_event={"created_at": "2026-07-01T00:00:00+00:00"})

    result = svc.llm_usage(30, USERS)

    s = result["summary"]
    assert s["calls"] == 5
    assert s["input_tokens"] == 7600
    assert s["output_tokens"] == 3800
    assert s["distinct_users"] == 2
    assert s["tracking_since"] == "2026-07-01T00:00:00+00:00"
    assert s["total_cost_usd"] > 0

    assert [d["date"] for d in result["daily"]] == ["2026-07-10", "2026-07-11"]
    assert result["daily"][0]["platform_cost_usd"] > 0
    assert result["daily"][1]["platform_cost_usd"] == 0

    by_user = {u["user_id"]: u for u in result["by_user"]}
    assert by_user[U1]["email"] == "alice@x.com"
    assert by_user[U1]["sessions"] == 2
    assert by_user[U1]["calls"] == 3
    assert "claude-sonnet-4-6" in by_user[U1]["models"]

    sessions_u1 = {r["session_id"]: r for r in result["sessions"][U1]}
    assert sessions_u1[S1]["title"] == "Kisqali TRx"
    assert sessions_u1[S1]["calls"] == 2
    assert sessions_u1[S2]["title"] is None

    assert len(result["platform"]) == 1
    p = result["platform"][0]
    assert (p["surface"], p["component"], p["model"]) == ("insights", "ExecutiveBrief", "gpt-4o")

    assert result["unpriced_models"] == []
    assert result["pricing_version"]


def test_unpriced_model_counted_but_not_costed():
    events = [_ev("mystery-lm-9", 1000, 1000, user=U1, session=S1)]
    svc = _service(events, first_event={"created_at": "2026-07-10T00:00:00+00:00"})
    result = svc.llm_usage(30, USERS)
    assert result["unpriced_models"] == ["mystery-lm-9"]
    assert result["summary"]["total_cost_usd"] == 0.0  # cost skipped, not faked
    assert result["summary"]["input_tokens"] == 1000  # tokens still honest


def test_empty_window():
    svc = _service([], first_event=None)
    result = svc.llm_usage(7, USERS)
    assert result["summary"]["calls"] == 0
    assert result["summary"]["tracking_since"] is None
    assert result["by_user"] == []
    assert result["daily"] == []
    assert result["platform"] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_observability_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/services/llm_observability_service.py`**

```python
"""Aggregation for GET /api/admin/observability/llm-usage (spec 2026-07-12).

Reads llm_usage_events and computes cost at READ time via llm_pricing, so
pricing corrections apply retroactively. Chat rows (user_id set) roll up per
user and per session; NULL-user rows aggregate into the platform section.
Unpriced models contribute tokens/calls but no cost and are listed in
unpriced_models — never silently costed. Sync methods by design: the route
runs them via asyncio.to_thread like its admin.py siblings.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.services.llm_pricing import PRICING_VERSION, cost_usd

logger = logging.getLogger(__name__)

_PAGE = 1000
_IN_CHUNK = 100  # keep .in_() URL length bounded

_EVENT_COLUMNS = (
    "created_at, provider, model, input_tokens, output_tokens, "
    "surface, component, user_id, session_id"
)


class LLMObservabilityService:
    def __init__(self, client: Optional[Any] = None) -> None:
        if client is None:
            from src.api.dependencies.supabase_client import get_supabase

            client = get_supabase()
        self.client = client

    # ------------------------------------------------------------- fetch ----

    def _fetch_events(self, since_iso: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = (
                self.client.table("llm_usage_events")
                .select(_EVENT_COLUMNS)
                .gte("created_at", since_iso)
                .order("id", desc=False)
                .range(offset, offset + _PAGE - 1)
                .execute()
                .data
                or []
            )
            events.extend(page)
            if len(page) < _PAGE:
                return events
            offset += _PAGE

    def _tracking_since(self) -> Optional[str]:
        rows = (
            self.client.table("llm_usage_events")
            .select("created_at")
            .order("id", desc=False)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0]["created_at"] if rows else None

    def _conversations(self, session_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(session_ids), _IN_CHUNK):
            chunk = session_ids[i : i + _IN_CHUNK]
            rows = (
                self.client.table("chatbot_conversations")
                .select("session_id, title, created_at")
                .in_("session_id", chunk)
                .execute()
                .data
                or []
            )
            for row in rows:
                out[row["session_id"]] = row
        return out

    # --------------------------------------------------------- aggregate ----

    def llm_usage(self, days: int, users: List[Dict[str, Any]]) -> Dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self._fetch_events(since.isoformat())
        emails = {u.get("id"): u.get("email") for u in users}

        unpriced: Set[str] = set()

        def _cost(event: Dict[str, Any], input_t: int, output_t: int) -> Optional[float]:
            cost = cost_usd(event.get("model") or "", input_t, output_t)
            if cost is None:
                unpriced.add(event.get("model") or "")
            return cost

        summary: Dict[str, Any] = {
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": 0,
            "distinct_users": 0,
            "days": days,
            "tracking_since": self._tracking_since(),
        }
        daily: Dict[str, Dict[str, Any]] = {}
        per_user: Dict[str, Dict[str, Any]] = {}
        per_session: Dict[str, Dict[str, Any]] = {}
        platform: Dict[Tuple[str, Optional[str], str], Dict[str, Any]] = {}

        for event in events:
            input_t = int(event.get("input_tokens") or 0)
            output_t = int(event.get("output_tokens") or 0)
            cost = _cost(event, input_t, output_t)
            model = event.get("model") or "unknown"

            summary["calls"] += 1
            summary["input_tokens"] += input_t
            summary["output_tokens"] += output_t
            if cost:
                summary["total_cost_usd"] += cost

            day = (event.get("created_at") or "")[:10]
            bucket = daily.setdefault(
                day,
                {"date": day, "chat_cost_usd": 0.0, "platform_cost_usd": 0.0, "tokens": 0},
            )
            bucket["tokens"] += input_t + output_t

            user_id = event.get("user_id")
            if user_id:
                if cost:
                    bucket["chat_cost_usd"] += cost
                user_row = per_user.setdefault(
                    user_id,
                    {
                        "user_id": user_id,
                        "email": emails.get(user_id),
                        "session_ids": set(),
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "models": set(),
                    },
                )
                session_id = event.get("session_id") or "unknown"
                user_row["session_ids"].add(session_id)
                user_row["calls"] += 1
                user_row["input_tokens"] += input_t
                user_row["output_tokens"] += output_t
                if cost:
                    user_row["cost_usd"] += cost
                user_row["models"].add(model)

                session_row = per_session.setdefault(
                    session_id,
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "first_event_at": event.get("created_at"),
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "models": set(),
                    },
                )
                session_row["calls"] += 1
                session_row["input_tokens"] += input_t
                session_row["output_tokens"] += output_t
                if cost:
                    session_row["cost_usd"] += cost
                session_row["models"].add(model)
            else:
                if cost:
                    bucket["platform_cost_usd"] += cost
                key = (event.get("surface") or "other", event.get("component"), model)
                platform_row = platform.setdefault(
                    key,
                    {
                        "surface": key[0],
                        "component": key[1],
                        "model": model,
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
                platform_row["calls"] += 1
                platform_row["input_tokens"] += input_t
                platform_row["output_tokens"] += output_t
                if cost:
                    platform_row["cost_usd"] += cost

        summary["distinct_users"] = len(per_user)
        summary["total_cost_usd"] = round(summary["total_cost_usd"], 6)

        conversations = self._conversations(sorted(per_session))

        by_user = []
        for row in per_user.values():
            by_user.append(
                {
                    "user_id": row["user_id"],
                    "email": row["email"],
                    "sessions": len(row["session_ids"]),
                    "calls": row["calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cost_usd": round(row["cost_usd"], 6),
                    "models": sorted(row["models"]),
                }
            )
        by_user.sort(key=lambda r: r["cost_usd"], reverse=True)

        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for row in per_session.values():
            conv = conversations.get(row["session_id"], {})
            sessions.setdefault(row["user_id"], []).append(
                {
                    "session_id": row["session_id"],
                    "title": conv.get("title"),
                    "started_at": conv.get("created_at") or row["first_event_at"],
                    "calls": row["calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cost_usd": round(row["cost_usd"], 6),
                    "models": sorted(row["models"]),
                }
            )
        for rows in sessions.values():
            rows.sort(key=lambda r: r["started_at"] or "", reverse=True)

        platform_rows = [
            {**row, "cost_usd": round(row["cost_usd"], 6)} for row in platform.values()
        ]
        platform_rows.sort(key=lambda r: r["cost_usd"], reverse=True)

        return {
            "summary": summary,
            "daily": [daily[d] for d in sorted(daily)],
            "by_user": by_user,
            "sessions": sessions,
            "platform": platform_rows,
            "pricing_version": PRICING_VERSION,
            "unpriced_models": sorted(m for m in unpriced if m),
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/test_services/test_llm_observability_service.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/llm_observability_service.py tests/unit/test_services/test_llm_observability_service.py
git commit -m "feat(observability): llm usage aggregation service

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Admin API endpoint

**Files:**
- Modify: `src/api/routes/admin.py` (imports ~line 18; new endpoint after `activity_overview`, line 336)
- Test: `tests/unit/test_api/test_routes/test_admin_llm_usage.py`

- [ ] **Step 1: Write the failing test**

```python
"""GET /admin/observability/llm-usage: thin route — admin-gated (Depends,
enforced like every admin.py sibling), runs the aggregation off-thread,
passes days + the user listing through."""

import asyncio

from src.api.routes.admin import llm_usage_overview


class _FakeAdminService:
    def list_users(self):
        return [{"id": "u1", "email": "a@x.com"}]


class _FakeObs:
    def __init__(self):
        self.calls = []

    def llm_usage(self, days, users):
        self.calls.append((days, users))
        return {"summary": {"days": days, "calls": 0}}


def test_llm_usage_overview_wires_days_and_users():
    obs = _FakeObs()
    result = asyncio.run(
        llm_usage_overview(
            days=7,
            admin={"id": "admin-1"},
            service=_FakeAdminService(),
            obs=obs,
        )
    )
    assert obs.calls == [(7, [{"id": "u1", "email": "a@x.com"}])]
    assert result["summary"]["days"] == 7
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_api/test_routes/test_admin_llm_usage.py -v`
Expected: FAIL — `ImportError: cannot import name 'llm_usage_overview'`

- [ ] **Step 3: Implement in `src/api/routes/admin.py`**

Add import (after the AdminUserService import block, line 25):

```python
from src.services.llm_observability_service import LLMObservabilityService
```

Add the singleton getter next to `get_admin_service` (line 44):

```python
_obs_service: Optional[LLMObservabilityService] = None


def get_llm_observability_service() -> LLMObservabilityService:
    global _obs_service
    if _obs_service is None:
        _obs_service = LLMObservabilityService()
    return _obs_service
```

Add the endpoint after `activity_overview` (line 336):

```python
@router.get("/observability/llm-usage")
async def llm_usage_overview(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
    obs: LLMObservabilityService = Depends(get_llm_observability_service),
) -> Dict[str, Any]:
    """LLM usage/tokens/cost: per-user + per-session (chat) and platform
    aggregates (spec 2026-07-12). Cost computed at read time from the
    pricing table; unpriced models surface in unpriced_models."""

    def _query() -> Dict[str, Any]:
        users = service.list_users()
        return obs.llm_usage(days, users)

    return await asyncio.to_thread(_query)
```

- [ ] **Step 4: Run tests + auth/drift guards**

Run:
```bash
.venv/bin/pytest tests/unit/test_api/test_routes/test_admin_llm_usage.py -v
.venv/bin/pytest tests/unit/test_security/test_auth_gating.py -q
```
Expected: all PASS (route lives on the existing gated `/admin` router — same tag, no new router, so no openapi_tags/sentinel drift updates are needed; the gating test run confirms).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/admin.py tests/unit/test_api/test_routes/test_admin_llm_usage.py
git commit -m "feat(observability): GET /admin/observability/llm-usage endpoint

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Frontend plumbing (types, client, query key, hook)

**Files:**
- Modify: `frontend/src/types/admin.ts` (append)
- Modify: `frontend/src/api/admin.ts` (append + extend type import)
- Modify: `frontend/src/lib/query-client.ts` (inside `admin: {...}`, after `auditFeed` line 565)
- Modify: `frontend/src/hooks/api/use-admin.ts` (append + extend imports)

- [ ] **Step 1: Append to `frontend/src/types/admin.ts`**

```ts
// --- LLM observability (mirrors GET /api/admin/observability/llm-usage) ---

export interface LlmUsageSummary {
  total_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  calls: number;
  distinct_users: number;
  days: number;
  tracking_since: string | null;
}

export interface LlmDailyUsage {
  date: string;
  chat_cost_usd: number;
  platform_cost_usd: number;
  tokens: number;
}

export interface LlmUserUsage {
  user_id: string;
  email: string | null;
  sessions: number;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  models: string[];
}

export interface LlmSessionUsage {
  session_id: string;
  title: string | null;
  started_at: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  models: string[];
}

export interface LlmPlatformUsage {
  surface: string;
  component: string | null;
  model: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface LlmUsageResponse {
  summary: LlmUsageSummary;
  daily: LlmDailyUsage[];
  by_user: LlmUserUsage[];
  sessions: Record<string, LlmSessionUsage[]>;
  platform: LlmPlatformUsage[];
  pricing_version: string;
  unpriced_models: string[];
}
```

- [ ] **Step 2: Append to `frontend/src/api/admin.ts`** (and add `LlmUsageResponse` to the type-import list at the top)

```ts
export function getLlmUsage(days = 30): Promise<LlmUsageResponse> {
  return get<LlmUsageResponse>(`${BASE}/observability/llm-usage`, { params: { days } });
}
```

- [ ] **Step 3: Add the query key in `frontend/src/lib/query-client.ts`** (after the `auditFeed` entry, line 565)

```ts
    llmUsage: (days: number) => [...queryKeys.admin.all(), 'llm-usage', days] as const,
```

- [ ] **Step 4: Append to `frontend/src/hooks/api/use-admin.ts`** (add `getLlmUsage` to the `@/api/admin` import list)

```ts
export function useLlmUsage(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.llmUsage(days),
    queryFn: () => getLlmUsage(days),
  });
}
```

- [ ] **Step 5: Typecheck**

Run: `cd frontend && npx tsc -p tsconfig.app.json --noEmit`
Expected: clean exit 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/admin.ts frontend/src/api/admin.ts frontend/src/lib/query-client.ts frontend/src/hooks/api/use-admin.ts
git commit -m "feat(observability): frontend api client, types, hook for llm usage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: ObservabilityTab component

**Files:**
- Create: `frontend/src/components/admin/ObservabilityTab.tsx`
- Create: `frontend/src/components/admin/ObservabilityTab.test.tsx`
- Modify: `frontend/src/components/admin/index.ts` (add export)

- [ ] **Step 1: Write the failing tests**

```tsx
/**
 * ObservabilityTab tests — stat cards, per-user expansion, platform section,
 * honest states (tracking banner, unpriced "—", empty window). recharts SVG
 * is not asserted (visual — verified live).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { LlmUsageResponse } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  useLlmUsage: vi.fn(),
}));

import * as adminHooks from '@/hooks/api/use-admin';
import { ObservabilityTab } from './ObservabilityTab';

const U1 = '11111111-1111-1111-1111-111111111111';

const PAYLOAD: LlmUsageResponse = {
  summary: {
    total_cost_usd: 1.234567,
    input_tokens: 120000,
    output_tokens: 45000,
    calls: 42,
    distinct_users: 2,
    days: 30,
    // dynamic so the "tracking began inside this window" banner assertion
    // never rots as real time passes
    tracking_since: new Date().toISOString(),
  },
  daily: [
    { date: '2026-07-12', chat_cost_usd: 0.9, platform_cost_usd: 0.33, tokens: 165000 },
  ],
  by_user: [
    {
      user_id: U1,
      email: 'alice@x.com',
      sessions: 2,
      calls: 30,
      input_tokens: 100000,
      output_tokens: 40000,
      cost_usd: 0.9,
      models: ['claude-sonnet-4-6'],
    },
  ],
  sessions: {
    [U1]: [
      {
        session_id: `${U1}~conv-a`,
        title: 'Kisqali TRx dip',
        started_at: '2026-07-12T10:00:00+00:00',
        calls: 20,
        input_tokens: 80000,
        output_tokens: 30000,
        cost_usd: 0.7,
        models: ['claude-sonnet-4-6'],
      },
    ],
  },
  platform: [
    {
      surface: 'insights',
      component: 'ExecutiveBrief',
      model: 'gpt-4o',
      calls: 12,
      input_tokens: 20000,
      output_tokens: 5000,
      cost_usd: 0.33,
    },
  ],
  pricing_version: '2026-07-12',
  unpriced_models: [],
};

const mockHook = (over: Partial<ReturnType<typeof buildResult>> = {}) => {
  (adminHooks.useLlmUsage as ReturnType<typeof vi.fn>).mockReturnValue(
    buildResult(over)
  );
};

function buildResult(over: object) {
  return { data: PAYLOAD, isLoading: false, isError: false, ...over };
}

beforeEach(() => vi.clearAllMocks());

describe('ObservabilityTab', () => {
  it('renders stat cards and the tracking banner', () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.getByText('Total cost')).toBeInTheDocument();
    expect(screen.getByText('$1.23')).toBeInTheDocument();
    expect(screen.getByText('LLM calls')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText(/Usage tracking began/)).toBeInTheDocument();
  });

  it('expands a user row to session breakdown', async () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.queryByText('Kisqali TRx dip')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /alice@x.com/ }));
    expect(screen.getByText('Kisqali TRx dip')).toBeInTheDocument();
  });

  it('renders the platform (non-chat) section', () => {
    mockHook();
    render(<ObservabilityTab />);
    expect(screen.getByText('Platform LLM usage (non-chat)')).toBeInTheDocument();
    expect(screen.getByText('ExecutiveBrief')).toBeInTheDocument();
  });

  it('lists unpriced models honestly', () => {
    mockHook({
      data: { ...PAYLOAD, unpriced_models: ['mystery-lm-9'] },
    });
    render(<ObservabilityTab />);
    expect(screen.getByText(/mystery-lm-9/)).toBeInTheDocument();
  });

  it('shows an explicit empty state', () => {
    mockHook({
      data: {
        ...PAYLOAD,
        summary: { ...PAYLOAD.summary, calls: 0, tracking_since: null },
        daily: [],
        by_user: [],
        sessions: {},
        platform: [],
      },
    });
    render(<ObservabilityTab />);
    expect(screen.getByText(/No LLM usage recorded/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd frontend && npx vitest run src/components/admin/ObservabilityTab.test.tsx`
Expected: FAIL — cannot resolve `./ObservabilityTab`.

- [ ] **Step 3: Implement `frontend/src/components/admin/ObservabilityTab.tsx`**

```tsx
/**
 * ObservabilityTab — LLM model, tokens, and $ cost per user / per session
 * (chat), plus platform (non-chat) spend. GET /api/admin/observability/llm-usage.
 *
 * Honesty rules (spec 2026-07-12): attribution is chat-only (everything else
 * is the Platform section); pre-feature history is "untracked", never
 * estimated; unpriced models render "—", never $0.
 */

import { Fragment, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useLlmUsage } from '@/hooks/api/use-admin';

const fmtInt = (n: number) => n.toLocaleString();
const fmtCost = (n: number | null | undefined) => {
  if (n == null) return '—';
  if (n === 0) return '$0';
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
};

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] p-4">
      <div className="text-xs uppercase text-[var(--color-muted-foreground)]">{label}</div>
      <div className="text-2xl font-semibold text-[var(--color-foreground)]">{value}</div>
    </div>
  );
}

function ModelChips({ models }: { models: string[] }) {
  return (
    <span className="flex flex-wrap gap-1">
      {models.map((m) => (
        <span
          key={m}
          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-xs text-[var(--color-muted-foreground)]"
        >
          {m}
        </span>
      ))}
    </span>
  );
}

const TH = 'px-3 py-2 text-left text-xs font-medium uppercase text-[var(--color-muted-foreground)]';
const TD = 'px-3 py-2 text-sm text-[var(--color-foreground)]';

export function ObservabilityTab() {
  const [days, setDays] = useState(30);
  const [expandedUser, setExpandedUser] = useState<string | null>(null);
  const { data, isLoading, isError } = useLlmUsage(days);

  if (isLoading) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">Loading LLM usage…</p>
    );
  }
  if (isError || !data) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">
        Failed to load LLM usage.
      </p>
    );
  }

  const { summary, daily, by_user, sessions, platform, unpriced_models } = data;
  const windowStart = Date.now() - days * 86_400_000;
  const trackingLater =
    summary.tracking_since && new Date(summary.tracking_since).getTime() > windowStart;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[var(--color-foreground)]">
            LLM observability
          </h2>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Models used, tokens consumed, and cost — per user and per chat session. Costs are
            computed from the pricing table (v{data.pricing_version}).
          </p>
        </div>
        <select
          aria-label="Time range"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
        >
          <option value={7}>7 days</option>
          <option value={30}>30 days</option>
          <option value={90}>90 days</option>
        </select>
      </div>

      {trackingLater && (
        <p className="rounded-lg border border-[var(--color-border)] bg-[var(--color-muted)] px-4 py-2 text-sm text-[var(--color-muted-foreground)]">
          Usage tracking began {new Date(summary.tracking_since as string).toLocaleDateString()};
          earlier sessions are untracked.
        </p>
      )}

      {unpriced_models.length > 0 && (
        <p className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
          Models missing from the pricing table (cost shown as —): {unpriced_models.join(', ')}
        </p>
      )}

      {summary.calls === 0 ? (
        <p className="rounded-lg border border-dashed border-[var(--color-border)] p-6 text-center text-sm text-[var(--color-muted-foreground)]">
          No LLM usage recorded in this window.
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Total cost" value={fmtCost(summary.total_cost_usd)} />
            <StatCard
              label="Tokens (in / out)"
              value={`${fmtInt(summary.input_tokens)} / ${fmtInt(summary.output_tokens)}`}
            />
            <StatCard label="LLM calls" value={fmtInt(summary.calls)} />
            <StatCard label="Active users" value={fmtInt(summary.distinct_users)} />
          </div>

          <section>
            <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
              Cost per day — chat vs platform
            </h3>
            <div className="h-64 rounded-lg border border-[var(--color-border)] p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={daily}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => fmtCost(v)} />
                  <Legend />
                  <Bar dataKey="chat_cost_usd" stackId="c" fill="#6366f1" name="Chat" />
                  <Bar dataKey="platform_cost_usd" stackId="c" fill="#10b981" name="Platform" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section>
            <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
              Usage by user (chat)
            </h3>
            <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    <th className={TH}>User</th>
                    <th className={TH}>Sessions</th>
                    <th className={TH}>Calls</th>
                    <th className={TH}>Tokens (in / out)</th>
                    <th className={TH}>Cost</th>
                    <th className={TH}>Models</th>
                  </tr>
                </thead>
                <tbody>
                  {by_user.map((u) => (
                    <Fragment key={u.user_id}>
                      <tr className="border-b border-[var(--color-border)]">
                        <td className={TD}>
                          <button
                            type="button"
                            aria-expanded={expandedUser === u.user_id}
                            onClick={() =>
                              setExpandedUser(expandedUser === u.user_id ? null : u.user_id)
                            }
                            className="font-medium text-[var(--color-primary)]"
                          >
                            {expandedUser === u.user_id ? '▾ ' : '▸ '}
                            {u.email ?? u.user_id}
                          </button>
                        </td>
                        <td className={TD}>{u.sessions}</td>
                        <td className={TD}>{fmtInt(u.calls)}</td>
                        <td className={TD}>
                          {fmtInt(u.input_tokens)} / {fmtInt(u.output_tokens)}
                        </td>
                        <td className={TD}>{fmtCost(u.cost_usd)}</td>
                        <td className={TD}>
                          <ModelChips models={u.models} />
                        </td>
                      </tr>
                      {expandedUser === u.user_id &&
                        (sessions[u.user_id] ?? []).map((s) => (
                          <tr key={s.session_id} className="border-b border-[var(--color-border)] bg-[var(--color-muted)]">
                            <td className={`${TD} pl-8`}>
                              {s.title ?? s.session_id.split('~')[1] ?? s.session_id}
                              <span className="ml-2 text-xs text-[var(--color-muted-foreground)]">
                                {s.started_at
                                  ? new Date(s.started_at).toLocaleString()
                                  : 'start unknown'}
                              </span>
                            </td>
                            <td className={TD}>—</td>
                            <td className={TD}>{fmtInt(s.calls)}</td>
                            <td className={TD}>
                              {fmtInt(s.input_tokens)} / {fmtInt(s.output_tokens)}
                            </td>
                            <td className={TD}>{fmtCost(s.cost_usd)}</td>
                            <td className={TD}>
                              <ModelChips models={s.models} />
                            </td>
                          </tr>
                        ))}
                    </Fragment>
                  ))}
                  {by_user.length === 0 && (
                    <tr>
                      <td colSpan={6} className={`${TD} text-center text-[var(--color-muted-foreground)]`}>
                        No attributed chat usage in this window.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h3 className="mb-1 text-sm font-medium text-[var(--color-foreground)]">
              Platform LLM usage (non-chat)
            </h3>
            <p className="mb-2 text-sm text-[var(--color-muted-foreground)]">
              Insights, agents, and background generation — not attributable to individual users.
            </p>
            <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    <th className={TH}>Surface</th>
                    <th className={TH}>Component</th>
                    <th className={TH}>Model</th>
                    <th className={TH}>Calls</th>
                    <th className={TH}>Tokens (in / out)</th>
                    <th className={TH}>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {platform.map((p) => (
                    <tr
                      key={`${p.surface}|${p.component}|${p.model}`}
                      className="border-b border-[var(--color-border)]"
                    >
                      <td className={TD}>{p.surface}</td>
                      <td className={TD}>{p.component ?? '—'}</td>
                      <td className={TD}>{p.model}</td>
                      <td className={TD}>{fmtInt(p.calls)}</td>
                      <td className={TD}>
                        {fmtInt(p.input_tokens)} / {fmtInt(p.output_tokens)}
                      </td>
                      <td className={TD}>{fmtCost(p.cost_usd)}</td>
                    </tr>
                  ))}
                  {platform.length === 0 && (
                    <tr>
                      <td colSpan={6} className={`${TD} text-center text-[var(--color-muted-foreground)]`}>
                        No platform LLM usage in this window.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Export from `frontend/src/components/admin/index.ts`** — add:

```ts
export { ObservabilityTab } from './ObservabilityTab';
```

- [ ] **Step 5: Run to verify pass**

Run: `cd frontend && npx vitest run src/components/admin/ObservabilityTab.test.tsx`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/admin/ObservabilityTab.tsx frontend/src/components/admin/ObservabilityTab.test.tsx frontend/src/components/admin/index.ts
git commit -m "feat(observability): ObservabilityTab component

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14: Wire the tab into Admin.tsx

**Files:**
- Modify: `frontend/src/pages/Admin.tsx`

- [ ] **Step 1: Make the three edits**

Line 15 area — add import:

```tsx
import { ObservabilityTab } from '@/components/admin/ObservabilityTab';
```

Line 18 — widen the type:

```tsx
type Tab = 'users' | 'activity' | 'observability';
```

Line 49 — extend the tablist array:

```tsx
        {(['users', 'activity', 'observability'] as const).map((t) => (
```

After line 77 (`{tab === 'activity' && ...}`) — render it:

```tsx
      {tab === 'observability' && <ObservabilityTab />}
```

Line 34 — update the subtitle:

```tsx
            Invite users, manage roles and brand access, review activity and LLM usage.
```

- [ ] **Step 2: Typecheck + full admin component tests**

Run:
```bash
cd frontend && npx tsc -p tsconfig.app.json --noEmit && npx vitest run src/components/admin/
```
Expected: clean typecheck; all admin tests PASS.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/Admin.tsx
git commit -m "feat(admin): observability tab on /admin

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 15: Full verification, push, PR

- [ ] **Step 1: Lint + format (both gates), scoped to changed files**

Run:
```bash
CHANGED=$(git diff --name-only main...HEAD -- '*.py' | tr '\n' ' ')
.venv/bin/ruff check $CHANGED
.venv/bin/ruff format --check $CHANGED
```
Expected: both clean. If format flags files, run `.venv/bin/ruff format <files>` and amend.

- [ ] **Step 2: Scoped mypy (changed files only — NEVER whole-tree on this box)**

Run:
```bash
.venv/bin/mypy --config-file pyproject.toml src/services/llm_pricing.py src/utils/llm_attribution.py src/services/llm_usage_recorder.py src/utils/llm_usage_callback.py src/utils/litellm_usage_logger.py src/services/llm_observability_service.py
```
Expected: no NEW errors in these files (CI's mypy gate is the ceiling arbiter).

- [ ] **Step 3: Run every test added by this plan**

Run:
```bash
.venv/bin/pytest \
  tests/unit/test_services/test_llm_pricing.py \
  tests/unit/test_utils/test_llm_attribution.py \
  tests/unit/test_services/test_llm_usage_recorder.py \
  tests/unit/test_utils/test_llm_usage_capture.py \
  tests/unit/test_services/test_llm_observability_service.py \
  tests/unit/test_api/test_routes/test_admin_llm_usage.py \
  tests/unit/test_api/test_routes/test_copilotkit_usage_stamping.py -v
```
Expected: all PASS.

- [ ] **Step 4: Push and open the PR**

```bash
git config --global http.https://github.com.proxy ""
git push -u origin feat/admin-llm-observability
gh pr create \
  --title "feat(admin): Observability tab — LLM usage, tokens, cost per user/session" \
  --body "$(cat <<'EOF'
## Summary
- Migration 104 `llm_usage_events` + two fail-open capture hooks (llm_factory LangChain callback; global litellm logger covering all dspy traffic)
- Contextvar attribution: chat runs carry user/session; everything else aggregates as platform-level (NULL user) — attribution is honest-only
- Persist-time stamping fills the previously-always-NULL `chatbot_messages.tokens_used/model_used` (and removes the env-var `model_used` fabrication in graph finalize)
- `GET /api/admin/observability/llm-usage` (admin-gated), cost computed at READ time from the pricing table (unpriced models surfaced, never $0)
- New Observability tab on /admin: stat cards, chat-vs-platform daily cost chart, per-user table expandable to sessions, platform section, honest empty/untracked states

Spec: docs/superpowers/specs/2026-07-12-admin-observability-tab-design.md
Capture gate: scripts/verify_llm_usage_capture.py PASSED on the droplet (real keys, streamed LangChain call + dspy call both captured with nonzero tokens).

## Test plan
- [ ] CI green (unit tests cover pricing, attribution, recorder, both hooks, aggregation, route, persist stamping; vitest covers the tab)
- [ ] Post-deploy: send one chat message → row in llm_usage_events attributed to the sender; tab renders it
EOF
)"
```

Expected: PR created; report the URL and wait for CI. **Do not merge or deploy — await explicit user go** (established feedback: bundle actions, await explicit go).

- [ ] **Step 5: Watch CI**

Run: `gh pr checks --watch` (or `gh run list --branch feat/admin-llm-observability`)
Expected: lint, mypy, backend tests, frontend build all green. Fix and push if not.

---

### Task 16: Post-merge live verification (runs only after the user merges + deploys)

- [ ] **Step 1: Confirm rows accrue in prod**

After deploy, send one chat message in the live UI (user's logged-in browser), then:

```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT provider, model, input_tokens, output_tokens, surface, user_id IS NOT NULL AS attributed, session_id
   FROM llm_usage_events ORDER BY id DESC LIMIT 10;"
```
Expected: a `surface='chat'` row with `attributed=t` and the sender's session_id; plus any background `insights`/`other` rows with `attributed=f`.

- [ ] **Step 2: Confirm the message row got stamped**

```bash
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT role, model_used, tokens_used FROM chatbot_messages ORDER BY created_at DESC LIMIT 4;"
```
Expected: newest assistant row has a real model id and nonzero tokens_used.

- [ ] **Step 3: Verify the tab live**

Open https://eznomics.site/admin → Observability (hard-reload first — open tabs keep old bundles). Expected: stat cards nonzero, the chat row attributed to the sending user, platform section populated as background jobs run.

---

## Out of scope (spec §9)

Budgets/alerts, historical token backfill (impossible), per-message drill-down UI, Prometheus/OTel export, embedding-model usage.
