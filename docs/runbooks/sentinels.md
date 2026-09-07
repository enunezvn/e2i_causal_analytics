# Sentinels Operations Runbook

**Version**: 1.1 | **Last Updated**: 2026-09-07 | **Status**: Living Document

Owner: Platform / Memory Subsystems

Companion plan: `.claude/plans/archive/e2i_memory_subsystems_implementation_plan_archived_20260520.md`
(Phase 3) — archived 2026-05-20; a **local, untracked planning note**, not in the
repo. Historical context only.

---

## Table of Contents

1. [What is a Sentinel?](#1-what-is-a-sentinel)
2. [YAML Configuration Schema](#2-yaml-configuration-schema)
3. [Lifecycle States](#3-lifecycle-states)
4. [Redis Alert Channel](#4-redis-alert-channel)
5. [Cooldown Semantics](#5-cooldown-semantics)
6. [Action Handlers](#6-action-handlers)
7. [Adding a New Sentinel](#7-adding-a-new-sentinel)
8. [Testing a Sentinel Locally](#8-testing-a-sentinel-locally)
9. [Silencing a Sentinel During Incident Response](#9-silencing-a-sentinel-during-incident-response)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What is a Sentinel?

A **sentinel** is a data-driven watcher that evaluates a SQL/Redis condition on a
schedule and fires an action when the condition matches. Sentinels replace
hardcoded Celery beat tasks for the common case of *"if condition X about
table Y happens, fire action Z."*

Two paths produce sentinels in the runtime database:

- **YAML config**: `config/sentinels.yaml` is read at API startup by
  `src.memory.sentinels.config_loader.load_sentinels_from_yaml`
  (`src/memory/sentinels/config_loader.py`). Operator-facing — uses the
  *plan vocabulary* (`data_drop`, `staleness_threshold`, `cohort_drift`,
  `schedule`).
- **REST API**: `POST /api/sentinels` registers a sentinel programmatically.
  Uses the *shipped registry vocabulary* (`threshold_breach`, `freshness`,
  `drift_score`, `new_causal_path`, `invalidation_count`).

A single Celery beat task evaluates each enabled sentinel; errors in one
sentinel never block others. The names you need in order to find it (verified
2026-09-07):

| | |
| --- | --- |
| Celery task name | `src.tasks.sentinel_dispatcher` |
| Python function | `sentinel_dispatcher` in `src/tasks/insight_lifecycle_tasks.py` |
| Beat schedule key | `insight-lifecycle-sentinels` in `src/workers/celery_app.py` |
| Schedule | `300.0` seconds (5 minutes), queue `quick` |
| Registry entry point | `dispatch_sentinels` in `src/memory/sentinels/registry.py` |

### Plan vocabulary vs shipped vocabulary

The YAML uses operator-friendly nouns; the registry uses internal/mechanistic
names. Translation is single-source-of-truth at
`PLAN_TRIGGER_TO_INTERNAL_PATTERN` in `src/memory/sentinels/config_loader.py`:

| Plan trigger (YAML)    | Shipped pattern (registry)  | Semantics                                       |
|------------------------|-----------------------------|-------------------------------------------------|
| `data_drop`            | `freshness`                 | Row age > `max_age_hours` on tracked table.     |
| `staleness_threshold`  | `invalidation_count`        | Rows where `invalidated_at IS NOT NULL`.        |
| `cohort_drift`         | `drift_score`               | Drift score from drift_monitor exceeds bound.   |
| `schedule`             | `new_causal_path`           | Rows in `causal_paths` since `last_fired_at`.   |

If you are editing YAML, use the **left column**. If you are calling the
REST API or reading registry source, use the **right column**.

---

## 2. YAML Configuration Schema

The canonical config is `config/sentinels.yaml`. Two top-level keys are
required:

```yaml
lifecycle_state: advisory   # required — see §3
sentinels:                  # required — list of sentinel objects
  - id: sentinel_<name>
    name: <human readable>
    trigger_type: <plan vocab>
    condition: {...}
    action: <action name>
    brands: ["all" | "<brand>" | ...]
    active: true | false
    cooldown_minutes: <int>
```

### Sentinel object schema

| Field              | Type        | Required | Description                                                                                |
|--------------------|-------------|----------|--------------------------------------------------------------------------------------------|
| `id`               | string      | yes      | Unique YAML identifier (used in logs).                                                     |
| `name`             | string      | yes      | Human-readable label; surfaced in alert payloads.                                          |
| `trigger_type`     | string      | yes      | One of `data_drop`, `staleness_threshold`, `cohort_drift`, `schedule`.                     |
| `condition`        | object      | yes      | Per-trigger config; see [Condition schemas](#condition-schemas).                           |
| `action`           | string      | yes      | One of `rerun_all_active_cohorts`, `notify_and_queue_reanalysis`, `flag_for_review`, `run_full_consolidation`. |
| `brands`           | list / `"all"` | yes   | Brand scope; `["all"]` requires ADMIN role at the API layer.                                |
| `active`           | bool        | yes      | `false` disables registration entirely.                                                    |
| `cooldown_minutes` | int (>=0)   | no       | Minutes between consecutive fires. `0` / unset = no cooldown.                              |

### Condition schemas

Each `trigger_type` requires a different `condition` shape:

#### `data_drop` (freshness)

```yaml
condition:
  table: <threshold-watchable table>  # required
  ts_column: <timestamp column>       # required
  max_age_hours: <float>              # required
```

**Constraints (`_validate_watch_target` in `src/memory/sentinels/registry.py`,
enforced at BOTH registration and evaluation time — review finding M8):**

- `table` must be in `THRESHOLD_WATCHABLE_TABLES` — **5 tables**, verified
  2026-09-07: `causal_paths`, `triggers`, `ml_predictions`, `episodic_memories`,
  `executive_insights`. This is a different (larger) allowlist from
  `INVALIDATION_AWARE_TABLES`, which constrains `staleness_threshold` only.
- `ts_column` must match `^[A-Za-z_][A-Za-z0-9_]*$` — a plain SQL identifier.
  `table`/`column` are operator-supplied and interpolated into a PostgREST
  projection, where `*`, `,`, `(`, `)` and `:` are meta-characters; the pattern
  blocks projection-widening and foreign-table exfiltration.

The same two constraints apply to `threshold_breach`.

Fires when any row in `table` has `ts_column < (now - max_age_hours)`.
Source: `_eval_freshness` in `src/memory/sentinels/registry.py`.

#### `staleness_threshold` (invalidation_count)

```yaml
condition:
  table: <invalidation-aware table>   # required — must be in INVALIDATION_AWARE_TABLES
  tier: <optional human label>        # optional — surfaced in alert payloads
```

Fires for every row where `invalidated_at IS NOT NULL`. Per
**Decision 3 = KEEP BINARY** (plan §"DECISIONS ADOPTED" 2026-05-19), each
match carries `staleness_score=1.0` — no graded staleness.

**Allowed tables** are validated at registration time
(`INVALIDATION_AWARE_TABLES` in `src/memory/sentinels/registry.py`):

- `triggers`
- `ml_predictions`
- `executive_insights`

Validation lives at `_validate_pattern_config`
(`_validate_pattern_config` in `src/memory/sentinels/registry.py`) and fails loudly with a
`ValueError` if you target a table without an `invalidated_at` column.

#### `cohort_drift` (drift_score)

```yaml
condition:
  max_drift_score: <float, 0.0-1.0>   # required
  cohort_id: <optional cohort id>     # optional
```

Fires when the latest `drift_monitor` output exceeds `max_drift_score`.
Source: `_eval_drift_score` in `src/memory/sentinels/registry.py`.

> Note: v1 returns `[]` unless a downstream integration provides drift
> outputs — the action plumbing is exercised by other trigger types.

#### `schedule` (new_causal_path)

```yaml
condition: {}     # empty — fires whenever new causal_paths exist
```

Fires for `causal_paths` rows with `created_at > last_fired_at`. Auto-bumps
`last_fired_at` after firing. Source: `_eval_new_causal_path` at
`_eval_new_causal_path` in `src/memory/sentinels/registry.py`.

---

## 3. Lifecycle States

Every sentinel YAML config MUST declare a top-level `lifecycle_state` key.
Five values are valid, defined as a canonical enum at
`GateLifecycleState` in `src/lifecycle/gate_lifecycle.py`:

| State          | Behavior                                                                            | Sentinel usage                                |
|----------------|-------------------------------------------------------------------------------------|-----------------------------------------------|
| `development`  | Plumbing exists, not connected. Never fires actions.                                | Rare — new untested sentinel pre-deploy.      |
| `advisory`     | Emits signals + queues Celery actions. Does NOT drop features, halt, or deny.       | **Default for sentinels.**                    |
| `calibrating`  | Same as advisory + emits `would_be_reject_rate` metric.                             | Use during initial sentinel tuning windows.   |
| `enforced`     | Verdict drops a feature, halts a pipeline, or denies a promotion. Requires signed lifecycle-change doc. | Rare for sentinels — they observe, they don't block. |
| `deprecated`   | Code remains for backward-compat reads but new emissions stop. Terminal.            | Use when superseded by a newer sentinel.      |

### Why sentinels default to `advisory`

Per the canonical semantics: sentinels publish to the `e2i:alerts` Redis
channel and queue Celery action tasks, but they do NOT drop features, halt
pipelines, or deny promotions. That is the textbook `advisory` state — emit
signals, do not block. The YAML header comment at
`config/sentinels.yaml:25-35` documents this rationale inline.

### Transition policy

Transitions are enforced by `scripts/check_lifecycle_state.py` +
`.github/workflows/lifecycle_state_guard.yml`. The allowed transitions live
in the `_ALLOWED_TRANSITIONS` map read by `is_transition_allowed` (`src/lifecycle/gate_lifecycle.py`):

- `development` → `advisory` or `deprecated`
- `advisory` → `calibrating` or `deprecated`
- `calibrating` → `advisory` (rollback) or `enforced` or `deprecated`
- `enforced` → `calibrating` (re-calibrate after drift) or `deprecated`
- `deprecated` → (terminal — no out-transitions)

Identity transitions (`X` → `X`) are explicitly NOT allowed.
`development` → `enforced` is intentionally blocked — the gate must spend
at least one `advisory` and one `calibrating` window before being enforced
(see the L3 note above `is_transition_allowed` in `src/lifecycle/gate_lifecycle.py`).

### The lifecycle state guard CI check

`scripts/check_lifecycle_state.py` substring-matches `threshold`, `cutoff`,
and `buffer` **anywhere** in YAML (value or key). Any config that contains
those substrings — including a value like `staleness_threshold` or a key
like `max_age_hours` — MUST declare `lifecycle_state` at the top level. The
guard runs on every push/PR via the lifecycle workflow.

---

## 4. Redis Alert Channel

The single cross-process alert channel is `e2i:alerts`, defined as a
module-level `Final[str]` constant at
the module-level constant in `src/tasks/sentinel_actions.py`:

```python
ALERTS_CHANNEL: Final[str] = "e2i:alerts"
```

### Publishers

Four sentinel-action handlers in `src/tasks/sentinel_actions.py` publish to
this channel via the best-effort `publish_alert()` helper
(`publish_alert` in `src/tasks/sentinel_actions.py`):

| Handler                          | Alert payload `type`         | Triggered by                              |
|----------------------------------|------------------------------|-------------------------------------------|
| `rerun_all_active_cohorts`       | `data_refresh`               | `data_drop` sentinels                     |
| `notify_and_queue_reanalysis`    | `staleness_alert`            | `staleness_threshold` sentinels           |
| `flag_for_review`                | `cohort_drift`               | `cohort_drift` sentinels                  |
| `run_full_consolidation`         | `full_consolidation_run`     | `schedule` sentinels                      |

`publish_alert` also fires from the `notify` action type in the dispatcher
(`_fire_action` in `src/memory/sentinels/registry.py`), emitting `type=sentinel_notify`.

### Subscribers

The primary subscriber is the CopilotKit SSE bridge at
`src/api/routes/staleness_alerts.py` (added in PR #394). It exposes:

```
GET /api/alerts/stream?brand=<brand>   AUTH
```

Returns `text/event-stream` with one SSE `event: alert` per matching
publish on `e2i:alerts`. Authentication is `Depends(require_auth)` at
the `alerts_stream` handler in `src/api/routes/staleness_alerts.py`.

### Alert payload shapes

All payloads are JSON dictionaries with brand-aware routing. The SSE
bridge filters by the connection's requested brand
(`AlertBridge._matches_brand` in `src/api/routes/staleness_alerts.py`); `"all"` in `brands` matches
every subscriber.

#### `data_refresh`

```json
{
  "type": "data_refresh",
  "sentinel_id": "<uuid>",
  "brands": ["all" | "<brand>", ...],
  "trigger_data": {...}
}
```

#### `staleness_alert`

```json
{
  "type": "staleness_alert",
  "sentinel_id": "<uuid>",
  "brands": ["<brand>"],
  "findings": [
    {
      "finding_id": "<insight_id>",
      "row_id": "<insight_id>",
      "brand": "<brand>",
      "table": "executive_insights",
      "invalidated_at": "<iso>",
      "staleness_score": 1.0
    },
    ...
  ]
}
```

The `findings` array contains the FULL stale-findings list as received by
the handler — see
`notify_and_queue_reanalysis` in `src/tasks/sentinel_actions.py`. The top-5
cap (`_REANALYSIS_CAP = 5`, same module) only applies INTERNALLY for the
per-finding `reanalyze_finding` Celery enqueue and log lines, not the
alert payload itself.

#### `cohort_drift`

```json
{
  "type": "cohort_drift",
  "sentinel_id": "<uuid>",
  "brands": ["<brand>"],
  "drift_data": {...}
}
```

`drift_data` is whatever `trigger_data.get("drift_data", trigger_data)`
resolves to (the tail of `notify_and_queue_reanalysis`) — typically the
full dispatcher `trigger_data` dict (sentinel match + action input).

#### `full_consolidation_run`

```json
{
  "type": "full_consolidation_run",
  "sentinel_id": "<uuid>",
  "brands": ["all"],
  "promoted_to_semantic": <int>,
  "promoted_to_procedural": <int>,
  "errors": [...]
}
```

Top-level fields are flattened from the consolidator result (see
`run_full_consolidation` in `src/tasks/sentinel_actions.py`).

#### `sentinel_notify` (action type `notify`)

```json
{
  "type": "sentinel_notify",
  "sentinel_id": "<uuid>",
  "sentinel_name": "<name>",
  "brand": "<brand>",
  "match": {...},
  "action_config": {...}
}
```

### Backpressure

Per-connection queue caps at 100 events
(the per-connection queue cap constant in `src/api/routes/staleness_alerts.py`). At capacity, the oldest event
is evicted (drop-oldest semantics) and a `WARNING` is logged. The
connection stays open — disconnecting a slow client would defeat the
alerting contract. Logs throttle to every 25th drop after the first.

---

## 5. Cooldown Semantics

Cooldown prevents a sentinel from re-firing too aggressively. Persisted
to the `sentinels.cooldown_minutes` column (migration
`database/memory/023_sentinel_cooldown.sql`).

### Semantics

Enforced by `_is_in_cooldown` at
`_is_in_cooldown` in `src/memory/sentinels/registry.py`:

| State                                                | Behavior                       |
|------------------------------------------------------|--------------------------------|
| `cooldown_minutes IS NULL`                           | No cooldown — always evaluate. |
| `cooldown_minutes == 0`                              | No cooldown — always evaluate. |
| `last_fired_at IS NULL`                              | Never fired — evaluate.        |
| `now - last_fired_at >= cooldown_minutes`            | Cooldown elapsed — evaluate.   |
| `now - last_fired_at < cooldown_minutes`             | **SKIP** — cooldown in effect. |

### Defaults

- **Migration 023** sets `DEFAULT 0` on the column (no cooldown gate).
  This preserves pre-#375 "always-fire" semantics. Switching to a 60-minute
  default would silently alter pre-existing sentinel behaviour.
- **YAML registration** uses whatever `cooldown_minutes` value is in the
  config; the loader passes it through to `register_sentinel`.
- **REST API registration** at `POST /api/sentinels` accepts
  `cooldown_minutes` as an optional field.

### Defense-in-depth constraints

Migration 023 also adds CHECK constraints to reject bad rows:

- `cooldown_minutes IS NULL OR cooldown_minutes >= 0`
- `cooldown_minutes IS NULL OR cooldown_minutes <= 525600` (365 days)

Catches operator typos (e.g. forgetting unit conversion).

### Defensive code

`_is_in_cooldown` defends against:

- **Boolean-as-int**: `cooldown_minutes=False` or `=True` smuggled past
  registration by direct DB write are treated as "no cooldown" since the
  gate is no longer trustworthy (Python's `True`/`False` are `int`
  subclasses).
- **Non-numeric values**: NaN / strings are treated as "no cooldown".
- **Unparseable `last_fired_at`**: treated as "never fired" so the
  sentinel can still evaluate.

### Reasonable defaults by trigger type

The shipped `config/sentinels.yaml` uses:

| Sentinel                          | Cooldown   | Rationale                                          |
|-----------------------------------|------------|----------------------------------------------------|
| `sentinel_optum_quarterly`        | 1440 (24h) | Optum CDM quarterly drop; daily is generous.       |
| `sentinel_staleness_alert`        | 360 (6h)   | Avoid swamping CopilotKit dashboards with re-fires.|
| `sentinel_pluvicto_cohort_drift`  | 2880 (48h) | Drift signals are durable; SME review is slow.     |
| `sentinel_weekly_consolidation`   | 10080 (7d) | Weekly cadence by design.                          |

---

## 6. Action Handlers

Each sentinel fires exactly one `action`. Four plan-specified action
handlers are implemented as async functions + Celery wrappers in
`src/tasks/sentinel_actions.py`:

### `rerun_all_active_cohorts`

Fires on `data_drop` sentinels. Publishes a `data_refresh` alert listing
all brands the operator wants re-run; downstream cohort-construction
pipelines pick this up via the existing trigger surface.

> The handler does NOT call `trigger_pipeline.delay(...)` directly — the
> pipeline-trigger surface is being reshaped under #237. The Redis alert
> is the portable contract.

### `notify_and_queue_reanalysis`

Fires on `staleness_threshold` sentinels. Publishes a `staleness_alert`
carrying the FULL stale-findings list (sorted by `staleness_score` —
currently 1.0 binary) AND enqueues a `reanalyze_finding` Celery task
for the top-5 most-stale findings (`_REANALYSIS_CAP = 5`,
`_REANALYSIS_CAP` in `src/tasks/sentinel_actions.py`). The top-5 cap applies ONLY to the
per-finding Celery enqueue + log lines, not the SSE alert payload
(see §4 for the on-the-wire shape). Broker outage on the per-finding
enqueue is best-effort: the Redis alert publication still goes out
regardless.

This is the **single-fire-with-list** path: the dispatcher calls this
handler exactly once per dispatcher tick with the full matches list
(rather than per-match). The bus event still fires per-match for
back-compat with the PR #250 contract. See
`_trigger_data_for_dispatch` in `src/memory/sentinels/registry.py`.

### `flag_for_review`

Fires on `cohort_drift` sentinels. Publishes a `cohort_drift` alert for
SME review. Sweepstakes the alert into CopilotKit dashboards via the SSE
bridge.

### `run_full_consolidation`

Fires on `schedule` sentinels. Runs a full `Consolidator.run()` pass
(`src/memory/lifecycle/consolidator.py`) and publishes a
`full_consolidation_run` heartbeat. The consolidator's `run()` performs:

1. `deduplicate_episodic` — collapses near-duplicate episodic rows.
2. `_promote_to_semantic` — stamps causal_paths as consolidated.
3. `_promote_to_procedural` — graduates procedural memories.

### Action wiring (dispatcher → Celery)

When a sentinel's `dispatch_agent` action carries an `agent_name` matching
one of the four plan-specified names, the dispatcher additionally calls
`celery_app.send_task(...)` so a worker enqueues the handler. Mapping is
single-source-of-truth at `PLAN_ACTION_TO_CELERY_TASK` in `src/memory/sentinels/registry.py`:

```python
PLAN_ACTION_TO_CELERY_TASK: Dict[str, str] = {
    "rerun_all_active_cohorts": "src.tasks.sentinel_actions.rerun_all_active_cohorts",
    "notify_and_queue_reanalysis": "src.tasks.sentinel_actions.notify_and_queue_reanalysis",
    "flag_for_review": "src.tasks.sentinel_actions.flag_for_review",
    "run_full_consolidation": "src.tasks.sentinel_actions.run_full_consolidation",
}
```

`agent_name` values outside this map flow bus-only (preserving back-compat
with the PR #250 contract).

---

## 7. Adding a New Sentinel

### Adding via YAML (recommended for operators)

1. Edit `config/sentinels.yaml`. Append a new entry following the schema
   in §2.
2. If your trigger type uses one of the substring patterns
   (`threshold`, `cutoff`, `buffer` anywhere in YAML), verify the
   top-level `lifecycle_state: advisory` is present. The lifecycle-state
   guard CI check will catch this if you forget.
3. If your trigger is `staleness_threshold`, confirm the `condition.table`
   is in `INVALIDATION_AWARE_TABLES`
   (`INVALIDATION_AWARE_TABLES` in `src/memory/sentinels/registry.py`) — currently `triggers`,
   `ml_predictions`, `executive_insights`. If you need a new
   invalidation-aware table, write a migration that adds an
   `invalidated_at` column (model: `database/memory/021_insight_lifecycle.sql`)
   AND extend the allow-list in registry.py.
4. Commit. The loader registers the new sentinel at API startup
   (`src/memory/sentinels/config_loader.py`). Re-loading the same
   YAML is idempotent — sentinels are matched on `(name, brand)` and
   skipped if already present.

### Adding via REST API

```bash
curl -X POST https://<host>/api/sentinels \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my new sentinel",
    "pattern_type": "threshold_breach",
    "pattern_config": {
      "table": "ml_predictions",
      "column": "prediction_value",
      "op": "<",
      "value": 0.5
    },
    "action_type": "dispatch_agent",
    "action_config": {"agent_name": "flag_for_review"},
    "brand": "remibrutinib",
    "cooldown_minutes": 60
  }'
```

**Roles** (`src/api/routes/sentinels.py`, verified 2026-09-07): every write
route requires **OPERATOR**; registering `brand="all"` additionally requires
**ADMIN** (a cross-brand sentinel). Other brands must be in the caller's granted
brand list. Reads (`GET`) require an authenticated user, scoped to their brands.

The route set is larger than a POST — enable/disable and delete are REST too:

| Route | Role | Purpose |
| --- | --- | --- |
| `POST /api/sentinels` | OPERATOR (ADMIN for `brand="all"`) | register |
| `GET /api/sentinels` | authenticated | list, brand-scoped |
| `GET /api/sentinels/{id}` | authenticated | fetch one |
| `PATCH /api/sentinels/{id}` | OPERATOR | enable/disable |
| `DELETE /api/sentinels/{id}` | OPERATOR | delete |

`PATCH` is the least invasive way to silence one sentinel without touching the
database directly — prefer it over the `UPDATE` statements in §9 when the API is
reachable.

### Adding a new pattern type (engineering work)

To add a new pattern type (e.g. `latency_spike`):

1. Add to `VALID_PATTERN_TYPES`
   (`VALID_PATTERN_TYPES` in `src/memory/sentinels/registry.py`).
2. Add validation in `_validate_pattern_config`
   (`_validate_pattern_config` in `src/memory/sentinels/registry.py`).
3. Add an `_eval_<pattern_type>` helper.
4. Add a dispatch arm in `evaluate_sentinel`
   (`evaluate_sentinel` in `src/memory/sentinels/registry.py`).
5. Add a DB enum value: `ALTER TYPE sentinel_pattern_type ADD VALUE IF NOT
   EXISTS '<pattern_type>';` — model: migration 024 at
   `database/memory/024_sentinel_invalidation_count_pattern.sql`. Skipping
   this step crashes production at YAML-loader insert time even when CI
   passes (the test suite mocks Supabase).
6. If the new pattern is plan-vocab-shaped, add it to
   `PLAN_TRIGGER_TO_INTERNAL_PATTERN`
   (`PLAN_TRIGGER_TO_INTERNAL_PATTERN` in `src/memory/sentinels/config_loader.py`).
7. Write unit tests against the registry — model:
   `tests/unit/test_memory/test_sentinel*.py` (8 modules) plus
   `tests/unit/test_security/test_sentinel_external_unreachable.py`.

---

## 8. Testing a Sentinel Locally

### Smoke test the YAML

```bash
python -c "
from src.memory.sentinels.config_loader import _load_yaml
config = _load_yaml('config/sentinels.yaml')
print('lifecycle_state:', config.get('lifecycle_state'))
print('sentinel count:', len(config.get('sentinels', [])))
"
```

### Evaluate a single sentinel against real Supabase

```bash
python -c "
import asyncio
from src.memory.sentinels.registry import evaluate_sentinel
from src.memory.services.factories import get_supabase_client

async def main():
    client = get_supabase_client()
    rows = client.table('sentinels').select('*').eq('name', 'High staleness alert').execute().data
    if not rows:
        print('sentinel not found')
        return
    sentinel = rows[0]
    matches = await evaluate_sentinel(sentinel)
    print(f'matches: {len(matches)}')
    for m in matches[:5]:
        print(' ', m)

asyncio.run(main())
"
```

### Trigger one dispatcher pass manually

```bash
python -c "
import asyncio
from src.memory.sentinels.registry import dispatch_sentinels

async def main():
    result = await dispatch_sentinels()
    print(f'examined={result.examined} fired={result.fired} '
          f'actions_taken={result.actions_taken}')
    if result.errors:
        print('errors:')
        for e in result.errors:
            print(' ', e)

asyncio.run(main())
"
```

### Listen to the alert channel

```bash
# Plain redis-cli
redis-cli -h <host> -a "$REDIS_PASSWORD" SUBSCRIBE e2i:alerts

# Or via the SSE bridge (after authenticating)
curl -N \
  -H "Authorization: Bearer $JWT" \
  "http://localhost:8000/api/alerts/stream?brand=remibrutinib"
```

### Publish a synthetic alert (for SSE bridge testing)

```bash
python -c "
import asyncio, json
from src.tasks.sentinel_actions import publish_alert

asyncio.run(publish_alert({
    'type': 'staleness_alert',
    'sentinel_id': 'test-sentinel',
    'brands': ['remibrutinib'],
    'findings': [
        {'finding_id': 'test-1', 'staleness_score': 1.0, 'brand': 'remibrutinib'}
    ]
}))
"
```

### Run the sentinel test suite

There is **no** `tests/unit/test_memory/test_sentinels/` directory — that path
exits 4 (usage error), not 0. The tests are individual modules:

```bash
# 8 sentinel modules under test_memory/ (verified 2026-09-07)
pytest tests/unit/test_memory/test_sentinel*.py -v

# plus the external-unreachable guard, which lives under test_security/
pytest tests/unit/test_security/test_sentinel_external_unreachable.py -v

# both at once
pytest tests/unit/test_memory/test_sentinel*.py \
       tests/unit/test_security/test_sentinel_external_unreachable.py -v
```

---

## 9. Silencing a Sentinel During Incident Response

When a sentinel is noisy during an incident (e.g. it's firing on every
dispatcher tick because of a flapping upstream), use one of these in
order of preference (least → most invasive):

### Option 1 — Bump cooldown (preferred)

> **On the droplet there is no local `psql` socket** — a bare `psql -c` fails.
> Use the same `docker exec` shape the migrations runbook uses (the Postgres
> container is named `supabase-db`; `docker port supabase-db` shows it published
> on host **5433**, not 5432):

```bash
# Set cooldown to 24 hours for one sentinel
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c \
  "UPDATE sentinels SET cooldown_minutes = 1440
   WHERE name = 'High staleness alert';"
```

Effect: sentinel is SKIPPED at the dispatcher's pre-evaluation cooldown
gate (`_is_in_cooldown`, called from `dispatch_sentinels` in
`src/memory/sentinels/registry.py`) until the cooldown
elapses. The dispatcher emits an INFO log line per skip
(`sentinel <id> in cooldown (cooldown_minutes=..., last_fired_at=...);
skipping`) — operators see the silencing in logs without action-handler
side effects.

### Option 2 — Disable a single sentinel

```bash
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c \
  "UPDATE sentinels SET enabled = false
   WHERE name = 'High staleness alert';"
```

Effect: dispatcher skips the sentinel entirely. The dispatcher loop is
unaffected; other sentinels continue to evaluate. Re-enable with
`SET enabled = true`.

### Option 3 — Disable in YAML and redeploy

In `config/sentinels.yaml`:

```yaml
  - id: sentinel_staleness_alert
    name: High staleness alert
    active: false   # <-- flip to false
```

```bash
# Also apply Option 2 to the existing DB row — REQUIRED.
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c \
  "UPDATE sentinels SET enabled = false WHERE name = 'High staleness alert';"
```

Effect: the YAML flag prevents future startup REGISTRATIONS from
re-enabling the row, but the loader does NOT modify already-registered DB
rows on subsequent boots (the existing row would keep firing if `enabled
= true`). Combine with Option 2 to actually stop firing now AND keep the
silence after redeploy.

### Option 4 — Stop the dispatcher entirely (nuclear)

```bash
# Stop the Celery beat / scheduler container. This halts ALL periodic
# tasks (not just sentinel_dispatcher). Disabling `celery control
# disable_events` is unrelated — that only stops event MONITORING, not
# beat scheduling or task execution.
# There is NO compose file at the repo root — always -f the docker/ one, or
# address the container directly.
docker compose -f docker/docker-compose.yml stop scheduler
# equivalently:
docker stop e2i_scheduler
```

Effect: NO sentinels run, and the API restart will not re-enable them
until the scheduler container is started again. Reserved for true
emergencies — `sentinel_dispatcher` shares the beat schedule with many
other periodic tasks (see `docs/ARCHITECTURE.md §3.5 Celery Beat
Schedule`), so this option starves drift monitoring, cache cleanup,
feedback loops, etc. Prefer Option 1 or 2 unless those have been
exhausted.

### After silencing — verify

Tail the API logs and confirm no sentinel-related INFO/WARNING lines
appear for ~10 minutes (2 dispatcher cycles). Then file a tracking issue
with the silencing rationale, who acked, and the un-silence plan.

---

## 10. Troubleshooting

### "Sentinel doesn't fire even though I expect it to"

1. Is it enabled?

   ```sql
   SELECT name, enabled, cooldown_minutes, last_fired_at, fire_count
   FROM sentinels WHERE name = '<name>';
   ```

2. Is it in cooldown?

   ```sql
   SELECT name, last_fired_at, cooldown_minutes,
          NOW() - last_fired_at AS elapsed
   FROM sentinels WHERE name = '<name>';
   ```

   If `elapsed < cooldown_minutes`, the sentinel is intentionally
   skipping. The dispatcher checks `_is_in_cooldown` BEFORE
   `evaluate_sentinel` (both in `dispatch_sentinels`,
   `src/memory/sentinels/registry.py`),
   so cooled-down rows do not even reach evaluation. Logs at INFO level
   confirm: `sentinel <id> in cooldown (cooldown_minutes=...,
   last_fired_at=...); skipping`
   (the cooldown-skip branch of `dispatch_sentinels`).

3. Did `evaluate_sentinel` actually return matches?

   Use the **§8 Evaluate a single sentinel** snippet above.

4. Is the dispatcher beat task running?

   `celery inspect scheduled` shows tasks a worker holds with an **ETA/countdown**
   — it does **not** list beat's periodic schedule, so `sentinel_dispatcher` will
   normally be absent from it even when beat is healthy. Look at what the
   scheduler actually emitted instead:

   ```bash
   docker logs e2i_scheduler --since 10m | grep sentinel_dispatcher
   ```

   > **UNVERIFIED (2026-09-07):** the `celery inspect scheduled` semantics above
   > are reasoned from Celery's documented behaviour, and the replacement command
   > was not executed — this doc pass was read-only. Treat the `docker logs` line
   > as the intended check and confirm it on first use.

### "Alert isn't reaching the CopilotKit dashboard"

1. Is Redis receiving the publish?

   ```bash
   redis-cli -h <host> -a "$REDIS_PASSWORD" SUBSCRIBE e2i:alerts
   # In another shell, force a sentinel via §8
   ```

2. Is the SSE bridge subscribing?

   Check API logs for `staleness-alerts bridge: subscriber loop failed`
   or similar. The bridge re-raises exceptions onto
   `task.exception()` — observable via test hooks but logs are the
   operational signal.

3. Does the publisher include the brand in `brands`?

   The bridge filters on `payload['brands']` — payloads without a
   matching brand are dropped by `_matches_brand`
   (`AlertBridge._matches_brand` in `src/api/routes/staleness_alerts.py`). A broadcast-without-brands
   is treated as out-of-scope.

4. Is the connection at queue depth 100?

   Check logs for `backpressure dropping oldest`
   (`AlertBridge._enqueue_with_backpressure` in `src/api/routes/staleness_alerts.py`). The connection stays
   open but drops oldest events.

### "Sentinel registration crashes at API startup"

Check the loader logs for `SentinelConfigLoadError`. Likely causes:

1. **Invalid YAML**: lint with `python -c "import yaml; yaml.safe_load(open('config/sentinels.yaml'))"`.
2. **Unknown trigger_type**: must be one of the four plan-vocab triggers.
3. **Unknown action**: must be one of the four plan-action names.
4. **Missing `lifecycle_state` key**: lifecycle guard rejects the config.
5. **`staleness_threshold` targeting a table without `invalidated_at`**:
   the `INVALIDATION_AWARE_TABLES` allow-list rejects with `ValueError`.
6. **DB enum out of sync**: if you added a new pattern type to Python but
   forgot the `ALTER TYPE ADD VALUE` migration, the insert fails with
   a Postgres enum violation. Model: migration 024 at
   `database/memory/024_sentinel_invalidation_count_pattern.sql`.

### "Cooldown not honored even when I set it"

Are you setting `cooldown_minutes = False` (or `True`)? Python's
`isinstance(False, int)` is `True`. The registration guard catches this
(`register_sentinel` in `src/memory/sentinels/registry.py`), but a direct DB write or
PostgREST call can smuggle a bool in. `_is_in_cooldown` treats both
booleans as "no cooldown" since the guarantee is no longer trustworthy.

Fix: `UPDATE sentinels SET cooldown_minutes = <int> WHERE ...`.

---

## References

- Plan: `.claude/plans/archive/e2i_memory_subsystems_implementation_plan_archived_20260520.md`
  (Phase 3) — archived; local, untracked planning note, historical
- Registry: `src/memory/sentinels/registry.py`
- YAML loader: `src/memory/sentinels/config_loader.py`
- Action handlers: `src/tasks/sentinel_actions.py`
- SSE bridge: `src/api/routes/staleness_alerts.py`
- Lifecycle enum: `src/lifecycle/gate_lifecycle.py`
- Migrations:
  - `database/memory/021_insight_lifecycle.sql` — `invalidated_at` columns
  - `database/memory/023_sentinel_cooldown.sql` — `cooldown_minutes` column
  - `database/memory/024_sentinel_invalidation_count_pattern.sql` — enum extension
