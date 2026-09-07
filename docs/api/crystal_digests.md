# Crystal Digests — CopilotKit API Reference

**Version**: 1.1 | **Last Updated**: 2026-09-07 | **Status**: Living Document

Owner: Platform / Memory Subsystems

Companion plan: `.claude/plans/archive/e2i_memory_subsystems_implementation_plan_archived_20260520.md`
(Phase 4) — a local, untracked planning note, archived 2026-05-20 and kept for
historical context only.

Source citations in this file name **symbols**, not line numbers: line numbers
drift silently and mislead the next reader. Where a number is unavoidable,
regenerate it with `grep -n` at write time.

---

## Table of Contents

1. [What is a Crystal Digest?](#1-what-is-a-crystal-digest)
2. [CrystalDigest Schema (15 fields)](#2-crystaldigest-schema-15-fields)
3. [`LLMCrystalNarrativeAudit` Trail](#3-llmcrystalnarrativeaudit-trail)
4. [Endpoint Reference](#4-endpoint-reference)
5. [Authentication](#5-authentication)
6. [Sample Requests + Responses](#6-sample-requests--responses)
7. [Real-time Updates (SSE)](#7-real-time-updates-sse)
8. [Lifecycle + Invalidation](#8-lifecycle--invalidation)

---

## 1. What is a Crystal Digest?

A **crystal digest** is a single durable `executive_insights` row that
aggregates 2+ related episodic memories (from different agents, same
brand, within a 7-day window, on the same `causal_path` or KPI) into a
cross-agent narrative for executive consumption.

The crystallizer (`Crystallizer` in
`src/memory/crystallization/crystallizer.py`) is **brand-strict**: it NEVER
co-aggregates across brands. Aggregation keys are
`(brand, region, kpi, time_window)`.

### Provenance filtering of the source memories

The episodic page query is passed through `apply_provenance_filter()`
(`src/repositories/provenance.py`) before paging, so **synthetic episodic
memories are excluded from real crystals** — the filter appends
`.eq('is_synthetic', False)` (#850/#977).

The one exception is a synthetic-gold showcase instance: when
`E2I_INCLUDE_SYNTHETIC` is set (`deployment_includes_synthetic()`) the
predicate is skipped for every reader, so the synthetic corpus crystallizes and
findings populate. This is reversible — unset the env and the strict gate
returns verbatim.

Note that `executive_insights` is itself **not** a provenance-tagged table
(`PROVENANCE_TAGGED_TABLES` in the same module); the filter applies to the
episodic memories a crystal is derived *from*, not to the crystal row.

### Schema shipped in two phases

- **Phase 1 (PR #250)**: 13 base fields (`insight_id`, `title`,
  `narrative`, `brand`, `region`, `kpi`, time window, source count, etc.).
- **Phase 4 (PR #384)**: 15 additional CrystalDigest fields per
  Decision 2 = HYBRID — 13 deterministic + 2 LLM-narrative prose
  fields. Migration: `database/memory/025_crystaldigest_schema_completion.sql`.

Per **Decision 3 = KEEP BINARY** (plan §"DECISIONS ADOPTED" 2026-05-19),
the `staleness_score` field is intentionally OMITTED. Staleness remains
boolean via `invalidated_at IS NULL` — see [§8](#8-lifecycle--invalidation).

---

## 2. CrystalDigest Schema (15 fields)

The 15 fields added by migration 025 split into three groups:

### Analytical (8 fields)

Source: `_derive_crystal_digest_fields()` in
`src/memory/crystallization/crystallizer.py`. All deterministically derived from
the source episodic memories' `raw_content` JSONB.

| Field                          | Type        | Source                                                                                |
|--------------------------------|-------------|---------------------------------------------------------------------------------------|
| `effect_size`                  | `float?`    | `raw_content.ate_estimate` — numeric ATE point estimate.                              |
| `effect_ci_lower`              | `float?`    | `raw_content.confidence_interval[0]` — 95% CI lower bound.                            |
| `effect_ci_upper`              | `float?`    | `raw_content.confidence_interval[1]` — 95% CI upper bound.                            |
| `effect_direction`             | `string?`   | Derived from sign + CI: `"positive"` / `"negative"` / `"null"`.                       |
| `cohort_size`                  | `int?`      | `raw_content.sample_size` — subject count.                                            |
| `confounders_controlled`       | `string[]`  | Union of `raw_content.confounders` across all members (sorted).                       |
| `sensitivity_checks_passed`    | `string[]`  | Union of `raw_content.refutation_passed_tests` (sorted).                              |
| `sensitivity_checks_failed`    | `string[]`  | Union of `raw_content.refutation_failed_tests` (sorted).                              |

#### `effect_direction` derivation

Logic in `_derive_crystal_digest_fields()`:

- If `effect_size`, `ci_lower`, `ci_upper` are all set AND `ci_lower <= 0 <= ci_upper`: `"null"`
- Else if `effect_size > 0`: `"positive"`
- Else if `effect_size < 0`: `"negative"`
- Else if `effect_size == 0`: `"null"`
- Fallback (no CI): sign-only direction from `effect_size`.

#### Why arrays are sorted

`confounders_controlled` and the two `sensitivity_checks_*` arrays are
sorted before the return in `_derive_crystal_digest_fields()` so JSONB diffs
are minimal and the row hash is reproducible across re-crystallization
passes. Source memories have no stable secondary ordering — without the
sort, encounter-order would leak into the response.

### Narrative-prose (2 fields)

| Field                       | Type      | Source                                              |
|-----------------------------|-----------|-----------------------------------------------------|
| `limitations`               | `string?` | LLM-generated (Haiku) OR deterministic heuristic.   |
| `recommended_next_analysis` | `string?` | LLM-generated (Haiku) OR deterministic heuristic.   |

Both fields are truncated to 500 chars at the audit boundary.

The LLM path is gated by the feature flag
`E2I_CRYSTAL_LLM_NARRATIVES_ENABLED`
(`LLM_NARRATIVE_ENV_VAR`, read by `_llm_narratives_enabled()` in
`src/memory/crystallization/crystallizer.py`). When the flag is off, the
deterministic heuristic (`_deterministic_narrative_prose()` in the same module)
emits short non-empty prose so dashboards don't show blank cells. Flag accepts
any of `{"1", "true", "yes", "on"}` (case-insensitive).

### Lineage (5 fields)

| Field                       | Type        | Source                                                                                   |
|-----------------------------|-------------|------------------------------------------------------------------------------------------|
| `provenance_chain_id`       | `string?`   | First 32 chars of SHA-256 hash of `causal_paths + member_ids` (sorted).                  |
| `provenance_depth`          | `int?`      | BFS hop count: 2 if any causal_path is wired, else 1.                                    |
| `consolidation_tier`        | `string?`   | `working` / `episodic` / `semantic` / `procedural` — highest tier among sources.         |
| `replication_count`         | `int?`      | Count of source episodic memories (= `source_count`).                                    |
| `data_version`              | `string?`   | First non-empty `raw_content.data_version` across sources.                               |

#### Tier inheritance

If any source memory carries a higher tier than `episodic`, the crystal
inherits the highest. Tier rank in `_derive_crystal_digest_fields()`:

```python
tier_rank = {"working": 0, "episodic": 1, "semantic": 2, "procedural": 3}
```

---

## 3. `LLMCrystalNarrativeAudit` Trail

When the LLM narrator path is enabled, the crystallizer emits an audit
sidecar to track the model call. Source:
`LLMCrystalNarrativeAudit` in `src/data/kg/types.py`.

### Schema

| Field                       | Type        | Description                                                                                  |
|-----------------------------|-------------|----------------------------------------------------------------------------------------------|
| `narrator_model`            | `string`    | Pinned model id. Default: `"claude-haiku-4-5-20251001"`.                                     |
| `key_finding`               | `string`    | LLM-generated 1-2 sentence headline. Truncated to 500 chars. Overrides title if non-empty.   |
| `limitations`               | `string`    | LLM-generated 1-2 sentence enumeration of limitations. Truncated to 500 chars.               |
| `recommended_next_analysis` | `string`    | LLM-generated 1-2 sentence follow-up guidance. Truncated to 500 chars.                       |
| `latency_ms`                | `float?`    | Wall-clock duration in milliseconds. `None` on flag-off or exception path.                   |
| `input_tokens`              | `int?`      | Prompt tokens consumed, from Anthropic SDK `response.usage`. `None` if not captured.         |
| `output_tokens`             | `int?`      | Completion tokens emitted. Same nullability as `input_tokens`.                               |
| `cost_usd`                  | `float?`    | Computed via `compute_haiku_cost_usd`. `None` when token counts could not be extracted.      |
| `input_prompt`              | `string`    | The **full prompt text sent to the LLM** (#391 security box 4), populated from `_build_narrator_prompt()`. Empty string on the flag-off / exception / legacy-caller paths. |

### Audit persistence

The audit struct **is** persisted, as one row per crystal in the
`crystal_narrative_audits` table (`database/memory/028_provenance_append_only.sql`).
Its prose fields also flow into the `executive_insights` row directly, so the
audit row is a sidecar rather than the only copy.

| Audit column | Notes |
|---|---|
| `audit_id` | UUID primary key |
| `insight_id` | FK → `executive_insights(insight_id)`, `ON DELETE CASCADE`, `UNIQUE` — so the write is idempotent across retries and orphans cannot accumulate |
| `narrator_model` | Pinned model id |
| `key_finding`, `limitations`, `recommended_next` | Audit copy of the prose columns (note the column is `recommended_next`, while the dataclass field is `recommended_next_analysis`) |
| `input_prompt` | `NOT NULL DEFAULT ''`. The empty string is itself a meaningful signal — "we did NOT send anything" |
| `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd` | Telemetry mirror |
| `created_at` | Insert timestamp |

**Why the prompt is captured**: the offline PHI scanner
(`scripts/audit_phi_in_crystal_narratives.py`) joins this table so it can audit
the LLM's **inputs** as well as its outputs for PHI/PII leaks. Without the row,
every real crystal lands with a NULL `audit_input_prompt` and only the outputs
(`key_finding`, `narrative`) are auditable.

**The write is best-effort, not crystallization-gating.** A failure is caught
narrowly, logged at WARNING, and the crystallization continues — losing one
audit sidecar is preferable to failing the whole pipeline. Persistent failures
are detectable with a left-join count of `executive_insights` rows that have no
`crystal_narrative_audits` row.

The audit struct also enforces the **single-writer** contract: narrator
emissions go through it, never directly to the row insert.

### Narrator dependency-injection

The narrator function `_invoke_llm_narrator` accepts an optional
`client_factory: Optional[Callable[[str], _AnthropicClientProtocol]]`
parameter
(`_AnthropicClientFactory` / `_invoke_llm_narrator()` in
`src/memory/crystallization/crystallizer.py`). This is the
DI hook for tests — production passes nothing and the default factory
constructs `anthropic.AsyncAnthropic(api_key=...)`.

The async SDK is required (`AsyncAnthropic`, not the sync
`anthropic.Anthropic`) because the crystallizer is async-end-to-end; a
sync client would block the event loop and stall the FastAPI / Celery
worker pool.

### Narrow exception handling

SDK-level transient errors fall back to an empty audit (the row insert
still completes with empty prose). Narrow catch tuple in
`_invoke_llm_narrator()`:

- `APIConnectionError`
- `APITimeoutError`
- `RateLimitError`
- `APIStatusError`

Programming errors (`TypeError`, `AttributeError`, `KeyError`) MUST
propagate so they surface in CI / DLQ instead of being silently swallowed.

### Key-shape check

`_invoke_llm_narrator()` short-circuits if `ANTHROPIC_API_KEY` does NOT
start with `"sk-ant-"`. Catches the CI
placeholder `test-key` so live-LM smoke tests don't 401 against the
Anthropic API.

---

## 4. Endpoint Reference

All endpoints live under the `/api/executive-insights` prefix
(`router` in `src/api/routes/executive_insights.py`).

### `GET /api/executive-insights`

List crystallized insights, brand-filtered.

**Auth**: AUTH (`Depends(require_auth)`)

**Query params**:

| Param              | Type       | Required | Default | Description                                              |
|--------------------|------------|----------|---------|----------------------------------------------------------|
| `brand`            | `string`   | no       | none    | Filter by brand. STRONGLY RECOMMENDED.                   |
| `region`           | `string`   | no       | none    | Filter by region.                                        |
| `include_recalled` | `bool`     | no       | `false` | Include recalled (overturned) insights.                  |
| `limit`            | `int`      | no       | `50`    | Max rows returned.                                       |

**Response**: `200 OK` with `List[ExecutiveInsightResponse]`. See
[§6](#6-sample-requests--responses) for the shape.

### `GET /api/executive-insights/portfolio-summary`

Per-brand aggregation of crystallized insights (#376 §D).

Aggregates across all non-recalled, non-invalidated crystals:

- count of insights per brand
- latest `crystallized_at` per brand
- average `effect_size` per brand (excluding NULL effect_size rows)

**Auth**: AUTH (`Depends(require_auth)`)

**Query params**: none.

**Response**: `200 OK` with `PortfolioSummaryResponse`:

```typescript
{
  by_brand: PortfolioBrandSummary[];
  total_brands: number;
  total_insights: number;
}

interface PortfolioBrandSummary {
  brand: string;
  insight_count: number;
  latest_crystallized_at: string | null;  // ISO 8601
  average_effect_size: number | null;
  effect_size_sample_count: number;       // denominator of the mean
}
```

> **Important**: this endpoint is declared BEFORE `/{insight_id}` in
> the route order (`get_portfolio_summary()` before `get_insight()` in
> `src/api/routes/executive_insights.py`),
> otherwise FastAPI matches `portfolio-summary` as an `insight_id`.

### `GET /api/executive-insights/{insight_id}`

Get one insight by ID.

**Auth**: AUTH (`Depends(require_auth)`)

**Path params**:

| Param        | Type     | Description                              |
|--------------|----------|------------------------------------------|
| `insight_id` | `string` | The crystal's `insight_id` (UUID).       |

**Response**:

- `200 OK` with `ExecutiveInsightResponse`.
- `404 Not Found` if no insight matches the ID.
- `410 Gone` if any provenance ancestor has been overturned/invalidated
  (returned by `InsightVerifierMiddleware`, not the route handler).

### `POST /api/executive-insights/crystallize`

Manually trigger crystallization for a brand/region pair.

**Auth**: OPERATOR (`Depends(require_operator)`)

**Body** (`CrystallizeRequest`):

```json
{
  "brand": "remibrutinib",
  "region": "northeast"
}
```

`region` is optional.

**Response**: `202 Accepted` with `CrystallizeResponse`:

```json
{
  "examined_groups": 3,
  "insights_created": 1,
  "edges_created": 5
}
```

### `GET /api/alerts/stream`

See [§7](#7-real-time-updates-sse).

---

## 5. Authentication

All read endpoints require any authenticated role (`Depends(require_auth)`).
The crystallization trigger requires the OPERATOR role
(`Depends(require_operator)`).

JWT format: Bearer token issued by Supabase Auth. The middleware stack
validates against Supabase in `JWTAuthMiddleware`
(`src/api/middleware/auth_middleware.py` — there is no `jwt_auth.py`). See
`docs/ARCHITECTURE.md §6.2` for the full JWT flow.

RBAC roles (in increasing privilege):

1. `VIEWER` (level 1) — dashboards, KPIs.
2. `ANALYST` (level 2) — causal inference, gap analysis. Sufficient for `require_auth`.
3. `OPERATOR` (level 3) — experiments, manual crystallization. Required for `POST /crystallize`.
4. `ADMIN` (level 4) — full system access.

---

## 6. Sample Requests + Responses

### List insights for one brand

```bash
curl -X GET 'https://eznomics.site/api/executive-insights?brand=remibrutinib&limit=10' \
  -H "Authorization: Bearer $JWT"
```

**Response** (truncated):

```json
[
  {
    "insight_id": "abc-123-def",
    "title": "remibrutinib in northeast: cross-agent finding on causal path xyz-789",
    "narrative": "Cross-agent crystallized insight for remibrutinib in northeast.\n...",
    "brand": "remibrutinib",
    "region": "northeast",
    "kpi": "trx_conversion_rate",
    "time_window_start": "2026-05-13T00:00:00+00:00",
    "time_window_end": "2026-05-19T23:59:59+00:00",
    "key_metrics": {
      "source_count": 3,
      "distinct_agents": 2,
      "agents": ["causal_impact", "gap_analyzer"],
      "causal_path_id": "xyz-789"
    },
    "recall": false,
    "recall_reason": null,
    "crystallized_at": "2026-05-20T13:42:11+00:00",
    "source_count": 3,

    "effect_size": 0.128,
    "effect_ci_lower": 0.041,
    "effect_ci_upper": 0.215,
    "effect_direction": "positive",
    "cohort_size": 1247,
    "confounders_controlled": ["age", "disease_severity", "prior_lines_of_therapy"],
    "sensitivity_checks_passed": ["placebo_test", "random_common_cause"],
    "sensitivity_checks_failed": [],

    "limitations": "small cohort (n=1247); standard limitations apply (deterministic heuristic).",
    "recommended_next_analysis": "replicate on an independent cohort to confirm generalizability.",

    "provenance_chain_id": "8a7c92e1b3f0d4a6c5e8b9f2a1c3d4e5",
    "provenance_depth": 2,
    "consolidation_tier": "semantic",
    "replication_count": 3,
    "data_version": "optum-v3.2-20260501"
  }
]
```

### Portfolio summary

```bash
curl -X GET 'https://eznomics.site/api/executive-insights/portfolio-summary' \
  -H "Authorization: Bearer $JWT"
```

**Response**:

```json
{
  "by_brand": [
    {
      "brand": "fabhalta",
      "insight_count": 4,
      "latest_crystallized_at": "2026-05-19T08:11:42+00:00",
      "average_effect_size": 0.087,
      "effect_size_sample_count": 4
    },
    {
      "brand": "kisqali",
      "insight_count": 7,
      "latest_crystallized_at": "2026-05-20T12:03:18+00:00",
      "average_effect_size": 0.211,
      "effect_size_sample_count": 6
    },
    {
      "brand": "remibrutinib",
      "insight_count": 12,
      "latest_crystallized_at": "2026-05-20T13:42:11+00:00",
      "average_effect_size": 0.128,
      "effect_size_sample_count": 12
    }
  ],
  "total_brands": 3,
  "total_insights": 23
}
```

Note `effect_size_sample_count` differing from `insight_count` for
`kisqali` — one insight has `effect_size IS NULL` and is excluded from
the mean.

### Manually trigger crystallization

```bash
curl -X POST 'https://eznomics.site/api/executive-insights/crystallize' \
  -H "Authorization: Bearer $JWT_OPERATOR" \
  -H "Content-Type: application/json" \
  -d '{"brand": "remibrutinib", "region": "northeast"}'
```

**Response** (`202 Accepted`):

```json
{
  "examined_groups": 5,
  "insights_created": 2,
  "edges_created": 11
}
```

---

## 7. Real-time Updates (SSE)

Crystallized insights can be invalidated by the staleness cascade
(`src/memory/lifecycle/invalidator.py`). When that happens, the
`sentinel_staleness_alert` (its `id:` entry in `config/sentinels.yaml`) fires and
publishes a `staleness_alert` payload to `e2i:alerts` Redis pub/sub.

CopilotKit dashboards subscribe via the SSE bridge at
`alerts_stream()` in `src/api/routes/staleness_alerts.py`:

### `GET /api/alerts/stream`

**Auth**: AUTH (`Depends(require_auth)`)

**Query params**:

| Param   | Type     | Required | Description                                                       |
|---------|----------|----------|-------------------------------------------------------------------|
| `brand` | `string` | yes      | Single-brand subscription (1-64 chars). Multi-brand is V2 scope.  |

**Response**: `text/event-stream` (SSE) that stays open until client
disconnects. Each event:

```
event: alert
data: {"type": "staleness_alert", "sentinel_id": "...", "brands": ["<brand>"], "findings": [...]}
```

`sse_starlette` emits a `: ping` comment line every 15 seconds to keep
intermediaries from closing idle connections. Clients should ignore
comment lines (per the SSE spec).

### Filter semantics

The bridge filters by the connection's requested `brand` value
(`AlertBridge._matches_brand()` in `src/api/routes/staleness_alerts.py`):

- `payload['brands']` is a list — match if `self.brand` appears in it.
- `"all"` in `brands` also matches every brand subscriber (mirrors
  invalidator convention).
- Missing or empty `brands` does NOT match any single-brand subscriber
  (broadcasts must populate the field).

### Backpressure

Per-connection bounded queue caps at 100 events
(`MAX_QUEUE_DEPTH`, applied by `AlertBridge._enqueue_with_backpressure()`).
At capacity: drop-oldest with
a WARNING log throttled to every 25th drop. The connection stays open.

### No replay

Subscribers that connect AFTER an alert was published do NOT receive it
(Redis pub/sub, not Streams). If a client drops mid-stream and
reconnects, the gap is permanent.

### Example client (JavaScript)

```javascript
const evtSource = new EventSource(
  `/api/alerts/stream?brand=${encodeURIComponent('remibrutinib')}`,
  { withCredentials: true }
);

evtSource.addEventListener('alert', (event) => {
  const payload = JSON.parse(event.data);
  if (payload.type === 'staleness_alert') {
    // Drop affected crystals from cache; trigger re-fetch
    payload.findings.forEach((f) => {
      crystalCache.invalidate(f.finding_id);
    });
  }
});

evtSource.onerror = (err) => {
  console.warn('SSE connection lost; browser will auto-reconnect', err);
};
```

---

## 8. Lifecycle + Invalidation

### Insight states

| State                    | Backing field                                | Behavior                                                                |
|--------------------------|----------------------------------------------|-------------------------------------------------------------------------|
| Active                   | `recall = false` AND `invalidated_at IS NULL`| Surfaced in list + summary endpoints.                                   |
| Recalled                 | `recall = true`                              | Hidden by default (`include_recalled=false`). Set when override.        |
| Invalidated              | `invalidated_at IS NOT NULL`                 | Surfaced in list; EXCLUDED from `portfolio-summary` aggregations.       |
| Overturned (ancestor)    | Detected via `verify_insight_chain` RPC      | `GET /{id}` returns `410 Gone` via `InsightVerifierMiddleware`.         |

### `invalidated_at` column

Added by migration 021 to `executive_insights`, `triggers`, and
`ml_predictions` (`database/memory/021_insight_lifecycle.sql`).
The portfolio summary endpoint explicitly excludes invalidated rows
(in `get_portfolio_summary()`):

```python
.is_("invalidated_at", "null")
```

### Cascade-invalidation flow

Sentinels DETECT staleness; the invalidator WRITES it. The sentinel fires
AFTER rows have already been marked `invalidated_at IS NOT NULL` upstream
by `src/memory/lifecycle/invalidator.py::cascade_invalidate`, which is
itself triggered by ancestor-overturn events (e.g. a `causal_path` is
recalled, the cascade walks the `insight_edges` DAG, stamping
`invalidated_at` on downstream artifacts).

Flow:

1. Upstream cascade (`src/memory/lifecycle/invalidator.py`) sets
   `invalidated_at` on affected `executive_insights` rows when an
   ancestor is overturned.
2. The next dispatcher tick runs `sentinel_staleness_alert` (cooldown
   permitting): the `invalidation_count` pattern enumerates rows where
   `invalidated_at IS NOT NULL`
   (`evaluate_sentinel()` in `src/memory/sentinels/registry.py`).
3. `notify_and_queue_reanalysis`
   (`src/tasks/sentinel_actions.py`) publishes a
   `staleness_alert` payload (full findings list) to `e2i:alerts`.
4. The handler ALSO enqueues up to 5 `reanalyze_finding` Celery tasks
   (`_REANALYSIS_CAP = 5` in `src/tasks/sentinel_actions.py`); each
   task publishes a `reanalysis_requested` event on
   `reanalysis:e2i:{brand}` for downstream orchestrator consumers
   (`reanalyze_finding()` in `src/tasks/insight_lifecycle_tasks.py`).
5. Any client subscribed to `GET /api/alerts/stream?brand=<brand>`
   receives the `staleness_alert` event via the SSE bridge and can
   invalidate its local cache for affected `insight_id`s. (Frontend
   consumer wiring is TBD — the SSE bridge is shipped and ready;
   see the JavaScript snippet in §7.)

### Decision 3 = KEEP BINARY

Staleness is boolean — `invalidated_at IS NULL` means "fresh", anything
else means "stale". There is no graded `staleness_score` column on
`executive_insights`. The `sentinel_staleness_alert` enumerates rows
binary-staleness-style and reports `staleness_score=1.0` per match (this
is a payload synthetic, not a column read).

Reversal cost is ~800-1,200 LoC if a future workflow surfaces a graded
need (4 migrations + helper change + test rewrites). See issue #376 for
the full reversal checklist.

---

## References

- Crystallizer source: `src/memory/crystallization/crystallizer.py`
- Route source: `src/api/routes/executive_insights.py`
- SSE bridge: `src/api/routes/staleness_alerts.py`
- Audit type: `LLMCrystalNarrativeAudit` in `src/data/kg/types.py`
- Migrations:
  - `database/memory/021_insight_lifecycle.sql` — base 13 fields + `invalidated_at`
  - `database/memory/025_crystaldigest_schema_completion.sql` — 15 new fields
  - `database/memory/028_provenance_append_only.sql` — `crystal_narrative_audits`
    table (one row per crystal, incl. `input_prompt` for the PHI scanner)
- Plan: `.claude/plans/archive/e2i_memory_subsystems_implementation_plan_archived_20260520.md`
  (Phase 4; local untracked planning note, archived 2026-05-20)
- Related ADRs: `docs/ARCHITECTURE.md §8 ADR-003 (Tri-Memory Architecture)`
