# Tool-composer learning loop: registry sync, composition records, measured reliability

**Date:** 2026-09-11
**Status:** design for the dispatcher's claim check; the owner decided on 2026-09-11 to build the loop
and to replace the migration-per-schema-change regime from PR #2012 with a startup sync
**Branch:** `claude/tool-composer-learning-loop` (worktree `.worktrees/lane-learning-loop`, off `56f8b8589`)
**Evidence:** live reads on the droplet database (`docker exec supabase-db psql`), taken 2026-09-11.
Every DDL/DML rehearsal ran inside `BEGIN … ROLLBACK`. The rehearsal scripts (`rehearse1.sql` to
`rehearse4.sql`, `draft_039.sql`) are in the session scratchpad; their results are copied below.

---

## 1. What the loop was designed to do

`database/ml/013_tool_composer_tables.sql` (header "Version 4.2.0, 2024-12-17") arrived in the first
platform commit `3e1c70cf4` (2025-12-20), together with design documents that were later untracked
(`889ff0f02`, "remove docs/ from git tracking"). Those documents are still readable with `git show 3e1c70cf4:<path>`.

| Source | What it says the loop does |
|---|---|
| ml/013 L12–13, L299 | `composer_episodes`: "Episodic memory for tool compositions — enables learning from experience"; `tool_performance`: "Tool execution metrics for learning" |
| ml/013 L456, L485 | `v_tool_reliability`: "Tool reliability metrics for planner optimization … for planner decisions" |
| ml/013 L981 | `update_tool_registry_metrics`: "Aggregate tool_performance into tool_registry baselines" |
| ml/013 L1010, L1044 | `find_similar_compositions`: "Finds similar successful compositions for plan optimization using vector similarity" |
| ml/013 L1141–1197 | `trg_log_step_performance`: log a `tool_performance` row when a step completes or fails |
| `docs/v4.2_IMPLEMENTATION_TODO.md` @3e1c70cf4 | "Database migrations complete, application integration pending". Planner: "Estimate total latency from `v_tool_reliability`", "Save plan to `composer_episodes.tool_plan`". Executor: "Initialize `composition_steps` for each tool call … trigger auto-logs to `tool_performance`". API: "POST /api/v1/compositions/{id}/feedback … Update `composer_episodes.success`, `user_rating`, `feedback_text`" |
| `docs/tool_composer_component_update_list.md` @3e1c70cf4 | `GET /api/v1/composer/tools/{name}/performance # Tool metrics`, `GET /api/v1/composer/active`, `GET /api/v1/classifier/accuracy`; a feedback_learner `CompositionCompleteEvent` |
| `docs/tool_composer_architecture.html` @3e1c70cf4 | "Tool Composer stores successful compositions in episodic memory and uses them to optimize future planning" |

**Why it was never connected (verified).** The Python work was built against a different specialist
spec, `.claude/specialists/tool-composer.md` @3e1c70cf4. That spec uses other table names, and for
memory says only "Episodic Memory: Read-only if needed".

The audit commit `fee8bc3e0` (2025-12-28) then closed the "learning" gaps with other stores:
- G1/G2 reuse went to the shared `episodic_memories` store (`memory_hooks.find_similar_compositions`).
- G8 became an in-memory latency EMA (`executor.update_tool_performance`).
- G3 became `register_from_database` / `sync_to_database`, which only mock tests call.

No commit ever wrote to `composer_episodes`, `composition_steps` or `tool_performance`. Nothing was
removed either, because nothing was ever wired. Two of the 013 objects could not be created as
originally written until `57bc91dd8` (2026-06-03) fixed them.

The classification half of the design is the exception: #1341 closed it. `classification_logs` is
written per classified turn, labelled nightly (`src/tasks/routing_label_tasks.py`), and read by
`v_classification_accuracy`. This design leaves that half alone.

## 2. What exists today (measured 2026-09-11)

### 2.1 Stores and volumes

| Store | Rows | Notes |
|---|---|---|
| `tool_registry` | 16 | 13 seeded by ml/013, 3 by ml/027. `success_rate` is 0.95 on 14 rows and 0.98 / 0.92 on the others; `avg_latency_ms` is 500 on 14 rows. These are seed defaults, not measurements. The code declares 500–5000 ms. |
| `tool_dependencies` | 11 | Corrected by ml/037 (PR #2012) |
| `tool_performance`, `composer_episodes`, `composition_steps` | 0 / 0 / 0 | No writer anywhere in `src/` or `scripts/` |
| `classification_logs` | 384 | TOOL_COMPOSER 11; 373 labelled. Written and read by the #1341 loop. |
| `audit_chain_entries`, agent `tool_composer` | workflow_start 23, decompose 22, plan 19, execute 7, synthesize 7 | 2026-07-29 → 2026-08-16. Mean durations: decompose 13.2 s, plan 5.3 s, execute 3.4 s, synthesize 26.8 s. 12 of 19 planned runs never reached `execute`, and no row says why: the `*_error` marker only arrived 2026-09-06. |
| `episodic_memories`, `event_type='composition_completed'` | 2 | Both `partial_success`. Tool sequences repeat `gap_calculator` 2–3×. |
| `procedural_memories`, `procedure_type='tool_composition'` | 6 | All from 2026-04-27 |
| `chatbot_analytics` | 942 turns | `tool_composer_tool` invoked in 2. `tool_composer_used` is never true. |
| `chatbot_message_feedback` | 3 | Last on 2026-07-07 |

**Consequence for every design choice below: production volume is about 20 compositions in six
weeks, and zero since 2026-08-16.** Per-tool samples are in single digits.

### 2.2 The runtime path

- **Entry points.**
  - Chat: `src/api/routes/chatbot_tools.py:1811` calls `compose_query()`, which builds a **new**
    `ToolComposer` per call.
  - Orchestrator: `ToolComposerAgent._get_composer` (`agent.py:359`) caches one per agent instance.
- **Phases.** `composer.py:243` `compose()` runs decompose → plan (`_build_kpi_causal_plan` for KPI
  questions, else `ToolPlanner.plan`) → execute (`PlanExecutor.execute`) → F6 fail-closed gate →
  synthesize → `_contribute_to_memory`.
- **Already recorded:**
  - Audit-chain phase rows (`composer.py:271–293, 875–909`).
  - An episodic `composition_completed` row with tool sequence, counts and confidence
    (`memory_hooks.py:332–444`), synthesized path only.
  - Procedural patterns for runs with confidence ≥ 0.7.
  - Redis working-memory cache.
  - Opik traces (`opik_tracer.py`; Opik is intentionally off on the droplet).
- **In-memory feedback that exists:**
  - `ToolFailureTracker` in `executor.py:241–365` keeps a per-tool EMA latency, success window and
    circuit breaker. It is created per `PlanExecutor`, and on the chat path that means **per request**.
    It never outlives a composition there and is never shared across the two gunicorn workers.
  - `update_tool_performance` (`executor.py:1312`) has only test callers.
  - The G6 plan and tool-output cache is a process singleton (`cache.py:428–435`).
  - The planner shows `schema.avg_execution_ms`, the declared number, in its prompt (`planner.py:390`),
    and the DSPy formatter does the same (`dspy_integration.py:225`). `_estimate_duration`
    (`planner.py:955`) sums the same declared numbers.
- **Similar-composition reuse is live** through `episodic_memories`:
  - Path: `planner._check_episodic_memory` → `memory_hooks.find_similar_compositions` →
    `search_episodic_by_text` (1536-dim embeddings, `min_similarity=0.7`, hydrated `raw_content`, #889).
  - It keeps rows with `raw_content.success`, which is true for PARTIAL runs.
  - A reference therefore recommends a sequence whose failed tools are not named: the two live rows
    both repeated `gap_calculator`.

### 2.3 Database facts that constrain the design

- **Grants.**
  - `anon` and `authenticated` have **no** privileges on any of the six tables. `SET ROLE anon;
    SELECT … FROM tool_registry` gives "permission denied".
  - `service_role` has full DML.
  - The API container has `SUPABASE_KEY`, whose JWT role claim is `anon`, and `SUPABASE_SERVICE_KEY`
    (role `service_role`). `get_async_supabase_client()` (`src/memory/services/factories.py:759`)
    prefers the service key.
  - A server-side write therefore works only through that client.
  - RLS is off on all six tables.
- **Default ACLs for role `postgres`** grant `anon`/`authenticated` `arwdDxt` on new tables and views,
  and `EXECUTE` on new functions. Every object this design creates must `REVOKE` explicitly.
  Today `update_tool_registry_metrics`, `find_similar_compositions` and `get_tool_execution_order`
  are EXECUTE-able by `anon`.
- **Rehearsal results**, all inside `BEGIN … ROLLBACK`:

| Probe | Result |
|---|---|
| `ToolRegistry.sync_to_database()` update payload (`registry.py:565`) | `ERROR: column "tier" of relation "tool_registry" does not exist`. Every update fails and is counted "skipped". |
| its insert payload | `ERROR: null value in column "category"` |
| insert `cohort_builder` with category `COHORT` | `invalid input value for enum tool_category: "COHORT"` |
| insert `cohort_builder` with source_agent `cohort_constructor` | `violates check constraint "valid_agent"` |
| insert `model_inference` (`PREDICTION`, `prediction_synthesizer`) | succeeds. It was simply never seeded. |
| step inserted with status COMPLETED | trigger does **not** fire (0 perf rows). It is `AFTER UPDATE` only. |
| step EXECUTING→COMPLETED, then COMPLETED→FAILED | 2 perf rows for one step (double count) |
| `v_composition_success_rate`, 1 episode with 2 planned tools | `total_compositions = 2`. The `LEFT JOIN tools_extracted` fans out per tool. |
| `update_tool_registry_metrics()` after one failed run | `success_rate = 0` written to the registry. There is no minimum-n guard. |
| `get_tool_execution_order(...)` | root tool returns `depends_on = {NULL}` |
| step for a tool with no `tool_registry` row | `violates foreign key constraint "composition_steps_tool_id_fkey"` |
| `ALTER TYPE tool_category ADD VALUE 'COHORT'` | allowed in a transaction (PG 15.8). The new value cannot be *used* in the same transaction. |

- **Other facts.**
  - pgvector is 0.8.0, and `pg_cron` is not installed.
  - PostgREST reloads its schema cache on DDL through the `pgrst_ddl_watch` event trigger (ml/036 L82).
  - `scripts/run_migrations.sh` applies `database/memory` before `database/ml`, so `e2i_agent_name`
    exists before any ml migration. It wraps each file in `--single-transaction`.

## 3. Goal: what a leader notices when the loop works

1. **Every composition is accounted for, including the ones that fail.** Each run has one row with its
   outcome (success / partial / failed / timeout), the phase it failed in and per-phase latency. Runs
   abandoned mid-flight (worker killed, client timeout) show as abandoned, not missing. Today 12 of 19
   planned runs vanish without a reason.
2. **Per-tool reliability separates three kinds of non-success:**
   - **health failures:** exception after retries, timeout;
   - **honest refusals:** `ToolRefusalError` / `ToolInputError`, meaning the data could not answer, which
     is a finding and not a fault;
   - **plan defects:** unresolvable reference, dependency unmet, tool not registered.

   Only health failures count against a tool.
3. **Verdict word first, then the numbers,** on an admin surface: "Too few runs to judge (n=3)",
   "Caveat: 6 of 24 runs failed (timeout)", "Reliable (n=52)".
4. **Similar-composition references name what failed** in the reference run, instead of presenting a
   partial run as a clean "successful composition".
5. **The planner sees measured latency and a reliability caveat only when the evidence clears a
   calibrated bar** (§7). At today's volume the bar is not reached, and the planner prompt is unchanged.
   That silence is the correct behaviour, and a test pins it.
6. **The DB registry matches the code on every boot** with no migration per schema change. That covers
   all 20 live tools, including `cohort_*` and `model_inference`.

## 4. Registry sync (replaces the migration-per-schema-change regime)

**Decision.** One SQL function, `sync_tool_registry(p_tools jsonb, p_dependencies jsonb,
p_max_deprecations int DEFAULT 3)`, called by a Python sync client:

- **At API startup:** a fire-and-forget task in `lifespan` after the Supabase client is initialised
  (`src/api/main.py` ~L278). Same pattern as the health heartbeat and the chatbot warm task, so boot
  is never delayed.
- **Lazily before the first composition record** in any process where the startup sync did not
  succeed, behind a process-level once-guard with an `asyncio.Lock`.

**Payload** (built in the new `src/agents/tool_composer/registry_sync.py`):
- Tool fields come from `create_default_tools()` (`tool_registry.py:313`): name, description, input
  schema, output schema (pydantic JSON Schema), declared `avg_latency_ms`, version.
- `category` comes from `TOOL_METADATA` and `source_agent` from the registered schema.
- Dependencies come from `DEPENDENCY_FIELD_MAPPINGS`.
- The payload is what `scripts/generate_tool_registry_sync_migration.py::build_payloads` renders today,
  minus the `NOT_SEEDED_IN_DB` exclusion.

**Idempotency.**
- Rows are upserted on `name`, with an `IS DISTINCT FROM` guard so an unchanged row is not rewritten
  and `updated_at` does not churn.
- Dependencies become exactly the payload set (upsert + delete of pairs not in the payload).
- Tools in the DB but not in code get `deprecated_at = now(), composable = false`. They are never
  deleted, because `composition_steps.tool_id` references them.
- Rehearsed: the first call inserted/updated, and a second identical call returned all-zero counts.

**Multi-worker race.** `pg_advisory_xact_lock(hashtext('sync_tool_registry'))` inside the function
serialises the two gunicorn workers (and any lazy call). The second caller sees no changes.

**Failure behaviour.**
- The Python client builds the payload only when `create_default_tools()` resolves every
  `TOOL_METADATA` entry. It raises `LookupError` otherwise, so a partially imported registry never syncs.
- The function refuses:
  - an empty payload (rehearsed: raises);
  - a payload that would deprecate more than `p_max_deprecations` tools. In rehearsal a 2-tool payload
    deprecated 15 tools and deleted all 11 dependencies, which is exactly the hazard the guard exists for.
- The function is atomic, so a bad row (unknown agent) rejects the whole sync (rehearsed) and leaves the
  previous state.
- Any failure is logged at WARNING with the error, and the API keeps running. Recording then reports
  the missing tools (§5.4).

**Schema changes (migration ml/039):**
- `ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'COHORT'`. Only the runtime sync uses the value,
  never the same migration transaction.
- `valid_agent`: replace the hard-coded six-agent list with
  `CHECK (source_agent = ANY (enum_range(NULL::e2i_agent_name)::text[]))`.
  - `e2i_agent_name` is the maintained agent taxonomy (memory migrations 018/029/048). It already has
    `cohort_constructor` and `cohort_profiler`, so adding an agent no longer needs an ml migration.
  - Rehearsed: `casual_impact` is refused.
- Drop `tool_registry.success_rate`.
  - Its values are seed defaults that look measured (0.95 on 14 rows). No code reads it; the only writer
    was `update_tool_registry_metrics`, which is also dropped (§6).
  - Measured reliability lives in `get_tool_reliability()`.
  - `avg_latency_ms` stays and is documented as the **declared** baseline written by the sync.
- `sync_tool_registry` gets `REVOKE ALL … FROM PUBLIC, anon, authenticated; GRANT EXECUTE … TO service_role`.

**Retirement of the #2012 regime.**
- Delete `scripts/generate_tool_registry_sync_migration.py`.
- Delete section 3 of `tests/unit/test_agents/test_tool_composer/test_registry_schema_drift_2003.py`
  (`_latest_sync_migration`, `test_db_sync_*`, `test_generator_output_is_what_the_drift_parser_reads`).
- Sections 1–2 stay unchanged: the live registry against the callables, and `create_default_tools`
  against the live registry.
- A new real-DB test proves that the payload applied by `sync_tool_registry` leaves `tool_registry` /
  `tool_dependencies` equal to the payload.
- `database/ml/037` stays as applied history. The ledger tracks filenames, so it is not edited.
- The comment at `tool_registry.py:214–216` and `docs/data/03-ML-PIPELINE-SCHEMA.md` §4.1 ("Rows are
  seeded by migration, never registered at runtime") are updated.

**`ToolRegistry.register_from_database` / `sync_to_database`** (`src/tool_registry/registry.py:432–604`)
are deleted along with their mock tests (`tests/unit/test_tool_registry/test_registry.py` ~L564–754).
- **Intent (G3, `fee8bc3e0`):** "Dynamic tool registration from database".
- **Reasons to delete:**
  - The owner chose code → DB.
  - `register_from_database` installs placeholder callables that raise `ToolNotFoundError`. Wired in,
    that would offer the planner tools that always fail.
  - `sync_to_database` fails on every row against the live schema (§2.3).
  - The new client replaces both.

## 5. Recording

### 5.1 Stores: which one is canonical for each signal

| Signal | Canonical store | Why |
|---|---|---|
| Tamper-evident phase provenance | `audit_chain_entries` (unchanged) | A hash chain shared by all agents. It has no per-step or per-tool rows, and its `query_text` feeds other readers (`/analytics`). |
| Composition record: outcome, failed phase, per-phase latency, plan | `composer_episodes` | Purpose-built columns. Episodic memory covers only the synthesized path and has no phase latency or failure phase. |
| Per-step outcome | `composition_steps` | Nothing else has it |
| Per-tool invocation metrics | `tool_performance` | Nothing else has it. `chatbot_analytics.tools_invoked` holds chat-tool names (`tool_composer_tool`), not composer tools. |
| Similarity retrieval for plan reuse | `episodic_memories` (unchanged) | Already embeds at 1536-dim with the shared embedding service and hydrates content (#889). A second embedding per composition would double embedding spend for the same retrieval. |
| Classification correctness | `classification_logs` (unchanged, #1341) | Already closed |

The stores join on `composition_id`, which episodic `raw_content` already carries (`memory_hooks.py`).

### 5.2 Schema fit (migration ml/040)

**`composer_episodes`:**
- Add:
  - `outcome text CHECK IN ('success','partial','failed','timeout')`, because `composition_status` has
    no PARTIAL and `status` keeps the terminal enum value;
  - `failed_phase text`;
  - `entry_point text` (`chat_tool` / `orchestrator_agent` / `direct`);
  - `brand`, `region`;
  - `is_synthetic boolean NOT NULL DEFAULT false`;
  - `tools_executed`, `tools_succeeded`;
  - `last_phase_at timestamptz NOT NULL DEFAULT now()`.
- `total_latency_ms DROP NOT NULL`. The row is inserted at start, before the total is known.
- `success`, `user_rating`, `feedback_text`, `feedback_at` keep their designed meaning: *user feedback*,
  NULL until a feedback link exists (owner decision O1).
- **Drop `query_embedding`, `idx_composer_episodes_embedding` and `find_similar_compositions()`** (§7.3).

**`composition_steps`:**
- Add:
  - `outcome_class text CHECK IN ('succeeded','cache_hit','refused','input_rejected','timeout','error',
    'plan_defect','dependency_unmet','circuit_open','not_registered')`;
  - `attempts int`;
  - `cache_hit boolean NOT NULL DEFAULT false`.
- Widen `serves_sub_question` to `varchar(100)`, because planner sub-question ids are free text.
- `status` uses the existing enum: COMPLETED for succeeded/cache_hit, FAILED otherwise.

**`tool_performance`:**
- Add:
  - `outcome_class text CHECK IN ('succeeded','refused','input_rejected','timeout','error')`;
  - `attempts int`;
  - a unique partial index on `step_id`, so one performance row per step.
- Rows are written only for steps where the tool was **invoked**. A cache hit, a plan defect, an unmet
  dependency, an open circuit or an unregistered tool never ran the tool and says nothing about its
  health.

**Replace the trigger** with an explicit insert inside the finish function. `trg_log_step_performance`
and `trigger_log_step_performance()` are dropped:
- the trigger fires on UPDATE only, so terminal-state inserts are lost;
- it double-counts COMPLETED→FAILED;
- the designed intent, "a performance row per finished step", is kept, but in one visible statement.

### 5.3 Write points

1. **`compose()` start.**
   - Generate `composition_id` up front and pass it into every `CompositionResult` built later,
     including `_create_error_result` and `_create_total_failure_result`.
   - Enqueue `composer_record_start` with the redacted query, session_id, user_id, entry_point
     (`context["entry_point"]`, set by the two entry points), brand, region, and
     `is_synthetic = deployment_includes_synthetic()` (`src/repositories/provenance.py:95`).
2. **After decompose / plan / execute**, enqueue `composer_record_phase` with the new status, phase
   latency, `sub_questions` (id, intent, question text redacted to 200 chars) and `tool_plan` (steps:
   step_id, tool_name, depends_on, input_mapping keys and `$step` references). Execute also sends
   `parallelizable_groups`.
3. **Terminal**, enqueue `composer_record_finish` with outcome, status, failed_phase, error (≤2000
   chars), per-phase and total latency, counts, and the steps built from `ExecutionTrace.step_results`.
   The terminal paths are:
   - success or partial (end of `compose`);
   - `_create_total_failure_result`;
   - `_fail_closed` (Decomposition / Planning / Execution / unexpected);
   - `asyncio.CancelledError`, recorded as `timeout` and then re-raised. This is the orchestrator's 180 s
     SLA cancel (`router.py:298`).

**Executor change (`executor.py`).** `StepResult` gains `outcome_class`, `attempts` and `cache_hit`
(`models/composition_models.py:204`). Each existing return site sets them:

| Site | Value |
|---|---|
| L522 | `dependency_unmet` |
| L549 | `plan_defect` |
| L611 | `cache_hit` |
| L631 | `circuit_open` |
| L652 | `not_registered` |
| L703 | `succeeded` |
| L753 | `input_rejected` / `refused` |
| L788 | `timeout` |
| L816 | `error`, attempts = max_retries + 1 |

The classes come from the exception type already caught at each site, never from error text.

### 5.4 Asynchronous and fail-open

- **Write path.** A new `CompositionRecorder` (`src/agents/tool_composer/learning_recorder.py`) chains
  its RPCs on a single background task per composition, so order is kept (start < phase < finish).
- **Bookkeeping.**
  - Each write has a 5 s timeout.
  - Tasks live in a module-level set with a `discard` done-callback, the `_pending_log_tasks` pattern
    from `intent_classifier.py:72/834`.
  - A failure logs one WARNING and never raises into `compose()`.
- **Composition latency.** Added latency is zero by construction, because nothing is awaited on the
  compose path. The live cert measures it (§9).
- **Unknown tools.** `composer_record_finish` returns `unknown_tools`: steps whose tool has no registry
  row. Rehearsed: a `cohort_builder` step was reported and the other 3 steps were recorded.
  - On a non-empty list the recorder runs the lazy sync once for the process.
  - It then re-sends only the missing steps through `composer_record_steps(p_composition_id, p_steps)`,
    which inserts steps plus performance rows for an already-finished episode and is idempotent on
    `(episode_id, step_number)`.
- **Transport.** The recorder and sync client call a small `rpc(name, params)` port.
  - Production: the service-role `get_async_supabase_client().rpc(...).execute()`.
  - Real-DB tests: a psycopg connection to a throwaway database calling `SELECT name(...)`. That is a
    real database through another transport, not a mock.
  - The PostgREST transport is proven in the live cert.
- **Multi-worker and Celery.** Writes are per composition, and no Celery task runs the composer
  (`grep` of `src/tasks`, `src/workers`: none). The only shared-state writer is the sync, which is
  serialised by the advisory lock.

### 5.5 PII and retention

- **Stored:**
  - **Query text:** `redact_query(query, max_len=500)` (`src/utils/redaction.py`), the same 500-char cap
    as the episodic `raw_content` (`memory_hooks.py`). `classification_logs` already keeps the full text
    in the same schema.
  - **Input params:** scalars (str ≤200 chars, numbers, bools), lists as `{"type":"list","len":n}`,
    dicts as `{"type":"dict","keys":[…]}`, DataFrames as `{"type":"frame","rows":r,"columns":c}`.
    Planner-bound column names are kept, because they are the plan. Entity id lists (for example
    `target_entities`) are never stored element-wise.
  - **Output:** keys only (`output_result = {"keys":[…]}`) plus the error message.
- **Not stored:** `synthesized_response` and `tool_outputs` stay NULL / `{}`. They can carry
  patient-level numbers, and learning does not need them.
- **Access:** `service_role` only. There are no `anon`/`authenticated` grants, and a real-DB test pins
  that for every new or recreated object. The admin API is `require_admin`.
- **Retention:** no purge job. At the measured volume (about 20 compositions and ≤8 steps each in six
  weeks) the tables grow by kilobytes a month. The runbook sets a re-evaluation trigger of
  `composer_episodes` > 50,000 rows.

## 6. Aggregation

**Decision: live reads, no materialisation, no schedule.** At this volume a scan of `tool_performance`
filtered by `executed_at` is sub-millisecond. There is no `pg_cron`, and a Celery beat entry would add
an ordering and staleness problem for no measurable gain.

- **`get_tool_reliability(p_days int DEFAULT 30)`**, SQL `STABLE`, `service_role` only. One row per
  active tool:
  - declared latency;
  - `n_invoked`, `n_succeeded`;
  - `n_refused` (refused + input_rejected);
  - `n_health_failures` (timeout + error);
  - p50 / p95 latency of **succeeded** invocations;
  - `last_executed_at`, most common health error.

  Rehearsed on a recorded episode: `causal_effect_estimator 1/1/0/0 p50 900`,
  `gap_calculator 1/0/1/0` (refused, not failed).
- **`v_tool_reliability`** is recreated as `SELECT * FROM get_tool_reliability(30)`. That keeps the
  designed name, with the 30-day window the reliability rule in §7 uses.
- **`v_composition_success_rate`** is recreated without the fan-out: daily counts by `outcome`,
  unfinished, and p50/p95 total latency. Rehearsed: 1 episode gives 1.
- **`v_active_compositions`** is recreated for runs not yet in a terminal status: `elapsed_ms` and
  `abandoned = last_phase_at < now() - interval '10 minutes'`. The composer SLA is 180 s, so 10 minutes
  cannot be a live run.
- **Dropped:** `update_tool_registry_metrics()`.
  - **Intent:** planner-visible reliability. That is now served by `get_tool_reliability` with n and
    failure classes.
  - **It is harmful as written:** no minimum n, so one failed run writes `success_rate = 0`.
  - **It has never had a caller.**
- **Untouched:** `v_classification_accuracy` (#1341) and `get_tool_execution_order()`. The latter has
  no consumer; the executor orders with `ExecutionPlan.get_execution_order()`. Its `{NULL}` defect is
  listed in §10.

## 7. Feedback into behaviour

### 7.1 Reliability rule, calibrated on planted truth at production n

The rule is a pure function in the new `src/agents/tool_composer/reliability.py`. It is the one place
that turns counts into a verdict; the planner and the admin API both call it.

| Verdict | Condition |
|---|---|
| `no_runs` | n_invoked = 0 |
| `too_few_runs` | 0 < n_invoked < 20 |
| `caveat` | n ≥ 20 and Wilson 95% **lower** bound of health-failure rate ≥ 10% |
| `reliable` | n ≥ 20 and Wilson 95% **upper** bound < 10% |
| `inconclusive` | otherwise |

Refusals are reported next to the verdict ("5 refused: data could not answer") and never enter the rate.

**Calibration** (scratch simulation, 20,000 Bernoulli draws per cell, seed 7). A healthy tool is
p = 0.02 or 0.05; a failing tool is p = 0.20 or 0.30.

| n | P(caveat), p=0.02 | P(caveat), p=0.05 | P(caveat), p=0.20 | P(caveat), p=0.30 | P(reliable), p=0.02 |
|---|---|---|---|---|---|
| 5 | 0.004 | 0.023 | 0.261 | 0.477 | 0 |
| 20 | 0.000 | 0.002 | 0.371 | 0.760 | 0 |
| 40 | 0.000 | 0.001 | 0.563 | 0.945 | 0.449 |
| 100 | 0.000 | 0.000 | 0.874 | 0.999 | 0.950 |

- **False caveats are rare at every n** (≤ 0.2% for a 5% tool at n ≥ 20).
- **A 30% failing tool is caught** 76% of the time at n = 20 and 94.5% at n = 40.
- **The n ≥ 20 floor removes the n = 5 false-caveat tail** (2.3% at p = 0.05).
- **At production n** (≤ 7 invocations per tool so far) every tool reads `too_few_runs`, and the planner
  prompt does not change. A test pins that silence.

### 7.2 What changes in planning

- **Where the numbers come from.** A `ToolReliabilityReader` calls `get_tool_reliability(30)` through
  the service client.
  - It caches per process for 300 s, so each worker makes one read per 5 minutes, not one per plan.
  - It is fail-open: on any error it returns `{}`, and planning uses the declared numbers as today.
- **Planner prompt** (`planner.py:371`) and **DSPy formatter** (`dspy_integration.py:189–225`) both go
  through one shared formatter:
  - Declared latency is labelled "(declared)".
  - At n ≥ 20 the measured median replaces it: "Measured median 1.8 s over 24 runs (30 days)".
  - On `caveat` a line is added: "Reliability caveat: 6 of 24 runs failed on tool errors (timeout)".
    Refusals are shown only as a count.
- **`_estimate_duration`** (`planner.py:955`) uses the measured median at n ≥ 20.
- **Deliberately not done (feedback-loop guards):**
  - no automatic removal of a tool from the offer;
  - no reranking;
  - no seeding of the circuit breaker;
  - no change to `_get_fallback_mapping`.

  A caveated tool that the LLM avoids gets fewer runs. If its count then drops below the floor, the
  caveat disappears. That oscillation is benign, and it is bounded by the 30-day window.
- **Unverified assumption (labelled):** a caveat line changes the LLM's tool choice when an alternative
  exists. Nothing measures that yet, and it cannot matter until a tool reaches n ≥ 20. The plan records
  it as a follow-up measurement, not a claim.

### 7.3 Similar-composition reuse

**Decision:** `episodic_memories` stays the canonical similarity store (§5.1).
`find_similar_compositions()` (SQL), `composer_episodes.query_embedding` and its ivfflat index are
dropped, because their design intent (reuse a successful plan for a similar query) is served by the
live episodic path. The SQL function has never had a caller, and the column has never held a row.

**New behaviour.**
- `memory_hooks.find_similar_compositions` hydrates each reference with its recorded steps: one
  `composition_steps` read by `composition_id` for the ≤ 3 references.
- `planner._format_episodic_context` (`planner.py:539`) renders "Tools used: causal_effect_estimator
  (succeeded), gap_calculator (refused: single-brand frame), rank_drivers (skipped: dependency unmet)".
- A reference with no recorded steps (the 2 pre-loop rows) renders exactly as today.
- Selection is unchanged. PARTIAL references stay eligible, but their failures are now visible.

### 7.4 In-memory G8

`PlanExecutor.update_tool_performance` (`executor.py:1312–1375`) is deleted with its mock tests
(`test_executor.py` ~L1611–1685).
- It has no production caller.
- It mutates the process-wide `ToolSchema.avg_execution_ms` from one executor's EMA. Wired in, one
  request's stats would leak into every later plan.
- Its intent (learned latency) is served by §7.2.

`ToolFailureTracker` stays. It drives the in-request circuit breaker and retry policy.

## 8. Observability surface

**An existing surface is extended, not a new page:**
- The Admin page, **Observability** tab (`frontend/src/pages/Admin.tsx:49–80`,
  `components/admin/ObservabilityTab.tsx`), which has a days selector and stat cards and is admin-only.
- Its API sits next to `GET /api/admin/observability/llm-usage` (`src/api/routes/admin.py:349`).

**API.** `GET /api/admin/observability/tool-composer?days=30` (`require_admin`) returns:
- `compositions`: counts by outcome, unfinished, abandoned, p50/p95 total latency;
- `tools[]`: name, category, verdict word, n_invoked, n_succeeded, n_refused, n_health_failures,
  p50/p95, declared latency, most common health error;
- `recent_failures[]`: last 10 failed or partial compositions with failed_phase and the failing step
  classes. Redacted query preview is 100 chars.

**UI.** A "Tool composer" section in the Observability tab:
- stat cards (compositions, success, partial, failed, abandoned);
- a per-tool table whose first column is the verdict word;
- the recent-failure list.

Contract hygiene, from earlier incidents:
- a route docstring is an OpenAPI change, so regenerate `api.ts` in the same task;
- a new field needs `api-schemas.ts` Zod plus a parse test.

**`/analytics` is not extended.** It is organised around `audit_chain_entries` per agent, and adding a
per-tool drill-down there would split one reading across two surfaces.

## 9. Verification

- **Red first, real database, no mocks.**
  - Migrations 039/040 and every SQL function are tested against a throwaway database on the droplet's
    own Postgres. This is the `scripts/test_migration_idempotency.sh` precedent: `DROP/CREATE DATABASE`
    as `supabase_admin`, `vector` extension, `e2i_agent_name` created with the values read (read-only) from the live enum, ml/013 → 027 → 037 →
    039 → 040, then re-apply 039/040 for idempotency.
  - The prod `postgres` database is never written by tests.
  - Opt-in through `E2I_DB_INTEGRATION=1`, as in `tests/integration/test_issue_825_schema_drift_realdb.py`,
    so CI (no DB) skips the tests.
- **Executor outcome classes.** Tested by running the real `PlanExecutor` on real registered tools and
  real frames:
  - a single-brand frame for a `gap_calculator` refusal;
  - a planned `$step_x.missing` reference;
  - a failing upstream for dependency unmet;
  - a second identical run for a cache hit;
  - an unregistered name;
  - a registered slow sync callable for a timeout (registered through `snapshot`/`restore_snapshot`).
- **Composer wiring.** An opt-in real-LLM run (`E2I_LIVE_LLM=1`) records to the throwaway database
  through the psycopg transport, plus the live cert.
- **Reliability rule.** The planted-truth test reproduces the §7.1 table bounds: false caveat ≤ 1% at
  p = 0.05 for n ∈ {20, 40, 100}; detection ≥ 90% at p = 0.30, n = 40; `too_few_runs` for every n < 20.
- **Grants.** For every created or recreated object, `has_table_privilege` /
  `has_function_privilege('anon'|'authenticated', …)` is false, and service_role is true.
- **Live cert on the deployed image** (record `image_sha: <sha>`):
  1. Boot logs show a sync with inserted=4 (`cohort_builder`, `cohort_validator`, `cohort_statistics`,
     `model_inference`) and the second worker with all-zero changes. DB: 20 active tools, 13 dependencies.
  2. A real chat composition produces one terminal `composer_episodes` row, steps equal to the plan's
     step count, and `tool_performance` rows equal to the invoked steps.
  3. A refusal-producing composition is recorded as `refused` and counted in `n_refused`, not
     `n_health_failures`.
  4. The admin endpoint and tab show "Too few runs to judge".
  5. The planner prompt log shows the declared latency: the gate is silent.
  6. PostgREST with the anon key: the RPCs and views are refused.
  7. Compose wall time within noise of the pre-deploy baseline, and recorder write durations logged
     at DEBUG are ≤ 5 s.
- **Rollback.**
  - Code: revert the PR. The recorder and sync are fail-open, so a code-only revert is safe with the
    migrations left in place.
  - DB: `database/ml/rollback_040.sql` / `rollback_039.sql`. They recreate the dropped function,
    trigger, column, index and views from their ml/013 definitions, restore the six-agent CHECK only
    when no row violates it, and drop the new functions and columns. Recorded rows in the new columns
    are lost.
  - `COHORT` cannot be removed from an enum (PG) and is harmless.
  - Rollback files are excluded from auto-apply by `run_migrations.sh`.

## 10. Non-goals

- **Linking `classification_logs.classification_id` to episodes.** The classifier writes fire-and-forget
  and returns no id to the dispatcher (`intent_classifier.py:826`). The column stays NULL.
- **Live per-step progress rows** (PENDING→EXECUTING updates per step). Steps are written once at
  terminal state, and phase status covers abandoned-run diagnosis.
- **A circuit breaker shared across requests or workers.**
- **`chatbot_analytics.tools_succeeded` / `tools_failed` / `tool_composer_used`,** which the chat writer
  never fills (`copilotkit.py:1859–1905`). That is a chat-analytics defect, separate from this loop.
- **Fixing `get_tool_execution_order()`'s `{NULL}`.** It has no consumer.
- **Procedural-memory `tool_composition` patterns.**
- **`tool_performance` rows from non-composer callers** (`called_by='agent'|'direct'`); none exist.
- **A retention/purge job** (§5.5).

## 11. Decisions that need the OWNER

- **O1. User feedback on compositions.** The design wanted `composer_episodes.success` / `user_rating` /
  `feedback_text` filled from user feedback (`v4.2_IMPLEMENTATION_TODO.md`).
  - Buildable now as a nightly task that mirrors #1341's explicit-feedback matcher:
    `chatbot_message_feedback` matched on session_id and time window.
  - `chatbot_message_feedback` holds **3 rows ever** (last 2026-07-07), so it would label about 0
    episodes today.
  - **Recommendation:** build it in this lane as the last functional task, so the feature is not left
    to be chased later. It reverses if the owner prefers to wait for feedback volume. The plan marks it
    as a task gated on this answer.
- **O2. The planner-prompt caveat.** §7.2's thresholds (n ≥ 20, 10% health-failure bar) are calibrated
  (§7.1), but whether a caveat line changes LLM tool choice is unmeasured, and it is dormant at current
  volume.
  - **Recommendation:** ship it dormant with the planted-truth test.
  - The fact that would reverse this: the owner wants no automatic prompt changes from telemetry,
    #1341-style human-gated authority. In that case the verdicts go only to the admin surface.
- **O3. Dropping designed DB objects:** `find_similar_compositions()`, `composer_episodes.query_embedding`
  plus the ivfflat index, `update_tool_registry_metrics()`, `trg_log_step_performance`,
  `tool_registry.success_rate`.
  - Each has verified intent that is served elsewhere, no caller ever, and a measured defect or a
    duplicate store (§4, §5.2, §6, §7.3).
  - **Recommendation:** drop.
  - Rollback files recreate them. The owner may prefer to keep the vector column for a future
    composer-specific similarity search.
