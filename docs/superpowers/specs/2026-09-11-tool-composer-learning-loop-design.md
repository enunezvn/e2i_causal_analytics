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
    exists before any ml migration. It wraps a file together with its ledger row in
    `--single-transaction`, **except** a file that contains `ALTER TYPE … ADD VALUE`, `CONCURRENTLY` or its
    own `COMMIT` (`scripts/run_migrations.sh:157–190`). Such a file runs statement by statement, and its
    ledger row is written only after a clean exit, so a partial failure is retried on the next deploy.

## 3. Goal: what a leader notices when the loop works

1. **Every composition is accounted for, including the ones that fail.** Each run has one row with its
   outcome (success / partial / failed / cancelled), the phase it failed in and per-phase latency. The
   steps that finished before a failure or a cancel are kept. Runs abandoned mid-flight (worker killed)
   show as abandoned, not missing. Today 12 of 19 planned runs vanish without a reason.

   Recording is best-effort by design. Nothing on the user's path waits for it, so a slow or unavailable
   database cannot delay or fail an answer. The loss is *measured*, not assumed:
   - every RPC can create the episode, so any single write that lands records the composition;
   - every write failure is counted;
   - the cert reconciles by **identity**: episodes join audit-chain `workflow_start` rows on
     `audit_workflow_id`, and both are checked against the list of compositions the cert itself triggered
     (§5.4, §9).

   Awaiting a write would not make recording durable: a crash one millisecond after an awaited start still
   loses the steps. It would only put the database on the user's latency path.
2. **Per-tool reliability separates three kinds of non-success:**
   - **health failures:** exception after retries, timeout;
   - **honest refusals:** `ToolRefusalError` / `ToolInputError`, meaning the data could not answer, which
     is a finding and not a fault;
   - **plan defects:** unresolvable reference, dependency unmet, tool not registered.

   Only health failures count against a tool.
3. **Verdict word first, then the numbers,** on an admin surface: "Too few runs to judge (n=3)",
   "Caveat: 6 of 24 runs failed (timeout)", "Reliable (n=52)".
4. **A plan that failed is not reused as if it had worked.**
   - Today the in-process plan cache (`cache.py:228–330`, 15-minute TTL) can hand a similar decomposition
     the steps of a plan whose execution failed, and skip the LLM planner to do it. The eligibility
     conditions are in §7.3.
   - Episodic references present a PARTIAL run as a clean "successful composition".
   - With the loop, a plan whose composition failed, or had a plan defect, is evicted from the plan
     cache. Episodic references recommend only the steps that succeeded and list what did not work (§7.3).
5. **Every planned step runs, in dependency order.** Today a step that the LLM leaves out of
   `parallel_groups` never executes, and a consumer grouped with its producer runs before its input
   exists (§7.4, measured).
6. **Measured reliability reaches the planner only after an experiment shows it changes the planner's
   choice** (§7.2).
   - The reliability rule is calibrated on planted truth (§7.1).
   - At today's volume it cannot fire. Its first effect is the admin surface.
7. **The DB registry matches the code on every boot** with no migration per schema change. That covers
   all 20 live tools, including `cohort_*` and `model_inference`.

## 4. Registry sync (replaces the migration-per-schema-change regime)

**Decision.** One SQL function, `sync_tool_registry(p_tools jsonb, p_dependencies jsonb,
p_max_deprecations int DEFAULT 3)`, called by a Python sync client:

- **At API startup:** a fire-and-forget task in `lifespan` after the Supabase client is initialised
  (`src/api/main.py` ~L278). Same pattern as the health heartbeat and the chatbot warm task, so boot
  is never delayed.
- **Lazily, on the recorder's background chain,** in any process where the startup sync did not succeed.
  It runs when a step write reports `unknown_tools`, behind a process-level once-guard with an
  `asyncio.Lock`, and it is never awaited by `compose()`.

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
serialises the two gunicorn workers (and any lazy call). The second caller sees no changes. Both
workers run the same image, so their payloads are identical.

**Mixed code versions.** Old and new code cannot sync at the same time:
- The API is a single container with a fixed name (`container_name: e2i_api`,
  `docker/docker-compose.yml:896`).
- Deploys and rollbacks recreate it with `up -d --no-deps --force-recreate` (`.github/workflows/deploy.yml`
  L480/485/490/1065), which stops the old container before the new one starts.
- Celery containers share the image but run no lifespan and no composer (§5.4), so they never sync.

A rollback to an older image re-syncs the registry to that image's tools. That is the intended
semantics: **the DB describes the code that is running.** No generation or ordering check is added,
because nothing can produce two live payloads at once.

**Failure behaviour.**
- The Python client builds the payload only when `create_default_tools()` resolves every
  `TOOL_METADATA` entry. It raises `LookupError` otherwise, so a partially imported registry never syncs.
- The function validates before any write and raises (no partial state) on:
  - an empty payload (rehearsed: raises);
  - a payload that would deprecate more than `p_max_deprecations` currently active tools. The count is
    computed first, before any mutation. In rehearsal an unguarded 2-tool payload deprecated 15 tools and
    deleted all 11 dependencies, which is exactly the hazard. The rehearsal draft did not yet carry the
    guard; the migration task adds it red-first.
  - a dependency whose consumer or producer is not in the tool payload;
  - a duplicate tool name in the payload.
- The function is atomic, so a bad row (unknown agent) rejects the whole sync (rehearsed) and leaves the
  previous state.
- Any failure is logged at WARNING with the error, and the API keeps running. Recording then reports
  the missing tools (§5.4).

**Schema changes — three migrations, split by the runner's transaction branch (§2.3):**
- **ml/039_tool_category_cohort.sql** holds only `ALTER TYPE tool_category ADD VALUE IF NOT EXISTS
  'COHORT'`.
  - The runner applies it un-wrapped.
  - It is one idempotent statement, so a failure cannot leave a partial state, and a re-run is a no-op.
  - Only the runtime sync uses the value.
- **ml/040_tool_registry_startup_sync.sql** holds everything else in this section plus
  `sync_tool_registry`. The runner applies it with its ledger row in one transaction.
- **ml/041_composer_learning_loop_recording.sql** holds §5.2 and §6 (wrapped, transactional).

The contents of ml/040:
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

### 5.2 Schema fit (migration ml/041)

**`composer_episodes`:**
- Add:
  - `audit_workflow_id uuid`, nullable: the audit chain's `workflow_id` for this run.
    - It comes from the **local** `audit_workflow_id` variable in `compose()` (`composer.py:272`, assigned
      at `:289` only when the audit start succeeds). It is never read from `context`, which a caller can
      pre-populate.
    - It is NULL when no audit service is wired or the audit start raised.
    - It joins episodes to audit rows. `audit_chain_entries` has no free-form input column to carry a
      composition_id. Its measured columns are entry_id, workflow_id, sequence_number, agent_name,
      agent_tier, action_type, created_at, duration_ms, input_hash, output_hash, validation_passed,
      confidence_score, refutation_results, previous_entry_id, previous_hash, entry_hash, user_id,
      session_id, query_text and brand.
  - `outcome text CHECK IN ('success','partial','failed','cancelled')`, because `composition_status` has
    no PARTIAL and `status` keeps the terminal enum value (COMPLETED / FAILED / TIMEOUT).
    - A cancel is `cancelled`, not `timeout`: `CancelledError` alone does not prove a deadline expired.
    - The orchestrator's 180 s SLA (`router.py:298`) is one cause; a client disconnect or a shutdown are
      others.
    - The enum value `TIMEOUT` stays unused until a caller passes an explicit deadline reason.
  - `failed_phase text`, `error_type text` (§5.3 item 4);
  - `plan_source text CHECK IN ('llm','plan_cache','kpi_deterministic')`: whether the plan came from the
    LLM planner, the in-process plan cache or `_build_kpi_causal_plan` (§7.3);
  - `entry_point text` (`chat_tool` / `orchestrator_agent` / `direct`);
  - `brand`, `region`;
  - `is_synthetic boolean NOT NULL DEFAULT false`;
  - `tools_executed`, `tools_succeeded`;
  - `last_activity_at timestamptz NOT NULL DEFAULT now()`, bumped by every recording RPC and by the
    liveness heartbeat (§5.4).
- `total_latency_ms DROP NOT NULL`. The row is inserted at start, before the total is known.
- `success`, `user_rating`, `feedback_text`, `feedback_at` keep their designed meaning: *user feedback*,
  NULL until a feedback link exists (owner decision O1).
- **Drop `query_embedding`, `idx_composer_episodes_embedding` and `find_similar_compositions()`** (§7.3).

**`composition_steps`:**
- Add:
  - `outcome_class text CHECK IN ('succeeded','cache_hit','refused','input_rejected','timeout','error',
    'plan_defect','dependency_unmet','circuit_open','not_registered')`;
  - `attempts int`;
  - `cache_hit boolean NOT NULL DEFAULT false`;
  - `error_type text`: the exception class name, for aggregation. No message is stored (§5.5).
- `step_number` is the step's index in `plan.steps`, fixed at plan time, so it does not depend on
  completion order. It makes `(episode_id, step_number)` a stable idempotency key (the existing
  `unique_step_in_episode`).
- `serves_sub_question` holds the **positional sub-question index** as text (`"0"`, `"1"`, …), never the
  LLM's id (§5.5). The column stays `varchar(20)`. `composition_steps.step_id` and
  `tool_performance.step_id` are database-generated UUIDs, unrelated to planner step ids.
- `status` uses the existing enum: COMPLETED for succeeded/cache_hit, FAILED otherwise.

**`tool_performance`:**
- Add:
  - `outcome_class text CHECK IN ('succeeded','refused','input_rejected','timeout','error')`;
  - `attempts int`;
  - `is_synthetic boolean NOT NULL DEFAULT false`, copied from the episode (§6);
  - `tool_version text`, copied from `tool_registry.version` at insert;
  - a unique partial index on `step_id`, so one performance row per step.
- Rows are written only for steps where the tool was **invoked**. A cache hit, a plan defect, an unmet
  dependency, an open circuit or an unregistered tool never ran the tool and says nothing about its
  health.
- **Unit of measurement: the step's final outcome.** A step that failed twice and then succeeded is one
  `succeeded` row with `attempts = 3`. Retries are exposed as `n_retried` (§6), not hidden, but they do
  not count as failures.

**Replace the trigger** with an explicit insert inside the step-recording function.
`trg_log_step_performance` and `trigger_log_step_performance()` are dropped:
- the trigger fires on UPDATE only, so terminal-state inserts are lost;
- it double-counts COMPLETED→FAILED;
- the designed intent, "a performance row per finished step", is kept, but in one visible statement.

### 5.3 Write points

1. **`compose()` start.**
   - Generate `composition_id` up front and pass it into every `CompositionResult` built later,
     including `_create_error_result` and `_create_total_failure_result`.
   - **After** the audit-start attempt (`composer.py:272–293`), build the episode **seed**: redacted
     query, session_id, user_id, entry_point (`context["entry_point"]`, set by the two entry points),
     brand, region, the local nullable `audit_workflow_id`, and
     `is_synthetic = deployment_includes_synthetic()` (`src/repositories/provenance.py:95`).
   - Enqueue `composer_record_start(seed)` on the recorder's chain. It is not awaited.
2. **After decompose / plan / execute**, update the recorder's in-memory **snapshot** and enqueue
   `composer_record_phase` with that phase's delta:
   - the new status and the phase latency;
   - `sub_questions`: positional index and normalized intent only (no text, no LLM ids, §5.5);
   - `tool_plan`: per step its step_number, tool_name, depends-on step numbers, and the structural input
     map of §5.5;
   - `plan_source`, and whether the execution order was repaired (§7.4);
   - `parallelizable_groups`, sent by execute, as lists of **step numbers**: the executed order after
     §7.4, with a flag when it was repaired.
3. **Per finished step, during execution.** `PlanExecutor.execute` gains an optional
   `on_step_result(step_number, StepResult)` callback. It is invoked the moment a step's `StepResult`
   exists:
   - single-step groups: after `_execute_step` returns (`executor.py:457`);
   - parallel groups: **inside** each task of `_execute_parallel` (`execute_with_semaphore`,
     `executor.py:923–925`), before `asyncio.gather` returns.

   The recorder enqueues `composer_record_steps` for each call.
   - A step that finished while a sibling in the same parallel group is still running is recorded even
     if the group is then cancelled.
   - A step that was still running when a cancel arrived has no result and is not recorded. The episode
     says `cancelled` in phase `execute`.
   - The callback is synchronous and only enqueues (no I/O), so it cannot slow or fail a step.
4. **Terminal**, enqueue `composer_record_finish` with the **complete snapshot**, so every field a lost
   phase write carried is restored:
   - outcome, status, failed_phase;
   - the episode's `error_type` (the caught exception's class, and the inner class for wrapped
     Decomposition/Planning/ExecutionError). No error text (§5.5);
   - every phase latency and the total;
   - sub-question indices and normalized intents, `tool_plan`, `plan_source`, the order-repair flag, and
     parallel groups as step numbers;
   - counts.

   The terminal paths are:
   - success or partial (end of `compose`);
   - `_create_total_failure_result`;
   - `_fail_closed` (Decomposition / Planning / Execution / unexpected);
   - `asyncio.CancelledError`: recorded as `cancelled`, then re-raised.

   Finish also re-sends the full step list. `composer_record_steps` is idempotent, so a step whose
   per-step write was lost is recovered here.

**Executor change (`executor.py`).** `StepResult` gains `outcome_class`, `attempts`, `cache_hit` and
`error_type` (`models/composition_models.py:204`). Each existing return site sets them:

| Site | outcome_class | attempts |
|---|---|---|
| L522 | `dependency_unmet` | 0 |
| L549 | `plan_defect` | 0 |
| L611 | `cache_hit` | 0 |
| L631 | `circuit_open` | 0 |
| L652 | `not_registered` | 0 |
| L703 | `succeeded` | attempt + 1 |
| L753 | `input_rejected` / `refused` | attempt + 1 |
| L788 | `timeout` (`SyncToolTimeout`) | attempt + 1 |
| L816 | `timeout` if the **last** attempt raised `asyncio.TimeoutError` (the async `wait_for` at L673), else `error` | max_retries + 1 |

- The classes come from the exception type already caught at each site, never from error text.
- The generic arm (L803) keeps the last exception object, not only its string, so the final class and
  `error_type` are exact.
- Today an async timeout lands in the generic arm and would read as `error`. That is the gap codex
  iteration 1 found.

### 5.4 Asynchronous and fail-open

A new `CompositionRecorder` (`src/agents/tool_composer/learning_recorder.py`) owns every write.

**Five RPCs. Every one carries the episode seed and begins with
`INSERT … ON CONFLICT (composition_id) DO NOTHING`,** so whichever write lands first creates the episode.
A failed start therefore never orphans later phase, step or finish data. Each RPC is idempotent, so a lost
response can be re-sent.

| RPC | Behaviour after the seed insert |
|---|---|
| `composer_record_start(p_seed)` | Nothing more; returns the episode_id |
| `composer_record_phase(p_seed, p_status, p_patch)` | Updates status and phase fields only while the episode is non-terminal |
| `composer_record_steps(p_seed, p_steps)` | `INSERT … ON CONFLICT (episode_id, step_number) DO NOTHING` for steps; performance rows for invoked classes only, `ON CONFLICT (step_id) DO NOTHING`. It works whether the episode is open or finished, and returns a stable receipt `{recorded, already_present, unknown_tools}`. |
| `composer_record_heartbeat(p_seed)` | Bumps `last_activity_at` only while the episode is non-terminal |
| `composer_record_finish(p_seed, p_final)` | Sets the terminal state and **all** phase latencies. The recorder keeps them in memory, so phase fields lost with a failed phase write are restored here. A second finish on a terminal episode returns `{recorded:false, already_terminal:true}` and changes nothing. |

**Ordering and retry.**
- The recorder chains start, phase, steps and finish writes on one background task per composition, so
  they apply in order.
- Each write has a 5 s timeout and **one** retry after 1 s on a transport error or timeout.
- After that the chain moves on. Every later write is self-sufficient because of the seed, so one lost
  write loses only its own delta, and finish restores phase fields and re-sends every step.
- Tasks live in a module-level set with a `discard` done-callback, the `_pending_log_tasks` pattern from
  `intent_classifier.py:72/834`.
- The API `lifespan` shutdown awaits the pending set for up to 5 s before closing clients, so a graceful
  deploy drains its in-flight records.

**Failure accounting.**
- A failed write logs one structured WARNING (rpc, composition_id, error class).
- It increments a Prometheus counter `composer_record_failures_total{rpc}`, following
  `src/api/routes/metrics.py`'s optional-client pattern.
- It never raises into `compose()`.
- Coverage is measured in the cert (§9), by identity:
  - episodes against audit-chain `workflow_start` rows on `audit_workflow_id`;
  - both against the cert's own list of triggered compositions;
  - steps against `ExecutionTrace` counts.

**Composition latency.** No write is awaited on the compose path: enqueueing is an in-memory append.
The lazy registry sync also runs on the background chain. A slow or unreachable database therefore adds
no latency and cannot push a request past its caller's deadline. A test pins that with an unreachable
transport under a near-expired deadline. The live cert compares compose wall time with the pre-deploy
baseline (§9).

**Unknown tools.** `composer_record_steps` reports steps whose tool has no registry row in
`unknown_tools`. Rehearsed with the draft finish function: a `cohort_builder` step was reported, and the
other 3 steps were recorded.
- On a non-empty list the recorder runs the lazy sync once for the process.
- It then re-sends the same steps. Idempotency makes that safe.
- **Transport.** The recorder and sync client call a small `rpc(name, params)` port.
  - Production: the service-role `get_async_supabase_client().rpc(...).execute()`.
  - Real-DB tests: a psycopg connection to a throwaway database calling `SELECT name(...)`. That is a
    real database through another transport, not a mock.
  - The PostgREST transport is proven in the live cert.
- **Multi-worker and Celery.** Writes are per composition, and no Celery task runs the composer
  (`grep` of `src/tasks`, `src/workers`: none). The only shared-state writer is the sync, which is
  serialised by the advisory lock.

### 5.5 PII and retention

The repo has no content scrubber: `redact_query` only truncates (`src/utils/redaction.py:25–41`). This
design therefore **does not persist raw data values** where the loop does not need them. It does not rely
on truncation to hide them.

- **Everything except the query is either a registry-validated identifier, a number or an enum.**
  Anything the LLM or the data authored is dropped or reduced to structure.
  - **Identifiers are positional.**
    - Sub-questions are stored by index (0…n−1) and steps by `step_number`.
    - LLM-authored `sub_question_id` / `step_id` strings are never stored. `step_name` is `step_<n>`.
    - `depends_on_steps` and `$step` references are remapped to step numbers.
  - **Intents are normalized** to the decomposer's declared vocabulary (`decomposer.py:54`: CAUSAL,
    COMPARATIVE, PREDICTIVE, DESCRIPTIVE, EXPERIMENTAL). Anything else becomes `OTHER`.
    `SubQuestion.intent` is a free `str` (`composition_models.py:68`).
  - **Question text is never stored,** neither sub-question text nor the planner's reasoning.
  - **Tool names** are stored only if registered. The planner already raises on unknown tools
    (`planner._validate_plan`).
  - **Input maps are keyed only by the tool's declared parameter names** (`ToolSchema.input_parameters`).
    Undeclared keys are counted (`{"undeclared_params": k}`), never named. Values are stored as structure:
    - a string that is a column name **in the database catalog** → `{"type":"column","name":c}`. Frame
      membership is not trusted: `executor.py:1236–1288` accepts caller-supplied frames, whose column
      names are caller-authored.
      - The allowlist is the catalog: the column names of relations in schema `public`, developer-authored
        DDL.
      - **The check runs in the serializer, before anything leaves the process.**
        - The recorder fetches the allowlist at API startup (the same fire-and-forget startup task as the
          registry sync, §4), then hourly, through an RPC
          `composer_public_column_names()` (service_role only). It returns the distinct `attname` of
          `pg_attribute ⋈ pg_class` where `relnamespace = 'public'::regnamespace`, `attnum > 0` and
          `NOT attisdropped`. Measured as service_role on 2026-09-11: the catalog is readable, and
          `treatment` / `region` / `brand` are in it while `PT-0001` is not.
        - **While the allowlist is unavailable** (not yet fetched, or the fetch failed), no string is kept
          as a name. Privacy fails closed. The finish snapshot is serialized at finish time, so a
          composition that started before the fetch completed still records names, provided the fetch
          has landed by then.
        - No frame is needed, so the plan snapshot gets the same treatment as finished steps.
      - A name that is not in the catalog becomes `{"type":"str","len":n}` in the payload itself. That
        covers derived feature columns, pivoted value-columns, and caller-authored names.
      - The recording RPCs repeat the same catalog check on every `{"type":"column"}` entry they receive
        (defence in depth). A name that fails there is stored as `{"type":"str","len":n}`.
    - a `$step` reference → `{"type":"ref","step":n,"field":f}`, where `f` is kept only if it is a
      field of the producer's registered output model, else `null`;
    - any other string → `{"type":"str","len":n}`, without the value;
    - numbers and bools are kept (effect sizes, alpha, top_n): settings, not identifiers;
    - lists → `{"type":"list","len":n}`; dicts → `{"type":"dict","len":n}`, with **no keys**, because
      dict keys can be data values such as segment names; DataFrames →
      `{"type":"frame","rows":r,"columns":c}`.
  - **Output:** only the keys that are fields of the tool's registered pydantic output model, plus
    `{"other_keys": k}`. Dynamic keys (for example per-segment dicts) are never named.
  - **Errors: no error text is persisted, at step or episode level.** `error_type` (exception class name)
    plus `outcome_class` (plus `failed_phase` for episodes) is what is stored and aggregated.
    - Refusal text can name data values: #1574's disclosure lists the entity groups a frame covered.
    - Generic exception text can echo inputs.
    - The composer already returns every failed step's reason text in the total-failure chat answer
      (`composer.py:1033–1065`), generic exceptions included. That is a pre-existing exposure of the
      chat surface, recorded in §10. It is not a justification for copying the text into a new store.
    - `composition_steps.error_message` and `composer_episodes.error_message` stay NULL.
    - **What is lost:** the admin surface can say "gap_calculator: refused (ToolRefusalError)", but not
      *why*. Structured refusal reason codes would restore that safely. They need changes in
      `tool_registrations.py`, which LANE-2015 and LANE-2016 are editing now, so they are a §10
      follow-up.
  - **The single serializer** (`learning_recorder.to_record()`) is the only path to the RPCs, and the
    RPC-side re-check is a second guard. The test plants a sentinel string in every LLM- and data-authored
    position:
    - question text, intent, sub_question_id, step_id;
    - a frame column name that is also used as a parameter value;
    - an undeclared parameter key, a string value, a dict key, a `$step` field;
    - a dynamic output key;
    - refusal, input-error and generic exception messages.

    It asserts the sentinel is absent from the serialized payload (what crosses the network) and from
    every persisted column after a real-DB round trip. A second case sends a hand-built payload whose
    `{"type":"column"}` entry names the sentinel, and asserts the RPC stores it as length-only.
- **Query text is stored, `redact_query(query, 500)`.** This is a deliberate decision, not a redaction
  claim.
  - The query is needed for the admin failure list and the reuse audit.
  - It is already persisted in four service-role stores, measured 2026-09-11:
    - `classification_logs.query_text`, full, up to 2,590 chars;
    - `audit_chain_entries.query_text`, full: 23 tool_composer rows, up to 532 chars, passed at
      `composer.py:286`;
    - episodic `raw_content.query`, 500 chars;
    - `chatbot_messages.content` for user turns, up to 2,590 chars.
  - `composer_episodes` adds no new exposure class, and access is the same (service_role only).
  - A platform PII scrubber is recorded in §10.
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

- **`get_tool_reliability(p_days int DEFAULT 30, p_include_synthetic boolean DEFAULT true)`**, SQL
  `STABLE`, `service_role` only. One row per active tool:
  - declared latency and current `version`;
  - `n_invoked`, `n_succeeded`;
  - `n_refused` (refused + input_rejected);
  - `n_health_failures` (timeout + error);
  - `n_health = n_succeeded + n_health_failures`, the denominator of the reliability rule (§7.1);
  - `n_retried` (succeeded with attempts > 1);
  - `n_synthetic`;
  - p50 / p95 latency of **succeeded** invocations;
  - `last_executed_at`, most common health `error_type`.

  Rehearsed with the draft (before `n_health` / provenance were added): `causal_effect_estimator 1/1/0/0
  p50 900`, `gap_calculator 1/0/1/0` (refused, not failed).
- **Provenance.** `p_include_synthetic = false` excludes rows whose episode ran under a deployment that
  includes synthetic substrate. Callers pass `deployment_includes_synthetic()`, the rule every
  provenance-gated reader follows (`src/repositories/provenance.py:95–110`):
  - On this showcase deployment (`E2I_INCLUDE_SYNTHETIC=true`, measured in `e2i_api`) the synthetic runs
    *are* the operational runs, so they count.
  - On a real-RWD deployment they are excluded from the planner signal and still shown as `n_synthetic`.
- **Versions.** `tool_version` is recorded on every performance row. It does not filter the rule,
  because versions are not bumped today (every live row is `1.0.0` or `4.4.0`), so a version split would
  be a label with nothing behind it. The 30-day window bounds how long a replaced implementation's
  history can count.
- **`v_tool_reliability`** is recreated as `SELECT * FROM get_tool_reliability(30, true)`. That keeps
  the designed name, with the 30-day window the reliability rule in §7 uses.
- **`v_composition_success_rate`** is recreated without the fan-out: daily counts by `outcome` (success,
  partial, failed, cancelled), unfinished, by `plan_source`, and p50/p95 total latency. Rehearsed:
  1 episode gives 1.
- **`v_active_compositions`** is recreated for runs not yet in a terminal status: `elapsed_ms` and
  `abandoned = last_activity_at < now() - interval '5 minutes'`.
  - **Why a heartbeat, not a deadline.** No deadline bounds every entry point. `compose_query()` awaits
    `compose()` directly (`composer.py:1118`), the executor's 120 s timeout applies per attempt
    (`executor.py:669–674`), and groups run sequentially (`:449–457`), so a legitimate composition can
    run past any fixed age. The 180 s SLA exists only on the orchestrator path (`router.py:298`).
  - **Heartbeat.** While a composition is in flight, the recorder runs a heartbeat task that calls
    `composer_record_heartbeat(p_seed)` every 60 s. The task starts with the start write and stops at
    finish or cancel.
  - `abandoned` therefore means **five missed heartbeats**: the worker died, or its event loop was
    blocked for five minutes. A slow but live run is never marked.
  - A late heartbeat or finish on an `abandoned` run simply bumps `last_activity_at` again. `abandoned`
    is derived, never stored.
- **Dropped:** `update_tool_registry_metrics()`.
  - **Intent:** planner-visible reliability. That is now served by `get_tool_reliability` with n and
    failure classes.
  - **It is harmful as written:** no minimum n, so one failed run writes `success_rate = 0`.
  - **It has never had a caller.**
- **Dropped:** `get_tool_execution_order(text[])`.
  - **Intent (verified):** `v4.2_IMPLEMENTATION_TODO.md` @3e1c70cf4: "Implement `optimize_plan()` using
    `get_tool_execution_order()` — Call Supabase function for topological sort — Identify parallelizable
    tool groups".
  - **History:** its only other references are "post-migration test" comments in ml/013 L1230 and ml/027
    (`b23c17355`).
  - **Why the function cannot serve that intent:** it orders *tool names* by *tool-level* dependencies.
    A plan is a list of *steps*, which can call one tool twice or use a consumer without its producer.
    Only the plan's `depends_on_steps` describes the real graph.
  - **The intent is real and currently unmet at runtime:** `ExecutionPlan.get_execution_order()` returns
    the LLM's `parallel_groups` unchecked (`models/composition_models.py:167–172`, no validation in
    `planner._validate_plan`). §7.4 repairs that in-process, at step level, and the function is dropped.
  - It also returns `{NULL}` for root tools (§2.3) and is EXECUTE-able by `anon`.
- **Untouched:** `v_classification_accuracy` (#1341).

## 7. Feedback into behaviour

### 7.1 Reliability rule, calibrated on planted truth at production n

The rule is a pure function in the new `src/agents/tool_composer/reliability.py`. It is the one place
that turns counts into a verdict. The admin API calls it, and so does the planner integration once
§7.2's experiment gate passes.

**Denominator.** `n_health = n_succeeded + n_health_failures`. Refusals never enter it, so twenty
refusals cannot qualify a tool for a verdict. The health-failure rate is `n_health_failures / n_health`.

| Verdict | Condition |
|---|---|
| `no_runs` | n_invoked = 0 |
| `too_few_runs` | n_health < 20. It covers tools that only refused: "no health evidence yet". |
| `caveat` | n_health ≥ 20 and Wilson 95% **lower** bound of the health-failure rate ≥ 10% |
| `reliable` | n_health ≥ 20 and Wilson 95% **upper** bound < 10% |
| `inconclusive` | otherwise |

- Refusals are reported next to the verdict ("5 refused: data could not answer") and never enter the rate.
- **Measured latency is shown only at `n_succeeded` ≥ 20** (admin surface, §8). Below that, the measured
  fields are `null` with the verdict "too few successful runs (n=k)". Declared latency is a separate,
  labelled field and is never substituted into a measured field.

**Calibration** (scratch simulation, 20,000 Bernoulli draws per cell, seed 7; n here is `n_health`).
A healthy tool is p = 0.02 or 0.05; a failing tool is p = 0.20 or 0.30. **The n = 5 row is the
*ungated* Wilson rule, shown only to justify the floor. The shipped rule returns `too_few_runs` at
n = 5.**

| n | P(caveat), p=0.02 | P(caveat), p=0.05 | P(caveat), p=0.20 | P(caveat), p=0.30 | P(reliable), p=0.02 |
|---|---|---|---|---|---|
| 5 | 0.004 | 0.023 | 0.261 | 0.477 | 0 |
| 20 | 0.000 | 0.002 | 0.371 | 0.760 | 0 |
| 40 | 0.000 | 0.001 | 0.563 | 0.945 | 0.449 |
| 100 | 0.000 | 0.000 | 0.874 | 0.999 | 0.950 |

- **False caveats are rare at every n** (≤ 0.2% for a 5% tool at n ≥ 20).
- **A 30% failing tool is caught** 76% of the time at n = 20 and 94.5% at n = 40.
- **The n ≥ 20 floor removes the n = 5 false-caveat tail** (2.3% at p = 0.05).
- **At production n** (≤ 7 invocations per tool so far) every tool reads `too_few_runs`. A test pins
  that.

### 7.2 Measured reliability into planning: built, experiment-gated

**Measured.**
- At production volume, the §7.1 rule returns `too_few_runs` for every tool, so no planner-visible
  reliability signal can exist for months.
- Whether a reliability line in the planning prompt changes the LLM's tool choice has never been
  measured. A caveat nobody has shown to change a decision is a label.
- `ExecutionPlan.estimated_duration_ms` (`planner._estimate_duration`) has **no consumer**. grep:
  its only other use copies it into an adapted plan (`planner.py:323`). Feeding it measured latency
  would change a number nobody reads, so this design does not.

**What this lane builds.**
1. **`ToolReliabilityReader`** (`src/agents/tool_composer/reliability.py`):
   - calls `get_tool_reliability(days, deployment_includes_synthetic())` through the service client;
   - `days` defaults to 30, which the planner integration uses. The admin API passes the requested window;
   - caches per process for 300 s, keyed by `(days, include_synthetic)`;
   - fail-open: returns `{}` on any error.

   The admin API uses it (§8).
2. **The planner integration, behind `TOOL_COMPOSER_RELIABILITY_IN_PLANNER` (default off).**
   - One shared formatter serves the planner prompt (`planner.py:371`) and the DSPy formatter
     (`dspy_integration.py:189–225`).
   - On `caveat` it adds exactly one line per caveated tool: "Reliability caveat: 6 of 24 runs failed on
     tool errors (timeout)". The flag controls nothing else.
   - The declared "Avg execution" line is unchanged.
     - **Measured latency is not substituted.** The LLM reading that line is the only planning consumer of
       latency: `estimated_duration_ms` has no reader, and there is no deadline-aware planning.
     - Whether a different number there changes the LLM's choice is as unmeasured as the caveat. It would
       need its own experiment arm, which this lane does not run.
     - Measured latency is shown on the admin surface (§8), and §10 records prompt substitution as
       deferred.
   - With the flag off, the prompt is byte-identical to today, and a test pins that.
3. **The experiment that decides the flag** (`scripts/benchmarks/tool_composer/reliability_caveat_experiment.py`).
   It is pre-registered in the script's docstring before any evaluation call.
   - **Items.** K ≥ 30 fixed real decompositions. Each has a sub-question that two registered tools can
     each answer (for example `psi_calculator` / `distribution_comparator`, or `risk_scorer` /
     `propensity_estimator`), plus a real frame from the cohort / KPI loaders the entry points use. The
     item set is frozen, and its hash is recorded.
   - **Target selection from a separate pilot.** A pilot of P = 10 different decompositions, run on
     today's prompt only, fixes per item family which tool is caveated: the one the planner picks more
     often. The pilot items are excluded from evaluation. This removes the selection bias of choosing
     the target from the evaluation baseline.
   - **Arms, paired per item.** A = the flag off, today's prompt. B = the flag on, with a planted
     `caveat` verdict for the target tool. That is the complete enabled prompt, byte-for-byte what
     production would send. Arm order is randomised per item with a fixed seed.
   - **Primary outcome, paired analysis.** Per item, whether the plan selects the target tool, in A and
     in B. Test: exact McNemar on the discordant pairs, one-sided. **Pass requires both:**
     - (b − c) / K ≥ 0.30, where b counts items that picked the target in A only and c items that picked
       it in B only;
     - p < 0.05.
   - **Validity outcomes, executable, per arm.**
     - Every returned plan goes through `planner._validate_plan` and the §7.4 execution-order checks.
     - It is then **executed** with the real `PlanExecutor` on the item's real frame.
     - Per item and arm, two binary outcomes. **Invalid:** `PlanningError`, a §7.4 validator rejection,
       or any `plan_defect` / `not_registered` step. **Total failure:** zero succeeded steps.
     - Every frozen item stays in the denominator. An item whose planner call errors counts as invalid
       in that arm.
     - **Pass also requires, on observed counts, each gate separately:**
       - invalid(B) ≤ invalid(A);
       - total_failure(B) ≤ total_failure(A);
       - median succeeded steps(B) ≥ median(A).

       No significance test is used for these guards. They are "not observed worse" gates, stated as
       such, not a non-inferiority claim.
   - **Cost.** Planner calls: P + 2K ≥ 70, thinking disabled, about 1,000 output tokens each
     (`composer.py:71–90`).
     - Plan executions run on the droplet against local services only. `model_inference` calls the local
       BentoML container: `BENTOML_SERVICE_URL=http://bentoml:3000` (`docker/docker-compose.yml:72`,
       read by `src/api/dependencies/bentoml_client.py:60`; measured in `e2i_api`). That is no external
       spend.
     - Executions are sequential, and `free -m` is checked before each item. The run stops below
       1500 MiB available.
     - Plans are neither restricted nor filtered: both arms offer the production tool set.
     - The planner spend needs the owner's authorization (O2).
   - **Result handling.** The result JSON and a summary are committed under `docs/demos/results/`. The flag
     default flips to on only in a follow-up commit that cites a passing result. A failing result leaves
     the flag off and records which condition failed.

**Feedback-loop guards, whatever the flag:**
- no automatic removal of a tool from the offer;
- no reranking;
- no circuit-breaker seeding;
- no change to `_get_fallback_mapping`.

A caveated tool that the LLM avoids gets fewer runs. If its `n_health` then drops below the floor, the
caveat disappears. That oscillation is benign, and the 30-day window bounds it.

### 7.3 Similar-composition reuse

**Decision:** `episodic_memories` stays the canonical similarity store (§5.1).
`find_similar_compositions()` (SQL), `composer_episodes.query_embedding` and its ivfflat index are
dropped, because their design intent (reuse a successful plan for a similar query) is served by the
live episodic path. The SQL function has never had a caller, and the column has never held a row.

**Intent of the reuse paths (verified).**
- G1/G2 (`fee8bc3e0`: "episodic memory for plan reuse") and #889 (the reference context "never
  fired") both intend to hand the planner **tool sequences that worked** for a similar question. The
  planner's own header reads: "The following successful compositions may inform your planning"
  (`planner.py:555`).
- G6 (`fee8bc3e0`: "plan similarity matching") intends to **skip planning** for a structurally similar
  decomposition.

Two measured defects break that intent. The loop fixes both.

**1. The plan cache reuses plans that failed.**

`ToolPlanner.plan` (`planner.py:212–228`):
- calls `get_similar_plan`: intent-set Jaccard plus dependency similarity ≥ 0.8, 15-minute TTL, process
  singleton;
- returns `_adapt_cached_plan`, which copies the cached steps and tool mappings (`planner.py:303–332`) and
  bypasses the LLM planner. The outcome-hint and treatment guards then run over the copied steps
  (`planner.py:221–228`), so a KPI outcome or treatment binding can still change. Everything else is reused
  as it was.
- caches every plan at planning time (`planner.py:293`), before execution, whatever happens next.

**Eligibility (verified):**
- the LLM planning path only: a KPI question that `_build_kpi_causal_plan` resolves never calls
  `ToolPlanner` (`composer.py:341–350`);
- a cached entry whose similarity is ≥ 0.8;
- a cached step count equal to the new sub-question count (`planner.py:314`);
- the same worker process.

The decomposition is **not** cached: `DecompositionCache` has no caller outside `cache.py`, measured by
grep. Every question is freshly decomposed, and eviction does not need to touch decompositions.

Under those conditions, a plan that failed (0 tools succeeded, a `plan_defect`, a `not_registered` step)
is served again to the next similar question in that worker for 15 minutes.

**New behaviour:**
- `PlanSimilarityCache` gets `evict(signature_key)`.
- `ExecutionPlan` carries the `plan_cache_key` it was stored or matched under.
- After execution, the composer evicts that key when the composition fails, or when any step's class is
  `plan_defect` / `not_registered`.
- `plan_source` is `llm`, `plan_cache` or `kpi_deterministic`, and the episode records it, so cache reuse
  and its outcomes are visible in the admin surface.
- The G6 intent ("plan similarity matching" to skip planning for similar work, `fee8bc3e0`) is kept:
  successful and partial-without-defect plans stay cached.
- **Deterministic and testable without an LLM:** a real `ToolPlanner` with the real cache and real
  `DecompositionResult` objects that meet the eligibility conditions. A failed trace evicts, and the next
  `plan()` for a similar decomposition does not return the cached steps. The positive control is a
  succeeded trace, after which it does.

**2. Episodic references recommend failed sequences.**

`memory_hooks.find_similar_compositions` keeps rows where `raw_content.success`, and `success` is true for
PARTIAL (`composer.py:488`). The rendered "Tools used" list therefore includes the tools that failed. The
two live rows repeat `gap_calculator`.

**New behaviour:**
- The hook hydrates each reference with its recorded steps: one `composition_steps` read by
  `composition_id` for the ≤ 3 references.
- `_format_episodic_context` renders:
  - "Tools that worked: causal_effect_estimator, cate_analyzer" as the recommended sequence, made only of
    `succeeded` / `cache_hit` steps in step order;
  - "Did not work for that question: gap_calculator (refused: data could not answer), rank_drivers
    (skipped: dependency unmet)".
- A reference with zero succeeded steps is dropped from the context.
- **References with no recorded steps** (the 2 live pre-loop rows, and any future row whose step writes
  were lost):
  - `raw_content.tools_succeeded == raw_content.tools_executed` means every tool worked. It renders as
    today.
  - Otherwise the reference cannot say which tools failed, so it is **dropped** rather than recommended.
  - Both live rows are `partial_success`, so both are dropped. Backfill is impossible: no per-step
    record of those runs exists anywhere (§2.1).
- **Tests:**
  - a real hydrated PARTIAL reference yields a recommended sequence without the refused tool;
  - a zero-success reference yields no context;
  - the two legacy raw_content shapes (copied from the live rows) are dropped;
  - a legacy all-success shape renders.
- **Unverified assumption (labelled):** that the LLM planner avoids the listed "did not work" tools. As in
  §7.2, only a paired LLM measurement would prove it. The deterministic part (what the planner is told to
  reuse) is what this lane guarantees.

### 7.4 Dependency-aware execution order (the intent of `get_tool_execution_order`)

**Measured (probe on `models/composition_models.py`, 2026-09-11).** `ExecutionPlan.get_execution_order()`
returns the LLM's `parallel_groups` unchecked:
- groups `[["a"]]` for steps a, b → `[["a"]]`: **step b never executes**;
- groups `[["a","b"]]` where b depends on a → `[["a","b"]]`: the consumer runs **in parallel with** its
  producer, and its `$a.…` reference cannot resolve;
- groups `[]` → one group per step **in list order**. That is correct only when producers precede
  consumers: steps `[b, a]` with b depending on a run b first, and b's reference cannot resolve.
- **Duplicate step ids** are not rejected. `_validate_plan` does not check them, `_check_cycles` collapses
  them into a dict (`planner.py:932`), and `ExecutionPlan.get_step()` returns the first match
  (`composition_models.py:155`). Two different steps named `a` execute the first one twice (codex
  iteration-4 probe).

`planner._validate_plan` checks tool names and dependency ids, not groups (`planner.py:662–712`). Cached
plans reuse the same groups (`planner.py:324`). The design intent ("dependency-aware execution DAG …
identify parallelizable tool groups", §6) is therefore unmet at runtime.

**New behaviour.**
1. **Duplicate step ids are a plan error.** A new `ExecutionPlan` model validator raises on duplicate
   `step_id`s, on a `depends_on_steps` entry naming no step, and on a dependency cycle. Every plan passes
   through it wherever it was built: LLM planner, cache adaptation, KPI builder, and directly
   constructed plans. It is the execution-boundary check.
   - A planner-built plan that fails it raises `PlanningError`, recorded as `failed` in phase `plan`.
2. `get_execution_order()` accepts the given groups only when:
   - every step appears exactly once;
   - no group names an unknown step;
   - every `depends_on_steps` entry sits in a strictly earlier group.

   Otherwise, **including when groups are empty**, it returns the topological levels of
   `depends_on_steps`: each level is the set of steps whose dependencies are all in earlier levels, in
   plan order within a level. The plan records
`execution_order_repaired` and the violated condition. The episode's `tool_plan` carries both, so the
frequency becomes measurable.

The deterministic KPI plan's groups (`composer.py:829–869`) satisfy all three conditions, so it is
unchanged. A test pins that.

### 7.5 In-memory G8

`PlanExecutor.update_tool_performance` (`executor.py:1312–1375`) is deleted with its mock tests
(`test_executor.py` ~L1611–1685).
- It has no production caller.
- It mutates the process-wide `ToolSchema.avg_execution_ms` from one executor's EMA. Wired in, one
  request's stats would leak into every later plan.
- Its intent (learned latency) is partly served: measured latency per tool is recorded and shown on the
  admin surface (§6, §8). Substituting it into the planning prompt is deferred until its own experiment
  arm shows it changes the LLM's choice (§7.2, §10).

`ToolFailureTracker` stays. It drives the in-request circuit breaker and retry policy.

## 8. Observability surface

**An existing surface is extended, not a new page:**
- The Admin page, **Observability** tab (`frontend/src/pages/Admin.tsx:49–80`,
  `components/admin/ObservabilityTab.tsx`), which has a days selector and stat cards and is admin-only.
- Its API sits next to `GET /api/admin/observability/llm-usage` (`src/api/routes/admin.py:349`).

**API.** `GET /api/admin/observability/tool-composer?days=30` (`require_admin`) returns:
- `compositions`: counts by outcome (success, partial, failed, cancelled), unfinished, abandoned, by
  `plan_source`, and p50/p95 total latency;
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

- **Red first, real database, no mocks.** The prod `postgres` database is never written by tests.
  Everything runs against throwaway databases on the droplet's own Postgres 15.8, the
  `scripts/test_migration_idempotency.sh` precedent (`DROP/CREATE DATABASE` as `supabase_admin`).
  - **Upgrade path, exactly as a deploy runs it.**
    1. `pg_dump --schema-only` of prod (a read-only dump) plus a data copy of `public.schema_migrations`,
       `tool_registry` and `tool_dependencies`, restored into `learning_loop_upgrade`. Every historical
       enum, CHECK, grant, default ACL, ml/036 object and the ledger arrive as prod has them.
       - The public schema references `extensions.*` and `auth.*`: 41 lines in a read-only
         `pg_dump -s -n public` probe on 2026-09-11, a 1.07 MB dump.
       - The restore therefore first creates the `extensions` schema with `pgcrypto` / `uuid-ossp`, and
         includes the `auth` schema definition (`-n auth`, schema only).
       - The first plan task proves the restore is clean before any test depends on it.
    2. **Failure first, on a fresh fixture.**
       - Point the runner at a temporary copy of the repository's `scripts/` and `database/` trees in
         which 041 ends with `SELECT 1/0;`, then run it in URL mode (`SUPABASE_DB_URL` = the throwaway
         database).
       - Expected: 039 applied un-wrapped and recorded, 040 applied wrapped and recorded, 041 failed.
         No 041 ledger row and none of 041's objects are present, because body and ledger row share one
         transaction (`scripts/run_migrations.sh:181–184`).
    3. **Then the real files.** Run the real `scripts/run_migrations.sh` against the same database. It must
       report exactly 041 pending and apply it. Run it again: 0 pending.
    4. **Branch assertions.** The test re-implements the runner's detector exactly: strip `--` comments,
       then match `ALTER TYPE … ADD VALUE`, `CONCURRENTLY` or a bare `COMMIT;` (`run_migrations.sh:175–177`).
       It asserts 039 matches and 040/041 do not, so neither can silently fall into the un-wrapped
       branch.
    5. **Idempotent re-apply.** Apply 039 (plain `psql`) and 040/041 (`psql --single-transaction`)
       directly a second time; no error and no row changes.
    6. Assert objects, grants and the sync results.
  - **Functional SQL tests** (sync guards, idempotent step/finish receipts, the dependency-endpoint
    check, `RETURNING (xmax = 0)` counts, reliability denominators, provenance filter) run on that
    upgraded database through psycopg transactions.
  - **Concurrency test:** two connections call `sync_tool_registry` at the same time. Both succeed, one
    reports all-zero changes, and the final rows equal the payload.
  - Opt-in through `E2I_DB_INTEGRATION=1`, as in `tests/integration/test_issue_825_schema_drift_realdb.py`,
    so CI (no DB) skips the tests.
  - **Not provable in a throwaway database:** PostgREST's schema-cache discovery of the new RPCs.
    PostgREST serves only the prod database. That is live-cert step 1.
- **Executor outcome classes.** Tested by running the real `PlanExecutor` on real registered tools and
  real frames:
  - a single-brand frame for a `gap_calculator` refusal;
  - a planned `$step_x.missing` reference;
  - a failing upstream for dependency unmet;
  - a second identical run for a cache hit;
  - an unregistered name;
  - a registered slow sync callable for `SyncToolTimeout`, and a registered slow async callable for the
    `wait_for` timeout, both classed `timeout`;
  - a callable that fails once and then succeeds: `succeeded`, attempts = 2.

  Test tools are registered through `snapshot`/`restore_snapshot`.
- **Per-step persistence under cancel.** A real plan whose parallel group holds one fast tool and one slow
  tool. The test cancels the execute task after the fast tool finished and while the slow one runs. It
  asserts:
  - the fast sibling's step is recorded;
  - the slow one is not;
  - the episode is `cancelled` in phase `execute`.

  The same shape with the fast tool in an earlier group is also covered.
- **Recorder failure modes** (psycopg transport against the throwaway database):
  - start write fails (database stopped for the start call only), then phase, steps and finish land: one
    episode with every field;
  - a phase write exhausts its retry: finish restores sub-question indices and intents, `tool_plan`,
    `plan_source`, groups
    and latencies;
  - audit identity: no audit service, audit start raising, and a caller-supplied stale
    `context["audit_workflow_id"]`. All three give an episode whose `audit_workflow_id` is NULL, or the
    run's real id, never the stale one;
  - a lost response re-sent: identical receipts, no duplicate rows;
  - finish twice: `already_terminal`;
  - shutdown drain: the pending set is flushed within 5 s;
  - liveness: with a test heartbeat period of 1 s and a 5-period window, a composition whose step runs
    10 s is never `abandoned`. A composition whose heartbeat task is cancelled mid-flight (the worker-death
    stand-in) is `abandoned` after the window.
- **No latency on the user path.** An unreachable transport (connection refused and a 10 s hang, both
  cases) under a composer call with a 1 s deadline. Composition time is unchanged against a no-recorder
  run within 50 ms, and no exception reaches the caller.
- **No raw values persisted.** The sentinel test of §5.5, through the real serializer and a real-DB round
  trip.
- **Plan-cache eviction.** A real `ToolPlanner` and a real cache, with non-KPI `DecompositionResult`
  objects that meet the §7.3 eligibility. A failed composition evicts, and the next similar decomposition
  does not get the cached steps. A succeeded composition keeps them (positive control). A
  `kpi_deterministic` plan never touches the cache.
- **Episodic references.** The four shapes listed in §7.3.
- **Execution order.** These shapes from §7.4 plus a valid plan and the KPI deterministic plan:
  - omitted step;
  - consumer grouped with its producer;
  - empty groups with a consumer listed before its producer;
  - duplicate step ids, which the model validator rejects;
  - a dependency on an unknown step and a cycle, which it also rejects. Invalid groups are repaired to
  topological levels and flagged; valid ones are returned unchanged. A real `PlanExecutor` run over the
  omitted-step plan executes every step.
- **Planner integration flag.** With `TOOL_COMPOSER_RELIABILITY_IN_PLANNER` unset, the planner and DSPy
  prompts are byte-identical to the pre-change prompt for the same registry. With it set and a `caveat`
  verdict, the caveat line appears.
- **Caveat experiment** (§7.2). The script's pure analysis functions (McNemar, the pass rule) are unit
  tested on constructed paired tables with known answers. The run itself happens only after the owner
  authorizes the spend (O2), and its result file and the flag decision are committed with it.
- **Composer wiring.** An opt-in real-LLM run (`E2I_LIVE_LLM=1`) records to the throwaway database
  through the psycopg transport, plus the live cert.
- **Reliability rule.** The planted-truth test reproduces the §7.1 table bounds: false caveat ≤ 1% at
  p = 0.05 for n ∈ {20, 40, 100}; detection ≥ 90% at p = 0.30, n = 40; `too_few_runs` for every n < 20.
- **Grants.** For every created or recreated object, `has_table_privilege` /
  `has_function_privilege('anon'|'authenticated', …)` is false, and service_role is true.
- **Live cert on the deployed image** (record `image_sha: <sha>`):
  1. **Sync.**
     - Boot logs show a sync with inserted=4 (`cohort_builder`, `cohort_validator`, `cohort_statistics`,
       `model_inference`) and the second worker with all-zero changes. That also proves PostgREST
       discovered the RPC.
     - DB: 20 active tools, 13 dependencies.
  2. **Recording.** A real chat composition produces one terminal `composer_episodes` row, steps equal to
     the trace's step count, and `tool_performance` rows equal to the invoked steps.
  3. **Refusals.** A refusal-producing composition is recorded as `refused` and counted in `n_refused`,
     not `n_health_failures`.
  4. **Admin surface.** The endpoint and tab show "Too few runs to judge".
  5. **Dormant gate.** The planner prompt log shows the declared latency.
  6. **Access.** PostgREST with the anon key refuses the RPCs and views.
  7. **Latency.** Compose wall time is within noise of the pre-deploy baseline for the same questions.
     Recorder write durations are logged at DEBUG, with p50 and max reported.
  8. **Coverage, by identity.**
     - The cert script gives every composition it triggers a unique session_id and records it with the
       request time. `composer_episodes.session_id` joins that list independently of the audit chain.
     - Over the cert window, three sets are compared:
       - the cert's triggered list;
       - `audit_chain_entries` `workflow_start` rows for `tool_composer`, by `workflow_id`;
       - `composer_episodes`, by session_id, and to audit rows by `audit_workflow_id`.
     - Every triggered composition must have an episode. Where its audit row exists, the episode's
       `audit_workflow_id` must equal that row's workflow_id.
     - A composition present in one store and missing from the other is listed with
       `composer_record_failures_total` and the recorder's WARNING lines.
     - Audit writes are also fail-open (`composer.py:272–294`), so a miss is attributed to whichever store
       lacks the row. The triggered list is the independent denominator.
  9. **Plan cache, observed, not forced.** The live cert reports the `plan_source` distribution and each
     `plan_cache` episode's outcome.
     - It does not try to force a cache hit live. Eligibility needs the same worker, a similar fresh LLM
       decomposition and equal step and sub-question counts (§7.3), so a forced pair proves nothing
       either way.
     - The eviction behaviour itself is certified by the deterministic integration test above.
- **Rollback.**
  - Code: revert the PR. The recorder and sync are fail-open, so a code-only revert is safe with the
    migrations left in place.
  - DB: `database/ml/rollback_041.sql` then `rollback_040.sql`.
    - They recreate the dropped function, trigger, column, index and views from their ml/013 definitions.
    - They restore the six-agent CHECK only when no row violates it. Otherwise they raise with the
      offending rows, and the operator decides.
    - They drop the new functions and columns. Recorded rows in the new columns are lost.
  - ml/039 has no rollback: `COHORT` cannot be removed from an enum (PG), and it is harmless.
  - Rollback files are excluded from auto-apply by `run_migrations.sh`.

## 10. Non-goals

- **Linking `classification_logs.classification_id` to episodes.** The classifier writes fire-and-forget
  and returns no id to the dispatcher (`intent_classifier.py:826`). The column stays NULL.
- **"Running" rows per step** (PENDING→EXECUTING updates). Each step is written once, when its result
  exists. Phase status and `abandoned` cover in-flight diagnosis.
- **A circuit breaker shared across requests or workers.**
- **`chatbot_analytics.tools_succeeded` / `tools_failed` / `tool_composer_used`,** which the chat writer
  never fills (`copilotkit.py:1859–1905`). That is a chat-analytics defect, separate from this loop.
- **Structured refusal reason codes** on `ToolRefusalError` / `ToolInputError`. They would let the admin
  surface say *why* a tool refused without storing free text (§5.5). They need changes in
  `tool_registrations.py`, which LANE-2015 and LANE-2016 are editing, so they are a follow-up issue to
  file after those lanes merge.
- **The chat surface's existing exposure of failure text.** The composer's total-failure answer already
  returns every failed step's error text, generic exceptions included (`composer.py:1033–1065`). Found in
  review, this is pre-existing, and it should be filed with the reason-code follow-up.
- **Procedural-memory `tool_composition` patterns.**
- **`tool_performance` rows from non-composer callers** (`called_by='agent'|'direct'`); none exist.
- **A retention/purge job** (§5.5).
- **Measured latency in the planning prompt or plan estimates.** The only planning consumer of latency is
  the LLM reading the declared "Avg execution" line. Its effect is unmeasured (§7.2). It needs its own
  experiment arm, or a deadline-aware planner, first.
- **A platform PII scrubber for query text.** Query text is already persisted in four stores (§5.5).
  - Changing `redact_query` alone would **not** cover them. All four bypass it:
    - `audit_chain_entries` receives `query_text=query` (`composer.py:286`);
    - `classification_logs` receives the raw query (`intent_classifier.py:115`);
    - episodic memory slices `query[:500]` (`memory_hooks.py:408`);
    - `chatbot_messages` stores the raw human message (`copilotkit.py:3603`, inserted at `:1670–1684`).
  - The follow-up must both implement scrubbing and wire it into each persistence path. This lane's
    composer_episodes writer goes through `redact_query`, so it needs no change then.
  This lane stores no raw parameter values and no generic exception text, so it adds no exposure class.

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
- **O2. Authorize the caveat experiment's LLM spend (§7.2).**
  - What it is: P + 2K ≥ 70 planner calls on the production provider, thinking disabled, plus local plan
    executions.
  - The planner integration is built behind a default-off flag either way. Only a passing, committed
    result turns it on.
  - **Recommendation:** authorize it, but run it once any tool reaches `n_health` ≥ 20. Before that the
    rule cannot fire, so a pass would change nothing in production.
  - The fact that would reverse this: the owner wants no automatic prompt changes from telemetry at all,
    #1341-style human-gated authority. In that case the flag and the experiment are removed, and
    verdicts go only to the admin surface.
- **O3. Dropping designed DB objects:** `find_similar_compositions()`, `update_tool_registry_metrics()`,
  `get_tool_execution_order()`, `composer_episodes.query_embedding` plus the ivfflat index,
  `trg_log_step_performance`, `tool_registry.success_rate`.
  - **The three functions and the column** (`find_similar_compositions`, `update_tool_registry_metrics`,
    `get_tool_execution_order`; `query_embedding`) have had no caller or writer since they were created.
    Their intents are served by live paths: episodic memory (§7.3), `get_tool_reliability` (§6), and
    step-level execution order (§7.4). `update_tool_registry_metrics` has a measured defect (no minimum
    n) and so does `get_tool_execution_order` (`{NULL}`), both in §2.3.
  - **The trigger** `trg_log_step_performance` is not "unused". It is attached, and it would fire, but its
    semantics are measured wrong: it misses inserted terminal steps and double-counts COMPLETED→FAILED
    (§2.3). Its intent (a performance row per finished step) is kept in the step RPC (§5.2).
  - **`tool_registry.success_rate`** holds seed defaults that look measured (§2.1), and nothing reads it.
  - **Recommendation:** drop.
  - Rollback files recreate them. The owner may prefer to keep the vector column for a future
    composer-specific similarity search.
