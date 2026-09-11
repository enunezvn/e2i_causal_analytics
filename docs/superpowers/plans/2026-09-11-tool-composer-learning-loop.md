# Tool-Composer Learning Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Connect the designed-but-never-wired tool-composer learning loop, so that:
- the DB tool registry matches the running code on every boot, with no migration per schema change;
- every composition and every finished step is recorded, with failure classes and without raw data values;
- per-tool reliability is measured with a rule calibrated on planted truth;
- failed plans stop being reused;
- every planned step runs in dependency order;
- an admin can read the verdicts;
- reliability reaches the planner prompt only after a pre-registered experiment says it changes the choice.

**Architecture:** Spec `docs/superpowers/specs/2026-09-11-tool-composer-learning-loop-design.md`. Read §2–§7
before Task 1.
- **Database:**
  - three migrations: ml/039 (enum), ml/040 (registry sync), ml/041 (recording);
  - SQL functions called over PostgREST RPC with the service-role client.
- **Python:**
  - `registry_sync.py`: payload plus sync client;
  - `learning_recorder.py`: serializer, recorder, heartbeat;
  - `reliability.py`: verdict rule, reader and prompt formatter;
  - edits to `executor.py`, `composer.py`, `planner.py`, `memory_hooks.py`, `models/composition_models.py`,
    `cache.py`.
- **Admin:** an endpoint plus a section in the existing Observability tab.
- **Test transport:** every real-DB test runs against a throwaway database built from a schema-only dump of
  prod, through a psycopg `RpcPort`. Prod is never written by tests.

**Tech stack:** Python 3.12, FastAPI, supabase-py (async and sync), psycopg 3.3, PostgreSQL 15.8 (droplet
`supabase-db`), pytest, React 18 + TypeScript + vitest + Zod.

---

## Conventions for every task

- **Worktree.**
  - `W=/home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-learning-loop`, branch
    `claude/tool-composer-learning-loop`.
  - Every Bash call starts `cd $W && …`, because the shared cwd hops.
  - Assert `git -C $W branch --show-current` before every commit.
  - NEW COMMITS ONLY: no amend, reset or rebase of reported commits.
  - Never touch `.worktrees/lane-2015`, `lane-2016`, `lane-e-2005`, `lane-f1-2006` or `lane-i-2009`.
- **Python:** `PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python`.
  - Every test command: `cd $W && $PY -m pytest <paths> -q -p no:cacheprovider -n 0`. **Always `-n 0`.**
  - Before any import-based probe, assert `src.__file__` starts with `$W/`.
- **Real-DB tests** are opt-in: `E2I_DB_INTEGRATION=1 $PY -m pytest … -n 0`.
  - They create and drop only the throwaway databases `learning_loop_upgrade` and `learning_loop_fail`.
  - The password is read at runtime with `docker exec supabase-db printenv POSTGRES_PASSWORD` and never
    written to any file or log.
  - The direct DB port is `127.0.0.1:5433`. Port 5432 is the pooler, which answers "Tenant or user not
    found", measured 2026-09-11.
- **Lint:** `$PY -m ruff check <files>`, then `$PY -m ruff format --check <files>` (ruff 0.14.10).
  - Type-check changed files only: `$PY -m mypy --config-file pyproject.toml <files>`.
  - **Never whole-tree mypy on this box.**
- **Memory:** run `free -m` before Tasks 1, 4, 7, 10, 17 and 19. Stop and report below 1500 MiB available.
  Never `pkill -f`.
- **Frontend:** `cd $W/frontend && npx vitest run <paths>`, then `npm run typecheck`. Never `prettier --write`.
- **Codex per task,** after green:
  - Run a ralph-style loop around
    `cd $W && free -m && flock <scratchpad>/codex.lock codex exec -C $W -s read-only "<brief>" < /dev/null > <scratchpad>/lane-learning-taskN-iterK.txt 2>&1`
    until `VERDICT: ACCEPT`.
  - Briefs name only the task's files, cap codex at ≤ 25 commands and ≤ 40 output lines each, and include
    the CLAUDE.md pushback paragraph verbatim.
  - Read the whole output, interim narration included. Subscription only: on an auth error, STOP.
- **Commit footer on every commit:**

```
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018uHnM6oxkd8sepkqAY3s3o
```

- **Nothing is pushed until Task 19,** and only after the dispatcher's go. Never squash.

## Migration numbering (checked 2026-09-11)

- `origin/main` (`56f8b8589`) and every worktree end at `database/ml/037`.
- LANE-2015 (#2015) may add `038_tool_registry_schema_sync.sql` if it changes a tool schema before this lane
  retires the regime. **This lane takes 039, 040 and 041 and leaves 038 free.**
- Task 0 and Task 19 re-check `git ls-tree origin/main database/ml/` and both parallel worktrees. On a
  collision, renumber this lane's files upward in one commit and update the tests that name them.
- The gap is harmless: the runner applies files in sort order, and `017→020` already skips.

## Overlap with the parallel lanes

| File | This lane | LANE-2015 / LANE-2016 | Resolution |
|---|---|---|---|
| `src/agents/tool_composer/tool_registrations.py` | **not touched** | both edit | none needed |
| `tests/unit/.../test_registry_schema_drift_2003.py` | deletes section 3 only (Task 6) | 2015 may rely on it to regenerate 038 | if 2015 merges first, keep its 038 file (applied history); section 3 is still deleted |
| `scripts/generate_tool_registry_sync_migration.py` | deleted (Task 6) | 2015 may run it | same as above |
| `src/agents/tool_composer/tool_registry.py` | comment at L214–216 (Task 6) | 2015 may edit `DEPENDENCY_FIELD_MAPPINGS` rows | textual merge only; the sync payload reads whatever the mappings say |
| `database/ml/038_*` | not created | 2015 may create | numbering above |

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `tests/integration/tool_composer_learning_loop/__init__.py`, `conftest.py`, `_pg.py` | throwaway-DB fixture (schema dump + ledger/registry data), psycopg `RpcPort`, runner helpers | 1 |
| `tests/integration/tool_composer_learning_loop/test_fixture_sanity.py` | fixture restores prod schema faithfully | 1 |
| `database/ml/039_tool_category_cohort.sql` | `ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'COHORT'` | 2 |
| `database/ml/040_tool_registry_startup_sync.sql` | `valid_agent` CHECK → `e2i_agent_name`; drop `success_rate`, `update_tool_registry_metrics`, `get_tool_execution_order`; `sync_tool_registry()`; grants | 2 |
| `tests/integration/tool_composer_learning_loop/test_040_registry_sync.py` | sync guards, idempotency, concurrency, grants | 2 |
| `database/ml/041_composer_learning_loop_recording.sql` | §5.2 columns, drops, 6 RPCs, `get_tool_reliability`, 3 views, grants | 3 |
| `tests/integration/tool_composer_learning_loop/test_041_recording.py` | RPC contracts, reliability counts, views, grants | 3 |
| `database/ml/rollback_041.sql`, `rollback_040.sql` | rollbacks (never auto-applied) | 4 |
| `tests/integration/tool_composer_learning_loop/test_migration_runner.py` | failure-first real-runner replay, branch detector, re-apply, rollbacks | 4 |
| `src/agents/tool_composer/rpc_port.py` | `RpcPort` protocol + `SupabaseRpcPort` (async) | 5 |
| `src/agents/tool_composer/registry_sync.py` | payload, `sync_tool_registry_once()`, `fetch_column_allowlist()`, `learning_loop_startup()` | 5 |
| `tests/unit/test_agents/test_tool_composer/test_registry_sync_payload.py`, `tests/integration/.../test_registry_sync_client.py` | payload and real-DB sync | 5 |
| `scripts/generate_tool_registry_sync_migration.py` (delete), drift test section 3 (delete), `src/tool_registry/registry.py` (delete DB methods), `tests/unit/test_tool_registry/test_registry.py` (delete their tests), `src/agents/tool_composer/tool_registry.py` (comment), `docs/data/03-ML-PIPELINE-SCHEMA.md` §4 | retire the #2012 regime | 6 |
| `src/agents/tool_composer/models/composition_models.py`, `executor.py` | `StepResult` classes, attempts, cache_hit, error_type; per-step callback | 7 |
| `tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py` | real-tool classification, cancel with a finished sibling | 7 |
| `models/composition_models.py`, `planner.py` | `ExecutionPlan` validator + dependency-aware `get_execution_order()` | 8 |
| `tests/unit/test_agents/test_tool_composer/test_execution_order_repair.py` | omitted step, co-grouped consumer, reversed order, duplicates, KPI plan | 8 |
| `src/agents/tool_composer/learning_recorder.py`, `src/api/routes/metrics.py` | serializer, recorder, heartbeat, drain, failure counter | 9 |
| `tests/unit/.../test_learning_recorder_serializer.py`, `tests/integration/.../test_learning_recorder_realdb.py` | sentinel test; failure modes; liveness; latency | 9 |
| `src/agents/tool_composer/composer.py`, `planner.py`, `cache.py`, `agent.py`, `src/api/routes/chatbot_tools.py`, `src/api/main.py` | recorder wiring, plan_source, eviction, entry_point, startup task + drain | 10 |
| `tests/unit/.../test_composer_recording_wiring.py`, `test_plan_cache_eviction.py`, `tests/integration/.../test_composer_live_llm.py` | wiring, eviction, opt-in live LLM | 10 |
| `src/agents/tool_composer/memory_hooks.py`, `planner.py` | step-hydrated references; worked / did-not-work rendering | 11 |
| `tests/unit/.../test_episodic_reference_rendering.py`, `tests/integration/.../test_reference_hydration_realdb.py` | four reference shapes; hydration | 11 |
| `src/agents/tool_composer/reliability.py`, `planner.py`, `dspy_integration.py` | Wilson verdicts, reader, flag-gated caveat formatter | 12 |
| `tests/unit/.../test_reliability_rule.py`, `test_planner_reliability_flag.py` | planted-truth calibration; byte-identical flag-off prompt | 12 |
| `src/agents/tool_composer/executor.py`, `tests/unit/.../test_executor.py` | delete G8 `update_tool_performance` + its tests | 13 |
| `src/services/tool_composer_observability_service.py`, `src/api/routes/admin.py`, `src/api/schemas/admin_tool_composer.py` | admin endpoint | 14 |
| `tests/unit/test_api/test_routes/test_admin_tool_composer.py`, `tests/integration/.../test_admin_tool_composer_realdb.py`, `frontend/src/types/generated/api.ts` | endpoint tests; regenerated contract | 14 |
| `frontend/src/api/admin.ts`, `frontend/src/lib/api-schemas.ts`, `frontend/src/hooks/api/use-admin.ts`, `frontend/src/components/admin/ToolComposerSection.tsx`, `ObservabilityTab.tsx`, tests | admin UI | 15 |
| `src/tasks/composition_feedback_tasks.py`, `src/workers/celery_app.py` | **gated on O1**: nightly feedback linker | 16 |
| `scripts/benchmarks/tool_composer/reliability_caveat_experiment.py`, `tests/unit/test_scripts/test_reliability_caveat_analysis.py` | experiment script and analysis tests; **the run is gated on O2** | 17 |
| `docs/data/03-ML-PIPELINE-SCHEMA.md`, `docs/runbooks/tool-composer-learning-loop.md` | schema doc + runbook | 18 |

---

### Task 0: Preflight (no code)

- [ ] **Step 1: Worktree and branch.**
  `cd $W && git branch --show-current && git status --short && git log --oneline -1`. Expect
  `claude/tool-composer-learning-loop`, a clean tree, and HEAD at the plan commit.
- [ ] **Step 2: Migration numbers.** `cd $W && git fetch -q origin main && git ls-tree --name-only origin/main database/ml/ | tail -3; ls ../lane-2015/database/ml ../lane-2016/database/ml | grep -E '^03[89]|^04'`.
  039–041 must be free; otherwise apply the numbering rule above.
- [ ] **Step 3: Baseline the tests this lane will touch** (record pass/fail counts in the task report):
  `cd $W && $PY -m pytest tests/unit/test_agents/test_tool_composer/test_executor.py tests/unit/test_agents/test_tool_composer/test_planner.py tests/unit/test_agents/test_tool_composer/test_registry_schema_drift_2003.py tests/unit/test_tool_registry/test_registry.py tests/unit/test_agents/test_tool_composer/test_memory_hooks_outcome_876.py tests/unit/test_api/test_routes/test_admin_llm_usage.py -q -p no:cacheprovider -n 0`
- [ ] **Step 4: `free -m`** and `codex --version`. No commit.

---

### Task 1: Throwaway-database fixture from a schema-only dump of prod

**Files:**
- Create `tests/integration/tool_composer_learning_loop/{__init__.py,conftest.py,_pg.py,test_fixture_sanity.py}`

**Why:** every later real-DB test needs prod's exact enums, CHECKs, grants, default ACLs and ledger (spec §9).

- [ ] **Step 1: Write the red test** `test_fixture_sanity.py`. Module-level
  `pytestmark = pytest.mark.skipif(os.getenv("E2I_DB_INTEGRATION") != "1", …)`.

```python
def test_fixture_matches_prod_ledger_and_registry(upgrade_db, prod_readonly):
    # prod_readonly: docker exec supabase-db psql -tA (SELECT only)
    assert upgrade_db.scalar("select count(*) from public.schema_migrations") == \
        prod_readonly.scalar("select count(*) from public.schema_migrations")
    assert upgrade_db.scalar("select count(*) from tool_registry") == 16
    assert upgrade_db.scalar("select count(*) from tool_dependencies") == 11
    assert upgrade_db.scalar("select 'ml/037_tool_registry_schema_sync.sql' in (select filename from public.schema_migrations)")
    assert upgrade_db.scalar("select to_regtype('e2i_agent_name') is not null")
    assert upgrade_db.scalar("select has_table_privilege('anon','public.tool_registry','SELECT')") is False
```

- [ ] **Step 2: Run it.** `E2I_DB_INTEGRATION=1 … -n 0`. Expect an ERROR: the fixture does not exist yet.
- [ ] **Step 3: Implement `_pg.py`.**
  - `admin_psql(sql, db="postgres")`: `docker exec -i supabase-db psql -U supabase_admin -d <db> -v ON_ERROR_STOP=1 -tA`.
  - `db_url(db)`: builds `postgresql://postgres:<pw>@127.0.0.1:5433/<db>`, with the password from
    `docker exec supabase-db printenv POSTGRES_PASSWORD`, held in memory only.
  - `class PgConn` with `.scalar/.rows/.execute`.
  - `class PsycopgRpcPort`: `async def call(name, params) -> Any` runs `SELECT name(%(p1)s, …)` with
    `jsonb` adaptation. It is the same contract as the production `SupabaseRpcPort` (Task 5).
  - `run_runner(project_root, db)`: `SUPABASE_DB_URL=db_url(db) bash <project_root>/scripts/run_migrations.sh`,
    returning (rc, stdout).
- [ ] **Step 4: Implement `conftest.py`** with session fixture `upgrade_db`:
  1. `DROP DATABASE IF EXISTS learning_loop_upgrade WITH (FORCE); CREATE DATABASE learning_loop_upgrade;`
  2. In the new database, as `supabase_admin`: `CREATE SCHEMA IF NOT EXISTS extensions; CREATE EXTENSION IF NOT EXISTS pgcrypto SCHEMA extensions; CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA extensions; CREATE EXTENSION IF NOT EXISTS vector;`
     plus the roles `anon`, `authenticated` and `service_role` if missing (the `test_058` pattern).
  3. Run `docker exec supabase-db pg_dump -U postgres -s -n auth -n public postgres`, restored with
     `psql -v ON_ERROR_STOP=0`. Capture every restore error line. **The test fails if any error line does
     not match an allowlist of Supabase-internal objects** (event triggers, publications) that the
     fixture records explicitly, never blanket-ignored.
  4. Run `pg_dump -U postgres -a -t public.schema_migrations -t public.tool_registry -t public.tool_dependencies postgres`
     and restore it.
  5. At teardown: `DROP DATABASE … WITH (FORCE)`.
- [ ] **Step 5: Run it.** Expect PASS. Also run the default suite without `E2I_DB_INTEGRATION`: SKIPPED.
- [ ] **Step 6: Lint, codex loop, commit** `test(learning-loop): throwaway DB restored from a schema-only dump of prod`.

---

### Task 2: ml/039 and ml/040 — COHORT and the registry sync function

**Files:**
- Create `database/ml/039_tool_category_cohort.sql`, `database/ml/040_tool_registry_startup_sync.sql`
- Test `tests/integration/tool_composer_learning_loop/test_040_registry_sync.py`

- [ ] **Step 1: Write red tests.** Apply 039 with plain psql and 040 with `--single-transaction`, the
  runner's branches, to `upgrade_db` in a module fixture. Tests:
  - `test_cohort_category_insertable_after_039_commit`: a `cohort_builder` row with `COHORT` and
    `cohort_constructor` inserts in a new transaction.
  - `test_valid_agent_follows_e2i_agent_name`: `casual_impact` raises `check_violation`;
    `cohort_profiler` passes.
  - `test_success_rate_and_dead_functions_dropped`: `success_rate` column absent;
    `to_regprocedure('update_tool_registry_metrics()')` and
    `to_regprocedure('get_tool_execution_order(text[])')` are NULL.
  - `test_sync_first_call_then_idempotent`: a payload of the 16 existing rows plus 4 new ones reports
    `inserted=4`; an identical second call is all zeros; `updated_at` is unchanged on the second call.
  - `test_sync_counts_use_xmax`: one changed description reports `updated=1, inserted=0`.
  - `test_sync_refuses_empty_payload`, `test_sync_refuses_duplicate_names`,
    `test_sync_refuses_dependency_endpoint_not_in_payload`.
  - `test_sync_refuses_more_than_max_deprecations_before_any_write`: a 2-tool payload raises, and
    `count(*) where deprecated_at is not null` stays 0 with all 11 dependencies present.
  - `test_sync_deprecates_within_limit`: payload minus one tool marks exactly that tool deprecated and
    `composable=false`, and deletes its dependency rows.
  - `test_sync_atomic_on_bad_agent`: a payload with one bad agent raises, and nothing changes.
  - `test_sync_concurrent_calls_serialize`: two psycopg connections call it simultaneously from two
    threads. Both succeed, one returns all zeros, and the final rows equal the payload.
  - `test_grants`: `has_function_privilege('anon'|'authenticated','sync_tool_registry(jsonb,jsonb,integer)','EXECUTE')`
    is false; for `service_role` it is true.
- [ ] **Step 2: Run.** Expect FAIL: the files do not exist.
- [ ] **Step 3: Write 039.** One statement plus a header comment explaining the un-wrapped runner branch
  (`scripts/run_migrations.sh:157–190`).
- [ ] **Step 4: Write 040.** Start from the rehearsed draft (session scratchpad `draft_039.sql`, sections
  1, 3 and 4) with these changes:
  - Signature `sync_tool_registry(p_tools jsonb, p_dependencies jsonb, p_max_deprecations integer DEFAULT 3)`.
  - **Validate before any write:** array non-empty; no duplicate `name`; every dependency endpoint in the
    payload names.
  - `v_would_deprecate := count(*) FROM tool_registry WHERE deprecated_at IS NULL AND name NOT IN payload`.
    Raise if `> p_max_deprecations`.
  - Then take the advisory lock and run the upsert, deprecation and dependency writes.
  - `DROP FUNCTION IF EXISTS update_tool_registry_metrics();` **before**
    `ALTER TABLE tool_registry DROP COLUMN IF EXISTS success_rate;`.
  - `DROP FUNCTION IF EXISTS get_tool_execution_order(text[]);`.
  - `REVOKE ALL … FROM PUBLIC, anon, authenticated; GRANT EXECUTE … TO service_role;`.
  - No `BEGIN`/`COMMIT` (`tests/integration/test_migrations_no_inner_txn.py`). No `ADD VALUE` or
    `CONCURRENTLY` text anywhere, comments included.
- [ ] **Step 5: Run.** Expect PASS. Also `$PY -m pytest tests/integration/test_migrations_no_inner_txn.py -n 0`
  passes.
- [ ] **Step 6: Codex loop, commit** `feat(db): ml/039 COHORT category, ml/040 sync_tool_registry with pre-write guards`.

---

### Task 3: ml/041 — recording schema, RPCs, reliability, views

**Files:**
- Create `database/ml/041_composer_learning_loop_recording.sql`
- Test `tests/integration/tool_composer_learning_loop/test_041_recording.py`

- [ ] **Step 1: Write red tests** (041 applied wrapped on top of Task 2's database).
  - **Schema:**
    - `test_columns_added`: every column in spec §5.2, including `audit_workflow_id`, `plan_source`,
      `error_type`, `last_activity_at`, `tool_performance.is_synthetic` and `tool_version`.
    - `test_outcome_and_class_checks`: an invalid value raises `check_violation`.
    - `test_dropped`: `query_embedding`, its index, `find_similar_compositions`, `trg_log_step_performance`
      and `trigger_log_step_performance` are gone.
  - **RPC contracts:**
    - `test_every_rpc_creates_episode_from_seed`: for each of start / phase / steps / heartbeat / finish
      called **first**, exactly one episode exists afterwards.
    - `test_phase_updates_only_non_terminal`.
    - `test_steps_idempotent_receipt`: the same steps twice give `already_present` equal to n and no
      duplicate `composition_steps` / `tool_performance` rows.
    - `test_steps_perf_rows_only_for_invoked_classes`: 10 classes in, perf rows only for succeeded,
      refused, input_rejected, timeout and error.
    - `test_steps_unknown_tools_reported_and_rest_recorded`.
    - `test_steps_work_after_finish`.
    - `test_finish_restores_snapshot_and_second_finish_is_noop`.
    - `test_heartbeat_bumps_only_non_terminal`.
    - `test_steps_for_returns_steps_in_order`: `composer_steps_for(ARRAY[…])` returns each composition's
      steps (step_number, tool_name, outcome_class) in `step_number` order. Task 11 reads references
      through it.
  - **Catalog:**
    - `test_catalog_allowlist_rpc`: `composer_public_column_names()` contains `treatment`, `region`,
      `brand`, and not `PT-0001`.
    - `test_rpc_rechecks_column_entries`: a hand-built step whose `input_params` holds
      `{"x":{"type":"column","name":"SENTINEL_7f3"}}` is stored as `{"type":"str","len":12}`.
  - **Reliability:**
    - `test_reliability_denominators`: plant 3 succeeded, 2 refused, 1 timeout and 1 error for a tool.
      Expect n_invoked 7, n_succeeded 3, n_refused 2, n_health_failures 2, n_health 5.
    - `test_reliability_n_retried_and_latency_only_succeeded`.
    - `test_reliability_provenance_filter`: with `p_include_synthetic=false`, synthetic rows are excluded
      from the counts and `n_synthetic` still shows them.
    - `test_reliability_window`: a row 31 days old is outside `p_days=30` and inside 60.
  - **Views:**
    - `test_success_rate_view_no_fanout`: 1 episode with 3 steps counts as 1.
    - `test_active_view_abandoned_derived`: `last_activity_at = now() - 6 min` is abandoned;
      `now() - 1 min` is not.
    - `test_grants_all_new_objects`: anon and authenticated are false and service_role true for the 6
      functions (`composer_record_{start,phase,steps,heartbeat,finish}`, `composer_public_column_names`,
      `composer_steps_for`), `get_tool_reliability` and the 3 views.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Write 041.** Start from the draft's sections 2, 5, 6 and 7 and apply the spec deltas:
  - **Seed first:** every RPC begins with an `INSERT … ON CONFLICT (composition_id) DO NOTHING` from `p_seed`.
  - **`composer_record_steps`:**
    - resolves `tool_id` by name;
    - re-checks `{"type":"column"}` entries against `pg_attribute ⋈ pg_class` in `public`;
    - `ON CONFLICT (episode_id, step_number) DO NOTHING`;
    - inserts perf rows with `is_synthetic` from the episode and `tool_version` from the registry,
      `ON CONFLICT (step_id) DO NOTHING`;
    - returns `{recorded, already_present, unknown_tools}`.
  - **`composer_record_finish`:** sets every snapshot field and the latencies; a terminal episode returns
    `{recorded:false, already_terminal:true}`.
  - **`get_tool_reliability(p_days integer DEFAULT 30, p_include_synthetic boolean DEFAULT true)`:** the
    columns of spec §6.
  - **Views:** `v_tool_reliability AS SELECT * FROM get_tool_reliability(30, true)`; the recreated
    `v_composition_success_rate` and `v_active_compositions` use `last_activity_at` and a 5-minute window.
  - **Grants:** REVOKE, then GRANT, on every object.
- [ ] **Step 4: Run.** Expect PASS. Re-apply 041 with `--single-transaction`: no error.
- [ ] **Step 5: Codex loop, commit** `feat(db): ml/041 composition recording RPCs, reliability function, fixed views`.

---

### Task 4: The real migration runner, failure first, and rollbacks

**Files:**
- Create `database/ml/rollback_041.sql`, `database/ml/rollback_040.sql`
- Test `tests/integration/tool_composer_learning_loop/test_migration_runner.py`

- [ ] **Step 1: Write red tests.**
  - `test_runner_failure_first_on_fresh_fixture`:
    1. Build `learning_loop_fail` with Task 1's fixture code.
    2. Copy `$W/scripts` and `$W/database` into `tmp_path/repo/`, then append `SELECT 1/0;` to the copy of 041.
    3. Run `run_runner(tmp_path/"repo", "learning_loop_fail")`. Expect rc ≠ 0.
    4. Assert the ledger holds `ml/039…` and `ml/040…` but not `ml/041…`, and that `composer_record_steps`
       does not exist.
    5. Run `run_runner($W, "learning_loop_fail")`. Expect rc 0 and exactly `ml/041…` applied.
    6. Run it again: output says 0 pending.
  - `test_runner_detector_branches`: re-implement the detector (strip `--…`, then
    `grep -qiE "ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE|CONCURRENTLY|^[[:space:]]*COMMIT[[:space:]]*;"`,
    copied from `run_migrations.sh:175–177`). Assert 039 matches and 040 and 041 do not.
  - `test_direct_reapply_idempotent`: apply 039 plain and 040/041 `--single-transaction` again; no error,
    and row counts are unchanged.
  - `test_rollbacks_restore_and_guard`, on a third fresh fixture:
    1. Apply 039–041, then run `rollback_041.sql` and `rollback_040.sql`.
    2. Assert the 013 objects are back (`find_similar_compositions`, the trigger, `query_embedding`,
       `success_rate`, `update_tool_registry_metrics`, `get_tool_execution_order`) and the new functions
       are gone.
    3. Assert `rollback_040` raises when a `cohort_constructor` row exists, naming it.
- [ ] **Step 2: Run.** Expect FAIL: no rollback files.
- [ ] **Step 3: Write the rollbacks.** Recreate from the ml/013 definitions (L1008–1044, L1054–1113,
  L1143–1197, L983–1003) and the ml/013 views L411–540. Then drop the 041/040 objects. The CHECK restore
  sits in a `DO` block that raises with the offending names.
- [ ] **Step 4: Run.** Expect PASS. `free -m` first: three fixtures are about 3 dumps.
- [ ] **Step 5: Codex loop, commit** `test(db): real-runner failure-first replay of ml/039–041, rollbacks`.

---

### Task 5: Registry sync client and payload

**Files:**
- Create `src/agents/tool_composer/rpc_port.py`, `src/agents/tool_composer/registry_sync.py`
- Tests `tests/unit/test_agents/test_tool_composer/test_registry_sync_payload.py`, `tests/integration/tool_composer_learning_loop/test_registry_sync_client.py`

- [ ] **Step 1: Write red unit tests.**
  - `test_payload_covers_exactly_the_live_tools`: names equal the drift test's `LIVE_TOOLS` (20).
  - `test_payload_categories_in_db_enum`: `ToolCategory` values ⊆ {013 values} ∪ {`COHORT`}. The DB side
    is parsed from the 013 and 039 files.
  - `test_payload_source_agents_are_known_agents`: ⊆ `AGENT_REGISTRY_CONFIG` keys
    (`src/agents/factory.py:80`).
  - `test_payload_dependencies_match_mappings`: equals `DEPENDENCY_FIELD_MAPPINGS`, 13 pairs.
  - `test_payload_refuses_partial_registry`: with the registry snapshot minus one tool (via
    `snapshot`/`restore_snapshot`), `build_sync_payload()` raises `LookupError`.
- [ ] **Step 2: Write red real-DB tests** on `upgrade_db` after 039–041:
  - `test_sync_client_applies_payload`: `await sync_tool_registry_once(PsycopgRpcPort(...))` → 20 active
    tools, 13 dependencies, and every row's `input_schema` / `output_schema` / `category` /
    `source_agent` / `avg_latency_ms` equal to the payload.
  - `test_second_call_in_process_is_noop`: the once-guard makes no second RPC. The port counts calls.
  - `test_failure_is_logged_not_raised`: a port pointed at a dropped database logs WARNING and returns
    `None`.
  - `test_column_allowlist_fetch_and_fail_closed`: `fetch_column_allowlist()` returns a frozenset holding
    `treatment`. When the port fails, it returns `None`, and the recorder treats `None` as an empty
    allowlist.
- [ ] **Step 3: Run.** Expect FAIL.
- [ ] **Step 4: Implement.**
  - `rpc_port.py`: `class RpcPort(Protocol): async def call(self, name: str, params: dict) -> Any`, and
    `SupabaseRpcPort`, which uses `await get_async_supabase_client()` then
    `(await client.rpc(name, params).execute()).data`.
  - `registry_sync.py`:
    - `build_sync_payload() -> tuple[list[dict], list[dict]]`, the generator's `build_payloads` minus
      `NOT_SEEDED_IN_DB`, plus category, source_agent, avg_latency_ms and version;
    - `sync_tool_registry_once(port=None)`, guarded by a module `asyncio.Lock` and a success flag;
    - `fetch_column_allowlist(port=None)`, cached with a 1 h refresh;
    - `learning_loop_startup()`, which runs both and logs the counts.
- [ ] **Step 5: Run.** Expect PASS. Unit tests `-n 0`; real-DB with `E2I_DB_INTEGRATION=1`.
- [ ] **Step 6: Codex loop, commit** `feat(tool-composer): registry sync client — payload from the live registry, once per process`.

---

### Task 6: Retire the migration-per-schema-change regime

**Files:**
- Delete `scripts/generate_tool_registry_sync_migration.py`
- Modify:
  - `tests/unit/test_agents/test_tool_composer/test_registry_schema_drift_2003.py`: delete from
    `SYNC_MARKER = …` (L617) to the end of section 3, and the module docstring's item 3 sentence;
  - `src/tool_registry/registry.py`: delete `register_from_database`, `_create_placeholder_callable` and
    `sync_to_database` (L432–604);
  - `tests/unit/test_tool_registry/test_registry.py`: delete the tests at L532–754 that call them;
  - `src/agents/tool_composer/tool_registry.py` L214–216;
  - `docs/data/03-ML-PIPELINE-SCHEMA.md` §4.1.

- [ ] **Step 1: Write the red test** `tests/unit/test_agents/test_tool_composer/test_registry_sync_regime.py`.
  - `test_no_code_references_the_retired_regime`: `git grep -n` over `src scripts tests` for
    `generate_tool_registry_sync_migration|register_from_database|sync_to_database|NOT_SEEDED_IN_DB`
    returns nothing outside this test file.
  - `test_registry_doc_describes_runtime_sync`: `03-ML-PIPELINE-SCHEMA.md` §4.1 no longer says "never
    registered at runtime", and names `sync_tool_registry`.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Delete and edit.**
  - Keep `ToolCategory` in `registry.py` only if `grep` shows a remaining importer. `get_tools_by_category`
    references it (L606): keep both unless unused; report either way.
  - The comment at `tool_registry.py:214–216` becomes: "The DB `tool_registry` / `tool_dependencies`
    rows are synced from these definitions at API startup by `registry_sync.sync_tool_registry_once()`
    (ml/040)."
  - Leave `database/ml/037` untouched: it is applied history.
- [ ] **Step 4: Run** the new test, the whole drift test file (sections 1–2 must stay green) and
  `tests/unit/test_tool_registry/test_registry.py`, all `-n 0`. Expect PASS, with counts matching the
  Task 0 baseline minus the deleted tests.
- [ ] **Step 5: Codex loop.** The brief includes the intent evidence (spec §4 on G3 / `fee8bc3e0`, the
  owner decision). Commit `refactor(tool-registry): retire the sync-migration regime and the unwired DB registration`.

---

### Task 7: Executor outcome classes and the per-step callback

**Files:** modify `models/composition_models.py` (`StepResult`) and `executor.py`. Test
`tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py`.

- [ ] **Step 1: Write red tests.** Use the real registry and real tools; register test callables with
  `snapshot`/`restore_snapshot` in a fixture.
  - `test_refusal_class`: a real `gap_calculator` step on a single-brand frame gives `refused`,
    attempts 1, error_type `ToolRefusalError`.
  - `test_input_rejected_class`: a real tool with a `None` for a required float gives `input_rejected`.
  - `test_plan_defect_class`: `input_mapping={"x": "$missing.field"}` gives `plan_defect`, attempts 0.
  - `test_dependency_unmet_class`: an upstream failing step gives `dependency_unmet`.
  - `test_cache_hit_class`: the same deterministic step twice gives `succeeded` then `cache_hit`, and
    `cache_hit=True`.
  - `test_not_registered_class`.
  - `test_sync_timeout_class`: a registered sync callable sleeping past `timeout_seconds=1`.
  - `test_async_timeout_class`: a registered async callable sleeping past 1 s gives `timeout`, attempts =
    max_retries + 1, error_type `TimeoutError`.
  - `test_retry_then_success_attempts`: a callable failing once then succeeding gives `succeeded`,
    attempts 2.
  - `test_error_class_keeps_last_exception_type`.
  - `test_circuit_open_class`: 3 failures, then `circuit_open`.
  - `test_callback_called_per_step_single_and_parallel`: a plan with a single group and a 2-step parallel
    group gets one callback per step, with the right step numbers.
  - `test_cancel_keeps_finished_parallel_sibling`: a parallel group of a fast callable and a slow one.
    Cancel the `execute` task after the fast one's callback fires. The callback saw the fast step and not
    the slow one, and `CancelledError` propagates.
  - `test_escaping_exception_keeps_finished_sibling`: the slow sibling raises during input construction
    (a context object whose attribute access raises). The fast sibling's callback fired.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Implement.**
  - `StepResult` gains `outcome_class: Optional[str] = None`, `attempts: int = 0`, `cache_hit: bool = False`
    and `error_type: Optional[str] = None`. Every return site in `_execute_step` sets them per spec
    §5.3's table.
  - The generic arm stores `last_exc` and classes `timeout` when `isinstance(last_exc, (asyncio.TimeoutError, TimeoutError))`.
  - `execute(plan, context, on_step_result: Optional[Callable[[int, StepResult], None]] = None)`: the
    step number is `plan.steps.index(step)`, computed once as a dict.
  - `_execute_parallel` wraps each task: `result = await self._execute_step(...)`, then
    `_safe_callback(on_step_result, n, result)`, then `return result`.
  - The callback is wrapped in `try/except Exception`, logged, and never raised.
- [ ] **Step 4: Run** the new file plus the existing `test_executor.py`, `test_dependency_skip.py`,
  `test_nonretryable_refusals_1600.py` and `test_sync_tool_bounded_timeout_1592.py`, all `-n 0`. Expect
  PASS.
- [ ] **Step 5: Codex loop, commit** `feat(tool-composer): executor records an outcome class, attempts and a per-step callback`.

---

### Task 8: Dependency-aware execution order

**Files:** modify `models/composition_models.py` (`ExecutionPlan`) and `planner.py`. Test
`tests/unit/test_agents/test_tool_composer/test_execution_order_repair.py`.

- [ ] **Step 1: Write red tests** (the three measured probes from spec §7.4 plus the rest).
  - `test_omitted_step_is_scheduled`: groups `[["a"]]`, steps a, b(dep a) → `[["a"],["b"]]`, and
    `plan.execution_order_repaired == "step_missing_from_groups"`.
  - `test_consumer_grouped_with_producer_is_split` → `[["a"],["b"]]`.
  - `test_empty_groups_follow_dependencies_not_list_order`: steps `[b(dep a), a]` → `[["a"],["b"]]`.
  - `test_valid_groups_unchanged`: `[["a","c"],["b"]]` with b depending on a → unchanged; repaired is None.
  - `test_duplicate_step_ids_rejected`, `test_unknown_dependency_rejected` and `test_cycle_rejected`
    raise `ValueError` at construction.
  - `test_error_result_plan_still_constructs`: `steps=[]`.
  - `test_kpi_plan_groups_unchanged`: build `_build_kpi_causal_plan` from a real KPI-shaped frame
    (`test_composer_kpi_plan.py` fixture).
  - `test_planner_wraps_validator_error_in_planning_error`: `ToolPlanner._build_execution_steps` output
    with duplicate ids reaching `ExecutionPlan(...)` inside `plan()` raises `PlanningError`.
  - `test_executor_runs_every_step_of_omitted_step_plan`: a real `PlanExecutor` on real tools; `trace.tools_executed == 2`.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Implement.**
  - A `@model_validator(mode="after")` on `ExecutionPlan` checks uniqueness, known dependencies and
    acyclicity (DFS).
  - A private attribute `_repair_reason` and a property `execution_order_repaired`.
  - `get_execution_order()` validates `parallel_groups` against the three conditions; otherwise it
    computes Kahn levels in plan order.
  - In `planner.plan`, the `ExecutionPlan(...)` construction sits inside the existing `try`, so the
    `ValueError` surfaces as `PlanningError`. `_adapt_cached_plan` already catches.
- [ ] **Step 4: Run** the new file plus `test_models.py`, `test_planner.py`, `test_composer_kpi_plan.py`
  and `test_fail_closed_zero_tools_f6.py`, all `-n 0`.
- [ ] **Step 5: Codex loop, commit** `fix(tool-composer): every planned step runs, in dependency order`.

---

### Task 9: The recorder — serializer, chain, heartbeat, drain, counter

**Files:** create `src/agents/tool_composer/learning_recorder.py`; modify `src/api/routes/metrics.py`.
Tests `tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py`,
`tests/integration/tool_composer_learning_loop/test_learning_recorder_realdb.py`.

- [ ] **Step 1: Write red serializer tests** (pure, real model objects, no DB).
  - `test_sentinel_absent_everywhere`: build a real `DecompositionResult`, `ExecutionPlan` and
    `ExecutionTrace` with `SENTINEL = "PT-SENTINEL-9c1"` planted in each of these positions:
    - sub-question text, intent, sub_question_id, step_id;
    - an undeclared input key, a string value, a dict key, a `$step` field name;
    - a dynamic output key;
    - a refusal, an input-error and a generic `KeyError` message;
    - a frame column that is also used as a parameter value.

    Assert `SENTINEL not in json.dumps(to_record(...))` with allowlist `frozenset({"treatment"})`.
  - `test_catalog_column_kept_and_non_catalog_reduced`: `"treatment"` gives
    `{"type":"column","name":"treatment"}`; `"geo_x"` gives `{"type":"str","len":5}`.
  - `test_allowlist_unavailable_keeps_no_names`: allowlist `None` means every string is `str`/len.
  - `test_positional_ids_and_remapped_refs`: LLM ids `kpi_ate`, `kpi_rank` become step numbers; refs
    become `{"type":"ref","step":1,"field":"edge_list"}` only when `edge_list` is in the producer's output
    model, else `field: null`.
  - `test_intent_normalized`: `"causal"` gives `CAUSAL`; `"sell more"` gives `OTHER`.
  - `test_output_keys_only_model_fields`.
  - `test_dict_values_length_only`.
- [ ] **Step 2: Write red real-DB tests** (`PsycopgRpcPort` on the Task 3 database).
  - `test_start_failure_then_later_writes_create_episode`: a port that raises on `composer_record_start`
    only still leaves one episode with every field after finish.
  - `test_exhausted_phase_retry_restored_by_finish`: a port failing `composer_record_phase` twice leaves
    the finish snapshot restoring `tool_plan`, `plan_source`, groups and latencies.
  - `test_lost_response_resend_no_duplicates`: the port raises after executing (response lost); the retry
    gives an identical DB state.
  - `test_unknown_tools_trigger_lazy_sync_then_resend`: start the database without cohort rows (sync not
    run). A step for `cohort_builder` makes the recorder run `sync_tool_registry_once` and re-send, and the
    step is recorded.
  - `test_heartbeat_liveness`: heartbeat 1 s, window 5 s, step callable sleeping 10 s → never abandoned.
    Cancel the heartbeat task → abandoned after 6 s.
  - `test_drain_flushes_pending`: enqueue 20 writes, then `await drain(timeout=5)`: all rows present.
  - `test_no_latency_on_user_path`:
    - two ports: connection refused, and a 10 s hang;
    - `recorder.start/phase/step/finish` calls complete in < 50 ms total, with no exception;
    - a composition-shaped coroutine under `asyncio.wait_for(…, 1)` finishes within 50 ms of the same
      coroutine with no recorder.
  - `test_failure_counter_increments`: `composer_record_failures_total{rpc="composer_record_start"}`
    increases when Prometheus is available.
- [ ] **Step 3: Run.** Expect FAIL.
- [ ] **Step 4: Implement** `learning_recorder.py`:
  - `to_record()` and its helpers (spec §5.5 rules);
  - `CompositionRecorder(composition_id, seed, port=None, heartbeat_s=60)` with
    `start() / phase(status, delta) / step(n, StepResult) / finish(final) / cancelled(phase)`, all
    synchronous enqueues onto one chained `asyncio.Task`;
  - `_write(rpc, params)`: 5 s `wait_for`, one retry after 1 s, then WARNING plus counter;
  - the heartbeat task;
  - module `_pending: set[asyncio.Task]` with a `discard` callback, and `async def drain(timeout)`.

  In `metrics.py`, add `composer_record_failures_total` (labels `rpc`) to `_init_metrics`, plus an
  `inc_composer_record_failure(rpc)` helper that no-ops without Prometheus.
- [ ] **Step 5: Run.** Expect PASS. `free -m` first.
- [ ] **Step 6: Codex loop,** with the brief pointing at spec §5.3–§5.5. Commit
  `feat(tool-composer): composition recorder — structure-only serializer, seeded idempotent writes, heartbeat`.

---

### Task 10: Wire recording into the composer; plan_source; plan-cache eviction; startup and drain

**Files:** modify `composer.py`, `planner.py`, `cache.py`, `agent.py`, `src/api/routes/chatbot_tools.py`
(only `context["entry_point"]`) and `src/api/main.py` (the lifespan startup task and drain). Tests
`tests/unit/test_agents/test_tool_composer/test_composer_recording_wiring.py`, `test_plan_cache_eviction.py`,
`tests/integration/tool_composer_learning_loop/test_composer_live_llm.py`.

- [ ] **Step 1: Write red deterministic tests.**
  - `test_seed_uses_local_audit_id_not_context`: `ToolComposer._recording_seed(query, context, audit_workflow_id)`
    is a pure helper. It returns the `audit_workflow_id` argument and ignores `context["audit_workflow_id"]`.
    Three cases: argument `None` with no context key; argument `None` with a stale context uuid; argument
    set with a different stale context uuid.
  - `test_audit_block_yields_none_when_absent_or_raising`: extract the audit start into
    `_start_audit(audit_service, query, context) -> Optional[UUID]`. It returns `None` for
    `audit_service=None`, and `None` for a real `AuditChainService` constructed on a client whose
    `start_workflow` raises: a real client pointed at a dropped database, not a mock.
  - The call-site order (seed built after the audit block) is exercised by the opt-in live-LLM test in
    Step 3, run twice: audit service wired and unset.
  - `test_composition_id_consistent_across_results`: every `CompositionResult` from `_fail_closed`,
    `_create_total_failure_result` and success carries the up-front id.
  - `test_cancelled_recorded_then_reraised`: cancel during execute. The recorder saw `cancelled` with
    phase `execute`, and `CancelledError` propagates.
  - `test_entry_point_set_by_both_entry_points`: `chatbot_tools` context and `ToolComposerAgent.run`
    merged context carry `chat_tool` / `orchestrator_agent`. Assert at the call boundary with a real
    context dict; `compose` is not mocked.
- [ ] **Step 2: Write red eviction tests** (real `ToolPlanner`, real `ToolComposerCacheManager`, real
  `DecompositionResult` objects that meet spec §7.3 eligibility, no LLM on the cached path).
  - `test_failed_composition_evicts_cached_plan`: `cache_plan(d1, p)`, then
    `composer._after_execution(p, trace_all_failed)`, then `planner._cache_manager.get_similar_plan(d2)` is
    `None`.
  - `test_plan_defect_step_evicts`.
  - `test_succeeded_composition_keeps_cache` (positive control).
  - `test_plan_source_values`: `llm` / `plan_cache` / `kpi_deterministic` are set on the plan by the
    three paths.
  - `test_kpi_plan_never_touches_cache`.
- [ ] **Step 3: Write the opt-in live-LLM test** `test_composer_live_llm.py`, skipped unless
  `E2I_LIVE_LLM=1` and `E2I_DB_INTEGRATION=1`. Real `compose()` on a real KPI question, with the recorder on
  `PsycopgRpcPort`. Run it twice, audit service unset and then wired. Assert:
  - one terminal episode each;
  - steps equal to the trace's step results;
  - perf rows equal to the invoked classes;
  - `audit_workflow_id` NULL for the unset run and equal to the audit id for the wired run.

  Run it once on the droplet (≈ 2 real compositions of LLM spend, within the owner's standing lane
  protocol for real results), and paste the summary into the task report.
- [ ] **Step 4: Run.** Expect FAIL.
- [ ] **Step 5: Implement.**
  - `compose()`:
    - `composition_id = f"comp_{uuid4().hex[:8]}"` first;
    - the seed built after the audit block;
    - `recorder = CompositionRecorder(...)`, then `recorder.start()`;
    - a phase call after each phase;
    - `self.executor.execute(plan, context, on_step_result=recorder.step)`;
    - `finish` on every exit;
    - `except asyncio.CancelledError: recorder.cancelled(current_phase); raise`.
  - `_after_execution(plan, trace)` does the eviction.
  - `planner.plan` sets `plan.plan_source` and `plan.plan_cache_key`.
  - `PlanSimilarityCache.evict(key)` plus a `ToolComposerCacheManager.evict_plan(key)` facade.
  - `main.py` lifespan: after the Supabase init,
    `learning_task = asyncio.create_task(learning_loop_startup())`. In `finally`, before the other
    cleanup, `await drain(timeout=5)`, wrapped in try/except.
- [ ] **Step 6: Run** the new tests plus `test_composer.py`, `test_composer_kpi_plan.py`,
  `test_fail_closed_zero_tools_f6.py`, `test_cache.py`, `test_planner.py`, `test_integration.py` and
  `tests/unit/test_api/test_audit_chain_lifespan_wiring.py`, all `-n 0`.
- [ ] **Step 7: Codex loop, commit** `feat(tool-composer): record every composition, evict failed plans from the plan cache`.

---

### Task 11: Episodic references say what worked

**Files:** modify `memory_hooks.py` (`find_similar_compositions` hydration) and `planner.py`
(`_format_episodic_context`). Tests `tests/unit/test_agents/test_tool_composer/test_episodic_reference_rendering.py`,
`tests/integration/tool_composer_learning_loop/test_reference_hydration_realdb.py`.

- [ ] **Step 1: Write red tests.**
  - **Pure formatter, on real shapes:**
    - `test_partial_reference_recommends_only_succeeded`: steps succeeded, refused and
      dependency_unmet. "Tools that worked" lists only the succeeded tool; "Did not work" lists the other
      two with their classes.
    - `test_zero_success_reference_dropped`.
    - `test_legacy_partial_reference_dropped`: the two live `raw_content` shapes copied verbatim from
      `episodic_memories`, `partial_success` with no steps.
    - `test_legacy_all_success_reference_renders_as_today`.
  - **Real DB:** `test_hydration_reads_steps_by_composition_id`. Record an episode with 3 steps via the
    RPCs, then run `hydrate_reference_steps([{"raw_content":{"composition_id":…}}], port)` and get the 3
    steps in `step_number` order, through `composer_steps_for` (created in Task 3).
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Implement.** The hook calls `composer_steps_for` (ml/041, Task 3) through the `RpcPort`
  for the ≤ 3 references. The formatter implements the four shapes. No new migration.
- [ ] **Step 4: Run** plus `test_planner.py`, `test_memory_hooks_outcome_876.py` and
  `tests/integration/test_raw_content_hydration_reads_889.py` (opt-in), all `-n 0`.
- [ ] **Step 5: Codex loop, commit** `feat(tool-composer): similar-composition references recommend only the steps that worked`.

---

### Task 12: Reliability rule, reader and flag-gated planner caveat

**Files:** create `src/agents/tool_composer/reliability.py`; modify `planner.py` (`_format_tools_for_prompt`)
and `dspy_integration.py` (L189–225). Tests `tests/unit/test_agents/test_tool_composer/test_reliability_rule.py`,
`test_planner_reliability_flag.py`.

- [ ] **Step 1: Write red tests.**
  - **Verdicts:** `test_verdict_table` covers every row of spec §7.1:
    - n_invoked 0 → `no_runs`;
    - n_health 19 → `too_few_runs`;
    - 20 refusals and 0 health → `too_few_runs`;
    - n_health 40 with 0 failures → `reliable`;
    - n_health 40 with 12 failures → `caveat`;
    - n_health 40 with 3 failures → `inconclusive`.
  - **`test_planted_truth_calibration`:** 20,000 draws, seed 7, `numpy.random.default_rng`. With the
    shipped rule:
    - P(caveat) ≤ 0.01 at p = 0.05 for n ∈ {20, 40, 100};
    - P(caveat) ≥ 0.90 at p = 0.30 with n = 40;
    - `too_few_runs` for all n < 20.

    Runtime < 5 s; if not, vectorise.
  - `test_latency_null_below_20_successes`.
  - **Reader:**
    - `test_reader_cache_keyed_by_window_and_provenance`: port call counts for (30, True) twice, then
      (60, True).
    - `test_reader_fail_open`: a failing port returns `{}`.
  - **Flag:**
    - `test_flag_off_prompt_byte_identical`: capture `_format_tools_for_prompt()` and the DSPy tool block
      from `main` (via `git show 56f8b8589:…` into a fixture) against the new code with the flag unset;
      both are equal.
    - `test_flag_on_caveat_line_only`: the reader returns `caveat` for one tool. Exactly one added line
      per caveated tool, and the "Avg execution" line is unchanged.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Implement** `wilson()`, `verdict(counts) -> Verdict`, `ToolReliabilityReader` (async, TTL 300 s,
  key `(days, include_synthetic)`) and `format_tool_block(tool, verdict_or_none)`, which the planner and
  DSPy both call. The flag is read fresh per call: `os.getenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "")`
  truthy.
- [ ] **Step 4: Run** plus `test_planner.py`, `test_dspy_integration.py` and
  `test_planner_token_budget_1365.py`, all `-n 0`.
- [ ] **Step 5: Codex loop, commit** `feat(tool-composer): calibrated reliability verdicts; planner caveat behind a default-off flag`.

---

### Task 13: Retire in-memory G8

**Files:** modify `executor.py` (delete `update_tool_performance`, L1312–1375) and
`tests/unit/test_agents/test_tool_composer/test_executor.py` (delete the class at ~L1610–1686).

- [ ] **Step 1: Red test** in `test_registry_sync_regime.py`: `test_g8_update_tool_performance_removed`,
  a grep over `src tests` for `update_tool_performance(`.
- [ ] **Step 2: Delete, run** `test_executor.py` and the regime test with `-n 0`.
- [ ] **Step 3: Codex loop** (brief carries spec §7.5's intent evidence). Commit
  `refactor(tool-composer): remove the unwired in-memory G8 latency writer`.

---

### Task 14: Admin endpoint

**Files:**
- Create `src/services/tool_composer_observability_service.py`, `src/api/schemas/admin_tool_composer.py`
- Modify `src/api/routes/admin.py`
- Tests `tests/unit/test_api/test_routes/test_admin_tool_composer.py`, `tests/integration/tool_composer_learning_loop/test_admin_tool_composer_realdb.py`
- Regenerate `frontend/src/types/generated/api.ts`

- [ ] **Step 1: Write red tests.**
  - **Real DB, service level:** seed episodes, steps and perf rows via the RPCs; the service on
    `PsycopgRpcPort` / psycopg. The response has:
    - `compositions` counts by outcome, unfinished, abandoned and `plan_source`;
    - `tools[]` with `verdict` first plus the counts, measured latency `null` below 20 successes, and
      `declared_latency_ms` separate;
    - `recent_failures[]` (≤ 10), with failed_phase and step classes;
    - `query_preview` ≤ 100 chars;
    - window respected; `include_synthetic` taken from `deployment_includes_synthetic()`.
  - **Route (unit):** `require_admin` enforced (403 for non-admin, following `test_admin_llm_usage.py`'s
    pattern); `days` bounds 1–365; schema of the response model.
- [ ] **Step 2: Run.** Expect FAIL.
- [ ] **Step 3: Implement.**
  - The sync service uses `get_supabase_client()` (service key) and `client.rpc("get_tool_reliability", …)`
    plus table selects. It calls `reliability.verdict()`.
  - The route is `GET /admin/observability/tool-composer`, run via `asyncio.to_thread`, with a docstring.
    **The docstring is an OpenAPI change.**
- [ ] **Step 4: Regenerate the contract** the way `verify-types.yml` does. `free -m` first; this imports
  the app.
  - `cd $W && $PY -m scripts.export_openapi --output /tmp/…/openapi.json && cd frontend && npx openapi-typescript <that> -o src/types/generated/api.ts`
  - then `git diff --stat src/types/generated/api.ts` (only the new path and schemas) and `npm run typecheck`.
- [ ] **Step 5: Run** the tests, `-n 0`.
- [ ] **Step 6: Codex loop, commit** `feat(admin): tool-composer observability endpoint — verdict first, then the numbers`.

---

### Task 15: Admin UI section

**Files:** modify `frontend/src/api/admin.ts`, `frontend/src/lib/api-schemas.ts`,
`frontend/src/hooks/api/use-admin.ts` and `frontend/src/components/admin/ObservabilityTab.tsx`; create
`frontend/src/components/admin/ToolComposerSection.tsx` plus tests `ToolComposerSection.test.tsx` and an
`api-schemas` parse test.

- [ ] **Step 1: Write red tests.**
  - **Zod parse:** a real response captured from Task 14's real-DB test is committed as a fixture JSON.
    It parses, and no key is stripped: assert a deep-equal round trip. Zod strips unknown keys (memory
    lesson).
  - **Component:**
    - the verdict word renders in the first column;
    - "Too few runs to judge (n=3)" for too-few rows;
    - measured latency shows "—" when null, with declared latency labelled "declared";
    - stat cards: compositions, success, partial, failed, cancelled, abandoned;
    - the days selector shared with the tab drives the query key.
- [ ] **Step 2: Run** `npx vitest run <files>`. Expect FAIL.
- [ ] **Step 3: Implement,** following `ObservabilityTab.tsx`'s existing stat-card and table patterns.
- [ ] **Step 4: Run** vitest on the new and existing `ObservabilityTab.test.tsx`, plus `npm run typecheck`.
- [ ] **Step 5: Codex loop, commit** `feat(admin-ui): tool composer section in the Observability tab`.

---

### Task 16 (GATED on owner decision O1): nightly composition feedback linker

Skip this task unless the owner answered O1 "build". If they did:
- **Files:** `src/tasks/composition_feedback_tasks.py`; a beat entry in `src/workers/celery_app.py` at a fixed
  slot clear of 02:00, Mon-03:00, 04:30, 05:30 and 06:00. The spec proposes 05:00.
- **Behaviour:** mirror `routing_label_tasks.py`'s explicit-feedback matcher. `chatbot_message_feedback`
  is matched on session_id and time window to `composer_episodes`. Thumbs up sets `success=true`; thumbs
  down sets `success=false` with `feedback_text` NULL (no free text, spec §5.5); `feedback_at` is set.
  The labeler never mutates behaviour.
- **TDD:** real-DB tests on the fixture (match, no match, two episodes in a session, idempotent re-run)
  → implement → codex → commit.

---

### Task 17: Caveat experiment script (build ungated; RUN gated on O2)

**Files:** create `scripts/benchmarks/tool_composer/reliability_caveat_experiment.py`; test
`tests/unit/test_scripts/test_reliability_caveat_analysis.py`. Check the directory is on the CI allowlist.
If `tests/unit/test_scripts/` is new, put the test in `tests/unit/test_agents/test_tool_composer/` instead.

- [ ] **Step 1: Write red analysis tests** on constructed paired tables with known answers:
  - `mcnemar_one_sided(b, c)` equals `scipy.stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue`;
  - `pass_rule(pairs)` needs effect ≥ 0.30 and p < 0.05 **and** the three observed-count guards;
  - each guard failing on its own fails the rule;
  - items whose planner call errored count as invalid in that arm.
- [ ] **Step 2: Implement.**
  - The pre-registration docstring: items, pilot, arms, outcomes, pass rule, memory guard.
  - `--pilot` / `--evaluate` modes; frozen item-set hash; seeded arm order.
  - Planner calls through the production factory (`get_chat_llm`, thinking off).
  - Plan execution with the real `PlanExecutor` on real frames from the entry points' loaders.
  - `free -m` checked before each item; abort below 1500 MiB.
  - Results written to `docs/demos/results/<date>_reliability_caveat_experiment/`.
- [ ] **Step 3: Run** the analysis tests with `-n 0`. **Do not run the experiment.** Codex loop, then
  commit `feat(bench): pre-registered reliability-caveat experiment (analysis tested; run awaits authorization)`.
- [ ] **Step 4 (after the O2 authorization only):** run the pilot, then the evaluation. Commit the results.
  Flip the flag default in a separate commit **only** on a pass.

---

### Task 18: Documentation

**Files:** modify `docs/data/03-ML-PIPELINE-SCHEMA.md` (§4.1 registry sync; §4.4–4.6 new columns, RPCs,
reliability function, views; the flowchart at L1625–1632); create `docs/runbooks/tool-composer-learning-loop.md`.

The runbook covers:
- reading the admin section;
- the `composer_record_failures_total` meaning;
- the coverage reconciliation query;
- the retention re-evaluation trigger (> 50,000 episodes);
- rollback steps (code revert first, then `rollback_041.sql`, then `rollback_040.sql` by hand);
- the flag and experiment.

- [ ] **Step 1: Red doc test** in `test_registry_sync_regime.py`: the schema doc names
  `composer_record_steps`, `get_tool_reliability` and `sync_tool_registry`, and the runbook exists and
  names both rollback files.
- [ ] **Step 2: Write the docs, run the test, codex loop, commit** `docs(learning-loop): schema reference and runbook`.

---

### Task 19: Whole diff, PR, deploy, live certification

- [ ] **Step 1: Re-check migration numbers and rebase need.** `git fetch origin main`. If LANE-2015 or
  LANE-2016 merged, `git merge origin/main` (**no rebase of reported commits**). Re-run Task 0 Step 2, then
  the full lane test list `-n 0` and the opt-in real-DB suite.
- [ ] **Step 2: Lint and types.**
  - `ruff check` and `ruff format --check` on every changed Python file.
  - `mypy` on changed `src/` files only.
  - Frontend `npm run typecheck` and the vitest files.
- [ ] **Step 3: Whole-diff codex loop** (`git diff origin/main...HEAD`, pointed at file groups in turn)
  until ACCEPT.
- [ ] **Step 4: STOP and report to the dispatcher** with the commit list and verification output. Push and
  open the PR **only on the dispatcher's go**. The PR body ends with the attribution lines.
- [ ] **Step 5: After merge, watch the deploy** with a bounded foreground loop and read the container tag.
  - Record `image_sha: <sha>` from `docker ps --format '{{.Image}}' | grep e2i-api`.
  - Migrations 039–041 are applied by `deploy.yml`'s `run_migrations.sh`. Verify
    `select filename from schema_migrations where filename like 'ml/04%'` (read-only).
- [ ] **Step 6: Live cert** (spec §9, items 1–9), outputs saved to
  `docs/demos/results/<date>_learning_loop_cert/`.
  1. Boot logs of both workers: sync counts (`inserted=4` then zeros); DB 20 active tools and 13
     dependencies.
  2. A real chat composition gives one terminal episode, steps equal to the trace, and perf rows equal to
     the invoked steps.
  3. A refusal-producing composition is counted `refused`, not a health failure.
  4. The admin endpoint and tab show "Too few runs to judge".
  5. The planner prompt log shows no caveat line (flag off).
  6. PostgREST with the anon key refuses `rpc/composer_record_start`, `rpc/get_tool_reliability` and
     `v_tool_reliability`.
  7. Compose wall time is within noise of the pre-deploy baseline for the same questions, with recorder
     write durations reported.
  8. Coverage by identity: cert session_ids against episodes against audit `workflow_start` rows.
  9. The `plan_source` distribution is reported.

  After the cert:
  - write the memory note;
  - on any failed step, report it as a finding, not a pass.
- [ ] **Step 7: Rollback readiness** (documented, not executed): code revert PR; `rollback_041.sql` /
  `rollback_040.sql` by hand via `docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < …`.

---

## Self-review against the spec

| Spec section | Task(s) |
|---|---|
| §4 registry sync, guards, mixed versions, retirement | 2, 4, 5, 6, 10 (startup) |
| §5.2 schema | 3 |
| §5.3 write points, executor classes, per-step callback | 7, 10 |
| §5.4 seeded idempotent RPCs, retry, heartbeat, drain, counter, no user-path latency | 3, 9, 10 |
| §5.5 structure-only serialization, catalog allowlist, no error text | 3 (RPC re-check), 9 |
| §6 reliability function, views, drops | 2, 3 |
| §7.1 rule + calibration | 12 |
| §7.2 reader, flag, experiment | 12, 17 |
| §7.3 plan-cache eviction, references | 10, 11 |
| §7.4 execution order | 8 |
| §7.5 G8 | 13 |
| §8 admin API + UI | 14, 15 |
| §9 verification, cert, rollback | 1, 4, every task, 19 |
| §11 O1 | 16 (gated) |
| §11 O2 | 17 step 4 (gated) |
| §11 O3 | 2, 3, 4 (rollbacks recreate) |
