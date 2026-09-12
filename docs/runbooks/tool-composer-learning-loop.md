# Runbook — tool-composer learning loop

What this covers: the recording loop added in 2026-09 (spec
`docs/superpowers/specs/2026-09-11-tool-composer-learning-loop-design.md`), its admin surface, its
metrics, and how to turn it off or roll it back.

The loop records what the tool composer actually did — one episode per composition, one row per
step, one performance row per invoked tool — so that reliability is *measured* rather than
declared. It never changes planning behaviour except through the flag described below, which is
off by default.

---

## 1. The flags

| Flag | Default | What it does |
|---|---|---|
| `TOOL_COMPOSER_LEARNING_LOOP_ENABLED` | off in code, `true` in `docker/docker-compose.yml` | Turns recording on for the process, and schedules the startup registry sync. |
| `TOOL_COMPOSER_RELIABILITY_IN_PLANNER` | off everywhere | Adds one reliability caveat line per caveated tool to the planning prompt. Ships off; see §6. |
| `E2I_INCLUDE_SYNTHETIC` | off | Whether synthetic-substrate runs count in the reliability numbers and the admin page. |

**Turning recording off.** Set `TOOL_COMPOSER_LEARNING_LOOP_ENABLED=false` and recreate the API
containers. Compositions keep working; nothing is recorded. Note the startup registry sync rides
the same flag — see §7.

---

## 2. Reading the admin section

Admin → Observability → *Tool composer* (`GET /api/admin/observability/tool-composer?days=30`).

- **Stat cards** account for every composition in the window: success, partial, failed, cancelled,
  unfinished, abandoned. "Abandoned" means unfinished and silent past the heartbeat window — the
  worker went away mid-composition.
- **The tool table leads with the verdict word**, because that is the reading; the counts explain
  it. `n_health = succeeded + health failures` is the denominator. **Refusals never enter it**: a
  tool declining to answer a question its data cannot support is behaving correctly.
- **"Too few runs to judge (n=k)"** is the honest state below 20 health runs, and is what every
  tool shows at today's volume. It is not a problem to fix.
- **Measured latency shows "—" until 20 successful runs.** The registry's *declared* latency is a
  separate, labelled column and never stands in for a measurement.
- **Recent failures** name the phase and the step classes that caused them, with a 100-character
  redacted query preview. No error text is stored anywhere in these tables (spec §5.5).

---

## 3. `composer_record_failures_total`

A Prometheus counter, labelled by `rpc`, incremented when a recording write fails after its retry.

Recording is fail-open by design: a failed write never fails a composition, so this counter is the
only place a recording outage is visible. A rising count means episodes are incomplete, not that
users are affected. Check the API logs for the matching structured warning (it carries the rpc,
the composition id and the error class, never the error text).

---

## 4. Coverage reconciliation

Are we recording everything we should be? Compare episodes against the audit chain, which is
written on a different path:

```sql
-- Compositions the audit chain saw, in the last day, that have no episode.
SELECT a.workflow_id, a.created_at
FROM audit_chain_entries a
LEFT JOIN composer_episodes e ON e.audit_workflow_id = a.workflow_id
WHERE a.agent_name = 'tool_composer'
  AND a.action_type = 'workflow_start'
  AND a.created_at > now() - interval '1 day'
  AND e.episode_id IS NULL
ORDER BY a.created_at DESC;
```

Rows here mean the recorder could not write (see §3). The reverse direction — episodes with a NULL
`audit_workflow_id` — is normal: the audit service is not always configured, and the composition is
recorded regardless.

```sql
-- Steps recorded against tools invoked, same window.
SELECT e.composition_id, e.tools_executed, count(s.step_id) AS steps_recorded
FROM composer_episodes e
LEFT JOIN composition_steps s USING (episode_id)
WHERE e.created_at > now() - interval '1 day'
GROUP BY e.composition_id, e.tools_executed
HAVING count(s.step_id) <> e.tools_executed;
```

---

## 5. Retention

Re-evaluate retention when `composer_episodes` passes **50,000 rows**:

```sql
SELECT count(*) FROM composer_episodes;
```

Nothing prunes these tables today. At that size, decide a retention window with the owner before
the tables affect backup or restore times. `query_text` is already redacted and bounded, so the
question is size, not privacy.

---

## 6. The reliability caveat experiment

Whether a caveat line changes the LLM's tool choice has never been measured, which is why
`TOOL_COMPOSER_RELIABILITY_IN_PLANNER` ships off.

`scripts/benchmarks/tool_composer/reliability_caveat_experiment.py` carries the pre-registration:
frozen hashed item set, seeded paired arms, exact one-sided McNemar, and a conjunctive pass rule
(effect ≥ 0.30, p < 0.05, and three "not observed worse" guards). Its analysis half is tested; the
collection half is deliberately unimplemented.

**Running it needs a fresh owner authorization (gate O2), and only makes sense once some tool has
reached `n_health` ≥ 20** — below the floor every verdict is `too_few_runs` and there is nothing to
caveat. The script refuses to run without `--i-have-authorization`. Flip the flag default only in a
follow-up commit that cites a passing result; a failing result leaves the flag off and records
which condition failed.

---

## 7. The startup registry sync

At API startup, when the learning-loop flag is set, `learning_loop_startup()` calls
`sync_tool_registry` (ml/040), which makes `tool_registry` and `tool_dependencies` equal to the
running code's registered tools. It replaced the previous regime of generating a migration whenever
a tool's schema changed.

**Consequence worth knowing:** with `TOOL_COMPOSER_LEARNING_LOOP_ENABLED` off, nothing re-syncs the
registry. If the flag is turned off long-term, either re-enable the sync independently of recording
or keep a static sync migration in the tree.

Boot log to look for: `tool-composer learning loop: registry sync {...}; column allowlist N names`.

---

## 8. Rollback

In this order:

1. **Revert the code first.** The RPCs are additive, so an older image simply stops calling them;
   the tables can stay. This is usually the whole fix.
2. **Then, only if the schema must go back**, apply the rollbacks by hand, newest first:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 --single-transaction \
  < database/ml/rollback_041.sql
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 --single-transaction \
  < database/ml/rollback_040.sql
```

Both are idempotent and each removes its own `schema_migrations` row.

**Rolling back 041 destroys recorded data, and you should expect that.** It drops the columns this
lane added — `outcome`, `failed_phase`, `error_type`, `plan_source`, `entry_point`, `attempts`,
`audit_workflow_id`, `is_synthetic` and the rest — with whatever values they hold. The guards it
does carry are narrower than that: they stop the rollback when an episode is still unfinished or
when rows exist that the older shape cannot represent, naming the offending rows so you can clear
them deliberately. They do **not** preserve the recording columns' contents.

So: revert the code first and leave the schema alone unless the schema itself is the problem. If
you do roll 041 back, take a dump of `composer_episodes`, `composition_steps` and
`tool_performance` first if the recorded history matters.

`rollback_041.sql` restores the ml/013 shapes it replaced; `rollback_040.sql` restores
`update_tool_registry_metrics()`, `tool_registry.success_rate` (with the ml/027 values) and
`get_tool_execution_order(text[])`.

---

## 9. Where things live

| Thing | Where |
|---|---|
| Recording RPCs (`composer_record_start/phase/steps/heartbeat/finish`), `composer_steps_for`, `get_tool_reliability` | `database/ml/041_composer_learning_loop_recording.sql` |
| `sync_tool_registry` | `database/ml/040_tool_registry_startup_sync.sql` |
| Recorder, reliability rule, registry sync client | `src/agents/tool_composer/{learning_recorder,reliability,registry_sync}.py` |
| Admin endpoint and aggregation | `src/api/routes/admin.py`, `src/services/tool_composer_observability_service.py` |
| Admin UI section | `frontend/src/components/admin/ToolComposerSection.tsx` |
| Nightly feedback linker | `src/tasks/composition_feedback_tasks.py` |
