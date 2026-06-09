# Issue #821 — `await get_supabase_client()` (await-on-sync) + `Repository(client=)` kwarg fix

**Date:** 2026-06-09
**Branch:** `fix/issue-821-await-sync-supabase` (off `origin/main` @ `cbbbd8b6`)
**Root cause class:** Supabase client *wiring* (same as PR #820's experiment-monitor H3b)

## What #821 is (the client-wiring root cause)

Two latent `TypeError`-producing patterns that the surrounding `except` swallows,
masking a crash as an empty / skipped / fallback result:

1. **`await get_supabase_client()`** — `get_supabase_client` is a **sync** factory
   (`def` at `src/memory/services/factories.py:674`). Awaiting it raises
   `TypeError: object Client can't be used in 'await' expression`. Fix: use the
   async factory `get_async_supabase_client()` (every one of these sites then does
   `await client.<query>.execute()`, which requires the async client).

2. **`SomeRepository(client=...)`** — `BaseRepository.__init__` takes
   `supabase_client=` (it sets `self.client`). Passing `client=` raises
   `TypeError: __init__() got an unexpected keyword argument 'client'` → the repo
   helper's `except` returns `None` → silent fallback to sample/mock data.

### Sites fixed (8 await-on-sync + 3 kwarg)

| File | Line(s) | Pattern | Fix | Downstream |
|---|---|---|---|---|
| `src/tasks/ab_testing_tasks.py` | 356, 534, 946, 1033 | await-sync | → `get_async_supabase_client()` | `await ...execute()` |
| `src/tasks/drift_monitoring_tasks.py` | 404 | await-sync | → async | `await ...execute()` |
| `src/tasks/feedback_loop_tasks.py` | 148, 418 | await-sync | → async | `await ...execute()` / `await client.rpc(...)` |
| `src/mlops/optuna_optimizer.py` | 762 | await-sync | → async (import moved factories) | `await ...insert().execute()` |
| `src/api/routes/copilotkit.py` | 908 | kwarg **+ async** (compound) | helper made `async`, `await get_async_supabase_client()`, `supabase_client=`, caller awaits | repo `get_by_tier`→`get_many` awaits execute |
| `src/agents/ml_foundation/observability_connector/agent.py` | 91 | kwarg only | `supabase_client=` (sync client kept) | repo uses **sync** `.execute()` |
| `src/agents/ml_foundation/observability_connector/nodes/metrics_aggregator.py` | 30 | kwarg only | `supabase_client=` (sync client kept) | repo uses **sync** `.execute()` |

**NOT touched (verified correct):** `ab_testing_tasks.py:740` — `client = get_supabase_client()`
called **without** `await` and used with **sync** `.execute()` (deliberate A/B-side
sync convention). The `feedback_learner` sites use the defensive
`await _maybe_await(get_supabase_client())` adapter (correct).

## Faithful verification (real docker `supabase-db`, NO mocks)

- `tests/unit/test_supabase_client_misuse_guard.py` — static AST guard, **CI-runnable
  (no DB)**, pins both patterns to **zero** across `src/`. RED before fix found
  exactly the 8 + 3 sites; GREEN after.
- `tests/integration/test_async_supabase_client_realdb.py` — opt-in
  (`E2I_DB_INTEGRATION=1`, `-n0`). 9 passed, 1 xfail. Real-data anchors:
  **621 running ml_experiments**, **13 agent_registry rows**, **5313 ml_observability_spans**.
  RED reproduced the exact `TypeError`s; GREEN proves the async/sync client is now
  acquired and reaches real data (e.g. async `AgentRegistryRepository.get_active_agents()`
  returns the 13 real agents; observability repo reads the 5313 real spans; optuna
  persists a real `ml_hpo_studies` row then cleans it up).

## SEPARATE pre-existing bugs EXPOSED by this fix (NOT #821 — schema drift)

Fixing the client-wiring **unmasked** a distinct second root cause: **code-vs-DB column
drift**. These were never reachable before (the await/kwarg `TypeError` fired first).
They are intentionally **not** fixed here — they need their own intent investigation
(and possibly migrations), and guessing column names would violate the repo's
REASON-BEFORE-RULES / CHEAPEST-DISPROOF discipline.

| Symptom (live DB) | Code | Real schema | Notes |
|---|---|---|---|
| `column ml_experiments.name does not exist` (42703) | `ab_testing_tasks.py` queries `select("id, name, config")` (enrollment_health_check:362, srm_detection_sweep, check_all_active_experiments) | real cols: `experiment_name`, **no `config`** (structured fields instead) | deep mismatch; enrollment-health logic reads `exp["config"]` which does not exist |
| `column ml_drift_history.detected_at does not exist` (hint: `detected_by`) | `drift_monitoring_tasks.py:cleanup_old_drift_history` deletes `lt("detected_at", cutoff)` | real timestamp col: `created_at` | likely a 1-line column rename, but unverified semantics |
| `column agent_registry.tier does not exist` | `AgentRegistryRepository.get_by_tier` filters `{"tier": int}`; `copilotkit._fetch_agents_from_db` loops `get_by_tier(range(1,6))` | real col: `agent_tier` is **text categories** (`coordination`, `causal_analytics`, …), not int 1-5; only **13 of 21** roster agents present | NOT a clean rename; `_fetch_agents_from_db`'s integer-tier approach is wrong for the real schema |

**Recommended follow-up:** tracked in **#825** — reconcile these tasks/repos with the
live schema (the agent-registry one is the highest value — it unblocks copilotkit
`get_agent_status` real data; the client-wiring half is already fixed and proven by
`test_agent_registry_repository_reaches_real_agents`).

The xfail `test_fetch_agents_from_db_end_to_end_blocked_by_tier_schema_drift` keeps the
agent-registry schema bug visible in the suite (it will xpass once the follow-up lands).
