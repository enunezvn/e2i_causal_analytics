# Lane A — migration 148 rehearsal, negative control, loader dry-run, deploy-gate banner (2026-09-22)

**Verdict: PASS on all four.** Migration 148 applies cleanly in the runner's own byte shape and rolls back leaving nothing behind; the loader validates the real export and reports the missing table honestly; the deploy's real-DB gate saw the migration as pending and passed.

## 1. Runner-shaped rehearsal inside `BEGIN … ROLLBACK` (prod `supabase-db`, worktree at `c9be0e9a5`)

The bytes piped are exactly what `scripts/run_migrations.sh::apply_dir()` builds: the file, then the runner's own
`INSERT INTO public.schema_migrations(filename) VALUES ('148_optum_biologic_persistence_causal.sql') ON CONFLICT DO NOTHING;`
— wrapped here in `BEGIN;` / `ROLLBACK;` with two probes before the rollback.

```
BEGIN
CREATE TABLE
CREATE INDEX
CREATE INDEX
INSERT 0 1
 cols 
------
   83
(1 row)

                 filename                  
-------------------------------------------
 148_optum_biologic_persistence_causal.sql
(1 row)

ROLLBACK
NULL|0
```

83 columns = the 81 exported fields (the contract test's `_export_record_keys()`) + `created_at` + `updated_at`. `INSERT 0 1` is the runner's ledger row (the file writes none of its own). The last line is the negative control, read AFTER the rollback: `to_regclass` is NULL and the ledger holds zero `148%` rows — nothing persisted.

## 2. Loader dry-run against the real export (table absent, `6332934d6`)

```
======================================================================
optum_biologic_persistence_causal loader (DRY RUN)  input=/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet
======================================================================
WOULD WRITE: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}
  persistent_at_180d_g28: positives={'XOLAIR': 8081, 'DUPIXENT': 3125} rate={'XOLAIR': 0.734, 'DUPIXENT': 0.744}
  discontinued_180d: positives={'XOLAIR': 1162, 'DUPIXENT': 511} rate={'XOLAIR': 0.1056, 'DUPIXENT': 0.1217}
  biologic_switch_180d_flag: positives={'XOLAIR': 106, 'DUPIXENT': 52} rate={'XOLAIR': 0.0096, 'DUPIXENT': 0.0124}
  persistent_at_180d: positives={'XOLAIR': 5759, 'DUPIXENT': 1458} rate={'XOLAIR': 0.5231, 'DUPIXENT': 0.3471}
LIVE TABLE: unreachable (migration 148 not applied, or no client)
DRY RUN complete. No rows written. Re-run with --execute to write.
```

Exit 0, nothing written. `LIVE TABLE: unreachable` is the honest report (migration 148 not applied yet) — never a zero split.

## 3. The deploy's blocking real-DB gate, run in the worktree (the faithful pre-deploy experiment)

```
env -u E2I_DB_SIMULATE_PENDING -u E2I_LIVE_LLM E2I_DB_INTEGRATION=1 .venv/bin/python -m pytest -n 0 -p no:cacheprovider -q -rs tests/unit/test_database/learning_loop/
```

Banner (the proof the gate saw this lane's migration):

```
learning-loop real-DB: prod ledger 230 keys; pending ['148_optum_biologic_persistence_causal.sql']; simulated None; deployed template learning_loop_deployed; registry sync {'updated': 0, 'inserted': 0, 'deprecated': 0, 'dependencies_deleted': 0, 'dependencies_upserted': 0}; tools the pre-upgrade schema refused []
```

Result: `243 passed, 8 skipped, 12 warnings in 263.62s`. The 8 skips are the pre-existing "upgrade path through ml/039–044: already applied" skips, unrelated to 148. MemAvailable before the run: 5,368 MiB (the gate refuses below 2,048).

## Sequencing (spec §5, §7)

The deploy applies 148 through `run_migrations.sh` (`database/**` is a deploy trigger). The owner-GO production load (`--execute`) and the live API cert follow the merge — see the plan's Task 12.
