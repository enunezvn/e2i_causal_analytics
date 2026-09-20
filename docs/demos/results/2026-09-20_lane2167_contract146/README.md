# Lane #2167 — applying the per_hcp_rollup CONTRACT migration (146)

Evidence for `database/migrations/146_drop_legacy_per_hcp_count_columns.sql` being
applied to **production** and then moved out of `database/deferred/`.

Host is the droplet, which is PROD == DEV; the database is the local self-contained
Supabase (`supabase-db` container). Every command below ran against that database.

## Preconditions, re-measured on the day (not copied from the issue)

| # | Precondition | Measurement |
|---|---|---|
| 1 | 144's deploy completed, health checks passed | `schema_migrations` row `144_per_hcp_trigger_count_columns.sql` @ `2026-09-19 16:28:26Z`; **two** later deploys landed; every app container on main HEAD `349858db5`, all healthy |
| 2 | No deployable rollback target reads the legacy names | `deploy.yml:739-742` sets `PREV_SHA` to the **running** sha = `349858db5`; `tests/unit/test_etl/test_per_hcp_legacy_column_names_absent.py` green on that tree |
| 3 | The per-HCP rollup ETL ran on the new code | newest `business_metrics.created_at` = `2026-09-19 17:08:02Z` (**after** 144); 13,609 rows carried both names; **0** disagreeing |

## 1. Rehearsal — `rehearsal.sql`

The real file, the real database, wrapped in `BEGIN … ROLLBACK` with probes on both
sides. Run at ~01:48Z, minutes before the commit.

```
BEFORE view cols            29 × 4
AFTER  legacy cols          0          AFTER trigger 0     AFTER function 0
AFTER  view cols            36 × 4  (v_train / v_test / v_validation / v_holdout)
VALUES preserved            n=25489   trx 21786 → delivered 21786
                                      nrx 11906 → accepted  11906
                                      total_rx 24682 → total 24682     all_match = t
ledger row written in-txn;  v_train still returns 7130 rows
```

**Negative control**, immediately after the `ROLLBACK` — the rehearsal changed nothing:

```
legacy cols live  3     trigger live  1     ledger 146 rows  0     v_train cols  29
```

**Rigor note, raised by the codex iter1 review and recorded rather than quietly fixed.** The
numbers in this section are TRANSCRIBED from the rehearsal run; its raw psql stdout was not
captured to a file, unlike `apply_stdout.txt` and `post_apply_probe.txt`, which are raw
captures. The rehearsal cannot be re-run to produce that file, because its "BEFORE" state (three
legacy columns present) no longer exists — the apply consumed it. What DOES corroborate these
numbers, independently of this prose:

* `apply_stdout.txt` + `post_apply_probe.txt` — the real apply reproduced every rehearsed number.
* `second_application_raw.txt` — a raw capture of a rehearsal that IS still reproducible.
* The codex reviewer re-ran an equivalent live rehearsal at review time and it corroborated them.

`rehearsal.sql` is the exact wrapper that was used, so the method is reproducible even though
that particular run's stdout is not.

## 2. Backup — `legacy_counts_backup_preapply.csv.gz`

`(metric_id, trx_count, nrx_count, total_rx_count)` for all 25,489 rows, dumped
before the commit. `DROP COLUMN` destroys data and the reversal SQL cannot invent it
back, so the values were on disk first. Uncompressed sha256
`c1051b90d65cb28c55b3c16dcff6ef47cd11da643fe511137c9304379ef9bb08`, 25,490 lines
(header + 25,489 rows).

## 3. Apply — `apply_stdout.txt`

```
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
  --single-transaction < database/deferred/146_drop_legacy_per_hcp_count_columns.sql
```

File sha256 `cc9d952f783ccf041d18a9bdd02606f1c781a24c852fc912a8ef6b8ab9114fb4` (the
pre-move path, which is what was applied). Started and finished `2026-09-20T01:51:04Z`,
psql **exit 0**, 16 statements.

## 4. Post-apply — `post_apply_probe.txt`

```
legacy cols remaining          0
canonical cols present         3
sync trigger                   0
sync function                  0
ledger row                     deferred/146_drop_legacy_per_hcp_count_columns.sql
rows                           25489
sum delivered/accepted/total   21786/11906/24682     <- identical to pre-apply
v_train cols                   36      v_holdout cols  36      v_train rows  7130
```

`GET /health` → **200**; api / frontend / worker_light ×2 / worker_medium all healthy.

## 5. The second, automatic application — `second_application_raw.txt`, `second_application_probe.sql`

The hand apply recorded the key the file had then; the moved file records the RUNNER's key. Prod
does not hold that key, so the first deploy after the move applies 146 again. The migration header
claims that is inert, so it was measured against the live, ALREADY-CONTRACTED schema inside
`BEGIN … ROLLBACK`.

The probe is built the way `apply_dir()` builds it — the file body plus **the runner's own
appended** `INSERT … ON CONFLICT DO NOTHING` under the bare-basename key, not a psql `\i` — so the
ledger write the deploy will actually perform is inside the rehearsal. (This is the one thing
`rehearsal.sql` never exercised; the codex iter1 review raised it.)

```
rows true · canonical sums true · table col count true
view privileges true · view definitions true
      (relacl and md5(pg_get_viewdef) compared for all four split views)
ledger rows for 146: 146_drop_legacy_per_hcp_count_columns.sql
                   + deferred/146_drop_legacy_per_hcp_count_columns.sql
```

The two INSERT results in the raw output are the point: `INSERT 0 1` from the file's own ledger
write, then `INSERT 0 0` from the runner's appended one — the runner's write is a no-op precisely
because the file already recorded the same key. psql exit 0. Negative control after the ROLLBACK
(legacy columns / 146 ledger rows / v_train width):

```
0 / deferred/146_drop_legacy_per_hcp_count_columns.sql / 36
```

## 6. Other verification in this directory

* `deploy_gate_realdb_suite.txt` — the deploy's own blocking gate, run on this branch. It derives
  what is pending from prod's ledger and reports `pending
  ['146_drop_legacy_per_hcp_count_columns.sql']`, so its green rehearsed this change on a
  throwaway copy of prod rather than ignoring it.
* `fresh_db_scour_coverage.txt` — the real runner, `--dry-run`, against an EMPTY ephemeral
  postgres: the fresh-database path this move newly puts 146 on. Read the section appended after
  the codex iter2 review, not just the ALL PASS banner: the script's safety loop only detects
  wrongful INCLUSION of destructive filenames and is structurally blind to wrongful EXCLUSION,
  which is the direction that matters here, so the apply set itself is recorded. 146 is in it, and
  the runner emits 144 two lines ahead of it — the expand-before-contract ordering visible in the
  real runner's real output rather than in a `sorted()` call inside a test. 228 pending files,
  cross-validating against the deploy gate's independent "prod ledger 228 keys".
* `../2026-09-15_trx_canonical/iter9_13_146_rehearsal_20260919.txt` — the lane-#2114-era raw
  evidence for the precondition block's four refusal/pass cases and the view ACLs. The migration
  header cites it; the codex iter1 review found it was **untracked**, so this lane commits it.

## 7. Ledger

The hand application recorded the key the file had at the time,
`deferred/146_drop_legacy_per_hcp_count_columns.sql`. That row is **kept** — it is the
only queryable record that this DDL reached production by hand on this date. The moved
file records the runner's key (the bare basename), so the first deploy after the move
applies it once more, idempotently, and writes a second row. The migration's header
states exactly what that second application does and does not change.
