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

## 5. Ledger

The hand application recorded the key the file had at the time,
`deferred/146_drop_legacy_per_hcp_count_columns.sql`. That row is **kept** — it is the
only queryable record that this DDL reached production by hand on this date. The moved
file records the runner's key (the bare basename), so the first deploy after the move
applies it once more, idempotently, and writes a second row. The migration's header
states exactly what that second application does and does not change.
