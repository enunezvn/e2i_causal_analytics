# Postgres slow-query log configuration

Box 4 of issue #391 PERFORMANCE slice: set up Postgres query slow log for
queries > 100ms.

The original spec was for a FalkorDB slow log; Decision 1 (RATIFY) replaced
the FalkorDB-as-primary-graph with Postgres provenance edges, so the slow-
query log target moves with it. This document captures how to enable the
slow log in the project's Supabase Postgres environment and how to surface
findings.

## Target threshold

```
log_min_duration_statement = 100ms
```

Any query whose total execution time exceeds 100ms emits a `LOG`-level
entry in the Postgres server log with the query text + duration. 100ms is
the issue #391 box-4 target verbatim.

## Recommended supporting GUCs

These complement the slow log so the captured entries are actionable:

| GUC | Recommended | Purpose |
|-----|-------------|---------|
| `log_min_duration_statement` | `100ms` | The slow-log threshold (box 4). |
| `log_statement` | `'none'` | Don't double-log all statements; let slow log do the filtering. |
| `log_duration` | `off` | Same — `log_min_duration_statement` already includes duration. |
| `log_line_prefix` | `'%t [%p] %u@%d/%a '` | Timestamp / PID / user@db / app — make entries grep-able. |
| `log_lock_waits` | `on` | Surface lock-contention as separate `LOG` entries (often the root cause of >100ms queries). |
| `track_io_timing` | `on` | Populate `pg_stat_statements.shared_blks_read_time` so EXPLAIN attributes time to I/O vs CPU. |

## Enabling on a self-hosted Postgres (development)

For local development against a self-hosted Postgres (e.g. via
`docker/docker-compose.yml`), apply the GUCs in `postgresql.conf` or via
`ALTER SYSTEM`:

```sql
-- Apply persistently (writes to postgresql.auto.conf).
ALTER SYSTEM SET log_min_duration_statement = '100ms';
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET log_line_prefix = '%t [%p] %u@%d/%a ';
-- Reload (no restart required for these settings).
SELECT pg_reload_conf();
```

Verify the threshold is active:

```sql
SHOW log_min_duration_statement;  -- expect: 100ms
```

Slow-query entries land in the Postgres server log (commonly
`/var/log/postgresql/postgresql-*.log` on a system install; for docker, see
`docker logs <postgres_container>`).

## Enabling on Supabase (production / staging)

Supabase exposes Postgres GUCs via the dashboard at
**Project → Settings → Database → Database settings**. Required steps:

1. Navigate to **Settings → Database**.
2. In the **Custom Postgres Config** section, add the lines below to the
   "Add config" pane:

```
log_min_duration_statement = 100ms
log_lock_waits = on
track_io_timing = on
log_line_prefix = '%t [%p] %u@%d/%a '
```

3. Click **Apply changes**. Supabase reloads the config without a restart
   for these GUCs.
4. Verify on a SQL session: `SHOW log_min_duration_statement;` should
   return `100ms`.

For programmatic access via the Supabase Management API:

```bash
curl -X POST "https://api.supabase.com/v1/projects/{project-ref}/config/database/postgres" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"log_min_duration_statement": 100, "log_lock_waits": true, "track_io_timing": true}'
```

(See <https://supabase.com/docs/reference/api/v1-update-postgres-config> —
the `log_min_duration_statement` value is in **milliseconds** when set via
the API.)

## Reading slow-log entries on Supabase

Slow-log entries are written to Supabase's PostgreSQL logs. Two ways to read:

1. **Supabase Logs Explorer**: Project → Logs → Postgres. Filter by
   `event_message ~ 'duration: '` to surface only slow-query entries.
2. **Supabase Management API (`/v1/projects/{ref}/analytics/endpoints/logs.all`)**:

   ```bash
   curl -X GET "https://api.supabase.com/v1/projects/{project-ref}/analytics/endpoints/logs.all?sql=SELECT%20*%20FROM%20postgres_logs%20WHERE%20event_message%20LIKE%20%27duration%3A%25%27%20ORDER%20BY%20timestamp%20DESC%20LIMIT%20100" \
     -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
   ```

3. **`pg_stat_statements`**: complement the slow log with the canonical
   query-statistics view (cumulative, not log-shaped, but cheaper for
   long-running aggregation):

   ```sql
   -- Top 20 slowest cumulative queries.
   SELECT
     query,
     calls,
     mean_exec_time,
     max_exec_time,
     stddev_exec_time,
     total_exec_time,
     shared_blks_read_time + shared_blks_write_time AS io_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 20;
   ```

   `pg_stat_statements` is enabled by default on Supabase Postgres; if not,
   add `pg_stat_statements` to `shared_preload_libraries` (requires restart).

## Reading slow-log entries on self-hosted Postgres

```bash
# Tail the live slow log
tail -f /var/log/postgresql/postgresql-*.log | grep "duration:"

# Aggregate via pgbadger (recommended for trend reporting)
pgbadger /var/log/postgresql/postgresql-*.log --slow-statement 100ms \
  --out slow-query-report.html
```

## Surface for the CI guard

The PERFORMANCE CI workflow (`.github/workflows/benchmarks.yml`) does NOT
attempt to enforce the slow-log threshold at run time — slow-log is a
**production observability surface**, not a test gate. Enforcement is
operational, not code-test-based:

* Periodic review of slow-query logs (weekly digest, owned by the on-call
  rotation).
* Any query whose `mean_exec_time` in `pg_stat_statements` exceeds 100ms
  should be flagged as an optimization candidate (index review, EXPLAIN
  ANALYZE, query rewrite).

For the corresponding monitoring instrumentation (Opik / MLflow), see PR
#391 monitoring-slice (sibling parallel-dispatch).

## Re-baseline & change-management

This document is configuration + ops, not code. Changes (e.g., relaxing
the threshold to 200ms) require:

1. PR linking issue #391 with a justification.
2. Updating the threshold value in BOTH this document and any IaC manifest
   that applies it (Supabase Management API call OR
   `postgresql.auto.conf` in the self-hosted compose file).
3. A short doc update describing what changed and why.
