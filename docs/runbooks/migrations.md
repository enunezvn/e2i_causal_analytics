# Database Migrations Runbook (PROD = MANUAL)

**Status**: Operator runbook | **Last verified against code**: 2026-06-03

This is a reader-facing companion. The **code and config are the source of
truth** — every command, flag, and line reference below was transcribed from
the files cited. If they have drifted since 2026-06-03, trust the files:

- `.github/workflows/deploy.yml` (the conditional migration step)
- `scripts/run_migrations.sh` (the `SUPABASE_DB_URL` + `psql` runner)
- `database/migrations/*.sql` (the migration files)
- `tests/integration/test_migrations_no_inner_txn.py` (the no-inner-txn lint)

---

## TL;DR

**Production database migrations are MANUAL.** Nothing in CI or the deploy
pipeline applies a migration to the live droplet database. When you add a file
under `database/migrations/`, you must apply it by hand by exec-ing into the
self-hosted Supabase Postgres container on the droplet:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
  < database/migrations/<file>.sql
```

Then verify it landed with an explicit cast/select disproof (see
[Verify it applied](#5-verify-it-applied)).

---

## 1. Why migrations do NOT auto-apply

The deploy workflow has a migration step, but it is **gated on
`SUPABASE_DB_URL` being set in the deploy environment** and the droplet env
does not set that variable.

`deploy.yml` (the SSH deploy script, around lines 244-248):

```bash
# Run migrations only when the DB URL is present in the deploy env;
# run_migrations.sh hard-exits otherwise, which `set -e` would abort on.
if [ -n "${SUPABASE_DB_URL:-}" ]; then
  bash scripts/run_migrations.sh
else
  echo "==> Skipping migrations: SUPABASE_DB_URL unset"
fi
```

The droplet's `.env` carries only Supabase **REST** credentials (anon/service
keys + URL); it has **no `SUPABASE_DB_URL`** Postgres connection string. So on
every production deploy this branch takes the `else` path and prints
`==> Skipping migrations: SUPABASE_DB_URL unset`. **No migration is ever
applied by the deploy.**

No other CI workflow applies migrations against a live database either. The
only migration-related CI is the **lint** in
`tests/integration/test_migrations_no_inner_txn.py`, which is filesystem-only
(no DB) — it checks file *shape*, not application.

| Path | Applies to live prod DB? | Why |
| --- | --- | --- |
| `deploy.yml` migration step | No | Gated on `SUPABASE_DB_URL`; droplet has none → "Skipping migrations" |
| `scripts/run_migrations.sh` | Only where `SUPABASE_DB_URL` is exported | Hard-exits otherwise (see below) |
| CI `test_migrations_no_inner_txn.py` | No | Filesystem-only lint, no DB connection |
| **`docker exec -i supabase-db psql ...`** | **Yes — manual** | The real droplet apply path |

---

## 2. The authoritative manual apply command

The droplet runs a self-hosted Supabase **Docker** stack. The Postgres
container is named `supabase-db`. You have direct `docker` access on the
droplet, so apply a migration by piping the SQL straight into `psql` inside
that container:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
  < database/migrations/<file>.sql
```

Flag notes:

| Flag | Effect |
| --- | --- |
| `exec -i` | Keep stdin open so the redirected `.sql` file is fed to `psql` |
| `-U postgres -d postgres` | Connect as superuser to the `postgres` database |
| `-v ON_ERROR_STOP=1` | Abort on the first SQL error instead of plowing on (do NOT omit) |

Migrations **029, 055, 056, 057** (issue #607, agent-taxonomy reconciliation)
were applied to the droplet exactly this way by hand.

---

## 3. Why NOT `scripts/run_migrations.sh` on the droplet

`run_migrations.sh` is the ledger-tracking runner, but it **cannot run on the
droplet** because it hard-requires a `SUPABASE_DB_URL` connection string and a
`psql` binary on the host — neither of which the droplet provides to that
script's environment.

`run_migrations.sh` (lines 46-56):

```bash
if [ -z "${SUPABASE_DB_URL:-}" ]; then
  echo -e "${RED}ERROR: SUPABASE_DB_URL environment variable is required${NC}"
  echo "Example: export SUPABASE_DB_URL='postgresql://user:pass@host:5432/dbname'"
  exit 1
fi
# ...
if ! command -v psql &> /dev/null; then
  echo -e "${RED}ERROR: psql not found. Install postgresql-client.${NC}"
  exit 1
fi
```

Use `run_migrations.sh` only in an environment where you have exported a
working `SUPABASE_DB_URL` (e.g. a workstation with network access to a
Postgres endpoint and `postgresql-client` installed). It applies each pending
file under `psql --single-transaction` and records it in
`public.schema_migrations`. It also supports a dry run:

```bash
export SUPABASE_DB_URL='postgresql://user:pass@host:5432/dbname'
./scripts/run_migrations.sh --dry-run   # list pending without applying
./scripts/run_migrations.sh             # apply pending migrations
```

---

## 4. CONSTRAINT: no inner transaction (no `BEGIN` / `COMMIT`)

Migration files **must not contain a script-level transaction-control
statement** — no `BEGIN;`, `COMMIT;`, `ROLLBACK;`, `END;`, `ABORT;`, or
`START TRANSACTION;`. The runner (`scripts/run_migrations.sh`) already wraps
each file in `psql --single-transaction`, which owns the outer transaction
(the migration body **plus** the `INSERT INTO public.schema_migrations`
bookkeeping row). An inner `COMMIT;` would prematurely commit before the
bookkeeping insert — leaving a migration applied but unrecorded (silent ledger
drift); an inner `ROLLBACK;` is the inverse hazard.

This is enforced by the CI lint
`tests/integration/test_migrations_no_inner_txn.py`, which fails any migration
file containing such a statement at script level.

**Exempt — `DO` blocks and PL/pgSQL / `BEGIN ATOMIC` function bodies.** The
lint tracks `$tag$ ... $tag$` dollar-quoted boundaries and `BEGIN ATOMIC ...
END;` openers, and **does not flag** `BEGIN ... END` that lives inside a
function body or a `DO $$ ... $$` block. Those are control-flow blocks, not
transaction control. Example of a permitted pattern:

```sql
DO $$
BEGIN
  IF NOT EXISTS (...) THEN
    ALTER TABLE ...;
  END IF;
END
$$;
```

The canonical clean-shape reference file is
`database/migrations/039_drop_triggers_join_from_feedback_loop.sql`.

> Note on idempotency: enum migrations like `055_*` use
> `ALTER TYPE ... ADD VALUE IF NOT EXISTS ...`, which is idempotent and safe
> to re-run, and (in older Postgres) cannot run inside a transaction block —
> another reason migrations stay transaction-control-free.

---

## 5. Verify it applied

The `docker exec` path **bypasses `public.schema_migrations` version
tracking** — that ledger is only written by `run_migrations.sh`. So a
docker-exec apply leaves **no row** in `schema_migrations`; do not rely on that
table to know what landed on the droplet.

Instead, verify success with an explicit **cast/select disproof** against the
object the migration created. For example, after adding `experiment_monitor`
to an enum (mig 055), confirm the value now casts (a value that does not exist
raises `22P02 invalid_text_representation`):

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT 'experiment_monitor'::e2i_agent_name;"
# Expected: returns the value. A 22P02 error means the migration did NOT apply.
```

For an enum, you can also list the full label set:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT enumlabel FROM pg_enum e
     JOIN pg_type t ON e.enumtypid = t.oid
    WHERE t.typname = 'agent_name_enum'
    ORDER BY e.enumsortorder;"
```

Pick a disproof that is specific to what the migration changed (a new column,
table, type, or enum value) and that **errors** when the change is absent.

---

## 6. CAUTION: the claude.ai Supabase MCP is NOT the droplet

The `claude.ai` "Remote Supabase MCP" tools (`list_tables`, `execute_sql`,
`apply_migration`, etc.) read a **different, cloud-hosted Supabase project** —
**not** the self-hosted droplet stack. It is **non-faithful** for droplet
state: schema and data you see through that MCP can differ from production.

For the truth about what is actually on the droplet, **exec into
`supabase-db`** and query it directly:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c "<your query>"
```

---

## 7. Operator checklist (apply a new migration to prod)

1. Confirm the file is in `database/migrations/` and merged to `main`.
2. Confirm it has **no script-level `BEGIN`/`COMMIT`/`ROLLBACK`/`END`/`ABORT`**
   (CI lint `test_migrations_no_inner_txn.py` should already be green).
3. SSH to the droplet; `cd` to the project checkout
   (`/home/enunez/Projects/e2i_causal_analytics`).
4. Apply:
   `docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < database/migrations/<file>.sql`
5. Verify with a cast/select disproof against the migration's new object
   (Section 5). A clean exit + the disproof passing = applied.
6. Remember: nothing recorded it in `schema_migrations`. Track applied
   migrations out of band (PR/commit + this runbook's note).

---

## Cross-reference

- Deployment / admin operations: `DEPLOYMENT_ADMIN.md` (and the existing
  top-level `DEPLOYMENT.md`) — note its rollback section: the deploy's
  migration step is conditional and **skipped on the droplet** (migrations are
  applied manually, per this runbook).
- Backlog item **C3** in
  `docs/DOCUMENTATION_UPDATE_INDEX_20260603.md` (this runbook fulfills it).
- The v3 success-criteria + QC gate doc: `docs/model_success_criteria.md`.
