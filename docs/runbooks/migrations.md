# Database Migrations Runbook (PROD = AUTO on deploy, manual path available)

**Status**: Operator runbook | **Last verified against code**: 2026-07-18

This is a reader-facing companion. The **code and config are the source of
truth** — every command, flag, and line reference below was transcribed from
the files cited. If they have drifted since 2026-07-18, trust the files:

- `.github/workflows/deploy.yml` (the unconditional migration step)
- `scripts/run_migrations.sh` (the auto-detecting, ledger-tracking runner)
- `database/**/*.sql` (the migration files — ALL schema dirs are in scope)
- `tests/integration/test_migrations_no_inner_txn.py` (the no-inner-txn lint)

> **History note**: until mid-2026 this runbook correctly said "PROD = MANUAL —
> nothing in CI applies a migration", because the deploy's migration step was
> gated on `SUPABASE_DB_URL`, which the droplet doesn't set. That gate is gone:
> the runner grew a docker-exec mode and the deploy now calls it
> unconditionally. If you find prose elsewhere claiming prod migrations are
> manual-only, it predates this change.

---

## TL;DR

**Migrations apply AUTOMATICALLY on every production deploy.** The deploy
workflow runs `scripts/run_migrations.sh` unconditionally. The runner
auto-detects its connection:

1. `SUPABASE_DB_URL` set → `psql` against that URL (CI / remote / workstation)
2. else → `docker exec` into the `supabase-db` container (the droplet's
   self-hosted Supabase stack exposes only REST creds, no DB URL)

It scans **all** `database/` schema dirs (`migrations`, `memory`, `core`, `ml`,
`causal`, `chat`, `rag`, `audit`), applies pending files in order, and records
each in `public.schema_migrations`. Files are idempotent and the ledger was
baselined (PRs #676 + #682), so a deploy re-scour is a clean no-op — only
genuinely-new files apply.

**So the normal flow is: merge the migration to `main` → the next deploy
applies it.** The manual `docker exec ... psql` path below remains valid for
urgent/out-of-band applies — but note it bypasses the ledger (§5).

---

## 1. How migrations apply on deploy

`deploy.yml` (SSH deploy script, after the `git reset --hard origin/main`):

```bash
# Apply DB migrations. run_migrations.sh auto-detects the connection:
# SUPABASE_DB_URL when set (CI/remote), else `docker exec` into the
# supabase-db container. ...
bash scripts/run_migrations.sh
```

There is **no conditional gate** — the step runs every deploy. If a migration
fails, `set -e` fails the deploy before the app-services flip (the droplet
keeps serving the pre-deploy version).

What the runner does (`scripts/run_migrations.sh`):

- **Connection auto-detect**: `SUPABASE_DB_URL` mode needs a host `psql`;
  docker mode needs `docker exec supabase-db` to work. Neither → hard error.
- **Scope**: all 8 `database/` dirs, each namespaced in the ledger by a key
  prefix (e.g. `ml/011_...` vs plain `011_...`) so identically-numbered files
  never collide. Numbers are also not unique **within** a directory:
  `database/migrations/` has shared-number pairs (two `099_*.sql`, two
  `101_*.sql`). The ledger keys by full filename, so both members of a pair
  are tracked and applied independently — a shared number is harmless, but
  prefer the next unused number for new files.
- **Safety skips**: `*_validation_queries.sql`, `rollback_*.sql`, and
  `*_rollback.sql` are never auto-applied (rollback utils DROP live objects).
- **Transaction wrapping**: each file normally applies under
  `psql --single-transaction` (body + ledger insert commit together). Files
  containing non-transactional DDL (`ALTER TYPE ... ADD VALUE`,
  `... CONCURRENTLY`) or a self-managed `COMMIT;` are detected and applied
  **unwrapped**, with the ledger row inserted separately only after a clean
  exit — a file that fails mid-way stays untracked and idempotently retries
  next deploy.
- **Ledger**: `public.schema_migrations (filename, applied_at)` — written in
  BOTH url and docker modes.
- `--dry-run` lists pending without applying; `--baseline` records
  everything-present as applied without running it (one-time adoption on an
  already-migrated DB — already done for prod).

| Path | Applies to live prod DB? | Notes |
| --- | --- | --- |
| `deploy.yml` migration step | **Yes — every deploy** | Unconditional; docker-exec mode on the droplet; ledger-tracked |
| `scripts/run_migrations.sh` run by hand | Yes | Same behavior; safe to run any time (pending-only) |
| CI `test_migrations_no_inner_txn.py` | No | Filesystem-only lint, no DB connection |
| `docker exec -i supabase-db psql ...` by hand | Yes — manual | Urgent/out-of-band path; **bypasses the ledger** (§5) |

---

## 2. The manual apply command (out-of-band path)

For an urgent apply that can't wait for a deploy (or to apply something the
runner skips), the droplet runs a self-hosted Supabase **Docker** stack with
the Postgres container named `supabase-db`:

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

Prefer running the tracked runner instead when possible — same droplet, ledger
maintained:

```bash
./scripts/run_migrations.sh --dry-run   # list pending
./scripts/run_migrations.sh             # apply pending, ledger-tracked
```

If you DO apply a file raw via `docker exec`, the next deploy's re-scour will
see it as "pending" (no ledger row) and re-run it — harmless **only because**
migration files are required to be idempotent. Non-idempotent SQL must go
through the runner.

---

## 3. Running the runner off-droplet

In any environment with a `SUPABASE_DB_URL` and `postgresql-client`, the runner
uses url mode:

```bash
export SUPABASE_DB_URL='postgresql://user:pass@host:5432/dbname'
./scripts/run_migrations.sh --dry-run
./scripts/run_migrations.sh
```

`SUPABASE_DB_CONTAINER` overrides the container name for docker mode (default
`supabase-db`).

---

## 4. CONSTRAINT: no inner transaction (no `BEGIN` / `COMMIT`)

Migration files **must not contain a script-level transaction-control
statement** — no `BEGIN;`, `COMMIT;`, `ROLLBACK;`, `END;`, `ABORT;`, or
`START TRANSACTION;`. The runner wraps each file in
`psql --single-transaction`, which owns the outer transaction (the migration
body **plus** the `INSERT INTO public.schema_migrations` bookkeeping row). An
inner `COMMIT;` would prematurely commit before the bookkeeping insert —
leaving a migration applied but unrecorded (silent ledger drift); an inner
`ROLLBACK;` is the inverse hazard. (The runner detects such files and applies
them unwrapped with separate tracking — see §1 — but the lint keeps the
default shape clean.)

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
> the runner applies such files unwrapped automatically.

---

## 5. Verify it applied

**Runner path**: check the ledger —

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT filename, applied_at FROM public.schema_migrations ORDER BY applied_at DESC LIMIT 10;"
```

**Raw `docker exec` path**: bypasses the ledger entirely — no row is written,
so do not rely on `schema_migrations` to know what a manual apply landed.

Either way, the decisive check is an explicit **cast/select disproof** against
the object the migration created. For example, after adding
`experiment_monitor` to an enum (mig 055), confirm the value now casts (a value
that does not exist raises `22P02 invalid_text_representation`):

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

## 7. Operator checklist (ship a new migration to prod)

1. Put the file in the right `database/` dir with the next number; keep it
   **idempotent** (`IF NOT EXISTS` / `IF EXISTS` guards).
2. Confirm it has **no script-level `BEGIN`/`COMMIT`/`ROLLBACK`/`END`/`ABORT`**
   (CI lint `test_migrations_no_inner_txn.py` should already be green).
3. Merge to `main`. If the same PR touches deploy-triggering paths (`src/`,
   `config/`, `frontend/`, deps), the deploy fires and **applies the migration
   automatically**. A docs-only or migration-only merge does NOT trigger a
   deploy — for those, either wait for the next deploy or run the runner (or
   the manual `docker exec` apply) on the droplet yourself.
4. Verify with the ledger and/or a cast/select disproof against the migration's
   new object (§5).

---

## Cross-reference

- `DEPLOYMENT.md` — "Production Deploy (CI/CD)" section (the migration step's
  place in the gated rollout ordering).
- The v3 success-criteria + QC gate doc: `docs/model_success_criteria.md`.
