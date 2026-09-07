# Database Migrations Runbook (PROD = AUTO on deploy, manual path available)

**Status**: Operator runbook | **Last verified against code**: 2026-09-07

This is a reader-facing companion. The **code and config are the source of
truth** — every command, flag, and line reference below was transcribed from
the files cited. If they have drifted since 2026-09-07, trust the files:

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
2. else → `docker exec` into the `supabase-db` container

> **The droplet DOES define `SUPABASE_DB_URL`** — `grep -c '^SUPABASE_DB_URL=' .env`
> → 1 (measured 2026-09-07), pointing at `127.0.0.1:5432`, the Supavisor pooler.
> (`supabase-db` itself publishes **5433** on the host: `docker port supabase-db`
> → `5432/tcp -> 0.0.0.0:5433`.) Deploys take docker mode anyway for a different
> reason: **the SSH deploy step does not export `.env`**, so the variable is
> simply not in the runner's environment there.
>
> **Consequence, and it is a real trap:** a shell that has sourced `.env` — which
> an operator's shell on this box very often has — flips the runner into **url
> mode**. Same database, different code path and different `psql` binary
> requirement (url mode needs a host `psql`; the droplet's is inside the
> container). If you want the deploy's exact behaviour by hand, unset it:
> `env -u SUPABASE_DB_URL ./scripts/run_migrations.sh --dry-run`.

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

`deploy.yml` (SSH deploy script). Placement matters: the migration step runs
**after** the tree has been reset to the deploy's *resolved target sha* (which is
not always `origin/main` — see [`deploy-operations.md`](deploy-operations.md) §1)
and **after** the published-image assertion, so a deploy that cannot find its GHCR
images fails with *nothing migrated* (#1780/#1785):

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
  The droplet has the variable in `.env` but not in the deploy's SSH environment,
  so deploys land in docker mode — see the note in the TL;DR before running the
  runner by hand from a shell that has sourced `.env`.
- **Scope**: all 8 `database/` dirs, each namespaced in the ledger by a key
  prefix (e.g. `ml/011_...` vs plain `011_...`) so identically-numbered files
  never collide. Numbers are also not unique **within** a directory:
  `database/migrations/` has several shared-number pairs (**5 shared numbers as
  of 2026-09-07** — re-measure rather than trusting this count, it grows):

  ```bash
  ls database/migrations/*.sql | sed 's|.*/||' | grep -oE '^[0-9]+' | sort | uniq -d
  ```

  The ledger keys by full filename, so every member of a pair is tracked and
  applied independently — a shared number is harmless, but prefer the next
  unused number for new files.
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
the object the migration created — and it must be cast against the object *that
migration* touched. **There are two agent-name enums in this database**, and
picking the wrong one gives a check that cannot fail:

| Enum | Owner | Labels (2026-09-07) |
| --- | --- | --- |
| `agent_name_enum` | observability schema — **what migration 055 altered** | 21 |
| `e2i_agent_name` | memory schema (`database/memory/029_*`) | 23 |

Both currently contain `experiment_monitor`, so
`SELECT 'experiment_monitor'::e2i_agent_name;` succeeds **whether or not
migration 055 ever applied**. It is a false positive, not a verification. Cast
against `agent_name_enum`:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT 'experiment_monitor'::agent_name_enum;"
# Expected: returns the value. A 22P02 invalid_text_representation means
# migration 055 did NOT apply.
```

(`database/migrations/055_add_missing_agents_to_agent_name_enum.sql` runs
`ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS …` for `cohort_constructor`
and `experiment_monitor`.)

For an enum, you can also list the full label set — substitute the type the
migration actually names:

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT enumlabel FROM pg_enum e
     JOIN pg_type t ON e.enumtypid = t.oid
    WHERE t.typname = 'agent_name_enum'
    ORDER BY e.enumsortorder;"
```

Pick a disproof that is specific to what the migration changed (a new column,
table, type, or enum value), that **errors** when the change is absent, and that
names the same object the migration file names. A check that passes on an
unmigrated database is worse than no check.

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
3. Merge to `main`. If the same PR touches deploy-triggering paths, the deploy
   fires and **applies the migration automatically**. The trigger list is
   `on.push.paths` in `.github/workflows/deploy.yml` — read it there rather than
   trusting a copy; as of 2026-09-07 it is `src/**`, `config/**`,
   `docker/Dockerfile`, the three compose files the deploy `-f`s,
   `docker/frontend/**`, `frontend/**`, `requirements.txt`, `requirements.lock`,
   `pyproject.toml`, `patches/**`, `scripts/bentoml/**`, `docker/bentoml/**`,
   `scripts/deploy/**`, **`scripts/**`** (the whole tree is `COPY`ed into the
   production image, #1783) and **`data/kg_cache/**`** (the KG Layer-2 caches are
   baked into the image, #1607/#1783).

   **`database/**` is deliberately NOT a trigger.** Migration files are not image
   inputs — the runner reads them from the droplet checkout after the reset — so a
   migration-only merge (like a docs-only one) does **not** fire a deploy. For
   those, either wait for the next deploy to pick it up, or run the runner (or the
   manual `docker exec` apply) on the droplet yourself.
4. Verify with the ledger and/or a cast/select disproof against the migration's
   new object (§5).

---

## Cross-reference

- `DEPLOYMENT.md` — "Production Deploy (CI/CD)" section (the migration step's
  place in the gated rollout ordering).
- The v3 success-criteria + QC gate doc: `docs/model_success_criteria.md`.
