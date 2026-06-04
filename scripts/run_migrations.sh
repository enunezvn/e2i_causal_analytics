#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Database Migration Runner
# =============================================================================
# Applies SQL migrations from database/migrations/ AND database/memory/ in
# alphabetical order, tracking applied migrations in public.schema_migrations.
#
# Connection (auto-detected, in priority order):
#   1. SUPABASE_DB_URL  - psql connection string (CI / remote), OR
#   2. docker exec into $SUPABASE_DB_CONTAINER (default: supabase-db) when the
#      runner executes on the droplet itself (the self-hosted Supabase stack
#      exposes no SUPABASE_DB_URL — REST creds only — so the deploy historically
#      SKIPPED migrations; docker-exec mode closes that gap).
#
# Usage:
#   ./scripts/run_migrations.sh              # Apply pending migrations
#   ./scripts/run_migrations.sh --dry-run    # List pending without applying
#
# Environment:
#   SUPABASE_DB_URL        - PostgreSQL connection string (optional; see above)
#   SUPABASE_DB_CONTAINER  - docker container running Postgres (default supabase-db)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_CONTAINER="${SUPABASE_DB_CONTAINER:-supabase-db}"
DRY_RUN=false

# Migration directories applied in order, each as "dir::keyprefix". The keyprefix
# namespaces the schema_migrations.filename so identically-numbered files in
# different dirs (e.g. ml/011 vs migrations/011) never collide.
#
# ALL schema dirs are now in scope: the previously-deferred dirs (core, ml,
# causal, chat, rag, audit) were reconciled to the droplet's deployed reality and
# tracked/baselined in PR #676 + #682 (every in-scope file is idempotent and
# already recorded in public.schema_migrations, so a deploy re-scour is a clean
# no-op; only genuinely-new files apply). See the reconciliation plan +
# completion notes in docs/reports/memory-system-review-20260603.md.
#
# SAFETY: rollback_*/*_rollback and *_validation_queries files are NOT forward
# migrations and are excluded in apply_dir() below — a rollback util (e.g.
# ml/rollback_023.sql) DROPs live tables and must never be auto-applied.
MIGRATION_DIRS=(
  "$PROJECT_ROOT/database/migrations::"
  "$PROJECT_ROOT/database/memory::memory/"
  "$PROJECT_ROOT/database/core::core/"
  "$PROJECT_ROOT/database/ml::ml/"
  "$PROJECT_ROOT/database/causal::causal/"
  "$PROJECT_ROOT/database/chat::chat/"
  "$PROJECT_ROOT/database/rag::rag/"
  "$PROJECT_ROOT/database/audit::audit/"
)

BASELINE=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --baseline) BASELINE=true ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--baseline]"
      echo "  --dry-run   List pending migrations without applying"
      echo "  --baseline  Record every present migration as applied WITHOUT running it."
      echo "              One-time adoption of tracking on an already-migrated DB so"
      echo "              future deploys only apply genuinely-new files. Run this AFTER"
      echo "              applying any known-pending backlog (else it marks those skipped)."
      exit 0
      ;;
  esac
done

# ----------------------------------------------------------------------------
# Connection mode detection
# ----------------------------------------------------------------------------
MODE=""
if [ -n "${SUPABASE_DB_URL:-}" ]; then
  MODE="url"
  command -v psql >/dev/null 2>&1 || { echo -e "${RED}ERROR: psql not found (needed for SUPABASE_DB_URL mode).${NC}"; exit 1; }
elif command -v docker >/dev/null 2>&1 && docker exec "$DB_CONTAINER" true >/dev/null 2>&1; then
  MODE="docker"
else
  echo -e "${RED}ERROR: no database connection available.${NC}"
  echo "Set SUPABASE_DB_URL, or run on a host where 'docker exec $DB_CONTAINER' works."
  exit 1
fi

# psql wrapper — reads a SQL script from stdin and/or takes psql args.
run_psql() {
  if [ "$MODE" = "url" ]; then
    psql "$SUPABASE_DB_URL" "$@"
  else
    docker exec -i "$DB_CONTAINER" psql -U postgres -d postgres "$@"
  fi
}

echo -e "${GREEN}=== E2I Database Migration Runner (mode=${MODE}) ===${NC}"

# Tracking table
run_psql -q >/dev/null <<'SQL'
CREATE TABLE IF NOT EXISTS public.schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SQL

APPLIED=$(run_psql -t -A -c "SELECT filename FROM public.schema_migrations ORDER BY filename;")

PENDING=0
APPLIED_COUNT=0

apply_dir() {
  local dir="$1" prefix="$2"
  [ -d "$dir" ] || return 0
  local migration_file filename key txn
  for migration_file in "$dir"/*.sql; do
    [ -f "$migration_file" ] || continue
    filename=$(basename "$migration_file")
    # Skip non-forward-migration files: read-only validation query bundles AND
    # destructive rollback utilities. A rollback_*/*_rollback file DROPs live
    # objects (e.g. ml/rollback_023.sql DROPs estimator_evaluations) and would
    # silently destroy prod data if auto-applied. .cypher files are already
    # excluded by the *.sql glob.
    case "$filename" in
      *_validation_queries.sql|rollback_*.sql|*_rollback.sql) continue ;;
    esac
    key="${prefix}${filename}"

    if echo "$APPLIED" | grep -qxF "$key"; then
      continue
    fi

    PENDING=$((PENDING + 1))
    if [ "$DRY_RUN" = true ]; then
      echo -e "${YELLOW}[PENDING]${NC} $key"
      continue
    fi

    if [ "$BASELINE" = true ]; then
      run_psql -q -c "INSERT INTO public.schema_migrations(filename) VALUES ('${key//\'/\'\'}') ON CONFLICT DO NOTHING;" >/dev/null
      echo -e "${YELLOW}[BASELINED]${NC} $key"
      APPLIED_COUNT=$((APPLIED_COUNT + 1))
      continue
    fi

    echo -n "Applying $key ... "
    # Decide whether this file may be wrapped in one --single-transaction. Two
    # statement classes must NOT be wrapped:
    #   * non-transactional DDL — `ALTER TYPE ... ADD VALUE`, `CREATE/DROP INDEX
    #     CONCURRENTLY` — errors with "cannot run inside a transaction block".
    #   * self-managed transactions — a file containing its own COMMIT; the
    #     wrapper's txn is silently ENDED by that COMMIT, so an appended tracking
    #     INSERT (or any later statement) rides outside it. We must instead apply
    #     the body alone and record tracking SEPARATELY, only after a clean exit,
    #     so a file that fails after its own COMMIT is left UNTRACKED (the next
    #     deploy idempotently retries) rather than half-applied-but-recorded.
    # Detection strips `--` line comments first so a comment mentioning these
    # keywords does not false-trigger (a real statement's keyword precedes any
    # `--` on its line, so stripping never hides a genuine match).
    # Anchor on the bare word CONCURRENTLY (covers CREATE/DROP INDEX and REINDEX,
    # single- or multi-line) — a false positive only costs per-file atomicity
    # (harmless on idempotent files), whereas a false NEGATIVE would wrap a
    # non-transactional statement and abort the deploy, so we err toward un-wrap.
    tracked_inline=true
    if sed 's/--.*$//' "$migration_file" | grep -qiE \
         "ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE|CONCURRENTLY|^[[:space:]]*COMMIT[[:space:]]*;"; then
      tracked_inline=false
    fi

    apply_ok=false
    if [ "$tracked_inline" = true ]; then
      # Normal file: body + tracking row commit (or roll back) atomically.
      if { cat "$migration_file"; printf "\nINSERT INTO public.schema_migrations(filename) VALUES ('%s') ON CONFLICT DO NOTHING;\n" "${key//\'/\'\'}"; } \
           | run_psql -v ON_ERROR_STOP=1 --single-transaction -q; then
        apply_ok=true
      fi
    else
      # Un-wrappable / self-managed-txn file: apply the body un-wrapped, then
      # record tracking in a SEPARATE invocation only if the body exited clean.
      if run_psql -v ON_ERROR_STOP=1 -q < "$migration_file"; then
        run_psql -q -c "INSERT INTO public.schema_migrations(filename) VALUES ('${key//\'/\'\'}') ON CONFLICT DO NOTHING;" >/dev/null
        apply_ok=true
      fi
    fi

    if [ "$apply_ok" = true ]; then
      echo -e "${GREEN}OK${NC}"
      APPLIED_COUNT=$((APPLIED_COUNT + 1))
    else
      echo -e "${RED}FAILED${NC}"
      echo -e "${RED}Migration $key failed. Aborting.${NC}"
      exit 1
    fi
  done
}

for spec in "${MIGRATION_DIRS[@]}"; do
  dir="${spec%%::*}"
  prefix="${spec##*::}"
  apply_dir "$dir" "$prefix"
done

if [ "$DRY_RUN" = true ]; then
  if [ "$PENDING" -eq 0 ]; then
    echo -e "${GREEN}No pending migrations.${NC}"
  else
    echo -e "${YELLOW}$PENDING migration(s) pending.${NC}"
  fi
else
  if [ "$APPLIED_COUNT" -eq 0 ]; then
    echo -e "${GREEN}Database is up to date. No migrations to apply.${NC}"
  else
    echo -e "${GREEN}Applied $APPLIED_COUNT migration(s) successfully.${NC}"
  fi
fi
