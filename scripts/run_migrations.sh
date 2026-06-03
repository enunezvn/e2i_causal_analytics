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

# Migration directories applied in this order. database/memory/ holds the
# agentic-memory + lifecycle/sentinel/crystallization schema that the old
# runner never scanned (it only looked at database/migrations/).
MIGRATION_DIRS=(
  "$PROJECT_ROOT/database/migrations::"
  "$PROJECT_ROOT/database/memory::memory/"
)

for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run  List pending migrations without applying"
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
    # Skip non-migration helpers (read-only validation query bundles). .cypher
    # files are excluded by the *.sql glob.
    case "$filename" in *_validation_queries.sql) continue ;; esac
    key="${prefix}${filename}"

    if echo "$APPLIED" | grep -qxF "$key"; then
      continue
    fi

    PENDING=$((PENDING + 1))
    if [ "$DRY_RUN" = true ]; then
      echo -e "${YELLOW}[PENDING]${NC} $key"
      continue
    fi

    echo -n "Applying $key ... "
    # `ALTER TYPE ... ADD VALUE` cannot be wrapped together with usage of the
    # new value in a single transaction, so enum migrations run un-wrapped.
    txn="--single-transaction"
    if grep -qiE "ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE" "$migration_file"; then
      txn=""
    fi

    if { cat "$migration_file"; printf "\nINSERT INTO public.schema_migrations(filename) VALUES ('%s') ON CONFLICT DO NOTHING;\n" "$key"; } \
         | run_psql -v ON_ERROR_STOP=1 $txn -q; then
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
