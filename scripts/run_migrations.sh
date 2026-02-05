#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Database Migration Runner
# =============================================================================
# Applies SQL migrations from database/migrations/ in alphabetical order,
# tracking applied migrations in public.schema_migrations.
#
# Usage:
#   ./scripts/run_migrations.sh              # Apply pending migrations
#   ./scripts/run_migrations.sh --dry-run    # List pending without applying
#
# Environment:
#   SUPABASE_DB_URL  - PostgreSQL connection string (required)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MIGRATIONS_DIR="$PROJECT_ROOT/database/migrations"
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --dry-run)
      DRY_RUN=true
      ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"
      echo ""
      echo "Options:"
      echo "  --dry-run  List pending migrations without applying"
      echo "  --help     Show this help message"
      exit 0
      ;;
  esac
done

# Validate environment
if [ -z "${SUPABASE_DB_URL:-}" ]; then
  echo -e "${RED}ERROR: SUPABASE_DB_URL environment variable is required${NC}"
  echo "Example: export SUPABASE_DB_URL='postgresql://user:pass@host:5432/dbname'"
  exit 1
fi

# Validate psql is available
if ! command -v psql &> /dev/null; then
  echo -e "${RED}ERROR: psql not found. Install postgresql-client.${NC}"
  exit 1
fi

# Validate migrations directory exists
if [ ! -d "$MIGRATIONS_DIR" ]; then
  echo -e "${RED}ERROR: Migrations directory not found: $MIGRATIONS_DIR${NC}"
  exit 1
fi

echo -e "${GREEN}=== E2I Database Migration Runner ===${NC}"

# Create schema_migrations table if it doesn't exist
psql "$SUPABASE_DB_URL" -q <<'SQL'
CREATE TABLE IF NOT EXISTS public.schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SQL

# Get list of already-applied migrations
APPLIED=$(psql "$SUPABASE_DB_URL" -t -A -c "SELECT filename FROM public.schema_migrations ORDER BY filename;")

# Find and process migration files
PENDING=0
APPLIED_COUNT=0

for migration_file in "$MIGRATIONS_DIR"/*.sql; do
  [ -f "$migration_file" ] || continue
  filename=$(basename "$migration_file")

  # Check if already applied
  if echo "$APPLIED" | grep -qxF "$filename"; then
    continue
  fi

  PENDING=$((PENDING + 1))

  if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[PENDING]${NC} $filename"
    continue
  fi

  echo -n "Applying $filename ... "

  # Apply migration in a transaction
  if psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 --single-transaction -q <<SQL
\i $migration_file
INSERT INTO public.schema_migrations (filename) VALUES ('$filename');
SQL
  then
    echo -e "${GREEN}OK${NC}"
    APPLIED_COUNT=$((APPLIED_COUNT + 1))
  else
    echo -e "${RED}FAILED${NC}"
    echo -e "${RED}Migration $filename failed. Aborting.${NC}"
    exit 1
  fi
done

if [ "$DRY_RUN" = true ]; then
  if [ "$PENDING" -eq 0 ]; then
    echo -e "${GREEN}No pending migrations.${NC}"
  else
    echo -e "${YELLOW}$PENDING migration(s) pending.${NC}"
  fi
else
  if [ "$APPLIED_COUNT" -eq 0 ] && [ "$PENDING" -eq 0 ]; then
    echo -e "${GREEN}Database is up to date. No migrations to apply.${NC}"
  else
    echo -e "${GREEN}Applied $APPLIED_COUNT migration(s) successfully.${NC}"
  fi
fi
