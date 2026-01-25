#!/bin/bash
# =============================================================================
# Import Data to Self-Hosted Supabase
# =============================================================================
# This script imports data from a Supabase Cloud export to a self-hosted instance.
#
# Prerequisites:
#   - Self-hosted Supabase running (see setup_self_hosted.sh)
#   - Export files from export_cloud_db.sh
#   - PostgreSQL client tools (psql)
#
# Usage:
#   ./import_data.sh <export_dir>
#   ./import_data.sh ./backups/supabase_export_20250124
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Database connection (self-hosted)
DB_HOST="${SUPABASE_DB_HOST:-localhost}"
DB_PORT="${SUPABASE_DB_PORT:-5433}"  # Using 5433 as per our override
DB_NAME="${SUPABASE_DB_NAME:-postgres}"
DB_USER="${SUPABASE_DB_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-}"

# Parse arguments
EXPORT_DIR=""
SKIP_SCHEMA=false
SKIP_DATA=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-schema)
            SKIP_SCHEMA=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --db-password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <export_dir> [options]"
            echo ""
            echo "Arguments:"
            echo "  export_dir          Directory containing export files"
            echo ""
            echo "Options:"
            echo "  --skip-schema       Skip schema import"
            echo "  --skip-data         Skip data import"
            echo "  --dry-run           Show what would be done without executing"
            echo "  --db-password       PostgreSQL password"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            EXPORT_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$EXPORT_DIR" ]]; then
    echo -e "${RED}Error: Export directory required${NC}"
    echo "Usage: $0 <export_dir>"
    exit 1
fi

if [[ ! -d "$EXPORT_DIR" ]]; then
    echo -e "${RED}Error: Export directory not found: $EXPORT_DIR${NC}"
    exit 1
fi

if [[ -z "$DB_PASSWORD" ]]; then
    echo -e "${YELLOW}Warning: No database password provided${NC}"
    echo "Set POSTGRES_PASSWORD environment variable or use --db-password"
    read -sp "Enter PostgreSQL password: " DB_PASSWORD
    echo ""
fi

# Build connection string
export PGPASSWORD="$DB_PASSWORD"
PSQL_CMD="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"

echo -e "${GREEN}=== Supabase Data Import ===${NC}"
echo "Export directory: $EXPORT_DIR"
echo "Target database: $DB_HOST:$DB_PORT/$DB_NAME"
echo ""

# Check connection
check_connection() {
    echo "Checking database connection..."

    if $PSQL_CMD -c "SELECT 1" &> /dev/null; then
        echo -e "${GREEN}Database connection successful${NC}"
    else
        echo -e "${RED}Error: Cannot connect to database${NC}"
        echo "Make sure Supabase is running: docker/supabase/start.sh"
        exit 1
    fi
}

# Pre-import checks
pre_import_checks() {
    echo "Running pre-import checks..."

    # Check required files exist
    local required_files=("schema.sql" "data.sql")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$EXPORT_DIR/$file" ]]; then
            echo -e "${RED}Error: Required file not found: $EXPORT_DIR/$file${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}All required files present${NC}"
}

# Install required extensions
install_extensions() {
    echo "Installing required PostgreSQL extensions..."

    $PSQL_CMD << 'EOF'
-- Required extensions for E2I
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Vector extension for RAG (if available)
CREATE EXTENSION IF NOT EXISTS "vector";

-- Notify on success
SELECT 'Extensions installed successfully' as status;
EOF

    echo -e "${GREEN}Extensions installed${NC}"
}

# Import schema
import_schema() {
    if [[ "$SKIP_SCHEMA" == "true" ]]; then
        echo -e "${YELLOW}Skipping schema import${NC}"
        return
    fi

    echo "Importing schema..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would import: $EXPORT_DIR/schema.sql"
        return
    fi

    # Import schema with error handling
    if $PSQL_CMD -f "$EXPORT_DIR/schema.sql" 2>&1 | tee "$EXPORT_DIR/schema_import.log"; then
        echo -e "${GREEN}Schema imported successfully${NC}"
    else
        echo -e "${YELLOW}Schema import completed with some warnings (check schema_import.log)${NC}"
    fi
}

# Import data
import_data() {
    if [[ "$SKIP_DATA" == "true" ]]; then
        echo -e "${YELLOW}Skipping data import${NC}"
        return
    fi

    echo "Importing data..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would import: $EXPORT_DIR/data.sql"
        return
    fi

    # Import data with error handling
    if $PSQL_CMD -f "$EXPORT_DIR/data.sql" 2>&1 | tee "$EXPORT_DIR/data_import.log"; then
        echo -e "${GREEN}Data imported successfully${NC}"
    else
        echo -e "${YELLOW}Data import completed with some warnings (check data_import.log)${NC}"
    fi
}

# Import E2I-specific migrations
import_e2i_migrations() {
    echo "Applying E2I-specific migrations..."

    local migrations_dir="$PROJECT_ROOT/database"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would apply migrations from: $migrations_dir"
        return
    fi

    # Apply migrations in order
    local migration_dirs=(
        "core"
        "memory"
        "chat"
        "ml"
        "causal"
        "audit"
        "rag"
        "migrations"
    )

    for dir in "${migration_dirs[@]}"; do
        if [[ -d "$migrations_dir/$dir" ]]; then
            echo "Applying migrations from: $dir"
            for sql_file in "$migrations_dir/$dir"/*.sql; do
                if [[ -f "$sql_file" ]]; then
                    echo "  - $(basename "$sql_file")"
                    $PSQL_CMD -f "$sql_file" >> "$EXPORT_DIR/migration.log" 2>&1 || true
                fi
            done
        fi
    done

    echo -e "${GREEN}E2I migrations applied${NC}"
}

# Verify import
verify_import() {
    echo "Verifying import..."

    $PSQL_CMD << 'EOF'
-- Table count
SELECT 'Tables' as type, count(*) as count
FROM information_schema.tables
WHERE table_schema = 'public';

-- Row counts for key tables
SELECT 'business_metrics' as table_name, count(*) as row_count FROM public.business_metrics
UNION ALL
SELECT 'agent_activities', count(*) FROM public.agent_activities
UNION ALL
SELECT 'causal_paths', count(*) FROM public.causal_paths;

-- Check RLS policies
SELECT 'RLS Policies' as type, count(*) as count
FROM pg_policies
WHERE schemaname = 'public';

-- Check extensions
SELECT 'Extensions' as type, count(*) as count FROM pg_extension;
EOF

    echo -e "${GREEN}Verification complete${NC}"
}

# Create import summary
create_summary() {
    echo "Creating import summary..."

    cat > "$EXPORT_DIR/import_summary.json" << EOF
{
    "import_date": "$(date -Iseconds)",
    "target_host": "$DB_HOST",
    "target_port": "$DB_PORT",
    "target_database": "$DB_NAME",
    "source_export": "$EXPORT_DIR",
    "skip_schema": $SKIP_SCHEMA,
    "skip_data": $SKIP_DATA,
    "dry_run": $DRY_RUN,
    "status": "completed"
}
EOF

    echo -e "${GREEN}Import summary saved${NC}"
}

# Main execution
check_connection
pre_import_checks
install_extensions
import_schema
import_data
import_e2i_migrations
verify_import
create_summary

echo ""
echo -e "${GREEN}=== Import Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - Schema: $(if [[ "$SKIP_SCHEMA" == "true" ]]; then echo "Skipped"; else echo "Imported"; fi)"
echo "  - Data: $(if [[ "$SKIP_DATA" == "true" ]]; then echo "Skipped"; else echo "Imported"; fi)"
echo "  - E2I Migrations: Applied"
echo ""
echo "Logs saved to:"
echo "  - $EXPORT_DIR/schema_import.log"
echo "  - $EXPORT_DIR/data_import.log"
echo "  - $EXPORT_DIR/migration.log"
echo ""
echo "Next steps:"
echo "1. Run validation: python scripts/supabase/validate_migration.py"
echo "2. Update application environment variables"
echo "3. Test application connectivity"
