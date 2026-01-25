#!/bin/bash
# =============================================================================
# Export Supabase Cloud Database
# =============================================================================
# This script exports schema, data, and configuration from Supabase Cloud
# for migration to a self-hosted instance.
#
# Prerequisites:
#   - Supabase CLI installed (npm install -g supabase)
#   - Project linked (supabase link --project-ref <ref>)
#   - OR: Cloud database connection string available
#
# Usage:
#   ./export_cloud_db.sh [output_dir]
#   ./export_cloud_db.sh --connection-string "postgresql://..."
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/backups/supabase_export_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

# Parse arguments
CONNECTION_STRING=""
USE_CLI=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --connection-string)
            CONNECTION_STRING="$2"
            USE_CLI=false
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [output_dir] [--connection-string <postgres_url>]"
            echo ""
            echo "Options:"
            echo "  output_dir           Directory to save exports (default: backups/supabase_export_<timestamp>)"
            echo "  --connection-string  PostgreSQL connection string for direct export"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            OUTPUT_DIR="$1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}=== Supabase Cloud Export ===${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to export using Supabase CLI
export_with_cli() {
    echo -e "${YELLOW}Using Supabase CLI for export...${NC}"

    # Check if CLI is installed
    if ! command -v supabase &> /dev/null; then
        echo -e "${RED}Error: Supabase CLI not installed${NC}"
        echo "Install with: npm install -g supabase"
        exit 1
    fi

    # Check if project is linked
    if [[ ! -f "$PROJECT_ROOT/supabase/.temp/project-ref" ]]; then
        echo -e "${YELLOW}Project not linked. Attempting to link...${NC}"
        echo "Please run: supabase link --project-ref <your-project-ref>"
        exit 1
    fi

    cd "$PROJECT_ROOT"

    # Export schema (excludes Supabase-managed schemas)
    echo "Exporting schema..."
    supabase db dump -f "$OUTPUT_DIR/schema.sql"
    echo -e "${GREEN}Schema exported to: $OUTPUT_DIR/schema.sql${NC}"

    # Export data
    echo "Exporting data..."
    supabase db dump -f "$OUTPUT_DIR/data.sql" --data-only --use-copy
    echo -e "${GREEN}Data exported to: $OUTPUT_DIR/data.sql${NC}"

    # Export roles
    echo "Exporting roles..."
    supabase db dump -f "$OUTPUT_DIR/roles.sql" --role-only || true
    echo -e "${GREEN}Roles exported to: $OUTPUT_DIR/roles.sql${NC}"

    # Export migration history
    echo "Exporting migration history..."
    supabase db dump -f "$OUTPUT_DIR/migration_history.sql" --use-copy --data-only --schema supabase_migrations || true
    echo -e "${GREEN}Migration history exported to: $OUTPUT_DIR/migration_history.sql${NC}"
}

# Function to export using pg_dump directly
export_with_pgdump() {
    echo -e "${YELLOW}Using pg_dump for export...${NC}"

    # Check if pg_dump is installed
    if ! command -v pg_dump &> /dev/null; then
        echo -e "${RED}Error: pg_dump not installed${NC}"
        echo "Install with: apt-get install postgresql-client"
        exit 1
    fi

    # Export schema
    echo "Exporting schema..."
    pg_dump "$CONNECTION_STRING" \
        --schema-only \
        --no-owner \
        --no-privileges \
        --no-subscriptions \
        --exclude-schema=auth \
        --exclude-schema=storage \
        --exclude-schema=realtime \
        --exclude-schema=supabase_functions \
        --exclude-schema=extensions \
        --exclude-schema=graphql \
        --exclude-schema=graphql_public \
        --exclude-schema=pgsodium \
        --exclude-schema=vault \
        -f "$OUTPUT_DIR/schema.sql"
    echo -e "${GREEN}Schema exported to: $OUTPUT_DIR/schema.sql${NC}"

    # Export data
    echo "Exporting data..."
    pg_dump "$CONNECTION_STRING" \
        --data-only \
        --no-owner \
        --no-privileges \
        --use-copy \
        --exclude-schema=auth \
        --exclude-schema=storage \
        --exclude-schema=realtime \
        --exclude-schema=supabase_functions \
        --exclude-schema=extensions \
        --exclude-schema=graphql \
        --exclude-schema=graphql_public \
        --exclude-schema=pgsodium \
        --exclude-schema=vault \
        -f "$OUTPUT_DIR/data.sql"
    echo -e "${GREEN}Data exported to: $OUTPUT_DIR/data.sql${NC}"

    # Export full backup (for safety)
    echo "Creating full backup..."
    pg_dump "$CONNECTION_STRING" \
        --no-owner \
        --no-privileges \
        --format=custom \
        -f "$OUTPUT_DIR/full_backup.dump"
    echo -e "${GREEN}Full backup exported to: $OUTPUT_DIR/full_backup.dump${NC}"
}

# Export auth schema modifications (if any)
export_auth_modifications() {
    echo "Checking for auth schema modifications..."

    if [[ -n "$CONNECTION_STRING" ]]; then
        # Export auth triggers and policies
        psql "$CONNECTION_STRING" -t -A << 'EOF' > "$OUTPUT_DIR/auth_modifications.sql"
-- Export auth triggers on public tables
SELECT pg_get_triggerdef(t.oid) || ';'
FROM pg_trigger t
JOIN pg_class c ON t.tgrelid = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE n.nspname = 'public'
AND t.tgname LIKE '%auth%';

-- Export RLS policies related to auth
SELECT 'CREATE POLICY ' || pol.polname || ' ON ' || c.relname ||
       CASE pol.polcmd
         WHEN 'r' THEN ' FOR SELECT'
         WHEN 'a' THEN ' FOR INSERT'
         WHEN 'w' THEN ' FOR UPDATE'
         WHEN 'd' THEN ' FOR DELETE'
         ELSE ' FOR ALL'
       END ||
       ' USING (' || pg_get_expr(pol.polqual, pol.polrelid) || ');'
FROM pg_policy pol
JOIN pg_class c ON pol.polrelid = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE n.nspname = 'public'
AND pg_get_expr(pol.polqual, pol.polrelid) LIKE '%auth.%';
EOF
    fi
}

# Main execution
if [[ "$USE_CLI" == "true" ]]; then
    export_with_cli
else
    export_with_pgdump
fi

export_auth_modifications

# Create manifest
echo "Creating export manifest..."
cat > "$OUTPUT_DIR/manifest.json" << EOF
{
    "export_date": "$(date -Iseconds)",
    "source": "supabase_cloud",
    "method": "$( [[ "$USE_CLI" == "true" ]] && echo "supabase_cli" || echo "pg_dump" )",
    "files": [
        "schema.sql",
        "data.sql",
        "roles.sql",
        "migration_history.sql",
        "auth_modifications.sql"
    ],
    "e2i_version": "4.3",
    "notes": "Export for migration to self-hosted Supabase"
}
EOF

echo ""
echo -e "${GREEN}=== Export Complete ===${NC}"
echo "Files exported to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review exported files"
echo "2. Set up self-hosted Supabase (see setup_self_hosted.sh)"
echo "3. Import data (see import_data.sh)"
