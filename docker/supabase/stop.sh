#!/bin/bash
# =============================================================================
# Stop Supabase Self-Hosted Services
# =============================================================================
# This script stops the self-hosted Supabase services.
#
# Usage:
#   ./stop.sh [--remove-volumes]
# =============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPABASE_DIR="${SUPABASE_DIR:-/opt/supabase/docker}"
E2I_OVERRIDE="$SCRIPT_DIR/docker-compose.override.yml"

# Parse arguments
REMOVE_VOLUMES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove-volumes)
            REMOVE_VOLUMES="-v"
            echo -e "${YELLOW}WARNING: This will delete all Supabase data!${NC}"
            read -p "Are you sure? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--remove-volumes]"
            echo ""
            echo "Options:"
            echo "  --remove-volumes  Remove data volumes (WARNING: deletes all data)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$SUPABASE_DIR" ]]; then
    echo "Supabase directory not found: $SUPABASE_DIR"
    exit 1
fi

cd "$SUPABASE_DIR"

echo "Stopping Supabase services..."
docker compose -f docker-compose.yml -f "$E2I_OVERRIDE" down $REMOVE_VOLUMES

echo -e "${GREEN}Supabase services stopped${NC}"
