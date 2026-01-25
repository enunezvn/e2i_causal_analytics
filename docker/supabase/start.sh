#!/bin/bash
# =============================================================================
# Start Supabase Self-Hosted Services
# =============================================================================
# This script starts the self-hosted Supabase services with E2I integration.
#
# Prerequisites:
#   - Supabase cloned to /opt/supabase
#   - .env configured in /opt/supabase/docker/
#   - Docker and Docker Compose installed
#
# Usage:
#   ./start.sh [--detach] [--pull]
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
DETACH="-d"
PULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-detach)
            DETACH=""
            shift
            ;;
        --pull)
            PULL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--no-detach] [--pull]"
            echo ""
            echo "Options:"
            echo "  --no-detach    Run in foreground (default: detached)"
            echo "  --pull         Pull latest images before starting"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate setup
if [[ ! -d "$SUPABASE_DIR" ]]; then
    echo -e "${YELLOW}Supabase not found at $SUPABASE_DIR${NC}"
    echo "Please run: scripts/supabase/setup_self_hosted.sh"
    exit 1
fi

if [[ ! -f "$SUPABASE_DIR/.env" ]]; then
    echo -e "${YELLOW}Environment file not found${NC}"
    echo "Please copy and configure: docker/supabase/.env.template -> $SUPABASE_DIR/.env"
    exit 1
fi

cd "$SUPABASE_DIR"

# Ensure E2I network exists
echo "Ensuring E2I network exists..."
docker network create e2i-backend-network 2>/dev/null || true

# Pull latest images if requested
if [[ "$PULL" == "true" ]]; then
    echo "Pulling latest Supabase images..."
    docker compose -f docker-compose.yml -f "$E2I_OVERRIDE" pull
fi

# Start services
echo -e "${GREEN}Starting Supabase services...${NC}"
docker compose -f docker-compose.yml -f "$E2I_OVERRIDE" up $DETACH

echo ""
echo -e "${GREEN}Supabase services started!${NC}"
echo ""
echo "Service URLs:"
echo "  - API Gateway:  http://localhost:8000"
echo "  - Studio:       http://localhost:3001"
echo "  - PostgreSQL:   localhost:5433"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/rest/v1/ -H 'apikey: YOUR_ANON_KEY'"
echo ""
echo "View logs:"
echo "  docker compose -f docker-compose.yml -f $E2I_OVERRIDE logs -f"
