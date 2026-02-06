#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Deploy Script (Docker Compose Dev Mode)
# =============================================================================
# Deploys the latest code changes. The API and frontend auto-reload via
# volume mounts (uvicorn --reload / Vite HMR). Workers need an explicit restart
# because Celery does not support auto-reload in production.
#
# Usage:
#   ./scripts/deploy.sh           # Pull + restart workers
#   ./scripts/deploy.sh --build   # Pull + rebuild images + restart all
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_CMD="docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml"

cd "$PROJECT_DIR"

# Parse flags
BUILD=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--build]"
            echo "  --build   Rebuild images (use when requirements.txt or Dockerfile changes)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== E2I Deploy ==="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "=== Pulling latest code ==="
git pull

if $BUILD; then
    echo ""
    echo "=== Rebuilding images ==="
    $COMPOSE_CMD build api frontend worker_light worker_medium scheduler

    echo ""
    echo "=== Restarting services (with new images) ==="
    $COMPOSE_CMD up -d --no-deps api frontend worker_light worker_medium scheduler
else
    echo ""
    echo "=== Restarting workers (code change pickup) ==="
    $COMPOSE_CMD restart worker_light worker_medium scheduler
fi

echo ""
echo "=== Deployed $(git rev-parse --short HEAD) ==="
echo "API: auto-reloads via uvicorn --reload"
echo "Frontend: auto-reloads via Vite HMR"
echo "Workers: restarted"
