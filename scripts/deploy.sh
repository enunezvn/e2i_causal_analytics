#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Deploy Script (Docker Compose Dev Mode)
# =============================================================================
# Deploys the latest code changes. The API and frontend auto-reload via
# volume mounts (uvicorn --reload / Vite HMR). Workers need an explicit restart
# because Celery does not support auto-reload in production.
#
# Safety features:
#   - Saves current SHA before pull (enables rollback)
#   - Runs database migrations (idempotent — skips applied)
#   - Health check loop after restart (30 attempts, 10s apart)
#   - Auto-rollback on health check failure
#
# Usage:
#   ./scripts/deploy.sh           # Pull + migrate + restart workers + verify
#   ./scripts/deploy.sh --build   # Pull + migrate + rebuild + restart + verify
#   ./scripts/deploy.sh --skip-migrations  # Skip migration step
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_CMD="docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml"
HEALTH_URL="http://localhost:8000/health"
HEALTH_MAX_ATTEMPTS=30
HEALTH_INTERVAL=10

cd "$PROJECT_DIR"

# Parse flags
BUILD=false
SKIP_MIGRATIONS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        --skip-migrations) SKIP_MIGRATIONS=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--build] [--skip-migrations]"
            echo "  --build             Rebuild images (use when requirements.txt or Dockerfile changes)"
            echo "  --skip-migrations   Skip database migration step"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== E2I Deploy ==="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Secure .env permissions
chmod 600 "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.dev" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Save current SHA for rollback
# ---------------------------------------------------------------------------
PREV_SHA=$(git rev-parse HEAD)
echo "=== Current SHA: $(git rev-parse --short HEAD) ==="

echo ""
echo "=== Pulling latest code ==="
git pull

NEW_SHA=$(git rev-parse HEAD)
if [ "$PREV_SHA" = "$NEW_SHA" ]; then
    echo "Already up to date."
fi

# ---------------------------------------------------------------------------
# Run database migrations (idempotent)
# ---------------------------------------------------------------------------
if ! $SKIP_MIGRATIONS; then
    echo ""
    echo "=== Running database migrations ==="
    if [ -f "$PROJECT_DIR/scripts/run_migrations.sh" ] && [ -n "${SUPABASE_DB_URL:-}" ]; then
        bash "$PROJECT_DIR/scripts/run_migrations.sh" || {
            echo "WARNING: Migration failed. Continuing deploy — review manually."
        }
    else
        echo "Skipping migrations (no SUPABASE_DB_URL or script not found)"
    fi
fi

# ---------------------------------------------------------------------------
# Build / restart services
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Health check loop
# ---------------------------------------------------------------------------
echo ""
echo "=== Verifying deployment health ==="
HEALTHY=false
for i in $(seq 1 $HEALTH_MAX_ATTEMPTS); do
    if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Health check passed (attempt $i/$HEALTH_MAX_ATTEMPTS)"
        HEALTHY=true
        break
    fi
    echo "  Waiting for API... (attempt $i/$HEALTH_MAX_ATTEMPTS)"
    sleep $HEALTH_INTERVAL
done

if ! $HEALTHY; then
    echo ""
    echo "=== HEALTH CHECK FAILED — ROLLING BACK ==="
    echo "Reverting to $PREV_SHA"
    git checkout "$PREV_SHA"

    if $BUILD; then
        $COMPOSE_CMD build api frontend worker_light worker_medium scheduler
        $COMPOSE_CMD up -d --no-deps api frontend worker_light worker_medium scheduler
    else
        $COMPOSE_CMD restart worker_light worker_medium scheduler
    fi

    echo "Rolled back to $(git rev-parse --short HEAD)"
    echo "Deploy FAILED. Investigate the issue and re-deploy."
    exit 1
fi

# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------
echo ""
echo "=== Deployed $(git rev-parse --short HEAD) ==="
echo "API: auto-reloads via uvicorn --reload"
echo "Frontend: auto-reloads via Vite HMR"
echo "Workers: restarted"
echo "Health: verified"
