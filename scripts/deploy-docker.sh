#!/usr/bin/env bash
# ==============================================================================
# E2I Causal Analytics - Docker Deployment Script
# ==============================================================================
# Usage: ./scripts/deploy-docker.sh [image_tag]
#
# This script is called by the CD pipeline (deploy.yml) or can be run manually.
# It pulls the latest image from GHCR, swaps containers, and verifies health.
# ==============================================================================

set -euo pipefail

REGISTRY="ghcr.io"
IMAGE_NAME="${GHCR_IMAGE:-ghcr.io/enunezvn/e2i-api}"
TAG="${1:-latest}"
HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=60
CHECK_INTERVAL=2

log() { echo "[deploy] $(date '+%H:%M:%S') $*"; }

# --- Pull new image ---
log "Pulling ${IMAGE_NAME}:${TAG}"
docker pull "${IMAGE_NAME}:${TAG}"

# --- Tag for rollback ---
if docker image inspect e2i-api:current >/dev/null 2>&1; then
    log "Tagging current image as 'previous' for rollback"
    docker tag e2i-api:current e2i-api:previous
fi
docker tag "${IMAGE_NAME}:${TAG}" e2i-api:current

# --- Restart service ---
log "Restarting API service"
cd /opt/e2i_causal_analytics
if docker compose ps --services 2>/dev/null | grep -q api; then
    docker compose up -d --no-deps api
else
    sudo systemctl restart e2i-api
fi

# --- Health check ---
log "Waiting for health check (max ${MAX_WAIT}s)..."
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
        log "Health check PASSED (${elapsed}s)"
        log "Deployment successful: ${IMAGE_NAME}:${TAG}"
        exit 0
    fi
    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

# --- Rollback on failure ---
log "Health check FAILED after ${MAX_WAIT}s"
if docker image inspect e2i-api:previous >/dev/null 2>&1; then
    log "Rolling back to previous image"
    docker tag e2i-api:previous e2i-api:current
    sudo systemctl restart e2i-api
    log "Rollback complete — previous image restored"
fi
exit 1
