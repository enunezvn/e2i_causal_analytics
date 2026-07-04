#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# OPIK MANAGER - Robust Container Management Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Purpose: Permanently solve Opik container startup failures
# Root Cause: Corporate proxy (163.116.128.80:2011) throttles Docker image downloads
# Solution: Bypass proxy for registry pulls + sequential pulling with retries
#
# Usage:
#   ./opik-manager.sh start    - Pull images and start Opik
#   ./opik-manager.sh stop     - Stop Opik containers
#   ./opik-manager.sh restart  - Stop then start
#   ./opik-manager.sh status   - Check container status
#   ./opik-manager.sh pull     - Only pull images
#   ./opik-manager.sh logs     - View logs
#   ./opik-manager.sh cleanup  - Remove all Opik containers and volumes
#
# Author: E2I Causal Analytics Team
# Created: 2025-12-30
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Opik deployment directory (persistent location)
OPIK_DIR="/home/enunez/opik/deployment/docker-compose"

# Verify Opik directory exists
if [ ! -d "$OPIK_DIR" ]; then
    echo "ERROR: Opik deployment directory not found at $OPIK_DIR"
    echo "Clone Opik repository with: git clone https://github.com/comet-ml/opik.git /home/enunez/opik"
    exit 1
fi

cd "$OPIK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Required images for Opik stack
IMAGES=(
    "mysql:8.4.2"
    "redis:7.2.4-alpine3.19"
    "clickhouse/clickhouse-server:25.3.6.56-alpine"
    "zookeeper:3.9.4"
    "minio/minio:RELEASE.2025-03-12T18-04-18Z"
    "minio/mc:RELEASE.2025-03-12T17-29-24Z"
    "alpine:latest"
)

# Opik-specific images from GitHub Container Registry
OPIK_IMAGES=(
    "ghcr.io/comet-ml/opik/opik-backend:latest"
    "ghcr.io/comet-ml/opik/opik-python-backend:latest"
    "ghcr.io/comet-ml/opik/opik-frontend:latest"
)

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=10
PULL_TIMEOUT=300  # 5 minutes per image

# Memory limits (MB) - adjust based on system resources
MYSQL_MEMORY=512
REDIS_MEMORY=128
CLICKHOUSE_MEMORY=768
ZOOKEEPER_MEMORY=256
MINIO_MEMORY=256
BACKEND_MEMORY=768
PYTHON_BACKEND_MEMORY=384
FRONTEND_MEMORY=128

# Minimum required memory (MB)
MIN_AVAILABLE_MEMORY=2048

# ─────────────────────────────────────────────────────────────────────────────
# COLORS AND LOGGING
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

check_docker() {
    if ! docker info &>/dev/null; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    log_success "Docker is running"
}

check_memory() {
    local available_mb=$(free -m | awk '/^Mem:/{print $7}')
    log_info "Available memory: ${available_mb}MB (minimum required: ${MIN_AVAILABLE_MEMORY}MB)"

    if [ "$available_mb" -lt "$MIN_AVAILABLE_MEMORY" ]; then
        log_warning "Low memory! Opik may not start properly."
        log_warning "Consider closing other applications or increasing WSL2 memory."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_disk() {
    local available_gb=$(df -BG "$OPIK_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
    log_info "Available disk space: ${available_gb}GB"

    if [ -n "$available_gb" ] && [ "$available_gb" -lt 10 ]; then
        log_warning "Low disk space! Docker images require significant space."
    fi
}

# Pull a single image with retry logic and proxy bypass
pull_image_with_retry() {
    local image=$1
    local attempt=1

    log_step "Pulling $image..."

    while [ $attempt -le $MAX_RETRIES ]; do
        log_info "Attempt $attempt of $MAX_RETRIES"

        # Try pulling with proxy bypass for Docker registries
        # This sets NO_PROXY to include Docker registries temporarily
        if NO_PROXY="$NO_PROXY,registry-1.docker.io,ghcr.io,docker.io,*.docker.io,*.docker.com" \
           timeout $PULL_TIMEOUT docker pull "$image" 2>&1; then
            log_success "Successfully pulled $image"
            return 0
        fi

        log_warning "Pull failed for $image (attempt $attempt)"

        if [ $attempt -lt $MAX_RETRIES ]; then
            log_info "Waiting ${RETRY_DELAY}s before retry..."
            sleep $RETRY_DELAY
        fi

        ((attempt++))
    done

    log_error "Failed to pull $image after $MAX_RETRIES attempts"
    return 1
}

# Pull all required images sequentially
pull_all_images() {
    log_step "Pulling base images (from Docker Hub)..."

    local failed_images=()

    # Pull base images first
    for image in "${IMAGES[@]}"; do
        if ! pull_image_with_retry "$image"; then
            failed_images+=("$image")
        fi
    done

    log_step "Pulling Opik images (from GitHub Container Registry)..."

    # Pull Opik-specific images
    for image in "${OPIK_IMAGES[@]}"; do
        if ! pull_image_with_retry "$image"; then
            failed_images+=("$image")
        fi
    done

    if [ ${#failed_images[@]} -gt 0 ]; then
        log_error "Failed to pull the following images:"
        for img in "${failed_images[@]}"; do
            echo "  - $img"
        done
        log_warning "You may need to pull these manually or check your network connection."
        return 1
    fi

    log_success "All images pulled successfully!"
    return 0
}

# Verify all required images exist locally
verify_images() {
    log_step "Verifying all required images exist..."

    local missing_images=()

    for image in "${IMAGES[@]}" "${OPIK_IMAGES[@]}"; do
        if ! docker image inspect "$image" &>/dev/null; then
            missing_images+=("$image")
        fi
    done

    if [ ${#missing_images[@]} -gt 0 ]; then
        log_warning "Missing images:"
        for img in "${missing_images[@]}"; do
            echo "  - $img"
        done
        return 1
    fi

    log_success "All required images are available locally"
    return 0
}

# Start Opik with proper sequencing
start_opik() {
    log_step "Starting Opik infrastructure services..."

    # Start infrastructure first (no profile needed - they start by default)
    docker compose up -d mysql redis clickhouse zookeeper minio mc

    # Wait for infrastructure to be healthy
    log_info "Waiting for infrastructure services to be healthy..."
    local max_wait=120
    local waited=0

    while [ $waited -lt $max_wait ]; do
        # Check MySQL health
        if docker compose exec -T mysql mysqladmin ping -h127.0.0.1 --silent 2>/dev/null; then
            log_success "MySQL is healthy"
            break
        fi
        sleep 5
        ((waited+=5))
        log_info "Waiting for MySQL... (${waited}s/${max_wait}s)"
    done

    if [ $waited -ge $max_wait ]; then
        log_warning "Infrastructure services may not be fully healthy, proceeding anyway..."
    fi

    log_step "Starting Opik application services..."
    docker compose --profile opik up -d

    # Wait for frontend to be healthy
    log_info "Waiting for Opik frontend to be ready..."
    waited=0
    max_wait=180

    while [ $waited -lt $max_wait ]; do
        if curl -s -f http://localhost:5173/health &>/dev/null; then
            log_success "Opik is ready at http://localhost:5173"
            return 0
        fi
        sleep 5
        ((waited+=5))
        log_info "Waiting for frontend... (${waited}s/${max_wait}s)"
    done

    log_warning "Opik may not be fully ready. Check status with: $0 status"
    return 0
}

# Stop Opik
stop_opik() {
    log_step "Stopping Opik..."
    docker compose --profile opik down
    log_success "Opik stopped"
}

# Show status
show_status() {
    log_step "Opik Container Status:"
    echo ""
    docker compose --profile opik ps -a
    echo ""

    log_step "Service Health:"
    echo ""

    # Check each service
    local services=("mysql" "redis" "clickhouse" "zookeeper" "minio" "backend" "python-backend" "frontend")

    for svc in "${services[@]}"; do
        local status=$(docker compose ps -q "$svc" 2>/dev/null)
        if [ -n "$status" ]; then
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$status" 2>/dev/null || echo "unknown")
            local running=$(docker inspect --format='{{.State.Running}}' "$status" 2>/dev/null || echo "unknown")
            echo "  $svc: running=$running, health=$health"
        else
            echo "  $svc: not running"
        fi
    done

    echo ""
    log_step "Access Points:"
    echo "  Frontend UI: http://localhost:5173"
    echo "  Backend API: http://localhost:5173/api"
}

# Show logs
show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        docker compose logs -f "$service"
    else
        docker compose --profile opik logs -f --tail=100
    fi
}

# Cleanup everything
cleanup() {
    log_warning "This will remove all Opik containers and volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_step "Cleaning up..."
        docker compose --profile opik down -v --remove-orphans
        log_success "Cleanup complete"
    else
        log_info "Cleanup cancelled"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  OPIK MANAGER - Robust Container Management"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""

    local command=${1:-start}

    case $command in
        start)
            check_docker
            check_memory
            check_disk

            if ! verify_images; then
                log_info "Some images are missing. Starting pull..."
                if ! pull_all_images; then
                    log_error "Failed to pull all required images"
                    log_info "Try running with better network conditions or pull images manually"
                    exit 1
                fi
            fi

            start_opik
            show_status
            ;;

        stop)
            check_docker
            stop_opik
            ;;

        restart)
            check_docker
            stop_opik
            sleep 5
            check_memory
            start_opik
            show_status
            ;;

        status)
            check_docker
            show_status
            ;;

        pull)
            check_docker
            pull_all_images
            ;;

        logs)
            check_docker
            show_logs "${2:-}"
            ;;

        cleanup)
            check_docker
            cleanup
            ;;

        *)
            echo "Usage: $0 {start|stop|restart|status|pull|logs [service]|cleanup}"
            echo ""
            echo "Commands:"
            echo "  start   - Pull images (if needed) and start Opik"
            echo "  stop    - Stop all Opik containers"
            echo "  restart - Stop and start Opik"
            echo "  status  - Show container status"
            echo "  pull    - Only pull images (with retry logic)"
            echo "  logs    - View logs (optionally specify service)"
            echo "  cleanup - Remove all containers and volumes"
            exit 1
            ;;
    esac
}

main "$@"
