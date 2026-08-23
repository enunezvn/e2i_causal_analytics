#!/bin/bash
# docker_cleanup.sh - Conservative Docker disk cleanup
#
# Targets only truly orphaned resources — never removes images that
# compose services depend on. Avoids `docker image prune -a` which
# forces full rebuilds.
#
# What it prunes (safe):
#   - Build cache (largest win)
#   - Dangling images (<none>:<none> layers from rebuilds)
#   - Exited containers older than 24h
#   - Dangling volumes (anonymous, unreferenced — NOT named volumes)
#   - Unused networks (no connected containers)
#
# What it SKIPS:
#   - `docker image prune -a` (removes images for stopped/defined services)
#   - Named volumes (always preserved)
#   - Any image referenced by a running or compose-defined container
#
# Usage:
#   ./docker_cleanup.sh              # Run cleanup
#   ./docker_cleanup.sh --dry-run    # Show what would be removed
#
# Cron: Sundays at 3am via setup_cron.sh

set -euo pipefail

LOG_FILE="/var/log/e2i/docker_cleanup.log"
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $1"; }

usage() {
    cat << 'USAGE'
Usage: docker_cleanup.sh [OPTIONS]

Conservative Docker disk cleanup — removes only orphaned resources.

Options:
  --dry-run    Show what would be removed without deleting anything
  --verbose    Print additional detail
  -h, --help   Show this help message
USAGE
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --verbose)  VERBOSE=true; shift ;;
        -h|--help)  usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Ensure docker is available
if ! command -v docker &>/dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Timestamp for log
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

{
    echo "========================================"
    echo "Docker cleanup started at $(timestamp)"
    echo "Dry run: $DRY_RUN"
    echo "========================================"

    # Show current disk usage
    log_step "Current Docker disk usage:"
    docker system df
    echo ""

    TOTAL_FREED=0

    # 1. Build cache (typically the biggest win)
    log_step "Pruning build cache..."
    if [[ "$DRY_RUN" == true ]]; then
        build_cache=$(docker system df --format '{{.Reclaimable}}' 2>/dev/null | head -1 || echo "unknown")
        log_info "[DRY RUN] Build cache reclaimable: $build_cache"
    else
        docker builder prune -f 2>&1 || log_warn "Build cache prune returned non-zero"
    fi
    echo ""

    # 2. Dangling images (<none>:<none>)
    log_step "Pruning dangling images..."
    dangling_count=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
    if [[ $dangling_count -gt 0 ]]; then
        log_info "Found $dangling_count dangling image(s)"
        if [[ "$DRY_RUN" == true ]]; then
            docker images -f "dangling=true" --format "  {{.Repository}}:{{.Tag}} ({{.Size}})"
        else
            docker image prune -f 2>&1
        fi
    else
        log_info "No dangling images found"
    fi
    echo ""

    # 3. Exited containers older than 24h
    log_step "Pruning exited containers (>24h old)..."
    if [[ "$DRY_RUN" == true ]]; then
        exited=$(docker ps -a --filter "status=exited" --filter "until=24h" --format "  {{.Names}} ({{.Status}})" 2>/dev/null)
        if [[ -n "$exited" ]]; then
            log_info "[DRY RUN] Would remove:"
            echo "$exited"
        else
            log_info "No exited containers older than 24h"
        fi
    else
        docker container prune -f --filter "until=24h" 2>&1
    fi
    echo ""

    # 4. Dangling volumes (anonymous, unreferenced)
    log_step "Pruning dangling volumes..."
    dangling_vols=$(docker volume ls -f "dangling=true" -q 2>/dev/null | wc -l)
    if [[ $dangling_vols -gt 0 ]]; then
        log_info "Found $dangling_vols dangling volume(s)"
        if [[ "$DRY_RUN" == true ]]; then
            docker volume ls -f "dangling=true" --format "  {{.Name}}"
        else
            docker volume prune -f 2>&1
        fi
    else
        log_info "No dangling volumes found"
    fi
    echo ""

    # 5. Unused networks
    log_step "Pruning unused networks..."
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would prune networks with no connected containers"
        docker network ls --filter "type=custom" --format "  {{.Name}}" 2>/dev/null || true
    else
        docker network prune -f 2>&1
    fi
    echo ""

    # Final disk usage
    log_step "Docker disk usage after cleanup:"
    docker system df
    echo ""

    # Disk usage summary
    disk_usage=$(df -h / | awk 'NR==2 {print $5}')
    log_info "Root filesystem usage: $disk_usage"

    echo "========================================"
    echo "Docker cleanup finished at $(timestamp)"
    echo "========================================"

} 2>&1 | if [[ -w "$(dirname "$LOG_FILE")" ]]; then
    tee -a "$LOG_FILE"
else
    cat
fi
