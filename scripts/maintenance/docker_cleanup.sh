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

LOG_FILE="${LOG_FILE:-/var/log/e2i/docker_cleanup.log}"

# #1798: a success stamp is the ONLY honest "this job completed" signal.
# The log is written by ANYTHING that invokes this script -- a --dry-run, a run
# that aborts halfway, a human debugging by hand -- so log mtime answers "was
# this file written", not "did this job complete". Keying the freshness check on
# log mtime produced a false OK for a job that had not run in 56 days.
# Only a real, completed run touches this.
write_success_stamp() {
    local _dir _name
    _dir=$(dirname "$LOG_FILE")
    _name=$(basename "$0" .sh)
    touch "${_dir}/.${_name}.success" 2>/dev/null || true
}

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
        # #1798: this was `--format '{{.Reclaimable}}' | head -1`, which takes the
        # IMAGES row -- docker system df prints one row per type. It reported
        # "Build cache reclaimable: 19.81GB" while build cache was 0B. Select the
        # row by its Type instead of by position.
        build_cache=$(docker system df --format '{{.Type}}\t{{.Reclaimable}}' 2>/dev/null \
            | awk -F'\t' '$1=="Build Cache"{print $2}' || true)
        build_cache=${build_cache:-unknown}
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
        # #1798: `until` is NOT a valid `docker ps` filter -- the daemon rejects it
        # ("invalid filter 'until'"). Under `set -euo pipefail` that non-zero exit
        # killed the whole run here, so --dry-run never reached the volume or
        # network steps. It IS valid for `docker container prune` (the real branch
        # below), which is why only the preview was broken. List exited containers
        # without the filter and say plainly that the prune applies the age cut.
        exited=$(docker ps -a --filter "status=exited" --format "  {{.Names}} ({{.Status}})" 2>/dev/null || true)
        if [[ -n "$exited" ]]; then
            log_info "[DRY RUN] Exited containers (the real prune removes those >24h):"
            echo "$exited"
        else
            log_info "No exited containers"
        fi
    else
        docker container prune -f --filter "until=24h" 2>&1
    fi
    echo ""

    # 4. Dangling volumes (anonymous, unreferenced)
    log_step "Pruning dangling volumes..."
    # #1798: the preview must match the action.
    #
    # `docker volume prune -f` (no --all) removes ONLY ANONYMOUS volumes. On
    # Docker 29.1.3 `-a` is documented as "Remove all unused volumes, not just
    # anonymous ones", and a real run left every named volume intact. But
    # `docker volume ls -f dangling=true` lists ALL unused volumes INCLUDING
    # named ones, so the preview used to name e2i_grafana_data, e2i_loki_data,
    # e2i_prometheus_data and e2i_promtail_positions as removal candidates that
    # the prune would never touch -- telling an operator their observability data
    # was about to be deleted when it was not.
    #
    # Anonymous volumes are named with a 64-char hex id; that is the set prune
    # actually targets, so that is the set we count and show.
    anon_vols=$(docker volume ls -f "dangling=true" -q 2>/dev/null \
        | grep -E '^[0-9a-f]{64}$' || true)
    dangling_vols=$(printf '%s' "$anon_vols" | grep -c . || true)
    if [[ ${dangling_vols:-0} -gt 0 ]]; then
        log_info "Found $dangling_vols anonymous volume(s) the prune would remove"
        if [[ "$DRY_RUN" == true ]]; then
            printf '  %s\n' $anon_vols
        else
            docker volume prune -f 2>&1
        fi
    else
        log_info "No anonymous volumes to prune (named volumes are never touched)"
    fi
    echo ""

    # 5. Unused networks
    log_step "Pruning unused networks..."
    if [[ "$DRY_RUN" == true ]]; then
        # #1798: this listed EVERY custom network, but `docker network prune`
        # removes only those with NO connected containers. On the prod box that
        # named e2i_network and supabase-network -- which carry the whole running
        # stack -- as if they were about to be deleted. Ask each network how many
        # containers it has, and show only the ones prune would actually take.
        log_info "[DRY RUN] Networks with no connected containers:"
        _idle_nets=""
        for _net in $(docker network ls --filter "type=custom" --format '{{.Name}}' 2>/dev/null || true); do
            _n=$(docker network inspect -f '{{len .Containers}}' "$_net" 2>/dev/null || echo 1)
            [[ "${_n:-1}" == "0" ]] && _idle_nets="${_idle_nets}${_net}"$'\n'
        done
        if [[ -n "${_idle_nets// }" ]]; then
            printf '  %s\n' $_idle_nets
        else
            log_info "  none"
        fi
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
    if [[ "$DRY_RUN" != true ]]; then
        write_success_stamp
    fi
    echo "Docker cleanup finished at $(timestamp)"
    echo "========================================"

} 2>&1 | if [[ -w "$(dirname "$LOG_FILE")" ]]; then
    tee -a "$LOG_FILE"
else
    cat
fi
