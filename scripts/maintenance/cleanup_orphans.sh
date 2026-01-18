#!/bin/bash
# cleanup_orphans.sh - Clean up orphan/zombie processes that cause terminal freezes
#
# Common culprits:
# - exec(eval...) - Node.js/esbuild orphan processes from Vite HMR
# - Defunct/zombie Python processes
# - Orphaned npm/node processes
#
# Usage: ./cleanup_orphans.sh [--dry-run] [--verbose]

set -euo pipefail

DRY_RUN=false
VERBOSE=false
LOG_FILE="/var/log/e2i/orphan_cleanup.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--verbose]"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE" 2>/dev/null || echo "$msg"
}

log_verbose() {
    if $VERBOSE; then
        log "$1"
    fi
}

# Track statistics
KILLED_COUNT=0
TOTAL_FOUND=0

kill_orphans() {
    local pattern="$1"
    local description="$2"

    log_verbose "Checking for: $description"

    # Find matching processes (exclude grep itself and this script)
    local pids
    pids=$(ps aux | grep -E "$pattern" | grep -v "grep" | grep -v "cleanup_orphans" | awk '{print $2}' || true)

    if [[ -z "$pids" ]]; then
        log_verbose "  No $description found"
        return
    fi

    local count
    count=$(echo "$pids" | wc -l)
    TOTAL_FOUND=$((TOTAL_FOUND + count))

    log "Found $count $description process(es)"

    for pid in $pids; do
        # Verify process still exists
        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        # Get process details for logging
        local proc_info
        proc_info=$(ps -p "$pid" -o pid,ppid,etime,rss,comm --no-headers 2>/dev/null || echo "$pid unknown")

        if $DRY_RUN; then
            log "  [DRY RUN] Would kill PID $pid: $proc_info"
        else
            log "  Killing PID $pid: $proc_info"

            # Try graceful SIGTERM first
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log "  Force killing PID $pid (SIGKILL)"
                kill -9 "$pid" 2>/dev/null || true
            fi

            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    done
}

log "=== Orphan Process Cleanup Started ==="
if $DRY_RUN; then
    log "Running in DRY RUN mode - no processes will be killed"
fi

# 1. exec(eval...) - Node.js/esbuild orphans from Vite HMR
kill_orphans 'exec\(eval' "exec(eval) Node.js/esbuild orphan"

# 2. Orphaned esbuild processes consuming high memory
kill_orphans 'esbuild.*service' "orphaned esbuild service"

# 3. Defunct/zombie processes (state Z)
zombie_pids=$(ps aux | awk '$8 ~ /Z/ {print $2}' || true)
if [[ -n "$zombie_pids" ]]; then
    count=$(echo "$zombie_pids" | wc -l)
    log "Found $count zombie process(es)"
    TOTAL_FOUND=$((TOTAL_FOUND + count))

    for pid in $zombie_pids; do
        # Zombies can't be killed directly - kill parent instead
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "")
        if [[ -n "$ppid" && "$ppid" != "1" ]]; then
            if $DRY_RUN; then
                log "  [DRY RUN] Would signal parent $ppid to reap zombie $pid"
            else
                log "  Signaling parent $ppid to reap zombie $pid"
                kill -SIGCHLD "$ppid" 2>/dev/null || true
                KILLED_COUNT=$((KILLED_COUNT + 1))
            fi
        fi
    done
fi

# 4. Orphaned vite dev server processes (not attached to terminal)
kill_orphans 'node.*vite' "orphaned vite dev server"

# 5. Old npm processes running for more than 1 hour
# This catches stuck npm install/run processes
old_npm=$(ps -eo pid,etime,comm | grep npm | awk '$2 ~ /^[0-9]+-/ || $2 ~ /^[0-9]+:[0-9]+:[0-9]+/ && substr($2,1,2) > 1 {print $1}' || true)
if [[ -n "$old_npm" ]]; then
    count=$(echo "$old_npm" | wc -l)
    log "Found $count long-running npm process(es)"
    TOTAL_FOUND=$((TOTAL_FOUND + count))

    for pid in $old_npm; do
        if $DRY_RUN; then
            log "  [DRY RUN] Would kill long-running npm PID $pid"
        else
            log "  Killing long-running npm PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    done
fi

# Summary
log "=== Cleanup Complete ==="
log "Total orphan processes found: $TOTAL_FOUND"
if $DRY_RUN; then
    log "Dry run - no processes were killed"
else
    log "Processes killed: $KILLED_COUNT"
fi

# Exit with code indicating if orphans were found (useful for monitoring)
if [[ $TOTAL_FOUND -gt 0 ]]; then
    exit 0  # Success, but orphans were found and handled
else
    exit 0  # Success, no orphans found
fi
