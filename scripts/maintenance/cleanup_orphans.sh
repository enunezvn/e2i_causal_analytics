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
LOG_FILE="${LOG_FILE:-/var/log/e2i/orphan_cleanup.log}"

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
# #1801: a host-side `kill -SIGCHLD` cannot reap a zombie whose parent is PID 1
# inside a container. PID 1 in a namespace ignores signals it has no handler for,
# and the parent here is third-party code (supabase-meta's node server, running
# without an init shim). All 40 zombies on the prod box are in exactly that
# shape, so every 15 minutes this script was attempting an impossible reap and --
# since #1799 made the counter honest -- logging "REAPED 0 of 40" forever. Honest
# but useless noise, and noise is what trains people to stop reading the log that
# hid #1798 for eight weeks.
#
# PROC_ROOT is overridable so this is testable without a real container.
# Unreadable cgroup => NOT containerized: fail toward attempting the reap, since
# a failed signal is harmless and skipping a reap we could have done is not.
parent_is_containerized() {
    local ppid="$1"
    local cg="${PROC_ROOT:-/proc}/${ppid}/cgroup"
    [[ -r "$cg" ]] || return 1
    grep -qE 'docker-|/docker/|containerd|kubepods' "$cg" 2>/dev/null
}

count_zombies() { ps aux | awk '$8 ~ /Z/ {print $2}'; }
zombie_pids=$(count_zombies || true)
if [[ -n "$zombie_pids" ]]; then
    count=$(echo "$zombie_pids" | wc -l)
    log "Found $count zombie process(es)"
    TOTAL_FOUND=$((TOTAL_FOUND + count))

    signalled=0
    unreapable=0
    _reported_ppids=""
    for pid in $zombie_pids; do
        # Zombies can't be killed directly - signal the parent to reap instead.
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "")
        if [[ -n "$ppid" && "$ppid" != "1" ]]; then
            if parent_is_containerized "$ppid"; then
                # Stand down and say why, rather than signalling into a void and
                # reporting the failure again on the next tick (#1801).
                unreapable=$((unreapable + 1))
                if [[ -z "${_reported_ppids// }" || "$_reported_ppids" != *"|$ppid|"* ]]; then
                    log "  SKIP: parent $ppid is PID 1 in a container - not host-reapable (#1801)"
                    _reported_ppids="${_reported_ppids}|$ppid|"
                fi
            elif $DRY_RUN; then
                log "  [DRY RUN] Would signal parent $ppid to reap zombie $pid"
            else
                log "  Signaling parent $ppid to reap zombie $pid"
                kill -SIGCHLD "$ppid" 2>/dev/null || true
                signalled=$((signalled + 1))
            fi
        fi
    done

    # #1798: signalling a parent is NOT reaping a zombie. This used to do
    # KILLED_COUNT=$((KILLED_COUNT + 1)) per SIGCHLD sent, so the 2026-08-23 run
    # logged "Processes killed: 40" while all 40 zombies were still there. A
    # parent that ignores SIGCHLD (or is itself stuck) reaps nothing. Re-observe
    # the zombie set and report the difference, so the number means something.
    if ! $DRY_RUN; then
        sleep 1
        remaining=$(count_zombies | wc -l)
        REAPED=$((count - remaining))
        (( REAPED < 0 )) && REAPED=0
        log "  Signalled $signalled parent(s); REAPED $REAPED of $count zombie(s); $remaining remain"
        if [[ ${unreapable:-0} -gt 0 ]]; then
            log "  ($unreapable of those are not host-reapable: containerized PID 1 - see #1801)"
        fi
        KILLED_COUNT=$((KILLED_COUNT + REAPED))
    fi
fi

# 4. Orphaned vite dev server processes (not attached to terminal)
kill_orphans 'node.*vite' "orphaned vite dev server"

# 5. Old npm processes running for more than 1 hour
# This catches stuck npm install/run processes
# #1798: the age threshold is a NAMED knob, and the comparison is NUMERIC.
# It used to be an inline `substr($2,1,2) > 1`. substr() returns a STRING, so
# "02" > 1 is a LEXICAL compare and is FALSE ('0' < '1') -- the rule advertised
# "more than 1 hour" and actually fired at 10 hours, exempting everything from
# 1h to 9h59m for the script's entire life.
#
# The default stays at 10h ON PURPOSE. 10h is the behaviour that has actually
# been running in production; npm is now also used for long-lived MCP servers
# (e.g. `npm exec chrome-devtools-mcp`), which a 1h threshold would kill hourly.
# Lowering it is a deliberate decision, not a bug fix -- change it here.
NPM_MAX_AGE_HOURS="${NPM_MAX_AGE_HOURS:-10}"
old_npm=$(ps -eo pid,etime,comm | grep npm | awk -v maxh="$NPM_MAX_AGE_HOURS" '$2 ~ /^[0-9]+-/ || ($2 ~ /^[0-9]+:[0-9]+:[0-9]+/ && substr($2,1,2)+0 >= maxh) {print $1}' || true)
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
if ! $DRY_RUN; then write_success_stamp; fi
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
