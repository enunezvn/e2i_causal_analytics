#!/bin/bash
# memory_monitor.sh - Monitor memory usage and alert on high consumption
#
# Features:
# - Alerts when memory exceeds threshold
# - Identifies top memory consumers
# - Auto-cleanup of known memory hogs (esbuild, node orphans)
# - Logging with rotation support
#
# Usage: ./memory_monitor.sh [--threshold 80] [--auto-cleanup] [--webhook URL]

set -euo pipefail

# Configuration defaults
MEMORY_THRESHOLD=80          # Alert when memory usage exceeds this percentage
SWAP_THRESHOLD=50            # Alert when swap usage exceeds this percentage
AUTO_CLEANUP=false           # Automatically clean up orphan processes
LOG_FILE="/var/log/e2i/memory_monitor.log"
ALERT_COOLDOWN_FILE="/tmp/e2i_memory_alert_cooldown"
ALERT_COOLDOWN_SECONDS=300   # 5 minutes between alerts
WEBHOOK_URL=""               # Optional: Slack/Discord webhook for alerts

# Known memory-hungry processes to watch
MEMORY_HOGS=("esbuild" "node" "python" "vite")
MEMORY_HOG_THRESHOLD_MB=1500  # Alert if individual process exceeds this

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold|-t)
            MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --swap-threshold)
            SWAP_THRESHOLD="$2"
            shift 2
            ;;
        --auto-cleanup|-c)
            AUTO_CLEANUP=true
            shift
            ;;
        --webhook|-w)
            WEBHOOK_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --threshold N      Memory usage alert threshold (default: 80%)"
            echo "  --swap-threshold N     Swap usage alert threshold (default: 50%)"
            echo "  -c, --auto-cleanup     Auto-cleanup orphan processes on high memory"
            echo "  -w, --webhook URL      Webhook URL for alerts (Slack/Discord)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

log() {
    local level="$1"
    local msg="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $msg" | tee -a "$LOG_FILE" 2>/dev/null || echo "[$timestamp] [$level] $msg"
}

# Get memory stats
get_memory_stats() {
    local mem_info
    mem_info=$(free -m | grep Mem)

    TOTAL_MEM=$(echo "$mem_info" | awk '{print $2}')
    USED_MEM=$(echo "$mem_info" | awk '{print $3}')
    FREE_MEM=$(echo "$mem_info" | awk '{print $4}')
    AVAILABLE_MEM=$(echo "$mem_info" | awk '{print $7}')
    CACHED_MEM=$(echo "$mem_info" | awk '{print $6}')

    # Calculate usage percentage (using available memory for accurate reading)
    MEM_USED_PCT=$((100 - (AVAILABLE_MEM * 100 / TOTAL_MEM)))
}

# Get swap stats
get_swap_stats() {
    local swap_info
    swap_info=$(free -m | grep Swap)

    TOTAL_SWAP=$(echo "$swap_info" | awk '{print $2}')
    USED_SWAP=$(echo "$swap_info" | awk '{print $3}')

    if [[ $TOTAL_SWAP -gt 0 ]]; then
        SWAP_USED_PCT=$((USED_SWAP * 100 / TOTAL_SWAP))
    else
        SWAP_USED_PCT=0
    fi
}

# Get top memory-consuming processes
get_top_processes() {
    echo "Top 10 memory consumers:"
    ps aux --sort=-%mem | head -11 | tail -10 | awk '{printf "  PID: %-8s MEM: %-6s CMD: %s\n", $2, $4"%", $11}'
}

# Check for memory-hogging processes
check_memory_hogs() {
    local alerts=""

    for proc in "${MEMORY_HOGS[@]}"; do
        # Get all matching processes with their memory usage
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                local pid mem_mb cmd
                pid=$(echo "$line" | awk '{print $1}')
                mem_mb=$(echo "$line" | awk '{print $2}')
                cmd=$(echo "$line" | awk '{print $3}')

                if [[ $mem_mb -gt $MEMORY_HOG_THRESHOLD_MB ]]; then
                    alerts="$alerts\n  - $cmd (PID: $pid) using ${mem_mb}MB"
                fi
            fi
        done < <(ps aux | grep -i "$proc" | grep -v grep | awk '{printf "%s %d %s\n", $2, $6/1024, $11}')
    done

    if [[ -n "$alerts" ]]; then
        echo -e "Memory hogs detected (>${MEMORY_HOG_THRESHOLD_MB}MB):$alerts"
        return 1
    fi
    return 0
}

# Send alert (log + optional webhook)
send_alert() {
    local level="$1"
    local title="$2"
    local message="$3"

    # Check cooldown
    if [[ -f "$ALERT_COOLDOWN_FILE" ]]; then
        local last_alert
        last_alert=$(cat "$ALERT_COOLDOWN_FILE")
        local now
        now=$(date +%s)
        if [[ $((now - last_alert)) -lt $ALERT_COOLDOWN_SECONDS ]]; then
            log "DEBUG" "Alert cooldown active, skipping notification"
            return
        fi
    fi

    # Update cooldown
    date +%s > "$ALERT_COOLDOWN_FILE"

    log "$level" "$title: $message"

    # Send to webhook if configured
    if [[ -n "$WEBHOOK_URL" ]]; then
        local color="warning"
        [[ "$level" == "CRITICAL" ]] && color="danger"

        local payload
        payload=$(cat <<EOF
{
    "text": ":warning: *$title*",
    "attachments": [{
        "color": "$color",
        "text": "$message",
        "footer": "E2I Memory Monitor | $(hostname)",
        "ts": $(date +%s)
    }]
}
EOF
)
        curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Run cleanup script if available
run_cleanup() {
    local cleanup_script
    cleanup_script="$(dirname "$0")/cleanup_orphans.sh"

    if [[ -x "$cleanup_script" ]]; then
        log "INFO" "Running orphan cleanup..."
        "$cleanup_script" --verbose 2>&1 | while IFS= read -r line; do
            log "CLEANUP" "$line"
        done
        return 0
    else
        log "WARN" "Cleanup script not found or not executable: $cleanup_script"
        return 1
    fi
}

# Main monitoring logic
main() {
    log "INFO" "=== Memory Monitor Check ==="

    get_memory_stats
    get_swap_stats

    log "INFO" "Memory: ${USED_MEM}MB / ${TOTAL_MEM}MB (${MEM_USED_PCT}% used, ${AVAILABLE_MEM}MB available)"
    log "INFO" "Swap: ${USED_SWAP}MB / ${TOTAL_SWAP}MB (${SWAP_USED_PCT}% used)"

    local alert_triggered=false
    local alert_messages=""

    # Check memory threshold
    if [[ $MEM_USED_PCT -ge $MEMORY_THRESHOLD ]]; then
        alert_triggered=true
        alert_messages="Memory usage at ${MEM_USED_PCT}% (threshold: ${MEMORY_THRESHOLD}%)"
        log "WARN" "$alert_messages"
    fi

    # Check swap threshold
    if [[ $SWAP_USED_PCT -ge $SWAP_THRESHOLD ]]; then
        alert_triggered=true
        local swap_msg="Swap usage at ${SWAP_USED_PCT}% (threshold: ${SWAP_THRESHOLD}%)"
        alert_messages="$alert_messages\n$swap_msg"
        log "WARN" "$swap_msg"
    fi

    # Check for individual memory hogs
    if ! hog_output=$(check_memory_hogs); then
        alert_triggered=true
        alert_messages="$alert_messages\n$hog_output"
        log "WARN" "$hog_output"
    fi

    # Take action if alert triggered
    if $alert_triggered; then
        # Log top processes
        log "INFO" "$(get_top_processes)"

        # Determine alert level
        local level="WARNING"
        if [[ $MEM_USED_PCT -ge 95 ]]; then
            level="CRITICAL"
        fi

        # Send alert
        send_alert "$level" "E2I Memory Alert" "$alert_messages"

        # Auto-cleanup if enabled
        if $AUTO_CLEANUP; then
            log "INFO" "Auto-cleanup enabled, attempting to free memory..."
            if run_cleanup; then
                # Re-check memory after cleanup
                sleep 2
                get_memory_stats
                log "INFO" "Post-cleanup memory: ${MEM_USED_PCT}% used (${AVAILABLE_MEM}MB available)"
            fi
        fi
    else
        log "INFO" "Memory usage within normal limits"
    fi

    log "INFO" "=== Check Complete ==="
}

# Run main
main
