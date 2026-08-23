#!/bin/bash
# check_maintenance_freshness.sh - Is the maintenance cron actually running? (#1798)
#
# On 2026-06-30 `/etc/cron.d/e2i-maintenance` silently stopped executing: root's
# password entered a forced-change state, so `pam_unix` rejected every job and
# cron logged "Authentication token is no longer valid" on each tick without
# running the command. Nothing noticed for EIGHT WEEKS -- including
# `memory_monitor.sh --auto-cleanup`, the memory relief valve on a box that runs
# production and development on the same host.
#
# This script answers "did those jobs run when they were supposed to?" by
# comparing each job's log mtime against the interval DERIVED FROM THE CRONTAB
# ITSELF. It deliberately does not keep its own table of intervals: a second
# copy would drift from the real schedule, which is how a hand-maintained table
# has bitten this repo before (#1791).
#
# IMPORTANT -- do NOT install this as a cron job. A staleness check that runs
# from the crontab it is checking is dead exactly when the thing it checks is
# dead. Call it from somewhere independent of cron: a deploy step, a health
# check, or by hand.
#
# Usage:
#   ./check_maintenance_freshness.sh                       # default crontab
#   ./check_maintenance_freshness.sh --crontab PATH        # explicit
#   ./check_maintenance_freshness.sh --tolerance 3         # allow 3x interval
#
# Exit codes:
#   0  every job with a log is fresh
#   1  at least one job's log is stale or missing
#   2  the crontab could not be read

set -uo pipefail

CRONTAB="/etc/cron.d/e2i-maintenance"
TOLERANCE=2          # a job may be this many intervals late before it is stale
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --crontab) CRONTAB="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -r "$CRONTAB" ]]; then
    echo "ERROR: cannot read crontab: $CRONTAB" >&2
    exit 2
fi

# Derive an interval, in seconds, from the five cron schedule fields.
# Prints the interval, or "UNKNOWN" for any shape we do not positively
# recognise. Guessing an interval for an unrecognised schedule would produce a
# confident wrong verdict, so we report rather than fabricate.
schedule_interval_seconds() {
    local min="$1" hour="$2" dom="$3" mon="$4" dow="$5"

    # Only the shapes actually used by this crontab are recognised.
    if [[ "$min" =~ ^\*/([0-9]+)$ && "$hour" == "*" && "$dom" == "*" && "$mon" == "*" && "$dow" == "*" ]]; then
        echo $(( ${BASH_REMATCH[1]} * 60 )); return
    fi
    if [[ "$min" =~ ^[0-9]+$ && "$hour" =~ ^\*/([0-9]+)$ && "$dom" == "*" && "$mon" == "*" && "$dow" == "*" ]]; then
        echo $(( ${BASH_REMATCH[1]} * 3600 )); return
    fi
    if [[ "$min" =~ ^[0-9]+$ && "$hour" =~ ^[0-9]+$ && "$dom" == "*" && "$mon" == "*" && "$dow" == "*" ]]; then
        echo 86400; return   # daily
    fi
    if [[ "$min" =~ ^[0-9]+$ && "$hour" =~ ^[0-9]+$ && "$dom" == "*" && "$mon" == "*" && "$dow" =~ ^[0-9]+$ ]]; then
        echo 604800; return  # weekly
    fi
    echo "UNKNOWN"
}

now=$(date +%s)
stale=0
checked=0

echo "Maintenance freshness check (crontab: $CRONTAB, tolerance: ${TOLERANCE}x)"

while IFS= read -r line; do
    # Skip comments, blanks and environment assignments (SHELL=, PATH=).
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[A-Z_]+= ]] && continue

    read -r f1 f2 f3 f4 f5 _rest <<< "$line"
    [[ -z "${_rest:-}" ]] && continue

    # The log is whatever the job appends to. A job with no `>>` redirect (the
    # log-rotation entry) writes nothing, so there is nothing to check -- that
    # is not a failure, and it must not be reported as one.
    if [[ ! "$line" =~ \>\>[[:space:]]*([^[:space:]]+) ]]; then
        continue
    fi
    logpath="${BASH_REMATCH[1]}"
    logname=$(basename "$logpath")

    interval=$(schedule_interval_seconds "$f1" "$f2" "$f3" "$f4" "$f5")
    if [[ "$interval" == "UNKNOWN" ]]; then
        # Fail OPEN on "could not ask": report it, do not invent a limit and do
        # not fail the check on a schedule we simply do not parse.
        echo "  $logname: UNKNOWN SCHEDULE ($f1 $f2 $f3 $f4 $f5) - not checked"
        continue
    fi

    checked=$((checked + 1))
    limit=$(( interval * TOLERANCE ))

    if [[ ! -e "$logpath" ]]; then
        echo "  $logname: MISSING - no log at $logpath (expected every ${interval}s)"
        stale=$((stale + 1))
        continue
    fi

    mtime=$(stat -c %Y "$logpath" 2>/dev/null || echo 0)
    age=$(( now - mtime ))

    if (( age > limit )); then
        echo "  $logname: STALE - last write ${age}s ago, limit ${limit}s (every ${interval}s)"
        stale=$((stale + 1))
    else
        echo "  $logname: OK (age ${age}s, limit ${limit}s)"
    fi
done < "$CRONTAB"

if (( stale > 0 )); then
    echo "FAIL: $stale of $checked maintenance job(s) have not run on schedule."
    echo "      Check that cron can execute them: a PAM account failure (e.g. an"
    echo "      expired password on the job's user) makes cron fire every tick and"
    echo "      run nothing. See #1798."
    exit 1
fi

echo "OK: $checked maintenance job(s) fresh."
exit 0
