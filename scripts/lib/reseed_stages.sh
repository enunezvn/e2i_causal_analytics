# shellcheck shell=bash
# =============================================================================
# Stage runner for scripts/reseed_synthetic.sh (#1577) — sourceable, testable.
# =============================================================================
# WHY: the reseed wrapper ran its five stages under bare `set -euo pipefail`,
# so stage 1 (the loader) exiting 1 on partial failure killed the wrapper
# before ANY later stage — the kpi_history backfill, weekly capture, goldstd
# retrain, and A/B refresh stages never executed via cron between 2026-07-06
# and 2026-08-10 (#1577: zero stage markers in ~/logs/e2i-reseed.log, no run
# ever printed the done line). This runner captures each stage's failure,
# still runs the remaining stages, always reaches the final done line, and
# exits nonzero iff any stage failed — an honest aggregate with no silent
# truncation.
#
# Contract (pinned by tests/unit/test_scripts/test_reseed_stages.py):
#
#   reseed_run_stage NAME CMD [ARGS...]
#     Prints "=== NAME start <ts> ===", runs CMD, then prints
#     "=== NAME done <ts> ===" on success or
#     "=== NAME FAILED (exit N) <ts> ===" on failure, recording NAME in
#     RESEED_STAGE_FAILURES. ALWAYS returns 0, so `set -e` in the caller
#     cannot abort the remaining stages.
#
#     bash semantics note: CMD runs inside an `if` condition, where `set -e`
#     is suspended. A multi-command stage FUNCTION must chain its commands
#     with `&&` so the function's exit status reflects the first failure
#     rather than the last command's status.
#
#   reseed_finish LABEL
#     Always prints the "=== LABEL done <ts> ... ===" line with a status
#     summary; returns 0 iff no stage failed, 1 otherwise. Call it as the
#     script's last command so the script's exit code IS the aggregate.

RESEED_STAGE_FAILURES=()

reseed_run_stage() {
    local stage_name="$1"
    shift
    echo "=== ${stage_name} start $(date -Is) ==="
    if "$@"; then
        echo "=== ${stage_name} done $(date -Is) ==="
    else
        local rc=$?
        echo "=== ${stage_name} FAILED (exit ${rc}) $(date -Is) ==="
        RESEED_STAGE_FAILURES+=("${stage_name}")
    fi
    return 0
}

reseed_finish() {
    local label="${1:-reseed_synthetic}"
    if [[ ${#RESEED_STAGE_FAILURES[@]} -gt 0 ]]; then
        echo "=== ${label} done $(date -Is) (FAILED stages: ${RESEED_STAGE_FAILURES[*]}) ==="
        return 1
    fi
    echo "=== ${label} done $(date -Is) (all stages OK) ==="
    return 0
}
