#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Synthetic Substrate Maintenance (Cron-Ready Wrapper)
# =============================================================================
# DEFAULT (no args): FRONTIER APPEND — grow the frozen substrate with the
# trailing deterministic weekly cohorts (src/ml/synthetic/frontier_append.py).
# Appends new-patient cohorts + downstream events at the calendar frontier and
# refreshes user_sessions (MAU/WAU, #1127) WITHOUT rewriting history, so the
# weekly run no longer re-fires drift alerts on the gold-standard models and
# the dataset actually grows week over week.
#
# RECOVERY ONLY: `reseed_synthetic.sh --full` runs the legacy full-size
# --anchor-to-now destructive reseed (rewrites every synthetic timestamp AND —
# because the anchored/calendar paths consume different RNG draw counts —
# every attribute value under the same PKs). This invalidates the frozen
# substrate the gold-standard models were trained on and re-fires an honest
# drift storm. Use only when the substrate is corrupt beyond appending.
#
# STAGE RESILIENCE (#1577): every stage runs through the runner in
# scripts/lib/reseed_stages.sh — a failed stage prints a FAILED marker and the
# REMAINING STAGES STILL RUN; the final done line is always reached with a
# status summary, and the exit code is nonzero iff any stage failed. Before
# this, the loader's exit-1-on-partial-failure killed the wrapper under
# `set -e` and the kpi backfill / capture / retrain / A/B stages never ran via
# cron (every weekly run 2026-07-06..2026-08-10). The loader's exit semantics
# are deliberately unchanged — partial failure still fails its stage loudly.
#
# Environment gotchas this wrapper encodes (do not "simplify" away):
#   - `dotenv` is NOT on the bare shell PATH — only .venv/bin/dotenv works
#   - PYTHONPATH must be the repo root for src.* imports
#   - LOKY_MAX_CPU_COUNT=1 keeps joblib from over-forking on the droplet
#   - (--full) full size is REQUIRED for KPI targets: --small yields ~250
#     users and MAU/WAU (#1115, targets 2000/1200) can never reach GOOD
#
# Crontab entry (weekly, Monday 3 AM — UNCHANGED across the append cutover).
# Log under $HOME — /var/log is NOT writable by the cron user on this host (a
# root-owned dir kills the redirect with Permission denied BEFORE the script
# runs, silently no-oping the job):
#   0 3 * * 1 /home/enunez/Projects/e2i_causal_analytics/scripts/reseed_synthetic.sh >> /home/enunez/logs/e2i-reseed.log 2>&1
#
# Extra args are forwarded to load_synthetic_data.py, EXCEPT --skip-retrain,
# which this wrapper consumes to opt out of the gold-standard model retrain
# stage (scripts/retrain_goldstd.sh).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# shellcheck source=lib/reseed_stages.sh
source "$SCRIPT_DIR/lib/reseed_stages.sh"

# Environment preflight — not a stage: nothing can run without the venv, so
# failing loud and early here is correct.
if [[ ! -x .venv/bin/dotenv || ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/dotenv or .venv/bin/python missing — run from a checkout with the project venv installed" >&2
    exit 1
fi

# Consume --skip-retrain (anywhere in the args) before mode detection so it is
# never forwarded to load_synthetic_data.py.
RETRAIN=1
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-retrain" ]]; then
        RETRAIN=0
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

MODE="--append-frontier"
if [[ "${1:-}" == "--full" ]]; then
    shift
    MODE="--anchor-to-now"
    echo "=== reseed_synthetic start $(date -Is) (RECOVERY: full-size, --anchor-to-now) ==="
else
    echo "=== reseed_synthetic start $(date -Is) (frontier append) ==="
fi

# "$@" is function-local in bash — capture the forwarded args so the stage
# functions below can reach them.
FORWARD_ARGS=("${@+"$@"}")

stage_loader() {
    PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 \
        .venv/bin/dotenv -f .env run -- \
        .venv/bin/python scripts/load_synthetic_data.py "$MODE" \
        "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
}

# Rebuild kpi_history from the substrate. Replace semantics (delete per
# (kpi_id, source), then upsert) stay correct in BOTH modes: after an append,
# history months recompute to the same values (source rows are frozen) and the
# current month picks up the new frontier rows; after a --full reseed the
# shifted timeline is rewritten wholesale, which is exactly what replace
# semantics exist for.
stage_kpi_backfill() {
    PYTHONPATH="$PROJECT_ROOT" \
        .venv/bin/dotenv -f .env run -- \
        .venv/bin/python -m src.kpi.history_backfill
}

# Present-state KPIs (coverage/eligibility, non-recastable windows) can't be
# backfilled honestly — record this week's live reading instead (append-only;
# same calculator path as the API). After a --full reseed the old captures
# describe a substrate that no longer exists, so purge them first; the `&&`
# chain matters (see the lib's bash-semantics note): a failed purge must skip
# the capture — capturing on top of stale rows is the exact corruption the
# purge exists to prevent — and fail the stage.
stage_weekly_capture() {
    if [[ "$MODE" == "--anchor-to-now" ]]; then
        PYTHONPATH="$PROJECT_ROOT" \
            .venv/bin/dotenv -f .env run -- \
            .venv/bin/python -m src.kpi.history_capture --purge &&
            PYTHONPATH="$PROJECT_ROOT" \
                .venv/bin/dotenv -f .env run -- \
                .venv/bin/python -m src.kpi.history_capture
    else
        PYTHONPATH="$PROJECT_ROOT" \
            .venv/bin/dotenv -f .env run -- \
            .venv/bin/python -m src.kpi.history_capture
    fi
}

# Retrain the 12 gold-standard staging models on the substrate that just grew
# (or, in --full recovery mode, was rewritten — after which the old fits
# describe data that no longer exists, so retraining is not optional in
# spirit). Idempotent; see scripts/retrain_goldstd.sh. Opt out with
# --skip-retrain.
stage_goldstd_retrain() {
    "$SCRIPT_DIR/retrain_goldstd.sh"
}

# Refresh the Shard-09 A/B substrate (ml_experiments + assignments/enrollments/
# results) in place. The frontier append above does NOT touch these tables, so
# without this step the experiment_monitor staleness alerts alarm on a frozen
# substrate (found frozen at its last manual full load — 2026-07-11 /experiments
# review). Deterministic uuid5 ids -> pure in-place refresh. Skipped in --full
# mode because the full generate path already rebuilds the substrate.
# Still ordered last: nothing else in this wrapper reads the AB tables. (The
# pre-#1577 rationale — that ordering was the only thing protecting the other
# stages from an A/B failure under `set -e` — is obsolete: the stage runner
# now guarantees every stage runs regardless of earlier failures.) "$@" is
# forwarded so --dry-run stays write-free (the purge is gated on it);
# --refresh-ab itself ignores --small.
stage_ab_refresh() {
    PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 \
        .venv/bin/dotenv -f .env run -- \
        .venv/bin/python scripts/load_synthetic_data.py --refresh-ab \
        "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
}

reseed_run_stage "loader" stage_loader
reseed_run_stage "kpi_history backfill" stage_kpi_backfill
reseed_run_stage "kpi_history weekly capture" stage_weekly_capture

if [[ "$RETRAIN" == "1" ]]; then
    reseed_run_stage "goldstd retrain" stage_goldstd_retrain
else
    echo "=== goldstd retrain SKIPPED (--skip-retrain) $(date -Is) ==="
fi

if [[ "$MODE" == "--append-frontier" ]]; then
    reseed_run_stage "A/B substrate refresh" stage_ab_refresh
fi

# Aggregate verdict: always prints the done line; exits nonzero iff any stage
# failed. MUST stay the last command — its status is the script's exit code.
reseed_finish "reseed_synthetic"
