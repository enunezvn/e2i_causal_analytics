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
# Extra args are forwarded to load_synthetic_data.py.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/dotenv || ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/dotenv or .venv/bin/python missing — run from a checkout with the project venv installed" >&2
    exit 1
fi

MODE="--append-frontier"
if [[ "${1:-}" == "--full" ]]; then
    shift
    MODE="--anchor-to-now"
    echo "=== reseed_synthetic start $(date -Is) (RECOVERY: full-size, --anchor-to-now) ==="
else
    echo "=== reseed_synthetic start $(date -Is) (frontier append) ==="
fi

PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 \
    .venv/bin/dotenv -f .env run -- \
    .venv/bin/python scripts/load_synthetic_data.py "$MODE" "$@"

# Rebuild kpi_history from the substrate. Replace semantics (delete per
# (kpi_id, source), then upsert) stay correct in BOTH modes: after an append,
# history months recompute to the same values (source rows are frozen) and the
# current month picks up the new frontier rows; after a --full reseed the
# shifted timeline is rewritten wholesale, which is exactly what replace
# semantics exist for.
echo "=== kpi_history backfill start $(date -Is) ==="

PYTHONPATH="$PROJECT_ROOT" \
    .venv/bin/dotenv -f .env run -- \
    .venv/bin/python -m src.kpi.history_backfill

echo "=== kpi_history backfill done $(date -Is) ==="

echo "=== reseed_synthetic done $(date -Is) ==="
