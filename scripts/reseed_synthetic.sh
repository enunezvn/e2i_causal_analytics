#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Synthetic Substrate Reseed (Cron-Ready Wrapper)
# =============================================================================
# Runs the full-size, frontier-anchored synthetic reseed with the exact
# environment scripts/load_synthetic_data.py needs. Encodes the working
# invocation so operators (and cron) never trip the known failure modes:
#
#   - `dotenv` is NOT on the bare shell PATH — only .venv/bin/dotenv works
#   - PYTHONPATH must be the repo root for src.* imports
#   - LOKY_MAX_CPU_COUNT=1 keeps joblib from over-forking on the droplet
#   - full size is REQUIRED for KPI targets: --small yields ~250 users and
#     MAU/WAU (#1115, targets 2000/1200) can never reach GOOD
#
# WHY a periodic reseed matters (#1127): MAU/WAU registry queries are
# deliberately NOW()-anchored while user_sessions is a fixed 90-day history
# ending at the last reseed — more than 7 days without a reseed and WAU
# decays to CRITICAL (30 days for MAU). The reseed is idempotent
# (deterministic PKs — PRs #1105/#1106/#1120) and takes ~3 min full-size.
#
# Crontab entry (weekly, Monday 3 AM). Log under $HOME — /var/log is NOT
# writable by the cron user on this host (a root-owned dir kills the redirect
# with Permission denied BEFORE the script runs, silently no-oping the job):
#   0 3 * * 1 /home/enunez/Projects/e2i_causal_analytics/scripts/reseed_synthetic.sh >> /home/enunez/logs/e2i-reseed.log 2>&1
#
# Extra args are forwarded to load_synthetic_data.py (after --anchor-to-now).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/dotenv || ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/dotenv or .venv/bin/python missing — run from a checkout with the project venv installed" >&2
    exit 1
fi

echo "=== reseed_synthetic start $(date -Is) (full-size, --anchor-to-now) ==="

PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 \
    .venv/bin/dotenv -f .env run -- \
    .venv/bin/python scripts/load_synthetic_data.py --anchor-to-now "$@"

echo "=== reseed_synthetic done $(date -Is) ==="
