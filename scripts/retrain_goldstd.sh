#!/bin/bash
# =============================================================================
# E2I Causal Analytics - Gold-Standard Model Retrain (Cron-Ready Wrapper)
# =============================================================================
# Retrains the 12 gold-standard STAGING models — {initiation, persistence,
# discontinuation} x {Remibrutinib, Fabhalta, Kisqali} at patient grain plus
# hcp_adoption x 3 brands at HCP grain — on the CURRENT synthetic substrate,
# and re-records their walk-forward + holdout metric trends.
#
# Everything downstream is idempotent / re-run safe by construction:
#   - ml_model_registry rows UPSERT in place on the (model_name, model_version)
#     unique key, preserving row ids and therefore every RESTRICT FK from
#     ml_performance_metrics / ml_drift_history / ml_monitoring_alerts
#   - metric rows are delete-by-(model_id, source)-then-insert
#   - all 12 models stay stage='staging'; cohort_deployer hard-refuses
#     stage='production', so the serving ensemble is never touched
#   - the walk-forward re-windows by journey month, so new frontier months
#     become new backtest points with no code change
#
# Measured cost (2026-07-04, this droplet): ~43 s wall / ~570 MiB peak RSS for
# one slot; ~9 min for all 12 sequentially.
#
# NOT included here (heavier, service-restarting choreography — run manually
# when the serving layer should catch up): SHAP serving-bundle re-materialize
# + bentoml restart + SHAP cache refresh. See scripts/sync_goldstd_serving.py.
#
# Environment gotchas (same as reseed_synthetic.sh — do not "simplify" away):
#   - `dotenv` is NOT on the bare shell PATH — only .venv/bin/dotenv works
#   - PYTHONPATH must be the repo root for src.* imports
#   - LOKY_MAX_CPU_COUNT=1 keeps joblib from over-forking on the droplet
#
# Called weekly by scripts/reseed_synthetic.sh AFTER the frontier append and
# kpi_history backfill (opt out there with --skip-retrain). Also runnable
# standalone:
#   ./scripts/retrain_goldstd.sh >> /home/enunez/logs/e2i-reseed.log 2>&1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/dotenv || ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv/bin/dotenv or .venv/bin/python missing — run from a checkout with the project venv installed" >&2
    exit 1
fi

echo "=== goldstd retrain start $(date -Is) (9 patient slots) ==="

PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 E2I_DB_INTEGRATION=1 \
    .venv/bin/dotenv -f .env run -- \
    .venv/bin/python -m src.mlops.gold_standard_eval.run_patient_cohorts

echo "=== goldstd retrain patient slots done $(date -Is) (3 HCP slots) ==="

PYTHONPATH="$PROJECT_ROOT" LOKY_MAX_CPU_COUNT=1 E2I_DB_INTEGRATION=1 \
    .venv/bin/dotenv -f .env run -- \
    .venv/bin/python -m src.mlops.gold_standard_eval.run_hcp_cohorts

echo "=== goldstd retrain done $(date -Is) ==="
