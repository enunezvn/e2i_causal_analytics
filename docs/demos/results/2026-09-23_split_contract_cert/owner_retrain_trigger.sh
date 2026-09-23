#!/usr/bin/env bash
# OWNER-RUN ONLY (prod write: creates an ml_retraining_history row and runs a real retrain on worker_medium).
# Faithful proof for PR #2241 / migration 151: one manual retrain of initiation_kisqali_goldstd_lr_v1
# from its persisted cohort contract (table dict + treatment_initiated + synthetic_csu manifest).
# Run from the repo root on the droplet:  bash <this file>
set -euo pipefail
exec > >(tee -a /tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/625a5714-efae-4077-ad0e-68c512e5a6c7/scratchpad/trigger_run.out) 2>&1
cd /home/enunez/Projects/e2i_causal_analytics
set -a; . ./.env; set +a
API="${E2I_API_BASE:-https://eznomics.site}"
MODEL_ID="4ec55d13-46c8-4df4-9ec8-7723fad67fb3"   # registry uuid of initiation_kisqali_goldstd_lr_v1 (model_version='1.0' is NOT unique — use the uuid)

# 1) admin JWT — the scripts/sync_goldstd_serving.py::_admin_token mechanism (GoTrue password grant)
TOKEN=$(curl -sS "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@e2i.local\",\"password\":\"$E2I_ADMIN_PASSWORD\"}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

# 2) the persisted contract is the input (no explicit data_source/target → the row's contract is used)
echo "== registry contract"; docker exec -i supabase-db psql -U postgres -d postgres -At -F' | ' \
  -c "SELECT model_name, cohort_target_outcome, cohort_feature_manifest_source, left(cohort_data_source,80) FROM ml_model_registry WHERE id='$MODEL_ID'"
echo "== history rows before"; docker exec -i supabase-db psql -U postgres -d postgres -At -c "SELECT count(*) FROM ml_retraining_history"

# 3) trigger (auto_approve so the job is enqueued immediately on the analytics queue → worker_medium)
curl -sS -X POST "$API/api/monitoring/retraining/trigger/$MODEL_ID?triggered_by=owner_split_contract_proof" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"reason":"manual","notes":"PR #2241 / migration 151 faithful proof","auto_approve":true}' | python3 -m json.tool

# 4) watch (re-run this line until status is completed or failed; a run takes minutes)
echo "== watch"; docker exec -i supabase-db psql -U postgres -d postgres -At -F' | ' \
  -c "SELECT id, status, trigger_reason, old_metric_value, new_metric_value, improvement, left(notes,200) AS notes, triggered_at, completed_at FROM ml_retraining_history ORDER BY created_at DESC LIMIT 1"
# PASS criteria: status=completed; performance_after (validation AUC) in a sane band vs the goldstd reference 0.8505
# (≈1.0 would mean leakage = plausible-wrong → FAIL); the registry row unchanged (already non-NULL; heal refuses conflicts).
# Known limit (#2242): the retrained candidate is registered under a generated experiment id, not as a new version of this row.
