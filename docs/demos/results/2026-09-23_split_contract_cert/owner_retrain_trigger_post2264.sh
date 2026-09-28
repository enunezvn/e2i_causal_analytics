#!/usr/bin/env bash
# OWNER-RUN ONLY (prod write: creates an ml_retraining_history row, runs a real retrain on
# worker_medium and — if every gate passes — registers a candidate row in ml_model_registry).
# Live proof after PRs #2261 (#2242), #2262 (#2257/#2255), #2264 (#2248 option (a), migration 155):
# one manual retrain of initiation_kisqali_goldstd_lr_v1 from its persisted cohort contract.
# Run from the repo root on the droplet AFTER the deploy of 6720959b8 is live:  bash <this file>
set -euo pipefail
cd /home/enunez/Projects/e2i_causal_analytics
set -a; . ./.env; set +a
API="${E2I_API_BASE:-https://eznomics.site}"
MODEL_ID="4ec55d13-46c8-4df4-9ec8-7723fad67fb3"   # registry uuid of initiation_kisqali_goldstd_lr_v1
PSQL=(docker exec -i supabase-db psql -U postgres -d postgres -At -F ' | ')

echo "== preflight: deployed image must contain 6720959b8"
IMG=$(docker inspect e2i_api --format '{{.Config.Image}}' | sed 's/.*://')
git merge-base --is-ancestor 6720959b8 "$IMG" && echo "image $IMG contains #2264" || { echo "ABORT: image $IMG predates #2264"; exit 1; }
echo "== parent row (must show calibration_method=sigmoid, provenance synthetic_gold)"
"${PSQL[@]}" -c "SELECT model_name, model_version, hyperparameters->>'calibration_method', training_provenance, experiment_id FROM ml_model_registry WHERE id='$MODEL_ID'"
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ); echo "T0=$T0"

# admin JWT (GoTrue password grant, as scripts/sync_goldstd_serving.py::_admin_token)
TOKEN=$(curl -sS "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@e2i.local\",\"password\":\"$E2I_ADMIN_PASSWORD\"}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -sS -X POST "$API/api/monitoring/retraining/trigger/$MODEL_ID?triggered_by=owner_post2264_proof" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"reason":"manual","notes":"post-#2264 live proof (#2242/#2255/#2257/#2248)","auto_approve":true}' | python3 -m json.tool

echo "== waiting for the job to finish (a run takes ~2 min)"
for i in $(seq 1 30); do
  S=$("${PSQL[@]}" -c "SELECT status FROM ml_retraining_history WHERE created_at >= '$T0' ORDER BY created_at DESC LIMIT 1")
  case "$S" in completed|failed) break;; esac; sleep 20
done

echo "== 1) history row (PASS: completed, new_metric_value ~0.83 vs goldstd 0.8505, new_model_version = <1.0>_retrained_<ts>_<hex>)"
"${PSQL[@]}" -c "SELECT id, status, old_model_version, new_model_version, old_metric_value, new_metric_value, left(notes,300) FROM ml_retraining_history WHERE created_at >= '$T0' ORDER BY created_at DESC LIMIT 1"
echo "== 2) candidate registry row (PASS: same model_name, new version, SAME experiment_id as parent, provenance synthetic_gold, calibration sigmoid)"
"${PSQL[@]}" -c "SELECT model_name, model_version, experiment_id, training_provenance, hyperparameters->>'calibration_method', stage FROM ml_model_registry WHERE model_name='initiation_kisqali_goldstd_lr_v1' ORDER BY model_version"
echo "== 3) episodic rows since T0 (PASS for #2157/#2120: all SIX agents incl. model_deployer, ONE shared audit_workflow_id, session NULL)"
"${PSQL[@]}" -c "SELECT agent_name, coalesce(session_id::text,'NULL'), raw_content->>'audit_workflow_id' FROM episodic_memories WHERE agent_name IN ('scope_definer','data_preparer','feature_analyzer','model_selector','model_trainer','model_deployer') AND created_at >= '$T0' ORDER BY created_at"
echo "== 4) no new experiment rows named after the physical label (PASS: 0 — #2257)"
"${PSQL[@]}" -c "SELECT count(*) FROM ml_experiments WHERE created_at >= '$T0'"
echo "== 5) worker log: calibration + criteria"
docker logs e2i-causal-analytics-worker_medium-1 --since "$T0" 2>&1 | grep -E "calibration_method|Calibration:|criteria_source|success_criteria_met|Retraining .*(completed|failed)" | grep -v "Calculating Metrics" | tail -12
