# #2157 live retrain promotion certification (2026-09-28)

Model: `initiation_kisqali_goldstd_lr_v1` (parent registry row `4ec55d13-46c8-4df4-9ec8-7723fad67fb3`, v1.0,
sigmoid, synthetic_gold, experiment `35a2cd41-4d85-4034-b1a0-1c2a61e3766e`). Trigger:
`owner_retrain_trigger_post2264.sh` (admin `POST /api/monitoring/retraining/trigger/{id}`, `auto_approve=true`).

## Run 1: prod 815478760 (after #2302 / #2296): FAIL at a new point
Job `073c38eb-586b-4f83-adb3-4a7ec0d1d20b`, T0 16:17:40Z. The #2296 fix held: the training run persisted, the model
logged (`models:/m-a747…`), the deployer wrote registry row `c524db0f` and MLflow v3. BentoML 1.4.39 prints
`Successfully built Bento(tag="name:v3")`, and the tag parser kept `tag="` → `bento_validation_error`. The job
failed closed with a misattributed message; there was no model_deployer episodic row, and the Stage-4 log showed
`primary_metric=0.0000`. Files: `live_retrain_post2302_*`.

## Run 2: prod c16bdc8a9 (contains #2306 + #2307): PASS
Job `836578bf-0456-433d-9593-dbd373581579`, T0 19:34Z. File: `live_retrain_c16bdc8a9_PASS_20260928.out`.
1. History row `completed`, `new_metric_value=0.835903`, new version `1.0_retrained_20260928_1934_725a94`, notes name
   the candidate row `cff4f2b5-a87e-4947-aa2a-243a7fb0ee45` and "register/promote only — no Bento packaged, no
   endpoint deployed".
2. Candidate registry row: same model_name, new version, same experiment_id, synthetic_gold, sigmoid, `staging`;
   MLflow v4 (run `830aef651bba4bf5bb7202a96e614e7a`) in Staging, linked by `mlflow_run_id` / `mlflow_model_uri`.
3. Episodic rows: all SIX agents (scope_definer, data_preparer, model_selector, model_trainer, feature_analyzer,
   model_deployer) on ONE `audit_workflow_id` `db585587-0996-4478-bbf5-7fa90e70b3eb`, session NULL.
4. No new experiment rows (#2257).
5. Worker: sigmoid calibration ECE 0.1068 → 0.0295, success_criteria_met=True, Stage-4 `primary_metric=0.8363`.

## Follow-ups filed
#2308 (register-only `ml_deployments` row marked `active` with no endpoint), #2310 (staging candidates picked up
by explain / KPI n_models / drift sweeps), #2311 (task result `mlflow_model_version=None`), #2304 (flaky timing test).
