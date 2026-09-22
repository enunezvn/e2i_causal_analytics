-- Migration 150: the per-model COHORT CONTRACT on the model catalog (#2207 follow-up,
-- owner decision 2026-09-22 on PR #2223 blocker 1).
--
-- WHY: the daily retraining sweep (`retraining-evaluation-daily` ->
-- check_retraining_for_all_models -> evaluate_retraining_need) can only enqueue a
-- retraining job that names the committed cohort it retrains on: the job runs
-- MLFoundationPipeline on `data_source` (the Supabase table / file bundle the
-- data_preparer loads) with `target_outcome` (the prediction target column), and
-- fails closed without them (_cohort_input_from_training_config). No persisted model
-- record carried that identity — ml_model_registry had no data_source column and
-- ml_experiments only prediction_target/brand — so the sweep evaluated and refused to
-- enqueue for every model (retraining_blocked_reason="no_cohort_contract"), and
-- ml_retraining_history stayed at 0 rows. The registry row is the entity the sweep
-- iterates (is_synthetic=false, stage in production/staging), so the contract lives
-- there: no join, no new table, no change to the #894-hardened is_synthetic scoping.
--
-- WRITERS: (1) the model_deployer's registry writer at training time
-- (registry_manager._persist_model_registry_row <- MLFoundationPipeline.run input_data);
-- (2) the manual trigger route POST /monitoring/retraining/trigger/{model_id}, which
-- heals a NULL contract from an explicit, complete request (RetrainingTriggerService).
-- READER: the drift-monitor Supabase connector projection -> the sweep's cohort dict.
--
-- Additive + nullable (the 083 pattern): ADD COLUMN IF NOT EXISTS, COMMENT, NO backfill
-- (see the end of this file), no constraint. Safe to re-run.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS cohort_data_source TEXT;
ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS cohort_target_outcome TEXT;
ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS cohort_feature_manifest_source TEXT;

COMMENT ON COLUMN ml_model_registry.cohort_data_source IS
    'Cohort contract (#2207): the committed cohort table/batch the model was trained on '
    '-- the data_source MLFoundationPipeline loads for a retrain (a Supabase table name, '
    'or the JSON of a file-source dict {"type": "file_dir"|"files", ...}). The scheduled '
    'retraining sweep may only enqueue a retrain when BOTH cohort_data_source and '
    'cohort_target_outcome are present; NULL = unknown, the sweep stays blocked for this '
    'model until a training run or an explicit manual trigger persists it.';
COMMENT ON COLUMN ml_model_registry.cohort_target_outcome IS
    'Cohort contract (#2207): the prediction target the model was trained on -- the '
    'target_outcome MLFoundationPipeline predicts on a retrain (must be a column of the '
    'frame cohort_data_source loads); NULL = unknown — never backfilled from '
    'ml_experiments.prediction_target, which is an experiment label, not a column. The '
    'sweep may only enqueue a retrain when BOTH cohort_data_source and cohort_target_outcome are present.';
COMMENT ON COLUMN ml_model_registry.cohort_feature_manifest_source IS
    'Cohort contract (#2207): the RESOLVED Layer-5 feature-manifest source '
    '(csu/optum/synthetic, src/data/manifests/resolution.py) the model was trained under; '
    'NULL = none/unknown. Optional pass-through of the retrain contract.';

-- ---------------------------------------------------------------------------
-- NO backfill (dispatcher review 2026-09-22)
-- ---------------------------------------------------------------------------
-- Neither value is provable for the 14 pre-existing real rows (12
-- `<cohort>_<brand>_goldstd_lr_v1` + 2 archived `csu_treatment_initiation_lr_*`):
--   * cohort_target_outcome: ml_experiments.prediction_target holds the experiment's
--     LABEL (initiation_kisqali, hcp_adoption_remibrutinib, csu_treatment_initiation,
--     ...), which is a column of NO live table (information_schema.columns, 2026-09-22:
--     0 rows for those names; only hcp_brand_adoption.adopted exists) — the goldstd
--     models were trained on spec.label_column (treatment_initiated / persistent_180d /
--     discontinued_180d / adopted, src/mlops/gold_standard_eval/cohort_spec.py). A
--     contract column holding a plausible-looking non-column would make the sweep
--     enqueue a job with a bogus target the moment a data source is set, and the
--     conflict-refusing heal would then never let the correct value land.
--   * cohort_data_source: the goldstd frames were host-built by FeatureBuilder.load_frame
--     (patient grain: patient_journeys filtered by brand/split; HCP grain:
--     hcp_brand_adoption JOIN hcp_profiles), not ONE table data_loader could reload; the
--     csu rows were trained on an in-process dataset (prediction_synthesizer_deploy).
-- Honest state: all 14 rows carry NULL contracts and the sweep stays blocked for them
-- until an operator triggers POST /monitoring/retraining/trigger/{model_id} with
-- data_source + target_outcome; that job's COMPLETED run heals the row.
