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
-- Additive + nullable (the 083 pattern): ADD COLUMN IF NOT EXISTS, COMMENT, a NULL-only
-- backfill from an authoritative source, no constraint. Safe to re-run.
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
    'frame cohort_data_source loads). Backfilled from ml_experiments.prediction_target '
    '(the experiment''s declared target) for the pre-existing real models. The sweep may '
    'only enqueue a retrain when BOTH cohort_data_source and cohort_target_outcome are present.';
COMMENT ON COLUMN ml_model_registry.cohort_feature_manifest_source IS
    'Cohort contract (#2207): the RESOLVED Layer-5 feature-manifest source '
    '(csu/optum/synthetic, src/data/manifests/resolution.py) the model was trained under; '
    'NULL = none/unknown. Optional pass-through of the retrain contract.';

-- ---------------------------------------------------------------------------
-- Backfill (NULL-only, real models only)
-- ---------------------------------------------------------------------------
-- cohort_target_outcome: the experiment's declared prediction target, via the FK. The
-- 14 is_synthetic=false rows on 2026-09-22 (12 `<cohort>_<brand>_goldstd_lr_v1` +
-- 2 archived `csu_treatment_initiation_lr_*`) all have a populated
-- ml_experiments.prediction_target (initiation_kisqali, hcp_adoption_remibrutinib,
-- csu_treatment_initiation, ...). Idempotent: NULL-only.
UPDATE ml_model_registry r
   SET cohort_target_outcome = e.prediction_target
  FROM ml_experiments e
 WHERE e.id = r.experiment_id
   AND r.cohort_target_outcome IS NULL
   AND r.is_synthetic = false
   AND e.prediction_target IS NOT NULL
   AND e.prediction_target <> '';

-- cohort_data_source: NOT backfilled -- the mapping is not provable as a
-- (table, target column) contract the retrain path could load:
--   * The 12 goldstd models were trained by
--     src/mlops/gold_standard_eval/cohort_deployer.train_cohort_model on frames built by
--     FeatureBuilder.load_frame (patient grain: patient_journeys filtered by brand/split;
--     HCP grain: hcp_brand_adoption JOIN hcp_profiles) with spec.label_column as the label
--     (treatment_initiated / persistent_180d / discontinued_180d / adopted --
--     src/mlops/gold_standard_eval/cohort_spec.py), NOT with the experiment's
--     prediction_target as a column: `initiation_kisqali` etc. is a column of no live
--     table (information_schema.columns, 2026-09-22: 0 rows for those 8 names). The
--     MLFoundationPipeline retrain loads ONE table (data_loader._load_from_supabase) and
--     reads scope_spec.prediction_target from it, so no cohort_data_source value would
--     reproduce that training; a fabricated one would enqueue jobs that fail at data-prep.
--   * The 2 csu_treatment_initiation_lr_* rows (archived; outside the sweep's
--     production/staging scope) were trained by src/mlops/prediction_synthesizer_deploy
--     .train_target_models on an in-process dataset, not a named table.
-- The sweep therefore stays honestly blocked for these 14 rows; the manual trigger route
-- heals the contract once an operator supplies a loadable data_source + target.
