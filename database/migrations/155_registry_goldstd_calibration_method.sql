-- Migration 155: record the post-hoc calibration method of the 12 goldstd registry rows
-- (#2248 option (a), owner decision 2026-09-23: a retrain uses the calibration method of
-- the model it retrains).
--
-- WHY: the retrain path (execute_model_retraining -> MLFoundationPipeline -> model_trainer)
-- let the evaluator's auto policy pick the calibrator. With > 100 validation positives it
-- picks isotonic, whose exact-0/1 plateaus failed maximum_calibration_slope_deviation on the
-- faithful re-run of job c4252b47 (initiation_kisqali_goldstd_lr_v1: slope 0.690, deviation
-- 0.310 > cap 0.161). The goldstd model it retrains is Platt-calibrated; with the same
-- method every gate passes at unchanged thresholds (deviation 0.012). The method of record
-- is read from ml_model_registry.hyperparameters.calibration_method
-- (src/services/cohort_contract.py::recorded_calibration_method); a row without it keeps
-- the auto policy and the retrain says so.
--
-- PROVABLE (measured 2026-09-23, read-only): every one of the 12 registered artifacts
-- (the row's artifact_path on the host AND the /app/data/ml_artifacts copy the workers
-- mount) unpickles to CalibratedClassifierCV with method='sigmoid', cv=3 -- the estimator
-- src/mlops/gold_standard_eval/cohort_deployer.py::train_cohort_model builds
-- (CalibratedClassifierCV(base, method="sigmoid", cv=cv)). The rows' algorithm is
-- 'logistic_regression_calibrated'; that string alone is NOT proof (the bare-LR
-- fallback for a minority class < 2 carries it too), hence the per-row artifact read.
-- Future goldstd registrations record the method themselves (register_cohort_model reads
-- it off the artifact it registers); pipeline deploys record the method the trainer
-- applied (model_trainer agent -> ml_training_runs.hyperparameters -> registry_manager).
--
-- Each UPDATE is pinned to the AUDITED registration: the row id AND its trained_at (every
-- register_model_row upsert restamps trained_at, so a row re-registered with a different
-- artifact after this audit -- e.g. the bare-LR fallback -- no longer matches and is left
-- alone; register_cohort_model now records such an artifact's method itself). The key is
-- merged in, never overwritten (a row that already records a method is skipped), scoped to
-- is_synthetic = false. Data-only: no DDL. hyperparameters is '{}' on all 12 today.

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = 'b933a5d0-cf0d-44cd-ac75-d34fab2259ea'
   AND model_name = 'discontinuation_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:58.395706+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = 'f2da0bd5-4d6b-47c4-9d30-6c8b4b59de46'
   AND model_name = 'discontinuation_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:05:22.83531+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '5d0486b4-fd74-463e-aab0-d91ac9fe0fca'
   AND model_name = 'discontinuation_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:44.114318+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = 'dab9a91b-708d-4b0a-a332-803cef3d3245'
   AND model_name = 'hcp_adoption_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:06:23.393962+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '8d1244df-7c38-435e-820f-1f201e51af24'
   AND model_name = 'hcp_adoption_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:06:38.973136+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '8b7a1e4e-be3c-469f-822b-46d9ab3dad81'
   AND model_name = 'hcp_adoption_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:05:57.641562+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = 'a11a50e2-42ed-4233-8711-bbdf233d3a0d'
   AND model_name = 'initiation_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:12.266514+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '4ec55d13-46c8-4df4-9ec8-7723fad67fb3'
   AND model_name = 'initiation_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:31.021107+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '2db8b0e0-1d9e-4e18-bf7a-576a4796610d'
   AND model_name = 'initiation_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:02:51.799003+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '3f5011b5-b16a-4ee7-afd8-4be9862e15d4'
   AND model_name = 'persistence_fabhalta_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:08.079086+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = 'd3a926ac-e24a-4847-ac15-152d2983e466'
   AND model_name = 'persistence_kisqali_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:04:23.038637+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');

UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE id = '64728d92-378a-407b-b670-4eb3337a63c1'
   AND model_name = 'persistence_remibrutinib_goldstd_lr_v1'
   AND trained_at = '2026-09-21 03:03:49.02015+00'
   AND is_synthetic = false
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');
