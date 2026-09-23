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
-- Future goldstd registrations record the method themselves
-- (register_cohort_model(calibration_method=calibration_method_of(model))).
--
-- Each row: hyperparameters is '{}' today; the key is merged in, never overwritten (the
-- UPDATE skips a row that already records a method), scoped to the real fitted rows
-- (is_synthetic = false, the algorithm the artifact proves). Data-only: no DDL.
UPDATE ml_model_registry
   SET hyperparameters = COALESCE(hyperparameters, '{}'::jsonb)
                         || '{"calibration_method": "sigmoid"}'::jsonb
 WHERE model_name IN (
         'initiation_fabhalta_goldstd_lr_v1',
         'initiation_kisqali_goldstd_lr_v1',
         'initiation_remibrutinib_goldstd_lr_v1',
         'persistence_fabhalta_goldstd_lr_v1',
         'persistence_kisqali_goldstd_lr_v1',
         'persistence_remibrutinib_goldstd_lr_v1',
         'discontinuation_fabhalta_goldstd_lr_v1',
         'discontinuation_kisqali_goldstd_lr_v1',
         'discontinuation_remibrutinib_goldstd_lr_v1',
         'hcp_adoption_fabhalta_goldstd_lr_v1',
         'hcp_adoption_kisqali_goldstd_lr_v1',
         'hcp_adoption_remibrutinib_goldstd_lr_v1'
       )
   AND is_synthetic = false
   AND algorithm = 'logistic_regression_calibrated'
   AND NOT (COALESCE(hyperparameters, '{}'::jsonb) ? 'calibration_method');
