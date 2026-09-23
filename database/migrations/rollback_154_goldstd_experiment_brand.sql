-- Rollback for migration 154 (NOT auto-applied: the runner skips rollback_* files).
-- Restores the pre-154 state measured 2026-09-23: every goldstd experiment carried
-- brand = 'Remibrutinib'. Scoped to the same rows 154 derives (created_by =
-- 'gold_standard_eval' with a canonical goldstd name, plus the two all-brand
-- experiments). Rows already 'Remibrutinib' are untouched, so a second run is a no-op.
-- Note: this re-introduces the #2256 mis-attribution on purpose; it exists only to undo 154.
UPDATE ml_experiments
   SET brand = 'Remibrutinib'
 WHERE created_by = 'gold_standard_eval'
   AND (
           experiment_name ~ '^(initiation|persistence|discontinuation|hcp_adoption)_[a-z]+_goldstd_eval_v1$'
        OR (experiment_name, prediction_target) IN (
               ('persistence_goldstd_eval_v1', 'pnh_persistence'),
               ('discontinuation_goldstd_eval_v1', 'pnh_discontinuation'))
       )
   AND brand IS DISTINCT FROM 'Remibrutinib';
