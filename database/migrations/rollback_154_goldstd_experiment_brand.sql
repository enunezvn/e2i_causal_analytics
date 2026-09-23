-- Rollback for migration 154 (NOT auto-applied: the runner skips rollback_* files).
-- Restores the pre-154 state measured 2026-09-23: every goldstd experiment carried
-- brand = 'Remibrutinib'. It reverts ONLY what 154 wrote: the same name/target
-- derivation, the exact post-154 value (the derived brand, or NULL for the two
-- all-brand rows), and only the measured pre-154 rows (all 15 were created
-- 2026-06-14, hence the 2026-06-15 cutoff), so a goldstd experiment created later by
-- the fixed code, or a row corrected to some other value, is left alone. A second run
-- matches zero rows.
-- Note: this re-introduces the #2256 mis-attribution on purpose; it exists only to undo 154.

-- 8 per-brand rows 154 moved to Kisqali/Fabhalta.
UPDATE ml_experiments e
   SET brand = 'Remibrutinib'
  FROM (SELECT unnest(enum_range(NULL::brand_type))::text AS label) b
 WHERE e.created_by = 'gold_standard_eval'
   AND e.created_at < '2026-06-15T00:00:00Z'
   AND lower(b.label) = substring(
           e.experiment_name
           FROM '^(?:initiation|persistence|discontinuation|hcp_adoption)_([a-z]+)_goldstd_eval_v1$')
   AND e.prediction_target = substring(e.experiment_name FROM '^(.*)_goldstd_eval_v1$')
   AND e.brand = b.label::brand_type
   AND e.brand <> 'Remibrutinib';

-- 2 all-brand rows 154 moved to NULL.
UPDATE ml_experiments
   SET brand = 'Remibrutinib'
 WHERE created_by = 'gold_standard_eval'
   AND created_at < '2026-06-15T00:00:00Z'
   AND (experiment_name, prediction_target) IN (
           ('persistence_goldstd_eval_v1', 'pnh_persistence'),
           ('discontinuation_goldstd_eval_v1', 'pnh_discontinuation'))
   AND brand IS NULL;
