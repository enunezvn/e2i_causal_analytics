-- Migration 154: ml_experiments.brand for the goldstd experiments is the COHORT's brand,
-- not 'Remibrutinib' on every row (#2256).
--
-- WHY: src/mlops/prediction_synthesizer_deploy.py _get_or_create_experiment wrote the
-- module constant BRAND = "Remibrutinib" on every INSERT. That constant is correct for the
-- #840 serving deploy (its models are trained on the CSU/Remibrutinib synthetic scenario),
-- but the gold-standard eval pipeline (src/mlops/gold_standard_eval/cohort_deployer.py)
-- reused the helper for every cohort, so all 15 goldstd experiments were stamped
-- 'Remibrutinib' (measured 2026-09-23: brand|count = Remibrutinib|15), including the
-- Kisqali and Fabhalta cohorts (the hcp_adoption_kisqali / _fabhalta ones back
-- production-stage champions) and the two all-brand cohorts. The code now passes
-- spec.brand; this file corrects the rows already written.
--
-- DERIVATION (no hand-typed brand per row):
--   * 12 per-brand experiments: the brand token in the canonical name
--     goldstd_experiment_name(cohort, brand) = f"{cohort}_{brand.lower()}_goldstd_eval_v1"
--     (cohort_spec.py), matched case-insensitively to a brand_type enum label, and
--     required to agree with the prediction_target f"{cohort}_{brand.lower()}" the same
--     spec wrote. Cross-check: the 9 patient rows' registry contracts (migration 151,
--     cohort_data_source filters.brand) must agree, or this migration aborts.
--   * 2 all-brand experiments (targets pnh_persistence / pnh_discontinuation): the
--     PERSISTENCE / DISCONTINUATION CohortSpecs have brand=None (FeatureBuilder applies no
--     brand partition: trained on ALL brands), so brand is NULL (the column is nullable).
--     Their experiments were created 2026-06-14 19:46/19:47, after the brand=None specs
--     landed (bf579a650, 18:48). They have no registry rows, so no second source exists.
--   * initiation_goldstd_eval_v1 (csu_treatment_initiation): INITIATION.brand is
--     'Remibrutinib', so it is already correct and is not touched.
--
-- SAFETY: scoped to created_by = 'gold_standard_eval'. Every UPDATE is a compare-and-set on
-- the wrong constant (brand = 'Remibrutinib'), so a row something else has already healed
-- is never overwritten and a second application matches zero rows (idempotent). The
-- tr_ml_experiments_updated trigger bumps updated_at on the corrected rows.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

-- Pre-check: the registry contracts (the second, independent source) must not contradict
-- the name-derived brand of any per-brand experiment.
DO $$
DECLARE
    n_conflicts integer;
BEGIN
    SELECT count(*) INTO n_conflicts
      FROM ml_experiments e
      JOIN ml_model_registry r ON r.experiment_id = e.id
      JOIN (SELECT unnest(enum_range(NULL::brand_type))::text AS label) b
        ON lower(b.label) = substring(
               e.experiment_name
               FROM '^(?:initiation|persistence|discontinuation|hcp_adoption)_([a-z]+)_goldstd_eval_v1$')
     WHERE e.created_by = 'gold_standard_eval'
       AND r.cohort_data_source LIKE '{%'
       AND (r.cohort_data_source::jsonb -> 'filters') ? 'brand'
       AND (r.cohort_data_source::jsonb -> 'filters' ->> 'brand') IS DISTINCT FROM b.label;
    IF n_conflicts > 0 THEN
        RAISE EXCEPTION
            'migration 154: % registry contract(s) disagree with the name-derived experiment brand; refusing to guess',
            n_conflicts;
    END IF;
END
$$;

-- 12 per-brand experiments: brand from the canonical experiment name, consistent with
-- the prediction_target.
UPDATE ml_experiments e
   SET brand = b.label::brand_type
  FROM (SELECT unnest(enum_range(NULL::brand_type))::text AS label) b
 WHERE e.created_by = 'gold_standard_eval'
   AND lower(b.label) = substring(
           e.experiment_name
           FROM '^(?:initiation|persistence|discontinuation|hcp_adoption)_([a-z]+)_goldstd_eval_v1$')
   AND e.prediction_target = substring(e.experiment_name FROM '^(.*)_goldstd_eval_v1$')
   AND e.brand = 'Remibrutinib'
   AND e.brand IS DISTINCT FROM b.label::brand_type;

-- 2 all-brand experiments: the PERSISTENCE / DISCONTINUATION specs have brand=None.
UPDATE ml_experiments
   SET brand = NULL
 WHERE created_by = 'gold_standard_eval'
   AND (experiment_name, prediction_target) IN (
           ('persistence_goldstd_eval_v1', 'pnh_persistence'),
           ('discontinuation_goldstd_eval_v1', 'pnh_discontinuation'))
   AND brand = 'Remibrutinib';
