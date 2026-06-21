-- T9 (2026-06-21): prognostic persistence drivers on patient_journeys.
-- Additive, nullable, pre-index (drawn INDEPENDENTLY of treatment_arm in the DGP)
-- => leakage-safe. These two columns feed the enriched discontinuation/persistence
-- structural equation (src/ml/synthetic/generators/cohort_outcomes.py) alongside the
-- already-existing insurance_type + age_at_diagnosis, lifting achievable model AUC
-- from ~0.70 to a realistic ~0.78-0.82 and making /feature-importance rank 7
-- covariates instead of 3. See docs/superpowers/specs/2026-06-21-persistence-dgp-enrichment-design.md.
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS comorbidity_burden  SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS prior_therapy_lines SMALLINT;

COMMENT ON COLUMN patient_journeys.comorbidity_burden  IS 'Pre-index comorbidity count (0-5); prognostic driver of 180d persistence (T9).';
COMMENT ON COLUMN patient_journeys.prior_therapy_lines IS 'Pre-index prior therapy lines (0-3); prognostic driver of 180d persistence (T9).';
