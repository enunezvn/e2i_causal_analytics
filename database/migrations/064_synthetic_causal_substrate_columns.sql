-- ============================================================================
-- Migration 064 (M2): per-unit causal substrate columns on patient_journeys
-- (synthetic-causal-validation). Absorbs Shard 06's orphaned outcome DDL.
-- ============================================================================
-- The DGP (Shard 03) writes a per-unit treatment arm + propensity + segment;
-- the cohort-outcomes shard (Shard 06) writes the disc/persistence outcome
-- columns. These MUST exist before any generator load. data_lag_hours already
-- EXISTS (no DDL); ml_predictions.{treatment_effect_estimate,
-- heterogeneous_effect, segment_assignment} already exist (no DDL here).
-- Canonical names only (never treatment/outcome/severity_score/age/cohort).
-- Additive + idempotent: existing rows default NULL until the generator loads.
-- ----------------------------------------------------------------------------

ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS treatment_arm      SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS propensity_score   DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS segment_assignment TEXT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS discontinued_180d  SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS persistent_180d    SMALLINT;

COMMENT ON COLUMN patient_journeys.treatment_arm IS
    'Binary treatment arm T (0/1) assigned by the synthetic DGP (Shard 03). Added by migration 064 (M2).';
COMMENT ON COLUMN patient_journeys.propensity_score IS
    'Designed propensity e(X)=P(T=1|X) from the synthetic DGP (overlap-respecting). Added by migration 064 (M2).';
COMMENT ON COLUMN patient_journeys.segment_assignment IS
    'Effect-modifier segment label (high_severity/medium_severity/low_severity) driving CATE. Added by migration 064 (M2).';
COMMENT ON COLUMN patient_journeys.discontinued_180d IS
    'Discontinuation cohort outcome (1=discontinued within 180d). Filtered treatment_initiated=1. Added by migration 064 (M2).';
COMMENT ON COLUMN patient_journeys.persistent_180d IS
    'Persistence cohort outcome (1=persistent at 180d). Filtered treatment_initiated=1. Added by migration 064 (M2).';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
