-- ============================================================================
-- Migration 088: commercial-arms + binarized-adherence columns on
-- patient_journeys (dgp-commercial-arms-enrichment). Additive + idempotent.
-- All columns front-loaded; NULL until the generator's per-phase wiring fills
-- them. Phase 0 populates adherent_180d/low_gap_180d/adherence_rate/gap_days;
-- the arm + per-arm-propensity + insurance_access_score columns stay NULL until
-- Phases 1-3. Canonical names only. Mirrors migration 064's contract.
-- ----------------------------------------------------------------------------

-- Phase 0: binarized adherence outcomes + raw proxies
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS adherent_180d   SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS low_gap_180d    SMALLINT;
-- adherence_rate / gap_days were added by migration 033 (which always runs first),
-- so these ADD-IF-NOT-EXISTS are no-ops on a real DB; declared here only for fresh-DB
-- self-containment and typed to MATCH migration 033 (adherence_rate NUMERIC, gap_days
-- INTEGER — whole refill-gap days; the generator emits integer days).
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS adherence_rate  NUMERIC;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS gap_days        INTEGER;

-- Phases 1-3: new arms + per-arm propensity + numeric insurance proxy (NULL now)
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS copay_support                 SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS psp_enrolled                  SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS rep_detailing_high            SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS sample_dropped                SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS copay_support_propensity      DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS psp_enrolled_propensity       DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS rep_detailing_high_propensity DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS sample_dropped_propensity     DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS insurance_access_score        DOUBLE PRECISION;

COMMENT ON COLUMN patient_journeys.adherent_180d IS
    'Binarized adherence outcome (1 = PDC adherence_rate >= 0.8 at 180d). Recoverable effect of treatment_arm. Added by migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.low_gap_180d IS
    'Binarized refill-gap outcome (1 = gap_days <= 30 at 180d). Recoverable effect of treatment_arm. Added by migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.adherence_rate IS
    'Continuous PDC proxy of the adherence latent (covariate). Populated by the generator. Migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.gap_days IS
    'Continuous refill-gap-days proxy (inverse of adherence latent; covariate). Migration 088 (Phase 0).';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
