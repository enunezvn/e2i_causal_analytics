-- ============================================================================
-- Migration 112: COMM-ARMS Phase 4 — trigger_accepted arm columns on
-- patient_journeys. Additive + idempotent; mirrors migration 088's contract.
-- Columns stay NULL until the next synthetic reseed populates them
-- (PatientGenerator Phase 4 wiring; the loader whitelist carries both).
-- ----------------------------------------------------------------------------

ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS trigger_accepted            SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS trigger_accepted_propensity DOUBLE PRECISION;

COMMENT ON COLUMN patient_journeys.trigger_accepted IS
    'COMM-ARMS Phase 4 commercial arm: >=1 NBA trigger for this patient was accepted by the rep (0/1). Confounded on disease_severity + engagement_score; enters ONLY the treatment_initiated latent. The triggers table''s acceptance_status is generated consistently with this arm (arm=1 <=> >=1 accepted trigger). Added by migration 112.';
COMMENT ON COLUMN patient_journeys.trigger_accepted_propensity IS
    'Estimable propensity P(trigger_accepted=1 | disease_severity, engagement_score), clipped [0.01, 0.99] for overlap. Added by migration 112.';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
