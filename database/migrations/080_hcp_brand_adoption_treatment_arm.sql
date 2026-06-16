-- ============================================================================
-- Migration 080: hcp_brand_adoption.treatment_arm — TRACKING migration
-- ============================================================================
-- WHY: the binary HCP-adoption treatment arm (`treatment_arm`, 0/1) and its
-- CHECK constraint (`ck_hcp_brand_adoption_treatment_arm`) were applied to the
-- LIVE DB out-of-band by the synthetic data-generating process (DGP) when it
-- populated the hcp_brand_adoption cohort (created by migration 076). This
-- migration exists ONLY to track that column + constraint in version control so
-- a FRESH database built from the migration chain reproduces the same shape.
--
-- IDEMPOTENT / NO-OP ON LIVE: verified live 2026-06-16 the column AND the
-- constraint already exist, so both guards below short-circuit and this migration
-- is a no-op on the production DB. A bare `ADD CONSTRAINT` (without the
-- pg_constraint existence guard) would ERROR on live — hence the DO-block guard
-- mirroring migration 072's pattern.
--
-- The GET /api/causal/treatment-effects endpoint (hcp_adoption cohort) reads
-- hcp_brand_adoption.treatment_arm (treatment) -> adopted (outcome), with
-- peer_influence_score (from hcp_profiles) as the confounder. treatment_arm is
-- confounded by HCP centrality by construction, so a naive difference-in-means
-- over-states the effect; the endpoint de-confounds via DoWhy/EconML backdoor.
-- ============================================================================

-- Column: idempotent add (no-op when already present on the live DB).
ALTER TABLE hcp_brand_adoption
    ADD COLUMN IF NOT EXISTS treatment_arm INTEGER;

COMMENT ON COLUMN hcp_brand_adoption.treatment_arm IS
    'Binary HCP-adoption treatment arm (0/1). The exposure for the hcp_adoption '
    'cohort in GET /api/causal/treatment-effects. Confounded by HCP centrality '
    '(peer_influence_score in hcp_profiles) by construction — de-confounded via '
    'backdoor adjustment in the causal pipeline, never reported as a raw mean diff.';

-- CHECK constraint: guarded add so it can never fail on an environment that
-- already holds it (the live DB) while still recreating it on a fresh DB.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_hcp_brand_adoption_treatment_arm'
          AND conrelid = 'hcp_brand_adoption'::regclass
    ) THEN
        ALTER TABLE hcp_brand_adoption
            ADD CONSTRAINT ck_hcp_brand_adoption_treatment_arm
            CHECK (treatment_arm IS NULL OR treatment_arm IN (0, 1));
        RAISE NOTICE 'hcp_brand_adoption: added ck_hcp_brand_adoption_treatment_arm';
    END IF;
END $$;
