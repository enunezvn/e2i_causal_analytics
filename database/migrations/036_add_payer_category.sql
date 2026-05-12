-- Migration: 036_add_payer_category
-- Issue #156 item 6: extend payer vocabulary from 3-way INSURANCE_TYPE_MAP
-- to 8-value payer_category. Persist raw source fields alongside the derived
-- value to enable re-derivation without re-ETL.
--
-- The legacy `insurance_type` column is preserved for backwards compatibility.
-- Deprecation is a follow-up PR.
--
-- Forward-only. No rollback (the additive columns are nullable and have no
-- back-references).

BEGIN;

-- 8-value vocabulary per issue #156 item 6.
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS payer_category VARCHAR(30)
        CHECK (
            payer_category IS NULL
            OR payer_category IN (
                'commercial',
                'commercial_exchange',
                'medicare',
                'medicare_advantage',
                'medicare_lis_dual',
                'medicaid',
                'cash',
                'other'
            )
        );

-- Raw source fields for audit and re-derivation. Mirrors the precedent set
-- by source_combination_method (also persisted on patient_journeys).
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS payer_bus_raw VARCHAR(10);
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS payer_product_raw VARCHAR(20);
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS payer_health_exch_raw BOOLEAN;
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS payer_lis_dual_raw BOOLEAN;

-- Index for downstream cohort filtering by payer category. Partial index
-- excludes the NULL rows which are inert for analytics.
CREATE INDEX IF NOT EXISTS idx_patient_journeys_payer_category
    ON patient_journeys (payer_category)
    WHERE payer_category IS NOT NULL;

COMMIT;
