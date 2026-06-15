-- ============================================================================
-- Migration 076: hcp_brand_adoption — per-brand, temporal HCP adoption cohort
-- ============================================================================
-- The gold-standard model-eval suite (src/mlops/gold_standard_eval/) already
-- serves 9 patient-grain models (initiation/persistence/discontinuation × 3
-- brands). The 4th cohort in the 4×3 vision is HCP-grain BRAND ADOPTION × 3
-- brands. hcp_profiles already carries the leakage-safe adoption substrate
-- (peer_influence_score, influence_network_size, years_experience, specialty,
-- geographic_region, adoption_category) BUT only a single static label per HCP
-- and NO brand or temporal dimension — so a per-brand, walk-forward-able cohort
-- cannot be expressed on hcp_profiles alone.
--
-- This table adds exactly that missing grain: one row per (hcp_id, brand) with
--   * adopted / adoption_category : the leakage-safe label from the SHARED DGP
--     _compute_adoption() (src/ml/synthetic/generators/hcp_adoption_artifact.py),
--     evaluated per-brand via _BRAND_ADOPT_SCALE.
--   * consideration_date : the temporal axis the walk-forward backtest rolls
--     over (the FeatureBuilder HCP load path aliases it to journey_start_date,
--     so walk_forward.py / recorder.py / cohort_deployer.py are reused untouched).
--   * data_split : champion train/validation vs holdout headline split.
-- The predictive FEATURES are NOT duplicated here — they live in hcp_profiles
-- and are JOINed at load time (single SSOT for HCP attributes).
--
-- Leakage-safe by construction: adoption is derived ONLY from exogenous
-- centrality + the confounded treatment arm (see hcp_adoption_artifact.py);
-- the leaky columns (days_to_first / first_adoption_dt / adopter_rank) are never
-- materialized. `consideration_date` is a row attribute, never a model feature.
--
-- Additive + idempotent: new table, IF NOT EXISTS throughout. is_synthetic
-- defaults FALSE ("real"); the gold-standard generator writes rows with
-- is_synthetic = TRUE, and the load path filters is_synthetic = TRUE.
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hcp_brand_adoption (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hcp_id              VARCHAR(20) NOT NULL REFERENCES hcp_profiles(hcp_id),
    brand               brand_type NOT NULL,
    consideration_date  DATE NOT NULL,
    adopted             INTEGER NOT NULL,
    adoption_category   VARCHAR(20),
    data_split          data_split_type NOT NULL DEFAULT 'unassigned',
    is_synthetic        BOOLEAN NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_hcp_brand_adoption UNIQUE (hcp_id, brand),
    CONSTRAINT ck_hcp_brand_adoption_adopted CHECK (adopted IN (0, 1))
);

-- Load query filters brand + is_synthetic; walk-forward rolls over
-- consideration_date. Composite (brand, is_synthetic) serves the per-brand load.
CREATE INDEX IF NOT EXISTS idx_hcp_brand_adoption_brand_synth
    ON hcp_brand_adoption (brand, is_synthetic);
CREATE INDEX IF NOT EXISTS idx_hcp_brand_adoption_consideration
    ON hcp_brand_adoption (consideration_date);

COMMENT ON TABLE hcp_brand_adoption IS
    'Per-brand, temporal HCP adoption cohort for the gold-standard model-eval '
    'suite (migration 076). Grain = (hcp_id, brand). Label (adopted / '
    'adoption_category) from the shared leakage-safe _compute_adoption DGP; '
    'consideration_date is the walk-forward axis (aliased to journey_start_date '
    'at load); features are JOINed from hcp_profiles (single SSOT). '
    'is_synthetic = TRUE marks gold-standard rows.';
COMMENT ON COLUMN hcp_brand_adoption.consideration_date IS
    'Temporal axis for the walk-forward backtest (HCP first considered the '
    'brand). Aliased to journey_start_date by the FeatureBuilder HCP load path '
    'so walk_forward.py (_DATE_COL=journey_start_date) is reused unchanged. '
    'Row attribute only — NEVER a model feature.';
COMMENT ON COLUMN hcp_brand_adoption.adopted IS
    '0/1 adoption label (model target); 1 == ADOPTER. Mirrors the integer '
    'label convention of patient_journeys.{treatment_initiated,persistent_180d}.';
