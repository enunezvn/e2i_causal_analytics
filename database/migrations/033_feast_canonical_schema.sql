-- =============================================================================
-- Migration 033: Canonical schema for Feast feature views; drops bridging views
-- =============================================================================
-- Promotes the bridging columns (patient/digital/prescribing tiers, brand_id on
-- triggers, response timing, brand-derived composites, per-HCP business_metrics
-- columns, event_timestamp on territory_metrics) into the canonical tables, then
-- drops the four bridging views and the synthetic seed table created by
-- migrations 031 + 032. Feast feature views in 6B-infra-3 will be repointed to
-- the canonical tables via feature_repo/data_sources.py.
--
-- Tables modified:
--   1. hcp_profiles      add tier columns + years_of_practice; backfill
--                        territory_id NULLs to 'UNASSIGNED' sentinel.
--   2. triggers          add brand_id (default 'UNKNOWN' sentinel) + several
--                        STORED generated columns (trigger_date, is_responded,
--                        channel, response_time_hours, conversion_flag,
--                        hcp_brand_id) + nullable roi_estimate.
--   3. patient_journeys  add adherence_rate, refill_count, gap_days (nullable
--                        for ETL backfill) + STORED generated columns
--                        (event_date, is_churned, brand_id, patient_brand_id).
--   4. business_metrics  add nullable hcp_id (per-HCP rows; pre-existing
--                        per-brand+region aggregate rows stay hcp_id IS NULL),
--                        event_timestamp generated, hcp_brand_id generated,
--                        and per-HCP semantics columns (trx_count, nrx_count,
--                        total_rx_count, market_share, conversion_rate,
--                        engagement_score, call_frequency).
--   5. territory_metrics add event_timestamp generated column (so Feast can
--                        use a real timestamp column instead of metric_date).
--
-- Sentinels (chosen because canonical lookup tables do not yet exist):
--   * hcp_profiles.territory_id = 'UNASSIGNED'   (no sales_rep -> territory map)
--   * triggers.brand_id         = 'UNKNOWN'      (no clean trigger->brand join)
--   These sentinels surface in real ETL (6B-infra-2*) as flags to populate.
--
-- Generated columns: ALL expressions are IMMUTABLE (DATE(), EXTRACT(),
-- comparisons, COALESCE, string concat, enum->TEXT casts). No NOW(), random(),
-- or CURRENT_DATE -- those would be rejected by PostgreSQL inside STORED
-- generated columns.
--
-- View drops at end (after columns + backfill, so canonical replacement is in
-- place when bridge schema disappears):
--   DROP VIEW   feast_hcp_profile_source
--   DROP VIEW   feast_trigger_response_source
--   DROP VIEW   feast_patient_journey_source
--   DROP VIEW   feast_business_metrics_source
--   DROP TABLE  feast_business_metrics_seed
--
-- Transaction: the migration runner (scripts/run_migrations.sh) wraps every
-- migration with `psql --single-transaction -v ON_ERROR_STOP=1`. Do NOT add
-- BEGIN/COMMIT in here -- psql rejects nested BEGIN under --single-transaction.
-- The schema_migrations row is also auto-inserted by the runner; we do not
-- write one ourselves.
--
-- Idempotency: every ALTER uses ADD COLUMN IF NOT EXISTS; UPDATEs are guarded
-- by NULL checks; DROPs use IF EXISTS. Safe to re-run defensively even though
-- the runner gates re-runs by filename.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 033.1  hcp_profiles: tier columns, years_of_practice alias, territory_id backfill
-- -----------------------------------------------------------------------------
-- Tier columns are PLAIN TEXT (not GENERATED) per plan 6B-infra-1: they are
-- "derived in app code; column exists for materialize". The 6B-infra-2* ETLs
-- will populate them from total_patient_volume / digital_engagement_score /
-- prescribing_volume using the same CASE bands the bridging view used.
ALTER TABLE hcp_profiles ADD COLUMN IF NOT EXISTS patient_volume_tier TEXT;
ALTER TABLE hcp_profiles ADD COLUMN IF NOT EXISTS digital_engagement_tier TEXT;
ALTER TABLE hcp_profiles ADD COLUMN IF NOT EXISTS prescribing_tier TEXT;

-- years_of_practice: alias of years_experience, populated via UPDATE in this
-- migration. Keeping in sync going forward is out of scope (a later block adds
-- a trigger or the app writes both); 6B-infra-1 only seeds the initial value.
ALTER TABLE hcp_profiles ADD COLUMN IF NOT EXISTS years_of_practice INTEGER;
UPDATE hcp_profiles
   SET years_of_practice = years_experience
 WHERE years_of_practice IS NULL
   AND years_experience  IS NOT NULL;

-- territory_id backfill: stamp NULL rows with 'UNASSIGNED' sentinel.
-- A sales_reps -> territory_id lookup table does not exist in canonical
-- v3 schema (verified against database/core/e2i_ml_complete_v3_schema.sql),
-- so the plan's preferred path (sales_rep_id -> territory_id) is unavailable
-- and we fall back to the explicit sentinel choice noted in the plan.
UPDATE hcp_profiles
   SET territory_id = 'UNASSIGNED'
 WHERE territory_id IS NULL;

-- Post-backfill assertion: zero NULL territory_id rows. ON_ERROR_STOP=1 in the
-- runner ensures a non-zero count rolls back the entire migration.
DO $$
DECLARE
    null_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO null_count FROM hcp_profiles WHERE territory_id IS NULL;
    IF null_count <> 0 THEN
        RAISE EXCEPTION 'territory_id backfill failed: % NULL rows remain (expected 0)', null_count;
    END IF;
END
$$;

-- -----------------------------------------------------------------------------
-- 033.2  triggers: brand_id sentinel, generated timing/flag columns, hcp_brand_id
-- -----------------------------------------------------------------------------
-- brand_id added with NOT NULL DEFAULT 'UNKNOWN' so the existing rows pick up
-- the sentinel; we then DROP DEFAULT so future inserts must supply a value.
-- A heuristic backfill from trigger_reason / business_metrics is NOT attempted:
-- trigger_reason is free-form TEXT, and business_metrics rows are aggregate
-- per (brand, region) -- there is no clean join key to a single (hcp, brand).
-- 6B-infra-2a (per-HCP business_metrics ETL) will produce the per-HCP brand
-- mapping going forward; back-stamping pre-existing trigger rows is out of
-- scope for this migration.
ALTER TABLE triggers ADD COLUMN IF NOT EXISTS brand_id TEXT NOT NULL DEFAULT 'UNKNOWN';
ALTER TABLE triggers ALTER COLUMN brand_id DROP DEFAULT;

-- trigger_date: DATE(trigger_timestamp) -- canonical timestamp, no 1h-backdate
-- hack like the bridging view used. This does mean parity tests that rely on
-- "recent" timestamps must be driven by ETL writing fresh trigger_timestamp
-- values; that is 6B-infra-2*'s responsibility, not this migration's.
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS trigger_date DATE
    GENERATED ALWAYS AS (DATE(trigger_timestamp)) STORED;

-- is_responded: surfaces the same flag the bridging view exposed.
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS is_responded BOOLEAN
    GENERATED ALWAYS AS (acceptance_status IN ('accepted','responded')) STORED;

-- channel: plain alias for delivery_channel so Feast feature views can use the
-- canonical name without a view layer.
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS channel TEXT
    GENERATED ALWAYS AS (delivery_channel) STORED;

-- response_time_hours: hours between delivery and acceptance. Per plan, uses
-- (acceptance_timestamp - delivery_timestamp) -- migration 031's view used
-- (action_timestamp - trigger_timestamp); the canonical name is more accurate.
-- NULL when either timestamp is NULL (subtraction of NULL TIMESTAMPTZ -> NULL).
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS response_time_hours NUMERIC
    GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (acceptance_timestamp - delivery_timestamp)) / 3600.0) STORED;

-- conversion_flag: outcome_value > 0 per plan. Note the bridging view used
-- > 0.5; we deliberately follow the plan as written. outcome_value is
-- DECIMAL(4,3) in [0, 1], so > 0 means any non-zero outcome counts as a
-- conversion (stricter requires the new ETL to refine outcome_value semantics).
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS conversion_flag BOOLEAN
    GENERATED ALWAYS AS (outcome_value > 0) STORED;

-- roi_estimate: nullable, real ETL fills. Plain numeric column (not generated).
ALTER TABLE triggers ADD COLUMN IF NOT EXISTS roi_estimate NUMERIC;

-- hcp_brand_id: composite key surfacing for Feast (replaces the SQL
-- string-concat hack in feature_repo/data_sources.py). Must be added AFTER
-- brand_id because PostgreSQL requires referenced columns to already exist.
-- COALESCE on brand_id is defensive: brand_id is NOT NULL so COALESCE is a
-- no-op today, but mirrors the data_sources.py pattern and survives a future
-- schema relaxation.
ALTER TABLE triggers
    ADD COLUMN IF NOT EXISTS hcp_brand_id TEXT
    GENERATED ALWAYS AS (hcp_id || '_' || COALESCE(brand_id, 'UNKNOWN')) STORED;

-- -----------------------------------------------------------------------------
-- 033.3  patient_journeys: ETL columns + generated event_date / is_churned / brand aliases
-- -----------------------------------------------------------------------------
-- ETL-populated columns (nullable; 6B-infra-2b populates from real refill
-- adherence data). Generated counterparts come AFTER because event_date and
-- the brand aliases derive from journey_start_date / brand which already exist.
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS adherence_rate NUMERIC;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS refill_count    INTEGER;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS gap_days        INTEGER;

ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS event_date DATE
    GENERATED ALWAYS AS (DATE(journey_start_date)) STORED;

-- journey_status is journey_status_type (an enum). Cast to TEXT before IN
-- comparison so the generated expression matches migration 032's view shape
-- and avoids any future enum-equality surprises.
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS is_churned BOOLEAN
    GENERATED ALWAYS AS (journey_status::TEXT IN ('churned','discontinued')) STORED;

-- brand is brand_type (an enum); cast to TEXT for the generated alias so
-- Feast feature views consume a plain string column. Same applies to the
-- composite patient_brand_id below.
ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS brand_id TEXT
    GENERATED ALWAYS AS (brand::TEXT) STORED;

ALTER TABLE patient_journeys
    ADD COLUMN IF NOT EXISTS patient_brand_id TEXT
    GENERATED ALWAYS AS (patient_id || '_' || brand::TEXT) STORED;

-- -----------------------------------------------------------------------------
-- 033.4  business_metrics: per-HCP semantics + generated event_timestamp / hcp_brand_id
-- -----------------------------------------------------------------------------
-- hcp_id added as nullable FK so per-HCP rows (real ETL 6B-infra-2a output)
-- can coexist with the existing per-(brand, region) aggregate rows
-- (which keep hcp_id IS NULL). FK references hcp_profiles(hcp_id) so an
-- HCP delete cascades to a NULL hcp_id (default ON DELETE NO ACTION is fine
-- because we are not enforcing strict per-HCP integrity yet).
ALTER TABLE business_metrics
    ADD COLUMN IF NOT EXISTS hcp_id VARCHAR(20) REFERENCES hcp_profiles(hcp_id);

-- event_timestamp: cast metric_date (DATE) -> TIMESTAMPTZ for Feast, which
-- requires a TIMESTAMPTZ event-time column. Cast inside generated column is
-- IMMUTABLE.
ALTER TABLE business_metrics
    ADD COLUMN IF NOT EXISTS event_timestamp TIMESTAMPTZ
    GENERATED ALWAYS AS (metric_date::TIMESTAMPTZ) STORED;

-- hcp_brand_id: COALESCE(hcp_id, '_AGG') so existing aggregate rows (hcp_id
-- IS NULL) get a stable composite key '_AGG_<brand>' rather than NULL. brand
-- is brand_type enum -> cast to TEXT for the string concat.
ALTER TABLE business_metrics
    ADD COLUMN IF NOT EXISTS hcp_brand_id TEXT
    GENERATED ALWAYS AS (COALESCE(hcp_id, '_AGG') || '_' || brand::TEXT) STORED;

-- Per-HCP semantics columns (nullable for pre-existing aggregate rows; ETL
-- populates per-HCP rows with real values).
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS trx_count        INTEGER;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS nrx_count        INTEGER;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS total_rx_count   INTEGER;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS market_share     NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS conversion_rate  NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS engagement_score NUMERIC;
ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS call_frequency   NUMERIC;

-- -----------------------------------------------------------------------------
-- 033.5  territory_metrics: add generated event_timestamp (created by 031)
-- -----------------------------------------------------------------------------
-- territory_metrics was created in migration 031.A1.3 (lines 75-86). Wrap with
-- IF NOT EXISTS so the migration is safe to re-run.
ALTER TABLE territory_metrics
    ADD COLUMN IF NOT EXISTS event_timestamp TIMESTAMPTZ
    GENERATED ALWAYS AS (metric_date::TIMESTAMPTZ) STORED;

-- -----------------------------------------------------------------------------
-- 033.6  Drop bridging views + synthetic seed table
-- -----------------------------------------------------------------------------
-- All canonical replacements are now in place; data_sources.py rewrite
-- (6B-infra-3) will repoint Feast feature views to the canonical tables.
DROP VIEW  IF EXISTS feast_hcp_profile_source;
DROP VIEW  IF EXISTS feast_trigger_response_source;
DROP VIEW  IF EXISTS feast_patient_journey_source;
DROP VIEW  IF EXISTS feast_business_metrics_source;
DROP TABLE IF EXISTS feast_business_metrics_seed;
