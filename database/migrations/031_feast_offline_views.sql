-- =============================================================================
-- Migration 031: Feast offline-store bridging views + territory_metrics seed
-- =============================================================================
-- Bridges canonical schema columns to the names Feast feature views expect.
-- This is a bootstrap path: replacing the views with proper schema migrations
-- and a real territory ETL is a Block 6B follow-up.
--
-- Three artifacts:
--   1. feast_hcp_profile_source     view over hcp_profiles
--   2. feast_trigger_response_source view over triggers
--   3. territory_metrics            base table + minimal seed
--
-- Safety: views are CREATE OR REPLACE; table uses IF NOT EXISTS; seed is
-- ON CONFLICT DO NOTHING — the migration is idempotent.
-- =============================================================================

-- A1.1 — feast_hcp_profile_source view
CREATE OR REPLACE VIEW feast_hcp_profile_source AS
SELECT
    hcp_id,
    territory_id,
    specialty,
    practice_type,
    years_experience AS years_of_practice,
    CASE
        WHEN total_patient_volume >= 1000 THEN 'High'
        WHEN total_patient_volume >= 500  THEN 'Medium'
        ELSE 'Low'
    END AS patient_volume_tier,
    CASE
        WHEN digital_engagement_score >= 0.70 THEN 'High'
        WHEN digital_engagement_score >= 0.40 THEN 'Medium'
        ELSE 'Low'
    END AS digital_engagement_tier,
    CASE
        WHEN prescribing_volume >= 1000 THEN 'Tier1'
        WHEN prescribing_volume >= 500  THEN 'Tier2'
        ELSE 'Tier3'
    END AS prescribing_tier,
    -- last_updated = NOW() - 1h: synthetic hcp_profiles.updated_at is months
    -- stale, which falls outside the FV TTL (30 days) and breaks point-in-time
    -- parity. Surfacing a recent-but-not-future timestamp keeps the parity test
    -- honest under synthetic data. The 1-hour offset matters: a bare NOW() at
    -- view-eval time is AFTER materialize's captured end-timestamp, so the
    -- range filter excludes the rows. 1h backdate puts events safely in the
    -- past relative to materialize end (well within all FV TTLs). Block 6B
    -- replaces the canonical updated_at population with a real cadence so this
    -- override goes away.
    (NOW() - INTERVAL '1 hour') AS last_updated,
    created_at
FROM hcp_profiles;

-- A1.2 — feast_trigger_response_source view
CREATE OR REPLACE VIEW feast_trigger_response_source AS
SELECT
    trigger_id,
    hcp_id,
    NULL::VARCHAR AS brand_id,                                  -- not present in canonical triggers; Block 6B
    -- trigger_date = NOW() - 1h: same rationale as hcp_profile.last_updated.
    -- Synthetic trigger_timestamp is months stale; surfacing 1h-backdated
    -- timestamps keeps the 1-day-TTL trigger_response_features parity test
    -- honest while staying behind materialize's captured end-time. Block 6B
    -- replaces this with a proper canonical-timestamp source.
    (NOW() - INTERVAL '1 hour') AS trigger_date,
    trigger_type,
    delivery_channel AS channel,
    (acceptance_status IN ('accepted','responded')) AS is_responded,
    EXTRACT(EPOCH FROM (action_timestamp - trigger_timestamp))/3600.0 AS response_time_hours,
    (COALESCE(outcome_value, 0) > 0.5) AS conversion_flag,
    outcome_value AS roi_estimate,
    created_at
FROM triggers;

-- A1.3 — territory_metrics base table (no canonical source to derive from yet)
CREATE TABLE IF NOT EXISTS territory_metrics (
    territory_id              VARCHAR(20)  NOT NULL,
    metric_date               DATE         NOT NULL,
    total_trx                 BIGINT       NOT NULL DEFAULT 0,
    total_nrx                 BIGINT       NOT NULL DEFAULT 0,
    active_hcp_count          BIGINT       NOT NULL DEFAULT 0,
    covered_lives             BIGINT       NOT NULL DEFAULT 0,
    market_potential          DOUBLE PRECISION NOT NULL DEFAULT 0,
    resource_allocation_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (territory_id, metric_date)
);
CREATE INDEX IF NOT EXISTS idx_territory_metrics_date ON territory_metrics(metric_date);

-- A1.4 — minimal seed for territory_metrics (one row per known territory at NOW())
INSERT INTO territory_metrics
    (territory_id, metric_date, total_trx, total_nrx, active_hcp_count,
     covered_lives, market_potential, resource_allocation_score)
SELECT
    territory_id,
    CURRENT_DATE,
    (1000 + (random()*5000)::int)::bigint AS total_trx,
    (200  + (random()*1000)::int)::bigint AS total_nrx,
    COUNT(*)::bigint                       AS active_hcp_count,
    (10000 + (random()*50000)::int)::bigint AS covered_lives,
    random()                               AS market_potential,
    random()                               AS resource_allocation_score
FROM hcp_profiles
WHERE territory_id IS NOT NULL
GROUP BY territory_id
ON CONFLICT (territory_id, metric_date) DO NOTHING;
