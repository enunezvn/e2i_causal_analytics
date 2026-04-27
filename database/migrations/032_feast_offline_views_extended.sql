-- =============================================================================
-- Migration 032: Extended Feast offline-store bridging views
-- =============================================================================
-- Bridges patient_journeys + (synthesizes from) business_metrics so the
-- remaining 6 FVs can materialize. Per-HCP business metrics are SYNTHETIC;
-- real ETL is Block 6B.
--
-- Three artifacts:
--   1. feast_patient_journey_source     view over patient_journeys
--   2. feast_business_metrics_seed       per-HCP synthetic table
--   3. feast_business_metrics_source     view over the seed
--
-- Safety: views are CREATE OR REPLACE; seed table uses IF NOT EXISTS; the
-- INSERT is ON CONFLICT DO NOTHING — the migration is idempotent.
-- =============================================================================

-- 032.1 — feast_patient_journey_source view
CREATE OR REPLACE VIEW feast_patient_journey_source AS
SELECT
    patient_journey_id::VARCHAR              AS journey_id,
    patient_id::VARCHAR,
    brand::VARCHAR                           AS brand_id,
    journey_start_date::TIMESTAMPTZ          AS event_date,
    journey_start_date::TIMESTAMPTZ          AS therapy_start_date,
    COALESCE(journey_duration_days, 0)::INTEGER AS days_on_therapy,
    NULL::REAL                               AS adherence_rate,
    NULL::INTEGER                            AS refill_count,
    NULL::INTEGER                            AS gap_days,
    (journey_status::TEXT IN ('churned','discontinued'))::BOOLEAN AS is_churned,
    COALESCE(risk_score, 0)::REAL            AS churn_risk_score,
    created_at
FROM patient_journeys;

-- 032.2 — feast_business_metrics_seed table (per-HCP synthetic)
CREATE TABLE IF NOT EXISTS feast_business_metrics_seed (
    hcp_id            VARCHAR(20)  NOT NULL,
    territory_id      VARCHAR(20),
    brand_id          VARCHAR(50)  NOT NULL,
    event_timestamp   TIMESTAMPTZ  NOT NULL,
    trx_count         INTEGER      NOT NULL DEFAULT 0,
    nrx_count         INTEGER      NOT NULL DEFAULT 0,
    total_rx_count    INTEGER      NOT NULL DEFAULT 0,
    market_share      REAL         NOT NULL DEFAULT 0,
    conversion_rate   REAL         NOT NULL DEFAULT 0,
    engagement_score  REAL         NOT NULL DEFAULT 0,
    call_frequency    REAL         NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (hcp_id, brand_id, event_timestamp)
);
CREATE INDEX IF NOT EXISTS idx_feast_bms_event ON feast_business_metrics_seed(event_timestamp);

-- 032.3 — Initial seed (one row per (hcp, brand) at NOW())
INSERT INTO feast_business_metrics_seed
    (hcp_id, territory_id, brand_id, event_timestamp,
     trx_count, nrx_count, total_rx_count,
     market_share, conversion_rate, engagement_score, call_frequency)
SELECT
    hp.hcp_id,
    hp.territory_id,
    b.brand::TEXT                AS brand_id,
    NOW()                         AS event_timestamp,
    (random()*100)::INTEGER       AS trx_count,
    (random()*30)::INTEGER        AS nrx_count,
    (random()*130)::INTEGER       AS total_rx_count,
    random()::REAL                AS market_share,
    random()::REAL                AS conversion_rate,
    (random()*100)::REAL          AS engagement_score,
    (random()*10)::REAL           AS call_frequency
FROM hcp_profiles hp
CROSS JOIN (
    SELECT DISTINCT brand::TEXT AS brand FROM business_metrics LIMIT 5
) b
ON CONFLICT (hcp_id, brand_id, event_timestamp) DO NOTHING;

-- 032.4 — feast_business_metrics_source view
CREATE OR REPLACE VIEW feast_business_metrics_source AS
SELECT * FROM feast_business_metrics_seed;
