-- ============================================================================
-- Migration 025: Feast Feature Store Tracking Tables
-- ============================================================================
-- Purpose: Add Feast-specific tables for feature view and materialization tracking
--
-- Tables:
--   1. ml_feast_feature_views - Track Feast feature view configurations
--   2. ml_feast_materialization_jobs - Track materialization runs
--   3. ml_feast_feature_freshness - Track feature freshness per view
--
-- Author: E2I Causal Analytics
-- Date: 2025-12-29
-- ============================================================================

BEGIN;

-- ============================================================================
-- TABLE 1: ml_feast_feature_views
-- Track Feast feature view configurations and metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_feast_feature_views (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Feature view identification
    name VARCHAR(255) NOT NULL,
    project VARCHAR(100) DEFAULT 'e2i_causal_analytics',
    description TEXT,

    -- Entity configuration
    entities JSONB DEFAULT '[]',  -- List of entity names
    entity_join_keys JSONB DEFAULT '[]',  -- Join keys for entities

    -- Feature definitions
    features JSONB DEFAULT '[]',  -- List of feature names and types
    feature_count INTEGER DEFAULT 0,

    -- Source configuration
    source_type VARCHAR(50),  -- 'batch', 'stream', 'request'
    source_name VARCHAR(255),
    source_config JSONB DEFAULT '{}',

    -- TTL and online store settings
    ttl_seconds INTEGER,
    online_enabled BOOLEAN DEFAULT true,
    batch_source_enabled BOOLEAN DEFAULT true,

    -- Tags and metadata
    tags JSONB DEFAULT '{}',
    owner VARCHAR(100),

    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,  -- Soft delete

    -- Unique constraint on name + project
    UNIQUE (name, project)
);

-- Indexes for ml_feast_feature_views
CREATE INDEX idx_feast_views_name ON ml_feast_feature_views(name);
CREATE INDEX idx_feast_views_project ON ml_feast_feature_views(project);
CREATE INDEX idx_feast_views_source ON ml_feast_feature_views(source_type);
CREATE INDEX idx_feast_views_active ON ml_feast_feature_views(deleted_at)
    WHERE deleted_at IS NULL;

-- ============================================================================
-- TABLE 2: ml_feast_materialization_jobs
-- Track materialization job runs and their metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_feast_materialization_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to feature view
    feature_view_id UUID REFERENCES ml_feast_feature_views(id) ON DELETE CASCADE,
    feature_view_name VARCHAR(255) NOT NULL,  -- Denormalized for queries

    -- Job identification
    job_id VARCHAR(255),  -- External job ID if applicable
    job_type VARCHAR(50) DEFAULT 'incremental',  -- 'full', 'incremental'

    -- Time range materialized
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,

    -- Execution status
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'running', 'success', 'failed'
    error_message TEXT,

    -- Metrics
    rows_materialized INTEGER DEFAULT 0,
    bytes_written BIGINT,
    duration_seconds FLOAT,

    -- Online store metrics
    online_store_rows_written INTEGER,
    online_store_latency_ms FLOAT,

    -- Resource usage
    cpu_seconds FLOAT,
    memory_peak_mb FLOAT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Indexes for ml_feast_materialization_jobs
CREATE INDEX idx_feast_jobs_view ON ml_feast_materialization_jobs(feature_view_id);
CREATE INDEX idx_feast_jobs_view_name ON ml_feast_materialization_jobs(feature_view_name);
CREATE INDEX idx_feast_jobs_status ON ml_feast_materialization_jobs(status);
CREATE INDEX idx_feast_jobs_created ON ml_feast_materialization_jobs(created_at DESC);
CREATE INDEX idx_feast_jobs_time_range ON ml_feast_materialization_jobs(start_time, end_time);

-- Note: Partial index with NOW() removed - PostgreSQL requires IMMUTABLE functions in index predicates
-- The composite index idx_feast_jobs_created provides efficient queries for recent jobs
-- Alternative: Use application-level filtering or scheduled index rebuilds if needed

-- ============================================================================
-- TABLE 3: ml_feast_feature_freshness
-- Track feature freshness metrics per feature view
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_feast_feature_freshness (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to feature view
    feature_view_id UUID REFERENCES ml_feast_feature_views(id) ON DELETE CASCADE,
    feature_view_name VARCHAR(255) NOT NULL,

    -- Freshness timestamp
    recorded_at TIMESTAMPTZ DEFAULT NOW(),

    -- Freshness metrics
    last_materialization_time TIMESTAMPTZ,
    staleness_seconds INTEGER,  -- Time since last materialization
    data_lag_seconds INTEGER,  -- Lag between source and online store

    -- Feature statistics
    null_rate FLOAT,  -- Percentage of null values
    unique_count INTEGER,  -- Count of unique values (for low-cardinality features)

    -- Health status
    freshness_status VARCHAR(50) DEFAULT 'unknown',  -- 'fresh', 'stale', 'critical', 'unknown'

    -- Thresholds used
    staleness_threshold_seconds INTEGER,
    critical_threshold_seconds INTEGER
);

-- Indexes for ml_feast_feature_freshness
CREATE INDEX idx_feast_freshness_view ON ml_feast_feature_freshness(feature_view_id);
CREATE INDEX idx_feast_freshness_view_name ON ml_feast_feature_freshness(feature_view_name);
CREATE INDEX idx_feast_freshness_time ON ml_feast_feature_freshness(recorded_at DESC);
CREATE INDEX idx_feast_freshness_status ON ml_feast_feature_freshness(freshness_status);

-- Note: Partial index with NOW() removed - PostgreSQL requires IMMUTABLE functions in index predicates
-- The composite index (feature_view_id, recorded_at) via idx_feast_freshness_time provides efficient queries
-- Alternative: Use application-level filtering or scheduled index rebuilds if needed

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to get latest freshness for a feature view
CREATE OR REPLACE FUNCTION get_feast_feature_freshness(p_feature_view_id UUID)
RETURNS TABLE (
    feature_view_id UUID,
    feature_view_name VARCHAR(255),
    recorded_at TIMESTAMPTZ,
    staleness_seconds INTEGER,
    freshness_status VARCHAR(50),
    last_materialization_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.feature_view_id,
        f.feature_view_name,
        f.recorded_at,
        f.staleness_seconds,
        f.freshness_status,
        f.last_materialization_time
    FROM ml_feast_feature_freshness f
    WHERE f.feature_view_id = p_feature_view_id
    ORDER BY f.recorded_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get materialization summary for a feature view
CREATE OR REPLACE FUNCTION get_feast_materialization_summary(
    p_feature_view_id UUID,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    total_jobs INTEGER,
    successful_jobs INTEGER,
    failed_jobs INTEGER,
    total_rows_materialized BIGINT,
    avg_duration_seconds FLOAT,
    success_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_jobs,
        COUNT(*) FILTER (WHERE m.status = 'success')::INTEGER as successful_jobs,
        COUNT(*) FILTER (WHERE m.status = 'failed')::INTEGER as failed_jobs,
        COALESCE(SUM(m.rows_materialized), 0)::BIGINT as total_rows_materialized,
        AVG(m.duration_seconds) as avg_duration_seconds,
        (COUNT(*) FILTER (WHERE m.status = 'success')::FLOAT / NULLIF(COUNT(*), 0) * 100) as success_rate
    FROM ml_feast_materialization_jobs m
    WHERE m.feature_view_id = p_feature_view_id
    AND m.created_at > NOW() - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to update feature view freshness based on latest materialization
CREATE OR REPLACE FUNCTION update_feast_freshness(p_feature_view_id UUID)
RETURNS VARCHAR(50) AS $$
DECLARE
    v_last_mat_time TIMESTAMPTZ;
    v_staleness INTEGER;
    v_status VARCHAR(50);
    v_view_name VARCHAR(255);
    v_ttl INTEGER;
BEGIN
    -- Get feature view info
    SELECT name, ttl_seconds INTO v_view_name, v_ttl
    FROM ml_feast_feature_views
    WHERE id = p_feature_view_id;

    -- Get latest successful materialization
    SELECT completed_at INTO v_last_mat_time
    FROM ml_feast_materialization_jobs
    WHERE feature_view_id = p_feature_view_id
    AND status = 'success'
    ORDER BY completed_at DESC
    LIMIT 1;

    -- Calculate staleness
    IF v_last_mat_time IS NULL THEN
        v_staleness := NULL;
        v_status := 'unknown';
    ELSE
        v_staleness := EXTRACT(EPOCH FROM (NOW() - v_last_mat_time))::INTEGER;

        -- Determine status based on TTL (if set)
        IF v_ttl IS NOT NULL THEN
            IF v_staleness > v_ttl * 2 THEN
                v_status := 'critical';
            ELSIF v_staleness > v_ttl THEN
                v_status := 'stale';
            ELSE
                v_status := 'fresh';
            END IF;
        ELSE
            -- Default thresholds: 1 hour = fresh, 6 hours = stale, 24 hours = critical
            IF v_staleness > 86400 THEN
                v_status := 'critical';
            ELSIF v_staleness > 21600 THEN
                v_status := 'stale';
            ELSE
                v_status := 'fresh';
            END IF;
        END IF;
    END IF;

    -- Insert freshness record
    INSERT INTO ml_feast_feature_freshness (
        feature_view_id,
        feature_view_name,
        last_materialization_time,
        staleness_seconds,
        freshness_status,
        staleness_threshold_seconds,
        critical_threshold_seconds
    ) VALUES (
        p_feature_view_id,
        v_view_name,
        v_last_mat_time,
        v_staleness,
        v_status,
        COALESCE(v_ttl, 3600),
        COALESCE(v_ttl * 2, 86400)
    );

    RETURN v_status;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Feature views with latest freshness status
CREATE OR REPLACE VIEW v_feast_feature_views_status AS
SELECT
    fv.id,
    fv.name,
    fv.project,
    fv.feature_count,
    fv.source_type,
    fv.online_enabled,
    fv.ttl_seconds,
    fv.created_at,
    ff.freshness_status,
    ff.staleness_seconds,
    ff.last_materialization_time,
    ff.recorded_at as freshness_checked_at
FROM ml_feast_feature_views fv
LEFT JOIN LATERAL (
    SELECT *
    FROM ml_feast_feature_freshness
    WHERE feature_view_id = fv.id
    ORDER BY recorded_at DESC
    LIMIT 1
) ff ON true
WHERE fv.deleted_at IS NULL;

-- View: Recent materialization jobs with feature view info
CREATE OR REPLACE VIEW v_feast_recent_materializations AS
SELECT
    mj.id,
    mj.feature_view_name,
    mj.job_type,
    mj.status,
    mj.start_time,
    mj.end_time,
    mj.rows_materialized,
    mj.duration_seconds,
    mj.created_at,
    fv.source_type,
    fv.online_enabled
FROM ml_feast_materialization_jobs mj
JOIN ml_feast_feature_views fv ON mj.feature_view_id = fv.id
WHERE mj.created_at > NOW() - INTERVAL '7 days'
ORDER BY mj.created_at DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE ml_feast_feature_views IS 'Feast feature view configurations and metadata';
COMMENT ON TABLE ml_feast_materialization_jobs IS 'Feast materialization job runs and metrics';
COMMENT ON TABLE ml_feast_feature_freshness IS 'Feature freshness tracking per view';
COMMENT ON FUNCTION get_feast_feature_freshness IS 'Get the most recent freshness record for a feature view';
COMMENT ON FUNCTION get_feast_materialization_summary IS 'Get aggregated materialization stats for a feature view';
COMMENT ON FUNCTION update_feast_freshness IS 'Update freshness status based on latest materialization';
COMMENT ON VIEW v_feast_feature_views_status IS 'Feature views with their current freshness status';
COMMENT ON VIEW v_feast_recent_materializations IS 'Recent materialization jobs from last 7 days';

-- ============================================================================
-- GRANTS
-- ============================================================================
-- Uncomment and adjust based on your roles:
-- GRANT SELECT, INSERT, UPDATE ON ml_feast_feature_views TO e2i_app;
-- GRANT SELECT, INSERT ON ml_feast_materialization_jobs TO e2i_app;
-- GRANT SELECT, INSERT ON ml_feast_feature_freshness TO e2i_app;
-- GRANT EXECUTE ON FUNCTION get_feast_feature_freshness TO e2i_app;
-- GRANT EXECUTE ON FUNCTION get_feast_materialization_summary TO e2i_app;
-- GRANT EXECUTE ON FUNCTION update_feast_freshness TO e2i_app;
-- GRANT SELECT ON v_feast_feature_views_status TO e2i_app;
-- GRANT SELECT ON v_feast_recent_materializations TO e2i_app;

COMMIT;
