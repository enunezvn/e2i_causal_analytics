-- ============================================================================
-- Migration 024: BentoML Service Tracking Tables
-- ============================================================================
-- Purpose: Add BentoML-specific tables for service health and serving metrics
--
-- Tables:
--   1. ml_bentoml_services - Track BentoML service deployments
--   2. ml_bentoml_serving_metrics - Time-series serving performance metrics
--
-- Author: E2I Causal Analytics
-- Date: 2025-12-29
-- ============================================================================

BEGIN;

-- ============================================================================
-- TABLE 1: ml_bentoml_services
-- Track BentoML service deployments with health monitoring
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_bentoml_services (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Service identification
    service_name VARCHAR(255) NOT NULL,
    bento_tag VARCHAR(255) NOT NULL,
    bento_version VARCHAR(100),

    -- Link to model registry and deployment
    model_registry_id UUID REFERENCES ml_model_registry(id),
    deployment_id UUID REFERENCES ml_deployments(id),

    -- Container info
    container_image TEXT,
    container_tag VARCHAR(255),

    -- Service configuration
    replicas INTEGER DEFAULT 1,
    resources JSONB DEFAULT '{"cpu": "1", "memory": "2Gi"}',
    environment VARCHAR(50) DEFAULT 'staging',

    -- Health tracking
    health_status VARCHAR(50) DEFAULT 'unknown',
    -- Values: 'healthy', 'unhealthy', 'degraded', 'unknown', 'starting', 'stopped'
    last_health_check TIMESTAMPTZ,
    health_check_failures INTEGER DEFAULT 0,

    -- Endpoint info
    serving_endpoint TEXT,
    internal_endpoint TEXT,

    -- Lifecycle
    status deployment_status_enum DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    created_by VARCHAR(100),
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}'
);

-- Indexes for ml_bentoml_services
CREATE INDEX idx_bentoml_services_name ON ml_bentoml_services(service_name);
CREATE INDEX idx_bentoml_services_tag ON ml_bentoml_services(bento_tag);
CREATE INDEX idx_bentoml_services_model ON ml_bentoml_services(model_registry_id);
CREATE INDEX idx_bentoml_services_deployment ON ml_bentoml_services(deployment_id);
CREATE INDEX idx_bentoml_services_status ON ml_bentoml_services(status);
CREATE INDEX idx_bentoml_services_health ON ml_bentoml_services(health_status);
CREATE INDEX idx_bentoml_services_env ON ml_bentoml_services(environment);
CREATE INDEX idx_bentoml_services_active ON ml_bentoml_services(status)
    WHERE status = 'active';

-- ============================================================================
-- TABLE 2: ml_bentoml_serving_metrics
-- Time-series serving performance metrics for BentoML services
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_bentoml_serving_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign key to service
    service_id UUID NOT NULL REFERENCES ml_bentoml_services(id) ON DELETE CASCADE,

    -- Timestamp (for time-series queries)
    recorded_at TIMESTAMPTZ DEFAULT NOW(),

    -- Throughput metrics
    requests_total INTEGER DEFAULT 0,
    requests_per_second FLOAT,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,

    -- Latency metrics (in milliseconds)
    avg_latency_ms FLOAT,
    p50_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    max_latency_ms FLOAT,

    -- Error metrics
    error_rate FLOAT,  -- Percentage (0.0 - 100.0)
    error_types JSONB DEFAULT '{}',  -- {"ValidationError": 5, "TimeoutError": 2}

    -- Resource utilization
    memory_mb FLOAT,
    memory_percent FLOAT,
    cpu_percent FLOAT,

    -- Prediction metrics
    predictions_count INTEGER DEFAULT 0,
    batch_size_avg FLOAT,

    -- Model-specific metrics
    model_load_time_ms FLOAT,
    inference_time_avg_ms FLOAT
);

-- Indexes for ml_bentoml_serving_metrics
CREATE INDEX idx_bentoml_metrics_service ON ml_bentoml_serving_metrics(service_id);
CREATE INDEX idx_bentoml_metrics_time ON ml_bentoml_serving_metrics(recorded_at DESC);
CREATE INDEX idx_bentoml_metrics_service_time ON ml_bentoml_serving_metrics(service_id, recorded_at DESC);

-- Partial index for recent metrics (last 24 hours)
CREATE INDEX idx_bentoml_metrics_recent ON ml_bentoml_serving_metrics(service_id, recorded_at)
    WHERE recorded_at > NOW() - INTERVAL '24 hours';

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to get latest metrics for a service
CREATE OR REPLACE FUNCTION get_latest_bentoml_metrics(p_service_id UUID)
RETURNS TABLE (
    service_id UUID,
    recorded_at TIMESTAMPTZ,
    requests_per_second FLOAT,
    avg_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    error_rate FLOAT,
    memory_mb FLOAT,
    cpu_percent FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.service_id,
        m.recorded_at,
        m.requests_per_second,
        m.avg_latency_ms,
        m.p95_latency_ms,
        m.error_rate,
        m.memory_mb,
        m.cpu_percent
    FROM ml_bentoml_serving_metrics m
    WHERE m.service_id = p_service_id
    ORDER BY m.recorded_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get aggregated metrics over a time period
CREATE OR REPLACE FUNCTION get_bentoml_metrics_summary(
    p_service_id UUID,
    p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '1 hour',
    p_end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    total_requests INTEGER,
    avg_rps FLOAT,
    avg_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    avg_error_rate FLOAT,
    avg_memory_mb FLOAT,
    avg_cpu_percent FLOAT,
    data_points INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(SUM(m.requests_total), 0)::INTEGER as total_requests,
        AVG(m.requests_per_second) as avg_rps,
        AVG(m.avg_latency_ms) as avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY m.p95_latency_ms) as p95_latency_ms,
        AVG(m.error_rate) as avg_error_rate,
        AVG(m.memory_mb) as avg_memory_mb,
        AVG(m.cpu_percent) as avg_cpu_percent,
        COUNT(*)::INTEGER as data_points
    FROM ml_bentoml_serving_metrics m
    WHERE m.service_id = p_service_id
    AND m.recorded_at BETWEEN p_start_time AND p_end_time;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to update service health based on recent metrics
CREATE OR REPLACE FUNCTION update_bentoml_service_health(p_service_id UUID)
RETURNS VARCHAR(50) AS $$
DECLARE
    v_error_rate FLOAT;
    v_avg_latency FLOAT;
    v_new_health VARCHAR(50);
BEGIN
    -- Get average metrics from last 5 minutes
    SELECT
        AVG(error_rate),
        AVG(avg_latency_ms)
    INTO v_error_rate, v_avg_latency
    FROM ml_bentoml_serving_metrics
    WHERE service_id = p_service_id
    AND recorded_at > NOW() - INTERVAL '5 minutes';

    -- Determine health status based on thresholds
    IF v_error_rate IS NULL THEN
        v_new_health := 'unknown';
    ELSIF v_error_rate > 10.0 THEN
        v_new_health := 'unhealthy';
    ELSIF v_error_rate > 5.0 OR v_avg_latency > 1000 THEN
        v_new_health := 'degraded';
    ELSE
        v_new_health := 'healthy';
    END IF;

    -- Update service health
    UPDATE ml_bentoml_services
    SET
        health_status = v_new_health,
        last_health_check = NOW(),
        updated_at = NOW()
    WHERE id = p_service_id;

    RETURN v_new_health;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Active BentoML services with latest metrics
CREATE OR REPLACE VIEW v_bentoml_services_health AS
SELECT
    s.id,
    s.service_name,
    s.bento_tag,
    s.environment,
    s.status,
    s.health_status,
    s.replicas,
    s.serving_endpoint,
    s.last_health_check,
    m.requests_per_second,
    m.avg_latency_ms,
    m.p95_latency_ms,
    m.error_rate,
    m.memory_mb,
    m.cpu_percent,
    m.recorded_at as metrics_recorded_at
FROM ml_bentoml_services s
LEFT JOIN LATERAL (
    SELECT *
    FROM ml_bentoml_serving_metrics
    WHERE service_id = s.id
    ORDER BY recorded_at DESC
    LIMIT 1
) m ON true
WHERE s.status = 'active';

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE ml_bentoml_services IS 'BentoML service deployments with health monitoring';
COMMENT ON TABLE ml_bentoml_serving_metrics IS 'Time-series serving performance metrics for BentoML services';
COMMENT ON FUNCTION get_latest_bentoml_metrics IS 'Get the most recent metrics for a BentoML service';
COMMENT ON FUNCTION get_bentoml_metrics_summary IS 'Get aggregated metrics over a time period';
COMMENT ON FUNCTION update_bentoml_service_health IS 'Update service health based on recent metrics';
COMMENT ON VIEW v_bentoml_services_health IS 'Active BentoML services with their latest metrics';

-- ============================================================================
-- GRANTS
-- ============================================================================
-- Uncomment and adjust based on your roles:
-- GRANT SELECT, INSERT, UPDATE ON ml_bentoml_services TO e2i_app;
-- GRANT SELECT, INSERT ON ml_bentoml_serving_metrics TO e2i_app;
-- GRANT EXECUTE ON FUNCTION get_latest_bentoml_metrics TO e2i_app;
-- GRANT EXECUTE ON FUNCTION get_bentoml_metrics_summary TO e2i_app;
-- GRANT EXECUTE ON FUNCTION update_bentoml_service_health TO e2i_app;
-- GRANT SELECT ON v_bentoml_services_health TO e2i_app;

COMMIT;
