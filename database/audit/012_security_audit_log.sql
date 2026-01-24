-- =============================================================================
-- Security Audit Log Table
-- E2I Causal Analytics - Phase 4 Security Hardening
--
-- Stores security-related events for compliance and threat detection:
-- - Authentication events (login, logout, token issues)
-- - Authorization events (access denied, privilege changes)
-- - Rate limiting events (threshold hits, blocks)
-- - API security events (invalid requests, suspicious activity)
-- - Data access events (sensitive data, exports)
-- =============================================================================

-- Drop existing table if recreating
-- DROP TABLE IF EXISTS security_audit_log;

-- =============================================================================
-- Main Security Audit Log Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS security_audit_log (
    -- Primary identification
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message TEXT NOT NULL,

    -- Actor information
    user_id VARCHAR(255),
    user_email VARCHAR(255),
    user_roles JSONB DEFAULT '[]'::jsonb,

    -- Request context
    request_id VARCHAR(100),
    correlation_id VARCHAR(100),
    client_ip INET,
    user_agent TEXT,
    endpoint VARCHAR(500),
    http_method VARCHAR(10),

    -- Resource context
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action_attempted VARCHAR(100),
    action_result VARCHAR(50),

    -- Error details
    error_code VARCHAR(50),
    error_details TEXT,

    -- Additional metadata (flexible JSON field)
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Record tracking
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes for Common Query Patterns
-- =============================================================================

-- Time-based queries (most common for recent events)
CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp
ON security_audit_log (timestamp DESC);

-- Event type filtering
CREATE INDEX IF NOT EXISTS idx_security_audit_event_type
ON security_audit_log (event_type);

-- Severity filtering (for alerts)
CREATE INDEX IF NOT EXISTS idx_security_audit_severity
ON security_audit_log (severity);

-- User-based queries (audit trails)
CREATE INDEX IF NOT EXISTS idx_security_audit_user_id
ON security_audit_log (user_id);

-- IP-based queries (threat detection)
CREATE INDEX IF NOT EXISTS idx_security_audit_client_ip
ON security_audit_log (client_ip);

-- Request correlation
CREATE INDEX IF NOT EXISTS idx_security_audit_request_id
ON security_audit_log (request_id);

-- Composite index for common filtered queries
CREATE INDEX IF NOT EXISTS idx_security_audit_type_severity_time
ON security_audit_log (event_type, severity, timestamp DESC);

-- JSONB index for metadata queries
CREATE INDEX IF NOT EXISTS idx_security_audit_metadata
ON security_audit_log USING gin (metadata);

-- =============================================================================
-- Partitioning Setup (for high-volume deployments)
-- =============================================================================
-- Note: For production with high event volume, consider partitioning by time:
--
-- CREATE TABLE security_audit_log_partitioned (
--     LIKE security_audit_log INCLUDING ALL
-- ) PARTITION BY RANGE (timestamp);
--
-- CREATE TABLE security_audit_log_y2026m01 PARTITION OF security_audit_log_partitioned
--     FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- Recent critical events (last 24 hours)
CREATE OR REPLACE VIEW v_security_critical_events AS
SELECT *
FROM security_audit_log
WHERE severity IN ('error', 'critical')
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Authentication failures summary
CREATE OR REPLACE VIEW v_auth_failures_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    client_ip,
    user_email,
    COUNT(*) as failure_count,
    MAX(timestamp) as last_attempt
FROM security_audit_log
WHERE event_type = 'auth.login.failure'
  AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp), client_ip, user_email
HAVING COUNT(*) > 3
ORDER BY hour DESC, failure_count DESC;

-- Rate limit violations summary
CREATE OR REPLACE VIEW v_rate_limit_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    client_ip,
    endpoint,
    COUNT(*) as violation_count,
    SUM(CASE WHEN event_type = 'rate_limit.blocked' THEN 1 ELSE 0 END) as blocks
FROM security_audit_log
WHERE event_type LIKE 'rate_limit.%'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp), client_ip, endpoint
ORDER BY hour DESC, violation_count DESC;

-- Suspicious activity by IP
CREATE OR REPLACE VIEW v_suspicious_ips AS
SELECT
    client_ip,
    COUNT(*) as total_events,
    COUNT(DISTINCT event_type) as event_types,
    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_events,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM security_audit_log
WHERE severity IN ('warning', 'error', 'critical')
  AND timestamp > NOW() - INTERVAL '24 hours'
  AND client_ip IS NOT NULL
GROUP BY client_ip
HAVING COUNT(*) > 10 OR COUNT(CASE WHEN severity = 'critical' THEN 1 END) > 0
ORDER BY critical_events DESC, total_events DESC;

-- Data access audit trail
CREATE OR REPLACE VIEW v_sensitive_data_access AS
SELECT
    timestamp,
    user_id,
    user_email,
    resource_type,
    resource_id,
    client_ip,
    metadata->>'data_classification' as data_classification
FROM security_audit_log
WHERE event_type LIKE 'data.%'
  AND timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

-- =============================================================================
-- RLS Policies (Enable Row Level Security)
-- =============================================================================

-- Enable RLS
ALTER TABLE security_audit_log ENABLE ROW LEVEL SECURITY;

-- Admin can read all
CREATE POLICY security_audit_admin_read ON security_audit_log
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM user_roles ur
            WHERE ur.user_id = auth.uid()::text
              AND ur.role IN ('admin', 'security_admin')
        )
    );

-- Service role can insert
CREATE POLICY security_audit_service_insert ON security_audit_log
    FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Regular users can see their own events
CREATE POLICY security_audit_user_own_read ON security_audit_log
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid()::text);

-- =============================================================================
-- Functions for Querying and Analysis
-- =============================================================================

-- Get event statistics for a time period
CREATE OR REPLACE FUNCTION get_security_event_stats(
    p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    p_end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    event_type VARCHAR,
    severity VARCHAR,
    event_count BIGINT,
    unique_users BIGINT,
    unique_ips BIGINT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        sal.event_type,
        sal.severity,
        COUNT(*)::BIGINT as event_count,
        COUNT(DISTINCT sal.user_id)::BIGINT as unique_users,
        COUNT(DISTINCT sal.client_ip)::BIGINT as unique_ips
    FROM security_audit_log sal
    WHERE sal.timestamp BETWEEN p_start_time AND p_end_time
    GROUP BY sal.event_type, sal.severity
    ORDER BY event_count DESC;
END;
$$;

-- Check if IP should be blocked (brute force detection)
CREATE OR REPLACE FUNCTION check_ip_should_block(
    p_client_ip INET,
    p_failure_threshold INT DEFAULT 10,
    p_window_minutes INT DEFAULT 15
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_failure_count INT;
BEGIN
    SELECT COUNT(*)
    INTO v_failure_count
    FROM security_audit_log
    WHERE client_ip = p_client_ip
      AND event_type = 'auth.login.failure'
      AND timestamp > NOW() - (p_window_minutes || ' minutes')::INTERVAL;

    RETURN v_failure_count >= p_failure_threshold;
END;
$$;

-- Get recent events for a user (compliance audit)
CREATE OR REPLACE FUNCTION get_user_security_audit(
    p_user_id VARCHAR,
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    event_id UUID,
    event_type VARCHAR,
    severity VARCHAR,
    timestamp TIMESTAMPTZ,
    message TEXT,
    client_ip INET,
    endpoint VARCHAR,
    action_result VARCHAR
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        sal.event_id,
        sal.event_type,
        sal.severity,
        sal.timestamp,
        sal.message,
        sal.client_ip,
        sal.endpoint,
        sal.action_result
    FROM security_audit_log sal
    WHERE sal.user_id = p_user_id
      AND sal.timestamp > NOW() - (p_days || ' days')::INTERVAL
    ORDER BY sal.timestamp DESC
    LIMIT 1000;
END;
$$;

-- =============================================================================
-- Cleanup and Retention
-- =============================================================================

-- Function to purge old audit logs (call via scheduled job)
CREATE OR REPLACE FUNCTION purge_old_security_logs(
    p_retention_days INT DEFAULT 90
)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_deleted INT;
BEGIN
    DELETE FROM security_audit_log
    WHERE timestamp < NOW() - (p_retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS v_deleted = ROW_COUNT;

    -- Log the purge operation itself
    INSERT INTO security_audit_log (
        event_type, severity, message, metadata
    ) VALUES (
        'admin.system.access',
        'info',
        'Security audit log purged',
        jsonb_build_object(
            'retention_days', p_retention_days,
            'records_deleted', v_deleted
        )
    );

    RETURN v_deleted;
END;
$$;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE security_audit_log IS 'Centralized security audit log for compliance and threat detection';
COMMENT ON COLUMN security_audit_log.event_type IS 'Categorized event type (auth.login.success, rate_limit.exceeded, etc.)';
COMMENT ON COLUMN security_audit_log.severity IS 'Event severity: debug, info, warning, error, critical';
COMMENT ON COLUMN security_audit_log.metadata IS 'Flexible JSON field for event-specific data';
COMMENT ON VIEW v_security_critical_events IS 'Quick access to critical security events in last 24 hours';
COMMENT ON VIEW v_auth_failures_summary IS 'Aggregated view of authentication failures for brute force detection';
COMMENT ON FUNCTION check_ip_should_block IS 'Check if an IP should be blocked based on recent auth failures';
