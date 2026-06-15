-- Migration 031: expose is_synthetic on ml_model_health_dashboard
-- =============================================================================
-- WS-BACKEND (codex round-2 RESOLUTION-4). The model-health denominator must
-- exclude synthetic experiment artifacts STRUCTURALLY, not rely on the mutable
-- "synthetic rows happen to lack a ml_performance_metrics row" assumption.
--
-- ml_model_registry already carries is_synthetic (migration 069). The dashboard
-- view dropped it, so src/api/routes/health_score._fetch_model_health could not
-- filter on provenance and instead leaned on latest_metric_value IS NOT NULL.
-- Live registry snapshot (2026-06-15): stage IN (production, staging) holds
--   (is_synthetic=False) 2 production + 12 staging  -> the 14 REAL models
--   (is_synthetic=True)  360 production + 360 staging -> metric-less artifacts
-- All 12 gold-standard models are is_synthetic=False, so an is_synthetic=False
-- guard surfaces exactly the real models and hides nothing the platform wants.
--
-- CREATE OR REPLACE VIEW appends is_synthetic as the LAST column (Postgres only
-- permits adding columns at the end). Additive + idempotent: existing readers
-- select an explicit column list that does not name is_synthetic, so they are
-- unaffected; re-running during the batched deploy is harmless.
-- =============================================================================

CREATE OR REPLACE VIEW ml_model_health_dashboard AS
SELECT
    m.id as model_id,
    m.model_name as model_name,  -- prod ml_model_registry column is model_name, not name
    m.stage as model_stage,
    -- Latest drift status
    COALESCE(d.has_drift, FALSE) as has_active_drift,
    d.max_drift_severity,
    d.drift_check_count,
    -- Active alerts
    COALESCE(a.total_active, 0) as active_alerts,
    COALESCE(a.critical_count, 0) as critical_alerts,
    -- Latest performance
    p.latest_metric_value,
    p.metric_name as primary_metric,
    p.is_degraded as performance_degraded,
    -- Overall health
    CASE
        WHEN a.critical_count > 0 THEN 'critical'
        WHEN a.high_count > 0 OR d.max_drift_severity = 'critical' THEN 'warning'
        WHEN d.has_drift OR p.is_degraded THEN 'attention'
        ELSE 'healthy'
    END as health_status,
    -- Provenance (migration 031): appended LAST so CREATE OR REPLACE VIEW accepts
    -- the column add. Lets the health route exclude synthetic experiment
    -- artifacts structurally instead of relying on a null-metric coincidence.
    m.is_synthetic as is_synthetic
FROM ml_model_registry m
LEFT JOIN (
    SELECT
        model_id,
        bool_or(drift_detected) as has_drift,
        MAX(severity::text) as max_drift_severity,
        COUNT(*) as drift_check_count
    FROM ml_drift_history
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY model_id
) d ON m.id = d.model_id
LEFT JOIN ml_active_alerts_summary a ON m.id = a.model_id
LEFT JOIN LATERAL (
    SELECT metric_name, metric_value as latest_metric_value, is_degraded
    FROM ml_performance_metrics
    WHERE model_id = m.id
    ORDER BY measured_at DESC
    LIMIT 1
) p ON true
WHERE m.stage IN ('production', 'staging');
