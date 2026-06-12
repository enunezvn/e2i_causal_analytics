-- ============================================================================
-- Migration 075 (#894): exclude synthetic spans from v_agent_latency_summary
-- ============================================================================
-- ml_observability_spans is is_synthetic-tagged (migration 069) and the
-- synthetic mlops generator loads spans with recent started_at values
-- (600/816 live rows synthetic at filing time), but the latency-summary view
-- (the source of ObservabilitySpanRepository.get_latency_stats -> agent-
-- visible LatencyStats) aggregated them indistinguishably from real traffic.
-- Same pattern as migration 067's KPI-view wrapping: real mode is enforced
-- server-side in the view; synthetic spans remain readable via direct table
-- reads with include_synthetic=True.
-- Idempotent (CREATE OR REPLACE, additive WHERE only — column list unchanged).
-- Source definition: database/ml/mlops_tables.sql:538.

CREATE OR REPLACE VIEW v_agent_latency_summary AS
SELECT
    agent_name,
    agent_tier,
    COUNT(*) as total_spans,
    AVG(duration_ms) as avg_duration_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_ms,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as error_rate,
    SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as fallback_rate,
    SUM(total_tokens) as total_tokens_used
FROM ml_observability_spans
WHERE started_at > NOW() - INTERVAL '24 hours'
  AND is_synthetic = false
GROUP BY agent_name, agent_tier;
