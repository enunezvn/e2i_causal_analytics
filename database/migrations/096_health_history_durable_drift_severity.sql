-- ============================================================================
-- 096_health_history_durable_drift_severity.sql
-- Durable system-health history + honest drift-severity rollup on the model
-- health dashboard (2026-07-06 /system-health review).
--
-- WHY (two independent findings, one page):
--
--   1. The /system-health "Health Trend" and "History" charts read
--      GET /health-score/history, which serves a process-local in-memory list
--      (per gunicorn worker, wiped on every deploy/restart). The "7-day
--      health score history" label was never true — the data is minutes-scale
--      and worker-random. A real 30-day trend needs a durable table; the
--      backend code has carried a "durable health-history table + scheduled
--      full check" follow-up note since #927. This is that follow-up.
--
--   2. ml_model_health_dashboard computed max(severity::text) — an
--      ALPHABETICAL max over 'none'>'medium'>'low'>'high'>'critical', so
--      max_drift_severity was garbage (a 'critical' is masked by any
--      coexisting value; 8 low detections surfaced as 'none'). Worse, the
--      health_status CASE escalated ANY detected drift to 'attention', which
--      the API maps to 'degraded': every Monday the 3AM frontier-append cron
--      legitimately shifts feature distributions, a handful of per-feature
--      checks fire at LOW severity, and all 12 gold-standard models render
--      "degraded" + a warning alert each — alarmist wording for expected
--      weekly data movement (zero active alerts, no performance degradation).
--
-- WHAT THIS MIGRATION DOES:
--
--   1. Creates health_check_history: one row per TRUSTED full-scope health
--      check (the API rate-limits writes and enforces measured/partial
--      provenance; the CHECK constraint makes the DB refuse untrusted rows
--      outright — a placeholder score must never replay as historical truth).
--
--   2. Recreates ml_model_health_dashboard:
--        - max_drift_severity: max() on the drift_severity_enum (enum order,
--          none<low<medium<high<critical), cast to text AFTER aggregation.
--        - health_status 'attention' now requires drift severity >= 'medium'
--          (or a measured performance degradation). Low-severity drift keeps
--          has_active_drift=true for observability (and full drift detail
--          stays on /monitoring) but no longer flips the model status the
--          dashboard words as "degraded".
--      Consumers verified 2026-07-06: only src/api/routes/health_score.py
--      (GET /health-score/models + the health-score agent's MetricsStore
--      adapter) read health_status; nothing else selects from this view.
--
-- NOTE: no BEGIN/COMMIT here — scripts/run_migrations.sh applies migrations
-- with psql --single-transaction; script-level transaction control would
-- break it.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Durable health-check history
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS health_check_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_id TEXT NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Atomic multi-worker dedup: writers compute epoch // 600 and upsert with
    -- ON CONFLICT DO NOTHING, so concurrent workers that all pass the app-side
    -- rate-limit probe still yield exactly one row per 10-min bucket. Two
    -- instants in the same bucket are by definition <600s apart, so this can
    -- never reject a legitimately spaced write. (A stored column, not
    -- GENERATED: extract(epoch FROM timestamptz) is STABLE, not IMMUTABLE.)
    time_bucket BIGINT NOT NULL
        CONSTRAINT health_check_history_one_row_per_bucket UNIQUE,
    overall_health_score NUMERIC(5, 2) NOT NULL,
    health_grade TEXT NOT NULL,
    -- Dimension scores are 0-1 and NULLABLE: null means the dimension was not
    -- measured for that check — never coerced to a fabricated 0.
    component_health_score NUMERIC(4, 3),
    model_health_score NUMERIC(4, 3),
    pipeline_health_score NUMERIC(4, 3),
    agent_health_score NUMERIC(4, 3),
    critical_issues_count INTEGER NOT NULL DEFAULT 0,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    -- Only trusted provenances may be recorded. 'placeholder' (dev mock) and
    -- 'unknown' (fail-closed default) are fabricated-or-empty scores; recording
    -- them would replot as historical truth the very values the live dashboard
    -- refuses to render. Enforced here so no future writer can regress it.
    data_provenance TEXT NOT NULL
        CONSTRAINT health_check_history_trusted_provenance
        CHECK (data_provenance IN ('measured', 'partial')),
    check_scope TEXT NOT NULL DEFAULT 'full'
);

-- The read path is "recent window, newest first" (days-bounded chart + list).
CREATE INDEX IF NOT EXISTS idx_health_check_history_checked_at
    ON health_check_history (checked_at DESC);

GRANT SELECT, INSERT, DELETE ON health_check_history TO service_role;
GRANT SELECT ON health_check_history TO authenticated;

-- Daily aggregates in SQL so the API reads at most `days` rows for a chart
-- window (never the raw row stream — no PostgREST row-cap concerns). A day is
-- 'measured' only when EVERY contributing check was measured.
CREATE OR REPLACE VIEW health_check_history_daily AS
SELECT
    (checked_at AT TIME ZONE 'UTC')::date AS day,
    round(avg(overall_health_score), 2) AS avg_score,
    min(overall_health_score) AS min_score,
    max(overall_health_score) AS max_score,
    count(*) AS checks_count,
    CASE
        WHEN bool_and(data_provenance = 'measured') THEN 'measured'
        ELSE 'partial'
    END AS data_provenance
FROM health_check_history
WHERE check_scope = 'full'
GROUP BY 1;

GRANT SELECT ON health_check_history_daily TO service_role;
GRANT SELECT ON health_check_history_daily TO authenticated;

-- ----------------------------------------------------------------------------
-- 2. Model health dashboard: ordered severity + medium+ attention gate
-- ----------------------------------------------------------------------------
-- Same output columns (names, types, order) as the 031 definition, so
-- CREATE OR REPLACE is safe.

CREATE OR REPLACE VIEW ml_model_health_dashboard AS
SELECT
    m.id AS model_id,
    m.model_name,
    m.stage AS model_stage,
    COALESCE(d.has_drift, false) AS has_active_drift,
    -- Enum max (definition order none<low<medium<high<critical), cast AFTER
    -- aggregation. The previous max(severity::text) was an alphabetical max.
    d.max_severity::text AS max_drift_severity,
    d.drift_check_count,
    COALESCE(a.total_active, 0::bigint) AS active_alerts,
    COALESCE(a.critical_count, 0::bigint) AS critical_alerts,
    p.latest_metric_value,
    p.metric_name AS primary_metric,
    p.is_degraded AS performance_degraded,
    CASE
        WHEN a.critical_count > 0 THEN 'critical'::text
        WHEN a.high_count > 0
            OR d.max_severity >= 'critical'::drift_severity_enum THEN 'warning'::text
        -- Low-severity drift is expected weekly noise (the Monday frontier
        -- append shifts distributions); it stays visible via has_active_drift
        -- and /monitoring but only medium+ drift or a measured performance
        -- degradation marks the model for attention.
        WHEN (d.has_drift AND d.max_severity >= 'medium'::drift_severity_enum)
            OR p.is_degraded THEN 'attention'::text
        ELSE 'healthy'::text
    END AS health_status,
    m.is_synthetic
FROM ml_model_registry m
LEFT JOIN (
    SELECT
        ml_drift_history.model_id,
        bool_or(ml_drift_history.drift_detected) AS has_drift,
        max(ml_drift_history.severity) AS max_severity,
        count(*) AS drift_check_count
    FROM ml_drift_history
    WHERE ml_drift_history.created_at >= (now() - '24:00:00'::interval)
    GROUP BY ml_drift_history.model_id
) d ON m.id = d.model_id
LEFT JOIN ml_active_alerts_summary a ON m.id = a.model_id
LEFT JOIN LATERAL (
    SELECT
        ml_performance_metrics.metric_name,
        ml_performance_metrics.metric_value AS latest_metric_value,
        ml_performance_metrics.is_degraded
    FROM ml_performance_metrics
    WHERE ml_performance_metrics.model_id = m.id
    ORDER BY ml_performance_metrics.measured_at DESC
    LIMIT 1
) p ON true
WHERE m.stage = ANY (ARRAY['production'::model_stage_enum, 'staging'::model_stage_enum]);
