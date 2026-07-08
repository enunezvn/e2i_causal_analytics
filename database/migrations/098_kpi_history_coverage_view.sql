-- Migration 098: kpi_history coverage view
-- =========================================
-- Per-(kpi_id, brand) coverage summary of the materialized KPI history so the
-- Time-Series page can (a) badge dropdown entries that have a real series and
-- (b) offer only the brand scopes that actually exist for the selected KPI
-- (e.g. WS3-BI-007 NBRx is per-brand ONLY — a global request is empty by
-- design, not by accident).
--
-- Read by GET /api/kpis/history/coverage (KPIHistoryRepository.get_coverage).

CREATE OR REPLACE VIEW v_kpi_history_coverage AS
SELECT
    kpi_id,
    brand,
    COUNT(*)         AS points,
    MIN(metric_date) AS first_date,
    MAX(metric_date) AS last_date
FROM kpi_history
GROUP BY kpi_id, brand;

COMMENT ON VIEW v_kpi_history_coverage IS
    'Coverage summary of kpi_history per (kpi_id, brand): point count + date span. Backs GET /api/kpis/history/coverage.';

-- PostgREST caches the schema; reload so the new view is visible to the REST API.
NOTIFY pgrst, 'reload schema';
