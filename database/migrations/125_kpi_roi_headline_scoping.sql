-- ============================================================================
-- Migration 125: WS3-BI-010 brand/region-scoped ROI headline (#1534)
-- ============================================================================
-- The ROI headline query has been a 0-param portfolio-wide aggregate since the
-- 044 allowlist restore, while every surface that batches it (dashboard grid,
-- insights_strategic grounding, chatbot kpi_calculate_tool) passes brand/
-- region context and labels the figure with it ("the brand/region the figure
-- was computed for" — d72ac745). #1534 makes the headline honor that scope.
--
-- This registers a 2-nullable-param SCOPED variant of the exact 089 headline:
-- same AVG(roi), same inclusive >= 30-day window, same frontier anchoring,
-- same data_through disclosure — plus the 124/business_impact_trx filter
-- idiom ($1 brand enum ::text equality, $2 region enum ::text
-- case-insensitive, both nullable). Called with [NULL, NULL] it is
-- value-identical to the 0-param query (executable invariant:
-- tests/integration/test_roi_headline_scoping_1534_live.py).
--
-- The frontier (MAX(metric_date)) stays GLOBAL, not scope-narrowed: the
-- window is the same calendar window for every scope (matching the 124 band),
-- so a brand with no recent rows returns an honest NULL average instead of
-- silently sliding its window into the past. The calculator fails loud on
-- that NULL when scope was requested (agent_activities has no brand/region
-- dimension — measured 2026-08-10 — so no scoped fallback exists).
--
-- ADDITIVE registry rows only; the 0-param ids stay registered and untouched
-- (existing callers: scripts/check_kpi_coverage.py, M4 tests) — no deploy
-- skew between old code and new registry in either direction. DB application
-- follows the droplet recipe (psql --single-transaction owns the outer txn -
-- no COMMIT here).
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_roi_business_metrics_scoped', $kpi$SELECT base.*, (SELECT MAX(metric_date) FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics)::date AS data_through FROM (SELECT AVG(roi) AS avg_roi FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics WHERE metric_date >= (SELECT MAX(metric_date) FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics) - INTERVAL '30 days' AND roi IS NOT NULL AND ($1::text IS NULL OR brand::text = $1) AND ($2::text IS NULL OR LOWER(region::text) = LOWER($2))) base$kpi$, 2, $note$#1534: brand/region-scoped ROI headline ($1 brand, $2 region, both nullable; [NULL,NULL] == the 0-param query); M4 default-exclude synthetic; 089 frontier-anchored (GLOBAL frontier - window never slides per scope)$note$),
    ('business_impact_roi_business_metrics_scoped_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(metric_date) FROM business_metrics)::date AS data_through FROM (SELECT AVG(roi) AS avg_roi FROM business_metrics WHERE metric_date >= (SELECT MAX(metric_date) FROM business_metrics) - INTERVAL '30 days' AND roi IS NOT NULL AND ($1::text IS NULL OR brand::text = $1) AND ($2::text IS NULL OR LOWER(region::text) = LOWER($2))) base$kpi$, 2, $note$#1534 scoped ROI headline (includes synthetic - showcase deployments); 089 frontier-anchored (GLOBAL frontier)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
