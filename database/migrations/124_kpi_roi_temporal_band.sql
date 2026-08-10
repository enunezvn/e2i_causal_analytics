-- ============================================================================
-- Migration 124: WS3-BI-010 per-slice trailing-12-month temporal-variability
--                band (#1532, supersedes #1527)
-- ============================================================================
-- The observed-ROI KPI (WS3-BI-010) is a pooled point estimate with no
-- dispersion. #1527 established that no interval is possible within the
-- 30-day headline window: business_metrics ROI data is MONTHLY, so every
-- (metric_name, brand, region) slice has exactly n=1 there (measured
-- 2026-08-10: 9,840 ROI rows, one per slice per month, 164 months deep) and a
-- pooled STDDEV would measure cross-slice heterogeneity, not uncertainty.
--
-- This registers the #1532 re-scoped estimand instead: per-slice descriptive
-- statistics over the trailing 12 months of data (n<=12 monthly observations
-- per slice), from which the calculator assembles a TEMPORAL-VARIABILITY BAND
-- - the range of the slice's recent monthly ROI values. It is NOT a
-- confidence interval and is never named as one (the #1526 sensitivity_band
-- naming discipline); suppression below a minimum n and all naming live in
-- the calculator (src/kpi/calculators/business_impact.py).
--
-- Idioms carried over:
-- * 089 frontier-anchoring: the window ends at MAX(metric_date), not NOW()
--   (the substrate is calendar-fixed), and data_through discloses that as-of
--   date on every row.
-- * 066/M4 synthetic gating: base statement default-excludes synthetic rows;
--   the _include_synthetic twin serves showcase deployments (this substrate
--   is currently 100% synthetic, so real-mode returns honest-empty).
-- * business_impact_trx filter idiom: $1 brand (enum ::text equality),
--   $2 region (enum ::text, case-insensitive) - both nullable.
-- * HALF-OPEN window (metric_date > frontier - 12 months): with monthly rows,
--   an inclusive >= would admit 13 observations whenever the frontier lands
--   exactly on a row date (measured 2026-08-10: 13 vs 12 on Kisqali/
--   northeast/trx with a 2026-08-01 frontier) - the (F-12mo, F] form keeps
--   the n<=12 contract unconditionally.
--
-- ADDITIVE registry rows only; no DDL, no data changes, every existing id
-- untouched. DB application follows the droplet recipe (psql
-- --single-transaction owns the outer txn - no COMMIT here).
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_roi_temporal_band', $kpi$SELECT s.*, (SELECT MAX(metric_date) FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics)::date AS data_through FROM (SELECT metric_name, brand::text AS brand, region::text AS region, COUNT(*) AS n, AVG(roi) AS roi_mean, STDDEV(roi) AS roi_stddev, MIN(roi) AS roi_min, MAX(roi) AS roi_max FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics WHERE roi IS NOT NULL AND metric_date > (SELECT MAX(metric_date) FROM (SELECT * FROM business_metrics WHERE is_synthetic = false) business_metrics) - INTERVAL '12 months' AND ($1::text IS NULL OR brand::text = $1) AND ($2::text IS NULL OR LOWER(region::text) = LOWER($2)) GROUP BY metric_name, brand::text, region::text ORDER BY metric_name, brand::text, region::text) s$kpi$, 2, $note$#1532: per-slice trailing-12-month ROI temporal-variability stats ($1 brand, $2 region, both nullable); NOT a confidence interval - the band names temporal variability of monthly values; M4 default-exclude synthetic; 089 frontier-anchored$note$),
    ('business_impact_roi_temporal_band_include_synthetic', $kpi$SELECT s.*, (SELECT MAX(metric_date) FROM business_metrics)::date AS data_through FROM (SELECT metric_name, brand::text AS brand, region::text AS region, COUNT(*) AS n, AVG(roi) AS roi_mean, STDDEV(roi) AS roi_stddev, MIN(roi) AS roi_min, MAX(roi) AS roi_max FROM business_metrics WHERE roi IS NOT NULL AND metric_date > (SELECT MAX(metric_date) FROM business_metrics) - INTERVAL '12 months' AND ($1::text IS NULL OR brand::text = $1) AND ($2::text IS NULL OR LOWER(region::text) = LOWER($2)) GROUP BY metric_name, brand::text, region::text ORDER BY metric_name, brand::text, region::text) s$kpi$, 2, $note$#1532 temporal-variability stats (includes synthetic - showcase deployments); 089 frontier-anchored$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
