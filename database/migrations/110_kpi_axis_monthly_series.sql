-- ============================================================================
-- 110_kpi_axis_monthly_series.sql
-- Monthly TIME-SERIES variants of the business_impact Rx queries, grouped by
-- patient axis: severity tier (patient_journeys.segment_assignment) and
-- line-of-therapy (patient_journeys.prior_therapy_lines).
--
-- WHY: the chat chart renderer (renderKpiTrend) reads the materialized
-- kpi_history table, which has no patient-segment dimension (migration 079 —
-- intentionally: threading an axis into that read would silently return
-- unsegmented history). Migration 105 added axis-scoped POINT queries
-- (`*_segment` / `*_line`, one value per call), but charting a trend per tier
-- via those would need one RPC round-trip per month per tier. These `_monthly_
-- by_*` variants return the FULL monthly series for ALL tiers of an axis in a
-- single call: rows of (month_start, bucket, value), plus the global
-- prescription date range (data_min/data_max) so the API can drop partial
-- edge months exactly like the kpi_history backfill does
-- (src/kpi/history_backfill.py::_complete_months — a leading month is
-- complete only when data starts on its 1st, a trailing month only when data
-- reaches its last day).
--
-- COHERENCE with the headline series: the brand-level kpi_history TRx/NRx
-- points are calendar-month COUNTs over treatment_events prescriptions
-- (_backfill_trx/_backfill_nrx), and NBRx is the month of each patient's
-- first brand Rx (_backfill_nbrx). The statements below use the identical
-- month bucketing (date_trunc('month', event_date)) and event predicates, so
-- the per-tier lines partition the headline brand line month by month.
--
-- JOIN KEY = patient_id (NOT patient_journey_id) — same rationale as
-- migration 105: treatment_events.patient_journey_id is NULL on ~17% of
-- prescriptions (~45% of NRx), while patient_id links 100% and
-- patient_journeys is 1:1 on patient_id (25,018/25,018 distinct; zero
-- patients carry >1 segment_assignment or >1 prior_therapy_lines), so an
-- INNER JOIN neither drops nor fans out rows. Rows whose journey has a NULL
-- axis value are excluded (they are equally unreachable through the per-tier
-- migration-105 variants).
--
-- ADDITIVE (not in-place), mirroring 077/084/105: new parallel query ids
-- only; certified base queries stay byte-for-byte unchanged. The
-- `_include_synthetic` twins (used when the synthetic-gold showcase flag is
-- on, via src/kpi/synthetic_mode.py suffixing) drop the
-- `(SELECT * ... WHERE is_synthetic=false)` wrappers.
--
-- Params: [brand] (max_params = 1); NULL brand = all brands. No window
-- params — the full series is small (months x <=4 buckets) and the API
-- applies start/end filters, mirroring /api/kpis/{id}/history semantics.
--
-- Exactly 12 new query ids:
--   {trx,nrx,nbrx} x {monthly_by_segment, monthly_by_line} x
--   {plain, _include_synthetic}
--
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry+RPC, which
-- wraps statements in SELECT row_to_json(...) — SETOF json supports these
-- multi-row results), 089 (frontier/base statement shapes), 105 (axis
-- predicates + patient_id-join validation).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- TRx monthly by severity tier ----
    ('business_impact_trx_monthly_by_segment', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND pj.segment_assignment IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly TRx series by severity tier$note$),
    ('business_impact_trx_monthly_by_segment_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND pj.segment_assignment IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly TRx series by severity tier (includes synthetic)$note$),
    -- ---- NRx monthly by severity tier ----
    ('business_impact_nrx_monthly_by_segment', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND pj.segment_assignment IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NRx series by severity tier$note$),
    ('business_impact_nrx_monthly_by_segment_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND pj.segment_assignment IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NRx series by severity tier (includes synthetic)$note$),
    -- ---- NBRx monthly by severity tier (month of each patient's FIRST brand Rx) ----
    ('business_impact_nbrx_monthly_by_segment', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) SELECT date_trunc('month', fb.first_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM first_brand fb JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = fb.patient_id WHERE pj.segment_assignment IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NBRx series by severity tier$note$),
    ('business_impact_nbrx_monthly_by_segment_include_synthetic', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) SELECT date_trunc('month', fb.first_date)::date AS month_start, LOWER(pj.segment_assignment::text) AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM first_brand fb JOIN patient_journeys pj ON pj.patient_id = fb.patient_id WHERE pj.segment_assignment IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NBRx series by severity tier (includes synthetic)$note$),
    -- ---- TRx monthly by line-of-therapy ----
    ('business_impact_trx_monthly_by_line', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND pj.prior_therapy_lines IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly TRx series by line of therapy$note$),
    ('business_impact_trx_monthly_by_line_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND pj.prior_therapy_lines IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly TRx series by line of therapy (includes synthetic)$note$),
    -- ---- NRx monthly by line-of-therapy ----
    ('business_impact_nrx_monthly_by_line', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND pj.prior_therapy_lines IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NRx series by line of therapy$note$),
    ('business_impact_nrx_monthly_by_line_include_synthetic', $kpi$SELECT date_trunc('month', te.event_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND pj.prior_therapy_lines IS NOT NULL AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NRx series by line of therapy (includes synthetic)$note$),
    -- ---- NBRx monthly by line-of-therapy ----
    ('business_impact_nbrx_monthly_by_line', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) SELECT date_trunc('month', fb.first_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM first_brand fb JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = fb.patient_id WHERE pj.prior_therapy_lines IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NBRx series by line of therapy$note$),
    ('business_impact_nbrx_monthly_by_line_include_synthetic', $kpi$WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) SELECT date_trunc('month', fb.first_date)::date AS month_start, pj.prior_therapy_lines::text AS bucket, COUNT(*) AS value, (SELECT MIN(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_min, (SELECT MAX(event_date) FROM treatment_events WHERE event_type::text = 'prescription')::date AS data_max FROM first_brand fb JOIN patient_journeys pj ON pj.patient_id = fb.patient_id WHERE pj.prior_therapy_lines IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 1, $note$monthly NBRx series by line of therapy (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';
