-- ============================================================================
-- 128_kpi_conversion_rate_brand_region.sql
-- Brand+region joint variant for Conversion Rate (WS3-BI-009), issue #1575.
--
-- WHY: the conversion-rate family had `_brand` [brand] (111) and `_region`
-- [region] (077/089) legs but no joint — its certified base is param-less by
-- original design, so unlike every Rx-volume sibling (whose NULLable $1 brand
-- makes their `_region` leg [brand, region] already), "conversion rate for
-- Kisqali in the west" could only be answered brand-scoped or region-scoped.
-- `_calc_conversion_rate` failed loud on the combination and the chat layer
-- answered "KPI unavailable" for a combination that is unserved, not
-- unservable. Registry sweep (2026-08-13 live dump): conversion_rate is the
-- ONLY business_impact family with this split-legs gap.
--
-- DERIVATION (byte-based, not hand-written): each statement is the vetted
-- migration-111 `_segment` statement with the patient-axis CTE swapped —
-- axis_patients on segment_assignment -> region_patients on
-- LOWER(geographic_region::text) = LOWER($2) (the 077/078 region idiom).
-- Everything else is byte-identical to the dry-run-verified 111 shape:
--   * SAME-BRAND semantics: NULL-tolerant $1 on the triggered CTE
--     (triggers.brand_id), the converted CTE (t.brand_id) AND the converting
--     prescription (te.brand) — with $1 NULL the statement reduces to the
--     `_region` leg's semantics (111 header, dry-run-verified);
--   * region is patient MEMBERSHIP via patient_id on BOTH CTEs (1:1 with
--     patient_journeys; NEVER treatment_events.patient_journey_id — NULL on
--     ~45% of NRx events, #1208);
--   * frontier-anchored at the GLOBAL triggers MAX with data_through (089;
--     the anchoring generator's replay stops at 089 by design, so post-089
--     rows are born anchored — drift-locked in
--     tests/unit/test_kpi/test_mig128_registry_presence.py);
--   * the 30-day trigger->Rx conversion horizon is the KPI's definition and
--     is untouched; the region cut bounds WHICH triggers/patients count.
--
-- Params: $1 brand (NULL-tolerant), $2 region — the 113 `_brand_region`
-- ordering, routed by `_calc_conversion_rate` via brand_region_query_id.
-- Windowed brand+region stays UNREGISTERED (honest failure preserved).
--
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry+RPC),
-- 066 (synthetic twins), 089 (frontier anchoring), 111 (family brand legs).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_conversion_rate_brand_region', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (WITH region_patients AS (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)), triggered AS (SELECT COUNT(DISTINCT trigger_id) AS total FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days' AND ($1::text IS NULL OR brand_id = $1) AND patient_id IN (SELECT patient_id FROM region_patients)), converted AS (SELECT COUNT(DISTINCT t.trigger_id) AS total FROM (SELECT * FROM triggers WHERE is_synthetic = false) t INNER JOIN (SELECT * FROM treatment_events WHERE is_synthetic = false) te ON te.patient_id = t.patient_id WHERE t.trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days' AND ($1::text IS NULL OR t.brand_id = $1) AND t.patient_id IN (SELECT patient_id FROM region_patients) AND te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) AND te.event_date >= t.trigger_timestamp::date AND te.event_date <= (t.trigger_timestamp + INTERVAL '30 days')::date) SELECT converted.total::float / NULLIF(triggered.total, 0) AS conversion_rate FROM triggered, converted) base$kpi$, 2, $note$#1575 brand+region-scoped conversion rate (same-brand trigger->Rx, frontier 30d); 111 _segment shape with the patient axis swapped to patient_journeys.geographic_region membership ($1 brand NULL-tolerant, $2 region)$note$),
    ('business_impact_conversion_rate_brand_region_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (WITH region_patients AS (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)), triggered AS (SELECT COUNT(DISTINCT trigger_id) AS total FROM triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days' AND ($1::text IS NULL OR brand_id = $1) AND patient_id IN (SELECT patient_id FROM region_patients)), converted AS (SELECT COUNT(DISTINCT t.trigger_id) AS total FROM triggers t INNER JOIN treatment_events te ON te.patient_id = t.patient_id WHERE t.trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days' AND ($1::text IS NULL OR t.brand_id = $1) AND t.patient_id IN (SELECT patient_id FROM region_patients) AND te.event_type::text = 'prescription' AND ($1::text IS NULL OR te.brand::text = $1) AND te.event_date >= t.trigger_timestamp::date AND te.event_date <= (t.trigger_timestamp + INTERVAL '30 days')::date) SELECT converted.total::float / NULLIF(triggered.total, 0) AS conversion_rate FROM triggered, converted) base$kpi$, 2, $note$#1575 brand+region-scoped conversion rate (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new statements serve immediately.
NOTIFY pgrst, 'reload schema';
