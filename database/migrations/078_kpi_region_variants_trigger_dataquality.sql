-- ============================================================================
-- 078_kpi_region_variants_trigger_dataquality.sql
-- Region-scoped variants of the trigger_performance + data_quality KPI queries
-- (Increment 2). Same ADDITIVE, zero-touch approach as migration 077: the base
-- queries stay byte-for-byte unchanged (certified gates unaffected, RPC
-- param-count safe), and the calculator routes to these `*_region` ids ONLY
-- when a region is selected.
--
-- Region param is $1 (max_params 1) for every query here. Join paths:
--   * trigger_performance.* : triggers.patient_id -> patient_journeys
--     .geographic_region (triggers carries no region column). recall also scopes
--     its treatment_events cohort the same way.
--   * data_quality_completeness_pass_rate : patient_journeys.geographic_region
--     directly.
--   * data_quality_source_coverage_patients/hcps : the covered side filters the
--     source table's region; the universe total filters reference_universe.region
--     AND universe_type (the region cut REQUIRES the universe_type filter to
--     avoid mixing hcp+patient targets — more correct than the brand-only base).
-- Region match is case-insensitive (LOWER both sides). The *_include_synthetic
-- variants drop the (SELECT ... WHERE is_synthetic=false) wrappers.
--
-- DELIBERATELY OMITTED: data_quality_geographic_consistency. It measures the gap
-- between the geographic DISTRIBUTION of patients and the reference universe
-- (max share-difference ACROSS regions) — a cross-region metric. Scoping it to a
-- single region is semantically meaningless, so it has no region variant and the
-- dashboard keeps its portfolio-level value when a region is selected.
--
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry+RPC), 066 (twins).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- trigger_performance: acceptance_rate (region=$1) ----
    ('trigger_performance_acceptance_rate_region', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS acceptance_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped acceptance rate$note$),
    ('trigger_performance_acceptance_rate_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS acceptance_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped acceptance rate (includes synthetic)$note$),

    -- ---- trigger_performance: action_rate_uplift (region=$1) ----
    ('trigger_performance_action_rate_uplift_region', $kpi$WITH arms AS (SELECT control_group_flag, COUNT(*) FILTER (WHERE action_taken IS NOT NULL)::float / NULLIF(COUNT(*), 0) AS action_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE control_group_flag IS NOT NULL AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1)) GROUP BY control_group_flag) SELECT (t.action_rate - c.action_rate) / NULLIF(c.action_rate, 0) AS action_rate_uplift, t.action_rate AS treatment_rate, c.action_rate AS control_rate FROM (SELECT action_rate FROM arms WHERE control_group_flag = false) t, (SELECT action_rate FROM arms WHERE control_group_flag = true) c$kpi$, 1, $note$region-scoped action rate uplift$note$),
    ('trigger_performance_action_rate_uplift_region_include_synthetic', $kpi$WITH arms AS (SELECT control_group_flag, COUNT(*) FILTER (WHERE action_taken IS NOT NULL)::float / NULLIF(COUNT(*), 0) AS action_rate FROM triggers WHERE control_group_flag IS NOT NULL AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1)) GROUP BY control_group_flag) SELECT (t.action_rate - c.action_rate) / NULLIF(c.action_rate, 0) AS action_rate_uplift, t.action_rate AS treatment_rate, c.action_rate AS control_rate FROM (SELECT action_rate FROM arms WHERE control_group_flag = false) t, (SELECT action_rate FROM arms WHERE control_group_flag = true) c$kpi$, 1, $note$region-scoped action rate uplift (includes synthetic)$note$),

    -- ---- trigger_performance: cfr (region=$1) ----
    ('trigger_performance_cfr_region', $kpi$SELECT COUNT(CASE WHEN change_failed THEN 1 END)::float / NULLIF(COUNT(CASE WHEN previous_trigger_id IS NOT NULL THEN 1 END), 0) AS cfr FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped change-fail rate$note$),
    ('trigger_performance_cfr_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN change_failed THEN 1 END)::float / NULLIF(COUNT(CASE WHEN previous_trigger_id IS NOT NULL THEN 1 END), 0) AS cfr FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped change-fail rate (includes synthetic)$note$),

    -- ---- trigger_performance: false_alert_rate (region=$1) ----
    ('trigger_performance_false_alert_rate_region', $kpi$SELECT COUNT(CASE WHEN false_positive_flag THEN 1 END)::float / NULLIF(COUNT(*), 0) AS false_alert_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped false alert rate$note$),
    ('trigger_performance_false_alert_rate_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN false_positive_flag THEN 1 END)::float / NULLIF(COUNT(*), 0) AS false_alert_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped false alert rate (includes synthetic)$note$),

    -- ---- trigger_performance: lead_time (region=$1) ----
    ('trigger_performance_lead_time_region', $kpi$SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_days) AS median_lead_time FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE lead_time_days IS NOT NULL AND trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped lead time$note$),
    ('trigger_performance_lead_time_region_include_synthetic', $kpi$SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_days) AS median_lead_time FROM triggers WHERE lead_time_days IS NOT NULL AND trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped lead time (includes synthetic)$note$),

    -- ---- trigger_performance: override_rate (region=$1) ----
    ('trigger_performance_override_rate_region', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS override_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped override rate$note$),
    ('trigger_performance_override_rate_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0) AS override_rate FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped override rate (includes synthetic)$note$),

    -- ---- trigger_performance: precision (region=$1) ----
    ('trigger_performance_precision_region', $kpi$SELECT COUNT(CASE WHEN outcome_tracked AND outcome_value > 0 THEN 1 END)::float / NULLIF(COUNT(CASE WHEN outcome_tracked THEN 1 END), 0) AS precision FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped precision$note$),
    ('trigger_performance_precision_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN outcome_tracked AND outcome_value > 0 THEN 1 END)::float / NULLIF(COUNT(CASE WHEN outcome_tracked THEN 1 END), 0) AS precision FROM triggers WHERE trigger_timestamp >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))$kpi$, 1, $note$region-scoped precision (includes synthetic)$note$),

    -- ---- trigger_performance: recall (region=$1; scopes the treatment_events cohort) ----
    ('trigger_performance_recall_region', $kpi$WITH positive_outcomes AS (SELECT DISTINCT patient_id FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE event_type::text = 'prescription' AND event_date >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))), trigger_preceded AS (SELECT DISTINCT po.patient_id FROM positive_outcomes po INNER JOIN (SELECT * FROM triggers WHERE is_synthetic = false) t ON t.patient_id = po.patient_id WHERE t.trigger_timestamp < (SELECT MIN(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.patient_id = po.patient_id AND te.event_type::text = 'prescription')) SELECT COUNT(DISTINCT tp.patient_id)::float / NULLIF(COUNT(DISTINCT po.patient_id), 0) AS recall FROM positive_outcomes po LEFT JOIN trigger_preceded tp ON tp.patient_id = po.patient_id$kpi$, 1, $note$region-scoped recall$note$),
    ('trigger_performance_recall_region_include_synthetic', $kpi$WITH positive_outcomes AS (SELECT DISTINCT patient_id FROM treatment_events WHERE event_type::text = 'prescription' AND event_date >= NOW() - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))), trigger_preceded AS (SELECT DISTINCT po.patient_id FROM positive_outcomes po INNER JOIN triggers t ON t.patient_id = po.patient_id WHERE t.trigger_timestamp < (SELECT MIN(event_date) FROM treatment_events te WHERE te.patient_id = po.patient_id AND te.event_type::text = 'prescription')) SELECT COUNT(DISTINCT tp.patient_id)::float / NULLIF(COUNT(DISTINCT po.patient_id), 0) AS recall FROM positive_outcomes po LEFT JOIN trigger_preceded tp ON tp.patient_id = po.patient_id$kpi$, 1, $note$region-scoped recall (includes synthetic)$note$),

    -- ---- data_quality: completeness_pass_rate (region=$1) ----
    ('data_quality_completeness_pass_rate_region', $kpi$SELECT AVG(CASE WHEN patient_id IS NOT NULL AND brand IS NOT NULL AND event_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS pass_rate FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE created_at >= NOW() - INTERVAL '30 days' AND LOWER(geographic_region::text) = LOWER($1)$kpi$, 1, $note$region-scoped completeness pass rate$note$),
    ('data_quality_completeness_pass_rate_region_include_synthetic', $kpi$SELECT AVG(CASE WHEN patient_id IS NOT NULL AND brand IS NOT NULL AND event_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS pass_rate FROM patient_journeys WHERE created_at >= NOW() - INTERVAL '30 days' AND LOWER(geographic_region::text) = LOWER($1)$kpi$, 1, $note$region-scoped completeness pass rate (includes synthetic)$note$),

    -- ---- data_quality: source_coverage_patients (region=$1) ----
    ('data_quality_source_coverage_patients_region', $kpi$SELECT COUNT(DISTINCT pj.patient_id) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'patient' AND LOWER(region::text) = LOWER($1)), 0) AS total FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj WHERE LOWER(pj.geographic_region::text) = LOWER($1)$kpi$, 1, $note$region-scoped patient source coverage$note$),
    ('data_quality_source_coverage_patients_region_include_synthetic', $kpi$SELECT COUNT(DISTINCT pj.patient_id) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'patient' AND LOWER(region::text) = LOWER($1)), 0) AS total FROM patient_journeys pj WHERE LOWER(pj.geographic_region::text) = LOWER($1)$kpi$, 1, $note$region-scoped patient source coverage (includes synthetic)$note$),

    -- ---- data_quality: source_coverage_hcps (region=$1) ----
    ('data_quality_source_coverage_hcps_region', $kpi$SELECT (SELECT COUNT(DISTINCT hcp_id) FROM (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hcp_profiles WHERE coverage_status = true AND LOWER(geographic_region::text) = LOWER($1)) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'hcp' AND LOWER(region::text) = LOWER($1)), 0) AS total$kpi$, 1, $note$region-scoped HCP source coverage$note$),
    ('data_quality_source_coverage_hcps_region_include_synthetic', $kpi$SELECT (SELECT COUNT(DISTINCT hcp_id) FROM hcp_profiles WHERE coverage_status = true AND LOWER(geographic_region::text) = LOWER($1)) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'hcp' AND LOWER(region::text) = LOWER($1)), 0) AS total$kpi$, 1, $note$region-scoped HCP source coverage (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;
