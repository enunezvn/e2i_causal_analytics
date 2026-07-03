-- ============================================================================
-- 090_kpi_tr006_override_delivered_denominator.sql
-- WS2-TR-006 (Override Rate): align the registry SQL denominator with the
-- documented formula `count(overridden) / count(delivered)` (#1119).
--
-- WHY: migrations 078/089 registered a denominator of ALL rows with a
-- non-null acceptance_status. The trigger DGP never emits a NULL status
-- (undelivered triggers carry 'pending'), so that denominator degenerates to
-- COUNT(*) over the window -- it counts pending and failed-delivery triggers
-- that were never in front of a rep, diluting the rate and diverging from the
-- documented semantics (config/kpi_definitions.yaml:486,
-- docs/data/06-KPI-REFERENCE.md WS2-TR-006).
--
-- WHAT "DELIVERED" MEANS: triggers.delivery_status IN ('delivered','viewed').
-- 'viewed' is strictly post-delivery in the delivery lifecycle
-- (trigger_generator.py gates any acceptance disposition on
-- delivery_status IN ('delivered','viewed')), so a viewed trigger IS a
-- delivered trigger. The numerator stays count(acceptance_status =
-- 'overridden') over the same window; the DGP only emits 'overridden' on
-- delivered/viewed rows, so numerator <= denominator by construction.
--
-- COMPANION (same PR, #1118/#1119): the trigger DGP now emits the
-- 'overridden' acceptance_status arm (P=0.14 of delivered -- just under the
-- 0.15 target) and populates false_positive_flag; the KPI stops being
-- structurally-0.0 after the operator reseeds.
--
-- TR-005 (false_alert_rate) AUDITED, NOT CHANGED: its registered denominator
-- is COUNT(*) over the window, which matches its documented formula
-- `count(false_positive) / total_triggers` exactly. (TR-004 acceptance_rate
-- shares the old non-null-status denominator pattern; it is documented as
-- count(accepted)/count(delivered) and is tracked separately, not here.)
--
-- Preserves the 089 frontier-anchoring contract verbatim: windows end at
-- MAX(trigger_timestamp) over each query's own domain (never NOW()) and every
-- statement exposes the data_through provenance column. Arity unchanged.
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry), 089.
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('trigger_performance_override_rate', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days') base$kpi$, 0, $note$M4: default-exclude synthetic; 089 frontier-anchored (window ends at domain MAX, not NOW()); 090 denominator = delivered (delivery_status delivered/viewed) per documented formula count(overridden)/count(delivered) (#1119)$note$),
    ('trigger_performance_override_rate_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days') base$kpi$, 0, $note$M4 opt-in: INCLUDES synthetic (validation runs only); 089 frontier-anchored (window ends at domain MAX, not NOW()); 090 denominator = delivered (delivery_status delivered/viewed) per documented formula count(overridden)/count(delivered) (#1119)$note$),
    ('trigger_performance_override_rate_region', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))) base$kpi$, 1, $note$region-scoped override rate; 089 frontier-anchored (window ends at domain MAX, not NOW()); 090 denominator = delivered (delivery_status delivered/viewed) per documented formula count(overridden)/count(delivered) (#1119)$note$),
    ('trigger_performance_override_rate_region_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))) base$kpi$, 1, $note$region-scoped override rate (includes synthetic); 089 frontier-anchored (window ends at domain MAX, not NOW()); 090 denominator = delivered (delivery_status delivered/viewed) per documented formula count(overridden)/count(delivered) (#1119)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the updated rows are visible.
NOTIFY pgrst, 'reload schema';
