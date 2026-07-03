-- ============================================================================
-- 092_kpi_tr004_acceptance_delivered_denominator.sql
-- WS2-TR-004 (Acceptance Rate): align the registry SQL denominator with the
-- documented formula `count(accepted) / count(delivered)` (#1124).
--
-- WHY: migrations 044/066/078/089 registered a denominator of ALL rows with a
-- non-null acceptance_status. The trigger DGP never emits a NULL status
-- (undelivered triggers carry 'pending'), so that denominator degenerates to
-- COUNT(*) over the window -- it counts pending and failed-delivery triggers
-- that were never in front of a rep, diluting the rate and diverging from the
-- documented semantics (config/kpi_definitions.yaml WS2-TR-004,
-- docs/data/06-KPI-REFERENCE.md WS2-TR-004). Since #1122 added the
-- 'overridden' acceptance arm, those rows also inflate the denominator.
-- Measured live 2026-07-03 (30-day window, include-synthetic domain):
-- accepted 9,785 / all-non-null 23,225 = 0.4213 -> CRITICAL (< 0.45), vs
-- accepted 9,785 / delivered 19,663 = 0.4976 -> WARNING. The inflated
-- denominator flips the status band.
--
-- WHAT "DELIVERED" MEANS: triggers.delivery_status IN ('delivered','viewed'),
-- the convention established by migration 090 (#1119) for TR-006. 'viewed' is
-- strictly post-delivery in the delivery lifecycle (trigger_generator.py
-- gates any acceptance disposition on delivery_status IN
-- ('delivered','viewed')), so a viewed trigger IS a delivered trigger. The
-- numerator stays count(acceptance_status = 'accepted') over the same window;
-- the DGP only emits 'accepted' on delivered/viewed rows, so
-- numerator <= denominator by construction.
--
-- TR-004 was explicitly flagged as the out-of-scope follow-up of this
-- divergence class in PR #1122 (migration 090); this migration closes it.
--
-- Preserves the 089 frontier-anchoring contract verbatim: windows end at
-- MAX(trigger_timestamp) over each query's own domain (never NOW()) and every
-- statement exposes the data_through provenance column. Arity unchanged.
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry), 089, 090.
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('trigger_performance_acceptance_rate', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days') base$kpi$, 0, $note$M4: default-exclude synthetic; 089 frontier-anchored (window ends at domain MAX, not NOW()); 092 denominator = delivered (delivery_status delivered/viewed) per documented formula count(accepted)/count(delivered) (#1124)$note$),
    ('trigger_performance_acceptance_rate_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days') base$kpi$, 0, $note$M4 opt-in: INCLUDES synthetic (validation runs only); 089 frontier-anchored (window ends at domain MAX, not NOW()); 092 denominator = delivered (delivery_status delivered/viewed) per documented formula count(accepted)/count(delivered) (#1124)$note$),
    ('trigger_performance_acceptance_rate_region', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))) base$kpi$, 1, $note$region-scoped acceptance rate; 089 frontier-anchored (window ends at domain MAX, not NOW()); 092 denominator = delivered (delivery_status delivered/viewed) per documented formula count(accepted)/count(delivered) (#1124)$note$),
    ('trigger_performance_acceptance_rate_region_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM triggers WHERE trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '30 days' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($1))) base$kpi$, 1, $note$region-scoped acceptance rate (includes synthetic); 089 frontier-anchored (window ends at domain MAX, not NOW()); 092 denominator = delivered (delivery_status delivered/viewed) per documented formula count(accepted)/count(delivered) (#1124)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the updated rows are visible.
NOTIFY pgrst, 'reload schema';
