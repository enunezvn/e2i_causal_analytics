-- ============================================================================
-- 127_kpi_brand_specific_region_variants.sql
-- Region-scoped variants of the brand_specific KPI queries (#1564).
--
-- WHY: the region axis (#1536/#1538, migrations 077/078/113/125) covered 3 of
-- the 6 KPI calculator families; brand_specific had no region variants, so
-- region+brand asks ("Kisqali oncologist reach in the northeast") always
-- answered portfolio-level under the honest not_applicable hedge. Region
-- exists in every BR source: patient_journeys.geographic_region for the
-- patient-based KPIs (BR-001/003/004), hcp_profiles.geographic_region for the
-- HCP-based ones (BR-002/005).
--
-- WHY ADDITIVE (not in-place): same contract as migration 077 — the base
-- statements feed certified reads; parallel `*_region` ids are routed to ONLY
-- when a region is selected, so region=None stays byte-identical to today.
--
-- Each variant mirrors the LATEST registered SQL of its base (BR-001/BR-004
-- from 066, BR-002 primary from 044 view semantics, BR-002 fallback + BR-005
-- from 089 frontier-anchored, BR-003 from 091 structural-zero guard) plus one
-- region predicate:
--   * patient membership joins on patient_id (1:1 with patient_journeys),
--     NEVER treatment_events.patient_journey_id — that FK is NULL on ~45% of
--     NRx events and silently drops rows (#1208, corrected idiom of 105/111);
--   * region match is case-insensitive LOWER(geographic_region::text)=LOWER($n)
--     (077/078 idiom);
--   * the windowed statements (BR-002 fallback, BR-005) keep their mig-089
--     frontier anchor at the GLOBAL domain MAX (a per-region frontier would
--     shift maturation cutoffs; kpi_history backfill precedent keeps cutoffs
--     global);
--   * BR-003 keeps the #1116 pnh_events_total guard TABLE-WIDE: substrate
--     coverage ("concept never recorded anywhere") is not a per-region fact —
--     a regional 0.0 with events elsewhere is a genuine regional 0%.
--
-- BR-002 note: the certified chain is primary (v_kpi_intent_to_prescribe,
-- latest survey month) -> fallback (trailing 90 days of data). The view has
-- no region column, so the `_primary_region` variant reproduces the view's
-- defining aggregation (quality-flagged monthly average) from hcp_intent_surveys
-- joined to the surveyed HCP's hcp_profiles.geographic_region, evaluated at
-- the REGION's latest quality-flagged survey month (the region's most recent
-- reading — the per-region parallel of ORDER BY survey_month DESC LIMIT 1).
--
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry+RPC),
-- 066 (synthetic twins), 089 (frontier anchoring), 091 (BR-003 guard).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- BR-001 Remi AH Uncontrolled ($1 UAS7 threshold, $2 region) ----
    ('brand_specific_remi_ah_uncontrolled_region', $kpi$WITH per_patient AS (SELECT patient_id, bool_or((lab_values->>'value')::numeric >= $1::numeric) AS uncontrolled FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events WHERE brand::text = 'Remibrutinib' AND event_subtype = 'baseline_antihistamine' AND drug_class = 'R06A' AND lab_values->>'assay' = 'UAS7' AND patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)) GROUP BY patient_id) SELECT COUNT(*) FILTER (WHERE uncontrolled)::float / NULLIF(COUNT(*), 0) AS uncontrolled_rate FROM per_patient$kpi$, 2, $note$#1564 region-scoped BR-001; region via patient_id -> patient_journeys.geographic_region ($1 threshold, $2 region)$note$),
    ('brand_specific_remi_ah_uncontrolled_region_include_synthetic', $kpi$WITH per_patient AS (SELECT patient_id, bool_or((lab_values->>'value')::numeric >= $1::numeric) AS uncontrolled FROM treatment_events WHERE brand::text = 'Remibrutinib' AND event_subtype = 'baseline_antihistamine' AND drug_class = 'R06A' AND lab_values->>'assay' = 'UAS7' AND patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)) GROUP BY patient_id) SELECT COUNT(*) FILTER (WHERE uncontrolled)::float / NULLIF(COUNT(*), 0) AS uncontrolled_rate FROM per_patient$kpi$, 2, $note$#1564 region-scoped BR-001 (includes synthetic)$note$),

    -- ---- BR-002 primary: latest quality-flagged survey month IN the region ($1 region) ----
    ('brand_specific_remi_intent_delta_primary_region', $kpi$WITH regional AS (SELECT s.survey_date, s.intent_to_prescribe_change FROM (SELECT * FROM hcp_intent_surveys WHERE is_synthetic = false) s INNER JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = s.hcp_id WHERE s.brand::text = 'Remibrutinib' AND s.response_quality_flag = TRUE AND LOWER(hp.geographic_region::text) = LOWER($1)) SELECT AVG(intent_to_prescribe_change) AS intent_delta FROM regional WHERE DATE_TRUNC('month', survey_date) = (SELECT MAX(DATE_TRUNC('month', survey_date)) FROM regional)$kpi$, 1, $note$#1564 region-scoped BR-002 primary; v_kpi_intent_to_prescribe semantics (quality-flagged monthly avg) at the region's latest survey month; region via hcp_id -> hcp_profiles.geographic_region$note$),
    ('brand_specific_remi_intent_delta_primary_region_include_synthetic', $kpi$WITH regional AS (SELECT s.survey_date, s.intent_to_prescribe_change FROM hcp_intent_surveys s INNER JOIN hcp_profiles hp ON hp.hcp_id = s.hcp_id WHERE s.brand::text = 'Remibrutinib' AND s.response_quality_flag = TRUE AND LOWER(hp.geographic_region::text) = LOWER($1)) SELECT AVG(intent_to_prescribe_change) AS intent_delta FROM regional WHERE DATE_TRUNC('month', survey_date) = (SELECT MAX(DATE_TRUNC('month', survey_date)) FROM regional)$kpi$, 1, $note$#1564 region-scoped BR-002 primary (includes synthetic)$note$),

    -- ---- BR-002 fallback: trailing 90d of data, GLOBAL frontier anchor ($1 region) ----
    ('brand_specific_remi_intent_delta_fallback_region', $kpi$SELECT base.*, (SELECT MAX(survey_date) FROM (SELECT * FROM hcp_intent_surveys WHERE is_synthetic = false) hcp_intent_surveys)::date AS data_through FROM (SELECT AVG(s.intent_to_prescribe_change) AS intent_delta FROM (SELECT * FROM hcp_intent_surveys WHERE is_synthetic = false) s INNER JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = s.hcp_id WHERE s.brand::text = 'Remibrutinib' AND s.survey_date >= (SELECT MAX(survey_date) FROM (SELECT * FROM hcp_intent_surveys WHERE is_synthetic = false) hcp_intent_surveys) - INTERVAL '90 days' AND s.intent_to_prescribe_change IS NOT NULL AND LOWER(hp.geographic_region::text) = LOWER($1)) base$kpi$, 1, $note$#1564 region-scoped BR-002 fallback; mirrors 089 frontier anchor (GLOBAL survey MAX); region via hcp_id -> hcp_profiles.geographic_region$note$),
    ('brand_specific_remi_intent_delta_fallback_region_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(survey_date) FROM hcp_intent_surveys)::date AS data_through FROM (SELECT AVG(s.intent_to_prescribe_change) AS intent_delta FROM hcp_intent_surveys s INNER JOIN hcp_profiles hp ON hp.hcp_id = s.hcp_id WHERE s.brand::text = 'Remibrutinib' AND s.survey_date >= (SELECT MAX(survey_date) FROM hcp_intent_surveys) - INTERVAL '90 days' AND s.intent_to_prescribe_change IS NOT NULL AND LOWER(hp.geographic_region::text) = LOWER($1)) base$kpi$, 1, $note$#1564 region-scoped BR-002 fallback (includes synthetic)$note$),

    -- ---- BR-003 Fabhalta PNH Tested ($1 region) — guard stays TABLE-WIDE ----
    ('brand_specific_fabhalta_pnh_tested_region', $kpi$WITH eligible AS (SELECT DISTINCT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE brand::text = 'Fabhalta' AND primary_diagnosis_code = 'D59.5' AND LOWER(geographic_region::text) = LOWER($1)) SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.patient_id = e.patient_id AND te.event_subtype = 'pnh_flow_cytometry' AND te.loinc_codes && ARRAY['55164-8','35468-8','90735-2','44007-3']))::float / NULLIF(COUNT(*), 0) AS tested_rate, (SELECT COUNT(*) FROM treatment_events WHERE is_synthetic = false AND event_subtype = 'pnh_flow_cytometry')::int AS pnh_events_total FROM eligible e$kpi$, 1, $note$#1564 region-scoped BR-003; region on the D59.5 eligibility cohort (patient_journeys.geographic_region); pnh_events_total stays TABLE-WIDE for the #1116 structural-zero guard$note$),
    ('brand_specific_fabhalta_pnh_tested_region_include_synthetic', $kpi$WITH eligible AS (SELECT DISTINCT patient_id FROM patient_journeys WHERE brand::text = 'Fabhalta' AND primary_diagnosis_code = 'D59.5' AND LOWER(geographic_region::text) = LOWER($1)) SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM treatment_events te WHERE te.patient_id = e.patient_id AND te.event_subtype = 'pnh_flow_cytometry' AND te.loinc_codes && ARRAY['55164-8','35468-8','90735-2','44007-3']))::float / NULLIF(COUNT(*), 0) AS tested_rate, (SELECT COUNT(*) FROM treatment_events WHERE event_subtype = 'pnh_flow_cytometry')::int AS pnh_events_total FROM eligible e$kpi$, 1, $note$#1564 region-scoped BR-003 (includes synthetic)$note$),

    -- ---- BR-004 Kisqali Dx Adoption ($1 region) ----
    -- first-Rx dates stay computed over ALL events (the true first Rx); the
    -- region cut selects patients by their journey's region on the join.
    ('brand_specific_kisqali_dx_adoption_region', $kpi$WITH first_kisqali AS (SELECT te.patient_id, MIN(te.event_date) AS first_rx_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.brand::text = 'Kisqali' AND te.event_type::text = 'prescription' GROUP BY te.patient_id) SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (fk.first_rx_date - pj.journey_start_date)) AS median_days FROM first_kisqali fk INNER JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = fk.patient_id WHERE pj.journey_start_date IS NOT NULL AND fk.first_rx_date >= pj.journey_start_date AND LOWER(pj.geographic_region::text) = LOWER($1)$kpi$, 1, $note$#1564 region-scoped BR-004; region via the existing patient_journeys join (true first-Rx kept global per patient)$note$),
    ('brand_specific_kisqali_dx_adoption_region_include_synthetic', $kpi$WITH first_kisqali AS (SELECT te.patient_id, MIN(te.event_date) AS first_rx_date FROM treatment_events te WHERE te.brand::text = 'Kisqali' AND te.event_type::text = 'prescription' GROUP BY te.patient_id) SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (fk.first_rx_date - pj.journey_start_date)) AS median_days FROM first_kisqali fk INNER JOIN patient_journeys pj ON pj.patient_id = fk.patient_id WHERE pj.journey_start_date IS NOT NULL AND fk.first_rx_date >= pj.journey_start_date AND LOWER(pj.geographic_region::text) = LOWER($1)$kpi$, 1, $note$#1564 region-scoped BR-004 (includes synthetic)$note$),

    -- ---- BR-005 Kisqali Oncologist Reach ($1 region) — both CTEs filtered, GLOBAL frontier ----
    ('brand_specific_kisqali_oncologist_reach_region', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers)::date AS data_through FROM (WITH oncologists AS (SELECT COUNT(DISTINCT hcp_id) AS total FROM (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hcp_profiles WHERE specialty ILIKE '%oncolog%' AND LOWER(geographic_region::text) = LOWER($1)), engaged AS (SELECT COUNT(DISTINCT t.hcp_id) AS total FROM (SELECT * FROM triggers WHERE is_synthetic = false) t INNER JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = t.hcp_id WHERE hp.specialty ILIKE '%oncolog%' AND LOWER(hp.geographic_region::text) = LOWER($1) AND t.brand_id = 'Kisqali' AND t.trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers) - INTERVAL '90 days') SELECT engaged.total::float / NULLIF(oncologists.total, 0) AS reach FROM oncologists, engaged) base$kpi$, 1, $note$#1564 region-scoped BR-005; engaged-in-region / oncologists-in-region (hcp_profiles.geographic_region on BOTH CTEs); mirrors 089 GLOBAL frontier anchor$note$),
    ('brand_specific_kisqali_oncologist_reach_region_include_synthetic', $kpi$SELECT base.*, (SELECT MAX(trigger_timestamp) FROM triggers)::date AS data_through FROM (WITH oncologists AS (SELECT COUNT(DISTINCT hcp_id) AS total FROM hcp_profiles WHERE specialty ILIKE '%oncolog%' AND LOWER(geographic_region::text) = LOWER($1)), engaged AS (SELECT COUNT(DISTINCT t.hcp_id) AS total FROM triggers t INNER JOIN hcp_profiles hp ON hp.hcp_id = t.hcp_id WHERE hp.specialty ILIKE '%oncolog%' AND LOWER(hp.geographic_region::text) = LOWER($1) AND t.brand_id = 'Kisqali' AND t.trigger_timestamp >= (SELECT MAX(trigger_timestamp) FROM triggers) - INTERVAL '90 days') SELECT engaged.total::float / NULLIF(oncologists.total, 0) AS reach FROM oncologists, engaged) base$kpi$, 1, $note$#1564 region-scoped BR-005 (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new statements serve immediately.
NOTIFY pgrst, 'reload schema';
