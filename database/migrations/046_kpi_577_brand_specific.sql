-- ============================================================================
-- Migration 046: #577 Tier 2 — wire BR-001 (Remi AH-uncontrolled %) + BR-003
-- (Fabhalta % PNH-tested) by SURGICALLY augmenting the synthetic RWD cohort with
-- the missing clinical concepts, using REAL ontology codes, then registering the
-- two read-only kpi_query allowlist statements.
-- ============================================================================
-- Issue #577 (follow-up to #574). BR-001/BR-003 were fail-loud because the cohort
-- lacked the concepts they measure: there was no antihistamine baseline-therapy and
-- no UAS7 disease-activity reading (BR-001), and the brand diagnosis codes were
-- brand-blind mangled placeholders (ICD10_C50 etc., mostly NULL) with only dummy
-- LOINCs (BR-003). The whole local DB is synthetic RWD, so the HONEST fix is to emit
-- the missing concepts with REAL codes (canonical source: src/ml/synthetic/clinical_codes.py
-- + the durable generator change in src/ml/data_generator.py) and let the calculators
-- compute over them. This is NOT a relabel (the #574 trap — e.g. remapping
-- antihistamine->Remibrutinib): we ADD genuinely-new rows and correct the mangled dx
-- codes, never dress existing mismatched data as the missing concept.
--
-- WHY A SURGICAL SEED (not a generator truncate+reload): treatment_events is 83%
-- Pipeline-B rows (7106 `trx_` of 8607) that reference patient_journeys.patient_id;
-- a full regenerate would orphan them and destroy most of the table. So this migration
-- (a) UPDATEs dx codes in place (preserving every patient_id, zero orphans) and (b)
-- APPENDS the new antihistamine / PNH-flow events. Idempotent (NOT EXISTS guards +
-- deterministic hashtext selection). Snapshot taken before first apply.
--
-- Real codes (all validated during the #577 investigation):
--   CSU dx L50.1/L50.8/L50.9; PNH dx D59.5; HR+ BC dx C50.x.
--   Antihistamine drug_class ATC R06A; RxCUIs cetirizine 20610 / fexofenadine 87636 /
--     loratadine 28889 / desloratadine 275635.
--   UAS7 (range 0-42); uncontrolled = UAS7 >= 7 (EAACI guideline, PMID 34536239).
--   PNH flow-cytometry LOINC 55164-8 / 35468-8 / 90735-2 / 44007-3 (56659-3 is NOT a
--     real LOINC and is deliberately excluded).
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target for this work. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/046_kpi_577_brand_specific.sql
-- ----------------------------------------------------------------------------

-- (A) Correct the brand-blind / mostly-NULL primary_diagnosis_code to REAL ICD-10
--     (in place; preserves patient_id and all referencing treatment_events).
UPDATE public.patient_journeys SET
    primary_diagnosis_code = CASE brand::text
        WHEN 'Remibrutinib' THEN (ARRAY['L50.1','L50.8','L50.9'])[1 + (abs(hashtext(patient_id || 'csu')) % 3)]
        WHEN 'Fabhalta'     THEN 'D59.5'
        WHEN 'Kisqali'      THEN (ARRAY['C50.1','C50.2','C50.9'])[1 + (abs(hashtext(patient_id || 'bc')) % 3)]
        ELSE primary_diagnosis_code END,
    primary_diagnosis_desc = CASE brand::text
        WHEN 'Remibrutinib' THEN 'Chronic spontaneous urticaria'
        WHEN 'Fabhalta'     THEN 'Paroxysmal nocturnal hemoglobinuria'
        WHEN 'Kisqali'      THEN 'Malignant neoplasm of breast'
        ELSE primary_diagnosis_desc END
WHERE brand::text IN ('Remibrutinib','Fabhalta','Kisqali');

-- (B) Emit ONE prior baseline-antihistamine prescription per Remibrutinib (CSU) patient
--     journey, carrying a real RxCUI/ATC drug_class and a UAS7 reading. UAS7 value is
--     deterministic per journey (hashtext) so the ~45% uncontrolled prevalence is stable
--     across re-runs; NOT EXISTS makes the whole insert idempotent.
INSERT INTO public.treatment_events
    (treatment_event_id, patient_journey_id, patient_id, hcp_id, event_date, event_type,
     event_subtype, brand, drug_name, drug_class, days_from_diagnosis, lab_values)
SELECT
    'ah577_' || pj.patient_journey_id,
    pj.patient_journey_id,
    pj.patient_id,
    pj.hcp_id,
    COALESCE(pj.journey_start_date, CURRENT_DATE) - 30,           -- pre-index baseline therapy
    'prescription'::event_type,
    'baseline_antihistamine',
    'Remibrutinib'::brand_type,
    (ARRAY['cetirizine','fexofenadine','loratadine','desloratadine'])[1 + (abs(hashtext(pj.patient_id || 'ah')) % 4)],
    'R06A',                                                       -- ATC antihistamines for systemic use
    -1 * (1 + abs(hashtext(pj.patient_id || 'dfd')) % 180),       -- days_from_diagnosis < 0 (pre-index)
    jsonb_build_object(
        'assay', 'UAS7',
        'value', CASE WHEN (abs(hashtext(pj.patient_id || 'uas7')) % 100) < 45
                      THEN 7  + (abs(hashtext(pj.patient_id || 'uval')) % 36)   -- uncontrolled 7-42
                      ELSE      (abs(hashtext(pj.patient_id || 'uval')) % 7)    -- controlled 0-6
                 END,
        'unit', 'score')
FROM public.patient_journeys pj
WHERE pj.brand::text = 'Remibrutinib'
  AND NOT EXISTS (
      SELECT 1 FROM public.treatment_events te
      WHERE te.patient_id = pj.patient_id AND te.event_subtype = 'baseline_antihistamine');

-- (C) Emit a PNH flow-cytometry lab_test for ~65% of Fabhalta (D59.5) patient journeys,
--     carrying a REAL PNH-flow LOINC. Deterministic 65% membership (hashtext) + NOT EXISTS
--     -> idempotent; the remaining ~35% stay untested so tested/eligible is a real ratio.
INSERT INTO public.treatment_events
    (treatment_event_id, patient_journey_id, patient_id, hcp_id, event_date, event_type,
     event_subtype, brand, loinc_codes, lab_values)
SELECT
    'pnh577_' || pj.patient_journey_id,
    pj.patient_journey_id,
    pj.patient_id,
    pj.hcp_id,
    COALESCE(pj.journey_start_date, CURRENT_DATE),
    'lab_test'::event_type,
    'pnh_flow_cytometry',
    'Fabhalta'::brand_type,
    ARRAY[(ARRAY['55164-8','35468-8','90735-2','44007-3'])[1 + (abs(hashtext(pj.patient_id || 'loinc')) % 4)]],
    jsonb_build_object('assay', 'PNH_clone',
                       'value', round((abs(hashtext(pj.patient_id || 'clone')) % 9500) / 100.0, 2),
                       'unit', '%')
FROM public.patient_journeys pj
WHERE pj.brand::text = 'Fabhalta'
  AND pj.primary_diagnosis_code = 'D59.5'
  AND (abs(hashtext(pj.patient_id || 'tested')) % 100) < 65
  AND NOT EXISTS (
      SELECT 1 FROM public.treatment_events te
      WHERE te.patient_id = pj.patient_id AND te.event_subtype = 'pnh_flow_cytometry');

-- (D) Register the two read-only KPI statements (allowlist; executed only via kpi_query).
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('brand_specific_remi_ah_uncontrolled', $kpi$SELECT COUNT(*) FILTER (WHERE (lab_values->>'value')::numeric >= $1::numeric)::float / NULLIF(COUNT(*), 0) AS uncontrolled_rate FROM treatment_events WHERE brand::text = 'Remibrutinib' AND event_subtype = 'baseline_antihistamine' AND lab_values->>'assay' = 'UAS7'$kpi$, 1, $note$BR-001: % of antihistamine(R06A)-treated CSU patients whose UAS7 >= $1 (uncontrolled; guideline cutoff 7, PMID 34536239). Denominator = patients with a baseline_antihistamine event; NULL (fail-loud) if that cohort is empty.$note$),
    ('brand_specific_fabhalta_pnh_tested', $kpi$WITH eligible AS (SELECT DISTINCT patient_id FROM patient_journeys WHERE brand::text = 'Fabhalta' AND primary_diagnosis_code = 'D59.5') SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM treatment_events te WHERE te.patient_id = e.patient_id AND te.event_subtype = 'pnh_flow_cytometry' AND te.loinc_codes && ARRAY['55164-8','35468-8','90735-2','44007-3']))::float / NULLIF(COUNT(*), 0) AS tested_rate FROM eligible e$kpi$, 0, $note$BR-003: % of PNH-eligible (D59.5) patients with a flow-cytometry lab_test carrying a real PNH LOINC. NULL (fail-loud) if no D59.5 cohort; genuine 0.0 if a cohort exists but none tested.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_ids are callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
