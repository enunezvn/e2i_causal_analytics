-- ============================================================================
-- Migration 086: #1064 — seed the CSU baseline-antihistamine cohort so BR-001
-- (Remi AH-Uncontrolled %) computes. DATA seed only (INSERT into
-- treatment_events). Idempotent. Depends on: 046 (BR-001 registry), 063
-- (is_synthetic), 066 (BR-001 _include_synthetic twin).
-- ----------------------------------------------------------------------------
-- ROOT CAUSE (verified read-only against the deployed/demo supabase 2026-06-20):
-- BR-001 (brand_specific_remi_ah_uncontrolled) returns null — but, UNLIKE
-- WS3-BI-003, this is NOT synthetic exclusion. BR-001 already HAS an
-- ``_include_synthetic`` twin (migration 066) and is in SYNTHETIC_TWINNED_QUERY_IDS,
-- so under the demo flag it reads treatment_events RAW (synthetic included). It
-- still returns null because the demo DB has ZERO rows matching the BR-001 cohort
-- filter: ``drug_class='R06A'`` = 0 and ``event_subtype='baseline_antihistamine'``
-- = 0 ANYWHERE (synthetic or real), despite 31,862 Remibrutinib treatment_events.
-- The synthetic generator DOES emit these (src/ml/data_generator.py:660-691), but
-- the deployed demo cohort was seeded before/without them. So the fix is DATA, not
-- a twin: generate the missing baseline-antihistamine events for the existing
-- Remibrutinib CSU patients.
--
-- WHY THIS IS HONEST (not a hardcoded KPI value): we generate one real, fully
-- coded baseline-antihistamine PRESCRIPTION event per Remibrutinib CSU patient —
-- ATC R06A drug_class, a real H1-antihistamine RxCUI, and a UAS7 disease-activity
-- reading — mirroring src/ml/data_generator.py:660-691 and the constants in
-- src/ml/synthetic/clinical_codes.py (ANTIHISTAMINES, ANTIHISTAMINE_ATC_CLASS=R06A,
-- UAS7_ASSAY, UAS7_UNCONTROLLED_THRESHOLD=7, UAS7_UNCONTROLLED_PREVALENCE=0.45).
-- Rows are is_synthetic=true (the demo substrate). BR-001 then COMPUTES the
-- realized uncontrolled fraction (UAS7>=7) from these rows — it is not asserted.
--
-- COHORT: the 8,420 Remibrutinib patients carrying a real CSU ICD-10 dx
-- (L50.1/L50.8/L50.9) in patient_journeys. UAS7 is drawn uncontrolled (7..42) with
-- p=0.45 else controlled (0..6), matching the generator's prevalence parameter.
--
-- IDEMPOTENT: keyed on a deterministic treatment_event_id (TE_BR001_<journey_id>)
-- with ON CONFLICT DO NOTHING, and guarded by NOT EXISTS so a patient who already
-- carries a baseline_antihistamine event (e.g. from a future full regenerate) is
-- never double-seeded. Re-applying is a no-op.
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/086_seed_csu_antihistamine_cohort.sql
-- ----------------------------------------------------------------------------

INSERT INTO public.treatment_events (
    treatment_event_id, patient_journey_id, patient_id, hcp_id, event_date,
    event_type, event_subtype, brand, drug_name, drug_ndc, drug_class,
    days_from_diagnosis, loinc_codes, lab_values, sequence_number,
    data_source, source_timestamp, data_split, created_at, is_synthetic
)
SELECT
    'TE_BR001_' || pj.patient_journey_id,
    pj.patient_journey_id,
    pj.patient_id,
    NULL,                                                   -- hcp_id (nullable; not used by BR-001)
    (CURRENT_DATE - ((1 + floor(random() * 180))::int)),    -- a baseline event in the prior ~6 months
    'prescription'::event_type,
    'baseline_antihistamine',
    'Remibrutinib'::brand_type,
    ah.drug_name,
    'RXCUI' || ah.rxcui,
    'R06A',                                                 -- ANTIHISTAMINE_ATC_CLASS
    -((1 + floor(random() * 180))::int),                    -- days_from_diagnosis (pre-dx baseline)
    ARRAY[]::text[],
    jsonb_build_object(
        'assay', 'UAS7',
        'value', CASE WHEN random() < 0.45                  -- UAS7_UNCONTROLLED_PREVALENCE
                      THEN (7 + floor(random() * 36))::int  -- uncontrolled: UAS7 7..42
                      ELSE floor(random() * 7)::int          -- controlled:   UAS7 0..6
                 END,
        'unit', 'score'
    ),
    0,
    pj.data_source,
    pj.source_timestamp,
    pj.data_split,
    NOW(),
    true
FROM public.patient_journeys pj
CROSS JOIN LATERAL (
    -- pick ONE antihistamine consistently (drug_name + matching RxCUI) per row
    SELECT (ARRAY['cetirizine', 'fexofenadine', 'loratadine', 'desloratadine'])[i] AS drug_name,
           (ARRAY['20610', '87636', '28889', '275635'])[i] AS rxcui
    FROM (SELECT (1 + floor(random() * 4))::int AS i) pick
) ah
WHERE pj.brand::text = 'Remibrutinib'
  AND pj.primary_diagnosis_code::text IN ('L50.1', 'L50.8', 'L50.9')
  AND NOT EXISTS (
      SELECT 1 FROM public.treatment_events te
      WHERE te.patient_id = pj.patient_id
        AND te.event_subtype = 'baseline_antihistamine'
  )
ON CONFLICT (treatment_event_id) DO NOTHING;

-- (No COMMIT; run_migrations.sh / psql --single-transaction owns the outer txn.)
