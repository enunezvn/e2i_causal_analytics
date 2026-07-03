-- ============================================================================
-- Migration 091: #1116 — BR-003 (Fabhalta % PNH Tested) structural-zero guard.
-- ----------------------------------------------------------------------------
-- ROOT CAUSE (adversarially verified 2026-07-03, issue #1116): BR-003 rendered a
-- plausible-real 0.0% -> CRITICAL, but the numerator was STRUCTURALLY zero — no
-- pnh_flow_cytometry event existed anywhere in treatment_events, while the D59.5
-- denominator held 8,412 patients. The pnh emission lived only in the LEGACY
-- generator (src/ml/data_generator.py:692) and one-shot migration 046 block C,
-- whose rows did not survive later full regenerates; the ACTIVE generator
-- (src/ml/synthetic/generators/treatment_generator.py) never emitted the concept.
-- BR-001 survived only because migration 086 re-seeded its cohort; BR-003 had no
-- analogous re-seed.
--
-- THIS MIGRATION (registry-only; no data writes): extend the two BR-003 registry
-- statements to ALSO return ``pnh_events_total`` — the table-wide count of
-- pnh_flow_cytometry events (synthetic-excluded in the base statement, raw in the
-- _include_synthetic twin, mirroring migration 066's gating). The calculator
-- (src/kpi/calculators/brand_specific.py::_calc_fabhalta_pnh_tested) fails loud
-- (-> UNKNOWN) when tested_rate = 0.0 AND pnh_events_total = 0: a concept absent
-- from the entire table is a substrate/pipeline coverage gap, not a business 0%.
-- A GENUINE 0% (events exist in the table, none for the eligible cohort) still
-- returns 0.0 -> CRITICAL — the guard cannot mask a real zero reading.
--
-- The durable numerator fix is the generator change shipped with this migration
-- (TreatmentGenerator emits one deterministic pnh_flow_cytometry lab_test per
-- ~65% of D59.5 Fabhalta patients); a treatment_events reseed materializes it:
--   python scripts/load_synthetic_data.py --anchor-to-now
--
-- NOTE: if migration 066 is ever REGENERATED from its codegen and re-applied, it
-- overwrites these two statements with the single-column originals — re-apply
-- this migration afterwards (migrations are ordered, so a full chain replay
-- ends in the correct state).
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/091_kpi_br003_structural_zero_guard.sql
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('brand_specific_fabhalta_pnh_tested', $kpi$WITH eligible AS (SELECT DISTINCT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE brand::text = 'Fabhalta' AND primary_diagnosis_code = 'D59.5') SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.patient_id = e.patient_id AND te.event_subtype = 'pnh_flow_cytometry' AND te.loinc_codes && ARRAY['55164-8','35468-8','90735-2','44007-3']))::float / NULLIF(COUNT(*), 0) AS tested_rate, (SELECT COUNT(*) FROM treatment_events WHERE is_synthetic = false AND event_subtype = 'pnh_flow_cytometry')::int AS pnh_events_total FROM eligible e$kpi$, 0, $note$BR-003 (#1116): tested/eligible over the D59.5 cohort (synthetic-excluded, M4) + pnh_events_total = table-wide pnh_flow_cytometry count for the structural-zero fail-loud guard. NULL tested_rate (fail-loud) if no D59.5 cohort; pnh_events_total=0 with a populated cohort -> calculator raises (coverage gap, not a business 0%).$note$),
    ('brand_specific_fabhalta_pnh_tested_include_synthetic', $kpi$WITH eligible AS (SELECT DISTINCT patient_id FROM patient_journeys WHERE brand::text = 'Fabhalta' AND primary_diagnosis_code = 'D59.5') SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM treatment_events te WHERE te.patient_id = e.patient_id AND te.event_subtype = 'pnh_flow_cytometry' AND te.loinc_codes && ARRAY['55164-8','35468-8','90735-2','44007-3']))::float / NULLIF(COUNT(*), 0) AS tested_rate, (SELECT COUNT(*) FROM treatment_events WHERE event_subtype = 'pnh_flow_cytometry')::int AS pnh_events_total FROM eligible e$kpi$, 0, $note$BR-003 (#1116) opt-in twin: INCLUDES synthetic (demo/validation runs). Same structural-zero guard column as the base statement.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the updated statements serve immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
