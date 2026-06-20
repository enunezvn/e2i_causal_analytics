-- ============================================================================
-- Migration 085: #1064 — synthetic-inclusive twin for WS3-BI-003
-- patient_touch_rate. View + kpi_query_registry INSERT only (no schema/data
-- change). Idempotent. Depends on: 050 (base view + registry), 063 (is_synthetic
-- columns), 067 (synthetic-excluding v_patient_eligibility).
-- ----------------------------------------------------------------------------
-- ROOT CAUSE (verified read-only against the deployed/demo supabase 2026-06-20):
-- WS3-BI-003 patient_touch_rate returns null in the demo env. The demo
-- patient_journeys are 100% is_synthetic=true (25,000/25,000), and migration 067
-- CREATE-OR-REPLACE'd ``v_patient_eligibility`` to default-EXCLUDE synthetic rows
-- (``FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false)``), so the
-- eligible cohort is empty -> COUNT(*)=0 -> NULLIF -> touch_rate null.
--
-- The migration-066 bulk synthetic-twin pass only covered query_ids that read a
-- TAGGABLE TABLE directly (it wraps the table in a default-exclude subquery and
-- registers a verbatim ``_include_synthetic`` twin). patient_touch_rate reads a
-- VIEW (v_patient_eligibility), so it had NO twin and ``resolve_kpi_query_id``
-- had nothing to swap to under the ``E2I_KPI_INCLUDE_SYNTHETIC`` demo flag.
--
-- FIX (the issue's option 1 — honest in a demo context; values are clearly
-- synthetic and badged ``data_source="synthetic"`` by the provenance layer):
--   (A) a synthetic-INCLUSIVE eligibility view that is migration 067's
--       v_patient_eligibility VERBATIM minus the two ``WHERE is_synthetic = false``
--       wraps — the ONLY difference is synthetic inclusion, so the twin and base
--       stay structurally identical and cannot drift in their eligibility logic.
--   (B) the ``business_impact_patient_touch_rate_include_synthetic`` registry
--       twin, identical to the 050 base query except it reads the inclusive view.
-- src/kpi/synthetic_mode.py adds ``business_impact_patient_touch_rate`` to
-- SYNTHETIC_TWINNED_QUERY_IDS so resolve_kpi_query_id swaps to this twin under
-- the demo flag (drift-locked by tests/unit/test_kpi/test_synthetic_mode.py,
-- which now parses 066 + 085).
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/085_kpi_patient_touch_rate_include_synthetic.sql
-- ----------------------------------------------------------------------------

-- (A) Synthetic-INCLUSIVE eligibility view. Identical to migration 067's
-- v_patient_eligibility except it reads patient_journeys / triggers RAW (no
-- is_synthetic=false wrap), so the labeled-synthetic demo cohort is eligible.
CREATE OR REPLACE VIEW public.v_patient_eligibility_include_synthetic AS
SELECT DISTINCT pj.patient_id,
    pj.brand,
    (EXISTS ( SELECT 1
           FROM triggers t
          WHERE t.patient_id::text = pj.patient_id::text
            AND (t.delivery_status::text = ANY (ARRAY['delivered'::character varying, 'viewed'::character varying]::text[])))) AS has_delivered_touch
   FROM patient_journeys pj
  WHERE pj.primary_diagnosis_code::text = ANY (ARRAY['C50.1'::character varying, 'C50.2'::character varying, 'C50.9'::character varying, 'D59.5'::character varying, 'L50.1'::character varying, 'L50.8'::character varying, 'L50.9'::character varying]::text[]);

GRANT SELECT ON public.v_patient_eligibility_include_synthetic TO service_role;

-- (B) Synthetic-inclusive twin of business_impact_patient_touch_rate. Byte-for-byte
-- the migration-050 base query except the FROM clause names the inclusive view, so
-- the [0,1] scale / brand-band / patient-level collapse / fail-loud-on-empty
-- semantics are guaranteed identical to the base. $1 = optional brand filter.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_patient_touch_rate_include_synthetic', $kpi$WITH elig AS (SELECT patient_id, bool_or(has_delivered_touch) AS has_delivered_touch FROM public.v_patient_eligibility_include_synthetic WHERE ($1 = '' OR brand::text = $1) GROUP BY patient_id) SELECT COUNT(*) FILTER (WHERE has_delivered_touch)::float / NULLIF(COUNT(*), 0) AS touch_rate FROM elig$kpi$, 1, $note$#1064 demo/review twin of WS3-BI-003 patient_touch_rate: INCLUDES synthetic-tagged rows (reads v_patient_eligibility_include_synthetic instead of the default synthetic-excluding v_patient_eligibility). Resolved ONLY when E2I_KPI_INCLUDE_SYNTHETIC/E2I_INCLUDE_SYNTHETIC is set (synthetic-gold demo). Same FRACTION touched/eligible in [0,1], same fail-loud NULL on an empty eligible cohort, same optional $1 brand band as the base.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; run_migrations.sh / psql --single-transaction owns the outer txn.)
