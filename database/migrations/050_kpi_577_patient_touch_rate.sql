-- ============================================================================
-- Migration 050: #577 WS3-BI-003 — wire patient_touch_rate to REAL data via a
-- code-anchored eligibility view + the kpi_query allowlist. View + registry
-- INSERT only — NO schema-column or data-value change (a pure view+registry
-- migration like 047).
-- ============================================================================
-- Issue #577 (follow-up to #574). WS3-BI-003 previously raised a fail-loud
-- RuntimeError ("patient_journeys has no is_eligible column"). An adversarial
-- design-review (verified end-to-end against the live self-contained supabase)
-- established that no is_eligible column is needed: eligibility is DERIVED from
-- the real primary_diagnosis_code already in patient_journeys.
--
-- WHY THIS IS HONEST (and not the #574 relabel trap):
--   * ELIGIBLE = patient carries a brand-qualifying REAL ICD-10 diagnosis. The
--     7-code set is the EMITTED source-of-truth from
--     src/ml/synthetic/clinical_codes.py BRAND_DIAGNOSIS (replicated verbatim in
--     migration 046): Kisqali = C50.1/C50.2/C50.9 (HR+ breast cancer),
--     Fabhalta = D59.5 (PNH; D59.6 is deliberately EXCLUDED — it is NOT
--     Marchiafava-Micheli PNH), Remibrutinib = L50.1/L50.8/L50.9 (CSU). This is
--     the same honest pattern accepted for BR-003 (derive PNH eligibility from
--     real D59.5 dx, not a blanket flag), and NOT the ~93%-NULL journey_status.
--     Anchored on the dx CODE, not the brand LABEL, so it survives brand drift.
--     Exact IN() membership (verified == prefix-LIKE over live data, 0 whitespace
--     contamination) — never a LIKE 'C50.%' prefix that would silently widen
--     eligibility if a non-emitted C50.x code ever appears.
--   * TOUCH = a trigger that ACTUALLY reached the patient (delivery_status IN
--     ('delivered','viewed')). This is the load-bearing honesty lever: counting
--     "any trigger" is the DEGENERATE/dishonest framing (99.5% live, since
--     nearly every patient has some trigger row); delivered-only is 90.7% and
--     reflects a real 236-patient reach gap (triggers that pended/failed/expired
--     and never became a touchpoint). The faithful e2e proves any-trigger >
--     delivered by a material margin, so the chosen definition is demonstrably
--     the non-degenerate one.
--   * On the CURRENT synthetic data the eligibility filter is near-universal
--     (2700 of 2702 journeys carry a qualifying dx; the only 2 excluded rows
--     have a NULL dx), so it does NOT presently narrow a clinical sub-population
--     — the delivered-touch definition is the operative non-degeneracy lever.
--     The code-anchored eligibility is kept for future-proofing (when non-target
--     dx codes appear) and BR-003 parity, not as a current discriminator.
--
-- SCALE / THRESHOLD CONTRACT: touch_rate is the FRACTION touched/eligible in
-- [0,1] (sibling parity with business_impact_conversion_rate / _hcp_coverage,
-- which return ::float / NULLIF(...,0)). The config/kpi_definitions.yaml target
-- (0.40) is a fraction, so returning 100*ratio would be a mis-scale bug. NOTE:
-- this delivered-touch definition intentionally differs from the YAML "formula"
-- shorthand (`patients_with_trigger / eligible_patients`), which literally
-- describes the rejected degenerate any-trigger ratio; threshold recalibration
-- (live value 0.907 reads GOOD by >2x the 0.40 target) is a separate follow-up,
-- deliberately NOT changed here.
--
-- NO DATA CHANGE: dx codes are already real + brand-aligned (migration 046) and
-- triggers.delivery_status already discriminates reach, so there is no surgical
-- seed, no /tmp snapshot, and no src/ml/data_generator.py mirror (the brand->
-- ICD-10 mapping the view needs already lives in clinical_codes.BRAND_DIAGNOSIS).
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the faithful
-- target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/050_kpi_577_patient_touch_rate.sql
-- ----------------------------------------------------------------------------

-- (A) Code-anchored patient eligibility + delivered-touch flag.
-- ELIGIBLE = primary_diagnosis_code in the brand-qualifying ICD-10 set (the 7
-- EMITTED codes). has_delivered_touch = patient has >=1 trigger that actually
-- reached them. brand is carried as a column so the registry query can apply an
-- optional brand band. (Live: zero multi-brand eligible patients, so the DISTINCT
-- yields one row per eligible patient_id = 2700 rows. The registry query (B) still
-- collapses to patient-level before counting, so a future multi-brand patient
-- cannot double-count the denominator.)
CREATE OR REPLACE VIEW public.v_patient_eligibility AS
SELECT DISTINCT
    pj.patient_id,
    pj.brand,
    EXISTS (
        SELECT 1 FROM public.triggers t
        WHERE t.patient_id = pj.patient_id
          AND t.delivery_status IN ('delivered', 'viewed')
    ) AS has_delivered_touch
FROM public.patient_journeys pj
WHERE pj.primary_diagnosis_code IN
    ('C50.1', 'C50.2', 'C50.9', 'D59.5', 'L50.1', 'L50.8', 'L50.9');

-- The kpi_query RPC runs SECURITY DEFINER as the owner, so the registry SELECT
-- reads this view regardless of caller grants; this explicit grant is
-- defense-in-depth (it is STRICTER than the default Supabase view grants — it
-- does NOT expose the view to anon; the RPC is the only client-reachable surface).
GRANT SELECT ON public.v_patient_eligibility TO service_role;

-- (B) Register the read-only WS3-BI-003 statement (allowlist; executed only via kpi_query).
-- $1 = optional brand filter ('' => all eligible patients across all brands;
-- e.g. 'Fabhalta' => that brand band only). Empty-string sentinel idiom
-- (047/048/049 standard); the param is ALWAYS a non-null TEXT, never NULL.
-- max_params = 1 is the EXACT required arity (the kpi_query RPC raises on a
-- count mismatch) — the calculator always passes exactly one element.
-- The CTE collapses to patient-level (GROUP BY patient_id, bool_or over the
-- brand-independent touch flag) BEFORE counting, so the denominator is distinct
-- ELIGIBLE PATIENTS — never patient×brand pairs. On current data this is a no-op
-- (one eligible row per patient; rate unchanged at 0.9074), but it makes the
-- metric robust if a patient ever becomes eligible for multiple brands.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_patient_touch_rate', $kpi$WITH elig AS (SELECT patient_id, bool_or(has_delivered_touch) AS has_delivered_touch FROM public.v_patient_eligibility WHERE ($1 = '' OR brand::text = $1) GROUP BY patient_id) SELECT COUNT(*) FILTER (WHERE has_delivered_touch)::float / NULLIF(COUNT(*), 0) AS touch_rate FROM elig$kpi$, 1, $note$WS3-BI-003 patient_touch_rate: FRACTION of code-anchored ELIGIBLE patients (primary_diagnosis_code in the brand-qualifying ICD-10 set, via v_patient_eligibility — NOT the absent is_eligible flag #574, NOT a brand-label match) who have >=1 DELIVERED trigger (delivery_status IN ('delivered','viewed') — an actual touchpoint; pending/failed/expired excluded, so NOT the degenerate any-trigger=99.5% relabel). $1 = optional brand filter ('' = all brands; e.g. 'Fabhalta'). Returns touch_rate as a [0,1] fraction (sibling parity with conversion_rate/hcp_coverage); NULL (fail-loud) when the eligible denominator is empty; a genuine 0.0 is legitimate. Live: overall 2450/2700=0.9074, Fabhalta 0.8944, Kisqali 0.9150, Remibrutinib 0.9129.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
