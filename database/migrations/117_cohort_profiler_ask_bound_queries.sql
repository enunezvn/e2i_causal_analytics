-- ============================================================================
-- 117_cohort_profiler_ask_bound_queries.sql
-- Ask-bound allowlist statements for the cohort_profiler agent (#1356,
-- parts 1 + 2 of the ratified 2026-07-29 `extend:cohort_profiler` ruling).
--
-- WHY: the 2026-07-29 #1337 benchmark (q11/q15, both surfaces) confirmed the
-- profiler was query-INSENSITIVE — the brand and every inclusion criterion in
-- the ask were ignored, and no agent could serve HCP-entity cohorts with
-- quantitative KPI thresholds. These two statement families let the agent BIND
-- the ask's parameters ($1..$N) through the migration-044 kpi_query allowlist
-- RPC (never raw SQL from the agent):
--
--   * cohort_profiler_hcp_trx_cohort — per-HCP TRx aggregation over an
--     explicit half-open [$2, $3) window with a strict > $4 threshold,
--     joined to hcp_profiles for the specialty / priority-tier segment axes.
--     Substrate = treatment_events prescription rows, IDENTICAL to the
--     platform TRx KPI (business_impact_trx), so HCP cohort numbers stay in
--     lock-step with the KPI dashboard. Rows: one per (specialty,
--     priority_tier) group with n_hcps / total_trx / max_trx — the agent
--     marginalizes in Python; bounded cardinality regardless of threshold.
--   * cohort_profiler_patient_criteria_profile — the mig-105 NRx breakdown
--     (prescription events, sequence 1, most recent 30 days of data) joined to
--     patient_journeys to bind servable inclusion criteria: brand ($1) and
--     age-at-diagnosis bounds ($2 exclusive min, $3 exclusive max —
--     age_at_diagnosis is populated on all rows, verified READ-ONLY
--     2026-07-30). Grouped by (segment_assignment, prior_therapy_lines) so one
--     call yields the criteria-bound headline + both segment-axis marginals.
--     The `_windowed` sibling (codex iter-2) swaps the 30-day anchor for the
--     ask's explicit [$2,$3) window with $4 = exclusive min age (the RPC's
--     4-param cap means max-age cannot also bind there — the agent discloses
--     it as NOT applied for that combo).
--
-- Criteria the data model can NOT serve (e.g. "diagnosed in 2024": zero
-- 'diagnosis' events in treatment_events; journey_start_date is only a
-- documented proxy — see the 044 kisqali_dx_adoption note) get NO statement
-- here on purpose: the agent fails closed honestly instead of approximating.
--
-- Synthetic gating follows the ADDITIVE-variant idiom (mig 077/084/105/116):
-- base statements wrap taggable tables in (SELECT * FROM t WHERE
-- is_synthetic = false); the `_include_synthetic` twins are the unwrapped
-- originals. These ids are deliberately ABSENT from
-- SYNTHETIC_TWINNED_QUERY_IDS (locked to migrations 066/085/095 by CI) — the
-- agent's _profiler_query_id() appends the suffix under the showcase flag,
-- exactly like synthetic_mode.region_query_id.
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- HCP-entity cohort: per-HCP TRx over [$2,$3) with TRx > $4 ----
    ('cohort_profiler_hcp_trx_cohort', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int) SELECT hp.specialty, hp.priority_tier, COUNT(*) AS n_hcps, SUM(c.trx) AS total_trx, MAX(c.trx) AS max_trx FROM cohort c JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = c.hcp_id GROUP BY hp.specialty, hp.priority_tier ORDER BY n_hcps DESC$kpi$, 4, $note$#1356 HCP cohort: params $1 brand (nullable), $2/$3 half-open date window, $4 exclusive TRx floor; substrate = TRx KPI prescription rows$note$),
    ('cohort_profiler_hcp_trx_cohort_include_synthetic', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM treatment_events te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int) SELECT hp.specialty, hp.priority_tier, COUNT(*) AS n_hcps, SUM(c.trx) AS total_trx, MAX(c.trx) AS max_trx FROM cohort c JOIN hcp_profiles hp ON hp.hcp_id = c.hcp_id GROUP BY hp.specialty, hp.priority_tier ORDER BY n_hcps DESC$kpi$, 4, $note$#1356 HCP cohort (includes synthetic)$note$),
    -- ---- Patient criteria-bound profile: brand + age bounds, both axes ----
    ('cohort_profiler_patient_criteria_profile', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= (SELECT MAX(event_date) FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) t WHERE t.event_type::text = 'prescription') - INTERVAL '30 days' AND ($1::text IS NULL OR te.brand::text = $1) AND ($2::int IS NULL OR pj.age_at_diagnosis > $2::int) AND ($3::int IS NULL OR pj.age_at_diagnosis < $3::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 3, $note$#1356 criteria-bound patient profile: params $1 brand (nullable), $2 exclusive min age, $3 exclusive max age (both nullable); mig-105 NRx window anchor$note$),
    ('cohort_profiler_patient_criteria_profile_include_synthetic', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= (SELECT MAX(event_date) FROM treatment_events t WHERE t.event_type::text = 'prescription') - INTERVAL '30 days' AND ($1::text IS NULL OR te.brand::text = $1) AND ($2::int IS NULL OR pj.age_at_diagnosis > $2::int) AND ($3::int IS NULL OR pj.age_at_diagnosis < $3::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 3, $note$#1356 criteria-bound patient profile (includes synthetic)$note$),
    -- ---- Patient criteria-bound profile, WINDOWED sibling (codex iter-2) ----
    -- Binds the ask's explicit [$2,$3) window in place of the 30-day anchor.
    -- The kpi_query RPC caps at 4 positional params, so only the MIN age bound
    -- ($4, exclusive) can ride along — a max-age bound in a windowed ask is
    -- disclosed as NOT applied by the agent, never silently dropped.
    ('cohort_profiler_patient_criteria_profile_windowed', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) AND ($4::int IS NULL OR pj.age_at_diagnosis > $4::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 4, $note$#1356 iter-2 windowed criteria profile: params $1 brand (nullable), $2/$3 half-open date window, $4 exclusive min age (nullable)$note$),
    ('cohort_profiler_patient_criteria_profile_windowed_include_synthetic', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) AND ($4::int IS NULL OR pj.age_at_diagnosis > $4::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 4, $note$#1356 iter-2 windowed criteria profile (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';
