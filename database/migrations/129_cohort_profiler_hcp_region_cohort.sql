-- ============================================================================
-- Migration 129: region-servable HCP cohort statements for cohort_profiler
-- (#1693 full fix — the follow-up migration 120's header explicitly deferred).
-- ============================================================================
-- WHY: eval turn 4.1 (2026-08-18) asked for "oncologists in the Northeast
-- region" and received the FULL 3,428-HCP / 37,006-TRx cross-region cohort
-- presented as Northeast-scoped (real Northeast universe: 1,086 of 5,000 HCPs
-- — measured against hcp_profiles). Two layers failed: the ask parser never
-- recognized geographic terms (so the #1356 per-criterion accounting could not
-- fire), and no allowlisted statement could bind a region even if it had.
--
-- This migration fixes the DB half: the migration-117 HCP cohort statement
-- with a 5th positional param — $5 region, matched case-insensitively against
-- hcp_profiles.geographic_region (enum region_type: northeast / midwest /
-- south / west), nullable like every other filter. The migration-120 RPC
-- already supports 5-param binds (cap raised 4 -> 6 by #1388).
--
-- ADDITIVE-variant idiom throughout (mig 117/118/120): these are NEW registry
-- rows; the 4-param `cohort_profiler_hcp_trx_cohort[_include_synthetic]` ids
-- are untouched and remain valid, so pre-migration code keeps working and the
-- agent selects the `_region` sibling only when the ask names a region.
-- Synthetic gating: base rows wrap taggable tables in (SELECT * FROM t WHERE
-- is_synthetic = false); the `_include_synthetic` twins are the unwrapped
-- originals, selected by the agent's _profiler_query_id() under the showcase
-- flag (deliberately ABSENT from SYNTHETIC_TWINNED_QUERY_IDS, locked by CI).
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- HCP cohort, region-bound: per-HCP TRx over [$2,$3), TRx > $4, region = $5 ----
    ('cohort_profiler_hcp_trx_cohort_region', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int) SELECT hp.specialty, hp.priority_tier, COUNT(*) AS n_hcps, SUM(c.trx) AS total_trx, MAX(c.trx) AS max_trx FROM cohort c JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = c.hcp_id WHERE ($5::text IS NULL OR LOWER(hp.geographic_region::text) = LOWER($5)) GROUP BY hp.specialty, hp.priority_tier ORDER BY n_hcps DESC$kpi$, 5, $note$#1693 region-bound HCP cohort: params $1 brand (nullable), $2/$3 half-open date window, $4 exclusive TRx floor, $5 region vs hcp_profiles.geographic_region (nullable, case-insensitive); substrate = TRx KPI prescription rows, identical to the mig-117 base$note$),
    ('cohort_profiler_hcp_trx_cohort_region_include_synthetic', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM treatment_events te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int) SELECT hp.specialty, hp.priority_tier, COUNT(*) AS n_hcps, SUM(c.trx) AS total_trx, MAX(c.trx) AS max_trx FROM cohort c JOIN hcp_profiles hp ON hp.hcp_id = c.hcp_id WHERE ($5::text IS NULL OR LOWER(hp.geographic_region::text) = LOWER($5)) GROUP BY hp.specialty, hp.priority_tier ORDER BY n_hcps DESC$kpi$, 5, $note$#1693 region-bound HCP cohort (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';
