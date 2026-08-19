-- ============================================================================
-- Migration 130: HCP volume-tier segmentation statements for cohort_profiler
-- (#1736 — make the eval-4.3 promised deliverable real).
-- ============================================================================
-- WHY: eval turn 4.3 ("Segment HCPs by prescription volume into high, medium,
-- and low tiers") promised, after brand selection, "counts per tier" — but the
-- profiler supported only a single min_exclusive TRx threshold (mig-117 $4),
-- so the promised per-tier breakdown was undeliverable on the offered path.
-- The promise survived two eval runs (post1708 + post1730 grades_n3n4 4.3);
-- the ratified direction is to EXTEND the capability (issue #1736, option b;
-- lineage: the #1356 extend:cohort_profiler ruling).
--
-- TIER DEFINITION (data-grounded, measured READ-ONLY 2026-08-19): the data
-- model carries NO stored volume tier — hcp_profiles.prescribing_tier and
-- prescribing_volume are NULL on all 5,000 rows, and priority_tier (populated
-- 1-5) is a DISTINCT targeting concept (ontology: volume + brand affinity +
-- accessibility) that must not be conflated with a volume axis. Tiers are
-- therefore COMPUTED from the same per-HCP TRx substrate as the mig-117 HCP
-- cohort (treatment_events prescription rows — lock-step with the platform
-- TRx KPI): value-based terciles, cut points percentile_disc(1/3) / (2/3) of
-- the per-HCP TRx distribution WITHIN the queried scope (brand / window /
-- threshold / region), buckets assigned BY VALUE so equal TRx always shares a
-- tier (NTILE would split ties arbitrarily). The cut points ride along in
-- every row so the agent can disclose them as measured, scope-relative values.
-- Measured reference points (all-brands, [2026-04-01,2026-07-01), floor 0):
-- 545 HCPs, cuts 2/5, low=219 medium=155 high=171; northeast-scoped cuts are
-- 1/5 (vs 2/5 global) — which is why the region filter MUST precede the cuts.
--
-- ADDITIVE-variant idiom throughout (mig 117/129): these are NEW registry
-- rows; the single-threshold `cohort_profiler_hcp_trx_cohort*` ids are
-- untouched and remain valid — the agent selects a `_volume_tiers` statement
-- only when the ask names volume tiers. Synthetic gating: base rows wrap
-- taggable tables in (SELECT * FROM t WHERE is_synthetic = false); the
-- `_include_synthetic` twins are the unwrapped originals, selected by the
-- agent's _profiler_query_id() under the showcase flag (deliberately ABSENT
-- from SYNTHETIC_TWINNED_QUERY_IDS, locked by CI).
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- volume tiers: per-HCP TRx over [$2,$3), TRx > $4, terciles in-scope ----
    ('cohort_profiler_hcp_volume_tiers', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int), scoped AS (SELECT c.hcp_id, c.trx, hp.specialty FROM cohort c JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = c.hcp_id), cuts AS (SELECT percentile_disc(1.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_low_max, percentile_disc(2.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_medium_max FROM scoped) SELECT CASE WHEN s.trx <= cuts.cut_low_max THEN 'low' WHEN s.trx <= cuts.cut_medium_max THEN 'medium' ELSE 'high' END AS volume_tier, s.specialty, cuts.cut_low_max, cuts.cut_medium_max, COUNT(*) AS n_hcps, SUM(s.trx) AS total_trx, MIN(s.trx) AS min_trx, MAX(s.trx) AS max_trx FROM scoped s CROSS JOIN cuts GROUP BY 1, s.specialty, cuts.cut_low_max, cuts.cut_medium_max ORDER BY 1, n_hcps DESC$kpi$, 4, $note$#1736 HCP volume tiers: params $1 brand (nullable), $2/$3 half-open date window, $4 exclusive TRx floor; buckets = value-based terciles (percentile_disc 1/3 & 2/3) of per-HCP TRx WITHIN this scope, cut points returned per row; substrate = TRx KPI prescription rows, identical to the mig-117 cohort CTE$note$),
    ('cohort_profiler_hcp_volume_tiers_include_synthetic', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM treatment_events te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int), scoped AS (SELECT c.hcp_id, c.trx, hp.specialty FROM cohort c JOIN hcp_profiles hp ON hp.hcp_id = c.hcp_id), cuts AS (SELECT percentile_disc(1.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_low_max, percentile_disc(2.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_medium_max FROM scoped) SELECT CASE WHEN s.trx <= cuts.cut_low_max THEN 'low' WHEN s.trx <= cuts.cut_medium_max THEN 'medium' ELSE 'high' END AS volume_tier, s.specialty, cuts.cut_low_max, cuts.cut_medium_max, COUNT(*) AS n_hcps, SUM(s.trx) AS total_trx, MIN(s.trx) AS min_trx, MAX(s.trx) AS max_trx FROM scoped s CROSS JOIN cuts GROUP BY 1, s.specialty, cuts.cut_low_max, cuts.cut_medium_max ORDER BY 1, n_hcps DESC$kpi$, 4, $note$#1736 HCP volume tiers (includes synthetic)$note$),
    -- ---- region-bound sibling (mig-129 idiom): $5 region INSIDE the scope that feeds the cuts ----
    ('cohort_profiler_hcp_volume_tiers_region', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int), scoped AS (SELECT c.hcp_id, c.trx, hp.specialty FROM cohort c JOIN (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hp ON hp.hcp_id = c.hcp_id WHERE ($5::text IS NULL OR LOWER(hp.geographic_region::text) = LOWER($5))), cuts AS (SELECT percentile_disc(1.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_low_max, percentile_disc(2.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_medium_max FROM scoped) SELECT CASE WHEN s.trx <= cuts.cut_low_max THEN 'low' WHEN s.trx <= cuts.cut_medium_max THEN 'medium' ELSE 'high' END AS volume_tier, s.specialty, cuts.cut_low_max, cuts.cut_medium_max, COUNT(*) AS n_hcps, SUM(s.trx) AS total_trx, MIN(s.trx) AS min_trx, MAX(s.trx) AS max_trx FROM scoped s CROSS JOIN cuts GROUP BY 1, s.specialty, cuts.cut_low_max, cuts.cut_medium_max ORDER BY 1, n_hcps DESC$kpi$, 5, $note$#1736 region-bound HCP volume tiers: params as base + $5 region vs hcp_profiles.geographic_region (nullable, case-insensitive), applied BEFORE the tercile cuts so the tiers are terciles of the region-scoped cohort (measured 2026-08-19: northeast cuts 1/5 vs global 2/5)$note$),
    ('cohort_profiler_hcp_volume_tiers_region_include_synthetic', $kpi$WITH cohort AS (SELECT te.hcp_id, COUNT(*) AS trx FROM treatment_events te WHERE te.event_type::text = 'prescription' AND te.hcp_id IS NOT NULL AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) GROUP BY te.hcp_id HAVING COUNT(*) > $4::int), scoped AS (SELECT c.hcp_id, c.trx, hp.specialty FROM cohort c JOIN hcp_profiles hp ON hp.hcp_id = c.hcp_id WHERE ($5::text IS NULL OR LOWER(hp.geographic_region::text) = LOWER($5))), cuts AS (SELECT percentile_disc(1.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_low_max, percentile_disc(2.0/3.0) WITHIN GROUP (ORDER BY trx) AS cut_medium_max FROM scoped) SELECT CASE WHEN s.trx <= cuts.cut_low_max THEN 'low' WHEN s.trx <= cuts.cut_medium_max THEN 'medium' ELSE 'high' END AS volume_tier, s.specialty, cuts.cut_low_max, cuts.cut_medium_max, COUNT(*) AS n_hcps, SUM(s.trx) AS total_trx, MIN(s.trx) AS min_trx, MAX(s.trx) AS max_trx FROM scoped s CROSS JOIN cuts GROUP BY 1, s.specialty, cuts.cut_low_max, cuts.cut_medium_max ORDER BY 1, n_hcps DESC$kpi$, 5, $note$#1736 region-bound HCP volume tiers (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';
