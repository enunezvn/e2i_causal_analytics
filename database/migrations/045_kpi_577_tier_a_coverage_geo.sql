-- ============================================================================
-- Migration 045: #577 Tier A — wire DQ-002 (HCP source coverage) + DQ-006
-- (geographic consistency) to REAL data via the kpi_query allowlist.
-- ============================================================================
-- Issue #577 (follow-up to #574). Both metrics previously raised a fail-loud
-- RuntimeError because their calculators referenced sources that do not exist
-- (DQ-002 claimed a missing ``reference_hcps`` table; DQ-006 joined
-- ``agent_activities.hcp_id`` which has no such column). The real sources DO
-- exist — they were never wired:
--
--   * DQ-002: covered HCPs = hcp_profiles.coverage_status = true; reference
--     universe = SUM(reference_universe.target_count) WHERE universe_type='hcp'.
--     This is a GLOBAL coverage ratio (no brand param): hcp_profiles has no brand
--     column, so the numerator is not brand-attributable — banding only the
--     denominator by brand would be an incoherent ratio (global covered HCPs over
--     a single brand's target universe), so we do not offer a brand param here.
--     Per-brand HCP coverage needs a brand-attributable coverage source (future).
--   * DQ-006: the AUTHORITATIVE formula (config/kpi_definitions.yaml +
--     docs/data/06-KPI-REFERENCE.md) is max_region(|share_source - share_universe|),
--     NOT the pre-#574 region-self-consistency the broken join implied. Source
--     distribution = patient_journeys by geographic_region; universe = the
--     universe_type='patient' rows of reference_universe by region.
--
-- Both statements are read-only SELECT/WITH (satisfy kpi_query_registry's CHECK),
-- exposed only through the kpi_query SECURITY DEFINER RPC (migration 044).
--
-- NOTE: deploy.yml runs migrations only when SUPABASE_DB_URL is set, so this must
-- be applied to the target Supabase for the two query_ids to exist there; until
-- then the two calculators fail-closed (pre-existing state). The local
-- self-contained supabase is the faithful target for this work.
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('data_quality_source_coverage_hcps', $kpi$SELECT (SELECT COUNT(DISTINCT hcp_id) FROM hcp_profiles WHERE coverage_status = true) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'hcp'), 0) AS total$kpi$, 0, $note$GLOBAL HCP coverage (no brand param): covered = distinct HCPs with coverage_status=true; denominator = SUM(reference_universe.target_count) for universe_type='hcp'. hcp_profiles has no brand column so the numerator is not brand-attributable; banding only the denominator would be an incoherent ratio, so DQ-002 is global-only. Per-brand HCP coverage needs a brand-attributable coverage source (future #577 follow-up).$note$),
    ('data_quality_geographic_consistency', $kpi$WITH src AS (SELECT geographic_region::text AS region, COUNT(*)::numeric AS n FROM patient_journeys WHERE geographic_region IS NOT NULL AND ($1::text IS NULL OR brand::text = $1) GROUP BY geographic_region), src_share AS (SELECT region, n / NULLIF(SUM(n) OVER (), 0) AS share FROM src), uni AS (SELECT region::text AS region, SUM(total_count)::numeric AS n FROM reference_universe WHERE universe_type = 'patient' AND ($1::text IS NULL OR brand::text = $1) GROUP BY region), uni_share AS (SELECT region, n / NULLIF(SUM(n) OVER (), 0) AS share FROM uni) SELECT MAX(ABS(COALESCE(s.share, 0) - COALESCE(u.share, 0))) AS max_gap FROM src_share s FULL OUTER JOIN uni_share u ON s.region = u.region$kpi$, 1, $note$WS1-DQ-006 authoritative formula max_region(|share_source - share_universe|): source share from patient_journeys by geographic_region, universe share from reference_universe(universe_type='patient') by region; region cast to text (geographic_region is region_type enum). FULL OUTER JOIN so a region present in only one side still contributes its full share as the gap. $1 optionally bands by brand.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_ids are callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
