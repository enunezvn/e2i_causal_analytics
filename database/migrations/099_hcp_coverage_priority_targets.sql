-- Migration 099: honest WS3-BI-004 (HCP Coverage)
-- ================================================
-- The KPI was doubly broken on synthetic instances:
--   1. The synthetic loader never wrote hcp_profiles.priority_tier (NULL on
--      every row) -> the denominator (tier 1-2 targets) was empty -> the KPI
--      fail-louded "no data for HCP coverage".
--   2. coverage_status inherited its column DEFAULT TRUE on every row (a
--      100%-coverage artifact, not data).
--   3. The registry numerator counted ALL covered HCPs while the denominator
--      counted only tier 1-2 — fixing the data alone would produce a nonsense
--      >100% "coverage". The KPI's definition is "percentage of PRIORITY HCPs
--      with active engagement" (kpi_definitions.yaml), so the numerator must be
--      scoped to the same tier 1-2 universe.
--
-- The generator now emits both columns (src/ml/synthetic/generators/
-- hcp_generator.py); this migration re-scopes the registry queries and heals
-- the live rows the same deterministic way the generator would.

-- 1) Registry: numerator scoped to priority targets (base + synthetic twin).
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('business_impact_hcp_coverage', $kpi$SELECT COUNT(CASE WHEN coverage_status = true AND priority_tier <= 2 THEN 1 END)::float / NULLIF(COUNT(CASE WHEN priority_tier <= 2 THEN 1 END), 0) AS coverage FROM (SELECT * FROM hcp_profiles WHERE is_synthetic = false) hcp_profiles$kpi$, 0, $note$M099: numerator scoped to tier<=2 priority targets (definition: covered priority HCPs / priority HCPs)$note$),
    ('business_impact_hcp_coverage_include_synthetic', $kpi$SELECT COUNT(CASE WHEN coverage_status = true AND priority_tier <= 2 THEN 1 END)::float / NULLIF(COUNT(CASE WHEN priority_tier <= 2 THEN 1 END), 0) AS coverage FROM hcp_profiles$kpi$, 0, $note$M099 opt-in: INCLUDES synthetic; numerator scoped to tier<=2 priority targets$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- 2) Data heal for pre-099 rows, mirroring the generator's logic:
--    priority_tier = patient-volume quintile (1 = highest-volume fifth).
--    Deterministic (volume desc, hcp_id tiebreak); only fills NULLs, so a
--    post-099 reseed (generator-provided tiers) is never overwritten.
WITH ranked AS (
    SELECT hcp_id,
           NTILE(5) OVER (ORDER BY total_patient_volume DESC, hcp_id) AS tier
    FROM hcp_profiles
)
UPDATE hcp_profiles hp
SET priority_tier = r.tier
FROM ranked r
WHERE r.hcp_id = hp.hcp_id
  AND hp.priority_tier IS NULL;

--    coverage_status: tier-weighted engagement (field force prioritizes tier
--    1-2), deterministic per hcp_id so the heal is idempotent. Applies only to
--    synthetic rows still carrying the all-TRUE default artifact; real
--    (is_synthetic = false) rows are never touched.
UPDATE hcp_profiles
SET coverage_status = (abs(hashtext(hcp_id)) % 100) <
    CASE priority_tier
        WHEN 1 THEN 95
        WHEN 2 THEN 90
        WHEN 3 THEN 70
        WHEN 4 THEN 50
        ELSE 30
    END
WHERE is_synthetic = true;
