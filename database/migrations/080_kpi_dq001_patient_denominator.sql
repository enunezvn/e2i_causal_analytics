-- =============================================================================
-- 080: WS1-DQ-001 patient source-coverage denominator must be patient-only.
-- =============================================================================
-- WS1-DQ-001 (patient source coverage) = COUNT(DISTINCT patient) / SUM(target_count).
--
-- BUG: the global/brand query variants summed `target_count` over ALL
-- reference_universe rows (universe_type IN ('patient','hcp')) — diluting patient
-- coverage by the HCP universe, AND inconsistent with the by-region variants,
-- which already filter `universe_type = 'patient'`. The mismatch made per-region
-- coverage and per-brand/global coverage use different denominators (live: global
-- ~0.85 vs region cuts > 100% once the universe was scaled to a realistic size).
--
-- FIX: add `universe_type = 'patient'` to the global/brand denominators (base +
-- the include_synthetic twin), so coverage is patient-over-patient on EVERY cut
-- (global / brand / region). HCP coverage (WS1-DQ-002) already filters
-- universe_type='hcp'; geographic consistency already filters 'patient' — neither
-- is touched. Idempotent (re-running sets the same SQL).
-- =============================================================================

UPDATE kpi_query_registry
SET sql = $SQL$SELECT COUNT(DISTINCT pj.patient_id) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'patient' AND ($1::text IS NULL OR brand::text = $1)), 0) AS total FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj WHERE ($1::text IS NULL OR pj.brand::text = $1)$SQL$
WHERE query_id = 'data_quality_source_coverage_patients';

UPDATE kpi_query_registry
SET sql = $SQL$SELECT COUNT(DISTINCT pj.patient_id) AS covered, COALESCE((SELECT SUM(target_count) FROM reference_universe WHERE universe_type = 'patient' AND ($1::text IS NULL OR brand::text = $1)), 0) AS total FROM patient_journeys pj WHERE ($1::text IS NULL OR pj.brand::text = $1)$SQL$
WHERE query_id = 'data_quality_source_coverage_patients_include_synthetic';
