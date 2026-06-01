-- ============================================================================
-- Migration 054: #580 - fix DQ-009 (Time-to-Release) unit mismatch (days -> hours)
-- ============================================================================
-- Issue #580 (follow-up to #574/#577). The kpi_query_registry row
-- 'data_quality_time_to_release' (seeded in migration 044) returned
--   SELECT (avg_ttr_hours / 24.0) AS median_ttr_days FROM v_kpi_time_to_release LIMIT 1
-- i.e. DAYS under a "median" name -- while config/kpi_definitions.yaml WS1-DQ-009
-- declares unit: hours with thresholds target=24 / warning=48 / critical=72 (HOURS).
-- Returned value (days) and thresholds (hours) were in different units, so DQ-009 was
-- EXCLUDED from the calculator's _LOWER_IS_BETTER_IDS to dodge a wrong-unit evaluation.
-- This migration aligns the registry to HOURS so DQ-009 can be evaluated lower-is-better
-- against the yaml hour thresholds.
--
-- ANTI-FABRICATION: the view v_kpi_time_to_release.avg_ttr_hours is an AVG (not a
-- median). The old alias "median_ttr_days" was a double misnomer (median + days). The
-- honest alias is avg_ttr_hours -- it is literally AVG(time_to_release_hours), in hours,
-- matching the yaml unit. No "median" anywhere; no "PROXY ... hours/24 -> days"
-- disclaimer needed because there is no longer a stat being mislabeled.
--
-- The companion calculator change ships in the same PR:
--   * src/kpi/calculators/data_quality.py: _calc_time_to_release reads avg_ttr_hours
--     (was median_ttr_days); "WS1-DQ-009" added to _LOWER_IS_BETTER_IDS.
--   * src/kpi/calculator.py: _first_numeric_from_row canonical key median_ttr_days -> avg_ttr_hours.
--
-- Idempotent ON CONFLICT upsert -- safe whether migration 044 has been applied or not
-- (a from-scratch replay reaches: 044 seeds the days-row, then 054 upserts the hours-row
-- -> correct final state). Migration 044 is intentionally left byte-for-byte unchanged
-- (editing an already-applied migration desyncs its checksum; 054 is the sole delivery
-- mechanism, mirroring the 045-053 append-only discipline).
--
-- NOTE: deploy.yml runs migrations only when SUPABASE_DB_URL is set, so this (like
-- 044-053) must be applied to the running self-contained supabase-db for the corrected
-- row to be served; until then DQ-009 keeps returning the pre-existing days value.
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('data_quality_time_to_release', $kpi$SELECT avg_ttr_hours FROM v_kpi_time_to_release LIMIT 1$kpi$, 0, $note$avg TTR in HOURS (v_kpi_time_to_release.avg_ttr_hours = AVG(time_to_release_hours)); lower-is-better; matches kpi_definitions.yaml WS1-DQ-009 thresholds (target 24 / warning 48 / critical 72, hours). #580 unit fix: previously returned avg_ttr_hours/24.0 AS median_ttr_days (DAYS, mislabeled median).$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the corrected statement is served immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
