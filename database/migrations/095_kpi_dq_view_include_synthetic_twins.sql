-- ============================================================================
-- Migration 095: synthetic-inclusive twins + deterministic re-registration for
-- the four view-backed WS1 data-quality KPIs (DQ-003 cross_source_match,
-- DQ-004 stacking_lift, DQ-007 data_lag, DQ-009 time_to_release), plus the
-- etl_pipeline_metrics status backfill DQ-009 additionally needs.
-- Registry rows + one scoped data fix only (no schema change). Idempotent.
-- Depends on: 050 (base registry rows), 063 (is_synthetic columns),
-- 066 (synthetic-exclusion gate + twin family), 067 (synthetic-excluding views).
-- ----------------------------------------------------------------------------
-- ROOT CAUSE (verified read-only against the deployed supabase 2026-07-06):
-- the four KPIs read views (v_kpi_cross_source_match / v_kpi_stacking_lift /
-- v_kpi_data_lag / v_kpi_time_to_release) that migration 067 made
-- default-EXCLUDE synthetic rows. On a synthetic-gold instance (100% of
-- data_source_tracking / patient_journeys / etl_pipeline_metrics rows are
-- is_synthetic=true) the views are empty, and — exactly like #1064 /
-- migration 085's patient_touch_rate — the migration-066 bulk twin pass only
-- covered TABLE-reading query_ids, so these four had no _include_synthetic
-- twin for resolve_kpi_query_id to swap to. Result: /data-quality renders
-- "No data" for all four despite 50,000 fresh rows per source table.
--
-- WHY THE BASE ROWS ARE RE-REGISTERED TOO (not just twinned): the 050 base
-- statements were `SELECT <col> FROM <view> LIMIT 1` with no ORDER BY over
-- per-(date[, source/pipeline]) grouped views — a NONDETERMINISTIC single
-- group's value posing as the KPI. Base and twin are re-registered together as
-- the same deterministic trailing-30-day aggregate, anchored to the data
-- frontier (MAX date in the underlying table — the migration-089 convention),
-- differing ONLY in the 066-style `(SELECT * FROM <t> WHERE is_synthetic =
-- false)` wrap. Byte-identical modulo the wrap, so base and twin cannot drift
-- in their KPI semantics. Result-column names/units are unchanged
-- (match_rate, lift_score, median_lag_days, avg_ttr_hours) — the
-- src/kpi/calculators/data_quality.py contracts hold verbatim.
--
-- DQ-009's SECOND blocker: v_kpi_time_to_release (and the re-registered query
-- below) filter status='success' per the e2i_ml_complete_v3_schema.sql
-- contract ('success'|'partial'|'failed'), but the synthetic generator
-- (coverage_tables_generator.py) wrote status='completed' on every row — an
-- unsanctioned value. Fixed at the source (generator now writes 'success')
-- and backfilled here, scoped to is_synthetic=true so real pipeline rows are
-- never rewritten. The health-score consumer (_map_pipeline_status) treats
-- both spellings as non-failed, so the backfill is behavior-neutral there.
--
-- src/kpi/synthetic_mode.py adds the four base ids to
-- SYNTHETIC_TWINNED_QUERY_IDS (drift-locked by
-- tests/unit/test_kpi/test_synthetic_mode.py, which now parses 066 + 085 + 095).
--
-- deploy.yml SKIPS migrations; the local self-contained supabase is the
-- faithful target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/095_kpi_dq_view_include_synthetic_twins.sql
-- ----------------------------------------------------------------------------

-- (A) DQ-009 data backfill: retag the generator's unsanctioned status value on
-- synthetic rows only. Idempotent (second run matches zero rows).
UPDATE public.etl_pipeline_metrics
SET status = 'success'
WHERE status = 'completed'
  AND is_synthetic = true;

-- (B) The four KPI statements, base (synthetic-excluding wrap) + twin
-- (verbatim minus the wrap). ON CONFLICT upserts so the 050 base rows are
-- re-registered in place.

-- WS1-DQ-003 Cross-source Match Rate: records-weighted match ratio over the
-- trailing 30 days at the data frontier.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('data_quality_cross_source_match',
     $kpi$SELECT (SUM(records_matched)::numeric / NULLIF(SUM(records_received), 0))::float AS match_rate FROM (SELECT * FROM public.data_source_tracking WHERE is_synthetic = false) dst WHERE tracking_date >= (SELECT MAX(tracking_date) FROM (SELECT * FROM public.data_source_tracking WHERE is_synthetic = false) dst2) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-003 base (re-registered by 095): records-weighted cross-source match rate over the trailing 30 days anchored at the non-synthetic data frontier. Replaces the 050 `SELECT match_rate FROM v_kpi_cross_source_match LIMIT 1` — a nondeterministic single (date, source) row with no ORDER BY. Synthetic-excluding (066-style wrap); byte-identical to its _include_synthetic twin modulo the wrap.$note$),
    ('data_quality_cross_source_match_include_synthetic',
     $kpi$SELECT (SUM(records_matched)::numeric / NULLIF(SUM(records_received), 0))::float AS match_rate FROM public.data_source_tracking dst WHERE tracking_date >= (SELECT MAX(tracking_date) FROM public.data_source_tracking dst2) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-003 demo/review twin: INCLUDES synthetic-tagged rows (reads data_source_tracking raw). Resolved ONLY when E2I_KPI_INCLUDE_SYNTHETIC/E2I_INCLUDE_SYNTHETIC is set (synthetic-gold showcase). Same records-weighted trailing-30d match rate, same match_rate column, as the 095 base.$note$),

-- WS1-DQ-004 Stacking Lift: mean stacking lift over the trailing 30 days at
-- the data frontier (fractional, e.g. 0.175 = 17.5%).
    ('data_quality_stacking_lift',
     $kpi$SELECT AVG(stacking_lift_percentage)::float AS lift_score FROM (SELECT * FROM public.data_source_tracking WHERE is_synthetic = false) dst WHERE tracking_date >= (SELECT MAX(tracking_date) FROM (SELECT * FROM public.data_source_tracking WHERE is_synthetic = false) dst2) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-004 base (re-registered by 095): mean stacking_lift_percentage over the trailing 30 days anchored at the non-synthetic data frontier. Replaces the 050 `SELECT avg_lift_pct AS lift_score FROM v_kpi_stacking_lift LIMIT 1` — a nondeterministic single tracking_date with no ORDER BY. Synthetic-excluding (066-style wrap); byte-identical to its _include_synthetic twin modulo the wrap.$note$),
    ('data_quality_stacking_lift_include_synthetic',
     $kpi$SELECT AVG(stacking_lift_percentage)::float AS lift_score FROM public.data_source_tracking dst WHERE tracking_date >= (SELECT MAX(tracking_date) FROM public.data_source_tracking dst2) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-004 demo/review twin: INCLUDES synthetic-tagged rows (reads data_source_tracking raw). Resolved ONLY when E2I_KPI_INCLUDE_SYNTHETIC/E2I_INCLUDE_SYNTHETIC is set (synthetic-gold showcase). Same trailing-30d mean lift, same lift_score column, as the 095 base.$note$),

-- WS1-DQ-007 Data Lag (Median): true median lag in days over the trailing 30
-- days at the data frontier (a real percentile over rows — the 050 statement
-- read ONE arbitrary (date, source) group's median from v_kpi_data_lag).
    ('data_quality_data_lag',
     $kpi$SELECT (percentile_cont(0.5) WITHIN GROUP (ORDER BY data_lag_hours) / 24.0)::float AS median_lag_days FROM (SELECT * FROM public.patient_journeys WHERE is_synthetic = false) pj WHERE data_lag_hours IS NOT NULL AND created_at >= (SELECT MAX(created_at) FROM (SELECT * FROM public.patient_journeys WHERE is_synthetic = false) pj2 WHERE pj2.data_lag_hours IS NOT NULL) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-007 base (re-registered by 095): true median data_lag_hours (in days) over the trailing 30 days anchored at the non-synthetic data frontier. Replaces the 050 `SELECT (median_lag_hours / 24.0) ... FROM v_kpi_data_lag LIMIT 1` — a nondeterministic single (date, source) group's median with no ORDER BY. Synthetic-excluding (066-style wrap); byte-identical to its _include_synthetic twin modulo the wrap.$note$),
    ('data_quality_data_lag_include_synthetic',
     $kpi$SELECT (percentile_cont(0.5) WITHIN GROUP (ORDER BY data_lag_hours) / 24.0)::float AS median_lag_days FROM public.patient_journeys pj WHERE data_lag_hours IS NOT NULL AND created_at >= (SELECT MAX(created_at) FROM public.patient_journeys pj2 WHERE pj2.data_lag_hours IS NOT NULL) - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-007 demo/review twin: INCLUDES synthetic-tagged rows (reads patient_journeys raw). Resolved ONLY when E2I_KPI_INCLUDE_SYNTHETIC/E2I_INCLUDE_SYNTHETIC is set (synthetic-gold showcase). Same trailing-30d true median in days, same median_lag_days column, as the 095 base.$note$),

-- WS1-DQ-009 Time-to-Release: mean TTR in HOURS over the trailing 30 days of
-- successful runs at the data frontier (unit contract per #580: hours,
-- thresholds 24/48/72). Window is anchored on the latest SUCCESSFUL run so a
-- trailing failure streak narrows the window rather than emptying it.
    ('data_quality_time_to_release',
     $kpi$SELECT AVG(time_to_release_hours)::float AS avg_ttr_hours FROM (SELECT * FROM public.etl_pipeline_metrics WHERE is_synthetic = false) epm WHERE status = 'success' AND run_start >= (SELECT MAX(run_start) FROM (SELECT * FROM public.etl_pipeline_metrics WHERE is_synthetic = false) epm2 WHERE epm2.status = 'success') - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-009 base (re-registered by 095): mean time_to_release_hours over the trailing 30 days of status='success' runs anchored at the latest non-synthetic successful run. Replaces the 054 `SELECT avg_ttr_hours FROM v_kpi_time_to_release LIMIT 1` — a nondeterministic single (date, pipeline) group with no ORDER BY. Synthetic-excluding (066-style wrap); byte-identical to its _include_synthetic twin modulo the wrap.$note$),
    ('data_quality_time_to_release_include_synthetic',
     $kpi$SELECT AVG(time_to_release_hours)::float AS avg_ttr_hours FROM public.etl_pipeline_metrics epm WHERE status = 'success' AND run_start >= (SELECT MAX(run_start) FROM public.etl_pipeline_metrics epm2 WHERE epm2.status = 'success') - INTERVAL '30 days'$kpi$,
     0,
     $note$WS1-DQ-009 demo/review twin: INCLUDES synthetic-tagged rows (reads etl_pipeline_metrics raw). Resolved ONLY when E2I_KPI_INCLUDE_SYNTHETIC/E2I_INCLUDE_SYNTHETIC is set (synthetic-gold showcase). Same trailing-30d mean TTR in hours over status='success' runs (see the 095 backfill in (A)), same avg_ttr_hours column, as the 095 base.$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_ids are callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; run_migrations.sh / psql --single-transaction owns the outer txn.)
