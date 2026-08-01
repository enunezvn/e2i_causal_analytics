-- ============================================================================
-- 122_cohort_profiler_windowed_maxage_bind.sql
-- Restore the cohort_profiler WINDOWED criteria statement's MAX-age bound as a
-- 5th positional param, now the kpi_query RPC allows 6 (#1402).
--
-- WHY: migration 117 registered
-- `cohort_profiler_patient_criteria_profile_windowed` (+ its
-- `_include_synthetic` twin) with only FOUR positional params — brand ($1),
-- the half-open [$2, $3) window, and the exclusive MIN-age bound ($4). The
-- exclusive MAX-age bound was DROPPED because the migration-044 kpi_query()
-- RPC capped at 4 positional params: the mig-117 header and the
-- cohort_profiler agent's `_analyze_patients` disclosed a windowed max-age ask
-- as NOT-applied (never silently dropped). Migration 120 (#1388) raised that
-- cap to 6 (`ELSIF n = 5` / `ELSIF n = 6`), so the constraint that forced the
-- drop no longer exists — this is the follow-up PR #1396 explicitly deferred.
--
-- WHAT: upsert BOTH windowed rows with the MAX-age bound restored at $5
-- (`($5::int IS NULL OR pj.age_at_diagnosis < $5::int)`, exclusive + nullable,
-- mirroring the $4 min-age idiom) and `max_params` bumped 4 -> 5. Param order
-- is purely additive — $1 brand, $2/$3 half-open window, $4 exclusive min age,
-- $5 exclusive max age — leaving the existing $1..$4 positions byte-for-byte.
-- Every other mig-117 statement (the non-windowed criteria family, the HCP
-- cohort family) is untouched.
--
-- Synthetic gating follows the ADDITIVE-variant idiom (mig 077/084/105/116/117):
-- the base statement wraps taggable tables in (SELECT * FROM t WHERE
-- is_synthetic = false); the `_include_synthetic` twin is the unwrapped
-- original. These ids stay ABSENT from SYNTHETIC_TWINNED_QUERY_IDS (locked to
-- migrations 066/085/095 by CI) — the agent's _profiler_query_id() appends the
-- suffix under the showcase flag.
-- ----------------------------------------------------------------------------

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- Patient criteria-bound profile, WINDOWED sibling — max-age restored ($5) ----
    ('cohort_profiler_patient_criteria_profile_windowed', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) te JOIN (SELECT * FROM patient_journeys WHERE is_synthetic = false) pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) AND ($4::int IS NULL OR pj.age_at_diagnosis > $4::int) AND ($5::int IS NULL OR pj.age_at_diagnosis < $5::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 5, $note$#1402 windowed criteria profile: params $1 brand (nullable), $2/$3 half-open date window, $4 exclusive min age (nullable), $5 exclusive max age (nullable)$note$),
    ('cohort_profiler_patient_criteria_profile_windowed_include_synthetic', $kpi$SELECT pj.segment_assignment::text AS severity, pj.prior_therapy_lines AS therapy_line, COUNT(*) AS nrx FROM treatment_events te JOIN patient_journeys pj ON pj.patient_id = te.patient_id WHERE te.event_type::text = 'prescription' AND te.sequence_number = 1 AND te.event_date >= $2::date AND te.event_date < $3::date AND ($1::text IS NULL OR te.brand::text = $1) AND ($4::int IS NULL OR pj.age_at_diagnosis > $4::int) AND ($5::int IS NULL OR pj.age_at_diagnosis < $5::int) GROUP BY 1, 2 ORDER BY 1, 2$kpi$, 5, $note$#1402 windowed criteria profile (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the updated statement is visible.
NOTIFY pgrst, 'reload schema';
