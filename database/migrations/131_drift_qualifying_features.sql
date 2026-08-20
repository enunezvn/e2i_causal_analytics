-- ============================================================================
-- Migration 131: drift_qualifying_features() — the dispatcher's substrate
-- probe for drift_monitor chat dispatch (#1747).
-- ============================================================================
-- WHY: every chat-path orchestrator dispatch of drift_monitor died at input
-- coercion ('features_to_monitor Field required') because the dispatcher had
-- no input resolver for it. The resolver (mirror of the #874 gap / #1726 het
-- pattern) must ground features_to_monitor in the REAL feature store — which
-- means answering, at dispatch time: "which registered features have enough
-- feature_values samples in BOTH drift windows (baseline and current) for the
-- agent's detectors to actually compute?" The agent's DataDriftNode requires
-- >= 30 samples per window (_min_samples); a bound feature below that yields
-- only an honest per-feature 'insufficient data' — fine for a user-NAMED
-- feature, pointless for a dispatcher-SELECTED one.
--
-- Why a SQL function: the per-feature two-window HAVING-count is a group-by
-- aggregate; PostgREST aggregates are disabled on this deployment (PGRST123,
-- measured 2026-08-20) and paging the raw window rows through the REST API at
-- dispatch time is not viable (~26k rows/60d at 1k rows/page). One RPC call
-- per candidate window keeps the dispatch-path probe bounded and exact.
--
-- Windows mirror DataDriftNode._fetch_data + the supabase connector's
-- .gte(start).lte(end) CLOSED intervals exactly (codex iter-1 MED):
-- current = [now()-Nd, now()], baseline = [now()-2Nd, now()-Nd]. Two
-- consequences of that fidelity: a row at the exact now()-Nd instant counts in
-- BOTH windows (the connector genuinely does this), and future-dated rows
-- (clock skew, scheduled loads) are EXCLUDED — the connector would never fetch
-- them, so counting them would qualify features on support the detectors can't
-- see. Provenance mirrors apply_provenance_filter
-- default-exclude (is_synthetic IS NOT TRUE ... NULL counts as real) on BOTH
-- the features registry (#894: the registry is is_synthetic-tagged) and the
-- feature_values rows; p_include_synthetic=true lifts both, matching the
-- #872/#880 opt-in the resolver threads through.
--
-- Measured at authoring time (2026-08-20): feature_values is 100%
-- synthetic-tagged (444,068 rows, zero non-synthetic in any month), so
-- real-mode returns 0 rows at every window; with synthetic included the 7d
-- default window has 0 qualifying features and 30d has 15. Those numbers are
-- the substrate's, not the function's — the function just reports them.
--
-- SECURITY DEFINER + EXECUTE grants: the dispatcher probe runs on the API's
-- anon-key sync client (same client as the #874 gap probe). The function
-- exposes only aggregate counts per feature name — no row-level data.
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.drift_qualifying_features(
    p_window_days       integer,
    p_min_samples       integer DEFAULT 30,
    p_include_synthetic boolean DEFAULT false
)
RETURNS TABLE (
    feature_name text,
    baseline_n   bigint,
    current_n    bigint
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
    WITH windowed AS (
        SELECT
            fv.feature_id,
            count(*) FILTER (
                WHERE fv.event_timestamp >= now() - make_interval(days => p_window_days)
                  AND fv.event_timestamp <= now()
            ) AS current_n,
            count(*) FILTER (
                WHERE fv.event_timestamp >= now() - make_interval(days => 2 * p_window_days)
                  AND fv.event_timestamp <= now() - make_interval(days => p_window_days)
            ) AS baseline_n
        FROM public.feature_values fv
        WHERE fv.event_timestamp >= now() - make_interval(days => 2 * p_window_days)
          AND fv.event_timestamp <= now()
          AND (p_include_synthetic OR fv.is_synthetic IS NOT TRUE)
        GROUP BY fv.feature_id
    )
    SELECT
        f.name::text AS feature_name,
        w.baseline_n,
        w.current_n
    FROM windowed w
    JOIN public.features f ON f.id = w.feature_id
    WHERE (p_include_synthetic OR f.is_synthetic IS NOT TRUE)
      AND w.baseline_n >= p_min_samples
      AND w.current_n  >= p_min_samples
    ORDER BY w.current_n DESC, f.name ASC
$$;

COMMENT ON FUNCTION public.drift_qualifying_features(integer, integer, boolean) IS
    '#1747: features with >= p_min_samples feature_values in BOTH drift windows '
    '(closed intervals mirroring the connector''s gte/lte: baseline '
    '[now-2N, now-N], current [now-N, now]) under the provenance predicate — '
    'the dispatcher''s drift_monitor substrate probe. Ordered by '
    'current-window support (best-supported first).';

GRANT EXECUTE ON FUNCTION public.drift_qualifying_features(integer, integer, boolean)
    TO anon, authenticated, service_role;
