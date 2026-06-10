-- Migration 063: add the `business_impact_hcp_reach` allowlist query for the
-- Home "HCPs Reached" KPI tile (H1 fix).
--
-- WHY: the Home tiles previously read the SYNTHETIC `business_metrics` table (or
-- fell back to hardcoded `_FALLBACK_KPIS`), showing fabricated values as real.
-- The honest fix re-points get_kpi_summary() at the REAL allowlisted KPI queries
-- (treatment_events via kpi_query RPC). TRx/NRx/NBRx/coverage/share/conversion
-- already exist in the registry (migration 044); "HCPs Reached" did not.
--
-- Definition: distinct HCPs with brand prescription activity in the last 30 days,
-- from the REAL treatment_events table (has hcp_id + brand). Mirrors
-- business_impact_trx (same table/window/optional-brand-filter) so the count is
-- brand-scoped and consistent with TRx. Honest note: when the source data is
-- stale (current prod treatment_events ends 2025-12-22) this is a real 0, not a
-- fabricated number -- a DATA-freshness matter, not a code bug.

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note)
VALUES (
    'business_impact_hcp_reach',
    $kpi$SELECT COUNT(DISTINCT hcp_id) AS hcp_reach FROM treatment_events WHERE event_date >= NOW() - INTERVAL '30 days' AND ($1::text IS NULL OR brand::text = $1)$kpi$,
    1,
    $note$H1: distinct HCPs reached (brand prescription activity, 30d) for the Home tile; real treatment_events, brand enum ::text cast, brand filter optional$note$
)
ON CONFLICT (query_id) DO UPDATE
    SET sql = EXCLUDED.sql,
        max_params = EXCLUDED.max_params,
        note = EXCLUDED.note;
