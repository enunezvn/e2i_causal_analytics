-- Migration 064: add the `business_impact_data_through` allowlist query.
--
-- WHY: the Home KPI tiles (PR #820) honestly read the real treatment_events
-- substrate, but prod data is stale (ends 2025-12-22), so the 30-day-window KPIs
-- read 0. Showing a bare "0" reads as "broken". The FE now renders a 0/null tile
-- as "No recent activity -- data through <date>", where <date> is computed
-- DYNAMICALLY from this query (NOT hardcoded) -- the latest prescription
-- event_date in treatment_events, i.e. the data-coverage end. It auto-advances
-- when fresh data lands.
--
-- Brand-agnostic (data coverage is a global freshness fact). Read-only SELECT,
-- runs only via the vetted kpi_query() RPC.

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note)
VALUES (
    'business_impact_data_through',
    $kpi$SELECT max(event_date) AS data_through FROM treatment_events WHERE event_type::text = 'prescription'$kpi$,
    0,
    $note$Latest prescription event_date in treatment_events -- data-coverage end for the Home tiles' dynamic "data through <date>" honesty label$note$
)
ON CONFLICT (query_id) DO UPDATE
    SET sql = EXCLUDED.sql,
        max_params = EXCLUDED.max_params,
        note = EXCLUDED.note;
