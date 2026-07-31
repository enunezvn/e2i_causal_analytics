-- ============================================================================
-- Migration 120: raise the kpi_query positional-param cap 4 -> 6 (#1388)
-- ============================================================================
-- The kpi_query allowlist executor (migration 044) hand-unrolls positional binds
-- and hard-raises once a call carries more than four positional params (the
-- terminal ELSE arm of the n=0..4 unroll).
--
-- That four was a DESIGN cap from when no registered statement needed more --
-- NOT a Postgres limit (PL/pgSQL EXECUTE ... USING has no such ceiling; we unroll
-- because USING has no VARIADIC form). Consequence: a KPI ask that needs region +
-- an explicit time window cannot co-bind -- brand, region, trigger_type,
-- window_start, window_end = 5 params. The #1360 trigger-effectiveness
-- `_windowed` variants (migration 118) therefore had to DROP the region axis, and
-- the calculator failed closed on region+window (honest, but a capability gap).
--
-- Part 1 EXTENDS the unroll to 6 params (n=5, n=6 arms). The SECURITY MODEL is
-- UNCHANGED: still only registry-vetted statements run (looked up by id), params
-- are still positionally bound via EXECUTE ... USING (never string-interpolated),
-- the arity check (n <> expected_n) still rejects wrong-count binds, and the
-- terminal RAISE still hard-caps -- just at 6 instead of 4. No new client surface.
--
-- Part 2 registers the 5-param regioned+windowed trigger-effectiveness variants
-- (`trigger_effectiveness_<metric>_windowed_region[_include_synthetic]`): the
-- migration-118 windowed SQL with region re-added ($2, via the patient_journeys
-- join -- triggers carry no region column, the 078/118 idiom) and the params
-- shifted so brand=$1, region=$2, trigger_type=$3, window=[$4,$5). These are
-- ADDITIVE registry rows; every existing id, calculator, and the migration-118
-- `_windowed` (region-dropped) forms are untouched and still valid.
--
-- cohort_profiler's migration-117 windowed variant (which dropped its MAX-age
-- bound at the same 4-param wall) can now regain that 5th param too, but that is
-- a DIFFERENT agent/selection path and axis set -- filed as follow-up, not bundled
-- here (the issue names trigger-effectiveness first).
--
-- Zero data changes. DB application is deferred to batch-deploy (this file is
-- authored in the repo; the dispatcher applies it on the droplet).
-- ----------------------------------------------------------------------------

-- ----------------------------------------------------------------------------
-- Part 1: extend the positional unroll 4 -> 6 (CREATE OR REPLACE; body mirrors
-- migration 044 verbatim except the added n=5/n=6 arms and the 6-param RAISE).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.kpi_query(query_id text, params jsonb DEFAULT '[]'::jsonb)
RETURNS SETOF json
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = public, pg_temp
AS $func$
DECLARE
    stmt text;
    expected_n int;
    param_arr text[];
    n int;
    wrapped text;
BEGIN
    SELECT r.sql, r.max_params INTO stmt, expected_n
      FROM public.kpi_query_registry r
     WHERE r.query_id = kpi_query.query_id;

    IF stmt IS NULL THEN
        RAISE EXCEPTION 'kpi_query: unknown query_id %', query_id USING ERRCODE = '22023';
    END IF;

    -- params must be a JSON array (reject objects/scalars that would bind wrong).
    IF params IS NOT NULL AND jsonb_typeof(params) <> 'array' THEN
        RAISE EXCEPTION 'kpi_query: params must be a JSON array (got %)', jsonb_typeof(params)
            USING ERRCODE = '22023';
    END IF;

    -- Bind $1..$N positionally from the jsonb array (text-typed; statements cast as needed).
    SELECT array_agg(elem #>> '{}' ORDER BY ord)
      INTO param_arr
      FROM jsonb_array_elements(COALESCE(params, '[]'::jsonb)) WITH ORDINALITY AS t(elem, ord);
    param_arr := COALESCE(param_arr, ARRAY[]::text[]);
    n := COALESCE(array_length(param_arr, 1), 0);

    -- Enforce the registry's declared arity (no wrong-count binding).
    IF n <> expected_n THEN
        RAISE EXCEPTION 'kpi_query: % expects % param(s), got %', query_id, expected_n, n
            USING ERRCODE = '22023';
    END IF;

    -- stmt is a TRUSTED, registry-vetted statement (not client SQL): safe to wrap.
    wrapped := format('SELECT row_to_json(_sub) FROM (%s) AS _sub', stmt);

    IF n = 0 THEN
        RETURN QUERY EXECUTE wrapped;
    ELSIF n = 1 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1];
    ELSIF n = 2 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2];
    ELSIF n = 3 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2], param_arr[3];
    ELSIF n = 4 THEN
        RETURN QUERY EXECUTE wrapped USING param_arr[1], param_arr[2], param_arr[3], param_arr[4];
    ELSIF n = 5 THEN
        RETURN QUERY EXECUTE wrapped
            USING param_arr[1], param_arr[2], param_arr[3], param_arr[4], param_arr[5];
    ELSIF n = 6 THEN
        RETURN QUERY EXECUTE wrapped
            USING param_arr[1], param_arr[2], param_arr[3], param_arr[4], param_arr[5], param_arr[6];
    ELSE
        RAISE EXCEPTION 'kpi_query: at most 6 positional parameters supported (got %)', n;
    END IF;
END;
$func$;

COMMENT ON FUNCTION public.kpi_query(text, jsonb) IS
    'Run a vetted read-only KPI statement from kpi_query_registry by id (#574). '
    'Clients pass query_id + params; never raw SQL. SECURITY DEFINER over allowlist. '
    'Positional unroll supports up to 6 params (#1388 raised the design cap from 4).';

-- ----------------------------------------------------------------------------
-- Part 2: regioned+windowed trigger-effectiveness statements (#1388).
-- The migration-118 `_windowed` SQL with region re-added at $2 (patient_journeys
-- join) and params shifted: $1 brand, $2 region, $3 trigger_type, $4/$5 half-open
-- [start, end) window on trigger_timestamp -- all filters nullable. No
-- data_through column (the window is explicit -- the WS3 _windowed idiom).
-- ----------------------------------------------------------------------------
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    -- ---- Trigger Precision (WS2-TR-001, v2 definition, migration 113) ----
    ('trigger_effectiveness_precision_windowed_region', $kpi$SELECT COUNT(*) FILTER (WHERE acceptance_status = 'accepted' AND outcome_tracked AND outcome_value > 0)::float / NULLIF(COUNT(*) FILTER (WHERE acceptance_status = 'accepted' AND outcome_tracked), 0) AS precision FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed precision: $1 brand, $2 region (patient_journeys join), $3 trigger_type (nullable), $4/$5 half-open window; recent windows under-count until the conversion window matures$note$),
    ('trigger_effectiveness_precision_windowed_region_include_synthetic', $kpi$SELECT COUNT(*) FILTER (WHERE acceptance_status = 'accepted' AND outcome_tracked AND outcome_value > 0)::float / NULLIF(COUNT(*) FILTER (WHERE acceptance_status = 'accepted' AND outcome_tracked), 0) AS precision FROM triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed precision (includes synthetic)$note$),
    -- ---- Acceptance Rate (WS2-TR-004, delivered denominator 090/092) ----
    ('trigger_effectiveness_acceptance_rate_windowed_region', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed acceptance rate: $1 brand, $2 region, $3 trigger_type (nullable), $4/$5 half-open window$note$),
    ('trigger_effectiveness_acceptance_rate_windowed_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS acceptance_rate FROM triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed acceptance rate (includes synthetic)$note$),
    -- ---- Override Rate (WS2-TR-006, delivered denominator 090) ----
    ('trigger_effectiveness_override_rate_windowed_region', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed override rate: $1 brand, $2 region, $3 trigger_type (nullable), $4/$5 half-open window$note$),
    ('trigger_effectiveness_override_rate_windowed_region_include_synthetic', $kpi$SELECT COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN delivery_status IN ('delivered', 'viewed') THEN 1 END), 0) AS override_rate FROM triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)$kpi$, 5, $note$#1388 regioned+windowed override rate (includes synthetic)$note$),
    -- ---- Trigger Funnel Conversion (WS2-TR-009) ----
    ('trigger_effectiveness_funnel_conversion_windowed_region', $kpi$SELECT base.*, base.n_actioned::float / NULLIF(base.n_delivered, 0) AS funnel_conversion FROM (SELECT COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed')) AS n_delivered, COUNT(*) FILTER (WHERE delivery_status = 'viewed') AS n_viewed, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted') AS n_accepted, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted' AND action_taken IS NOT NULL) AS n_actioned, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted' AND action_taken IS NOT NULL AND outcome_tracked AND outcome_value > 0) AS n_outcome FROM (SELECT * FROM triggers WHERE is_synthetic = false) triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)) base$kpi$, 5, $note$#1388 regioned+windowed funnel: $1 brand, $2 region, $3 trigger_type (nullable), $4/$5 half-open window$note$),
    ('trigger_effectiveness_funnel_conversion_windowed_region_include_synthetic', $kpi$SELECT base.*, base.n_actioned::float / NULLIF(base.n_delivered, 0) AS funnel_conversion FROM (SELECT COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed')) AS n_delivered, COUNT(*) FILTER (WHERE delivery_status = 'viewed') AS n_viewed, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted') AS n_accepted, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted' AND action_taken IS NOT NULL) AS n_actioned, COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed') AND acceptance_status = 'accepted' AND action_taken IS NOT NULL AND outcome_tracked AND outcome_value > 0) AS n_outcome FROM triggers WHERE trigger_timestamp >= $4::timestamptz AND trigger_timestamp < $5::timestamptz AND ($1::text IS NULL OR brand_id::text = $1) AND ($2::text IS NULL OR patient_id IN (SELECT patient_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))) AND ($3::text IS NULL OR trigger_type::text = $3)) base$kpi$, 5, $note$#1388 regioned+windowed funnel (includes synthetic)$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the new ids are visible.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
