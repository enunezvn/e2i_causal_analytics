-- =============================================================================
-- 109: Random entity sampling RPC for cohort-level SHAP importance
-- =============================================================================
-- /api/explain/global previously sampled a deterministic prefix (first N ids
-- ordered by id) of the cohort source table. Sequential synthetic ids correlate
-- with generation order (e.g. frontier-append rows land at the tail), so a
-- prefix is not a representative draw. PostgREST cannot ORDER BY random(), so
-- the uniform draw lives here as an RPC.
--
-- p_source is whitelisted (not interpolated) — the two cohort grains only.
-- VOLATILE on purpose: random() must re-evaluate per call.

CREATE OR REPLACE FUNCTION public.sample_entity_ids(p_source text, p_limit integer)
RETURNS TABLE(entity_id text)
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    IF p_limit IS NULL OR p_limit < 1 OR p_limit > 500 THEN
        RAISE EXCEPTION 'sample_entity_ids: p_limit out of range (1..500): %', p_limit;
    END IF;

    IF p_source = 'hcp_profiles' THEN
        RETURN QUERY
        SELECT hcp_id::text
        FROM public.hcp_profiles
        ORDER BY random()
        LIMIT p_limit;
    ELSIF p_source = 'patient_journeys' THEN
        -- DISTINCT first: a patient with several journey rows must not get a
        -- higher draw probability than a single-journey patient.
        RETURN QUERY
        SELECT pid
        FROM (SELECT DISTINCT patient_id::text AS pid FROM public.patient_journeys) d
        ORDER BY random()
        LIMIT p_limit;
    ELSE
        RAISE EXCEPTION 'sample_entity_ids: unknown entity source: %', p_source;
    END IF;
END;
$$;

COMMENT ON FUNCTION public.sample_entity_ids(text, integer) IS
    'Uniform random draw of cohort entity ids for /api/explain/global (migration 109).';
