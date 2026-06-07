-- ============================================================================
-- Migration 035: search_episodic_memory — add min_importance + days_back filters
--                (audit 2026-06-05, L1 / issue #694)
-- ============================================================================
--
-- WHAT: REWIRE. `EpisodicSearchFilters.min_importance` / `days_back` were declared
--       in src/memory/episodic_memory.py but never forwarded — the RPC had no
--       params for them, so the declared filters silently did nothing (L1). This
--       adds two OPTIONAL params + WHERE clauses, making the declared filters
--       functional. `episodic_memories.importance_score` (001:197) and
--       `occurred_at` (001:193) already exist.
--
-- WHY DROP+CREATE (not CREATE OR REPLACE): adding params changes the function
--       signature, so REPLACE would leave a second overload. We DROP the exact
--       9-arg signature and CREATE the 11-arg one, re-GRANTing to match the
--       existing ACL (anon, authenticated, service_role). Wrapped in a
--       transaction so the LIVE RPC is never absent.
--
-- Idempotent: DROP ... IF EXISTS + CREATE; safe to re-apply.
-- ============================================================================

BEGIN;

DROP FUNCTION IF EXISTS search_episodic_memory(
    vector, FLOAT, INT, memory_event_type, e2i_agent_name, VARCHAR, VARCHAR, VARCHAR, VARCHAR
);

CREATE OR REPLACE FUNCTION search_episodic_memory(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10,
    filter_event_type memory_event_type DEFAULT NULL,
    filter_agent e2i_agent_name DEFAULT NULL,
    filter_brand VARCHAR DEFAULT NULL,
    filter_region VARCHAR DEFAULT NULL,
    filter_patient_id VARCHAR DEFAULT NULL,
    filter_hcp_id VARCHAR DEFAULT NULL,
    filter_min_importance FLOAT DEFAULT NULL,
    filter_days_back INT DEFAULT NULL
)
RETURNS TABLE (
    memory_id UUID,
    event_type memory_event_type,
    description TEXT,
    entities JSONB,
    patient_journey_id VARCHAR,
    patient_id VARCHAR,
    hcp_id VARCHAR,
    trigger_id VARCHAR,
    agent_name e2i_agent_name,
    brand VARCHAR,
    region VARCHAR,
    occurred_at TIMESTAMPTZ,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        em.memory_id,
        em.event_type,
        em.description,
        em.entities,
        em.patient_journey_id,
        em.patient_id,
        em.hcp_id,
        em.trigger_id,
        em.agent_name,
        em.brand,
        em.region,
        em.occurred_at,
        1 - (em.embedding <=> query_embedding) AS similarity
    FROM episodic_memories em
    WHERE
        1 - (em.embedding <=> query_embedding) > match_threshold
        AND (filter_event_type IS NULL OR em.event_type = filter_event_type)
        AND (filter_agent IS NULL OR em.agent_name = filter_agent)
        AND (filter_brand IS NULL OR em.brand = filter_brand)
        AND (filter_region IS NULL OR em.region = filter_region)
        AND (filter_patient_id IS NULL OR em.patient_id = filter_patient_id)
        AND (filter_hcp_id IS NULL OR em.hcp_id = filter_hcp_id)
        -- L1 (#694): the previously-inert filters, now functional.
        AND (filter_min_importance IS NULL OR em.importance_score >= filter_min_importance)
        AND (filter_days_back IS NULL OR em.occurred_at >= NOW() - make_interval(days => filter_days_back))
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

GRANT EXECUTE ON FUNCTION search_episodic_memory(
    vector, FLOAT, INT, memory_event_type, e2i_agent_name, VARCHAR, VARCHAR, VARCHAR, VARCHAR, FLOAT, INT
) TO anon, authenticated, service_role;

COMMIT;
