-- ============================================================================
-- Migration 036: atomic increment_procedure_outcome RPC (audit 2026-06-05, L2 / #694)
-- ============================================================================
--
-- WHAT: `update_procedure_outcome` (src/memory/procedural_memory.py) did a
--       read-modify-write — SELECT usage_count/success_count, then UPDATE to
--       count+1 — which loses updates under concurrent outcomes (L2). This
--       server-side RPC performs the increment ATOMICALLY in a single UPDATE.
--
-- NOTE: `procedural_memories.success_rate` is GENERATED ALWAYS, so it recomputes
--       from usage_count/success_count automatically — do NOT set it here.
--       Returns the number of rows updated (0 => procedure not found, so the
--       caller can keep its "not found" warning).
--
-- Idempotent (CREATE OR REPLACE). Write RPC: granted to authenticated +
-- service_role (not anon).
-- ============================================================================

CREATE OR REPLACE FUNCTION increment_procedure_outcome(
    p_procedure_id UUID,
    p_success BOOLEAN
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows INTEGER;
BEGIN
    UPDATE procedural_memories
    SET usage_count = COALESCE(usage_count, 0) + 1,
        success_count = COALESCE(success_count, 0) + CASE WHEN p_success THEN 1 ELSE 0 END,
        updated_at = NOW()
    WHERE procedure_id = p_procedure_id;
    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows;
END;
$$;

GRANT EXECUTE ON FUNCTION increment_procedure_outcome(UUID, BOOLEAN)
    TO authenticated, service_role;
