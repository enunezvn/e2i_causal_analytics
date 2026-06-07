-- ============================================================================
-- Migration 036: atomic increment_procedure_outcome RPC (audit 2026-06-05, L2 / #694)
-- ============================================================================
--
-- WHAT: `update_procedure_outcome` (src/memory/procedural_memory.py) did a
--       read-modify-write — SELECT usage_count/success_count, then UPDATE to
--       count+1 — which loses updates under concurrent outcomes (L2). This
--       server-side RPC performs the increment ATOMICALLY in a single UPDATE.
--
-- RETURN SHAPE: RETURNS TABLE(rows_updated INTEGER), NOT a bare scalar. A scalar
--       (RETURNS INTEGER) is serialized by PostgREST as a bare JSON number, which
--       supabase-py's APIResponse rejects (its `data` must be a LIST) → APIError
--       on every call. A TABLE-returning function returns a JSON array, which
--       supabase-py parses. We emit a row ONLY when a procedure was actually
--       updated, so "not found" returns an EMPTY array and the caller's
--       `if not result.data:` cleanly detects it.
--
-- NOTE: `procedural_memories.success_rate` is GENERATED ALWAYS, so it recomputes
--       from usage_count/success_count automatically — do NOT set it here.
--
-- DROP+CREATE (not CREATE OR REPLACE): an earlier revision shipped this as
-- RETURNS INTEGER; the return type can't be changed in place. Idempotent
-- (DROP ... IF EXISTS). Write RPC: granted to authenticated + service_role.
-- ============================================================================

DROP FUNCTION IF EXISTS increment_procedure_outcome(UUID, BOOLEAN);

CREATE OR REPLACE FUNCTION increment_procedure_outcome(
    p_procedure_id UUID,
    p_success BOOLEAN
) RETURNS TABLE (rows_updated INTEGER)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE procedural_memories
    SET usage_count = COALESCE(usage_count, 0) + 1,
        success_count = COALESCE(success_count, 0) + CASE WHEN p_success THEN 1 ELSE 0 END,
        updated_at = NOW()
    WHERE procedure_id = p_procedure_id;

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    -- Emit a row only when something was updated; "not found" => empty result.
    IF rows_updated > 0 THEN
        RETURN NEXT;
    END IF;
    RETURN;
END;
$$;

GRANT EXECUTE ON FUNCTION increment_procedure_outcome(UUID, BOOLEAN)
    TO authenticated, service_role;
