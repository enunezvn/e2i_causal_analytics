-- ============================================================================
-- Migration 134: guarded causal_paths promotion (lane 1, spec §4.3)
-- ============================================================================
-- WHAT: two functions.
--   public.dag_structure_rejected(p_dag_version_hash text, p_brand text)
--     → boolean. The expert-review chronology rule
--     (src/causal_engine/expert_review_gate.py, ExpertReviewGate
--     ._latest_adjudication / check_rejection) in SQL: a structure is rejected
--     when the NEWEST non-pending review row for the hash (and brand, when a
--     brand is given) is 'rejected' and no pending row is newer than it. A
--     NULL hash means "no structure to check" and reads false. An EMPTY-STRING
--     brand means "no brand" (NULLIF) -- the Python reader filters with
--     ``if brand:`` (src/repositories/expert_review.py get_reviews_for_dag), so
--     '' must be unfiltered here too or a same-hash rejection is missed
--     (pre-execution review 2026-09-08, codex HIGH). A pending row with the SAME
--     created_at as the rejection does NOT reopen it (strict >); Task 3b gives
--     ExpertReviewGate._latest_adjudication the same tie rule, so the probe
--     and the promote read a tie identically.
--   CONCURRENCY: promote_causal_path_guarded takes LOCK TABLE expert_reviews
--     IN SHARE MODE before its UPDATE (released with the RPC's transaction, a
--     few ms). SHARE conflicts with ROW EXCLUSIVE, so a resolve (UPDATE of the
--     pending row) or a renew (INSERT of a new pending row that could then be
--     rejected) racing the promote either committed first -- READ COMMITTED
--     gives the UPDATE below a fresh snapshot that sees it -- or waits and lands
--     strictly after the promote; plain reads are not blocked. A row-level FOR
--     SHARE was not enough: it cannot cover a row that does not exist yet
--     (pre-execution review iter-2 + iter-3, codex HIGH x2; row lock and table
--     lock both measured live 2026-09-08, Step 4c).
--   public.promote_causal_path_guarded(p_path_id, p_new_status,
--     p_allowed_current text[], p_dag_version_hash, p_brand) → jsonb
--     One UPDATE that moves causal_paths.validation_status only when the
--     current status is allowed AND the structure is not rejected -- the
--     rejection predicate is evaluated INSIDE the UPDATE statement, so there
--     is no window between reading the verdict and writing the status.
--     Returns {"moved": 0|1, "rejected": bool}.
-- WHY: CausalPathRepository.set_validation_status conditioned the promote on
--   the current status only; a rejection committed between the RefutationNode's
--   read-only probe and its status write was not seen (#1985 residue).
-- SAFETY: SECURITY INVOKER; service_role EXECUTE only (precedent
--   database/ml/036 record_discovered_dag); idempotent (CREATE OR REPLACE);
--   the asserting DO block below RAISEs on a grant regression. Migration 119's
--   trigger on 'validated' stays the second line of defence.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.dag_structure_rejected(
    p_dag_version_hash text,
    p_brand text DEFAULT NULL
) RETURNS boolean
LANGUAGE sql
STABLE
SET search_path = public
AS $$
    WITH latest_np AS (
        SELECT r.approval_status, r.created_at
        FROM public.expert_reviews r
        WHERE r.dag_version_hash = p_dag_version_hash
          AND (NULLIF(p_brand, '') IS NULL OR r.brand = p_brand)
          AND r.approval_status <> 'pending'
        ORDER BY r.created_at DESC
        LIMIT 1
    )
    SELECT COALESCE(
        (SELECT l.approval_status = 'rejected'
                AND NOT EXISTS (
                    SELECT 1
                    FROM public.expert_reviews p
                    WHERE p.dag_version_hash = p_dag_version_hash
                      AND (NULLIF(p_brand, '') IS NULL OR p.brand = p_brand)
                      AND p.approval_status = 'pending'
                      AND p.created_at > l.created_at  -- a tie is NOT a reopen
                )
         FROM latest_np l),
        false
    );
$$;

COMMENT ON FUNCTION public.dag_structure_rejected(text, text) IS
    'Lane 1 (migration 134): the expert-review chronology rule in SQL -- true when '
    'the newest non-pending review of this DAG hash (and brand, when given) is '
    'rejected and no pending review is newer. Python mirror: ExpertReviewGate'
    '._latest_adjudication. NULL hash reads false ("unchecked").';

CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(
    p_path_id text,
    p_new_status text,
    p_allowed_current text[],
    p_dag_version_hash text DEFAULT NULL,
    p_brand text DEFAULT NULL
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
    v_moved integer := 0;
    v_rejected boolean := false;
BEGIN
    IF p_path_id IS NULL OR p_new_status IS NULL OR p_allowed_current IS NULL THEN
        RAISE EXCEPTION 'promote_causal_path_guarded: p_path_id, p_new_status and p_allowed_current are required';
    END IF;

    IF p_dag_version_hash IS NOT NULL THEN
        -- Pin the review chronology for the rest of this transaction (see
        -- header): blocks concurrent review INSERT/UPDATE/DELETE, never reads.
        LOCK TABLE public.expert_reviews IN SHARE MODE;
    END IF;

    UPDATE public.causal_paths
       SET validation_status = p_new_status
     WHERE path_id = p_path_id
       AND validation_status = ANY (p_allowed_current)
       AND NOT public.dag_structure_rejected(p_dag_version_hash, p_brand);
    GET DIAGNOSTICS v_moved = ROW_COUNT;

    IF v_moved = 0 THEN
        v_rejected := public.dag_structure_rejected(p_dag_version_hash, p_brand);
    END IF;

    RETURN jsonb_build_object('moved', v_moved, 'rejected', v_rejected);
END;
$$;

COMMENT ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) IS
    'Lane 1 (migration 134): the RefutationNode''s SOLE promoter write. Moves '
    'causal_paths.validation_status only when the current status is in '
    'p_allowed_current AND dag_structure_rejected(hash, brand) is false, in one '
    'statement. Returns {"moved": 0|1, "rejected": bool}.';

REVOKE ALL ON FUNCTION public.dag_structure_rejected(text, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.dag_structure_rejected(text, text) TO service_role;
REVOKE ALL ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.promote_causal_path_guarded(text, text, text[], text, text) TO service_role;

DO $$
DECLARE
    v_fn text;
BEGIN
    FOREACH v_fn IN ARRAY ARRAY[
        'public.dag_structure_rejected(text, text)',
        'public.promote_causal_path_guarded(text, text, text[], text, text)'
    ] LOOP
        IF NOT has_function_privilege('service_role', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: service_role cannot EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('anon', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: anon can still EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('authenticated', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 134: authenticated can still EXECUTE %', v_fn;
        END IF;
    END LOOP;
    -- Behavioural smoke: an unknown hash is never "rejected"; an absent path never moves.
    IF public.dag_structure_rejected('migration-134-no-such-hash', NULL) THEN
        RAISE EXCEPTION 'migration 134: unknown hash reads as rejected';
    END IF;
    IF (public.promote_causal_path_guarded('migration-134-no-such-path', 'validated',
            ARRAY['pending'], NULL, NULL) ->> 'moved')::int <> 0 THEN
        RAISE EXCEPTION 'migration 134: a non-existent path moved';
    END IF;
END $$;
