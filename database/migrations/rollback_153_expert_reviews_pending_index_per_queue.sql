-- ROLLBACK for migration 153 (pending-review uniqueness per queue, #2244). NOT a
-- forward migration: scripts/run_migrations.sh skips rollback_*.sql in apply_dir(),
-- which is what makes it safe to ship this file beside 153.
--
-- Restores migration 140's exact index (uq_er_pending_estimand ON (estimand_key)
-- WHERE approval_status = 'pending') and removes the two per-queue indexes 153
-- created, and restores the two chronology readers 153 replaced to their
-- pre-153 bodies (public.dag_structure_rejected from migration 134,
-- public.is_dag_approved from ml/010), both queues again. 153 wrote no rows, so
-- there is nothing to restore in the data.
--
-- PRECONDITION (asserted below, BEFORE anything is dropped): no estimand may hold a
-- pending row in BOTH queues -- a pending initial_dag review beside a pending
-- dag_approval/quarterly_audit consult. 140's coarser index cannot be created over
-- such a pair; resolve (approve / reject / supersede) one of them first. The check
-- runs first so the failure names the estimands instead of surfacing as a 23505
-- from CREATE UNIQUE INDEX after the drops (inside --single-transaction nothing is
-- lost either way, but the message would be less useful).
--
-- The ledger row is deleted so a later deploy re-applies 153 instead of skipping it
-- (run_migrations.sh skips any filename already recorded in schema_migrations).
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_153_expert_reviews_pending_index_per_queue.sql

DO $$
DECLARE
    v_colliding TEXT;
BEGIN
    SELECT string_agg(estimand_key, ', ' ORDER BY estimand_key) INTO v_colliding
      FROM (
          SELECT estimand_key
            FROM public.expert_reviews
           WHERE approval_status = 'pending'
           GROUP BY estimand_key
          HAVING count(*) > 1
      ) AS dup;
    IF v_colliding IS NOT NULL THEN
        RAISE EXCEPTION 'rollback 153: estimand(s) % hold a pending review in both queues; '
            'resolve one per estimand before restoring the single-slot index (migration 140)',
            v_colliding;
    END IF;
END $$;

DROP INDEX IF EXISTS uq_er_pending_estimand_runtime;
DROP INDEX IF EXISTS uq_er_pending_estimand_structural;

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand
    ON public.expert_reviews (estimand_key)
    WHERE approval_status = 'pending';

COMMENT ON INDEX uq_er_pending_estimand IS
    'At most one PENDING expert review per estimand (migration 140); a new structure on '
    'the same estimand appends a version (expert_review_versions, migration 141).';

-- migration 134's body, verbatim (both queues).
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

-- ml/010's body, verbatim (both queues).
CREATE OR REPLACE FUNCTION public.is_dag_approved(
    p_dag_hash VARCHAR(64),
    p_brand VARCHAR(50) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM expert_reviews
        WHERE dag_version_hash = p_dag_hash
          AND approval_status = 'approved'
          AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
          AND (p_brand IS NULL OR brand = p_brand)
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.is_dag_approved(VARCHAR, VARCHAR) IS
    'Check if a DAG (by hash) has an active expert approval';

DELETE FROM public.schema_migrations
 WHERE filename = '153_expert_reviews_pending_index_per_queue.sql';
