-- ============================================================================
-- Migration 153: pending-review uniqueness per QUEUE, not per estimand (#2244)
-- ============================================================================
-- WHAT: replaces migration 140's uq_er_pending_estimand
--   ON expert_reviews (estimand_key) WHERE approval_status = 'pending'
-- with two partial unique indexes on the same key that split on review_type:
--   uq_er_pending_estimand_runtime     ... AND review_type <> 'initial_dag'
--   uq_er_pending_estimand_structural  ... AND review_type =  'initial_dag'
--
-- WHY: two producers write PENDING rows for the same estimand with different
--   meanings, and they are two independent review queues:
--   * the RUNTIME queue -- ExpertReviewGate's auto-created consult
--     (review_type = 'dag_approval', src/causal_engine/expert_review_gate.py)
--     and the renewals renew_review mints on it ('quarterly_audit', #2090,
--     which the gate adopts as its own consult on the 23505 recovery). ONE
--     pending row per estimand across these types is migration 140's contract
--     and stays: a structure change UPDATES the open consult (141) instead of
--     minting a sibling, and creation order is adjudication order.
--   * the STRUCTURAL-AUTHOR queue -- Lane B's offline, pre-run authoring
--     review (review_type = 'initial_dag', migration 152; written by
--     scripts/author_cohort_dag.py --review, read back ONLY by
--     src/data/kg/structural_prior_loader.py, which fail-closes on any other
--     type). The gate must never adopt one of these as its consult: it would
--     append its runtime structure to the AUTHORED snapshot (the loader then
--     refuses the row on its hash check, and the reviewer sees a mutated
--     DAG), and in the other order Lane B's insert lost the race and its
--     evidence write matched zero rows (PR #2246 header, spec §3 item 4).
--   140's index keyed on the estimand ALONE, so the two queues shared one
--   slot. Measured against the verbatim 140 DDL in prod's own Postgres image
--   (tests/unit/test_database/test_migration_153_pending_index_review_type.py):
--   with a pending initial_dag row, both a dag_approval and a quarterly_audit
--   insert on the same estimand fail with 23505. Prod today: 41 dag_approval
--   rows, 0 pending, 0 initial_dag (read-only probe 2026-09-23) -- latent, and
--   Lane B's real authoring runs are about to open the first initial_dag rows.
--
-- WHY two indexes and not (estimand_key, review_type): a per-type key would
--   let a dag_approval consult and a quarterly_audit renewal be pending side by
--   side on one estimand -- the sibling-row problem 140 removed and #2090
--   relies on. The split is by QUEUE; today that is initial_dag vs everything
--   else. A future author-side type must be added to BOTH predicates here (and
--   to STRUCTURAL_AUTHOR_REVIEW_TYPE in src/repositories/expert_review.py).
--
-- ALSO (codex r3): two DB-side readers of the review chronology ranked BOTH
--   queues and are replaced below with the runtime-queue predicate:
--   * public.dag_structure_rejected(hash, brand) (migration 134) -- the
--     predicate INSIDE promote_causal_path_guarded, the RefutationNode's sole
--     promoter write (src/repositories/causal_path.py). Unfixed, a Lane B
--     initial_dag REJECTION of a hash would refuse a promotion the Python gate
--     does not refuse, and a newer initial_dag APPROVAL could mask a runtime
--     rejection and let a refused structure promote.
--   * public.is_dag_approved(hash, brand) (ml/010) -- the schema-owned mirror
--     of ExpertReviewRepository.is_dag_approved (no in-repo caller; service_role
--     callable). Unfixed, it would answer true on an initial_dag approval where
--     the Python reader now answers false.
--   Both are CREATE OR REPLACE with identical signatures (privileges and
--   ownership are kept by Postgres); 134's grants are re-asserted below.
--   The operator views (v_pending_expert_reviews, ml/010) keep BOTH queues on
--   purpose: the Expert Reviews page shows the owner every review.
--
-- SAFETY: index DDL plus the two CREATE OR REPLACE FUNCTION statements; no row
--   is written or deleted. The new keys
--   PARTITION the old one (every row that was unique under 140 is unique under
--   its queue), so the creates cannot fail on existing rows. Plain CREATE INDEX
--   (not CONCURRENTLY): run_migrations.sh wraps this file in --single-
--   transaction, so the drop and both creates land together or not at all --
--   there is no window without pending uniqueness. The old index is dropped
--   FIRST and the new ones carry NEW names, so IF NOT EXISTS can never no-op on
--   the old definition. Table lock is brief (41 rows).
--   DEPLOY WINDOW (codex r1/r2 HIGH, dispositioned -- owner decision): deploy.sh
--   applies this file BEFORE flipping containers, so for the length of the
--   flip the OLD image (gate and recovery not queue-aware) runs on the NEW
--   schema. Two cases in that window, both needing a pending initial_dag row
--   on the estimand of a runtime consult:
--   (a) the initial_dag row already existed: the old gate adopts it exactly as
--       it does today under 140 -- the status-quo harm, not a new one.
--   (b) NEW to the window (codex r2): a pending dag_approval consult exists,
--       and Lane B's `author_cohort_dag.py --review` inserts its initial_dag row
--       AFTER this file commits and BEFORE the flip. Under 140 that insert hit
--       23505 and the script exited 3 (loud, nothing corrupted); under this
--       index it succeeds, and an old-image consult on that estimand during the
--       rest of the flip takes the NEWEST pending row -- the authored one --
--       and advances its snapshot to the runtime structure. This requires the
--       owner's paid authoring run to write inside the container flip
--       (minutes) on an estimand a runtime consult then hits; the operational
--       guard is: do not run `--review` while a deploy is in progress. Prod
--       holds 0 pending and 0 initial_dag rows today.
--   The reverse order (new image on the old schema) is never produced by
--   deploy.sh and would fail CLOSED (the new gate ignores the authored row,
--   its insert hits 140's index, the queue-scoped recovery finds nothing ->
--   BLOCKED). A two-deploy rollout (readers first, index second) trades (b)
--   for that fail-closed state during its first stage and needs a second
--   deploy; it is offered to the owner in PR #2244's lane, not taken here.
-- REVERSE: database/migrations/rollback_153_expert_reviews_pending_index_per_queue.sql
--   (restores 140's exact index; refuses while both queues hold a pending row
--   on one estimand; deletes this file's ledger row so a re-apply is not
--   skipped). Never edit 140 itself: this file supersedes its index and points
--   back to it.
-- ============================================================================

DROP INDEX IF EXISTS uq_er_pending_estimand;

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand_runtime
    ON public.expert_reviews (estimand_key)
    WHERE approval_status = 'pending'
      AND review_type <> 'initial_dag';

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand_structural
    ON public.expert_reviews (estimand_key)
    WHERE approval_status = 'pending'
      AND review_type = 'initial_dag';

COMMENT ON INDEX uq_er_pending_estimand_runtime IS
    'At most one PENDING review per estimand in the RUNTIME queue (migration 153, #2244; '
    'replaces 140''s uq_er_pending_estimand): the gate''s dag_approval consult and the '
    'quarterly_audit renewals minted on it share this slot; a new structure on the same '
    'estimand appends a version (expert_review_versions, migration 141).';

COMMENT ON INDEX uq_er_pending_estimand_structural IS
    'At most one PENDING review per estimand in the STRUCTURAL-AUTHOR queue (migration 153, '
    '#2244): the Lane B initial_dag review (migration 152) written by '
    'scripts/author_cohort_dag.py --review and read back only by structural_prior_loader.';

-- ---------------------------------------------------------------------------
-- DB-side chronology readers: runtime queue only (see ALSO in the header).
-- Body identical to migration 134's except the review_type predicate on BOTH
-- scans (the latest adjudication AND the reopen check).
-- ---------------------------------------------------------------------------

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
          AND r.review_type <> 'initial_dag'  -- migration 153: runtime queue only
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
                      AND p.review_type <> 'initial_dag'  -- migration 153
                      AND p.approval_status = 'pending'
                      AND p.created_at > l.created_at  -- a tie is NOT a reopen
                )
         FROM latest_np l),
        false
    );
$$;

COMMENT ON FUNCTION public.dag_structure_rejected(text, text) IS
    'Lane 1 (migration 134; runtime queue only since migration 153, #2244): the '
    'expert-review chronology rule in SQL -- true when the newest non-pending RUNTIME '
    'review (not a Lane B initial_dag row) of this DAG hash (and brand, when given) is '
    'rejected and no pending runtime review is newer. Python mirror: ExpertReviewGate'
    '._latest_adjudication over runtime_review_queue(...). NULL hash reads false.';

-- ml/010's schema-owned mirror of ExpertReviewRepository.is_dag_approved: same
-- signature, same body, plus the runtime-queue predicate.
CREATE OR REPLACE FUNCTION public.is_dag_approved(
    p_dag_hash VARCHAR(64),
    p_brand VARCHAR(50) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM public.expert_reviews
        WHERE dag_version_hash = p_dag_hash
          AND approval_status = 'approved'
          AND review_type <> 'initial_dag'  -- migration 153: runtime queue only
          AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
          AND (p_brand IS NULL OR brand = p_brand)
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.is_dag_approved(VARCHAR, VARCHAR) IS
    'Check if a DAG (by hash) has an active RUNTIME-queue expert approval (migration 153, '
    '#2244: a Lane B initial_dag approval never counts; mirrors '
    'ExpertReviewRepository.is_dag_approved).';

DO $$
DECLARE
    v_fn text;
BEGIN
    -- 134's grant posture must survive the replace (CREATE OR REPLACE keeps
    -- privileges; asserted rather than assumed).
    FOREACH v_fn IN ARRAY ARRAY['public.dag_structure_rejected(text, text)'] LOOP
        IF NOT has_function_privilege('service_role', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 153: service_role cannot EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('anon', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 153: anon can still EXECUTE %', v_fn;
        END IF;
        IF has_function_privilege('authenticated', v_fn, 'EXECUTE') THEN
            RAISE EXCEPTION 'migration 153: authenticated can still EXECUTE %', v_fn;
        END IF;
    END LOOP;
    -- Behavioural smoke: an unknown hash is neither rejected nor approved.
    IF public.dag_structure_rejected('migration-153-no-such-hash', NULL) THEN
        RAISE EXCEPTION 'migration 153: unknown hash reads as rejected';
    END IF;
    IF public.is_dag_approved('migration-153-no-such-hash', NULL) THEN
        RAISE EXCEPTION 'migration 153: unknown hash reads as approved';
    END IF;
END $$;
