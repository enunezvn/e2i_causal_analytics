-- ============================================================================
-- Migration 140: expert_reviews keyed on the ESTIMAND (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: (1) estimand_key TEXT = lower(brand):treatment:outcome, backfilled from the
--   three columns every row already carries, then NOT NULL; (2) pending uniqueness
--   moves from (dag_version_hash, brand) [migration 062] to (estimand_key), so a
--   covariate change on the same estimand UPDATES the pending review (a new
--   structure version, migration 141) instead of minting a sibling; (3) every
--   pending row minted by a BLOCK-band run is resolved as 'superseded' -- a BLOCK
--   run is terminal before the review is consulted, so approving it changes
--   nothing (measured 2026-09-13: 38 pending, all gate=block, 28 estimands).
-- WHY the adjustment set is NOT in the key: a covariate change is the event that
--   must update a review, not mint one. The hash and adjustment set are the
--   VERSION (141), not the identity.
-- SAFETY: additive except the index swap; the supersede is a guarded UPDATE that
--   raises (rolling the whole file back under run_migrations.sh's
--   --single-transaction) unless it touches exactly the rows counted at the top
--   of the block. Nothing is deleted. dag_version_hash and its approval indexes
--   are untouched; approval lookups by hash stay valid.
-- ============================================================================

ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS estimand_key TEXT;

UPDATE public.expert_reviews
SET estimand_key = lower(COALESCE(brand, '')) || ':' || lower(COALESCE(treatment_variable, ''))
                   || ':' || lower(COALESCE(outcome_variable, ''))
WHERE estimand_key IS NULL;

ALTER TABLE public.expert_reviews ALTER COLUMN estimand_key SET NOT NULL;

COMMENT ON COLUMN public.expert_reviews.estimand_key IS
    'Review identity (migration 140, #1991 debt 3): lower(brand):treatment:outcome. '
    'dag_version_hash is the structure VERSION the review currently covers, not its identity.';

-- Supersede the BLOCK-band pending rows BEFORE the new uniqueness lands (several
-- share an estimand -- 4 for Remibrutinib treatment_arm -> persistent_180d).
DO $$
DECLARE
    v_expected INTEGER;
    v_done INTEGER;
BEGIN
    SELECT count(*) INTO v_expected FROM public.expert_reviews
     WHERE approval_status = 'pending' AND analysis_context LIKE '%gate=block%';

    UPDATE public.expert_reviews
       SET approval_status = 'superseded',
           resolved_at = now(),
           comments_json = COALESCE(comments_json, '{}'::jsonb)
               || jsonb_build_object('superseded_reason',
                  'block-band review: the run was terminal before the review was consulted, '
                  'so an approval could not change an outcome (#1991 debt 3, migration 140)')
     WHERE approval_status = 'pending' AND analysis_context LIKE '%gate=block%';
    GET DIAGNOSTICS v_done = ROW_COUNT;

    IF v_done <> v_expected THEN
        RAISE EXCEPTION 'migration 140: superseded % rows, expected %', v_done, v_expected;
    END IF;
END $$;

DROP INDEX IF EXISTS uq_er_pending_dag_brand;

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand
    ON public.expert_reviews (estimand_key)
    WHERE approval_status = 'pending';

CREATE INDEX IF NOT EXISTS idx_er_estimand_created
    ON public.expert_reviews (estimand_key, created_at DESC);

COMMENT ON INDEX uq_er_pending_estimand IS
    'At most one PENDING expert review per estimand (migration 140); a new structure on '
    'the same estimand appends a version (expert_review_versions, migration 141).';
