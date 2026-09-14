-- ============================================================================
-- Migration 140: expert_reviews keyed on the ESTIMAND (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: (1) estimand_key TEXT GENERATED ALWAYS AS (...) STORED =
--   lower(brand):treatment:outcome -- a generated column, not a backfilled
--   plain column, so neither the OLD nor the NEW api image ever writes it
--   (deploy.sh runs migrations at line ~91, BEFORE flipping containers at
--   line ~109; migration 136's rule applies here too: "no default, no
--   constraint; the old image's writer keeps working during the deploy
--   window" -- a plain column + ALTER ... SET NOT NULL would make
--   ExpertReviewRepository.create_review's insert 23502 for the whole
--   window, since it strips None values and recovers only 23505/duplicate,
--   src/repositories/expert_review.py:215-252). Every operand is COALESCEd
--   to '', so the expression can never evaluate to NULL -- Postgres does not
--   let a generated column declare NOT NULL directly, but the expression's
--   own totality gives the same guarantee. When brand IS NULL the key falls
--   in the ':treatment:outcome' bucket (6 live rows today); Python callers
--   must compute the identical lower/COALESCE/':' expression (estimand_key_for)
--   for any lookup to hit the same row PostgREST would derive.
--   (2) pending uniqueness moves from (dag_version_hash, brand) [migration 062]
--   to (estimand_key), so a covariate change on the same estimand UPDATES the
--   pending review (a new structure version, migration 141) instead of
--   minting a sibling; (3) every pending row minted by a BLOCK-band run is
--   resolved as 'superseded' -- a BLOCK run is terminal before the review is
--   consulted, so approving it changes nothing (measured 2026-09-13: 38
--   pending, all gate=block, 28 estimands, 7 of those estimands hold more
--   than one pending row -- the supersede MUST run before
--   uq_er_pending_estimand exists or that index's own creation would fail on
--   the duplicates).
-- WHY the adjustment set is NOT in the key: a covariate change is the event that
--   must update a review, not mint one. The hash and adjustment set are the
--   VERSION (141), not the identity.
-- WHY updated_at is not used for 141's version backfill: trg_er_updated_at
--   (BEFORE UPDATE) bumps updated_at on ANY update to the row, including this
--   migration's own supersede UPDATE -- migration 136 already established
--   updated_at is not a decision time. 141 must derive version ordering from
--   created_at (or its own new column), not updated_at.
-- SAFETY: additive except the index swap; the supersede is a guarded UPDATE
--   that raises (rolling the whole file back under run_migrations.sh's
--   --single-transaction) unless it touches exactly the rows counted at the
--   top of the block. A precondition before that UPDATE also raises if any
--   pending row's comments_json is non-NULL and not a JSON object (live shape
--   today: object 2, NULL 39 -- the `||` jsonb concatenation used in the
--   supersede would silently produce an array instead of raising if a
--   pending row ever held a non-object comments_json). Nothing is deleted.
--   dag_version_hash and its approval indexes are untouched; approval
--   lookups by hash stay valid.
-- REVERSE (manual, not run by this file): DROP INDEX IF EXISTS
--   idx_er_estimand_created; DROP INDEX IF EXISTS uq_er_pending_estimand;
--   PRECONDITION for BOTH statements below: uq_er_pending_dag_brand keys on
--     (dag_version_hash, COALESCE(brand,'')), which is COARSER than the estimand
--     it replaces -- two pending reviews of the same brand and structure that
--     differ only in treatment/outcome are legal under uq_er_pending_estimand
--     and forbidden under the restored index. So check for collisions across
--     EVERY row that will be pending after the reverse: the rows ALREADY PENDING
--     (minted since this migration, which can collide with EACH OTHER -- the
--     CREATE UNIQUE INDEX below then fails before anything is restored) AND the
--     superseded rows the UPDATE restores (which can collide with those and with
--     one another). Resolve or exclude every colliding row FIRST; otherwise the
--     reverse fails mid-transaction, or a restored row takes a queue slot a live
--     pending review is using.
--   CREATE UNIQUE INDEX uq_er_pending_dag_brand ON public.expert_reviews
--     USING btree (dag_version_hash, COALESCE(brand, ''::character varying))
--     WHERE ((approval_status)::text = 'pending'::text);
--   UPDATE public.expert_reviews SET approval_status = 'pending',
--     resolved_at = NULL,
--     comments_json = comments_json - 'superseded_reason'
--     WHERE comments_json ? 'superseded_reason';
--   ALTER TABLE public.expert_reviews DROP COLUMN estimand_key;
--   also DELETE FROM public.schema_migrations WHERE filename =
--   '140_expert_reviews_estimand_key.sql' -- run_migrations.sh's ledger check
--   (line ~117) skips any file already recorded there, so leaving that row in
--   place would make a later re-apply of this file silently no-op instead of
--   recreating the column and the pending-uniqueness index.
-- ============================================================================

ALTER TABLE public.expert_reviews
    ADD COLUMN IF NOT EXISTS estimand_key TEXT
    GENERATED ALWAYS AS (
        lower(COALESCE(brand, '')) || ':' || lower(COALESCE(treatment_variable, ''))
        || ':' || lower(COALESCE(outcome_variable, ''))
    ) STORED;

COMMENT ON COLUMN public.expert_reviews.estimand_key IS
    'Review identity (migration 140, #1991 debt 3): GENERATED ALWAYS AS '
    'lower(brand):treatment:outcome STORED -- writers must never send this '
    'column, Postgres derives it. dag_version_hash is the structure VERSION '
    'the review currently covers, not its identity.';

-- Supersede the BLOCK-band pending rows BEFORE the new uniqueness lands (several
-- share an estimand -- 4 for Remibrutinib treatment_arm -> persistent_180d).
DO $$
DECLARE
    v_expected INTEGER;
    v_done INTEGER;
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.expert_reviews
         WHERE approval_status = 'pending'
           AND comments_json IS NOT NULL
           AND jsonb_typeof(comments_json) <> 'object'
    ) THEN
        RAISE EXCEPTION 'migration 140: a pending row has a non-object comments_json';
    END IF;

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
