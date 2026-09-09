-- ============================================================================
-- Migration 136: expert_reviews.resolved_at (lane 1, codex whole-diff HIGH F1)
-- ============================================================================
-- WHAT: one nullable column, public.expert_reviews.resolved_at TIMESTAMPTZ --
--   the time an operator resolved the review, for BOTH statuses. Written by
--   ExpertReviewRepository.submit_review (src/repositories/expert_review.py)
--   as now() together with the resolver's reviewer_name / reviewer_email,
--   which the resolve route derives from the authenticated operator
--   (src/api/routes/expert_review.py resolve_review).
-- WHY: the linked-review card showed Reviewer = reviewer_name ?? reviewer_id
--   and Decided = approved_at ?? created_at. reviewer_id holds the REQUESTER
--   (the originating query id: src/agents/causal_impact/nodes/refutation.py
--   -> expert_review_gate.py create_review(reviewer_id=requester_id)), no
--   live row carries reviewer_name / reviewer_email (0 of 40 on 2026-09-09),
--   and a rejection wrote no timestamp at all (approved_at is set only on
--   approval) -- so every resolved row would have named a query id as the
--   reviewer and its creation date as the decision date. This column plus the
--   writer change make the decision time recordable; the card renders only
--   what was recorded.
-- NO BACKFILL: rows resolved before this migration stay NULL. updated_at is
--   trigger-maintained and is NOT a decision time (the one live rejected row
--   had its cached agent assessment written after its rejection). Unknown
--   stays unknown.
-- SAFETY: additive, idempotent (ADD COLUMN IF NOT EXISTS), no default, no
--   constraint; the old image's writer keeps working during the deploy window.
--   No BEGIN/COMMIT of its own -- scripts/run_migrations.sh wraps the file in
--   --single-transaction.
-- ============================================================================

ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

COMMENT ON COLUMN public.expert_reviews.resolved_at IS
    'Lane 1 (migration 136): when an operator resolved this review, for BOTH '
    'statuses (approved and rejected). Written as now() by '
    'ExpertReviewRepository.submit_review together with the resolver''s '
    'reviewer_name / reviewer_email. NULL for rows resolved before this '
    'migration -- deliberately not backfilled, the trigger-maintained updated_at '
    'is not a decision time.';
