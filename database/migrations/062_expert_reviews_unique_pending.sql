-- Migration 062: M-reach1 — prevent concurrent DUPLICATE pending expert reviews
-- ============================================================================
-- Two concurrent causal_impact runs on the SAME dag_version_hash (+brand) could
-- both pass the "no pending review" check in
-- ExpertReviewGate.check_approval (src/causal_engine/expert_review_gate.py) and
-- both INSERT a pending row, leaving an orphan duplicate in the review queue.
--
-- A partial UNIQUE index makes the second concurrent INSERT fail; the repository
-- recovers gracefully by returning the winner's pending review_id
-- (src/repositories/expert_review.py::create_review, M-reach1).
--
-- COALESCE(brand, '') so a NULL brand collides with another NULL-brand pending
-- row (a bare UNIQUE treats NULLs as DISTINCT, which would defeat the guard).
-- dag_version_hash NULL is left distinct on purpose: the gate only creates
-- reviews with a real DAG hash, so a NULL-hash row is not a concurrent-dedup
-- target. Mirrors the existing partial index idx_er_active_approvals (which
-- guards approval_status='approved').
--
-- Idempotent (IF NOT EXISTS). Verified 0 pending rows / 0 duplicate groups on the
-- live DB before authoring, so the index builds cleanly.

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_dag_brand
    ON expert_reviews (dag_version_hash, (COALESCE(brand, '')))
    WHERE approval_status = 'pending';

COMMENT ON INDEX uq_er_pending_dag_brand IS
    'M-reach1: at most one PENDING expert review per (dag_version_hash, brand) — '
    'blocks concurrent duplicate review-queue rows. NULL brand normalized via '
    'COALESCE so two NULL-brand pending rows collide.';
