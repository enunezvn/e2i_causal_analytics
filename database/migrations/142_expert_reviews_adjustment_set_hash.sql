-- ============================================================================
-- Migration 142: expert_reviews.adjustment_set_hash (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: give the REVIEW ROW the second half of its current version identity. The
--   row already carries dag_version_hash; this adds the adjustment-set hash of
--   the structure the review currently covers, so the pair
--   (dag_version_hash, adjustment_set_hash) identifies that structure the same
--   way migration 141's version rows already identify theirs. NULL means the
--   adjustment set is UNKNOWN -- never "empty", which is the distinct canonical
--   value sha256("[]") (src/causal_engine/dag_hash.py compute_adjustment_set_hash).
-- WHY (codex round 2, three HIGH findings with one root cause): compute_dag_hash
--   deliberately EXCLUDES adjustment sets, so the DAG hash alone cannot see a
--   covariate-only change. Every guard that binds to the REVIEW ROW was therefore
--   matching on half an identity, and an ADJUSTMENT-ONLY advance (same DAG, new
--   adjustment set) walked past all three of them:
--     1. submit_review -- a form opened on (h1, {W}) still resolved the review
--        after a run advanced it to (h1, {Z}): the hash filter matched, so a
--        reviewer signed off a structure they were never shown.
--     2. update_agent_assessment -- an advisory assessment BUILT from (h1, {W})
--        still persisted after the advance to (h1, {Z}) cleared the cache, so the
--        review UI showed a grading of a structure the review no longer covers.
--     3. advance_review -- two concurrent adjustment-only advances (h1, B) and
--        (h1, C) both passed the hash-only compare-and-set, leaving the review on
--        B while the timeline's latest row was C; later C runs then saw both the
--        hash and the latest pair match and skipped the repair forever.
--   None of the three is fixable in Python while the row cannot SAY which
--   adjustment set it is on. The version TIMELINE keeps its own
--   adjustment_set_hash (141): that column is the history, this one is the
--   review's CURRENT state, and the two answer different questions.
-- WHY VARCHAR(64): the same type as the column it partners with --
--   expert_reviews.dag_version_hash is character varying(64) (measured live
--   2026-09-14) and both are hex SHA-256 digests. Two halves of one identity
--   stored in different types would invite a silent cast at comparison time.
-- WHY no DEFAULT, no CHECK, no BACKFILL: there is nothing on expert_reviews to
--   derive the hash FROM. dag_structure_json holds the snapshot, but a review
--   whose snapshot is NULL, or one written before the gate computed the hash at
--   all, has no recoverable adjustment set -- and manufacturing sha256("[]") for
--   it would assert an EMPTY adjustment set, a plausible-wrong value that reads
--   exactly like a real measurement. Migration 141 left its 41 backfilled version
--   rows NULL for this same reason. Unknown stays unknown; the first run that
--   advances or re-encounters a review populates it honestly, and the Python
--   guards match NULL explicitly (an IS NULL filter) rather than ignoring it.
-- SAFETY: purely additive and therefore safe through the DEPLOY WINDOW, the rule
--   migrations 136 and 140 follow: run_migrations.sh applies this file BEFORE the
--   api image flips, so the OLD image -- whose create_review payload never
--   mentions adjustment_set_hash -- keeps inserting successfully against a
--   nullable, defaultless column it cannot see. Nothing is dropped, renamed or
--   made required; no row is rewritten; no lock is held beyond the ALTER's own
--   (PG11+ adds a nullable column without a table rewrite). No own transaction
--   (run_migrations.sh wraps the file in --single-transaction and appends the
--   ledger row).
-- REVERSE (manual, not run by this file):
--   ALTER TABLE public.expert_reviews DROP COLUMN adjustment_set_hash;
--   also DELETE FROM public.schema_migrations WHERE filename =
--   '142_expert_reviews_adjustment_set_hash.sql' -- run_migrations.sh's ledger
--   check (line ~117) skips any file already recorded there, so leaving that row
--   in place would make a later re-apply of this file silently no-op instead of
--   re-adding the column.
-- ============================================================================

ALTER TABLE public.expert_reviews
    ADD COLUMN IF NOT EXISTS adjustment_set_hash VARCHAR(64);

COMMENT ON COLUMN public.expert_reviews.adjustment_set_hash IS
    'The adjustment-set half of the review''s CURRENT version identity (migration 142, '
    '#1991 debt 3). Paired with dag_version_hash it names the structure the review covers; '
    'NULL means the adjustment set is unknown (a pre-142 row, or a run with no structure in '
    'scope), which is NOT the empty set. The version timeline expert_review_versions keeps '
    'its own adjustment_set_hash for the history.';
