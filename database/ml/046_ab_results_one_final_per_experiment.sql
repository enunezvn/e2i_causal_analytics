-- ============================================================================
-- Migration: 046_ab_results_one_final_per_experiment.sql
-- Purpose: One FINAL result row per experiment, enforced at the database (#2206, owner fix)
-- Dependencies: 021_ab_results_tables.sql (ab_experiment_results, ab_fidelity_comparisons)
-- ============================================================================
--
-- NUMBERING: 046, not 045. PR #2223 (persist the HPO study) owns
-- ml/045_persist_hpo_study_rpc.sql on its own lane; both PRs merge, so this file skips the
-- number rather than collide. The live ledger holds ml/040..ml/044 at writing time.
--
-- THE RACE THIS CLOSES. compute_experiment_results(final) runs under Celery late-ack: a worker
-- lost after persisting the final row gets the task again. The task checks for an existing
-- final row before recomputing (idempotent in the common case), but the check and the INSERT
-- are separated by the outcome-feed load, so two deliveries racing within that window both
-- pass the check and both insert. ab_experiment_results had no unique key beyond its primary
-- key (measured 2026-09-22: pkey + 4 plain btrees, 360 rows, every one 'final', ZERO
-- duplicates). Downstream, the fidelity producer takes the NEWEST final row
-- (results_analysis.compare_experiment_to_twin) and rolls it into
-- digital_twin_models.fidelity_score, so a duplicate would make the scored row a coin flip.
--
-- WHY A PARTIAL INDEX (history vs singleton, the migrations/119 rationale): only 'final' is a
-- singleton, and only because the code says so. 'interim' rows in this table are legitimate
-- HISTORY — compute_experiment_results defaults to analysis_type='interim', has no scheduled
-- producer, and an operator may recompute it at successive milestones (the scheduled interim
-- sweep persists to ab_interim_analyses, a different table with its own unique_analysis_number).
-- 'post_hoc' is exploratory by definition. A table-wide unique key on
-- (experiment_id, analysis_type) would forbid both, so the rule is scoped by predicate.
--
-- WRITER CONTRACT (src/repositories/ab_results.py): PostgREST's upsert can only emit
-- ON CONFLICT (experiment_id) DO NOTHING, and Postgres rejects that against a PARTIAL index
-- ("there is no unique or exclusion constraint matching the ON CONFLICT specification" —
-- measured on this schema). So the writer keeps a plain INSERT and treats a 23505 on a FINAL
-- row as "another delivery already persisted the final row" (FinalResultAlreadyPersisted);
-- the task then skips its recompute and still enqueues fidelity tracking.
--
-- ORDER OF OPERATIONS (all in the runner's single transaction):
--   1) guarded dedupe of any pre-existing duplicate finals per experiment, keeping the
--      EARLIEST computed_at, and FIRST repointing ab_fidelity_comparisons.results_id of the
--      losers to the survivor — that FK is ON DELETE SET NULL, so deleting a loser without the
--      repoint would silently null a comparison's audit pointer. Skipped entirely once the
--      index exists (second apply / deploy re-scour): with the index in place there cannot be
--      duplicates. Measured 0 duplicates live, so on prod this block changes nothing.
--   2) the partial unique index (IF NOT EXISTS — the accepted idempotent form).
-- No NOTIFY pgrst: an index is not a schema-cache change.
--
-- Rollback: database/ml/rollback_046.sql (drops the index; the dedupe is not reversible).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1) Guarded dedupe: repoint, then delete. No-op when the index already exists
--    or when no experiment holds more than one final row.
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    v_repointed integer := 0;
    v_deleted   integer := 0;
BEGIN
    IF to_regclass('public.uq_ab_results_one_final_per_experiment') IS NOT NULL THEN
        RAISE NOTICE 'ml/046: uq_ab_results_one_final_per_experiment already present; dedupe skipped';
        RETURN;
    END IF;

    -- losers = every final row per experiment except the earliest computed_at
    -- (ties: earliest created_at, then lowest id — deterministic across re-runs).
    CREATE TEMP TABLE ml046_losers ON COMMIT DROP AS
    WITH keepers AS (
        SELECT DISTINCT ON (experiment_id) experiment_id, id AS keeper_id
        FROM public.ab_experiment_results
        WHERE analysis_type = 'final'
        ORDER BY experiment_id, computed_at ASC NULLS LAST, created_at ASC NULLS LAST, id ASC
    ),
    losers AS (
        SELECT r.id AS loser_id, k.keeper_id
        FROM public.ab_experiment_results r
        JOIN keepers k ON k.experiment_id = r.experiment_id
        WHERE r.analysis_type = 'final'
          AND r.id <> k.keeper_id
    )
    SELECT loser_id, keeper_id FROM losers;

    -- Repoint audit pointers BEFORE the delete (FK is ON DELETE SET NULL).
    UPDATE public.ab_fidelity_comparisons c
       SET results_id = l.keeper_id
      FROM ml046_losers l
     WHERE c.results_id = l.loser_id;
    GET DIAGNOSTICS v_repointed = ROW_COUNT;

    DELETE FROM public.ab_experiment_results r
     USING ml046_losers l
     WHERE r.id = l.loser_id;
    GET DIAGNOSTICS v_deleted = ROW_COUNT;

    IF v_deleted > 0 OR v_repointed > 0 THEN
        RAISE NOTICE 'ml/046: removed % duplicate final result row(s); repointed % fidelity comparison(s)',
            v_deleted, v_repointed;
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 2) The singleton rule for FINAL rows only. interim / post_hoc rows stay
--    repeatable history (see header).
-- ----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_ab_results_one_final_per_experiment
    ON public.ab_experiment_results (experiment_id)
    WHERE analysis_type = 'final';

COMMENT ON INDEX public.uq_ab_results_one_final_per_experiment IS
    'ml/046 (#2206): one FINAL result row per experiment. Partial: interim and post_hoc rows are '
    'repeatable history. The writer treats a 23505 on a final insert as "another delivery '
    'already persisted the final row" (PostgREST cannot emit ON CONFLICT against a partial index).';
