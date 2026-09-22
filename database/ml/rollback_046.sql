-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/046_ab_results_one_final_per_experiment.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand,
-- together with the code revert (the writer's FinalResultAlreadyPersisted handling is harmless
-- without the index — a 23505 simply never occurs — so the order does not matter):
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_046.sql
--
-- Drops the partial unique index, restoring the pre-046 state where two deliveries of
-- compute_experiment_results(final) racing within one round-trip can both persist a final row.
--
-- The 046 dedupe is NOT reversible: any duplicate final rows it removed are gone (their fidelity
-- comparisons were repointed to the surviving row, so no audit pointer was lost). Measured 0
-- duplicates on prod at writing time, so on prod there was nothing to remove.
--
-- Also deletes the ml/046 ledger row, so the next deploy re-applies it.
-- Idempotent: the drop is IF EXISTS, so a second run changes nothing.
-- ============================================================================

DROP INDEX IF EXISTS public.uq_ab_results_one_final_per_experiment;

DELETE FROM public.schema_migrations
 WHERE filename = 'ml/046_ab_results_one_final_per_experiment.sql';
