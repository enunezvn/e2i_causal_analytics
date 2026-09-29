-- Rollback for migration 164 (NOT auto-applied: the runner skips rollback_* files).
-- Run it as the runbook runs rollbacks: psql as postgres, one transaction.
--
-- Returns exactly the rows 164 rewrote (they carry its selection_backfill marker) to their
-- pre-164 shape: algorithm_name / selection_score back to JSON null, the primary_reason and
-- selection_backfill keys removed, and the pre-164 description. Rows written by the fixed hook
-- carry no marker and are never touched.
--
-- It also removes 164's ledger row, so a later deploy re-applies 164 (the 158 rollback does the
-- same). A second run matches zero rows.

UPDATE episodic_memories em
   SET raw_content = (em.raw_content - 'selection_backfill' - 'primary_reason')
                     || jsonb_build_object('algorithm_name', NULL, 'selection_score', NULL),
       description = 'Model Selection: unknown (unknown). Score: 0.00. Reason: N/A'
 WHERE em.agent_name = 'model_selector'
   AND em.event_type = 'model_selection_completed'
   AND em.raw_content -> 'selection_backfill' ->> 'migration'
       = '164_backfill_model_selector_episodic_selection';

DELETE FROM public.schema_migrations
 WHERE filename = '164_backfill_model_selector_episodic_selection.sql';
