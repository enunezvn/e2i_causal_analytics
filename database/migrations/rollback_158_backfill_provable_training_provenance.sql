-- Rollback for migration 158 (NOT auto-applied: the runner skips rollback_* files).
-- Run it as the runbook runs rollbacks: psql as postgres, one transaction.
--
-- Returns the two rows 158 wrote to NULL, under the same proof predicates 158 used, only while
-- they still carry the value 158 set. It cannot tell 158's write from an identical one: the
-- only other writer of these rows is the prediction_synthesizer_deploy CLI re-registering the
-- same (model_name, model_version), which since #2259 writes the same 'synthetic_gold'. After
-- a rollback, re-running that CLI (or re-applying 158) restores the value.
--
-- It also removes 158's ledger row. The runner skips a ledgered file, so without this a later
-- deploy would never re-apply 158 and the rows would stay NULL (the 143/144 rollbacks do the
-- same). A second run matches zero rows.

UPDATE ml_model_registry r
   SET training_provenance = NULL
  FROM ml_experiments e
 WHERE e.id = r.experiment_id
   AND r.id IN ('d765b451-12df-46df-955f-63359b506b52', '5fd7826b-28d7-491b-b9b1-8b5494dbe1ff')
   AND r.training_provenance = 'synthetic_gold'
   AND r.is_synthetic = false
   AND e.experiment_name = 'csu_treatment_initiation_live_v1'
   AND r.model_name IN ('csu_treatment_initiation_lr_full_v1',
                        'csu_treatment_initiation_lr_balanced_v1')
   AND r.model_version = '1.0'
   AND r.artifact_path LIKE '%/ml_artifacts/csu_treatment_initiation/%';

DELETE FROM public.schema_migrations
 WHERE filename = '158_backfill_provable_training_provenance.sql';
