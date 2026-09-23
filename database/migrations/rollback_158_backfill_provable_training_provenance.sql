-- Rollback for migration 158 (NOT auto-applied: the runner skips rollback_* files).
-- Returns the two rows 158 wrote to NULL, only while they still carry the value 158 set.
-- A second run matches zero rows.

UPDATE ml_model_registry
   SET training_provenance = NULL
 WHERE id IN ('d765b451-12df-46df-955f-63359b506b52', '5fd7826b-28d7-491b-b9b1-8b5494dbe1ff')
   AND training_provenance = 'synthetic_gold';
