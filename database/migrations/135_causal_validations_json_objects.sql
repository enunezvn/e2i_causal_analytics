-- ============================================================================
-- Migration 135: causal_validations evidence columns hold JSON OBJECTS (lane 1)
-- ============================================================================
-- WHAT: decode every row whose details_json / test_config is a JSON *string*
--   into the object it encodes. The Python writer json.dumps'ed into the jsonb
--   column since 0742b81f6, so every agent-path row was a string (480 live rows
--   on 2026-09-08, all estimate_source = causal_impact_query); the 545 rows the
--   migration-119 DGP seed wrote were already objects. Idempotent: a re-run
--   finds nothing to decode.
-- WHY: the lane's per-resample evidence must be queryable
--   (jsonb_array_length(details_json->'subset_effects')) and testable in ONE
--   shape (owner decision 2026-09-09).
-- SAFETY: pure data fix, no DDL, deliberately NO CHECK constraint -- migrations
--   run before the container flips, and a constraint would make the OLD image's
--   writer fail during that window. The writer fix ships in the same deploy;
--   Task 14 certifies zero string rows afterwards. Rehearsed BEGIN/ROLLBACK
--   2026-09-09: 480 -> 0 string rows in both columns, every decoded row readable.
-- ============================================================================

UPDATE public.causal_validations
   SET details_json = (details_json #>> '{}')::jsonb
 WHERE jsonb_typeof(details_json) = 'string';

UPDATE public.causal_validations
   SET test_config = (test_config #>> '{}')::jsonb
 WHERE jsonb_typeof(test_config) = 'string';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.causal_validations
         WHERE jsonb_typeof(details_json) = 'string' OR jsonb_typeof(test_config) = 'string'
    ) THEN
        RAISE EXCEPTION 'migration 135: string-shaped evidence rows remain';
    END IF;
END $$;
