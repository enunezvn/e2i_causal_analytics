-- ============================================================================
-- Migration 137: expert_reviews JSON columns hold JSON OBJECTS (#1992)
-- ============================================================================
-- WHAT: decode every row whose checklist_json / comments_json /
--   agent_assessment_json / dag_structure_json is a JSON *string* into the
--   object it encodes. src/repositories/expert_review.py's writers
--   json.dumps'ed into each jsonb column, so PostgREST stored a JSON string
--   scalar instead of an object (measured live 2026-09-09, rehearsed
--   BEGIN/ROLLBACK: checklist_json 3, comments_json 2, agent_assessment_json
--   6, dag_structure_json 40, of 40 total rows).
--   Idempotent: a re-run finds nothing to decode.
-- WHY: mirrors migration 135 (causal_validations) for the same defect class
--   (#1992) -- the evidence must be queryable and testable in ONE shape.
-- SAFETY: pure data fix, no DDL, deliberately NO CHECK constraint --
--   migrations run before the container flips, and a constraint would make
--   the OLD image's writer fail during that window. The writer fix ships in
--   the same deploy. (col #>> '{}')::jsonb RAISES 'invalid input syntax for
--   type json' on a string cell that is not valid JSON text (positive-
--   controlled with a plain-text cell and a bare NaN token); run inside the
--   migration runner's transaction, that error aborts the whole deploy
--   before the container flips, nothing half-applied. Live data has ZERO
--   such rows today (measured 2026-09-09) on all four columns; a plain
--   prefix check (does the string start with '{' or '[') is NOT a reliable
--   recovery query -- it MISSES a malformed cell that happens to start with
--   '{' or '[' (codex round 2 LOW), and the live server is PostgreSQL 15.8,
--   where pg_input_is_valid() is not available. Recovery: attempt the cast
--   row by row and catch invalid_text_representation, e.g.
--   DO $$ DECLARE r record; c text; BEGIN
--     FOREACH c IN ARRAY ARRAY['checklist_json','comments_json',
--         'agent_assessment_json','dag_structure_json'] LOOP
--       FOR r IN EXECUTE format(
--           'SELECT review_id, %I #>> ''{}'' AS s FROM public.expert_reviews'
--           ' WHERE jsonb_typeof(%I) = ''string''', c, c) LOOP
--         BEGIN PERFORM r.s::jsonb;
--         EXCEPTION WHEN invalid_text_representation THEN
--           RAISE NOTICE 'malformed %: %', c, r.review_id;
--         END;
--       END LOOP;
--     END LOOP;
--   END $$;
--   Positive-controlled ROLLBACK-only on the live DB 2026-09-09: a throwaway
--   row with checklist_json = '{not json' (a string scalar starting with
--   '{' that is not valid JSON) produced exactly one NOTICE naming that
--   row's review_id; run again against the real rows with no throwaway
--   insert, it reported none.
--   TRANSITIONAL WINDOW (codex round 1 MED): run_migrations.sh tracks this
--   file by filename in public.schema_migrations and never re-runs it, so
--   any row the OLD (pre-fix) image writes between this migration running
--   and the container flip stays string-shaped for good -- NOT re-decoded
--   by a later deploy. Same decision as migration 135's precedent for
--   causal_validations (owner decision 6): no trigger, no constraint to
--   close that window. The four UPDATE statements above are idempotent and
--   safe to re-run by hand at any time; if post-deploy certification finds
--   any string cells remaining, re-run them once the new writer is live.
-- ============================================================================

UPDATE public.expert_reviews
   SET checklist_json = (checklist_json #>> '{}')::jsonb
 WHERE jsonb_typeof(checklist_json) = 'string';

UPDATE public.expert_reviews
   SET comments_json = (comments_json #>> '{}')::jsonb
 WHERE jsonb_typeof(comments_json) = 'string';

UPDATE public.expert_reviews
   SET agent_assessment_json = (agent_assessment_json #>> '{}')::jsonb
 WHERE jsonb_typeof(agent_assessment_json) = 'string';

UPDATE public.expert_reviews
   SET dag_structure_json = (dag_structure_json #>> '{}')::jsonb
 WHERE jsonb_typeof(dag_structure_json) = 'string';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.expert_reviews
         WHERE jsonb_typeof(checklist_json) = 'string'
            OR jsonb_typeof(comments_json) = 'string'
            OR jsonb_typeof(agent_assessment_json) = 'string'
            OR jsonb_typeof(dag_structure_json) = 'string'
    ) THEN
        RAISE EXCEPTION 'migration 137: string-shaped expert_reviews rows remain';
    END IF;
END $$;
