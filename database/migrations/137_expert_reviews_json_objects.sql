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
--   the same deploy.
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
