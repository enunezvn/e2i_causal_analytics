-- Migration 164: repair the model_selector episodic rows written before #2325.
--
-- WHY: store_model_selection read algorithm_name / selection_score / primary_reason at the top
-- level of the agent output; run() nests them under model_candidate / selection_rationale.
-- Every model_selection_completed row (188/188 in prod, read-only 2026-09-29) therefore holds
--   raw_content.algorithm_name = null, raw_content.selection_score = null, and
--   description = 'Model Selection: unknown (unknown). Score: 0.00. Reason: N/A'.
-- The description is not inert: the chat RAG returns it verbatim (hybrid_vector_search and
-- hybrid_fulltext_search select em.description AS content for every agent), and the memory
-- API lists it (GET /memory/episodic, the Memory Architecture page). "Score: 0.00" reads as a
-- measured score of zero.
--
-- SOURCE: each row's OWN raw_content. The rationale text the agent stored with the row starts
-- "Selected <algorithm> (score: <x.xxx>)" (rationale_generator._build_rationale_text, the same
-- format since the file was added in 3e1c70cf4), and its primary_reason is stored beside it.
-- 188/188 prod rows match. No join to MLflow or any other table is needed or made.
--
-- WHAT IS NOT RECOVERED, and stays absent: algorithm_family / algorithm_class (the row never
-- stored the primary's family; the description says "family not recorded"). The score is the
-- rationale's 3-decimal rendering (0.758, not 0.7580625); selection_backfill records that.
--
-- NOT CHANGED: the embedding. It was computed from the broken description; re-embedding needs
-- the embedding API and is an optional owner follow-up. The stale vector still points at
-- "model selection", so what a vector hit now returns is the corrected text.
--
-- SAFETY: compare-and-set. Only model_selector model_selection_completed rows whose
-- description is still exactly the broken text, whose structured fields are still null, and
-- whose rationale text matches the pattern. A second application matches zero rows.
--
-- APPLIED BY: the first deploy after merge (deploy.yml runs scripts/run_migrations.sh BEFORE
-- the services flip). A model_selector run landing between the migration and the flip is
-- written by the OLD hook and stays broken; check after the deploy with
--   select count(*) from episodic_memories where agent_name = 'model_selector'
--      and description = 'Model Selection: unknown (unknown). Score: 0.00. Reason: N/A';
-- and, if non-zero, re-run this file's statement by hand (it is idempotent).
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

WITH parsed AS (
    SELECT em.memory_id,
           regexp_match(
               em.raw_content -> 'selection_rationale' ->> 'selection_rationale',
               '^Selected ([^ ()]+) \(score: ([0-9]+\.[0-9]{3})\)'
           ) AS m,
           NULLIF(btrim(em.raw_content -> 'selection_rationale' ->> 'primary_reason'), '')
               AS primary_reason
      FROM episodic_memories em
     WHERE em.agent_name = 'model_selector'
       AND em.event_type = 'model_selection_completed'
       AND em.description = 'Model Selection: unknown (unknown). Score: 0.00. Reason: N/A'
       AND em.raw_content ->> 'algorithm_name' IS NULL
       AND em.raw_content ->> 'selection_score' IS NULL
       AND jsonb_typeof(em.raw_content -> 'selection_rationale') = 'object'
)
UPDATE episodic_memories em
   SET raw_content = em.raw_content || jsonb_build_object(
           'algorithm_name', p.m[1],
           'selection_score', p.m[2]::double precision,
           'primary_reason', p.primary_reason,
           'selection_backfill', jsonb_build_object(
               'migration', '164_backfill_model_selector_episodic_selection',
               'issue', 2325,
               'source', 'raw_content.selection_rationale.selection_rationale',
               'selection_score_decimals', 3
           )
       ),
       description = format(
           'Model Selection: %s (family not recorded). Score: %s. Reason: %s',
           p.m[1], p.m[2], COALESCE(p.primary_reason, 'not recorded')
       )
  FROM parsed p
 WHERE em.memory_id = p.memory_id
   AND p.m IS NOT NULL
   -- The old-state predicates again on the row being updated, so a row changed after the
   -- CTE's snapshot is left alone.
   AND em.agent_name = 'model_selector'
   AND em.event_type = 'model_selection_completed'
   AND em.description = 'Model Selection: unknown (unknown). Score: 0.00. Reason: N/A'
   AND em.raw_content ->> 'algorithm_name' IS NULL
   AND em.raw_content ->> 'selection_score' IS NULL;
