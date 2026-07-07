-- Migration 097: expert_reviews DAG structure + agent assessment
-- ==============================================================
-- The expert-review queue (R6-F2) stored only the one-way SHA256
-- dag_version_hash, so the /expert-reviews UI could not show the reviewer the
-- DAG under review, and there was nowhere to cache an agent-generated
-- assessment of the reviewer checklist questions.
--
--   dag_structure_json    sanitized CausalGraph snapshot captured at review
--                         creation (nodes, edges, treatment/outcome nodes,
--                         adjustment sets, discovery provenance). Nullable:
--                         rows created before this migration cannot be
--                         backfilled (the hash is one-way) and render an
--                         honest "structure not captured" fallback.
--   agent_assessment_json cached agent (LLM/deterministic) assessment of the
--                         reviewer checklist questions, generated on demand
--                         via POST /expert-reviews/{id}/assessment. Advisory
--                         only — kept SEPARATE from checklist_json, which
--                         remains the human reviewer's own record.
--
-- NOTE: no BEGIN/COMMIT here — the migration runner wraps the file.

ALTER TABLE expert_reviews
    ADD COLUMN IF NOT EXISTS dag_structure_json JSONB,
    ADD COLUMN IF NOT EXISTS agent_assessment_json JSONB;

COMMENT ON COLUMN expert_reviews.dag_structure_json IS
    'Sanitized causal-graph snapshot (nodes/edges/treatment/outcome/adjustment sets) captured when the review was created; NULL for rows created before migration 097 (hash is one-way, not backfillable)';
COMMENT ON COLUMN expert_reviews.agent_assessment_json IS
    'Cached advisory agent assessment of the reviewer checklist questions (verdict + evidence-grounded rationale per item); generated on demand, never a substitute for the human checklist_json';
