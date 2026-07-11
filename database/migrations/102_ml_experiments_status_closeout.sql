-- Migration 102: close out perpetually-"running" ml_experiments registry rows
-- ===========================================================================
-- ml_experiments doubles as (a) the A/B experiment registry and (b) an
-- MLflow-style lineage registry for ML pipeline scoping/eval/deploy events.
-- The status column was designed with a full lifecycle
-- (draft/running/completed/stopped/archived, DB default 'running') but no
-- writer ever set it and no code path ever transitions it: the repository's
-- MLExperiment.to_dict() omits status entirely, so every scope_definer /
-- gold_standard_eval / prediction_synthesizer_deploy insert silently
-- inherited 'running' and stayed there forever (oldest rows: 2026-04-25).
--
-- Live incident 2026-07-11: 692 scope_definer lineage rows (18 distinct
-- scope names — each Tier-0 pipeline execution blind-inserted a duplicate),
-- 15 gold_standard_eval rows and 1 prediction_synthesizer_deploy row — all
-- with zero enrolled participants and zero attached training runs — were
-- counted as "running experiments" by the experiment monitor (955 vs the
-- true 360-experiment A/B portfolio), the /experiments running-count
-- endpoint, and the ab_testing interim-analysis sweeps.
--
-- These rows record work that finished at insert time (a scope definition,
-- a completed eval, a deployment) → 'completed' is their honest state.
-- The 360 synthetic_loader A/B rows are genuinely running and are not
-- touched. The scope_definer insert path now writes status explicitly and
-- reuses the existing row per scope name (see src/agents/ml_foundation/
-- scope_definer/agent.py), so this backfill is not re-creatable.
--
-- NOTE: no BEGIN/COMMIT — the migration runner wraps each file in its own
-- transaction (see 093 incident note).

UPDATE ml_experiments
SET status = 'completed',
    updated_at = now()
WHERE status = 'running'
  AND created_by IN (
      'scope_definer',
      'gold_standard_eval',
      'prediction_synthesizer_deploy'
  );

COMMENT ON COLUMN ml_experiments.status IS
    'Lifecycle: draft/running/completed/stopped/archived. Only rows that '
    'represent an actively enrolling A/B experiment may be ''running''; '
    'pipeline-lineage rows (scope_definer, gold_standard_eval, '
    'prediction_synthesizer_deploy) are written as ''completed''.';
