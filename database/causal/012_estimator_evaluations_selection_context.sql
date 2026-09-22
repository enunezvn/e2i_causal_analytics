-- ============================================================================
-- Migration causal/012: estimator_evaluations — selection context (#2207)
-- ============================================================================
-- Why: estimator_evaluations (causal/011, 220df4cec) has had a writer since
-- 2025-12 — EnergyScoreMLflowTracker._log_to_database — that no live path ever
-- instantiated (0 rows on 2026-09-22). Wiring it onto the live energy-score path
-- (the causal_impact estimation node, a QUERY-time path) exposes a design gap in
-- 011: experiment_id is NOT NULL and references ml_experiments(id), but the tracker
-- never holds an ml_experiments UUID — it holds an MLflow experiment id (an int
-- string) or a random uuid4 fallback. Every insert the writer could ever make
-- would have violated the FK (or failed the uuid cast). The natural key of a
-- query-time evaluation is the query, not an ML experiment.
--
-- What: keep experiment_id (nullable now, FK intact) for the ML-experiment use
-- 011 designed for, and add the query-time context: the query/session that ran
-- the selection, the run id that groups the N rows of ONE selection, the
-- variables, brand/region, data provenance and the MLflow run when one exists.
-- v_selection_comparison partitioned by experiment_id; with NULL experiment_ids
-- every query-time row would collapse into one partition, so it now partitions
-- by selection_run_id (each 011-era row, had any existed, gets its own).
--
-- Additive. No backfill (there are no rows), no data rewrite. Idempotent.
-- ============================================================================

ALTER TABLE estimator_evaluations
    ALTER COLUMN experiment_id DROP NOT NULL;

ALTER TABLE estimator_evaluations
    ADD COLUMN IF NOT EXISTS selection_run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    ADD COLUMN IF NOT EXISTS query_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS mlflow_run_id VARCHAR(100),
    ADD COLUMN IF NOT EXISTS treatment_variable VARCHAR(255),
    ADD COLUMN IF NOT EXISTS outcome_variable VARCHAR(255),
    ADD COLUMN IF NOT EXISTS brand VARCHAR(100),
    ADD COLUMN IF NOT EXISTS region VARCHAR(100),
    ADD COLUMN IF NOT EXISTS data_source VARCHAR(50);

COMMENT ON COLUMN estimator_evaluations.experiment_id IS
    'ml_experiments(id) when the selection ran inside an ML experiment; NULL for a query-time selection (the causal_impact agent), whose key is query_id / selection_run_id (#2207).';
COMMENT ON COLUMN estimator_evaluations.selection_run_id IS
    'Groups the N rows (one per estimator evaluated) of ONE EstimatorSelector.select() call (#2207).';
COMMENT ON COLUMN estimator_evaluations.query_id IS
    'CausalImpactState.query_id of the query that ran the selection (#2207).';
COMMENT ON COLUMN estimator_evaluations.session_id IS
    'CausalImpactState.session_id (working-memory session) when known (#2207).';
COMMENT ON COLUMN estimator_evaluations.mlflow_run_id IS
    'MLflow run id when the selection was logged under an MLflow run (start_selection_run); NULL on the query-time path (#2207).';
COMMENT ON COLUMN estimator_evaluations.data_source IS
    'Provenance of the frame the estimators were fit on, as the estimation node saw it (e.g. real table/cohort name, or ''synthetic'' for the explicit synthetic opt-in) (#2207).';

CREATE INDEX IF NOT EXISTS idx_estimator_evals_query_id
    ON estimator_evaluations(query_id);
CREATE INDEX IF NOT EXISTS idx_estimator_evals_selection_run
    ON estimator_evaluations(selection_run_id);

-- Re-key the legacy-vs-energy comparison on the selection run (011 keyed it on
-- experiment_id, which is NULL for every query-time row).
CREATE OR REPLACE VIEW v_selection_comparison AS
WITH ranked_evals AS (
    SELECT
        selection_run_id,
        estimator_type,
        energy_score,
        success,
        was_selected,
        ROW_NUMBER() OVER (PARTITION BY selection_run_id ORDER BY estimator_priority) AS priority_rank,
        ROW_NUMBER() OVER (PARTITION BY selection_run_id ORDER BY energy_score NULLS LAST) AS energy_rank
    FROM estimator_evaluations
    WHERE success = TRUE
),
comparisons AS (
    SELECT
        selection_run_id,
        -- What legacy (first_success) would have selected
        MAX(CASE WHEN priority_rank = 1 THEN estimator_type END) AS legacy_selection,
        MAX(CASE WHEN priority_rank = 1 THEN energy_score END) AS legacy_energy_score,
        -- What energy score selection chose
        MAX(CASE WHEN energy_rank = 1 THEN estimator_type END) AS energy_selection,
        MAX(CASE WHEN energy_rank = 1 THEN energy_score END) AS best_energy_score,
        -- Actual selection
        MAX(CASE WHEN was_selected THEN estimator_type END) AS actual_selection
    FROM ranked_evals
    GROUP BY selection_run_id
)
SELECT
    COUNT(*) AS total_experiments,
    SUM(CASE WHEN legacy_selection = energy_selection THEN 1 ELSE 0 END) AS same_selection,
    SUM(CASE WHEN legacy_selection != energy_selection THEN 1 ELSE 0 END) AS different_selection,
    ROUND(100.0 * SUM(CASE WHEN legacy_selection != energy_selection THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved,
    ROUND(AVG(legacy_energy_score - best_energy_score)::numeric, 4) AS avg_energy_improvement
FROM comparisons
WHERE legacy_selection IS NOT NULL AND energy_selection IS NOT NULL;

COMMENT ON VIEW v_selection_comparison IS
    'Compares legacy first-success selection vs energy score selection per selection run (re-keyed from experiment_id by causal/012, #2207)';
