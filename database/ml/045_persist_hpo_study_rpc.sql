-- ============================================================================
-- Migration ml/045: persist_hpo_study(jsonb, jsonb) — atomic study + trial set (#2207)
-- ============================================================================
-- Why: OptunaOptimizer.save_to_database wrote ml_hpo_studies and ml_hpo_trials
-- through separate PostgREST statements (study upsert, one upsert per trial, a
-- delete of stale trailing trials). An in-memory Optuna rerun reuses the study
-- name, so between those statements a reader could see the new parent with the
-- old run's trials, or a partially written set (codex r5). One SQL function is
-- one transaction: the study is upserted on its UNIQUE study_name and its trial
-- set is REPLACED, or nothing changes.
--
-- Contract: p_study carries the ml_hpo_studies columns as JSON (experiment_id is
-- an ml_experiments(id) uuid or null); p_trials is a JSON array of ml_hpo_trials
-- rows (trial_number, state, params, value, intermediate_values, datetime_start,
-- datetime_complete, duration_seconds, user_attrs, system_attrs). Returns the
-- study id. Idempotent (CREATE OR REPLACE). SECURITY INVOKER, pinned search_path,
-- executable by service_role (the backend client), like ml/040.
--
-- Objective columns (codex r7): 016 declared best_value / trial value as
-- numeric(10,6), which overflows above 9999.999999 — a legitimate large
-- regression objective (an rmse in the millions) would abort the whole atomic
-- call. Widened to double precision (what the Python side holds anyway); the
-- casts below follow. Both tables were at 0 rows when this shipped.
-- ============================================================================

ALTER TABLE ml_hpo_studies ALTER COLUMN best_value TYPE DOUBLE PRECISION;
ALTER TABLE ml_hpo_trials ALTER COLUMN value TYPE DOUBLE PRECISION;

CREATE OR REPLACE FUNCTION public.persist_hpo_study(
    p_study JSONB,
    p_trials JSONB DEFAULT '[]'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
    v_study_id UUID;
BEGIN
    IF p_study IS NULL OR COALESCE(p_study->>'study_name', '') = '' THEN
        RAISE EXCEPTION 'persist_hpo_study: p_study.study_name is required';
    END IF;
    IF jsonb_typeof(COALESCE(p_trials, '[]'::jsonb)) <> 'array' THEN
        RAISE EXCEPTION 'persist_hpo_study: p_trials must be a JSON array';
    END IF;

    INSERT INTO ml_hpo_studies (
        study_name, experiment_id, algorithm_name, problem_type, direction,
        sampler_name, pruner_name, metric, search_space, fixed_params,
        n_trials, n_completed, n_pruned, n_failed,
        best_trial_number, best_value, best_params, duration_seconds,
        status, completed_at, updated_at
    )
    VALUES (
        p_study->>'study_name',
        NULLIF(p_study->>'experiment_id', '')::uuid,
        COALESCE(p_study->>'algorithm_name', 'unknown'),
        COALESCE(p_study->>'problem_type', 'binary_classification'),
        COALESCE(p_study->>'direction', 'maximize'),
        COALESCE(p_study->>'sampler_name', 'TPESampler'),
        COALESCE(p_study->>'pruner_name', 'MedianPruner'),
        COALESCE(p_study->>'metric', 'roc_auc'),
        COALESCE(p_study->'search_space', '{}'::jsonb),
        COALESCE(p_study->'fixed_params', '{}'::jsonb),
        COALESCE((p_study->>'n_trials')::int, 0),
        COALESCE((p_study->>'n_completed')::int, 0),
        COALESCE((p_study->>'n_pruned')::int, 0),
        COALESCE((p_study->>'n_failed')::int, 0),
        NULLIF(p_study->>'best_trial_number', '')::int,
        NULLIF(p_study->>'best_value', '')::double precision,
        COALESCE(p_study->'best_params', '{}'::jsonb),
        NULLIF(p_study->>'duration_seconds', '')::numeric,
        COALESCE(p_study->>'status', 'completed'),
        NULLIF(p_study->>'completed_at', '')::timestamptz,
        now()
    )
    ON CONFLICT (study_name) DO UPDATE SET
        experiment_id     = EXCLUDED.experiment_id,
        algorithm_name    = EXCLUDED.algorithm_name,
        problem_type      = EXCLUDED.problem_type,
        direction         = EXCLUDED.direction,
        sampler_name      = EXCLUDED.sampler_name,
        pruner_name       = EXCLUDED.pruner_name,
        metric            = EXCLUDED.metric,
        search_space      = EXCLUDED.search_space,
        fixed_params      = EXCLUDED.fixed_params,
        n_trials          = EXCLUDED.n_trials,
        n_completed       = EXCLUDED.n_completed,
        n_pruned          = EXCLUDED.n_pruned,
        n_failed          = EXCLUDED.n_failed,
        best_trial_number = EXCLUDED.best_trial_number,
        best_value        = EXCLUDED.best_value,
        best_params       = EXCLUDED.best_params,
        duration_seconds  = EXCLUDED.duration_seconds,
        status            = EXCLUDED.status,
        completed_at      = EXCLUDED.completed_at,
        updated_at        = now()
    RETURNING id INTO v_study_id;

    -- Replace the trial set: the child rows are exactly this run's trials or,
    -- on any error below, unchanged (the whole function is one transaction).
    DELETE FROM ml_hpo_trials WHERE study_id = v_study_id;

    INSERT INTO ml_hpo_trials (
        study_id, trial_number, state, params, value, intermediate_values,
        datetime_start, datetime_complete, duration_seconds, user_attrs, system_attrs
    )
    SELECT
        v_study_id,
        (t->>'trial_number')::int,
        COALESCE(t->>'state', 'COMPLETE'),
        COALESCE(t->'params', '{}'::jsonb),
        NULLIF(t->>'value', '')::double precision,
        COALESCE(t->'intermediate_values', '{}'::jsonb),
        NULLIF(t->>'datetime_start', '')::timestamptz,
        NULLIF(t->>'datetime_complete', '')::timestamptz,
        NULLIF(t->>'duration_seconds', '')::numeric,
        COALESCE(t->'user_attrs', '{}'::jsonb),
        COALESCE(t->'system_attrs', '{}'::jsonb)
    FROM jsonb_array_elements(COALESCE(p_trials, '[]'::jsonb)) AS t;

    RETURN v_study_id;
END;
$$;

COMMENT ON FUNCTION public.persist_hpo_study(jsonb, jsonb) IS
    'Atomically upserts one ml_hpo_studies row (on study_name) and replaces its ml_hpo_trials set (#2207). Called by OptunaOptimizer.save_to_database.';

GRANT EXECUTE ON FUNCTION public.persist_hpo_study(jsonb, jsonb) TO service_role;
