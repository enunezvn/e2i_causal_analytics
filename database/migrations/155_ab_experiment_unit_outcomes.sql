-- Migration 155: ab_experiment_unit_outcomes — the per-experiment, per-unit,
-- time-indexed OUTCOME feed of the twin fidelity loop (option d1, owner decision
-- 2026-09-23, Part of #2207).
--
-- WHY: the fidelity loop compares the twin's PREDICTED effect with the MEASURED
-- effect of a real/synthetic A/B experiment. The measured side is
-- ExperimentOutcomeRepository.load_arrays (src/repositories/experiment_outcome.py),
-- which until now could only join an experiment's assignments to the per-HCP
-- business_metrics rollup. Measured 2026-09-23 on this deployment: (a) none of the
-- 360 synthetic experiments' prediction_target values (pnh_persistence /
-- csu_treatment_initiation / kisqali_dx_adoption) maps to a business_metrics
-- column, so compute_experiment_results returned `skipped` for every one; (b) even
-- with a mapped column the joined ATE is a STRUCTURAL NULL (adopted -0.032,
-- conversion_rate -0.007 vs the stored +0.097): the synthetic generator draws each
-- unit's outcome in memory, uses it only for the aggregate ab_experiment_results
-- row, and discards it — the per-HCP outcome tables come from independent DGPs.
-- Each HCP sits in ~12 experiments per brand, so no per-(hcp, brand) column can
-- carry per-experiment outcomes: the outcome must be keyed (experiment, unit) and
-- time-indexed. This table is that feed.
--
-- CONTRACT: one observed outcome per (experiment_id, unit_id, metric_name);
-- metric_name == ml_experiments.prediction_target == ab_experiment_results
-- .primary_metric; observed_at is when the outcome was observed (never before the
-- unit's assigned_at). load_arrays reads THIS table first and falls back to the
-- business_metrics join only when an experiment has no rows here.
-- WRITERS: synthetic experiments — the generator
-- (src/ml/synthetic/generators/experiment_generator.py) writes each unit's drawn
-- outcome from the SAME draw its ab_experiment_results row is computed from, so
-- the measured ATE reproduces the planted per-channel truth. Real experiments —
-- a real outcome feed would write here; NONE EXISTS TODAY (real experiments keep
-- the business_metrics fallback until one does).
-- Every synthetic row is is_synthetic = true (#894 provenance family; the table
-- joins PROVENANCE_TAGGED_TABLES in src/repositories/provenance.py). Purged and
-- reloaded by scripts/load_synthetic_data.py --refresh-ab (before enrollments /
-- results / assignments — it FK-references assignments).
-- Additive + idempotent (IF NOT EXISTS). Never touches existing rows.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

CREATE TABLE IF NOT EXISTS public.ab_experiment_unit_outcomes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- the unit's assignment (variant lives there); cascades with the assignment
    assignment_id   UUID NOT NULL REFERENCES public.ab_experiment_assignments(id) ON DELETE CASCADE,
    -- denormalised for the (experiment, metric) read; RESTRICT mirrors the
    -- assignments -> ml_experiments FK (an experiment with outcomes is not droppable)
    experiment_id   UUID NOT NULL REFERENCES public.ml_experiments(id) ON DELETE RESTRICT,
    unit_id         VARCHAR(255) NOT NULL,
    metric_name     VARCHAR(255) NOT NULL,
    outcome_value   DOUBLE PRECISION NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    is_synthetic    BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT ab_unit_outcomes_one_per_experiment_unit_metric
        UNIQUE (experiment_id, unit_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_ab_unit_outcomes_experiment_metric
    ON public.ab_experiment_unit_outcomes (experiment_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_ab_unit_outcomes_assignment
    ON public.ab_experiment_unit_outcomes (assignment_id);

COMMENT ON TABLE public.ab_experiment_unit_outcomes IS
    'Per-experiment, per-unit, time-indexed outcome feed: ONE observed outcome per '
    '(experiment_id, unit_id, metric_name). The MEASURED side of the twin fidelity loop '
    '(option d1, 2026-09-23, Part of #2207): ExperimentOutcomeRepository.load_arrays reads '
    'this table FIRST and falls back to the business_metrics per-HCP join only when an '
    'experiment has no rows here. Synthetic experiments are fed by the generator from the '
    'SAME per-unit draw as ab_experiment_results (is_synthetic = true); real experiments '
    'need a real outcome feed writing here -- none exists today.';
COMMENT ON COLUMN public.ab_experiment_unit_outcomes.metric_name IS
    'The outcome measured: equals ml_experiments.prediction_target and '
    'ab_experiment_results.primary_metric for the experiment.';
COMMENT ON COLUMN public.ab_experiment_unit_outcomes.outcome_value IS
    'The unit''s observed outcome for metric_name (a 0/1 for binary endpoints).';
COMMENT ON COLUMN public.ab_experiment_unit_outcomes.observed_at IS
    'When the outcome was observed; never before the unit''s '
    'ab_experiment_assignments.assigned_at. load_arrays(window_days=...) windows on it.';
COMMENT ON COLUMN public.ab_experiment_unit_outcomes.is_synthetic IS
    'Provenance tag (#894 family): true for generator-written rows, default false. '
    'Real-mode readers default-exclude synthetic rows via apply_provenance_filter.';
