-- ============================================================================
-- Migration 032: Routing-classifier metrics snapshots (#1341 Phase 2)
--
-- Phase 1 (PR #1342) populates classification_logs.was_correct via the nightly
-- labeler; v_classification_accuracy then aggregates per-day per-pattern
-- accuracy live from those rows. This table adds the ONE thing the view cannot:
-- a per-run TIME SERIES of the whole-run safety telemetry (engagement rate,
-- abstention correctness, LLM-layer share, label-source mix) so trends across
-- nights are queryable — the standing safety signal any future active-mode
-- promotion is judged against. One small row per labeler run.
--
-- Written fail-open by src/tasks/routing_label_tasks.py after each cycle; the
-- labeler degrades to log-only if this table is absent, so applying this
-- migration is not a hard dependency of the Phase-1 labeler.
--
-- Created: 2026-07-31
-- ============================================================================

CREATE TABLE IF NOT EXISTS routing_classifier_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Run identity
    run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    task_id VARCHAR(100),
    window_days INTEGER NOT NULL,

    -- Volume
    total INTEGER NOT NULL DEFAULT 0,
    labeled INTEGER NOT NULL DEFAULT 0,

    -- Whole-run safety telemetry
    overall_accuracy_pct NUMERIC,          -- pipeline-vs-judge agreement over labeled rows
    engagement_rate NUMERIC,               -- share committing to a route at active_floor
    active_floor NUMERIC NOT NULL,         -- MIN_ACTIVE_CONFIDENCE used for engagement
    llm_layer_share NUMERIC,               -- share that engaged the LLM layer

    -- Abstention correctness (over-abstention is the #1337 finding)
    abstention_total INTEGER NOT NULL DEFAULT 0,
    abstention_correct INTEGER NOT NULL DEFAULT 0,
    abstention_incorrect INTEGER NOT NULL DEFAULT 0,

    -- Breakdowns
    per_pattern JSONB NOT NULL DEFAULT '{}',    -- {pattern: {total, correct, incorrect, awaiting, accuracy_pct}}
    label_sources JSONB NOT NULL DEFAULT '{}',  -- {explicit_feedback, implicit_outcome, llm_judge, llm_judge_abstain}

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_routing_classifier_metrics_run_at
    ON routing_classifier_metrics (run_at DESC);

COMMENT ON TABLE routing_classifier_metrics IS
    'Per-run routing-classifier safety telemetry time series (#1341 Phase 2); one row per nightly labeler cycle.';
