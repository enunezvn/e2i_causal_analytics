-- ============================================================================
-- Migration 069: is_synthetic provenance on the Shard-09 substrate tables
-- (synthetic-causal-validation).
-- ============================================================================
-- Shard 01's migration 063 (M1) tagged the 12 tables the analytics read paths
-- and the kpi_query RPC touched at that point. Shard 09 adds breadth substrate
-- to the experiment / MLOps / observability / feedback tables, which 063 did NOT
-- cover. Without a provenance flag on these tables the loader cannot tag the
-- rows it inserts and Shard 07's default-exclude reads cannot distinguish the
-- synthetic breadth rows from real ones.
--
-- These 8 tables lack is_synthetic on the faithful DB (verified via
-- information_schema.columns). The other Shard-09 tables already carry it:
--   - causal_paths, ab_experiment_assignments, user_sessions, hcp_intent_surveys
--     (migration 063), and data_source_tracking / etl_pipeline_metrics /
--     ml_annotations (earlier KPI-577 substrate migrations).
--
-- Additive + idempotent: existing rows default FALSE ("real") -- correct, since
-- nothing synthetic has been loaded into these tables yet.
-- ----------------------------------------------------------------------------

ALTER TABLE ml_experiments         ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_model_registry      ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_training_runs       ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_deployments         ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ab_experiment_enrollments ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ab_experiment_results  ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_observability_spans ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE learning_signals       ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN ml_experiments.is_synthetic IS
    'TRUE = synthetic causal-validation substrate (Shard 09), excluded by '
    'default from real analyses. Added by migration 069.';

-- Partial indexes (only the synthetic minority is indexed -> tiny) on the
-- high-fanout breadth tables.
CREATE INDEX IF NOT EXISTS idx_ml_observability_spans_is_synthetic ON ml_observability_spans (is_synthetic) WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_learning_signals_is_synthetic       ON learning_signals (is_synthetic)       WHERE is_synthetic;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
