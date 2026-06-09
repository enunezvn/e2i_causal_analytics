-- ============================================================================
-- Migration 063 (M1): is_synthetic provenance column (synthetic-causal-validation)
-- ============================================================================
-- Adds a dedicated synthetic/real provenance flag to every taggable table the
-- analytics read paths and the kpi_query RPC touch. Synthetic rows set TRUE;
-- reads exclude is_synthetic=true BY DEFAULT (enforced server-side in
-- migration 066/M4 for the RPC path; in repositories/connectors by Shard 07).
-- data_source (patient_journeys/treatment_events) is a free-text SOURCE label,
-- NOT a synthetic flag, so a uniform boolean is added there too.
-- Additive + idempotent: existing rows default FALSE ("real") -- correct, since
-- nothing synthetic has been loaded yet.
-- ----------------------------------------------------------------------------

ALTER TABLE triggers                   ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE business_metrics           ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ml_predictions             ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE agent_activities           ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE causal_paths               ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE patient_journeys           ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE treatment_events           ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE hcp_profiles               ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE user_sessions              ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE hcp_intent_surveys         ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE episodic_memories          ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE ab_experiment_assignments  ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN treatment_events.is_synthetic IS
    'TRUE = row produced by the synthetic-causal-validation generator stack '
    '(known ground-truth ATE/CATE). Excluded by default from real analyses; '
    'opt-in via the kpi_query *_include_synthetic statements or the Shard 07 '
    'real/validation mode flag. Added by migration 063 (M1).';

-- Partial indexes (only the synthetic minority is indexed -> tiny) on the
-- high-fanout tables.
CREATE INDEX IF NOT EXISTS idx_treatment_events_is_synthetic ON treatment_events (is_synthetic) WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_triggers_is_synthetic         ON triggers (is_synthetic)         WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_ml_predictions_is_synthetic   ON ml_predictions (is_synthetic)   WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_business_metrics_is_synthetic ON business_metrics (is_synthetic) WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_patient_journeys_is_synthetic ON patient_journeys (is_synthetic) WHERE is_synthetic;
CREATE INDEX IF NOT EXISTS idx_episodic_memories_is_synthetic ON episodic_memories (is_synthetic) WHERE is_synthetic;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
