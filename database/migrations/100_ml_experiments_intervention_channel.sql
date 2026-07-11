-- Migration 100: ml_experiments.intervention_channel
-- ==================================================
-- The /experiments page review (2026-07-11) found the synthetic A/B portfolio
-- unexplainable: 360 clone rows ("Synthetic causal-validation experiment
-- (Shard 09)") with no machine-readable record of WHICH intervention each
-- experiment tests. The redesigned Shard-09 generator gives every experiment
-- one channel from the user-approved digital-twin intervention taxonomy
-- (src/digital_twin/effect/provider.py INTERVENTION_CATALOG), and both the
-- experiment_monitor payload (per-card badge) and POST /insights/experiments
-- (per-channel effect ranking from ab_experiment_results) group by it.
--
-- Nullable by design: real (is_synthetic=false) experiments predate the
-- taxonomy and legitimately have no channel; consumers must treat NULL as
-- "channel not recorded", never fabricate one.
--
-- NOTE: no BEGIN/COMMIT — the migration runner wraps each file in its own
-- transaction (see 093 incident note).

ALTER TABLE ml_experiments
    ADD COLUMN IF NOT EXISTS intervention_channel VARCHAR(50);

COMMENT ON COLUMN ml_experiments.intervention_channel IS
    'Digital-twin intervention taxonomy value this experiment tests '
    '(INTERVENTION_CATALOG in src/digital_twin/effect/provider.py). '
    'NULL = channel not recorded (e.g. real pre-taxonomy experiments).';

CREATE INDEX IF NOT EXISTS idx_ml_experiments_intervention_channel
    ON ml_experiments (intervention_channel)
    WHERE intervention_channel IS NOT NULL;
