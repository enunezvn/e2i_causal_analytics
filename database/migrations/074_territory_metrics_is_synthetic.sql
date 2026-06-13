-- ============================================================================
-- Migration 074: is_synthetic provenance on territory_metrics (issue #895)
-- ============================================================================
-- territory_metrics (created by migration 031, extended by 033) was never
-- covered by the is_synthetic provenance family (063 tagged the analytics
-- read-path tables, 069 the Shard-09 substrate tables). The daily territory
-- rollup ETL (src/etl/territory_metrics_etl.py) aggregates per-HCP
-- business_metrics rollup rows plus raw triggers -- both provenance-tagged --
-- into a table with NO provenance column at all: second-order laundering.
-- This migration adds the column so the rollup SQL can inherit
-- is_synthetic = bool(any synthetic input) per derived row.
--
-- Additive + idempotent: existing rows default FALSE ("real") -- correct,
-- since the table is empty on the faithful DB (031's seed rows were removed
-- by the 2026-06-11 synthetic-gold cleanup) and nothing synthetic has been
-- written to it yet.
-- ----------------------------------------------------------------------------

ALTER TABLE territory_metrics ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN territory_metrics.is_synthetic IS
    'TRUE = derived from at least one synthetic-tagged input row (per-HCP '
    'business_metrics rollups, triggers, hcp_profiles). Provenance is '
    'inherited by the territory rollup ETL (bool_or over the aggregated '
    'cell, issue #895); excluded by default from real analyses. Added by '
    'migration 074.';

-- Partial index (only the synthetic minority is indexed -> tiny), matching
-- the 063/069 family pattern.
CREATE INDEX IF NOT EXISTS idx_territory_metrics_is_synthetic ON territory_metrics (is_synthetic) WHERE is_synthetic;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
