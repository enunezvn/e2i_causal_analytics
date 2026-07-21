-- ============================================================================
-- Migration 115: Backlog #45 PR-A — synthetic claims ARRIVAL plane columns on
-- treatment_events. Additive + idempotent (DDL ONLY; the completion-factor
-- nowcast registry SQL is migration 116, owned by PR-B).
--
-- Columns stay NULL until the next synthetic reseed populates them via the
-- stamp_claim_arrival post-generation pass (seed+10; parameters authored in
-- config/domain_vocabulary.yaml data_constraints.adjudication_lag_dgp; the
-- loader whitelist carries both). Migration-113-safe by construction: NO base
-- KPI filters on these columns — the base TRx/NRx/NBRx/recall SQL keeps
-- reading event_date (the omniscient/mature truth), and the provisional view
-- filters this NEW column instead of masking event_date (the falsified
-- as-of-cutoff of migration 113 item 3).
-- ----------------------------------------------------------------------------

ALTER TABLE treatment_events ADD COLUMN IF NOT EXISTS claim_available_date  DATE;
ALTER TABLE treatment_events ADD COLUMN IF NOT EXISTS adjudication_lag_days INTEGER;

COMMENT ON COLUMN treatment_events.claim_available_date IS
    'Synthetic claims arrival plane (backlog #45): date this claims-derived event became visible in the data = event_date + adjudication_lag_days. NON-CLAIMS/CRM events and pre-#45 rows are NULL. NO base KPI filters on this column — it feeds only the completion-factor nowcast overlay (migration 116 / PR-B). Additive; migration-113-safe. Added by migration 115.';
COMMENT ON COLUMN treatment_events.adjudication_lag_days IS
    'Synthetic claims arrival plane (backlog #45): the drawn adjudication lag in days (gamma per source class, authored in domain_vocabulary.yaml data_constraints.adjudication_lag_dgp; transparency + estimator input). 0 for configured zero-lag/CRM event types; NULL when event_date is absent and on pre-#45 rows. Added by migration 115.';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
