-- ROLLBACK for migration 150 (per-model cohort contract, #2207 follow-up). NOT a forward
-- migration: scripts/run_migrations.sh skips rollback_*.sql in apply_dir(), which is what
-- makes it safe to ship this file beside 150.
--
-- 150 is a pure EXPAND (three nullable TEXT columns, no backfill), so a failed
-- container replacement needs no schema recovery: pre-lane code never reads the columns
-- and MLModelRegistry.to_dict() only emits them when set. This file exists to return the
-- schema to its exact pre-150 shape (re-rehearsal, or abandoning the lane). It removes
-- ONLY what 150 added (the columns are NULL for every pre-existing row).
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_150_ml_registry_cohort_contract.sql

ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS cohort_feature_manifest_source;
ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS cohort_target_outcome;
ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS cohort_data_source;
