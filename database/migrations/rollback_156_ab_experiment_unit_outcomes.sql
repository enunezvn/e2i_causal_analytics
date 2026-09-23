-- Rollback for migration 156 (NOT auto-applied: the runner skips rollback_* files).
-- Drops the per-experiment unit outcome feed. The synthetic rows are reproducible
-- from scripts/load_synthetic_data.py --refresh-ab (deterministic ids); no real
-- writer exists yet, so no real data is lost.
DROP TABLE IF EXISTS public.ab_experiment_unit_outcomes;

-- Retire the ledger row in the SAME transaction as the drop (the 143/144 rollback
-- pattern; codex r2 MED): otherwise scripts/run_migrations.sh still believes 156 is
-- applied and skips recreating the table on the next deploy while the loader,
-- --refresh-ab and load_arrays already depend on it. Apply this file with
-- `psql --single-transaction` so the drop and the ledger retirement commit together.
DELETE FROM public.schema_migrations WHERE filename = '155_ab_experiment_unit_outcomes.sql';
