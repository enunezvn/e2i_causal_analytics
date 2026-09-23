-- Rollback for migration 155 (NOT auto-applied: the runner skips rollback_* files).
-- Drops the per-experiment unit outcome feed. The synthetic rows are reproducible
-- from scripts/load_synthetic_data.py --refresh-ab (deterministic ids); no real
-- writer exists yet, so no real data is lost.
DROP TABLE IF EXISTS public.ab_experiment_unit_outcomes;
