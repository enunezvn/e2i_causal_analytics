-- Rollback for migration 148 (NOT auto-applied: the runner skips rollback_* files).
-- Drops the Lane A causal cohort table. The rows are reproducible from the
-- persistence_causal export (scripts/convert_optum_mart.py + load_optum_causal_cohort.py).
DROP TABLE IF EXISTS public.optum_biologic_persistence_causal;
