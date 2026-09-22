-- Rollback for migration 149 (NOT auto-applied: the runner skips rollback_* files).
-- Drops the Lane C causal cohort table. The synthetic backing rows are
-- reproducible from scripts/build_csu_escalation_synthetic_cohort.py (fixed seed);
-- real rows, once loaded, from the post-launch csu_escalation_causal export.
DROP TABLE IF EXISTS public.csu_escalation_causal;
