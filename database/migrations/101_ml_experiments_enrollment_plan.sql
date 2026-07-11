-- Migration 101: ml_experiments enrollment plan (target + planned duration)
-- ==========================================================================
-- The experiment_monitor health check was judging enrollment progress against
-- config.get("target_sample_size", 1000) — but ml_experiments has never had a
-- config column, so every experiment was measured against a fabricated
-- 1000-in-30-days plan (required pace ~16.7/day, above any realistic rate).
-- Live incident 2026-07-11: all 25 monitored experiments flagged "warning"
-- (Healthy = 0) purely from that fabricated denominator, while their actual
-- enrollment rates (5.7–15/day) were healthy.
--
-- These columns give the monitor's target concept (information fraction,
-- behind-plan health, interim milestones) real per-experiment data:
--   target_enrollment     — planned number of enrolled units
--   planned_duration_days — planned enrollment window in days
--
-- Nullable by design: real (is_synthetic=false) experiments are ML-scoping
-- rows with no A/B enrollment plan; consumers must treat NULL as "no plan
-- recorded" (skip plan-relative checks, report fraction as unknown), never
-- fabricate a default target.
--
-- NOTE: no BEGIN/COMMIT — the migration runner wraps each file in its own
-- transaction (see 093 incident note).

ALTER TABLE ml_experiments
    ADD COLUMN IF NOT EXISTS target_enrollment INTEGER,
    ADD COLUMN IF NOT EXISTS planned_duration_days INTEGER;

COMMENT ON COLUMN ml_experiments.target_enrollment IS
    'Planned A/B enrollment (units). NULL = no enrollment plan recorded — '
    'consumers must not substitute a default target.';

COMMENT ON COLUMN ml_experiments.planned_duration_days IS
    'Planned A/B enrollment window (days). NULL = no enrollment plan recorded.';
