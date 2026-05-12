-- ----------------------------------------------------------------------------
-- Migration 035: extend journey_stage_type enum (issue #155 §2)
--
-- PR #152 (annotated gap-analysis workbook column G row 2) surfaced that the
-- legacy 5-value journey_stage_type enum (diagnosis / initial_treatment /
-- treatment_optimization / maintenance / treatment_switch) cannot represent
-- the 7-stage engagement funnel data scientists need for downstream KPIs:
--
--   aware          dx-only, no specialist visit yet
--   considering    dx + specialist visit (HCP engagement signal pre-Rx)
--   prescribed     Rx written but not yet dispensed
--   first_fill     Rx dispensed once (treatment initiation event)
--   adherent       MPR >= 0.8 in observation window
--   discontinued   gap > threshold between fills without re-fill
--   maintained     adherent for >= 6 months (stable long-term)
--
-- WARNING: PostgreSQL `ALTER TYPE ... ADD VALUE` is FORWARD-ONLY. Once an
-- enum value lands in any production database it cannot be removed without
-- a full type-rebuild migration. The 7 names above were finalised in PR for
-- issue #155 — DO NOT add additional values without supervisor review.
--
-- Existing 5 values are preserved. The 7 new values join the union so
-- downstream code can gradually migrate; converters / synthetic generators
-- now emit the granular values per PR #152 row 2 derivation rules.
-- ----------------------------------------------------------------------------

-- Add each new value with IF NOT EXISTS so this migration is idempotent on
-- dev/staging databases that may have partially applied a prior attempt.

ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'aware';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'considering';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'prescribed';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'first_fill';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'adherent';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'discontinued';
ALTER TYPE journey_stage_type ADD VALUE IF NOT EXISTS 'maintained';

-- Post-condition (manual verification):
--   SELECT enumlabel FROM pg_enum
--     WHERE enumtypid = 'journey_stage_type'::regtype::oid
--     ORDER BY enumsortorder;
-- Expected: 12 values total (5 legacy + 7 new).
