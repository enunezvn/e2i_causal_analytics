-- ============================================================================
-- E2I Sentinel Cooldown Column — issue #375 (Phase 3 hardening)
-- Migration: 023_sentinel_cooldown.sql
--
-- Adds:
--   - sentinels.cooldown_minutes  : integer minutes between consecutive fires
--                                   (NULL = no cooldown, prior behaviour)
--
-- The Python dispatcher
--   src.memory.sentinels.registry::dispatch_sentinels
-- enforces the gate:
--     SKIP if (now - last_fired_at) < cooldown_minutes
-- A NULL or 0 cooldown means "always evaluate" (back-compat with PR #250).
--
-- Safety net (defense-in-depth):
-- A CHECK constraint prevents bad rows from being persisted:
--   cooldown_minutes IS NULL OR cooldown_minutes >= 0
-- A second CHECK rejects cooldowns greater than ~1 year to catch obvious
-- operator typos (e.g. forgetting unit conversion):
--   cooldown_minutes IS NULL OR cooldown_minutes <= 525600  -- 365 days * 24h * 60min
-- ============================================================================

BEGIN;

-- Idempotent column add — older deployments without the column get it;
-- newer ones are no-op.
ALTER TABLE sentinels
    ADD COLUMN IF NOT EXISTS cooldown_minutes INTEGER;

COMMENT ON COLUMN sentinels.cooldown_minutes IS
'Minimum minutes between consecutive fires. NULL or 0 = no cooldown (always evaluate). Enforced by src.memory.sentinels.registry::dispatch_sentinels and bounded by the CHECK constraint below.';

-- The DO-block wrapper is the idiomatic Postgres pattern for "add constraint
-- only if absent" (no IF NOT EXISTS on ADD CONSTRAINT pre-PG18). Narrow
-- exception class — invalid_object_definition + duplicate_object only.
-- This is a load-bearing pattern: NEVER use `WHEN OTHERS` because it would
-- mask real errors (foreign-key violations, etc.).
DO $$
BEGIN
    ALTER TABLE sentinels
        ADD CONSTRAINT chk_sentinels_cooldown_nonneg
        CHECK (cooldown_minutes IS NULL OR cooldown_minutes >= 0);
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN invalid_object_definition THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE sentinels
        ADD CONSTRAINT chk_sentinels_cooldown_bounded
        CHECK (cooldown_minutes IS NULL OR cooldown_minutes <= 525600);
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN invalid_object_definition THEN NULL;
END $$;

-- Index on (enabled, last_fired_at) supports the dispatcher's
-- "fetch enabled sentinels ordered by last_fired_at" pattern when the
-- gate is applied in SQL (future optimisation; today the gate is in
-- Python). Indexing both columns now keeps the query plan stable as
-- the dispatcher's query evolves.
CREATE INDEX IF NOT EXISTS idx_sentinels_enabled_last_fired
    ON sentinels(enabled, last_fired_at)
    WHERE enabled = TRUE;

COMMIT;
