-- ============================================================================
-- E2I Sentinel Cooldown Column — issue #375 (Phase 3 hardening)
-- Migration: 023_sentinel_cooldown.sql
--
-- Adds:
--   - sentinels.cooldown_minutes  : integer minutes between consecutive fires
--                                   (DEFAULT 0 = no cooldown gate; semantically
--                                   identical to NULL but operator-explicit)
--
-- The Python dispatcher
--   src.memory.sentinels.registry::dispatch_sentinels
-- enforces the gate:
--     SKIP if (now - last_fired_at) < cooldown_minutes
-- A NULL or 0 cooldown means "always evaluate" (back-compat with PR #250).
--
-- iter-1 M2 fix (#375 codex iter-0 M2):
-- The column previously had no DEFAULT and no backfill, leaving existing
-- and newly-created rows as NULL. We add DEFAULT 0 and a one-shot backfill
-- so:
--   * New rows insert with cooldown_minutes = 0 unless the caller sets one
--   * Existing rows (created before this migration) move from NULL → 0
-- Why 0 (not 60): PR #250 shipped "no cooldown" semantics. Switching to a
-- 60-minute default would silently alter pre-#375 sentinel behaviour. The
-- dispatcher treats 0 and NULL identically (both = "no gate"), so 0 is the
-- safe-default that ALSO surfaces explicit operator intent. Operators who
-- want a cooldown gate must set it explicitly (YAML or POST /api/sentinels).
--
-- Safety net (defense-in-depth):
-- A CHECK constraint prevents bad rows from being persisted:
--   cooldown_minutes IS NULL OR cooldown_minutes >= 0
-- A second CHECK rejects cooldowns greater than ~1 year to catch obvious
-- operator typos (e.g. forgetting unit conversion):
--   cooldown_minutes IS NULL OR cooldown_minutes <= 525600  -- 365 days * 24h * 60min
-- The IS NULL alternative in the CHECK predicates is retained for forward
-- compatibility (operators who want to revert to NULL semantics via DDL
-- still validate).
-- ============================================================================

BEGIN;

-- Idempotent column add — older deployments without the column get it;
-- newer ones are no-op. DEFAULT 0 on the column makes new inserts explicit
-- about "no cooldown gate".
ALTER TABLE sentinels
    ADD COLUMN IF NOT EXISTS cooldown_minutes INTEGER DEFAULT 0;

-- Backfill (#375 iter-1 M2): for deployments that ran an earlier
-- 023 without the DEFAULT, existing rows are NULL. Move them to 0 so the
-- column is fully populated and dispatcher gate behaviour is identical to
-- newly-created rows. Idempotent: the second run hits 0 rows.
UPDATE sentinels SET cooldown_minutes = 0 WHERE cooldown_minutes IS NULL;

COMMENT ON COLUMN sentinels.cooldown_minutes IS
'Minimum minutes between consecutive fires. 0 (default) or NULL = no cooldown gate (always evaluate). Enforced by src.memory.sentinels.registry::dispatch_sentinels and bounded by the CHECK constraint below. iter-1 M2: DEFAULT 0 + backfill from NULL.';

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
