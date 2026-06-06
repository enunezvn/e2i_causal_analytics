-- =============================================================================
-- Drop the inert falkordb_synced / falkordb_sync_at columns (audit 2026-06-05, F4)
-- Migration: 034_drop_inert_falkordb_sync_columns.sql
-- Date: 2026-06-06
-- =============================================================================
--
-- AUDIT F4: `semantic_memory_cache.falkordb_synced` (BOOLEAN) and
-- `falkordb_sync_at` (TIMESTAMPTZ), added in 001, were never written or read —
-- there is no Supabase->FalkorDB sync-back job. Verified inert:
--   * `grep -rniE 'falkordb_synced|falkordb_sync_at' src/`      -> 0 references
--   * `grep -rniE 'falkordb_synced|falkordb_sync_at' database/`  -> only the 001
--     column definitions; the populating RPC `sync_hcp_patient_relationships_to_cache`
--     (001b) does NOT reference either column in its INSERT column lists.
--
-- The S3 commit (9cb0dc19) removed the inert `semantic_cache_ttl_minutes` config
-- control but DEFERRED this DROP COLUMN to a live-DB op (see the decision log
-- docs/plans/memory-remediation-decisions-20260605.md, D2). This migration closes
-- that deferred item in-tree.
--
-- SUBSTRATE PRESERVED (audit D2, "do not omit features fished for later"): the
-- `semantic_memory_cache` table and its populating RPC are KEPT as deploy-seed-only
-- scaffolding for a future FalkorDB->Supabase hot-cache mirror. Only the two dead
-- sync-state columns are dropped. NOTHING IS FORECLOSED: if/when that mirror is
-- built, a sync-state column is trivially re-added (this migration + git history
-- document the original shape).
--
-- Idempotent (IF EXISTS); safe to re-apply.
-- =============================================================================

ALTER TABLE IF EXISTS semantic_memory_cache
    DROP COLUMN IF EXISTS falkordb_synced,
    DROP COLUMN IF EXISTS falkordb_sync_at;
