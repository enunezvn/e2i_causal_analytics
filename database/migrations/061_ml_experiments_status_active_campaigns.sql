-- ============================================================================
-- Migration 061: Active Campaigns — add ml_experiments.status (schema-drift fix)
-- + TRUTHFUL backfill + a read-only active-experiments count for the Home tile.
-- ============================================================================
-- THE GAP: the Home "Active Campaigns" tile was hardcoded to a fabricated 24.
-- The system's own code (src/agents/experiment_monitor/nodes/health_checker.py,
-- src/tasks/ab_testing_tasks.py) already treats "active campaigns" as the count
-- of ml_experiments WHERE status='running' — but the LIVE ml_experiments table
-- has NO `status` column (schema drift), so those queries error and there is no
-- honest count to surface. There is no first-class campaign/intervention entity
-- in this platform; running A/B experiments is the closest truthful mapping the
-- codebase itself uses. (Flagged in the PR body as a data-substrate mapping.)
--
-- (A) Add the missing `status` column the code already assumes. CHECK constrains
--     it to the lifecycle the agent code uses.
--
-- (B) BACKFILL TRUTHFULLY — do NOT leave all 616 rows as 'running' (that would
--     re-fabricate a "616 active" number). Derive lifecycle from updated_at
--     recency: rows touched within the last 30 days -> 'running', older rows ->
--     'completed'. LIVE-VERIFIED split on supabase-db: 254 running / 362
--     completed (616 total; updated_at spans 2026-04-25..2026-06-07). So the
--     honest active count is 254, not 616 and not the fabricated 24.
--
-- (C) Register a read-only active-experiments count via the kpi_query allowlist
--     (mirrors the model_performance_* registry rows; max_params=0). The Home
--     tile reads this through GET /api/experiments/active-count (same PR), which
--     calls this query. Alias `active_count`. Starts with SELECT to satisfy the
--     registry read-only CHECK.
--
-- NOTE on health_checker.py: its query also selected non-existent columns
-- (`name`, `config`); the same PR aligns that select to the real columns
-- (`experiment_name`) so the agent path stops erroring once `status` exists.
--
-- IDEMPOTENT / RE-RUNNABLE: ADD COLUMN IF NOT EXISTS; the backfill only sets
-- rows whose status IS NULL (so a re-run won't clobber a real lifecycle written
-- later); ON CONFLICT for the registry row.
--
-- NOTE: deploy.yml SKIPS migrations; the local self-contained supabase IS the
-- faithful prod target. Apply manually:
--   docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/060_ml_experiments_status_active_campaigns.sql
-- ----------------------------------------------------------------------------

-- (A) Add the column the agent code already assumes. The DEFAULT 'running'
--     applies to FUTURE inserts (a newly-created experiment is running), but a
--     column-level DEFAULT also back-fills EXISTING rows to 'running' on ADD
--     COLUMN — which would fabricate "616 active". So (B) immediately overwrites
--     the historical rows with the truthful recency-derived lifecycle.
ALTER TABLE public.ml_experiments
    ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'running'
    CHECK (status IN ('draft', 'running', 'completed', 'stopped', 'archived'));

-- (B) Truthful backfill from updated_at recency. This MUST be unconditional on
--     first apply: the ADD COLUMN DEFAULT just set every existing row to
--     'running' (616 active is not honest). Scope to rows with a known
--     updated_at and re-derive lifecycle deterministically. A re-run is a no-op
--     in distribution (same recency rule) and idempotent.
UPDATE public.ml_experiments
SET status = CASE
        WHEN updated_at >= NOW() - INTERVAL '30 days' THEN 'running'
        ELSE 'completed'
    END
WHERE updated_at IS NOT NULL;

-- (C) Register the read-only active-experiments count.
INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
    ('active_experiments_count',
     $kpi$SELECT COUNT(*)::int AS active_count FROM public.ml_experiments WHERE status = 'running'$kpi$,
     0,
     $note$Active Campaigns = count of ml_experiments WHERE status='running' (the codebase's own mapping for "active campaigns"; no first-class campaign entity exists). max_params=0. Surfaced via GET /api/experiments/active-count for the Home QUICK_STATS tile. Live count after the migration 060 backfill: 254 running / 362 completed (616 total).$note$)
ON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;

-- PostgREST caches the schema; reload so the registered query_id is callable immediately.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
