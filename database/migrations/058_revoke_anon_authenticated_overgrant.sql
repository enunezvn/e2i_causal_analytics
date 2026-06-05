-- ============================================================================
-- Migration 058: REVOKE the anon/authenticated over-grant on the public schema
--                + DROP the anon-callable arbitrary-SQL RPC (M9 / issue #703)
-- ============================================================================
--
-- WHAT:
--   1. DROP public.execute_custom_sql(text) — an anon-executable SECURITY DEFINER
--      function that string-concats client SQL and runs it as postgres
--      (`EXECUTE 'SELECT jsonb_agg(t) FROM (' || sql_query || ') t'`). It is a
--      TOTAL bypass of any table/view REVOKE (anon could read pg_authid etc.).
--   2. REVOKE ALL privileges on every public TABLE + VIEW (+ materialized view)
--      from the anon and authenticated roles.
--   3. Neutralise the Supabase DEFAULT PRIVILEGES that re-grant ALL to
--      anon/authenticated on every newly-created public table/view, for BOTH
--      roles that carry that default (postgres and supabase_admin).
--
-- WHY (M9): the Supabase default `GRANT ALL ON ... TO anon, authenticated` left
--   ~101 no-RLS public tables (+105 anon-granted views) readable/writable by the
--   anon role via the internet-bound Kong gateway with the public anon key —
--   PHI-bearing tables (patient_journeys, hcp_profiles, ml_predictions, the
--   memory tables) included. Proven live: anon GET /rest/v1/<table> returned 206
--   with row counts. anon also held INSERT/UPDATE/DELETE/TRUNCATE (tamper/wipe).
--
-- SAFE TO REVOKE (proven, see PR): (a) the frontend NEVER calls PostgREST — it
--   uses Supabase only for GoTrue auth (auth schema / supabase_auth_admin role,
--   untouched here) and all data via FastAPI /api; (b) the backend authenticates
--   as SERVICE-ROLE after the paired code change (factories.py get_supabase_client
--   / get_async_supabase_client), and service_role bypasses RLS + retains grants.
--   Kong logs: 0 browser hits on /rest/v1; service-role parity verified live.
--
-- DEPLOY ORDER (coupled): the service-role code change must be LIVE before this
--   migration is applied (else the backend, still on anon, loses table access).
--   merge PR -> auto-deploy ships service-role -> verify backend healthy ->
--   THEN apply this migration manually on the droplet.
--
-- execute_custom_sql DROP is intent-verified: chat/009 created it + tried to
--   REVOKE it (FROM PUBLIC, authenticated — but MISSED the anon role, and the
--   default ACL re-grants EXECUTE to anon); migrations/044 replaced it with the
--   guarded kpi_query allow-list and calls it "unsafe"; 0 code consumers; never
--   invoked (Kong logs show only execute_sql -> 404, a different dead name).
--
-- SCOPE (per owner decision): tables + views, REVOKE only (RLS deferred). Other
--   anon-granted public FUNCTIONS (e.g. the guarded kpi_query allow-list) and
--   SEQUENCES are intentionally left in place and tracked as follow-ups; only the
--   dangerous execute_custom_sql is removed here.
--
-- IDEMPOTENT: DROP ... IF EXISTS; REVOKE is inherently re-runnable; ALTER DEFAULT
--   PRIVILEGES REVOKE is re-runnable. Safe to re-apply.
--
-- NO SCRIPT-LEVEL BEGIN/COMMIT: scripts/run_migrations.sh wraps every migration in
--   `psql --single-transaction`, so the runner owns the outer transaction (a bare
--   BEGIN/COMMIT here would fragment it — see test_migrations_no_inner_txn.py and
--   the canonical shape in 039_drop_triggers_join_from_feedback_loop.sql). For a
--   MANUAL apply, run atomically with: psql --single-transaction -f <this file>
--   (the DO blocks below are PL/pgSQL function bodies, not script-level txn control).
-- ============================================================================

-- 1. Remove the arbitrary-SQL bypass primitive (vestigial + dangerous).
DROP FUNCTION IF EXISTS public.execute_custom_sql(text);

-- 2. Revoke anon/authenticated on every public TABLE and VIEW.
--    (REVOKE ... ON ALL TABLES covers ordinary tables, partitioned tables, and
--    views; materialized views are handled separately in step 2b.)
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;

-- 2b. Materialized views are NOT covered by "ON ALL TABLES"; revoke each.
DO $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT c.relname
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relkind = 'm'
    LOOP
        EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', r.relname);
    END LOOP;
END $$;

-- 3. Stop FUTURE public tables/views from re-inheriting the default GRANT ALL.
--    The granting defaults are owned by BOTH postgres and supabase_admin
--    (pg_default_acl). The postgres default governs OUR flow: migrations are
--    applied as postgres, so new tables are created by postgres and pick up the
--    postgres default — revoke it directly. The supabase_admin default is
--    best-effort: postgres is NOT a member of supabase_admin, so applied as
--    postgres that ALTER is skipped with a NOTICE (the postgres REVOKE already
--    covers migration-created tables). Apply as supabase_admin to neutralise
--    both. Kept in one migration so the static test sees both roles.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON TABLES FROM anon, authenticated;
DO $$
BEGIN
    -- ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public REVOKE ALL ON TABLES FROM anon, authenticated
    EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public REVOKE ALL ON TABLES FROM anon, authenticated';
    RAISE NOTICE 'migration 058: neutralised the supabase_admin default-privilege grant to anon/authenticated';
EXCEPTION WHEN insufficient_privilege OR undefined_object THEN
    -- insufficient_privilege: applier is not a member of supabase_admin (e.g. run
    --   as postgres) — the postgres default REVOKE above already covers
    --   migration-created tables; re-apply as supabase_admin to also neutralise this.
    -- undefined_object: supabase_admin role absent (e.g. a vanilla Postgres) — n/a.
    RAISE NOTICE 'migration 058: SKIPPED supabase_admin default-privilege REVOKE (not a member of, or absent, supabase_admin). The postgres default REVOKE covers migration-created tables; re-apply as supabase_admin to also neutralise the supabase_admin default.';
END $$;
