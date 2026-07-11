-- ============================================================================
-- Migration 101: Admin user management — activity log + auth-schema RPCs
-- ============================================================================
-- Spec: docs/superpowers/specs/2026-07-11-admin-user-management-design.md
-- 1) user_activity_log: per-minute pre-aggregated API activity (bounded rows)
-- 2) record_user_activity(jsonb): additive upsert flush target (PostgREST
--    upsert cannot merge counts, so flush goes through this RPC)
-- 3) admin_get_login_activity / admin_get_platform_activity /
--    admin_get_user_recent_events: SECURITY DEFINER readers over
--    auth.audit_log_entries (PostgREST cannot reach the auth schema)
-- 4) chatbot_user_profiles backfill: every auth user gets a profile row;
--    profile role synced from app_metadata where app_metadata has a role
-- 5) purge_old_user_activity(days): retention (manual/cron follow-up)
-- Grants: RPCs are service_role-only. Table RLS mirrors 012 (admins read).

CREATE TABLE IF NOT EXISTS user_activity_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    user_email TEXT,
    endpoint_group TEXT NOT NULL,
    http_method TEXT NOT NULL DEFAULT 'GET',
    bucket_minute TIMESTAMPTZ NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_activity_bucket
        UNIQUE (user_id, endpoint_group, http_method, bucket_minute)
);

CREATE INDEX IF NOT EXISTS idx_user_activity_user_time
    ON user_activity_log (user_id, bucket_minute DESC);
CREATE INDEX IF NOT EXISTS idx_user_activity_time
    ON user_activity_log (bucket_minute DESC);

ALTER TABLE user_activity_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS user_activity_admin_read ON user_activity_log;
CREATE POLICY user_activity_admin_read ON user_activity_log
    FOR SELECT TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM chatbot_user_profiles p
            WHERE p.id = auth.uid() AND p.role = 'admin'
        )
    );
-- service_role bypasses RLS; no insert policy needed for authenticated.

-- Additive-merge flush target. p_rows: [{user_id, user_email, endpoint_group,
-- http_method, bucket_minute, request_count}, ...]
CREATE OR REPLACE FUNCTION record_user_activity(p_rows JSONB)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    INSERT INTO user_activity_log
        (user_id, user_email, endpoint_group, http_method, bucket_minute, request_count)
    SELECT (r->>'user_id')::uuid,
           r->>'user_email',
           r->>'endpoint_group',
           COALESCE(r->>'http_method', 'GET'),
           (r->>'bucket_minute')::timestamptz,
           COALESCE((r->>'request_count')::int, 1)
    FROM jsonb_array_elements(p_rows) AS r
    ON CONFLICT (user_id, endpoint_group, http_method, bucket_minute)
    DO UPDATE SET request_count = user_activity_log.request_count + EXCLUDED.request_count;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;

-- Login/auth history per user (or platform-wide when p_user_id IS NULL).
CREATE OR REPLACE FUNCTION admin_get_login_activity(
    p_user_id UUID DEFAULT NULL,
    p_days INTEGER DEFAULT 90
)
RETURNS TABLE (day DATE, event_type TEXT, event_count BIGINT)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, auth
AS $$
    SELECT date(e.created_at) AS day,
           COALESCE(e.payload->>'action', 'unknown') AS event_type,
           count(*) AS event_count
    FROM auth.audit_log_entries e
    WHERE e.created_at >= now() - make_interval(days => p_days)
      AND (p_user_id IS NULL OR (e.payload->>'actor_id')::uuid = p_user_id)
    GROUP BY 1, 2
    ORDER BY 1;
$$;

-- Platform overview: logins/day + distinct active users/day.
CREATE OR REPLACE FUNCTION admin_get_platform_activity(p_days INTEGER DEFAULT 30)
RETURNS TABLE (day DATE, logins BIGINT, active_users BIGINT)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, auth
AS $$
    SELECT date(e.created_at) AS day,
           count(*) FILTER (WHERE e.payload->>'action' = 'login') AS logins,
           count(DISTINCT e.payload->>'actor_id')
               FILTER (WHERE e.payload->>'action' IN ('login', 'token_refreshed')) AS active_users
    FROM auth.audit_log_entries e
    WHERE e.created_at >= now() - make_interval(days => p_days)
    GROUP BY 1
    ORDER BY 1;
$$;

-- Recent raw auth events for one user (feed).
CREATE OR REPLACE FUNCTION admin_get_user_recent_events(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 25
)
RETURNS TABLE (occurred_at TIMESTAMPTZ, action TEXT)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, auth
AS $$
    SELECT e.created_at AS occurred_at,
           COALESCE(e.payload->>'action', 'unknown') AS action
    FROM auth.audit_log_entries e
    WHERE (e.payload->>'actor_id')::uuid = p_user_id
    ORDER BY e.created_at DESC
    LIMIT LEAST(p_limit, 200);
$$;

CREATE OR REPLACE FUNCTION purge_old_user_activity(p_days INTEGER DEFAULT 180)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    DELETE FROM user_activity_log
    WHERE bucket_minute < now() - make_interval(days => p_days);
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;

-- Lock the RPCs to service_role only (backend admin service).
REVOKE ALL ON FUNCTION record_user_activity(JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION admin_get_login_activity(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION admin_get_platform_activity(INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION admin_get_user_recent_events(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION purge_old_user_activity(INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION record_user_activity(JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION admin_get_login_activity(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION admin_get_platform_activity(INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION admin_get_user_recent_events(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION purge_old_user_activity(INTEGER) TO service_role;

-- Backfill: every auth user has a chatbot profile row (old users predate the
-- trigger — verified: admin@e2i.local had none).
INSERT INTO chatbot_user_profiles (id, email)
SELECT u.id, u.email
FROM auth.users u
ON CONFLICT (id) DO NOTHING;

-- Sync profile role/is_admin FROM app_metadata where app_metadata carries a
-- valid role (JWT is authoritative). The reverse direction (NULL jwt role,
-- profile role set) requires the auth admin API and runs via
-- AdminUserService.reconcile_role_stores (plan Task 16).
UPDATE chatbot_user_profiles p
SET role = (u.raw_app_meta_data->>'role')::user_role,
    is_admin = ((u.raw_app_meta_data->>'role') = 'admin')
FROM auth.users u
WHERE u.id = p.id
  AND u.raw_app_meta_data->>'role' IN ('viewer', 'analyst', 'operator', 'admin')
  AND (p.role IS DISTINCT FROM (u.raw_app_meta_data->>'role')::user_role
       OR p.is_admin IS DISTINCT FROM ((u.raw_app_meta_data->>'role') = 'admin'));
