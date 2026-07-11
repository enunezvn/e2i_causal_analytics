# Admin User Management & Activity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/admin` surface for eznomics.site — invite users via copyable links, manage roles/brands, disable/enable/delete users, and view per-user + platform activity over time — per the approved spec `docs/superpowers/specs/2026-07-11-admin-user-management-design.md`.

**Architecture:** Thin admin layer over Supabase GoTrue: new `/api/admin/*` router behind existing `require_admin`, a service-role `AdminUserService`, migration 101 (activity table + SECURITY DEFINER RPCs over `auth.audit_log_entries`), a bounded fail-open activity middleware, the one-line `security_audit` DB-sink fix, and a React `/admin` page + public `/accept-invite` page.

**Tech Stack:** Python 3.12 / FastAPI / supabase-py 2.27 (sync) / pytest real-DB integration tests (`E2I_DB_INTEGRATION=1` opt-in, no mocks); React 18 / TypeScript / TanStack Query / recharts / vitest + testing-library (vi.mock convention).

**Verified ground truth this plan encodes (do NOT re-litigate; re-verify only if a test contradicts it):**
- `auth.admin.generate_link({"type":"invite","email":...})` → `.properties.hashed_token`; the SPA completes it same-origin via `supabase.auth.verifyOtp({type:'invite', token_hash})` → session → `updateUser({password})` → future sign-ins work. Proven live 2026-07-11.
- Reinvite: pending user → invite-type regenerates (old link invalidated); active user → GoTrue 422 "already been registered" → use `{"type":"recovery"}` link instead.
- Ban does NOT kill existing access tokens and `banned_until` is unreadable via gotrue-py. Disable therefore = ban + `app_metadata.disabled: true`, enforced fail-closed in `verify_supabase_token`.
- Role stores are TWO: JWT `app_metadata` (authoritative for API) + `chatbot_user_profiles.role`/`is_admin` (RLS). `user_profiles` does NOT exist in prod.
- `chatbot_user_profiles` row is auto-created per auth user by trigger `on_auth_user_created_chatbot` — but old users may lack rows (e.g. `admin@e2i.local`): always UPSERT.
- `security_audit_log` columns exactly match `SecurityAuditEvent.to_dict()` keys; the only bug is `from src.api.deps import get_supabase` (module doesn't exist → real one is `src.api.dependencies`).
- `auth.audit_log_entries.payload` shape: `{"action": "login"|"token_refreshed"|"user_deleted"|..., "actor_id": "<uuid>", "actor_username": ..., "traits": {...}}`.
- Starlette middleware: LAST `add_middleware` call = OUTERMOST (runs first inbound). `JWTAuthMiddleware` sets `request.state.user` (auth_middleware.py:335). Activity middleware must be added BEFORE the JWT add call in code (= inner = sees `state.user`).
- Frontend: `env.supabaseUrl` defaults to `window.location.origin` (same-origin prod); `isAdmin` comes from `useAuth()` (`frontend/src/hooks/use-auth.ts:134`); api functions use `get/post/patch/del` from `@/lib/api-client`; page tests `vi.mock` the hooks module.

**Execution constraints (user-mandated):**
- Worktree isolation; TDD red-first; single PR at the end; **CI/deploy batched — ONE push at the very end** (OOM history; racing deploys). Do not push intermediate commits.
- No mocking in product paths. Backend tests hit the REAL local Supabase stack. Frontend unit tests use the repo's vi.mock convention; final verification is live on eznomics.site.
- Droplet memory policy: targeted pytest/mypy only (never whole-tree); watch `free -m` during heavy steps.
- Migrations applied manually: `docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < file`. Idempotent, no BEGIN/COMMIT.
- Convergence: after Task 14, run ralph-loop + codex:codex-rescue audits until fixed point (no findings), then PR.

**Test commands (host .venv, targeted):**
```bash
# backend real-DB integration (opt-in gate, mirrors tests/integration convention)
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_user_service_realdb.py -p no:cacheprovider -v
# backend unit (no DB)
.venv/bin/pytest tests/unit/test_activity_buffer.py -v
# frontend
cd frontend && npx vitest run src/pages/Admin.test.tsx && npx tsc -p tsconfig.app.json --noEmit
```

---

### Task 0: Worktree + branch

**Files:** none (setup)

- [ ] **Step 0.1:** From the repo root (local `main`, which already carries the spec commits):

```bash
git worktree add /home/enunez/Projects/e2i_worktrees/admin-user-mgmt -b feat/admin-user-management
cd /home/enunez/Projects/e2i_worktrees/admin-user-mgmt
ln -sf /home/enunez/Projects/e2i_causal_analytics/.env .env   # worktrees lack .env (memory: digital-twin session)
ln -sfn /home/enunez/Projects/e2i_causal_analytics/.venv .venv  # reuse host venv for targeted pytest
```

All subsequent git commands run FROM THE WORKTREE dir (standing feedback 2026-06-30). All file paths below are relative to the worktree root.

- [ ] **Step 0.2:** Copy this plan + spec are already in the branch (they were committed to main before branching). Verify: `git log --oneline -3` shows the spec/plan commits.

---

### Task 1: Migration 101 — activity table, RPCs, backfill, purge

**Files:**
- Create: `database/migrations/101_admin_user_activity.sql`

- [ ] **Step 1.1: Write the migration** (idempotent, autocommit-safe, NO BEGIN/COMMIT):

```sql
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
-- profile role set) requires the auth admin API and runs in Task 13
-- (AdminUserService.reconcile_role_stores).
UPDATE chatbot_user_profiles p
SET role = (u.raw_app_meta_data->>'role')::user_role,
    is_admin = ((u.raw_app_meta_data->>'role') = 'admin')
FROM auth.users u
WHERE u.id = p.id
  AND u.raw_app_meta_data->>'role' IN ('viewer', 'analyst', 'operator', 'admin')
  AND (p.role IS DISTINCT FROM (u.raw_app_meta_data->>'role')::user_role
       OR p.is_admin IS DISTINCT FROM ((u.raw_app_meta_data->>'role') = 'admin'));
```

- [ ] **Step 1.2: Dry-run on the droplet** (transactional preview — this migration has no ALTER TYPE, so a txn wrapper is safe):

```bash
{ echo "BEGIN;"; cat database/migrations/101_admin_user_activity.sql; echo "ROLLBACK;"; } | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1
```

Expected: runs to `ROLLBACK` with no errors (NOTICEs are fine).

- [ ] **Step 1.3: Apply for real** (additive + idempotent; safe mid-development, user authorized migrations):

```bash
docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < database/migrations/101_admin_user_activity.sql
```

- [ ] **Step 1.4: Verify:**

```bash
docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT count(*) FROM user_activity_log"                     # 0
docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT * FROM admin_get_platform_activity(30) LIMIT 3"     # rows with real login counts
docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT count(*) FROM chatbot_user_profiles"                # == count of auth.users (8)
docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT p.role FROM chatbot_user_profiles p JOIN auth.users u ON u.id=p.id WHERE u.email='etn3724@gmail.com'"  # admin
```

- [ ] **Step 1.5: Commit**

```bash
git add database/migrations/101_admin_user_activity.sql
git commit -m "feat(admin): migration 101 — user_activity_log, auth-audit RPCs, profile backfill"
```

---

### Task 2: Fix the security-audit DB sink (red-first)

**Files:**
- Test: `tests/integration/test_security_audit_sink_realdb.py` (create)
- Modify: `src/utils/security_audit.py:736` (one line)

- [ ] **Step 2.1: Write the failing test:**

```python
"""Red-first pin for the security_audit_log DB sink (spec: admin feature).

ROOT CAUSE (verified 2026-07-11): get_security_audit_service() imports
``from src.api.deps import get_supabase`` — src.api.deps does not exist
(the real module is src.api.dependencies). The ImportError is swallowed,
self.db stays None, and every security event ever emitted went to
stdout only. security_audit_log has 0 rows in prod despite months of
auth-failure logging.

Real-DB test (no mocks): requires docker supabase-db + service key in env.
    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_security_audit_sink_realdb.py -p no:cacheprovider -v
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


def test_security_audit_service_persists_to_database():
    from src.api.dependencies import get_supabase
    from src.utils.security_audit import (
        get_security_audit_service,
        reset_security_audit_service,
    )

    reset_security_audit_service()
    try:
        service = get_security_audit_service()
        # The factory must have wired a real client (this is the red assertion:
        # today the broken import leaves service.db None).
        assert service.db is not None, (
            "security audit service has no DB sink — src.api.deps import bug"
        )

        marker = f"sink-test-{uuid.uuid4()}"
        service.log_auth_failure(
            user_email="sink-test@example.invalid",
            client_ip="127.0.0.1",
            reason=marker,
        )

        client = get_supabase()
        rows = (
            client.table("security_audit_log")
            .select("event_id, event_type, error_details")
            .eq("error_details", marker)
            .execute()
        )
        assert len(rows.data) == 1, f"expected 1 persisted row, got {rows.data}"
        assert rows.data[0]["event_type"] == "auth.login.failure"

        # cleanup
        client.table("security_audit_log").delete().eq("error_details", marker).execute()
    finally:
        reset_security_audit_service()
```

- [ ] **Step 2.2: Run to verify it FAILS:**

```bash
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_security_audit_sink_realdb.py -p no:cacheprovider -v
```

Expected: FAIL at `assert service.db is not None`.

- [ ] **Step 2.3: Fix the import** in `src/utils/security_audit.py` (inside `get_security_audit_service`, ~line 736):

```python
        # Try to get Supabase client if configured
        supabase_client = None
        try:
            from src.api.dependencies import get_supabase

            supabase_client = get_supabase()
        except Exception:
            pass  # Running without database is fine
```

(Only the import line changes: `src.api.deps` → `src.api.dependencies`. `get_supabase` is re-exported from `src/api/dependencies/__init__.py`.)

- [ ] **Step 2.4: Run to verify it PASSES** (same command). If the insert fails on a column mismatch, the error surfaces in the test — `to_dict()` keys were verified to match the table exactly, so a failure here means the DB diverged: inspect with `docker exec supabase-db psql ... "\d security_audit_log"` before changing code.

- [ ] **Step 2.5: Commit**

```bash
git add tests/integration/test_security_audit_sink_realdb.py src/utils/security_audit.py
git commit -m "fix(security-audit): wire DB sink — src.api.deps import never existed"
```

---

### Task 3: `app_metadata.disabled` fail-closed check in token verification (red-first)

**Files:**
- Test: `tests/integration/test_auth_disabled_flag_realdb.py` (create)
- Modify: `src/api/dependencies/auth.py:321` (verify_supabase_token)

- [ ] **Step 3.1: Write the failing test:**

```python
"""Disable must lock out EXISTING tokens immediately (spec verified fact 6).

DISPROVED by live experiment 2026-07-11: a GoTrue ban does NOT invalidate an
existing access token (get_user still succeeds; gotrue-py's User model has no
banned_until). Ban alone leaves a <=1h window. The fix: disable sets
app_metadata.disabled=true and verify_supabase_token fails closed on it —
get_user returns FRESH app_metadata per request, so lockout is immediate.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_auth_disabled_flag_realdb.py -p no:cacheprovider -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

EMAIL = "etn3724+admtest-disabled@gmail.com"
PASSWORD = "AdmTest#2026-disabled"


@pytest.fixture()
def disposable_user():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    admin = create_client(
        url,
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"],
    )
    anon = create_client(url, os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"])
    for u in admin.auth.admin.list_users():
        if u.email == EMAIL:
            admin.auth.admin.delete_user(u.id)
    created = admin.auth.admin.create_user(
        {"email": EMAIL, "password": PASSWORD, "email_confirm": True}
    )
    session = anon.auth.sign_in_with_password({"email": EMAIL, "password": PASSWORD})
    yield admin, created.user.id, session.session.access_token
    admin.auth.admin.delete_user(created.user.id)


@pytest.mark.asyncio
async def test_disabled_flag_rejects_live_token(disposable_user):
    from src.api.dependencies.auth import verify_supabase_token

    admin, user_id, token = disposable_user

    # Token is valid before the flag
    user = await verify_supabase_token(token)
    assert user is not None and user["email"] == EMAIL

    # Set the disabled flag (what /disable will do) — same LIVE token must now fail
    admin.auth.admin.update_user_by_id(user_id, {"app_metadata": {"disabled": True}})
    assert await verify_supabase_token(token) is None, (
        "disabled user's existing token must be rejected immediately"
    )

    # Clearing the flag restores access (what /enable will do)
    admin.auth.admin.update_user_by_id(user_id, {"app_metadata": {"disabled": False}})
    user = await verify_supabase_token(token)
    assert user is not None
```

- [ ] **Step 3.2: Run to verify it FAILS** (second assertion — token still accepted):

```bash
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_auth_disabled_flag_realdb.py -p no:cacheprovider -v
```

- [ ] **Step 3.3: Implement** in `src/api/dependencies/auth.py` — inside `verify_supabase_token`, right after `if response and response.user:` (line ~321), BEFORE building `user_data`:

```python
        if response and response.user:
            app_metadata = response.user.app_metadata or {}
            # Admin-disabled users are locked out immediately. GoTrue bans do
            # NOT invalidate already-issued access tokens (verified 2026-07-11),
            # so /api/admin disable sets app_metadata.disabled and we fail
            # closed on the FRESH app_metadata get_user returns per request.
            if app_metadata.get("disabled"):
                logger.warning(
                    "Rejected token for disabled user: %s", response.user.email
                )
                return None
            user_data = {
                "id": response.user.id,
                "email": response.user.email,
                "role": response.user.role,
                "aud": response.user.aud,
                "created_at": str(response.user.created_at) if response.user.created_at else None,
                "app_metadata": app_metadata,
                "user_metadata": response.user.user_metadata or {},
            }
```

- [ ] **Step 3.4: Run to verify it PASSES** (same command).

- [ ] **Step 3.5: Commit**

```bash
git add tests/integration/test_auth_disabled_flag_realdb.py src/api/dependencies/auth.py
git commit -m "feat(auth): fail closed on app_metadata.disabled — bans don't kill live tokens"
```

---

### Task 4: ActivityBuffer (pure, unit-tested, OOM-safe)

**Files:**
- Test: `tests/unit/test_activity_buffer.py` (create)
- Create: `src/api/middleware/activity_tracking.py` (buffer class only in this task)

- [ ] **Step 4.1: Write the failing tests:**

```python
"""Unit tests for the bounded activity aggregation buffer (no DB, no mocks —
pure data structure). OOM-safety is the point: the buffer must CAP distinct
buckets and drop (counting drops) rather than grow."""

from src.api.middleware.activity_tracking import ActivityBuffer

UID = "11111111-1111-1111-1111-111111111111"


def test_record_aggregates_same_bucket():
    buf = ActivityBuffer(max_buckets=10, flush_interval_s=9999, flush_threshold=9999)
    for _ in range(5):
        buf.record(UID, "a@x.com", "causal", "GET", "2026-07-11T15:00:00+00:00")
    rows = buf.drain()
    assert len(rows) == 1
    assert rows[0]["request_count"] == 5
    assert rows[0]["endpoint_group"] == "causal"
    assert rows[0]["user_id"] == UID


def test_distinct_buckets_are_separate_rows():
    buf = ActivityBuffer(max_buckets=10, flush_interval_s=9999, flush_threshold=9999)
    buf.record(UID, "a@x.com", "causal", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "kpis", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "causal", "POST", "2026-07-11T15:00:00+00:00")
    assert len(buf.drain()) == 3


def test_cap_drops_new_buckets_but_still_counts_existing():
    buf = ActivityBuffer(max_buckets=2, flush_interval_s=9999, flush_threshold=9999)
    buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00")
    buf.record(UID, "a@x.com", "g2", "GET", "2026-07-11T15:00:00+00:00")
    # new bucket beyond cap -> dropped
    buf.record(UID, "a@x.com", "g3", "GET", "2026-07-11T15:00:00+00:00")
    # existing bucket still increments at cap
    buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00")
    assert buf.dropped == 1
    rows = {r["endpoint_group"]: r["request_count"] for r in buf.drain()}
    assert rows == {"g1": 2, "g2": 1}


def test_flush_threshold_and_drain_resets():
    buf = ActivityBuffer(max_buckets=100, flush_interval_s=9999, flush_threshold=2)
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00") is False
    assert buf.record(UID, "a@x.com", "g2", "GET", "2026-07-11T15:00:00+00:00") is True
    assert buf.drain() != []
    assert buf.drain() == []  # drained


def test_time_based_flush(monkeypatch):
    import src.api.middleware.activity_tracking as at

    t = {"now": 1000.0}
    monkeypatch.setattr(at.time, "monotonic", lambda: t["now"])
    buf = ActivityBuffer(max_buckets=100, flush_interval_s=30.0, flush_threshold=9999)
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:00+00:00") is False
    t["now"] = 1031.0
    assert buf.record(UID, "a@x.com", "g1", "GET", "2026-07-11T15:00:01+00:00") is True
```

- [ ] **Step 4.2: Run to verify FAIL** (`ModuleNotFoundError`):

```bash
.venv/bin/pytest tests/unit/test_activity_buffer.py -v
```

- [ ] **Step 4.3: Implement** `src/api/middleware/activity_tracking.py`:

```python
"""Per-request user activity tracking (admin feature, spec 2026-07-11).

ActivityBuffer is a BOUNDED in-memory aggregator: (user, endpoint_group,
method, minute) -> count. Bounded because this box has OOM history — past the
cap, NEW buckets are dropped (counted in .dropped); existing buckets keep
incrementing. Flush drains to the record_user_activity RPC (additive upsert),
fired as a background task from the middleware. Everything here is fail-open:
tracking must never block or break a request.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ActivityBuffer:
    """Bounded aggregation of per-request activity into per-minute buckets."""

    def __init__(
        self,
        max_buckets: int = 2048,
        flush_interval_s: float = 30.0,
        flush_threshold: int = 500,
    ):
        self.max_buckets = max_buckets
        self.flush_interval_s = flush_interval_s
        self.flush_threshold = flush_threshold
        self.dropped = 0
        self._buckets: Dict[Tuple[str, Optional[str], str, str, str], int] = {}
        self._last_flush = time.monotonic()

    def record(
        self,
        user_id: str,
        user_email: Optional[str],
        endpoint_group: str,
        http_method: str,
        bucket_minute_iso: str,
    ) -> bool:
        """Record one request. Returns True when the caller should flush."""
        key = (user_id, user_email, endpoint_group, http_method, bucket_minute_iso)
        if key not in self._buckets and len(self._buckets) >= self.max_buckets:
            self.dropped += 1
        else:
            self._buckets[key] = self._buckets.get(key, 0) + 1
        return self.should_flush()

    def should_flush(self) -> bool:
        if not self._buckets:
            return False
        if len(self._buckets) >= self.flush_threshold:
            return True
        return (time.monotonic() - self._last_flush) >= self.flush_interval_s

    def drain(self) -> List[Dict[str, Any]]:
        """Return accumulated rows (RPC payload shape) and reset the buffer."""
        rows = [
            {
                "user_id": k[0],
                "user_email": k[1],
                "endpoint_group": k[2],
                "http_method": k[3],
                "bucket_minute": k[4],
                "request_count": count,
            }
            for k, count in self._buckets.items()
        ]
        self._buckets = {}
        self._last_flush = time.monotonic()
        return rows
```

- [ ] **Step 4.4: Run to verify PASS** (same command).

- [ ] **Step 4.5: Commit**

```bash
git add tests/unit/test_activity_buffer.py src/api/middleware/activity_tracking.py
git commit -m "feat(admin): bounded ActivityBuffer for per-minute API activity aggregation"
```

---

### Task 5: ActivityTrackingMiddleware + flush + wiring

**Files:**
- Modify: `src/api/middleware/activity_tracking.py` (append middleware + flush)
- Test: `tests/integration/test_activity_tracking_realdb.py` (create)
- Modify: `src/api/main.py` (import ~line 62; add_middleware after the InsightVerifier block ~line 845)

- [ ] **Step 5.1: Write the failing integration test:**

```python
"""Middleware flush writes REAL rows via the record_user_activity RPC and the
RPC merges counts additively (no mocks).

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_activity_tracking_realdb.py -p no:cacheprovider -v
"""

import os
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


@pytest.fixture()
def db():
    from supabase import create_client

    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"],
    )
    test_user = str(uuid.uuid4())
    yield client, test_user
    client.table("user_activity_log").delete().eq("user_id", test_user).execute()


@pytest.mark.asyncio
async def test_flush_rows_persists_and_merges(db):
    from src.api.middleware.activity_tracking import flush_rows

    client, test_user = db
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    row = {
        "user_id": test_user,
        "user_email": "activity-test@example.invalid",
        "endpoint_group": "causal",
        "http_method": "GET",
        "bucket_minute": minute,
        "request_count": 3,
    }
    await flush_rows([row])
    await flush_rows([dict(row, request_count=2)])  # same bucket -> additive merge

    got = (
        client.table("user_activity_log")
        .select("request_count, endpoint_group")
        .eq("user_id", test_user)
        .execute()
    )
    assert len(got.data) == 1
    assert got.data[0]["request_count"] == 5


@pytest.mark.asyncio
async def test_middleware_records_authenticated_api_requests(db):
    """Drive the middleware directly with a minimal ASGI app — the real app's
    JWT layer sets request.state.user; here we set it the same way and assert
    the buffer bucketing + skip rules (non-/api paths, missing user, non-UUID
    test users are all skipped)."""
    from fastapi import FastAPI, Request
    from starlette.testclient import TestClient

    from src.api.middleware.activity_tracking import (
        ActivityBuffer,
        ActivityTrackingMiddleware,
    )

    client, test_user = db
    app = FastAPI()
    buf = ActivityBuffer(flush_interval_s=99999, flush_threshold=99999)

    @app.middleware("http")
    async def fake_jwt(request: Request, call_next):
        request.state.user = {"id": test_user, "email": "activity-test@example.invalid"}
        return await call_next(request)

    app.add_middleware(ActivityTrackingMiddleware, buffer=buf)
    # NOTE inverted order vs main.py: add_middleware prepends, so the LAST
    # added (fake_jwt via decorator runs first) — decorator middleware runs
    # outermost here, matching prod where JWT is outer to activity tracking.

    @app.get("/api/causal/estimate")
    async def causal():
        return {"ok": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    tc = TestClient(app)
    tc.get("/api/causal/estimate")
    tc.get("/api/causal/estimate")
    tc.get("/healthz")  # non-/api -> not recorded

    rows = buf.drain()
    assert len(rows) == 1
    assert rows[0]["endpoint_group"] == "causal"
    assert rows[0]["request_count"] == 2
```

- [ ] **Step 5.2: Run to verify FAIL** (no `flush_rows` / `ActivityTrackingMiddleware`).

- [ ] **Step 5.3: Implement** — append to `src/api/middleware/activity_tracking.py`:

```python
async def flush_rows(rows: List[Dict[str, Any]]) -> None:
    """Persist drained rows via the additive-upsert RPC. Fail-open."""
    if not rows:
        return
    try:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
        if client is None:
            return
        await asyncio.to_thread(
            lambda: client.rpc("record_user_activity", {"p_rows": rows}).execute()
        )
    except Exception:
        logger.warning("activity flush failed (fail-open, %d rows lost)", len(rows), exc_info=True)


def _endpoint_group(path: str) -> Optional[str]:
    """'/api/causal/estimate' -> 'causal'. Bounded cardinality by design."""
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "api" and parts[2]:
        return parts[2]
    return None


class ActivityTrackingMiddleware(BaseHTTPMiddleware):
    """Records (user, endpoint_group, minute) for authenticated /api requests.

    Must be INNER to JWTAuthMiddleware (added BEFORE it in main.py — Starlette
    add_middleware prepends, so earlier-added = inner = sees request.state.user).
    Fail-open everywhere; flushes fire-and-forget so requests never wait on DB.
    """

    def __init__(self, app, buffer: Optional[ActivityBuffer] = None):
        super().__init__(app)
        self.buffer = buffer or ActivityBuffer()

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        try:
            user = getattr(request.state, "user", None)
            group = _endpoint_group(request.url.path)
            if user and user.get("id") and group:
                try:
                    uuid.UUID(str(user["id"]))  # skip TESTING_MODE's non-UUID user
                except ValueError:
                    return response
                minute = (
                    datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
                )
                if self.buffer.record(
                    str(user["id"]), user.get("email"), group, request.method, minute
                ):
                    rows = self.buffer.drain()
                    asyncio.get_running_loop().create_task(flush_rows(rows))
        except Exception:
            logger.warning("activity tracking failed (fail-open)", exc_info=True)
        return response
```

- [ ] **Step 5.4: Wire into `src/api/main.py`.** Add the import after line 62 (`from src.api.middleware.auth_middleware import JWTAuthMiddleware`):

```python
from src.api.middleware.activity_tracking import ActivityTrackingMiddleware
```

Insert the middleware registration AFTER the InsightVerifier block (after line ~845 `logger.info("Insight Verifier: ENABLED ...")`) and BEFORE the JWT block at ~line 847 — earlier-added = inner = runs after JWT inbound, so `request.state.user` is populated:

```python
# User activity tracking (admin feature) — bounded per-minute aggregation of
# authenticated /api requests, flushed in the background. Added BEFORE
# JWTAuthMiddleware so it is INNER to it (sees request.state.user).
app.add_middleware(ActivityTrackingMiddleware)
logger.info("Activity Tracking: ENABLED (per-minute buckets -> user_activity_log)")
```

- [ ] **Step 5.5: Run to verify PASS:**

```bash
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_activity_tracking_realdb.py tests/unit/test_activity_buffer.py -p no:cacheprovider -v
```

- [ ] **Step 5.6: Commit**

```bash
git add src/api/middleware/activity_tracking.py src/api/main.py tests/integration/test_activity_tracking_realdb.py
git commit -m "feat(admin): activity tracking middleware — fail-open, bounded, batched RPC flush"
```

---

### Task 6: AdminUserService — client, listing, invite/links

**Files:**
- Create: `src/services/admin_user_service.py`
- Test: `tests/integration/test_admin_user_service_realdb.py` (create)

- [ ] **Step 6.1: Write the failing tests (listing + invite + links):**

```python
"""AdminUserService against the REAL local Supabase stack (no mocks).

Disposable users use the +admsvc email tag; every test cleans up after itself.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_user_service_realdb.py -p no:cacheprovider -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

TAG = "+admsvc"


@pytest.fixture()
def svc():
    from src.services.admin_user_service import AdminUserService

    service = AdminUserService()
    yield service
    # cleanup ALL disposable users this file may have created
    for u in service.admin_client.auth.admin.list_users():
        if u.email and TAG in u.email:
            service.admin_client.auth.admin.delete_user(u.id)


def test_list_users_merges_auth_and_profile(svc):
    users = svc.list_users()
    assert len(users) >= 8  # the 8 real users
    me = next(u for u in users if u["email"] == "etn3724@gmail.com")
    assert me["role"] == "admin"
    assert me["status"] == "active"
    assert me["last_sign_in_at"] is not None
    # profile join fields present (backfilled by migration 101)
    assert "total_messages" in me and "last_active_at" in me


def test_invite_creates_pending_user_with_role_and_link(svc):
    email = f"etn3724{TAG}-inv@gmail.com"
    result = svc.invite_user(email=email, role="analyst", brands=["Kisqali"], full_name="Inv Test")
    assert result["invite_link"].startswith("https://eznomics.site/accept-invite?token_hash=")
    # dual-write landed
    user = next(u for u in svc.list_users() if u["email"] == email)
    assert user["role"] == "analyst"
    assert user["brands"] == ["Kisqali"]
    assert user["status"] == "invited"  # never signed in
    profile = (
        svc.admin_client.table("chatbot_user_profiles")
        .select("role, is_admin")
        .eq("id", user["id"])
        .execute()
    )
    assert profile.data[0]["role"] == "analyst"
    assert profile.data[0]["is_admin"] is False


def test_invite_duplicate_email_raises_conflict(svc):
    from src.services.admin_user_service import AdminConflictError

    email = f"etn3724{TAG}-dup@gmail.com"
    svc.invite_user(email=email, role="viewer", brands=["all"])
    with pytest.raises(AdminConflictError):
        svc.invite_user(email=email, role="viewer", brands=["all"])


def test_invite_link_completes_verify_otp(svc):
    """The returned token_hash must actually work — the whole feature hinges
    on this (proven live 2026-07-11; this test pins it forever)."""
    from urllib.parse import parse_qs, urlparse

    from supabase import create_client

    email = f"etn3724{TAG}-otp@gmail.com"
    result = svc.invite_user(email=email, role="viewer", brands=["all"])
    token_hash = parse_qs(urlparse(result["invite_link"]).query)["token_hash"][0]
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    verified = anon.auth.verify_otp({"type": "invite", "token_hash": token_hash})
    assert verified.session is not None
    anon.auth.update_user({"password": "AdmSvc#2026-otp"})
    anon.auth.sign_out()
    fresh = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    signed = fresh.auth.sign_in_with_password({"email": email, "password": "AdmSvc#2026-otp"})
    assert signed.session is not None


def test_reinvite_pending_reissues_and_active_falls_back_to_recovery(svc):
    email = f"etn3724{TAG}-re@gmail.com"
    first = svc.invite_user(email=email, role="viewer", brands=["all"])
    user_id = next(u["id"] for u in svc.list_users() if u["email"] == email)

    second = svc.reinvite_user(user_id)
    assert second["invite_link"] != first["invite_link"]
    assert second["link_type"] == "invite"

    # activate the user, then reinvite must fall back to recovery
    from urllib.parse import parse_qs, urlparse

    from supabase import create_client

    token_hash = parse_qs(urlparse(second["invite_link"]).query)["token_hash"][0]
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    anon.auth.verify_otp({"type": "invite", "token_hash": token_hash})
    anon.auth.update_user({"password": "AdmSvc#2026-re"})

    third = svc.reinvite_user(user_id)
    assert third["link_type"] == "recovery"


def test_recovery_link_for_active_user(svc):
    me = next(u for u in svc.list_users() if u["email"] == "etn3724@gmail.com")
    link = svc.recovery_link(me["id"])
    assert link["invite_link"].startswith("https://eznomics.site/accept-invite?token_hash=")
    assert link["link_type"] == "recovery"


def test_invalid_role_and_brand_rejected(svc):
    from src.services.admin_user_service import AdminValidationError

    with pytest.raises(AdminValidationError):
        svc.invite_user(email=f"etn3724{TAG}-bad@gmail.com", role="superuser", brands=["all"])
    with pytest.raises(AdminValidationError):
        svc.invite_user(email=f"etn3724{TAG}-bad@gmail.com", role="viewer", brands=["Humira"])
```

- [ ] **Step 6.2: Run to verify FAIL** (`ModuleNotFoundError: src.services.admin_user_service`).

- [ ] **Step 6.3: Implement** `src/services/admin_user_service.py`:

```python
"""Admin user management over Supabase GoTrue (spec 2026-07-11).

Server-only: holds a SERVICE-ROLE client. Never import from frontend-reachable
code paths without require_admin in front.

Design facts this code encodes (all live-verified 2026-07-11 — see spec):
- Invite = generate_link(type='invite') -> hashed_token -> our own URL on
  E2I_PUBLIC_APP_URL (GoTrue SMTP is fake; GOTRUE_URI_ALLOW_LIST excludes the
  site, so we never use action_link).
- Reinvite: pending -> invite-type reissues; active -> GoTrue rejects ->
  recovery-type link.
- Disable = ban (blocks sign-in/refresh) + app_metadata.disabled=true
  (immediate API lockout via verify_supabase_token). banned_until is not
  readable through gotrue-py, so app_metadata.disabled IS the status flag.
- Role dual-write: app_metadata (API-authoritative) + chatbot_user_profiles
  (RLS). user_profiles does not exist in prod.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from supabase import create_client

logger = logging.getLogger(__name__)

VALID_ROLES = ("viewer", "analyst", "operator", "admin")
VALID_BRANDS = ("Kisqali", "Fabhalta", "Remibrutinib", "all")
BAN_DURATION = "876000h"  # ~100 years
PUBLIC_APP_URL = os.environ.get("E2I_PUBLIC_APP_URL", "https://eznomics.site")


class AdminServiceError(Exception):
    """Base error; routes map subclasses to HTTP statuses."""


class AdminValidationError(AdminServiceError):
    """Invalid role/brand/input -> 422."""


class AdminConflictError(AdminServiceError):
    """Duplicate email / GoTrue state conflict -> 409."""


class AdminGuardError(AdminServiceError):
    """Self-targeting or last-admin protection -> 403."""


class AdminNotFoundError(AdminServiceError):
    """Unknown user id -> 404."""


def _validate(role: str, brands: List[str]) -> None:
    if role not in VALID_ROLES:
        raise AdminValidationError(f"invalid role {role!r}; must be one of {VALID_ROLES}")
    if not brands:
        raise AdminValidationError("brands must be non-empty (use ['all'] for cross-brand)")
    for b in brands:
        if b not in VALID_BRANDS:
            raise AdminValidationError(f"invalid brand {b!r}; must be one of {VALID_BRANDS}")


class AdminUserService:
    """All GoTrue admin + profile dual-write operations."""

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or os.environ.get(
            "SUPABASE_SERVICE_KEY", ""
        )
        if not url or not key:
            raise AdminServiceError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")
        self.admin_client = create_client(url, key)

    # ------------------------------------------------------------------ users

    def _get_auth_user(self, user_id: str):
        try:
            return self.admin_client.auth.admin.get_user_by_id(user_id).user
        except Exception as e:
            raise AdminNotFoundError(f"user {user_id} not found: {e}") from e

    @staticmethod
    def _status(user) -> str:
        meta = user.app_metadata or {}
        if meta.get("disabled"):
            return "disabled"
        if user.last_sign_in_at is None:
            return "invited"
        return "active"

    def list_users(self) -> List[Dict[str, Any]]:
        auth_users = self.admin_client.auth.admin.list_users(page=1, per_page=1000)
        profiles = (
            self.admin_client.table("chatbot_user_profiles")
            .select("id, role, total_conversations, total_messages, last_active_at")
            .execute()
        )
        by_id = {p["id"]: p for p in profiles.data}
        out = []
        for u in auth_users:
            meta = u.app_metadata or {}
            p = by_id.get(u.id, {})
            out.append(
                {
                    "id": u.id,
                    "email": u.email,
                    "full_name": (u.user_metadata or {}).get("full_name"),
                    "role": meta.get("role") or p.get("role") or "viewer",
                    "brands": meta.get("brands") or [],
                    "status": self._status(u),
                    "created_at": str(u.created_at) if u.created_at else None,
                    "last_sign_in_at": str(u.last_sign_in_at) if u.last_sign_in_at else None,
                    "total_conversations": p.get("total_conversations") or 0,
                    "total_messages": p.get("total_messages") or 0,
                    "last_active_at": p.get("last_active_at"),
                }
            )
        return sorted(out, key=lambda x: x["created_at"] or "", reverse=True)

    # ---------------------------------------------------------------- profile

    def _upsert_profile(self, user_id: str, email: str, role: str) -> None:
        self.admin_client.table("chatbot_user_profiles").upsert(
            {
                "id": user_id,
                "email": email,
                "role": role,
                "is_admin": role == "admin",
            },
            on_conflict="id",
        ).execute()

    # ----------------------------------------------------------------- invite

    @staticmethod
    def _accept_link(hashed_token: str) -> str:
        return f"{PUBLIC_APP_URL}/accept-invite?token_hash={quote(hashed_token)}"

    def invite_user(
        self,
        email: str,
        role: str,
        brands: List[str],
        full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        _validate(role, brands)
        try:
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "invite", "email": email}
            )
        except Exception as e:
            if "already been registered" in str(e):
                raise AdminConflictError(f"{email} is already registered") from e
            raise
        user = resp.user
        attrs: Dict[str, Any] = {"app_metadata": {"role": role, "brands": brands}}
        if full_name:
            attrs["user_metadata"] = {"full_name": full_name}
        self.admin_client.auth.admin.update_user_by_id(user.id, attrs)
        self._upsert_profile(user.id, email, role)
        return {
            "user_id": user.id,
            "email": email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": "invite",
        }

    def reinvite_user(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        try:
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "invite", "email": user.email}
            )
            link_type = "invite"
        except Exception as e:
            if "already been registered" not in str(e):
                raise
            resp = self.admin_client.auth.admin.generate_link(
                {"type": "recovery", "email": user.email}
            )
            link_type = "recovery"
        return {
            "user_id": user.id,
            "email": user.email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": link_type,
        }

    def recovery_link(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        resp = self.admin_client.auth.admin.generate_link(
            {"type": "recovery", "email": user.email}
        )
        return {
            "user_id": user.id,
            "email": user.email,
            "invite_link": self._accept_link(resp.properties.hashed_token),
            "link_type": "recovery",
        }
```

- [ ] **Step 6.4: Run to verify PASS:**

```bash
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_user_service_realdb.py -p no:cacheprovider -v
```

- [ ] **Step 6.5: Commit**

```bash
git add src/services/admin_user_service.py tests/integration/test_admin_user_service_realdb.py
git commit -m "feat(admin): AdminUserService — merged listing, invite/reinvite/recovery links"
```

---

### Task 7: AdminUserService — role update, disable/enable, delete guards, activity, reconcile

**Files:**
- Modify: `src/services/admin_user_service.py` (append methods)
- Modify: `tests/integration/test_admin_user_service_realdb.py` (append tests)

- [ ] **Step 7.1: Append the failing tests:**

```python
def _mk(svc, suffix, role="viewer", password="AdmSvc#2026-x"):
    """Create + activate a disposable user, return (id, email)."""
    email = f"etn3724{TAG}-{suffix}@gmail.com"
    created = svc.admin_client.auth.admin.create_user(
        {
            "email": email,
            "password": password,
            "email_confirm": True,
            "app_metadata": {"role": role, "brands": ["all"]},
        }
    )
    svc._upsert_profile(created.user.id, email, role)
    return created.user.id, email


def test_update_user_dual_writes_role(svc):
    uid, email = _mk(svc, "upd")
    svc.update_user(uid, role="operator", brands=["Fabhalta"], acting_admin_id="not-the-target")
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("role") == "operator"
    assert (u.app_metadata or {}).get("brands") == ["Fabhalta"]
    p = svc.admin_client.table("chatbot_user_profiles").select("role, is_admin").eq("id", uid).execute()
    assert p.data[0]["role"] == "operator" and p.data[0]["is_admin"] is False


def test_disable_sets_flag_and_blocks_signin_enable_reverses(svc):
    from supabase import create_client

    uid, email = _mk(svc, "dis")
    svc.disable_user(uid, acting_admin_id="not-the-target")
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("disabled") is True
    anon = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"],
    )
    with pytest.raises(Exception, match="[Bb]anned"):
        anon.auth.sign_in_with_password({"email": email, "password": "AdmSvc#2026-x"})

    svc.enable_user(uid)
    u = svc._get_auth_user(uid)
    assert not (u.app_metadata or {}).get("disabled")
    signed = anon.auth.sign_in_with_password({"email": email, "password": "AdmSvc#2026-x"})
    assert signed.session is not None


def test_delete_removes_user_and_cascades_profile(svc):
    uid, _ = _mk(svc, "del")
    svc.delete_user(uid, acting_admin_id="not-the-target")
    from src.services.admin_user_service import AdminNotFoundError

    with pytest.raises(AdminNotFoundError):
        svc._get_auth_user(uid)
    p = svc.admin_client.table("chatbot_user_profiles").select("id").eq("id", uid).execute()
    assert p.data == []  # ON DELETE CASCADE


def test_self_targeting_guards(svc):
    from src.services.admin_user_service import AdminGuardError

    uid, _ = _mk(svc, "self", role="admin")
    with pytest.raises(AdminGuardError):
        svc.delete_user(uid, acting_admin_id=uid)
    with pytest.raises(AdminGuardError):
        svc.disable_user(uid, acting_admin_id=uid)


def test_last_admin_guards(svc):
    """Deleting/demoting/disabling the LAST enabled admin is refused. Uses only
    disposable admins: temporarily demote is simulated by counting — the real
    admins (>=1: etn3724) always exist, so we assert the guard math on a world
    where our disposable admin is NOT the last one (guard passes), then verify
    the counting helper directly."""
    from src.services.admin_user_service import AdminGuardError

    uid, _ = _mk(svc, "lastadm", role="admin")
    # There are >=2 admins now (real etn3724 + disposable) -> delete allowed
    svc.delete_user(uid, acting_admin_id="not-the-target")

    # Guard math: with exactly the real admin left, deleting THAT admin must
    # be refused. We do NOT touch the real admin — assert via the counter +
    # a direct guard call.
    admins = svc._enabled_admin_ids()
    assert len(admins) >= 1
    if len(admins) == 1:
        with pytest.raises(AdminGuardError):
            svc._guard_not_last_admin(admins[0], "delete")


def test_activity_readers_return_real_history(svc):
    # Platform: real auth.audit_log_entries exist since 2026-02 (login events)
    platform = svc.platform_activity(days=365)
    assert platform["days"]  # non-empty
    assert any(d["logins"] > 0 for d in platform["days"])

    # Per-user: the real admin has login history
    me = next(u for u in svc.list_users() if u["email"] == "etn3724@gmail.com")
    activity = svc.user_activity(me["id"], days=365)
    assert any(d["event_type"] == "login" and d["event_count"] > 0 for d in activity["auth_events"])
    assert isinstance(activity["api_activity"], list)
    assert isinstance(activity["recent_events"], list)
    assert activity["chat"]["total_messages"] >= 0


def test_reconcile_role_stores(svc):
    """Users with NULL jwt role but a profile role get app_metadata backfilled."""
    uid, email = _mk(svc, "recon")
    # strip the jwt role to simulate legacy drift
    svc.admin_client.auth.admin.update_user_by_id(
        uid, {"app_metadata": {"role": None, "brands": None}}
    )
    svc.admin_client.table("chatbot_user_profiles").update({"role": "analyst"}).eq("id", uid).execute()

    report = svc.reconcile_role_stores()
    assert any(r["user_id"] == uid and r["action"] == "app_metadata_backfilled" for r in report)
    u = svc._get_auth_user(uid)
    assert (u.app_metadata or {}).get("role") == "analyst"
```

- [ ] **Step 7.2: Run to verify FAIL** (missing methods).

- [ ] **Step 7.3: Implement** — append to `AdminUserService`:

```python
    # ----------------------------------------------------------------- guards

    def _enabled_admin_ids(self) -> List[str]:
        return [
            u.id
            for u in self.admin_client.auth.admin.list_users(page=1, per_page=1000)
            if (u.app_metadata or {}).get("role") == "admin"
            and not (u.app_metadata or {}).get("disabled")
        ]

    def _guard_not_self(self, user_id: str, acting_admin_id: str, action: str) -> None:
        if user_id == acting_admin_id:
            raise AdminGuardError(f"admins cannot {action} their own account")

    def _guard_not_last_admin(self, user_id: str, action: str) -> None:
        admins = self._enabled_admin_ids()
        if user_id in admins and len(admins) <= 1:
            raise AdminGuardError(f"cannot {action} the last enabled admin")

    # ------------------------------------------------------------------ write

    def update_user(
        self,
        user_id: str,
        acting_admin_id: str,
        role: Optional[str] = None,
        brands: Optional[List[str]] = None,
        full_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        new_role = role or meta.get("role") or "viewer"
        new_brands = brands if brands is not None else (meta.get("brands") or ["all"])
        _validate(new_role, new_brands)
        if meta.get("role") == "admin" and new_role != "admin":
            self._guard_not_last_admin(user_id, "demote")
        attrs: Dict[str, Any] = {
            "app_metadata": {**meta, "role": new_role, "brands": new_brands}
        }
        if full_name is not None:
            attrs["user_metadata"] = {**(user.user_metadata or {}), "full_name": full_name}
        self.admin_client.auth.admin.update_user_by_id(user_id, attrs)
        self._upsert_profile(user_id, user.email, new_role)
        return {"user_id": user_id, "role": new_role, "brands": new_brands}

    def disable_user(self, user_id: str, acting_admin_id: str) -> Dict[str, Any]:
        self._guard_not_self(user_id, acting_admin_id, "disable")
        self._guard_not_last_admin(user_id, "disable")
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        meta["disabled"] = True
        self.admin_client.auth.admin.update_user_by_id(
            user_id, {"app_metadata": meta, "ban_duration": BAN_DURATION}
        )
        return {"user_id": user_id, "status": "disabled"}

    def enable_user(self, user_id: str) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        meta = dict(user.app_metadata or {})
        meta["disabled"] = False
        self.admin_client.auth.admin.update_user_by_id(
            user_id, {"app_metadata": meta, "ban_duration": "none"}
        )
        return {"user_id": user_id, "status": "active" if user.last_sign_in_at else "invited"}

    def delete_user(self, user_id: str, acting_admin_id: str) -> Dict[str, Any]:
        self._guard_not_self(user_id, acting_admin_id, "delete")
        self._guard_not_last_admin(user_id, "delete")
        user = self._get_auth_user(user_id)  # 404 before delete
        self.admin_client.auth.admin.delete_user(user_id)
        return {"user_id": user_id, "email": user.email, "deleted": True}

    # --------------------------------------------------------------- activity

    def user_activity(self, user_id: str, days: int = 90) -> Dict[str, Any]:
        user = self._get_auth_user(user_id)
        auth_events = self.admin_client.rpc(
            "admin_get_login_activity", {"p_user_id": user_id, "p_days": days}
        ).execute()
        api_rows = (
            self.admin_client.table("user_activity_log")
            .select("endpoint_group, http_method, bucket_minute, request_count")
            .eq("user_id", user_id)
            .order("bucket_minute", desc=True)
            .limit(5000)
            .execute()
        )
        recent = self.admin_client.rpc(
            "admin_get_user_recent_events", {"p_user_id": user_id, "p_limit": 25}
        ).execute()
        profile = (
            self.admin_client.table("chatbot_user_profiles")
            .select("total_conversations, total_messages, last_active_at")
            .eq("id", user_id)
            .execute()
        )
        chat = profile.data[0] if profile.data else {
            "total_conversations": 0,
            "total_messages": 0,
            "last_active_at": None,
        }
        return {
            "user_id": user_id,
            "email": user.email,
            "auth_events": auth_events.data or [],
            "api_activity": api_rows.data or [],
            "recent_events": recent.data or [],
            "chat": chat,
        }

    def platform_activity(self, days: int = 30) -> Dict[str, Any]:
        rows = self.admin_client.rpc(
            "admin_get_platform_activity", {"p_days": days}
        ).execute()
        return {"days": rows.data or []}

    # -------------------------------------------------------------- reconcile

    def reconcile_role_stores(self) -> List[Dict[str, Any]]:
        """One-time drift repair: users with NO jwt role but a profile role get
        app_metadata backfilled from the profile (migration 101 already synced
        the other direction, jwt -> profile)."""
        report: List[Dict[str, Any]] = []
        profiles = (
            self.admin_client.table("chatbot_user_profiles")
            .select("id, role")
            .execute()
        )
        by_id = {p["id"]: p.get("role") for p in profiles.data}
        for u in self.admin_client.auth.admin.list_users(page=1, per_page=1000):
            meta = dict(u.app_metadata or {})
            if not meta.get("role") and by_id.get(u.id):
                meta["role"] = by_id[u.id]
                meta.setdefault("brands", ["all"])
                self.admin_client.auth.admin.update_user_by_id(
                    u.id, {"app_metadata": meta}
                )
                report.append({"user_id": u.id, "action": "app_metadata_backfilled",
                               "role": meta["role"]})
        return report
```

- [ ] **Step 7.4: Run to verify PASS** (same test command as 6.4).

- [ ] **Step 7.5: Commit**

```bash
git add src/services/admin_user_service.py tests/integration/test_admin_user_service_realdb.py
git commit -m "feat(admin): role dual-write, disable/enable, guarded delete, activity readers, reconcile"
```

---

### Task 8: Admin API routes + audit events + main.py wiring

**Files:**
- Create: `src/api/routes/admin.py`
- Test: `tests/integration/test_admin_routes_realdb.py` (create)
- Modify: `src/api/main.py` (router import alphabetically after `agents` ~line 70; include after the audit_router include ~line 1154)

- [ ] **Step 8.1: Write the failing tests** — REAL JWTs via real sign-ins; the full app (real middleware chain):

```python
"""Admin routes against the REAL app + REAL Supabase (no mocks, real JWTs).

A disposable viewer and a disposable admin are created via the service-role
client and signed in via the anon client — their tokens exercise the actual
RBAC path (JWTAuthMiddleware -> require_admin).

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_routes_realdb.py -p no:cacheprovider -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

TAG = "+admroute"
PASSWORD = "AdmRoute#2026-x"


@pytest.fixture(scope="module")
def harness():
    # Real-auth harness: TESTING_MODE must be OFF or the middleware injects a
    # mock admin and the RBAC assertions are meaningless.
    assert os.environ.get("E2I_TESTING_MODE", "").lower() not in ("1", "true"), (
        "unset E2I_TESTING_MODE for real-auth route tests"
    )
    from fastapi.testclient import TestClient
    from supabase import create_client

    from src.api.main import app

    url = os.environ["SUPABASE_URL"]
    admin_client = create_client(
        url, os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"]
    )
    anon_key = os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_KEY"]

    def make_user(suffix, role):
        email = f"etn3724{TAG}-{suffix}@gmail.com"
        for u in admin_client.auth.admin.list_users():
            if u.email == email:
                admin_client.auth.admin.delete_user(u.id)
        created = admin_client.auth.admin.create_user(
            {
                "email": email,
                "password": PASSWORD,
                "email_confirm": True,
                "app_metadata": {"role": role, "brands": ["all"]},
            }
        )
        anon = create_client(url, anon_key)
        session = anon.auth.sign_in_with_password({"email": email, "password": PASSWORD})
        return created.user.id, session.session.access_token

    admin_id, admin_token = make_user("admin", "admin")
    viewer_id, viewer_token = make_user("viewer", "viewer")

    yield TestClient(app), admin_token, viewer_token, admin_client, admin_id

    for u in admin_client.auth.admin.list_users():
        if u.email and TAG in u.email:
            admin_client.auth.admin.delete_user(u.id)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_viewer_gets_403_admin_gets_200(harness):
    client, admin_token, viewer_token, *_ = harness
    assert client.get("/api/admin/users", headers=_auth(viewer_token)).status_code == 403
    assert client.get("/api/admin/users").status_code == 401  # no token
    resp = client.get("/api/admin/users", headers=_auth(admin_token))
    assert resp.status_code == 200
    assert any(u["email"] == "etn3724@gmail.com" for u in resp.json()["users"])


def test_invite_flow_and_conflict(harness):
    client, admin_token, *_ = harness
    body = {"email": f"etn3724{TAG}-inv@gmail.com", "role": "viewer", "brands": ["Kisqali"]}
    resp = client.post("/api/admin/users/invite", json=body, headers=_auth(admin_token))
    assert resp.status_code == 200
    assert resp.json()["invite_link"].startswith("https://eznomics.site/accept-invite?")
    dup = client.post("/api/admin/users/invite", json=body, headers=_auth(admin_token))
    assert dup.status_code == 409
    bad = client.post(
        "/api/admin/users/invite",
        json={"email": "x@y.z", "role": "root", "brands": ["all"]},
        headers=_auth(admin_token),
    )
    assert bad.status_code == 422


def test_role_update_disable_enable_delete(harness):
    client, admin_token, _, admin_client, _ = harness
    invited = client.post(
        "/api/admin/users/invite",
        json={"email": f"etn3724{TAG}-lc@gmail.com", "role": "viewer", "brands": ["all"]},
        headers=_auth(admin_token),
    ).json()
    uid = invited["user_id"]

    assert (
        client.patch(
            f"/api/admin/users/{uid}", json={"role": "analyst"}, headers=_auth(admin_token)
        ).status_code
        == 200
    )
    assert (
        client.post(f"/api/admin/users/{uid}/disable", headers=_auth(admin_token)).status_code
        == 200
    )
    assert (
        client.post(f"/api/admin/users/{uid}/enable", headers=_auth(admin_token)).status_code
        == 200
    )
    assert (
        client.delete(f"/api/admin/users/{uid}", headers=_auth(admin_token)).status_code == 200
    )


def test_self_delete_forbidden(harness):
    client, admin_token, _, _, admin_id = harness
    resp = client.delete(f"/api/admin/users/{admin_id}", headers=_auth(admin_token))
    assert resp.status_code == 403


def test_activity_endpoints(harness):
    client, admin_token, *_ = harness
    overview = client.get(
        "/api/admin/activity/overview?days=365", headers=_auth(admin_token)
    )
    assert overview.status_code == 200
    assert any(d["logins"] > 0 for d in overview.json()["days"])

    users = client.get("/api/admin/users", headers=_auth(admin_token)).json()["users"]
    me = next(u for u in users if u["email"] == "etn3724@gmail.com")
    activity = client.get(
        f"/api/admin/users/{me['id']}/activity?days=365", headers=_auth(admin_token)
    )
    assert activity.status_code == 200
    assert activity.json()["auth_events"]


def test_admin_actions_are_audited(harness):
    client, admin_token, _, admin_client, _ = harness
    email = f"etn3724{TAG}-aud@gmail.com"
    invited = client.post(
        "/api/admin/users/invite",
        json={"email": email, "role": "viewer", "brands": ["all"]},
        headers=_auth(admin_token),
    ).json()
    rows = (
        admin_client.table("security_audit_log")
        .select("event_type, resource_id")
        .eq("resource_id", invited["user_id"])
        .execute()
    )
    assert any(r["event_type"] == "admin.user.modified" for r in rows.data)
    admin_client.table("security_audit_log").delete().eq(
        "resource_id", invited["user_id"]
    ).execute()


def test_audit_feed_endpoint(harness):
    client, admin_token, *_ = harness
    resp = client.get("/api/admin/audit?days=30", headers=_auth(admin_token))
    assert resp.status_code == 200
    assert isinstance(resp.json()["events"], list)
```

- [ ] **Step 8.2: Run to verify FAIL** (404s — router absent).

- [ ] **Step 8.3: Implement** `src/api/routes/admin.py`:

```python
"""Admin user management endpoints (spec 2026-07-11). ALL endpoints require
the admin role via require_admin; the router is NOT in PUBLIC_PATHS, so the
JWT middleware gates it before RBAC even runs. Every mutation is audited to
security_audit_log via the (now-fixed) SecurityAuditService."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr, Field

from src.api.dependencies.auth import require_admin
from src.services.admin_user_service import (
    AdminConflictError,
    AdminGuardError,
    AdminNotFoundError,
    AdminServiceError,
    AdminUserService,
    AdminValidationError,
)
from src.utils.security_audit import (
    SecurityAuditEvent,
    SecurityEventSeverity,
    SecurityEventType,
    get_security_audit_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

_service: Optional[AdminUserService] = None


def get_admin_service() -> AdminUserService:
    global _service
    if _service is None:
        _service = AdminUserService()
    return _service


def _audit(
    event_type: SecurityEventType,
    admin: Dict[str, Any],
    request: Request,
    message: str,
    target_user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit-log an admin action. Never blocks the action (existing convention)."""
    try:
        get_security_audit_service().log_event(
            SecurityAuditEvent(
                event_id=uuid4(),
                event_type=event_type,
                severity=SecurityEventSeverity.WARNING,
                timestamp=datetime.now(timezone.utc),
                message=message,
                user_id=str(admin.get("id")),
                user_email=admin.get("email"),
                client_ip=request.headers.get("x-real-ip") or (
                    request.client.host if request.client else None
                ),
                endpoint=str(request.url.path),
                http_method=request.method,
                resource_type="auth_user",
                resource_id=target_user_id,
                action_result="success",
                metadata=metadata or {},
            )
        )
    except Exception:
        logger.warning("admin audit logging failed (non-blocking)", exc_info=True)


def _map_error(e: AdminServiceError) -> HTTPException:
    if isinstance(e, AdminValidationError):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, AdminConflictError):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, AdminGuardError):
        return HTTPException(status_code=403, detail=str(e))
    if isinstance(e, AdminNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=502, detail=f"auth service error: {e}")


# ------------------------------------------------------------------- schemas


class InviteRequest(BaseModel):
    email: EmailStr
    role: str = "viewer"
    brands: List[str] = Field(default_factory=lambda: ["all"])
    full_name: Optional[str] = None


class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    brands: Optional[List[str]] = None
    full_name: Optional[str] = None


class UsersResponse(BaseModel):
    users: List[Dict[str, Any]]


class LinkResponse(BaseModel):
    user_id: str
    email: str
    invite_link: str
    link_type: str


# ----------------------------------------------------------------- endpoints


@router.get("/users", response_model=UsersResponse)
async def list_users(
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> UsersResponse:
    users = await asyncio.to_thread(service.list_users)
    return UsersResponse(users=users)


@router.post("/users/invite", response_model=LinkResponse)
async def invite_user(
    body: InviteRequest,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(
            service.invite_user, body.email, body.role, body.brands, body.full_name
        )
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Invited {body.email} as {body.role}",
        target_user_id=result["user_id"],
        metadata={"action": "invite", "role": body.role, "brands": body.brands},
    )
    return LinkResponse(**result)


@router.post("/users/{user_id}/reinvite", response_model=LinkResponse)
async def reinvite_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(service.reinvite_user, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Re-invited {result['email']} ({result['link_type']} link)",
        target_user_id=user_id,
        metadata={"action": "reinvite", "link_type": result["link_type"]},
    )
    return LinkResponse(**result)


@router.post("/users/{user_id}/recovery-link", response_model=LinkResponse)
async def recovery_link(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> LinkResponse:
    try:
        result = await asyncio.to_thread(service.recovery_link, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"Generated recovery link for {result['email']}",
        target_user_id=user_id,
        metadata={"action": "recovery_link"},
    )
    return LinkResponse(**result)


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            service.update_user,
            user_id,
            str(admin.get("id")),
            body.role,
            body.brands,
            body.full_name,
        )
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_ROLE_CHANGE,
        admin,
        request,
        f"Updated user {user_id}: role={result['role']} brands={result['brands']}",
        target_user_id=user_id,
        metadata={"action": "update", **result},
    )
    return result


@router.post("/users/{user_id}/disable")
async def disable_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.disable_user, user_id, str(admin.get("id")))
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_PERMISSION_REVOKED,
        admin,
        request,
        f"Disabled user {user_id}",
        target_user_id=user_id,
        metadata={"action": "disable"},
    )
    return result


@router.post("/users/{user_id}/enable")
async def enable_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.enable_user, user_id)
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.AUTHZ_PERMISSION_GRANTED,
        admin,
        request,
        f"Enabled user {user_id}",
        target_user_id=user_id,
        metadata={"action": "enable"},
    )
    return result


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(service.delete_user, user_id, str(admin.get("id")))
    except AdminServiceError as e:
        raise _map_error(e) from e
    _audit(
        SecurityEventType.ADMIN_USER_MODIFIED,
        admin,
        request,
        f"DELETED user {result['email']}",
        target_user_id=user_id,
        metadata={"action": "delete", "email": result["email"]},
    )
    return result


@router.get("/users/{user_id}/activity")
async def user_activity(
    user_id: str,
    days: int = Query(default=90, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(service.user_activity, user_id, days)
    except AdminServiceError as e:
        raise _map_error(e) from e


@router.get("/activity/overview")
async def activity_overview(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    return await asyncio.to_thread(service.platform_activity, days)


@router.get("/audit")
async def admin_audit_feed(
    days: int = Query(default=30, ge=1, le=365),
    admin: Dict[str, Any] = Depends(require_admin),
    service: AdminUserService = Depends(get_admin_service),
) -> Dict[str, Any]:
    def _query() -> List[Dict[str, Any]]:
        from datetime import timedelta

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = (
            service.admin_client.table("security_audit_log")
            .select(
                "event_id, event_type, severity, timestamp, message, "
                "user_email, resource_id, metadata"
            )
            .gte("timestamp", since)
            .order("timestamp", desc=True)
            .limit(200)
            .execute()
        )
        return rows.data or []

    return {"events": await asyncio.to_thread(_query)}
```

- [ ] **Step 8.4: Wire into `src/api/main.py`.** Import (alphabetical, after the `agents` import at line 70):

```python
from src.api.routes.admin import router as admin_router
```

Include (after the audit_router include at ~line 1154):

```python
# Admin user management endpoints (/api/admin/*) — require_admin on every route
app.include_router(admin_router, prefix="/api")
```

- [ ] **Step 8.5: Run to verify PASS** (watch memory — `free -m` before/after; the app import is heavy):

```bash
free -m
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_admin_routes_realdb.py -p no:cacheprovider -v
free -m
```

- [ ] **Step 8.6: Commit**

```bash
git add src/api/routes/admin.py src/api/main.py tests/integration/test_admin_routes_realdb.py
git commit -m "feat(admin): /api/admin router — users CRUD, links, activity, audited mutations"
```

---

### Task 9: Backend quality checkpoint

**Files:** none (verification)

- [ ] **Step 9.1:** Scoped lint + types (changed files ONLY — droplet policy):

```bash
.venv/bin/ruff check src/api/routes/admin.py src/services/admin_user_service.py src/api/middleware/activity_tracking.py src/utils/security_audit.py src/api/dependencies/auth.py src/api/main.py
.venv/bin/mypy --config-file pyproject.toml src/api/routes/admin.py src/services/admin_user_service.py src/api/middleware/activity_tracking.py
```

Expected: clean (fix anything reported; CI's full gates are the arbiter later).

- [ ] **Step 9.2:** Full targeted backend suite once more, then commit any fixes:

```bash
E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_security_audit_sink_realdb.py tests/integration/test_auth_disabled_flag_realdb.py tests/integration/test_activity_tracking_realdb.py tests/integration/test_admin_user_service_realdb.py tests/integration/test_admin_routes_realdb.py tests/unit/test_activity_buffer.py -p no:cacheprovider -q
```

- [ ] **Step 9.3:** Confirm zero disposable users remain: `docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT email FROM auth.users WHERE email LIKE '%+adm%'"` → empty.

---

### Task 10: Frontend API layer (types, client functions, query keys, hooks)

**Files:**
- Create: `frontend/src/types/admin.ts`
- Create: `frontend/src/api/admin.ts`
- Create: `frontend/src/hooks/api/use-admin.ts`
- Modify: `frontend/src/lib/query-client.ts` (add `admin` key group inside `queryKeys`)
- Modify: `frontend/src/hooks/api/index.ts` (export the new hooks)

- [ ] **Step 10.1:** `frontend/src/types/admin.ts`:

```typescript
/** Admin user management types (mirrors /api/admin/* responses). */

export type AdminRole = 'viewer' | 'analyst' | 'operator' | 'admin';
export type AdminUserStatus = 'active' | 'invited' | 'disabled';
export type AdminBrand = 'Kisqali' | 'Fabhalta' | 'Remibrutinib' | 'all';

export interface AdminUser {
  id: string;
  email: string;
  full_name: string | null;
  role: AdminRole;
  brands: string[];
  status: AdminUserStatus;
  created_at: string | null;
  last_sign_in_at: string | null;
  total_conversations: number;
  total_messages: number;
  last_active_at: string | null;
}

export interface AdminUsersResponse {
  users: AdminUser[];
}

export interface InviteRequest {
  email: string;
  role: AdminRole;
  brands: string[];
  full_name?: string;
}

export interface LinkResponse {
  user_id: string;
  email: string;
  invite_link: string;
  link_type: 'invite' | 'recovery';
}

export interface UpdateUserRequest {
  role?: AdminRole;
  brands?: string[];
  full_name?: string;
}

export interface AuthEventBucket {
  day: string;
  event_type: string;
  event_count: number;
}

export interface ApiActivityRow {
  endpoint_group: string;
  http_method: string;
  bucket_minute: string;
  request_count: number;
}

export interface UserActivityResponse {
  user_id: string;
  email: string;
  auth_events: AuthEventBucket[];
  api_activity: ApiActivityRow[];
  recent_events: { occurred_at: string; action: string }[];
  chat: {
    total_conversations: number;
    total_messages: number;
    last_active_at: string | null;
  };
}

export interface PlatformActivityResponse {
  days: { day: string; logins: number; active_users: number }[];
}

export interface AuditFeedResponse {
  events: {
    event_id: string;
    event_type: string;
    severity: string;
    timestamp: string;
    message: string;
    user_email: string | null;
    resource_id: string | null;
    metadata: Record<string, unknown>;
  }[];
}
```

- [ ] **Step 10.2:** `frontend/src/api/admin.ts`:

```typescript
/**
 * Admin API Client (/api/admin/*)
 * All endpoints require an admin JWT — the shared apiClient attaches it.
 * @module api/admin
 */

import { get, post, patch, del } from '@/lib/api-client';
import type {
  AdminUsersResponse,
  AuditFeedResponse,
  InviteRequest,
  LinkResponse,
  PlatformActivityResponse,
  UpdateUserRequest,
  UserActivityResponse,
} from '@/types/admin';

const BASE = '/admin';

export function listUsers(): Promise<AdminUsersResponse> {
  return get<AdminUsersResponse>(`${BASE}/users`);
}

export function inviteUser(body: InviteRequest): Promise<LinkResponse> {
  return post<LinkResponse, InviteRequest>(`${BASE}/users/invite`, body);
}

export function reinviteUser(userId: string): Promise<LinkResponse> {
  return post<LinkResponse>(`${BASE}/users/${userId}/reinvite`);
}

export function recoveryLink(userId: string): Promise<LinkResponse> {
  return post<LinkResponse>(`${BASE}/users/${userId}/recovery-link`);
}

export function updateUser(
  userId: string,
  body: UpdateUserRequest
): Promise<{ user_id: string; role: string; brands: string[] }> {
  return patch(`${BASE}/users/${userId}`, body);
}

export function disableUser(userId: string): Promise<{ user_id: string; status: string }> {
  return post(`${BASE}/users/${userId}/disable`);
}

export function enableUser(userId: string): Promise<{ user_id: string; status: string }> {
  return post(`${BASE}/users/${userId}/enable`);
}

export function deleteUser(
  userId: string
): Promise<{ user_id: string; email: string; deleted: boolean }> {
  return del(`${BASE}/users/${userId}`);
}

export function getUserActivity(userId: string, days = 90): Promise<UserActivityResponse> {
  return get<UserActivityResponse>(`${BASE}/users/${userId}/activity?days=${days}`);
}

export function getPlatformActivity(days = 30): Promise<PlatformActivityResponse> {
  return get<PlatformActivityResponse>(`${BASE}/activity/overview?days=${days}`);
}

export function getAuditFeed(days = 30): Promise<AuditFeedResponse> {
  return get<AuditFeedResponse>(`${BASE}/audit?days=${days}`);
}
```

(If `post`/`del` signatures differ on optional body — check `frontend/src/lib/api-client.ts:351/433` overloads — adjust the no-body calls to `post(url, undefined)` as required by the actual types.)

- [ ] **Step 10.3:** Add to `queryKeys` in `frontend/src/lib/query-client.ts` (same shape as the existing `audit` group at line 543):

```typescript
  /**
   * Admin user management queries
   */
  admin: {
    all: () => [...queryKeys.all, 'admin'] as const,
    users: () => [...queryKeys.admin.all(), 'users'] as const,
    userActivity: (userId: string, days: number) =>
      [...queryKeys.admin.all(), 'activity', userId, days] as const,
    platformActivity: (days: number) =>
      [...queryKeys.admin.all(), 'platform-activity', days] as const,
    auditFeed: (days: number) =>
      [...queryKeys.admin.all(), 'audit-feed', days] as const,
  },
```

- [ ] **Step 10.4:** `frontend/src/hooks/api/use-admin.ts`:

```typescript
/**
 * Admin React Query hooks (/api/admin/*).
 * Mutations invalidate the users list — no optimistic updates (spec: refetch
 * after mutation).
 * @module hooks/api/use-admin
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  deleteUser,
  disableUser,
  enableUser,
  getAuditFeed,
  getPlatformActivity,
  getUserActivity,
  inviteUser,
  listUsers,
  recoveryLink,
  reinviteUser,
  updateUser,
} from '@/api/admin';
import type { InviteRequest, UpdateUserRequest } from '@/types/admin';

export function useAdminUsers() {
  return useQuery({
    queryKey: queryKeys.admin.users(),
    queryFn: listUsers,
  });
}

export function useUserActivity(userId: string | null, days = 90) {
  return useQuery({
    queryKey: queryKeys.admin.userActivity(userId ?? 'none', days),
    queryFn: () => getUserActivity(userId as string, days),
    enabled: Boolean(userId),
  });
}

export function usePlatformActivity(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.platformActivity(days),
    queryFn: () => getPlatformActivity(days),
  });
}

export function useAuditFeed(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.auditFeed(days),
    queryFn: () => getAuditFeed(days),
  });
}

function useInvalidateUsers() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ queryKey: queryKeys.admin.all() });
}

export function useInviteUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({
    mutationFn: (body: InviteRequest) => inviteUser(body),
    onSuccess: invalidate,
  });
}

export function useReinviteUser() {
  return useMutation({ mutationFn: (userId: string) => reinviteUser(userId) });
}

export function useRecoveryLink() {
  return useMutation({ mutationFn: (userId: string) => recoveryLink(userId) });
}

export function useUpdateUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({
    mutationFn: ({ userId, body }: { userId: string; body: UpdateUserRequest }) =>
      updateUser(userId, body),
    onSuccess: invalidate,
  });
}

export function useDisableUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({ mutationFn: disableUser, onSuccess: invalidate });
}

export function useEnableUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({ mutationFn: enableUser, onSuccess: invalidate });
}

export function useDeleteUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({ mutationFn: deleteUser, onSuccess: invalidate });
}
```

- [ ] **Step 10.5:** Export from `frontend/src/hooks/api/index.ts` (append, following the file's section-banner style):

```typescript
// =============================================================================
// ADMIN USER MANAGEMENT HOOKS
// =============================================================================

export {
  useAdminUsers,
  useUserActivity,
  usePlatformActivity,
  useAuditFeed,
  useInviteUser,
  useReinviteUser,
  useRecoveryLink,
  useUpdateUser,
  useDisableUser,
  useEnableUser,
  useDeleteUser,
} from './use-admin';
```

- [ ] **Step 10.6:** Type-check (the ONLY faithful tsc invocation — bare `npx tsc --noEmit` is a FALSE GREEN in this repo):

```bash
cd frontend && npx tsc -p tsconfig.app.json --noEmit
```

- [ ] **Step 10.7: Commit**

```bash
git add frontend/src/types/admin.ts frontend/src/api/admin.ts frontend/src/hooks/api/use-admin.ts frontend/src/hooks/api/index.ts frontend/src/lib/query-client.ts
git commit -m "feat(admin-ui): admin API types, client functions, query keys, hooks"
```

---

### Task 11: AcceptInvite page (public) — red-first

**Files:**
- Test: `frontend/src/pages/AcceptInvite.test.tsx` (create)
- Create: `frontend/src/pages/AcceptInvite.tsx`
- Modify: `frontend/src/router/routes.tsx` (lazy import + public route before the `*` catch-all)

- [ ] **Step 11.1: Write the failing test:**

```typescript
/**
 * AcceptInvite Page Tests
 * =======================
 * The page verifies an invite token_hash via supabase.auth.verifyOtp, then
 * lets the invitee set a password (updateUser) and enter the app.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import AcceptInvite from './AcceptInvite';

const mockVerifyOtp = vi.fn();
const mockUpdateUser = vi.fn();

vi.mock('@/lib/supabase', () => ({
  supabase: {
    auth: {
      verifyOtp: (...args: unknown[]) => mockVerifyOtp(...args),
      updateUser: (...args: unknown[]) => mockUpdateUser(...args),
    },
  },
  isSupabaseConfigured: () => true,
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const mod = await importOriginal<typeof import('react-router-dom')>();
  return { ...mod, useNavigate: () => mockNavigate };
});

function renderAt(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <AcceptInvite />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('AcceptInvite', () => {
  it('shows an error when the link has no token_hash', () => {
    renderAt('/accept-invite');
    expect(screen.getByText(/invalid invite link/i)).toBeInTheDocument();
    expect(mockVerifyOtp).not.toHaveBeenCalled();
  });

  it('verifies the token and shows the set-password form', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    renderAt('/accept-invite?token_hash=abc123');
    await waitFor(() =>
      expect(mockVerifyOtp).toHaveBeenCalledWith({ type: 'invite', token_hash: 'abc123' })
    );
    expect(await screen.findByLabelText(/new password/i)).toBeInTheDocument();
  });

  it('shows expired-link error when verification fails', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: null, user: null },
      error: { message: 'Email link is invalid or has expired' },
    });
    renderAt('/accept-invite?token_hash=stale');
    expect(await screen.findByText(/invalid or has expired/i)).toBeInTheDocument();
  });

  it('sets the password and navigates into the app', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    mockUpdateUser.mockResolvedValue({ data: { user: {} }, error: null });
    renderAt('/accept-invite?token_hash=abc123');
    const pw = await screen.findByLabelText(/new password/i);
    const confirm = screen.getByLabelText(/confirm password/i);
    await userEvent.type(pw, 'Str0ng!Passw0rd');
    await userEvent.type(confirm, 'Str0ng!Passw0rd');
    await userEvent.click(screen.getByRole('button', { name: /set password/i }));
    await waitFor(() =>
      expect(mockUpdateUser).toHaveBeenCalledWith({ password: 'Str0ng!Passw0rd' })
    );
    expect(mockNavigate).toHaveBeenCalledWith('/', { replace: true });
  });

  it('rejects mismatched passwords without calling updateUser', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    renderAt('/accept-invite?token_hash=abc123');
    await userEvent.type(await screen.findByLabelText(/new password/i), 'Str0ng!Passw0rd');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'different');
    await userEvent.click(screen.getByRole('button', { name: /set password/i }));
    expect(await screen.findByText(/passwords do not match/i)).toBeInTheDocument();
    expect(mockUpdateUser).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 11.2:** `cd frontend && npx vitest run src/pages/AcceptInvite.test.tsx` → FAIL (module missing).

- [ ] **Step 11.3: Implement** `frontend/src/pages/AcceptInvite.tsx` (style-match Login.tsx — check its card/container classes and reuse them):

```tsx
/**
 * AcceptInvite Page (PUBLIC route /accept-invite)
 * ===============================================
 * Invite links minted by /api/admin/users/invite land here:
 *   /accept-invite?token_hash=<hashed_token>
 * Flow: verifyOtp({type:'invite'}) -> session -> set password -> enter app.
 * Recovery links from the admin also use this page (verifyOtp falls back to
 * type:'recovery' when the invite type is rejected).
 */

import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { supabase } from '@/lib/supabase';

type Phase = 'verifying' | 'set-password' | 'done' | 'error';

export default function AcceptInvite() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tokenHash = searchParams.get('token_hash');

  const [phase, setPhase] = useState<Phase>(tokenHash ? 'verifying' : 'error');
  const [error, setError] = useState<string | null>(
    tokenHash ? null : 'Invalid invite link — no token found. Ask your admin for a new link.'
  );
  const [email, setEmail] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!tokenHash) return;
    let cancelled = false;
    (async () => {
      // Invite links use type 'invite'; admin recovery links reuse this page
      // with type 'recovery' — try invite first, fall back.
      let result = await supabase.auth.verifyOtp({ type: 'invite', token_hash: tokenHash });
      if (result.error) {
        result = await supabase.auth.verifyOtp({ type: 'recovery', token_hash: tokenHash });
      }
      if (cancelled) return;
      if (result.error || !result.data.session) {
        setError(result.error?.message ?? 'Verification failed');
        setPhase('error');
        return;
      }
      setEmail(result.data.user?.email ?? null);
      setPhase('set-password');
    })();
    return () => {
      cancelled = true;
    };
  }, [tokenHash]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setError(null);
    setSubmitting(true);
    const { error: updateError } = await supabase.auth.updateUser({ password });
    setSubmitting(false);
    if (updateError) {
      setError(updateError.message);
      return;
    }
    setPhase('done');
    navigate('/', { replace: true });
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--color-background)] px-4">
      <div className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-8 shadow-sm">
        <h1 className="mb-2 text-2xl font-semibold text-[var(--color-foreground)]">
          Welcome to E2I Analytics
        </h1>

        {phase === 'verifying' && (
          <p className="text-[var(--color-muted-foreground)]">Verifying your invite…</p>
        )}

        {phase === 'error' && (
          <div role="alert" className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
            {error}
          </div>
        )}

        {phase === 'set-password' && (
          <form onSubmit={handleSubmit} className="mt-4 space-y-4">
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {email ? `Signed in as ${email}. ` : ''}Choose a password to finish setting up
              your account.
            </p>
            <div>
              <label htmlFor="new-password" className="mb-1 block text-sm font-medium">
                New password
              </label>
              <input
                id="new-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label htmlFor="confirm-password" className="mb-1 block text-sm font-medium">
                Confirm password
              </label>
              <input
                id="confirm-password"
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
                autoComplete="new-password"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
              />
            </div>
            {error && (
              <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
                {error}
              </div>
            )}
            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {submitting ? 'Saving…' : 'Set password'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 11.4:** Register the route in `frontend/src/router/routes.tsx`. Lazy import (with the other page imports):

```typescript
const AcceptInvite = lazy(() => import('@/pages/AcceptInvite'));
```

Public route (next to `/reset-password`, BEFORE the protected block — no nav entry, no ProtectedRoute):

```tsx
  // Invite acceptance (public, like /login): target of admin-minted invite
  // and recovery links (/api/admin/users/invite).
  {
    path: '/accept-invite',
    element: (
      <LazyPage>
        <AcceptInvite />
      </LazyPage>
    ),
  },
```

- [ ] **Step 11.5:** Run to PASS: `cd frontend && npx vitest run src/pages/AcceptInvite.test.tsx`

- [ ] **Step 11.6: Commit**

```bash
git add frontend/src/pages/AcceptInvite.tsx frontend/src/pages/AcceptInvite.test.tsx frontend/src/router/routes.tsx
git commit -m "feat(admin-ui): public /accept-invite page — verifyOtp + set password"
```

---

### Task 12: Admin page — Users tab (red-first)

**Files:**
- Test: `frontend/src/pages/Admin.test.tsx` (create)
- Create: `frontend/src/pages/Admin.tsx`
- Create: `frontend/src/components/admin/UsersTable.tsx`
- Create: `frontend/src/components/admin/InviteUserDialog.tsx`
- Create: `frontend/src/components/admin/EditUserDialog.tsx`
- Create: `frontend/src/components/admin/ConfirmDeleteDialog.tsx`
- Create: `frontend/src/components/admin/index.ts`

The page follows the repo's page conventions (header + content, MainLayout is applied by the router shell — check how `SystemHealth.tsx` opens and mirror its outer div/классes). Components stay small and focused; `Admin.tsx` only owns tab state.

- [ ] **Step 12.1: Write the failing tests** (vi.mock the hooks module — Documentation.test.tsx convention):

```typescript
/**
 * Admin Page Tests — Users tab behaviors.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Admin from './Admin';
import type { AdminUser } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  useAdminUsers: vi.fn(),
  useUserActivity: vi.fn(),
  usePlatformActivity: vi.fn(),
  useAuditFeed: vi.fn(),
  useInviteUser: vi.fn(),
  useReinviteUser: vi.fn(),
  useRecoveryLink: vi.fn(),
  useUpdateUser: vi.fn(),
  useDisableUser: vi.fn(),
  useEnableUser: vi.fn(),
  useDeleteUser: vi.fn(),
}));
vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => ({ isAdmin: true, user: { id: 'me-id', email: 'me@x.com' } }),
}));

import * as adminHooks from '@/hooks/api/use-admin';

const USERS: AdminUser[] = [
  {
    id: 'me-id',
    email: 'me@x.com',
    full_name: 'Me Admin',
    role: 'admin',
    brands: ['all'],
    status: 'active',
    created_at: '2026-02-01T00:00:00Z',
    last_sign_in_at: '2026-07-10T00:00:00Z',
    total_conversations: 5,
    total_messages: 40,
    last_active_at: '2026-07-10T00:00:00Z',
  },
  {
    id: 'u2',
    email: 'viewer@x.com',
    full_name: null,
    role: 'viewer',
    brands: ['Kisqali'],
    status: 'invited',
    created_at: '2026-07-01T00:00:00Z',
    last_sign_in_at: null,
    total_conversations: 0,
    total_messages: 0,
    last_active_at: null,
  },
];

const idleMutation = () => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false,
  data: undefined,
  error: null,
  reset: vi.fn(),
});

beforeEach(() => {
  vi.clearAllMocks();
  (adminHooks.useAdminUsers as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { users: USERS },
    isLoading: false,
    isError: false,
  });
  (adminHooks.usePlatformActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { days: [] },
    isLoading: false,
  });
  (adminHooks.useAuditFeed as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { events: [] },
    isLoading: false,
  });
  (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  for (const name of [
    'useInviteUser',
    'useReinviteUser',
    'useRecoveryLink',
    'useUpdateUser',
    'useDisableUser',
    'useEnableUser',
    'useDeleteUser',
  ] as const) {
    (adminHooks[name] as ReturnType<typeof vi.fn>).mockReturnValue(idleMutation());
  }
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Admin page — Users tab', () => {
  it('renders the users table with roles and statuses', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /administration/i })).toBeInTheDocument();
    const table = screen.getByRole('table');
    expect(within(table).getByText('me@x.com')).toBeInTheDocument();
    expect(within(table).getByText('viewer@x.com')).toBeInTheDocument();
    expect(within(table).getByText(/invited/i)).toBeInTheDocument();
  });

  it('opens the invite dialog and shows the one-time link on success', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      user_id: 'u3',
      email: 'new@x.com',
      invite_link: 'https://eznomics.site/accept-invite?token_hash=xyz',
      link_type: 'invite',
    });
    (adminHooks.useInviteUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /invite user/i }));
    await userEvent.type(screen.getByLabelText(/email/i), 'new@x.com');
    await userEvent.click(screen.getByRole('button', { name: /^send invite$/i }));
    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    expect(
      await screen.findByText(/eznomics\.site\/accept-invite\?token_hash=xyz/)
    ).toBeInTheDocument();
    expect(screen.getByText(/shown once/i)).toBeInTheDocument();
  });

  it('delete requires typing the email to confirm', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({ deleted: true });
    (adminHooks.useDeleteUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    const row = screen.getByText('viewer@x.com').closest('tr')!;
    await userEvent.click(within(row).getByRole('button', { name: /delete/i }));
    const confirmBtn = screen.getByRole('button', { name: /delete permanently/i });
    expect(confirmBtn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/type the email/i), 'viewer@x.com');
    expect(confirmBtn).toBeEnabled();
    await userEvent.click(confirmBtn);
    await waitFor(() => expect(mutateAsync).toHaveBeenCalledWith('u2'));
  });

  it('does not offer delete/disable on your own row', () => {
    renderPage();
    const myRow = screen.getByText('me@x.com').closest('tr')!;
    expect(within(myRow).queryByRole('button', { name: /delete/i })).toBeNull();
    expect(within(myRow).queryByRole('button', { name: /disable/i })).toBeNull();
  });
});
```

- [ ] **Step 12.2:** `cd frontend && npx vitest run src/pages/Admin.test.tsx` → FAIL.

- [ ] **Step 12.3: Implement the components.** Keep each file focused. `frontend/src/components/admin/UsersTable.tsx` (table + row actions; props take the users array, current admin id, and action callbacks), `InviteUserDialog.tsx` (form: email, full name, role select, brand multi-select; success state renders the link in a `<code>` block + copy-to-clipboard button + "This link is shown once — copy it now" notice), `EditUserDialog.tsx` (role select + brand multi-select + name), `ConfirmDeleteDialog.tsx` (type-the-email gate; `aria-label="Type the email to confirm"` on the input; confirm button disabled until exact match). `Admin.tsx`:

```tsx
/**
 * Admin Page (/admin — admin-only via ProtectedRoute requireAdmin)
 * Users tab: invite / role & brands / disable / delete / links.
 * Activity tab: platform + per-user activity over time, admin audit feed.
 */

import { useState } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { useAdminUsers } from '@/hooks/api/use-admin';
import { UsersTable } from '@/components/admin/UsersTable';
import { InviteUserDialog } from '@/components/admin/InviteUserDialog';
import { ActivityTab } from '@/components/admin/ActivityTab';

type Tab = 'users' | 'activity';

export default function Admin() {
  const [tab, setTab] = useState<Tab>('users');
  const [inviteOpen, setInviteOpen] = useState(false);
  const { user } = useAuth();
  const { data, isLoading, isError } = useAdminUsers();

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Administration</h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Invite users, manage roles and brand access, and review activity.
          </p>
        </div>
        {tab === 'users' && (
          <button
            type="button"
            onClick={() => setInviteOpen(true)}
            className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white"
          >
            Invite user
          </button>
        )}
      </div>

      <div role="tablist" className="flex gap-2 border-b border-[var(--color-border)]">
        {(['users', 'activity'] as const).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize ${
              tab === t
                ? 'border-b-2 border-[var(--color-primary)] text-[var(--color-foreground)]'
                : 'text-[var(--color-muted-foreground)]'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'users' && (
        <UsersTable
          users={data?.users ?? []}
          currentUserId={user?.id ?? ''}
          isLoading={isLoading}
          isError={isError}
        />
      )}
      {tab === 'activity' && <ActivityTab users={data?.users ?? []} />}

      <InviteUserDialog open={inviteOpen} onClose={() => setInviteOpen(false)} />
    </div>
  );
}
```

(UsersTable owns Edit/Confirm dialogs + the disable/enable/delete/reinvite/recovery mutations from `use-admin`; self-row hides delete/disable — the backend guards are the real enforcement, the UI just doesn't offer footguns. Full code is written at implementation time following the test contract above; ActivityTab arrives in Task 13 — for THIS task create a placeholder-free minimal `ActivityTab` that renders the platform chart section header and "select a user" prompt so the page compiles and tests pass.)

- [ ] **Step 12.4:** Run to PASS: `cd frontend && npx vitest run src/pages/Admin.test.tsx`

- [ ] **Step 12.5: Commit**

```bash
git add frontend/src/pages/Admin.tsx frontend/src/pages/Admin.test.tsx frontend/src/components/admin/
git commit -m "feat(admin-ui): Admin page Users tab — invite dialog, role editor, guarded delete"
```

---

### Task 13: Admin page — Activity tab (charts + feeds)

**Files:**
- Create: `frontend/src/components/admin/ActivityTab.tsx` (replace the Task-12 minimal version)
- Create: `frontend/src/components/admin/ActivityTab.test.tsx`

- [ ] **Step 13.1: Write the failing tests:**

```typescript
/**
 * ActivityTab Tests — platform chart, per-user drill-down, audit feed.
 * recharts renders SVG in jsdom; assert on section headings + data presence
 * hooks were called with the right args (charts themselves are visual —
 * verified live at the end).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ActivityTab } from './ActivityTab';
import type { AdminUser } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  usePlatformActivity: vi.fn(),
  useUserActivity: vi.fn(),
  useAuditFeed: vi.fn(),
}));
import * as adminHooks from '@/hooks/api/use-admin';

const USERS = [
  { id: 'u1', email: 'a@x.com' } as AdminUser,
  { id: 'u2', email: 'b@x.com' } as AdminUser,
];

beforeEach(() => {
  vi.clearAllMocks();
  (adminHooks.usePlatformActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      days: [
        { day: '2026-07-01', logins: 4, active_users: 3 },
        { day: '2026-07-02', logins: 6, active_users: 5 },
      ],
    },
    isLoading: false,
  });
  (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  (adminHooks.useAuditFeed as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      events: [
        {
          event_id: 'e1',
          event_type: 'admin.user.modified',
          severity: 'warning',
          timestamp: '2026-07-11T10:00:00Z',
          message: 'Invited new@x.com as viewer',
          user_email: 'me@x.com',
          resource_id: 'u9',
          metadata: {},
        },
      ],
    },
    isLoading: false,
  });
});

describe('ActivityTab', () => {
  it('renders platform activity section with data', () => {
    render(<ActivityTab users={USERS} />);
    expect(screen.getByRole('heading', { name: /platform activity/i })).toBeInTheDocument();
  });

  it('drills into a selected user', async () => {
    (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        user_id: 'u2',
        email: 'b@x.com',
        auth_events: [{ day: '2026-07-01', event_type: 'login', event_count: 2 }],
        api_activity: [
          {
            endpoint_group: 'causal',
            http_method: 'GET',
            bucket_minute: '2026-07-01T10:00:00Z',
            request_count: 7,
          },
        ],
        recent_events: [{ occurred_at: '2026-07-01T10:00:00Z', action: 'login' }],
        chat: { total_conversations: 1, total_messages: 9, last_active_at: null },
      },
      isLoading: false,
    });
    render(<ActivityTab users={USERS} />);
    await userEvent.selectOptions(screen.getByLabelText(/select user/i), 'u2');
    expect(adminHooks.useUserActivity).toHaveBeenLastCalledWith('u2', expect.any(Number));
    expect(screen.getByRole('heading', { name: /user activity/i })).toBeInTheDocument();
    expect(screen.getByText(/9/)).toBeInTheDocument(); // chat messages stat
  });

  it('renders the admin audit feed', () => {
    render(<ActivityTab users={USERS} />);
    expect(screen.getByRole('heading', { name: /admin audit/i })).toBeInTheDocument();
    expect(screen.getByText(/invited new@x\.com as viewer/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 13.2:** Run → FAIL. **Step 13.3: Implement** `ActivityTab.tsx`: platform `LineChart` (recharts: `ResponsiveContainer` + `LineChart` with `logins` and `active_users` lines over `day`), a user `<select aria-label="Select user">` driving `useUserActivity(selectedId, 90)`, per-user login `BarChart` (auth_events filtered to `login`), API-activity summary aggregated by endpoint_group (`BarChart`), chat counters as stat cards, recent-events list, and the `useAuditFeed(30)` table under an "Admin audit" heading. **Step 13.4:** Run to PASS.

- [ ] **Step 13.5: Commit**

```bash
git add frontend/src/components/admin/ActivityTab.tsx frontend/src/components/admin/ActivityTab.test.tsx
git commit -m "feat(admin-ui): Activity tab — platform/user activity charts + admin audit feed"
```

---

### Task 14: Nav + routing wiring (admin-gated)

**Files:**
- Modify: `frontend/src/router/routes.tsx` (RouteConfig.adminOnly + config entry + protected route)
- Modify: `frontend/src/components/layout/Sidebar.tsx` (filter adminOnly by isAdmin)
- Test: extend `frontend/src/pages/Admin.test.tsx` is NOT needed; create `frontend/src/router/admin-nav.test.tsx`

- [ ] **Step 14.1: Write the failing test:**

```typescript
/**
 * Admin nav gating — /admin appears in the sidebar ONLY for admins, and the
 * route registry carries adminOnly + requireAdmin protection.
 */
import { describe, it, expect } from 'vitest';
import { getNavigationSections, routeConfigs, routes } from './routes';

describe('admin nav registry', () => {
  it('has an adminOnly /admin entry in the system section', () => {
    const admin = routeConfigs.find((r) => r.path === '/admin');
    expect(admin).toBeDefined();
    expect(admin!.adminOnly).toBe(true);
    expect(admin!.section).toBe('system');
    expect(admin!.showInNav).toBe(true);
  });

  it('getNavigationSections(false) excludes adminOnly routes', () => {
    const sections = getNavigationSections(false);
    const paths = sections.flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).not.toContain('/admin');
  });

  it('getNavigationSections(true) includes /admin', () => {
    const sections = getNavigationSections(true);
    const paths = sections.flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).toContain('/admin');
  });

  it('defaults to excluding adminOnly (backwards-compatible no-arg call)', () => {
    const paths = getNavigationSections().flatMap((s) => s.routes.map((r) => r.path));
    expect(paths).not.toContain('/admin');
  });

  it('registers /admin and /accept-invite as router routes', () => {
    const paths = routes.map((r) => r.path);
    expect(paths).toContain('/admin');
    expect(paths).toContain('/accept-invite');
  });
});
```

- [ ] **Step 14.2:** Run → FAIL. **Step 14.3: Implement** in `routes.tsx`:

Add to `RouteConfig`:

```typescript
  /** Only shown in nav (and only reachable) for admins. */
  adminOnly?: boolean;
```

Add the config entry at the END of the system section (after Feedback Learning):

```typescript
  {
    path: '/admin',
    title: 'Administration',
    description: 'Invite and manage users, roles, and activity',
    icon: 'shield-check',
    section: 'system',
    showInNav: true,
    adminOnly: true,
  },
```

Lazy import + protected route (with the other protected routes):

```typescript
const Admin = lazy(() => import('@/pages/Admin'));
```

```tsx
  {
    path: '/admin',
    element: (
      <ProtectedRoute requireAdmin>
        <LazyPage>
          <Admin />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
```

Change `getNavigationSections` to accept the flag (default false — non-admin callers and existing tests unchanged):

```typescript
export function getNavigationSections(includeAdmin = false): NavSectionGroup[] {
  const navRoutes = getNavigationRoutes().filter(
    (route) => includeAdmin || !route.adminOnly
  );
  return NAV_SECTION_ORDER.map(({ key, label }) => ({
    key,
    label,
    routes: navRoutes.filter((route) => (route.section ?? 'main') === key),
  })).filter((group) => group.routes.length > 0);
}
```

In `Sidebar.tsx` — add the import and pass the flag (line ~15 and ~225):

```typescript
import { useAuth } from '@/hooks/use-auth';
```

```typescript
  const { isAdmin } = useAuth();
  const navSections = getNavigationSections(isAdmin);
```

- [ ] **Step 14.4:** Run to PASS: `cd frontend && npx vitest run src/router/admin-nav.test.tsx`. Also re-run the sidebar's existing tests if any: `npx vitest run src/components/layout`.

- [ ] **Step 14.5: Frontend quality checkpoint:**

```bash
cd frontend && npx tsc -p tsconfig.app.json --noEmit && npx vitest run src/pages/Admin.test.tsx src/pages/AcceptInvite.test.tsx src/components/admin src/router/admin-nav.test.tsx
```

(Do NOT run prettier --write across files — no prettier gate in this repo, it just pollutes diffs.)

- [ ] **Step 14.6: Commit**

```bash
git add frontend/src/router/routes.tsx frontend/src/router/admin-nav.test.tsx frontend/src/components/layout/Sidebar.tsx
git commit -m "feat(admin-ui): admin-gated /admin nav entry + requireAdmin route"
```

---

### Task 15: Convergence — ralph-loop + codex-rescue to fixed point

**Files:** whatever the audits surface

- [ ] **Step 15.1:** Invoke ralph-wiggum:ralph-loop over the branch diff with codex:codex-rescue as the auditor. The codex brief MUST include the CLAUDE.md pushback paragraph verbatim:

> If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

Focus areas for the auditor: (1) security of the admin surface (privilege escalation, missing require_admin, service-key leakage into responses), (2) the disabled-flag enforcement path, (3) guard completeness (last-admin, self-target), (4) activity middleware fail-open + memory bounds, (5) invite-link handling (token in URL — verify no token logging).

- [ ] **Step 15.2:** Fix findings with the same TDD discipline (red test → fix → green), one commit per finding batch. Re-run the audit until it returns NO findings (fixed point). Re-run the full targeted test batch (Task 9.2 + Task 14.5) after each fix round.

---

### Task 16: One-time reconcile + final pre-PR verification

**Files:** none (operations)

- [ ] **Step 16.1:** Run the role-store reconcile against prod (this is the app_metadata side the SQL backfill couldn't do):

```bash
docker exec -i e2i_api python <<'EOF'
import sys
sys.path.insert(0, "/app")
from src.services.admin_user_service import AdminUserService
report = AdminUserService().reconcile_role_stores()
for r in report:
    print(r)
print(f"reconciled: {len(report)} users")
EOF
```

NOTE: the running container predates this branch — if the module isn't baked in yet, defer this step to Task 17 post-deploy (it's idempotent; run it then). Record which path was taken.

- [ ] **Step 16.2:** Verify zero drift remains:

```bash
docker exec supabase-db psql -U postgres -d postgres -tA -c "SELECT u.email, u.raw_app_meta_data->>'role', p.role FROM auth.users u LEFT JOIN chatbot_user_profiles p ON p.id = u.id WHERE COALESCE(u.raw_app_meta_data->>'role','') IS DISTINCT FROM COALESCE(p.role::text,'')"
```

Expected: empty (or only rows explained in the PR body).

- [ ] **Step 16.3:** Memory check + full targeted suites one final time (`free -m`; backend batch from 9.2; frontend batch from 14.5).

---

### Task 17: PR → batched CI → merge → deploy → LIVE verification

**Files:** none (release)

- [ ] **Step 17.1:** Push ONCE (the batched-CI moment) and open the PR:

```bash
git config --global http.https://github.com.proxy ""
git push -u origin feat/admin-user-management
gh pr create --title "feat(admin): user management & activity — /admin page, invites, roles, activity tracking" --body-file /tmp/pr_body.md
```

PR body: summary of spec/plan, the four verified-disproof callouts (SMTP, ban semantics, user_profiles absence, reinvite semantics), migration-100 note (already applied to the droplet — additive + idempotent), test evidence (targeted run outputs), and the standard footer. Verify the body landed (`gh pr view --json body | head`) — `--body-file` has silently failed before.

- [ ] **Step 17.2:** Watch CI (`gh pr checks --watch`). Fix red gates with targeted commits; push fixes as they come (still one logical batch — no deploy happens until merge).

- [ ] **Step 17.3:** Merge preserving history (frontend-touching PRs need `--admin`):

```bash
gh pr merge --merge --admin
```

- [ ] **Step 17.4:** Deploy: the push-to-main CI pipeline force-recreates prod from sha-tagged GHCR images. Watch `gh run list --branch main --limit 3` until green. Mid-deploy 502s are expected — hard-reload before diagnosing.

- [ ] **Step 17.5:** If Step 16.1 was deferred: run the reconcile now inside the NEW container (same heredoc).

- [ ] **Step 17.6: LIVE end-to-end verification** (chrome-devtools MCP against the user's logged-in Chrome, verify by RENDERED CONTENT):
  1. Hard-reload eznomics.site → sidebar shows "Administration" (admin account).
  2. /admin → Users tab lists the 8 real users with correct roles/statuses.
  3. Invite `etn3724+livetest@gmail.com` as viewer/Kisqali → copy the link → open it in a fresh incognito page → set a password → confirm entry to the app, NO admin nav item, and Kisqali-scoped data.
  4. Activity tab → platform chart shows real login history (data since 2026-02); drill into your own user → login events present; API activity accrues after browsing (make a few page visits, refetch).
  5. Disable the livetest user → confirm their existing session gets 401s (refresh their tab) and sign-in is blocked. Enable → sign-in works.
  6. Delete the livetest user (type-to-confirm) → row gone; `security_audit_log` shows the full action trail (psql check).
  7. Confirm memory is healthy: `free -m`, `docker stats --no-stream e2i_api`.

- [ ] **Step 17.7:** Save the session memory file (feature summary, gotchas discovered, verification evidence) per the memory instructions.

---

## Self-review checklist (run after writing, before executing)

- Spec coverage: invite (copyable link) ✓ T6/T8/T11/T12; reinvite+recovery ✓ T6/T8; roles+brands dual-write ✓ T7/T8/T12; disable/enable immediate lockout ✓ T3/T7/T8; guarded delete ✓ T7/T8/T12; activity over time (auth history + API + chat) ✓ T1/T5/T7/T8/T13; audit trail + sink fix ✓ T2/T8/T13; nav/RBAC UI gating ✓ T14; backfill/reconcile ✓ T1/T16; batched CI ✓ T17; live verify ✓ T17.6.
- Placeholders: Task 12.3 delegates full dialog internals to the test contract (deliberate: tests in 12.1 ARE the specification; every behavior asserted there must be implemented — this is not a TBD).
- Type consistency: `AdminUserService` method names match route calls (`invite_user/reinvite_user/recovery_link/update_user/disable_user/enable_user/delete_user/user_activity/platform_activity/reconcile_role_stores`); frontend api fn names match hook imports; `LinkResponse`/`link_type` consistent across backend + frontend; `getNavigationSections(includeAdmin)` default preserves existing callers.
