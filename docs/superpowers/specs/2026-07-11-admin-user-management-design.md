# Admin User Management & Activity — Design Spec

**Date:** 2026-07-11
**Status:** Approved by user (approach + design sections + addendum, 2026-07-11)
**Requested by:** etn3724@gmail.com (platform admin on eznomics.site)

## Problem

The deployed platform (https://eznomics.site/) has a full 4-tier RBAC model
(`viewer < analyst < operator < admin`, hierarchical, brand-scoped) enforced by the
backend, and the logged-in user is an admin — but there is **no admin surface**:
no way to invite or remove users, change their role/brand privileges, or view
their activity over time. `require_admin`'s docstring already promises
"user management" (`src/api/dependencies/auth.py:551`); nothing implements it.

## Verified facts the design rests on (cheapest-disproof results)

All verified against the live droplet stack (PROD == DEV == this host), not assumed:

1. **`auth.admin.*` works today.** `docker exec e2i_api python` with
   `create_client("http://172.17.0.1:54321", SUPABASE_SERVICE_KEY)` →
   `auth.admin.list_users()` returned all 8 real users with `app_metadata` roles.
   The service-role key is already in the container env (`docker-compose.yml`
   `x-common-env`). No new secret needed.
2. **Email invites can never deliver.** GoTrue is configured with a fake SMTP host
   (`GOTRUE_SMTP_HOST=supabase-mail` — no such container) and
   `GOTRUE_MAILER_AUTOCONFIRM=true`, `GOTRUE_DISABLE_SIGNUP=true`. Therefore the
   invite flow must produce a **copyable link**, not an email. This also means the
   existing ForgotPassword page silently cannot deliver reset emails — an existing
   gap this feature closes with admin-generated recovery links.
3. **Real activity history exists.** `auth.audit_log_entries` holds 1,088 events
   since 2026-02-09 (288 login, 398 token_refreshed, 396 token_revoked,
   2 user_signedup). Chat usage counters live on `chatbot_user_profiles`
   (total_conversations, total_messages, last_active_at).
4. **`security_audit_log` has 0 rows and the root cause is known.**
   `get_security_audit_service()` (`src/utils/security_audit.py:736`) does
   `from src.api.deps import get_supabase` — module doesn't exist (real module:
   `src.api.dependencies`). The ImportError is silently swallowed, so the DB sink
   is never wired and every security event ever emitted went to stdout only.
5. **Role stores have already drifted.** Three stores exist:
   JWT `app_metadata.role` (what the API trusts), `chatbot_user_profiles.role`
   (what RLS `has_role()` trusts), and `user_profiles.is_admin` (legacy, created by
   the `on_auth_user_created` trigger). Observed drift: `admin@e2i.local` has
   jwt_role=admin but no profile row; several users have NULL jwt_role but profile
   role=viewer.
6. **Backend role enforcement is real-time.** `verify_supabase_token()` calls
   `auth.get_user(token)` per request, which returns *fresh* `app_metadata` — so
   demotions apply on the next request, and GoTrue bans lock out existing tokens
   immediately. No token-expiry window to worry about (frontend session claims go
   stale until refresh, which is harmless).
7. **`GOTRUE_URI_ALLOW_LIST` does not include eznomics.site** and
   `GOTRUE_SITE_URL=http://138.197.4.36`. The invite link therefore must NOT rely
   on GoTrue's `action_link` redirect. Instead: extract `hashed_token` from
   `generate_link` and build `https://eznomics.site/accept-invite?token_hash=…`;
   the SPA completes verification same-origin via `supabase.auth.verifyOtp()`.
   No GoTrue config changes needed.

## Architecture (Approach A — thin admin layer over Supabase auth, user-selected)

Zero changes to the existing token-verification path.

- **Router:** `src/api/routes/admin.py`, `APIRouter(prefix="/admin", tags=["Admin"])`,
  registered `app.include_router(admin_router, prefix="/api")` (majority convention
  in `main.py`). Every endpoint `Depends(require_admin)`. NOT added to
  `PUBLIC_PATHS` — JWT-gated by the middleware by default.
- **Service:** `src/services/admin_user_service.py` — holds a server-only
  service-role Supabase client and all business logic; routes stay thin.
- **DB:** migration `database/migrations/100_admin_user_activity.sql` (next free
  number after the 099 collision):
  - `user_activity_log` table — pre-aggregated per-minute buckets:
    `(id bigserial PK, user_id uuid, endpoint_group text, method text,
    bucket_minute timestamptz, request_count int, created_at)` with index
    `(user_id, bucket_minute)`. Bounded cardinality by design (group × minute, not
    per-request rows).
  - Two `SECURITY DEFINER` RPCs (service_role-only EXECUTE) because PostgREST
    cannot reach the `auth` schema: `admin_get_login_activity(p_user_id, p_days)`
    and `admin_get_platform_activity(p_days)` reading `auth.audit_log_entries`.
  - Role-drift backfill: reconcile the three role stores (JWT `app_metadata` is
    authoritative where set; profiles created/updated to match; users with NULL
    jwt_role get their profile role written into `app_metadata` — implemented in
    the service as a one-time reconcile invoked post-migration, since
    `app_metadata` writes go through the auth admin API, not SQL).
  - `purge_old_user_activity(p_days int DEFAULT 180)` retention function
    (invoked manually or by existing cron infra; not auto-scheduled in this PR).
  - Idempotent (`IF NOT EXISTS` / `CREATE OR REPLACE`); no BEGIN/COMMIT (droplet
    psql applies autocommit); applied manually via
    `docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < file`.
- **Activity capture middleware:** `src/api/middleware/activity_tracking.py`,
  added after JWTAuthMiddleware (so `request.state.user` exists). Appends
  `(user_id, endpoint_group, minute)` to a **bounded** in-memory aggregation dict
  (hard cap on distinct keys; increments-only past the cap with a dropped-events
  counter — OOM-safe). A background task flushes every 30 s or 500 buckets as one
  batched insert. Fail-open: any error is logged and the request proceeds;
  tracking never blocks or breaks traffic. Only authenticated, non-public API
  paths are recorded; path is normalized to a small `endpoint_group` set (first
  path segment after `/api/`) to bound cardinality.
- **Audit sink fix:** correct the `src.api.deps` → `src.api.dependencies` import
  in `src/utils/security_audit.py` so security events persist to
  `security_audit_log` (the table, RLS, and helper functions already exist —
  `database/audit/012_security_audit_log.sql`). All admin mutations emit events
  using the enum values that already anticipate this feature
  (`ADMIN_USER_MODIFIED`, `AUTHZ_ROLE_CHANGE`, `AUTHZ_PERMISSION_GRANTED/REVOKED`).
- **Frontend:**
  - `frontend/src/pages/Admin.tsx` (+ colocated `Admin.test.tsx`), route `/admin`
    wrapped in `<ProtectedRoute requireAdmin>` (prop exists, currently unused).
  - Nav registry (`frontend/src/router/routes.tsx`): new `adminOnly: true` flag on
    the route config; `Sidebar.tsx` filters `adminOnly` entries using the existing
    `isAdmin` selector from `auth-store.ts`.
  - `frontend/src/pages/AcceptInvite.tsx` — **public** route `/accept-invite`
    (like `/login`): reads `token_hash` from the URL, calls
    `supabase.auth.verifyOtp({ type: 'invite', token_hash })`, then a set-password
    form (`auth.updateUser({ password })`), then redirects into the app. Graceful
    handling of expired/invalid/already-used links.

## Role model

Existing hierarchy unchanged: `viewer(1) < analyst(2) < operator(3) < admin(4)`.
Brand grants: `Kisqali`, `Fabhalta`, `Remibrutinib`, or `all` (cross-brand).
Role changes are a **tri-write**: `app_metadata.role` + `app_metadata.brands`
(auth admin API), `chatbot_user_profiles.role`/`is_admin`, and
`user_profiles.is_admin`. `app_metadata` is authoritative for the API;
`chatbot_user_profiles.role` must match for RLS correctness.

## Endpoints (all `Depends(require_admin)`, all mutations audited)

| Endpoint | Behavior |
|---|---|
| `GET /api/admin/users` | Merged list: `auth.admin.list_users()` + profile join + status (active / disabled / invited-pending via `last_sign_in_at`/`banned_until`) + last sign-in + chat usage |
| `POST /api/admin/users/invite` | `{email, full_name?, role, brands}` → `generate_link(type='invite')` → set `app_metadata` → sync profile stores (same tri-write as role changes) → return one-time copyable link `https://eznomics.site/accept-invite?token_hash=…` |
| `POST /api/admin/users/{id}/reinvite` | Fresh link for a pending (never-signed-in) user; actual GoTrue behavior for re-generation is pinned by a test, with fallback to a recovery-type link if invite-type regeneration is rejected |
| `PATCH /api/admin/users/{id}` | `{role?, brands?, full_name?}` — tri-write role stores |
| `POST /api/admin/users/{id}/disable` | GoTrue ban (`ban_duration` ≈ 100 y); locks out existing sessions immediately; reversible |
| `POST /api/admin/users/{id}/enable` | `ban_duration: 'none'` |
| `DELETE /api/admin/users/{id}` | Hard delete via `auth.admin.delete_user`; profile rows cascade. Guards: **no self-delete; no deleting or demoting the last active admin.** UI requires type-to-confirm |
| `POST /api/admin/users/{id}/recovery-link` | Copyable password-reset link (closes the dead-ForgotPassword-email gap) |
| `GET /api/admin/users/{id}/activity?days=90` | Login history (RPC over `auth.audit_log_entries`), API activity buckets (`user_activity_log`), chat counters, recent events feed |
| `GET /api/admin/activity/overview?days=30` | Platform: daily active users, logins/day, API events/day |
| `GET /api/admin/audit?days=30` | Admin-action feed from `security_audit_log` |

Base URL for generated links comes from config (`E2I_PUBLIC_APP_URL`, default
`https://eznomics.site`) — not hardcoded.

## Frontend UI

`/admin` page, two tabs:

- **Users tab:** table (email, name, role badge, brands, status, last sign-in,
  chat usage, created) with per-row actions: edit role/brands, disable/enable,
  delete (type-the-email-to-confirm dialog), generate recovery link. "Invite
  User" button → dialog (email, name, role, brands) → success state shows the
  one-time copyable link with a copy button and an explicit "this link is shown
  once" notice.
- **Activity tab:** platform activity chart over time (DAU, logins/day, API
  events/day), per-user drill-down (login-history chart + API-activity chart +
  recent-events feed), and the admin audit feed.

Data fetching follows the repo's existing api-client + react-query hook
conventions; charts follow the existing recharts component patterns.

## Data flow

- **Invite:** admin dialog → `POST invite` → `generate_link` creates the user row
  immediately (status "invited-pending" until first sign-in) → metadata + profile
  written → link displayed once → invitee opens it on eznomics.site →
  `verifyOtp` establishes a session → set password → sign-in works thereafter.
- **Role change:** `update_user_by_id(app_metadata)` + profile updates → backend
  enforcement immediate (fresh `get_user` per request).
- **Disable:** ban → GoTrue rejects the banned user's tokens on next `get_user` →
  401 everywhere immediately.
- **Activity:** request → middleware buffer → batched insert → aggregation
  endpoints → charts.

## Error handling

- Guards return structured 4xx: 409 email-already-exists, 403 self-delete /
  last-admin violations, 422 invalid role or brand (brands restricted to the
  known 3 + `all`).
- Supabase/GoTrue failures map to 502 with detail; no partial tri-writes without
  surfacing which store failed (best-effort rollback of `app_metadata` if profile
  write fails, and the response says exactly what state the user is in).
- Audit-log failures never block the underlying action (existing convention).
- Activity middleware is fail-open with bounded memory and a dropped-events
  counter in logs.
- Frontend: toasts on failure, buttons disabled during mutations,
  refetch-after-mutation (no optimistic updates).

## Testing (TDD red-first, real systems, no mocks in product paths)

- **Backend pytest against the real local Supabase stack** (service key available
  in env). Tests create disposable users (`+admtest` email tags) and always clean
  up. Pinned real behaviors:
  - RBAC: non-admin gets 403 — using a **real JWT** obtained by signing in a
    disposable viewer via the anon client.
  - Invite: user created; returned link's `token_hash` actually completes
    `verifyOtp`; role/brands land in `app_metadata` AND profile.
  - Reinvite semantics (pin actual GoTrue behavior).
  - Role change: tri-write lands in all stores.
  - Disable: previously-valid token is rejected immediately; enable restores it.
  - Delete: auth user gone, profile rows cascaded; self-delete and last-admin
    guards enforced.
  - Activity: middleware buffer flushes real rows; aggregation endpoints return
    them; buffer cap drops (not grows) past the limit.
  - Audit sink: an admin action produces a real row in `security_audit_log`
    (this test is red today — the import bug — and proves the fix).
  - Droplet memory policy: **targeted test runs only** (`pytest tests/…/test_admin*`),
    never whole-tree; `free -m` / `docker stats` monitored during runs.
- **Frontend vitest** colocated per convention (`Admin.test.tsx`,
  `AcceptInvite.test.tsx`) using the repo's existing MSW test harness (test
  doubles in tests are fine; the no-mocking rule bars fake values in product
  paths).
- **Live verification at the end:** real invite for a disposable email → open
  link in the browser → set password → log in → confirm viewer-level access and
  no admin nav → clean up the disposable user. Frontend verified by rendered
  content (per standing feedback), backend by real API responses.

## Out of scope (explicit, not omitted silently)

- `user_sessions` table (WS3 MAU/WAU gap): session-grained latent schema with no
  write path; the per-minute `user_activity_log` serves the admin charts better.
  Left untouched; populating it is a separate future feature.
- SMTP/email delivery: not being fixed; the copyable-link design deliberately
  routes around it.
- The existing `/signup` page: `GOTRUE_DISABLE_SIGNUP=true` already makes it
  inert; invite flow supersedes it. No changes in this PR.
- Auto-scheduled activity retention: purge function ships; cron wiring is a
  follow-up.

## Rollout

Worktree isolation → TDD red-first → ralph-loop with codex rescue audits to
fixed point → migration 100 applied manually on the droplet → single PR
(merge commit, never squash) → **CI/deploy batched as one push at the very end**
(OOM history; racing deploys converge) → post-deploy live verification on
eznomics.site.
