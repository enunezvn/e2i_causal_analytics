# E2I Frontend Authentication Integration Plan

**Created**: 2025-01-08
**Completed**: 2026-01-09
**Status**: ✅ Complete
**Backend Auth**: Complete (JWT + Supabase on droplet 159.89.180.27:8001)

---

## Overview

Integrate Supabase Auth with the React frontend to enable user login/signup and protect authenticated routes.

### Current State
- **Backend**: JWT auth via Supabase deployed (v4.2.0)
- **Frontend**: ✅ Auth fully implemented and deployed
- **Supabase env vars**: ✅ Configured in production `.env` on droplet
- **Test user**: `testuser@example.com` / `TestUser2024` - ✅ Verified working

### Implementation Summary
All phases were already implemented in the codebase. The deployment task completed:
1. Created `.env` file on droplet with Supabase credentials
2. Rebuilt frontend with environment variables embedded
3. Restarted `e2i-frontend` systemd service
4. Verified login flow with test user credentials

### Tech Stack (Frontend)
- React 18.3.1 + TypeScript
- React Router 7.11.0
- Zustand (state management)
- React Query (data fetching)
- Tailwind CSS + Radix UI (shadcn/ui style)
- Axios (API client)

---

## Implementation Phases

### Phase 1: Supabase Client Setup
**Files to create/modify:**
- [ ] `frontend/src/lib/supabase.ts` - Supabase client singleton
- [ ] `frontend/package.json` - Add `@supabase/supabase-js`

**Tasks:**
1. Install Supabase JS client
2. Create typed Supabase client using existing env vars
3. Export client and auth helpers

**Verification:**
```bash
cd frontend && npm install @supabase/supabase-js
# Test import in browser console
```

---

### Phase 2: Auth Store (Zustand)
**Files to create/modify:**
- [ ] `frontend/src/stores/auth-store.ts` - Auth state management

**Tasks:**
1. Create auth store with user, session, loading states
2. Add actions: setUser, setSession, logout, initialize
3. Persist session to localStorage
4. Add selectors: isAuthenticated, isAdmin

**Verification:**
```typescript
// Unit test auth store actions
```

---

### Phase 3: Auth Context & Provider
**Files to create/modify:**
- [ ] `frontend/src/providers/AuthProvider.tsx` - Auth context
- [ ] `frontend/src/hooks/use-auth.ts` - Auth hook
- [ ] `frontend/src/main.tsx` - Wrap app with AuthProvider

**Tasks:**
1. Create AuthProvider with Supabase session listener
2. Handle auth state changes (login, logout, token refresh)
3. Create `useAuth()` hook for consuming auth state
4. Integrate with React Query for cache invalidation

**Verification:**
```bash
# Check auth state in React DevTools
```

---

### Phase 4: Login Page
**Files to create/modify:**
- [ ] `frontend/src/pages/Login.tsx` - Login page component
- [ ] `frontend/src/router/routes.tsx` - Add /login route

**Tasks:**
1. Create login form (email, password) using existing UI components
2. Handle form validation with react-hook-form + zod
3. Show loading state during auth
4. Handle errors (invalid credentials, network)
5. Redirect to dashboard on success
6. Add "Forgot password" link (future)

**Verification:**
```bash
# Manual test: login with testuser@example.com
```

---

### Phase 5: Signup Page
**Files to create/modify:**
- [ ] `frontend/src/pages/Signup.tsx` - Signup page component
- [ ] `frontend/src/router/routes.tsx` - Add /signup route

**Tasks:**
1. Create signup form (email, password, confirm password)
2. Validate password strength
3. Handle Supabase signup with email confirmation
4. Show success message / redirect

**Verification:**
```bash
# Manual test: create new user
```

---

### Phase 6: Protected Routes
**Files to create/modify:**
- [ ] `frontend/src/components/auth/ProtectedRoute.tsx` - Route guard
- [ ] `frontend/src/components/auth/index.ts` - Exports
- [ ] `frontend/src/router/routes.tsx` - Wrap protected routes

**Tasks:**
1. Create ProtectedRoute component
2. Redirect to /login if not authenticated
3. Show loading spinner during auth check
4. Preserve intended destination for post-login redirect

**Verification:**
```bash
# Test: access /dashboard without login -> redirects to /login
# Test: login -> redirects back to /dashboard
```

---

### Phase 7: API Client Auth Integration
**Files to create/modify:**
- [ ] `frontend/src/lib/api-client.ts` - Add auth interceptor

**Tasks:**
1. Add request interceptor to attach Bearer token
2. Add response interceptor for 401 handling
3. Auto-logout on token expiration
4. Refresh token handling (optional)

**Verification:**
```bash
# Test: API call to protected endpoint succeeds with token
# Test: 401 response triggers logout
```

---

### Phase 8: Header Auth UI
**Files to create/modify:**
- [ ] `frontend/src/components/layout/Header.tsx` - Add user menu
- [ ] `frontend/src/components/auth/UserMenu.tsx` - Dropdown menu

**Tasks:**
1. Show user email/avatar when logged in
2. Add dropdown with: Profile, Settings, Logout
3. Show Login button when not authenticated
4. Handle logout action

**Verification:**
```bash
# Visual test: header shows correct state
```

---

### Phase 9: Testing
**Files to create/modify:**
- [ ] `frontend/src/__tests__/auth/` - Auth test directory
- [ ] MSW handlers for auth endpoints

**Tasks:**
1. Unit tests for auth-store
2. Unit tests for useAuth hook
3. Integration tests for Login/Signup pages
4. E2E test for login flow (Playwright)

**Test batches (resource-friendly):**
- Batch 1: Store tests (no rendering)
- Batch 2: Hook tests (minimal rendering)
- Batch 3: Page tests (full rendering)
- Batch 4: E2E tests (browser)

---

## File Summary

| Phase | File | Action |
|-------|------|--------|
| 1 | `frontend/src/lib/supabase.ts` | Create |
| 1 | `frontend/package.json` | Modify |
| 2 | `frontend/src/stores/auth-store.ts` | Create |
| 3 | `frontend/src/providers/AuthProvider.tsx` | Create |
| 3 | `frontend/src/hooks/use-auth.ts` | Create |
| 3 | `frontend/src/main.tsx` | Modify |
| 4 | `frontend/src/pages/Login.tsx` | Create |
| 4 | `frontend/src/router/routes.tsx` | Modify |
| 5 | `frontend/src/pages/Signup.tsx` | Create |
| 5 | `frontend/src/router/routes.tsx` | Modify |
| 6 | `frontend/src/components/auth/ProtectedRoute.tsx` | Create |
| 6 | `frontend/src/components/auth/index.ts` | Create |
| 7 | `frontend/src/lib/api-client.ts` | Modify |
| 8 | `frontend/src/components/layout/Header.tsx` | Modify |
| 8 | `frontend/src/components/auth/UserMenu.tsx` | Create |

---

## Environment Variables

Already configured in `frontend/src/config/env.ts`:
```typescript
VITE_SUPABASE_URL=https://isbcwfupvuxzglpvqwvx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbG...
```

---

## Progress Tracking

### Phase 1: Supabase Client Setup
- [ ] Install @supabase/supabase-js
- [ ] Create supabase.ts client

### Phase 2: Auth Store
- [ ] Create auth-store.ts
- [ ] Test store actions

### Phase 3: Auth Provider
- [ ] Create AuthProvider.tsx
- [ ] Create use-auth.ts hook
- [ ] Integrate in main.tsx

### Phase 4: Login Page
- [ ] Create Login.tsx
- [ ] Add /login route
- [ ] Test login flow

### Phase 5: Signup Page
- [ ] Create Signup.tsx
- [ ] Add /signup route
- [ ] Test signup flow

### Phase 6: Protected Routes
- [ ] Create ProtectedRoute.tsx
- [ ] Wrap dashboard routes
- [ ] Test redirect behavior

### Phase 7: API Client
- [ ] Add auth interceptor
- [ ] Handle 401 responses
- [ ] Test authenticated API calls

### Phase 8: Header UI
- [ ] Create UserMenu.tsx
- [ ] Update Header.tsx
- [ ] Test logout

### Phase 9: Testing
- [ ] Batch 1: Store tests
- [ ] Batch 2: Hook tests
- [ ] Batch 3: Page tests
- [ ] Batch 4: E2E tests

---

## Verification Checklist

- [ ] Login with testuser@example.com works
- [ ] Signup creates new user
- [ ] Protected routes redirect to login
- [ ] API calls include Bearer token
- [ ] Logout clears session
- [ ] Token refresh works (if implemented)
- [ ] 401 responses trigger re-login

---

## Notes

- Keep phases small for context-window efficiency
- Test each phase before moving to next
- Use existing UI components (Button, Input, Card from Radix UI)
- Follow existing patterns (Zustand stores, React Query hooks)
