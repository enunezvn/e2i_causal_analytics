# Droplet Dependency Audit Plan

## ✅ COMPLETE (2026-01-08)

Audit of backend (Python) and frontend (npm) dependencies on the E2I droplet (159.89.180.27).

### Final Status

| Category | Status | Result |
|----------|--------|--------|
| **Frontend Security** | ✅ RESOLVED | 0 HIGH, 4 MODERATE (unfixable - CopilotKit transitive) |
| **Backend Conflicts** | ✅ OK | 2 known dependency conflicts (documented, stable) |
| **Outdated Packages** | ✅ UPDATED | Safe packages updated 2026-01-08 |
| **Test Environment** | ✅ FIXED | All 1706 tests passing (was 40 failures) |
| **Build Verification** | ✅ PASSED | Local and droplet builds successful |

---

## Execution Log (2026-01-08)

### Actions Completed

1. **HIGH vulnerability already fixed** - react-router-dom@7.12.0 (was 7.11.0)
2. **Build verification** - Passed after fixing TypeScript errors:
   - Removed unused `copilotKitHandlers` from handlers.ts
   - Removed unused `passthrough` import from handlers.ts
   - Synced vite.config.ts with underscore-prefixed unused params
3. **Safe packages updated**:
   - @tanstack/react-query: 5.90.12 → 5.90.16
   - react-hook-form: 7.69.0 → 7.70.0
   - zod: 4.2.1 → 4.3.5
   - typescript-eslint: 8.50.0 → 8.52.0
   - msw: 2.12.4 → 2.12.7
   - jsdom: 27.3.0 → 27.4.0
4. **Post-update build** - Passed
5. **Test environment fixed** - All 1706 tests now pass:
   - Added `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `VITE_API_URL` to vitest.config.ts
   - Added `renderWithAllProviders` wrapper to test utils (QueryClient + Router)
   - Updated Home.test.tsx to use proper provider wrapper
   - Updated env.ts to respect explicit `VITE_API_URL` in dev mode
6. **Droplet verification** - 51 test files, 1662 tests passed on droplet

### Remaining Vulnerabilities (4 MODERATE - unfixable)

All from CopilotKit transitive dependency chain:
```
@copilotkit/react-ui → react-syntax-highlighter → refractor → prismjs <1.30.0
```
**Status**: Requires CopilotKit upstream fix. Monitor for updates.

---

## Backend (Python) Audit

### Overview
- **Total Dependencies**: 40 direct, 332 locked (transitive)
- **Python Version**: 3.12+
- **Package Manager**: pip with pyproject.toml

### Installed Versions on Droplet (Verified)
| Package | Installed | Required | Status |
|---------|-----------|----------|--------|
| fastapi | 0.115.14 | >=0.115.0 | OK |
| pandas | 2.3.3 | >=2.2.0 | OK |
| mlflow | 3.7.0 | >=2.16.0 | OK |
| opik | 1.9.60 | >=0.2.0 | OK |

### Known Dependency Conflicts
1. **multiprocess/dill conflict** (pyproject.toml line 89)
   - `multiprocess>=0.70.16,<0.70.18` pinned to avoid dill>=0.4.0
   - Reason: feast requires dill~=0.3.0
   - Status: Documented, workaround in place

2. **tenacity conflict** (pyproject.toml line 83)
   - graphiti-core requires tenacity>=9.0.0
   - Some feast dependencies want older tenacity
   - Status: Documented, separate requirements-ragas.txt exists

### Backend Status: HEALTHY (No action required)

---

## Frontend (npm) Audit

### Overview
- **Total Dependencies**: 60 (33 production + 27 development)
- **Lock File**: package-lock.json (v3, 15k lines)
- **Node/npm**: Standard versions

### CRITICAL: Security Vulnerabilities

#### HIGH Severity (1)
**react-router-dom@7.11.0** - 3 security issues:
- CSRF in Action/Server Action Request Processing
- XSS via Open Redirects
- SSR XSS in ScrollRestoration

**Fix Available**: `npm audit fix` → updates to 7.12.0

#### MODERATE Severity (5)
**PrismJS DOM Clobbering** via dependency chain:
```
@copilotkit/react-ui → react-syntax-highlighter → refractor → prismjs <1.30.0
```
**Status**: Requires CopilotKit update (transitive dependency)

### Outdated Packages (17 total)

#### Safe to Update (patch/minor)
| Package | Current | Latest | Priority |
|---------|---------|--------|----------|
| react-router-dom | 7.11.0 | 7.12.0 | URGENT (security) |
| @tanstack/react-query | 5.90.12 | 5.90.16 | Low |
| react-hook-form | 7.69.0 | 7.70.0 | Low |
| zod | 4.2.1 | 4.3.5 | Low |
| typescript-eslint | 8.50.0 | 8.52.0 | Low |
| msw | 2.12.4 | 2.12.7 | Low |
| jsdom | 27.3.0 | 27.4.0 | Low |

#### Major Updates Available (requires testing)
| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| react | 18.3.1 | 19.2.3 | Major - defer |
| react-dom | 18.3.1 | 19.2.3 | Major - defer |
| vite | 6.4.1 | 7.3.1 | Major - defer |
| framer-motion | 11.18.2 | 12.24.12 | Major - defer |
| typescript | 5.6.3 | 5.9.3 | Minor - low risk |

---

## Recommended Actions

### Immediate (This Session)

1. **Fix HIGH security vulnerability**
   ```bash
   ssh root@159.89.180.27
   cd /root/Projects/e2i_causal_analytics/frontend
   npm audit fix
   ```

2. **Verify fix**
   ```bash
   npm audit
   npm run build
   npm run test:run
   ```

### Short-term (This Week)

3. **Update safe packages**
   ```bash
   npm update @tanstack/react-query react-hook-form zod typescript-eslint msw jsdom
   ```

4. **Run full test suite**
   ```bash
   npm run test:run
   npm run test:e2e
   ```

### Deferred (Next Quarter)

- React 18 → 19 migration (breaking changes)
- Vite 6 → 7 migration
- Monitor CopilotKit for PrismJS fix

---

## Verification Steps

After applying fixes:

1. **Security check**
   ```bash
   npm audit  # Should show 0 HIGH vulnerabilities
   ```

2. **Build verification**
   ```bash
   npm run build  # Should complete without errors
   ```

3. **Test suite**
   ```bash
   npm run test:run      # Unit tests
   npm run test:e2e      # E2E tests (if configured)
   ```

4. **Application check**
   - Verify frontend loads at http://localhost:5174
   - Check React Router navigation works
   - Verify no console errors

---

## Files Modified (2026-01-08)

### On Droplet (synced from local)
- `frontend/package.json` - version updates
- `frontend/package-lock.json` - lock file regeneration
- `frontend/vite.config.ts` - fixed unused variable warnings
- `frontend/vitest.config.ts` - added test environment variables
- `frontend/src/config/env.ts` - respect explicit VITE_API_URL in dev mode
- `frontend/src/test/utils.tsx` - added renderWithAllProviders wrapper
- `frontend/src/pages/Home.test.tsx` - use proper provider wrapper
- `frontend/src/mocks/handlers.ts` - removed unused copilotKitHandlers

### Commits
- `b93bd59` - fix(mocks): remove unused copilotKitHandlers and passthrough import
- `b0741ec` - fix(test): add test env vars and proper provider wrappers
