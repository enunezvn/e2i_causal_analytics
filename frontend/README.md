# E2I Causal Analytics — Frontend

**Status**: Living reference | **Last verified against code**: 2026-07-18

React 18 + TypeScript single-page app for the E2I Causal Analytics platform:
30+ lazy-loaded pages across six navigation sections (Causal Analytics,
Predictive Modeling, Decisions & Optimization, Data & Reference, System &
Platform, plus the home/main group), an embedded CopilotKit chat sidebar, and
Supabase (GoTrue) authentication. The **code is the source of truth** — file
paths below point at it.

---

## Quick start

Prerequisites: Node 20, npm.

```bash
cd frontend
npm ci
cp .env.example .env          # then fill in the VITE_* values (see below)
npm run dev                   # Vite dev server on http://localhost:5174
```

The dev server runs on port **5174** (not Vite's default 5173 —
`vite.config.ts`) and proxies `/api` to `VITE_API_URL` (default
`http://localhost:8000`), so a locally running backend is picked up
automatically.

---

## Stack

| Concern | Choice | Where |
|---|---|---|
| Build | Vite 6, TypeScript ~5.6 (`tsc -b` before build) | `vite.config.ts` |
| Routing | react-router-dom 7, `createBrowserRouter`, **every route lazy-loaded** | `src/router/` |
| Server state | TanStack Query 5 | `src/lib/query-client.ts` |
| Client state | Zustand 5 (`auth-store`, `filter-store`, `ui-store`) | `src/stores/` |
| Styling / UI | Tailwind CSS v4 + shadcn/ui (`new-york`, Radix primitives, lucide icons) | `tailwind.config.js`, `components.json`, `src/components/ui/` |
| Charts / viz | Recharts, Plotly, D3, Cytoscape (knowledge graph) | per-page |
| Chat | CopilotKit (pinned exact version — upgrade deliberately) | `src/providers/`, `src/components/chat/` |
| Auth | @supabase/supabase-js (GoTrue) | `src/lib/supabase.ts` |
| HTTP + validation | axios + zod response schemas | `src/lib/api-client.ts`, `src/lib/api-schemas.ts` |
| Tests | Vitest (jsdom + MSW), Playwright e2e | `vitest.config.ts`, `playwright.config.ts` |

There is **no prettier** — linting is ESLint (flat config, `eslint.config.js`).

---

## Project layout

```
frontend/src/
├── pages/          # one component per page (30+), all React.lazy route chunks
├── router/         # index.tsx (createBrowserRouter) + routes.tsx (route +
│                   #   nav metadata; getNavigationSections() drives the sidebar)
├── api/            # per-domain API modules (causal, kpi, segments, admin, …)
│                   #   with colocated *.test.ts
├── lib/            # api-client (axios + auth interceptor), api-schemas (zod),
│                   #   query-client, supabase, utils
├── stores/         # zustand: auth-store, filter-store, ui-store (persisted)
├── components/     # ui/ (shadcn), layout/, auth/ (ProtectedRoute), chat/, theme/
├── providers/      # AuthProvider, E2ICopilotProvider (agent registry + actions)
├── hooks/          # use-auth and friends
├── config/env.ts   # single accessor for all VITE_* env vars
└── types/          # hand-written committed types + generated/ (gitignored)
```

---

## Routing & auth

- Route definitions and sidebar metadata live together in
  `src/router/routes.tsx` (`routeConfigs`); the sidebar renders whatever
  `getNavigationSections()` returns — add a page by adding one entry there.
- Public routes: `/login`, `/signup`, `/forgot-password`, `/reset-password`,
  `/accept-invite`. Everything else sits behind
  `src/components/auth/ProtectedRoute.tsx`, which **fails closed**: a
  config-error screen if Supabase env is unset, spinner until the auth store
  initializes, redirect to `/login` (preserving the destination) when
  unauthenticated. `/admin` additionally requires `requireAdmin`.
- The Supabase client is a lazy singleton (`src/lib/supabase.ts`), session
  persisted in localStorage under `e2i-auth-token`. The axios interceptor in
  `src/lib/api-client.ts` attaches `Authorization: Bearer <access_token>` on
  every API call (plus an `X-Correlation-ID`), and never attaches a token when
  Supabase is unconfigured.
- Theme default is **light** (`src/stores/ui-store.ts`) — users opt into
  dark/system via the header toggle; `ThemeManager` applies it to `<html>`.

---

## API layer & generated types

Hand-written request/response types in `src/types/*.ts` are the **committed**
source of truth for the compiler. OpenAPI-generated types exist as a local
verification aid:

```bash
npm run generate:types        # from a local backend  → src/types/generated/api.ts
npm run generate:types:prod   # from the live deployment
```

`src/types/generated/api.ts` is **gitignored and not generated in CI**. Two
hard rules (from `src/types/generated/README.md`):

1. **Never commit a `src/` file that imports `@/types/generated/api`** — CI
   type-checks without the generated file, so the import is an instant TS2307
   build failure. Use generated types locally to diff against the hand-written
   ones, then fix the hand-written type.
2. When a page shows impossible values (e.g. a fabricated "0%" where the
   backend sent `null`), **diff the hand-written type against the generated
   one first** — hand-written/backend drift has caused exactly this class of
   production bug before.

Responses can be validated at runtime with the zod schemas in
`src/lib/api-schemas.ts` (opt-in `*Validated` helpers).

Query-client defaults (`src/lib/query-client.ts`): 5-min `staleTime`, 10-min
`gcTime`, no retry on 4xx (except 408/429), exponential backoff,
`refetchOnWindowFocus` in prod only, and polling pauses while the tab is
hidden.

---

## Chat (CopilotKit)

`CopilotKit` is mounted at the router root with
`runtimeUrl = ${apiUrl}/copilotkit/`; the chat UI is `E2IChatSidebar`
(mounted in `Layout.tsx`). `E2ICopilotProvider` registers the platform's
agent registry and CopilotKit actions (in-chat navigation, filter changes,
inline KPI trend charts). Chat is toggled by `VITE_COPILOT_ENABLED` — when
unset it follows the build mode (**on in prod, off in dev**); CI e2e builds
force it off.

---

## Environment variables

All frontend env vars are `VITE_*` and are **bundled into the client at build
time — never put secrets here**. Precedence: `.env`/`.env.local` (gitignored)
→ `.env.production` (tracked, public build-time defaults) →
`.env.production.local` (gitignored). Accessors live in `src/config/env.ts`.

| Variable | Purpose |
|---|---|
| `VITE_API_URL` | Backend base URL (default: relative `/api`; also the dev-proxy target) |
| `VITE_SUPABASE_URL` | Supabase project URL (auth) |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/publishable key (RLS-gated — safe to ship) |
| `VITE_COPILOT_ENABLED` | Chat toggle; default follows prod/dev build mode |
| `VITE_APP_VERSION` | Build metadata shown in the UI |
| `VITE_DEBUG` | Verbose client logging (dev only) |
| `VITE_MSW_ENABLED` | Dev-only Mock Service Worker; must be false/unset in prod |

---

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server (port 5174, `/api` proxy) |
| `npm run build` | `tsc -b` + production build |
| `npm run typecheck` | `tsc -b --noEmit` |
| `npm run lint` | ESLint |
| `npm run test` / `test:run` / `test:coverage` | Vitest (watch / once / with coverage) |
| `npm run test:e2e` / `test:e2e:ui` | Playwright against a local build/dev server |
| `npm run test:e2e:noserver` | Playwright live specs against a running deployment |
| `npm run check:dist` | Asserts no MSW/dev artifacts in `dist/` |
| `npm run generate:types` / `generate:types:prod` | Regenerate OpenAPI types (local aid, gitignored) |

---

## Testing

- **Unit (Vitest)**: jsdom + Testing Library + MSW; tests colocated under
  `src/`. Coverage thresholds are enforced in `vitest.config.ts`.
- **E2E (Playwright)**: page specs in `e2e/specs/`, Chromium-only and sharded
  in CI. A **quarantine ratchet** (`e2e/.quarantine.json`) lets flaky specs be
  quarantined but only ever shrinks in CI; `_smoke.spec.ts` can never be
  quarantined.
- **Live validation** (`playwright.noserver.config.ts`): specs like
  `e2e/live-goldstd-validation.spec.ts` run against the real deployment. They
  need `BASE_URL`, `E2I_LOGIN_EMAIL`, and `E2I_ADMIN_PASSWORD` (the spec
  **skips** when the password is unset). See root `.env.example` §7 for the
  admin credentials' other consumers.

---

## CI & deploy

The **Frontend Tests** workflow (`.github/workflows/frontend-tests.yml`,
triggered by `frontend/**`) runs: quarantine ratchet → lint + typecheck →
unit tests with coverage → production build (chat forced off) + `check:dist`
→ sharded Playwright e2e → a `ci-success` gate job.

Deploys: `frontend/**` is in `deploy.yml`'s trigger paths, so **any merge to
`main` touching `frontend/` fires a production deploy**. The
`build-and-push-frontend` job builds `docker/frontend/Dockerfile`
(`target: production`, context = repo root) and pushes
`ghcr.io/<owner>/e2i-frontend` tagged with the commit SHA; the droplet pulls
the pre-built image (`--no-build`). The production stage is `nginx:alpine`
serving the static build (non-root, `/health` endpoint), and the image build
**fails on dev artifacts**: a bundled `mockServiceWorker.js`, MSW code in
`dist/assets`, or a dev Supabase URL baked into the bundle. Production
`VITE_*` values come from the tracked `frontend/.env.production` at build
time. See root `DEPLOYMENT.md` for the full pipeline (migrations, rollout
gates, rollback).

---

## Gotchas

- **No `manualChunks` / vendor split** in `vite.config.ts` — a forced React
  vendor chunk broke CJS→ESM interop and blanked the page (#919, reverted).
  Code-splitting is per-route via `React.lazy` only.
- The CopilotKit packages are **pinned to an exact version** — upgrade
  deliberately, not via `^` drift.
- MSW is dev-only; the prod build strips the worker and `check:dist` + the
  Docker build guards enforce it.
- When verifying a code-split page landed in prod, grep for its **lazy chunk**
  (or a string literal from the page), not `index-*.js`.

---

## Cross-reference

- Root `README.md` — platform overview (agents, tiers, backend stack)
- `DEPLOYMENT.md` — the CI/CD deploy pipeline this frontend ships through
- `src/types/generated/README.md` — the full generated-types contract
