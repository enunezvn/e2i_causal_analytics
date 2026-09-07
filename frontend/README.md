# E2I Causal Analytics — Frontend

**Status**: Living reference | **Last verified against code**: 2026-09-07

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
| Charts / viz | Recharts (pages + the hand-tuned chat trend chart), D3, Cytoscape (knowledge graph); `flint-chart` → Plotly for chat-rendered charts | per-page, `src/lib/flint-chart.ts` |
| Chat | CopilotKit `1.51.2` (pinned **exact** — upgrade deliberately) over the AG-UI protocol | `src/providers/`, `src/components/chat/` |
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
├── hooks/          # use-auth and friends, plus…
│   └── api/        #   24 per-domain TanStack Query hooks (use-causal, use-kpi,
│                   #   use-segments, …). index.ts re-exports only 10 of them —
│                   #   import the rest from their own module.
├── config/env.ts   # accessor for the VITE_* env vars (two exceptions — see
│                   #   "Environment variables")
├── mocks/          # MSW: handlers.ts, browser.ts (dev), server.ts (vitest),
│                   #   MSWBanner + fixture data — dev/test only, never in dist/
├── test/           # vitest setup.ts, utils.tsx, __mocks__/ (incl. copilotkit),
│                   #   plus meta-tests that pin the e2e wiring & quarantine ratchet
├── assets/         # static imports bundled by Vite (currently just react.svg)
└── types/          # hand-written types + generated/api.ts (committed contract baseline)
```

---

## Routing & auth

- Route definitions and sidebar metadata live together in
  `src/router/routes.tsx`, but in **three separate hand-written lists** — there
  is no single registry, and missing one of the three is the usual "the page
  builds but the sidebar link 404s" bug. Adding a page means all three edits:

  1. **The `lazy()` import** at the top of the file (31 today) —
     `const MyPage = lazy(() => import('@/pages/MyPage'));`. This is what makes
     the page its own route chunk.
  2. **A `routeConfigs` entry** (`export const routeConfigs: RouteConfig[]`,
     L69) — `path`, `title`, `description`, plus the nav metadata `icon`,
     `section` (`main | causal | predictive | decisions | data | system`),
     `showInNav`, and `adminOnly`. It feeds **two** consumers, so it is not
     "the sidebar file":
     - the sidebar, via `getNavigationSections(includeAdmin)` (L636), which
       filters on a truthy `showInNav`, drops `adminOnly` entries unless
       admin, and groups the rest by `section`. `showInNav` is optional
       (`showInNav?: boolean`), so a page is kept out of the nav by omitting it
       or setting it `false` — today every entry sets it `true`;
     - the header page title, via `getRouteConfig(path)` (L608), consumed by
       `components/layout/Header.tsx:107`.

     So a page whose `routeConfigs` entry is missing does not fail loudly: the
     header falls back to the generic title `Dashboard` (`Header.tsx:108`,
     `currentRoute?.title ?? 'Dashboard'`) and the page silently renders under
     a plausible-looking wrong title. `RouteConfig.description` is carried but
     reaches no UI — `Header.tsx` is `getRouteConfig`'s only consumer, and it
     reads `.title` only. An entry here with no element below renders a link to
     nothing.
  3. **A `RouteObject` element** in `export const routes: RouteObject[]` (L285)
     — the actual react-router table. Protected pages wrap the component in
     `<ProtectedRoute>` (`<ProtectedRoute requireAdmin>` for `/admin`, L590)
     and then in `<LazyPage>`, which supplies the `Suspense` fallback; the
     public auth routes wrap in `<LazyPage>` only.

  Verify the shape before editing:
  `grep -nE '^export const (routeConfigs|routes)' src/router/routes.tsx`.
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
- **A 401 is refreshed once and the request replayed** (`api-client.ts`, PR #1891)
  — never by clearing the Supabase session. A 401 is rejected at the auth gate
  before any handler runs, so replaying any method is side-effect safe. The
  response interceptor resolves a fresher session (reading the stored one
  first, calling `refreshSession()` only when the stored token *is* the
  rejected one), stamps `e2iAuthReplayed` on the config so a second 401 is
  terminal, and shares **one** in-flight resolution per rejected token so a
  burst of 401s does not spend its single replay on a token the API just
  refused. Tests: `src/lib/api-client.refresh401.test.ts`.
- Theme default is **light** (`src/stores/ui-store.ts`) — users opt into
  dark/system via the header toggle; `ThemeManager` applies it to `<html>`.

---

## API layer & generated types

Hand-written request/response types in `src/types/*.ts` are the source of
truth **for application code**. Alongside them, `src/types/generated/api.ts`
is a **committed contract baseline** (re-tracked 2026-07): CI's
`verify-types` workflow regenerates it from the backend schema on every
schema-affecting PR and **fails if it differs** from the committed file.

```bash
make generate-types           # repo root — CI-identical static export (use this to fix a drift-gate failure)
npm run generate:types        # from a local running backend
npm run generate:types:prod   # from the live deployment
```

Working rules:

1. **A backend PR that changes the API schema must regenerate and commit
   `api.ts`** (`make generate-types`) — otherwise the verify-types drift gate
   fails. The diff doubles as a visible record of the contract change; update
   the affected hand-written types in the same PR.
2. When a page shows impossible values (e.g. a fabricated "0%" where the
   backend sent `null`), **diff the hand-written type against the generated
   one first** — hand-written/backend drift has caused exactly this class of
   production bug before.
3. Importing from `@/types/generated/api` no longer breaks CI (the file is
   always present now), but hand-written types remain the convention for
   application code.

Responses can be validated at runtime with the zod schemas in
`src/lib/api-schemas.ts` (opt-in `*Validated` helpers).

**Column labels are not a frontend concern.** Display names and definitions for
gold-standard treatment/outcome columns come from the backend SSOT
`src/insights/column_labels.py` (#1895), served as a `labels` map on
`GET /causal/variables` and `GET /segments/datasets`. Every page that prints a
column name goes through `columnLabel()` in `src/lib/column-labels.ts`, whose
fallback mirrors the backend auto-label byte-for-byte (underscores → spaces,
then Python `str.capitalize()`). Hard-coding a label in a component is how the
same column ended up reading differently on `/causal-analysis` and
`/segment-analysis`.

### Query-client and long-running analyses

Query defaults (`src/lib/query-client.ts`): 5-min `staleTime`, 10-min `gcTime`,
no retry on 4xx (except 408/429), exponential backoff, `refetchOnWindowFocus`
in prod only, and polling pauses while the tab is hidden.

**Mutations never retry** (`mutations.retry: 0`, #1846 → PR #1852). Mutations
are POSTs and almost none are idempotent: an axios client timeout
(`ECONNABORTED`) or a 5xx arrives *after* the server may already have queued
the job or written the row, so a blind re-run is a duplicate — and on the
heavy-compute endpoints the retry lands on the slot the first run still holds
and is rejected as "compute capacity saturated". `ERR_NETWORK` is not proof the
request never arrived either, and the offline case is already handled by
`networkMode: 'online'` pausing the mutation. A hook whose POST is *provably*
idempotent opts in with its own `retry`.

Long analyses are therefore **started once, then polled** — never re-POSTed:

- `POST /segments/analyze` returns a durable `analysis_id`;
  `waitForSegmentAnalysis()` (`src/api/segments.ts`) is **GET-only** and polls
  every 2 s.
- Ceilings are measured, not guessed (`SegmentAnalysis.tsx` L112–113): 300 s
  single-brand, 600 s all-brands (~2.8× the rows). Prod runs of the
  single-brand hierarchical CausalML uplift forest measured 73/125/153/208 s on
  2026-08-29/30; the earlier 120 s cap threw "timed out" on runs that then
  completed unseen (PR #1836).
- On expiry the page does **not** show a destructive error: it keeps the
  `analysis_id` and offers **"Keep waiting"**, which re-attaches by GET (issue
  #1841 → PR #1844). Running the analysis again would start a second
  server-side run.

---

## Chat (CopilotKit / AG-UI)

`CopilotKit` is mounted at the router root with
`runtimeUrl = ${apiUrl}/copilotkit/`; the chat UI is `E2IChatSidebar`
(mounted in `Layout.tsx`). `E2ICopilotProvider` registers the platform's agent
registry and the CopilotKit actions (in-chat navigation, filter changes, the
two chart actions below). Chat is toggled by `VITE_COPILOT_ENABLED` — when
unset it follows the build mode (**on in prod, off in dev**); CI e2e builds
force it off. The backend contract for everything below is
[`docs/api/chat.md`](../docs/api/chat.md).

> ⚠️ **CopilotKit hooks throw outside a `<CopilotKit>` provider** — and since
> chat is off in dev, that is the default local state. See *Gotchas*.

### What the page tells the agent

Three distinct channels, deliberately not one:

- **`state.filters` — the typed dashboard-filter channel.**
  `AgentFiltersBridge` (`src/components/chat/AgentFiltersBridge.tsx`, rendered
  by the provider) pushes the active `E2IFilters` into the `"default"` agent's
  CoAgent shared state via `useCoAgent`, and the backend declares a matching
  `filters` channel. The backend renders the non-sentinel values into an
  **ACTIVE DASHBOARD FILTERS** prompt suffix whose instruction is
  *resolve-don't-ask*, so "how are we doing?" answers for the selected brand
  instead of asking which one (PRs #1724, #1750, #1755). This channel is typed
  rather than a plain readable **because `"All"` / `"All US"` are
  no-selection sentinels** and a raw readable cannot say so. The bridge merges
  rather than clobbers (agent state also carries the actions channel) and keys
  its effect on the serialized filters, because `setState`'s identity follows
  live state and would otherwise loop.
- **`context` — the readables channel.** Every `useCopilotReadable` on the page
  rides the `agent/run` body and reaches the prompt as **ON-SCREEN APP
  CONTEXT** (PR #1818) — comments claiming readables never leave the browser
  were reading an echo of a payload the route had itself zeroed; measured
  2026-08-26, they do arrive. Values are stringified by
  `passThroughText` in `src/providers/copilotReadableConverters.ts`, which
  takes the **last** argument: CopilotKit 1.51.2 *types* `convert(description,
  value)` but its runtime calls `convert(value)`, so a positional
  `(description, value) =>` converter stringifies `undefined`.
- **`usePageChatContext(summary)` — a page's prose summary of what it is
  showing.** Eight pages publish one (Home, CausalAnalysis, FeatureImportance,
  GapAnalysis, PredictiveAnalytics, ResourceOptimization, SegmentAnalysis,
  TimeSeries). It also grounds the opener pills below.

### Suggestion pills

`ConversationSuggestions` in `E2IChatSidebar` fetches pills from
`POST /chat/suggestions` — openers grounded in the page context before the user
has typed, follow-ups from the transcript after. Passing a `suggestions` array
bypasses CopilotKit's own suggestion engine, which is deliberately **not**
enabled (`suggestions="auto"` would be a second LLM call per turn). The backend
grounds the prompt in a capability catalog and post-filters unsupported pills
(PR #1900); on a 502 the sidebar falls back to its static context-aware pills.

### Charts in chat

Two generative-UI actions, both of which **fetch the numbers themselves — the
model only says *what* to chart**:

| Action | Covers | Renders |
|---|---|---|
| `renderKpiTrend` | line-over-time of the six Rx-volume/commercial KPIs (`trx`, `nrx`, `nbrx`, `trx_share`, `conversion_rate`, `roi`), optionally split by severity tier or line of therapy | hand-tuned Recharts (`components/chat/KpiTrendChart.tsx`) |
| `renderChart` | every other registry KPI and every other shape — 12 chart types from `Line Chart` to `Heatmap`/`KPI Card` (PR #1383) | `flint-chart` → Plotly (`components/chat/FlintChart.tsx`) |

- **The KPI catalog is generated.** `src/lib/kpi-catalog.generated.ts` (45 KPIs
  with alias forms and a Flint semantic type) is produced from
  `config/kpi_definitions.yaml` by `scripts/gen_kpi_catalog.py`. Regenerate with
  `python3 scripts/gen_kpi_catalog.py` after editing the YAML —
  `kpi-catalog.test.ts` fails if it drifts. Do not hand-edit it.
  `src/lib/kpi-chart-router.ts` maps a named KPI to whichever endpoint can serve
  it (materialized history, patient-axis history, current value, batch compare).
- **The data rule** (`src/lib/flint-chart.ts`): Flint accepts inline rows, which
  makes "let the model write the spec" the obvious-looking integration and also
  the one that would let a model emit plausible-wrong pharma figures straight
  into a chart. Rows only ever enter through `assembleKpiFigure`, from real API
  responses.
- **The validation shim exists because Flint does not validate.** Measured
  against `flint-chart@0.4.1`: an encoding naming an absent field compiles
  silently to a nominal-typed spec and renders an empty chart; an unrecognised
  semantic type is likewise accepted. Both values originate with an LLM, so
  `validateChartRequest` checks them first and a bad request surfaces as an
  explicit error state instead of a blank plot.
- **Both the compiler and Plotly are dynamically imported.** Importing
  `lib/flint-chart` statically from the provider added ~136 kB gzip to the main
  chunk (1,054 → 1,191 kB — figure recorded at `components/chat/FlintChart.tsx:14`,
  not re-measured here); loading it inside `FlintChart` returns the main
  chunk to baseline. Keep it that way — this is why the router hands over a
  *logical* encoding rather than Flint template channels.

---

## Environment variables

All frontend env vars are `VITE_*` and are **bundled into the client at build
time — never put secrets here**. Precedence: `.env`/`.env.local` (gitignored)
→ `.env.production` (tracked, public build-time defaults) →
`.env.production.local` (gitignored).

| Variable | Purpose | Read in |
|---|---|---|
| `VITE_API_URL` | Backend base URL (default: relative `/api`; also the dev-proxy target) | `config/env.ts` |
| `VITE_SUPABASE_URL` | Supabase project URL (auth). Empty → `window.location.origin` — see the gotcha below | `config/env.ts` |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/publishable key (RLS-gated — safe to ship) | `config/env.ts` |
| `VITE_COPILOT_ENABLED` | Chat toggle; default follows prod/dev build mode | `config/env.ts` |
| `VITE_APP_VERSION` | Build metadata shown in the UI | `config/env.ts` |
| `VITE_DEBUG` | Forces the gated logger on in a production bundle | `lib/logger.ts` |
| `VITE_MSW_ENABLED` | Set to `false` to disable MSW when the mock worker is started; dev-only either way | `mocks/browser.ts` |

`src/config/env.ts` is the accessor for the first five. Note it reads the two
Supabase vars through the dynamic-key helper `getEnvVar('SUPABASE_URL')`, so a
literal `grep VITE_SUPABASE_URL src/config/env.ts` finds nothing. `lib/supabase.ts`
is the **only** consumer of `env.supabaseUrl` / `env.supabaseAnonKey`;
`providers/AuthProvider.tsx` imports the ready-made `supabase` client and
`isSupabaseConfigured` from it and never touches the env accessor. The only two vars read **outside**
`config/env.ts` are `VITE_DEBUG` and `VITE_MSW_ENABLED`; `api-client.ts` appears
to honour `VITE_DEBUG` but does so indirectly, through the logger.

`.env` hygiene: `frontend/.env` must contain **VITE-only** keys — see
[`docs/runbooks/frontend-env-and-csp.md`](../docs/runbooks/frontend-env-and-csp.md).

---

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server (port 5174, `/api` proxy) |
| `npm run build` | `tsc -b` + production build |
| `npm run preview` | `vite preview` — serve the built `dist/` locally |
| `npm run typecheck` | `tsc -b --noEmit` |
| `npm run lint` | ESLint |
| `npm run test` / `test:run` / `test:coverage` | Vitest (watch / once / with coverage) |
| `npm run test:ui` | Vitest in its browser UI |
| `npm run test:e2e` / `test:e2e:ui` | Playwright against a local build/dev server |
| `npm run test:e2e:noserver` | Playwright live specs against a running deployment |
| `npm run check:dist` | Asserts no MSW/dev artifacts in `dist/` |
| `npm run generate:types` / `generate:types:prod` | Regenerate OpenAPI types (committed baseline; prefer `make generate-types` for CI parity) |

---

## Testing

- **Unit (Vitest)**: jsdom + Testing Library + MSW; tests colocated under
  `src/`. Coverage thresholds are enforced in `vitest.config.ts`.
- **E2E (Playwright)**: page specs in `e2e/specs/`, Chromium-only and sharded
  in CI. A **quarantine ratchet** (`e2e/.quarantine.json`) lets flaky specs be
  quarantined but only ever shrinks in CI; `_smoke.spec.ts` can never be
  quarantined.
- **Live validation** (`playwright.noserver.config.ts`, which is
  `playwright.config.ts` with `webServer: undefined`): the four `e2e/live-*.spec.ts`
  files run against a **real running deployment**, not a local build — they are
  post-deploy certification. Note they are **not excluded at config level**:
  `testMatch` collects `**/e2e/**/*.spec.ts` and `testIgnore` drops only
  quarantined `**/specs/` entries, so CI's bare `npx playwright test --shard=N/4`
  does distribute them across the shards. What keeps them inert there is the
  runtime skip below, not a filter.

  | Spec | Env it needs |
  |---|---|
  | `live-goldstd-validation.spec.ts` | `BASE_URL`, `E2I_LOGIN_EMAIL`, `E2I_ADMIN_PASSWORD` |
  | `live-1749-brand-filter-sync.spec.ts` | the same **plus `E2I_RUN_LIVE_CERTS=1`** |
  | `live-1752-1753-filter-sync.spec.ts` | the same **plus `E2I_RUN_LIVE_CERTS=1`** |
  | `live-digital-twin-8-interventions.spec.ts` | `BASE_URL`, `E2I_LOGIN_EMAIL`, `E2I_LOGIN_PASSWORD` |

  Every one of them **skips rather than fails** when its credentials are
  missing, and the two filter-sync certs skip even with credentials unless
  `E2I_RUN_LIVE_CERTS=1` is set explicitly — so a bare `npx playwright test`
  never fires them by accident. **A skipped live cert is not a pass**: check the
  run reports the tests as executed. See root `.env.example` §7 for the admin
  credentials' other consumers.

---

## CI & deploy

The **Frontend Tests** workflow (`.github/workflows/frontend-tests.yml`,
triggered by `frontend/**`) runs: quarantine ratchet → lint + typecheck →
unit tests with coverage → production build (chat forced off) + `check:dist`
→ sharded Playwright e2e → `e2e-report` → a `ci-success` gate job.

Deploys: `frontend/**` is in `deploy.yml`'s trigger paths, so **any merge to
`main` touching `frontend/` fires a production deploy**. The
`build-and-push-frontend` job builds `docker/frontend/Dockerfile`
(`target: production`, context = repo root) and pushes
`ghcr.io/<owner>/e2i-frontend` tagged with the commit SHA; the droplet pulls
the pre-built image (`--no-build`). The production stage is `nginx:alpine`
serving the static build (non-root, `/health` endpoint), and the image build
**fails on dev artifacts**: a bundled `mockServiceWorker.js`, MSW code in
`dist/assets`, or `localhost:8443` (the dev Supabase URL) baked into the
bundle. Production `VITE_*` values come from the tracked
`frontend/.env.production` at build time.

Two gates worth knowing when a frontend deploy misbehaves:

- **`ensure-main-image`** (issue #1780 → PR #1782) runs after both build jobs
  and before the SSH step. If `main` moved past the sha this run built, it
  builds and pushes images for the *new* `main` HEAD, so the droplet never has
  to build locally. It counts a sha as built only when **both** `e2i-api` and
  `e2i-frontend` have a manifest at that exact tag, tags the sha only (never
  `latest`), and is deliberately **fail-soft** — every give-up path degrades to
  the pre-#1780 behaviour rather than blocking the deploy, so a red step here is
  an operator signal, not a failed deploy.
- **The health gate and rollback.** After the flip the deploy polls
  `http://localhost:8000/health` for up to 30 × 2 s. On failure it calls
  `rollback_to_prev api frontend worker_light worker_medium scheduler`, which
  re-pulls the **previous sha's** GHCR images (`--no-build`: the React/esbuild
  build is the OOM that turned the 2026-06-23 rollback into a double-fault) and
  then exits 1. So a rollback target only exists if that sha's frontend image is
  in GHCR — which is exactly what `ensure-main-image` guarantees.

See root `DEPLOYMENT.md` for the full pipeline (migrations, feast gates,
BentoML refresh) and
[`docs/runbooks/frontend-serving-flip.md`](../docs/runbooks/frontend-serving-flip.md)
for serving-mode changes.

---

## Gotchas

- **`useCopilotReadable` / `useCopilotAction` throw outside a `<CopilotKit>`
  provider — and chat is OFF in dev.** So a page that calls one unconditionally
  works in prod and crashes on the developer's machine (and in CI, which builds
  with chat forced off). Gate every call behind
  `useCopilotEnabled()` from `src/providers/E2ICopilotProvider.tsx` (defined at
  L448, re-exported from `src/providers/index.ts`). It reads a context that
  defaults to `false`, so it is safe *itself* to call anywhere.

  The rules of hooks mean the gate has to be a **component boundary**, not an
  `if` around the hook call. Both call sites that register throwing hooks do it
  the same way: the provider's `CopilotHooksConnector` returns `null` instead of
  rendering `CopilotHooksInner`, and `PredictiveAnalytics` renders
  `{copilotEnabled && <CohortReadable … />}` — the readable lives in that child
  (PR #1818). `E2IChatSidebar` and `E2IChatPopup` use the same flag to decide
  whether to mount chat at all; `use-e2i-filters`, `use-e2i-highlights` and
  `use-user-preferences` use it to fall back to local state off-provider.
  `grep -rln useCopilotEnabled src/ | grep -v test` lists 8 files — 7 call sites
  plus the `providers/index.ts` re-export.
- **`VITE_SUPABASE_URL` is intentionally EMPTY in `.env.production` — do not
  "fix" it.** `config/env.ts` treats an empty value as unset and falls back to
  `window.location.origin`, so supabase-js targets the same origin the app is
  served from; host nginx (`docker/nginx/host-nginx.conf`) proxies the paths
  supabase-js v2 appends (`/auth/v1`, `/rest/v1`, `/realtime/v1`, `/storage/v1`)
  to the self-hosted Kong on `127.0.0.1:54321`. This resolved a mixed-content
  login failure. The empty mode-specific value also **shadows** any
  `VITE_SUPABASE_URL` in a local untracked `frontend/.env` during `vite build`
  (verified on vite 6.4.1), so dev values cannot leak into a prod bundle — and
  the Dockerfile fails the image build if `localhost:8443` reaches the assets.
- **No `manualChunks` / vendor split** in `vite.config.ts` — a forced React
  vendor chunk broke CJS→ESM interop and blanked the page (PR #919, reverted).
  Code-splitting is per-route via `React.lazy` only.
- The CopilotKit packages are **pinned to an exact version** (`1.51.2`) —
  upgrade deliberately, not via `^` drift. Its runtime and its types disagree in
  at least one place (see the readables converter above), so an upgrade needs
  the chat paths re-exercised, not just a green typecheck.
- MSW is dev-only; the prod build strips the worker and `check:dist` + the
  Docker build guards enforce it.
- When verifying a code-split page landed in prod, grep for its **lazy chunk**
  (or a string literal from the page), not `index-*.js`.

---

## Cross-reference

- Root `README.md` — platform overview (agents, tiers, backend stack)
- `DEPLOYMENT.md` — the CI/CD deploy pipeline this frontend ships through
- [`docs/api/chat.md`](../docs/api/chat.md) — the backend contract behind the
  chat sidebar: AG-UI endpoints, the `state.filters` and `context` channels,
  the empty-delta invariant, bound tools, and `/chat/suggestions`
- [`docs/runbooks/frontend-env-and-csp.md`](../docs/runbooks/frontend-env-and-csp.md)
  — `.env` hygiene and CSP reconciliation
- [`docs/runbooks/frontend-serving-flip.md`](../docs/runbooks/frontend-serving-flip.md)
  — changing how the built bundle is served
- `src/types/generated/README.md` — the full generated-types contract
- `e2e/README.md` — when to use `test:e2e` vs `test:e2e:noserver`
