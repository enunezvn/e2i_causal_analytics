# Frontend e2e Tests (Playwright)

End-to-end specs live in `e2e/specs/`, page objects in `e2e/pages/`, shared
fixtures in `e2e/fixtures/`, shared utilities in `e2e/utils/`.

## Running

Two scripts are wired in `package.json`:

| Script | When to use | Server lifecycle |
|---|---|---|
| `npm run test:e2e` | Default. Local dev or CI. | Playwright spawns its own server per `playwright.config.ts` (`webServer.command`): `npm run dev` locally, `npx serve -s dist -l 5174` in CI. Honors `reuseExistingServer: !process.env.CI`, so it will attach to an existing local server on port 5174 instead of spawning a duplicate. |
| `npm run test:e2e:noserver` | You already manage the server externally. | Playwright never spawns or attaches a `webServer`; it just hits the `baseURL` you point it at. Useful when (a) the dev/static server is on a non-default port, (b) you want fast Playwright startup with no spawn overhead, or (c) multiple concurrent test runs are sharing one externally-managed server. |

### Examples

```bash
# Default (Playwright owns the server)
npm run test:e2e
npm run test:e2e -- e2e/specs/data-quality.spec.ts

# External server (you own the server)
npx serve -s dist -l 5174 &
npm run test:e2e:noserver -- e2e/specs/data-quality.spec.ts
```

To customize the URL Playwright connects to, set `BASE_URL` before invoking, or
edit `playwright.config.ts`.

## Quarantine

Specs listed in `e2e/.quarantine.json` are excluded from collection via
`testIgnore` in `playwright.config.ts`. The list exists because the 2026-05-17
live-wiring wave (PRs #312–#320) broke 8 specs, and 8 more pre-wave specs have
been broken since at least 2026-02-02 (issue #332). Without quarantine, the
`e2e-tests` workflow stays RED on every PR and the signal is muted by routine
admin-merges.

The manifest has two fields:

- `budget` (int) — the maximum number of quarantined specs allowed. Enforced
  one-way down by the `quarantine-ratchet` CI job.
- `specs` (string[]) — filenames in `e2e/specs/` that are currently excluded.
  `len(specs)` must equal `budget`.

### To un-quarantine a spec

In a single PR:

1. Delete the spec's entry from `.quarantine.json` `specs`.
2. Decrement `budget` by 1.
3. Fix the spec so it passes against current main.

The CI ratchet rejects any PR that raises `budget` above the base ref's
`budget`, and rejects any state where `budget != len(specs)`.

### To run a quarantined spec locally for diagnosis

```bash
E2E_QUARANTINE_OFF=1 npx playwright test e2e/specs/<spec-file>
```

This bypasses the manifest's `testIgnore` for the current shell only.

### `_smoke.spec.ts` is special

`_smoke.spec.ts` is the floor signal — four route-availability checks that
don't depend on auth seeding or API mocks. It MUST NOT be added to the
quarantine list (the ratchet enforces this). It exists so the `e2e-tests`
shards have something to collect while the quarantine list is non-empty
(Playwright exits non-zero on "no tests found").

## Adding a spec

1. Create `e2e/specs/<area>.spec.ts` (+ optional `e2e/pages/<area>.page.ts` for
   page objects).
2. Use `e2e/fixtures/api-mocks.ts` for shared API stubbing patterns, or
   `page.route()` inline in a `beforeEach` for spec-local mocks (see
   `Shared fixtures vs inline overrides` below).
3. Run locally with either `test:e2e` script.

## Protocol for live-wiring PRs

When you wire `src/pages/Foo.tsx` from mocked data to a live BE endpoint, you
MUST do ONE of:

- Update `e2e/specs/foo.spec.ts` and `e2e/pages/foo.page.ts` in the SAME PR to
  assert against the live-wired DOM, OR
- Add `foo.spec.ts` to `e2e/.quarantine.json` (bump `budget`) and open a
  tracking issue with labels `bug,ci,frontend,e2e-quarantine`.

Admin-merging through a freshly-red e2e baseline is not acceptable — it
permanently mutes the regression signal for everyone else.

## Shared fixtures vs inline overrides

`e2e/fixtures/api-mocks.ts` is consumed by every spec via
`await mockApiRoutes(page)` in `beforeEach`. It mocks the cross-page endpoints
(auth, copilotkit, kpis, etc.). Per-spec endpoints should be stubbed inline via
`page.route()` registered AFTER `mockApiRoutes(page)` — Playwright's
last-registered-route-wins dispatch resolves the per-spec stub first. Avoid
editing `api-mocks.ts` from a per-spec PR; its blast radius spans 15+ specs.

## Configs

- `playwright.config.ts` — base config (web server, browsers, sharding,
  timeout, quarantine `testIgnore`).
- `playwright.noserver.config.ts` — extends the base, sets
  `webServer: undefined`. Invoked by `test:e2e:noserver`.

Do not introduce a third variant. Extend one of the two above (or wire a new
script into `package.json` + a row in the "Running" table above).
