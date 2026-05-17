# Frontend e2e Tests (Playwright)

End-to-end specs live in `e2e/specs/`, page objects in `e2e/pages/`, fixtures in
`e2e/fixtures/`, shared utilities in `e2e/utils/`.

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

## Adding a spec

1. Create `e2e/specs/<area>.spec.ts` (+ optional `e2e/pages/<area>.page.ts` for
   page objects).
2. Use `e2e/fixtures/api-mocks.ts` for shared API stubbing patterns, or
   `page.route()` inline in a `beforeEach` for spec-local mocks. Do NOT modify
   `fixtures/api-mocks.ts` for spec-local needs — it is consumed by many specs.
3. Run locally with either script above.

## Configs

- `playwright.config.ts` — base config (web server, browsers, sharding, timeout).
- `playwright.noserver.config.ts` — extends the base, sets `webServer: undefined`. Invoked by `test:e2e:noserver`.

Do not introduce a third variant. Extend one of the two above (or wire a new
script into `package.json` + a row in the table above).
