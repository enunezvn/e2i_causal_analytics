// Playwright config used when an external dev/static server is already running
// (e.g. `npm run dev` in another shell, a `npx serve -s dist` for the built
// bundle, or a CI step that owns the server lifecycle). Skips the global
// `webServer` block defined in `playwright.config.ts` so Playwright does not
// try to spawn its own server.
//
// Usage:
//   npm run test:e2e:noserver -- <spec>
//   npx playwright test --config=playwright.noserver.config.ts <spec>
//
// See `e2e/README.md` for when to use this vs the default `test:e2e` script.
import baseConfig from './playwright.config'
import { defineConfig } from '@playwright/test'

export default defineConfig({
  ...baseConfig,
  webServer: undefined,
})
