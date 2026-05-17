// Local Playwright config used when an external dev/static server is already
// running (e.g. `npm run dev` in another shell, or a CI step that owns the
// server lifecycle). Skips the global `webServer` block defined in
// `playwright.config.ts` so Playwright does not try to spawn its own server.
//
// Usage: `npx playwright test --config=playwright.noserver.config.ts <spec>`
import baseConfig from './playwright.config'
import { defineConfig } from '@playwright/test'

export default defineConfig({
  ...baseConfig,
  webServer: undefined,
})
