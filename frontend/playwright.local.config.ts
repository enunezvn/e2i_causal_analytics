import baseConfig from './playwright.config'
import { defineConfig } from '@playwright/test'

export default defineConfig({
  ...baseConfig,
  // Skip Playwright's webServer — we manage the static server externally to
  // avoid contention with sibling sessions.
  webServer: undefined,
})
