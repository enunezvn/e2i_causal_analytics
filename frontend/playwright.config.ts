import { defineConfig, devices } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// frontend/package.json has "type": "module"; __dirname is not defined in ESM.
const __dirname = dirname(fileURLToPath(import.meta.url))

const quarantine = JSON.parse(
  readFileSync(join(__dirname, 'e2e/.quarantine.json'), 'utf8'),
) as { budget: number; specs: string[] }

/**
 * Playwright configuration for E2E testing.
 * See https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './e2e',
  testMatch: ['**/specs/**/*.spec.ts', '**/e2e/**/*.spec.ts'],
  // Specs listed in e2e/.quarantine.json are excluded until fixed. See
  // e2e/README.md for the un-quarantine protocol. Override locally with
  // E2E_QUARANTINE_OFF=1 npx playwright test <spec> for diagnosis.
  testIgnore: process.env.E2E_QUARANTINE_OFF
    ? []
    : quarantine.specs.map((f) => `**/specs/${f}`),
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 0 : 0,  // Disable retries to fail fast and get clear results
  workers: process.env.CI ? 1 : 4,
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
  ],
  timeout: 30000,
  expect: {
    timeout: 10000,
  },
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:5174',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 15000,
    navigationTimeout: 30000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // Firefox and WebKit are optional - only run when ALL_BROWSERS is set
    // In CI, we use Chromium only for faster, more reliable tests
    ...(process.env.ALL_BROWSERS
      ? [
          {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
          },
          {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
          },
        ]
      : []),
  ],
  webServer: {
    // In CI, serve the pre-built dist folder; locally, use dev server
    command: process.env.CI ? 'npx serve -s dist -l 5174' : 'npm run dev',
    url: 'http://localhost:5174',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
  outputDir: 'test-results/',
})
