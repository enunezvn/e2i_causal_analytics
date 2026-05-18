import { test, expect } from '@playwright/test'

// MUST NOT be added to e2e/.quarantine.json. The quarantine-ratchet CI job
// fails if this file appears in the manifest. Reason: with all 16 broken
// specs excluded via testIgnore, Playwright would otherwise collect 0 tests
// and exit non-zero ("No tests found"), keeping e2e-tests RED even when the
// quarantine is doing its job. The smoke spec is the floor signal — it only
// asserts that the SPA shell renders, so ProtectedRoute redirects to /login
// still pass (the login page is served with a non-empty <title>).
//
// 4+ tests so each of the 4 e2e-tests shards collects at least one.
test.describe('Smoke — routes serve HTML', () => {
  const paths = ['/', '/login', '/causal-discovery', '/monitoring']
  for (const path of paths) {
    test(`GET ${path} returns < 400 with a non-empty <title>`, async ({ page }) => {
      const response = await page.goto(path, { waitUntil: 'domcontentloaded' })
      expect(response?.status() ?? 0).toBeLessThan(400)
      await expect(page).toHaveTitle(/.+/)
    })
  }
})
