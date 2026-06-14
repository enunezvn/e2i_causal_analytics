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

// The HTML-only smoke above is NOT sufficient: the <title> "E2I Causal
// Analytics" lives statically in index.html, so it stays green even when the
// React app fails to MOUNT (blank white page). That exact gap let a prod-only
// blank page ship (a manualChunks vendor-react split broke React's CJS init in
// the production rollup bundle — invisible to dev/vitest, and the title check
// passed anyway). This block closes the gap: it asserts React actually mounts
// (#root gets children) and that React's init does not throw. It MUST run
// against the PRODUCTION bundle (`npx serve -s dist`, i.e. CI), where the bug
// manifested — the dev server does not use manualChunks.
test.describe('Smoke — React app actually mounts (prod bundle)', () => {
  for (const path of ['/', '/login']) {
    test(`GET ${path} mounts React (#root populated, no init error)`, async ({ page }) => {
      const fatalErrors: string[] = []
      page.on('pageerror', (e) => fatalErrors.push(String(e?.message ?? e)))
      page.on('console', (m) => {
        if (m.type() === 'error') fatalErrors.push(m.text())
      })

      await page.goto(path, { waitUntil: 'load' })

      // React mounts into <div id="root">. A blank page leaves it empty.
      await expect(page.locator('#root')).not.toBeEmpty({ timeout: 15000 })

      // The blank-page signature: React core failing to initialize.
      const blankSig = fatalErrors.filter((e) =>
        /Cannot set properties of undefined|setting 'Children'/.test(e),
      )
      expect(blankSig, `React init error(s): ${blankSig.join(' | ')}`).toHaveLength(0)
    })
  }
})
