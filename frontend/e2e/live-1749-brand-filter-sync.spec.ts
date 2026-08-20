/**
 * LIVE certification (NOT mocked) of the #1749 brand-filter sync fix against
 * the DEPLOYED site. Certifies the exact seam that was broken: the Home brand
 * dropdown and the copilot chat surfaces now share ONE filter state.
 *
 * Deterministic discriminators (both fail on the pre-fix bundle):
 *  1. Fresh Home load, chat opened → the POST /chat/suggestions opener payload
 *     must OMIT `brand` (pre-fix it always carried the hardcoded provider
 *     default `brand: "Remibrutinib"` regardless of the page).
 *  2. Select Brand: Kisqali on Home → a new opener request must fire carrying
 *     `brand: "Kisqali"` (pre-fix the dropdown wrote page-local useState only,
 *     so the payload stayed "Remibrutinib" — the reported defect).
 *
 * Run AFTER deploy (E2I_RUN_LIVE_CERTS is the explicit opt-in — without it
 * the spec skips even when credentials resolve, so a plain `npx playwright
 * test` on a box whose repo .env holds the password can never hit prod by
 * accident):
 *   E2I_RUN_LIVE_CERTS=1 BASE_URL=https://eznomics.site \
 *     npx playwright test --config playwright.noserver.config.ts \
 *     e2e/live-1749-brand-filter-sync.spec.ts --project=chromium --reporter=line
 *
 * Intentionally NOT under e2e/specs (which auto-applies api-mocks); raw runner
 * so the page talks to the REAL backend.
 */
import { test, expect, type Page, type Request } from '@playwright/test'
import { readFileSync } from 'node:fs'

const RUN_LIVE = process.env.E2I_RUN_LIVE_CERTS === '1'
const BASE = process.env.BASE_URL || 'https://eznomics.site'
const EMAIL = process.env.E2I_LOGIN_EMAIL || 'admin@e2i.local'

/** Same credential source as the deploy scripts: E2I_ADMIN_PASSWORD from the
 *  environment, falling back to the repo root .env (dotenv-style) so the
 *  secret never transits a shell command line. */
function resolvePassword(): string {
  if (process.env.E2I_ADMIN_PASSWORD) return process.env.E2I_ADMIN_PASSWORD
  for (const envPath of [
    '/home/enunez/Projects/e2i_causal_analytics/.env',
    `${process.cwd()}/../.env`,
  ]) {
    try {
      const line = readFileSync(envPath, 'utf8')
        .split('\n')
        .find((l) => l.startsWith('E2I_ADMIN_PASSWORD='))
      if (line) return line.slice('E2I_ADMIN_PASSWORD='.length).trim()
    } catch {
      /* next candidate */
    }
  }
  return ''
}
// Resolve the password only for opted-in runs: on the droplet the repo .env
// always resolves, and reading it during an ordinary CI/dev collection pass
// would arm a prod-facing spec nobody asked for.
const PASSWORD = RUN_LIVE ? resolvePassword() : ''

interface SuggestionsPayload {
  messages: unknown[]
  page: string
  brand?: string
  page_context?: string
}

async function login(page: Page): Promise<void> {
  await page.goto(`${BASE}/login`)
  await page.locator('#email').fill(EMAIL)
  await page.locator('#password').fill(PASSWORD)
  await page.locator('button[type="submit"]').filter({ hasText: /Sign in/i }).click()
  await page.waitForURL((u) => !u.pathname.includes('/login'), { timeout: 30000 })
}

function collectSuggestionPayloads(page: Page, sink: SuggestionsPayload[]): void {
  page.on('request', (req: Request) => {
    if (req.method() === 'POST' && req.url().includes('/chat/suggestions')) {
      try {
        sink.push(JSON.parse(req.postData() ?? '{}') as SuggestionsPayload)
      } catch {
        /* non-JSON body — ignore */
      }
    }
  })
}

async function openChat(page: Page): Promise<void> {
  // The FAB exposes no accessible name (icon-only button), so the class pair
  // that gives it its distinctive round 14x14 shape is the most stable handle
  // available today. If this breaks, prefer adding an aria-label app-side
  // over chasing new classes here.
  await page.locator('button.rounded-full.h-14').first().click()
  await expect(page.getByText('E2I Assistant')).toBeVisible({ timeout: 15000 })
}

test.describe('LIVE #1749 brand filter sync', () => {
  test.skip(
    !RUN_LIVE || !PASSWORD,
    'live cert is opt-in: set E2I_RUN_LIVE_CERTS=1 (and have E2I_ADMIN_PASSWORD resolvable)'
  )

  test('fresh Home load: opener /chat/suggestions omits brand (honest All default)', async ({
    page,
  }) => {
    const payloads: SuggestionsPayload[] = []
    collectSuggestionPayloads(page, payloads)

    await login(page)
    await page.goto(`${BASE}/`)
    await openChat(page)

    // Opener fetch is debounced 800ms and re-fires on page-context upgrades.
    await expect
      .poll(() => payloads.length, { timeout: 20000, message: 'opener request fired' })
      .toBeGreaterThan(0)

    for (const p of payloads) {
      // Pre-fix bundle: every one of these carried brand: 'Remibrutinib'.
      expect(p.brand, `payload must omit brand at default, got ${JSON.stringify(p.brand)}`).toBeUndefined()
    }
    // The page_context readable must state the honest default, not a brand.
    const withCtx = payloads.filter((p) => p.page_context)
    for (const p of withCtx) {
      expect(p.page_context).toContain('Brand filter: All')
      expect(p.page_context).not.toContain('Brand filter: Remibrutinib')
    }
  })

  test('selecting Kisqali on Home: opener request carries brand Kisqali', async ({ page }) => {
    const payloads: SuggestionsPayload[] = []
    collectSuggestionPayloads(page, payloads)

    await login(page)
    await page.goto(`${BASE}/`)

    // Select Brand: Kisqali in the Home dropdown. Semantic locator: the Radix
    // trigger has role=combobox and renders the selected brand label ('All
    // Brands' at the honest default), which distinguishes it from the Region
    // combobox next to it.
    await page
      .getByRole('combobox')
      .filter({ hasText: /^(All Brands|Remibrutinib|Fabhalta|Kisqali)/ })
      .first()
      .click()
    // Option a11y name includes the indication: "Kisqali (HR+/HER2- BC)".
    await page.getByRole('option', { name: /^Kisqali/ }).click()

    await openChat(page)

    await expect
      .poll(
        () => payloads.filter((p) => p.brand === 'Kisqali').length,
        { timeout: 20000, message: 'opener request with brand Kisqali fired' }
      )
      .toBeGreaterThan(0)

    const kisqali = payloads.filter((p) => p.brand === 'Kisqali')
    for (const p of kisqali.filter((q) => q.page_context)) {
      expect(p.page_context).toContain('Brand filter: Kisqali')
    }
    // No request after selection may still claim the old hardcoded default.
    expect(payloads.every((p) => p.brand !== 'Remibrutinib')).toBe(true)

    // Observational: record what the pills actually say (LLM pills vary; the
    // static fallback is deterministic — either way none should push the old
    // hardcoded Remibrutinib default).
    const pillTexts = await page
      .locator('button:has-text("📊"), button:has-text("📈"), button:has-text("🔍"), button:has-text("💡")')
      .allTextContents()
    console.log('[cert #1749] visible suggestion pills:', JSON.stringify(pillTexts))
  })
})
