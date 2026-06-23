/**
 * LIVE validation (NOT mocked) of the gold-standard Feature-Importance page for
 * all 4 cohorts × 3 brands against the DEPLOYED site. This is the frontend proof
 * that the _BASE7 enrichment + hcp_adoption family fix actually render — the gap
 * the API-only checks could not close.
 *
 * Run AFTER deploy + the serving activation (bundles + Feast materialize +
 * bentoml restart + cache refresh):
 *
 *   BASE_URL=https://eznomics.site E2I_ADMIN_PASSWORD=... \
 *     npx playwright test --config playwright.noserver.config.ts \
 *     e2e/live-goldstd-validation.spec.ts --project=chromium --reporter=line
 *
 * It is intentionally NOT under e2e/specs (which auto-applies api-mocks); it uses
 * the raw @playwright/test runner so the page talks to the REAL backend.
 */
import { test, expect } from '@playwright/test'

const BASE = process.env.BASE_URL || 'https://eznomics.site'
const EMAIL = process.env.E2I_LOGIN_EMAIL || 'admin@e2i.local'
const PASSWORD = process.env.E2I_ADMIN_PASSWORD || ''

// Distinctive substrings for the 4 NEW prognostic drivers (T9/T11 _BASE7). At
// least one must appear in the rendered feature rankings for every patient
// cohort once the enrichment is live; pre-fix the page shows only the base 3.
const DRIVER_SUBSTRINGS = ['insurance', 'comorbidity', 'therapy', 'diagnosis']

const COHORTS = [
  { label: 'Initiation', patient: true },
  { label: 'Persistence', patient: true },
  { label: 'Discontinuation', patient: true },
  { label: 'HCP Adoption', patient: false },
]
const BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali']

async function login(page: import('@playwright/test').Page): Promise<void> {
  await page.goto(`${BASE}/login`)
  await page.locator('#email').fill(EMAIL)
  await page.locator('#password').fill(PASSWORD)
  await page.locator('button[type="submit"]').filter({ hasText: /Sign in/i }).click()
  await page.waitForURL((u) => !u.pathname.includes('/login'), { timeout: 30000 })
}

async function pickFromSelect(
  page: import('@playwright/test').Page,
  trigger: import('@playwright/test').Locator,
  optionLabel: string,
): Promise<void> {
  await trigger.click()
  await page.getByRole('option', { name: new RegExp(`^${optionLabel}$`, 'i') }).first().click()
}

test.describe('LIVE gold-standard Feature Importance', () => {
  test.skip(!PASSWORD, 'E2I_ADMIN_PASSWORD not set')

  for (const cohort of COHORTS) {
    for (const brand of BRANDS) {
      test(`${cohort.label} / ${brand} renders ${cohort.patient ? 'enriched 7-cov' : 'HCP 5-cov'}`, async ({
        page,
      }) => {
        await login(page)
        await page.goto(`${BASE}/feature-importance`)

        const modelSelector = page
          .locator('button.w-\\[190px\\], button:has-text("Select cohort")')
          .first()
        const brandSelector = page
          .locator('button.w-\\[150px\\], button:has-text("Select brand")')
          .first()

        await pickFromSelect(page, modelSelector, cohort.label)
        await pickFromSelect(page, brandSelector, brand)

        // Wait for the cohort feature-ranking rows to populate.
        await page
          .locator('.rounded-lg.cursor-pointer')
          .first()
          .waitFor({ state: 'visible', timeout: 45000 })

        const rows = page.locator('.rounded-lg.cursor-pointer')
        const text = (await rows.allInnerTexts()).join(' \n ').toLowerCase()
        const slug = `${cohort.label}_${brand}`.replace(/\s/g, '_')
        await page.screenshot({ path: `/tmp/fi_${slug}.png`, fullPage: true })
        // eslint-disable-next-line no-console
        console.log(`[${cohort.label}/${brand}] rankings text:\n${text.slice(0, 400)}`)

        // No "no data" / error masquerade.
        await expect(page.getByText(/failed|unavailable|no data|error/i)).toHaveCount(0)

        if (cohort.patient) {
          const found = DRIVER_SUBSTRINGS.filter((d) => text.includes(d))
          expect(
            found.length,
            `expected >=1 enriched driver in the rankings, found [${found}] for ${slug}`,
          ).toBeGreaterThan(0)
        } else {
          // HCP adoption: peer influence / network / specialty / experience render.
          expect(text).toMatch(/peer|influence|specialty|experience|network/i)
        }
      })
    }
  }
})
