import { test, expect } from '@playwright/test'
import { InterventionImpactPage } from '../pages/intervention-impact.page'
import { harnessBase } from '../fixtures/page-harness'

/**
 * `/intervention-impact` is retired (T10) — unified into `/causal-analysis`
 * (Treatment Effects tab) with the other tabs covered by their canonical pages.
 * This spec asserts the redirect: visiting the old route lands on
 * `/causal-analysis` with its header (NOT a 404 / NotFound).
 */
test.describe('Intervention Impact (retired → redirect)', () => {
  test('redirects /intervention-impact to /causal-analysis', async ({ page }) => {
    await harnessBase(page)
    const impactPage = new InterventionImpactPage(page)
    await page.goto(impactPage.url)
    await page.waitForLoadState('networkidle')
    await expect(page).toHaveURL(/causal-analysis/)
    await expect(impactPage.redirectedHeader).toBeVisible()
  })
})
