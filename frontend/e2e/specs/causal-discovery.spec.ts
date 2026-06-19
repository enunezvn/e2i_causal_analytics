import { test, expect } from '@playwright/test'
import { CausalDiscoveryPage } from '../pages/causal-discovery.page'
import { harnessBase } from '../fixtures/page-harness'

/**
 * `/causal-discovery` is retired — unified into the agent-led `/causal-analysis`
 * page. This spec asserts the redirect: visiting the old route lands on
 * `/causal-analysis` with its header (NOT a 404 / NotFound). The leaderboard +
 * manual panel behavior is covered by causal-analysis.spec.ts and the component
 * unit tests.
 */
test.describe('Causal Discovery (retired → redirect)', () => {
  test('redirects /causal-discovery to /causal-analysis', async ({ page }) => {
    await harnessBase(page)
    const causalPage = new CausalDiscoveryPage(page)
    await page.goto(causalPage.url)
    await page.waitForLoadState('networkidle')
    await expect(page).toHaveURL(/causal-analysis/)
    await expect(causalPage.redirectedHeader).toBeVisible()
  })
})
