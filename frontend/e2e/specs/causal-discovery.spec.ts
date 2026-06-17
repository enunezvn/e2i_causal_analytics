import { test, expect } from '@playwright/test'
import { CausalDiscoveryPage } from '../pages/causal-discovery.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

/**
 * Causal Discovery is a validated-effects leaderboard: click "Discover causal
 * effects" and the agent validates + ranks each candidate question. These specs
 * assert the page's HONEST states — header, agent-driven badge, the run control,
 * and the empty state before a run. The full agent run (minutes) and the ranked
 * leaderboard rendering are covered by the component unit tests.
 */
test.describe('Causal Discovery Page', () => {
  let causalPage: CausalDiscoveryPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    causalPage = new CausalDiscoveryPage(page)
    await causalPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(causalPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(causalPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(causalPage.pageHeader).toBeVisible()
    })

    test('should display the confidence/impact ranking description', async () => {
      await expect(causalPage.pageDescription).toBeVisible()
    })

    test('should show the Agent-driven badge', async () => {
      await expect(causalPage.agentDrivenBadge).toBeVisible()
    })
  })

  test.describe('Run control', () => {
    test('should display the Discover causal effects button', async () => {
      await expect(causalPage.discoverButton).toBeVisible()
    })
  })

  test.describe('Honest empty state', () => {
    test('shows the empty state before any discovery is run', async () => {
      await expect(causalPage.emptyState).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await causalPage.goto()
      await expect(causalPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await causalPage.goto()
      await expect(causalPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await causalPage.goto()
      await expect(causalPage.mainContent).toBeVisible()
    })
  })
})
