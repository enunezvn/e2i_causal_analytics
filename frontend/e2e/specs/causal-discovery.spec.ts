import { test, expect } from '@playwright/test'
import { CausalDiscoveryPage } from '../pages/causal-discovery.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

/**
 * Causal Discovery is now an agent-driven one-click flow: pick the causal
 * question, the agent learns the DAG from data + estimates + refutes. These
 * specs assert the page's HONEST states — header, agent-driven badge, the
 * question form, and the empty state before a run. The rendered result (learned
 * DAG, effect, gate) is covered by the component unit tests (jsdom); the e2e
 * here locks the page loads and presents the agent-driven entry point.
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

    test('should display the agent-driven page description', async () => {
      await expect(causalPage.pageDescription).toBeVisible()
    })

    test('should show the Agent-driven badge', async () => {
      await expect(causalPage.agentDrivenBadge).toBeVisible()
    })
  })

  test.describe('Question form (the only user input)', () => {
    test('should display the treatment selector', async () => {
      await expect(causalPage.treatmentSelect).toBeVisible()
    })

    test('should display the outcome selector', async () => {
      await expect(causalPage.outcomeSelect).toBeVisible()
    })

    test('should display the Discover & Analyze button', async () => {
      await expect(causalPage.analyzeButton).toBeVisible()
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
