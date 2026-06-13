/**
 * Resource Optimization Page E2E Tests (#19 coverage gap)
 * =======================================================
 *
 * `/resource-optimization` was a routed data page with NO e2e coverage. It is
 * the Tier-4 Resource Optimizer dashboard (scipy-backed mathematical
 * optimization). These specs stub the REAL endpoints the page calls
 * (`GET /api/resources/health`, `GET /api/resources/scenarios`) and assert
 * HONEST states:
 *   - solver ready -> "Solver Ready" badge
 *   - solver down -> "Solver Unavailable" badge (NOT a fake-ready badge)
 *   - no optimization run yet -> EmptyState "Run an optimization to see results"
 *
 * The health + scenarios responses are Zod-validated by the api-client
 * (ResourceHealthResponseWireSchema / ScenarioListResponseWireSchema), so the
 * stubs are faithful mirrors of the live contract.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { ResourceOptimizationPage } from '../pages/resource-optimization.page'
import { harnessBase } from '../fixtures/page-harness'

const HEALTH_READY = {
  status: 'healthy',
  agent_available: true,
  scipy_available: true,
  last_optimization: new Date().toISOString(),
  optimizations_24h: 3,
}

const HEALTH_DOWN = {
  status: 'degraded',
  agent_available: false,
  scipy_available: false,
  last_optimization: null,
  optimizations_24h: 0,
}

const SCENARIOS_EMPTY = {
  total_count: 0,
  scenarios: [],
}

async function stubResourceEndpoints(
  page: Page,
  opts: { down?: boolean } = {},
): Promise<void> {
  await page.route('**/api/resources/health**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opts.down ? HEALTH_DOWN : HEALTH_READY),
    })
  })

  await page.route('**/api/resources/scenarios**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(SCENARIOS_EMPTY),
    })
  })
}

test.describe('Resource Optimization Page', () => {
  let resPage: ResourceOptimizationPage

  test.describe('Solver ready', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubResourceEndpoints(page)
      resPage = new ResourceOptimizationPage(page)
      await resPage.goto()
    })

    test('loads at /resource-optimization', async ({ page }) => {
      await expect(page).toHaveURL(/resource-optimization/)
    })

    test('displays the page header', async () => {
      await expect(resPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(resPage.pageDescription).toBeVisible()
    })

    test('renders the "Solver Ready" badge from real health data', async () => {
      await expect(resPage.solverReadyBadge).toBeVisible()
    })

    test('shows honest empty state before any optimization is run', async () => {
      await expect(resPage.emptyState).toBeVisible()
    })
  })

  test.describe('Solver unavailable (falsifiability)', () => {
    test('renders the "Solver Unavailable" badge when the solver is down', async ({ page }) => {
      await harnessBase(page)
      await stubResourceEndpoints(page, { down: true })
      resPage = new ResourceOptimizationPage(page)
      await resPage.goto()

      await expect(resPage.solverUnavailableBadge).toBeVisible()
      await expect(resPage.solverReadyBadge).toBeHidden()
    })
  })
})
