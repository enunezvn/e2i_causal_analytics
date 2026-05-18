import { test, expect, type Page, type Route } from '@playwright/test'
import { SystemHealthPage } from '../pages/system-health.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// Inline MSW-style stubs for endpoints the System Health page hits via
// useAlerts / useMonitoringRuns / useQuickHealthCheck / useAgentHealth /
// usePipelineHealth / useHealthHistory. These are not covered by the shared
// `mockApiRoutes` fixture, so we register them per-spec (the page falls back
// to SAMPLE_* fixtures when these 404, but stubbing them deterministically
// silences react-query retries that otherwise race with assertion windows).
async function mockSystemHealthRoutes(page: Page): Promise<void> {
  await page.route('**/api/monitoring/alerts**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        alerts: [],
        total: 0,
        active_count: 0,
        page: 1,
        page_size: 10,
      }),
    })
  })

  await page.route('**/api/monitoring/runs**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ runs: [], total: 0 }),
    })
  })

  // Health-score endpoints (api-client baseURL is `/api`, so /health-score/*
  // is served as /api/health-score/*).
  await page.route('**/api/health-score/quick**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        overall_health_score: 92,
        health_grade: 'A',
        component_health_score: 0.95,
        model_health_score: 0.88,
        pipeline_health_score: 0.82,
        agent_health_score: 0.92,
        critical_issues: [],
        warnings: [],
        recommendations: [],
        timestamp: new Date().toISOString(),
      }),
    })
  })

  await page.route('**/api/health-score/agents**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ agents: [], available_count: 0, total_agents: 0 }),
    })
  })

  await page.route('**/api/health-score/pipelines**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ pipelines: [] }),
    })
  })

  await page.route('**/api/health-score/history**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        checks: [],
        trend: 'stable',
        avg_health_score: 89.5,
        total_checks: 0,
      }),
    })
  })
}

// Resilient navigation: the route loads a Vite-lazy chunk (SystemHealth-*.js).
// Under high test concurrency, the dynamic import occasionally fails with
// "Failed to fetch dynamically imported module", surfacing as the page-level
// error boundary with a "Try Again" button. Clicking Try Again retries the
// import; we retry up to 3x before giving up.
async function gotoSystemHealth(page: Page, healthPage: SystemHealthPage): Promise<void> {
  await healthPage.goto()
  const errorText = /Failed to fetch dynamically imported module/i
  const heading = page.getByRole('heading', { name: /^System Health$/i }).first()

  for (let attempt = 0; attempt < 4; attempt += 1) {
    // Wait for either the in-page heading OR the dynamic-import failure
    // banner, whichever resolves first.
    await Promise.race([
      heading.waitFor({ state: 'visible', timeout: 8000 }).catch(() => {}),
      page.getByText(errorText).first().waitFor({ state: 'visible', timeout: 8000 }).catch(() => {}),
    ])

    const errorVisible = await page.getByText(errorText).first().isVisible().catch(() => false)
    if (!errorVisible) {
      // Heading is up or no error surfaced — proceed.
      return
    }

    const retry = page.getByRole('button', { name: /Try Again/i }).first()
    if (!(await retry.isVisible().catch(() => false))) {
      // Error visible but no retry — best-effort full reload.
      await page.reload()
      continue
    }
    await retry.click()
    // Allow the chunk fetch to settle before re-checking.
    await page.waitForTimeout(800)
  }
}

test.describe('System Health Page', () => {
  let healthPage: SystemHealthPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    await mockSystemHealthRoutes(page)
    healthPage = new SystemHealthPage(page)
    await gotoSystemHealth(page, healthPage)
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(healthPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(healthPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(healthPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(healthPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Overview Stats', () => {
    test('should display overview stats', async () => {
      const hasStats = await healthPage.verifyOverviewStatsDisplayed()
      expect(hasStats).toBeTruthy()
    })

    test('should show Services stat', async () => {
      await expect(healthPage.servicesCard).toBeVisible()
    })

    test('should show Model Health stat', async () => {
      await expect(healthPage.modelHealthCard).toBeVisible()
    })

    test('should show Active Alerts stat', async () => {
      await expect(healthPage.activeAlertsCard).toBeVisible()
    })
  })

  test.describe('Service Status', () => {
    test('should display service status section', async () => {
      const hasStatus = await healthPage.verifyServiceStatusDisplayed()
      expect(hasStatus).toBeTruthy()
    })

    test('should show API Gateway service', async () => {
      await expect(healthPage.apiGatewayService).toBeVisible()
    })

    test('should show PostgreSQL service', async () => {
      await expect(healthPage.postgresService).toBeVisible()
    })

    test('should show Redis Cache service', async () => {
      await expect(healthPage.redisService).toBeVisible()
    })

    test('should show FalkorDB service', async () => {
      await expect(healthPage.falkordbService).toBeVisible()
    })
  })

  test.describe('Model Health', () => {
    test('should display model health section', async () => {
      const hasModels = await healthPage.verifyModelHealthDisplayed()
      expect(hasModels).toBeTruthy()
    })

    test('should show Propensity Model', async () => {
      await expect(healthPage.propensityModel).toBeVisible()
    })
  })

  test.describe('Active Alerts', () => {
    test('should display alerts section', async () => {
      const hasAlerts = await healthPage.verifyAlertsDisplayed()
      expect(hasAlerts).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(healthPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await healthPage.clickRefresh()
      await healthPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await gotoSystemHealth(page, healthPage)
      await expect(healthPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await gotoSystemHealth(page, healthPage)
      await expect(healthPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await gotoSystemHealth(page, healthPage)
      await expect(healthPage.mainContent).toBeVisible()
    })
  })
})
