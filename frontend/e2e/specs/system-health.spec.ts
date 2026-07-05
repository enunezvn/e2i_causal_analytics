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
  // is served as /api/health-score/*). Every real backend payload now carries
  // data_provenance and the page renders scores only for trusted provenance
  // (isTrustedProvenance), so the stubs mirror that wire contract — an
  // untrusted stub would exercise only the honest-empty path (codex PR-4 R6).
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
        data_provenance: 'measured',
      }),
    })
  })

  await page.route('**/api/health-score/agents**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        agents: [],
        available_count: 0,
        total_agents: 0,
        data_provenance: 'measured',
      }),
    })
  })

  await page.route('**/api/health-score/pipelines**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ pipelines: [], data_provenance: 'measured' }),
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
// error boundary with a "Try Again" button. We retry the navigation by full
// page.reload() until the in-page heading appears (up to 5 attempts).
async function gotoSystemHealth(page: Page, healthPage: SystemHealthPage): Promise<void> {
  const errorText = /Failed to fetch dynamically imported module/i
  const heading = page.getByRole('heading', { name: /^System Health$/i }).first()

  await healthPage.goto()

  for (let attempt = 0; attempt < 5; attempt += 1) {
    // Wait for the in-page heading (the only reliable signal that the lazy
    // chunk loaded and the page actually rendered, vs. the ErrorBoundary).
    try {
      await heading.waitFor({ state: 'visible', timeout: 10000 })
      return // heading visible — page loaded successfully.
    } catch {
      // Heading didn't appear within 10s — check for ErrorBoundary.
    }

    const errorVisible = await page.getByText(errorText).first().isVisible().catch(() => false)
    if (!errorVisible) {
      // No error and no heading either — bail; test will surface the issue.
      return
    }

    // ErrorBoundary present — full reload (more reliable than Try Again button
    // under concurrent chunk-fetch contention).
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForTimeout(500)
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

    // F-002: SAMPLE_SERVICES data was deleted from production paths.
    // The page now renders an empty state when no API service-status
    // endpoint is wired, so the individual service-name assertions
    // below are no longer applicable. Once the API hook exists, this
    // block should re-assert on the real service names.
    test.skip('should show API Gateway service (skipped post F-002)', async () => {
      await expect(healthPage.apiGatewayService).toBeVisible()
    })

    test.skip('should show PostgreSQL service (skipped post F-002)', async () => {
      await expect(healthPage.postgresService).toBeVisible()
    })

    test.skip('should show Redis Cache service (skipped post F-002)', async () => {
      await expect(healthPage.redisService).toBeVisible()
    })

    test.skip('should show FalkorDB service (skipped post F-002)', async () => {
      await expect(healthPage.falkordbService).toBeVisible()
    })
  })

  test.describe('Model Health', () => {
    test('should display model health section', async () => {
      const hasModels = await healthPage.verifyModelHealthDisplayed()
      expect(hasModels).toBeTruthy()
    })

    // F-002: SAMPLE_MODELS data was deleted from production paths.
    test.skip('should show Propensity Model (skipped post F-002)', async () => {
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
