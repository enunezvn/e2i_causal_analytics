import { test, expect, Route } from '@playwright/test'
import { MonitoringPage } from '../pages/monitoring.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// ---------------------------------------------------------------------------
// Inline mocks for the live Monitoring endpoints.
//
// Monitoring.tsx (live-wired in PR #318) consumes three endpoints not mocked
// by the shared fixture (`api-mocks.ts`):
//
//   GET /api/monitoring/runs        → MonitoringRunsResponse
//   GET /api/monitoring/alerts      → AlertListResponse
//   GET /api/monitoring/health/:id  → ModelHealthSummary
//
// We register these via inline `page.route()` calls AFTER `mockApiRoutes()` so
// Playwright dispatches the latest-registered handler first (#318 already
// shipped `/api/monitoring/drift/*` mocks in the shared fixture, but those are
// orthogonal to the three endpoints above).
//
// Shape: keep payloads narrow but valid against `frontend/src/types/monitoring.ts`.
// ---------------------------------------------------------------------------

const MOCK_RUNS = {
  model_id: 'propensity_v2.1.0',
  total_runs: 3,
  runs: [
    {
      id: 'run-1',
      model_version: 'propensity_v2.1.0',
      run_type: 'scheduled',
      // Recent timestamp so the client-side time-range filter (defaults to 24h)
      // keeps the row visible.
      started_at: new Date(Date.now() - 60_000).toISOString(),
      completed_at: new Date(Date.now() - 30_000).toISOString(),
      features_checked: 42,
      drift_detected_count: 2,
      alerts_generated: 1,
      duration_ms: 18_500,
    },
    {
      id: 'run-2',
      model_version: 'propensity_v2.1.0',
      run_type: 'manual',
      started_at: new Date(Date.now() - 5 * 60_000).toISOString(),
      completed_at: new Date(Date.now() - 4 * 60_000).toISOString(),
      features_checked: 38,
      drift_detected_count: 0,
      alerts_generated: 0,
      duration_ms: 22_400,
    },
    {
      id: 'run-3',
      model_version: 'propensity_v2.1.0',
      run_type: 'scheduled',
      started_at: new Date(Date.now() - 30 * 60_000).toISOString(),
      completed_at: new Date(Date.now() - 29 * 60_000).toISOString(),
      features_checked: 40,
      drift_detected_count: 1,
      alerts_generated: 1,
      duration_ms: 19_100,
    },
  ],
}

const MOCK_ALERTS = {
  total_count: 2,
  active_count: 2,
  alerts: [
    {
      id: 'alert-1',
      model_version: 'propensity_v2.1.0',
      alert_type: 'drift_detected',
      severity: 'high',
      title: 'Feature drift detected',
      description: 'tenure_months PSI exceeded warning threshold.',
      status: 'active',
      triggered_at: new Date(Date.now() - 10 * 60_000).toISOString(),
    },
    {
      id: 'alert-2',
      model_version: 'propensity_v2.1.0',
      alert_type: 'performance_degradation',
      severity: 'critical',
      title: 'AUC dropped below SLO',
      description: 'Rolling AUC fell below 0.80 over the last 24h.',
      status: 'active',
      triggered_at: new Date(Date.now() - 20 * 60_000).toISOString(),
    },
  ],
}

const MOCK_HEALTH = {
  model_id: 'propensity_v2.1.0',
  overall_health: 'warning' as const,
  last_check: new Date().toISOString(),
  drift_score: 0.42,
  active_alerts: 2,
  last_retrained: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  performance_trend: 'stable' as const,
  recommendations: ['Investigate tenure_months PSI', 'Schedule a refresh of the holdout snapshot'],
}

async function mockMonitoringEndpoints(
  page: Parameters<typeof mockApiRoutes>[0],
): Promise<void> {
  await page.route('**/api/monitoring/alerts**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MOCK_ALERTS),
    })
  })

  await page.route('**/api/monitoring/runs**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MOCK_RUNS),
    })
  })

  await page.route('**/api/monitoring/health/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MOCK_HEALTH),
    })
  })
}

test.describe('Monitoring Page', () => {
  let monitoringPage: MonitoringPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    // Register live-monitoring mocks AFTER `mockApiRoutes` so Playwright matches
    // our specific endpoints first (last-registered wins).
    await mockMonitoringEndpoints(page)
    monitoringPage = new MonitoringPage(page)
    await monitoringPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(monitoringPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(monitoringPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(monitoringPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(monitoringPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Overview Metrics', () => {
    test('should display overview metrics', async () => {
      const hasMetrics = await monitoringPage.verifyOverviewMetricsDisplayed()
      expect(hasMetrics).toBeTruthy()
    })

    test('should show Total Runs metric', async () => {
      await expect(monitoringPage.totalRunsCard).toBeVisible()
    })

    test('should show Drift Rate metric', async () => {
      await expect(monitoringPage.driftRateCard).toBeVisible()
    })

    test('should show Avg Run Duration metric', async () => {
      await expect(monitoringPage.avgRunDurationCard).toBeVisible()
    })

    test('should show Active Alerts metric', async () => {
      await expect(monitoringPage.activeAlertsCard).toBeVisible()
    })

    test('should show Drift Events metric', async () => {
      await expect(monitoringPage.driftEventsCard).toBeVisible()
    })

    test('should show Health Score metric', async () => {
      await expect(monitoringPage.healthScoreCard).toBeVisible()
    })
  })

  test.describe('Time Range Selector', () => {
    test('should display time range selector', async () => {
      await expect(monitoringPage.timeRangeSelector).toBeVisible()
    })

    test('should allow time range selection', async () => {
      await monitoringPage.selectTimeRange('24 Hours')
      // Give the runs query time to refetch with new `days` param.
      await monitoringPage.page.waitForTimeout(500)
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(monitoringPage.modelSelector).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await monitoringPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Drift Trend tab', async () => {
      await expect(monitoringPage.driftTrendTab).toBeVisible()
    })

    test('should show Runs tab', async () => {
      await expect(monitoringPage.runsTab).toBeVisible()
    })

    test('should show Errors tab', async () => {
      await expect(monitoringPage.errorsTab).toBeVisible()
    })

    test('should show System tab', async () => {
      await expect(monitoringPage.systemTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await monitoringPage.clickTab('Errors')
      await monitoringPage.page.waitForTimeout(500)
    })
  })

  test.describe('Drift Trend Tab', () => {
    test('should display drift trend telemetry', async () => {
      const hasTelemetry = await monitoringPage.verifyDriftTrendDisplayed()
      expect(hasTelemetry).toBeTruthy()
    })
  })

  test.describe('Runs Tab', () => {
    test('should display monitoring runs when tab clicked', async () => {
      // The live page renames the "User Activity" tab → "Runs" but keeps the
      // underlying TabsTrigger value="activity".
      await monitoringPage.clickTab('^Runs$')
      const hasRuns = await monitoringPage.verifyRunsDisplayed()
      expect(hasRuns).toBeTruthy()
    })
  })

  test.describe('Errors Tab', () => {
    test('should display alert feed when tab clicked', async () => {
      await monitoringPage.clickTab('Errors')
      const hasAlerts = await monitoringPage.verifyAlertFeedDisplayed()
      expect(hasAlerts).toBeTruthy()
    })
  })

  test.describe('System Tab', () => {
    test('should display model health when tab clicked', async () => {
      await monitoringPage.clickTab('System')
      const hasSystem = await monitoringPage.verifyModelHealthDisplayed()
      expect(hasSystem).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(monitoringPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await monitoringPage.clickRefresh()
      await monitoringPage.page.waitForTimeout(500)
    })

    test('should have export button', async () => {
      await expect(monitoringPage.exportButton).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await monitoringPage.goto()
      await expect(monitoringPage.mainContent).toBeVisible()
    })
  })
})
