import { test, expect, Page, Route } from '@playwright/test'
import { ModelPerformancePage } from '../pages/model-performance.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

/**
 * Inline mocks for endpoints that ModelPerformance.tsx hits after PR #317
 * live-wired it:
 *   - GET /api/models/status                (useModelsStatus)
 *   - GET /api/monitoring/performance/<id>/trend    (usePerformanceTrend)
 *   - GET /api/monitoring/performance/<id>/alerts   (usePerformanceAlerts)
 *   - GET /api/monitoring/performance/<id>/compare/<other> (useModelComparison)
 *
 * Registered BEFORE the page-level mockApiRoutes catch-all so they win in
 * Playwright's registration-order dispatch.
 *
 * Refs #332. Per the issue contract we keep these page-local instead of
 * editing the shared `frontend/e2e/fixtures/api-mocks.ts`.
 */
async function mockModelPerformanceRoutes(page: Page): Promise<void> {
  const now = new Date().toISOString()
  const models = [
    {
      model_name: 'HCP Tier Classifier',
      status: 'healthy',
      endpoint: 'https://api.example.com/models/hcp-tier-classifier',
      last_check: now,
    },
    {
      model_name: 'Patient Risk Predictor',
      status: 'healthy',
      endpoint: 'https://api.example.com/models/patient-risk-predictor',
      last_check: now,
    },
  ]

  const trendResponse = (modelId: string) => ({
    model_id: modelId,
    metric_name: 'accuracy',
    current_value: 0.92,
    baseline_value: 0.9,
    change_percent: 2.22,
    trend: 'improving',
    is_significant: true,
    alert_threshold_breached: false,
    history: [
      { metric_name: 'accuracy', metric_value: 0.9, recorded_at: now },
      { metric_name: 'accuracy', metric_value: 0.91, recorded_at: now },
      { metric_name: 'accuracy', metric_value: 0.92, recorded_at: now },
    ],
  })

  const alertsResponse = (modelId: string) => ({
    model_id: modelId,
    alert_count: 0,
    alerts: [],
  })

  const comparisonResponse = (modelId: string, otherId: string) => ({
    model_id: modelId,
    other_model_id: otherId,
    metric_name: 'accuracy',
    model_value: 0.92,
    other_model_value: 0.89,
    difference: 0.03,
    difference_percent: 3.37,
    better_model: modelId,
  })

  // /api/models/status  (useModelsStatus -> getModelsStatus -> `${MODELS_BASE}/status`)
  await page.route('**/api/models/status**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        total_models: models.length,
        healthy_count: models.length,
        unhealthy_count: 0,
        models,
        timestamp: now,
      }),
    })
  })

  // /api/monitoring/performance/<id>/compare/<other>  (must register BEFORE the
  // less-specific /trend + /alerts routes, otherwise the latter swallow this URL).
  await page.route(
    '**/api/monitoring/performance/*/compare/**',
    async (route: Route) => {
      const url = route.request().url()
      const match = url.match(/\/performance\/([^/]+)\/compare\/([^/?]+)/)
      const modelId = match?.[1] ? decodeURIComponent(match[1]) : 'unknown'
      const otherId = match?.[2] ? decodeURIComponent(match[2]) : 'unknown'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(comparisonResponse(modelId, otherId)),
      })
    }
  )

  // /api/monitoring/performance/<id>/trend
  await page.route(
    '**/api/monitoring/performance/*/trend**',
    async (route: Route) => {
      const url = route.request().url()
      const match = url.match(/\/performance\/([^/]+)\/trend/)
      const modelId = match?.[1] ? decodeURIComponent(match[1]) : 'unknown'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(trendResponse(modelId)),
      })
    }
  )

  // /api/monitoring/performance/<id>/alerts
  await page.route(
    '**/api/monitoring/performance/*/alerts**',
    async (route: Route) => {
      const url = route.request().url()
      const match = url.match(/\/performance\/([^/]+)\/alerts/)
      const modelId = match?.[1] ? decodeURIComponent(match[1]) : 'unknown'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(alertsResponse(modelId)),
      })
    }
  )
}

test.describe('Model Performance Page', () => {
  let modelPage: ModelPerformancePage

  test.beforeEach(async ({ page }) => {
    // Page-local mocks MUST register before the shared catch-all so Playwright
    // dispatches them first (registration-order routing).
    await mockModelPerformanceRoutes(page)
    await mockApiRoutes(page)
    modelPage = new ModelPerformancePage(page)
    await modelPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(modelPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(modelPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(modelPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(modelPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(modelPage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async ({ page }) => {
      // Wait for live /api/models/status to populate the dropdown options
      // before attempting to interact with it.
      await modelPage.waitForModelOptions()

      // Select a model OTHER than the auto-selected first option
      // ("HCP Tier Classifier") so the test would actually fail if model
      // selection were broken. We then assert the page issued the
      // performance trend request for the newly-selected model, which is
      // the load-bearing side effect of a working selection.
      const trendForPatient = page.waitForRequest(
        (req) =>
          req.method() === 'GET' &&
          /\/api\/monitoring\/performance\/Patient(%20|\+|\s)Risk(%20|\+|\s)Predictor\/trend/.test(
            req.url()
          ),
        { timeout: 5000 }
      )

      await modelPage.selectModel('Patient Risk Predictor')

      // Falsifiability anchor: if `onValueChange` no longer propagates to
      // the trend query, this request never fires and the test fails.
      await trendForPatient
    })
  })

  test.describe('KPI Cards', () => {
    test('should display KPI cards', async () => {
      const hasKpis = await modelPage.verifyKPICardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Accuracy metric', async () => {
      await expect(modelPage.accuracyCard).toBeVisible()
    })

    // After PR #317 the trend-driven KPI tiles are Current/Baseline/Change/Trend
    // (the previous Precision/Recall/F1 mock-data tiles were removed). We assert
    // on the live KPI titles emitted by ModelPerformance.tsx.
    test('should show Baseline metric', async () => {
      await expect(modelPage.baselineCard).toBeVisible()
    })

    test('should show Change metric', async () => {
      await expect(modelPage.changeCard).toBeVisible()
    })

    test('should show Trend metric', async () => {
      await expect(modelPage.trendCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await modelPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Confusion Matrix tab as first tab', async () => {
      // Note: There is no Overview tab - first tab is Confusion Matrix
      await expect(modelPage.confusionMatrixTab).toBeVisible()
    })

    test('should show ROC Curve tab', async () => {
      await expect(modelPage.rocCurveTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await modelPage.clickTab('ROC Curve')
      await modelPage.page.waitForTimeout(500)
    })
  })

  test.describe('Overview Tab', () => {
    test('should display performance metrics', async () => {
      const hasMetrics = await modelPage.verifyMetricsDisplayed()
      expect(hasMetrics).toBeTruthy()
    })
  })

  test.describe('Confusion Matrix Tab', () => {
    test('should display confusion matrix when tab clicked', async () => {
      await modelPage.clickTab('Confusion Matrix')
      const hasMatrix = await modelPage.verifyConfusionMatrixDisplayed()
      expect(hasMatrix).toBeTruthy()
    })
  })

  test.describe('ROC Curve Tab', () => {
    test('should display ROC curve when tab clicked', async () => {
      await modelPage.clickTab('ROC Curve')
      const hasRoc = await modelPage.verifyROCCurveDisplayed()
      expect(hasRoc).toBeTruthy()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await modelPage.goto()
      await expect(modelPage.mainContent).toBeVisible()
    })
  })
})
