import { test, expect, Page, Route } from '@playwright/test'
import { PredictiveAnalyticsPage } from '../pages/predictive-analytics.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

/**
 * Predictive Analytics e2e suite (post-PR #319 wiring; closes #332).
 *
 * The page was rewired from a synthetic dashboard to a live-data form
 * backed by /api/models/predict/{model_name}. The selector + form only
 * render when /api/models/status returns at least one model, and the
 * feature inputs only render once /api/models/{name}/info responds.
 *
 * api-mocks.ts is intentionally not extended — those endpoints are
 * specific to this page and we inline the routes here so the shared
 * fixture stays small. We register them AFTER mockApiRoutes() so they
 * win when multiple handlers match (Playwright dispatches routes in
 * reverse registration order — last-added first).
 */

const MOCK_MODELS = [
  {
    model_name: 'Conversion Model',
    status: 'healthy',
    endpoint: '/api/models/predict/Conversion%20Model',
    last_check: new Date().toISOString(),
  },
  {
    model_name: 'Churn Model',
    status: 'healthy',
    endpoint: '/api/models/predict/Churn%20Model',
    last_check: new Date().toISOString(),
  },
]

const MOCK_STATUS_RESPONSE = {
  total_models: MOCK_MODELS.length,
  healthy_count: MOCK_MODELS.length,
  unhealthy_count: 0,
  models: MOCK_MODELS,
  timestamp: new Date().toISOString(),
}

const MOCK_MODEL_INFO = {
  name: 'Conversion Model',
  version: '1.0.0',
  type: 'classification',
  description: 'Test conversion model',
  input_schema: {
    hcp_id: 'string',
    territory: 'string',
    visits: 'number',
  },
  trained_at: '2026-01-01T00:00:00Z',
  metadata: {},
}

async function mockPredictionRoutes(page: Page): Promise<void> {
  // GET /api/models/status — drives the selector
  await page.route('**/api/models/status*', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MOCK_STATUS_RESPONSE),
    })
  })

  // GET /api/models/{name}/info — drives the feature form
  // Match BEFORE the catch-all so it isn't shadowed.
  await page.route('**/api/models/*/info*', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(MOCK_MODEL_INFO),
    })
  })

  // POST /api/models/predict/{name} — only used by the drill-down / what-if;
  // harmless if never triggered by the basic suite.
  await page.route('**/api/models/predict/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model_name: 'Conversion Model',
        prediction: 0.87,
        confidence: 0.91,
        latency_ms: 42,
        model_version: '1.0.0',
        timestamp: new Date().toISOString(),
      }),
    })
  })
}

test.describe('Predictive Analytics Page', () => {
  let predictivePage: PredictiveAnalyticsPage

  test.beforeEach(async ({ page }) => {
    // Register shared catch-alls first, then our page-specific routes so
    // ours win in reverse-LIFO matching when patterns overlap.
    await mockApiRoutes(page)
    await mockPredictionRoutes(page)
    predictivePage = new PredictiveAnalyticsPage(page)
    await predictivePage.goto()
    // The live-data UI is gated on /api/models/status resolving and at
    // least one model being present. Wait for the resulting Active Model
    // card to render so subsequent assertions don't race the React Query
    // cache hydration. Tolerate failure here — the Page Load + Responsive
    // suites only need the bare page shell, so a missed model render
    // shouldn't fail those tests.
    await predictivePage.activeModelLabel
      .waitFor({ state: 'visible', timeout: 10000 })
      .catch(() => {})
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(predictivePage.mainContent).toBeVisible({
        timeout: TIMEOUTS.PAGE_LOAD,
      })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(predictivePage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(predictivePage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(predictivePage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(predictivePage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      await predictivePage.selectModel('Churn')
      // Auto-selection of the first model means the selector is already
      // populated; verify the Active Model card still renders after switch.
      await expect(predictivePage.activeModelLabel).toBeVisible()
    })
  })

  test.describe('Active Model Summary', () => {
    test('should display Active Model card after models load', async () => {
      const ok = await predictivePage.verifyActiveModelCard()
      expect(ok).toBeTruthy()
    })
  })

  test.describe('Cohort Scoring', () => {
    test('should display Ranked Targets card', async () => {
      const ok = await predictivePage.verifyRankedTargetsCard()
      expect(ok).toBeTruthy()
    })

    test('should display Score holdout cohort button', async () => {
      await expect(predictivePage.scoreCohortButton).toBeVisible()
    })
  })

  test.describe('Prediction Detail', () => {
    test('should display Prediction Detail card', async () => {
      const ok = await predictivePage.verifyPredictionDetailCard()
      expect(ok).toBeTruthy()
    })

    test('should show placeholder before any drill-down', async () => {
      await expect(predictivePage.predictionDetailPlaceholder).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await predictivePage.goto()
      await expect(predictivePage.mainContent).toBeVisible()
    })
  })
})
