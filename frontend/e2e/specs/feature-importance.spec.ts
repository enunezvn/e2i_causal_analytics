import { test, expect, type Page, type Route } from '@playwright/test'
import { FeatureImportancePage } from '../pages/feature-importance.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// =============================================================================
// LOCAL MOCK OVERRIDES (post-PR #316 live-wired contract)
// =============================================================================
//
// PR #316 rewired src/pages/FeatureImportance.tsx onto the real `/api/explain/*`
// endpoints. The shared `mockApiRoutes` fixture serves a legacy `{ features:
// [...] }` payload for `**/api/explain/**`, which:
//   1. Leaves the model selector empty (model list never resolves), and
//   2. Means the Model Info / Base Value / Top Feature card never renders
//      because `hasExplanation` is false.
//
// We register more specific routes here AFTER mockApiRoutes so Playwright's
// route-dispatch ordering picks ours first (last-registered wins on
// per-request match). This keeps the fix scoped to spec + page-object per
// the agent contract (do not modify api-mocks.ts).
// =============================================================================

const MOCK_MODEL_TYPES = ['propensity', 'churn_prediction'] as const

const MOCK_FEATURES = [
  {
    feature_name: 'prior_visits',
    feature_value: 4,
    shap_value: 0.31,
    contribution_direction: 'positive' as const,
    contribution_rank: 1,
  },
  {
    feature_name: 'rx_count_90d',
    feature_value: 12,
    shap_value: 0.18,
    contribution_direction: 'positive' as const,
    contribution_rank: 2,
  },
  {
    feature_name: 'days_since_last_fill',
    feature_value: 28,
    shap_value: -0.12,
    contribution_direction: 'negative' as const,
    contribution_rank: 3,
  },
]

async function stubExplainEndpoints(page: Page): Promise<void> {
  // Models list — drives the Select dropdown options.
  await page.route('**/api/explain/models', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        supported_models: MOCK_MODEL_TYPES.map((model_type) => ({
          model_type,
          latest_version: '4.7.0',
          explainer_type: 'TreeExplainer',
          avg_latency_ms: 42,
        })),
        total_models: MOCK_MODEL_TYPES.length,
      }),
    })
  })

  // Predict — fires on Explain / Refresh click. Returns the ExplainResponse
  // shape declared in src/types/explain.ts.
  await page.route('**/api/explain/predict', async (route: Route) => {
    const request = route.request()
    let body: { patient_id?: string; model_type?: string } = {}
    try {
      body = (request.postDataJSON() ?? {}) as typeof body
    } catch {
      // postData wasn't JSON — fall back to defaults.
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        explanation_id: 'e2e-mock-explanation-id',
        request_timestamp: new Date().toISOString(),
        patient_id: body.patient_id ?? 'patient_e2e_001',
        model_type: body.model_type ?? MOCK_MODEL_TYPES[0],
        model_version_id: '4.7.0',
        prediction_class: 'positive',
        prediction_probability: 0.78,
        base_value: 0.25,
        top_features: MOCK_FEATURES,
        shap_sum: MOCK_FEATURES.reduce((acc, f) => acc + f.shap_value, 0),
        computation_time_ms: 42,
        audit_stored: false,
      }),
    })
  })

  // History — empty list keeps the History tab benign.
  await page.route('**/api/explain/history/**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        patient_id: 'patient_e2e_001',
        total_explanations: 0,
        explanations: [],
      }),
    })
  })
}

test.describe('Feature Importance Page', () => {
  let featurePage: FeatureImportancePage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    // Register the explain-specific overrides AFTER the shared mocks so
    // Playwright picks ours first for `/api/explain/*` URLs.
    await stubExplainEndpoints(page)
    featurePage = new FeatureImportancePage(page)
    await featurePage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(featurePage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(featurePage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(featurePage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(featurePage.pageDescription).toBeVisible()
    })
  })

  test.describe('Model Selector', () => {
    test('should display model selector', async () => {
      await expect(featurePage.modelSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      // `Propensity` is the formatModelLabel() output for model_type='propensity'
      // — one of the entries we serve from the stubbed /api/explain/models above.
      await featurePage.selectModel('Propensity')
      await featurePage.page.waitForTimeout(300)
    })
  })

  test.describe('Model Info', () => {
    test('should display model info', async () => {
      const hasModelInfo = await featurePage.verifyModelInfoDisplayed()
      expect(hasModelInfo).toBeTruthy()
    })

    test('should show Base Value stat', async () => {
      // Base Value only renders post-Explain; drive the mutation first.
      await featurePage.runExplanation()
      await expect(featurePage.baseValueDisplay).toBeVisible()
    })

    test('should show Top Feature stat', async () => {
      await featurePage.runExplanation()
      await expect(featurePage.topFeatureDisplay).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await featurePage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Bar Chart tab', async () => {
      await expect(featurePage.barChartTab).toBeVisible()
    })

    test('should show Beeswarm tab', async () => {
      await expect(featurePage.beeswarmTab).toBeVisible()
    })

    test('should show Waterfall tab', async () => {
      await expect(featurePage.waterfallTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await featurePage.clickTab('Beeswarm')
      await featurePage.page.waitForTimeout(300)
    })
  })

  test.describe('Bar Chart Tab', () => {
    test('should display bar chart by default', async () => {
      const hasBarChart = await featurePage.verifyBarChartDisplayed()
      expect(hasBarChart).toBeTruthy()
    })
  })

  test.describe('Beeswarm Tab', () => {
    test('should display beeswarm when tab clicked', async () => {
      await featurePage.clickTab('Beeswarm')
      const hasBeeswarm = await featurePage.verifyBeeswarmDisplayed()
      expect(hasBeeswarm).toBeTruthy()
    })
  })

  test.describe('Waterfall Tab', () => {
    test('should display waterfall when tab clicked', async () => {
      await featurePage.clickTab('Waterfall')
      const hasWaterfall = await featurePage.verifyWaterfallDisplayed()
      expect(hasWaterfall).toBeTruthy()
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(featurePage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      // Refresh is disabled until a patient ID is set; clickRefresh() handles
      // that by driving a baseline explanation when the button isn't enabled.
      await featurePage.clickRefresh()
      await featurePage.page.waitForTimeout(300)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await featurePage.goto()
      await expect(featurePage.mainContent).toBeVisible()
    })
  })
})
