import { test, expect, type Page, type Route } from '@playwright/test'
import { FeatureImportancePage } from '../pages/feature-importance.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// =============================================================================
// LOCAL MOCK OVERRIDES (post-PR #985 cohort/individual redesign)
// =============================================================================
//
// PR #985 redesigned src/pages/FeatureImportance.tsx into two modes:
//   - **Cohort (global)** — the DEFAULT. On arrival it calls
//     `GET /api/explain/global` (mean |SHAP| over a real cohort sample) and
//     `GET /api/explain/models`. The summary card (Base Value / Top Feature) and
//     the Bar/Beeswarm viz tabs are populated WITHOUT any entity selection.
//   - **Individual** — reached via the "Individual" tab. It lists real entity
//     IDs from `GET /api/explain/sample-entities` and AUTO-RUNS
//     `POST /api/explain/predict` on selection (no Explain button). Waterfall +
//     History viz tabs only exist here.
//
// The shared `mockApiRoutes` fixture serves a legacy `{ features: [...] }`
// payload for `**/api/explain/**`, which does not match any of the redesigned
// response shapes. We register more specific routes here AFTER mockApiRoutes so
// Playwright's last-registered-wins dispatch picks ours first for the
// `/api/explain/*` URLs. This keeps the fix scoped to spec + page-object (do not
// modify api-mocks.ts).
// =============================================================================

// Only the real gold-standard cohort families are explainable; the page filters
// the model list down to `is_gold_standard` (or, as a fallback, the
// GOLD_STANDARD_COHORTS taxonomy). Serve two of those so the cohort selector is
// populated and selectable. `formatModelLabel('initiation')` → "Initiation".
const MOCK_MODEL_TYPES = ['initiation', 'persistence'] as const

// Cohort-level (global) features — GlobalImportanceFeature[] shape from
// src/types/explain.ts (mean_abs_shap / mean_shap / mean_feature_value /
// contribution_rank). The page maps these into the bar chart + summary card.
// `prior_visits` is intentionally rank 1 so the "Top Feature" stat renders
// "prior visits" (the falsifiability anchor below).
const MOCK_GLOBAL_FEATURES = [
  {
    feature_name: 'prior_visits',
    mean_abs_shap: 0.31,
    mean_shap: 0.27,
    mean_feature_value: 4,
    contribution_rank: 1,
  },
  {
    feature_name: 'rx_count_90d',
    mean_abs_shap: 0.18,
    mean_shap: 0.18,
    mean_feature_value: 12,
    contribution_rank: 2,
  },
  {
    feature_name: 'days_since_last_fill',
    mean_abs_shap: 0.12,
    mean_shap: -0.12,
    mean_feature_value: 28,
    contribution_rank: 3,
  },
]

// Per-entity SHAP points — GlobalImportancePoint[] shape (real beeswarm dots).
const MOCK_GLOBAL_POINTS = [
  { feature_name: 'prior_visits', shap_value: 0.34, feature_value: 5 },
  { feature_name: 'prior_visits', shap_value: 0.21, feature_value: 3 },
  { feature_name: 'rx_count_90d', shap_value: 0.2, feature_value: 14 },
  { feature_name: 'rx_count_90d', shap_value: 0.15, feature_value: 9 },
  { feature_name: 'days_since_last_fill', shap_value: -0.14, feature_value: 31 },
  { feature_name: 'days_since_last_fill', shap_value: -0.09, feature_value: 22 },
]

// Real entity IDs for the individual-mode picker.
const MOCK_ENTITY_IDS = ['patient_e2e_001', 'patient_e2e_002', 'patient_e2e_003'] as const

// Local (per-entity) feature contributions — FeatureContribution[] shape.
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
  // Models list — drives the cohort Select dropdown options. Mark the served
  // types as gold-standard so the page's `is_gold_standard` filter keeps them.
  await page.route('**/api/explain/models', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        supported_models: MOCK_MODEL_TYPES.map((model_type, i) => ({
          model_type,
          latest_version: '1.0.0',
          explainer_type: 'LinearExplainer',
          is_gold_standard: true,
          description: `${model_type} gold-standard cohort model`,
          avg_latency_ms: 42 + i,
        })),
        total_models: MOCK_MODEL_TYPES.length,
      }),
    })
  })

  // Cohort (global) feature importance — fires on arrival in the DEFAULT mode.
  // GlobalFeatureImportanceResponse shape from src/types/explain.ts. `**` after
  // the path matches the `?model_type=…&brand=…&sample_size=…` query string.
  await page.route('**/api/explain/global**', async (route: Route) => {
    const url = new URL(route.request().url())
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model_type: url.searchParams.get('model_type') ?? MOCK_MODEL_TYPES[0],
        brand: url.searchParams.get('brand') ?? 'Remibrutinib',
        model_name: 'initiation_remibrutinib_goldstd_lr_v1',
        // base_value 0.25 → formatBaseValue() renders "0.250" (anchor below).
        base_value: 0.25,
        sample_size: 25,
        requested_sample_size: 25,
        computation_method: 'LinearExplainer',
        computed_at: new Date().toISOString(),
        cached: false,
        features: MOCK_GLOBAL_FEATURES,
        points: MOCK_GLOBAL_POINTS,
      }),
    })
  })

  // Sample entities — drives the individual-mode picker. SampleEntitiesResponse
  // shape. `**` matches the `?model_type=…&limit=…` query string.
  await page.route('**/api/explain/sample-entities**', async (route: Route) => {
    const url = new URL(route.request().url())
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model_type: url.searchParams.get('model_type') ?? MOCK_MODEL_TYPES[0],
        grain: 'patient',
        id_field: 'patient_id',
        entities: MOCK_ENTITY_IDS,
      }),
    })
  })

  // Predict — auto-fires when an entity is picked in individual mode. Returns
  // the ExplainResponse shape declared in src/types/explain.ts.
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
        patient_id: body.patient_id ?? MOCK_ENTITY_IDS[0],
        model_type: body.model_type ?? MOCK_MODEL_TYPES[0],
        model_version_id: '1.0.0',
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
        patient_id: MOCK_ENTITY_IDS[0],
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
    test('should display cohort selector', async () => {
      await expect(featurePage.modelSelector).toBeVisible()
    })

    test('should display brand selector', async () => {
      await expect(featurePage.brandSelector).toBeVisible()
    })

    test('should allow model selection', async () => {
      // `Initiation` is the formatModelLabel() output for model_type='initiation'
      // — one of the gold-standard entries we serve from /api/explain/models.
      await featurePage.selectModel('Initiation')
      await featurePage.page.waitForTimeout(300)
    })
  })

  test.describe('Model Info', () => {
    test('should display model info', async () => {
      // DEFAULT (cohort) mode: the summary card is populated on arrival from
      // /api/explain/global — no entity selection needed.
      const hasModelInfo = await featurePage.verifyModelInfoDisplayed()
      expect(hasModelInfo).toBeTruthy()
    })

    test('should show Base Value stat', async () => {
      // Cohort mode renders `formatBaseValue(global.base_value)`; the global mock
      // returns `base_value=0.25` → formatted "0.250".
      await expect(featurePage.baseValueDisplay).toBeVisible()
      // Falsifiability anchor: a 200 with the wrong global shape (no base_value)
      // would render "—" instead. Scoped to modelInfoCard so an unrelated
      // "0.250" elsewhere on the page cannot satisfy the assertion.
      await expect(featurePage.modelInfoCard.getByText('0.250')).toBeVisible()
    })

    test('should show Top Feature stat', async () => {
      await expect(featurePage.topFeatureDisplay).toBeVisible()
      // Top Feature renders `features[0]?.feature_name.replace(/_/g, ' ')`. The
      // global mock ranks `prior_visits` first → "prior visits". A 200 missing
      // `features` would render "—" and fail this. SCOPED to the summary card so
      // the same name appearing in the Feature Rankings list / chart labels
      // cannot satisfy a fallen-back "—" Top Feature stat.
      await expect(
        featurePage.modelInfoCard.getByText(/prior visits/i),
      ).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await featurePage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show mode tabs (Cohort + Individual)', async () => {
      await expect(featurePage.cohortModeTab).toBeVisible()
      await expect(featurePage.individualModeTab).toBeVisible()
    })

    test('should show Bar Chart tab', async () => {
      await expect(featurePage.barChartTab).toBeVisible()
    })

    test('should show Beeswarm tab', async () => {
      await expect(featurePage.beeswarmTab).toBeVisible()
    })

    test('should show Waterfall tab in individual mode', async () => {
      // Waterfall is an individual-mode-only viz tab (PR #985). It does not
      // exist on the default cohort view.
      await expect(featurePage.waterfallTab).toHaveCount(0)
      await featurePage.switchToIndividualMode()
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
      // Waterfall lives only in individual mode; drive the per-entity
      // explanation first so the tab exists and has data, then click it.
      await featurePage.runIndividualExplanation()
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
      // In the default cohort mode the Refresh button is enabled as soon as the
      // global query settles — no entity needed — so it can be clicked directly.
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
