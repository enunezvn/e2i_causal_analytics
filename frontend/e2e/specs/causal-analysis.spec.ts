/**
 * Causal Analysis Page E2E Tests (#19 coverage gap)
 * =================================================
 *
 * `/causal-analysis` was a routed data page with NO e2e coverage. It is the
 * multi-library (DoWhy/EconML/CausalML) hierarchical-CATE dashboard. These
 * specs stub the REAL backend endpoints the page calls
 * (`GET /api/causal/health`, `GET /api/causal/estimators`) and assert HONEST
 * states:
 *   - healthy service -> "Causal Engine Healthy" banner + KPI overview
 *   - degraded service -> "Service Issue" banner (NOT a fake-healthy banner)
 *   - no analysis run yet -> EmptyState "No hierarchical CATE analysis available"
 *
 * The health/estimator responses are Zod-validated by the api-client
 * (CausalHealthResponseWireSchema / EstimatorListResponseWireSchema), so the
 * stubs below are faithful, shape-correct mirrors of the live contract.
 */

import { test, expect, type Page, type Route } from '@playwright/test'
import { CausalAnalysisPage } from '../pages/causal-analysis.page'
import { harnessBase } from '../fixtures/page-harness'

const HEALTH_HEALTHY = {
  status: 'healthy',
  libraries_available: { dowhy: true, econml: true, causalml: true },
  estimators_loaded: 6,
  pipeline_orchestrator_ready: true,
  hierarchical_analyzer_ready: true,
  last_analysis: new Date().toISOString(),
  analysis_count_24h: 9,
  average_latency_ms: 1840,
  error: null,
}

const HEALTH_DEGRADED = {
  status: 'degraded',
  libraries_available: { dowhy: true, econml: false, causalml: false },
  estimators_loaded: 2,
  pipeline_orchestrator_ready: false,
  hierarchical_analyzer_ready: false,
  last_analysis: null,
  analysis_count_24h: 0,
  average_latency_ms: null,
  error: 'econml import failed',
}

const ESTIMATORS = {
  estimators: [
    {
      name: 'CausalForestDML',
      library: 'econml',
      estimator_type: 'causal_forest',
      description: 'Causal Forest with double ML',
      best_for: ['heterogeneous effects'],
      parameters: ['n_estimators'],
      supports_confidence_intervals: true,
      supports_heterogeneous_effects: true,
    },
    {
      name: 'LinearDML',
      library: 'econml',
      estimator_type: 'linear_dml',
      description: 'Linear Double ML',
      best_for: ['linear effects'],
      parameters: ['model_y'],
      supports_confidence_intervals: true,
      supports_heterogeneous_effects: false,
    },
  ],
  total: 2,
  by_library: { econml: ['CausalForestDML', 'LinearDML'] },
}

async function stubCausalEndpoints(
  page: Page,
  opts: { degraded?: boolean } = {},
): Promise<void> {
  await page.route('**/api/causal/health**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(opts.degraded ? HEALTH_DEGRADED : HEALTH_HEALTHY),
    })
  })

  await page.route('**/api/causal/estimators**', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(ESTIMATORS),
    })
  })
}

test.describe('Causal Analysis Page', () => {
  let causalPage: CausalAnalysisPage

  test.describe('Healthy service', () => {
    test.beforeEach(async ({ page }) => {
      await harnessBase(page)
      await stubCausalEndpoints(page)
      causalPage = new CausalAnalysisPage(page)
      await causalPage.goto()
    })

    test('loads at /causal-analysis', async ({ page }) => {
      await expect(page).toHaveURL(/causal-analysis/)
    })

    test('displays the page header', async () => {
      await expect(causalPage.pageHeader).toBeVisible()
    })

    test('displays the page description', async () => {
      await expect(causalPage.pageDescription).toBeVisible()
    })

    test('renders the healthy service banner from real health data', async () => {
      await expect(causalPage.healthyBanner).toBeVisible()
    })

    test('renders the analyses-24h count from real health data', async ({ page }) => {
      await expect(page.getByText(/9 analyses completed in the last 24 hours/i)).toBeVisible()
    })

    test('renders the KPI overview cards', async () => {
      await expect(causalPage.librariesCard).toBeVisible()
      await expect(causalPage.estimatorsCard).toBeVisible()
    })

    test('shows honest empty state before any analysis is run', async () => {
      await expect(causalPage.emptyState).toBeVisible()
    })
  })

  test.describe('Degraded service (falsifiability)', () => {
    // If the page ignored the real health payload and always rendered a
    // hard-coded "healthy" banner, this assertion would fail.
    test('renders the "Service Issue" banner when the engine is degraded', async ({ page }) => {
      await harnessBase(page)
      await stubCausalEndpoints(page, { degraded: true })
      causalPage = new CausalAnalysisPage(page)
      await causalPage.goto()

      await expect(causalPage.serviceIssueBanner).toBeVisible()
      await expect(causalPage.healthyBanner).toBeHidden()
    })
  })
})
