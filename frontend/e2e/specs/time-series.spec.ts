import { test, expect, type Page, type Route } from '@playwright/test'
import { TimeSeriesPage } from '../pages/time-series.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// =============================================================================
// LOCAL MOCK OVERRIDES (post-PR #313 live-wired contract)
// =============================================================================
//
// PR #313 (issue #302) rewired src/pages/TimeSeries.tsx onto the live
// `/api/monitoring/performance/{model_id}/trend` endpoint (via
// `usePerformanceTrend`) and the `/api/kpis/*` endpoints (via
// `useKPIValue` / `useKPIMetadata` / `useKPIList`). The shared
// `mockApiRoutes` fixture does NOT stub the performance-trend URL, so the
// page renders a `QueryErrorState` and the metric / time-range Selects
// + chart never appear.
//
// Per the agent contract on issue #332, the api-mocks fixture and
// production code are off-limits — we register more specific routes here
// AFTER `mockApiRoutes` so Playwright's route-dispatch ordering picks ours
// first (last-registered wins on per-request match).
// =============================================================================

function buildTrendHistory(days: number): Array<{ recorded_at: string; metric_value: number }> {
  const now = Date.now()
  const points: Array<{ recorded_at: string; metric_value: number }> = []
  for (let i = days; i >= 0; i--) {
    const ts = new Date(now - i * 24 * 60 * 60 * 1000).toISOString()
    // Smooth-ish sinusoidal series to give a non-trivial chart.
    const value = 0.85 + 0.05 * Math.sin(i / 7)
    points.push({ recorded_at: ts, metric_value: Number(value.toFixed(4)) })
  }
  return points
}

async function stubTimeSeriesEndpoints(page: Page): Promise<void> {
  // Performance trend — drives the default "Model performance" tab.
  // Match via regex so the model_id segment can contain dots / slashes / dashes
  // without tripping over Playwright's `**` glob semantics.
  // Anchor at path end (`?` or end-of-string) so this does NOT swallow any
  // future `/trend/...` sub-paths (e.g. `/trend/baseline`).
  await page.route(/\/api\/monitoring\/performance\/[^/]+\/trend(?:\?|$)/, async (route: Route) => {
    const url = new URL(route.request().url())
    const modelId = decodeURIComponent(url.pathname.split('/').slice(-2)[0] ?? 'propensity_v2.1.0')
    const metric = url.searchParams.get('metric_name') ?? 'accuracy'
    const requestedDays = Number(url.searchParams.get('days') ?? '90')
    // Backend caps `days` at 90 (see src/api/routes/monitoring.py around L1058).
    // Mirror that constraint so a test that selects a > 90-day range cannot
    // silently mask a production 422.
    const days = Math.min(requestedDays, 90)
    const history = buildTrendHistory(days)
    const current = history[history.length - 1].metric_value
    const baseline = history[0].metric_value
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model_id: modelId,
        metric_name: metric,
        current_value: current,
        baseline_value: baseline,
        change_percent: Number((((current - baseline) / baseline) * 100).toFixed(2)),
        trend: current > baseline ? 'improving' : 'stable',
        is_significant: false,
        alert_threshold_breached: false,
        history,
      }),
    })
  })

  // Per-KPI value — drives the "KPI history" tab when the user switches modes.
  // We override the shared `/api/kpis/{id}?use_cache=true` handler so the
  // payload carries a `metadata.history` series rather than `metadata: {}`.
  // Use a regex anchored at the path end so we DON'T intercept
  // `/api/kpis/{id}/metadata` (which the shared mock already serves correctly).
  await page.route(/\/api\/kpis\/[^/?]+(?:\?|$)/, async (route: Route) => {
    const url = new URL(route.request().url())
    const segments = url.pathname.split('/').filter(Boolean)
    const last = segments[segments.length - 1]
    if (!last || last === 'workstreams' || last === 'health') {
      await route.fallback()
      return
    }
    const kpiId = decodeURIComponent(last)
    const history = buildTrendHistory(90).map((p) => ({
      recorded_at: p.recorded_at,
      value: p.metric_value * 100,
    }))
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        kpi_id: kpiId,
        value: history[history.length - 1].value,
        status: 'good',
        calculated_at: new Date().toISOString(),
        cached: false,
        metadata: { history },
      }),
    })
  })
}

test.describe('Time Series Page', () => {
  let timeSeriesPage: TimeSeriesPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    // Register TS-specific overrides AFTER the shared mocks so Playwright
    // picks ours first for `/api/monitoring/performance/.../trend` URLs.
    await stubTimeSeriesEndpoints(page)
    timeSeriesPage = new TimeSeriesPage(page)
    await timeSeriesPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(timeSeriesPage.mainContent).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should display page title', async ({ page }) => {
      await expect(page).toHaveTitle(timeSeriesPage.pageTitle)
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should finish loading within timeout', async ({ page }) => {
      await assertNotLoading(page, TIMEOUTS.PAGE_LOAD)
    })

    test('should display page header', async () => {
      await expect(timeSeriesPage.pageHeader).toBeVisible()
    })

    test('should display page description', async () => {
      await expect(timeSeriesPage.pageDescription).toBeVisible()
    })
  })

  test.describe('Metric Selector', () => {
    test('should display metric selector', async () => {
      await expect(timeSeriesPage.metricSelector).toBeVisible()
    })

    test('should allow metric selection', async () => {
      // The metric Select exposes Accuracy / Precision / Recall / F1 / AUC-ROC
      // (see METRIC_OPTIONS in TimeSeries.tsx).
      await timeSeriesPage.selectMetric('Precision')
      await timeSeriesPage.page.waitForTimeout(500)
    })
  })

  test.describe('Time Range Selector', () => {
    test('should display time range selector', async () => {
      await expect(timeSeriesPage.timeRangeSelector).toBeVisible()
    })

    test('should allow time range selection', async () => {
      // TIME_RANGES exposes 30 Days / 60 Days / 90 Days / 6 Months / 1 Year.
      // The backend caps `days` at 90 (see src/api/routes/monitoring.py
      // L1058), so we pick "60 Days" — a value that stays within range and
      // therefore exercises a real, not-error path through the hook.
      await timeSeriesPage.selectTimeRange('60 Days')
      await timeSeriesPage.page.waitForTimeout(500)
    })
  })

  test.describe('KPI Cards', () => {
    test('should display KPI cards', async () => {
      const hasKpis = await timeSeriesPage.verifyKPICardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Current Value stat', async () => {
      await expect(timeSeriesPage.currentValueCard).toBeVisible()
    })

    test('should show Trend stat', async () => {
      // The "Trend Summary" card renders only once `performanceTrend.data`
      // resolves, and that query is disabled until a model ID is entered
      // (usePerformanceTrend → `enabled: !!model_id`; DEFAULT_MODEL_ID is '').
      // So drive the page like a user: entering a model ID enables the fetch —
      // fulfilled here by the inline `/trend` mock — which surfaces the card.
      // Without this, the mock never receives a request and the card (and the
      // KPI summary stats) stay empty: the failure seen in CI shard 4 (#941).
      await timeSeriesPage.enterModelId('csu_treatment_initiation_lr_balanced_v1')
      await expect(timeSeriesPage.trendCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await timeSeriesPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Trend tab', async () => {
      // "Trend tab" → "Model performance" in the post-#302 UI.
      await expect(timeSeriesPage.trendTab).toBeVisible()
    })

    test('should show Seasonality tab', async () => {
      // "Seasonality tab" → "KPI history" in the post-#302 UI.
      await expect(timeSeriesPage.seasonalityTab).toBeVisible()
    })

    test('should show Anomalies tab', async () => {
      // The post-#302 UI no longer has a dedicated Anomalies tab — fall
      // back to the KPI history tab so this assertion still proves the
      // tablist is intact.
      await expect(timeSeriesPage.anomaliesTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      await timeSeriesPage.clickTab('Seasonality')
      await timeSeriesPage.page.waitForTimeout(500)
    })
  })

  test.describe('Trend Tab', () => {
    test('should display trend chart', async () => {
      const hasTrend = await timeSeriesPage.verifyTrendChartDisplayed()
      expect(hasTrend).toBeTruthy()
    })
  })

  test.describe('Seasonality Tab', () => {
    test('should display seasonality when tab clicked', async () => {
      // "Seasonality" → "KPI history" tab.
      await timeSeriesPage.clickTab('Seasonality')
      const hasDecomp = await timeSeriesPage.verifyDecompositionDisplayed()
      expect(hasDecomp).toBeTruthy()
    })
  })

  test.describe('Anomalies Tab', () => {
    test('should display anomalies when tab clicked', async () => {
      // No dedicated Anomalies tab post-#302 — the legacy alias maps to the
      // "KPI history" tab. Assert that the tab actually became active AND
      // its content (KPI History card) rendered — *not* the static page
      // description, which is mounted regardless of tab state and would
      // give a false-positive if tab switching were broken.
      await timeSeriesPage.clickTab('Anomalies')
      await expect(timeSeriesPage.kpiHistoryTab).toHaveAttribute(
        'aria-selected',
        'true',
        { timeout: 5000 },
      )
      await expect(timeSeriesPage.kpiHistoryCard).toBeVisible({ timeout: 10000 })
    })
  })

  test.describe('Actions', () => {
    test('should have refresh button', async () => {
      await expect(timeSeriesPage.refreshButton).toBeVisible()
    })

    test('should allow refresh', async () => {
      await timeSeriesPage.clickRefresh()
      await timeSeriesPage.page.waitForTimeout(500)
    })
  })

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 })
      await timeSeriesPage.goto()
      await expect(timeSeriesPage.mainContent).toBeVisible()
    })

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 })
      await timeSeriesPage.goto()
      await expect(timeSeriesPage.mainContent).toBeVisible()
    })

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 })
      await timeSeriesPage.goto()
      await expect(timeSeriesPage.mainContent).toBeVisible()
    })
  })
})
