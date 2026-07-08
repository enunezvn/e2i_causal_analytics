import { test, expect, type Page, type Route } from '@playwright/test'
import { TimeSeriesPage } from '../pages/time-series.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

// =============================================================================
// LOCAL MOCK OVERRIDES (KPI-history contract)
// =============================================================================
//
// The Time Series page is the single-mode KPI-history home: it reads
// `GET /api/kpis` (useKPIList) for the grouped select, `GET /api/kpis/
// history/coverage` (useKPIHistoryCoverage) for the has-history badges +
// brand scopes, `GET /api/kpis/{id}/history` (useKPIHistory) for the chart,
// `GET /api/kpis/{id}` (useKPIValue) for the status card and `GET
// /api/kpis/{id}/metadata` (useKPIMetadata) for the display name. The shared
// `mockApiRoutes` fixture only knows two WS1 KPIs and no `/history` surface,
// so we register more specific routes here AFTER `mockApiRoutes` —
// Playwright's route dispatch picks the last-registered matching route
// first. Per the agent contract on issue #332, the api-mocks fixture and
// production code are off-limits.
//
// The former "Model performance" mode (and its `/api/monitoring/performance/
// {model_id}/trend` stub) moved to the Model Performance page — sibling PR.
//
// Workstream values MUST match the live registry (`ws3_business`, NOT the
// pre-rewrite `ws3_business_impact` — that stale key is exactly how this
// spec caught the dropdown's silent-vanish failure mode in CI).
// =============================================================================

// Deliberately unordered — the page groups by workstream + sorts by KPI id.
// The WS1-MP-* and CM-* entries MUST be filtered out of the dropdown (they
// are served on /model-performance / per-analysis surfaces respectively).
const KPI_FIXTURES = [
  kpiFixture('WS1-DQ-001', 'Source Coverage - Patients', 'ws1_data_quality'),
  kpiFixture('WS3-BI-010', 'Return on Investment', 'ws3_business'),
  kpiFixture('WS2-TR-005', 'Alert Yield', 'ws2_triggers'),
  kpiFixture('WS3-BI-007', 'New-to-Brand Prescriptions (NBRx)', 'ws3_business'),
  kpiFixture('WS1-MP-001', 'ROC-AUC', 'ws1_model_performance'),
  kpiFixture('CM-001', 'Average Treatment Effect (ATE)', 'causal_metrics'),
]

// The dropdown must offer exactly the non-hidden fixtures.
const VISIBLE_KPI_IDS = ['WS1-DQ-001', 'WS3-BI-010', 'WS2-TR-005', 'WS3-BI-007']

// Coverage map: which KPIs have a real series, in which brand scopes.
// WS1-DQ-001 is deliberately absent (-> "no history yet" badge); WS3-BI-007
// NBRx is per-brand ONLY (no '' global scope) — the page must snap its brand
// select to a real brand instead of showing a false empty-state.
const COVERAGE_FIXTURES = [
  {
    kpi_id: 'WS3-BI-010',
    brands: [''],
    points: 24,
    first_date: '2024-07-01',
    last_date: '2026-06-01',
  },
  {
    kpi_id: 'WS2-TR-005',
    brands: [''],
    points: 35,
    first_date: '2023-08-01',
    last_date: '2026-06-01',
  },
  {
    kpi_id: 'WS3-BI-007',
    brands: ['Fabhalta', 'Kisqali', 'Remibrutinib'],
    points: 105,
    first_date: '2023-08-01',
    last_date: '2026-06-01',
  },
]

function kpiFixture(id: string, name: string, workstream: string) {
  return {
    id,
    name,
    definition: `${name} definition`,
    formula: 'numerator / denominator',
    calculation_type: 'direct',
    workstream,
    tables: [],
    columns: [],
    threshold: { target: 85, warning: 70, critical: 50 },
    unit: '%',
    frequency: 'monthly',
    primary_causal_library: 'none',
  }
}

async function stubTimeSeriesEndpoints(page: Page): Promise<void> {
  // KPI list — populates the grouped KPI select.
  await page.route(/\/api\/kpis(?:\?|$)/, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ kpis: KPI_FIXTURES, total: KPI_FIXTURES.length }),
    })
  })

  // History coverage — drives the "no history yet" badges and the per-KPI
  // brand scopes. Static path, so it must be stubbed explicitly (it would
  // otherwise fall through to the shared fixture and error).
  await page.route(/\/api\/kpis\/history\/coverage(?:\?|$)/, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ coverage: COVERAGE_FIXTURES, total: COVERAGE_FIXTURES.length }),
    })
  })

  // Per-KPI metadata — drives the chart's display name.
  await page.route(/\/api\/kpis\/[^/]+\/metadata(?:\?|$)/, async (route: Route) => {
    const url = new URL(route.request().url())
    const kpiId = decodeURIComponent(url.pathname.split('/').slice(-2)[0] ?? 'WS3-BI-010')
    const kpi = KPI_FIXTURES.find((k) => k.id === kpiId) ?? kpiFixture(kpiId, kpiId)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(kpi),
    })
  })

  // Per-KPI history — drives the chart. The page reads the real monthly
  // series from `GET /api/kpis/{id}/history` (useKPIHistory); when it's
  // empty the page renders an honest empty-state instead of a chart. Stub a
  // non-empty monthly series so the chart (and the recharts SVG that
  // `verifyHistoryChartDisplayed` asserts) actually renders. The default
  // time range is 5 years, so ~24 monthly points all survive the client-side
  // range filter.
  await page.route(/\/api\/kpis\/[^/]+\/history(?:\?|$)/, async (route: Route) => {
    const url = new URL(route.request().url())
    const kpiId = decodeURIComponent(url.pathname.split('/').slice(-2)[0] ?? 'WS3-BI-010')
    const now = Date.now()
    const points = []
    for (let i = 23; i >= 0; i--) {
      const d = new Date(now - i * 30 * 24 * 60 * 60 * 1000)
      points.push({
        metric_date: d.toISOString().slice(0, 10),
        value: Number((1.8 + 0.05 * Math.sin(i / 3)).toFixed(4)),
        status: 'warning',
      })
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        kpi_id: kpiId,
        brand: '',
        region: '',
        count: points.length,
        points,
      }),
    })
  })

  // Per-KPI value — drives the "Current KPI Status" card. We override the
  // shared `/api/kpis/{id}?use_cache=true` handler so the payload is
  // well-formed. Use a regex anchored at the path end so we DON'T intercept
  // `/api/kpis/{id}/metadata` or `/api/kpis/{id}/history` (handled above).
  await page.route(/\/api\/kpis\/[^/?]+(?:\?|$)/, async (route: Route) => {
    const url = new URL(route.request().url())
    const segments = url.pathname.split('/').filter(Boolean)
    const last = segments[segments.length - 1]
    if (!last || last === 'workstreams' || last === 'health' || last === 'batch') {
      await route.fallback()
      return
    }
    const kpiId = decodeURIComponent(last)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        kpi_id: kpiId,
        value: 1.85,
        status: 'warning',
        calculated_at: new Date().toISOString(),
        cached: false,
        metadata: {},
      }),
    })
  })
}

test.describe('Time Series Page', () => {
  let timeSeriesPage: TimeSeriesPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    // Register TS-specific overrides AFTER the shared mocks so Playwright
    // picks ours first for `/api/kpis/...` URLs.
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

    test('should display the KPI-focused page description', async () => {
      await expect(timeSeriesPage.pageDescription).toBeVisible()
    })

    test('should not render the retired Model performance mode', async ({ page }) => {
      // The mode tabs and cohort/brand selects moved to Model Performance
      // (sibling PR) — none of them may render here.
      await expect(page.getByRole('tab')).toHaveCount(0)
      await expect(page.locator('#ts-cohort')).toHaveCount(0)
      await expect(page.locator('#ts-brand')).toHaveCount(0)
      await expect(page.getByText('Performance Trend')).toHaveCount(0)
    })
  })

  test.describe('KPI Selector', () => {
    test('should display KPI selector', async () => {
      await expect(timeSeriesPage.kpiSelector).toBeVisible()
    })

    test('should default to Return on Investment (WS3-BI-010)', async () => {
      // WS3-BI-010 is the deepest real series in kpi_history, so the page
      // lands on a populated chart instead of an empty-state.
      await expect(timeSeriesPage.kpiSelector).toContainText('Return on Investment')
      // The KPI id is visible in the trigger — families are identifiable.
      await expect(timeSeriesPage.kpiSelector).toContainText('WS3-BI-010')
    })

    test('should group KPIs by workstream with visible ids, hiding MP-*/CM-*', async ({
      page,
    }) => {
      await timeSeriesPage.kpiSelector.click()
      const options = page.getByRole('option')
      // Exactly the non-hidden fixtures — WS1-MP-* and CM-* must NOT render.
      await expect(options).toHaveCount(VISIBLE_KPI_IDS.length)
      const texts = await options.allTextContents()
      for (const id of VISIBLE_KPI_IDS) {
        expect(texts.some((t) => t.includes(id))).toBe(true)
      }
      expect(texts.some((t) => t.includes('WS1-MP') || t.includes('ROC-AUC'))).toBe(false)
      expect(texts.some((t) => t.includes('CM-001') || t.includes('(ATE)'))).toBe(false)
      // Workstream group labels render for the families present.
      await expect(page.getByText('Data Quality (WS1)')).toBeVisible()
      await expect(page.getByText('Trigger Performance (WS2)')).toBeVisible()
      await expect(page.getByText('Business Impact (WS3)')).toBeVisible()
      // History-less entries are labeled honestly (WS1-DQ-001 has no
      // coverage entry; WS3-BI-010 has a real series).
      expect(texts.find((t) => t.includes('WS1-DQ-001'))).toMatch(/no history yet/i)
      expect(texts.find((t) => t.includes('WS3-BI-010'))).not.toMatch(/no history yet/i)
      await page.keyboard.press('Escape')
    })

    test('should allow KPI selection', async () => {
      await timeSeriesPage.selectKpi('Alert Yield')
      await expect(timeSeriesPage.kpiSelector).toContainText('Alert Yield')
      // The chart re-renders for the newly selected KPI.
      const hasChart = await timeSeriesPage.verifyHistoryChartDisplayed()
      expect(hasChart).toBeTruthy()
    })

    test('per-brand-only KPI shows a brand select snapped to a real brand', async ({
      page,
    }) => {
      // A global NBRx series is undefined by design — on selection the page
      // must surface the brand select snapped to the first covered brand
      // (previously these 105 real points were unreachable: the page never
      // passed a brand, so NBRx falsely rendered as an empty-state).
      await timeSeriesPage.selectKpi('New-to-Brand')
      const brandSelect = page.getByRole('combobox', { name: /^brand$/i })
      await expect(brandSelect).toBeVisible()
      await expect(brandSelect).toContainText('Fabhalta')
      // No "All Brands" option for a per-brand-only KPI.
      await brandSelect.click()
      await expect(page.getByRole('option', { name: /All Brands/i })).toHaveCount(0)
      await expect(page.getByRole('option', { name: 'Kisqali' })).toBeVisible()
      await page.keyboard.press('Escape')
    })
  })

  test.describe('Time Range Selector', () => {
    test('should display time range selector', async () => {
      await expect(timeSeriesPage.timeRangeSelector).toBeVisible()
    })

    test('should allow time range selection', async () => {
      // TIME_RANGES exposes 30/60/90 Days, 6 Months, 1 Year, 5 Years — the
      // cutoff is applied client-side over the monthly history.
      await timeSeriesPage.selectTimeRange('1 Year')
      await expect(timeSeriesPage.timeRangeSelector).toContainText('1 Year')
    })
  })

  test.describe('Summary Cards', () => {
    test('should display summary stat cards', async () => {
      const hasKpis = await timeSeriesPage.verifyKPICardsDisplayed()
      expect(hasKpis).toBeTruthy()
    })

    test('should show Current Value stat', async () => {
      await expect(timeSeriesPage.currentValueCard).toBeVisible()
    })

    test('should show Data Points stat', async () => {
      await expect(timeSeriesPage.dataPointsCard).toBeVisible()
    })
  })

  test.describe('KPI History', () => {
    test('should display the history chart', async () => {
      const hasChart = await timeSeriesPage.verifyHistoryChartDisplayed()
      expect(hasChart).toBeTruthy()
    })

    test('should display the Current KPI Status card', async () => {
      await expect(timeSeriesPage.currentKpiStatusCard).toBeVisible()
    })

    test('should show an honest empty-state when a KPI has no history', async ({ page }) => {
      // Re-stub the history endpoint to return no points (registered after
      // the beforeEach stubs, so it wins), then reload.
      await page.route(/\/api\/kpis\/[^/]+\/history(?:\?|$)/, async (route: Route) => {
        const url = new URL(route.request().url())
        const kpiId = decodeURIComponent(url.pathname.split('/').slice(-2)[0] ?? 'WS3-BI-010')
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ kpi_id: kpiId, brand: '', region: '', count: 0, points: [] }),
        })
      })
      await timeSeriesPage.goto()
      await expect(timeSeriesPage.kpiHistoryEmptyState).toBeVisible({ timeout: 10000 })
      // The point-in-time status card still renders.
      await expect(timeSeriesPage.currentKpiStatusCard).toBeVisible()
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

    test('should have export button', async () => {
      await expect(timeSeriesPage.exportButton).toBeVisible()
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
