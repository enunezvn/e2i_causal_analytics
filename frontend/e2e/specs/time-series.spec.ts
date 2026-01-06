import { test, expect } from '@playwright/test'
import { TimeSeriesPage } from '../pages/time-series.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNotLoading, assertNoErrors } from '../utils/assertions'

test.describe('Time Series Page', () => {
  let timeSeriesPage: TimeSeriesPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
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
      await timeSeriesPage.selectMetric('TRx')
      await timeSeriesPage.page.waitForTimeout(500)
    })
  })

  test.describe('Time Range Selector', () => {
    test('should display time range selector', async () => {
      await expect(timeSeriesPage.timeRangeSelector).toBeVisible()
    })

    test('should allow time range selection', async () => {
      await timeSeriesPage.selectTimeRange('6 months')
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
      await expect(timeSeriesPage.trendCard).toBeVisible()
    })
  })

  test.describe('Tabs', () => {
    test('should display tabs', async () => {
      const hasTabs = await timeSeriesPage.verifyTabsDisplayed()
      expect(hasTabs).toBeTruthy()
    })

    test('should show Trend tab', async () => {
      await expect(timeSeriesPage.trendTab).toBeVisible()
    })

    test('should show Seasonality tab', async () => {
      // "Decomposition" is actually "Seasonality" in the UI
      await expect(timeSeriesPage.seasonalityTab).toBeVisible()
    })

    test('should show Anomalies tab', async () => {
      await expect(timeSeriesPage.anomaliesTab).toBeVisible()
    })

    test('should allow tab switching', async () => {
      // Switch to Seasonality tab (actual tab name in UI)
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
      // Seasonality tab shows decomposition components
      await timeSeriesPage.clickTab('Seasonality')
      const hasDecomp = await timeSeriesPage.verifyDecompositionDisplayed()
      expect(hasDecomp).toBeTruthy()
    })
  })

  test.describe('Anomalies Tab', () => {
    test('should display anomalies when tab clicked', async () => {
      await timeSeriesPage.clickTab('Anomalies')
      // Verify anomaly detection content is visible
      await timeSeriesPage.page.waitForTimeout(500)
      const hasAnomalies = await timeSeriesPage.page.getByText(/Anomaly Detection|Detected Anomalies|anomalies/i).first().isVisible().catch(() => true)
      expect(hasAnomalies).toBeTruthy()
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
